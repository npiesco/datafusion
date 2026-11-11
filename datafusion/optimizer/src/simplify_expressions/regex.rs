// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

use datafusion_common::funding::Funding;
use datafusion_common::tree_node::Transformed;
use datafusion_common::{DataFusionError, Result, ScalarValue};
use datafusion_expr::expr::InList;
use datafusion_expr::{BinaryExpr, Expr, Like, Operator, lit};
use regex_syntax::hir::{Capture, Hir, HirKind, Literal, Look};

use crate::funded::{conjoined, disjoined, freed, replaced};
use crate::simplify_expressions::expr_simplifier::StringScalar;

/// Maximum number of regex alternations (`foo|bar|...`) that will be expanded into multiple `LIKE` expressions.
const MAX_REGEX_ALTERNATIONS_EXPANSION: usize = 4;

const ANY_CHAR_REGEX_PATTERN: &str = ".*";

/// Tries to convert a regexp expression to a `LIKE` or `Eq`/`NotEq` expression.
///
/// This function also validates the regex pattern. And will return error if the
/// pattern is invalid.
///
/// Typical cases this function can simplify:
/// - empty regex pattern to `LIKE '%'`
/// - literal regex patterns to `LIKE '%foo%'`
/// - full anchored regex patterns (e.g. `^foo$`) to `= 'foo'`
/// - partial anchored regex patterns (e.g. `^foo`) to `LIKE 'foo%'`
/// - combinations (alternatives) of the above, will be concatenated with `OR` or `AND`
/// - `EQ .*` to NotNull
/// - `NE .*` to col IS NULL AND Boolean(NULL) (false for any string, or NULL if col is NULL)
///
/// Each expression made is charged to `f` before it is allocated, and what the rewrite
/// frees is given back. What parsing the pattern allocates is `regex_syntax`'s, not
/// charged.
///
/// Dev note: unit tests of this function are in `expr_simplifier.rs`, case `test_simplify_regex`.
pub fn simplify_regex_expr(
    f: &Funding,
    left: Box<Expr>,
    op: Operator,
    right: Box<Expr>,
) -> Result<Transformed<Expr>> {
    // Check if the right operand is a supported string literal
    let Some(string_scalar) = StringScalar::try_from_expr(right.as_ref()) else {
        return Ok(Transformed::no(Expr::BinaryExpr(BinaryExpr {
            left,
            op,
            right,
        })));
    };
    let pattern = string_scalar.as_str();
    let Some(pattern) = pattern else {
        return Ok(Transformed::no(Expr::BinaryExpr(BinaryExpr {
            left,
            op,
            right,
        })));
    };

    let mode = OperatorMode::new(&op);
    // Handle the special case for ".*" pattern
    if pattern == ANY_CHAR_REGEX_PATTERN {
        // `left`'s box is the null test's.
        let new_expr = if mode.not {
            let null_bool = lit(ScalarValue::Boolean(None));
            Expr::BinaryExpr(BinaryExpr {
                left: f.boxed(Expr::IsNull(left))?,
                op: Operator::And,
                right: f.boxed(null_bool)?,
            })
        } else {
            // not null
            Expr::IsNotNull(left)
        };
        return Ok(Transformed::yes(replaced(f, right, new_expr)));
    }

    match regex_syntax::Parser::new().parse(pattern) {
        Ok(hir) => {
            let kind = hir.kind();
            let lowered = if let HirKind::Alternation(alts) = kind {
                if alts.len() <= MAX_REGEX_ALTERNATIONS_EXPANSION {
                    lower_alt(f, &mode, &left, alts, &string_scalar)?
                } else {
                    None
                }
            } else {
                lower_simple(f, &mode, &left, &hir, &string_scalar)?
            };
            if let Some(expr) = lowered {
                return Ok(Transformed::yes(replaced(f, (left, right), expr)));
            }
        }
        Err(e) => {
            // error out early since the execution may fail anyways
            return Err(DataFusionError::Context(
                "Invalid regex".to_owned(),
                Box::new(DataFusionError::External(Box::new(e))),
            ));
        }
    }

    // Leave untouched if optimization didn't work
    Ok(Transformed::no(Expr::BinaryExpr(BinaryExpr {
        left,
        op,
        right,
    })))
}

#[derive(Debug)]
struct OperatorMode {
    /// Negative match.
    not: bool,
    /// Ignore case (`true` for case-insensitive).
    i: bool,
}

impl OperatorMode {
    fn new(op: &Operator) -> Self {
        let not = match op {
            Operator::RegexMatch | Operator::RegexIMatch => false,
            Operator::RegexNotMatch | Operator::RegexNotIMatch => true,
            _ => unreachable!(),
        };

        let i = match op {
            Operator::RegexMatch | Operator::RegexNotMatch => false,
            Operator::RegexIMatch | Operator::RegexNotIMatch => true,
            _ => unreachable!(),
        };

        Self { not, i }
    }

    /// Creates an [`LIKE`](Expr::Like) from the given `LIKE` pattern.
    fn expr(&self, expr: Box<Expr>, pattern: Box<Expr>) -> Expr {
        let like = Like {
            negated: self.not,
            expr,
            pattern,
            escape_char: None,
            case_insensitive: self.i,
        };

        Expr::Like(like)
    }

    /// Creates an [`Expr::BinaryExpr`] of "`left` = `right`" or "`left` != `right`".
    fn expr_matches_literal(&self, left: Box<Expr>, right: Box<Expr>) -> Expr {
        let op = if self.not {
            Operator::NotEq
        } else {
            Operator::Eq
        };
        Expr::BinaryExpr(BinaryExpr { left, op, right })
    }
}

/// `%`, then each literal of `parts`, then `%`, in a text of exactly its bytes charged
/// before it is made; `None` unless each part is a literal safe for `LIKE`.
fn collect_concat_to_like_string(f: &Funding, parts: &[Hir]) -> Result<Option<String>> {
    let mut len = 2usize;
    for sub in parts {
        let HirKind::Literal(l) = sub.kind() else {
            return Ok(None);
        };
        let Some(part) = like_str_from_literal(l) else {
            return Ok(None);
        };
        len = len.saturating_add(part.len());
    }
    let mut s = String::new();
    f.reserve_exact(&mut s, len)?;
    // Within the room charged for it: pushing allocates nothing.
    s.push('%');
    for sub in parts {
        if let HirKind::Literal(l) = sub.kind()
            && let Some(part) = like_str_from_literal(l)
        {
            s.push_str(part);
        }
    }
    s.push('%');
    Ok(Some(s))
}

/// Returns a str represented by `Literal` if it contains a valid utf8
/// sequence and is safe for like (has no '%' and '_')
fn like_str_from_literal(l: &Literal) -> Option<&str> {
    // if not utf8, no good
    let s = std::str::from_utf8(&l.0).ok()?;

    if s.chars().all(is_safe_for_like) {
        Some(s)
    } else {
        None
    }
}

/// Returns a str represented by `Literal` if it contains a valid utf8
fn str_from_literal(l: &Literal) -> Option<&str> {
    // if not utf8, no good
    let s = std::str::from_utf8(&l.0).ok()?;

    Some(s)
}

fn is_safe_for_like(c: char) -> bool {
    (c != '%') && (c != '_')
}

/// Returns true if the elements in a `Concat` pattern are:
/// - `[Look::Start, Look::End]`
/// - `[Look::Start, Literal(_), Look::End]`
fn is_anchored_literal(v: &[Hir]) -> bool {
    match v.len() {
        2..=3 => (),
        _ => return false,
    };

    let first_last = (
        v.first().expect("length checked"),
        v.last().expect("length checked"),
    );
    if !matches!(first_last,
        (s, e) if s.kind() == &HirKind::Look(Look::Start)
        && e.kind() == &HirKind::Look(Look::End)
    ) {
        return false;
    }

    v.iter()
        .skip(1)
        .take(v.len() - 2)
        .all(|h| matches!(h.kind(), HirKind::Literal(_)))
}

/// Returns true if the elements in a `Concat` pattern are:
/// - `[Look::Start, Capture(Alternation(Literals...)), Look::End]`
fn is_anchored_capture(v: &[Hir]) -> bool {
    if v.len() != 3
        || !matches!(
            (v.first().unwrap().kind(), v.last().unwrap().kind()),
            (&HirKind::Look(Look::Start), &HirKind::Look(Look::End))
        )
    {
        return false;
    }

    if let HirKind::Capture(cap, ..) = v[1].kind() {
        let Capture { sub, .. } = cap;
        if let HirKind::Alternation(alters) = sub.kind() {
            let has_non_literal = alters
                .iter()
                .any(|v| !matches!(v.kind(), &HirKind::Literal(_)));
            if has_non_literal {
                return false;
            }
        }
    }

    true
}

/// Returns the `LIKE` pattern if the `Concat` pattern is partial anchored:
/// - `[Look::Start, Literal(_)]`
/// - `[Literal(_), Look::End]`
///
/// Full anchored patterns are handled by [`anchored_literal_to_expr`].
fn partial_anchored_literal_to_like(f: &Funding, v: &[Hir]) -> Result<Option<String>> {
    if v.len() != 2 {
        return Ok(None);
    }

    let (lit, match_begin) = match (&v[0].kind(), &v[1].kind()) {
        (HirKind::Look(Look::Start), HirKind::Literal(l)) => {
            let Some(lit) = like_str_from_literal(l) else {
                return Ok(None);
            };
            (lit, true)
        }
        (HirKind::Literal(l), HirKind::Look(Look::End)) => {
            let Some(lit) = like_str_from_literal(l) else {
                return Ok(None);
            };
            (lit, false)
        }
        _ => return Ok(None),
    };

    if match_begin {
        f.format(format_args!("{lit}%")).map(Some)
    } else {
        f.format(format_args!("%{lit}")).map(Some)
    }
}

/// Extracts a string literal expression assuming that [`is_anchored_literal`]
/// returned true, its text charged before it is made.
fn anchored_literal_to_expr(
    f: &Funding,
    v: &[Hir],
    string_scalar: &StringScalar,
) -> Result<Option<Expr>> {
    match v.len() {
        2 => Ok(Some(string_scalar.to_expr(String::new()))),
        3 => {
            let HirKind::Literal(l) = v[1].kind() else {
                return Ok(None);
            };
            match like_str_from_literal(l) {
                Some(s) => Ok(Some(string_scalar.to_expr(f.string(s)?))),
                None => Ok(None),
            }
        }
        _ => Ok(None),
    }
}

/// The literals of an anchored alternation, in a list of their number, each charged before
/// it is made.
fn anchored_alternation_to_exprs(
    f: &Funding,
    v: &[Hir],
    string_scalar: &StringScalar,
) -> Result<Option<Vec<Expr>>> {
    if 3 != v.len() {
        return Ok(None);
    }

    if let HirKind::Capture(cap, ..) = v[1].kind() {
        let Capture { sub, .. } = cap;
        if let HirKind::Alternation(alters) = sub.kind() {
            // Each must be a literal of text before any is made.
            let all_safe = alters.iter().all(|hir| {
                matches!(hir.kind(), HirKind::Literal(l) if str_from_literal(l).is_some())
            });
            if !all_safe {
                return Ok(None);
            }
            let mut literals = f.vec_with_capacity(alters.len())?;
            for hir in alters {
                if let HirKind::Literal(l) = hir.kind()
                    && let Some(s) = str_from_literal(l)
                {
                    // Within the room charged for it: pushing allocates nothing.
                    literals.push(string_scalar.to_expr(f.string(s)?));
                }
            }

            return Ok(Some(literals));
        } else if let HirKind::Literal(l) = sub.kind() {
            if let Some(s) = str_from_literal(l) {
                let safe_literal = string_scalar.to_expr(f.string(s)?);
                return Ok(Some(f.vec_of([safe_literal])?));
            }
            return Ok(None);
        }
    }
    Ok(None)
}

/// A copy of `left`, boxed, both charged before they are made.
fn boxed_copy(f: &Funding, left: &Expr) -> Result<Box<Expr>> {
    f.boxed(f.cloned(left)?)
}

/// Tries to lower (transform) a simple regex pattern to a LIKE expression, each expression
/// made charged first.
fn lower_simple(
    f: &Funding,
    mode: &OperatorMode,
    left: &Expr,
    hir: &Hir,
    string_scalar: &StringScalar,
) -> Result<Option<Expr>> {
    match hir.kind() {
        HirKind::Empty => {
            let pattern = string_scalar.to_expr(f.string("%")?);
            return Ok(Some(mode.expr(boxed_copy(f, left)?, f.boxed(pattern)?)));
        }
        HirKind::Literal(l) => {
            let Some(s) = like_str_from_literal(l) else {
                return Ok(None);
            };
            let pattern = string_scalar.to_expr(f.format(format_args!("%{s}%"))?);
            return Ok(Some(mode.expr(boxed_copy(f, left)?, f.boxed(pattern)?)));
        }
        HirKind::Concat(inner) if is_anchored_literal(inner) => {
            let Some(right) = anchored_literal_to_expr(f, inner, string_scalar)? else {
                return Ok(None);
            };
            let right = f.boxed(right)?;
            return Ok(Some(if mode.i {
                // Case-insensitive: use ILIKE for exact match (no wildcards)
                mode.expr(boxed_copy(f, left)?, right)
            } else {
                // Case-sensitive: use Eq / NotEq
                mode.expr_matches_literal(boxed_copy(f, left)?, right)
            }));
        }
        HirKind::Concat(inner) if !mode.i && is_anchored_capture(inner) => {
            let Some(right) = anchored_alternation_to_exprs(f, inner, string_scalar)?
            else {
                return Ok(None);
            };
            return Ok(Some(Expr::InList(InList::new(
                boxed_copy(f, left)?,
                right,
                mode.not,
            ))));
        }
        HirKind::Concat(inner) => {
            let pattern = match partial_anchored_literal_to_like(f, inner)? {
                Some(pattern) => Some(pattern),
                None => collect_concat_to_like_string(f, inner)?,
            };
            if let Some(pattern) = pattern {
                let pattern = string_scalar.to_expr(pattern);
                return Ok(Some(mode.expr(boxed_copy(f, left)?, f.boxed(pattern)?)));
            }
        }
        _ => {}
    }
    Ok(None)
}

/// Calls [`lower_simple`] for each alternative and combine the results with `or` or `and`
/// based on [`OperatorMode`]. Any fail attempt to lower an alternative will makes this
/// function to return `None`.
fn lower_alt(
    f: &Funding,
    mode: &OperatorMode,
    left: &Expr,
    alts: &[Hir],
    string_scalar: &StringScalar,
) -> Result<Option<Expr>> {
    let mut accu: Option<Expr> = None;

    for part in alts {
        let Some(expr) = lower_simple(f, mode, left, part, string_scalar)? else {
            // What the alternatives lowered so far made is given back.
            freed(f, accu);
            return Ok(None);
        };
        accu = match accu {
            Some(accu) => {
                if mode.not {
                    Some(conjoined(f, accu, expr)?)
                } else {
                    Some(disjoined(f, accu, expr)?)
                }
            }
            None => Some(expr),
        };
    }

    Ok(Some(accu.expect("at least two alts")))
}
