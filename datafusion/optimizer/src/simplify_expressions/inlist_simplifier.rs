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

//! This module implements a rule that simplifies the values for `InList`s

use super::THRESHOLD_INLINE_INLIST;

use datafusion_common::funding::Funding;
use datafusion_common::tree_node::{Transformed, TreeNodeRewriter};
use datafusion_common::{Result, internal_err};
use datafusion_expr::expr::InList;
use datafusion_expr::{Expr, Operator};

use crate::funded::binary;

pub(super) struct ShortenInListSimplifier<'a> {
    funding: &'a Funding,
}

impl<'a> ShortenInListSimplifier<'a> {
    pub(super) fn new(funding: &'a Funding) -> Self {
        Self { funding }
    }
}

impl TreeNodeRewriter for ShortenInListSimplifier<'_> {
    type Node = Expr;

    fn f_up(&mut self, expr: Expr) -> Result<Transformed<Expr>> {
        // if expr is a single column reference:
        // expr IN (A, B, ...) --> (expr = A) OR (expr = B) OR (expr = C)
        let shortened = matches!(
            &expr,
            Expr::InList(InList { expr, list, .. })
                if !list.is_empty()
                    && (
                        // For lists with only 1 value we allow more complex expressions to be simplified
                        // e.g SUBSTR(c1, 2, 3) IN ('1') -> SUBSTR(c1, 2, 3) = '1'
                        // for more than one we avoid repeating this potentially expensive
                        // expressions
                        list.len() == 1
                            || list.len() <= THRESHOLD_INLINE_INLIST
                                && expr.try_as_col().is_some()
                    )
        );
        if shortened {
            let Expr::InList(InList {
                mut expr,
                mut list,
                negated,
            }) = expr
            else {
                return internal_err!("Expected an IN list");
            };
            let funding = self.funding;
            let (op, join) = if negated {
                (Operator::NotEq, Operator::And)
            } else {
                (Operator::Eq, Operator::Or)
            };
            // Each item compared to a copy of `expr`, charged first, the last to `expr`
            // itself; the list's block and `expr`'s box are given back.
            //
            // Note that `A and B and C and D` is a left-deep tree structure
            // as such we want to maintain this structure as much as possible
            // to avoid reordering the expression during each optimization
            // pass.
            //
            // Left-deep tree structure for `A and B and C and D`:
            // ```
            //        &
            //       / \
            //      &   D
            //     / \
            //    &   C
            //   / \
            //  A   B
            // ```
            let count = list.len();
            let mut joined: Option<Expr> = None;
            for (at, item) in list.drain(..).enumerate() {
                let compared = if at + 1 < count {
                    funding.cloned(expr.as_ref())?
                } else {
                    std::mem::take(expr.as_mut())
                };
                let comparison = binary(funding, compared, op, item)?;
                joined = Some(match joined {
                    None => comparison,
                    Some(acc) => binary(funding, acc, join, comparison)?,
                });
            }
            funding.released(list);
            funding.released_box(expr);
            return match joined {
                Some(joined) => Ok(Transformed::yes(joined)),
                None => internal_err!("Expected a non-empty IN list"),
            };
        }

        Ok(Transformed::no(expr))
    }
}
