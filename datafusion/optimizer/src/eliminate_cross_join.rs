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

//! [`EliminateCrossJoin`] converts `CROSS JOIN` to `INNER JOIN` if join predicates are available.
use crate::{OptimizerConfig, OptimizerRule};
use std::sync::Arc;

use crate::funded::{
    apply_plan, binary, equijoin_sides, freed, hashable, map_inputs_mut,
    map_subqueries_mut, recomputed, schema_columns, unwrapped, vacated,
};
use crate::join_key_set::JoinKeySet;
use datafusion_common::funding::{Funding, Held};
use datafusion_common::tree_node::{Transformed, TreeNodeRecursion};
use datafusion_common::{NullEquality, Result};
use datafusion_expr::expr::{BinaryExpr, Expr};
use datafusion_expr::logical_plan::{
    Filter, Join, JoinConstraint, JoinType, LogicalPlan, Projection,
};
use datafusion_expr::{Operator, build_join_schema};

#[derive(Default, Debug)]
pub struct EliminateCrossJoin;

impl EliminateCrossJoin {
    #[expect(missing_docs)]
    pub fn new() -> Self {
        Self {}
    }
}

/// Eliminate cross joins by rewriting them to inner joins when possible.
///
/// # Example
/// The initial plan for this query:
/// ```sql
/// select ... from a, b where a.x = b.y and b.xx = 100;
/// ```
///
/// Looks like this:
/// ```text
/// Filter(a.x = b.y AND b.xx = 100)
///  Cross Join
///   TableScan a
///   TableScan b
/// ```
///
/// After the rule is applied, the plan will look like this:
/// ```text
/// Filter(b.xx = 100)
///   InnerJoin(a.x = b.y)
///     TableScan a
///     TableScan b
/// ```
///
/// # Other Examples
/// * 'select ... from a, b where a.x = b.y and b.xx = 100;'
/// * 'select ... from a, b where (a.x = b.y and b.xx = 100) or (a.x = b.y and b.xx = 200);'
/// * 'select ... from a, b, c where (a.x = b.y and b.xx = 100 and a.z = c.z)
/// * or (a.x = b.y and b.xx = 200 and a.z=c.z);'
/// * 'select ... from a, b where a.x > b.y'
///
/// For above queries, the join predicate is available in filters and they are moved to
/// join nodes appropriately
///
/// This fix helps to improve the performance of TPCH Q19. issue#78
impl OptimizerRule for EliminateCrossJoin {
    fn supports_rewrite(&self) -> bool {
        true
    }

    #[cfg_attr(feature = "recursive_protection", recursive::recursive)]
    fn rewrite(
        &self,
        plan: LogicalPlan,
        config: &dyn OptimizerConfig,
    ) -> Result<Transformed<LogicalPlan>> {
        // Fast path: nothing to do if the plan contains no `Join` nodes.
        // Without this guard the rule still falls through to
        // `rewrite_children`, which walks the entire plan, processes
        // uncorrelated subqueries, and rewrites every direct child in place
        // (copy-on-write) — paid by every query in the
        // logical optimizer pipeline. Same shape as the
        // `plan_has_subqueries` fast-path landed in #22298.
        let funding = config.funding();
        if !plan_has_joins(&funding, &plan)? {
            return Ok(Transformed::no(plan));
        }

        let plan_schema = Arc::clone(plan.schema());
        let mut possible_join_keys = JoinKeySet::new(&funding);
        let mut all_inputs: Held<Vec<LogicalPlan>> = funding.held();
        let mut all_filters: Held<Vec<Expr>> = funding.held();
        let mut null_equality = NullEquality::NullEqualsNothing;

        let parent_predicate = if let LogicalPlan::Filter(filter) = plan {
            // if input isn't a join that can potentially be rewritten
            // avoid unwrapping the input
            let rewritable = matches!(
                filter.input.as_ref(),
                LogicalPlan::Join(Join {
                    join_type: JoinType::Inner,
                    ..
                })
            );

            if !rewritable {
                // recursively try to rewrite children
                return rewrite_children(self, LogicalPlan::Filter(filter), config);
            }

            if !can_flatten_join_inputs(&filter.input) {
                return Ok(Transformed::no(LogicalPlan::Filter(filter)));
            }

            let Filter {
                input, predicate, ..
            } = filter;

            // Extract null_equality setting from the input join
            if let LogicalPlan::Join(join) = input.as_ref() {
                null_equality = join.null_equality;
            }

            flatten_join_inputs(
                &funding,
                unwrapped(&funding, input)?,
                &mut possible_join_keys,
                &mut all_inputs,
                &mut all_filters,
            )?;

            extract_possible_join_keys(&predicate, &mut possible_join_keys)?;
            Some(predicate)
        } else {
            match plan {
                LogicalPlan::Join(Join {
                    join_type: JoinType::Inner,
                    null_equality: original_null_equality,
                    ..
                }) => {
                    if !can_flatten_join_inputs(&plan) {
                        return Ok(Transformed::no(plan));
                    }
                    flatten_join_inputs(
                        &funding,
                        plan,
                        &mut possible_join_keys,
                        &mut all_inputs,
                        &mut all_filters,
                    )?;
                    null_equality = original_null_equality;
                    None
                }
                _ => {
                    // recursively try to rewrite children
                    return rewrite_children(self, plan, config);
                }
            }
        };

        // Join keys are handled locally: whether each possible key joins two inputs.
        let mut all_join_keys: Held<Vec<bool>> = funding.held();
        all_join_keys.reserve_exact(possible_join_keys.len())?;
        // Room for each is charged: extending allocates nothing.
        all_join_keys.extend(std::iter::repeat_n(false, possible_join_keys.len()));
        let mut left = all_inputs.remove(0);
        while !all_inputs.is_empty() {
            left = find_inner_join(
                &funding,
                left,
                &mut all_inputs,
                &possible_join_keys,
                &mut all_join_keys,
                null_equality,
            )?;
        }

        left = rewrite_children(self, left, config)?.data;

        if &plan_schema != left.schema() {
            left = LogicalPlan::Projection(Projection::try_new_with_schema(
                schema_columns(&funding, &plan_schema)?,
                funding.arc(left)?,
                Arc::clone(&plan_schema),
            )?);
        }

        if !all_filters.is_empty() {
            // Add any filters on top - PushDownFilter can push filters down to applicable join
            let mut predicate = all_filters.swap_remove(0);
            for filter in all_filters.drain(..) {
                predicate = binary(&funding, predicate, Operator::And, filter)?;
            }
            left = LogicalPlan::Filter(Filter::try_new(predicate, funding.arc(left)?)?);
        }

        let Some(predicate) = parent_predicate else {
            return Ok(Transformed::yes(left));
        };

        // If there are no join keys then do nothing:
        if !all_join_keys.contains(&true) {
            Filter::try_new(predicate, funding.arc(left)?)
                .map(|filter| Transformed::yes(LogicalPlan::Filter(filter)))
        } else {
            // Remove join expressions from filter:
            match remove_join_expressions(
                &funding,
                predicate,
                &possible_join_keys,
                &all_join_keys,
            )? {
                Some(filter_expr) => Filter::try_new(filter_expr, funding.arc(left)?)
                    .map(|filter| Transformed::yes(LogicalPlan::Filter(filter))),
                _ => Ok(Transformed::yes(left)),
            }
        }
    }

    fn name(&self) -> &str {
        "eliminate_cross_join"
    }
}

/// Returns `true` if `plan` contains at least one [`LogicalPlan::Join`]
/// node, either directly in its tree *or* inside an embedded subquery
/// plan reachable through `Expr::ScalarSubquery` / `Expr::InSubquery`
/// / `Expr::Exists` / `Expr::SetComparison`.
///
/// Used as a fast-path gate at the top of [`EliminateCrossJoin::rewrite`]
/// so that join-free plans skip the full recursive rewrite. Subquery
/// traversal matters because `rewrite_children` also dives into
/// uncorrelated subqueries; ignoring
/// them here would skip optimizing a `CROSS JOIN` that sits only inside
/// an `IN (SELECT ... FROM a, b)`-style predicate.
///
/// [`apply_plan`] walks this node, every child and every subquery plan, as
/// `LogicalPlan::apply_with_subqueries` does, without listing any node's inputs.
fn plan_has_joins(funding: &Funding, plan: &LogicalPlan) -> Result<bool> {
    let mut found = false;
    apply_plan(funding, plan, true, &mut |node| {
        if matches!(node, LogicalPlan::Join(_)) {
            found = true;
            Ok(TreeNodeRecursion::Stop)
        } else {
            Ok(TreeNodeRecursion::Continue)
        }
    })?;
    Ok(found)
}

/// `plan` rewritten by `optimizer` in place.
fn rewritten_in_place(
    optimizer: &EliminateCrossJoin,
    plan: &mut LogicalPlan,
    config: &dyn OptimizerConfig,
) -> Result<bool> {
    let rewritten = optimizer.rewrite(std::mem::take(plan), config)?;
    *plan = rewritten.data;
    Ok(rewritten.transformed)
}

fn rewrite_children(
    optimizer: &EliminateCrossJoin,
    mut plan: LogicalPlan,
    config: &dyn OptimizerConfig,
) -> Result<Transformed<LogicalPlan>> {
    // Process uncorrelated subqueries in expressions, then direct children.
    let mut changed = map_subqueries_mut(&mut plan, &mut |subquery| {
        if matches!(
            subquery,
            LogicalPlan::Subquery(sq) if sq.outer_ref_columns.is_empty()
        ) {
            rewritten_in_place(optimizer, subquery, config)
        } else {
            Ok(false)
        }
    })?;
    changed |= map_inputs_mut(&config.funding(), &mut plan, &mut |input| {
        rewritten_in_place(optimizer, input, config)
    })?;

    // recompute schema if the plan was transformed
    if changed {
        plan = recomputed(&config.funding(), plan)?;
    }
    Ok(Transformed::new_transformed(plan, changed))
}

/// Recursively accumulate possible_join_keys and inputs from inner joins
/// (including cross joins).
///
/// Assumes can_flatten_join_inputs has returned true and thus the plan can be
/// flattened. Adds all leaf inputs to `all_inputs` and join_keys to
/// possible_join_keys, each list charged as it grows, each join's list of keys given back
fn flatten_join_inputs(
    funding: &Funding,
    plan: LogicalPlan,
    possible_join_keys: &mut JoinKeySet,
    all_inputs: &mut Held<Vec<LogicalPlan>>,
    all_filters: &mut Held<Vec<Expr>>,
) -> Result<()> {
    match plan {
        LogicalPlan::Join(join) if join.join_type == JoinType::Inner => {
            let Join {
                left,
                right,
                mut on,
                filter,
                schema,
                ..
            } = join;
            if let Some(filter) = filter {
                all_filters.push(filter)?;
            }
            possible_join_keys.insert_all_owned(on.drain(..))?;
            freed(funding, (on, schema));
            flatten_join_inputs(
                funding,
                unwrapped(funding, left)?,
                possible_join_keys,
                all_inputs,
                all_filters,
            )?;
            flatten_join_inputs(
                funding,
                unwrapped(funding, right)?,
                possible_join_keys,
                all_inputs,
                all_filters,
            )?;
        }
        _ => {
            all_inputs.push(plan)?;
        }
    };
    Ok(())
}

/// Returns true if the plan is a Join or Cross join could be flattened with
/// `flatten_join_inputs`
///
/// Must stay in sync with `flatten_join_inputs`
fn can_flatten_join_inputs(plan: &LogicalPlan) -> bool {
    // can only flatten inner / cross joins
    let LogicalPlan::Join(join) = plan else {
        return false;
    };
    if join.join_type != JoinType::Inner {
        return false;
    }

    [&join.left, &join.right].into_iter().all(|child| {
        !matches!(
            child.as_ref(),
            LogicalPlan::Join(Join {
                join_type: JoinType::Inner,
                ..
            })
        ) || can_flatten_join_inputs(child)
    })
}

/// Finds the next to join with the left input plan,
///
/// Finds the next `right` from `rights` that can be joined with `left_input`
/// plan based on the join keys in `possible_join_keys`.
///
/// If such a matching `right` is found:
/// 1. Marks the matching join keys in `all_join_keys`.
/// 2. Returns `left_input JOIN right ON (all join keys)`, the keys copied.
///
/// If no matching `right` is found:
/// 1. Removes the first plan from `rights`
/// 2. Returns `left_input CROSS JOIN right`.
fn find_inner_join(
    funding: &Funding,
    left_input: LogicalPlan,
    rights: &mut Held<Vec<LogicalPlan>>,
    possible_join_keys: &JoinKeySet,
    all_join_keys: &mut [bool],
    null_equality: NullEquality,
) -> Result<LogicalPlan> {
    for i in 0..rights.len() {
        let right_input = &rights[i];
        let mut join_keys = vec![];

        for (at, (l, r)) in possible_join_keys.iter().enumerate() {
            let Some(swapped) =
                equijoin_sides(l, r, left_input.schema(), right_input.schema())?
            else {
                continue;
            };
            let (valid_l, valid_r) = if swapped { (r, l) } else { (l, r) };

            // Save join keys
            if hashable(valid_l, left_input.schema())? {
                let key = (funding.cloned(valid_l)?, funding.cloned(valid_r)?);
                funding.push(&mut join_keys, key)?;
                all_join_keys[at] = true;
            }
        }

        // Found one or more matching join keys
        if !join_keys.is_empty() {
            let right_input = rights.remove(i);
            return inner_join(
                funding,
                left_input,
                right_input,
                join_keys,
                null_equality,
            );
        }
    }

    // no matching right plan had any join keys, cross join with the first right
    // plan
    let right = rights.remove(0);
    inner_join(funding, left_input, right, vec![], null_equality)
}

/// `left` joined to `right` on `on`, the join's schema and inputs' blocks charged before
/// they are made.
fn inner_join(
    funding: &Funding,
    left: LogicalPlan,
    right: LogicalPlan,
    on: Vec<(Expr, Expr)>,
    null_equality: NullEquality,
) -> Result<LogicalPlan> {
    let join_schema = funding.arc(build_join_schema(
        left.schema(),
        right.schema(),
        &JoinType::Inner,
        funding,
    )?)?;

    Ok(LogicalPlan::Join(Join {
        left: funding.arc(left)?,
        right: funding.arc(right)?,
        join_type: JoinType::Inner,
        join_constraint: JoinConstraint::On,
        on,
        filter: None,
        schema: join_schema,
        null_equality,
        null_aware: false,
    }))
}

/// Extract join keys from a WHERE clause
fn extract_possible_join_keys(expr: &Expr, join_keys: &mut JoinKeySet) -> Result<()> {
    if let Expr::BinaryExpr(BinaryExpr { left, op, right }) = expr {
        match op {
            Operator::Eq => {
                // insert handles ensuring  we don't add the same Join keys multiple times
                join_keys.insert(left, right)?;
            }
            Operator::And => {
                extract_possible_join_keys(left, join_keys)?;
                extract_possible_join_keys(right, join_keys)?;
            }
            // Fix for issue#78 join predicates from inside of OR expr also pulled up properly.
            Operator::Or => {
                let mut left_join_keys = JoinKeySet::new(join_keys.funding());
                let mut right_join_keys = JoinKeySet::new(join_keys.funding());

                extract_possible_join_keys(left, &mut left_join_keys)?;
                extract_possible_join_keys(right, &mut right_join_keys)?;

                join_keys.insert_intersection(&left_join_keys, &right_join_keys)?;
            }
            _ => (),
        };
    }
    Ok(())
}

/// Remove join expressions from a filter expression, each removed freed and given back
///
/// # Returns
/// * `Some()` when there are few remaining predicates in filter_expr
/// * `None` otherwise
fn remove_join_expressions(
    funding: &Funding,
    expr: Expr,
    possible_join_keys: &JoinKeySet,
    all_join_keys: &[bool],
) -> Result<Option<Expr>> {
    let is_join_key = |left: &Expr, right: &Expr| {
        possible_join_keys
            .position(left, right)
            .is_some_and(|at| all_join_keys[at])
    };
    Ok(match expr {
        Expr::BinaryExpr(BinaryExpr {
            left,
            op: Operator::Eq,
            right,
        }) if is_join_key(left.as_ref(), right.as_ref()) => {
            // was a join key, so remove it
            freed(funding, (left, right));
            None
        }
        // Fix for issue#78 join predicates from inside of OR expr also pulled up properly.
        Expr::BinaryExpr(BinaryExpr {
            mut left,
            op,
            mut right,
        }) if matches!(op, Operator::And | Operator::Or) => {
            let l = remove_join_expressions(
                funding,
                std::mem::replace(left.as_mut(), vacated()),
                possible_join_keys,
                all_join_keys,
            )?;
            let r = remove_join_expressions(
                funding,
                std::mem::replace(right.as_mut(), vacated()),
                possible_join_keys,
                all_join_keys,
            )?;
            match (l, r) {
                (Some(ll), Some(rr)) => {
                    *left = ll;
                    *right = rr;
                    Some(Expr::BinaryExpr(BinaryExpr::new(left, op, right)))
                }
                (Some(kept), None) | (None, Some(kept)) if op == Operator::And => {
                    freed(funding, (left, right));
                    Some(kept)
                }
                // When either `left` or `right` is empty, it means they are `true`
                // so OR'ing anything with them will also be true
                (l, r) => {
                    freed(funding, (left, right));
                    freed(funding, (l, r));
                    None
                }
            }
        }
        _ => Some(expr),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimizer::OptimizerContext;
    use crate::test::*;

    use datafusion_expr::{
        Operator::{And, Or},
        binary_expr, col, lit,
        logical_plan::builder::LogicalPlanBuilder,
    };
    use insta::assert_snapshot;

    macro_rules! assert_optimized_plan_equal {
        (
            $plan:expr,
            @ $expected:literal $(,)?
        ) => {{
            let starting_schema = Arc::clone($plan.schema());
            let rule = EliminateCrossJoin::new();
            let Transformed {transformed: is_plan_transformed, data: optimized_plan, ..} = rule.rewrite($plan, &OptimizerContext::new()).unwrap();
            let formatted_plan = optimized_plan.display_indent_schema();
            // Ensure the rule was actually applied
            assert!(is_plan_transformed, "failed to optimize plan");
            // Verify the schema remains unchanged
            assert_eq!(&starting_schema, optimized_plan.schema());
            assert_snapshot!(
                formatted_plan,
                @ $expected,
            );

            Ok(())
        }};
    }

    #[test]
    fn eliminate_cross_with_simple_and() -> Result<()> {
        let t1 = test_table_scan_with_name("t1")?;
        let t2 = test_table_scan_with_name("t2")?;

        // could eliminate to inner join since filter has Join predicates
        let plan = LogicalPlanBuilder::from(t1)
            .cross_join(t2)?
            .filter(binary_expr(
                col("t1.a").eq(col("t2.a")),
                And,
                col("t2.c").lt(lit(20u32)),
            ))?
            .build()?;

        assert_optimized_plan_equal!(
            plan,
            @ r"
        Filter: t2.c < UInt32(20) [a:UInt32, b:UInt32, c:UInt32, a:UInt32, b:UInt32, c:UInt32]
          Inner Join: t1.a = t2.a [a:UInt32, b:UInt32, c:UInt32, a:UInt32, b:UInt32, c:UInt32]
            TableScan: t1 [a:UInt32, b:UInt32, c:UInt32]
            TableScan: t2 [a:UInt32, b:UInt32, c:UInt32]
        "
        )
    }

    #[test]
    fn eliminate_cross_with_simple_or() -> Result<()> {
        let t1 = test_table_scan_with_name("t1")?;
        let t2 = test_table_scan_with_name("t2")?;

        // could not eliminate to inner join since filter OR expression and there is no common
        // Join predicates in left and right of OR expr.
        let plan = LogicalPlanBuilder::from(t1)
            .cross_join(t2)?
            .filter(binary_expr(
                col("t1.a").eq(col("t2.a")),
                Or,
                col("t2.b").eq(col("t1.a")),
            ))?
            .build()?;

        assert_optimized_plan_equal!(
            plan,
            @ r"
        Filter: t1.a = t2.a OR t2.b = t1.a [a:UInt32, b:UInt32, c:UInt32, a:UInt32, b:UInt32, c:UInt32]
          Cross Join: [a:UInt32, b:UInt32, c:UInt32, a:UInt32, b:UInt32, c:UInt32]
            TableScan: t1 [a:UInt32, b:UInt32, c:UInt32]
            TableScan: t2 [a:UInt32, b:UInt32, c:UInt32]
        "
        )
    }

    #[test]
    fn eliminate_cross_with_and() -> Result<()> {
        let t1 = test_table_scan_with_name("t1")?;
        let t2 = test_table_scan_with_name("t2")?;

        // could eliminate to inner join
        let plan = LogicalPlanBuilder::from(t1)
            .cross_join(t2)?
            .filter(binary_expr(
                binary_expr(col("t1.a").eq(col("t2.a")), And, col("t2.c").lt(lit(20u32))),
                And,
                binary_expr(col("t1.a").eq(col("t2.a")), And, col("t2.c").eq(lit(10u32))),
            ))?
            .build()?;

        assert_optimized_plan_equal!(
            plan,
            @ r"
        Filter: t2.c < UInt32(20) AND t2.c = UInt32(10) [a:UInt32, b:UInt32, c:UInt32, a:UInt32, b:UInt32, c:UInt32]
          Inner Join: t1.a = t2.a [a:UInt32, b:UInt32, c:UInt32, a:UInt32, b:UInt32, c:UInt32]
            TableScan: t1 [a:UInt32, b:UInt32, c:UInt32]
            TableScan: t2 [a:UInt32, b:UInt32, c:UInt32]
        "
        )
    }

    #[test]
    fn eliminate_cross_with_or() -> Result<()> {
        let t1 = test_table_scan_with_name("t1")?;
        let t2 = test_table_scan_with_name("t2")?;

        // could eliminate to inner join since Or predicates have common Join predicates
        let plan = LogicalPlanBuilder::from(t1)
            .cross_join(t2)?
            .filter(binary_expr(
                binary_expr(col("t1.a").eq(col("t2.a")), And, col("t2.c").lt(lit(15u32))),
                Or,
                binary_expr(
                    col("t1.a").eq(col("t2.a")),
                    And,
                    col("t2.c").eq(lit(688u32)),
                ),
            ))?
            .build()?;

        assert_optimized_plan_equal!(
            plan,
            @ r"
        Filter: t2.c < UInt32(15) OR t2.c = UInt32(688) [a:UInt32, b:UInt32, c:UInt32, a:UInt32, b:UInt32, c:UInt32]
          Inner Join: t1.a = t2.a [a:UInt32, b:UInt32, c:UInt32, a:UInt32, b:UInt32, c:UInt32]
            TableScan: t1 [a:UInt32, b:UInt32, c:UInt32]
            TableScan: t2 [a:UInt32, b:UInt32, c:UInt32]
        "
        )
    }

    #[test]
    fn eliminate_cross_not_possible_simple() -> Result<()> {
        let t1 = test_table_scan_with_name("t1")?;
        let t2 = test_table_scan_with_name("t2")?;

        // could not eliminate to inner join
        let plan = LogicalPlanBuilder::from(t1)
            .cross_join(t2)?
            .filter(binary_expr(
                binary_expr(col("t1.a").eq(col("t2.a")), And, col("t2.c").lt(lit(15u32))),
                Or,
                binary_expr(
                    col("t1.b").eq(col("t2.b")),
                    And,
                    col("t2.c").eq(lit(688u32)),
                ),
            ))?
            .build()?;

        assert_optimized_plan_equal!(
            plan,
            @ r"
        Filter: t1.a = t2.a AND t2.c < UInt32(15) OR t1.b = t2.b AND t2.c = UInt32(688) [a:UInt32, b:UInt32, c:UInt32, a:UInt32, b:UInt32, c:UInt32]
          Cross Join: [a:UInt32, b:UInt32, c:UInt32, a:UInt32, b:UInt32, c:UInt32]
            TableScan: t1 [a:UInt32, b:UInt32, c:UInt32]
            TableScan: t2 [a:UInt32, b:UInt32, c:UInt32]
        "
        )
    }

    #[test]
    fn eliminate_cross_not_possible() -> Result<()> {
        let t1 = test_table_scan_with_name("t1")?;
        let t2 = test_table_scan_with_name("t2")?;

        // could not eliminate to inner join
        let plan = LogicalPlanBuilder::from(t1)
            .cross_join(t2)?
            .filter(binary_expr(
                binary_expr(col("t1.a").eq(col("t2.a")), And, col("t2.c").lt(lit(15u32))),
                Or,
                binary_expr(col("t1.a").eq(col("t2.a")), Or, col("t2.c").eq(lit(688u32))),
            ))?
            .build()?;

        assert_optimized_plan_equal!(
            plan,
            @ r"
        Filter: t1.a = t2.a AND t2.c < UInt32(15) OR t1.a = t2.a OR t2.c = UInt32(688) [a:UInt32, b:UInt32, c:UInt32, a:UInt32, b:UInt32, c:UInt32]
          Cross Join: [a:UInt32, b:UInt32, c:UInt32, a:UInt32, b:UInt32, c:UInt32]
            TableScan: t1 [a:UInt32, b:UInt32, c:UInt32]
            TableScan: t2 [a:UInt32, b:UInt32, c:UInt32]
        "
        )
    }

    #[test]
    fn eliminate_cross_possible_nested_inner_join_with_filter() -> Result<()> {
        let t1 = test_table_scan_with_name("t1")?;
        let t2 = test_table_scan_with_name("t2")?;
        let t3 = test_table_scan_with_name("t3")?;

        // could not eliminate to inner join with filter
        let plan = LogicalPlanBuilder::from(t1)
            .join(
                t3,
                JoinType::Inner,
                (vec!["t1.a"], vec!["t3.a"]),
                Some(col("t1.a").gt(lit(20u32))),
            )?
            .join(t2, JoinType::Inner, (vec!["t1.a"], vec!["t2.a"]), None)?
            .filter(col("t1.a").gt(lit(15u32)))?
            .build()?;

        assert_optimized_plan_equal!(
            plan,
            @ r"
        Filter: t1.a > UInt32(15) [a:UInt32, b:UInt32, c:UInt32, a:UInt32, b:UInt32, c:UInt32, a:UInt32, b:UInt32, c:UInt32]
          Filter: t1.a > UInt32(20) [a:UInt32, b:UInt32, c:UInt32, a:UInt32, b:UInt32, c:UInt32, a:UInt32, b:UInt32, c:UInt32]
            Inner Join: t1.a = t2.a [a:UInt32, b:UInt32, c:UInt32, a:UInt32, b:UInt32, c:UInt32, a:UInt32, b:UInt32, c:UInt32]
              Inner Join: t1.a = t3.a [a:UInt32, b:UInt32, c:UInt32, a:UInt32, b:UInt32, c:UInt32]
                TableScan: t1 [a:UInt32, b:UInt32, c:UInt32]
                TableScan: t3 [a:UInt32, b:UInt32, c:UInt32]
              TableScan: t2 [a:UInt32, b:UInt32, c:UInt32]
        "
        )
    }

    #[test]
    /// ```txt
    /// filter: a.id = b.id and a.id = c.id
    ///   cross_join a (bc)
    ///     cross_join b c
    /// ```
    /// Without reorder, it will be
    /// ```txt
    ///   inner_join a (bc) on a.id = b.id and a.id = c.id
    ///     cross_join b c
    /// ```
    /// Reorder it to be
    /// ```txt
    ///   inner_join (ab)c and a.id = c.id
    ///     inner_join a b on a.id = b.id
    /// ```
    fn reorder_join_to_eliminate_cross_join_multi_tables() -> Result<()> {
        let t1 = test_table_scan_with_name("t1")?;
        let t2 = test_table_scan_with_name("t2")?;
        let t3 = test_table_scan_with_name("t3")?;

        // could eliminate to inner join
        let plan = LogicalPlanBuilder::from(t1)
            .cross_join(t2)?
            .cross_join(t3)?
            .filter(binary_expr(
                binary_expr(col("t3.a").eq(col("t1.a")), And, col("t3.c").lt(lit(15u32))),
                And,
                binary_expr(col("t3.a").eq(col("t2.a")), And, col("t3.b").lt(lit(15u32))),
            ))?
            .build()?;

        assert_optimized_plan_equal!(
            plan,
            @ r"
        Filter: t3.c < UInt32(15) AND t3.b < UInt32(15) [a:UInt32, b:UInt32, c:UInt32, a:UInt32, b:UInt32, c:UInt32, a:UInt32, b:UInt32, c:UInt32]
          Projection: t1.a, t1.b, t1.c, t2.a, t2.b, t2.c, t3.a, t3.b, t3.c [a:UInt32, b:UInt32, c:UInt32, a:UInt32, b:UInt32, c:UInt32, a:UInt32, b:UInt32, c:UInt32]
            Inner Join: t3.a = t2.a [a:UInt32, b:UInt32, c:UInt32, a:UInt32, b:UInt32, c:UInt32, a:UInt32, b:UInt32, c:UInt32]
              Inner Join: t1.a = t3.a [a:UInt32, b:UInt32, c:UInt32, a:UInt32, b:UInt32, c:UInt32]
                TableScan: t1 [a:UInt32, b:UInt32, c:UInt32]
                TableScan: t3 [a:UInt32, b:UInt32, c:UInt32]
              TableScan: t2 [a:UInt32, b:UInt32, c:UInt32]
        "
        )
    }

    #[test]
    fn eliminate_cross_join_multi_tables() -> Result<()> {
        let t1 = test_table_scan_with_name("t1")?;
        let t2 = test_table_scan_with_name("t2")?;
        let t3 = test_table_scan_with_name("t3")?;
        let t4 = test_table_scan_with_name("t4")?;

        // could eliminate to inner join
        let plan1 = LogicalPlanBuilder::from(t1)
            .cross_join(t2)?
            .filter(binary_expr(
                binary_expr(col("t1.a").eq(col("t2.a")), And, col("t2.c").lt(lit(15u32))),
                Or,
                binary_expr(
                    col("t1.a").eq(col("t2.a")),
                    And,
                    col("t2.c").eq(lit(688u32)),
                ),
            ))?
            .build()?;

        let plan2 = LogicalPlanBuilder::from(t3)
            .cross_join(t4)?
            .filter(binary_expr(
                binary_expr(
                    binary_expr(
                        col("t3.a").eq(col("t4.a")),
                        And,
                        col("t4.c").lt(lit(15u32)),
                    ),
                    Or,
                    binary_expr(
                        col("t3.a").eq(col("t4.a")),
                        And,
                        col("t3.c").eq(lit(688u32)),
                    ),
                ),
                Or,
                binary_expr(
                    col("t3.a").eq(col("t4.a")),
                    And,
                    col("t3.b").eq(col("t4.b")),
                ),
            ))?
            .build()?;

        let plan = LogicalPlanBuilder::from(plan1)
            .cross_join(plan2)?
            .filter(binary_expr(
                binary_expr(col("t3.a").eq(col("t1.a")), And, col("t4.c").lt(lit(15u32))),
                Or,
                binary_expr(
                    col("t3.a").eq(col("t1.a")),
                    And,
                    col("t4.c").eq(lit(688u32)),
                ),
            ))?
            .build()?;

        assert_optimized_plan_equal!(
            plan,
            @ r"
        Filter: t4.c < UInt32(15) OR t4.c = UInt32(688) [a:UInt32, b:UInt32, c:UInt32, a:UInt32, b:UInt32, c:UInt32, a:UInt32, b:UInt32, c:UInt32, a:UInt32, b:UInt32, c:UInt32]
          Inner Join: t1.a = t3.a [a:UInt32, b:UInt32, c:UInt32, a:UInt32, b:UInt32, c:UInt32, a:UInt32, b:UInt32, c:UInt32, a:UInt32, b:UInt32, c:UInt32]
            Filter: t2.c < UInt32(15) OR t2.c = UInt32(688) [a:UInt32, b:UInt32, c:UInt32, a:UInt32, b:UInt32, c:UInt32]
              Inner Join: t1.a = t2.a [a:UInt32, b:UInt32, c:UInt32, a:UInt32, b:UInt32, c:UInt32]
                TableScan: t1 [a:UInt32, b:UInt32, c:UInt32]
                TableScan: t2 [a:UInt32, b:UInt32, c:UInt32]
            Filter: t4.c < UInt32(15) OR t3.c = UInt32(688) OR t3.b = t4.b [a:UInt32, b:UInt32, c:UInt32, a:UInt32, b:UInt32, c:UInt32]
              Inner Join: t3.a = t4.a [a:UInt32, b:UInt32, c:UInt32, a:UInt32, b:UInt32, c:UInt32]
                TableScan: t3 [a:UInt32, b:UInt32, c:UInt32]
                TableScan: t4 [a:UInt32, b:UInt32, c:UInt32]
        "
        )
    }

    #[test]
    fn eliminate_cross_join_multi_tables_1() -> Result<()> {
        let t1 = test_table_scan_with_name("t1")?;
        let t2 = test_table_scan_with_name("t2")?;
        let t3 = test_table_scan_with_name("t3")?;
        let t4 = test_table_scan_with_name("t4")?;

        // could eliminate to inner join
        let plan1 = LogicalPlanBuilder::from(t1)
            .cross_join(t2)?
            .filter(binary_expr(
                binary_expr(col("t1.a").eq(col("t2.a")), And, col("t2.c").lt(lit(15u32))),
                Or,
                binary_expr(
                    col("t1.a").eq(col("t2.a")),
                    And,
                    col("t2.c").eq(lit(688u32)),
                ),
            ))?
            .build()?;

        // could eliminate to inner join
        let plan2 = LogicalPlanBuilder::from(t3)
            .cross_join(t4)?
            .filter(binary_expr(
                binary_expr(
                    binary_expr(
                        col("t3.a").eq(col("t4.a")),
                        And,
                        col("t4.c").lt(lit(15u32)),
                    ),
                    Or,
                    binary_expr(
                        col("t3.a").eq(col("t4.a")),
                        And,
                        col("t3.c").eq(lit(688u32)),
                    ),
                ),
                Or,
                binary_expr(
                    col("t3.a").eq(col("t4.a")),
                    And,
                    col("t3.b").eq(col("t4.b")),
                ),
            ))?
            .build()?;

        // could not eliminate to inner join
        let plan = LogicalPlanBuilder::from(plan1)
            .cross_join(plan2)?
            .filter(binary_expr(
                binary_expr(col("t3.a").eq(col("t1.a")), And, col("t4.c").lt(lit(15u32))),
                Or,
                binary_expr(col("t3.a").eq(col("t1.a")), Or, col("t4.c").eq(lit(688u32))),
            ))?
            .build()?;

        assert_optimized_plan_equal!(
            plan,
            @ r"
        Filter: t3.a = t1.a AND t4.c < UInt32(15) OR t3.a = t1.a OR t4.c = UInt32(688) [a:UInt32, b:UInt32, c:UInt32, a:UInt32, b:UInt32, c:UInt32, a:UInt32, b:UInt32, c:UInt32, a:UInt32, b:UInt32, c:UInt32]
          Cross Join: [a:UInt32, b:UInt32, c:UInt32, a:UInt32, b:UInt32, c:UInt32, a:UInt32, b:UInt32, c:UInt32, a:UInt32, b:UInt32, c:UInt32]
            Filter: t2.c < UInt32(15) OR t2.c = UInt32(688) [a:UInt32, b:UInt32, c:UInt32, a:UInt32, b:UInt32, c:UInt32]
              Inner Join: t1.a = t2.a [a:UInt32, b:UInt32, c:UInt32, a:UInt32, b:UInt32, c:UInt32]
                TableScan: t1 [a:UInt32, b:UInt32, c:UInt32]
                TableScan: t2 [a:UInt32, b:UInt32, c:UInt32]
            Filter: t4.c < UInt32(15) OR t3.c = UInt32(688) OR t3.b = t4.b [a:UInt32, b:UInt32, c:UInt32, a:UInt32, b:UInt32, c:UInt32]
              Inner Join: t3.a = t4.a [a:UInt32, b:UInt32, c:UInt32, a:UInt32, b:UInt32, c:UInt32]
                TableScan: t3 [a:UInt32, b:UInt32, c:UInt32]
                TableScan: t4 [a:UInt32, b:UInt32, c:UInt32]
        "
        )
    }

    #[test]
    fn eliminate_cross_join_multi_tables_2() -> Result<()> {
        let t1 = test_table_scan_with_name("t1")?;
        let t2 = test_table_scan_with_name("t2")?;
        let t3 = test_table_scan_with_name("t3")?;
        let t4 = test_table_scan_with_name("t4")?;

        // could eliminate to inner join
        let plan1 = LogicalPlanBuilder::from(t1)
            .cross_join(t2)?
            .filter(binary_expr(
                binary_expr(col("t1.a").eq(col("t2.a")), And, col("t2.c").lt(lit(15u32))),
                Or,
                binary_expr(
                    col("t1.a").eq(col("t2.a")),
                    And,
                    col("t2.c").eq(lit(688u32)),
                ),
            ))?
            .build()?;

        // could not eliminate to inner join
        let plan2 = LogicalPlanBuilder::from(t3)
            .cross_join(t4)?
            .filter(binary_expr(
                binary_expr(
                    binary_expr(
                        col("t3.a").eq(col("t4.a")),
                        And,
                        col("t4.c").lt(lit(15u32)),
                    ),
                    Or,
                    binary_expr(
                        col("t3.a").eq(col("t4.a")),
                        And,
                        col("t3.c").eq(lit(688u32)),
                    ),
                ),
                Or,
                binary_expr(col("t3.a").eq(col("t4.a")), Or, col("t3.b").eq(col("t4.b"))),
            ))?
            .build()?;

        // could eliminate to inner join
        let plan = LogicalPlanBuilder::from(plan1)
            .cross_join(plan2)?
            .filter(binary_expr(
                binary_expr(col("t3.a").eq(col("t1.a")), And, col("t4.c").lt(lit(15u32))),
                Or,
                binary_expr(
                    col("t3.a").eq(col("t1.a")),
                    And,
                    col("t4.c").eq(lit(688u32)),
                ),
            ))?
            .build()?;

        assert_optimized_plan_equal!(
            plan,
            @ r"
        Filter: t4.c < UInt32(15) OR t4.c = UInt32(688) [a:UInt32, b:UInt32, c:UInt32, a:UInt32, b:UInt32, c:UInt32, a:UInt32, b:UInt32, c:UInt32, a:UInt32, b:UInt32, c:UInt32]
          Inner Join: t1.a = t3.a [a:UInt32, b:UInt32, c:UInt32, a:UInt32, b:UInt32, c:UInt32, a:UInt32, b:UInt32, c:UInt32, a:UInt32, b:UInt32, c:UInt32]
            Filter: t2.c < UInt32(15) OR t2.c = UInt32(688) [a:UInt32, b:UInt32, c:UInt32, a:UInt32, b:UInt32, c:UInt32]
              Inner Join: t1.a = t2.a [a:UInt32, b:UInt32, c:UInt32, a:UInt32, b:UInt32, c:UInt32]
                TableScan: t1 [a:UInt32, b:UInt32, c:UInt32]
                TableScan: t2 [a:UInt32, b:UInt32, c:UInt32]
            Filter: t3.a = t4.a AND t4.c < UInt32(15) OR t3.a = t4.a AND t3.c = UInt32(688) OR t3.a = t4.a OR t3.b = t4.b [a:UInt32, b:UInt32, c:UInt32, a:UInt32, b:UInt32, c:UInt32]
              Cross Join: [a:UInt32, b:UInt32, c:UInt32, a:UInt32, b:UInt32, c:UInt32]
                TableScan: t3 [a:UInt32, b:UInt32, c:UInt32]
                TableScan: t4 [a:UInt32, b:UInt32, c:UInt32]
        "
        )
    }

    #[test]
    fn eliminate_cross_join_multi_tables_3() -> Result<()> {
        let t1 = test_table_scan_with_name("t1")?;
        let t2 = test_table_scan_with_name("t2")?;
        let t3 = test_table_scan_with_name("t3")?;
        let t4 = test_table_scan_with_name("t4")?;

        // could not eliminate to inner join
        let plan1 = LogicalPlanBuilder::from(t1)
            .cross_join(t2)?
            .filter(binary_expr(
                binary_expr(col("t1.a").eq(col("t2.a")), Or, col("t2.c").lt(lit(15u32))),
                Or,
                binary_expr(
                    col("t1.a").eq(col("t2.a")),
                    And,
                    col("t2.c").eq(lit(688u32)),
                ),
            ))?
            .build()?;

        // could eliminate to inner join
        let plan2 = LogicalPlanBuilder::from(t3)
            .cross_join(t4)?
            .filter(binary_expr(
                binary_expr(
                    binary_expr(
                        col("t3.a").eq(col("t4.a")),
                        And,
                        col("t4.c").lt(lit(15u32)),
                    ),
                    Or,
                    binary_expr(
                        col("t3.a").eq(col("t4.a")),
                        And,
                        col("t3.c").eq(lit(688u32)),
                    ),
                ),
                Or,
                binary_expr(
                    col("t3.a").eq(col("t4.a")),
                    And,
                    col("t3.b").eq(col("t4.b")),
                ),
            ))?
            .build()?;

        // could eliminate to inner join
        let plan = LogicalPlanBuilder::from(plan1)
            .cross_join(plan2)?
            .filter(binary_expr(
                binary_expr(col("t3.a").eq(col("t1.a")), And, col("t4.c").lt(lit(15u32))),
                Or,
                binary_expr(
                    col("t3.a").eq(col("t1.a")),
                    And,
                    col("t4.c").eq(lit(688u32)),
                ),
            ))?
            .build()?;

        assert_optimized_plan_equal!(
            plan,
            @ r"
        Filter: t4.c < UInt32(15) OR t4.c = UInt32(688) [a:UInt32, b:UInt32, c:UInt32, a:UInt32, b:UInt32, c:UInt32, a:UInt32, b:UInt32, c:UInt32, a:UInt32, b:UInt32, c:UInt32]
          Inner Join: t1.a = t3.a [a:UInt32, b:UInt32, c:UInt32, a:UInt32, b:UInt32, c:UInt32, a:UInt32, b:UInt32, c:UInt32, a:UInt32, b:UInt32, c:UInt32]
            Filter: t1.a = t2.a OR t2.c < UInt32(15) OR t1.a = t2.a AND t2.c = UInt32(688) [a:UInt32, b:UInt32, c:UInt32, a:UInt32, b:UInt32, c:UInt32]
              Cross Join: [a:UInt32, b:UInt32, c:UInt32, a:UInt32, b:UInt32, c:UInt32]
                TableScan: t1 [a:UInt32, b:UInt32, c:UInt32]
                TableScan: t2 [a:UInt32, b:UInt32, c:UInt32]
            Filter: t4.c < UInt32(15) OR t3.c = UInt32(688) OR t3.b = t4.b [a:UInt32, b:UInt32, c:UInt32, a:UInt32, b:UInt32, c:UInt32]
              Inner Join: t3.a = t4.a [a:UInt32, b:UInt32, c:UInt32, a:UInt32, b:UInt32, c:UInt32]
                TableScan: t3 [a:UInt32, b:UInt32, c:UInt32]
                TableScan: t4 [a:UInt32, b:UInt32, c:UInt32]
        "
        )
    }

    #[test]
    fn eliminate_cross_join_multi_tables_4() -> Result<()> {
        let t1 = test_table_scan_with_name("t1")?;
        let t2 = test_table_scan_with_name("t2")?;
        let t3 = test_table_scan_with_name("t3")?;
        let t4 = test_table_scan_with_name("t4")?;

        // could eliminate to inner join
        // filter: (t1.a = t2.a OR t2.c < 15) AND (t1.a = t2.a AND tc.2 = 688)
        let plan1 = LogicalPlanBuilder::from(t1)
            .cross_join(t2)?
            .filter(binary_expr(
                binary_expr(col("t1.a").eq(col("t2.a")), Or, col("t2.c").lt(lit(15u32))),
                And,
                binary_expr(
                    col("t1.a").eq(col("t2.a")),
                    And,
                    col("t2.c").eq(lit(688u32)),
                ),
            ))?
            .build()?;

        // could eliminate to inner join
        let plan2 = LogicalPlanBuilder::from(t3).cross_join(t4)?.build()?;

        // could eliminate to inner join
        // filter:
        //   ((t3.a = t1.a AND t4.c < 15) OR (t3.a = t1.a AND t4.c = 688))
        //     AND
        //   ((t3.a = t4.a AND t4.c < 15) OR (t3.a = t4.a AND t3.c = 688) OR (t3.a = t4.a AND t3.b = t4.b))
        let plan = LogicalPlanBuilder::from(plan1)
            .cross_join(plan2)?
            .filter(binary_expr(
                binary_expr(
                    binary_expr(
                        col("t3.a").eq(col("t1.a")),
                        And,
                        col("t4.c").lt(lit(15u32)),
                    ),
                    Or,
                    binary_expr(
                        col("t3.a").eq(col("t1.a")),
                        And,
                        col("t4.c").eq(lit(688u32)),
                    ),
                ),
                And,
                binary_expr(
                    binary_expr(
                        binary_expr(
                            col("t3.a").eq(col("t4.a")),
                            And,
                            col("t4.c").lt(lit(15u32)),
                        ),
                        Or,
                        binary_expr(
                            col("t3.a").eq(col("t4.a")),
                            And,
                            col("t3.c").eq(lit(688u32)),
                        ),
                    ),
                    Or,
                    binary_expr(
                        col("t3.a").eq(col("t4.a")),
                        And,
                        col("t3.b").eq(col("t4.b")),
                    ),
                ),
            ))?
            .build()?;

        assert_optimized_plan_equal!(
            plan,
            @ r"
        Filter: (t4.c < UInt32(15) OR t4.c = UInt32(688)) AND (t4.c < UInt32(15) OR t3.c = UInt32(688) OR t3.b = t4.b) [a:UInt32, b:UInt32, c:UInt32, a:UInt32, b:UInt32, c:UInt32, a:UInt32, b:UInt32, c:UInt32, a:UInt32, b:UInt32, c:UInt32]
          Inner Join: t3.a = t4.a [a:UInt32, b:UInt32, c:UInt32, a:UInt32, b:UInt32, c:UInt32, a:UInt32, b:UInt32, c:UInt32, a:UInt32, b:UInt32, c:UInt32]
            Inner Join: t1.a = t3.a [a:UInt32, b:UInt32, c:UInt32, a:UInt32, b:UInt32, c:UInt32, a:UInt32, b:UInt32, c:UInt32]
              Filter: t2.c = UInt32(688) [a:UInt32, b:UInt32, c:UInt32, a:UInt32, b:UInt32, c:UInt32]
                Inner Join: t1.a = t2.a [a:UInt32, b:UInt32, c:UInt32, a:UInt32, b:UInt32, c:UInt32]
                  TableScan: t1 [a:UInt32, b:UInt32, c:UInt32]
                  TableScan: t2 [a:UInt32, b:UInt32, c:UInt32]
              TableScan: t3 [a:UInt32, b:UInt32, c:UInt32]
            TableScan: t4 [a:UInt32, b:UInt32, c:UInt32]
        "
        )
    }

    #[test]
    fn eliminate_cross_join_multi_tables_5() -> Result<()> {
        let t1 = test_table_scan_with_name("t1")?;
        let t2 = test_table_scan_with_name("t2")?;
        let t3 = test_table_scan_with_name("t3")?;
        let t4 = test_table_scan_with_name("t4")?;

        // could eliminate to inner join
        let plan1 = LogicalPlanBuilder::from(t1).cross_join(t2)?.build()?;

        // could eliminate to inner join
        let plan2 = LogicalPlanBuilder::from(t3).cross_join(t4)?.build()?;

        // could eliminate to inner join
        // Filter:
        //  ((t3.a = t1.a AND t4.c < 15) OR (t3.a = t1.a AND t4.c = 688))
        //      AND
        //  ((t3.a = t4.a AND t4.c < 15) OR (t3.a = t4.a AND t3.c = 688) OR (t3.a = t4.a AND t3.b = t4.b))
        //      AND
        //  ((t1.a = t2.a OR t2.c < 15) AND (t1.a = t2.a AND t2.c = 688))
        let plan = LogicalPlanBuilder::from(plan1)
            .cross_join(plan2)?
            .filter(binary_expr(
                binary_expr(
                    binary_expr(
                        binary_expr(
                            col("t3.a").eq(col("t1.a")),
                            And,
                            col("t4.c").lt(lit(15u32)),
                        ),
                        Or,
                        binary_expr(
                            col("t3.a").eq(col("t1.a")),
                            And,
                            col("t4.c").eq(lit(688u32)),
                        ),
                    ),
                    And,
                    binary_expr(
                        binary_expr(
                            binary_expr(
                                col("t3.a").eq(col("t4.a")),
                                And,
                                col("t4.c").lt(lit(15u32)),
                            ),
                            Or,
                            binary_expr(
                                col("t3.a").eq(col("t4.a")),
                                And,
                                col("t3.c").eq(lit(688u32)),
                            ),
                        ),
                        Or,
                        binary_expr(
                            col("t3.a").eq(col("t4.a")),
                            And,
                            col("t3.b").eq(col("t4.b")),
                        ),
                    ),
                ),
                And,
                binary_expr(
                    binary_expr(
                        col("t1.a").eq(col("t2.a")),
                        Or,
                        col("t2.c").lt(lit(15u32)),
                    ),
                    And,
                    binary_expr(
                        col("t1.a").eq(col("t2.a")),
                        And,
                        col("t2.c").eq(lit(688u32)),
                    ),
                ),
            ))?
            .build()?;

        assert_optimized_plan_equal!(
            plan,
            @ r"
        Filter: (t4.c < UInt32(15) OR t4.c = UInt32(688)) AND (t4.c < UInt32(15) OR t3.c = UInt32(688) OR t3.b = t4.b) AND t2.c = UInt32(688) [a:UInt32, b:UInt32, c:UInt32, a:UInt32, b:UInt32, c:UInt32, a:UInt32, b:UInt32, c:UInt32, a:UInt32, b:UInt32, c:UInt32]
          Inner Join: t3.a = t4.a [a:UInt32, b:UInt32, c:UInt32, a:UInt32, b:UInt32, c:UInt32, a:UInt32, b:UInt32, c:UInt32, a:UInt32, b:UInt32, c:UInt32]
            Inner Join: t1.a = t3.a [a:UInt32, b:UInt32, c:UInt32, a:UInt32, b:UInt32, c:UInt32, a:UInt32, b:UInt32, c:UInt32]
              Inner Join: t1.a = t2.a [a:UInt32, b:UInt32, c:UInt32, a:UInt32, b:UInt32, c:UInt32]
                TableScan: t1 [a:UInt32, b:UInt32, c:UInt32]
                TableScan: t2 [a:UInt32, b:UInt32, c:UInt32]
              TableScan: t3 [a:UInt32, b:UInt32, c:UInt32]
            TableScan: t4 [a:UInt32, b:UInt32, c:UInt32]
        "
        )
    }

    #[test]
    fn eliminate_cross_join_with_expr_and() -> Result<()> {
        let t1 = test_table_scan_with_name("t1")?;
        let t2 = test_table_scan_with_name("t2")?;

        // could eliminate to inner join since filter has Join predicates
        let plan = LogicalPlanBuilder::from(t1)
            .cross_join(t2)?
            .filter(binary_expr(
                (col("t1.a") + lit(100u32)).eq(col("t2.a") * lit(2u32)),
                And,
                col("t2.c").lt(lit(20u32)),
            ))?
            .build()?;

        assert_optimized_plan_equal!(
            plan,
            @ r"
        Filter: t2.c < UInt32(20) [a:UInt32, b:UInt32, c:UInt32, a:UInt32, b:UInt32, c:UInt32]
          Inner Join: t1.a + UInt32(100) = t2.a * UInt32(2) [a:UInt32, b:UInt32, c:UInt32, a:UInt32, b:UInt32, c:UInt32]
            TableScan: t1 [a:UInt32, b:UInt32, c:UInt32]
            TableScan: t2 [a:UInt32, b:UInt32, c:UInt32]
        "
        )
    }

    #[test]
    fn eliminate_cross_with_expr_or() -> Result<()> {
        let t1 = test_table_scan_with_name("t1")?;
        let t2 = test_table_scan_with_name("t2")?;

        // could not eliminate to inner join since filter OR expression and there is no common
        // Join predicates in left and right of OR expr.
        let plan = LogicalPlanBuilder::from(t1)
            .cross_join(t2)?
            .filter(binary_expr(
                (col("t1.a") + lit(100u32)).eq(col("t2.a") * lit(2u32)),
                Or,
                col("t2.b").eq(col("t1.a")),
            ))?
            .build()?;

        assert_optimized_plan_equal!(
            plan,
            @ r"
        Filter: t1.a + UInt32(100) = t2.a * UInt32(2) OR t2.b = t1.a [a:UInt32, b:UInt32, c:UInt32, a:UInt32, b:UInt32, c:UInt32]
          Cross Join: [a:UInt32, b:UInt32, c:UInt32, a:UInt32, b:UInt32, c:UInt32]
            TableScan: t1 [a:UInt32, b:UInt32, c:UInt32]
            TableScan: t2 [a:UInt32, b:UInt32, c:UInt32]
        "
        )
    }

    #[test]
    fn eliminate_cross_with_common_expr_and() -> Result<()> {
        let t1 = test_table_scan_with_name("t1")?;
        let t2 = test_table_scan_with_name("t2")?;

        // could eliminate to inner join
        let common_join_key = (col("t1.a") + lit(100u32)).eq(col("t2.a") * lit(2u32));
        let plan = LogicalPlanBuilder::from(t1)
            .cross_join(t2)?
            .filter(binary_expr(
                binary_expr(common_join_key.clone(), And, col("t2.c").lt(lit(20u32))),
                And,
                binary_expr(common_join_key, And, col("t2.c").eq(lit(10u32))),
            ))?
            .build()?;

        assert_optimized_plan_equal!(
            plan,
            @ r"
        Filter: t2.c < UInt32(20) AND t2.c = UInt32(10) [a:UInt32, b:UInt32, c:UInt32, a:UInt32, b:UInt32, c:UInt32]
          Inner Join: t1.a + UInt32(100) = t2.a * UInt32(2) [a:UInt32, b:UInt32, c:UInt32, a:UInt32, b:UInt32, c:UInt32]
            TableScan: t1 [a:UInt32, b:UInt32, c:UInt32]
            TableScan: t2 [a:UInt32, b:UInt32, c:UInt32]
        "
        )
    }

    #[test]
    fn eliminate_cross_with_common_expr_or() -> Result<()> {
        let t1 = test_table_scan_with_name("t1")?;
        let t2 = test_table_scan_with_name("t2")?;

        // could eliminate to inner join since Or predicates have common Join predicates
        let common_join_key = (col("t1.a") + lit(100u32)).eq(col("t2.a") * lit(2u32));
        let plan = LogicalPlanBuilder::from(t1)
            .cross_join(t2)?
            .filter(binary_expr(
                binary_expr(common_join_key.clone(), And, col("t2.c").lt(lit(15u32))),
                Or,
                binary_expr(common_join_key, And, col("t2.c").eq(lit(688u32))),
            ))?
            .build()?;

        assert_optimized_plan_equal!(
            plan,
            @ r"
        Filter: t2.c < UInt32(15) OR t2.c = UInt32(688) [a:UInt32, b:UInt32, c:UInt32, a:UInt32, b:UInt32, c:UInt32]
          Inner Join: t1.a + UInt32(100) = t2.a * UInt32(2) [a:UInt32, b:UInt32, c:UInt32, a:UInt32, b:UInt32, c:UInt32]
            TableScan: t1 [a:UInt32, b:UInt32, c:UInt32]
            TableScan: t2 [a:UInt32, b:UInt32, c:UInt32]
        "
        )
    }

    #[test]
    fn reorder_join_with_expr_key_multi_tables() -> Result<()> {
        let t1 = test_table_scan_with_name("t1")?;
        let t2 = test_table_scan_with_name("t2")?;
        let t3 = test_table_scan_with_name("t3")?;

        // could eliminate to inner join
        let plan = LogicalPlanBuilder::from(t1)
            .cross_join(t2)?
            .cross_join(t3)?
            .filter(binary_expr(
                binary_expr(
                    (col("t3.a") + lit(100u32)).eq(col("t1.a") * lit(2u32)),
                    And,
                    col("t3.c").lt(lit(15u32)),
                ),
                And,
                binary_expr(
                    (col("t3.a") + lit(100u32)).eq(col("t2.a") * lit(2u32)),
                    And,
                    col("t3.b").lt(lit(15u32)),
                ),
            ))?
            .build()?;

        assert_optimized_plan_equal!(
            plan,
            @ r"
        Filter: t3.c < UInt32(15) AND t3.b < UInt32(15) [a:UInt32, b:UInt32, c:UInt32, a:UInt32, b:UInt32, c:UInt32, a:UInt32, b:UInt32, c:UInt32]
          Projection: t1.a, t1.b, t1.c, t2.a, t2.b, t2.c, t3.a, t3.b, t3.c [a:UInt32, b:UInt32, c:UInt32, a:UInt32, b:UInt32, c:UInt32, a:UInt32, b:UInt32, c:UInt32]
            Inner Join: t3.a + UInt32(100) = t2.a * UInt32(2) [a:UInt32, b:UInt32, c:UInt32, a:UInt32, b:UInt32, c:UInt32, a:UInt32, b:UInt32, c:UInt32]
              Inner Join: t1.a * UInt32(2) = t3.a + UInt32(100) [a:UInt32, b:UInt32, c:UInt32, a:UInt32, b:UInt32, c:UInt32]
                TableScan: t1 [a:UInt32, b:UInt32, c:UInt32]
                TableScan: t3 [a:UInt32, b:UInt32, c:UInt32]
              TableScan: t2 [a:UInt32, b:UInt32, c:UInt32]
        "
        )
    }

    #[test]
    fn preserve_null_equality_setting() -> Result<()> {
        let t1 = test_table_scan_with_name("t1")?;
        let t2 = test_table_scan_with_name("t2")?;

        // Create an inner join with NullEquality::NullEqualsNull
        let join_schema = Arc::new(build_join_schema(
            t1.schema(),
            t2.schema(),
            &JoinType::Inner,
            &Funding::default(),
        )?);

        let inner_join = LogicalPlan::Join(Join {
            left: Arc::new(t1),
            right: Arc::new(t2),
            join_type: JoinType::Inner,
            join_constraint: JoinConstraint::On,
            on: vec![],
            filter: None,
            schema: join_schema,
            null_equality: NullEquality::NullEqualsNull, // Test preservation
            null_aware: false,
        });

        // Apply filter that can create join conditions
        let plan = LogicalPlanBuilder::from(inner_join)
            .filter(binary_expr(
                col("t1.a").eq(col("t2.a")),
                And,
                col("t2.c").lt(lit(20u32)),
            ))?
            .build()?;

        let rule = EliminateCrossJoin::new();
        let optimized_plan = rule.rewrite(plan, &OptimizerContext::new())?.data;

        // Verify that null_equality is preserved in the optimized plan
        fn check_null_equality_preserved(plan: &LogicalPlan) -> bool {
            match plan {
                LogicalPlan::Join(join) => {
                    // All joins in the optimized plan should preserve null equality
                    if join.null_equality == NullEquality::NullEqualsNothing {
                        return false;
                    }
                    // Recursively check child plans
                    plan.inputs()
                        .iter()
                        .all(|input| check_null_equality_preserved(input))
                }
                _ => {
                    // Recursively check child plans for non-join nodes
                    plan.inputs()
                        .iter()
                        .all(|input| check_null_equality_preserved(input))
                }
            }
        }

        assert!(
            check_null_equality_preserved(&optimized_plan),
            "null_equality setting should be preserved after optimization"
        );

        Ok(())
    }

    // ---------------- fast-path tests ----------------

    /// `plan_has_joins` detects a `Join` at the root of the plan.
    #[test]
    fn plan_has_joins_detects_root_join() -> Result<()> {
        let plan = LogicalPlanBuilder::from(test_table_scan_with_name("t1")?)
            .cross_join(test_table_scan_with_name("t2")?)?
            .build()?;
        assert!(plan_has_joins(&Funding::default(), &plan)?);
        Ok(())
    }

    /// `plan_has_joins` detects a `Join` nested under other operators.
    #[test]
    fn plan_has_joins_detects_nested_join() -> Result<()> {
        let plan = LogicalPlanBuilder::from(test_table_scan_with_name("t1")?)
            .cross_join(test_table_scan_with_name("t2")?)?
            .filter(col("t1.a").eq(col("t2.a")))?
            .project(vec![col("t1.a")])?
            .build()?;
        assert!(plan_has_joins(&Funding::default(), &plan)?);
        Ok(())
    }

    /// Join-free plans return `false` so the fast-path in `rewrite` can
    /// bail out before doing any recursion.
    #[test]
    fn plan_has_joins_returns_false_for_join_free_plan() -> Result<()> {
        let plan = LogicalPlanBuilder::from(test_table_scan_with_name("t1")?)
            .filter(col("a").gt(lit(0_i32)))?
            .project(vec![col("a"), col("b")])?
            .build()?;
        assert!(!plan_has_joins(&Funding::default(), &plan)?);
        Ok(())
    }

    /// `plan_has_joins` walks into embedded subquery plans — e.g. an
    /// outer `Filter` whose predicate is `IN (SELECT ... FROM a, b)`
    /// where the inner plan contains a `CROSS JOIN`. Without this the
    /// fast-path would silently skip optimizing joins-in-subqueries
    /// because `LogicalPlan::apply` doesn't descend into subquery
    /// plan trees.
    #[test]
    fn plan_has_joins_detects_join_inside_subquery() -> Result<()> {
        use datafusion_expr::in_subquery;

        // Subquery plan that itself contains a join.
        let subquery_plan =
            LogicalPlanBuilder::from(test_table_scan_with_name("sub_t1")?)
                .cross_join(test_table_scan_with_name("sub_t2")?)?
                .project(vec![col("sub_t1.a")])?
                .build()?;

        // Outer plan with NO direct Join — only the IN subquery reaches one.
        let outer = LogicalPlanBuilder::from(test_table_scan_with_name("t1")?)
            .filter(in_subquery(col("a"), Arc::new(subquery_plan)))?
            .project(vec![col("a")])?
            .build()?;

        assert!(
            plan_has_joins(&Funding::default(), &outer)?,
            "plan_has_joins must descend into subquery plans"
        );
        Ok(())
    }

    /// `EliminateCrossJoin::rewrite` short-circuits on join-free plans:
    /// no recursion into `rewrite_children`, no `Transformed::yes`,
    /// the plan comes back identical.
    #[test]
    fn rewrite_short_circuits_when_plan_has_no_joins() -> Result<()> {
        let plan = LogicalPlanBuilder::from(test_table_scan_with_name("t1")?)
            .filter(col("a").gt(lit(0_i32)))?
            .project(vec![col("a"), col("b")])?
            .build()?;

        let starting_display = plan.display_indent_schema().to_string();
        let starting_schema = Arc::clone(plan.schema());

        let rule = EliminateCrossJoin::new();
        let Transformed {
            transformed,
            data: optimized_plan,
            ..
        } = rule.rewrite(plan, &OptimizerContext::new())?;

        assert!(
            !transformed,
            "join-free plan should not be marked as transformed"
        );
        assert_eq!(&starting_schema, optimized_plan.schema());
        assert_eq!(
            starting_display,
            optimized_plan.display_indent_schema().to_string()
        );
        Ok(())
    }

    #[test]
    fn eliminating_a_cross_join_is_refused_unfunded() -> Result<()> {
        let t1 = test_table_scan_with_name("t1")?;
        let t2 = test_table_scan_with_name("t2")?;
        let plan = LogicalPlanBuilder::from(t1)
            .cross_join(t2)?
            .filter(col("t1.a").eq(col("t2.a")))?
            .build()?;

        let refused = EliminateCrossJoin::new()
            .rewrite(plan, &refusing_context())
            .unwrap_err();

        assert!(refused.refused().is_some());
        Ok(())
    }
}
