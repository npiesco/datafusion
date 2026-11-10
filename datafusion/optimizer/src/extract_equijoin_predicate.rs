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

//! [`ExtractEquijoinPredicate`] identifies equality join (equijoin) predicates
use crate::funded::{conjuncts_owned, equijoin_sides, hashable, joined};
use crate::optimizer::ApplyOrder;
use crate::{OptimizerConfig, OptimizerRule};
use datafusion_common::funding::{Funding, Held};
use datafusion_common::tree_node::Transformed;
use datafusion_common::{DFSchema, assert_or_internal_err};
use datafusion_common::{NullEquality, Result};
use datafusion_expr::{BinaryExpr, Expr, Join, LogicalPlan, Operator};
// equijoin predicate
type EquijoinPredicate = (Expr, Expr);

/// Optimizer that splits conjunctive join predicates into equijoin
/// predicates and (other) filter predicates.
///
/// Join algorithms are often highly optimized for equality predicates such as `x = y`,
/// often called `equijoin` predicates, so it is important to locate such predicates
/// and treat them specially.
///
/// For example, `SELECT ... FROM A JOIN B ON (A.x = B.y AND B.z > 50)`
/// has one equijoin predicate (`A.x = B.y`) and one filter predicate (`B.z > 50`).
/// See [datafusion_expr::utils::find_valid_equijoin_key_pair] for more information on
/// what predicates are considered equijoins.
#[derive(Default, Debug)]
pub struct ExtractEquijoinPredicate;

impl ExtractEquijoinPredicate {
    #[expect(missing_docs)]
    pub fn new() -> Self {
        Self {}
    }
}

impl OptimizerRule for ExtractEquijoinPredicate {
    fn supports_rewrite(&self) -> bool {
        true
    }

    fn name(&self) -> &str {
        "extract_equijoin_predicate"
    }

    fn apply_order(&self) -> Option<ApplyOrder> {
        Some(ApplyOrder::BottomUp)
    }

    fn rewrite(
        &self,
        plan: LogicalPlan,
        config: &dyn OptimizerConfig,
    ) -> Result<Transformed<LogicalPlan>> {
        match plan {
            LogicalPlan::Join(Join {
                left,
                right,
                mut on,
                filter: Some(expr),
                join_type,
                join_constraint,
                schema,
                mut null_equality,
                null_aware,
            }) => {
                let funding = config.funding();
                let left_schema = left.schema();
                let right_schema = right.schema();
                let had_keys = !on.is_empty();
                let mut conjuncts = conjuncts_owned(&funding, expr)?;
                let mut extracted = split_op_and_other_join_predicates(
                    &funding,
                    &mut conjuncts,
                    &mut on,
                    left_schema,
                    right_schema,
                    Operator::Eq,
                )?;

                // Equi-join operators like HashJoin support a special behavior
                // that evaluates `NULL = NULL` as true instead of NULL. Therefore,
                // we transform `t1.c1 IS NOT DISTINCT FROM t2.c1` into an equi-join
                // and set the `NullEquality` configuration in the join operator.
                // This allows certain queries to use Hash Join instead of
                // Nested Loop Join, resulting in better performance.
                //
                // Only convert when there are NO equijoin predicates, to be conservative.
                if !had_keys && !extracted && !conjuncts.is_empty() {
                    extracted = split_op_and_other_join_predicates(
                        &funding,
                        &mut conjuncts,
                        &mut on,
                        left_schema,
                        right_schema,
                        Operator::IsNotDistinctFrom,
                    )?;
                    if extracted {
                        // According to `is not distinct from`'s semantics, it's
                        // safe to override it
                        null_equality = NullEquality::NullEqualsNull;
                    }
                }

                let join = LogicalPlan::Join(Join {
                    left,
                    right,
                    on,
                    filter: joined(&funding, conjuncts, Operator::And)?,
                    join_type,
                    join_constraint,
                    schema,
                    null_equality,
                    null_aware,
                });
                Ok(if extracted {
                    Transformed::yes(join)
                } else {
                    Transformed::no(join)
                })
            }
            _ => Ok(Transformed::no(plan)),
        }
    }
}

/// Moves the `operator` predicates among `conjuncts` that are join keys of `left_schema`
/// and `right_schema` into `on`, its growth charged first and their boxes given back;
/// whether any were.
///
/// # Example
///
/// For the conjuncts of `a.id = b.id AND a.x > 10 AND b.x > b.id` and [`Operator::Eq`],
/// `(a.id, b.id)` is moved into `on`, and `a.x > 10` and `b.x > b.id` stay:
/// - It is an equality expression like `lhs == rhs`
/// - All column references in `lhs` are from the left schema, and all in `rhs` are from
///   the right schema (or the other way round, the sides then swapped)
/// - Both sides have hashable types
///
/// # Note
/// For [`Operator::IsNotDistinctFrom`], the caller is responsible for setting
/// `NullEquality::NullEqualsNull` in the join operator: `IS NOT DISTINCT FROM` treats
/// `NULL = NULL` as true, so a valid plan using the extracted keys looks like:
///
/// HashJoin
/// - on: `a.id = b.id` (equality)
/// - join_filter: `(a.x > 10) AND (b.x > b.id)`
/// - nulls_equals_null: `true`
fn split_op_and_other_join_predicates(
    funding: &Funding,
    conjuncts: &mut Held<Vec<Expr>>,
    on: &mut Vec<EquijoinPredicate>,
    left_schema: &DFSchema,
    right_schema: &DFSchema,
    operator: Operator,
) -> Result<bool> {
    assert_or_internal_err!(
        matches!(operator, Operator::Eq | Operator::IsNotDistinctFrom),
        "split_op_and_other_join_predicates only supports 'Eq' or 'IsNotDistinctFrom' operators, \
        but received: {:?}",
        operator
    );

    let mut extracted = false;
    let mut index = 0;
    while index < conjuncts.len() {
        let swapped = match &conjuncts[index] {
            Expr::BinaryExpr(BinaryExpr { left, op, right }) if *op == operator => {
                match equijoin_sides(left, right, left_schema, right_schema)? {
                    Some(swapped) => {
                        let (left_key, right_key) = if swapped {
                            (right, left)
                        } else {
                            (left, right)
                        };
                        (hashable(left_key, left_schema)?
                            && hashable(right_key, right_schema)?)
                        .then_some(swapped)
                    }
                    None => None,
                }
            }
            _ => None,
        };
        if let Some(swapped) = swapped
            && let Expr::BinaryExpr(BinaryExpr { left, right, .. }) =
                conjuncts.remove(index)
        {
            let (left, right) = (funding.unboxed(left), funding.unboxed(right));
            funding.push(
                on,
                if swapped {
                    (right, left)
                } else {
                    (left, right)
                },
            )?;
            extracted = true;
        } else {
            index += 1;
        }
    }
    Ok(extracted)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::OptimizerContext;
    use crate::assert_optimized_plan_eq_display_indent_snapshot;
    use crate::funded::Freed;
    use crate::test::*;
    use arrow::datatypes::DataType;
    use datafusion_common::config::ConfigOptions;
    use datafusion_common::funding::PlanningFunding;
    use datafusion_expr::{
        ExprSchemable, JoinType, col, lit, logical_plan::builder::LogicalPlanBuilder,
    };
    use std::sync::Arc;

    macro_rules! assert_optimized_plan_equal {
        (
            $plan:expr,
            @ $expected:literal $(,)?
        ) => {{
            let rule: Arc<dyn crate::OptimizerRule + Send + Sync> = Arc::new(ExtractEquijoinPredicate {});
            assert_optimized_plan_eq_display_indent_snapshot!(
                rule,
                $plan,
                @ $expected,
            )
        }};
    }

    #[test]
    fn join_with_only_column_equi_predicate() -> Result<()> {
        let t1 = test_table_scan_with_name("t1")?;
        let t2 = test_table_scan_with_name("t2")?;

        let plan = LogicalPlanBuilder::from(t1)
            .join_on(t2, JoinType::Left, Some(col("t1.a").eq(col("t2.a"))))?
            .build()?;

        assert_optimized_plan_equal!(
            plan,
            @r"
        Left Join: t1.a = t2.a [a:UInt32, b:UInt32, c:UInt32, a:UInt32;N, b:UInt32;N, c:UInt32;N]
          TableScan: t1 [a:UInt32, b:UInt32, c:UInt32]
          TableScan: t2 [a:UInt32, b:UInt32, c:UInt32]
        "
        )
    }

    #[test]
    fn join_with_only_equi_expr_predicate() -> Result<()> {
        let t1 = test_table_scan_with_name("t1")?;
        let t2 = test_table_scan_with_name("t2")?;

        let plan = LogicalPlanBuilder::from(t1)
            .join_on(
                t2,
                JoinType::Left,
                Some((col("t1.a") + lit(10i64)).eq(col("t2.a") * lit(2u32))),
            )?
            .build()?;

        assert_optimized_plan_equal!(
            plan,
            @r"
        Left Join: t1.a + Int64(10) = t2.a * UInt32(2) [a:UInt32, b:UInt32, c:UInt32, a:UInt32;N, b:UInt32;N, c:UInt32;N]
          TableScan: t1 [a:UInt32, b:UInt32, c:UInt32]
          TableScan: t2 [a:UInt32, b:UInt32, c:UInt32]
        "
        )
    }

    #[test]
    fn join_with_only_none_equi_predicate() -> Result<()> {
        let t1 = test_table_scan_with_name("t1")?;
        let t2 = test_table_scan_with_name("t2")?;

        let plan = LogicalPlanBuilder::from(t1)
            .join_on(
                t2,
                JoinType::Left,
                Some(
                    (col("t1.a") + lit(10i64))
                        .gt_eq(col("t2.a") * lit(2u32))
                        .and(col("t1.b").lt(lit(100i32))),
                ),
            )?
            .build()?;

        assert_optimized_plan_equal!(
            plan,
            @r"
        Left Join:  Filter: t1.a + Int64(10) >= t2.a * UInt32(2) AND t1.b < Int32(100) [a:UInt32, b:UInt32, c:UInt32, a:UInt32;N, b:UInt32;N, c:UInt32;N]
          TableScan: t1 [a:UInt32, b:UInt32, c:UInt32]
          TableScan: t2 [a:UInt32, b:UInt32, c:UInt32]
        "
        )
    }

    #[test]
    fn join_with_expr_both_from_filter_and_keys() -> Result<()> {
        let t1 = test_table_scan_with_name("t1")?;
        let t2 = test_table_scan_with_name("t2")?;

        let plan = LogicalPlanBuilder::from(t1)
            .join_with_expr_keys(
                t2,
                JoinType::Left,
                (
                    vec![col("t1.a") + lit(11u32)],
                    vec![col("t2.a") * lit(2u32)],
                ),
                Some(
                    (col("t1.a") + lit(10i64))
                        .eq(col("t2.a") * lit(2u32))
                        .and(col("t1.b").lt(lit(100i32))),
                ),
            )?
            .build()?;

        assert_optimized_plan_equal!(
            plan,
            @r"
        Left Join: t1.a + UInt32(11) = t2.a * UInt32(2), t1.a + Int64(10) = t2.a * UInt32(2) Filter: t1.b < Int32(100) [a:UInt32, b:UInt32, c:UInt32, a:UInt32;N, b:UInt32;N, c:UInt32;N]
          TableScan: t1 [a:UInt32, b:UInt32, c:UInt32]
          TableScan: t2 [a:UInt32, b:UInt32, c:UInt32]
        "
        )
    }

    #[test]
    fn join_with_and_or_filter() -> Result<()> {
        let t1 = test_table_scan_with_name("t1")?;
        let t2 = test_table_scan_with_name("t2")?;

        let plan = LogicalPlanBuilder::from(t1)
            .join_on(
                t2,
                JoinType::Left,
                Some(
                    col("t1.c")
                        .eq(col("t2.c"))
                        .or((col("t1.a") + col("t1.b")).gt(col("t2.b") + col("t2.c")))
                        .and(
                            col("t1.a").eq(col("t2.a")).and(col("t1.b").eq(col("t2.b"))),
                        ),
                ),
            )?
            .build()?;

        assert_optimized_plan_equal!(
            plan,
            @r"
        Left Join: t1.a = t2.a, t1.b = t2.b Filter: t1.c = t2.c OR t1.a + t1.b > t2.b + t2.c [a:UInt32, b:UInt32, c:UInt32, a:UInt32;N, b:UInt32;N, c:UInt32;N]
          TableScan: t1 [a:UInt32, b:UInt32, c:UInt32]
          TableScan: t2 [a:UInt32, b:UInt32, c:UInt32]
        "
        )
    }

    #[test]
    fn join_with_multiple_table() -> Result<()> {
        let t1 = test_table_scan_with_name("t1")?;
        let t2 = test_table_scan_with_name("t2")?;
        let t3 = test_table_scan_with_name("t3")?;

        let input = LogicalPlanBuilder::from(t2)
            .join_on(
                t3,
                JoinType::Left,
                Some(
                    col("t2.a")
                        .eq(col("t3.a"))
                        .and((col("t2.a") + col("t3.b")).gt(lit(100u32))),
                ),
            )?
            .build()?;
        let plan = LogicalPlanBuilder::from(t1)
            .join_on(
                input,
                JoinType::Left,
                Some(
                    col("t1.a")
                        .eq(col("t2.a"))
                        .and((col("t1.c") + col("t2.c") + col("t3.c")).lt(lit(100u32))),
                ),
            )?
            .build()?;

        assert_optimized_plan_equal!(
            plan,
            @r"
        Left Join: t1.a = t2.a Filter: t1.c + t2.c + t3.c < UInt32(100) [a:UInt32, b:UInt32, c:UInt32, a:UInt32;N, b:UInt32;N, c:UInt32;N, a:UInt32;N, b:UInt32;N, c:UInt32;N]
          TableScan: t1 [a:UInt32, b:UInt32, c:UInt32]
          Left Join: t2.a = t3.a Filter: t2.a + t3.b > UInt32(100) [a:UInt32, b:UInt32, c:UInt32, a:UInt32;N, b:UInt32;N, c:UInt32;N]
            TableScan: t2 [a:UInt32, b:UInt32, c:UInt32]
            TableScan: t3 [a:UInt32, b:UInt32, c:UInt32]
        "
        )
    }

    #[test]
    fn join_with_multiple_table_and_eq_filter() -> Result<()> {
        let t1 = test_table_scan_with_name("t1")?;
        let t2 = test_table_scan_with_name("t2")?;
        let t3 = test_table_scan_with_name("t3")?;

        let input = LogicalPlanBuilder::from(t2)
            .join_on(
                t3,
                JoinType::Left,
                Some(
                    col("t2.a")
                        .eq(col("t3.a"))
                        .and((col("t2.a") + col("t3.b")).gt(lit(100u32))),
                ),
            )?
            .build()?;
        let plan = LogicalPlanBuilder::from(t1)
            .join_on(
                input,
                JoinType::Left,
                Some(col("t1.a").eq(col("t2.a")).and(col("t2.c").eq(col("t3.c")))),
            )?
            .build()?;

        assert_optimized_plan_equal!(
            plan,
            @r"
        Left Join: t1.a = t2.a Filter: t2.c = t3.c [a:UInt32, b:UInt32, c:UInt32, a:UInt32;N, b:UInt32;N, c:UInt32;N, a:UInt32;N, b:UInt32;N, c:UInt32;N]
          TableScan: t1 [a:UInt32, b:UInt32, c:UInt32]
          Left Join: t2.a = t3.a Filter: t2.a + t3.b > UInt32(100) [a:UInt32, b:UInt32, c:UInt32, a:UInt32;N, b:UInt32;N, c:UInt32;N]
            TableScan: t2 [a:UInt32, b:UInt32, c:UInt32]
            TableScan: t3 [a:UInt32, b:UInt32, c:UInt32]
        "
        )
    }

    #[test]
    fn join_with_alias_filter() -> Result<()> {
        let t1 = test_table_scan_with_name("t1")?;
        let t2 = test_table_scan_with_name("t2")?;

        let t1_schema = Arc::clone(t1.schema());
        let t2_schema = Arc::clone(t2.schema());

        // filter: t1.a + CAST(Int64(1), UInt32) = t2.a + CAST(Int64(2), UInt32) as t1.a + 1 = t2.a + 2
        let filter = Expr::eq(
            col("t1.a") + lit(1i64).cast_to(&DataType::UInt32, &t1_schema)?,
            col("t2.a") + lit(2i32).cast_to(&DataType::UInt32, &t2_schema)?,
        )
        .alias("t1.a + 1 = t2.a + 2");
        let plan = LogicalPlanBuilder::from(t1)
            .join_on(t2, JoinType::Left, Some(filter))?
            .build()?;

        assert_optimized_plan_equal!(
            plan,
            @r"
        Left Join: t1.a + CAST(Int64(1) AS UInt32) = t2.a + CAST(Int32(2) AS UInt32) [a:UInt32, b:UInt32, c:UInt32, a:UInt32;N, b:UInt32;N, c:UInt32;N]
          TableScan: t1 [a:UInt32, b:UInt32, c:UInt32]
          TableScan: t2 [a:UInt32, b:UInt32, c:UInt32]
        "
        )
    }

    struct Refusing;

    impl PlanningFunding for Refusing {
        fn charge(&self, _bytes: usize) -> bool {
            false
        }

        fn discharge(&self, _bytes: usize) {}
    }

    struct Unlimited;

    impl PlanningFunding for Unlimited {
        fn charge(&self, _bytes: usize) -> bool {
            true
        }

        fn discharge(&self, _bytes: usize) {}
    }

    fn join_with_key_in_filter() -> Result<LogicalPlan> {
        let t1 = test_table_scan_with_name("t1")?;
        let t2 = test_table_scan_with_name("t2")?;
        LogicalPlanBuilder::from(t1)
            .join_on(
                t2,
                JoinType::Inner,
                Some(col("t1.a").eq(col("t2.a")).and(col("t1.b").gt(lit(10u32)))),
            )?
            .build()
    }

    #[test]
    fn extracting_a_key_is_refused_unfunded() -> Result<()> {
        let mut options = ConfigOptions::default();
        options.funding = Funding::new(Arc::new(Refusing));
        let config = OptimizerContext::new_with_config_options(Arc::new(options));

        let refused = ExtractEquijoinPredicate::new()
            .rewrite(join_with_key_in_filter()?, &config)
            .unwrap_err();

        assert!(refused.refused().is_some());
        Ok(())
    }

    #[test]
    fn extracted_keys_hold_what_was_charged() -> Result<()> {
        let plan = join_with_key_in_filter()?;
        let mut options = ConfigOptions::default();
        options.funding = Funding::new(Arc::new(Unlimited));
        // Planned through the same funding.
        options.funding.charge(plan.freed_bytes())?;
        let config = OptimizerContext::new_with_config_options(Arc::new(options));

        let extracted = ExtractEquijoinPredicate::new().rewrite(plan, &config)?;

        assert!(extracted.transformed);
        // The key's boxes and the conjunction's are given back, the key list charged.
        assert_eq!(
            config.options().funding.outstanding(),
            extracted.data.freed_bytes()
        );
        Ok(())
    }
}
