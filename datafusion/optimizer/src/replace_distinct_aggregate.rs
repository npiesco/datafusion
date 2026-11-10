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

//! [`ReplaceDistinctWithAggregate`] replaces `DISTINCT ...` with `GROUP BY ...`

use crate::funded::{
    aliased, column_expr, freed, normalize_in_place, unwrapped, wildcard_columns,
};
use crate::optimizer::{ApplyOrder, ApplyOrder::BottomUp};
use crate::{OptimizerConfig, OptimizerRule};
use std::sync::Arc;

use datafusion_common::funding::Held;
use datafusion_common::tree_node::Transformed;
use datafusion_common::{Dependency, Result, ScalarValue};
use datafusion_expr::expr::AggregateFunction;
use datafusion_expr::logical_plan::builder::project;
use datafusion_expr::{Aggregate, Distinct, DistinctOn, Expr, LogicalPlan};
use datafusion_expr::{Limit, LogicalPlanBuilder, lit};

/// Optimizer that replaces logical [[Distinct]] with a logical [[Aggregate]]
///
/// ```text
/// SELECT DISTINCT a, b FROM tab
/// ```
///
/// Into
/// ```text
/// SELECT a, b FROM tab GROUP BY a, b
/// ```
///
/// On the other hand, for a `DISTINCT ON` query the replacement is
/// a bit more involved and effectively converts
/// ```text
/// SELECT DISTINCT ON (a) b FROM tab ORDER BY a DESC, c
/// ```
///
/// into
/// ```text
/// SELECT b FROM (
///     SELECT a, FIRST_VALUE(b ORDER BY a DESC, c) AS b
///     FROM tab
///     GROUP BY a
/// )
/// ORDER BY a DESC
/// ```
///
/// In case there are no columns, the [[Distinct]] is replaced by a [[Limit]]
///
/// ```text
/// SELECT DISTINCT * FROM empty_table
/// ```
///
/// Into
/// ```text
/// SELECT * FROM empty_table LIMIT 1
/// ```
#[derive(Default, Debug)]
pub struct ReplaceDistinctWithAggregate {}

impl ReplaceDistinctWithAggregate {
    #[expect(missing_docs)]
    pub fn new() -> Self {
        Self {}
    }
}

impl OptimizerRule for ReplaceDistinctWithAggregate {
    fn supports_rewrite(&self) -> bool {
        true
    }

    fn rewrite(
        &self,
        plan: LogicalPlan,
        config: &dyn OptimizerConfig,
    ) -> Result<Transformed<LogicalPlan>> {
        match plan {
            LogicalPlan::Distinct(Distinct::All(input)) => {
                let funding = config.funding();
                let group_expr = wildcard_columns(&funding, &input)?;

                if group_expr.is_empty() {
                    // Special case: there are no columns to group by, so we can't replace it by a group by
                    // however, we can replace it by LIMIT 1 because there is either no output or a single empty row
                    return Ok(Transformed::yes(LogicalPlan::Limit(Limit {
                        skip: None,
                        fetch: Some(funding.boxed(lit(1i64))?),
                        input,
                    })));
                }

                let field_count = input.schema().fields().len();
                for dep in input.schema().functional_dependencies().iter() {
                    // If the input is already unique on all of its columns (e.g.
                    // it is a GROUP BY over exactly these columns), the DISTINCT
                    // is a no-op and we can simply remove it. The dependency mode
                    // must be `Single`: a `Multi` dependence (e.g. a former key
                    // downgraded by a join) means equal rows may occur multiple
                    // times, so the DISTINCT still has work to do.
                    if dep.mode == Dependency::Single
                        && dep.source_indices.len() >= field_count
                        && dep.source_indices[..field_count]
                            .iter()
                            .enumerate()
                            .all(|(idx, f_idx)| idx == *f_idx)
                    {
                        freed(&funding, group_expr);
                        return Ok(Transformed::yes(unwrapped(&funding, input)?));
                    }
                }

                // Replace with aggregation:
                let aggr_plan = LogicalPlan::Aggregate(Aggregate::try_new(
                    input,
                    group_expr,
                    vec![],
                )?);
                Ok(Transformed::yes(aggr_plan))
            }
            LogicalPlan::Distinct(Distinct::On(DistinctOn {
                mut select_expr,
                mut on_expr,
                sort_expr,
                input,
                schema,
            })) => {
                let funding = config.funding();
                let expr_cnt = on_expr.len();

                // Construct the aggregation expression to be used to fetch the selected
                // expressions, each in place of the selection it is made of.
                let first_value_udaf: Arc<datafusion_expr::AggregateUDF> =
                    config.function_registry().unwrap().udaf("first_value")?;
                for e in &mut select_expr {
                    let selected =
                        std::mem::replace(e, Expr::Literal(ScalarValue::Null, None));
                    let order_by = match &sort_expr {
                        Some(order_by) => funding.to_vec(order_by)?,
                        None => vec![],
                    };
                    *e = Expr::AggregateFunction(AggregateFunction::new_udf(
                        Arc::clone(&first_value_udaf),
                        funding.vec_of([selected])?,
                        false,
                        None,
                        order_by,
                        None,
                    ));
                    normalize_in_place(&funding, &input, e)?;
                }
                for e in &mut on_expr {
                    normalize_in_place(&funding, &input, e)?;
                }

                // Build the aggregation plan
                let plan = LogicalPlan::Aggregate(Aggregate::try_new(
                    input,
                    on_expr,
                    select_expr,
                )?);
                // TODO use LogicalPlanBuilder directly rather than recreating the Aggregate
                // when https://github.com/apache/datafusion/issues/10485 is available
                let lpb = LogicalPlanBuilder::new_from_arc(funding.arc(plan)?, &funding);

                let plan = if let Some(mut sort_expr) = sort_expr {
                    // While sort expressions were used in the `FIRST_VALUE` aggregation itself above,
                    // this on it's own isn't enough to guarantee the proper output order of the grouping
                    // (`ON`) expression, so we need to sort those as well.

                    // truncate the sort_expr to the length of on_expr
                    for dropped in sort_expr.drain(expr_cnt.min(sort_expr.len())..) {
                        freed(&funding, dropped);
                    }

                    lpb.sort(sort_expr)?.build()?
                } else {
                    lpb.build()?
                };

                // Whereas the aggregation plan by default outputs both the grouping and the aggregation
                // expressions, for `DISTINCT ON` we only need to emit the original selection expressions.
                let mut project_exprs: Held<Vec<Expr>> = funding.held();
                project_exprs.reserve_exact(schema.fields().len())?;
                for ((new_qualifier, new_field), (old_qualifier, old_field)) in
                    plan.schema().iter().skip(expr_cnt).zip(schema.iter())
                {
                    project_exprs.push(aliased(
                        &funding,
                        column_expr(&funding, new_qualifier, new_field.name())?,
                        old_qualifier,
                        old_field.name(),
                    )?)?;
                }

                let plan = project(plan, project_exprs.drain(..), &funding)?;
                // The DISTINCT ON's schema is replaced by the projection's.
                freed(&funding, schema);

                Ok(Transformed::yes(plan))
            }
            _ => Ok(Transformed::no(plan)),
        }
    }

    fn name(&self) -> &str {
        "replace_distinct_aggregate"
    }

    fn apply_order(&self) -> Option<ApplyOrder> {
        Some(BottomUp)
    }
}

#[cfg(test)]
mod tests {
    use crate::assert_optimized_plan_eq_snapshot;
    use crate::replace_distinct_aggregate::ReplaceDistinctWithAggregate;
    use crate::test::*;
    use arrow::datatypes::{Fields, Schema};
    use std::sync::Arc;

    use crate::funded::Freed;
    use crate::{OptimizerConfig, OptimizerContext, OptimizerRule};
    use datafusion_common::config::ConfigOptions;
    use datafusion_common::funding::{Funding, PlanningFunding};
    use datafusion_common::{Column, Result};
    use datafusion_expr::{
        Expr, JoinType, LogicalPlan, col, logical_plan::builder::LogicalPlanBuilder,
        table_scan,
    };
    use datafusion_functions_aggregate::sum::sum;

    macro_rules! assert_optimized_plan_equal {
        (
            $plan:expr,
            @ $expected:literal $(,)?
        ) => {{
            let optimizer_ctx = OptimizerContext::new().with_max_passes(1);
            let rules: Vec<Arc<dyn crate::OptimizerRule + Send + Sync>> = vec![Arc::new(ReplaceDistinctWithAggregate::new())];
            assert_optimized_plan_eq_snapshot!(
                optimizer_ctx,
                rules,
                $plan,
                @ $expected,
            )
        }};
    }

    #[test]
    fn eliminate_redundant_distinct_simple() -> Result<()> {
        let table_scan = test_table_scan().unwrap();
        let plan = LogicalPlanBuilder::from(table_scan)
            .aggregate(vec![col("c")], Vec::<Expr>::new())?
            .project(vec![col("c")])?
            .distinct()?
            .build()?;

        assert_optimized_plan_equal!(plan, @r"
        Projection: test.c
          Aggregate: groupBy=[[test.c]], aggr=[[]]
            TableScan: test
        ")
    }

    #[test]
    fn eliminate_redundant_distinct_pair() -> Result<()> {
        let table_scan = test_table_scan().unwrap();
        let plan = LogicalPlanBuilder::from(table_scan)
            .aggregate(vec![col("a"), col("b")], Vec::<Expr>::new())?
            .project(vec![col("a"), col("b")])?
            .distinct()?
            .build()?;

        assert_optimized_plan_equal!(plan, @r"
        Projection: test.a, test.b
          Aggregate: groupBy=[[test.a, test.b]], aggr=[[]]
            TableScan: test
        ")
    }

    #[test]
    fn do_not_eliminate_distinct() -> Result<()> {
        let table_scan = test_table_scan().unwrap();
        let plan = LogicalPlanBuilder::from(table_scan)
            .project(vec![col("a"), col("b")])?
            .distinct()?
            .build()?;

        assert_optimized_plan_equal!(plan, @r"
        Aggregate: groupBy=[[test.a, test.b]], aggr=[[]]
          Projection: test.a, test.b
            TableScan: test
        ")
    }

    #[test]
    fn do_not_eliminate_distinct_with_aggr() -> Result<()> {
        let table_scan = test_table_scan().unwrap();
        let plan = LogicalPlanBuilder::from(table_scan)
            .aggregate(vec![col("a"), col("b"), col("c")], vec![sum(col("c"))])?
            .project(vec![col("a"), col("b")])?
            .distinct()?
            .build()?;

        assert_optimized_plan_equal!(plan, @r"
        Aggregate: groupBy=[[test.a, test.b]], aggr=[[]]
          Projection: test.a, test.b
            Aggregate: groupBy=[[test.a, test.b, test.c]], aggr=[[sum(test.c)]]
              TableScan: test
        ")
    }

    #[test]
    fn use_limit_1_when_no_columns() -> Result<()> {
        let plan = table_scan(Some("test"), &Schema::new(Fields::empty()), None)?
            .distinct()?
            .build()?;

        assert_optimized_plan_equal!(plan, @r"
        Limit: skip=0, fetch=1
          TableScan: test
        ")
    }

    #[test]
    fn distinct_over_using_join_groups_each_key_once() -> Result<()> {
        let plan = LogicalPlanBuilder::from(test_table_scan()?)
            .join_using(
                test_table_scan_with_name("test1")?,
                JoinType::Inner,
                vec![Column::from_name("a")],
            )?
            .distinct()?
            .build()?;

        assert_optimized_plan_equal!(plan, @r"
        Aggregate: groupBy=[[test.a, test.b, test.c, test1.b, test1.c]], aggr=[[]]
          Inner Join: Using test.a = test1.a
            TableScan: test
            TableScan: test1
        ")
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

    fn context(funding: Funding) -> OptimizerContext {
        let mut options = ConfigOptions::default();
        options.funding = funding;
        OptimizerContext::new_with_config_options(Arc::new(options))
    }

    #[test]
    fn grouping_a_distinct_is_refused_unfunded() -> Result<()> {
        let plan = LogicalPlanBuilder::from(test_table_scan()?)
            .project(vec![col("a"), col("b")])?
            .distinct()?
            .build()?;
        let config = context(Funding::new(Arc::new(Refusing)));

        let refused = ReplaceDistinctWithAggregate::new()
            .rewrite(plan, &config)
            .unwrap_err();

        assert!(refused.refused().is_some());
        Ok(())
    }

    #[test]
    fn eliminated_distinct_gives_back_what_it_held() -> Result<()> {
        let plan: LogicalPlan = LogicalPlanBuilder::from(test_table_scan()?)
            .aggregate(vec![col("c")], Vec::<Expr>::new())?
            .project(vec![col("c")])?
            .distinct()?
            .build()?;
        let config = context(Funding::new(Arc::new(Unlimited)));
        // Planned through the same funding.
        config.options().funding.charge(plan.freed_bytes())?;

        let replaced = ReplaceDistinctWithAggregate::new().rewrite(plan, &config)?;

        assert!(replaced.transformed);
        // The group list made to compare is given back, and the distinct's input block.
        assert_eq!(
            config.options().funding.outstanding(),
            replaced.data.freed_bytes()
        );
        Ok(())
    }
}
