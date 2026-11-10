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

//! [`EliminateDuplicatedExpr`] Removes redundant expressions

use crate::funded::{Freed, HashIndex, deduplicated};
use crate::optimizer::ApplyOrder;
use crate::{OptimizerConfig, OptimizerRule};
use datafusion_common::funding::Funding;
use datafusion_common::tree_node::Transformed;
use datafusion_common::{Result, get_required_sort_exprs_indices, internal_err};
use datafusion_expr::logical_plan::LogicalPlan;
use datafusion_expr::{Aggregate, Expr, Sort, SortExpr};

/// Optimization rule that eliminate duplicated expr.
#[derive(Default, Debug)]
pub struct EliminateDuplicatedExpr;

impl EliminateDuplicatedExpr {
    #[expect(missing_docs)]
    pub fn new() -> Self {
        Self {}
    }
}

/// `exprs` kept where `keep` holds for their position, in order; each dropped freed and
/// given back.
fn retained(
    funding: &Funding,
    exprs: &mut Vec<SortExpr>,
    mut keep: impl FnMut(usize) -> bool,
) {
    let mut at = 0;
    exprs.retain(|sort_expr| {
        let kept = keep(at);
        at += 1;
        if !kept {
            funding.discharge(sort_expr.freed_bytes());
        }
        kept
    });
}

/// The sort expressions of `exprs` after the first ordering by the same expression dropped,
/// and then those the input's functional dependencies make redundant: scratch charged to
/// `funding` and given back, each dropped expression freed and given back.
fn deduplicated_sort_exprs(
    funding: &Funding,
    sort: &Sort,
    exprs: &mut Vec<SortExpr>,
) -> Result<()> {
    let mut firsts = funding.held::<Vec<bool>>();
    {
        let mut keys = funding.held::<Vec<&Expr>>();
        keys.reserve_exact(exprs.len())?;
        // Room for each is charged: extending allocates nothing.
        keys.extend(exprs.iter().map(|sort_expr| &sort_expr.expr));
        let index = HashIndex::new(funding, &keys)?;
        firsts.reserve_exact(exprs.len())?;
        firsts.extend((0..exprs.len()).map(|at| index.is_first(at)));
    }
    retained(funding, exprs, |at| firsts[at]);

    let required = get_required_sort_exprs_indices(
        sort.input.schema().as_ref(),
        exprs.iter().map(|sort_expr| sort_expr.expr.schema_name()),
        funding,
    )?;
    if required.len() < exprs.len() {
        let mut next = required.iter().peekable();
        retained(funding, exprs, |at| next.next_if_eq(&&at).is_some());
    }
    Ok(())
}

impl OptimizerRule for EliminateDuplicatedExpr {
    fn apply_order(&self) -> Option<ApplyOrder> {
        Some(ApplyOrder::TopDown)
    }

    fn supports_rewrite(&self) -> bool {
        true
    }

    fn rewrite(
        &self,
        plan: LogicalPlan,
        config: &dyn OptimizerConfig,
    ) -> Result<Transformed<LogicalPlan>> {
        let funding = config.funding();
        match plan {
            LogicalPlan::Sort(mut sort) => {
                let len = sort.expr.len();
                let mut exprs = std::mem::take(&mut sort.expr);
                let deduplicated = deduplicated_sort_exprs(&funding, &sort, &mut exprs);
                sort.expr = exprs;
                deduplicated?;

                if sort.expr.is_empty() {
                    return internal_err!(
                        "FD pruning unexpectedly removed all ORDER BY expressions"
                    );
                }

                Ok(if len != sort.expr.len() {
                    Transformed::yes(LogicalPlan::Sort(sort))
                } else {
                    Transformed::no(LogicalPlan::Sort(sort))
                })
            }
            LogicalPlan::Aggregate(mut agg) => {
                let len = agg.group_expr.len();
                deduplicated(&funding, &mut agg.group_expr)?;
                if len == agg.group_expr.len() {
                    return Ok(Transformed::no(LogicalPlan::Aggregate(agg)));
                }

                Aggregate::try_new(agg.input, agg.group_expr, agg.aggr_expr)
                    .map(|f| Transformed::yes(LogicalPlan::Aggregate(f)))
            }
            _ => Ok(Transformed::no(plan)),
        }
    }
    fn name(&self) -> &str {
        "eliminate_duplicated_expr"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::OptimizerContext;
    use crate::assert_optimized_plan_eq_snapshot;
    use crate::test::*;
    use datafusion_common::config::ConfigOptions;
    use datafusion_common::funding::PlanningFunding;
    use datafusion_expr::{col, logical_plan::builder::LogicalPlanBuilder};
    use std::sync::Arc;

    macro_rules! assert_optimized_plan_equal {
        (
            $plan:expr,
            @ $expected:literal $(,)?
        ) => {{
            let optimizer_ctx = OptimizerContext::new().with_max_passes(1);
            let rules: Vec<Arc<dyn crate::OptimizerRule + Send + Sync>> =
                vec![Arc::new(EliminateDuplicatedExpr::new())];
            assert_optimized_plan_eq_snapshot!(
                optimizer_ctx,
                rules,
                $plan,
                @ $expected,
            )
        }};
    }

    #[test]
    fn eliminate_sort_expr() -> Result<()> {
        let table_scan = test_table_scan().unwrap();
        let plan = LogicalPlanBuilder::from(table_scan)
            .sort_by(vec![col("a"), col("a"), col("b"), col("c")])?
            .limit(5, Some(10))?
            .build()?;

        assert_optimized_plan_equal!(plan, @r"
        Limit: skip=5, fetch=10
          Sort: test.a ASC NULLS LAST, test.b ASC NULLS LAST, test.c ASC NULLS LAST
            TableScan: test
        ")
    }

    #[test]
    fn eliminate_sort_exprs_with_options() -> Result<()> {
        let table_scan = test_table_scan().unwrap();
        let sort_exprs = vec![
            col("a").sort(true, true),
            col("b").sort(true, false),
            col("a").sort(false, false),
            col("b").sort(false, true),
        ];
        let plan = LogicalPlanBuilder::from(table_scan)
            .sort(sort_exprs)?
            .limit(5, Some(10))?
            .build()?;

        assert_optimized_plan_equal!(plan, @r"
        Limit: skip=5, fetch=10
          Sort: test.a ASC NULLS FIRST, test.b ASC NULLS LAST
            TableScan: test
        ")
    }

    /// Funding without a limit.
    struct Unlimited;

    impl PlanningFunding for Unlimited {
        fn charge(&self, _bytes: usize) -> bool {
            true
        }

        fn discharge(&self, _bytes: usize) {}
    }

    #[test]
    fn eliminated_sort_exprs_are_given_back() -> Result<()> {
        let plan = LogicalPlanBuilder::from(test_table_scan()?)
            .sort_by(vec![col("a"), col("a"), col("b")])?
            .build()?;
        let mut options = ConfigOptions::default();
        options.funding = Funding::new(Arc::new(Unlimited));
        // Planned through the same funding.
        options.funding.charge(plan.freed_bytes())?;
        let config = OptimizerContext::new_with_config_options(Arc::new(options));

        let sorted = EliminateDuplicatedExpr::new().rewrite(plan, &config)?;

        assert!(sorted.transformed);
        // The duplicate is given back; the scratch finding it is too.
        assert_eq!(
            config.options().funding.outstanding(),
            sorted.data.freed_bytes()
        );
        Ok(())
    }
}
