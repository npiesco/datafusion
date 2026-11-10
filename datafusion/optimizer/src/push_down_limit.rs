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

//! [`PushDownLimit`] pushes `LIMIT` earlier in the query plan

use std::cmp::min;
use std::sync::Arc;

use crate::funded::{Freed, cloned_plan, freed, unwrapped};
use crate::optimizer::ApplyOrder;
use crate::{OptimizerConfig, OptimizerRule};

use datafusion_common::Result;
use datafusion_common::funding::Funding;
use datafusion_common::tree_node::Transformed;
use datafusion_common::utils::combine_limit;
use datafusion_expr::logical_plan::{Join, JoinType, Limit, LogicalPlan};
use datafusion_expr::{Expr, FetchType, SkipType, lit};

/// Optimization rule that tries to push down `LIMIT`.
//. It will push down through projection, limits (taking the smaller limit)
#[derive(Default, Debug)]
pub struct PushDownLimit {}

impl PushDownLimit {
    #[expect(missing_docs)]
    pub fn new() -> Self {
        Self {}
    }
}

/// Push down Limit.
impl OptimizerRule for PushDownLimit {
    fn supports_rewrite(&self) -> bool {
        true
    }

    fn rewrite(
        &self,
        plan: LogicalPlan,
        config: &dyn OptimizerConfig,
    ) -> Result<Transformed<LogicalPlan>> {
        push_down_limit(&config.funding(), plan)
    }

    fn name(&self) -> &str {
        "push_down_limit"
    }

    fn apply_order(&self) -> Option<ApplyOrder> {
        Some(ApplyOrder::TopDown)
    }
}

/// [`PushDownLimit`] over `plan`: each node made charged to `funding`, each freed given back.
fn push_down_limit(
    funding: &Funding,
    plan: LogicalPlan,
) -> Result<Transformed<LogicalPlan>> {
    let LogicalPlan::Limit(mut limit) = plan else {
        return Ok(Transformed::no(plan));
    };

    // Currently only rewrite if skip and fetch are both literals
    let SkipType::Literal(skip) = limit.get_skip_type()? else {
        return Ok(Transformed::no(LogicalPlan::Limit(limit)));
    };
    let FetchType::Literal(fetch) = limit.get_fetch_type()? else {
        return Ok(Transformed::no(LogicalPlan::Limit(limit)));
    };

    // Merge the Parent Limit and the Child Limit.
    if let LogicalPlan::Limit(child) = limit.input.as_ref() {
        let SkipType::Literal(child_skip) = child.get_skip_type()? else {
            return Ok(Transformed::no(LogicalPlan::Limit(limit)));
        };
        let FetchType::Literal(child_fetch) = child.get_fetch_type()? else {
            return Ok(Transformed::no(LogicalPlan::Limit(limit)));
        };
        let child_input = Arc::clone(&child.input);

        let (skip, fetch) = combine_limit(skip, fetch, child_skip, child_fetch);
        let skip = Some(bound(funding, limit.skip.take(), skip)?);
        let fetch = match fetch {
            Some(fetch) => Some(bound(funding, limit.fetch.take(), fetch)?),
            None => {
                freed(funding, limit.fetch.take());
                None
            }
        };
        // The child limit, its own input now shared, is freed.
        freed(funding, limit.input);
        let plan = LogicalPlan::Limit(Limit {
            skip,
            fetch,
            input: child_input,
        });

        // recursively reapply the rule on the new plan
        return push_down_limit(funding, plan);
    }

    // no fetch to push, so return the original plan
    let Some(fetch) = fetch else {
        return Ok(Transformed::no(LogicalPlan::Limit(limit)));
    };
    let bounds = (limit.skip, limit.fetch);

    match unwrapped(funding, limit.input)? {
        LogicalPlan::TableScan(mut scan) => {
            let rows_needed = if fetch != 0 { fetch + skip } else { 0 };
            let new_fetch = scan
                .fetch
                .map(|x| min(x, rows_needed))
                .or(Some(rows_needed));
            if new_fetch == scan.fetch {
                original_limit(funding, bounds, skip, fetch, LogicalPlan::TableScan(scan))
            } else {
                // push limit into the table scan itself
                scan.fetch = new_fetch;
                transformed_limit(
                    funding,
                    bounds,
                    skip,
                    fetch,
                    LogicalPlan::TableScan(scan),
                )
            }
        }
        LogicalPlan::Union(mut union) => {
            // push limits to each input of the union
            for input in union.inputs.iter_mut() {
                *input = make_arc_limit(funding, 0, fetch + skip, Arc::clone(input))?;
            }
            transformed_limit(funding, bounds, skip, fetch, LogicalPlan::Union(union))
        }

        LogicalPlan::Join(join) => {
            let join = push_down_join(funding, join, fetch + skip)?;
            let input = funding.arc(LogicalPlan::Join(join.data))?;
            let limit = make_limit(funding, bounds, skip, fetch, input)?;
            Ok(Transformed::new_transformed(limit, join.transformed))
        }

        LogicalPlan::Sort(mut sort) => {
            let new_fetch = {
                let sort_fetch = skip + fetch;
                Some(sort.fetch.map(|f| f.min(sort_fetch)).unwrap_or(sort_fetch))
            };
            if new_fetch == sort.fetch {
                if skip > 0 {
                    original_limit(funding, bounds, skip, fetch, LogicalPlan::Sort(sort))
                } else {
                    freed(funding, bounds);
                    Ok(Transformed::yes(LogicalPlan::Sort(sort)))
                }
            } else {
                sort.fetch = new_fetch;
                Ok(Transformed::yes(LogicalPlan::Limit(Limit {
                    skip: bounds.0,
                    fetch: bounds.1,
                    input: funding.arc(LogicalPlan::Sort(sort))?,
                })))
            }
        }
        LogicalPlan::Projection(mut proj) => {
            // commute
            let new_limit = LogicalPlan::Limit(Limit {
                skip: bounds.0,
                fetch: bounds.1,
                input: Arc::clone(&proj.input),
            });
            proj.input = funding.arc(new_limit)?;
            Ok(Transformed::yes(LogicalPlan::Projection(proj)))
        }
        LogicalPlan::SubqueryAlias(mut subquery_alias) => {
            // commute
            let new_limit = LogicalPlan::Limit(Limit {
                skip: bounds.0,
                fetch: bounds.1,
                input: Arc::clone(&subquery_alias.input),
            });
            subquery_alias.input = funding.arc(new_limit)?;
            Ok(Transformed::yes(LogicalPlan::SubqueryAlias(subquery_alias)))
        }
        LogicalPlan::Extension(extension_plan)
            if extension_plan.node.supports_limit_pushdown() =>
        {
            let children = extension_plan.node.inputs();
            let mut new_children = funding.vec_with_capacity(children.len())?;
            for child in children {
                let input = funding.arc(cloned_plan(funding, child)?)?;
                // Room for each is charged: pushing allocates nothing.
                new_children.push(make_limit(
                    funding,
                    (None, None),
                    0,
                    fetch + skip,
                    input,
                )?);
            }

            // Create a new extension node with updated inputs
            let child_plan = LogicalPlan::Extension(extension_plan);
            let exprs = child_plan.expressions();
            funding.charge(exprs.freed_bytes())?;
            let new_extension = child_plan.with_new_exprs(exprs, new_children)?;
            freed(funding, child_plan);

            transformed_limit(funding, bounds, skip, fetch, new_extension)
        }
        input => original_limit(funding, bounds, skip, fetch, input),
    }
}

/// A limit's skip and fetch, each in its box if it has one.
type Bounds = (Option<Box<Expr>>, Option<Box<Expr>>);

/// `value` as a literal, written into `slot`'s box if there is one (what it held freed and
/// given back), else boxed, the box charged first.
fn bound(funding: &Funding, slot: Option<Box<Expr>>, value: usize) -> Result<Box<Expr>> {
    let value = lit(value as i64);
    match slot {
        Some(mut boxed) => {
            freed(funding, std::mem::replace(boxed.as_mut(), value));
            Ok(boxed)
        }
        None => funding.boxed(value),
    }
}

/// Wrap the input plan with a limit node, its literals written into the boxes of `bounds`
/// where it has them, any others charged to `funding`
///
/// Original:
/// ```text
/// input
/// ```
///
/// Return
/// ```text
/// Limit: skip=skip, fetch=fetch
///  input
/// ```
fn make_limit(
    funding: &Funding,
    bounds: Bounds,
    skip: usize,
    fetch: usize,
    input: Arc<LogicalPlan>,
) -> Result<LogicalPlan> {
    Ok(LogicalPlan::Limit(Limit {
        skip: Some(bound(funding, bounds.0, skip)?),
        fetch: Some(bound(funding, bounds.1, fetch)?),
        input,
    }))
}

/// Wrap the input plan with a new limit node, in a block charged to `funding`
fn make_arc_limit(
    funding: &Funding,
    skip: usize,
    fetch: usize,
    input: Arc<LogicalPlan>,
) -> Result<Arc<LogicalPlan>> {
    funding.arc(make_limit(funding, (None, None), skip, fetch, input)?)
}

/// Returns the original limit (non transformed)
fn original_limit(
    funding: &Funding,
    bounds: Bounds,
    skip: usize,
    fetch: usize,
    input: LogicalPlan,
) -> Result<Transformed<LogicalPlan>> {
    let input = funding.arc(input)?;
    Ok(Transformed::no(make_limit(
        funding, bounds, skip, fetch, input,
    )?))
}

/// Returns the a transformed limit
fn transformed_limit(
    funding: &Funding,
    bounds: Bounds,
    skip: usize,
    fetch: usize,
    input: LogicalPlan,
) -> Result<Transformed<LogicalPlan>> {
    let input = funding.arc(input)?;
    Ok(Transformed::yes(make_limit(
        funding, bounds, skip, fetch, input,
    )?))
}

/// Adds a limit to the inputs of a join, if possible
fn push_down_join(
    funding: &Funding,
    mut join: Join,
    limit: usize,
) -> Result<Transformed<Join>> {
    use JoinType::*;

    // Cross join is the special case of inner join where there is no join condition. see [LogicalPlanBuilder::cross_join]
    fn is_cross_join(join: &Join) -> bool {
        join.join_type == Inner && join.on.is_empty() && join.filter.is_none()
    }

    let (left_limit, right_limit) = if is_cross_join(&join) {
        (Some(limit), Some(limit))
    } else {
        match join.join_type {
            Left => (Some(limit), None),
            Right => (None, Some(limit)),
            _ => (None, None),
        }
    };

    if left_limit.is_none() && right_limit.is_none() {
        return Ok(Transformed::no(join));
    }
    if let Some(limit) = left_limit {
        join.left = make_arc_limit(funding, 0, limit, join.left)?;
    }
    if let Some(limit) = right_limit {
        join.right = make_arc_limit(funding, 0, limit, join.right)?;
    }
    Ok(Transformed::yes(join))
}

#[cfg(test)]
mod test {
    use std::cmp::Ordering;
    use std::fmt::{Debug, Formatter};
    use std::vec;

    use super::*;
    use crate::assert_optimized_plan_eq_snapshot;
    use crate::test::*;

    use crate::OptimizerContext;
    use datafusion_common::DFSchemaRef;
    use datafusion_expr::{
        Expr, Extension, UserDefinedLogicalNodeCore, col, exists,
        logical_plan::builder::LogicalPlanBuilder,
    };
    use datafusion_functions_aggregate::expr_fn::max;

    macro_rules! assert_optimized_plan_equal {
        (
            $plan:expr,
            @ $expected:literal $(,)?
        ) => {{
            let optimizer_ctx = OptimizerContext::new().with_max_passes(1);
            let rules: Vec<Arc<dyn crate::OptimizerRule + Send + Sync>> = vec![Arc::new(PushDownLimit::new())];
            assert_optimized_plan_eq_snapshot!(
                optimizer_ctx,
                rules,
                $plan,
                @ $expected,
            )
        }};
    }

    #[derive(Debug, PartialEq, Eq, Hash)]
    pub struct NoopPlan {
        input: Vec<LogicalPlan>,
        schema: DFSchemaRef,
    }

    // Manual implementation needed because of `schema` field. Comparison excludes this field.
    impl PartialOrd for NoopPlan {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            self.input
                .partial_cmp(&other.input)
                // TODO (https://github.com/apache/datafusion/issues/17477) avoid recomparing all fields
                .filter(|cmp| *cmp != Ordering::Equal || self == other)
        }
    }

    impl UserDefinedLogicalNodeCore for NoopPlan {
        fn name(&self) -> &str {
            "NoopPlan"
        }

        fn inputs(&self) -> Vec<&LogicalPlan> {
            self.input.iter().collect()
        }

        fn schema(&self) -> &DFSchemaRef {
            &self.schema
        }

        fn expressions(&self) -> Vec<Expr> {
            self.input
                .iter()
                .flat_map(|child| child.expressions())
                .collect()
        }

        fn fmt_for_explain(&self, f: &mut Formatter) -> std::fmt::Result {
            write!(f, "NoopPlan")
        }

        fn with_exprs_and_inputs(
            &self,
            _exprs: Vec<Expr>,
            inputs: Vec<LogicalPlan>,
        ) -> Result<Self> {
            Ok(Self {
                input: inputs,
                schema: Arc::clone(&self.schema),
            })
        }

        fn supports_limit_pushdown(&self) -> bool {
            true // Allow limit push-down
        }
    }

    #[derive(Debug, PartialEq, Eq, Hash)]
    struct NoLimitNoopPlan {
        input: Vec<LogicalPlan>,
        schema: DFSchemaRef,
    }

    // Manual implementation needed because of `schema` field. Comparison excludes this field.
    impl PartialOrd for NoLimitNoopPlan {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            self.input
                .partial_cmp(&other.input)
                // TODO (https://github.com/apache/datafusion/issues/17477) avoid recomparing all fields
                .filter(|cmp| *cmp != Ordering::Equal || self == other)
        }
    }

    impl UserDefinedLogicalNodeCore for NoLimitNoopPlan {
        fn name(&self) -> &str {
            "NoLimitNoopPlan"
        }

        fn inputs(&self) -> Vec<&LogicalPlan> {
            self.input.iter().collect()
        }

        fn schema(&self) -> &DFSchemaRef {
            &self.schema
        }

        fn expressions(&self) -> Vec<Expr> {
            self.input
                .iter()
                .flat_map(|child| child.expressions())
                .collect()
        }

        fn fmt_for_explain(&self, f: &mut Formatter) -> std::fmt::Result {
            write!(f, "NoLimitNoopPlan")
        }

        fn with_exprs_and_inputs(
            &self,
            _exprs: Vec<Expr>,
            inputs: Vec<LogicalPlan>,
        ) -> Result<Self> {
            Ok(Self {
                input: inputs,
                schema: Arc::clone(&self.schema),
            })
        }

        fn supports_limit_pushdown(&self) -> bool {
            false // Disallow limit push-down by default
        }
    }
    #[test]
    fn limit_pushdown_basic() -> Result<()> {
        let table_scan = test_table_scan()?;
        let noop_plan = LogicalPlan::Extension(Extension {
            node: Arc::new(NoopPlan {
                input: vec![table_scan.clone()],
                schema: Arc::clone(table_scan.schema()),
            }),
        });

        let plan = LogicalPlanBuilder::from(noop_plan)
            .limit(0, Some(1000))?
            .build()?;

        assert_optimized_plan_equal!(
            plan,
            @r"
        Limit: skip=0, fetch=1000
          NoopPlan
            Limit: skip=0, fetch=1000
              TableScan: test, fetch=1000
        "
        )
    }

    #[test]
    fn limit_pushdown_with_skip() -> Result<()> {
        let table_scan = test_table_scan()?;
        let noop_plan = LogicalPlan::Extension(Extension {
            node: Arc::new(NoopPlan {
                input: vec![table_scan.clone()],
                schema: Arc::clone(table_scan.schema()),
            }),
        });

        let plan = LogicalPlanBuilder::from(noop_plan)
            .limit(10, Some(1000))?
            .build()?;

        assert_optimized_plan_equal!(
            plan,
            @r"
        Limit: skip=10, fetch=1000
          NoopPlan
            Limit: skip=0, fetch=1010
              TableScan: test, fetch=1010
        "
        )
    }

    #[test]
    fn limit_pushdown_multiple_limits() -> Result<()> {
        let table_scan = test_table_scan()?;
        let noop_plan = LogicalPlan::Extension(Extension {
            node: Arc::new(NoopPlan {
                input: vec![table_scan.clone()],
                schema: Arc::clone(table_scan.schema()),
            }),
        });

        let plan = LogicalPlanBuilder::from(noop_plan)
            .limit(10, Some(1000))?
            .limit(20, Some(500))?
            .build()?;

        assert_optimized_plan_equal!(
            plan,
            @r"
        Limit: skip=30, fetch=500
          NoopPlan
            Limit: skip=0, fetch=530
              TableScan: test, fetch=530
        "
        )
    }

    #[test]
    fn limit_pushdown_multiple_inputs() -> Result<()> {
        let table_scan = test_table_scan()?;
        let noop_plan = LogicalPlan::Extension(Extension {
            node: Arc::new(NoopPlan {
                input: vec![table_scan.clone(), table_scan.clone()],
                schema: Arc::clone(table_scan.schema()),
            }),
        });

        let plan = LogicalPlanBuilder::from(noop_plan)
            .limit(0, Some(1000))?
            .build()?;

        assert_optimized_plan_equal!(
            plan,
            @r"
        Limit: skip=0, fetch=1000
          NoopPlan
            Limit: skip=0, fetch=1000
              TableScan: test, fetch=1000
            Limit: skip=0, fetch=1000
              TableScan: test, fetch=1000
        "
        )
    }

    #[test]
    fn limit_pushdown_disallowed_noop_plan() -> Result<()> {
        let table_scan = test_table_scan()?;
        let no_limit_noop_plan = LogicalPlan::Extension(Extension {
            node: Arc::new(NoLimitNoopPlan {
                input: vec![table_scan.clone()],
                schema: Arc::clone(table_scan.schema()),
            }),
        });

        let plan = LogicalPlanBuilder::from(no_limit_noop_plan)
            .limit(0, Some(1000))?
            .build()?;

        assert_optimized_plan_equal!(
            plan,
            @r"
        Limit: skip=0, fetch=1000
          NoLimitNoopPlan
            TableScan: test
        "
        )
    }

    #[test]
    fn limit_pushdown_projection_table_provider() -> Result<()> {
        let table_scan = test_table_scan()?;

        let plan = LogicalPlanBuilder::from(table_scan)
            .project(vec![col("a")])?
            .limit(0, Some(1000))?
            .build()?;

        // Should push the limit down to table provider
        // When it has a select
        assert_optimized_plan_equal!(
            plan,
            @r"
        Projection: test.a
          Limit: skip=0, fetch=1000
            TableScan: test, fetch=1000
        "
        )
    }

    #[test]
    fn limit_push_down_take_smaller_limit() -> Result<()> {
        let table_scan = test_table_scan()?;

        let plan = LogicalPlanBuilder::from(table_scan)
            .limit(0, Some(1000))?
            .limit(0, Some(10))?
            .build()?;

        // Should push down the smallest limit
        // Towards table scan
        // This rule doesn't replace multiple limits
        assert_optimized_plan_equal!(
            plan,
            @r"
        Limit: skip=0, fetch=10
          TableScan: test, fetch=10
        "
        )
    }

    #[test]
    fn limit_doesnt_push_down_aggregation() -> Result<()> {
        let table_scan = test_table_scan()?;

        let plan = LogicalPlanBuilder::from(table_scan)
            .aggregate(vec![col("a")], vec![max(col("b"))])?
            .limit(0, Some(1000))?
            .build()?;

        // Limit should *not* push down aggregate node
        assert_optimized_plan_equal!(
            plan,
            @r"
        Limit: skip=0, fetch=1000
          Aggregate: groupBy=[[test.a]], aggr=[[max(test.b)]]
            TableScan: test
        "
        )
    }

    #[test]
    fn limit_should_push_down_union() -> Result<()> {
        let table_scan = test_table_scan()?;

        let plan = LogicalPlanBuilder::from(table_scan.clone())
            .union(LogicalPlanBuilder::from(table_scan).build()?)?
            .limit(0, Some(1000))?
            .build()?;

        // Limit should push down through union
        assert_optimized_plan_equal!(
            plan,
            @r"
        Limit: skip=0, fetch=1000
          Union
            Limit: skip=0, fetch=1000
              TableScan: test, fetch=1000
            Limit: skip=0, fetch=1000
              TableScan: test, fetch=1000
        "
        )
    }

    #[test]
    fn limit_push_down_sort() -> Result<()> {
        let table_scan = test_table_scan()?;

        let plan = LogicalPlanBuilder::from(table_scan)
            .sort_by(vec![col("a")])?
            .limit(0, Some(10))?
            .build()?;

        // Should push down limit to sort
        assert_optimized_plan_equal!(
            plan,
            @r"
        Limit: skip=0, fetch=10
          Sort: test.a ASC NULLS LAST, fetch=10
            TableScan: test
        "
        )
    }

    #[test]
    fn limit_push_down_sort_skip() -> Result<()> {
        let table_scan = test_table_scan()?;

        let plan = LogicalPlanBuilder::from(table_scan)
            .sort_by(vec![col("a")])?
            .limit(5, Some(10))?
            .build()?;

        // Should push down limit to sort
        assert_optimized_plan_equal!(
            plan,
            @r"
        Limit: skip=5, fetch=10
          Sort: test.a ASC NULLS LAST, fetch=15
            TableScan: test
        "
        )
    }

    #[test]
    fn multi_stage_limit_recursive_to_deeper_limit() -> Result<()> {
        let table_scan = test_table_scan()?;

        let plan = LogicalPlanBuilder::from(table_scan)
            .limit(0, Some(1000))?
            .aggregate(vec![col("a")], vec![max(col("b"))])?
            .limit(0, Some(10))?
            .build()?;

        // Limit should use deeper LIMIT 1000, but Limit 10 shouldn't push down aggregation
        assert_optimized_plan_equal!(
            plan,
            @r"
        Limit: skip=0, fetch=10
          Aggregate: groupBy=[[test.a]], aggr=[[max(test.b)]]
            Limit: skip=0, fetch=1000
              TableScan: test, fetch=1000
        "
        )
    }

    #[test]
    fn limit_pushdown_should_not_pushdown_limit_with_offset_only() -> Result<()> {
        let table_scan = test_table_scan()?;
        let plan = LogicalPlanBuilder::from(table_scan)
            .limit(10, None)?
            .build()?;

        // Should not push any limit down to table provider
        // When it has a select
        assert_optimized_plan_equal!(
            plan,
            @r"
        Limit: skip=10, fetch=None
          TableScan: test
        "
        )
    }

    #[test]
    fn limit_pushdown_with_offset_projection_table_provider() -> Result<()> {
        let table_scan = test_table_scan()?;

        let plan = LogicalPlanBuilder::from(table_scan)
            .project(vec![col("a")])?
            .limit(10, Some(1000))?
            .build()?;

        // Should push the limit down to table provider
        // When it has a select
        assert_optimized_plan_equal!(
            plan,
            @r"
        Projection: test.a
          Limit: skip=10, fetch=1000
            TableScan: test, fetch=1010
        "
        )
    }

    #[test]
    fn limit_pushdown_with_offset_after_limit() -> Result<()> {
        let table_scan = test_table_scan()?;

        let plan = LogicalPlanBuilder::from(table_scan)
            .project(vec![col("a")])?
            .limit(0, Some(1000))?
            .limit(10, None)?
            .build()?;

        assert_optimized_plan_equal!(
            plan,
            @r"
        Projection: test.a
          Limit: skip=10, fetch=990
            TableScan: test, fetch=1000
        "
        )
    }

    #[test]
    fn limit_pushdown_with_limit_after_offset() -> Result<()> {
        let table_scan = test_table_scan()?;

        let plan = LogicalPlanBuilder::from(table_scan)
            .project(vec![col("a")])?
            .limit(10, None)?
            .limit(0, Some(1000))?
            .build()?;

        assert_optimized_plan_equal!(
            plan,
            @r"
        Projection: test.a
          Limit: skip=10, fetch=1000
            TableScan: test, fetch=1010
        "
        )
    }

    #[test]
    fn limit_push_down_with_offset_take_smaller_limit() -> Result<()> {
        let table_scan = test_table_scan()?;

        let plan = LogicalPlanBuilder::from(table_scan)
            .limit(10, None)?
            .limit(0, Some(1000))?
            .limit(0, Some(10))?
            .build()?;

        assert_optimized_plan_equal!(
            plan,
            @r"
        Limit: skip=10, fetch=10
          TableScan: test, fetch=20
        "
        )
    }

    #[test]
    fn limit_doesnt_push_down_with_offset_aggregation() -> Result<()> {
        let table_scan = test_table_scan()?;

        let plan = LogicalPlanBuilder::from(table_scan)
            .aggregate(vec![col("a")], vec![max(col("b"))])?
            .limit(10, Some(1000))?
            .build()?;

        // Limit should *not* push down aggregate node
        assert_optimized_plan_equal!(
            plan,
            @r"
        Limit: skip=10, fetch=1000
          Aggregate: groupBy=[[test.a]], aggr=[[max(test.b)]]
            TableScan: test
        "
        )
    }

    #[test]
    fn limit_should_push_down_with_offset_union() -> Result<()> {
        let table_scan = test_table_scan()?;

        let plan = LogicalPlanBuilder::from(table_scan.clone())
            .union(LogicalPlanBuilder::from(table_scan).build()?)?
            .limit(10, Some(1000))?
            .build()?;

        // Limit should push down through union
        assert_optimized_plan_equal!(
            plan,
            @r"
        Limit: skip=10, fetch=1000
          Union
            Limit: skip=0, fetch=1010
              TableScan: test, fetch=1010
            Limit: skip=0, fetch=1010
              TableScan: test, fetch=1010
        "
        )
    }

    #[test]
    fn limit_offset_should_not_push_down_with_offset_join() -> Result<()> {
        let table_scan_1 = test_table_scan()?;
        let table_scan_2 = test_table_scan_with_name("test2")?;

        let plan = LogicalPlanBuilder::from(table_scan_1)
            .join(
                LogicalPlanBuilder::from(table_scan_2).build()?,
                JoinType::Inner,
                (vec!["a"], vec!["a"]),
                None,
            )?
            .limit(10, Some(1000))?
            .build()?;

        // Limit pushdown Not supported in Join
        assert_optimized_plan_equal!(
            plan,
            @r"
        Limit: skip=10, fetch=1000
          Inner Join: test.a = test2.a
            TableScan: test
            TableScan: test2
        "
        )
    }

    #[test]
    fn offset_limit_should_not_push_down_with_offset_join() -> Result<()> {
        let table_scan_1 = test_table_scan()?;
        let table_scan_2 = test_table_scan_with_name("test2")?;

        let plan = LogicalPlanBuilder::from(table_scan_1)
            .join(
                LogicalPlanBuilder::from(table_scan_2).build()?,
                JoinType::Inner,
                (vec!["a"], vec!["a"]),
                None,
            )?
            .limit(10, Some(1000))?
            .build()?;

        // Limit pushdown Not supported in Join
        assert_optimized_plan_equal!(
            plan,
            @r"
        Limit: skip=10, fetch=1000
          Inner Join: test.a = test2.a
            TableScan: test
            TableScan: test2
        "
        )
    }

    #[test]
    fn limit_offset_should_not_push_down_with_offset_sub_query() -> Result<()> {
        let table_scan_1 = test_table_scan_with_name("test1")?;
        let table_scan_2 = test_table_scan_with_name("test2")?;

        let subquery = LogicalPlanBuilder::from(table_scan_1)
            .project(vec![col("a")])?
            .filter(col("a").eq(col("test1.a")))?
            .build()?;

        let outer_query = LogicalPlanBuilder::from(table_scan_2)
            .project(vec![col("a")])?
            .filter(exists(Arc::new(subquery)))?
            .limit(10, Some(100))?
            .build()?;

        // Limit pushdown Not supported in sub_query
        assert_optimized_plan_equal!(
            outer_query,
            @r"
        Limit: skip=10, fetch=100
          Filter: EXISTS (<subquery>)
            Subquery:
              Filter: test1.a = test1.a
                Projection: test1.a
                  TableScan: test1
            Projection: test2.a
              TableScan: test2
        "
        )
    }

    #[test]
    fn offset_limit_should_not_push_down_with_offset_sub_query() -> Result<()> {
        let table_scan_1 = test_table_scan_with_name("test1")?;
        let table_scan_2 = test_table_scan_with_name("test2")?;

        let subquery = LogicalPlanBuilder::from(table_scan_1)
            .project(vec![col("a")])?
            .filter(col("a").eq(col("test1.a")))?
            .build()?;

        let outer_query = LogicalPlanBuilder::from(table_scan_2)
            .project(vec![col("a")])?
            .filter(exists(Arc::new(subquery)))?
            .limit(10, Some(100))?
            .build()?;

        // Limit pushdown Not supported in sub_query
        assert_optimized_plan_equal!(
            outer_query,
            @r"
        Limit: skip=10, fetch=100
          Filter: EXISTS (<subquery>)
            Subquery:
              Filter: test1.a = test1.a
                Projection: test1.a
                  TableScan: test1
            Projection: test2.a
              TableScan: test2
        "
        )
    }

    #[test]
    fn limit_should_push_down_left_outer_join_with_offset() -> Result<()> {
        let table_scan_1 = test_table_scan()?;
        let table_scan_2 = test_table_scan_with_name("test2")?;

        let plan = LogicalPlanBuilder::from(table_scan_1)
            .join(
                LogicalPlanBuilder::from(table_scan_2).build()?,
                JoinType::Left,
                (vec!["a"], vec!["a"]),
                None,
            )?
            .limit(10, Some(1000))?
            .build()?;

        // Limit pushdown Not supported in Join
        assert_optimized_plan_equal!(
            plan,
            @r"
        Limit: skip=10, fetch=1000
          Left Join: test.a = test2.a
            Limit: skip=0, fetch=1010
              TableScan: test, fetch=1010
            TableScan: test2
        "
        )
    }

    #[test]
    fn limit_should_push_down_right_outer_join() -> Result<()> {
        let table_scan_1 = test_table_scan()?;
        let table_scan_2 = test_table_scan_with_name("test2")?;

        let plan = LogicalPlanBuilder::from(table_scan_1)
            .join(
                LogicalPlanBuilder::from(table_scan_2).build()?,
                JoinType::Right,
                (vec!["a"], vec!["a"]),
                None,
            )?
            .limit(0, Some(1000))?
            .build()?;

        // Limit pushdown Not supported in Join
        assert_optimized_plan_equal!(
            plan,
            @r"
        Limit: skip=0, fetch=1000
          Right Join: test.a = test2.a
            TableScan: test
            Limit: skip=0, fetch=1000
              TableScan: test2, fetch=1000
        "
        )
    }

    #[test]
    fn limit_should_push_down_right_outer_join_with_offset() -> Result<()> {
        let table_scan_1 = test_table_scan()?;
        let table_scan_2 = test_table_scan_with_name("test2")?;

        let plan = LogicalPlanBuilder::from(table_scan_1)
            .join(
                LogicalPlanBuilder::from(table_scan_2).build()?,
                JoinType::Right,
                (vec!["a"], vec!["a"]),
                None,
            )?
            .limit(10, Some(1000))?
            .build()?;

        // Limit pushdown with offset supported in right outer join
        assert_optimized_plan_equal!(
            plan,
            @r"
        Limit: skip=10, fetch=1000
          Right Join: test.a = test2.a
            TableScan: test
            Limit: skip=0, fetch=1010
              TableScan: test2, fetch=1010
        "
        )
    }

    #[test]
    fn limit_push_down_cross_join() -> Result<()> {
        let table_scan_1 = test_table_scan()?;
        let table_scan_2 = test_table_scan_with_name("test2")?;

        let plan = LogicalPlanBuilder::from(table_scan_1)
            .cross_join(LogicalPlanBuilder::from(table_scan_2).build()?)?
            .limit(0, Some(1000))?
            .build()?;

        assert_optimized_plan_equal!(
            plan,
            @r"
        Limit: skip=0, fetch=1000
          Cross Join:
            Limit: skip=0, fetch=1000
              TableScan: test, fetch=1000
            Limit: skip=0, fetch=1000
              TableScan: test2, fetch=1000
        "
        )
    }

    #[test]
    fn skip_limit_push_down_cross_join() -> Result<()> {
        let table_scan_1 = test_table_scan()?;
        let table_scan_2 = test_table_scan_with_name("test2")?;

        let plan = LogicalPlanBuilder::from(table_scan_1)
            .cross_join(LogicalPlanBuilder::from(table_scan_2).build()?)?
            .limit(1000, Some(1000))?
            .build()?;

        assert_optimized_plan_equal!(
            plan,
            @r"
        Limit: skip=1000, fetch=1000
          Cross Join:
            Limit: skip=0, fetch=2000
              TableScan: test, fetch=2000
            Limit: skip=0, fetch=2000
              TableScan: test2, fetch=2000
        "
        )
    }

    #[test]
    fn merge_limit_result_empty() -> Result<()> {
        let scan = test_table_scan()?;

        let plan = LogicalPlanBuilder::from(scan)
            .limit(0, Some(1000))?
            .limit(1000, None)?
            .build()?;

        assert_optimized_plan_equal!(
            plan,
            @r"
        Limit: skip=1000, fetch=0
          TableScan: test, fetch=0
        "
        )
    }

    #[test]
    fn skip_great_than_fetch() -> Result<()> {
        let scan = test_table_scan()?;

        let plan = LogicalPlanBuilder::from(scan)
            .limit(0, Some(1))?
            .limit(1000, None)?
            .build()?;

        assert_optimized_plan_equal!(
            plan,
            @r"
        Limit: skip=1000, fetch=0
          TableScan: test, fetch=0
        "
        )
    }

    #[test]
    fn push_down_subquery_alias() -> Result<()> {
        let scan = test_table_scan()?;

        let plan = LogicalPlanBuilder::from(scan)
            .alias("a")?
            .limit(0, Some(1))?
            .limit(1000, None)?
            .build()?;

        assert_optimized_plan_equal!(
            plan,
            @r"
        SubqueryAlias: a
          Limit: skip=1000, fetch=0
            TableScan: test, fetch=0
        "
        )
    }

    /// Funding without a limit.
    struct Unlimited;

    impl datafusion_common::funding::PlanningFunding for Unlimited {
        fn charge(&self, _bytes: usize) -> bool {
            true
        }

        fn discharge(&self, _bytes: usize) {}
    }

    #[test]
    fn merged_limits_hold_what_was_charged() -> Result<()> {
        let plan = LogicalPlanBuilder::from(test_table_scan()?)
            .limit(0, Some(1000))?
            .limit(0, Some(10))?
            .build()?;
        let mut options = datafusion_common::config::ConfigOptions::default();
        options.funding = Funding::new(Arc::new(Unlimited));
        // Planned through the same funding.
        options.funding.charge(plan.freed_bytes())?;
        let config = OptimizerContext::new_with_config_options(Arc::new(options));

        let pushed = PushDownLimit::new().rewrite(plan, &config)?;

        assert!(pushed.transformed);
        // The child limit is given back, the limit kept over the scan charged.
        assert_eq!(
            config.options().funding.outstanding(),
            pushed.data.freed_bytes()
        );
        Ok(())
    }
}
