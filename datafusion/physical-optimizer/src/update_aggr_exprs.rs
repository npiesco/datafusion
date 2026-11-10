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

//! An optimizer rule that checks ordering requirements of aggregate expressions
//! and modifies the expressions to work more efficiently if possible.

use std::sync::Arc;

use datafusion_common::config::ConfigOptions;
use datafusion_common::funding::{Funding, Held};
use datafusion_common::tree_node::{Transformed, TransformedResult, TreeNode};
use datafusion_common::{Result, plan_datafusion_err};
use datafusion_physical_expr::aggregate::AggregateFunctionExpr;
use datafusion_physical_expr::{EquivalenceProperties, PhysicalSortRequirement};
use datafusion_physical_plan::aggregates::{AggregateExec, AggregateInputMode};
use datafusion_physical_plan::windows::get_ordered_partition_by_indices;
use datafusion_physical_plan::{ExecutionPlan, ExecutionPlanProperties};

use crate::PhysicalOptimizerRule;

/// This optimizer rule checks ordering requirements of aggregate expressions.
///
/// There are 3 kinds of aggregators in terms of ordering requirements:
/// - `AggregateOrderSensitivity::Insensitive`, meaning that ordering is not
///   important.
/// - `AggregateOrderSensitivity::HardRequirement`, meaning that the aggregator
///   requires a specific ordering.
/// - `AggregateOrderSensitivity::Beneficial`, meaning that the aggregator can
///   handle unordered input, but can run more efficiently if its input conforms
///   to a specific ordering.
///
/// This rule analyzes aggregate expressions of type `Beneficial` to see whether
/// their input ordering requirements are satisfied. If this is the case, the
/// aggregators are modified to run in a more efficient mode.
#[derive(Default, Debug)]
pub struct OptimizeAggregateOrder {}

impl OptimizeAggregateOrder {
    #[expect(missing_docs)]
    pub fn new() -> Self {
        Self::default()
    }
}

impl PhysicalOptimizerRule for OptimizeAggregateOrder {
    /// Applies the `OptimizeAggregateOrder` rule to the provided execution plan.
    ///
    /// This function traverses the execution plan tree, identifies `AggregateExec` nodes,
    /// and optimizes their aggregate expressions based on existing input orderings.
    /// If optimizations are applied, it returns a modified execution plan.
    ///
    /// # Arguments
    ///
    /// * `plan` - The root of the execution plan to optimize.
    /// * `_config` - Configuration options (currently unused).
    ///
    /// # Returns
    ///
    /// A `Result` containing the potentially optimized execution plan or an error.
    fn optimize(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        config: &ConfigOptions,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let funding = &config.funding;
        plan.transform_up(|plan| {
            if let Some(aggr_exec) = plan.downcast_ref::<AggregateExec>() {
                // Final stage implementations do not rely on ordering -- those
                // ordering fields may be pruned out by first stage aggregates.
                // Hence, necessary information for proper merge is added during
                // the first stage to the state field, which the final stage uses.
                if aggr_exec.mode().input_mode() == AggregateInputMode::Partial {
                    return Ok(Transformed::no(plan));
                }
                let input = aggr_exec.input();

                let groupby_exprs = aggr_exec.group_expr().input_exprs();
                // If the existing ordering satisfies a prefix of the GROUP BY
                // expressions, prefix requirements with this section. In this
                // case, aggregation will work more efficiently.
                let indices = get_ordered_partition_by_indices(&groupby_exprs, input)?;
                let mut requirement = funding.held::<Vec<PhysicalSortRequirement>>();
                requirement.reserve_exact(indices.len())?;
                for &idx in &indices {
                    requirement.push(PhysicalSortRequirement::new(
                        Arc::clone(&groupby_exprs[idx]),
                        None,
                    ))?;
                }

                let aggr_exprs = try_convert_aggregate_if_better(
                    aggr_exec.aggr_expr(),
                    &requirement,
                    input.equivalence_properties(),
                    funding,
                )?;

                let aggr_exec = aggr_exec.with_new_aggr_exprs(aggr_exprs)?;

                Ok(Transformed::yes(funding.arc(aggr_exec)? as _))
            } else {
                Ok(Transformed::no(plan))
            }
        })
        .data()
    }

    fn name(&self) -> &str {
        "OptimizeAggregateOrder"
    }

    fn schema_check(&self) -> bool {
        true
    }
}

/// Tries to convert each aggregate expression to a potentially more efficient
/// version.
///
/// # Parameters
///
/// * `aggr_exprs` - A vector of `AggregateFunctionExpr` representing the
///   aggregate expressions to be optimized.
/// * `prefix_requirement` - An array slice representing the ordering
///   requirements preceding the aggregate expressions.
/// * `eq_properties` - A reference to the `EquivalenceProperties` object
///   containing ordering information.
///
/// # Returns
///
/// Returns `Ok(converted_aggr_exprs)` if the conversion process completes
/// successfully. Any errors occurring during the conversion process are
/// passed through.
fn try_convert_aggregate_if_better(
    aggr_exprs: &[Arc<AggregateFunctionExpr>],
    prefix_requirement: &[PhysicalSortRequirement],
    eq_properties: &EquivalenceProperties,
    funding: &Funding,
) -> Result<Arc<[Arc<AggregateFunctionExpr>]>> {
    let mut converted = funding.vec_with_capacity(aggr_exprs.len())?;
    for aggr_expr in aggr_exprs {
        let aggr_expr = Arc::clone(aggr_expr);
        let order_bys = aggr_expr.order_bys();
        // If the aggregate expression benefits from input ordering, and
        // there is an actual ordering enabling this, try to update the
        // aggregate expression to benefit from the existing ordering.
        // Otherwise, leave it as is.
        let aggr_expr = if !aggr_expr.order_sensitivity().is_beneficial()
            || order_bys.is_empty()
        {
            aggr_expr
        } else {
            let requirement = |reverse: bool| -> Result<Held<Vec<_>>> {
                let mut requirement = funding.held::<Vec<PhysicalSortRequirement>>();
                requirement.reserve_exact(prefix_requirement.len() + order_bys.len())?;
                requirement.extend(prefix_requirement.iter().cloned());
                requirement.extend(order_bys.iter().map(|e| {
                    PhysicalSortRequirement::from(if reverse {
                        e.reverse()
                    } else {
                        e.clone()
                    })
                }));
                Ok(requirement)
            };
            if eq_properties
                .ordering_satisfy_requirement(requirement(false)?.drain(..))?
            {
                // Existing ordering satisfies the aggregator requirements:
                aggr_expr.with_beneficial_ordering(true)?
            } else if eq_properties
                .ordering_satisfy_requirement(requirement(true)?.drain(..))?
            {
                // Converting to reverse enables more efficient execution
                // given the existing ordering (if possible):
                match aggr_expr.reverse_expr() {
                    Some(reversed) => {
                        funding.arc(reversed)?.with_beneficial_ordering(true)?
                    }
                    None => aggr_expr.with_beneficial_ordering(true)?,
                }
            } else {
                // There is no beneficial ordering present -- aggregation
                // will still work albeit in a less efficient mode.
                aggr_expr.with_beneficial_ordering(false)?
            }
            .map(|expr| funding.arc(expr))
            .transpose()?
            .ok_or_else(|| {
                plan_datafusion_err!(
                    "Expects an aggregate expression that can benefit from input ordering"
                )
            })?
        };
        converted.push(aggr_expr);
    }
    funding.arc_from_vec(converted)
}
