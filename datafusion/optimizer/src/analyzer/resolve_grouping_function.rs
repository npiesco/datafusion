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

//! Analyzed rule to replace TableScan references
//! such as DataFrames and Views and inlines the LogicalPlan.

use std::cmp::Ordering;
use std::sync::Arc;

use crate::analyzer::AnalyzerRule;
use crate::funded::{
    Freed, HashIndex, aliased, binary, column_expr, freed, nullable_cast,
    transform_up_plan_with_subqueries,
};

use arrow::datatypes::DataType;
use datafusion_common::config::ConfigOptions;
use datafusion_common::funding::{Funding, Held};
use datafusion_common::tree_node::{Transformed, TransformedResult};
use datafusion_common::{DFSchema, Result, ScalarValue, internal_err, plan_err};
use datafusion_expr::expr::{AggregateFunction, Alias};
use datafusion_expr::logical_plan::LogicalPlan;
use datafusion_expr::{Aggregate, Expr, GroupingSet, Operator, Projection};
use itertools::Itertools;

/// Replaces grouping aggregation function with value derived from internal grouping id
#[derive(Default, Debug)]
pub struct ResolveGroupingFunction;

impl ResolveGroupingFunction {
    pub fn new() -> Self {
        Self {}
    }
}

impl AnalyzerRule for ResolveGroupingFunction {
    fn analyze(&self, plan: LogicalPlan, config: &ConfigOptions) -> Result<LogicalPlan> {
        let funding = &config.funding;
        transform_up_plan_with_subqueries(funding, plan, |plan| {
            analyze_internal(plan, funding)
        })
        .data()
    }

    fn name(&self) -> &str {
        "resolve_grouping_function"
    }
}

/// The grouping expressions of an aggregate, each found at its index in the internal
/// grouping id, in place of a map from each to its index.
///
/// For more details on how the grouping id bitmap works the documentation for
/// [[Aggregate::INTERNAL_GROUPING_ID]]
struct GroupingBits<'a> {
    exprs: &'a [&'a Expr],
    index: HashIndex<'a, &'a Expr>,
}

impl GroupingBits<'_> {
    /// The index of `expr`'s bit in the internal grouping id.
    fn get(&self, expr: &Expr) -> Option<usize> {
        self.index
            .position(expr)
            .map(|at| self.exprs.len() - 1 - at)
    }

    /// How many distinct grouping expressions there are.
    fn len(&self) -> usize {
        (0..self.exprs.len())
            .filter(|at| self.index.is_first(*at))
            .count()
    }

    /// The distinct grouping expressions.
    fn distinct(&self) -> impl Iterator<Item = &Expr> {
        self.exprs
            .iter()
            .enumerate()
            .filter(|(at, _)| self.index.is_first(*at))
            .map(|(_, expr)| *expr)
    }
}

/// The grouping expressions of `group_expr` ([`grouping_set_to_exprlist`]), in a list charged
/// before it is made.
///
/// [`grouping_set_to_exprlist`]: datafusion_expr::utils::grouping_set_to_exprlist
fn grouping_exprs<'a>(
    funding: &Funding,
    group_expr: &'a [Expr],
) -> Result<Held<Vec<&'a Expr>>> {
    let mut exprs = funding.held::<Vec<&Expr>>();
    match group_expr.first() {
        Some(Expr::GroupingSet(grouping_set)) => {
            if group_expr.len() > 1 {
                return plan_err!(
                    "Invalid group by expressions, GroupingSet must be the only expression"
                );
            }
            match grouping_set {
                GroupingSet::Rollup(list) | GroupingSet::Cube(list) => {
                    exprs.reserve_exact(list.len())?;
                    for expr in list {
                        exprs.push(expr)?;
                    }
                }
                GroupingSet::GroupingSets(groups) => {
                    for expr in groups.iter().flatten() {
                        if !exprs.contains(&expr) {
                            exprs.push(expr)?;
                        }
                    }
                }
            }
        }
        _ => {
            let index = HashIndex::new(funding, group_expr)?;
            for (at, expr) in group_expr.iter().enumerate() {
                if index.is_first(at) {
                    exprs.push(expr)?;
                }
            }
        }
    }
    Ok(exprs)
}

fn replace_grouping_exprs(
    funding: &Funding,
    input: Arc<LogicalPlan>,
    schema: &DFSchema,
    group_expr: Vec<Expr>,
    mut aggr_expr: Vec<Expr>,
) -> Result<LogicalPlan> {
    let is_grouping_set = matches!(group_expr.as_slice(), [Expr::GroupingSet(_)]);
    let exprs = grouping_exprs(funding, &group_expr)?;
    let bits = GroupingBits {
        exprs: &exprs,
        index: HashIndex::new(funding, &exprs)?,
    };
    let grouping_id_len = if is_grouping_set { 1 } else { 0 };
    let group_expr_len = schema.fields().len() - aggr_expr.len() - grouping_id_len;
    let grouping_id_type = is_grouping_set
        .then(|| {
            schema
                .field_with_name(None, Aggregate::INTERNAL_GROUPING_ID)
                .map(|f| f.data_type())
        })
        .transpose()?;
    let mut projection_exprs =
        funding.vec_with_capacity(group_expr_len + aggr_expr.len())?;
    for (qualifier, field) in schema.iter().take(group_expr_len) {
        projection_exprs.push(column_expr(funding, qualifier, field.name())?);
    }
    for (expr, (qualifier, field)) in aggr_expr
        .iter()
        .zip(schema.iter().skip(group_expr_len + grouping_id_len))
    {
        let projected = match expr {
            Expr::AggregateFunction(function) if is_grouping_function(expr) => {
                let grouping_expr =
                    grouping_function_on_id(funding, function, &bits, grouping_id_type)?;
                aliased(funding, grouping_expr, qualifier, field.name())?
            }
            Expr::Alias(Alias { relation, name, .. }) if is_grouping_function(expr) => {
                let function = unwrap_alias_to_grouping_function(expr)?;
                let grouping_expr =
                    grouping_function_on_id(funding, function, &bits, grouping_id_type)?;
                // Preserve the outermost user-provided alias
                aliased(funding, grouping_expr, relation.as_ref(), name)?
            }
            _ => column_expr(funding, qualifier, field.name())?,
        };
        projection_exprs.push(projected);
    }
    drop(bits);
    drop(exprs);
    // The grouping functions are computed by the projection: the aggregate keeps the rest,
    // in its own list, each one dropped freed and given back.
    aggr_expr.retain(|expr| {
        let kept = !is_grouping_function(expr);
        if !kept {
            funding.discharge(expr.freed_bytes());
        }
        kept
    });
    // Recreate aggregate without grouping functions
    let new_aggregate =
        LogicalPlan::Aggregate(Aggregate::try_new(input, group_expr, aggr_expr)?);
    // Create projection with grouping functions calculations
    let projection = LogicalPlan::Projection(Projection::try_new(
        projection_exprs,
        funding.arc(new_aggregate)?,
        funding,
    )?);
    Ok(projection)
}

fn analyze_internal(
    plan: LogicalPlan,
    funding: &Funding,
) -> Result<Transformed<LogicalPlan>> {
    match plan {
        LogicalPlan::Aggregate(Aggregate {
            input,
            group_expr,
            aggr_expr,
            schema,
            ..
        }) if contains_grouping_function(&aggr_expr) => {
            let plan = replace_grouping_exprs(
                funding,
                input,
                schema.as_ref(),
                group_expr,
                aggr_expr,
            )?;
            // The aggregate's schema is replaced by what the rewrite planned.
            freed(funding, schema);
            Ok(Transformed::yes(plan))
        }
        _ => Ok(Transformed::no(plan)),
    }
}

/// Recursively unwrap `Expr::Alias` nodes to reach the inner `AggregateFunction`.
/// Returns an error if the innermost expression is not an `AggregateFunction`,
/// which should not happen if `is_grouping_function` returned true.
fn unwrap_alias_to_grouping_function(expr: &Expr) -> Result<&AggregateFunction> {
    match expr {
        Expr::AggregateFunction(function) => Ok(function),
        Expr::Alias(Alias { expr, .. }) => unwrap_alias_to_grouping_function(expr),
        _ => plan_err!("Expected grouping aggregate function inside alias, got {expr}"),
    }
}

fn is_grouping_function(expr: &Expr) -> bool {
    // TODO: Do something better than name here should grouping be a built
    // in expression?
    match expr {
        Expr::AggregateFunction(AggregateFunction { func, .. }) => {
            func.name() == "grouping"
        }
        Expr::Alias(Alias { expr, .. }) => is_grouping_function(expr),
        _ => false,
    }
}

fn contains_grouping_function(exprs: &[Expr]) -> bool {
    exprs.iter().any(is_grouping_function)
}

/// Validate that the arguments to the grouping function are in the group by clause.
fn validate_args(function: &AggregateFunction, bits: &GroupingBits<'_>) -> Result<()> {
    let expr_not_in_group_by = function
        .params
        .args
        .iter()
        .find(|expr| bits.get(expr).is_none());
    if let Some(expr) = expr_not_in_group_by {
        plan_err!(
            "Argument {} to grouping function is not in grouping columns {}",
            expr,
            bits.distinct().map(|e| e.to_string()).join(", ")
        )
    } else {
        Ok(())
    }
}

fn grouping_function_on_id(
    funding: &Funding,
    function: &AggregateFunction,
    bits: &GroupingBits<'_>,
    // None means not a grouping set (result is always 0).
    grouping_id_type: Option<&DataType>,
) -> Result<Expr> {
    validate_args(function, bits)?;
    let args = &function.params.args;

    // Postgres allows grouping function for group by without grouping sets, the result is then
    // always 0
    let Some(grouping_id_type) = grouping_id_type else {
        return Ok(Expr::Literal(ScalarValue::from(0i32), None));
    };

    // Use the actual __grouping_id column type to size literals correctly. This
    // accounts for duplicate-ordinal bits that `Aggregate::grouping_id_type`
    // packs into the high bits of the column, which a simple count of grouping
    // expressions would miss.
    let literal = |value: usize| match grouping_id_type {
        DataType::UInt8 => Expr::Literal(ScalarValue::from(value as u8), None),
        DataType::UInt16 => Expr::Literal(ScalarValue::from(value as u16), None),
        DataType::UInt32 => Expr::Literal(ScalarValue::from(value as u32), None),
        DataType::UInt64 => Expr::Literal(ScalarValue::from(value as u64), None),
        other => panic!("unexpected __grouping_id type: {other}"),
    };
    let grouping_id_column =
        || column_expr(funding, None, Aggregate::INTERNAL_GROUPING_ID);
    let n = bits.len();
    if args.len() == n
        && args
            .iter()
            .rev()
            .enumerate()
            .all(|(idx, expr)| bits.get(expr) == Some(idx))
    {
        // Mask the ordinal bits above position `n` so only the semantic bitmask is visible.
        // checked_shl returns None when n >= 64 (all bits are semantic), mapping to u64::MAX.
        let semantic_mask: u64 = 1u64.checked_shl(n as u32).map_or(u64::MAX, |m| m - 1);
        let masked_id = binary(
            funding,
            grouping_id_column()?,
            Operator::BitwiseAnd,
            literal(semantic_mask as usize),
        )?;
        return nullable_cast(funding, masked_id, &DataType::Int32);
    }

    let mut combined: Option<Expr> = None;
    for (arg_idx, expr) in args.iter().rev().enumerate() {
        let Some(group_by_idx) = bits.get(expr) else {
            return internal_err!("Grouping sets should contains at least one element");
        };
        let group_by_bit = binary(
            funding,
            grouping_id_column()?,
            Operator::BitwiseAnd,
            literal(1 << group_by_idx),
        )?;
        let bit = match group_by_idx.cmp(&arg_idx) {
            Ordering::Less => binary(
                funding,
                group_by_bit,
                Operator::BitwiseShiftLeft,
                literal(arg_idx - group_by_idx),
            )?,
            Ordering::Greater => binary(
                funding,
                group_by_bit,
                Operator::BitwiseShiftRight,
                literal(group_by_idx - arg_idx),
            )?,
            Ordering::Equal => group_by_bit,
        };
        combined = Some(match combined {
            Some(left) => binary(funding, left, Operator::BitwiseOr, bit)?,
            None => bit,
        });
    }
    match combined {
        Some(expr) => nullable_cast(funding, expr, &DataType::Int32),
        None => internal_err!("Grouping sets should contains at least one element"),
    }
}
