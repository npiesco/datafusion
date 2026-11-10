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

//! Optimizer rule rewriting `SetComparison` subqueries (e.g. `= ANY`,
//! `> ALL`) into boolean expressions built from `EXISTS` subqueries
//! that capture SQL three-valued logic.

use std::mem;
use std::sync::Arc;

use crate::funded::{
    Released, apply_plan_exprs, binary, cloned_plan, column_expr,
    for_each_child_expr_mut, freed, map_plan_exprs_mut, merged_input_schema, merged_with,
    out_ref_exprs, transform_up_plan_with_subqueries,
};
use crate::{OptimizerConfig, OptimizerRule};
use datafusion_common::funding::Funding;
use datafusion_common::tree_node::{Transformed, TreeNode, TreeNodeRecursion};
use datafusion_common::{DFSchema, ExprSchema, Result, ScalarValue, plan_err};
use datafusion_expr::expr::{self, Exists, SetComparison, SetQuantifier};
use datafusion_expr::logical_plan::{Filter, Subquery};
use datafusion_expr::{DmlStatement, Expr, LogicalPlan, WriteOp, lit};

/// Rewrite `SetComparison` expressions to scalar subqueries that return the
/// correct boolean value (including SQL NULL semantics). After this rule
/// runs, later rules such as `ScalarSubqueryToJoin` can decorrelate and
/// remove the remaining subquery.
#[derive(Debug, Default)]
pub struct RewriteSetComparison;

impl RewriteSetComparison {
    /// Create a new `RewriteSetComparison` optimizer rule.
    pub fn new() -> Self {
        Self
    }

    fn rewrite_plan(
        &self,
        mut plan: LogicalPlan,
        funding: &Funding,
    ) -> Result<Transformed<LogicalPlan>> {
        if !holds_set_comparison(&plan) {
            return Ok(Transformed::no(plan));
        }
        let mut schema = merged_input_schema(funding, &plan)?;
        if let LogicalPlan::Dml(DmlStatement {
            op: WriteOp::MergeInto(_),
            table_name,
            target,
            ..
        }) = &plan
        {
            schema = merged_with(
                funding,
                schema,
                &DFSchema::try_from_qualified_schema(
                    table_name.clone(),
                    &target.schema(),
                )?,
            )?;
        }
        let (schema, bytes) = schema;
        let schema = Released::new(funding, schema, bytes);
        let changed = map_plan_exprs_mut(&mut plan, &mut |expr| {
            rewrite_set_comparisons(funding, expr, &schema)
        })?;
        Ok(Transformed::new_transformed(plan, changed))
    }
}

impl OptimizerRule for RewriteSetComparison {
    fn name(&self) -> &str {
        "rewrite_set_comparison"
    }

    fn rewrite(
        &self,
        plan: LogicalPlan,
        config: &dyn OptimizerConfig,
    ) -> Result<Transformed<LogicalPlan>> {
        let funding = config.funding();
        transform_up_plan_with_subqueries(&funding, plan, |plan| {
            self.rewrite_plan(plan, &funding)
        })
    }
}

/// Whether `plan` (not its inputs) holds a set comparison, found without listing anything.
fn holds_set_comparison(plan: &LogicalPlan) -> bool {
    let mut found = false;
    let _ = apply_plan_exprs(plan, &mut |expr| {
        expr.apply(|e| {
            if matches!(e, Expr::SetComparison(_)) {
                found = true;
                Ok(TreeNodeRecursion::Stop)
            } else {
                Ok(TreeNodeRecursion::Continue)
            }
        })
    });
    found
}

/// Each set comparison in `expr` (its children first) rewritten in place. Returns whether
/// any was.
fn rewrite_set_comparisons(
    funding: &Funding,
    expr: &mut Expr,
    outer_schema: &DFSchema,
) -> Result<bool> {
    let mut changed = false;
    for_each_child_expr_mut(expr, &mut |child| {
        changed |= rewrite_set_comparisons(funding, child, outer_schema)?;
        Ok(())
    })?;
    if let Expr::SetComparison(_) = expr {
        let Expr::SetComparison(set_comparison) = mem::take(expr) else {
            unreachable!()
        };
        *expr = build_set_comparison_subquery(funding, set_comparison, outer_schema)?;
        changed = true;
    }
    Ok(changed)
}

fn build_set_comparison_subquery(
    funding: &Funding,
    set_comparison: SetComparison,
    outer_schema: &DFSchema,
) -> Result<Expr> {
    let SetComparison {
        expr,
        subquery,
        op,
        quantifier,
    } = set_comparison;

    let subquery_schema = subquery.subquery.schema();
    if subquery_schema.fields().is_empty() {
        return plan_err!("single expression required.");
    }
    let mut left_expr = funding.unboxed(expr);
    to_outer_reference(&mut left_expr, outer_schema)?;
    // avoid `head_output_expr` for aggr/window plan, it will gives group-by expr if exists
    let (qualifier, field) = subquery_schema.qualified_field(0);
    let right_expr = column_expr(funding, qualifier, field.name())?;

    let comparison = binary(funding, left_expr, op, right_expr)?;

    // The comparison is copied into the first subquery and moved into the second.
    let first = funding.boxed(funding.cloned(&comparison)?)?;
    let first = match quantifier {
        SetQuantifier::Any => Expr::IsTrue(first),
        SetQuantifier::All => Expr::IsFalse(first),
    };
    let first_exists = exists_subquery(funding, &subquery, first)?;
    let null_exists =
        exists_subquery(funding, &subquery, Expr::IsNull(funding.boxed(comparison)?))?;
    freed(funding, subquery);

    let (first_then, else_expr) = match quantifier {
        SetQuantifier::Any => (lit(true), lit(false)),
        SetQuantifier::All => (lit(false), lit(true)),
    };
    let when_then_expr = funding.vec_of([
        (funding.boxed(first_exists)?, funding.boxed(first_then)?),
        (
            funding.boxed(null_exists)?,
            funding.boxed(Expr::Literal(ScalarValue::Boolean(None), None))?,
        ),
    ])?;
    Ok(Expr::Case(expr::Case {
        expr: None,
        when_then_expr,
        else_expr: Some(funding.boxed(else_expr)?),
    }))
}

/// `EXISTS (subquery WHERE filter)`: the copy of the subquery's node, its filter, their
/// blocks and its outer reference list charged before they are made.
fn exists_subquery(funding: &Funding, subquery: &Subquery, filter: Expr) -> Result<Expr> {
    let input = funding.arc(cloned_plan(funding, &subquery.subquery)?)?;
    let plan = LogicalPlan::Filter(Filter::try_new(filter, input)?);
    let outer_ref_columns = out_ref_exprs(funding, &plan)?;
    Ok(Expr::Exists(Exists {
        subquery: Subquery {
            subquery: funding.arc(plan)?,
            outer_ref_columns,
            spans: funding.cloned(&subquery.spans)?,
        },
        negated: false,
    }))
}

/// Each column of `expr` made an outer reference to `outer_schema`'s field, in place: the
/// column and the field are moved and shared, nothing allocated.
fn to_outer_reference(expr: &mut Expr, outer_schema: &DFSchema) -> Result<()> {
    for_each_child_expr_mut(expr, &mut |child| to_outer_reference(child, outer_schema))?;
    if let Expr::Column(col) = expr {
        let field = Arc::clone(outer_schema.field_from_column(col)?);
        let Expr::Column(col) = mem::take(expr) else {
            unreachable!()
        };
        *expr = Expr::OuterReferenceColumn(field, col);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::OptimizerContext;
    use crate::test::test_table_scan_with_name;
    use datafusion_common::Spans;
    use datafusion_common::config::ConfigOptions;
    use datafusion_common::funding::PlanningFunding;
    use datafusion_expr::logical_plan::builder::LogicalPlanBuilder;
    use datafusion_expr::{Operator, col};

    /// Funding refusing everything.
    struct Refusing;

    impl PlanningFunding for Refusing {
        fn charge(&self, _bytes: usize) -> bool {
            false
        }

        fn discharge(&self, _bytes: usize) {}
    }

    #[test]
    fn rewriting_a_set_comparison_is_refused_unfunded() -> Result<()> {
        let subquery = LogicalPlanBuilder::from(test_table_scan_with_name("sq")?)
            .project(vec![col("sq.c")])?
            .build()?;
        let comparison = Expr::SetComparison(SetComparison::new(
            Box::new(col("t.a")),
            Subquery {
                subquery: Arc::new(subquery),
                outer_ref_columns: vec![],
                spans: Spans::new(),
            },
            Operator::Eq,
            SetQuantifier::Any,
        ));
        let plan = LogicalPlanBuilder::from(test_table_scan_with_name("t")?)
            .filter(comparison)?
            .build()?;
        let mut options = ConfigOptions::default();
        options.funding = Funding::new(Arc::new(Refusing));
        let config = OptimizerContext::new_with_config_options(Arc::new(options));

        let refused = RewriteSetComparison::new()
            .rewrite(plan, &config)
            .unwrap_err();

        assert!(refused.refused().is_some());
        Ok(())
    }
}
