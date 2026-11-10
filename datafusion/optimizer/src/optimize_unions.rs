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

//! [`OptimizeUnions`]: removes `Union` nodes in the logical plan.
use crate::funded::{coerced_for_schema, freed, unwrapped};
use crate::optimizer::ApplyOrder;
use crate::{OptimizerConfig, OptimizerRule};
use datafusion_common::funding::Funding;
use datafusion_common::tree_node::Transformed;
use datafusion_common::{DFSchema, Result, internal_err};
use datafusion_expr::{Distinct, LogicalPlan, Projection, Union};
use std::sync::Arc;

#[derive(Default, Debug)]
/// An optimization rule that
/// 1. replaces nested unions with a single union.
/// 2. removes unions with a single input.
pub struct OptimizeUnions;

impl OptimizeUnions {
    #[expect(missing_docs)]
    pub fn new() -> Self {
        Self {}
    }
}

impl OptimizerRule for OptimizeUnions {
    fn name(&self) -> &str {
        "optimize_unions"
    }

    fn apply_order(&self) -> Option<ApplyOrder> {
        Some(ApplyOrder::BottomUp)
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
            LogicalPlan::Union(Union { mut inputs, schema }) if inputs.len() == 1 => {
                let input = inputs.pop().unwrap();
                freed(&funding, (inputs, schema));
                Ok(Transformed::yes(unwrapped(&funding, input)?))
            }
            LogicalPlan::Union(Union { inputs, schema }) => {
                let inputs = flattened(&funding, inputs, &schema, false)?;
                Ok(Transformed::yes(LogicalPlan::Union(Union {
                    inputs,
                    schema,
                })))
            }
            LogicalPlan::Distinct(Distinct::All(nested_plan))
                if matches!(nested_plan.as_ref(), LogicalPlan::Union(_)) =>
            {
                match unwrapped(&funding, nested_plan)? {
                    LogicalPlan::Union(Union { inputs, schema }) => {
                        let inputs = flattened(&funding, inputs, &schema, true)?;
                        let union =
                            funding.arc(LogicalPlan::Union(Union { inputs, schema }))?;
                        Ok(Transformed::yes(LogicalPlan::Distinct(Distinct::All(
                            union,
                        ))))
                    }
                    nested_plan => Ok(Transformed::no(LogicalPlan::Distinct(
                        Distinct::All(funding.arc(nested_plan)?),
                    ))),
                }
            }
            _ => Ok(Transformed::no(plan)),
        }
    }
}

/// The inputs of a union over `inputs`, each nested union's inputs in its place (a
/// `DISTINCT` over each first taken off when `distinct`), each coerced to `schema`: the
/// list and what it holds charged as they are made, `inputs`' block given back.
fn flattened(
    funding: &Funding,
    mut inputs: Vec<Arc<LogicalPlan>>,
    schema: &DFSchema,
    distinct: bool,
) -> Result<Vec<Arc<LogicalPlan>>> {
    let mut flattened = funding.vec_with_capacity(inputs.len())?;
    for input in inputs.drain(..) {
        let input = if distinct {
            extract_plan_from_distinct(funding, input)?
        } else {
            input
        };
        extract_plans_from_union(funding, input, &mut |plan| {
            let plan = funding.arc(coerced_for_schema(funding, plan, schema)?)?;
            funding.push(&mut flattened, plan)
        })?;
    }
    freed(funding, inputs);
    Ok(flattened)
}

/// Calls `emit` with each plan `plan` unnests into: a union's inputs, or a projection's
/// over each input of the union under it, or `plan` itself.
fn extract_plans_from_union(
    funding: &Funding,
    plan: Arc<LogicalPlan>,
    emit: &mut dyn FnMut(LogicalPlan) -> Result<()>,
) -> Result<()> {
    match unwrapped(funding, plan)? {
        LogicalPlan::Union(Union { mut inputs, schema }) => {
            for input in inputs.drain(..) {
                emit(unwrapped(funding, input)?)?;
            }
            freed(funding, (inputs, schema));
            Ok(())
        }
        // While unnesting, unwrap a Projection whose input is a nested Union,
        // flatten the inner Union, and push the same Projection down onto
        // each of the nested Union’s children.
        //
        // Example:
        //   Union { Projection { Union { plan1, plan2 } }, plan3 }
        //     => Union { Projection { plan1 }, Projection { plan2 }, plan3 }
        LogicalPlan::Projection(Projection {
            expr,
            input,
            schema,
            ..
        }) if matches!(input.as_ref(), LogicalPlan::Union(_)) => {
            let LogicalPlan::Union(Union {
                mut inputs,
                schema: union_schema,
            }) = unwrapped(funding, input)?
            else {
                return internal_err!("optimize_unions: projection lost its union input");
            };
            for input in inputs.drain(..) {
                emit(LogicalPlan::Projection(Projection::try_new_with_schema(
                    funding.to_vec(&expr)?,
                    input,
                    Arc::clone(&schema),
                )?))?;
            }
            freed(funding, (inputs, union_schema));
            freed(funding, (expr, schema));
            Ok(())
        }
        plan => emit(plan),
    }
}

/// `plan` with a `DISTINCT` over it taken off.
fn extract_plan_from_distinct(
    funding: &Funding,
    plan: Arc<LogicalPlan>,
) -> Result<Arc<LogicalPlan>> {
    if !matches!(plan.as_ref(), LogicalPlan::Distinct(Distinct::All(_))) {
        return Ok(plan);
    }
    match unwrapped(funding, plan)? {
        LogicalPlan::Distinct(Distinct::All(plan)) => Ok(plan),
        plan => funding.arc(plan),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::OptimizerContext;
    use crate::analyzer::Analyzer;
    use crate::analyzer::type_coercion::TypeCoercion;
    use crate::assert_optimized_plan_eq_snapshot;
    use crate::funded::Freed;
    use crate::test::unlimited_context;
    use arrow::datatypes::{DataType, Field, Schema};
    use datafusion_common::config::ConfigOptions;
    use datafusion_common::funding::PlanningFunding;
    use datafusion_expr::{col, logical_plan::table_scan};

    fn schema() -> Schema {
        Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("key", DataType::Utf8, false),
            Field::new("value", DataType::Float64, false),
        ])
    }

    macro_rules! assert_optimized_plan_equal {
        (
            $plan:expr,
            @ $expected:literal $(,)?
        ) => {{
            let options = ConfigOptions::default();
            let analyzed_plan = Analyzer::with_rules(vec![Arc::new(TypeCoercion::new())])
                .execute_and_check($plan, &options, |_, _| {})?;
            let optimizer_ctx = OptimizerContext::new().with_max_passes(1);
            let rules: Vec<Arc<dyn crate::OptimizerRule + Send + Sync>> = vec![Arc::new(OptimizeUnions::new())];
            assert_optimized_plan_eq_snapshot!(
                optimizer_ctx,
                rules,
                analyzed_plan,
                @ $expected,
            )
        }};
    }

    #[test]
    fn eliminate_nothing() -> Result<()> {
        let plan_builder = table_scan(Some("table"), &schema(), None)?;

        let plan = plan_builder.clone().union(plan_builder.build()?)?.build()?;

        assert_optimized_plan_equal!(plan, @r"
        Union
          TableScan: table
          TableScan: table
        ")
    }

    #[test]
    fn eliminate_distinct_nothing() -> Result<()> {
        let plan_builder = table_scan(Some("table"), &schema(), None)?;

        let plan = plan_builder
            .clone()
            .union_distinct(plan_builder.build()?)?
            .build()?;

        assert_optimized_plan_equal!(plan, @r"
        Distinct:
          Union
            TableScan: table
            TableScan: table
        ")
    }

    #[test]
    fn eliminate_nested_union() -> Result<()> {
        let plan_builder = table_scan(Some("table"), &schema(), None)?;

        let plan = plan_builder
            .clone()
            .union(plan_builder.clone().build()?)?
            .union(plan_builder.clone().build()?)?
            .union(plan_builder.build()?)?
            .build()?;

        assert_optimized_plan_equal!(plan, @r"
        Union
          TableScan: table
          TableScan: table
          TableScan: table
          TableScan: table
        ")
    }

    #[test]
    fn eliminate_nested_union_with_distinct_union() -> Result<()> {
        let plan_builder = table_scan(Some("table"), &schema(), None)?;

        let plan = plan_builder
            .clone()
            .union_distinct(plan_builder.clone().build()?)?
            .union(plan_builder.clone().build()?)?
            .union(plan_builder.build()?)?
            .build()?;

        assert_optimized_plan_equal!(plan, @r"
        Union
          Distinct:
            Union
              TableScan: table
              TableScan: table
          TableScan: table
          TableScan: table
        ")
    }

    #[test]
    fn eliminate_nested_distinct_union() -> Result<()> {
        let plan_builder = table_scan(Some("table"), &schema(), None)?;

        let plan = plan_builder
            .clone()
            .union(plan_builder.clone().build()?)?
            .union_distinct(plan_builder.clone().build()?)?
            .union(plan_builder.clone().build()?)?
            .union_distinct(plan_builder.build()?)?
            .build()?;

        assert_optimized_plan_equal!(plan, @r"
        Distinct:
          Union
            TableScan: table
            TableScan: table
            TableScan: table
            TableScan: table
            TableScan: table
        ")
    }

    #[test]
    fn eliminate_nested_distinct_union_with_distinct_table() -> Result<()> {
        let plan_builder = table_scan(Some("table"), &schema(), None)?;

        let plan = plan_builder
            .clone()
            .union_distinct(plan_builder.clone().distinct()?.build()?)?
            .union(plan_builder.clone().distinct()?.build()?)?
            .union_distinct(plan_builder.build()?)?
            .build()?;

        assert_optimized_plan_equal!(plan, @r"
        Distinct:
          Union
            TableScan: table
            TableScan: table
            TableScan: table
            TableScan: table
        ")
    }

    // We don't need to use project_with_column_index in logical optimizer,
    // after LogicalPlanBuilder::union, we already have all equal expression aliases
    #[test]
    fn eliminate_nested_union_with_projection() -> Result<()> {
        let plan_builder = table_scan(Some("table"), &schema(), None)?;

        let plan = plan_builder
            .clone()
            .union(
                plan_builder
                    .clone()
                    .project(vec![col("id").alias("table_id"), col("key"), col("value")])?
                    .build()?,
            )?
            .union(
                plan_builder
                    .project(vec![col("id").alias("_id"), col("key"), col("value")])?
                    .build()?,
            )?
            .build()?;

        assert_optimized_plan_equal!(plan, @r"
        Union
          TableScan: table
          Projection: table.id AS id, table.key, table.value
            TableScan: table
          Projection: table.id AS id, table.key, table.value
            TableScan: table
        ")
    }

    #[test]
    fn eliminate_nested_distinct_union_with_projection() -> Result<()> {
        let plan_builder = table_scan(Some("table"), &schema(), None)?;

        let plan = plan_builder
            .clone()
            .union_distinct(
                plan_builder
                    .clone()
                    .project(vec![col("id").alias("table_id"), col("key"), col("value")])?
                    .build()?,
            )?
            .union_distinct(
                plan_builder
                    .project(vec![col("id").alias("_id"), col("key"), col("value")])?
                    .build()?,
            )?
            .build()?;

        assert_optimized_plan_equal!(plan, @r"
        Distinct:
          Union
            TableScan: table
            Projection: table.id AS id, table.key, table.value
              TableScan: table
            Projection: table.id AS id, table.key, table.value
              TableScan: table
        ")
    }

    #[test]
    fn eliminate_nested_union_in_projection() -> Result<()> {
        let plan_builder = table_scan(Some("table"), &schema(), None)?;

        let plan = plan_builder
            .clone()
            .union(plan_builder.clone().build()?)?
            .project(vec![col("id").alias("table_id"), col("key"), col("value")])?
            .union(plan_builder.build()?)?
            .build()?;

        assert_optimized_plan_equal!(plan, @r"
        Union
          Projection: id AS table_id, key, value
            TableScan: table
          Projection: id AS table_id, key, value
            TableScan: table
          TableScan: table
        ")
    }

    #[test]
    fn eliminate_nested_union_with_type_cast_projection() -> Result<()> {
        let table_1 = table_scan(
            Some("table_1"),
            &Schema::new(vec![
                Field::new("id", DataType::Int64, false),
                Field::new("key", DataType::Utf8, false),
                Field::new("value", DataType::Float64, false),
            ]),
            None,
        )?;

        let table_2 = table_scan(
            Some("table_1"),
            &Schema::new(vec![
                Field::new("id", DataType::Int32, false),
                Field::new("key", DataType::Utf8, false),
                Field::new("value", DataType::Float32, false),
            ]),
            None,
        )?;

        let table_3 = table_scan(
            Some("table_1"),
            &Schema::new(vec![
                Field::new("id", DataType::Int16, false),
                Field::new("key", DataType::Utf8, false),
                Field::new("value", DataType::Float32, false),
            ]),
            None,
        )?;

        let plan = table_1
            .union(table_2.build()?)?
            .union(table_3.build()?)?
            .build()?;

        assert_optimized_plan_equal!(plan, @r"
        Union
          TableScan: table_1
          Projection: CAST(table_1.id AS Int64) AS id, table_1.key, CAST(table_1.value AS Float64) AS value
            TableScan: table_1
          Projection: CAST(table_1.id AS Int64) AS id, table_1.key, CAST(table_1.value AS Float64) AS value
            TableScan: table_1
        ")
    }

    #[test]
    fn eliminate_nested_distinct_union_with_type_cast_projection() -> Result<()> {
        let table_1 = table_scan(
            Some("table_1"),
            &Schema::new(vec![
                Field::new("id", DataType::Int64, false),
                Field::new("key", DataType::Utf8, false),
                Field::new("value", DataType::Float64, false),
            ]),
            None,
        )?;

        let table_2 = table_scan(
            Some("table_1"),
            &Schema::new(vec![
                Field::new("id", DataType::Int32, false),
                Field::new("key", DataType::Utf8, false),
                Field::new("value", DataType::Float32, false),
            ]),
            None,
        )?;

        let table_3 = table_scan(
            Some("table_1"),
            &Schema::new(vec![
                Field::new("id", DataType::Int16, false),
                Field::new("key", DataType::Utf8, false),
                Field::new("value", DataType::Float32, false),
            ]),
            None,
        )?;

        let plan = table_1
            .union_distinct(table_2.build()?)?
            .union_distinct(table_3.build()?)?
            .build()?;

        assert_optimized_plan_equal!(plan, @r"
        Distinct:
          Union
            TableScan: table_1
            Projection: CAST(table_1.id AS Int64) AS id, table_1.key, CAST(table_1.value AS Float64) AS value
              TableScan: table_1
            Projection: CAST(table_1.id AS Int64) AS id, table_1.key, CAST(table_1.value AS Float64) AS value
              TableScan: table_1
        ")
    }

    #[test]
    fn eliminate_one_union() -> Result<()> {
        let plan = table_scan(Some("table"), &schema(), None)?.build()?;
        let schema = Arc::clone(plan.schema());
        // note it is not possible to create a single input union via
        // LogicalPlanBuilder so create it manually here
        let plan = LogicalPlan::Union(Union {
            inputs: vec![Arc::new(plan)],
            schema,
        });

        // Note we can't use the same assert_optimized_plan_equal as creating a
        // single input union is not possible via LogicalPlanBuilder and other passes
        // throw errors / don't handle the schema correctly.
        assert_optimized_plan_eq_snapshot!(
            OptimizerContext::new().with_max_passes(1),
            vec![Arc::new(OptimizeUnions::new())],
            plan,
            @"TableScan: table"
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

    fn context(funding: Funding) -> OptimizerContext {
        let mut options = ConfigOptions::default();
        options.funding = funding;
        OptimizerContext::new_with_config_options(Arc::new(options))
    }

    fn nested_union() -> Result<LogicalPlan> {
        let plan_builder = table_scan(Some("table"), &schema(), None)?;
        plan_builder
            .clone()
            .union(plan_builder.clone().build()?)?
            .union(plan_builder.build()?)?
            .build()
    }

    #[test]
    fn flattening_a_union_is_refused_unfunded() -> Result<()> {
        let config = context(Funding::new(Arc::new(Refusing)));

        let refused = OptimizeUnions::new()
            .rewrite(nested_union()?, &config)
            .unwrap_err();

        assert!(refused.refused().is_some());
        Ok(())
    }

    #[test]
    fn flattened_union_holds_what_was_charged() -> Result<()> {
        let plan = nested_union()?;
        let config = context(Funding::new(Arc::new(Unlimited)));
        // Planned through the same funding.
        config.options().funding.charge(plan.freed_bytes())?;

        let flattened = OptimizeUnions::new().rewrite(plan, &config)?;

        assert!(flattened.transformed);
        assert_eq!(flattened.data.inputs().len(), 3);
        // The nested union's node, block and list are given back.
        assert_eq!(
            config.options().funding.outstanding(),
            flattened.data.freed_bytes()
        );
        Ok(())
    }

    #[test]
    fn a_single_input_union_gives_back_its_schema() -> Result<()> {
        let input = table_scan(Some("table"), &schema(), None)?.build()?;
        let schema = Arc::new(input.schema().as_ref().clone());
        let plan = LogicalPlan::Union(Union {
            inputs: vec![Arc::new(input)],
            schema,
        });
        let config = unlimited_context();
        config.options().funding.charge(plan.freed_bytes())?;

        let unwrapped = OptimizeUnions::new().rewrite(plan, &config)?;

        assert!(matches!(unwrapped.data, LogicalPlan::TableScan(_)));
        // The union's node, list and schema are given back.
        assert_eq!(
            config.options().funding.outstanding(),
            unwrapped.data.freed_bytes()
        );
        Ok(())
    }
}
