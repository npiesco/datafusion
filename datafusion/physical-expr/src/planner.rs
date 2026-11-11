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

use datafusion_common::funding::Funding;
use std::sync::Arc;

use crate::projection::funded_column;
use crate::scalar_subquery::ScalarSubqueryExpr;
use crate::{HigherOrderFunctionExpr, ScalarFunctionExpr};
use crate::{
    PhysicalExpr,
    expressions::{self, Literal, binary, like, similar_to},
};

use arrow::datatypes::Schema;
use datafusion_common::config::ConfigOptions;
use datafusion_common::metadata::FieldMetadata;
use datafusion_common::{
    DFSchema, Result, ScalarValue, TableReference, ToDFSchema, exec_err,
    internal_datafusion_err, not_impl_err, plan_datafusion_err, plan_err,
};
use datafusion_expr::execution_props::ExecutionProps;
use datafusion_expr::expr::{
    Alias, Cast, HigherOrderFunction, InList, Lambda, LambdaVariable, Placeholder,
    ScalarFunction,
};
use datafusion_expr::physical_planning_context::PhysicalPlanningContext;
use datafusion_expr::var_provider::VarType;
use datafusion_expr::var_provider::is_system_variables;
use datafusion_expr::{Between, BinaryExpr, Expr, Like, Operator, TryCast};

/// [PhysicalExpr] evaluate DataFusion expressions such as `A + 1`, or `CAST(c1
/// AS int)`.
///
/// [PhysicalExpr] are the physical counterpart to [Expr] used in logical
/// planning, and can be evaluated directly on a [RecordBatch]. They are
/// normally created from [Expr] by a [PhysicalPlanner] and can be created
/// directly using [create_physical_expr].
///
/// A Physical expression knows its type, nullability and how to evaluate itself.
///
/// [PhysicalPlanner]: https://docs.rs/datafusion/latest/datafusion/physical_planner/trait.PhysicalPlanner.html
/// [RecordBatch]: https://docs.rs/arrow/latest/arrow/record_batch/struct.RecordBatch.html
///
/// # Example: Create `PhysicalExpr` from `Expr`
/// ```
/// # use arrow::datatypes::{DataType, Field, Schema};
/// # use datafusion_common::DFSchema;
/// # use datafusion_expr::{Expr, col, lit};
/// # use datafusion_physical_expr::create_physical_expr;
/// # use datafusion_expr::execution_props::ExecutionProps;
/// # use datafusion_expr::physical_planning_context::PhysicalPlanningContext;
/// // For a logical expression `a = 1`, we can create a physical expression
/// let expr = col("a").eq(lit(1));
/// // To create a PhysicalExpr we need 1. a schema
/// let schema = Schema::new(vec![Field::new("a", DataType::Int32, true)]);
/// let df_schema = DFSchema::try_from(schema).unwrap();
/// // 2. ExecutionProps
/// let props = ExecutionProps::new();
/// // We can now create a PhysicalExpr. Expressions with no scalar
/// // subqueries use an empty `PhysicalPlanningContext`:
/// let physical_expr =
///     create_physical_expr(
///         &expr,
///         &df_schema,
///         &props,
///         &PhysicalPlanningContext::default(),
///         &datafusion_common::funding::Funding::default(),
///     )
///         .unwrap();
/// ```
///
/// # Example: Executing a PhysicalExpr to obtain [ColumnarValue]
/// ```
/// # use std::sync::Arc;
/// # use arrow::array::{cast::AsArray, BooleanArray, Int32Array, RecordBatch};
/// # use arrow::datatypes::{DataType, Field, Schema};
/// # use datafusion_common::{assert_batches_eq, DFSchema};
/// # use datafusion_expr::{Expr, col, lit, ColumnarValue};
/// # use datafusion_physical_expr::create_physical_expr;
/// # use datafusion_expr::execution_props::ExecutionProps;
/// # use datafusion_expr::physical_planning_context::PhysicalPlanningContext;
/// # let expr = col("a").eq(lit(1));
/// # let schema = Schema::new(vec![Field::new("a", DataType::Int32, true)]);
/// # let df_schema = DFSchema::try_from(schema.clone()).unwrap();
/// # let props = ExecutionProps::new();
/// // Given a PhysicalExpr, for `a = 1` we can evaluate it against a RecordBatch like this:
/// let physical_expr =
///     create_physical_expr(
///         &expr,
///         &df_schema,
///         &props,
///         &PhysicalPlanningContext::default(),
///         &datafusion_common::funding::Funding::default(),
///     )
///         .unwrap();
/// // Input of [1,2,3]
/// let input_batch = RecordBatch::try_from_iter(vec![
///   ("a", Arc::new(Int32Array::from(vec![1, 2, 3])) as _)
/// ]).unwrap();
/// // The result is a ColumnarValue (either an Array or a Scalar)
/// let result = physical_expr.evaluate(&input_batch).unwrap();
/// // In this case, a BooleanArray with the result of the comparison
/// let ColumnarValue::Array(arr) = result else {
///  panic!("Expected an array")
/// };
/// assert_eq!(arr.as_boolean(), &BooleanArray::from(vec![true, false, false]));
/// ```
///
/// [ColumnarValue]: datafusion_expr::ColumnarValue
///
/// Create a physical expression from a logical expression ([Expr]).
///
/// # Arguments
///
/// * `e` - The logical expression
/// * `input_dfschema` - The DataFusion schema for the input, used to resolve `Column` references
///   to qualified or unqualified fields by name.
/// * `execution_props` - Per-execution properties such as the query start time.
/// * `planning_ctx` - The [`PhysicalPlanningContext`] used to resolve
///   `Expr::ScalarSubquery` and `Expr::LambdaVariable` nodes. The physical
///   planner threads the subquery index map and shared results container from
///   its `ScalarSubqueryExec` construction into calls to
///   `create_physical_expr`; the lambda variable qualifiers are added by this
///   function itself as it descends into lambda bodies. Callers creating
///   physical expressions outside of physical planning should pass
///   `&PhysicalPlanningContext::default()`; converting a scalar subquery then returns a
///   planning error.
/// * `funding` - What the expression allocates is charged to it before it allocates;
///   the expression returned keeps what was charged for it. A charge refused fails the
///   conversion. Pass `&Funding::default()` outside of planning.
#[cfg_attr(feature = "recursive_protection", recursive::recursive)]
pub fn create_physical_expr(
    e: &Expr,
    input_dfschema: &DFSchema,
    execution_props: &ExecutionProps,
    planning_ctx: &PhysicalPlanningContext,
    funding: &Funding,
) -> Result<Arc<dyn PhysicalExpr>> {
    let input_schema = input_dfschema.as_arrow();
    let create = |expr: &Expr| {
        create_physical_expr(expr, input_dfschema, execution_props, planning_ctx, funding)
    };
    let create_all = |exprs: &[Expr]| {
        create_physical_exprs(
            exprs,
            input_dfschema,
            execution_props,
            planning_ctx,
            funding,
        )
    };
    let truth_test = |expr: &Expr, op: Operator, value: Option<bool>| {
        let lhs = create(expr)?;
        let rhs = funding.arc(Literal::new(ScalarValue::Boolean(value), funding)?)?;
        binary(lhs, op, rhs, input_schema, funding)
    };

    match e {
        Expr::Alias(Alias { expr, metadata, .. }) => {
            if let Expr::Literal(v, prior_metadata) = expr.as_ref() {
                let new_metadata = FieldMetadata::merge_options(
                    prior_metadata.as_ref(),
                    metadata.as_ref(),
                );
                Ok(funding.arc(Literal::new_with_metadata(
                    funding.cloned(v)?,
                    new_metadata,
                    funding,
                )?)?)
            } else {
                create(expr)
            }
        }
        Expr::Column(c) => {
            let idx = input_dfschema.index_of_column(c)?;
            funded_column(&c.name, idx, funding)
        }
        Expr::Literal(value, metadata) => Ok(funding.arc(Literal::new_with_metadata(
            funding.cloned(value)?,
            funding.cloned(metadata)?,
            funding,
        )?)?),
        Expr::ScalarVariable(_, variable_names) => {
            let system = is_system_variables(variable_names);
            let var_type = if system {
                VarType::System
            } else {
                VarType::UserDefined
            };
            match execution_props.get_var_provider(var_type) {
                Some(provider) => {
                    let scalar_value =
                        provider.get_value(funding.cloned(variable_names)?)?;
                    Ok(funding.arc(Literal::new(scalar_value, funding)?)?)
                }
                None if system => plan_err!("No system variable provider found"),
                None => plan_err!("No user defined variable provider found"),
            }
        }
        Expr::IsTrue(expr) => truth_test(expr, Operator::IsNotDistinctFrom, Some(true)),
        Expr::IsNotTrue(expr) => truth_test(expr, Operator::IsDistinctFrom, Some(true)),
        Expr::IsFalse(expr) => truth_test(expr, Operator::IsNotDistinctFrom, Some(false)),
        Expr::IsNotFalse(expr) => truth_test(expr, Operator::IsDistinctFrom, Some(false)),
        Expr::IsUnknown(expr) => truth_test(expr, Operator::IsNotDistinctFrom, None),
        Expr::IsNotUnknown(expr) => truth_test(expr, Operator::IsDistinctFrom, None),
        Expr::BinaryExpr(BinaryExpr { left, op, right }) => {
            // Create physical expressions for left and right operands
            let lhs = create(left)?;
            let rhs = create(right)?;
            // Note that the logical planner is responsible
            // for type coercion on the arguments (e.g. if one
            // argument was originally Int32 and one was
            // Int64 they will both be coerced to Int64).
            //
            // There should be no coercion during physical
            // planning.
            binary(lhs, *op, rhs, input_schema, funding)
        }
        Expr::Like(Like {
            negated,
            expr,
            pattern,
            escape_char,
            case_insensitive,
        }) => {
            // `\` is the implicit escape, see https://github.com/apache/datafusion/issues/13291
            if escape_char.unwrap_or('\\') != '\\' {
                return exec_err!(
                    "LIKE does not support escape_char other than the backslash (\\)"
                );
            }
            let physical_expr = create(expr)?;
            let physical_pattern = create(pattern)?;
            like(
                *negated,
                *case_insensitive,
                physical_expr,
                physical_pattern,
                input_schema,
                funding,
            )
        }
        Expr::SimilarTo(Like {
            negated,
            expr,
            pattern,
            escape_char,
            case_insensitive,
        }) => {
            if escape_char.is_some() {
                return exec_err!("SIMILAR TO does not support escape_char yet");
            }
            let physical_expr = create(expr)?;
            let physical_pattern = create(pattern)?;
            similar_to(
                *negated,
                *case_insensitive,
                physical_expr,
                physical_pattern,
                funding,
            )
        }
        Expr::Case(case) => {
            let expr = case.expr.as_deref().map(create).transpose()?;
            let mut when_then_expr =
                funding.held::<Vec<(Arc<dyn PhysicalExpr>, Arc<dyn PhysicalExpr>)>>();
            when_then_expr.reserve_exact(case.when_then_expr.len())?;
            for (when, then) in &case.when_then_expr {
                when_then_expr.push((create(when)?, create(then)?))?;
            }
            let else_expr = case.else_expr.as_deref().map(create).transpose()?;
            expressions::case(expr, when_then_expr.keep(), else_expr, funding)
        }
        Expr::Cast(Cast { expr, field }) => expressions::cast_with_target_field(
            create(expr)?,
            input_schema,
            field,
            None,
            funding,
        ),
        Expr::TryCast(TryCast { expr, field }) => {
            expressions::try_cast_with_target_field(
                create(expr)?,
                input_schema,
                field,
                funding,
            )
        }
        Expr::Not(expr) => expressions::not(create(expr)?, funding),
        Expr::Negative(expr) => {
            expressions::negative(create(expr)?, input_schema, funding)
        }
        Expr::IsNull(expr) => expressions::is_null(create(expr)?, funding),
        Expr::IsNotNull(expr) => expressions::is_not_null(create(expr)?, funding),
        Expr::ScalarFunction(ScalarFunction { func, args }) => {
            let physical_args = create_all(args)?;
            let config_options = match execution_props.config_options.as_ref() {
                Some(config_options) => Arc::clone(config_options),
                None => funding.arc(ConfigOptions::default())?,
            };

            Ok(funding.arc(ScalarFunctionExpr::try_new(
                Arc::clone(func),
                physical_args,
                input_schema,
                config_options,
                funding,
            )?)?)
        }
        Expr::Between(Between {
            expr,
            negated,
            low,
            high,
        }) => {
            let value_expr = create(expr)?;
            let low_expr = create(low)?;
            let high_expr = create(high)?;

            // rewrite the between into the two binary operators
            let binary_expr = binary(
                binary(
                    Arc::clone(&value_expr),
                    Operator::GtEq,
                    low_expr,
                    input_schema,
                    funding,
                )?,
                Operator::And,
                binary(
                    Arc::clone(&value_expr),
                    Operator::LtEq,
                    high_expr,
                    input_schema,
                    funding,
                )?,
                input_schema,
                funding,
            )?;

            if *negated {
                expressions::not(binary_expr, funding)
            } else {
                Ok(binary_expr)
            }
        }
        Expr::InList(InList {
            expr,
            list,
            negated,
        }) => match expr.as_ref() {
            Expr::Literal(ScalarValue::Utf8(None), _) => {
                Ok(funding.arc(Literal::new(ScalarValue::Boolean(None), funding)?)?)
            }
            _ => {
                let value_expr = create(expr)?;
                let list_exprs = create_all(list)?;
                expressions::in_list(
                    value_expr,
                    list_exprs,
                    negated,
                    input_schema,
                    funding,
                )
            }
        },
        Expr::ScalarSubquery(sq) => {
            match planning_ctx.index_of(sq) {
                Some(index) => {
                    let schema = sq.subquery.schema();
                    if schema.fields().len() != 1 {
                        return plan_err!(
                            "Scalar subquery must return exactly one column, got {}",
                            schema.fields().len()
                        );
                    }
                    let dt = funding.cloned(schema.field(0).data_type())?;
                    let nullable = schema.field(0).is_nullable();
                    Ok(funding.arc(ScalarSubqueryExpr::new(
                        dt,
                        nullable,
                        index,
                        planning_ctx.results().clone(),
                    ))?)
                }
                None => {
                    // Not found: either a correlated subquery that wasn't
                    // rewritten to a join, or an uncorrelated one that wasn't
                    // registered by the physical planner.
                    not_impl_err!(
                        "Physical plan does not support logical expression {e:?}"
                    )
                }
            }
        }
        Expr::Placeholder(Placeholder { id, .. }) => {
            exec_err!("Placeholder '{id}' was not provided a value for execution.")
        }
        Expr::HigherOrderFunction(invocation @ HigherOrderFunction { func, args }) => {
            let num_lambdas = args
                .iter()
                .filter(|arg| matches!(arg, Expr::Lambda(_)))
                .count();

            let mut lambda_parameters =
                invocation.lambda_parameters(input_dfschema)?.into_iter();

            if num_lambdas > lambda_parameters.len() {
                return plan_err!(
                    "{} lambda_parameters returned only {} values for {num_lambdas} lambdas",
                    func.name(),
                    lambda_parameters.len()
                );
            }

            let lambda_qualifier = 1 + input_dfschema
                .iter()
                .filter_map(|(qualifier, _field)| {
                    qualifier.and_then(|tbl| {
                        tbl.table().strip_prefix("lambda_")?.parse::<usize>().ok()
                    })
                })
                .max()
                .unwrap_or_default();

            let qualifier_name =
                funding.format(format_args!("lambda_{lambda_qualifier}"))?;
            let qualifier_table = funding.arc_str(&qualifier_name);
            funding.released(qualifier_name);
            let qualifier = TableReference::bare(qualifier_table?);

            let mut physical_args = funding.held::<Vec<Arc<dyn PhysicalExpr>>>();
            physical_args.reserve_exact(args.len())?;
            for arg in args {
                let physical_arg = match arg {
                    Expr::Lambda(lambda) => {
                        let lambda_parameters =
                            lambda_parameters.next().ok_or_else(|| {
                                internal_datafusion_err!(
                                    "lambda_parameters len should have been checked above"
                                )
                            })?;

                        let mut new_fields = funding.held::<Vec<_>>();
                        new_fields.reserve_exact(
                            input_dfschema.fields().len() + lambda.params.len(),
                        )?;
                        for (tbl, field) in input_dfschema.iter() {
                            new_fields.push((tbl.cloned(), Arc::clone(field)))?;
                        }
                        for (field, name) in
                            lambda_parameters.into_iter().zip(&lambda.params)
                        {
                            new_fields.push((
                                Some(qualifier.clone()),
                                funding.renamed_field(field, name)?,
                            ))?;
                        }

                        let new_fields = new_fields.keep();
                        let lambda_schema = DFSchema::new_with_metadata(
                            &new_fields,
                            input_dfschema.metadata().clone(),
                            funding,
                        )?;
                        funding.released(new_fields);

                        let planning_ctx = planning_ctx
                            .clone()
                            .with_qualified_lambda_variables(&qualifier, &lambda.params);

                        create_physical_expr(
                            arg,
                            &lambda_schema,
                            execution_props,
                            &planning_ctx,
                            funding,
                        )?
                    }
                    _ => create(arg)?,
                };
                physical_args.push(physical_arg)?;
            }

            let config_options = match execution_props.config_options.as_ref() {
                Some(config_options) => Arc::clone(config_options),
                None => funding.arc(ConfigOptions::default())?,
            };

            Ok(funding.arc(HigherOrderFunctionExpr::try_new_with_schema(
                Arc::clone(func),
                physical_args.keep(),
                input_schema,
                config_options,
                funding,
            )?)?)
        }
        Expr::Lambda(Lambda { params, body }) => {
            expressions::lambda(params, create(body)?, funding)
        }
        Expr::LambdaVariable(LambdaVariable {
            name,
            field,
            spans: _,
        }) => {
            let field = field.as_ref().ok_or_else(|| {
                plan_datafusion_err!("unresolved LambdaVariable {name}")
            })?;

            let qualifier =
                planning_ctx
                    .lambda_variable_qualifier(name)
                    .ok_or_else(|| {
                        plan_datafusion_err!(
                            "qualifier for lambda variable {name} not found"
                        )
                    })?;

            let index = input_dfschema
                .index_of_column_by_name(Some(qualifier), name)
                .ok_or_else(|| {
                    plan_datafusion_err!(
                        "lambda variable {qualifier}.{name} not found in planning schema"
                    )
                })?;

            let schema_field = input_dfschema.field(index);

            // LambdaVariable.field will be made optional as in Expr::Placeholder
            // and only LambdaVariable.name used, and field.name ignored,
            // so they're not enforced to match for logical expressions
            // Rename the field to match the schema one and use it's PartialEq impl instead
            // of checking property by property and fail if new properties get's added to it.
            // While not necessary, the sql planner does create lambda vars with matching names,
            // so this shouldn't allocate with a lambda var from it
            let renamed_field = funding.renamed_field(Arc::clone(field), name)?;

            if &renamed_field != schema_field {
                return plan_err!(
                    "LambdaVariable field and schema field mismatch {} != {}",
                    renamed_field,
                    schema_field
                );
            }

            Ok(funding.arc(expressions::LambdaVariable::new(
                index,
                Arc::clone(schema_field),
            ))?)
        }
        other => {
            not_impl_err!("Physical plan does not support logical expression {other:?}")
        }
    }
}

/// Create vector of Physical Expression from a vector of logical expression, the list
/// and what each expression allocates charged to `funding` first
///
/// See [`create_physical_expr`] for details on the `planning_ctx` argument.
pub fn create_physical_exprs<'a, I>(
    exprs: I,
    input_dfschema: &DFSchema,
    execution_props: &ExecutionProps,
    planning_ctx: &PhysicalPlanningContext,
    funding: &Funding,
) -> Result<Vec<Arc<dyn PhysicalExpr>>>
where
    I: IntoIterator<Item = &'a Expr>,
{
    let exprs = exprs.into_iter();
    let mut physical = funding.held::<Vec<_>>();
    physical.reserve_exact(exprs.size_hint().0)?;
    for expr in exprs {
        physical.push(create_physical_expr(
            expr,
            input_dfschema,
            execution_props,
            planning_ctx,
            funding,
        )?)?;
    }
    Ok(physical.keep())
}

/// Convert a logical expression to a physical expression (without any simplification, etc)
pub fn logical2physical(expr: &Expr, schema: &Schema) -> Arc<dyn PhysicalExpr> {
    // TODO this makes a deep copy of the Schema. Should take SchemaRef instead and avoid deep copy
    let df_schema = schema.clone().to_dfschema().unwrap();
    let execution_props = ExecutionProps::new();
    create_physical_expr(
        expr,
        &df_schema,
        &execution_props,
        &PhysicalPlanningContext::default(),
        &Funding::default(),
    )
    .unwrap()
}

#[cfg(test)]
mod tests {
    use arrow::array::{ArrayRef, BooleanArray, RecordBatch, StringArray};
    use arrow::datatypes::{DataType, Field};
    use arrow_schema::extension::{EXTENSION_TYPE_METADATA_KEY, EXTENSION_TYPE_NAME_KEY};
    use datafusion_common::funding::Funding;
    use datafusion_expr::{col, lit};

    use super::*;

    fn test_cast_schema() -> Schema {
        Schema::new(vec![Field::new("a", DataType::Int32, false)])
    }

    fn lower_cast_expr(expr: &Expr, schema: &Schema) -> Result<Arc<dyn PhysicalExpr>> {
        let df_schema = DFSchema::try_from(schema.clone())?;
        create_physical_expr(
            expr,
            &df_schema,
            &ExecutionProps::new(),
            &PhysicalPlanningContext::default(),
            &Funding::default(),
        )
    }

    fn as_planner_cast(physical: &Arc<dyn PhysicalExpr>) -> &expressions::CastExpr {
        physical
            .downcast_ref::<expressions::CastExpr>()
            .expect("planner should lower logical CAST to CastExpr")
    }

    fn as_planner_try_cast(
        physical: &Arc<dyn PhysicalExpr>,
    ) -> &expressions::TryCastExpr {
        physical
            .downcast_ref::<expressions::TryCastExpr>()
            .expect("planner should lower logical TRY_CAST to TryCastExpr")
    }

    #[test]
    fn test_create_physical_expr_scalar_input_output() -> Result<()> {
        let expr = col("letter").eq(lit("A"));

        let schema = Schema::new(vec![Field::new("letter", DataType::Utf8, false)]);
        let df_schema = DFSchema::try_from_qualified_schema(
            "data",
            &schema,
            &datafusion_common::funding::Funding::default(),
        )?;
        let p = create_physical_expr(
            &expr,
            &df_schema,
            &ExecutionProps::new(),
            &PhysicalPlanningContext::default(),
            &Funding::default(),
        )?;

        let batch = RecordBatch::try_new(
            Arc::new(schema),
            vec![Arc::new(StringArray::from_iter_values(vec![
                "A", "B", "C", "D",
            ]))],
        )?;
        let result = p.evaluate(&batch)?;
        let result = result.into_array(4).expect("Failed to convert to array");

        assert_eq!(
            &result,
            &(Arc::new(BooleanArray::from(vec![true, false, false, false,])) as ArrayRef)
        );

        Ok(())
    }

    #[test]
    fn test_cast_lowering_preserves_target_field_metadata() -> Result<()> {
        let schema = test_cast_schema();

        // Target field with both extension metadata and custom metadata.
        // With exact target metadata semantics, all target metadata should propagate.
        let target_field = Arc::new(
            Field::new("cast_target", DataType::Int64, true).with_metadata(
                [
                    (
                        EXTENSION_TYPE_NAME_KEY.to_string(),
                        "arrow.json".to_string(),
                    ),
                    (EXTENSION_TYPE_METADATA_KEY.to_string(), "{}".to_string()),
                    ("custom_target_meta".to_string(), "custom_value".to_string()),
                ]
                .into(),
            ),
        );
        let cast_expr = Expr::Cast(Cast::new_from_field(
            Box::new(col("a")),
            Arc::clone(&target_field),
        ));

        let physical = lower_cast_expr(&cast_expr, &schema)?;
        let cast = as_planner_cast(&physical);

        // The CastExpr stores the target type and all target metadata
        assert_eq!(cast.cast_type(), &DataType::Int64);
        let target_metadata = cast.target_metadata().expect("should have metadata");
        assert_eq!(
            target_metadata.get(EXTENSION_TYPE_NAME_KEY),
            Some(&"arrow.json".to_string())
        );
        assert_eq!(
            target_metadata.get(EXTENSION_TYPE_METADATA_KEY),
            Some(&"{}".to_string())
        );
        assert_eq!(cast.target_nullable(), Some(true));

        // return_field should have all target metadata (exact semantics)
        let returned = physical.return_field(&schema)?;
        assert_eq!(
            returned.metadata().get(EXTENSION_TYPE_NAME_KEY),
            Some(&"arrow.json".to_string())
        );
        assert_eq!(
            returned.metadata().get(EXTENSION_TYPE_METADATA_KEY),
            Some(&"{}".to_string())
        );
        // All target metadata should propagate with exact semantics
        assert_eq!(
            returned.metadata().get("custom_target_meta"),
            Some(&"custom_value".to_string()),
            "All target metadata should propagate with exact semantics"
        );
        assert!(physical.nullable(&schema)?);

        Ok(())
    }

    #[test]
    fn test_cast_lowering_preserves_standard_cast_semantics() -> Result<()> {
        let schema = test_cast_schema();
        let cast_expr = Expr::Cast(Cast::new(Box::new(col("a")), DataType::Int64));

        let physical = lower_cast_expr(&cast_expr, &schema)?;
        let cast = as_planner_cast(&physical);
        let returned_field = physical.return_field(&schema)?;

        assert_eq!(cast.cast_type(), &DataType::Int64);
        assert_eq!(returned_field.name(), "a");
        assert_eq!(returned_field.data_type(), &DataType::Int64);
        assert!(!physical.nullable(&schema)?);

        Ok(())
    }

    #[test]
    fn test_cast_lowering_preserves_same_type_field_semantics() -> Result<()> {
        let schema = test_cast_schema();

        // Same-type cast with extension metadata on target.
        // With exact target metadata semantics, all target metadata should propagate.
        let target_field = Arc::new(
            Field::new("same_type_cast", DataType::Int32, true).with_metadata(
                [
                    (
                        EXTENSION_TYPE_NAME_KEY.to_string(),
                        "arrow.opaque".to_string(),
                    ),
                    ("custom_meta".to_string(), "custom_value".to_string()),
                ]
                .into(),
            ),
        );

        for use_try_cast in [false, true] {
            // For error labelling
            let cast_name = if use_try_cast { "TRY_CAST" } else { "CAST" };

            let cast_expr = if use_try_cast {
                Expr::TryCast(TryCast::new_from_field(
                    Box::new(col("a")),
                    Arc::clone(&target_field),
                ))
            } else {
                Expr::Cast(Cast::new_from_field(
                    Box::new(col("a")),
                    Arc::clone(&target_field),
                ))
            };

            let physical = lower_cast_expr(&cast_expr, &schema)?;

            // Extract common fields - both CastExpr and TryCastExpr have these
            let (cast_type, target_metadata, target_nullable) = if use_try_cast {
                let cast = as_planner_try_cast(&physical);
                (cast.cast_type(), cast.target_metadata(), None)
            } else {
                let cast = as_planner_cast(&physical);
                (
                    cast.cast_type(),
                    cast.target_metadata(),
                    cast.target_nullable(),
                )
            };

            // Verify the physical expression stores correct metadata (same for both)
            assert_eq!(cast_type, &DataType::Int32, "{cast_name}: cast_type");
            let target_metadata = target_metadata.expect("should have metadata");
            assert_eq!(
                target_metadata.get(EXTENSION_TYPE_NAME_KEY),
                Some(&"arrow.opaque".to_string()),
                "{cast_name}: extension type name"
            );

            // Only CastExpr tracks target_nullable (TryCast is always nullable)
            if !use_try_cast {
                assert_eq!(target_nullable, Some(true), "{cast_name}: target_nullable");
            }

            // return_field should have all target metadata (exact semantics)
            let returned = physical.return_field(&schema)?;
            assert_eq!(
                returned.metadata().get(EXTENSION_TYPE_NAME_KEY),
                Some(&"arrow.opaque".to_string()),
                "{cast_name}: return_field extension type name"
            );
            // All target metadata should propagate with exact semantics
            assert_eq!(
                returned.metadata().get("custom_meta"),
                Some(&"custom_value".to_string()),
                "{cast_name}: All target metadata should propagate with exact semantics"
            );
            assert!(
                physical.nullable(&schema)?,
                "{cast_name}: should be nullable"
            );
        }

        Ok(())
    }

    /// Test that deeply nested expressions do not cause a stack overflow.
    ///
    /// This test only runs when the `recursive_protection` feature is enabled,
    /// as it would overflow the stack otherwise.
    #[test]
    #[cfg_attr(not(feature = "recursive_protection"), ignore)]
    fn test_deeply_nested_binary_expr() -> Result<()> {
        // Create a deeply nested binary expression tree: ((((a + a) + a) + a) + ... )
        // With 1000 levels of nesting, this would overflow the stack without recursion protection.
        let depth = 1000;

        let mut expr = col("a");
        for _ in 0..depth {
            expr = Expr::BinaryExpr(BinaryExpr {
                left: Box::new(expr),
                op: Operator::Plus,
                right: Box::new(col("a")),
            });
        }

        let schema = Schema::new(vec![Field::new("a", DataType::Int32, false)]);
        let df_schema = DFSchema::try_from(schema)?;

        // This should not stack overflow
        let _physical_expr = create_physical_expr(
            &expr,
            &df_schema,
            &ExecutionProps::new(),
            &PhysicalPlanningContext::default(),
            &Funding::default(),
        )?;

        Ok(())
    }
}
