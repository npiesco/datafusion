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

//! Optimizing within a [`Funding`]: what the optimizer and the analyzer copy, unshare and
//! free of a plan, charged before it is allocated and given back as it is freed.
//!
//! A plan's `Box` and `Arc` children are rewritten in place by [`TreeNode`] traversals: the
//! block a child was in is reused for what replaces it, so a traversal over a plan whose
//! `Arc` children are each held once allocates nothing (an `Expr` child list is collected in
//! place, into its own block). Only a shared `Arc` is copied, so before such a traversal
//! the plan is [`unshare`]d: each shared child copied, its copy charged first
//! ([`LogicalPlan`]'s [`CloneBytes`]). What a freed plan or expression gives
//! back is counted by [`Freed`]: the blocks dropping it frees (its lists, boxes and texts,
//! and each plan it held alone).
//!
//! What cannot be counted exactly is refused rather than guessed: a node whose copy
//! allocates blocks that cannot be counted (a statement, a DDL or DML node, an `EXPLAIN`, a
//! range partitioning, a table scan carrying statistics requests) is refused its copy.
//! Schemas, fields, functions and table names are shared, not copied. A schema, field or
//! table name is freed with what holds it when that is its last owner: what charged for it
//! is then given back with the plan that held it last.
//!
//! What is freed is given back through the handle the plan was charged through: a plan
//! handed to the optimizer is charged on the handle it optimizes within.

use std::borrow::Borrow;
use std::collections::hash_map::DefaultHasher;
use std::hash::{BuildHasher, BuildHasherDefault, Hash};
use std::sync::Arc;

use arrow::compute::can_cast_types;
use arrow::datatypes::{DataType, Field, FieldRef, Schema};
use datafusion_common::cloned::CloneBytes;
use datafusion_common::funding::{Funding, Held, arc_bytes, arc_slice_bytes};
use datafusion_common::tree_node::{Transformed, TreeNode, TreeNodeRecursion};
use datafusion_common::{
    Column, DFSchema, DFSchemaRef, DataFusionError, ExprSchema, RecursionUnnestOption,
    Result, ScalarValue, Spans, TableReference, internal_err, plan_err,
};
use datafusion_expr::dml::CopyTo;
use datafusion_expr::expr::{
    AggregateFunction, AggregateFunctionParams, Alias, Between, BinaryExpr, Case, Cast,
    Exists, GroupingSet, HigherOrderFunction, InList, InSubquery, Lambda, LambdaVariable,
    Like, Placeholder, ScalarFunction, SetComparison, Sort, TryCast,
    Unnest as UnnestExpr, WindowFunction, WindowFunctionParams,
};
use datafusion_expr::expr_rewriter::normalize_col;
use datafusion_expr::logical_plan::{
    Aggregate, Analyze, ColumnUnnestList, CreateMemoryTable, CreateView, DdlStatement,
    Distinct, DistinctOn, DmlStatement, EmptyRelation, Execute, Explain, Extension,
    Filter, Join, JoinConstraint, Limit, LogicalPlan, MergeIntoAction, Partitioning,
    Prepare, Projection, RecursiveQuery, Repartition, Sort as SortPlan, Statement,
    Subquery, SubqueryAlias, TableScan, Union, Unnest, Values, Window,
};
use datafusion_expr::utils::can_hash;
use datafusion_expr::{
    Expr, ExprSchemable, Operator, WindowFrame, WindowFrameBound, WriteOp,
};

fn sum(parts: impl IntoIterator<Item = usize>) -> usize {
    parts.into_iter().fold(0, usize::saturating_add)
}

/// What a value gives back when it is dropped.
pub(crate) trait Freed {
    /// The bytes dropping it frees, of the blocks planning charges: its lists (their room),
    /// boxes and texts, and each plan, schema, field and table name it holds alone. Blocks
    /// another owner still holds are not freed with it, and functions are always shared:
    /// they are not counted.
    fn freed_bytes(&self) -> usize;
}

macro_rules! frees_nothing {
    ($($t:ty),*) => {$(
        impl Freed for $t {
            fn freed_bytes(&self) -> usize {
                0
            }
        }
    )*};
}

frees_nothing!(usize, bool);

impl Freed for String {
    fn freed_bytes(&self) -> usize {
        self.capacity()
    }
}

impl<T: Freed> Freed for Vec<T> {
    fn freed_bytes(&self) -> usize {
        self.iter().fold(
            self.capacity().saturating_mul(size_of::<T>()),
            |bytes, item| bytes.saturating_add(item.freed_bytes()),
        )
    }
}

impl<T: Freed> Freed for Box<T> {
    fn freed_bytes(&self) -> usize {
        size_of::<T>().saturating_add(self.as_ref().freed_bytes())
    }
}

impl<T: Freed> Freed for Option<T> {
    fn freed_bytes(&self) -> usize {
        self.as_ref().map_or(0, Freed::freed_bytes)
    }
}

impl<A: Freed, B: Freed> Freed for (A, B) {
    fn freed_bytes(&self) -> usize {
        sum([self.0.freed_bytes(), self.1.freed_bytes()])
    }
}

impl Freed for Arc<LogicalPlan> {
    /// Held alone, its block and everything its plan holds; shared, nothing is freed.
    fn freed_bytes(&self) -> usize {
        if Arc::strong_count(self) == 1 && Arc::weak_count(self) == 0 {
            sum([arc_bytes::<LogicalPlan>(), self.as_ref().freed_bytes()])
        } else {
            0
        }
    }
}

impl Freed for Spans {
    fn freed_bytes(&self) -> usize {
        self.0
            .capacity()
            .saturating_mul(size_of::<datafusion_common::Span>())
    }
}

impl Freed for Column {
    /// Its name and spans, and its table name's texts it holds alone.
    fn freed_bytes(&self) -> usize {
        sum([
            self.relation.freed_bytes(),
            self.name.freed_bytes(),
            self.spans.freed_bytes(),
        ])
    }
}

/// Whether `arc` is its block's last owner: dropping it frees the block.
fn last_owner<T: ?Sized>(arc: &Arc<T>) -> bool {
    Arc::strong_count(arc) == 1 && Arc::weak_count(arc) == 0
}

/// The counted block of `len` items of `T` that a live block of theirs is in: it was
/// allocated, so it does not overflow.
fn slice_block<T>(len: usize) -> usize {
    arc_slice_bytes::<T>(len).unwrap_or(usize::MAX)
}

impl Freed for Arc<str> {
    /// Held alone, its counted block.
    fn freed_bytes(&self) -> usize {
        if last_owner(self) {
            slice_block::<u8>(self.len())
        } else {
            0
        }
    }
}

impl Freed for TableReference {
    /// Each of its texts it holds alone.
    fn freed_bytes(&self) -> usize {
        match self {
            TableReference::Bare { table } => table.freed_bytes(),
            TableReference::Partial { schema, table } => {
                sum([schema.freed_bytes(), table.freed_bytes()])
            }
            TableReference::Full {
                catalog,
                schema,
                table,
            } => sum([
                catalog.freed_bytes(),
                schema.freed_bytes(),
                table.freed_bytes(),
            ]),
        }
    }
}

impl Freed for FieldRef {
    /// Held alone, its block, its name and what its type holds. Its metadata map is copied
    /// uncharged: its table is not counted.
    fn freed_bytes(&self) -> usize {
        if last_owner(self) {
            sum([
                arc_bytes::<Field>(),
                self.name().capacity(),
                self.data_type().freed_bytes(),
            ])
        } else {
            0
        }
    }
}

impl Freed for Arc<Schema> {
    /// Held alone, its block, its list of fields (an Arrow schema planning makes has a list
    /// of its own) and each field it holds alone. Its metadata map is copied uncharged: its
    /// table is not counted.
    fn freed_bytes(&self) -> usize {
        if last_owner(self) {
            self.fields().iter().fold(
                sum([
                    arc_bytes::<Schema>(),
                    slice_block::<FieldRef>(self.fields().len()),
                ]),
                |bytes, field| bytes.saturating_add(field.freed_bytes()),
            )
        } else {
            0
        }
    }
}

impl Freed for DFSchema {
    /// Its Arrow schema held alone, its list of qualifiers with each one's texts it holds
    /// alone, and its functional dependencies' lists.
    fn freed_bytes(&self) -> usize {
        self.iter().fold(
            sum([
                self.inner().freed_bytes(),
                size_of::<Option<TableReference>>().saturating_mul(self.fields().len()),
                self.functional_dependencies().clone_bytes(),
            ]),
            |bytes, (qualifier, _)| {
                bytes.saturating_add(qualifier.map_or(0, Freed::freed_bytes))
            },
        )
    }
}

impl Freed for DFSchemaRef {
    /// Held alone, its block and what its schema holds.
    fn freed_bytes(&self) -> usize {
        if last_owner(self) {
            sum([arc_bytes::<DFSchema>(), self.as_ref().freed_bytes()])
        } else {
            0
        }
    }
}

impl Freed for DataType {
    fn freed_bytes(&self) -> usize {
        match self {
            DataType::Dictionary(key, value) => {
                sum([key.freed_bytes(), value.freed_bytes()])
            }
            // Every other type holds its parts shared, or inline.
            _ => 0,
        }
    }
}

impl Freed for ScalarValue {
    fn freed_bytes(&self) -> usize {
        match self {
            ScalarValue::Utf8(Some(text))
            | ScalarValue::Utf8View(Some(text))
            | ScalarValue::LargeUtf8(Some(text)) => text.capacity(),
            ScalarValue::Binary(Some(bytes))
            | ScalarValue::BinaryView(Some(bytes))
            | ScalarValue::LargeBinary(Some(bytes))
            | ScalarValue::FixedSizeBinary(_, Some(bytes)) => bytes.capacity(),
            ScalarValue::Union(Some((_, value)), _, _) => value.freed_bytes(),
            ScalarValue::Dictionary(key_type, value) => {
                sum([key_type.freed_bytes(), value.freed_bytes()])
            }
            ScalarValue::RunEndEncoded(_, _, value) => value.freed_bytes(),
            // Every other value is inline, or held in shared blocks.
            _ => 0,
        }
    }
}

impl Freed for Sort {
    fn freed_bytes(&self) -> usize {
        self.expr.freed_bytes()
    }
}

impl Freed for WindowFrameBound {
    fn freed_bytes(&self) -> usize {
        match self {
            WindowFrameBound::Preceding(value) | WindowFrameBound::Following(value) => {
                value.freed_bytes()
            }
            WindowFrameBound::CurrentRow => 0,
        }
    }
}

impl Freed for WindowFrame {
    fn freed_bytes(&self) -> usize {
        sum([self.start_bound.freed_bytes(), self.end_bound.freed_bytes()])
    }
}

impl Freed for Subquery {
    fn freed_bytes(&self) -> usize {
        sum([
            self.subquery.freed_bytes(),
            self.outer_ref_columns.freed_bytes(),
            self.spans.freed_bytes(),
        ])
    }
}

impl Freed for WindowFunction {
    fn freed_bytes(&self) -> usize {
        let WindowFunctionParams {
            args,
            partition_by,
            order_by,
            window_frame,
            filter,
            ..
        } = &self.params;
        sum([
            args.freed_bytes(),
            partition_by.freed_bytes(),
            order_by.freed_bytes(),
            window_frame.freed_bytes(),
            filter.freed_bytes(),
        ])
    }
}

impl Freed for Expr {
    fn freed_bytes(&self) -> usize {
        match self {
            Expr::Alias(Alias {
                expr,
                relation,
                name,
                ..
            }) => sum([
                expr.freed_bytes(),
                relation.freed_bytes(),
                name.freed_bytes(),
            ]),
            Expr::Column(column) => column.freed_bytes(),
            Expr::OuterReferenceColumn(field, column) => {
                sum([field.freed_bytes(), column.freed_bytes()])
            }
            Expr::ScalarVariable(field, names) => {
                sum([field.freed_bytes(), names.freed_bytes()])
            }
            Expr::Literal(value, _) => value.freed_bytes(),
            Expr::BinaryExpr(BinaryExpr { left, right, .. }) => {
                sum([left.freed_bytes(), right.freed_bytes()])
            }
            Expr::Like(Like { expr, pattern, .. })
            | Expr::SimilarTo(Like { expr, pattern, .. }) => {
                sum([expr.freed_bytes(), pattern.freed_bytes()])
            }
            Expr::Not(expr)
            | Expr::IsNotNull(expr)
            | Expr::IsNull(expr)
            | Expr::IsTrue(expr)
            | Expr::IsFalse(expr)
            | Expr::IsUnknown(expr)
            | Expr::IsNotTrue(expr)
            | Expr::IsNotFalse(expr)
            | Expr::IsNotUnknown(expr)
            | Expr::Negative(expr)
            | Expr::Unnest(UnnestExpr { expr, .. }) => expr.freed_bytes(),
            Expr::Cast(Cast { expr, field }) | Expr::TryCast(TryCast { expr, field }) => {
                sum([expr.freed_bytes(), field.freed_bytes()])
            }
            Expr::Between(Between {
                expr, low, high, ..
            }) => sum([expr.freed_bytes(), low.freed_bytes(), high.freed_bytes()]),
            Expr::Case(Case {
                expr,
                when_then_expr,
                else_expr,
            }) => sum([
                expr.freed_bytes(),
                when_then_expr.freed_bytes(),
                else_expr.freed_bytes(),
            ]),
            Expr::ScalarFunction(ScalarFunction { args, .. })
            | Expr::HigherOrderFunction(HigherOrderFunction { args, .. }) => {
                args.freed_bytes()
            }
            Expr::AggregateFunction(AggregateFunction {
                params:
                    AggregateFunctionParams {
                        args,
                        filter,
                        order_by,
                        ..
                    },
                ..
            }) => sum([
                args.freed_bytes(),
                filter.freed_bytes(),
                order_by.freed_bytes(),
            ]),
            Expr::WindowFunction(window) => window.freed_bytes(),
            Expr::InList(InList { expr, list, .. }) => {
                sum([expr.freed_bytes(), list.freed_bytes()])
            }
            Expr::Exists(Exists { subquery, .. }) | Expr::ScalarSubquery(subquery) => {
                subquery.freed_bytes()
            }
            Expr::InSubquery(InSubquery { expr, subquery, .. })
            | Expr::SetComparison(SetComparison { expr, subquery, .. }) => {
                sum([expr.freed_bytes(), subquery.freed_bytes()])
            }
            // Its options' block; what present options hold is parsed SQL, never funded
            // ([`CloneBytes`] refuses copying it).
            #[expect(deprecated)]
            Expr::Wildcard { options, .. } => size_of_val(options.as_ref()),
            Expr::GroupingSet(GroupingSet::Rollup(exprs))
            | Expr::GroupingSet(GroupingSet::Cube(exprs)) => exprs.freed_bytes(),
            Expr::GroupingSet(GroupingSet::GroupingSets(lists)) => lists.freed_bytes(),
            Expr::Placeholder(Placeholder { id, field }) => {
                sum([id.freed_bytes(), field.freed_bytes()])
            }
            Expr::Lambda(Lambda { params, body }) => {
                sum([params.freed_bytes(), body.freed_bytes()])
            }
            Expr::LambdaVariable(LambdaVariable {
                name, field, spans, ..
            }) => sum([name.freed_bytes(), field.freed_bytes(), spans.freed_bytes()]),
        }
    }
}

impl Freed for ColumnUnnestList {
    fn freed_bytes(&self) -> usize {
        self.output_column.freed_bytes()
    }
}

impl Freed for RecursionUnnestOption {
    fn freed_bytes(&self) -> usize {
        sum([
            self.input_column.freed_bytes(),
            self.output_column.freed_bytes(),
        ])
    }
}

impl Freed for LogicalPlan {
    /// What dropping this node frees: what it holds, and each input it holds alone.
    fn freed_bytes(&self) -> usize {
        let own = match self {
            LogicalPlan::Projection(Projection { expr, schema, .. }) => {
                sum([expr.freed_bytes(), schema.freed_bytes()])
            }
            LogicalPlan::Filter(Filter { predicate, .. }) => predicate.freed_bytes(),
            LogicalPlan::Window(Window {
                window_expr,
                schema,
                ..
            }) => sum([window_expr.freed_bytes(), schema.freed_bytes()]),
            LogicalPlan::Aggregate(Aggregate {
                group_expr,
                aggr_expr,
                schema,
                ..
            }) => sum([
                group_expr.freed_bytes(),
                aggr_expr.freed_bytes(),
                schema.freed_bytes(),
            ]),
            LogicalPlan::Sort(SortPlan { expr, .. }) => expr.freed_bytes(),
            LogicalPlan::Join(Join {
                on, filter, schema, ..
            }) => sum([on.freed_bytes(), filter.freed_bytes(), schema.freed_bytes()]),
            LogicalPlan::Repartition(Repartition {
                partitioning_scheme:
                    Partitioning::Hash(exprs, _) | Partitioning::DistributeBy(exprs),
                ..
            }) => exprs.freed_bytes(),
            LogicalPlan::Union(Union { inputs, schema }) => {
                // The inputs themselves are counted below, with every other node's.
                sum([
                    inputs
                        .capacity()
                        .saturating_mul(size_of::<Arc<LogicalPlan>>()),
                    schema.freed_bytes(),
                ])
            }
            LogicalPlan::TableScan(TableScan {
                table_name,
                projection,
                projected_schema,
                filters,
                ..
            }) => sum([
                table_name.freed_bytes(),
                projection.freed_bytes(),
                projected_schema.freed_bytes(),
                filters.freed_bytes(),
            ]),
            LogicalPlan::EmptyRelation(EmptyRelation { schema, .. }) => {
                schema.freed_bytes()
            }
            LogicalPlan::SubqueryAlias(SubqueryAlias { alias, schema, .. }) => {
                sum([alias.freed_bytes(), schema.freed_bytes()])
            }
            LogicalPlan::Subquery(Subquery {
                outer_ref_columns,
                spans,
                ..
            }) => sum([outer_ref_columns.freed_bytes(), spans.freed_bytes()]),
            LogicalPlan::Limit(Limit { skip, fetch, .. }) => {
                sum([skip.freed_bytes(), fetch.freed_bytes()])
            }
            LogicalPlan::Values(Values { values, schema }) => {
                sum([values.freed_bytes(), schema.freed_bytes()])
            }
            LogicalPlan::Distinct(Distinct::On(DistinctOn {
                on_expr,
                select_expr,
                sort_expr,
                schema,
                ..
            })) => sum([
                on_expr.freed_bytes(),
                select_expr.freed_bytes(),
                sort_expr.freed_bytes(),
                schema.freed_bytes(),
            ]),
            LogicalPlan::Unnest(Unnest {
                exec_columns,
                list_type_columns,
                struct_type_columns,
                dependency_indices,
                options,
                schema,
                ..
            }) => sum([
                exec_columns.freed_bytes(),
                list_type_columns.freed_bytes(),
                struct_type_columns.freed_bytes(),
                dependency_indices.freed_bytes(),
                options.recursions.freed_bytes(),
                schema.freed_bytes(),
            ]),
            LogicalPlan::RecursiveQuery(RecursiveQuery { name, .. }) => {
                name.freed_bytes()
            }
            LogicalPlan::Statement(Statement::Execute(Execute {
                parameters, ..
            })) => parameters.freed_bytes(),
            _ => 0,
        };
        let mut inputs = 0;
        for_each_input_arc(self, &mut |input| {
            inputs = sum([inputs, input.freed_bytes()]);
        });
        sum([own, inputs])
    }
}

/// A copy of `plan`'s node (its inputs shared), what the copy allocates charged before it
/// is made.
pub(crate) fn cloned_plan(funding: &Funding, plan: &LogicalPlan) -> Result<LogicalPlan> {
    funding.charge(plan.clone_bytes())?;
    Ok(plan.clone())
}

/// `value` dropped: what it frees is given back.
pub(crate) fn freed<T: Freed>(funding: &Funding, value: T) {
    let bytes = value.freed_bytes();
    drop(value);
    funding.discharge(bytes);
}

/// `plan` with its schema recomputed ([`LogicalPlan::recompute_schema`]): the schema it
/// replaces freed and given back.
pub(crate) fn recomputed(funding: &Funding, plan: LogicalPlan) -> Result<LogicalPlan> {
    let replaced = Arc::clone(plan.schema());
    let plan = plan.recompute_schema()?;
    freed(funding, replaced);
    Ok(plan)
}

/// A value made through a [`Funding`], what it holds (`bytes`) charged for it: given back
/// once it is freed with this.
pub(crate) struct Released<T> {
    value: Option<T>,
    bytes: usize,
    funding: Funding,
}

impl<T> Released<T> {
    /// `value`, `bytes` charged for it.
    pub(crate) fn new(funding: &Funding, value: T, bytes: usize) -> Self {
        Self {
            value: Some(value),
            bytes,
            funding: funding.clone(),
        }
    }
}

impl<T> std::ops::Deref for Released<T> {
    type Target = T;

    fn deref(&self) -> &T {
        // Present until it is dropped.
        self.value.as_ref().unwrap_or_else(|| unreachable!())
    }
}

impl<T> Drop for Released<T> {
    fn drop(&mut self) {
        drop(self.value.take());
        self.funding.discharge(self.bytes);
    }
}

/// A column of `name` under `qualifier`, its name charged before it is copied.
pub(crate) fn column_expr(
    funding: &Funding,
    qualifier: Option<&TableReference>,
    name: &str,
) -> Result<Expr> {
    funding.charge(sum([qualifier.cloned().clone_bytes(), name.len()]))?;
    Ok(Expr::Column(Column::new(qualifier.cloned(), name)))
}

/// A column for each of `schema`'s fields, in a list of their number: the list and each
/// name charged before they are made.
pub(crate) fn schema_columns(funding: &Funding, schema: &DFSchema) -> Result<Vec<Expr>> {
    let mut exprs = funding.vec_with_capacity(schema.fields().len())?;
    for (qualifier, field) in schema.iter() {
        exprs.push(column_expr(funding, qualifier, field.name())?);
    }
    Ok(exprs)
}

/// The columns of a USING join's keys.
fn using_keys(on: &[(Expr, Expr)]) -> impl Iterator<Item = &Column> + Clone {
    on.iter()
        .flat_map(|(left, right)| [left, right])
        .filter_map(Expr::get_as_join_column)
}

/// Whether `f` is true of the keys of any USING join in `plan`, as
/// [`LogicalPlan::using_columns`] finds them (subqueries included), without collecting
/// them; a key that is not a column is the error it reports.
fn any_using_join(
    funding: &Funding,
    plan: &LogicalPlan,
    f: &mut dyn FnMut(&[(Expr, Expr)]) -> bool,
) -> Result<bool> {
    let mut found = false;
    apply_plan(funding, plan, true, &mut |node| {
        if let LogicalPlan::Join(Join {
            join_constraint: JoinConstraint::Using,
            on,
            ..
        }) = node
        {
            for key in on.iter().flat_map(|(left, right)| [left, right]) {
                if key.get_as_join_column().is_none() {
                    return internal_err!(
                        "Invalid join key. Expected column, found {key:?}"
                    );
                }
            }
            found |= f(on);
        }
        Ok(TreeNodeRecursion::Continue)
    })?;
    Ok(found)
}

/// `expr` with its unqualified columns qualified as [`normalize_col`] qualifies them over
/// `plan`, in place: each qualifier charged before it is copied.
pub(crate) fn normalize_in_place(
    funding: &Funding,
    plan: &LogicalPlan,
    expr: &mut Expr,
) -> Result<()> {
    if let Expr::Column(column) = expr {
        return normalize_column(funding, plan, column);
    }
    for_each_child_expr_mut(expr, &mut |child| normalize_in_place(funding, plan, child))
}

/// `column` qualified as [`Column::normalize_with_schemas_and_ambiguity_check`] qualifies
/// it over `plan`'s schema, then over the schemas of the inputs it falls back to
/// ([`LogicalPlan::fallback_normalize_schemas`], listed charged); its spans dropped as
/// that makes it anew. A column it cannot qualify is reported as [`normalize_col`]
/// reports it.
fn normalize_column(
    funding: &Funding,
    plan: &LogicalPlan,
    column: &mut Column,
) -> Result<()> {
    if column.relation.is_some() {
        return Ok(());
    }
    let mut fallback: Held<Vec<&DFSchema>> = funding.held();
    if matches!(
        plan,
        LogicalPlan::Window(_)
            | LogicalPlan::Projection(_)
            | LogicalPlan::Aggregate(_)
            | LogicalPlan::Unnest(_)
            | LogicalPlan::Join(_)
    ) {
        apply_inputs(plan, &mut |input| {
            fallback.push(input.schema().as_ref())?;
            Ok(TreeNodeRecursion::Continue)
        })?;
    }
    let schema: &DFSchema = plan.schema();
    let name = column.name.as_str();
    for level in [std::slice::from_ref(&schema), &fallback[..]] {
        let named = || {
            level
                .iter()
                .flat_map(|schema| schema.iter())
                .filter(|(_, field)| field.name() == name)
        };
        let mut matches = named();
        let Some((qualifier, _)) = matches.next() else {
            continue;
        };
        if matches.next().is_some()
            && !any_using_join(funding, plan, &mut |on| {
                named().all(|(qualifier, _)| {
                    using_keys(on)
                        .any(|key| key.relation.as_ref() == qualifier && key.name == name)
                })
            })?
        {
            break;
        }
        funding.charge(qualifier.map_or(0, CloneBytes::clone_bytes))?;
        column.relation = qualifier.cloned();
        freed(funding, std::mem::take(&mut column.spans));
        return Ok(());
    }
    normalize_col(Expr::Column(column.clone()), plan)?;
    internal_err!("{column} is qualified over the plan's schemas by normalize_col alone")
}

/// A projection of `schema`'s columns over `input`, its list charged before it is made.
pub(crate) fn schema_projection(
    funding: &Funding,
    input: Arc<LogicalPlan>,
    schema: DFSchemaRef,
) -> Result<LogicalPlan> {
    let expr = schema_columns(funding, &schema)?;
    Ok(LogicalPlan::Projection(Projection::try_new_with_schema(
        expr, input, schema,
    )?))
}

/// `expr` cast to `data_type` (a nullable field of it), its box and field charged first.
pub(crate) fn nullable_cast(
    funding: &Funding,
    expr: Expr,
    data_type: &DataType,
) -> Result<Expr> {
    funding.charge(sum([
        arc_bytes::<Field>(),
        data_type.clone_bytes(),
        size_of::<Expr>(),
    ]))?;
    Ok(Expr::Cast(Cast::new(Box::new(expr), data_type.clone())))
}

/// What holds an expression's place while it is moved out to be rebuilt; it allocates
/// nothing.
pub(crate) fn vacated() -> Expr {
    Expr::Literal(ScalarValue::Null, None)
}

/// Whether `expr` has type `data_type` over `schema`: a column's type compared in place,
/// any other expression's through [`ExprSchemable::get_type`].
fn has_type(expr: &Expr, schema: &dyn ExprSchema, data_type: &DataType) -> Result<bool> {
    Ok(match expr {
        Expr::Column(column) => schema.data_type(column)? == data_type,
        _ => expr.get_type(schema)? == *data_type,
    })
}

/// `expr` cast to `data_type` over `schema` as [`ExprSchemable::cast_to`] casts it:
/// itself if it already has the type, a scalar subquery's plan cast under a projection,
/// any other expression boxed in a cast ([`nullable_cast`]); what each makes charged
/// first.
pub(crate) fn cast_to(
    funding: &Funding,
    expr: Expr,
    data_type: &DataType,
    schema: &dyn ExprSchema,
) -> Result<Expr> {
    let computed;
    let this_type = match &expr {
        Expr::Column(column) => schema.data_type(column)?,
        _ => {
            computed = expr.get_type(schema)?;
            &computed
        }
    };
    if this_type == data_type {
        return Ok(expr);
    }
    let can_cast = match (this_type, data_type) {
        // Always allow struct-to-struct casts; field matching happens at runtime
        (DataType::Struct(_), DataType::Struct(_)) => true,
        _ => can_cast_types(this_type, data_type),
    };
    if !can_cast {
        return plan_err!("Cannot automatically convert {this_type} to {data_type}");
    }
    match expr {
        Expr::ScalarSubquery(subquery) => Ok(Expr::ScalarSubquery(cast_subquery(
            funding, subquery, data_type,
        )?)),
        expr => nullable_cast(funding, expr, data_type),
    }
}

/// `subquery` with its one column cast to `data_type`, as
/// [`datafusion_expr::expr_schema::cast_subquery`] casts it: its projection's expression
/// copied and cast under a projection over the same input, or a cast column projected over
/// any other plan; the projection and its block charged first, the plan it replaces given
/// back.
pub(crate) fn cast_subquery(
    funding: &Funding,
    subquery: Subquery,
    data_type: &DataType,
) -> Result<Subquery> {
    if subquery.subquery.schema().field(0).data_type() == data_type {
        return Ok(subquery);
    }
    let Subquery {
        subquery: plan,
        outer_ref_columns,
        spans,
    } = subquery;
    let projection = if let LogicalPlan::Projection(projection) = plan.as_ref() {
        let expr = cast_to(
            funding,
            funding.cloned(&projection.expr[0])?,
            data_type,
            projection.input.schema(),
        )?;
        let projection = Projection::try_new(
            funding.vec_of([expr])?,
            Arc::clone(&projection.input),
            funding,
        )?;
        freed(funding, plan);
        projection
    } else {
        let (qualifier, field) = plan.schema().qualified_field(0);
        let expr = cast_to(
            funding,
            column_expr(funding, qualifier, field.name())?,
            data_type,
            plan.schema(),
        )?;
        Projection::try_new(funding.vec_of([expr])?, plan, funding)?
    };
    freed(funding, spans);
    Ok(Subquery {
        subquery: funding.arc(LogicalPlan::Projection(projection))?,
        outer_ref_columns,
        spans: Spans::new(),
    })
}

/// `exprs`, over `source`, cast in place to the types of `target`'s fields as
/// [`datafusion_expr::expr_rewriter::coerce_plan_expr_for_schema`] casts a projection's:
/// an alias's expression cast in its box (its qualifier and metadata dropped), a column's
/// cast aliased to its name, a wildcard left as it is.
fn coerce_exprs_in_place(
    funding: &Funding,
    exprs: &mut [Expr],
    source: &DFSchema,
    target: &DFSchema,
) -> Result<()> {
    for (idx, slot) in exprs.iter_mut().enumerate() {
        let new_type = target.field(idx).data_type();
        if has_type(slot, source, new_type)? {
            continue;
        }
        *slot = match std::mem::replace(slot, vacated()) {
            Expr::Alias(mut alias) => {
                let inner = std::mem::replace(alias.expr.as_mut(), vacated());
                *alias.expr = cast_to(funding, inner, new_type, source)?;
                alias.relation = None;
                alias.metadata = None;
                Expr::Alias(alias)
            }
            #[expect(deprecated)]
            wildcard @ Expr::Wildcard { .. } => wildcard,
            Expr::Column(column) => {
                // The original name is kept, to avoid the table name being added to it
                // when not explicitly set by the query.
                let name = funding.string(column.name())?;
                let cast = cast_to(funding, Expr::Column(column), new_type, source)?;
                Expr::Alias(Alias {
                    expr: funding.boxed(cast)?,
                    relation: None,
                    name,
                    metadata: None,
                })
            }
            other => cast_to(funding, other, new_type, source)?,
        };
    }
    Ok(())
}

/// `plan` with its output cast to `schema`'s types as
/// [`datafusion_expr::expr_rewriter::coerce_plan_expr_for_schema`] casts it: a
/// projection's expressions cast in place, any other plan projected with its mismatched
/// columns cast, or left as it is when none are; what is made charged first.
pub(crate) fn coerced_for_schema(
    funding: &Funding,
    plan: LogicalPlan,
    schema: &DFSchema,
) -> Result<LogicalPlan> {
    match plan {
        LogicalPlan::Projection(Projection {
            mut expr,
            input,
            schema: replaced,
            ..
        }) => {
            coerce_exprs_in_place(funding, &mut expr, input.schema(), schema)?;
            let projection = Projection::try_new(expr, input, funding)?;
            // The projection's schema is replaced by the one planned for its casts.
            freed(funding, replaced);
            Ok(LogicalPlan::Projection(projection))
        }
        plan => {
            let source = plan.schema();
            let mismatched = |idx: usize| {
                source.field(idx).data_type() != schema.field(idx).data_type()
            };
            if !(0..source.fields().len()).any(mismatched) {
                return Ok(plan);
            }
            let mut exprs = funding.vec_with_capacity(source.fields().len())?;
            for (idx, (qualifier, field)) in source.iter().enumerate() {
                let column = column_expr(funding, qualifier, field.name())?;
                // Room for each is charged: pushing allocates nothing.
                exprs.push(if mismatched(idx) {
                    let cast =
                        cast_to(funding, column, schema.field(idx).data_type(), source)?;
                    aliased(funding, cast, None, field.name())?
                } else {
                    column
                });
            }
            let input = funding.arc(plan)?;
            Ok(LogicalPlan::Projection(Projection::try_new(
                exprs, input, funding,
            )?))
        }
    }
}

/// `expr` named `name` under `relation`, its box and name charged first.
pub(crate) fn aliased(
    funding: &Funding,
    expr: Expr,
    relation: Option<&TableReference>,
    name: &str,
) -> Result<Expr> {
    funding.charge(sum([
        size_of::<Expr>(),
        relation.cloned().clone_bytes(),
        name.len(),
    ]))?;
    Ok(Expr::Alias(Alias {
        expr: Box::new(expr),
        relation: relation.cloned(),
        name: name.to_owned(),
        metadata: None,
    }))
}

/// `left op right`, its two boxes charged before they are made.
pub(crate) fn binary(
    funding: &Funding,
    left: Expr,
    op: Operator,
    right: Expr,
) -> Result<Expr> {
    funding.charge(sum([size_of::<Expr>(), size_of::<Expr>()]))?;
    Ok(Expr::BinaryExpr(BinaryExpr::new(
        Box::new(left),
        op,
        Box::new(right),
    )))
}

/// `left AND right`, its boxes charged first.
pub(crate) fn conjoined(funding: &Funding, left: Expr, right: Expr) -> Result<Expr> {
    binary(funding, left, Operator::And, right)
}

/// `left OR right`, its boxes charged first.
pub(crate) fn disjoined(funding: &Funding, left: Expr, right: Expr) -> Result<Expr> {
    binary(funding, left, Operator::Or, right)
}

/// `NOT expr` as [`std::ops::Not`] makes it: a negatable expression flipped in place, any
/// other boxed, its box charged first.
pub(crate) fn negated(funding: &Funding, expr: Expr) -> Result<Expr> {
    Ok(match expr {
        Expr::Like(_)
        | Expr::SimilarTo(_)
        | Expr::InList(_)
        | Expr::Exists(_)
        | Expr::InSubquery(_) => std::ops::Not::not(expr),
        _ => Expr::Not(funding.boxed(expr)?),
    })
}

/// What `keep` boxes, moved out of its box, and `dropped` freed: the box and what `dropped`
/// held given back.
pub(crate) fn kept<T: Freed>(funding: &Funding, keep: Box<Expr>, dropped: T) -> Expr {
    freed(funding, dropped);
    funding.unboxed(keep)
}

/// `made`, in place of `old`: what `old` held given back.
pub(crate) fn replaced<T: Freed>(funding: &Funding, old: T, made: Expr) -> Expr {
    freed(funding, old);
    made
}

/// Whether `f` holds for any conjunct of `expr` (`A AND B AND C` gives `A`, `B` and `C`,
/// aliases looked through), in order, found without allocating.
pub(crate) fn any_conjunct<'a>(
    expr: &'a Expr,
    f: &mut impl FnMut(&'a Expr) -> bool,
) -> bool {
    match expr {
        Expr::BinaryExpr(BinaryExpr {
            left,
            op: Operator::And,
            right,
        }) => any_conjunct(left, f) || any_conjunct(right, f),
        Expr::Alias(Alias { expr, .. }) => any_conjunct(expr, f),
        other => f(other),
    }
}

/// The conjuncts of `expr` ([`any_conjunct`]), in a list charged as it grows.
pub(crate) fn conjuncts<'a>(
    funding: &Funding,
    expr: &'a Expr,
) -> Result<Held<Vec<&'a Expr>>> {
    let mut list: Held<Vec<&'a Expr>> = funding.held();
    let mut refused = Ok(());
    any_conjunct(expr, &mut |conjunct| {
        refused = list.push(conjunct);
        refused.is_err()
    });
    refused.map(|()| list)
}

/// The conjuncts of `expr`, moved out of it: the boxes and alias names that held them
/// given back, the list they are moved into charged as it grows.
pub(crate) fn conjuncts_owned(funding: &Funding, expr: Expr) -> Result<Held<Vec<Expr>>> {
    fn push(funding: &Funding, expr: Expr, list: &mut Held<Vec<Expr>>) -> Result<()> {
        match expr {
            Expr::BinaryExpr(BinaryExpr {
                left,
                op: Operator::And,
                right,
            }) => {
                push(funding, funding.unboxed(left), list)?;
                push(funding, funding.unboxed(right), list)
            }
            Expr::Alias(Alias { expr, name, .. }) => {
                freed(funding, name);
                push(funding, funding.unboxed(expr), list)
            }
            other => list.push(other),
        }
    }
    let mut list = funding.held();
    push(funding, expr, &mut list)?;
    Ok(list)
}

/// `exprs` joined by `op` from the left (`(A op B) op C`), each join's boxes charged first;
/// `None` if there are none. The list's block is given back with it.
pub(crate) fn joined(
    funding: &Funding,
    mut exprs: Held<Vec<Expr>>,
    op: Operator,
) -> Result<Option<Expr>> {
    let mut joined = None;
    for expr in exprs.drain(..) {
        joined = Some(match joined {
            None => expr,
            Some(acc) => binary(funding, acc, op, expr)?,
        });
    }
    Ok(joined)
}

/// Whether `expr` refers to any column, and whether every column it refers to is in
/// `schema`: [`Expr::column_refs`] checked in place, nothing collected.
fn columns_in(expr: &Expr, schema: &DFSchema) -> Result<(bool, bool)> {
    let (mut any, mut all) = (false, true);
    expr.apply(|expr| {
        if let Expr::Column(column) = expr {
            any = true;
            all &= schema.is_column_from_schema(column);
        }
        Ok(TreeNodeRecursion::Continue)
    })?;
    Ok((any, all))
}

/// How `left = right` joins `left_schema` to `right_schema`, as
/// [`datafusion_expr::utils::find_valid_equijoin_key_pair`] finds it without collecting
/// columns or copying keys: `Some(false)` as written, `Some(true)` with the sides swapped,
/// `None` if they are not a key pair.
pub(crate) fn equijoin_sides(
    left: &Expr,
    right: &Expr,
    left_schema: &DFSchema,
    right_schema: &DFSchema,
) -> Result<Option<bool>> {
    let (left_any, left_in_left) = columns_in(left, left_schema)?;
    let (right_any, right_in_right) = columns_in(right, right_schema)?;
    if !left_any || !right_any {
        return Ok(None);
    }
    if left_in_left && right_in_right {
        return Ok(Some(false));
    }
    let (_, right_in_left) = columns_in(right, left_schema)?;
    let (_, left_in_right) = columns_in(left, right_schema)?;
    Ok((right_in_left && left_in_right).then_some(true))
}

/// Whether `expr` has a hashable type in `schema`: a column's type read in place, any other
/// expression's through [`ExprSchemable::get_type`].
pub(crate) fn hashable(expr: &Expr, schema: &DFSchema) -> Result<bool> {
    Ok(match expr {
        Expr::Column(column) => can_hash(schema.data_type(column)?),
        _ => can_hash(&expr.get_type(schema)?),
    })
}

/// What a null of `data_type` boxes: a dictionary's key type and value, a run-end encoded
/// value.
fn null_bytes(data_type: &DataType) -> usize {
    match data_type {
        DataType::Dictionary(key_type, value_type) => sum([
            key_type.clone_bytes(),
            size_of::<ScalarValue>(),
            null_bytes(value_type),
        ]),
        DataType::RunEndEncoded(_, value_field) => sum([
            size_of::<ScalarValue>(),
            null_bytes(value_field.data_type()),
        ]),
        _ => 0,
    }
}

/// A null of `data_type`, its boxes charged before they are made. A nested type's null is
/// an array arrow allocates, held shared ([`Freed`] for [`ScalarValue`]).
pub(crate) fn null_of(funding: &Funding, data_type: &DataType) -> Result<ScalarValue> {
    funding.charge(null_bytes(data_type))?;
    ScalarValue::try_new_null(data_type)
}

/// `list` with only the first of the items equal to each other, the rest freed and given
/// back, found through a [`HashIndex`].
pub(crate) fn deduplicated<T: Hash + Eq + Freed>(
    funding: &Funding,
    list: &mut Vec<T>,
) -> Result<()> {
    let mut firsts = funding.held::<Vec<bool>>();
    {
        let index = HashIndex::new(funding, list)?;
        firsts.reserve_exact(list.len())?;
        // Room for each is charged: extending allocates nothing.
        firsts.extend((0..list.len()).map(|at| index.is_first(at)));
    }
    let mut at = 0;
    list.retain(|item| {
        let first = firsts[at];
        at += 1;
        if !first {
            funding.discharge(item.freed_bytes());
        }
        first
    });
    Ok(())
}

pub(crate) fn hash_of<T: Hash + ?Sized>(item: &T) -> u64 {
    BuildHasherDefault::<DefaultHasher>::default().hash_one(item)
}

/// The items of a list found by their hashes, in place of a hash set: each item's hash
/// with where it is, sorted, in a list of their number charged before it is made.
pub(crate) struct HashIndex<'a, T> {
    items: &'a [T],
    hashes: Held<Vec<(u64, usize)>>,
}

impl<'a, T: Hash + Eq> HashIndex<'a, T> {
    pub(crate) fn new(funding: &Funding, items: &'a [T]) -> Result<Self> {
        let mut hashes: Held<Vec<(u64, usize)>> = funding.held();
        hashes.reserve_exact(items.len())?;
        // Room for each is charged: extending allocates nothing.
        hashes.extend(
            items
                .iter()
                .enumerate()
                .map(|(at, item)| (hash_of(item), at)),
        );
        hashes.sort_unstable();
        Ok(Self { items, hashes })
    }

    /// Where the first item equal to `item` is.
    pub(crate) fn position<Q>(&self, item: &Q) -> Option<usize>
    where
        T: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        let hash = hash_of(item);
        let start = self.hashes.partition_point(|(other, _)| *other < hash);
        self.hashes[start..]
            .iter()
            .take_while(|(other, _)| *other == hash)
            .map(|(_, at)| *at)
            .find(|at| self.items[*at].borrow() == item)
    }

    pub(crate) fn contains<Q>(&self, item: &Q) -> bool
    where
        T: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.position(item).is_some()
    }

    /// How many items are equal to `item`.
    pub(crate) fn count<Q>(&self, item: &Q) -> usize
    where
        T: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        let hash = hash_of(item);
        let start = self.hashes.partition_point(|(other, _)| *other < hash);
        self.hashes[start..]
            .iter()
            .take_while(|(other, _)| *other == hash)
            .filter(|(_, at)| self.items[*at].borrow() == item)
            .count()
    }

    /// Whether the item at `at` is the first of those equal to it.
    pub(crate) fn is_first(&self, at: usize) -> bool {
        self.position(&self.items[at]) == Some(at)
    }
}

/// An empty relation of `schema`'s rows, `plan` (what it replaces) freed and given back.
pub(crate) fn emptied(
    funding: &Funding,
    plan: LogicalPlan,
    schema: DFSchemaRef,
) -> LogicalPlan {
    freed(funding, plan);
    LogicalPlan::EmptyRelation(EmptyRelation {
        produce_one_row: false,
        schema,
    })
}

/// The schema of `plan`'s inputs merged ([`merge_schema`]), found without listing them,
/// with what it holds alone charged for it: each schema merged on the way freed and given
/// back.
///
/// [`merge_schema`]: datafusion_expr::utils::merge_schema
pub(crate) fn merged_input_schema(
    funding: &Funding,
    plan: &LogicalPlan,
) -> Result<(DFSchema, usize)> {
    if let Some(input) = only_input(plan) {
        let bytes = input.schema().clone_bytes();
        return Ok((funding.cloned(input.schema().as_ref())?, bytes));
    }
    let mut merged: Option<(DFSchema, usize)> = None;
    apply_inputs(plan, &mut |input| {
        let base = merged
            .as_ref()
            .map_or(DFSchema::empty_ref().as_ref(), |(schema, _)| schema);
        let before = funding.outstanding();
        let next = base.merge(input.schema(), funding)?;
        let bytes = funding.outstanding().saturating_sub(before);
        if let Some((_, held)) = merged.replace((next, bytes)) {
            funding.discharge(held);
        }
        Ok(TreeNodeRecursion::Continue)
    })?;
    match merged {
        Some(merged) => Ok(merged),
        None => Ok((funding.cloned(DFSchema::empty_ref().as_ref())?, 0)),
    }
}

/// `held`, a schema with what it holds alone charged for it, merged with `other`
/// ([`DFSchema::merge`]): the merged schema charged before it is made, `held` then freed and
/// given back.
pub(crate) fn merged_with(
    funding: &Funding,
    held: (DFSchema, usize),
    other: &DFSchema,
) -> Result<(DFSchema, usize)> {
    let (schema, held_bytes) = held;
    let before = funding.outstanding();
    let merged = schema.merge(other, funding)?;
    let bytes = funding.outstanding().saturating_sub(before);
    drop(schema);
    funding.discharge(held_bytes);
    Ok((merged, bytes))
}

/// `schema` with each field under `qualifier` ([`DFSchema::try_from_qualified_schema`]),
/// with what it charged for itself: its qualifiers' list and its Arrow schema's block.
pub(crate) fn qualified_schema(
    funding: &Funding,
    qualifier: &TableReference,
    schema: &Schema,
) -> Result<(DFSchema, usize)> {
    let made = DFSchema::try_from_qualified_schema(qualifier.clone(), schema, funding)?;
    let bytes = sum([
        schema
            .fields()
            .len()
            .saturating_mul(size_of::<Option<TableReference>>()),
        arc_bytes::<Schema>(),
    ]);
    Ok((made, bytes))
}

/// What dropping `expr` frees, but for the plans of its subqueries: those are charged as
/// they are rewritten, on their own.
pub(crate) fn freed_bytes_outside_subqueries(expr: &Expr) -> usize {
    let mut bytes = expr.freed_bytes();
    // Visiting expressions only, never the plans of their subqueries.
    let _ = expr.apply(|e| {
        if let Expr::Exists(Exists { subquery, .. })
        | Expr::ScalarSubquery(subquery)
        | Expr::InSubquery(InSubquery { subquery, .. })
        | Expr::SetComparison(SetComparison { subquery, .. }) = e
        {
            bytes = bytes.saturating_sub(subquery.subquery.freed_bytes());
        }
        Ok(TreeNodeRecursion::Continue)
    });
    bytes
}

/// What a value held `before` and holds `after` it was changed by code that allocates on its
/// own: the difference charged or given back once it is made.
pub(crate) fn rebalanced(funding: &Funding, before: usize, after: usize) -> Result<()> {
    if after > before {
        funding.charge(after - before)
    } else {
        funding.discharge(before - after);
        Ok(())
    }
}

/// Whether `value` displays as exactly `text`, compared as it is written, nothing allocated.
pub(crate) fn displays_as(value: impl std::fmt::Display, text: &str) -> bool {
    use std::fmt::Write;
    struct Rest<'a>(&'a str);
    impl Write for Rest<'_> {
        fn write_str(&mut self, part: &str) -> std::fmt::Result {
            match self.0.strip_prefix(part) {
                Some(rest) => {
                    self.0 = rest;
                    Ok(())
                }
                None => Err(std::fmt::Error),
            }
        }
    }
    let mut rest = Rest(text);
    write!(rest, "{value}").is_ok() && rest.0.is_empty()
}

/// The one input of `plan`, if it has exactly one, found without listing its inputs.
pub(crate) fn only_input(plan: &LogicalPlan) -> Option<&LogicalPlan> {
    let mut only = None;
    let mut count = 0usize;
    let _ = apply_inputs(plan, &mut |input| {
        count += 1;
        only = Some(input);
        Ok(if count > 1 {
            TreeNodeRecursion::Stop
        } else {
            TreeNodeRecursion::Continue
        })
    });
    if count == 1 { only } else { None }
}

/// `e` with `context` added, unless it is a refusal: a refusal is passed on as it is,
/// nothing allocated for it.
pub(crate) fn with_context<C: Into<String>>(
    e: DataFusionError,
    context: impl FnOnce() -> C,
) -> DataFusionError {
    if e.refused().is_some() {
        e
    } else {
        e.context(context())
    }
}

/// The plan `arc` holds, moved out of it: held alone, its block is freed and given back;
/// shared, it is copied, the copy charged first.
pub(crate) fn unwrapped(funding: &Funding, arc: Arc<LogicalPlan>) -> Result<LogicalPlan> {
    match Arc::try_unwrap(arc) {
        Ok(plan) => {
            funding.discharge(arc_bytes::<LogicalPlan>());
            Ok(plan)
        }
        Err(shared) => cloned_plan(funding, shared.as_ref()),
    }
}

/// The plan `arc` holds, to change in place: shared, it is first copied into a block of its
/// own, the copy and its block charged before they are made.
pub(crate) fn plan_mut<'a>(
    funding: &Funding,
    arc: &'a mut Arc<LogicalPlan>,
) -> Result<&'a mut LogicalPlan> {
    if Arc::get_mut(arc).is_none() {
        funding.charge(sum([
            LogicalPlan::clone_bytes(arc),
            arc_bytes::<LogicalPlan>(),
        ]))?;
    }
    Ok(Arc::make_mut(arc))
}

/// Calls `f` on each input `plan` holds in an `Arc` of its own. An extension's inputs are
/// its own to hold: they are not visited.
pub(crate) fn for_each_input_arc<'a>(
    plan: &'a LogicalPlan,
    f: &mut dyn FnMut(&'a Arc<LogicalPlan>),
) {
    match plan {
        LogicalPlan::Projection(Projection { input, .. })
        | LogicalPlan::Filter(Filter { input, .. })
        | LogicalPlan::Repartition(Repartition { input, .. })
        | LogicalPlan::Window(Window { input, .. })
        | LogicalPlan::Aggregate(Aggregate { input, .. })
        | LogicalPlan::Sort(SortPlan { input, .. })
        | LogicalPlan::Limit(Limit { input, .. })
        | LogicalPlan::SubqueryAlias(SubqueryAlias { input, .. })
        | LogicalPlan::Analyze(Analyze { input, .. })
        | LogicalPlan::Dml(DmlStatement { input, .. })
        | LogicalPlan::Copy(CopyTo { input, .. })
        | LogicalPlan::Unnest(Unnest { input, .. })
        | LogicalPlan::Distinct(Distinct::All(input))
        | LogicalPlan::Distinct(Distinct::On(DistinctOn { input, .. }))
        | LogicalPlan::Explain(Explain { plan: input, .. })
        | LogicalPlan::Subquery(Subquery {
            subquery: input, ..
        })
        | LogicalPlan::Ddl(DdlStatement::CreateMemoryTable(CreateMemoryTable {
            input,
            ..
        }))
        | LogicalPlan::Ddl(DdlStatement::CreateView(CreateView { input, .. }))
        | LogicalPlan::Statement(Statement::Prepare(Prepare { input, .. })) => f(input),
        LogicalPlan::Join(Join { left, right, .. }) => {
            f(left);
            f(right);
        }
        LogicalPlan::RecursiveQuery(RecursiveQuery {
            static_term,
            recursive_term,
            ..
        }) => {
            f(static_term);
            f(recursive_term);
        }
        LogicalPlan::Union(Union { inputs, .. }) => inputs.iter().for_each(f),
        LogicalPlan::Extension(_)
        | LogicalPlan::TableScan(_)
        | LogicalPlan::EmptyRelation(_)
        | LogicalPlan::Values(_)
        | LogicalPlan::DescribeTable(_)
        | LogicalPlan::Ddl(_)
        | LogicalPlan::Statement(_) => {}
    }
}

/// Calls `f` on each input of `plan`, until it stops, without listing them. An extension
/// lists its inputs itself ([`UserDefinedLogicalNode::inputs`]).
///
/// [`UserDefinedLogicalNode::inputs`]: datafusion_expr::UserDefinedLogicalNode::inputs
pub(crate) fn apply_inputs<'a>(
    plan: &'a LogicalPlan,
    f: &mut dyn FnMut(&'a LogicalPlan) -> Result<TreeNodeRecursion>,
) -> Result<TreeNodeRecursion> {
    if let LogicalPlan::Extension(Extension { node }) = plan {
        let mut tnr = TreeNodeRecursion::Continue;
        for input in node.inputs() {
            tnr = f(input)?;
            if tnr == TreeNodeRecursion::Stop {
                break;
            }
        }
        return Ok(tnr);
    }
    let mut tnr = TreeNodeRecursion::Continue;
    let mut failed = None;
    for_each_input_arc(plan, &mut |input| {
        if tnr == TreeNodeRecursion::Stop || failed.is_some() {
            return;
        }
        match f(input.as_ref()) {
            Ok(next) => tnr = next,
            Err(e) => failed = Some(e),
        }
    });
    match failed {
        Some(e) => Err(e),
        None => Ok(tnr),
    }
}

/// Calls `f` on each expression `plan` holds, until it stops, without listing them. An
/// unnest's columns are not expressions it holds; an extension lists its expressions itself
/// ([`UserDefinedLogicalNode::expressions`]).
///
/// [`UserDefinedLogicalNode::expressions`]: datafusion_expr::UserDefinedLogicalNode::expressions
pub(crate) fn apply_plan_exprs(
    plan: &LogicalPlan,
    f: &mut dyn FnMut(&Expr) -> Result<TreeNodeRecursion>,
) -> Result<TreeNodeRecursion> {
    fn each<'e>(
        exprs: impl IntoIterator<Item = &'e Expr>,
        f: &mut dyn FnMut(&Expr) -> Result<TreeNodeRecursion>,
    ) -> Result<TreeNodeRecursion> {
        let mut tnr = TreeNodeRecursion::Continue;
        for expr in exprs {
            tnr = f(expr)?;
            if tnr == TreeNodeRecursion::Stop {
                break;
            }
        }
        Ok(tnr)
    }
    match plan {
        LogicalPlan::Projection(Projection { expr, .. }) => each(expr, f),
        LogicalPlan::Values(Values { values, .. }) => each(values.iter().flatten(), f),
        LogicalPlan::Filter(Filter { predicate, .. }) => f(predicate),
        LogicalPlan::Repartition(Repartition {
            partitioning_scheme,
            ..
        }) => match partitioning_scheme {
            Partitioning::Hash(exprs, _) | Partitioning::DistributeBy(exprs) => {
                each(exprs, f)
            }
            Partitioning::Range(range) => {
                each(range.ordering().iter().map(|sort| &sort.expr), f)
            }
            Partitioning::RoundRobinBatch(_) => Ok(TreeNodeRecursion::Continue),
        },
        LogicalPlan::Window(Window { window_expr, .. }) => each(window_expr, f),
        LogicalPlan::Aggregate(Aggregate {
            group_expr,
            aggr_expr,
            ..
        }) => each(group_expr.iter().chain(aggr_expr), f),
        LogicalPlan::Join(Join { on, filter, .. }) => each(
            on.iter()
                .flat_map(|(left, right)| [left, right])
                .chain(filter),
            f,
        ),
        LogicalPlan::Sort(SortPlan { expr, .. }) => {
            each(expr.iter().map(|sort| &sort.expr), f)
        }
        LogicalPlan::Extension(Extension { node }) => each(&node.expressions(), f),
        LogicalPlan::TableScan(TableScan { filters, .. }) => each(filters, f),
        LogicalPlan::Distinct(Distinct::On(DistinctOn {
            on_expr,
            select_expr,
            sort_expr,
            ..
        })) => each(
            on_expr
                .iter()
                .chain(select_expr)
                .chain(sort_expr.iter().flatten().map(|sort| &sort.expr)),
            f,
        ),
        LogicalPlan::Limit(Limit { skip, fetch, .. }) => {
            each(skip.iter().chain(fetch).map(AsRef::as_ref), f)
        }
        LogicalPlan::Statement(Statement::Execute(Execute { parameters, .. })) => {
            each(parameters, f)
        }
        LogicalPlan::Dml(DmlStatement {
            op: WriteOp::MergeInto(merge_op),
            ..
        }) => {
            let mut tnr = f(&merge_op.on)?;
            for clause in &merge_op.clauses {
                if tnr == TreeNodeRecursion::Stop {
                    break;
                }
                tnr = each(clause.predicate.iter(), f)?;
                if tnr == TreeNodeRecursion::Stop {
                    break;
                }
                tnr = match &clause.action {
                    MergeIntoAction::Update(assignments) => {
                        each(assignments.iter().map(|(_, value)| value), f)?
                    }
                    MergeIntoAction::Insert { values, .. } => each(values, f)?,
                    MergeIntoAction::Delete => TreeNodeRecursion::Continue,
                };
            }
            Ok(tnr)
        }
        LogicalPlan::Unnest(_)
        | LogicalPlan::EmptyRelation(_)
        | LogicalPlan::RecursiveQuery(_)
        | LogicalPlan::Subquery(_)
        | LogicalPlan::SubqueryAlias(_)
        | LogicalPlan::Analyze(_)
        | LogicalPlan::Explain(_)
        | LogicalPlan::Union(_)
        | LogicalPlan::Distinct(Distinct::All(_))
        | LogicalPlan::Statement(_)
        | LogicalPlan::Dml(_)
        | LogicalPlan::Ddl(_)
        | LogicalPlan::Copy(_)
        | LogicalPlan::DescribeTable(_) => Ok(TreeNodeRecursion::Continue),
    }
}

/// Calls `f` on `plan` and every plan below it, until it stops, as
/// [`TreeNode::apply`] (or, `with_subqueries`, [`LogicalPlan::apply_with_subqueries`])
/// does, without listing any node's inputs.
#[cfg_attr(feature = "recursive_protection", recursive::recursive)]
pub(crate) fn apply_plan(
    funding: &Funding,
    plan: &LogicalPlan,
    with_subqueries: bool,
    f: &mut dyn FnMut(&LogicalPlan) -> Result<TreeNodeRecursion>,
) -> Result<TreeNodeRecursion> {
    match f(plan)? {
        TreeNodeRecursion::Continue => {}
        TreeNodeRecursion::Jump => return Ok(TreeNodeRecursion::Continue),
        TreeNodeRecursion::Stop => return Ok(TreeNodeRecursion::Stop),
    }
    if with_subqueries {
        let tnr = apply_plan_exprs(plan, &mut |expr| {
            expr.apply(|e| match e {
                Expr::Exists(Exists { subquery, .. })
                | Expr::InSubquery(InSubquery { subquery, .. })
                | Expr::SetComparison(SetComparison { subquery, .. })
                | Expr::ScalarSubquery(subquery) => {
                    // The subquery is visited as a node of its own, a copy charged while
                    // it is visited.
                    funding.charge(subquery.clone_bytes())?;
                    let node = LogicalPlan::Subquery(subquery.clone());
                    let visited = apply_plan(funding, &node, true, f);
                    freed(funding, node);
                    visited
                }
                _ => Ok(TreeNodeRecursion::Continue),
            })
        })?;
        if tnr == TreeNodeRecursion::Stop {
            return Ok(tnr);
        }
    }
    apply_inputs(plan, &mut |input| {
        apply_plan(funding, input, with_subqueries, f)
    })
}

/// The plans in `plan`, as [`LogicalPlan::apply_with_subqueries`] visits them (each
/// subquery as the node holding it, then its plans), counted without listing any.
#[cfg_attr(feature = "recursive_protection", recursive::recursive)]
pub(crate) fn count_plans(plan: &LogicalPlan) -> usize {
    let mut count = 1usize;
    let _ = apply_plan_exprs(plan, &mut |expr| {
        expr.apply(|e| {
            match e {
                Expr::Exists(Exists { subquery, .. })
                | Expr::InSubquery(InSubquery { subquery, .. })
                | Expr::SetComparison(SetComparison { subquery, .. })
                | Expr::ScalarSubquery(subquery) => {
                    count = sum([count, 1, count_plans(&subquery.subquery)]);
                }
                _ => {}
            }
            Ok(TreeNodeRecursion::Continue)
        })
    });
    let _ = apply_inputs(plan, &mut |input| {
        count = sum([count, count_plans(input)]);
        Ok(TreeNodeRecursion::Continue)
    });
    count
}

/// The outer reference columns `plan` and its inputs hold (not their subqueries'), each
/// once, in the order [`LogicalPlan::all_out_ref_exprs`] lists them: the list and each copy
/// charged before they are made.
pub(crate) fn out_ref_exprs(funding: &Funding, plan: &LogicalPlan) -> Result<Vec<Expr>> {
    #[cfg_attr(feature = "recursive_protection", recursive::recursive)]
    fn collect(
        funding: &Funding,
        plan: &LogicalPlan,
        found: &mut Held<Vec<Expr>>,
    ) -> Result<()> {
        apply_plan_exprs(plan, &mut |expr| {
            expr.apply(|e| {
                if matches!(e, Expr::OuterReferenceColumn(..)) && !found.contains(e) {
                    found.push(funding.cloned(e)?)?;
                }
                Ok(TreeNodeRecursion::Continue)
            })
        })?;
        apply_inputs(plan, &mut |input| {
            collect(funding, input, found)?;
            Ok(TreeNodeRecursion::Continue)
        })?;
        Ok(())
    }
    let mut found = funding.held::<Vec<Expr>>();
    collect(funding, plan, &mut found)?;
    Ok(found.keep())
}

/// Whether `plan` or its inputs (not their subqueries) hold an outer reference column,
/// found without listing them.
#[cfg_attr(feature = "recursive_protection", recursive::recursive)]
pub(crate) fn has_out_ref_exprs(plan: &LogicalPlan) -> bool {
    let mut found = false;
    let _ = apply_plan_exprs(plan, &mut |expr| {
        expr.apply(|e| {
            if matches!(e, Expr::OuterReferenceColumn(..)) {
                found = true;
                Ok(TreeNodeRecursion::Stop)
            } else {
                Ok(TreeNodeRecursion::Continue)
            }
        })
    });
    if !found {
        let _ = apply_inputs(plan, &mut |input| {
            found = has_out_ref_exprs(input);
            Ok(if found {
                TreeNodeRecursion::Stop
            } else {
                TreeNodeRecursion::Continue
            })
        });
    }
    found
}

/// Whether `plan` (not its inputs) holds a subquery expression.
pub(crate) fn has_subquery_exprs(plan: &LogicalPlan) -> bool {
    let mut found = false;
    let _ = apply_plan_exprs(plan, &mut |expr| {
        expr.apply(|e| {
            if is_subquery_expr(e) {
                found = true;
                Ok(TreeNodeRecursion::Stop)
            } else {
                Ok(TreeNodeRecursion::Continue)
            }
        })
    });
    found
}

fn is_subquery_expr(expr: &Expr) -> bool {
    matches!(
        expr,
        Expr::Exists(_)
            | Expr::InSubquery(_)
            | Expr::SetComparison(_)
            | Expr::ScalarSubquery(_)
    )
}

/// Calls `f` on each input `plan` holds in an `Arc` of its own, to change in place: each
/// shared input copied first ([`plan_mut`]). Returns whether `f` changed any.
///
/// An extension's inputs are rewritten through the extension itself: what that allocates is
/// the extension's ([`UserDefinedLogicalNode::with_exprs_and_inputs`]).
///
/// [`UserDefinedLogicalNode::with_exprs_and_inputs`]: datafusion_expr::UserDefinedLogicalNode::with_exprs_and_inputs
pub(crate) fn map_inputs_mut(
    funding: &Funding,
    plan: &mut LogicalPlan,
    f: &mut dyn FnMut(&mut LogicalPlan) -> Result<bool>,
) -> Result<bool> {
    Ok(match plan {
        LogicalPlan::Projection(Projection { input, .. })
        | LogicalPlan::Filter(Filter { input, .. })
        | LogicalPlan::Repartition(Repartition { input, .. })
        | LogicalPlan::Window(Window { input, .. })
        | LogicalPlan::Aggregate(Aggregate { input, .. })
        | LogicalPlan::Sort(SortPlan { input, .. })
        | LogicalPlan::Limit(Limit { input, .. })
        | LogicalPlan::SubqueryAlias(SubqueryAlias { input, .. })
        | LogicalPlan::Analyze(Analyze { input, .. })
        | LogicalPlan::Dml(DmlStatement { input, .. })
        | LogicalPlan::Copy(CopyTo { input, .. })
        | LogicalPlan::Unnest(Unnest { input, .. })
        | LogicalPlan::Distinct(Distinct::All(input))
        | LogicalPlan::Distinct(Distinct::On(DistinctOn { input, .. }))
        | LogicalPlan::Explain(Explain { plan: input, .. })
        | LogicalPlan::Subquery(Subquery {
            subquery: input, ..
        })
        | LogicalPlan::Ddl(DdlStatement::CreateMemoryTable(CreateMemoryTable {
            input,
            ..
        }))
        | LogicalPlan::Ddl(DdlStatement::CreateView(CreateView { input, .. }))
        | LogicalPlan::Statement(Statement::Prepare(Prepare { input, .. })) => {
            f(plan_mut(funding, input)?)?
        }
        LogicalPlan::Join(Join { left, right, .. }) => {
            let left = f(plan_mut(funding, left)?)?;
            let right = f(plan_mut(funding, right)?)?;
            left || right
        }
        LogicalPlan::RecursiveQuery(RecursiveQuery {
            static_term,
            recursive_term,
            ..
        }) => {
            let static_changed = f(plan_mut(funding, static_term)?)?;
            let recursive_changed = f(plan_mut(funding, recursive_term)?)?;
            static_changed || recursive_changed
        }
        LogicalPlan::Union(Union { inputs, .. }) => {
            let mut changed = false;
            for input in inputs {
                changed |= f(plan_mut(funding, input)?)?;
            }
            changed
        }
        LogicalPlan::Extension(_) => {
            let owned = std::mem::take(plan);
            let mut changed = false;
            let rewritten = owned.map_children(|mut input| {
                changed |= f(&mut input)?;
                Ok(Transformed::no(input))
            })?;
            *plan = rewritten.data;
            changed
        }
        LogicalPlan::TableScan(_)
        | LogicalPlan::EmptyRelation(_)
        | LogicalPlan::Values(_)
        | LogicalPlan::DescribeTable(_)
        | LogicalPlan::Ddl(_)
        | LogicalPlan::Statement(_) => false,
    })
}

/// Calls `f` on each subquery `plan`'s expressions hold, as the [`LogicalPlan::Subquery`]
/// node holding it, to change in place, in the order [`LogicalPlan::map_subqueries`] visits
/// them. Returns whether `f` changed any.
///
/// An extension's, a range partitioning's or a merge's expressions are rewritten through
/// [`LogicalPlan::map_subqueries`]: what that allocates is theirs.
pub(crate) fn map_subqueries_mut(
    plan: &mut LogicalPlan,
    f: &mut dyn FnMut(&mut LogicalPlan) -> Result<bool>,
) -> Result<bool> {
    if !has_subquery_exprs(plan) {
        return Ok(false);
    }
    let mut changed = false;
    let mut each = |exprs: &mut dyn Iterator<Item = &mut Expr>| -> Result<()> {
        for expr in exprs {
            changed |= expr_subqueries_mut(expr, f)?;
        }
        Ok(())
    };
    match plan {
        LogicalPlan::Projection(Projection { expr, .. }) => each(&mut expr.iter_mut())?,
        LogicalPlan::Values(Values { values, .. }) => {
            each(&mut values.iter_mut().flatten())?
        }
        LogicalPlan::Filter(Filter { predicate, .. }) => {
            each(&mut std::iter::once(predicate))?
        }
        LogicalPlan::Repartition(Repartition {
            partitioning_scheme:
                Partitioning::Hash(exprs, _) | Partitioning::DistributeBy(exprs),
            ..
        }) => each(&mut exprs.iter_mut())?,
        LogicalPlan::Window(Window { window_expr, .. }) => {
            each(&mut window_expr.iter_mut())?
        }
        LogicalPlan::Aggregate(Aggregate {
            group_expr,
            aggr_expr,
            ..
        }) => each(&mut group_expr.iter_mut().chain(aggr_expr.iter_mut()))?,
        LogicalPlan::Join(Join { on, filter, .. }) => each(
            &mut on
                .iter_mut()
                .flat_map(|(left, right)| [left, right])
                .chain(filter.iter_mut()),
        )?,
        LogicalPlan::Sort(SortPlan { expr, .. }) => {
            each(&mut expr.iter_mut().map(|sort| &mut sort.expr))?
        }
        LogicalPlan::TableScan(TableScan { filters, .. }) => {
            each(&mut filters.iter_mut())?
        }
        LogicalPlan::Distinct(Distinct::On(DistinctOn {
            on_expr,
            select_expr,
            sort_expr,
            ..
        })) => each(
            &mut on_expr
                .iter_mut()
                .chain(select_expr.iter_mut())
                .chain(sort_expr.iter_mut().flatten().map(|sort| &mut sort.expr)),
        )?,
        LogicalPlan::Limit(Limit { skip, fetch, .. }) => each(
            &mut skip
                .iter_mut()
                .chain(fetch.iter_mut())
                .map(|expr| expr.as_mut()),
        )?,
        LogicalPlan::Statement(Statement::Execute(Execute { parameters, .. })) => {
            each(&mut parameters.iter_mut())?
        }
        _ => {
            let owned = std::mem::take(plan);
            let rewritten = owned.map_subqueries(|mut subquery| {
                changed |= f(&mut subquery)?;
                Ok(Transformed::no(subquery))
            })?;
            *plan = rewritten.data;
        }
    }
    Ok(changed)
}

/// Calls `f` on each expression `plan` holds, to change in place, in the order
/// [`LogicalPlan::map_expressions`] visits them. Returns whether `f` changed any.
///
/// An extension's, a range partitioning's, an unnest's or a merge's expressions are
/// rewritten through [`LogicalPlan::map_expressions`]: what that allocates is theirs.
pub(crate) fn map_plan_exprs_mut(
    plan: &mut LogicalPlan,
    f: &mut dyn FnMut(&mut Expr) -> Result<bool>,
) -> Result<bool> {
    let mut changed = false;
    let mut each = |exprs: &mut dyn Iterator<Item = &mut Expr>| -> Result<()> {
        for expr in exprs {
            changed |= f(expr)?;
        }
        Ok(())
    };
    match plan {
        LogicalPlan::Projection(Projection { expr, .. }) => each(&mut expr.iter_mut())?,
        LogicalPlan::Values(Values { values, .. }) => {
            each(&mut values.iter_mut().flatten())?
        }
        LogicalPlan::Filter(Filter { predicate, .. }) => {
            each(&mut std::iter::once(predicate))?
        }
        LogicalPlan::Repartition(Repartition {
            partitioning_scheme:
                Partitioning::Hash(exprs, _) | Partitioning::DistributeBy(exprs),
            ..
        }) => each(&mut exprs.iter_mut())?,
        LogicalPlan::Window(Window { window_expr, .. }) => {
            each(&mut window_expr.iter_mut())?
        }
        LogicalPlan::Aggregate(Aggregate {
            group_expr,
            aggr_expr,
            ..
        }) => each(&mut group_expr.iter_mut().chain(aggr_expr.iter_mut()))?,
        LogicalPlan::Join(Join { on, filter, .. }) => each(
            &mut on
                .iter_mut()
                .flat_map(|(left, right)| [left, right])
                .chain(filter.iter_mut()),
        )?,
        LogicalPlan::Sort(SortPlan { expr, .. }) => {
            each(&mut expr.iter_mut().map(|sort| &mut sort.expr))?
        }
        LogicalPlan::TableScan(TableScan { filters, .. }) => {
            each(&mut filters.iter_mut())?
        }
        LogicalPlan::Distinct(Distinct::On(DistinctOn {
            on_expr,
            select_expr,
            sort_expr,
            ..
        })) => each(
            &mut on_expr
                .iter_mut()
                .chain(select_expr.iter_mut())
                .chain(sort_expr.iter_mut().flatten().map(|sort| &mut sort.expr)),
        )?,
        LogicalPlan::Limit(Limit { skip, fetch, .. }) => each(
            &mut skip
                .iter_mut()
                .chain(fetch.iter_mut())
                .map(|expr| expr.as_mut()),
        )?,
        LogicalPlan::Statement(Statement::Execute(Execute { parameters, .. })) => {
            each(&mut parameters.iter_mut())?
        }
        LogicalPlan::Extension(_)
        | LogicalPlan::Repartition(_)
        | LogicalPlan::Unnest(_)
        | LogicalPlan::Dml(DmlStatement {
            op: WriteOp::MergeInto(_),
            ..
        }) => {
            let owned = std::mem::take(plan);
            let rewritten = owned.map_expressions(|mut expr| {
                let expr_changed = f(&mut expr)?;
                Ok(Transformed::new_transformed(expr, expr_changed))
            })?;
            changed = rewritten.transformed;
            *plan = rewritten.data;
        }
        LogicalPlan::EmptyRelation(_)
        | LogicalPlan::RecursiveQuery(_)
        | LogicalPlan::Subquery(_)
        | LogicalPlan::SubqueryAlias(_)
        | LogicalPlan::Analyze(_)
        | LogicalPlan::Explain(_)
        | LogicalPlan::Union(_)
        | LogicalPlan::Distinct(Distinct::All(_))
        | LogicalPlan::Statement(_)
        | LogicalPlan::Dml(_)
        | LogicalPlan::Ddl(_)
        | LogicalPlan::Copy(_)
        | LogicalPlan::DescribeTable(_) => {}
    }
    Ok(changed)
}

/// Calls `f` on each subquery in `expr`, top down, as the [`LogicalPlan::Subquery`] node
/// holding it, to change in place.
#[cfg_attr(feature = "recursive_protection", recursive::recursive)]
fn expr_subqueries_mut(
    expr: &mut Expr,
    f: &mut dyn FnMut(&mut LogicalPlan) -> Result<bool>,
) -> Result<bool> {
    let mut changed = false;
    if is_subquery_expr(expr) {
        // The expression is moved out, its subquery into a plan node and back: nothing is
        // allocated.
        let (mut node, rebuild): (LogicalPlan, fn(Expr, Subquery) -> Expr) =
            match std::mem::take(expr) {
                Expr::Exists(Exists { subquery, negated }) => (
                    LogicalPlan::Subquery(subquery),
                    if negated {
                        |_, subquery| {
                            Expr::Exists(Exists {
                                subquery,
                                negated: true,
                            })
                        }
                    } else {
                        |_, subquery| {
                            Expr::Exists(Exists {
                                subquery,
                                negated: false,
                            })
                        }
                    },
                ),
                Expr::ScalarSubquery(subquery) => {
                    (LogicalPlan::Subquery(subquery), |_, subquery| {
                        Expr::ScalarSubquery(subquery)
                    })
                }
                held @ (Expr::InSubquery(_) | Expr::SetComparison(_)) => {
                    // Its other fields stay in place, around a placeholder subquery
                    // moved back out once the subquery is rewritten.
                    *expr = held;
                    return in_place_subquery(expr, f);
                }
                other => {
                    *expr = other;
                    return Ok(false);
                }
            };
        changed |= f(&mut node)?;
        match node {
            LogicalPlan::Subquery(subquery) => *expr = rebuild(Expr::default(), subquery),
            _ => return internal_err!("Transformation should return Subquery"),
        }
        return Ok(changed | expr_children_subqueries_mut(expr, f)?);
    }
    changed |= expr_children_subqueries_mut(expr, f)?;
    Ok(changed)
}

/// An `IN (subquery)` or a set comparison: its subquery rewritten as a plan node, then its
/// expression's subqueries.
fn in_place_subquery(
    expr: &mut Expr,
    f: &mut dyn FnMut(&mut LogicalPlan) -> Result<bool>,
) -> Result<bool> {
    let subquery = match expr {
        Expr::InSubquery(InSubquery { subquery, .. })
        | Expr::SetComparison(SetComparison { subquery, .. }) => subquery,
        _ => return Ok(false),
    };
    // A subquery node is swapped in for it; the placeholder left behind holds the same
    // plan, a count bumped, and its empty lists allocate nothing.
    let placeholder = Subquery {
        subquery: Arc::clone(&subquery.subquery),
        outer_ref_columns: Vec::new(),
        spans: Spans::new(),
    };
    let mut node = LogicalPlan::Subquery(std::mem::replace(subquery, placeholder));
    let changed = f(&mut node)?;
    match node {
        LogicalPlan::Subquery(rewritten) => {
            // The placeholder is dropped: its plan's count is given back.
            *subquery = rewritten;
        }
        _ => return internal_err!("Transformation should return Subquery"),
    }
    Ok(changed | expr_children_subqueries_mut(expr, f)?)
}

fn expr_children_subqueries_mut(
    expr: &mut Expr,
    f: &mut dyn FnMut(&mut LogicalPlan) -> Result<bool>,
) -> Result<bool> {
    let mut changed = false;
    for_each_child_expr_mut(expr, &mut |child| {
        changed |= expr_subqueries_mut(child, f)?;
        Ok(())
    })?;
    Ok(changed)
}

/// Calls `f` on each child of `expr`, to change in place, the children
/// [`TreeNode::apply_children`] visits.
pub(crate) fn for_each_child_expr_mut(
    expr: &mut Expr,
    f: &mut dyn FnMut(&mut Expr) -> Result<()>,
) -> Result<()> {
    match expr {
        Expr::Alias(Alias { expr, .. })
        | Expr::Unnest(UnnestExpr { expr, .. })
        | Expr::Not(expr)
        | Expr::IsNotNull(expr)
        | Expr::IsTrue(expr)
        | Expr::IsFalse(expr)
        | Expr::IsUnknown(expr)
        | Expr::IsNotTrue(expr)
        | Expr::IsNotFalse(expr)
        | Expr::IsNotUnknown(expr)
        | Expr::IsNull(expr)
        | Expr::Negative(expr)
        | Expr::Cast(Cast { expr, .. })
        | Expr::TryCast(TryCast { expr, .. })
        | Expr::InSubquery(InSubquery { expr, .. })
        | Expr::SetComparison(SetComparison { expr, .. })
        | Expr::Lambda(Lambda { body: expr, .. }) => f(expr),
        Expr::GroupingSet(GroupingSet::Rollup(exprs))
        | Expr::GroupingSet(GroupingSet::Cube(exprs))
        | Expr::ScalarFunction(ScalarFunction { args: exprs, .. })
        | Expr::HigherOrderFunction(HigherOrderFunction { args: exprs, .. }) => {
            exprs.iter_mut().try_for_each(f)
        }
        Expr::GroupingSet(GroupingSet::GroupingSets(lists)) => {
            lists.iter_mut().flatten().try_for_each(f)
        }
        #[expect(deprecated)]
        Expr::Column(_)
        | Expr::OuterReferenceColumn(_, _)
        | Expr::ScalarVariable(_, _)
        | Expr::Literal(_, _)
        | Expr::Exists(_)
        | Expr::ScalarSubquery(_)
        | Expr::Wildcard { .. }
        | Expr::Placeholder(_)
        | Expr::LambdaVariable(_) => Ok(()),
        Expr::BinaryExpr(BinaryExpr { left, right, .. }) => {
            f(left)?;
            f(right)
        }
        Expr::Like(Like { expr, pattern, .. })
        | Expr::SimilarTo(Like { expr, pattern, .. }) => {
            f(expr)?;
            f(pattern)
        }
        Expr::Between(Between {
            expr, low, high, ..
        }) => {
            f(expr)?;
            f(low)?;
            f(high)
        }
        Expr::Case(Case {
            expr,
            when_then_expr,
            else_expr,
        }) => {
            if let Some(expr) = expr {
                f(expr)?;
            }
            for (when, then) in when_then_expr {
                f(when)?;
                f(then)?;
            }
            match else_expr {
                Some(else_expr) => f(else_expr),
                None => Ok(()),
            }
        }
        Expr::AggregateFunction(AggregateFunction {
            params:
                AggregateFunctionParams {
                    args,
                    filter,
                    order_by,
                    ..
                },
            ..
        }) => {
            args.iter_mut().try_for_each(&mut *f)?;
            if let Some(filter) = filter {
                f(filter)?;
            }
            order_by.iter_mut().try_for_each(|sort| f(&mut sort.expr))
        }
        Expr::WindowFunction(window) => {
            let WindowFunctionParams {
                args,
                partition_by,
                order_by,
                filter,
                ..
            } = &mut window.params;
            args.iter_mut().try_for_each(&mut *f)?;
            partition_by.iter_mut().try_for_each(&mut *f)?;
            order_by.iter_mut().try_for_each(|sort| f(&mut sort.expr))?;
            match filter {
                Some(filter) => f(filter),
                None => Ok(()),
            }
        }
        Expr::InList(InList { expr, list, .. }) => {
            f(expr)?;
            list.iter_mut().try_for_each(f)
        }
    }
}

/// Every plan `plan` holds in an `Arc` (and, `with_subqueries`, every subquery's) held by
/// `plan` alone, each shared one copied first, the copy charged before it is made
/// ([`plan_mut`]): a [`TreeNode`] traversal over it then reuses every block it rewrites and
/// allocates nothing for the plans it visits.
#[cfg_attr(feature = "recursive_protection", recursive::recursive)]
pub(crate) fn unshare(
    funding: &Funding,
    plan: &mut LogicalPlan,
    with_subqueries: bool,
) -> Result<()> {
    if with_subqueries {
        map_subqueries_mut(plan, &mut |subquery| {
            unshare(funding, subquery, true)?;
            Ok(false)
        })?;
    }
    map_inputs_mut(funding, plan, &mut |input| {
        unshare(funding, input, with_subqueries)?;
        Ok(false)
    })?;
    Ok(())
}

/// [`LogicalPlan::transform_up_with_subqueries`] over `plan`, each shared input and subquery
/// copied first, charged ([`unshare`]).
pub(crate) fn transform_up_plan_with_subqueries<F>(
    funding: &Funding,
    mut plan: LogicalPlan,
    f: F,
) -> Result<Transformed<LogicalPlan>>
where
    F: FnMut(LogicalPlan) -> Result<Transformed<LogicalPlan>>,
{
    unshare(funding, &mut plan, true)?;
    plan.transform_up_with_subqueries(f)
}
