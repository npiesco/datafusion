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
//! ([`plan_clone_bytes`], built on [`CloneBytes`]). What a freed plan or expression gives
//! back is counted by [`Freed`]: the blocks dropping it frees (its lists, boxes and texts,
//! and each plan it held alone).
//!
//! What cannot be counted exactly is refused rather than guessed: a node whose copy
//! allocates blocks that cannot be counted (a statement, a DDL or DML node, an `EXPLAIN`, a
//! range partitioning, a table scan carrying statistics requests) is refused its copy.
//! Schemas, fields, functions and table names are shared, not copied: their blocks are
//! their maker's to give back.
//!
//! What is freed is given back through the handle the plan was charged through: a plan
//! handed to the optimizer is charged on the handle it optimizes within.

use std::sync::Arc;

use arrow::datatypes::DataType;
use datafusion_common::cloned::CloneBytes;
use datafusion_common::funding::{Funding, arc_bytes};
use datafusion_common::tree_node::{Transformed, TreeNode, TreeNodeRecursion};
use datafusion_common::{
    Column, DataFusionError, RecursionUnnestOption, Result, ScalarValue, Spans,
    internal_err,
};
use datafusion_expr::dml::CopyTo;
use datafusion_expr::expr::{
    AggregateFunction, AggregateFunctionParams, Alias, Between, BinaryExpr, Case, Cast,
    Exists, GroupingSet, HigherOrderFunction, InList, InSubquery, Lambda, LambdaVariable,
    Like, Placeholder, ScalarFunction, SetComparison, Sort, TryCast,
    Unnest as UnnestExpr, WindowFunction, WindowFunctionParams,
};
use datafusion_expr::logical_plan::{
    Aggregate, Analyze, ColumnUnnestList, CreateMemoryTable, CreateView, DdlStatement,
    Distinct, DistinctOn, DmlStatement, Execute, Explain, Extension, Filter, Join, Limit,
    LogicalPlan, MergeIntoAction, Partitioning, Prepare, Projection, RecursiveQuery,
    Repartition, Sort as SortPlan, Statement, Subquery, SubqueryAlias, TableScan, Union,
    Unnest, Values, Window,
};
use datafusion_expr::{Expr, WindowFrame, WindowFrameBound, WriteOp};

fn sum(parts: impl IntoIterator<Item = usize>) -> usize {
    parts.into_iter().fold(0, usize::saturating_add)
}

/// What a value gives back when it is dropped.
pub(crate) trait Freed {
    /// The bytes dropping it frees, of the blocks planning charges: its lists (their room),
    /// boxes and texts, and each plan it holds alone. Shared blocks (schemas, fields,
    /// functions, table names) are not freed with it: they are not counted.
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
    /// Its name and spans; its table name is shared.
    fn freed_bytes(&self) -> usize {
        sum([self.name.freed_bytes(), self.spans.freed_bytes()])
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
            Expr::Alias(Alias { expr, name, .. }) => {
                sum([expr.freed_bytes(), name.freed_bytes()])
            }
            Expr::Column(column) | Expr::OuterReferenceColumn(_, column) => {
                column.freed_bytes()
            }
            Expr::ScalarVariable(_, names) => names.freed_bytes(),
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
            | Expr::Cast(Cast { expr, .. })
            | Expr::TryCast(TryCast { expr, .. })
            | Expr::Unnest(UnnestExpr { expr, .. }) => expr.freed_bytes(),
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
            Expr::Placeholder(Placeholder { id, .. }) => id.freed_bytes(),
            Expr::Lambda(Lambda { params, body }) => {
                sum([params.freed_bytes(), body.freed_bytes()])
            }
            Expr::LambdaVariable(LambdaVariable { name, spans, .. }) => {
                sum([name.freed_bytes(), spans.freed_bytes()])
            }
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
            LogicalPlan::Projection(Projection { expr, .. }) => expr.freed_bytes(),
            LogicalPlan::Filter(Filter { predicate, .. }) => predicate.freed_bytes(),
            LogicalPlan::Window(Window { window_expr, .. }) => window_expr.freed_bytes(),
            LogicalPlan::Aggregate(Aggregate {
                group_expr,
                aggr_expr,
                ..
            }) => sum([group_expr.freed_bytes(), aggr_expr.freed_bytes()]),
            LogicalPlan::Sort(SortPlan { expr, .. }) => expr.freed_bytes(),
            LogicalPlan::Join(Join { on, filter, .. }) => {
                sum([on.freed_bytes(), filter.freed_bytes()])
            }
            LogicalPlan::Repartition(Repartition {
                partitioning_scheme:
                    Partitioning::Hash(exprs, _) | Partitioning::DistributeBy(exprs),
                ..
            }) => exprs.freed_bytes(),
            LogicalPlan::Union(Union { inputs, .. }) => {
                // The inputs themselves are counted below, with every other node's.
                inputs
                    .capacity()
                    .saturating_mul(size_of::<Arc<LogicalPlan>>())
            }
            LogicalPlan::TableScan(TableScan {
                projection,
                filters,
                ..
            }) => sum([projection.freed_bytes(), filters.freed_bytes()]),
            LogicalPlan::Subquery(Subquery {
                outer_ref_columns,
                spans,
                ..
            }) => sum([outer_ref_columns.freed_bytes(), spans.freed_bytes()]),
            LogicalPlan::Limit(Limit { skip, fetch, .. }) => {
                sum([skip.freed_bytes(), fetch.freed_bytes()])
            }
            LogicalPlan::Values(Values { values, .. }) => values.freed_bytes(),
            LogicalPlan::Distinct(Distinct::On(DistinctOn {
                on_expr,
                select_expr,
                sort_expr,
                ..
            })) => sum([
                on_expr.freed_bytes(),
                select_expr.freed_bytes(),
                sort_expr.freed_bytes(),
            ]),
            LogicalPlan::Unnest(Unnest {
                exec_columns,
                list_type_columns,
                struct_type_columns,
                dependency_indices,
                options,
                ..
            }) => sum([
                exec_columns.freed_bytes(),
                list_type_columns.freed_bytes(),
                struct_type_columns.freed_bytes(),
                dependency_indices.freed_bytes(),
                options.recursions.freed_bytes(),
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

/// The bytes a copy of `plan`'s node allocates (its inputs are shared, not copied), every
/// block exactly as `Clone::clone` sizes it; `usize::MAX`, refused, for a node whose copy
/// cannot be counted.
pub(crate) fn plan_clone_bytes(plan: &LogicalPlan) -> usize {
    match plan {
        LogicalPlan::Projection(Projection { expr, .. }) => expr.clone_bytes(),
        LogicalPlan::Filter(Filter { predicate, .. }) => predicate.clone_bytes(),
        LogicalPlan::Window(Window { window_expr, .. }) => window_expr.clone_bytes(),
        LogicalPlan::Aggregate(Aggregate {
            group_expr,
            aggr_expr,
            ..
        }) => sum([group_expr.clone_bytes(), aggr_expr.clone_bytes()]),
        LogicalPlan::Sort(SortPlan { expr, .. }) => expr.clone_bytes(),
        LogicalPlan::Join(Join { on, filter, .. }) => {
            sum([on.clone_bytes(), filter.clone_bytes()])
        }
        LogicalPlan::Repartition(Repartition {
            partitioning_scheme,
            ..
        }) => match partitioning_scheme {
            Partitioning::Hash(exprs, _) | Partitioning::DistributeBy(exprs) => {
                exprs.clone_bytes()
            }
            Partitioning::RoundRobinBatch(_) => 0,
            // Its split points are not counted.
            Partitioning::Range(_) => usize::MAX,
        },
        LogicalPlan::Union(Union { inputs, .. }) => inputs.clone_bytes(),
        LogicalPlan::TableScan(TableScan {
            projection,
            filters,
            statistics_requests,
            ..
        }) => {
            if statistics_requests.is_empty() {
                sum([projection.clone_bytes(), filters.clone_bytes()])
            } else {
                // A tree's nodes are not counted.
                usize::MAX
            }
        }
        LogicalPlan::EmptyRelation(_)
        | LogicalPlan::SubqueryAlias(_)
        | LogicalPlan::Extension(_)
        | LogicalPlan::DescribeTable(_)
        | LogicalPlan::Distinct(Distinct::All(_)) => 0,
        LogicalPlan::Subquery(subquery) => subquery.clone_bytes(),
        LogicalPlan::Limit(Limit { skip, fetch, .. }) => {
            sum([skip.clone_bytes(), fetch.clone_bytes()])
        }
        LogicalPlan::Values(Values { values, .. }) => values.clone_bytes(),
        LogicalPlan::Distinct(Distinct::On(DistinctOn {
            on_expr,
            select_expr,
            sort_expr,
            ..
        })) => sum([
            on_expr.clone_bytes(),
            select_expr.clone_bytes(),
            sort_expr.clone_bytes(),
        ]),
        LogicalPlan::Unnest(Unnest {
            exec_columns,
            list_type_columns,
            struct_type_columns,
            dependency_indices,
            options,
            ..
        }) => sum([
            exec_columns.clone_bytes(),
            list_type_columns.iter().fold(
                size_of::<(usize, ColumnUnnestList)>()
                    .saturating_mul(list_type_columns.len()),
                |bytes, (_, list)| bytes.saturating_add(list.output_column.clone_bytes()),
            ),
            struct_type_columns.clone_bytes(),
            dependency_indices.clone_bytes(),
            options.recursions.iter().fold(
                size_of::<RecursionUnnestOption>()
                    .saturating_mul(options.recursions.len()),
                |bytes, recursion| {
                    sum([
                        bytes,
                        recursion.input_column.clone_bytes(),
                        recursion.output_column.clone_bytes(),
                    ])
                },
            ),
        ]),
        LogicalPlan::RecursiveQuery(RecursiveQuery { name, .. }) => name.clone_bytes(),
        // What these hold is not all counted: they are not copied.
        LogicalPlan::Statement(_)
        | LogicalPlan::Explain(_)
        | LogicalPlan::Analyze(_)
        | LogicalPlan::Dml(_)
        | LogicalPlan::Ddl(_)
        | LogicalPlan::Copy(_) => usize::MAX,
    }
}

/// A copy of `plan`'s node (its inputs shared), what the copy allocates charged before it
/// is made.
pub(crate) fn cloned_plan(funding: &Funding, plan: &LogicalPlan) -> Result<LogicalPlan> {
    funding.charge(plan_clone_bytes(plan))?;
    Ok(plan.clone())
}

/// `value` dropped: what it frees is given back.
pub(crate) fn freed<T: Freed>(funding: &Funding, value: T) {
    let bytes = value.freed_bytes();
    drop(value);
    funding.discharge(bytes);
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
        funding.charge(sum([plan_clone_bytes(arc), arc_bytes::<LogicalPlan>()]))?;
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

/// [`TreeNode::transform_up`] over `plan`, each shared input copied first, charged
/// ([`unshare`]).
pub(crate) fn transform_up_plan<F>(
    funding: &Funding,
    mut plan: LogicalPlan,
    f: F,
) -> Result<Transformed<LogicalPlan>>
where
    F: FnMut(LogicalPlan) -> Result<Transformed<LogicalPlan>>,
{
    unshare(funding, &mut plan, false)?;
    plan.transform_up(f)
}

/// [`TreeNode::transform_down`] over `plan`, each shared input copied first, charged
/// ([`unshare`]).
pub(crate) fn transform_down_plan<F>(
    funding: &Funding,
    mut plan: LogicalPlan,
    f: F,
) -> Result<Transformed<LogicalPlan>>
where
    F: FnMut(LogicalPlan) -> Result<Transformed<LogicalPlan>>,
{
    unshare(funding, &mut plan, false)?;
    plan.transform_down(f)
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

/// [`LogicalPlan::transform_down_with_subqueries`] over `plan`, each shared input and
/// subquery copied first, charged ([`unshare`]).
pub(crate) fn transform_down_plan_with_subqueries<F>(
    funding: &Funding,
    mut plan: LogicalPlan,
    f: F,
) -> Result<Transformed<LogicalPlan>>
where
    F: FnMut(LogicalPlan) -> Result<Transformed<LogicalPlan>>,
{
    unshare(funding, &mut plan, true)?;
    plan.transform_down_with_subqueries(f)
}
