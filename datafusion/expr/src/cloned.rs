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

//! What cloning an expression allocates ([`CloneBytes`]), measured so the clone is charged
//! before it is made.

use datafusion_common::cloned::CloneBytes;

use crate::expr::{
    AggregateFunction, AggregateFunctionParams, Alias, Between, BinaryExpr, Case, Cast,
    Exists, HigherOrderFunction, InList, InSubquery, Lambda, LambdaVariable, Like,
    Placeholder, ScalarFunction, SetComparison, Sort, TryCast, Unnest, WildcardOptions,
    WindowFunction, WindowFunctionParams,
};
use crate::logical_plan::{
    Aggregate, ColumnUnnestList, Distinct, DistinctOn, Filter, Join, Limit, LogicalPlan,
    Partitioning, Projection, RecursiveQuery, Repartition, Sort as SortPlan, Subquery,
    SubqueryAlias, TableScan, Union, Unnest as UnnestPlan, Values, Window,
};
use crate::{Expr, GroupingSet, WindowFrame, WindowFrameBound};

fn sum(parts: impl IntoIterator<Item = usize>) -> usize {
    parts.into_iter().fold(0, usize::saturating_add)
}

/// The list block a clone of `items` allocates, not what its items own.
fn list_block<T>(items: &[T]) -> usize {
    items.len().saturating_mul(size_of::<T>())
}

impl CloneBytes for Expr {
    fn clone_bytes(&self) -> usize {
        match self {
            Expr::Alias(Alias {
                expr,
                relation,
                name,
                metadata,
            }) => sum([
                expr.clone_bytes(),
                relation.clone_bytes(),
                name.clone_bytes(),
                metadata.clone_bytes(),
            ]),
            Expr::Column(column) => column.clone_bytes(),
            Expr::ScalarVariable(_, names) => names.clone_bytes(),
            Expr::Literal(value, metadata) => {
                sum([value.clone_bytes(), metadata.clone_bytes()])
            }
            Expr::BinaryExpr(BinaryExpr { left, right, .. }) => {
                sum([left.clone_bytes(), right.clone_bytes()])
            }
            Expr::Like(Like { expr, pattern, .. })
            | Expr::SimilarTo(Like { expr, pattern, .. }) => {
                sum([expr.clone_bytes(), pattern.clone_bytes()])
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
            | Expr::Unnest(Unnest { expr, .. }) => expr.clone_bytes(),
            Expr::Between(Between {
                expr, low, high, ..
            }) => sum([expr.clone_bytes(), low.clone_bytes(), high.clone_bytes()]),
            Expr::Case(Case {
                expr,
                when_then_expr,
                else_expr,
            }) => sum([
                expr.clone_bytes(),
                when_then_expr.clone_bytes(),
                else_expr.clone_bytes(),
            ]),
            Expr::ScalarFunction(ScalarFunction { args, .. })
            | Expr::HigherOrderFunction(HigherOrderFunction { args, .. }) => {
                args.clone_bytes()
            }
            Expr::AggregateFunction(AggregateFunction { params, .. }) => {
                params.clone_bytes()
            }
            Expr::WindowFunction(window) => window.clone_bytes(),
            Expr::InList(InList { expr, list, .. }) => {
                sum([expr.clone_bytes(), list.clone_bytes()])
            }
            Expr::Exists(Exists { subquery, .. }) | Expr::ScalarSubquery(subquery) => {
                subquery.clone_bytes()
            }
            Expr::InSubquery(InSubquery { expr, subquery, .. })
            | Expr::SetComparison(SetComparison { expr, subquery, .. }) => {
                sum([expr.clone_bytes(), subquery.clone_bytes()])
            }
            #[expect(deprecated)]
            Expr::Wildcard { qualifier, options } => {
                sum([qualifier.clone_bytes(), options.clone_bytes()])
            }
            Expr::GroupingSet(set) => set.clone_bytes(),
            Expr::Placeholder(Placeholder { id, field }) => {
                sum([id.clone_bytes(), field.clone_bytes()])
            }
            Expr::OuterReferenceColumn(_, column) => column.clone_bytes(),
            Expr::Lambda(Lambda { params, body }) => {
                sum([params.clone_bytes(), body.clone_bytes()])
            }
            Expr::LambdaVariable(LambdaVariable { name, spans, .. }) => {
                sum([name.clone_bytes(), spans.clone_bytes()])
            }
        }
    }
}

impl CloneBytes for Sort {
    fn clone_bytes(&self) -> usize {
        self.expr.clone_bytes()
    }
}

impl CloneBytes for AggregateFunctionParams {
    fn clone_bytes(&self) -> usize {
        sum([
            self.args.clone_bytes(),
            self.filter.clone_bytes(),
            self.order_by.clone_bytes(),
        ])
    }
}

impl CloneBytes for WindowFunction {
    fn clone_bytes(&self) -> usize {
        let WindowFunctionParams {
            args,
            partition_by,
            order_by,
            window_frame,
            filter,
            ..
        } = &self.params;
        sum([
            args.clone_bytes(),
            partition_by.clone_bytes(),
            order_by.clone_bytes(),
            window_frame.clone_bytes(),
            filter.clone_bytes(),
        ])
    }
}

impl CloneBytes for WindowFrame {
    fn clone_bytes(&self) -> usize {
        sum([self.start_bound.clone_bytes(), self.end_bound.clone_bytes()])
    }
}

impl CloneBytes for WindowFrameBound {
    fn clone_bytes(&self) -> usize {
        match self {
            WindowFrameBound::Preceding(value) | WindowFrameBound::Following(value) => {
                value.clone_bytes()
            }
            WindowFrameBound::CurrentRow => 0,
        }
    }
}

impl CloneBytes for Subquery {
    fn clone_bytes(&self) -> usize {
        sum([
            self.outer_ref_columns.clone_bytes(),
            self.spans.clone_bytes(),
        ])
    }
}

impl CloneBytes for GroupingSet {
    fn clone_bytes(&self) -> usize {
        match self {
            GroupingSet::Rollup(exprs) | GroupingSet::Cube(exprs) => exprs.clone_bytes(),
            GroupingSet::GroupingSets(sets) => sets.clone_bytes(),
        }
    }
}

impl CloneBytes for WildcardOptions {
    /// The options SQL planning makes are all absent. Any present option holds parsed SQL,
    /// whose clone is not measured: it is never funded.
    fn clone_bytes(&self) -> usize {
        let WildcardOptions {
            ilike,
            exclude,
            except,
            replace,
            rename,
        } = self;
        if ilike.is_none()
            && exclude.is_none()
            && except.is_none()
            && replace.is_none()
            && rename.is_none()
        {
            0
        } else {
            usize::MAX
        }
    }
}

impl CloneBytes for LogicalPlan {
    /// The node alone: its inputs and schemas are counted blocks the clone shares. A node
    /// holding parts whose clone is not measured is never funded.
    fn clone_bytes(&self) -> usize {
        match self {
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
                Partitioning::RoundRobinBatch(_) => 0,
                Partitioning::Hash(exprs, _) | Partitioning::DistributeBy(exprs) => {
                    exprs.clone_bytes()
                }
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
                    sum([
                        projection.as_deref().map_or(0, list_block),
                        filters.clone_bytes(),
                    ])
                } else {
                    usize::MAX
                }
            }
            LogicalPlan::EmptyRelation(_)
            | LogicalPlan::SubqueryAlias(SubqueryAlias { .. })
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
            LogicalPlan::Unnest(UnnestPlan {
                exec_columns,
                list_type_columns,
                struct_type_columns,
                dependency_indices,
                options,
                ..
            }) => sum([
                exec_columns.clone_bytes(),
                list_block(list_type_columns.as_slice()).saturating_add(sum(
                    list_type_columns.iter().map(
                        |(_, ColumnUnnestList { output_column, .. })| {
                            output_column.clone_bytes()
                        },
                    ),
                )),
                list_block(struct_type_columns),
                list_block(dependency_indices),
                options.clone_bytes(),
            ]),
            LogicalPlan::RecursiveQuery(RecursiveQuery { name, .. }) => {
                name.clone_bytes()
            }
            LogicalPlan::Statement(_)
            | LogicalPlan::Explain(_)
            | LogicalPlan::Analyze(_)
            | LogicalPlan::Dml(_)
            | LogicalPlan::Ddl(_)
            | LogicalPlan::Copy(_) => usize::MAX,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{col, lit};

    /// An expression's clone is measured block by block: each box, list and name.
    #[test]
    fn an_expression_clone_is_measured_block_by_block() {
        let expr = col("a").eq(lit("text")).alias("named");
        let bytes = size_of::<Expr>() // the alias's box
            + "named".len()
            + 2 * size_of::<Expr>() // the comparison's two boxes
            + "a".len()
            + "text".len();
        assert_eq!(expr.clone_bytes(), bytes);
    }

    /// A plan node's clone is measured without its inputs, which the clone shares.
    #[test]
    fn a_plan_clone_is_measured_without_its_inputs() {
        let input = std::sync::Arc::new(LogicalPlan::Filter(Filter::new(
            col("inner").eq(lit(1)),
            std::sync::Arc::new(LogicalPlan::default()),
        )));
        let plan = LogicalPlan::Filter(Filter::new(col("a").eq(lit(1)), input));
        assert_eq!(plan.clone_bytes(), 2 * size_of::<Expr>() + "a".len());
    }
}
