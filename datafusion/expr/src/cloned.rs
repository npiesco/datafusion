// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! What cloning an expression allocates ([`CloneBytes`]), measured so the clone is charged
//! before it is made.

use datafusion_common::cloned::CloneBytes;

use crate::expr::{
    AggregateFunction, AggregateFunctionParams, Alias, Between, BinaryExpr, Case, Cast,
    Exists, HigherOrderFunction, InList, InSubquery, Lambda, LambdaVariable, Like,
    Placeholder, ScalarFunction, SetComparison, Sort, TryCast, Unnest, WildcardOptions,
    WindowFunction, WindowFunctionParams,
};
use crate::logical_plan::Subquery;
use crate::{Expr, GroupingSet, WindowFrame, WindowFrameBound};

fn sum(parts: impl IntoIterator<Item = usize>) -> usize {
    parts.into_iter().fold(0, usize::saturating_add)
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
}
