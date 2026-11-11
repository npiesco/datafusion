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

use crate::joins::utils::ColumnIndex;
use arrow::datatypes::SchemaRef;
use datafusion_common::JoinSide;
use datafusion_common::Result;
use datafusion_common::funding::Funding;
use datafusion_physical_expr_common::physical_expr::PhysicalExpr;
use std::{fmt::Display, sync::Arc};

/// Filter applied before join output. Fields are crate-public to allow
/// downstream implementations to experiment with custom joins.
#[derive(Debug, Clone)]
pub struct JoinFilter {
    /// Filter expression
    pub(crate) expression: Arc<dyn PhysicalExpr>,
    /// Column indices required to construct intermediate batch for filtering
    pub(crate) column_indices: Vec<ColumnIndex>,
    /// Physical schema of intermediate batch
    pub(crate) schema: SchemaRef,
}

/// For display in `EXPLAIN` plans, only expression with column names is needed,
/// it output expression like `(col1 + col2) = 0`
impl Display for JoinFilter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.expression.fmt_sql(f)
    }
}

impl JoinFilter {
    /// Creates new JoinFilter
    pub fn new(
        expression: Arc<dyn PhysicalExpr>,
        column_indices: Vec<ColumnIndex>,
        schema: SchemaRef,
    ) -> JoinFilter {
        JoinFilter {
            expression,
            column_indices,
            schema,
        }
    }

    /// Helper for building ColumnIndex vector from left and right indices, charged to
    /// `funding` first; the index vectors are given back.
    pub fn build_column_indices(
        left_indices: Vec<usize>,
        right_indices: Vec<usize>,
        funding: &Funding,
    ) -> Result<Vec<ColumnIndex>> {
        let mut column_indices = funding.held::<Vec<ColumnIndex>>();
        column_indices.reserve_exact(left_indices.len() + right_indices.len())?;
        for (indices, side) in [
            (&left_indices, JoinSide::Left),
            (&right_indices, JoinSide::Right),
        ] {
            for &index in indices {
                column_indices.push(ColumnIndex { index, side })?;
            }
        }
        funding.released(left_indices);
        funding.released(right_indices);
        Ok(column_indices.keep())
    }

    /// A copy of this filter, its column indices charged to `funding` first.
    pub fn funded_clone(&self, funding: &Funding) -> Result<JoinFilter> {
        Ok(JoinFilter::new(
            Arc::clone(&self.expression),
            funding.vec_from_slice(&self.column_indices)?,
            Arc::clone(&self.schema),
        ))
    }

    /// Filter expression
    pub fn expression(&self) -> &Arc<dyn PhysicalExpr> {
        &self.expression
    }

    /// Column indices for intermediate batch creation
    pub fn column_indices(&self) -> &[ColumnIndex] {
        &self.column_indices
    }

    /// Intermediate batch schema
    pub fn schema(&self) -> &SchemaRef {
        &self.schema
    }

    /// Rewrites the join filter if the inputs to the join are rewritten, the new column
    /// indices charged to `funding` first
    pub fn swap(&self, funding: &Funding) -> Result<JoinFilter> {
        let mut column_indices = funding.held::<Vec<ColumnIndex>>();
        column_indices.reserve_exact(self.column_indices.len())?;
        for idx in self.column_indices() {
            column_indices.push(ColumnIndex {
                index: idx.index,
                side: idx.side.negate(),
            })?;
        }

        Ok(JoinFilter::new(
            Arc::clone(self.expression()),
            column_indices.keep(),
            Arc::clone(self.schema()),
        ))
    }
}
