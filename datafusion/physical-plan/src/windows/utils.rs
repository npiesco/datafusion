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

use arrow::datatypes::{FieldRef, Schema};
use datafusion_common::Result;
use datafusion_common::funding::Funding;
use datafusion_physical_expr::window::WindowExpr;
use std::sync::Arc;

pub(crate) fn create_schema(
    input_schema: &Schema,
    window_expr: &[Arc<dyn WindowExpr>],
    funding: &Funding,
) -> Result<Schema> {
    let mut fields = funding.held::<Vec<FieldRef>>();
    fields.reserve_exact(input_schema.fields().len() + window_expr.len())?;
    fields.extend(input_schema.fields().iter().cloned());
    // append results to the schema
    for expr in window_expr {
        fields.push(expr.field()?)?;
    }
    Ok(Schema::new_with_metadata(
        funding.fields(&fields)?,
        input_schema.metadata().clone(),
    ))
}
