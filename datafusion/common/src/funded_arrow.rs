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

//! Arrow schema parts made through a [`Funding`], charged before they are allocated.

use arrow::datatypes::{FieldRef, Fields};

use crate::error::Result;
use crate::funding::{Funding, arc_slice_bytes};

impl Funding {
    /// A copy of `fields` in the counted block of a [`Fields`] of their number, charged
    /// before it is allocated.
    pub fn fields(&self, fields: &[FieldRef]) -> Result<Fields> {
        self.charge(arc_slice_bytes::<FieldRef>(fields.len())?)?;
        Ok(Fields::from(fields))
    }
}
