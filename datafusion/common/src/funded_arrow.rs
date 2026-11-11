// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! Arrow schema parts made through a [`Funding`], charged before they are allocated.

use std::sync::Arc;

use arrow::datatypes::{DataType, Field, FieldRef, Fields};

use crate::ScalarValue;
use crate::cloned::CloneBytes;
use crate::error::Result;
use crate::funding::{Funding, arc_bytes, arc_slice_bytes};

/// The blocks [`ScalarValue::data_type`] allocates for the type of `value`: a dictionary's
/// copied key type and boxed value type.
fn scalar_type_bytes(value: &ScalarValue) -> usize {
    match value {
        ScalarValue::Dictionary(key, value) => size_of::<DataType>()
            .saturating_add(key.clone_bytes())
            .saturating_add(scalar_type_bytes(value)),
        // Every other type is inline, or shares its parts' counted blocks.
        _ => 0,
    }
}

impl Funding {
    /// A copy of `fields` in the counted block of a [`Fields`] of their number, charged
    /// before it is allocated.
    pub fn fields(&self, fields: &[FieldRef]) -> Result<Fields> {
        self.charge(arc_slice_bytes::<FieldRef>(fields.len())?)?;
        Ok(Fields::from(fields))
    }

    /// The type of `value`, what making it allocates charged first.
    pub fn scalar_type(&self, value: &ScalarValue) -> Result<DataType> {
        self.charge(scalar_type_bytes(value))?;
        Ok(value.data_type())
    }

    /// `field` named `name`: itself if it already is, otherwise a copy so named in a counted
    /// block of its own, its block, name and type charged before they are allocated. A
    /// non-empty metadata map is copied uncharged: its table's size is not measured.
    pub fn renamed_field(&self, field: FieldRef, name: &str) -> Result<FieldRef> {
        if field.name() == name {
            return Ok(field);
        }
        let copied_name = field.name().len();
        self.charge(
            copied_name
                .saturating_add(field.data_type().clone_bytes())
                .saturating_add(arc_bytes::<Field>()),
        )?;
        let mut renamed = Field::clone(&field);
        renamed.set_name(self.string(name)?);
        // The copied name is freed as the new one replaces it.
        self.discharge(copied_name);
        Ok(Arc::new(renamed))
    }
}
