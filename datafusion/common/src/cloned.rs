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

//! What cloning a value allocates, measured without allocating, so the clone is charged
//! before it is made ([`Funding::cloned`]).
//!
//! A cloned list or text is given room for exactly what it holds, a cloned box a block of
//! its value's size, and a cloned counted block (`Arc`, `Rc`) is a count bumped: nothing.

use std::rc::Rc;
use std::sync::Arc;

use arrow::datatypes::DataType;

use crate::error::Result;
use crate::funding::{Funding, arc_bytes};
use crate::metadata::FieldMetadata;
use crate::spans::Spans;
use crate::{
    Column, DFSchema, RecursionUnnestOption, ScalarValue, TableReference, UnnestOptions,
};

/// A value whose clone is measured before it is made.
pub trait CloneBytes: Clone {
    /// The bytes `clone` allocates for it, every block exactly as the clone sizes it.
    fn clone_bytes(&self) -> usize;
}

/// The bytes cloning `items` allocates: their list's block, room for exactly them, and what
/// each item's clone allocates.
pub fn slice_clone_bytes<T: CloneBytes>(items: &[T]) -> usize {
    items
        .iter()
        .fold(size_of::<T>().saturating_mul(items.len()), |bytes, item| {
            bytes.saturating_add(item.clone_bytes())
        })
}

impl Funding {
    /// A clone of `value`, all it allocates charged before it is made.
    pub fn cloned<T: CloneBytes>(&self, value: &T) -> Result<T> {
        self.charge(value.clone_bytes())?;
        Ok(value.clone())
    }

    /// `value`, a clone made through this handle ([`Funding::cloned`]), freed: what its clone
    /// allocated is given back.
    pub fn released_clone<T: CloneBytes>(&self, value: T) {
        let bytes = value.clone_bytes();
        drop(value);
        self.discharge(bytes);
    }

    /// A copy of `items`, in a list with room for exactly them, all it allocates charged
    /// before it is made.
    pub fn to_vec<T: CloneBytes>(&self, items: &[T]) -> Result<Vec<T>> {
        self.charge(slice_clone_bytes(items))?;
        Ok(items.to_vec())
    }

    /// The value out of `shared`, a counted block charged through this handle
    /// ([`Funding::arc`]): when `shared` was its last handle the block is freed and given
    /// back; otherwise the value is copied, charged ([`Funding::cloned`]).
    pub fn unwrapped<T: CloneBytes>(&self, shared: Arc<T>) -> Result<T> {
        match Arc::try_unwrap(shared) {
            Ok(value) => {
                self.discharge(arc_bytes::<T>());
                Ok(value)
            }
            Err(shared) => self.cloned(shared.as_ref()),
        }
    }
}

macro_rules! clone_allocates_nothing {
    ($($ty:ty),* $(,)?) => {
        $(impl CloneBytes for $ty {
            fn clone_bytes(&self) -> usize {
                0
            }
        })*
    };
}

clone_allocates_nothing!(
    bool,
    char,
    u8,
    u16,
    u32,
    u64,
    usize,
    i8,
    i16,
    i32,
    i64,
    isize,
    TableReference,
    FieldMetadata,
);

impl CloneBytes for String {
    fn clone_bytes(&self) -> usize {
        self.len()
    }
}

impl<T: CloneBytes> CloneBytes for Vec<T> {
    fn clone_bytes(&self) -> usize {
        slice_clone_bytes(self)
    }
}

impl<T: CloneBytes> CloneBytes for Box<T> {
    fn clone_bytes(&self) -> usize {
        size_of::<T>().saturating_add(self.as_ref().clone_bytes())
    }
}

impl<T: CloneBytes> CloneBytes for Option<T> {
    fn clone_bytes(&self) -> usize {
        self.as_ref().map_or(0, CloneBytes::clone_bytes)
    }
}

impl<T: ?Sized> CloneBytes for Arc<T> {
    fn clone_bytes(&self) -> usize {
        0
    }
}

impl<T: ?Sized> CloneBytes for Rc<T> {
    fn clone_bytes(&self) -> usize {
        0
    }
}

impl<A: CloneBytes, B: CloneBytes> CloneBytes for (A, B) {
    fn clone_bytes(&self) -> usize {
        self.0.clone_bytes().saturating_add(self.1.clone_bytes())
    }
}

impl<A: CloneBytes, B: CloneBytes, C: CloneBytes> CloneBytes for (A, B, C) {
    fn clone_bytes(&self) -> usize {
        self.0
            .clone_bytes()
            .saturating_add(self.1.clone_bytes())
            .saturating_add(self.2.clone_bytes())
    }
}

impl CloneBytes for Spans {
    fn clone_bytes(&self) -> usize {
        self.0.len().saturating_mul(size_of::<crate::spans::Span>())
    }
}

impl CloneBytes for Column {
    fn clone_bytes(&self) -> usize {
        self.name.len().saturating_add(self.spans.clone_bytes())
    }
}

impl CloneBytes for DataType {
    fn clone_bytes(&self) -> usize {
        match self {
            DataType::Dictionary(key, value) => {
                key.clone_bytes().saturating_add(value.clone_bytes())
            }
            // Every other type holds its parts in counted blocks, or inline.
            _ => 0,
        }
    }
}

impl CloneBytes for ScalarValue {
    fn clone_bytes(&self) -> usize {
        match self {
            ScalarValue::Utf8(text)
            | ScalarValue::Utf8View(text)
            | ScalarValue::LargeUtf8(text) => text.clone_bytes(),
            ScalarValue::Binary(bytes)
            | ScalarValue::BinaryView(bytes)
            | ScalarValue::FixedSizeBinary(_, bytes)
            | ScalarValue::LargeBinary(bytes) => bytes.as_ref().map_or(0, Vec::len),
            ScalarValue::Union(value, _, _) => {
                value.as_ref().map_or(0, |(_, value)| value.clone_bytes())
            }
            ScalarValue::Dictionary(key, value) => {
                key.clone_bytes().saturating_add(value.clone_bytes())
            }
            ScalarValue::RunEndEncoded(_, _, value) => value.clone_bytes(),
            // Every other value is inline, or held in counted blocks.
            _ => 0,
        }
    }
}

impl CloneBytes for DFSchema {
    /// Its Arrow schema is a counted block; its qualifiers, one per field, and its
    /// functional dependencies are cloned into lists of their own.
    fn clone_bytes(&self) -> usize {
        size_of::<Option<TableReference>>()
            .saturating_mul(self.fields().len())
            .saturating_add(self.functional_dependencies().clone_bytes())
    }
}

impl CloneBytes for UnnestOptions {
    fn clone_bytes(&self) -> usize {
        self.recursions.clone_bytes()
    }
}

impl CloneBytes for RecursionUnnestOption {
    fn clone_bytes(&self) -> usize {
        self.input_column
            .clone_bytes()
            .saturating_add(self.output_column.clone_bytes())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::funding::PlanningFunding;
    use std::sync::Mutex;

    struct Limited {
        limit: usize,
        held: Mutex<usize>,
    }

    impl PlanningFunding for Limited {
        fn charge(&self, bytes: usize) -> bool {
            let mut held = self.held.lock().unwrap();
            if *held + bytes > self.limit {
                return false;
            }
            *held += bytes;
            true
        }

        fn discharge(&self, bytes: usize) {
            *self.held.lock().unwrap() -= bytes;
        }
    }

    /// A clone is charged exactly the blocks it allocates, before it makes them; one byte
    /// short, it is refused and nothing stays charged.
    #[test]
    fn a_clone_is_charged_exactly_its_blocks() {
        let column = Column::new(Some("t"), "column_a");
        let value = ScalarValue::Dictionary(
            Box::new(DataType::Int32),
            Box::new(ScalarValue::Utf8(Some("dictionary".into()))),
        );
        let made = (Box::new(column), vec![value]);
        let bytes = size_of::<Column>()
            + "column_a".len()
            + size_of::<ScalarValue>()
            + size_of::<DataType>()
            + size_of::<ScalarValue>()
            + "dictionary".len();
        assert_eq!(made.clone_bytes(), bytes);

        let source = Arc::new(Limited {
            limit: bytes - 1,
            held: Mutex::new(0),
        });
        let funding = Funding::new(Arc::clone(&source) as Arc<dyn PlanningFunding>);
        assert!(funding.cloned(&made).is_err());
        assert_eq!(*source.held.lock().unwrap(), 0);

        let source = Arc::new(Limited {
            limit: bytes,
            held: Mutex::new(0),
        });
        let funding = Funding::new(Arc::clone(&source) as Arc<dyn PlanningFunding>);
        let clone = funding.cloned(&made).unwrap();
        assert_eq!(*source.held.lock().unwrap(), bytes);
        funding.released_clone(clone);
        assert_eq!(*source.held.lock().unwrap(), 0);
    }
}
