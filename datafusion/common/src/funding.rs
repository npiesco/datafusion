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

//! What planning allocates, each allocation charged to a [`PlanningFunding`] before it is
//! made: refused, nothing is allocated for it, and planning fails with
//! [`DataFusionError::Refused`].
//!
//! One [`Funding`] handle is carried by the session's [`ConfigOptions`], so every planning
//! stage reaches the same funding: SQL-to-plan (through its `ContextProvider`'s options), the
//! logical optimizer (through its `OptimizerConfig`'s options), the physical planner (through
//! the session's options) and the physical optimizer rules (handed the options). Cloning the
//! handle bumps a reference count and allocates nothing.
//!
//! Plans and schemas are plain `Box`, `Vec`, `String` and `Arc`, so no node can carry its own
//! charge. What planning made and handed out stays charged instead, exactly what it holds,
//! until its holder frees it: the plan a planning call returns holds what the funding still
//! has charged through this handle since the call began ([`Funding::outstanding`]), for its
//! owner to give back once it frees the plan. What planning makes only to read, and frees
//! itself, is given back as it frees it ([`Held`], [`Funding::released`]). What a failed
//! attempt made is freed with it, so planning that recovers from a failure gives back what was
//! made since it began ([`Funding::unwound`]).
//!
//! An `Arc` cloned is a count bumped, not an allocation: only an `Arc` newly made is charged,
//! once, for its whole block ([`Funding::arc`]).
//!
//! [`ConfigOptions`]: crate::config::ConfigOptions

use std::alloc::Layout;
use std::fmt;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::error::{DataFusionError, Result};

/// What planning charges each allocation to before it makes it. Refused, planning allocates
/// nothing for it and fails with [`DataFusionError::Refused`].
///
/// What it charges for what a plan keeps stays charged: plans outlive planning, so their owner
/// gives back what they were charged once it frees them. What planning frees itself, it
/// discharges as it frees it.
pub trait PlanningFunding: Send + Sync {
    /// Charges `bytes` before they are allocated: `false` refuses them.
    fn charge(&self, bytes: usize) -> bool;
    /// Gives back `bytes` charged before, now freed.
    fn discharge(&self, bytes: usize);
}

/// The funding a session plans within, carried by its options: a handle on a
/// [`PlanningFunding`], and on what has been charged through it and not yet given back. With
/// none ([`Funding::default`]), nothing is refused, but every allocation made through it is
/// still made fallibly: what the allocator refuses fails as refused rather than aborts.
///
/// One statement is planned within a handle at a time: what it has outstanding is the
/// statement's.
#[derive(Clone, Default)]
pub struct Funding(Option<Arc<Funded>>);

struct Funded {
    source: Arc<dyn PlanningFunding>,
    /// What was charged through the handle and not yet given back.
    outstanding: AtomicUsize,
}

impl fmt::Debug for Funding {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.0 {
            Some(funded) => write!(
                f,
                "Funding {{ outstanding: {} }}",
                funded.outstanding.load(Ordering::Relaxed)
            ),
            None => f.write_str("Funding(unfunded)"),
        }
    }
}

/// The refusal of `needed` bytes: nothing was allocated for them.
pub const fn refused(needed: usize) -> DataFusionError {
    DataFusionError::Refused { needed }
}

/// The block `Arc::new` (or `Rc::new`) allocates for a value laid out as `value`: its two
/// counts, then the value, padded to the alignment of both.
pub fn counted_block(value: Layout) -> Result<usize> {
    let counts = Layout::new::<[usize; 2]>();
    let (block, _) = counts.extend(value).map_err(|_| refused(usize::MAX))?;
    Ok(block.pad_to_align().size())
}

/// The block `Arc::new(value)` (or `Rc::new(value)`) allocates for a `T`
/// ([`counted_block`]).
pub const fn arc_bytes<T>() -> usize {
    let counts = 2 * size_of::<usize>();
    let align = if align_of::<T>() > align_of::<usize>() {
        align_of::<T>()
    } else {
        align_of::<usize>()
    };
    (counts.next_multiple_of(align_of::<T>()) + size_of::<T>()).next_multiple_of(align)
}

/// The block an `Arc<[T]>` (or `Rc<[T]>`) of `len` items is allocated in
/// ([`counted_block`]).
pub fn arc_slice_bytes<T>(len: usize) -> Result<usize> {
    counted_block(Layout::array::<T>(len).map_err(|_| refused(usize::MAX))?)
}

/// A buffer that is grown: its room is charged before it is allocated.
pub trait Buffer: Default {
    /// The bytes one item takes.
    const ITEM: usize;
    /// The items it holds.
    fn filled(&self) -> usize;
    /// The items its block has room for.
    fn room(&self) -> usize;
    /// Room for exactly `additional` more items than it holds; `false`, and nothing allocated,
    /// if the allocator refused it.
    fn grow_exact(&mut self, additional: usize) -> bool;
    /// Its block reallocated to room for exactly `room` items, no fewer than it holds.
    fn shrink_to(&mut self, room: usize);
    /// The bytes its block takes.
    fn block_bytes(&self) -> usize {
        self.room() * Self::ITEM
    }
}

impl Buffer for String {
    const ITEM: usize = 1;

    fn filled(&self) -> usize {
        self.len()
    }

    fn room(&self) -> usize {
        self.capacity()
    }

    fn grow_exact(&mut self, additional: usize) -> bool {
        self.try_reserve_exact(additional).is_ok()
    }

    fn shrink_to(&mut self, room: usize) {
        String::shrink_to(self, room);
    }
}

impl<T> Buffer for Vec<T> {
    const ITEM: usize = size_of::<T>();

    fn filled(&self) -> usize {
        self.len()
    }

    fn room(&self) -> usize {
        self.capacity()
    }

    fn grow_exact(&mut self, additional: usize) -> bool {
        self.try_reserve_exact(additional).is_ok()
    }

    fn shrink_to(&mut self, room: usize) {
        Vec::shrink_to(self, room);
    }
}

/// Counts the bytes a text is written in, allocating nothing.
struct Measure(usize);

impl fmt::Write for Measure {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        self.0 += s.len();
        Ok(())
    }
}

impl Funding {
    /// Planning charged to `source`.
    pub fn new(source: Arc<dyn PlanningFunding>) -> Self {
        Self(Some(Arc::new(Funded {
            source,
            outstanding: AtomicUsize::new(0),
        })))
    }

    /// Whether anything can be refused.
    pub fn is_funded(&self) -> bool {
        self.0.is_some()
    }

    /// The bytes charged through this handle and not yet given back: what everything made
    /// through it, and not yet freed, holds.
    pub fn outstanding(&self) -> usize {
        self.0
            .as_ref()
            .map_or(0, |funded| funded.outstanding.load(Ordering::Acquire))
    }

    /// `bytes` charged before they are allocated: refused, [`DataFusionError::Refused`].
    pub fn charge(&self, bytes: usize) -> Result<()> {
        match &self.0 {
            Some(funded) if bytes > 0 => {
                if !funded.source.charge(bytes) {
                    return Err(refused(bytes));
                }
                funded.outstanding.fetch_add(bytes, Ordering::AcqRel);
                Ok(())
            }
            _ => Ok(()),
        }
    }

    /// `bytes` charged before, now freed: given back.
    pub fn discharge(&self, bytes: usize) {
        if let Some(funded) = &self.0
            && bytes > 0
        {
            funded.outstanding.fetch_sub(bytes, Ordering::AcqRel);
            funded.source.discharge(bytes);
        }
    }

    /// What planning made and handed out, charged through this handle and not given back,
    /// handed on to whoever holds it: it holds that charge from now, and gives it back to the
    /// source itself once it frees what planning made. The handle has nothing outstanding then,
    /// for the next statement planned within it.
    pub fn hand_over(&self) -> usize {
        self.0
            .as_ref()
            .map_or(0, |funded| funded.outstanding.swap(0, Ordering::AcqRel))
    }

    /// What was made through this handle since it had `before` outstanding has all been
    /// freed (planning recovered from the attempt that made it): given back.
    pub fn unwound(&self, before: usize) {
        self.discharge(self.outstanding().saturating_sub(before));
    }

    /// `value` in a block of its own, charged before it is allocated. The charge goes with
    /// the box: whoever frees it gives back `size_of::<T>()` ([`Funding::unboxed`],
    /// [`Funding::released_box`]).
    pub fn boxed<T>(&self, value: T) -> Result<Box<T>> {
        let bytes = size_of::<T>();
        if bytes == 0 {
            // Nothing is allocated for it.
            return Ok(Box::new(value));
        }
        let mut slot = self.vec_with_capacity(1)?;
        slot.push(value);
        // Reserved exactly, the room is one item: boxing the slice reallocates nothing.
        debug_assert_eq!(slot.capacity(), 1);
        let raw = Box::into_raw(slot.into_boxed_slice()).cast::<T>();
        // SAFETY: the boxed slice holds exactly one `T`, in a block allocated for room for one
        // `T`, whose layout `Layout::array::<T>(1)` is `Layout::new::<T>()`: the layout a
        // `Box<T>` frees it with. The slice's ownership moves into the box.
        Ok(unsafe { Box::from_raw(raw) })
    }

    /// `value`, if any, in a block of its own ([`Funding::boxed`]).
    pub fn maybe_boxed<T>(&self, value: Option<T>) -> Result<Option<Box<T>>> {
        value.map(|value| self.boxed(value)).transpose()
    }

    /// The value `boxed` holds, moved out of its block: the block, freed, is given back.
    pub fn unboxed<T>(&self, boxed: Box<T>) -> T {
        let value = *boxed;
        self.discharge(size_of::<T>());
        value
    }

    /// `boxed`, a box made through this handle, freed: its block is given back. What its
    /// value holds is not.
    pub fn released_box<T>(&self, boxed: Box<T>) {
        drop(self.unboxed(boxed));
    }

    /// `value` in a counted block of its own, the whole block charged before it is
    /// allocated: the charge is the refusal point, `Arc::new` allocating only once it is
    /// taken. The charge goes with the block, given back by whoever frees its last handle.
    pub fn arc<T>(&self, value: T) -> Result<Arc<T>> {
        self.charge(arc_bytes::<T>())?;
        Ok(Arc::new(value))
    }

    /// As [`Funding::arc`], counted without atomics.
    pub fn rc<T>(&self, value: T) -> Result<Rc<T>> {
        self.charge(arc_bytes::<T>())?;
        Ok(Rc::new(value))
    }

    /// A copy of `items` in a counted block of its own, the block charged before it is
    /// allocated. What each item's clone allocates is not charged here: it is its maker's to
    /// charge.
    pub fn arc_slice<T: Clone>(&self, items: &[T]) -> Result<Arc<[T]>> {
        self.charge(arc_slice_bytes::<T>(items.len())?)?;
        Ok(Arc::from(items))
    }

    /// A copy of `text` in a counted block of its own, charged before it is allocated.
    pub fn arc_str(&self, text: &str) -> Result<Arc<str>> {
        self.charge(arc_slice_bytes::<u8>(text.len())?)?;
        Ok(Arc::from(text))
    }

    /// `items`, a list made through this handle, moved into a counted block of its own:
    /// the block charged before it is allocated, and the list's own block, freed then, given
    /// back. Refused, the list is freed with the refusal and its block given back too.
    pub fn arc_from_vec<T>(&self, items: Vec<T>) -> Result<Arc<[T]>> {
        let list = items.block_bytes();
        if let Err(refused) =
            arc_slice_bytes::<T>(items.len()).and_then(|block| self.charge(block))
        {
            drop(items);
            self.discharge(list);
            return Err(refused);
        }
        let made = Arc::from(items);
        self.discharge(list);
        Ok(made)
    }

    /// An empty list with room for exactly `capacity` items, charged before it is allocated.
    pub fn vec_with_capacity<T>(&self, capacity: usize) -> Result<Vec<T>> {
        let mut made = Vec::new();
        self.reserve_exact(&mut made, capacity)?;
        Ok(made)
    }

    /// A copy of `items` in a list of exactly their number. What each item's clone allocates
    /// is not charged here: it is its maker's to charge.
    pub fn vec_from_slice<T: Clone>(&self, items: &[T]) -> Result<Vec<T>> {
        let mut made = self.vec_with_capacity(items.len())?;
        made.extend_from_slice(items);
        Ok(made)
    }

    /// Exactly `items`, in a list of their number.
    pub fn vec_of<T, const N: usize>(&self, items: [T; N]) -> Result<Vec<T>> {
        let mut made = self.vec_with_capacity(N)?;
        made.extend(items);
        Ok(made)
    }

    /// `item` pushed onto `list`, a list made through this handle, its growth charged first.
    /// Refused, `item` is freed with the attempt that made it ([`Funding::unwound`]).
    pub fn push<T>(&self, list: &mut Vec<T>, item: T) -> Result<()> {
        self.reserve(list, 1)?;
        list.push(item);
        Ok(())
    }

    /// A copy of `text` in exactly its own bytes, charged before it is allocated.
    pub fn string(&self, text: &str) -> Result<String> {
        let mut made = String::new();
        self.reserve_exact(&mut made, text.len())?;
        made.push_str(text);
        Ok(made)
    }

    /// `more` appended to `text`, a text made through this handle, its growth charged first.
    pub fn push_str(&self, text: &mut String, more: &str) -> Result<()> {
        self.reserve(text, more.len())?;
        text.push_str(more);
        Ok(())
    }

    /// `args` written out, measured first so exactly its bytes are charged and allocated.
    pub fn format(&self, args: fmt::Arguments<'_>) -> Result<String> {
        let mut measure = Measure(0);
        fmt::write(&mut measure, args).map_err(|_| refused(0))?;
        let mut made = String::new();
        self.reserve_exact(&mut made, measure.0)?;
        if fmt::write(&mut made, args).is_err() || made.len() != measure.0 {
            // Written differently the second time: freed, what it was charged is given
            // back.
            drop(made);
            self.discharge(measure.0);
            return Err(refused(measure.0));
        }
        Ok(made)
    }

    /// Room for `additional` more items in `buffer`, made through this handle: the whole
    /// block it grows into is charged before it is allocated (the block it grew out of is
    /// freed only once that is), and the block it grew out of is given back once it is freed.
    /// Its room at least doubles as it grows.
    pub fn reserve<B: Buffer>(&self, buffer: &mut B, additional: usize) -> Result<()> {
        let (filled, room) = (buffer.filled(), buffer.room());
        if room - filled >= additional {
            return Ok(());
        }
        let wanted = filled.checked_add(additional).ok_or(refused(usize::MAX))?;
        let grown = wanted.max(room.saturating_mul(2)).max(4);
        self.grow_to(buffer, grown)
    }

    /// Room for exactly `additional` more items in `buffer`, charged as [`Funding::reserve`]
    /// charges it.
    pub fn reserve_exact<B: Buffer>(
        &self,
        buffer: &mut B,
        additional: usize,
    ) -> Result<()> {
        let (filled, room) = (buffer.filled(), buffer.room());
        if room - filled >= additional {
            return Ok(());
        }
        let wanted = filled.checked_add(additional).ok_or(refused(usize::MAX))?;
        self.grow_to(buffer, wanted)
    }

    fn grow_to<B: Buffer>(&self, buffer: &mut B, grown: usize) -> Result<()> {
        let (filled, before) = (buffer.filled(), buffer.block_bytes());
        let bytes = grown.checked_mul(B::ITEM).ok_or(refused(usize::MAX))?;
        self.charge(bytes)?;
        if !buffer.grow_exact(grown - filled) {
            self.discharge(bytes);
            return Err(refused(bytes));
        }
        // The block it grew into, read back: what it holds beyond what was asked is charged
        // too, or else let go.
        let made = buffer.block_bytes();
        if made > bytes && self.charge(made - bytes).is_err() {
            buffer.shrink_to(grown);
        }
        let made = buffer.block_bytes();
        if made < bytes {
            self.discharge(bytes - made);
        }
        self.discharge(before);
        Ok(())
    }

    /// `buffer`, made through this handle, freed: its block is given back. What its items
    /// hold is not.
    pub fn released<B: Buffer>(&self, buffer: B) {
        let bytes = buffer.block_bytes();
        drop(buffer);
        self.discharge(bytes);
    }

    /// An empty buffer, allocating nothing until it grows, given back as it is freed unless
    /// it is kept by what is made of it ([`Held::keep`]).
    pub fn held<B: Buffer>(&self) -> Held<B> {
        Held {
            buffer: B::default(),
            funding: self.clone(),
        }
    }
}

/// A buffer planning made, its room charged to its funding: given back as it is freed,
/// unless it is kept by what is made of it.
pub struct Held<B: Buffer> {
    buffer: B,
    funding: Funding,
}

impl<B: Buffer> Held<B> {
    /// Room for `additional` more items, charged as [`Funding::reserve`] charges it.
    pub fn reserve(&mut self, additional: usize) -> Result<()> {
        self.funding.reserve(&mut self.buffer, additional)
    }

    /// Room for exactly `additional` more items, charged as [`Funding::reserve_exact`]
    /// charges it.
    pub fn reserve_exact(&mut self, additional: usize) -> Result<()> {
        self.funding.reserve_exact(&mut self.buffer, additional)
    }

    /// The buffer, its charge going with it: whoever frees it gives that back
    /// ([`Funding::released`]).
    pub fn keep(mut self) -> B {
        std::mem::take(&mut self.buffer)
    }
}

impl<T> Held<Vec<T>> {
    /// `item` pushed, its growth charged first.
    pub fn push(&mut self, item: T) -> Result<()> {
        self.funding.push(&mut self.buffer, item)
    }
}

impl Held<String> {
    /// `more` appended, its growth charged first.
    pub fn push_str(&mut self, more: &str) -> Result<()> {
        self.funding.push_str(&mut self.buffer, more)
    }
}

impl<B: Buffer> std::ops::Deref for Held<B> {
    type Target = B;

    fn deref(&self) -> &B {
        &self.buffer
    }
}

impl<B: Buffer> std::ops::DerefMut for Held<B> {
    /// The buffer, to change in place: growing it other than through the [`Held`] leaves its
    /// growth uncharged.
    fn deref_mut(&mut self) -> &mut B {
        &mut self.buffer
    }
}

impl<B: Buffer> Drop for Held<B> {
    fn drop(&mut self) {
        let buffer = std::mem::take(&mut self.buffer);
        self.funding.released(buffer);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    /// Funding of `limit` bytes, recording what it holds.
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

    fn limited(limit: usize) -> (Arc<Limited>, Funding) {
        let source = Arc::new(Limited {
            limit,
            held: Mutex::new(0),
        });
        let funding = Funding::new(Arc::clone(&source) as Arc<dyn PlanningFunding>);
        (source, funding)
    }

    fn held(source: &Limited) -> usize {
        *source.held.lock().unwrap()
    }

    /// A counted block is charged exactly what `Arc::new` allocates, before it does; one
    /// short, it is refused for all of it, and nothing stays charged.
    #[test]
    fn an_arc_is_charged_its_whole_block_before_it_is_made() {
        assert_eq!(arc_bytes::<u64>(), 24);
        assert_eq!(arc_bytes::<u8>(), 24);
        assert_eq!(arc_bytes::<[u64; 3]>(), 40);
        assert_eq!(arc_slice_bytes::<u32>(3).unwrap(), 32);

        let (source, funding) = limited(arc_bytes::<u64>() - 1);
        match funding.arc(7u64) {
            Err(DataFusionError::Refused { needed }) => {
                assert_eq!(needed, arc_bytes::<u64>())
            }
            other => panic!("refused, not {other:?}"),
        }
        assert_eq!(held(&source), 0);
        assert_eq!(funding.outstanding(), 0);

        let (source, funding) = limited(arc_bytes::<u64>());
        let made = funding.arc(7u64).unwrap();
        assert_eq!(held(&source), arc_bytes::<u64>());
        assert_eq!(funding.outstanding(), arc_bytes::<u64>());
        drop(made);
        funding.discharge(arc_bytes::<u64>());
        assert_eq!(held(&source), 0);
    }

    /// A list's block is charged before it grows into it, and the block it grew out of is
    /// given back; a transient list gives back all it held as it is freed.
    #[test]
    fn a_list_is_charged_each_block_it_grows_into_and_gives_back_the_last() {
        let (source, funding) = limited(1 << 10);
        let mut list: Held<Vec<u32>> = funding.held();
        for item in 0..9 {
            list.push(item).unwrap();
        }
        assert_eq!(held(&source), list.capacity() * size_of::<u32>());
        drop(list);
        assert_eq!(held(&source), 0);

        let made: Vec<u64> = funding.vec_with_capacity(5).unwrap();
        assert_eq!(made.capacity(), 5);
        assert_eq!(held(&source), 5 * size_of::<u64>());
        funding.released(made);
        assert_eq!(held(&source), 0);
    }

    /// A box, a copy of a text and a formatted text are charged exactly their blocks; refused,
    /// nothing is allocated or stays charged.
    #[test]
    fn a_box_and_a_text_are_charged_exactly_their_blocks() {
        let (source, funding) = limited(64);
        let boxed = funding.boxed([0u8; 40]).unwrap();
        assert_eq!(held(&source), 40);
        match funding.boxed([0u8; 25]) {
            Err(DataFusionError::Refused { needed: 25 }) => {}
            other => panic!("refused, not {other:?}"),
        }
        assert_eq!(held(&source), 40);
        funding.released_box(boxed);
        assert_eq!(held(&source), 0);

        let text = funding.format(format_args!("{}-{}", "column", 12)).unwrap();
        assert_eq!(text, "column-12");
        assert_eq!(held(&source), text.capacity());
        let copy = funding.string(&text).unwrap();
        assert_eq!(held(&source), 2 * text.len());
        funding.released(text);
        funding.released(copy);
        assert_eq!(held(&source), 0);
    }

    /// What an attempt made, all freed as planning recovers from it, is given back.
    #[test]
    fn an_unwound_attempt_gives_back_what_it_made() {
        let (source, funding) = limited(1 << 10);
        let kept = funding.string("kept").unwrap();
        let before = funding.outstanding();
        let attempt = (funding.string("one"), funding.boxed(3u64));
        drop(attempt);
        funding.unwound(before);
        assert_eq!(held(&source), kept.capacity());
        funding.released(kept);
        assert_eq!(held(&source), 0);
    }

    /// A list refused its counted block is freed, and its own block given back, whether the
    /// refusal is propagated directly or the attempt is unwound.
    #[test]
    fn a_list_refused_its_counted_block_gives_back_its_own() {
        let list_bytes = 4 * size_of::<u64>();
        let (source, funding) = limited(list_bytes);
        let items: Vec<u64> = funding.vec_from_slice(&[1, 2, 3, 4]).unwrap();
        assert_eq!(held(&source), list_bytes);
        match funding.arc_from_vec(items) {
            Err(DataFusionError::Refused { needed }) => {
                assert_eq!(needed, arc_slice_bytes::<u64>(4).unwrap())
            }
            other => panic!("refused, not {other:?}"),
        }
        assert_eq!(held(&source), 0);
        assert_eq!(funding.outstanding(), 0);

        let before = funding.outstanding();
        let items: Vec<u64> = funding.vec_from_slice(&[1, 2, 3, 4]).unwrap();
        assert!(funding.arc_from_vec(items).is_err());
        funding.unwound(before);
        assert_eq!(held(&source), 0);
        assert_eq!(funding.outstanding(), 0);

        let (source, funding) = limited(list_bytes + arc_slice_bytes::<u64>(4).unwrap());
        let items: Vec<u64> = funding.vec_from_slice(&[1, 2, 3, 4]).unwrap();
        let made = funding.arc_from_vec(items).unwrap();
        assert_eq!(held(&source), arc_slice_bytes::<u64>(4).unwrap());
        drop(made);
        funding.discharge(arc_slice_bytes::<u64>(4).unwrap());
        assert_eq!(held(&source), 0);
    }

    /// What planning made, handed over, is no longer the handle's: given back by its holder,
    /// straight to the source.
    #[test]
    fn a_plan_handed_over_leaves_the_handle_nothing_outstanding() {
        let (source, funding) = limited(1 << 10);
        let kept = funding.string("kept").unwrap();
        assert_eq!(funding.hand_over(), kept.capacity());
        assert_eq!(funding.outstanding(), 0);
        assert_eq!(held(&source), kept.capacity());
        funding.unwound(0);
        assert_eq!(held(&source), kept.capacity(), "nothing of it is unwound");
        source.discharge(kept.capacity());
        drop(kept);
        assert_eq!(held(&source), 0);
    }
}
