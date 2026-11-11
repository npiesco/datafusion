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
use std::future::Future;
use std::hash::{BuildHasher, Hash};
use std::iter::FusedIterator;
use std::ops::{Index, IndexMut};
use std::pin::Pin;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::task::{Context, Poll};

use hashbrown::DefaultHashBuilder;
use indexmap::Equivalent;

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
    #[expect(
        clippy::boxed_local,
        reason = "Consuming the owned heap block lets its funding be discharged."
    )]
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

    /// `future` in a block of its own, to be awaited behind a pointer (a recursive or
    /// dynamically dispatched planning step): the block charged before it is allocated, and
    /// given back as it is freed, once the future has finished or is dropped unfinished.
    pub fn held_future<'a, F>(&self, future: F) -> Result<HeldFuture<'a, F::Output>>
    where
        F: Future + Send + 'a,
    {
        let block = size_of::<F>();
        let boxed: Pin<Box<dyn Future<Output = F::Output> + Send + 'a>> =
            Box::into_pin(self.boxed(future)?);
        Ok(HeldFuture {
            future: Some(boxed),
            block,
            funding: self.clone(),
        })
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

/// A future planning boxed ([`Funding::held_future`]): its block, charged before it was
/// allocated, is given back as it is freed, as soon as the future finishes or once it is
/// dropped unfinished.
pub struct HeldFuture<'a, T> {
    future: Option<Pin<Box<dyn Future<Output = T> + Send + 'a>>>,
    block: usize,
    funding: Funding,
}

impl<T> HeldFuture<'_, T> {
    fn release(&mut self) {
        if let Some(future) = self.future.take() {
            drop(future);
            self.funding.discharge(self.block);
        }
    }
}

impl<T> Future for HeldFuture<'_, T> {
    type Output = T;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<T> {
        let this = self.get_mut();
        let Some(future) = this.future.as_mut() else {
            panic!("a held planning future polled after it finished");
        };
        let polled = future.as_mut().poll(cx);
        if polled.is_ready() {
            this.release();
        }
        polled
    }
}

impl<T> Drop for HeldFuture<'_, T> {
    fn drop(&mut self) {
        self.release();
    }
}

/// An insertion-ordered hash map backed entirely by three [`Vec`] buffers (`entries`,
/// `hashes`, and `slots`) whose exact heap capacities are pre-charged to a [`Funding`]
/// handle before every growth step and discharged when the map is dropped.
///
/// Unlike `IndexMap` or `hashbrown::HashMap`, [`FundedMap`] owns no opaque SwissTable
/// allocation: every byte it allocates on the global heap is accounted for in
/// `entries.block_bytes() + hashes.block_bytes() + slots.block_bytes()`.
pub struct FundedMap<K, V, S = DefaultHashBuilder> {
    entries: Vec<(K, V)>,
    hashes: Vec<u64>,
    slots: Vec<u32>,
    hash_builder: S,
    funding: Funding,
    charged: usize,
}

impl<K, V> FundedMap<K, V, DefaultHashBuilder> {
    /// Unfunded empty [`FundedMap`] for default initialization prior to rebinding.
    pub fn new() -> Self {
        Self::with_funding(&Funding::default())
    }

    /// An empty [`FundedMap`] tied to `funding`. Allocates zero bytes until the first
    /// reservation or insertion.
    pub fn with_funding(funding: &Funding) -> Self {
        Self::with_hasher(DefaultHashBuilder::default(), funding)
    }
}

impl<K, V, S: Default> FundedMap<K, V, S> {
    /// An empty [`FundedMap`] with default hasher `S` tied to `funding`.
    pub fn with_hasher_and_funding(funding: &Funding) -> Self {
        Self::with_hasher(S::default(), funding)
    }
}

impl<K, V, S: Default> Default for FundedMap<K, V, S> {
    fn default() -> Self {
        Self::with_hasher_and_funding(&Funding::default())
    }
}

impl<K, V, S> FundedMap<K, V, S> {
    /// Bytes per entry at the 50% probe-table load factor (`(K, V)` + `u64` hash + two
    /// `u32` probe slots).
    pub const ENTRY_BYTES: usize =
        size_of::<(K, V)>() + size_of::<u64>() + (2 * size_of::<u32>());

    /// An empty [`FundedMap`] with explicit `hash_builder` tied to `funding`.
    pub fn with_hasher(hash_builder: S, funding: &Funding) -> Self {
        Self {
            entries: Vec::new(),
            hashes: Vec::new(),
            slots: Vec::new(),
            hash_builder,
            funding: funding.clone(),
            charged: 0,
        }
    }

    /// Number of key-value pairs in the map.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the map contains no elements.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Number of elements the map can hold without reallocating any of its three buffers.
    pub fn capacity(&self) -> usize {
        self.entries
            .capacity()
            .min(self.hashes.capacity())
            .min(self.slots.len() / 2)
    }

    /// Exact bytes currently charged to this map's [`Funding`] handle.
    pub fn charged_bytes(&self) -> usize {
        self.charged
    }

    /// The [`Funding`] handle this map charges its capacity to.
    pub fn funding(&self) -> &Funding {
        &self.funding
    }

    /// Reference to the map's [`BuildHasher`].
    pub fn hasher(&self) -> &S {
        &self.hash_builder
    }

    /// Transfers this map's standing capacity charge from its current [`Funding`] handle to
    /// `next`, charging `next` first so a refusal leaves the map and its existing funding
    /// unchanged.
    pub fn rebind(&mut self, next: &Funding) -> Result<()> {
        if self.charged > 0 {
            next.charge(self.charged)?;
            self.funding.discharge(self.charged);
        }
        self.funding = next.clone();
        Ok(())
    }

    /// Detaches this map's standing charge from its [`Funding`] handle without discharging
    /// the handle, returning the detached byte count so the caller can assume responsibility
    /// for discharging it later.
    pub fn detach_funding(&mut self) -> usize {
        self.funding = Funding::default();
        std::mem::replace(&mut self.charged, 0)
    }

    /// Clears all entries while retaining the allocated capacity and its standing charge.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.hashes.clear();
        self.slots.fill(0);
    }

    /// Shortens the map, keeping the first `len` elements and dropping the rest while
    /// retaining allocated capacity and its standing charge.
    pub fn truncate(&mut self, len: usize) {
        if len < self.entries.len() {
            self.entries.truncate(len);
            self.hashes.truncate(len);
            self.rebuild_slots();
        }
    }

    fn required_slots(entry_cap: usize) -> Result<usize> {
        if entry_cap == 0 {
            return Ok(0);
        }
        if entry_cap >= (u32::MAX as usize) {
            return Err(refused(usize::MAX));
        }
        let min_slots = entry_cap
            .checked_mul(2)
            .ok_or_else(|| refused(usize::MAX))?
            .max(4);
        min_slots
            .checked_next_power_of_two()
            .ok_or_else(|| refused(usize::MAX))
    }

    fn rebuild_slots(&mut self) {
        self.slots.fill(0);
        if self.slots.is_empty() {
            return;
        }
        let mask = self.slots.len() - 1;
        for (idx, &hash) in self.hashes.iter().enumerate() {
            let mut slot = (hash as usize) & mask;
            while self.slots[slot] != 0 {
                slot = (slot + 1) & mask;
            }
            self.slots[slot] = (idx + 1) as u32;
        }
    }

    fn find_slot<Q>(
        &self,
        hash: u64,
        key: &Q,
    ) -> std::result::Result<(usize, usize), usize>
    where
        Q: ?Sized + Equivalent<K>,
    {
        debug_assert!(!self.slots.is_empty());
        let mask = self.slots.len() - 1;
        let mut slot = (hash as usize) & mask;
        loop {
            let raw = self.slots[slot];
            if raw == 0 {
                return Err(slot);
            }
            let idx = (raw - 1) as usize;
            if self.hashes[idx] == hash && key.equivalent(&self.entries[idx].0) {
                return Ok((slot, idx));
            }
            slot = (slot + 1) & mask;
        }
    }

    fn grow_to(&mut self, additional: usize, exact: bool) -> Result<()> {
        let filled = self.entries.len();
        let room = self.capacity();
        if room.saturating_sub(filled) >= additional {
            return Ok(());
        }
        let wanted = filled
            .checked_add(additional)
            .ok_or_else(|| refused(usize::MAX))?;
        let target_entries = if exact {
            wanted
        } else {
            wanted.max(room.saturating_mul(2)).max(4)
        };
        let old_entries_cap = self.entries.capacity();
        let old_hashes_cap = self.hashes.capacity();
        let new_entries_cap = target_entries.max(old_entries_cap);
        let new_hashes_cap = target_entries.max(old_hashes_cap);
        let new_slots_cap =
            Self::required_slots(target_entries)?.max(self.slots.capacity());

        if new_entries_cap > old_entries_cap {
            self.funding.grow_to(&mut self.entries, new_entries_cap)?;
            self.charged = self.entries.block_bytes()
                + self.hashes.block_bytes()
                + self.slots.block_bytes();
        }
        if new_hashes_cap > old_hashes_cap {
            if let Err(err) = self.funding.grow_to(&mut self.hashes, new_hashes_cap) {
                if old_entries_cap == 0 {
                    self.funding.released(std::mem::take(&mut self.entries));
                }
                self.charged = self.entries.block_bytes()
                    + self.hashes.block_bytes()
                    + self.slots.block_bytes();
                return Err(err);
            }
            self.charged = self.entries.block_bytes()
                + self.hashes.block_bytes()
                + self.slots.block_bytes();
        }
        if new_slots_cap > self.slots.capacity() {
            if let Err(err) = self.funding.grow_to(&mut self.slots, new_slots_cap) {
                if old_hashes_cap == 0 {
                    self.funding.released(std::mem::take(&mut self.hashes));
                }
                if old_entries_cap == 0 {
                    self.funding.released(std::mem::take(&mut self.entries));
                }
                self.charged = self.entries.block_bytes()
                    + self.hashes.block_bytes()
                    + self.slots.block_bytes();
                return Err(err);
            }
            self.charged = self.entries.block_bytes()
                + self.hashes.block_bytes()
                + self.slots.block_bytes();
        }

        if self.slots.len() < new_slots_cap {
            self.slots.resize(new_slots_cap, 0);
            self.rebuild_slots();
        }
        Ok(())
    }

    /// Reserves capacity for at least `additional` more entries, pre-charging the exact byte
    /// delta across `entries`, `hashes`, and `slots` before any allocation occurs.
    pub fn reserve(&mut self, additional: usize) -> Result<()> {
        self.grow_to(additional, false)
    }

    /// Reserves exact capacity for `additional` more entries, pre-charging the exact byte
    /// delta across `entries`, `hashes`, and `slots` before any allocation occurs.
    pub fn reserve_exact(&mut self, additional: usize) -> Result<()> {
        self.grow_to(additional, true)
    }

    /// Returns a key-value pair by insertion index.
    pub fn get_index(&self, index: usize) -> Option<(&K, &V)> {
        self.entries.get(index).map(|(k, v)| (k, v))
    }

    /// Returns a key and mutable value reference by insertion index.
    pub fn get_index_mut(&mut self, index: usize) -> Option<(&K, &mut V)> {
        self.entries.get_mut(index).map(|(k, v)| (&*k, v))
    }

    /// Returns the first key-value pair.
    pub fn first(&self) -> Option<(&K, &V)> {
        self.get_index(0)
    }

    /// Returns the last key-value pair.
    pub fn last(&self) -> Option<(&K, &V)> {
        self.entries.last().map(|(k, v)| (k, v))
    }

    /// Removes and returns the last key-value pair, retaining allocated capacity.
    pub fn pop(&mut self) -> Option<(K, V)> {
        let removed = self.entries.pop()?;
        self.hashes.pop();
        self.rebuild_slots();
        Some(removed)
    }

    /// Removes the key-value pair at `index` by swapping it with the last element.
    pub fn swap_remove_index(&mut self, index: usize) -> Option<(K, V)> {
        if index >= self.entries.len() {
            return None;
        }
        let removed = self.entries.swap_remove(index);
        self.hashes.swap_remove(index);
        self.rebuild_slots();
        Some(removed)
    }

    /// Removes the key-value pair at `index` and shifts subsequent elements down to preserve
    /// insertion order.
    pub fn shift_remove_index(&mut self, index: usize) -> Option<(K, V)> {
        if index >= self.entries.len() {
            return None;
        }
        let removed = self.entries.remove(index);
        self.hashes.remove(index);
        self.rebuild_slots();
        Some(removed)
    }

    /// Iterator over `(&K, &V)` in insertion order.
    pub fn iter(&self) -> FundedMapIter<'_, K, V> {
        FundedMapIter {
            inner: self.entries.iter(),
        }
    }

    /// Mutable iterator over `(&K, &mut V)` in insertion order.
    pub fn iter_mut(&mut self) -> FundedMapIterMut<'_, K, V> {
        FundedMapIterMut {
            inner: self.entries.iter_mut(),
        }
    }

    /// Iterator over `&K` in insertion order.
    pub fn keys(&self) -> FundedMapKeys<'_, K, V> {
        FundedMapKeys {
            inner: self.entries.iter(),
        }
    }

    /// Iterator over `&V` in insertion order.
    pub fn values(&self) -> FundedMapValues<'_, K, V> {
        FundedMapValues {
            inner: self.entries.iter(),
        }
    }

    /// Mutable iterator over `&mut V` in insertion order.
    pub fn values_mut(&mut self) -> FundedMapValuesMut<'_, K, V> {
        FundedMapValuesMut {
            inner: self.entries.iter_mut(),
        }
    }

    /// Retains only the elements specified by the predicate in insertion order, without
    /// reallocating. If `keep` panics after one or more elements have been removed, a drop
    /// guard compacts the remaining `hashes` tail to match `Vec::retain_mut` and rebuilds
    /// `slots` so the map's probe table remains valid after unwind.
    pub fn retain<F>(&mut self, mut keep: F)
    where
        F: FnMut(&K, &mut V) -> bool,
    {
        struct RetainGuard<'a, K, V, S> {
            map: &'a mut FundedMap<K, V, S>,
            orig_len: usize,
            read: usize,
            write: usize,
        }

        impl<K, V, S> Drop for RetainGuard<'_, K, V, S> {
            fn drop(&mut self) {
                if self.write < self.read {
                    self.map
                        .hashes
                        .copy_within(self.read..self.orig_len, self.write);
                    self.map.hashes.truncate(self.map.entries.len());
                    self.map.rebuild_slots();
                }
            }
        }

        let orig_len = self.entries.len();
        let mut guard = RetainGuard {
            map: self,
            orig_len,
            read: 0,
            write: 0,
        };
        {
            let RetainGuard {
                map, read, write, ..
            } = &mut guard;
            let entries = &mut map.entries;
            let hashes = &mut map.hashes;
            entries.retain_mut(|(k, v)| {
                let keep_item = keep(k, v);
                if keep_item {
                    hashes[*write] = hashes[*read];
                    *write += 1;
                }
                *read += 1;
                keep_item
            });
        }
    }
}

impl<K: Clone, V: Clone, S: Clone> FundedMap<K, V, S> {
    /// Clones this map into a new [`FundedMap`] charged to `funding`, pre-charging the exact
    /// capacity before allocating.
    pub fn clone_funded(&self, funding: &Funding) -> Result<Self> {
        let mut cloned = Self::with_hasher(self.hash_builder.clone(), funding);
        if !self.entries.is_empty() {
            cloned.reserve_exact(self.entries.len())?;
            cloned.entries.extend_from_slice(&self.entries);
            cloned.hashes.extend_from_slice(&self.hashes);
            cloned.rebuild_slots();
        }
        Ok(cloned)
    }
}

impl<K, V, S: BuildHasher> FundedMap<K, V, S> {
    /// Returns the insertion index of `key` if present.
    pub fn get_index_of<Q>(&self, key: &Q) -> Option<usize>
    where
        Q: ?Sized + Hash + Equivalent<K>,
    {
        if self.slots.is_empty() {
            return None;
        }
        let hash = self.hash_builder.hash_one(key);
        self.find_slot(hash, key).ok().map(|(_, idx)| idx)
    }

    /// Returns a reference to the value corresponding to `key`.
    pub fn get<Q>(&self, key: &Q) -> Option<&V>
    where
        Q: ?Sized + Hash + Equivalent<K>,
    {
        let idx = self.get_index_of(key)?;
        Some(&self.entries[idx].1)
    }

    /// Returns a mutable reference to the value corresponding to `key`.
    pub fn get_mut<Q>(&mut self, key: &Q) -> Option<&mut V>
    where
        Q: ?Sized + Hash + Equivalent<K>,
    {
        let idx = self.get_index_of(key)?;
        Some(&mut self.entries[idx].1)
    }

    /// Returns the key-value pair corresponding to `key`.
    pub fn get_key_value<Q>(&self, key: &Q) -> Option<(&K, &V)>
    where
        Q: ?Sized + Hash + Equivalent<K>,
    {
        let idx = self.get_index_of(key)?;
        let (k, v) = &self.entries[idx];
        Some((k, v))
    }

    /// Returns the index, key, and value corresponding to `key`.
    pub fn get_full<Q>(&self, key: &Q) -> Option<(usize, &K, &V)>
    where
        Q: ?Sized + Hash + Equivalent<K>,
    {
        let idx = self.get_index_of(key)?;
        let (k, v) = &self.entries[idx];
        Some((idx, k, v))
    }

    /// Returns the index, key, and mutable value corresponding to `key`.
    pub fn get_full_mut<Q>(&mut self, key: &Q) -> Option<(usize, &K, &mut V)>
    where
        Q: ?Sized + Hash + Equivalent<K>,
    {
        let idx = self.get_index_of(key)?;
        let (k, v) = &mut self.entries[idx];
        Some((idx, &*k, v))
    }

    /// Returns `true` if the map contains a value for `key`.
    pub fn contains_key<Q>(&self, key: &Q) -> bool
    where
        Q: ?Sized + Hash + Equivalent<K>,
    {
        self.get_index_of(key).is_some()
    }

    /// Removes `key` by swapping its entry with the last entry, returning the key-value pair.
    pub fn swap_remove_entry<Q>(&mut self, key: &Q) -> Option<(K, V)>
    where
        Q: ?Sized + Hash + Equivalent<K>,
    {
        let idx = self.get_index_of(key)?;
        self.swap_remove_index(idx)
    }

    /// Removes `key` and shifts subsequent entries down to preserve order, returning the
    /// key-value pair.
    pub fn shift_remove_entry<Q>(&mut self, key: &Q) -> Option<(K, V)>
    where
        Q: ?Sized + Hash + Equivalent<K>,
    {
        let idx = self.get_index_of(key)?;
        self.shift_remove_index(idx)
    }

    /// Removes `key` by swapping its entry with the last entry, returning its value.
    pub fn swap_remove<Q>(&mut self, key: &Q) -> Option<V>
    where
        Q: ?Sized + Hash + Equivalent<K>,
    {
        self.swap_remove_entry(key).map(|(_, v)| v)
    }

    /// Removes `key` and shifts subsequent entries down to preserve order, returning its
    /// value.
    pub fn shift_remove<Q>(&mut self, key: &Q) -> Option<V>
    where
        Q: ?Sized + Hash + Equivalent<K>,
    {
        self.shift_remove_entry(key).map(|(_, v)| v)
    }
}

impl<K: Hash + Eq, V, S: BuildHasher> FundedMap<K, V, S> {
    /// Inserts a key-value pair into the map, charging any required table growth before
    /// allocating. If the key already exists, updates the value in place without charging or
    /// reallocating.
    pub fn try_insert(&mut self, key: K, value: V) -> Result<Option<V>> {
        Ok(self.try_insert_full(key, value)?.1)
    }

    /// Inserts a key-value pair into the map and returns `(index, previous_value)`, charging
    /// any required table growth before allocating.
    pub fn try_insert_full(&mut self, key: K, value: V) -> Result<(usize, Option<V>)> {
        let hash = self.hash_builder.hash_one(&key);
        if !self.slots.is_empty()
            && let Ok((_, idx)) = self.find_slot(hash, &key)
        {
            let old = std::mem::replace(&mut self.entries[idx].1, value);
            return Ok((idx, Some(old)));
        }
        self.reserve(1)?;
        let Err(slot) = self.find_slot(hash, &key) else {
            unreachable!("key was absent before reserve");
        };
        let idx = self.entries.len();
        self.entries.push((key, value));
        self.hashes.push(hash);
        self.slots[slot] = (idx + 1) as u32;
        Ok((idx, None))
    }

    /// Returns a mutable reference to the value for `key`, inserting the result of `f()` if
    /// absent after charging any required table growth.
    pub fn get_or_try_insert_with<F>(&mut self, key: K, f: F) -> Result<&mut V>
    where
        F: FnOnce() -> Result<V>,
    {
        let hash = self.hash_builder.hash_one(&key);
        if !self.slots.is_empty()
            && let Ok((_, idx)) = self.find_slot(hash, &key)
        {
            return Ok(&mut self.entries[idx].1);
        }
        self.reserve(1)?;
        let value = f()?;
        let Err(slot) = self.find_slot(hash, &key) else {
            unreachable!("key was absent before reserve");
        };
        let idx = self.entries.len();
        self.entries.push((key, value));
        self.hashes.push(hash);
        self.slots[slot] = (idx + 1) as u32;
        Ok(&mut self.entries[idx].1)
    }

    /// Returns a mutable reference to the value for `key`, inserting `V::default()` if absent
    /// after charging any required table growth.
    pub fn get_or_try_insert_default(&mut self, key: K) -> Result<&mut V>
    where
        V: Default,
    {
        self.get_or_try_insert_with(key, || Ok(V::default()))
    }

    /// Fallibly extends the map with the elements of `iter`, charging capacity growth before
    /// allocating.
    pub fn try_extend<I>(&mut self, iter: I) -> Result<()>
    where
        I: IntoIterator<Item = (K, V)>,
    {
        let iter = iter.into_iter();
        let (lower, _) = iter.size_hint();
        if lower > 0 {
            self.reserve(lower)?;
        }
        for (k, v) in iter {
            self.try_insert(k, v)?;
        }
        Ok(())
    }
}

impl<K: fmt::Debug, V: fmt::Debug, S> fmt::Debug for FundedMap<K, V, S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_map().entries(self.iter()).finish()
    }
}

impl<K, V, S1, S2> PartialEq<FundedMap<K, V, S2>> for FundedMap<K, V, S1>
where
    K: Hash + Eq,
    V: PartialEq,
    S2: BuildHasher,
{
    fn eq(&self, other: &FundedMap<K, V, S2>) -> bool {
        self.len() == other.len() && self.iter().all(|(k, v)| other.get(k) == Some(v))
    }
}

impl<K: Hash + Eq, V: Eq, S: BuildHasher> Eq for FundedMap<K, V, S> {}

impl<K, V, Q: ?Sized, S> Index<&Q> for FundedMap<K, V, S>
where
    Q: Hash + Equivalent<K>,
    S: BuildHasher,
{
    type Output = V;

    fn index(&self, key: &Q) -> &V {
        self.get(key).expect("FundedMap: key not found")
    }
}

impl<K, V, Q: ?Sized, S> IndexMut<&Q> for FundedMap<K, V, S>
where
    Q: Hash + Equivalent<K>,
    S: BuildHasher,
{
    fn index_mut(&mut self, key: &Q) -> &mut V {
        self.get_mut(key).expect("FundedMap: key not found")
    }
}

impl<K, V, S> Index<usize> for FundedMap<K, V, S> {
    type Output = V;

    fn index(&self, index: usize) -> &V {
        &self.entries[index].1
    }
}

impl<K, V, S> IndexMut<usize> for FundedMap<K, V, S> {
    fn index_mut(&mut self, index: usize) -> &mut V {
        &mut self.entries[index].1
    }
}

impl<K, V, S> Drop for FundedMap<K, V, S> {
    fn drop(&mut self) {
        let entries = std::mem::take(&mut self.entries);
        let hashes = std::mem::take(&mut self.hashes);
        let slots = std::mem::take(&mut self.slots);
        drop(entries);
        drop(hashes);
        drop(slots);
        let charged = std::mem::replace(&mut self.charged, 0);
        self.funding.discharge(charged);
    }
}

/// Immutable iterator over `(&K, &V)` pairs of a [`FundedMap`].
pub struct FundedMapIter<'a, K, V> {
    inner: std::slice::Iter<'a, (K, V)>,
}

impl<K, V> Clone for FundedMapIter<'_, K, V> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

impl<'a, K, V> Iterator for FundedMapIter<'a, K, V> {
    type Item = (&'a K, &'a V);

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(|(k, v)| (k, v))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<K, V> DoubleEndedIterator for FundedMapIter<'_, K, V> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.inner.next_back().map(|(k, v)| (k, v))
    }
}

impl<K, V> ExactSizeIterator for FundedMapIter<'_, K, V> {
    fn len(&self) -> usize {
        self.inner.len()
    }
}

impl<K, V> FusedIterator for FundedMapIter<'_, K, V> {}

/// Mutable iterator over `(&K, &mut V)` pairs of a [`FundedMap`].
pub struct FundedMapIterMut<'a, K, V> {
    inner: std::slice::IterMut<'a, (K, V)>,
}

impl<'a, K, V> Iterator for FundedMapIterMut<'a, K, V> {
    type Item = (&'a K, &'a mut V);

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(|(k, v)| (&*k, v))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<K, V> DoubleEndedIterator for FundedMapIterMut<'_, K, V> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.inner.next_back().map(|(k, v)| (&*k, v))
    }
}

impl<K, V> ExactSizeIterator for FundedMapIterMut<'_, K, V> {
    fn len(&self) -> usize {
        self.inner.len()
    }
}

impl<K, V> FusedIterator for FundedMapIterMut<'_, K, V> {}

/// Iterator over `&K` keys of a [`FundedMap`].
pub struct FundedMapKeys<'a, K, V> {
    inner: std::slice::Iter<'a, (K, V)>,
}

impl<K, V> Clone for FundedMapKeys<'_, K, V> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

impl<'a, K, V> Iterator for FundedMapKeys<'a, K, V> {
    type Item = &'a K;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(|(k, _)| k)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<K, V> DoubleEndedIterator for FundedMapKeys<'_, K, V> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.inner.next_back().map(|(k, _)| k)
    }
}

impl<K, V> ExactSizeIterator for FundedMapKeys<'_, K, V> {
    fn len(&self) -> usize {
        self.inner.len()
    }
}

impl<K, V> FusedIterator for FundedMapKeys<'_, K, V> {}

/// Iterator over `&V` values of a [`FundedMap`].
pub struct FundedMapValues<'a, K, V> {
    inner: std::slice::Iter<'a, (K, V)>,
}

impl<K, V> Clone for FundedMapValues<'_, K, V> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

impl<'a, K, V> Iterator for FundedMapValues<'a, K, V> {
    type Item = &'a V;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(|(_, v)| v)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<K, V> DoubleEndedIterator for FundedMapValues<'_, K, V> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.inner.next_back().map(|(_, v)| v)
    }
}

impl<K, V> ExactSizeIterator for FundedMapValues<'_, K, V> {
    fn len(&self) -> usize {
        self.inner.len()
    }
}

impl<K, V> FusedIterator for FundedMapValues<'_, K, V> {}

/// Mutable iterator over `&mut V` values of a [`FundedMap`].
pub struct FundedMapValuesMut<'a, K, V> {
    inner: std::slice::IterMut<'a, (K, V)>,
}

impl<'a, K, V> Iterator for FundedMapValuesMut<'a, K, V> {
    type Item = &'a mut V;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(|(_, v)| v)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<K, V> DoubleEndedIterator for FundedMapValuesMut<'_, K, V> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.inner.next_back().map(|(_, v)| v)
    }
}

impl<K, V> ExactSizeIterator for FundedMapValuesMut<'_, K, V> {
    fn len(&self) -> usize {
        self.inner.len()
    }
}

impl<K, V> FusedIterator for FundedMapValuesMut<'_, K, V> {}

/// Owning iterator over `(K, V)` pairs of a [`FundedMap`] that discharges the entries buffer
/// when dropped.
pub struct FundedMapIntoIter<K, V> {
    inner: Option<std::vec::IntoIter<(K, V)>>,
    funding: Funding,
    charged: usize,
}

impl<K, V> Iterator for FundedMapIntoIter<K, V> {
    type Item = (K, V);

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.as_mut()?.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner
            .as_ref()
            .map_or((0, Some(0)), std::vec::IntoIter::size_hint)
    }
}

impl<K, V> DoubleEndedIterator for FundedMapIntoIter<K, V> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.inner.as_mut()?.next_back()
    }
}

impl<K, V> ExactSizeIterator for FundedMapIntoIter<K, V> {
    fn len(&self) -> usize {
        self.inner.as_ref().map_or(0, ExactSizeIterator::len)
    }
}

impl<K, V> FusedIterator for FundedMapIntoIter<K, V> {}

impl<K, V> Drop for FundedMapIntoIter<K, V> {
    fn drop(&mut self) {
        drop(self.inner.take());
        let charged = std::mem::replace(&mut self.charged, 0);
        self.funding.discharge(charged);
    }
}

impl<'a, K, V, S> IntoIterator for &'a FundedMap<K, V, S> {
    type Item = (&'a K, &'a V);
    type IntoIter = FundedMapIter<'a, K, V>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, K, V, S> IntoIterator for &'a mut FundedMap<K, V, S> {
    type Item = (&'a K, &'a mut V);
    type IntoIter = FundedMapIterMut<'a, K, V>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

impl<K, V, S> IntoIterator for FundedMap<K, V, S> {
    type Item = (K, V);
    type IntoIter = FundedMapIntoIter<K, V>;

    fn into_iter(mut self) -> Self::IntoIter {
        let entries = std::mem::take(&mut self.entries);
        let hashes = std::mem::take(&mut self.hashes);
        let slots = std::mem::take(&mut self.slots);
        let remaining_charge = if self.charged > 0 {
            self.funding.released(hashes);
            self.funding.released(slots);
            self.charged = 0;
            entries.block_bytes()
        } else {
            drop(hashes);
            drop(slots);
            0
        };
        let funding = self.funding.clone();
        FundedMapIntoIter {
            inner: Some(entries.into_iter()),
            funding,
            charged: remaining_charge,
        }
    }
}

/// An insertion-ordered hash set backed by [`FundedMap<T, (), S>`] whose exact heap
/// buffers are pre-charged to a [`Funding`] handle before every growth step and discharged
/// when the set is dropped.
pub struct FundedSet<T, S = DefaultHashBuilder> {
    map: FundedMap<T, (), S>,
}

impl<T> FundedSet<T, DefaultHashBuilder> {
    /// Unfunded empty [`FundedSet`] for default initialization prior to rebinding.
    pub fn new() -> Self {
        Self::with_funding(&Funding::default())
    }

    /// An empty [`FundedSet`] tied to `funding`. Allocates zero bytes until the first
    /// reservation or insertion.
    pub fn with_funding(funding: &Funding) -> Self {
        Self {
            map: FundedMap::with_funding(funding),
        }
    }
}

impl<T, S: Default> FundedSet<T, S> {
    /// An empty [`FundedSet`] with default hasher `S` tied to `funding`.
    pub fn with_hasher_and_funding(funding: &Funding) -> Self {
        Self {
            map: FundedMap::with_hasher_and_funding(funding),
        }
    }
}

impl<T, S: Default> Default for FundedSet<T, S> {
    fn default() -> Self {
        Self::with_hasher_and_funding(&Funding::default())
    }
}

impl<T, S> FundedSet<T, S> {
    /// Bytes per entry at the 50% probe-table load factor.
    pub const ENTRY_BYTES: usize = FundedMap::<T, (), S>::ENTRY_BYTES;

    /// An empty [`FundedSet`] with explicit `hash_builder` tied to `funding`.
    pub fn with_hasher(hash_builder: S, funding: &Funding) -> Self {
        Self {
            map: FundedMap::with_hasher(hash_builder, funding),
        }
    }

    /// Number of elements in the set.
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// Whether the set is empty.
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Number of elements the set can hold without reallocating.
    pub fn capacity(&self) -> usize {
        self.map.capacity()
    }

    /// Exact bytes currently charged to this set's [`Funding`] handle.
    pub fn charged_bytes(&self) -> usize {
        self.map.charged_bytes()
    }

    /// The [`Funding`] handle this set charges its capacity to.
    pub fn funding(&self) -> &Funding {
        self.map.funding()
    }

    /// Reference to the set's [`BuildHasher`].
    pub fn hasher(&self) -> &S {
        self.map.hasher()
    }

    /// Transfers this set's standing capacity charge to `next`, charging `next` before
    /// discharging the previous handle.
    pub fn rebind(&mut self, next: &Funding) -> Result<()> {
        self.map.rebind(next)
    }

    /// Detaches this set's standing charge from its [`Funding`] handle without discharging
    /// the handle, returning the detached byte count.
    pub fn detach_funding(&mut self) -> usize {
        self.map.detach_funding()
    }

    /// Clears all elements while retaining allocated capacity and its standing charge.
    pub fn clear(&mut self) {
        self.map.clear();
    }

    /// Shortens the set, keeping the first `len` elements and dropping the rest.
    pub fn truncate(&mut self, len: usize) {
        self.map.truncate(len);
    }

    /// Reserves capacity for at least `additional` more items, charging the capacity delta
    /// before allocating.
    pub fn reserve(&mut self, additional: usize) -> Result<()> {
        self.map.reserve(additional)
    }

    /// Reserves exact capacity for `additional` more items, charging the capacity delta
    /// before allocating.
    pub fn reserve_exact(&mut self, additional: usize) -> Result<()> {
        self.map.reserve_exact(additional)
    }

    /// Returns a reference to the element at `index`.
    pub fn get_index(&self, index: usize) -> Option<&T> {
        self.map.get_index(index).map(|(k, _)| k)
    }

    /// Returns the first element in insertion order.
    pub fn first(&self) -> Option<&T> {
        self.get_index(0)
    }

    /// Returns the last element in insertion order.
    pub fn last(&self) -> Option<&T> {
        self.map.last().map(|(k, _)| k)
    }

    /// Removes and returns the last element, retaining allocated capacity.
    pub fn pop(&mut self) -> Option<T> {
        self.map.pop().map(|(k, _)| k)
    }

    /// Removes and returns the element at `index` by swapping it with the last element.
    pub fn swap_remove_index(&mut self, index: usize) -> Option<T> {
        self.map.swap_remove_index(index).map(|(k, _)| k)
    }

    /// Removes and returns the element at `index`, shifting subsequent elements down.
    pub fn shift_remove_index(&mut self, index: usize) -> Option<T> {
        self.map.shift_remove_index(index).map(|(k, _)| k)
    }

    /// Iterator over `&T` in insertion order.
    pub fn iter(&self) -> FundedSetIter<'_, T> {
        FundedSetIter {
            inner: self.map.keys(),
        }
    }

    /// Retains only the elements specified by the predicate in insertion order.
    pub fn retain<F>(&mut self, mut keep: F)
    where
        F: FnMut(&T) -> bool,
    {
        self.map.retain(|k, ()| keep(k));
    }
}

impl<T: Clone, S: Clone> FundedSet<T, S> {
    /// Clones this set into a new [`FundedSet`] charged to `funding`, pre-charging the exact
    /// capacity before allocating.
    pub fn clone_funded(&self, funding: &Funding) -> Result<Self> {
        Ok(Self {
            map: self.map.clone_funded(funding)?,
        })
    }
}

impl<T, S: BuildHasher> FundedSet<T, S> {
    /// Returns the insertion index of `value` if present.
    pub fn get_index_of<Q>(&self, value: &Q) -> Option<usize>
    where
        Q: ?Sized + Hash + Equivalent<T>,
    {
        self.map.get_index_of(value)
    }

    /// Returns a reference to the value in the set matching `value`.
    pub fn get<Q>(&self, value: &Q) -> Option<&T>
    where
        Q: ?Sized + Hash + Equivalent<T>,
    {
        self.map.get_key_value(value).map(|(k, _)| k)
    }

    /// Returns `(index, &T)` for `value` if present.
    pub fn get_full<Q>(&self, value: &Q) -> Option<(usize, &T)>
    where
        Q: ?Sized + Hash + Equivalent<T>,
    {
        self.map.get_full(value).map(|(idx, k, _)| (idx, k))
    }

    /// Returns `true` if the set contains `value`.
    pub fn contains<Q>(&self, value: &Q) -> bool
    where
        Q: ?Sized + Hash + Equivalent<T>,
    {
        self.map.contains_key(value)
    }

    /// Removes `value` by swapping with the last element, returning `true` if it was present.
    pub fn swap_remove<Q>(&mut self, value: &Q) -> bool
    where
        Q: ?Sized + Hash + Equivalent<T>,
    {
        self.map.swap_remove_entry(value).is_some()
    }

    /// Removes `value` and shifts subsequent elements down, returning `true` if it was
    /// present.
    pub fn shift_remove<Q>(&mut self, value: &Q) -> bool
    where
        Q: ?Sized + Hash + Equivalent<T>,
    {
        self.map.shift_remove_entry(value).is_some()
    }

    /// Removes and returns `value` by swapping with the last element.
    pub fn swap_take<Q>(&mut self, value: &Q) -> Option<T>
    where
        Q: ?Sized + Hash + Equivalent<T>,
    {
        self.map.swap_remove_entry(value).map(|(k, _)| k)
    }

    /// Removes and returns `value`, shifting subsequent elements down.
    pub fn shift_take<Q>(&mut self, value: &Q) -> Option<T>
    where
        Q: ?Sized + Hash + Equivalent<T>,
    {
        self.map.shift_remove_entry(value).map(|(k, _)| k)
    }
}

impl<T: Hash + Eq, S: BuildHasher> FundedSet<T, S> {
    /// Inserts `value` into the set, charging any required table growth before allocating.
    /// Returns `Ok(true)` if the value was newly inserted, or `Ok(false)` if already present.
    pub fn try_insert(&mut self, value: T) -> Result<bool> {
        Ok(self.map.try_insert(value, ())?.is_none())
    }

    /// Inserts `value` into the set and returns `(index, newly_inserted)`, charging any
    /// required table growth before allocating.
    pub fn try_insert_full(&mut self, value: T) -> Result<(usize, bool)> {
        let (idx, prev) = self.map.try_insert_full(value, ())?;
        Ok((idx, prev.is_none()))
    }

    /// Fallibly extends the set with the elements of `iter`, charging capacity growth before
    /// allocating.
    pub fn try_extend<I>(&mut self, iter: I) -> Result<()>
    where
        I: IntoIterator<Item = T>,
    {
        let iter = iter.into_iter();
        let (lower, _) = iter.size_hint();
        if lower > 0 {
            self.reserve(lower)?;
        }
        for item in iter {
            self.try_insert(item)?;
        }
        Ok(())
    }

    /// Returns `true` if `self` has no elements in common with `other`.
    pub fn is_disjoint<S2>(&self, other: &FundedSet<T, S2>) -> bool
    where
        S2: BuildHasher,
    {
        self.iter().all(|item| !other.contains(item))
    }

    /// Returns `true` if all elements of `self` are contained in `other`.
    pub fn is_subset<S2>(&self, other: &FundedSet<T, S2>) -> bool
    where
        S2: BuildHasher,
    {
        self.len() <= other.len() && self.iter().all(|item| other.contains(item))
    }

    /// Returns `true` if all elements of `other` are contained in `self`.
    pub fn is_superset<S2>(&self, other: &FundedSet<T, S2>) -> bool {
        other.len() <= self.len() && other.iter().all(|item| self.contains(item))
    }
}

impl<T, S> Index<usize> for FundedSet<T, S> {
    type Output = T;

    fn index(&self, index: usize) -> &T {
        &self.map.entries[index].0
    }
}

impl<T: fmt::Debug, S> fmt::Debug for FundedSet<T, S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_set().entries(self.iter()).finish()
    }
}

impl<T, S1, S2> PartialEq<FundedSet<T, S2>> for FundedSet<T, S1>
where
    T: Hash + Eq,
    S2: BuildHasher,
{
    fn eq(&self, other: &FundedSet<T, S2>) -> bool {
        self.len() == other.len() && self.is_subset(other)
    }
}

impl<T: Hash + Eq, S: BuildHasher> Eq for FundedSet<T, S> {}

/// Immutable iterator over `&T` elements of a [`FundedSet`].
pub struct FundedSetIter<'a, T> {
    inner: FundedMapKeys<'a, T, ()>,
}

impl<T> Clone for FundedSetIter<'_, T> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

impl<'a, T> Iterator for FundedSetIter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<T> DoubleEndedIterator for FundedSetIter<'_, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.inner.next_back()
    }
}

impl<T> ExactSizeIterator for FundedSetIter<'_, T> {
    fn len(&self) -> usize {
        self.inner.len()
    }
}

impl<T> FusedIterator for FundedSetIter<'_, T> {}

/// Owning iterator over `T` elements of a [`FundedSet`] that discharges the entries buffer
/// when dropped.
pub struct FundedSetIntoIter<T> {
    inner: FundedMapIntoIter<T, ()>,
}

impl<T> Iterator for FundedSetIntoIter<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(|(k, ())| k)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<T> DoubleEndedIterator for FundedSetIntoIter<T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.inner.next_back().map(|(k, ())| k)
    }
}

impl<T> ExactSizeIterator for FundedSetIntoIter<T> {
    fn len(&self) -> usize {
        self.inner.len()
    }
}

impl<T> FusedIterator for FundedSetIntoIter<T> {}

impl<'a, T, S> IntoIterator for &'a FundedSet<T, S> {
    type Item = &'a T;
    type IntoIter = FundedSetIter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<T, S> IntoIterator for FundedSet<T, S> {
    type Item = T;
    type IntoIter = FundedSetIntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        FundedSetIntoIter {
            inner: self.map.into_iter(),
        }
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

    #[test]
    fn funded_map_retains_valid_slots_when_predicate_panics() {
        #[derive(Default)]
        struct SameHash;

        impl std::hash::Hasher for SameHash {
            fn finish(&self) -> u64 {
                0
            }

            fn write(&mut self, _bytes: &[u8]) {}
        }

        let (_, funding) = limited(4096);
        let mut map = FundedMap::with_hasher(
            std::hash::BuildHasherDefault::<SameHash>::default(),
            &funding,
        );
        for key in 0..3 {
            map.try_insert(key, key).unwrap();
        }
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            map.retain(|key, _| match *key {
                0 => false,
                1 => panic!("predicate failed"),
                _ => true,
            });
        }));
        assert!(result.is_err());
        assert_eq!(map.len(), 2);
        assert_eq!(map.get(&1), Some(&1));
        assert_eq!(map.get(&2), Some(&2));
        assert_eq!(map.get(&3), None);
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

    /// A boxed future is charged its block before it is boxed, refused without boxing it,
    /// and gives its block back as soon as it finishes, or as it is dropped unfinished.
    #[test]
    fn a_held_future_gives_back_its_block_once_finished_or_dropped() {
        let payload = [7u8; 48];
        let future = async move { payload.iter().map(|b| *b as usize).sum::<usize>() };
        let block = size_of_val(&future);
        assert!(block >= 48);

        let (source, funding) = limited(block - 1);
        match funding.held_future(future) {
            Err(DataFusionError::Refused { needed }) => assert_eq!(needed, block),
            Err(other) => panic!("refused, not {other:?}"),
            Ok(_) => panic!("refused, not boxed"),
        }
        assert_eq!(held(&source), 0);

        let (source, funding) = limited(block);
        let future = async move { payload.iter().map(|b| *b as usize).sum::<usize>() };
        let mut held_future = funding.held_future(future).unwrap();
        assert_eq!(held(&source), block);
        let mut cx = Context::from_waker(std::task::Waker::noop());
        assert_eq!(
            Pin::new(&mut held_future).poll(&mut cx),
            Poll::Ready(7 * 48)
        );
        assert_eq!(held(&source), 0);
        drop(held_future);
        assert_eq!(held(&source), 0);
        assert_eq!(funding.outstanding(), 0);

        let future = async move { payload.iter().map(|b| *b as usize).sum::<usize>() };
        let unfinished = funding.held_future(future).unwrap();
        assert_eq!(held(&source), block);
        drop(unfinished);
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

    /// A [`FundedMap`] charges its exact `Vec` buffers (`entries`, `hashes`, `slots`) before
    /// every growth step, refuses growth without mutating or leaking when underfunded,
    /// supports `rebind` and `clone_funded`, and discharges everything on drop.
    #[test]
    fn a_funded_map_charges_each_growth_and_releases_on_drop() {
        let entry_bytes = FundedMap::<u64, u64>::ENTRY_BYTES;
        assert_eq!(entry_bytes, 32);

        let (source, funding) = limited(3 * entry_bytes);
        let mut map: FundedMap<u64, u64> = FundedMap::with_funding(&funding);
        assert_eq!(held(&source), 0);
        assert_eq!(map.charged_bytes(), 0);
        assert!(map.is_empty());

        match map.try_insert(1, 10) {
            Err(DataFusionError::Refused { needed }) => {
                assert_eq!(needed, 8 * size_of::<u32>());
            }
            other => panic!("expected refusal, got {other:?}"),
        }
        assert!(map.is_empty());
        assert_eq!(held(&source), 0);
        assert_eq!(funding.outstanding(), 0);

        let (source, funding) = limited(16 * entry_bytes);
        let mut map: FundedMap<u64, u64> = FundedMap::with_funding(&funding);
        for i in 0..8u64 {
            assert_eq!(map.try_insert(i, i * 10).unwrap(), None);
        }
        assert_eq!(map.len(), 8);
        assert_eq!(map.charged_bytes(), 8 * entry_bytes);
        assert_eq!(held(&source), map.charged_bytes());
        assert_eq!(funding.outstanding(), map.charged_bytes());

        // Overwriting an existing key at full capacity neither allocates nor charges.
        assert_eq!(map.try_insert(3, 300).unwrap(), Some(30));
        assert_eq!(map.get(&3), Some(&300));
        assert_eq!(held(&source), 8 * entry_bytes);

        // Removing and re-inserting within reserved capacity does not charge extra bytes.
        assert_eq!(map.swap_remove(&7), Some(70));
        assert_eq!(map.shift_remove(&0), Some(0));
        assert_eq!(map.len(), 6);
        assert_eq!(map.try_insert(99, 990).unwrap(), None);
        assert_eq!(held(&source), 8 * entry_bytes);

        // Rebinding transfers the standing charge to the destination funding.
        let (other_source, other_funding) = limited(16 * entry_bytes);
        map.rebind(&other_funding).unwrap();
        assert_eq!(held(&source), 0);
        assert_eq!(funding.outstanding(), 0);
        assert_eq!(held(&other_source), map.charged_bytes());
        assert_eq!(other_funding.outstanding(), map.charged_bytes());

        // Cloning into a funded handle charges the clone's exact capacity before allocating.
        let cloned = map.clone_funded(&funding).unwrap();
        assert_eq!(cloned, map);
        assert_eq!(held(&source), cloned.charged_bytes());
        drop(cloned);
        assert_eq!(held(&source), 0);

        drop(map);
        assert_eq!(held(&other_source), 0);
        assert_eq!(other_funding.outstanding(), 0);
    }

    /// A [`FundedSet`] charges its exact `Vec` buffers before growth (including the
    /// transient `old + new` peak during `realloc`), refuses unfunded growth, and
    /// discharges on drop.
    #[test]
    fn a_funded_set_charges_each_growth_and_releases_on_drop() {
        let entry_bytes = FundedSet::<u64>::ENTRY_BYTES;
        assert_eq!(entry_bytes, 24);

        // Growing from 4 to 8 entries via `realloc` holds `entries` (64B) + `hashes` (64B)
        // + old `slots` (32B) + new `slots` (64B) = 224B at the `slots` `realloc` peak,
        // then discharges the old `slots` (32B) to settle at `8 * entry_bytes` (192B).
        let (tight_source, tight_funding) = limited(8 * entry_bytes);
        let mut tight_set: FundedSet<u64> = FundedSet::with_funding(&tight_funding);
        for i in 0..4u64 {
            assert!(tight_set.try_insert(i).unwrap());
        }
        assert!(matches!(
            tight_set.try_insert(4),
            Err(DataFusionError::Refused { .. })
        ));
        assert_eq!(tight_set.len(), 4);
        assert_eq!(held(&tight_source), tight_set.charged_bytes());
        drop(tight_set);
        assert_eq!(held(&tight_source), 0);

        let (source, funding) = limited(8 * entry_bytes + 8 * size_of::<u32>());
        let mut set: FundedSet<u64> = FundedSet::with_funding(&funding);
        for i in 0..8u64 {
            assert!(set.try_insert(i).unwrap());
        }
        assert!(!set.try_insert(3).unwrap());
        assert_eq!(set.len(), 8);
        assert_eq!(set.charged_bytes(), 8 * entry_bytes);
        assert_eq!(held(&source), set.charged_bytes());

        // Next insert triggers growth from 8 to 16 entries, which is refused on the
        // first `Vec` (`entries` needs 16 * 8 = 128B on top of 192B held > 224B).
        assert!(matches!(
            set.try_insert(8),
            Err(DataFusionError::Refused { .. })
        ));
        assert_eq!(set.len(), 8);
        assert_eq!(held(&source), set.charged_bytes());

        drop(set);
        assert_eq!(held(&source), 0);
        assert_eq!(funding.outstanding(), 0);
    }
}
