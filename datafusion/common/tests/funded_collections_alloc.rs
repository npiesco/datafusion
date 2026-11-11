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

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use datafusion_common::DataFusionError;
use datafusion_common::funding::{FundedMap, FundedSet, Funding, PlanningFunding};

thread_local! {
    static TRACK_ALLOC: Cell<bool> = const { Cell::new(false) };
    static LIVE_BYTES: Cell<usize> = const { Cell::new(0) };
    static FUNDED_BYTES: Cell<usize> = const { Cell::new(0) };
    static UNDERCHARGE_EVENTS: Cell<usize> = const { Cell::new(0) };
}

struct CountingAlloc;

unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let tracked = TRACK_ALLOC.with(Cell::get);
        if tracked {
            let peak_live = LIVE_BYTES.with(|c| c.get().saturating_add(layout.size()));
            let funded = FUNDED_BYTES.with(Cell::get);
            if peak_live > funded {
                UNDERCHARGE_EVENTS.with(|u| u.set(u.get() + 1));
            }
        }
        // SAFETY: forwarded directly to the system allocator with the same layout.
        let ptr = unsafe { System.alloc(layout) };
        if tracked && !ptr.is_null() {
            let next_live = LIVE_BYTES.with(|c| {
                let next = c.get().saturating_add(layout.size());
                c.set(next);
                next
            });
            let funded = FUNDED_BYTES.with(Cell::get);
            if next_live > funded {
                UNDERCHARGE_EVENTS.with(|u| u.set(u.get() + 1));
            }
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        if TRACK_ALLOC.with(Cell::get) {
            let next_live = LIVE_BYTES.with(|c| {
                let next = c.get().saturating_sub(layout.size());
                c.set(next);
                next
            });
            let funded = FUNDED_BYTES.with(Cell::get);
            if next_live > funded {
                UNDERCHARGE_EVENTS.with(|u| u.set(u.get() + 1));
            }
        }
        // SAFETY: forwarded directly to the system allocator with matching ptr and layout.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let tracked = TRACK_ALLOC.with(Cell::get);
        if tracked {
            // A moving realloc holds both the old block (`layout.size()`) and the new
            // block (`new_size`) simultaneously until the old block is freed; funded
            // bytes must cover that transient `old + new` peak before `realloc` runs.
            let peak_live = LIVE_BYTES.with(|c| c.get().saturating_add(new_size));
            let funded = FUNDED_BYTES.with(Cell::get);
            if peak_live > funded {
                UNDERCHARGE_EVENTS.with(|u| u.set(u.get() + 1));
            }
        }
        // SAFETY: forwarded directly to the system allocator with matching ptr, layout, and new_size.
        let new_ptr = unsafe { System.realloc(ptr, layout, new_size) };
        if tracked && !new_ptr.is_null() {
            let next_live = LIVE_BYTES.with(|c| {
                let next = c.get().saturating_add(new_size).saturating_sub(layout.size());
                c.set(next);
                next
            });
            let funded = FUNDED_BYTES.with(Cell::get);
            if next_live > funded {
                UNDERCHARGE_EVENTS.with(|u| u.set(u.get() + 1));
            }
        }
        new_ptr
    }
}

#[global_allocator]
static GLOBAL: CountingAlloc = CountingAlloc;

struct TrackedBudget {
    limit: usize,
    used: AtomicUsize,
}

impl TrackedBudget {
    fn new(limit: usize) -> Arc<Self> {
        Arc::new(Self {
            limit,
            used: AtomicUsize::new(0),
        })
    }

    fn used(&self) -> usize {
        self.used.load(Ordering::Acquire)
    }
}

impl PlanningFunding for TrackedBudget {
    fn charge(&self, bytes: usize) -> bool {
        let ok = self
            .used
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |cur| {
                let next = cur.checked_add(bytes)?;
                (next <= self.limit).then_some(next)
            })
            .is_ok();
        if ok && TRACK_ALLOC.with(Cell::get) {
            FUNDED_BYTES.with(|c| c.set(c.get().saturating_add(bytes)));
        }
        ok
    }

    fn discharge(&self, bytes: usize) {
        self.used.fetch_sub(bytes, Ordering::AcqRel);
        if TRACK_ALLOC.with(Cell::get) {
            FUNDED_BYTES.with(|c| c.set(c.get().saturating_sub(bytes)));
        }
    }
}

struct AllocScope;

impl AllocScope {
    fn enter() -> Self {
        LIVE_BYTES.with(|c| c.set(0));
        FUNDED_BYTES.with(|c| c.set(0));
        UNDERCHARGE_EVENTS.with(|c| c.set(0));
        TRACK_ALLOC.with(|c| c.set(true));
        Self
    }

    fn live_bytes(&self) -> usize {
        LIVE_BYTES.with(Cell::get)
    }

    fn funded_bytes(&self) -> usize {
        FUNDED_BYTES.with(Cell::get)
    }

    fn undercharge_events(&self) -> usize {
        UNDERCHARGE_EVENTS.with(Cell::get)
    }

    fn assert_reconciled(&self, expected: usize) {
        let undercharges = self.undercharge_events();
        let live = self.live_bytes();
        let funded = self.funded_bytes();
        assert_eq!(
            undercharges, 0,
            "allocator observed live_bytes > funded_bytes during growth/dealloc"
        );
        assert_eq!(
            funded, live,
            "reconciled funded_bytes ({funded}) must equal allocator live_bytes ({live})"
        );
        assert_eq!(
            funded, expected,
            "reconciled funded_bytes ({funded}) must equal map/set charged_bytes ({expected})"
        );
    }
}

impl Drop for AllocScope {
    fn drop(&mut self) {
        TRACK_ALLOC.with(|c| c.set(false));
    }
}

#[test]
fn funded_map_and_set_exact_global_allocator_accounting() {
    let budget_a = TrackedBudget::new(usize::MAX);
    let funding_a = Funding::new(Arc::clone(&budget_a) as Arc<dyn PlanningFunding>);
    let budget_b = TrackedBudget::new(usize::MAX);
    let funding_b = Funding::new(Arc::clone(&budget_b) as Arc<dyn PlanningFunding>);

    let scope = AllocScope::enter();
    scope.assert_reconciled(0);

    let mut map: FundedMap<u64, u64> = FundedMap::with_funding(&funding_a);
    scope.assert_reconciled(0);

    for i in 0..128u64 {
        let prev = map.try_insert(i, i * 10).unwrap();
        assert_eq!(prev, None);
        assert!(map.charged_bytes() > 0);
        scope.assert_reconciled(map.charged_bytes());
        assert_eq!(funding_a.outstanding(), scope.live_bytes());
        assert_eq!(budget_a.used(), scope.live_bytes());
    }

    // Overwriting existing keys must not allocate or change charges.
    let steady_bytes = map.charged_bytes();
    for i in 0..128u64 {
        let prev = map.try_insert(i, i * 20).unwrap();
        assert_eq!(prev, Some(i * 10));
        assert_eq!(map.charged_bytes(), steady_bytes);
        scope.assert_reconciled(steady_bytes);
    }

    // Removing entries preserves reserved capacity and exact accounting.
    assert_eq!(map.swap_remove(&0), Some(0));
    assert_eq!(map.shift_remove(&1), Some(20));
    assert_eq!(map.len(), 126);
    scope.assert_reconciled(steady_bytes);

    // Funded clone charges the destination funding for its exact allocation.
    let cloned = map.clone_funded(&funding_b).unwrap();
    assert_eq!(cloned.len(), map.len());
    scope.assert_reconciled(map.charged_bytes() + cloned.charged_bytes());
    assert_eq!(
        funding_a.outstanding() + funding_b.outstanding(),
        scope.live_bytes()
    );

    // Rebind transfers the charge between funders without allocating.
    let mut rebound = cloned;
    rebound.rebind(&funding_a).unwrap();
    assert_eq!(funding_b.outstanding(), 0);
    assert_eq!(budget_b.used(), 0);
    scope.assert_reconciled(map.charged_bytes() + rebound.charged_bytes());
    assert_eq!(funding_a.outstanding(), scope.live_bytes());

    drop(rebound);
    scope.assert_reconciled(map.charged_bytes());
    drop(map);
    scope.assert_reconciled(0);
    assert_eq!(funding_a.outstanding(), 0);
    assert_eq!(budget_a.used(), 0);

    // FundedSet exercises the same exact accounting through every growth step.
    let mut set: FundedSet<u64> = FundedSet::with_funding(&funding_a);
    for i in 0..128u64 {
        assert!(set.try_insert(i).unwrap());
        scope.assert_reconciled(set.charged_bytes());
    }
    assert!(!set.try_insert(64).unwrap());
    scope.assert_reconciled(set.charged_bytes());

    // Consuming a FundedSet via IntoIterator immediately frees and discharges
    // the auxiliary `hashes` and `slots` buffers while retaining `entries` until
    // the iterator itself drops.
    let entries_only_bytes = 128 * std::mem::size_of::<(u64, ())>();
    let mut iter = set.into_iter();
    scope.assert_reconciled(entries_only_bytes);
    assert_eq!(iter.next(), Some(0));
    scope.assert_reconciled(entries_only_bytes);
    drop(iter);
    scope.assert_reconciled(0);
    assert_eq!(funding_a.outstanding(), 0);
    assert_eq!(budget_a.used(), 0);
}

#[test]
fn funded_map_and_set_refusal_never_allocates_unfunded_bytes() {
    let tight_budget = TrackedBudget::new(96);
    let tight_funding =
        Funding::new(Arc::clone(&tight_budget) as Arc<dyn PlanningFunding>);

    let scope = AllocScope::enter();
    let mut map: FundedMap<u64, u64> = FundedMap::with_funding(&tight_funding);

    // First growth (4 entries: 4*16 + 4*8 + 8*4 = 128 bytes > 96) must be refused
    // before any heap allocation takes place.
    let err = map.try_insert(1, 10).unwrap_err();
    assert!(matches!(err, DataFusionError::Refused { .. }));
    assert!(map.is_empty());
    scope.assert_reconciled(0);
    assert_eq!(tight_funding.outstanding(), 0);
    assert_eq!(tight_budget.used(), 0);

    // Exact reservation of 2 entries (2*16 + 2*8 + 4*4 = 64 bytes <= 96) succeeds,
    // and the subsequent growth beyond 2 entries is refused with zero leak.
    map.reserve_exact(2).unwrap();
    assert_eq!(map.charged_bytes(), 64);
    scope.assert_reconciled(64);
    map.try_insert(1, 10).unwrap();
    map.try_insert(2, 20).unwrap();
    scope.assert_reconciled(64);

    let err = map.try_insert(3, 30).unwrap_err();
    assert!(matches!(err, DataFusionError::Refused { .. }));
    assert_eq!(map.len(), 2);
    scope.assert_reconciled(64);

    drop(map);
    scope.assert_reconciled(0);
    assert_eq!(tight_funding.outstanding(), 0);
    assert_eq!(tight_budget.used(), 0);
}
