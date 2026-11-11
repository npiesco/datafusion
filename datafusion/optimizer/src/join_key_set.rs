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

//! [JoinKeySet] for tracking the set of join keys in a plan.

use datafusion_common::Result;
use datafusion_common::funding::{Funding, Held};
use datafusion_expr::Expr;

use crate::funded::{Freed, freed, hash_of};

/// Tracks a set of equality Join keys
///
/// A join key is an expression that is used to join two tables via an equality
/// predicate such as `a.x = b.y`
///
/// This struct models `a.x + 5 = b.y AND a.z = b.z` as two join keys
/// 1. `(a.x + 5,  b.y)`
/// 2. `(a.z,      b.z)`
///
/// # Important properties:
///
/// 1. Retains insert order
/// 2. Can quickly look up if a pair of expressions are in the set.
///
/// The keys are held in a list in insert order and found through their hashes, sorted
/// with where each is in a list of their own: both lists are charged as they grow, and
/// given back with the keys the set holds when it is dropped.
pub struct JoinKeySet {
    keys: Held<Vec<(Expr, Expr)>>,
    hashes: Held<Vec<(u64, usize)>>,
    funding: Funding,
}

impl JoinKeySet {
    /// Create a new empty set, charged to `funding` as it grows
    pub fn new(funding: &Funding) -> Self {
        Self {
            keys: funding.held(),
            hashes: funding.held(),
            funding: funding.clone(),
        }
    }

    /// Where the join pair where left = right or right = left is
    pub fn position(&self, left: &Expr, right: &Expr) -> Option<usize> {
        self.find(left, right).or_else(|| self.find(right, left))
    }

    fn find(&self, left: &Expr, right: &Expr) -> Option<usize> {
        let hash = hash_of(&(left, right));
        let start = self.hashes.partition_point(|(other, _)| *other < hash);
        self.hashes[start..]
            .iter()
            .take_while(|(other, _)| *other == hash)
            .map(|(_, at)| *at)
            .find(|at| {
                let (l, r) = &self.keys[*at];
                l == left && r == right
            })
    }

    /// Return true if the set contains a join pair
    /// where left = right or right = left
    pub fn contains(&self, left: &Expr, right: &Expr) -> bool {
        self.position(left, right).is_some()
    }

    /// Insert the join key `(left = right)` into the set  if join pair `(right =
    /// left)` is not already in the set, its copy charged before it is made
    ///
    /// returns true if the pair was inserted
    pub fn insert(&mut self, left: &Expr, right: &Expr) -> Result<bool> {
        if self.contains(left, right) {
            return Ok(false);
        }
        let left = self.funding.cloned(left)?;
        let right = self.funding.cloned(right)?;
        self.inserted(left, right)?;
        Ok(true)
    }

    /// Same as [`Self::insert`] but avoids cloning expression if they
    /// are owned: a pair already in the set is freed and given back
    pub fn insert_owned(&mut self, left: Expr, right: Expr) -> Result<bool> {
        if self.contains(&left, &right) {
            freed(&self.funding, (left, right));
            return Ok(false);
        }
        self.inserted(left, right)?;
        Ok(true)
    }

    fn inserted(&mut self, left: Expr, right: Expr) -> Result<()> {
        let hash = hash_of(&(&left, &right));
        self.keys.reserve(1)?;
        self.hashes.reserve(1)?;
        let at = self.hashes.partition_point(|(other, _)| *other <= hash);
        // Room for each is reserved: neither allocates.
        self.hashes.insert(at, (hash, self.keys.len()));
        self.keys.push((left, right))
    }

    /// Same as [`Self::insert_owned`] for each pair
    ///
    /// returns true if any of the pairs were inserted
    pub fn insert_all_owned(
        &mut self,
        iter: impl IntoIterator<Item = (Expr, Expr)>,
    ) -> Result<bool> {
        let mut inserted = false;
        for (left, right) in iter.into_iter() {
            inserted |= self.insert_owned(left, right)?;
        }
        Ok(inserted)
    }

    /// Inserts any join keys that are common to both `s1` and `s2` into self
    pub fn insert_intersection(
        &mut self,
        s1: &JoinKeySet,
        s2: &JoinKeySet,
    ) -> Result<()> {
        // note can't use inner.intersection as we need to consider both (l, r)
        // and (r, l) in equality
        for (left, right) in s1.keys.iter() {
            if s2.contains(left, right) {
                self.insert(left, right)?;
            }
        }
        Ok(())
    }

    /// returns true if this set is empty
    #[cfg(test)]
    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }

    /// Return the length of this set
    pub fn len(&self) -> usize {
        self.keys.len()
    }

    /// The funding the set is charged to
    pub fn funding(&self) -> &Funding {
        &self.funding
    }

    /// Return an iterator over the join keys in this set
    pub fn iter(&self) -> impl Iterator<Item = (&Expr, &Expr)> {
        self.keys.iter().map(|(l, r)| (l, r))
    }
}

impl Drop for JoinKeySet {
    fn drop(&mut self) {
        let bytes = self.keys.as_slice().freed_bytes();
        self.keys.clear();
        self.funding.discharge(bytes);
    }
}

#[cfg(test)]
mod test {
    use crate::join_key_set::JoinKeySet;
    use datafusion_common::funding::Funding;
    use datafusion_expr::{Expr, col};

    #[test]
    fn a_qualifier_shared_by_every_key_is_freed_with_the_set() {
        use crate::funded::Freed;
        use datafusion_common::funding::PlanningFunding;
        use datafusion_common::{Column, TableReference};
        use std::sync::Arc;

        struct Unlimited;
        impl PlanningFunding for Unlimited {
            fn charge(&self, _bytes: usize) -> bool {
                true
            }

            fn discharge(&self, _bytes: usize) {}
        }

        let funding = Funding::new(Arc::new(Unlimited));
        let t = TableReference::bare("t");
        let column = |name: &str| Expr::Column(Column::new(Some(t.clone()), name));
        let pairs = vec![(column("a"), column("b")), (column("c"), column("d"))];
        drop(t);
        let each = pairs.iter().map(Freed::freed_bytes).sum::<usize>();
        assert!(
            pairs.as_slice().freed_bytes() > each,
            "the qualifier is shared"
        );
        funding.charge(pairs.as_slice().freed_bytes()).unwrap();
        let mut set = JoinKeySet::new(&funding);
        set.insert_all_owned(pairs).unwrap();
        drop(set);
        assert_eq!(funding.outstanding(), 0);
    }

    #[test]
    fn test_insert() {
        let mut set = JoinKeySet::new(&Funding::default());
        // new sets should be empty
        assert!(set.is_empty());

        // insert (a = b)
        assert!(set.insert(&col("a"), &col("b")).unwrap());
        assert!(!set.is_empty());

        // insert (a=b) again returns false
        assert!(!set.insert(&col("a"), &col("b")).unwrap());
        assert_eq!(set.len(), 1);

        // insert (b = a) , should be considered equivalent
        assert!(!set.insert(&col("b"), &col("a")).unwrap());
        assert_eq!(set.len(), 1);

        // insert (a = c) should be considered different
        assert!(set.insert(&col("a"), &col("c")).unwrap());
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn test_insert_owned() {
        let mut set = JoinKeySet::new(&Funding::default());
        assert!(set.insert_owned(col("a"), col("b")).unwrap());
        assert!(set.contains(&col("a"), &col("b")));
        assert!(set.contains(&col("b"), &col("a")));
        assert!(!set.contains(&col("a"), &col("c")));
    }

    #[test]
    fn test_contains() {
        let mut set = JoinKeySet::new(&Funding::default());
        assert!(set.insert(&col("a"), &col("b")).unwrap());
        assert!(set.contains(&col("a"), &col("b")));
        assert!(set.contains(&col("b"), &col("a")));
        assert!(!set.contains(&col("a"), &col("c")));

        assert!(set.insert(&col("a"), &col("c")).unwrap());
        assert!(set.contains(&col("a"), &col("c")));
        assert!(set.contains(&col("c"), &col("a")));
    }

    #[test]
    fn test_iterator() {
        // put in c = a and
        let mut set = JoinKeySet::new(&Funding::default());
        // put in c = a , b = c, and a = c and expect to get only the first 2
        set.insert(&col("c"), &col("a")).unwrap();
        set.insert(&col("b"), &col("c")).unwrap();
        set.insert(&col("a"), &col("c")).unwrap();
        assert_contents(&set, vec![(&col("c"), &col("a")), (&col("b"), &col("c"))]);
    }

    #[test]
    fn test_insert_intersection() {
        // a = b, b = c, c = d
        let mut set1 = JoinKeySet::new(&Funding::default());
        set1.insert(&col("a"), &col("b")).unwrap();
        set1.insert(&col("b"), &col("c")).unwrap();
        set1.insert(&col("c"), &col("d")).unwrap();

        // a = a, b = b, b = c, d = c
        // should only intersect on b = c and c = d
        let mut set2 = JoinKeySet::new(&Funding::default());
        set2.insert(&col("a"), &col("a")).unwrap();
        set2.insert(&col("b"), &col("b")).unwrap();
        set2.insert(&col("b"), &col("c")).unwrap();
        set2.insert(&col("d"), &col("c")).unwrap();

        let mut set = JoinKeySet::new(&Funding::default());
        // put something in there already
        set.insert(&col("x"), &col("y")).unwrap();
        set.insert_intersection(&set1, &set2).unwrap();

        assert_contents(
            &set,
            vec![
                (&col("x"), &col("y")),
                (&col("b"), &col("c")),
                (&col("c"), &col("d")),
            ],
        );
    }

    fn assert_contents(set: &JoinKeySet, expected: Vec<(&Expr, &Expr)>) {
        let contents: Vec<_> = set.iter().collect();
        assert_eq!(contents, expected);
    }

    #[test]
    fn test_insert_all_owned() {
        let mut set = JoinKeySet::new(&Funding::default());

        // insert (a=b), (b=c), (b=a)
        set.insert_all_owned(vec![
            (col("a"), col("b")),
            (col("b"), col("c")),
            (col("b"), col("a")),
        ])
        .unwrap();
        assert_eq!(set.len(), 2);
        assert!(set.contains(&col("a"), &col("b")));
        assert!(set.contains(&col("b"), &col("c")));
        assert!(set.contains(&col("b"), &col("a")));

        // should not contain (a=c)
        assert!(!set.contains(&col("a"), &col("c")));
    }
}
