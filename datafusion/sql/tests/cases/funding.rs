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

use std::sync::{Arc, Mutex};

use datafusion_common::Result;
use datafusion_common::funding::{Funding, PlanningFunding};
use datafusion_expr::LogicalPlan;
use datafusion_sql::parser::DFParser;
use datafusion_sql::planner::SqlToRel;

use crate::common::{MockContextProvider, MockSessionState};

/// Funding that refuses what would take it past `limit`, and keeps count of what it holds.
struct Limited {
    limit: usize,
    held: Mutex<usize>,
}

impl PlanningFunding for Limited {
    fn charge(&self, bytes: usize) -> bool {
        let mut held = self.held.lock().unwrap();
        match held.checked_add(bytes) {
            Some(total) if total <= self.limit => {
                *held = total;
                true
            }
            _ => false,
        }
    }

    fn discharge(&self, bytes: usize) {
        let mut held = self.held.lock().unwrap();
        *held = held
            .checked_sub(bytes)
            .expect("discharged more than was charged");
    }
}

/// `sql` planned within funding of `limit` bytes, and what the funding holds after.
fn plan_within(sql: &str, limit: usize) -> (Result<LogicalPlan>, usize) {
    let source = Arc::new(Limited {
        limit,
        held: Mutex::new(0),
    });
    let mut state = MockSessionState::default();
    state.config_options.funding =
        Funding::new(Arc::clone(&source) as Arc<dyn PlanningFunding>);
    let context = MockContextProvider { state };
    let planner = SqlToRel::new(&context);
    let mut statements = DFParser::parse_sql(sql).unwrap();
    let plan = planner.statement_to_plan(statements.pop_front().unwrap());
    let held = *source.held.lock().unwrap();
    (plan, held)
}

/// Planning `sql` charges exactly what the plan holds: within any less it is refused, and
/// nothing stays charged; within that much it is planned.
fn assert_funded(sql: &str) {
    let (plan, needed) = plan_within(sql, usize::MAX);
    let plan = plan.unwrap();
    assert!(needed > 0, "planning {sql} charged nothing");

    for limit in 0..needed {
        let (refused, held) = plan_within(sql, limit);
        let err = refused.expect_err("planned within less than it needs");
        assert!(
            err.refused().is_some(),
            "{sql} within {limit} of {needed} bytes failed other than refused: {err}"
        );
        assert_eq!(
            held, 0,
            "{sql} refused within {limit} bytes left {held} held"
        );
    }

    let (funded, held) = plan_within(sql, needed);
    assert_eq!(funded.unwrap(), plan);
    assert_eq!(held, needed);
}

#[test]
fn select_literal_where_false_is_funded() {
    assert_funded("SELECT 1 WHERE false");
}
