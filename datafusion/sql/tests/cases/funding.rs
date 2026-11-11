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

/// Funding that refuses what would take it past `limit`, and keeps count of what it holds and
/// the most it held at once.
struct Limited {
    limit: usize,
    held: Mutex<(usize, usize)>,
}

impl PlanningFunding for Limited {
    fn charge(&self, bytes: usize) -> bool {
        let mut held = self.held.lock().unwrap();
        match held.0.checked_add(bytes) {
            Some(total) if total <= self.limit => {
                *held = (total, held.1.max(total));
                true
            }
            _ => false,
        }
    }

    fn discharge(&self, bytes: usize) {
        let mut held = self.held.lock().unwrap();
        held.0 = held
            .0
            .checked_sub(bytes)
            .expect("discharged more than was charged");
    }
}

/// `sql` planned within funding of `limit` bytes, what the funding holds after, and the most it
/// held while planning.
fn plan_within(sql: &str, limit: usize) -> (Result<LogicalPlan>, usize, usize) {
    let source = Arc::new(Limited {
        limit,
        held: Mutex::new((0, 0)),
    });
    let mut state = MockSessionState::default();
    state.config_options.funding =
        Funding::new(Arc::clone(&source) as Arc<dyn PlanningFunding>);
    let context = MockContextProvider { state };
    let planner = SqlToRel::new(&context);
    let mut statements = DFParser::parse_sql(sql).unwrap();
    let plan = planner.statement_to_plan(statements.pop_front().unwrap());
    let (held, peak) = *source.held.lock().unwrap();
    (plan, held, peak)
}

/// Planning `sql` is charged all it makes: within less than the most it held at once it is
/// refused, nothing staying charged; within that much it is planned, holding what the plan
/// keeps.
fn assert_funded(sql: &str) {
    let (plan, kept, needed) = plan_within(sql, usize::MAX);
    let plan = plan.unwrap();
    assert!(kept > 0, "the plan of {sql} keeps nothing charged");

    for limit in 0..needed {
        let (refused, held, _) = plan_within(sql, limit);
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

    let (funded, held, _) = plan_within(sql, needed);
    assert_eq!(funded.unwrap(), plan);
    assert_eq!(held, kept);
}

#[test]
fn select_literal_where_false_is_funded() {
    assert_funded("SELECT 1 WHERE false");
}

#[test]
fn select_order_by_is_funded() {
    assert_funded("SELECT id, age FROM person ORDER BY age DESC, id");
}

#[test]
fn select_limit_offset_is_funded() {
    assert_funded("SELECT id FROM person LIMIT 5 OFFSET 2");
}

#[test]
fn select_distinct_is_funded() {
    assert_funded("SELECT DISTINCT state FROM person");
}

#[test]
fn select_group_by_having_is_funded() {
    assert_funded("SELECT age FROM person GROUP BY age HAVING age > 21");
}

#[test]
fn select_table_alias_is_funded() {
    assert_funded("SELECT p.id FROM person AS p WHERE p.age > 21");
}

#[test]
fn subquery_alias_is_funded_and_failure_frees_it() {
    assert_funded("SELECT a.id FROM (SELECT id FROM person) AS a");
    assert_funded("SELECT * FROM (SELECT * FROM j1 AS a CROSS JOIN j1 AS b) AS c");
    assert_failure_freed("SELECT a.no_such_column FROM (SELECT id FROM person) AS a");
}

#[test]
fn unnest_is_funded_and_failure_frees_it() {
    assert_funded("SELECT UNNEST(array_col) FROM unnest_table");
    assert_failure_freed(
        "SELECT no_such_column FROM (SELECT UNNEST(array_col) AS item FROM unnest_table) AS u",
    );
}

#[test]
fn select_joins_are_funded() {
    assert_funded("SELECT j1_id, j2_string FROM j1 JOIN j2 ON j1_id = j2_id");
    assert_funded("SELECT j1_id FROM j1 LEFT JOIN j2 ON j1_id = j2_id AND j2_id > 1");
    assert_funded("SELECT j1_id, j2_id FROM j1, j2");
    assert_failure_freed(
        "SELECT j1_id FROM j1 JOIN j2 ON no_such_column = j2_id",
    );
    assert_failure_freed("SELECT j1_id FROM j1 JOIN j1 ON true");
}

#[test]
fn merge_into_is_funded() {
    let (_, scan_kept, _) = plan_within(
        "MERGE INTO j1 USING j2 ON false WHEN MATCHED THEN DELETE",
        usize::MAX,
    );
    let (_, wider_target_kept, _) = plan_within(
        "MERGE INTO person USING j2 ON false WHEN MATCHED THEN DELETE",
        usize::MAX,
    );
    assert_eq!(
        wider_target_kept, scan_kept,
        "releasing MERGE target_schema must not over-discharge shared Arrow fields"
    );
    assert_funded("MERGE INTO person USING j2 ON person.id = j2_id WHEN MATCHED THEN DELETE");
    assert_failure_freed(
        "MERGE INTO person USING j2 ON no_such_column = j2_id WHEN MATCHED THEN DELETE",
    );
    assert_failure_freed("MERGE INTO j1 USING j1 ON true WHEN MATCHED THEN DELETE");
}

#[test]
fn select_from_cte_is_funded() {
    assert_funded("WITH t AS (SELECT id FROM person) SELECT id FROM t");
}

#[test]
fn set_operations_are_funded() {
    assert_funded("SELECT j1_id FROM j1 UNION ALL SELECT j2_id FROM j2");
    assert_funded("SELECT j1_id FROM j1 UNION SELECT j2_id FROM j2");
}

#[test]
fn wildcards_are_funded() {
    assert_funded("SELECT * FROM person");
    assert_funded("SELECT p.* FROM person AS p");
    assert_funded("SELECT * EXCLUDE (age, state) FROM person");
    assert_funded("SELECT * REPLACE (age + 1 AS age) FROM person");
    assert_funded("SELECT * FROM person AS a JOIN person AS b USING (id)");
}

/// Planning `sql` fails other than refused, giving back all it charged.
fn assert_failure_freed(sql: &str) {
    let (plan, held, _) = plan_within(sql, usize::MAX);
    let err = plan.expect_err("planned a query that does not plan");
    assert!(err.refused().is_none(), "{sql} was refused: {err}");
    assert_eq!(held, 0, "{sql} failed leaving {held} held");
}

#[test]
fn a_failed_select_list_gives_back_the_items_planned_before() {
    assert_failure_freed("SELECT * REPLACE (age + 1 AS age), no_such_column FROM person");
    assert_failure_freed(
        "SELECT * REPLACE (age + 1 AS age, no_such_column AS id) FROM person",
    );
}

#[test]
fn pipe_extend_is_funded() {
    assert_funded("SELECT * FROM person |> EXTEND age + 1 AS older");
}
