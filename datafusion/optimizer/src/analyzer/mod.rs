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

//! [`Analyzer`] and [`AnalyzerRule`]

use std::fmt::Debug;
use std::sync::Arc;

use log::debug;

use datafusion_common::Result;
use datafusion_common::cloned::CloneBytes;
use datafusion_common::config::ConfigOptions;
use datafusion_common::instant::Instant;
use datafusion_expr::expr_rewriter::FunctionRewrite;
use datafusion_expr::{InvariantLevel, LogicalPlan};

use crate::analyzer::resolve_grouping_function::ResolveGroupingFunction;
use crate::analyzer::type_coercion::TypeCoercion;
use crate::funded::{Released, with_context};
use crate::utils::log_plan;

use self::function_rewrite::ApplyFunctionRewrites;

pub mod function_rewrite;
pub mod resolve_grouping_function;
pub mod type_coercion;

/// [`AnalyzerRule`]s transform [`LogicalPlan`]s in some way to make
/// the plan valid prior to the rest of the DataFusion optimization process.
///
/// `AnalyzerRule`s are different than an [`OptimizerRule`](crate::OptimizerRule)s
/// which must preserve the semantics of the `LogicalPlan`, while computing
/// results in a more optimal way.
///
/// For example, an `AnalyzerRule` may resolve [`Expr`](datafusion_expr::Expr)s into more specific
/// forms such as a subquery reference, or do type coercion to ensure the types
/// of operands are correct.
///
/// Use [`SessionState::add_analyzer_rule`] to register additional
/// `AnalyzerRule`s.
///
/// [`SessionState::add_analyzer_rule`]: https://docs.rs/datafusion/latest/datafusion/execution/session_state/struct.SessionState.html#method.add_analyzer_rule
pub trait AnalyzerRule: Debug {
    /// Rewrite `plan`
    fn analyze(&self, plan: LogicalPlan, config: &ConfigOptions) -> Result<LogicalPlan>;

    /// A human readable name for this analyzer rule
    fn name(&self) -> &str;
}

/// Rule-based Analyzer.
///
/// Applies [`FunctionRewrite`]s and [`AnalyzerRule`]s to transform a
/// [`LogicalPlan`] in preparation for execution.
///
/// For example, the `Analyzer` applies type coercion to ensure the types of
/// operands match the types required by functions.
#[derive(Clone, Debug)]
pub struct Analyzer {
    /// Expr --> Function writes to apply prior to analysis passes
    pub function_rewrites: Vec<Arc<dyn FunctionRewrite + Send + Sync>>,
    /// All rules to apply
    pub rules: Vec<Arc<dyn AnalyzerRule + Send + Sync>>,
}

impl Default for Analyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl Analyzer {
    /// Create a new analyzer using the recommended list of rules
    pub fn new() -> Self {
        let rules: Vec<Arc<dyn AnalyzerRule + Send + Sync>> = vec![
            Arc::new(ResolveGroupingFunction::new()),
            Arc::new(TypeCoercion::new()),
        ];
        Self::with_rules(rules)
    }

    /// Create a new analyzer with the given rules
    pub fn with_rules(rules: Vec<Arc<dyn AnalyzerRule + Send + Sync>>) -> Self {
        Self {
            function_rewrites: vec![],
            rules,
        }
    }

    /// Add a function rewrite rule
    pub fn add_function_rewrite(
        &mut self,
        rewrite: Arc<dyn FunctionRewrite + Send + Sync>,
    ) {
        self.function_rewrites.push(rewrite);
    }

    /// return the list of function rewrites in this analyzer
    pub fn function_rewrites(&self) -> &[Arc<dyn FunctionRewrite + Send + Sync>] {
        &self.function_rewrites
    }

    /// Analyze the logical plan by applying analyzer rules, and
    /// do necessary check and fail the invalid plans
    ///
    /// What analyzing allocates is charged to the options' funding before it is
    /// allocated; failing, everything analyzing made is freed and given back.
    pub fn execute_and_check<F>(
        &self,
        plan: LogicalPlan,
        config: &ConfigOptions,
        observer: F,
    ) -> Result<LogicalPlan>
    where
        F: FnMut(&LogicalPlan, &dyn AnalyzerRule),
    {
        let before = config.funding.outstanding();
        let analyzed = self.analyze_with_rules(plan, config, observer);
        if analyzed.is_err() {
            config.funding.unwound(before);
        }
        analyzed
    }

    fn analyze_with_rules<F>(
        &self,
        plan: LogicalPlan,
        config: &ConfigOptions,
        mut observer: F,
    ) -> Result<LogicalPlan>
    where
        F: FnMut(&LogicalPlan, &dyn AnalyzerRule),
    {
        // verify the logical plan required invariants at the start, before analyzer
        plan.check_invariants(InvariantLevel::Always)
            .map_err(|e| with_context(e, || "Invalid input plan passed to Analyzer"))?;

        let start_time = Instant::now();
        let mut new_plan = plan;

        // Create an analyzer pass that rewrites `Expr`s to function_calls, as
        // appropriate: its list of rewrites copied, charged first, and freed with it.
        //
        // Note this is run before all other rules since it rewrites based on
        // the argument types (List or Scalar), and TypeCoercion may cast the
        // argument types from Scalar to List.
        let expr_to_function = if self.function_rewrites.is_empty() {
            None
        } else {
            Some(Released::new(
                &config.funding,
                ApplyFunctionRewrites::new(
                    config.funding.cloned(&self.function_rewrites)?,
                ),
                self.function_rewrites.clone_bytes(),
            ))
        };
        let rules = expr_to_function
            .iter()
            .map(|rule| &**rule as &dyn AnalyzerRule)
            .chain(
                self.rules
                    .iter()
                    .map(|rule| rule.as_ref() as &dyn AnalyzerRule),
            );

        // TODO add common rule executor for Analyzer and Optimizer
        for rule in rules {
            new_plan = rule
                .analyze(new_plan, config)
                .map_err(|e| with_context(e, || rule.name()))?;
            log_plan(format_args!("{}", rule.name()), &new_plan);
            observer(&new_plan, rule);
        }

        // verify at the end, after the last LP analyzer pass, that the plan is executable.
        new_plan
            .check_invariants(InvariantLevel::Executable)
            .map_err(|e| {
                with_context(e, || "Invalid (non-executable) plan after Analyzer")
            })?;

        log_plan(format_args!("Final analyzed plan"), &new_plan);
        debug!("Analyzer took {} ms", start_time.elapsed().as_millis());
        Ok(new_plan)
    }
}
