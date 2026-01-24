//! Information abstraction for grouping similar private states.
//!
//! This module provides traits for **information abstraction**, which groups
//! similar private game states into buckets that share strategy computation.
//!
//! # Important Distinction
//!
//! This is **information abstraction** (grouping private states), NOT action
//! abstraction (discretizing continuous action spaces like bet sizes).
//!
//! # How It Works
//!
//! - States in the same bucket **share regret/strategy accumulation**
//! - The game tree traversal still uses **real private states**
//! - Terminal payoffs are computed with **exact values**, preserving accuracy
//! - This reduces strategy space while maintaining value precision

/// Maps private game states to abstract buckets for information set grouping.
///
/// Implement this trait to define how private states should be grouped.
/// States mapping to the same bucket will share strategy computation in CFR.
pub trait InfoAbstraction<PrivateState> {
    /// Returns the bucket ID for a given private state.
    ///
    /// States returning the same bucket ID will be treated as strategically
    /// equivalent and share the same learned strategy.
    fn bucket(&self, state: &PrivateState) -> usize;

    /// Returns the total number of buckets.
    fn num_buckets(&self) -> usize;
}

/// Identity abstraction - no bucketing, each state maps to itself.
///
/// Use this when you want the full, unabstracted solution where every
/// distinct private state has its own strategy.
#[derive(Debug, Clone, Copy, Default)]
pub struct IdentityAbstraction {
    num_states: usize,
}

impl IdentityAbstraction {
    /// Creates an identity abstraction with the given number of states.
    pub fn new(num_states: usize) -> Self {
        Self { num_states }
    }
}

impl InfoAbstraction<u8> for IdentityAbstraction {
    fn bucket(&self, state: &u8) -> usize {
        *state as usize
    }

    fn num_buckets(&self) -> usize {
        self.num_states
    }
}

impl InfoAbstraction<usize> for IdentityAbstraction {
    fn bucket(&self, state: &usize) -> usize {
        *state
    }

    fn num_buckets(&self) -> usize {
        self.num_states
    }
}
