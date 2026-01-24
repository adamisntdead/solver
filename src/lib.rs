//! A counterfactual regret minimization (CFR) solver for n-player games.
//!
//! This library provides:
//! - [`Game`] and [`GameNode`] traits for defining games
//! - [`CfrSolver`] implementing discounted CFR
//!
//! # Example
//!
//! See the `rps` example for a complete Rock-Paper-Scissors implementation:
//! ```bash
//! cargo run --example rps
//! ```

pub mod cfr;
pub mod game;

pub use cfr::{CfrSolver, DiscountParams};
pub use game::{Game, GameNode};
