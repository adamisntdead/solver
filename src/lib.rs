//! A counterfactual regret minimization (CFR) solver for n-player games.
//!
//! This library provides:
//! - [`Game`] and [`GameNode`] traits for defining games
//! - [`CfrSolver`] implementing discounted CFR (requires `rand` feature)
//! - [`InfoAbstraction`] trait for grouping similar private states
//!
//! # Examples
//!
//! See the examples for complete implementations:
//! ```bash
//! cargo run --example rps      # Rock-Paper-Scissors
//! cargo run --example kuhn13   # Kuhn poker with 13 cards
//! ```

#[cfg(feature = "rand")]
pub mod cfr;
pub mod game;
pub mod info_abstraction;
pub mod tree;

#[cfg(feature = "rand")]
pub use cfr::{CfrSolver, CfrVariant, DiscountParams};
pub use game::{Game, GameNode};
pub use info_abstraction::{IdentityAbstraction, InfoAbstraction};
pub use tree::{
    Action, ActionTree, ActionTreeNode, BetSize, BetSizeOptions, BetType, MemoryEstimate,
    Position, PreflopConfig, Street, StreetConfig, TerminalResult, TreeConfig, TreeStats,
};
