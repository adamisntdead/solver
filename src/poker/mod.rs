//! Poker CFR solver with hand isomorphism.
//!
//! This module provides a range-based CFR solver for postflop poker that:
//! - Enumerates all valid (OOP combo, IP combo) pairs for faster convergence
//! - Uses hand isomorphism to reduce the state space via suit canonicalization
//! - Integrates with the ActionTree betting structure from the tree module
//!
//! # Architecture
//!
//! - [`Combo`] and [`Range`]: Represent hole card combinations and weighted ranges
//! - [`Board`]: Represents community cards with dead card tracking
//! - [`RiverIsomorphism`]: Maps combos to canonical buckets for state reduction
//! - [`MatchupTable`]: Precomputes valid matchups and showdown results
//! - [`PostflopGame`]: Implements the [`Game`] trait for CFR solving
//!
//! # Example
//!
//! ```ignore
//! use solver::poker::{PostflopGame, Range, Board};
//! use solver::{CfrSolver, CfrVariant};
//!
//! let board = Board::from_str("KhQsJs")?;
//! let oop_range = Range::from_str("AA,KK,QQ")?;
//! let ip_range = Range::from_str("AA,KK,AKs")?;
//!
//! let game = PostflopGame::new(config, board, oop_range, ip_range)?;
//! let mut solver = CfrSolver::new(&game, CfrVariant::CfrPlus);
//! solver.train(&game, 1000);
//! ```

pub mod board_parser;
pub mod hands;
pub mod isomorphism;
pub mod matchups;
pub mod postflop_game;
pub mod postflop_solver;
pub mod range_parser;

pub use board_parser::*;
pub use hands::{Board, Card, Combo, Range, NUM_COMBOS};
pub use isomorphism::{RiverIsomorphism, SuitMapping};
pub use matchups::MatchupTable;
pub use postflop_game::{PostflopConfig, PostflopGame, PostflopNode};
pub use postflop_solver::PostflopSolver;
pub use range_parser::*;

// Re-export tree types for convenience
pub use crate::tree::{IndexedActionTree, IndexedNode, IndexedNodeType};
