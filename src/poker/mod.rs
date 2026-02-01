//! Poker CFR solver with hand isomorphism and abstraction.
//!
//! This module provides a range-based CFR solver for postflop poker that:
//! - Enumerates all valid (OOP combo, IP combo) pairs for faster convergence
//! - Uses hand isomorphism to reduce the state space via suit canonicalization
//! - Supports pluggable hand abstraction for additional compression
//! - Integrates with the ActionTree betting structure from the tree module
//!
//! # Architecture
//!
//! - [`Combo`] and [`Range`]: Represent hole card combinations and weighted ranges
//! - [`Board`]: Represents community cards with dead card tracking
//! - [`BoardIsomorphism`]: Maps combos to canonical buckets (generalized for any board)
//! - [`RiverIsomorphism`]: Legacy type for 5-card boards
//! - [`HandAbstraction`]: Trait for hand abstraction (isomorphism + optional bucketing)
//! - [`SuitIsomorphism`]: Lossless abstraction using suit symmetry
//! - [`ComposedAbstraction`]: Two-layer abstraction (suit iso + info abstraction)
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

pub mod abstraction;
pub mod abstraction_gen;
pub mod abstraction_io;
pub mod builtin_abstractions;
pub mod board_parser;
pub mod clustering;
pub mod ehs;
pub mod hands;
pub mod indexer;
pub mod isomorphism;
pub mod matchups;
pub mod postflop_game;
pub mod postflop_solver;
pub mod range_parser;

pub use abstraction::{
    compute_valid_cards, create_abstraction, AggSIAbstraction, ComposedAbstraction,
    EHSAbstraction, EMDAbstraction, HandAbstraction, InfoAbstraction, SemiAggSIAbstraction,
    SuitIsomorphism, WinSplitAbstraction,
};
pub use abstraction_gen::{AbstractionConfig, AbstractionType, GeneratedAbstraction};
pub use abstraction_io::{
    is_gambit_format, load_abstraction, load_abstraction_from_bytes, save_abstraction,
    AbstractionIOError,
};
#[cfg(feature = "zstd")]
pub use abstraction_io::{load_abstraction_auto, load_gambit_abstraction};
pub use builtin_abstractions::{
    get_flop_abstraction, load_builtin_for_street, BuiltinAbstractionConfig,
    FLOP_BUCKETS, RIVER_BUCKETS, TURN_BUCKETS,
};
pub use clustering::{DistanceMetric, KMeansConfig, KMeansResult};
pub use ehs::{compute_all_ehs, compute_emd_features, compute_winsplit_features, EMD_NUM_BINS};
pub use indexer::{ImperfectRecallIndexer, SingleBoardIndexer, Street};
pub use board_parser::*;
pub use hands::{Board, Card, Combo, Range, NUM_COMBOS};
pub use isomorphism::{BoardIsomorphism, RiverIsomorphism, SuitMapping, INVALID_BUCKET};
pub use matchups::{compute_multiway_equity, MatchupTable};
pub use postflop_game::{PostflopConfig, PostflopGame, PostflopNode};
pub use postflop_solver::{LoadedAbstractions, PostflopSolver};
pub use range_parser::*;

// Re-export tree types for convenience
pub use crate::tree::{IndexedActionTree, IndexedNode, IndexedNodeType};
