//! Poker game tree construction and configuration.
//!
//! This module provides types and algorithms for building action trees
//! for poker games. It supports:
//! - Multi-way pots (2-6 players)
//! - Pre-flop and post-flop play
//! - Configurable bet sizes (Pio/Monker style)
//! - No-limit and pot-limit betting
//! - Memory estimation

pub mod action;
pub mod bet_size;
pub mod builder;
pub mod config;
pub mod memory;
pub mod position;

pub use action::{Action, BettingState, TerminalResult};
pub use bet_size::{BetSize, BetSizeOptions};
pub use builder::{ActionTree, ActionTreeNode};
pub use config::{BetType, PreflopConfig, Street, StreetConfig, TreeConfig};
pub use memory::MemoryEstimate;
pub use position::Position;
