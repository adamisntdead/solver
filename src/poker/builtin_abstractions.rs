//! Built-in hand abstractions for common configurations.
//!
//! This module provides pre-configured abstractions that match Gambit's standard
//! configuration:
//! - Flop: SemiAggSI (deterministic, generated on demand)
//! - Turn: AsymEMD with 64000 buckets (requires external file)
//! - River: WinSplit with 500 buckets (requires external file)
//!
//! # Usage
//!
//! ```ignore
//! use solver::poker::builtin_abstractions::get_flop_abstraction;
//!
//! // Get the built-in flop abstraction for any flop board
//! let abs = get_flop_abstraction(&board);
//! ```

use std::path::Path;

use crate::poker::abstraction::{InfoAbstraction, SemiAggSIAbstraction};
use crate::poker::abstraction_gen::GeneratedAbstraction;
use crate::poker::abstraction_io::load_abstraction;
use crate::poker::hands::Board;
use crate::tree::Street;

/// Standard abstraction bucket counts (matching Gambit's configuration).
pub const FLOP_BUCKETS: usize = 1170; // SemiAggSI typical count
pub const TURN_BUCKETS: usize = 64000; // AsymEMD
pub const RIVER_BUCKETS: usize = 500; // WinSplit

/// Get the built-in flop abstraction for a board.
///
/// SemiAggSI is deterministic and can be generated on-the-fly.
/// This function caches the abstraction by board signature.
pub fn get_flop_abstraction(board: &Board) -> SemiAggSIAbstraction {
    assert_eq!(board.cards.len(), 3, "Flop abstraction requires 3-card board");
    SemiAggSIAbstraction::new(board)
}

/// Configuration for built-in abstractions.
#[derive(Debug, Clone)]
pub struct BuiltinAbstractionConfig {
    /// Use built-in abstractions when available.
    pub enabled: bool,
    /// Path to turn abstraction file (optional).
    pub turn_file: Option<String>,
    /// Path to river abstraction file (optional).
    pub river_file: Option<String>,
}

impl Default for BuiltinAbstractionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            turn_file: None,
            river_file: None,
        }
    }
}

impl BuiltinAbstractionConfig {
    /// Create a config with no external files.
    pub fn builtin_only() -> Self {
        Self::default()
    }

    /// Create a config with external abstraction files.
    pub fn with_files(turn_file: Option<String>, river_file: Option<String>) -> Self {
        Self {
            enabled: true,
            turn_file,
            river_file,
        }
    }

    /// Disable all built-in abstractions.
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            turn_file: None,
            river_file: None,
        }
    }
}

/// Load a built-in abstraction for the given street.
///
/// Returns `None` if no built-in abstraction is available for the street.
pub fn load_builtin_for_street(
    street: Street,
    board: &Board,
    config: &BuiltinAbstractionConfig,
) -> Option<Box<dyn InfoAbstraction>> {
    if !config.enabled {
        return None;
    }

    match street {
        Street::Flop => {
            // SemiAggSI is generated on-the-fly (deterministic)
            if board.cards.len() == 3 {
                Some(Box::new(get_flop_abstraction(board)))
            } else {
                None
            }
        }
        Street::Turn => {
            // Try to load from external file
            if let Some(ref path) = config.turn_file {
                match load_abstraction(Path::new(path)) {
                    Ok(abs) => Some(Box::new(LoadedBuiltinAbstraction::new(abs))),
                    Err(_) => None,
                }
            } else {
                None
            }
        }
        Street::River => {
            // Try to load from external file
            if let Some(ref path) = config.river_file {
                match load_abstraction(Path::new(path)) {
                    Ok(abs) => Some(Box::new(LoadedBuiltinAbstraction::new(abs))),
                    Err(_) => None,
                }
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Wrapper around GeneratedAbstraction to implement InfoAbstraction.
struct LoadedBuiltinAbstraction {
    inner: GeneratedAbstraction,
}

impl LoadedBuiltinAbstraction {
    fn new(abs: GeneratedAbstraction) -> Self {
        Self { inner: abs }
    }
}

impl InfoAbstraction for LoadedBuiltinAbstraction {
    fn bucket(&self, iso_hand: usize, _context: usize) -> usize {
        if iso_hand < self.inner.assignments.len() {
            self.inner.assignments[iso_hand] as usize
        } else {
            0
        }
    }

    fn num_buckets(&self) -> usize {
        self.inner.num_buckets
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::poker::board_parser::parse_board;

    #[test]
    fn test_flop_abstraction_generation() {
        let board = parse_board("KhQsJs").unwrap();
        let abs = get_flop_abstraction(&board);

        // SemiAggSI should produce around 1000-1500 buckets
        assert!(abs.num_buckets() > 500);
        assert!(abs.num_buckets() < 2000);
    }

    #[test]
    fn test_builtin_config_disabled() {
        let config = BuiltinAbstractionConfig::disabled();
        let board = parse_board("KhQsJs").unwrap();

        assert!(load_builtin_for_street(Street::Flop, &board, &config).is_none());
    }

    #[test]
    fn test_builtin_config_enabled() {
        let config = BuiltinAbstractionConfig::builtin_only();
        let board = parse_board("KhQsJs").unwrap();

        let abs = load_builtin_for_street(Street::Flop, &board, &config);
        assert!(abs.is_some());
    }
}
