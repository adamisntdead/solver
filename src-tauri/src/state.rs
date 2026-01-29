//! Solver state management for Tauri.

use std::sync::Mutex;

use solver::poker::{PostflopGame, PostflopSolver};

/// Global solver state managed by Tauri.
pub struct SolverState {
    pub game: Mutex<Option<PostflopGame>>,
    pub solver: Mutex<Option<PostflopSolver>>,
    pub pot: Mutex<i32>,
}

impl SolverState {
    pub fn new() -> Self {
        SolverState {
            game: Mutex::new(None),
            solver: Mutex::new(None),
            pot: Mutex::new(0),
        }
    }
}

impl Default for SolverState {
    fn default() -> Self {
        Self::new()
    }
}
