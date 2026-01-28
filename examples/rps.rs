//! Rock-Paper-Scissors example with custom payoffs.
//!
//! Payoffs:
//! - Scissors wins/losses: ±2
//! - Other wins/losses: ±1
//! - Draws: 0
//!
//! Run with: `cargo run --example rps`

use solver::{CfrSolver, Game, GameNode};

/// The three possible actions in RPS.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    Rock = 0,
    Paper = 1,
    Scissors = 2,
}

impl Action {
    fn from_index(index: usize) -> Self {
        match index {
            0 => Action::Rock,
            1 => Action::Paper,
            2 => Action::Scissors,
            _ => panic!("Invalid action index: {}", index),
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Action::Rock => "Rock",
            Action::Paper => "Paper",
            Action::Scissors => "Scissors",
        }
    }
}

/// A node in the RPS game tree.
#[derive(Debug, Clone)]
pub enum RpsNode {
    Player0Turn,
    Player1Turn {
        p0_action: Action,
    },
    Terminal {
        p0_action: Action,
        p1_action: Action,
    },
}

// Info set IDs:
// 0 = Player 0's turn
// 1 = Player 1's turn

impl GameNode for RpsNode {
    fn is_terminal(&self) -> bool {
        matches!(self, RpsNode::Terminal { .. })
    }

    fn is_chance(&self) -> bool {
        false
    }

    fn current_player(&self) -> usize {
        match self {
            RpsNode::Player0Turn => 0,
            RpsNode::Player1Turn { .. } => 1,
            RpsNode::Terminal { .. } => panic!("No current player at terminal node"),
        }
    }

    fn num_actions(&self) -> usize {
        match self {
            RpsNode::Terminal { .. } => 0,
            _ => 3,
        }
    }

    fn play(&self, action: usize) -> Self {
        let action = Action::from_index(action);
        match self {
            RpsNode::Player0Turn => RpsNode::Player1Turn { p0_action: action },
            RpsNode::Player1Turn { p0_action } => RpsNode::Terminal {
                p0_action: *p0_action,
                p1_action: action,
            },
            RpsNode::Terminal { .. } => panic!("Cannot play at terminal node"),
        }
    }

    fn payoff(&self, player: usize) -> f64 {
        match self {
            RpsNode::Terminal {
                p0_action,
                p1_action,
            } => {
                let p0_payoff = compute_payoff(*p0_action, *p1_action);
                if player == 0 { p0_payoff } else { -p0_payoff }
            }
            _ => panic!("Payoff only available at terminal nodes"),
        }
    }

    fn info_set_id(&self) -> usize {
        match self {
            RpsNode::Player0Turn => 0,
            RpsNode::Player1Turn { .. } => 1,
            RpsNode::Terminal { .. } => panic!("No info set at terminal node"),
        }
    }
}

fn compute_payoff(p0: Action, p1: Action) -> f64 {
    use Action::*;
    match (p0, p1) {
        (Rock, Rock) | (Paper, Paper) | (Scissors, Scissors) => 0.0,
        (Scissors, Paper) => 2.0,
        (Scissors, Rock) => -2.0,
        (Paper, Scissors) => -2.0,
        (Rock, Scissors) => 2.0,
        (Rock, Paper) => -1.0,
        (Paper, Rock) => 1.0,
    }
}

/// The Rock-Paper-Scissors game.
#[derive(Debug, Clone)]
pub struct RpsGame;

impl Game for RpsGame {
    type Node = RpsNode;

    fn root(&self) -> Self::Node {
        RpsNode::Player0Turn
    }

    fn num_players(&self) -> usize {
        2
    }

    fn num_info_sets(&self) -> usize {
        2 // P0's turn, P1's turn
    }
}

fn main() {
    println!("Rock-Paper-Scissors CFR Solver");
    println!("==============================");
    println!();
    println!("Payoffs: Scissors wins/losses = ±2, others = ±1, draws = 0");
    println!();

    let game = RpsGame;
    // Use Linear CFR for symmetric convergence with alternating updates
    let mut solver = CfrSolver::new(&game, solver::CfrVariant::LinearCfr);

    let iterations = 100_000;
    println!("Training for {} iterations...", iterations);
    solver.train(&game, iterations);
    println!("Exploitability: {:.6}", solver.exploitability(&game));

    println!();
    println!("Computed Nash Equilibrium Strategies:");
    println!();

    let actions = [Action::Rock, Action::Paper, Action::Scissors];

    for (info_id, label) in [(0, "Player 0"), (1, "Player 1")] {
        if let Some(strategy) = solver.get_strategy(info_id) {
            println!("{}:", label);
            for (i, &prob) in strategy.iter().enumerate() {
                println!(
                    "  {:8}: {:.4} ({:.1}%)",
                    actions[i].name(),
                    prob,
                    prob * 100.0
                );
            }
            println!();
        }
    }

    println!("Expected Nash Equilibrium:");
    println!("  Due to scissors being worth ±2, the equilibrium shifts.");
    println!("  Players should play Rock more often to exploit Scissors,");
    println!("  and Scissors less often due to higher risk.");
}
