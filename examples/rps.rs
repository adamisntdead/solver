//! Rock-Paper-Scissors example with custom payoffs.
//!
//! Payoffs:
//! - Scissors wins/losses: ±2
//! - Other wins/losses: ±1
//! - Draws: 0
//!
//! Run with: `cargo run --example rps`

use solver::{CfrSolver, DiscountParams, Game, GameNode};

/// The three possible actions in RPS.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    Rock = 0,
    Paper = 1,
    Scissors = 2,
}

impl Action {
    /// Converts an action index to an Action.
    fn from_index(index: usize) -> Self {
        match index {
            0 => Action::Rock,
            1 => Action::Paper,
            2 => Action::Scissors,
            _ => panic!("Invalid action index: {}", index),
        }
    }

    /// Returns the action name as a string.
    pub fn name(&self) -> &'static str {
        match self {
            Action::Rock => "Rock",
            Action::Paper => "Paper",
            Action::Scissors => "Scissors",
        }
    }
}

/// A node in the RPS game tree.
///
/// The game is modeled as:
/// 1. Player 0 chooses (secretly)
/// 2. Player 1 chooses (secretly)
/// 3. Terminal: payoffs are determined
#[derive(Debug, Clone)]
pub enum RpsNode {
    /// Initial state: Player 0 to act
    Player0Turn,
    /// Player 0 has acted, Player 1 to act
    Player1Turn { p0_action: Action },
    /// Game over: both players have acted
    Terminal { p0_action: Action, p1_action: Action },
}

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
            _ => 3, // Rock, Paper, Scissors
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
            RpsNode::Terminal { p0_action, p1_action } => {
                let p0_payoff = compute_payoff(*p0_action, *p1_action);
                if player == 0 {
                    p0_payoff
                } else {
                    -p0_payoff // Zero-sum game
                }
            }
            _ => panic!("Payoff only available at terminal nodes"),
        }
    }

    fn info_set_key(&self) -> String {
        // Each player only knows it's their turn, not the opponent's action
        match self {
            RpsNode::Player0Turn => "P0".to_string(),
            RpsNode::Player1Turn { .. } => "P1".to_string(),
            RpsNode::Terminal { .. } => panic!("No info set at terminal node"),
        }
    }
}

/// Computes the payoff for player 0 given both players' actions.
///
/// Payoff structure:
/// - Scissors wins/losses: ±2
/// - Other wins/losses: ±1
/// - Draws: 0
fn compute_payoff(p0: Action, p1: Action) -> f64 {
    use Action::*;

    match (p0, p1) {
        // Draws
        (Rock, Rock) | (Paper, Paper) | (Scissors, Scissors) => 0.0,

        // P0 wins with Scissors (beats Paper) -> +2
        (Scissors, Paper) => 2.0,
        // P0 loses with Scissors (to Rock) -> -2
        (Scissors, Rock) => -2.0,

        // P1 wins with Scissors (beats Paper) -> P0 loses, -2
        (Paper, Scissors) => -2.0,
        // P1 loses with Scissors (to Rock) -> P0 wins, +2
        (Rock, Scissors) => 2.0,

        // Non-scissors matchups: ±1
        (Rock, Paper) => -1.0,  // P0 Rock loses to P1 Paper
        (Paper, Rock) => 1.0,   // P0 Paper beats P1 Rock
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
}

fn main() {
    println!("Rock-Paper-Scissors CFR Solver");
    println!("==============================");
    println!();
    println!("Payoffs: Scissors wins/losses = ±2, others = ±1, draws = 0");
    println!();

    let game = RpsGame;

    // Create solver with discounted CFR parameters
    let mut solver = CfrSolver::new(DiscountParams::default());

    // Train for iterations
    let iterations = 10_000;
    println!("Training for {} iterations...", iterations);
    solver.train(&game, iterations);
    println!("Exploitability: {:.6}", solver.exploitability(&game));

    // Print results
    println!();
    println!("Computed Nash Equilibrium Strategies:");
    println!();

    let actions = [Action::Rock, Action::Paper, Action::Scissors];

    for (info_set, label) in [("P0", "Player 0"), ("P1", "Player 1")] {
        if let Some(strategy) = solver.get_strategy(info_set) {
            println!("{}:", label);
            for (i, &prob) in strategy.iter().enumerate() {
                println!("  {:8}: {:.4} ({:.1}%)", actions[i].name(), prob, prob * 100.0);
            }
            println!();
        }
    }

    let exploitability = solver.exploitability(&game);
    println!("Exploitability: {:.6}", exploitability);
    println!();

    // Explain the expected Nash equilibrium
    println!("Expected Nash Equilibrium:");
    println!("  Due to scissors being worth ±2, the equilibrium shifts.");
    println!("  Players should play Rock more often to exploit Scissors,");
    println!("  and Scissors less often due to higher risk.");
}
