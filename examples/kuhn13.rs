//! Kuhn-13: Extended Kuhn Poker with 13 cards.
//!
//! This example demonstrates information abstraction by comparing:
//! - Full solution (no abstraction): 13 distinct card strategies
//! - Abstracted solution: Cards grouped into strength buckets
//!
//! Run with: `cargo run --example kuhn13`

use solver::{CfrSolver, Game, GameNode, IdentityAbstraction, InfoAbstraction};
use std::sync::Arc;

// ============================================================================
// Game-specific information abstraction
// ============================================================================

/// Buckets cards into N equal-sized strength tiers.
#[derive(Debug, Clone, Copy)]
pub struct StrengthBuckets {
    num_buckets: usize,
    max_card: usize,
}

impl StrengthBuckets {
    pub fn new(num_buckets: usize, max_card: usize) -> Self {
        Self {
            num_buckets,
            max_card,
        }
    }
}

impl InfoAbstraction<u8> for StrengthBuckets {
    fn bucket(&self, card: &u8) -> usize {
        let zero_indexed = (*card as usize) - 1;
        let bucket = (zero_indexed * self.num_buckets) / self.max_card;
        bucket.min(self.num_buckets - 1)
    }

    fn num_buckets(&self) -> usize {
        self.num_buckets
    }
}

// ============================================================================
// Kuhn-13 Game Implementation
// ============================================================================

const NUM_CARDS: u8 = 13;

// Decision points (for computing info set IDs):
// 0 = P0Turn
// 1 = P1AfterCheck
// 2 = P1AfterBet
// 3 = P0AfterCheckBet
const NUM_DECISION_POINTS: usize = 4;

/// Game state in Kuhn-13.
#[derive(Debug, Clone)]
pub enum Kuhn13Node<A: InfoAbstraction<u8>> {
    Deal {
        abstraction: Arc<A>,
    },
    P0Turn {
        p0_card: u8,
        p1_card: u8,
        abstraction: Arc<A>,
    },
    P1AfterCheck {
        p0_card: u8,
        p1_card: u8,
        abstraction: Arc<A>,
    },
    P1AfterBet {
        p0_card: u8,
        p1_card: u8,
        abstraction: Arc<A>,
    },
    P0AfterCheckBet {
        p0_card: u8,
        p1_card: u8,
        abstraction: Arc<A>,
    },
    ShowdownCheckCheck {
        p0_card: u8,
        p1_card: u8,
    },
    ShowdownBetCall {
        p0_card: u8,
        p1_card: u8,
    },
    P1Folded,
    P0Folded,
}

impl<A: InfoAbstraction<u8> + Clone> GameNode for Kuhn13Node<A> {
    fn is_terminal(&self) -> bool {
        matches!(
            self,
            Kuhn13Node::ShowdownCheckCheck { .. }
                | Kuhn13Node::ShowdownBetCall { .. }
                | Kuhn13Node::P1Folded
                | Kuhn13Node::P0Folded
        )
    }

    fn is_chance(&self) -> bool {
        matches!(self, Kuhn13Node::Deal { .. })
    }

    fn current_player(&self) -> usize {
        match self {
            Kuhn13Node::P0Turn { .. } | Kuhn13Node::P0AfterCheckBet { .. } => 0,
            Kuhn13Node::P1AfterCheck { .. } | Kuhn13Node::P1AfterBet { .. } => 1,
            _ => panic!("No current player at this node"),
        }
    }

    fn num_actions(&self) -> usize {
        match self {
            Kuhn13Node::Deal { .. } => (NUM_CARDS as usize) * (NUM_CARDS as usize - 1),
            Kuhn13Node::P0Turn { .. } | Kuhn13Node::P1AfterCheck { .. } => 2, // Check/Bet
            Kuhn13Node::P1AfterBet { .. } | Kuhn13Node::P0AfterCheckBet { .. } => 2, // Fold/Call
            _ => 0,
        }
    }

    fn play(&self, action: usize) -> Self {
        match self {
            Kuhn13Node::Deal { abstraction } => {
                let p0_card = (action / (NUM_CARDS as usize - 1)) as u8 + 1;
                let p1_offset = action % (NUM_CARDS as usize - 1);
                let p1_card = if p1_offset < (p0_card as usize - 1) {
                    p1_offset as u8 + 1
                } else {
                    p1_offset as u8 + 2
                };
                Kuhn13Node::P0Turn {
                    p0_card,
                    p1_card,
                    abstraction: abstraction.clone(),
                }
            }
            Kuhn13Node::P0Turn {
                p0_card,
                p1_card,
                abstraction,
            } => {
                if action == 0 {
                    Kuhn13Node::P1AfterCheck {
                        p0_card: *p0_card,
                        p1_card: *p1_card,
                        abstraction: abstraction.clone(),
                    }
                } else {
                    Kuhn13Node::P1AfterBet {
                        p0_card: *p0_card,
                        p1_card: *p1_card,
                        abstraction: abstraction.clone(),
                    }
                }
            }
            Kuhn13Node::P1AfterCheck {
                p0_card,
                p1_card,
                abstraction,
            } => {
                if action == 0 {
                    Kuhn13Node::ShowdownCheckCheck {
                        p0_card: *p0_card,
                        p1_card: *p1_card,
                    }
                } else {
                    Kuhn13Node::P0AfterCheckBet {
                        p0_card: *p0_card,
                        p1_card: *p1_card,
                        abstraction: abstraction.clone(),
                    }
                }
            }
            Kuhn13Node::P1AfterBet {
                p0_card, p1_card, ..
            } => {
                if action == 0 {
                    Kuhn13Node::P1Folded
                } else {
                    Kuhn13Node::ShowdownBetCall {
                        p0_card: *p0_card,
                        p1_card: *p1_card,
                    }
                }
            }
            Kuhn13Node::P0AfterCheckBet {
                p0_card, p1_card, ..
            } => {
                if action == 0 {
                    Kuhn13Node::P0Folded
                } else {
                    Kuhn13Node::ShowdownBetCall {
                        p0_card: *p0_card,
                        p1_card: *p1_card,
                    }
                }
            }
            _ => panic!("Cannot play at terminal node"),
        }
    }

    fn payoff(&self, player: usize) -> f64 {
        let p0_payoff = match self {
            Kuhn13Node::ShowdownCheckCheck { p0_card, p1_card } => {
                if p0_card > p1_card {
                    1.0
                } else {
                    -1.0
                }
            }
            Kuhn13Node::ShowdownBetCall {
                p0_card, p1_card, ..
            } => {
                if p0_card > p1_card {
                    2.0
                } else {
                    -2.0
                }
            }
            Kuhn13Node::P1Folded => 1.0,
            Kuhn13Node::P0Folded => -1.0,
            _ => panic!("Payoff only available at terminal nodes"),
        };
        if player == 0 { p0_payoff } else { -p0_payoff }
    }

    fn info_set_id(&self) -> usize {
        // Info set ID = decision_point * num_buckets + bucket
        match self {
            Kuhn13Node::P0Turn {
                p0_card,
                abstraction,
                ..
            } => 0 * abstraction.num_buckets() + abstraction.bucket(p0_card),
            Kuhn13Node::P1AfterCheck {
                p1_card,
                abstraction,
                ..
            } => 1 * abstraction.num_buckets() + abstraction.bucket(p1_card),
            Kuhn13Node::P1AfterBet {
                p1_card,
                abstraction,
                ..
            } => 2 * abstraction.num_buckets() + abstraction.bucket(p1_card),
            Kuhn13Node::P0AfterCheckBet {
                p0_card,
                abstraction,
                ..
            } => 3 * abstraction.num_buckets() + abstraction.bucket(p0_card),
            _ => panic!("No info set at this node"),
        }
    }
}

/// The Kuhn-13 game with configurable information abstraction.
pub struct Kuhn13Game<A: InfoAbstraction<u8>> {
    abstraction: Arc<A>,
}

impl<A: InfoAbstraction<u8>> Kuhn13Game<A> {
    pub fn new(abstraction: A) -> Self {
        Self {
            abstraction: Arc::new(abstraction),
        }
    }
}

impl<A: InfoAbstraction<u8> + Clone + Send + Sync> Game for Kuhn13Game<A> {
    type Node = Kuhn13Node<A>;

    fn root(&self) -> Self::Node {
        Kuhn13Node::Deal {
            abstraction: self.abstraction.clone(),
        }
    }

    fn num_players(&self) -> usize {
        2
    }

    fn num_info_sets(&self) -> usize {
        NUM_DECISION_POINTS * self.abstraction.num_buckets()
    }
}

// ============================================================================
// Main
// ============================================================================

fn main() {
    println!("Kuhn-13 Poker CFR Solver");
    println!("========================");
    println!();
    println!("Deck: Cards 1-13, higher card wins");
    println!("Rules: Ante 1, bet size 1, standard Kuhn betting");
    println!();

    // --- Full solution (no abstraction) ---
    println!("=== Full Solution (no abstraction) ===");
    println!("Each of the 13 cards has its own strategy.");
    println!();

    let full_game = Kuhn13Game::new(IdentityAbstraction::new(NUM_CARDS as usize + 1));
    let mut full_solver = CfrSolver::new(&full_game, solver::CfrVariant::CfrPlus);

    let iterations = 100_000; // CFR+ converges well with alternating updates

    let start = std::time::Instant::now();
    full_solver.train(&full_game, iterations);
    let train_time = start.elapsed();

    let start = std::time::Instant::now();
    let full_exploitability = full_solver.exploitability(&full_game);
    let exploit_time = start.elapsed();

    println!(
        "Training: {:?}, Exploitability calc: {:?}",
        train_time, exploit_time
    );
    println!("Exploitability: {:.6}", full_exploitability);
    println!();

    // Print some strategies (info_set_id = 0*13 + card for P0Turn)
    println!("Sample strategies (P0 opening action: Check/Bet):");
    for card in [1u8, 7, 13] {
        let info_id = 0 * 14 + card as usize; // IdentityAbstraction uses card value directly
        if let Some(strategy) = full_solver.get_strategy(info_id) {
            println!(
                "  Card {:2}: Check {:.1}%, Bet {:.1}%",
                card,
                strategy[0] * 100.0,
                strategy[1] * 100.0
            );
        }
    }
    println!();

    // --- Abstracted solution (3 buckets) ---
    println!("=== Abstracted Solution (3 buckets) ===");
    println!("Cards grouped: 1-4 (weak), 5-8 (medium), 9-13 (strong)");
    println!();

    let abstracted_game = Kuhn13Game::new(StrengthBuckets::new(3, NUM_CARDS as usize));
    let mut abstracted_solver = CfrSolver::new(&abstracted_game, solver::CfrVariant::CfrPlus);

    let start = std::time::Instant::now();
    abstracted_solver.train(&abstracted_game, iterations);
    let train_time = start.elapsed();

    let abstracted_exploitability = abstracted_solver.exploitability(&abstracted_game);
    println!("Training: {:?}", train_time);
    println!("Exploitability: {:.6}", abstracted_exploitability);
    println!();

    println!("Bucket strategies (P0 opening action: Check/Bet):");
    for (bucket, name) in [(0, "Weak (1-4)"), (1, "Medium (5-8)"), (2, "Strong (9-13)")] {
        let info_id = 0 * 3 + bucket; // decision_point * num_buckets + bucket
        if let Some(strategy) = abstracted_solver.get_strategy(info_id) {
            println!(
                "  {:14}: Check {:.1}%, Bet {:.1}%",
                name,
                strategy[0] * 100.0,
                strategy[1] * 100.0
            );
        }
    }
    println!();

    // --- Comparison ---
    println!("=== Comparison ===");
    println!(
        "Full solution exploitability:       {:.6}",
        full_exploitability
    );
    println!(
        "Abstracted solution exploitability: {:.6}",
        abstracted_exploitability
    );
}
