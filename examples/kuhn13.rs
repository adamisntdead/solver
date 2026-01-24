//! Kuhn-13: Extended Kuhn Poker with 13 cards.
//!
//! This example demonstrates information abstraction by comparing:
//! - Full solution (no abstraction): 13 distinct card strategies
//! - Abstracted solution: Cards grouped into strength buckets
//!
//! Run with: `cargo run --example kuhn13`

use std::sync::Arc;

use solver::{CfrSolver, DiscountParams, Game, GameNode, IdentityAbstraction, InfoAbstraction};

// ============================================================================
// Game-specific information abstraction
// ============================================================================

/// Buckets cards into N equal-sized strength tiers.
///
/// Example with 3 buckets for cards 1-13:
/// - Cards 1-4 → bucket 0 (weak)
/// - Cards 5-8 → bucket 1 (medium)
/// - Cards 9-13 → bucket 2 (strong)
#[derive(Debug, Clone, Copy)]
pub struct StrengthBuckets {
    num_buckets: usize,
    max_card: usize,
}

impl StrengthBuckets {
    pub fn new(num_buckets: usize, max_card: usize) -> Self {
        assert!(num_buckets > 0, "Must have at least one bucket");
        assert!(max_card > 0, "Max card must be positive");
        Self {
            num_buckets,
            max_card,
        }
    }
}

impl InfoAbstraction<u8> for StrengthBuckets {
    fn bucket(&self, card: &u8) -> usize {
        let card = *card as usize;
        assert!(card >= 1 && card <= self.max_card, "Card out of range");
        // Convert to 0-indexed, divide into buckets
        let zero_indexed = card - 1;
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

/// Actions in Kuhn poker.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    Check = 0,
    Bet = 1,
    Fold = 2,
    Call = 3,
}

impl Action {
    fn from_index(index: usize) -> Self {
        match index {
            0 => Action::Check,
            1 => Action::Bet,
            2 => Action::Fold,
            3 => Action::Call,
            _ => panic!("Invalid action index"),
        }
    }
}

/// Game state in Kuhn-13.
#[derive(Debug, Clone)]
pub enum Kuhn13Node<A: InfoAbstraction<u8>> {
    /// Chance node: dealing cards
    Deal {
        abstraction: Arc<A>,
    },
    /// Player 0 acts first
    P0Turn {
        p0_card: u8,
        p1_card: u8,
        abstraction: Arc<A>,
    },
    /// Player 1 responds to check
    P1AfterCheck {
        p0_card: u8,
        p1_card: u8,
        abstraction: Arc<A>,
    },
    /// Player 1 responds to bet
    P1AfterBet {
        p0_card: u8,
        p1_card: u8,
        abstraction: Arc<A>,
    },
    /// Player 0 responds to check-bet
    P0AfterCheckBet {
        p0_card: u8,
        p1_card: u8,
        abstraction: Arc<A>,
    },
    /// Showdown after check-check
    ShowdownCheckCheck {
        p0_card: u8,
        p1_card: u8,
    },
    /// Showdown after bet-call or check-bet-call
    ShowdownBetCall {
        p0_card: u8,
        p1_card: u8,
        bet_by_p0: bool,
    },
    /// P1 folded to P0's bet
    P1Folded,
    /// P0 folded to P1's bet (after check-bet)
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
            Kuhn13Node::P0Turn { .. } => 2,        // Check, Bet
            Kuhn13Node::P1AfterCheck { .. } => 2,  // Check, Bet
            Kuhn13Node::P1AfterBet { .. } => 2,    // Fold, Call
            Kuhn13Node::P0AfterCheckBet { .. } => 2, // Fold, Call
            _ => 0,
        }
    }

    fn play(&self, action: usize) -> Self {
        match self {
            Kuhn13Node::Deal { abstraction } => {
                // Decode action into card pair (p0_card, p1_card)
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
                let action = Action::from_index(action);
                match action {
                    Action::Check => Kuhn13Node::P1AfterCheck {
                        p0_card: *p0_card,
                        p1_card: *p1_card,
                        abstraction: abstraction.clone(),
                    },
                    Action::Bet => Kuhn13Node::P1AfterBet {
                        p0_card: *p0_card,
                        p1_card: *p1_card,
                        abstraction: abstraction.clone(),
                    },
                    _ => panic!("Invalid action for P0Turn"),
                }
            }
            Kuhn13Node::P1AfterCheck {
                p0_card,
                p1_card,
                abstraction,
            } => {
                let action = Action::from_index(action);
                match action {
                    Action::Check => Kuhn13Node::ShowdownCheckCheck {
                        p0_card: *p0_card,
                        p1_card: *p1_card,
                    },
                    Action::Bet => Kuhn13Node::P0AfterCheckBet {
                        p0_card: *p0_card,
                        p1_card: *p1_card,
                        abstraction: abstraction.clone(),
                    },
                    _ => panic!("Invalid action for P1AfterCheck"),
                }
            }
            Kuhn13Node::P1AfterBet { p0_card, p1_card, .. } => {
                // Actions are Fold=0, Call=1 (mapped from indices 2, 3)
                match action {
                    0 => Kuhn13Node::P1Folded,
                    1 => Kuhn13Node::ShowdownBetCall {
                        p0_card: *p0_card,
                        p1_card: *p1_card,
                        bet_by_p0: true,
                    },
                    _ => panic!("Invalid action for P1AfterBet"),
                }
            }
            Kuhn13Node::P0AfterCheckBet { p0_card, p1_card, .. } => {
                match action {
                    0 => Kuhn13Node::P0Folded,
                    1 => Kuhn13Node::ShowdownBetCall {
                        p0_card: *p0_card,
                        p1_card: *p1_card,
                        bet_by_p0: false,
                    },
                    _ => panic!("Invalid action for P0AfterCheckBet"),
                }
            }
            _ => panic!("Cannot play at terminal node"),
        }
    }

    fn payoff(&self, player: usize) -> f64 {
        // Pot starts with 2 (1 ante each), bets add 1 each
        let p0_payoff = match self {
            Kuhn13Node::ShowdownCheckCheck { p0_card, p1_card } => {
                // Pot = 2 (antes only), winner gets opponent's ante
                if p0_card > p1_card { 1.0 } else { -1.0 }
            }
            Kuhn13Node::ShowdownBetCall { p0_card, p1_card, .. } => {
                // Pot = 4 (antes + bets), winner gets opponent's ante + bet
                if p0_card > p1_card { 2.0 } else { -2.0 }
            }
            Kuhn13Node::P1Folded => 1.0,  // P0 wins P1's ante
            Kuhn13Node::P0Folded => -1.0, // P1 wins P0's ante
            _ => panic!("Payoff only available at terminal nodes"),
        };

        if player == 0 { p0_payoff } else { -p0_payoff }
    }

    fn info_set_key(&self) -> String {
        match self {
            Kuhn13Node::P0Turn { p0_card, abstraction, .. } => {
                let bucket = abstraction.bucket(p0_card);
                format!("P0:b{}:", bucket)
            }
            Kuhn13Node::P1AfterCheck { p1_card, abstraction, .. } => {
                let bucket = abstraction.bucket(p1_card);
                format!("P1:b{}:x", bucket)
            }
            Kuhn13Node::P1AfterBet { p1_card, abstraction, .. } => {
                let bucket = abstraction.bucket(p1_card);
                format!("P1:b{}:b", bucket)
            }
            Kuhn13Node::P0AfterCheckBet { p0_card, abstraction, .. } => {
                let bucket = abstraction.bucket(p0_card);
                format!("P0:b{}:xb", bucket)
            }
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
}

// ============================================================================
// Main: Compare abstracted vs non-abstracted solutions
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
    let mut full_solver = CfrSolver::new(DiscountParams::default());

    let iterations = 10_000;
    println!("Training for {} iterations...", iterations);
    full_solver.train(&full_game, iterations);
    let full_exploitability = full_solver.exploitability(&full_game);
    println!("Exploitability: {:.6}", full_exploitability);
    println!();

    // Print some strategies
    println!("Sample strategies (P0 opening action: Check/Bet):");
    for card in [1, 7, 13] {
        let key = format!("P0:b{}:", card);
        if let Some(strategy) = full_solver.get_strategy(&key) {
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
    let mut abstracted_solver = CfrSolver::new(DiscountParams::default());

    println!("Training for {} iterations...", iterations);
    abstracted_solver.train(&abstracted_game, iterations);
    let abstracted_exploitability = abstracted_solver.exploitability(&abstracted_game);
    println!("Exploitability: {:.6}", abstracted_exploitability);
    println!();

    println!("Bucket strategies (P0 opening action: Check/Bet):");
    for (bucket, name) in [(0, "Weak (1-4)"), (1, "Medium (5-8)"), (2, "Strong (9-13)")] {
        let key = format!("P0:b{}:", bucket);
        if let Some(strategy) = abstracted_solver.get_strategy(&key) {
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
    println!();
    println!("Note: Abstraction trades accuracy for reduced state space.");
    println!("The abstracted solution converges faster but may be more exploitable.");
}
