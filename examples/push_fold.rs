//! Push-Fold
//!
//! This example demonstrates a multi-player game using precomputed payoff tables.
//!
//! Run with: `cargo run --example push_fold`

// ============================================================================
// Poker Game Primitives
// ============================================================================

// We encode hands as card_id = 4 * rank + suit

/// A card, defined as an alias of `u8`.
/// 
/// We define `card_id = 4 * rank + suit` (where `0 <= card_id < 52`)
/// Rank: 2 => `0`, ..., A => `12`
type Card = u8;

/// Constant representing an undealt/unshown hand
const NOT_DEALT: Card = Card::MAX;




// ============================================================================
// Push Fold Game Implementation
// ============================================================================

use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
enum PushFoldAction {
    None,
    Fold,
    Push
}

impl fmt::Display for PushFoldAction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PushFoldAction::None => write!(f, "-"),
            PushFoldAction::Fold => write!(f, "F"),
            PushFoldAction::Push => write!(f, "P"),
        }
    }
}

/// Struct containing game tree configuration
#[derive(Debug, Default, Clone)]
struct PushFoldGameConfig {
    num_players: u8,
    effective_stack: i32,
    small_blind: i32,
    big_blind: i32
}

#[derive(Debug, Default, Clone)]
struct PushFoldGameTree {
    config: PushFoldGameConfig,
    root: PushFoldGameTreeNode,
    history: Vec<PushFoldAction>
}

#[derive(Debug, Default, Clone)]
struct PushFoldGameTreeNode {
    player: u8,
    actions: Vec<PushFoldAction>,
    children: Vec<PushFoldGameTreeNode>,
    history: Vec<PushFoldAction>,
}

impl PushFoldGameTreeNode {
    fn is_terminal(&self) -> bool {
        self.actions.is_empty()
    }

    fn print_tree(&self, indent: usize) {
        let prefix = "  ".repeat(indent);
        let history_str: String = self.history.iter().map(|a| format!("{}", a)).collect::<Vec<_>>().join("");

        if self.is_terminal() {
            // Terminal node - show outcome
            let pushers: Vec<u8> = self.history.iter().enumerate()
                .filter(|(_, a)| **a == PushFoldAction::Push)
                .map(|(i, _)| i as u8)
                .collect();

            let outcome = if pushers.is_empty() {
                "BB wins blinds".to_string()
            } else if pushers.len() == 1 {
                format!("P{} wins uncontested", pushers[0])
            } else {
                format!("Showdown: P{}", pushers.iter().map(|p| p.to_string()).collect::<Vec<_>>().join(" vs P"))
            };

            println!("{}[{}] -> {}", prefix, history_str, outcome);
        } else {
            // Decision node
            let actions_str: String = self.actions.iter().map(|a| format!("{}", a)).collect::<Vec<_>>().join("/");
            println!("{}[{}] P{} to act ({})", prefix, history_str, self.player, actions_str);

            for child in &self.children {
                child.print_tree(indent + 1);
            }
        }
    }
}

impl PushFoldGameTree {
    fn new(config: PushFoldGameConfig) -> Self {
        let mut ret = Self {
            config,
            ..Default::default()
        };
        ret.build_tree();

        ret
    }

    fn build_tree(&mut self) {
        // Our game tree is pretty simple.
        // For n players, for the first n - 1 we get two choices FOLD/PUSH
        // Then on the last player, if nobody has done the PUSH action,
        // the node is terminal (showdown). If someone has done that action,
        // then the final player also gets the choice and we again go to showdown.

        let mut root = PushFoldGameTreeNode {
            player: 0,
            actions: vec![PushFoldAction::Fold, PushFoldAction::Push],
            children: vec![],
            history: vec![]
        };

        self.build_tree_recursive(&mut root);
        self.root = root;
    }

    fn build_tree_recursive(&self, node: &mut PushFoldGameTreeNode) {
        // Base case: if no actions available, this is a terminal node
        if node.actions.is_empty() {
            return;
        }

        for action in node.actions.clone() {
            let next_player = node.player + 1;
            let mut new_history = node.history.clone();
            new_history.push(action.clone());

            // Determine available actions for the child node
            let avail_actions = if next_player >= self.config.num_players {
                // All players have acted - terminal node
                vec![]
            } else if next_player == self.config.num_players - 1 {
                // Last player (big blind)
                if new_history.contains(&PushFoldAction::Push) {
                    // Someone pushed, BB gets to respond
                    vec![PushFoldAction::Fold, PushFoldAction::Push]
                } else {
                    // Everyone folded, BB wins uncontested - terminal
                    vec![]
                }
            } else {
                // Middle players always get fold/push
                vec![PushFoldAction::Fold, PushFoldAction::Push]
            };

            let mut child_node = PushFoldGameTreeNode {
                player: next_player,
                history: new_history,
                actions: avail_actions,
                children: vec![]
            };

            // Recursively build children
            self.build_tree_recursive(&mut child_node);
            node.children.push(child_node);
        }
    }

    fn print(&self) {
        println!("Game Tree ({} players):", self.config.num_players);
        println!();
        self.root.print_tree(0);
    }
}

// ============================================================================
// Main
// ============================================================================

fn main() {
    println!("Push Fold Poker CFR Solver");
    println!("========================");
    println!();
    println!("Rules: 4 players, 20 chip starting stack. Blinds 1/2, all in or fold.");

    let game_config = PushFoldGameConfig {
        num_players: 4,
        effective_stack: 20,
        small_blind: 1,
        big_blind: 2
    };


    println!();
    println!("{:?}", game_config);

    let game_tree = PushFoldGameTree::new(game_config);

    println!();
    game_tree.print();

}
