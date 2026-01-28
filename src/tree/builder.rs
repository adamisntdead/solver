//! Action tree construction.

use crate::tree::action::{Action, BettingState, TerminalResult};
use crate::tree::bet_size::BetSize;
use crate::tree::config::{Street, TreeConfig};

/// A node in the action tree.
#[derive(Debug, Clone)]
pub enum ActionTreeNode {
    /// Player decision point.
    Player {
        /// Which player acts.
        player: usize,
        /// Legal actions at this node.
        actions: Vec<Action>,
        /// One child per action.
        children: Vec<ActionTreeNode>,
    },

    /// Chance node (card dealt).
    Chance {
        /// Which street we're transitioning to.
        street: Street,
        /// Child node (we abstract over specific cards).
        child: Box<ActionTreeNode>,
    },

    /// Terminal node.
    Terminal {
        /// Result type.
        result: TerminalResult,
        /// Final pot size.
        pot: i32,
    },
}

impl ActionTreeNode {
    /// Check if this is a terminal node.
    pub fn is_terminal(&self) -> bool {
        matches!(self, ActionTreeNode::Terminal { .. })
    }

    /// Check if this is a chance node.
    pub fn is_chance(&self) -> bool {
        matches!(self, ActionTreeNode::Chance { .. })
    }

    /// Check if this is a player node.
    pub fn is_player(&self) -> bool {
        matches!(self, ActionTreeNode::Player { .. })
    }

    /// Count all nodes in the subtree.
    pub fn node_count(&self) -> usize {
        match self {
            ActionTreeNode::Terminal { .. } => 1,
            ActionTreeNode::Chance { child, .. } => 1 + child.node_count(),
            ActionTreeNode::Player { children, .. } => {
                1 + children.iter().map(|c| c.node_count()).sum::<usize>()
            }
        }
    }

    /// Count terminal nodes in the subtree.
    pub fn terminal_count(&self) -> usize {
        match self {
            ActionTreeNode::Terminal { .. } => 1,
            ActionTreeNode::Chance { child, .. } => child.terminal_count(),
            ActionTreeNode::Player { children, .. } => {
                children.iter().map(|c| c.terminal_count()).sum()
            }
        }
    }

    /// Get the maximum depth of the subtree.
    pub fn max_depth(&self) -> usize {
        match self {
            ActionTreeNode::Terminal { .. } => 1,
            ActionTreeNode::Chance { child, .. } => 1 + child.max_depth(),
            ActionTreeNode::Player { children, .. } => {
                1 + children.iter().map(|c| c.max_depth()).max().unwrap_or(0)
            }
        }
    }

    /// Count player nodes (info set nodes) in the subtree.
    pub fn player_node_count(&self) -> usize {
        match self {
            ActionTreeNode::Terminal { .. } => 0,
            ActionTreeNode::Chance { child, .. } => child.player_node_count(),
            ActionTreeNode::Player { children, .. } => {
                1 + children.iter().map(|c| c.player_node_count()).sum::<usize>()
            }
        }
    }
}

/// The complete action tree.
#[derive(Debug)]
pub struct ActionTree {
    /// Tree configuration.
    pub config: TreeConfig,

    /// Root node of the tree.
    pub root: ActionTreeNode,

    /// Total node count.
    pub node_count: usize,

    /// Terminal node count.
    pub terminal_count: usize,

    /// Maximum tree depth.
    pub max_depth: usize,

    /// Player (info set) node count.
    pub player_node_count: usize,
}

impl ActionTree {
    /// Build a new action tree from configuration.
    pub fn new(config: TreeConfig) -> Result<Self, String> {
        config.validate()?;

        let mut builder = TreeBuilder::new(&config);
        let root = builder.build();

        let node_count = root.node_count();
        let terminal_count = root.terminal_count();
        let max_depth = root.max_depth();
        let player_node_count = root.player_node_count();

        Ok(Self {
            config,
            root,
            node_count,
            terminal_count,
            max_depth,
            player_node_count,
        })
    }

    /// Get tree statistics.
    pub fn stats(&self) -> TreeStats {
        TreeStats {
            node_count: self.node_count,
            terminal_count: self.terminal_count,
            max_depth: self.max_depth,
            player_node_count: self.player_node_count,
        }
    }
}

/// Tree statistics.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TreeStats {
    pub node_count: usize,
    pub terminal_count: usize,
    pub max_depth: usize,
    pub player_node_count: usize,
}

/// Node type for indexed tree.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndexedNodeType {
    /// Player decision node.
    Player { player: usize },
    /// Chance node (card dealt).
    Chance,
    /// Terminal: fold (winner gets pot).
    TerminalFold { winner: usize },
    /// Terminal: showdown.
    TerminalShowdown,
    /// Terminal: all-in runout.
    TerminalAllIn { num_players: usize },
}

/// A node in the indexed (flattened) action tree.
#[derive(Debug, Clone)]
pub struct IndexedNode {
    /// Node type.
    pub node_type: IndexedNodeType,
    /// Actions available (empty for terminal/chance).
    pub actions: Vec<Action>,
    /// Child node indices (one per action, or one for chance).
    pub children: Vec<usize>,
    /// Current pot size.
    pub pot: i32,
}

impl IndexedNode {
    /// Check if this is a terminal node.
    pub fn is_terminal(&self) -> bool {
        matches!(
            self.node_type,
            IndexedNodeType::TerminalFold { .. }
                | IndexedNodeType::TerminalShowdown
                | IndexedNodeType::TerminalAllIn { .. }
        )
    }

    /// Check if this is a chance node.
    pub fn is_chance(&self) -> bool {
        matches!(self.node_type, IndexedNodeType::Chance)
    }

    /// Check if this is a player node.
    pub fn is_player(&self) -> bool {
        matches!(self.node_type, IndexedNodeType::Player { .. })
    }

    /// Get the player to act (panics if not a player node).
    pub fn player(&self) -> usize {
        match self.node_type {
            IndexedNodeType::Player { player } => player,
            _ => panic!("Not a player node"),
        }
    }
}

/// Flattened action tree for O(1) node access.
///
/// The recursive ActionTreeNode is converted to a flat vector
/// where each node has indices to its children.
#[derive(Debug)]
pub struct IndexedActionTree {
    /// All nodes in the tree.
    pub nodes: Vec<IndexedNode>,
    /// Index of the root node (always 0).
    pub root_idx: usize,
    /// Number of player nodes (for info set counting).
    pub player_node_count: usize,
}

impl IndexedActionTree {
    /// Get the root node.
    pub fn root(&self) -> &IndexedNode {
        &self.nodes[self.root_idx]
    }

    /// Get a node by index.
    pub fn get(&self, idx: usize) -> &IndexedNode {
        &self.nodes[idx]
    }

    /// Get the number of nodes.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Check if the tree is empty.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }
}

impl ActionTree {
    /// Convert to an indexed (flattened) tree for efficient traversal.
    pub fn to_indexed(&self) -> IndexedActionTree {
        let mut nodes = Vec::with_capacity(self.node_count);
        let mut player_node_count = 0;

        fn flatten(
            node: &ActionTreeNode,
            nodes: &mut Vec<IndexedNode>,
            player_count: &mut usize,
        ) -> usize {
            let idx = nodes.len();

            match node {
                ActionTreeNode::Terminal { result, pot } => {
                    let node_type = match result {
                        TerminalResult::Fold { winner } => {
                            IndexedNodeType::TerminalFold { winner: *winner }
                        }
                        TerminalResult::Showdown => IndexedNodeType::TerminalShowdown,
                        TerminalResult::AllInRunout { num_players } => {
                            IndexedNodeType::TerminalAllIn {
                                num_players: *num_players,
                            }
                        }
                    };
                    nodes.push(IndexedNode {
                        node_type,
                        actions: Vec::new(),
                        children: Vec::new(),
                        pot: *pot,
                    });
                }
                ActionTreeNode::Chance { child, .. } => {
                    // Reserve spot for this node
                    nodes.push(IndexedNode {
                        node_type: IndexedNodeType::Chance,
                        actions: Vec::new(),
                        children: Vec::new(),
                        pot: 0,
                    });

                    let child_idx = flatten(child, nodes, player_count);
                    nodes[idx].children.push(child_idx);
                    // Copy pot from child
                    nodes[idx].pot = nodes[child_idx].pot;
                }
                ActionTreeNode::Player {
                    player,
                    actions,
                    children,
                } => {
                    *player_count += 1;

                    // Reserve spot for this node
                    nodes.push(IndexedNode {
                        node_type: IndexedNodeType::Player { player: *player },
                        actions: actions.clone(),
                        children: Vec::new(),
                        pot: 0,
                    });

                    let child_indices: Vec<usize> = children
                        .iter()
                        .map(|c| flatten(c, nodes, player_count))
                        .collect();

                    nodes[idx].children = child_indices;
                    // Pot at this node is the pot before any action
                    // We'll get it from the first terminal or estimate from structure
                    if let Some(&first_child) = nodes[idx].children.first() {
                        // Walk to first terminal to get pot context
                        let mut current = first_child;
                        while !nodes[current].is_terminal() && !nodes[current].children.is_empty() {
                            current = nodes[current].children[0];
                        }
                        nodes[idx].pot = nodes[current].pot;
                    }
                }
            }

            idx
        }

        flatten(&self.root, &mut nodes, &mut player_node_count);

        IndexedActionTree {
            nodes,
            root_idx: 0,
            player_node_count,
        }
    }
}

/// Internal tree builder.
struct TreeBuilder<'a> {
    config: &'a TreeConfig,
}

impl<'a> TreeBuilder<'a> {
    fn new(config: &'a TreeConfig) -> Self {
        Self { config }
    }

    fn build(&mut self) -> ActionTreeNode {
        // Determine starting street
        let starting_street = self.config.starting_street.unwrap_or_else(|| {
            // Auto-detect: preflop if configured, else first postflop street
            if self.config.preflop.is_some() {
                Street::Preflop
            } else if self.config.flop.is_some() {
                Street::Flop
            } else if self.config.turn.is_some() {
                Street::Turn
            } else if self.config.river.is_some() {
                Street::River
            } else {
                Street::Preflop // Fallback
            }
        });

        // Create initial state based on starting street
        let state = if starting_street == Street::Preflop && self.config.preflop.is_some() {
            BettingState::new(self.config)
        } else {
            // Postflop start
            let initial_pot = if self.config.starting_pot > 0 {
                self.config.starting_pot
            } else {
                // Default: 2/3 of effective stack as pot (typical postflop spot)
                (self.config.effective_stack() * 2 / 3).max(1)
            };
            BettingState::new_postflop(self.config, starting_street, initial_pot)
        };

        // Make sure the starting street config exists
        if starting_street != Street::Preflop && self.config.street_config(starting_street).is_none() {
            return ActionTreeNode::Terminal {
                result: TerminalResult::Showdown,
                pot: 0,
            };
        }

        self.build_recursive(state)
    }

    fn build_recursive(&mut self, state: BettingState) -> ActionTreeNode {
        // Check for terminal conditions
        if state.active_count() == 1 {
            // Everyone else folded
            let winner = (0..state.num_players)
                .find(|&i| state.is_active(i))
                .unwrap();
            return ActionTreeNode::Terminal {
                result: TerminalResult::Fold { winner },
                pot: state.pot,
            };
        }

        // Check if round is complete
        if state.round_complete || state.is_round_complete() {
            return self.handle_round_complete(state);
        }

        // Generate actions for current player
        let player = state.current_actor;
        let actions = self.generate_actions(&state);

        if actions.is_empty() {
            // No valid actions - this shouldn't happen
            return ActionTreeNode::Terminal {
                result: TerminalResult::Showdown,
                pot: state.pot,
            };
        }

        let mut children = Vec::with_capacity(actions.len());
        for &action in &actions {
            let mut next_state = state.clone();
            next_state.apply_action(action);
            children.push(self.build_recursive(next_state));
        }

        ActionTreeNode::Player {
            player,
            actions,
            children,
        }
    }

    fn handle_round_complete(&mut self, mut state: BettingState) -> ActionTreeNode {
        // Check for all-in runout
        if state.has_all_in {
            // Count players who can still act (have chips)
            let active_with_chips = (0..state.num_players)
                .filter(|&i| state.is_active(i) && state.stacks[i] > 0)
                .count();

            if active_with_chips <= 1 {
                // All-in runout - skip to showdown
                return ActionTreeNode::Terminal {
                    result: TerminalResult::AllInRunout {
                        num_players: state.active_count(),
                    },
                    pot: state.pot,
                };
            }
        }

        // Check if we've reached showdown (river complete)
        if state.street == Street::River {
            return ActionTreeNode::Terminal {
                result: TerminalResult::Showdown,
                pot: state.pot,
            };
        }

        // Move to next street
        let next_street = state.street.next().unwrap();

        // Check if we have config for the next street
        let has_next_street = match next_street {
            Street::Preflop => false, // Can't go back to preflop
            Street::Flop => self.config.flop.is_some(),
            Street::Turn => self.config.turn.is_some(),
            Street::River => self.config.river.is_some(),
        };

        if !has_next_street {
            // No config for next street - terminal
            return ActionTreeNode::Terminal {
                result: TerminalResult::Showdown,
                pot: state.pot,
            };
        }

        // Advance to next street
        state.advance_street();

        // Create chance node
        ActionTreeNode::Chance {
            street: next_street,
            child: Box::new(self.build_recursive(state)),
        }
    }

    fn generate_actions(&self, state: &BettingState) -> Vec<Action> {
        let player = state.current_actor;
        let stack = state.stacks[player];
        let current_bet = state.bets[player];
        let max_bet = *state.bets.iter().max().unwrap_or(&0);
        let to_call = max_bet - current_bet;
        let pot = state.pot;

        let mut actions = Vec::new();

        // Fold (if facing a bet)
        if to_call > 0 {
            actions.push(Action::Fold);
        }

        // Check or Call
        if to_call == 0 {
            actions.push(Action::Check);
        } else if to_call <= stack {
            actions.push(Action::Call(current_bet + to_call));
        } else {
            // Call for less (all-in)
            actions.push(Action::AllIn(current_bet + stack));
            return actions; // Can't raise when calling all-in
        }

        // Can we raise?
        if state.num_raises >= self.config.max_raises_per_round {
            return actions;
        }

        if stack <= to_call {
            return actions; // No chips to raise with
        }

        // Calculate bet sizing
        let min_raise = state.min_raise();
        let max_possible = state.max_bet(self.config.bet_type);
        let all_in = current_bet + stack;

        if min_raise > all_in {
            return actions; // Can't make minimum raise
        }

        // Get configured bet sizes
        let bet_amounts = self.get_bet_amounts(state, pot, to_call, max_bet, min_raise, all_in);

        // Add bet/raise actions
        for amount in bet_amounts {
            if amount >= min_raise && amount <= max_possible {
                let action = if to_call == 0 {
                    if amount == all_in {
                        Action::AllIn(amount)
                    } else {
                        Action::Bet(amount)
                    }
                } else if amount == all_in {
                    Action::AllIn(amount)
                } else {
                    Action::Raise(amount)
                };

                actions.push(action);
            }
        }

        // Apply bet merging
        actions = merge_bet_actions(actions, pot, max_bet, self.config.merge_threshold);

        // Ensure all-in is present if close to stack
        self.maybe_add_all_in(&mut actions, state, pot, all_in, max_possible);

        // Force all-in if threshold met
        self.force_all_in_if_needed(&mut actions, state, pot, all_in);

        // Remove duplicates and sort
        actions.sort();
        actions.dedup();

        actions
    }

    fn get_bet_amounts(
        &self,
        state: &BettingState,
        pot: i32,
        to_call: i32,
        max_bet: i32,
        min_raise: i32,
        all_in: i32,
    ) -> Vec<i32> {
        let mut amounts = Vec::new();
        let streets_remaining = state.street.streets_remaining();
        let player = state.current_actor;

        // Get bet sizes from config
        let bet_sizes = self.get_configured_sizes(state, player);

        for size in bet_sizes {
            let amount = match size {
                BetSize::PotRelative(ratio) => {
                    let effective_pot = pot + to_call;
                    max_bet + ((effective_pot as f64) * ratio).round() as i32
                }
                BetSize::PrevBetRelative(ratio) => {
                    if max_bet > 0 {
                        ((max_bet as f64) * ratio).round() as i32
                    } else {
                        continue; // Skip for opening bets
                    }
                }
                BetSize::Additive(chips) => max_bet + chips,
                BetSize::Geometric {
                    num_streets,
                    max_pot_pct: _,
                } => {
                    let n = if num_streets == 0 {
                        streets_remaining
                    } else {
                        num_streets
                    };
                    let geo_amount = size.resolve(pot, to_call, max_bet, all_in, n);
                    max_bet + geo_amount
                }
                BetSize::AllIn => all_in,
            };

            // Clamp to valid range
            let clamped = amount.clamp(min_raise, all_in);
            amounts.push(clamped);
        }

        // Always include all-in as an option
        amounts.push(all_in);

        // Only add min_raise if no other bet sizes were configured
        // (avoids adding useless "bet 1" for postflop spots)
        if amounts.len() <= 1 {
            amounts.push(min_raise);
        }

        amounts.sort();
        amounts.dedup();
        amounts
    }

    fn get_configured_sizes(&self, state: &BettingState, _player: usize) -> Vec<BetSize> {
        let street = state.street;
        let raise_count = state.num_raises as usize;

        // Get street config
        let street_config = match street {
            Street::Preflop => {
                // Use preflop config
                if let Some(ref pf) = self.config.preflop {
                    return match raise_count {
                        0 => pf
                            .open_sizes
                            .first()
                            .map(|s| s.raise.clone())  // Open raise uses raise sizes
                            .unwrap_or_default(),
                        1 => pf
                            .three_bet_sizes
                            .first()
                            .map(|s| s.raise.clone())
                            .unwrap_or_default(),
                        _ => pf
                            .four_bet_plus_sizes
                            .first()
                            .map(|s| s.reraise_plus.clone())
                            .unwrap_or_default(),
                    };
                }
                return vec![BetSize::PotRelative(0.75), BetSize::AllIn];
            }
            Street::Flop => self.config.flop.as_ref(),
            Street::Turn => self.config.turn.as_ref(),
            Street::River => self.config.river.as_ref(),
        };

        if let Some(sc) = street_config {
            if let Some(sizes) = sc.sizes.first() {
                return sizes.sizes_for_raise_count(raise_count as u8).to_vec();
            }
        }

        // Default sizes
        vec![
            BetSize::PotRelative(0.33),
            BetSize::PotRelative(0.67),
            BetSize::PotRelative(1.0),
            BetSize::AllIn,
        ]
    }

    fn maybe_add_all_in(
        &self,
        actions: &mut Vec<Action>,
        state: &BettingState,
        pot: i32,
        all_in: i32,
        max_possible: i32,
    ) {
        // Find maximum bet action
        let max_action_amount = actions
            .iter()
            .filter_map(|a| match a {
                Action::Bet(x) | Action::Raise(x) | Action::AllIn(x) => Some(*x),
                _ => None,
            })
            .max()
            .unwrap_or(0);

        let max_bet = *state.bets.iter().max().unwrap_or(&0);

        // Add all-in if max bet is less than threshold * pot
        let threshold = (pot as f64 * self.config.add_all_in_threshold).round() as i32;
        if max_action_amount < max_bet + threshold && all_in <= max_possible {
            // Check if all-in is already in actions
            if !actions.iter().any(|a| matches!(a, Action::AllIn(_))) {
                actions.push(Action::AllIn(all_in));
            }
        }
    }

    fn force_all_in_if_needed(
        &self,
        actions: &mut Vec<Action>,
        state: &BettingState,
        pot: i32,
        all_in: i32,
    ) {
        let player = state.current_actor;
        let stack = state.stacks[player];
        let max_bet = *state.bets.iter().max().unwrap_or(&0);

        // Calculate pot after a potential bet
        for action in actions.iter_mut() {
            if let Action::Bet(amount) | Action::Raise(amount) = action {
                let bet_size = *amount - max_bet;
                let pot_after_bet = pot + 2 * bet_size;
                let remaining_stack = stack - (*amount - state.bets[player]);

                // Force all-in if remaining stack is too small relative to pot
                let threshold = (pot_after_bet as f64 * self.config.force_all_in_threshold).round() as i32;
                if remaining_stack > 0 && remaining_stack <= threshold {
                    *action = Action::AllIn(all_in);
                }
            }
        }
    }
}

/// Merge similar bet actions using Pio algorithm.
///
/// Algorithm: Select the highest bet size (= X% of pot) and remove all bet actions
/// with a value (= Y% of pot) satisfying: (100 + X) / (100 + Y) < 1.0 + threshold
fn merge_bet_actions(
    mut actions: Vec<Action>,
    pot: i32,
    offset: i32,
    threshold: f64,
) -> Vec<Action> {
    if threshold <= 0.0 || pot <= 0 {
        return actions;
    }

    let get_amount = |action: &Action| -> Option<i32> {
        match action {
            Action::Bet(a) | Action::Raise(a) | Action::AllIn(a) => Some(*a),
            _ => None,
        }
    };

    // Sort by amount descending
    actions.sort_by(|a, b| {
        let amt_a = get_amount(a).unwrap_or(0);
        let amt_b = get_amount(b).unwrap_or(0);
        amt_b.cmp(&amt_a)
    });

    let mut result = Vec::new();
    let mut last_kept_amount: Option<i32> = None;

    for action in actions {
        if let Some(amount) = get_amount(&action) {
            let ratio = (amount - offset) as f64 / pot as f64;

            if let Some(last) = last_kept_amount {
                let last_ratio = (last - offset) as f64 / pot as f64;

                // Pio merge formula: keep if ratio is sufficiently different
                // (100 + X) / (100 + Y) >= 1.0 + threshold
                // Rearranged: Y <= (100 + X) / (1.0 + threshold) - 100
                let threshold_ratio = (last_ratio - threshold) / (1.0 + threshold);

                if ratio < threshold_ratio * 0.999 {
                    result.push(action);
                    last_kept_amount = Some(amount);
                }
                // else: skip this action (too close to last kept)
            } else {
                result.push(action);
                last_kept_amount = Some(amount);
            }
        } else {
            result.push(action);
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tree::bet_size::BetSizeOptions;
    use crate::tree::config::{PreflopConfig, StreetConfig};

    #[test]
    fn test_simple_tree() {
        let config = TreeConfig::new(2)
            .with_stack(100)
            .with_preflop(PreflopConfig::new(1, 2));

        let tree = ActionTree::new(config).unwrap();

        assert!(tree.node_count > 0);
        assert!(tree.terminal_count > 0);
        assert!(tree.max_depth > 0);
    }

    #[test]
    fn test_postflop_tree() {
        let sizes = BetSizeOptions::try_from_strs("33%, 67%, 100%", "2.5x, a").unwrap();

        let config = TreeConfig::new(2)
            .with_stack(100)
            .with_flop(StreetConfig::uniform(sizes));

        let mut state = BettingState::new_postflop(&config, Street::Flop, 10);
        state.round_complete = false;

        // Verify state setup
        assert_eq!(state.pot, 10);
        assert_eq!(state.street, Street::Flop);
    }

    #[test]
    fn test_tree_stats() {
        let config = TreeConfig::new(2)
            .with_stack(50)
            .with_preflop(PreflopConfig::new(1, 2));

        let tree = ActionTree::new(config).unwrap();
        let stats = tree.stats();

        assert!(stats.node_count >= stats.terminal_count);
        assert!(stats.player_node_count > 0);
    }

    #[test]
    fn test_merge_bet_actions() {
        let actions = vec![
            Action::Check,
            Action::Bet(50),
            Action::Bet(55),
            Action::Bet(100),
            Action::AllIn(200),
        ];

        let merged = merge_bet_actions(actions, 100, 0, 0.1);

        // 55 should be merged with either 50 or 100
        assert!(merged.len() < 5);
        assert!(merged.contains(&Action::Check));
        assert!(merged.contains(&Action::AllIn(200)));
    }

    #[test]
    fn test_node_counts() {
        let node = ActionTreeNode::Player {
            player: 0,
            actions: vec![Action::Fold, Action::Call(10)],
            children: vec![
                ActionTreeNode::Terminal {
                    result: TerminalResult::Fold { winner: 1 },
                    pot: 10,
                },
                ActionTreeNode::Terminal {
                    result: TerminalResult::Showdown,
                    pot: 20,
                },
            ],
        };

        assert_eq!(node.node_count(), 3);
        assert_eq!(node.terminal_count(), 2);
        assert_eq!(node.player_node_count(), 1);
    }
}
