//! Postflop poker game implementation for CFR solving.
//!
//! This module provides:
//! - [`PostflopGame`]: Implements the [`Game`] trait
//! - [`PostflopNode`]: Implements the [`GameNode`] trait
//!
//! # Range-vs-Range Approach
//!
//! The game starts with a "dealing" chance node that enumerates all valid
//! (OOP combo, IP combo) matchups. The CFR algorithm naturally handles this
//! by recursing into each matchup.
//!
//! # Information Set Encoding
//!
//! Each information set is encoded as:
//! ```text
//! info_set_id = (player * tree_size + node_index) * num_buckets + bucket
//! ```

use std::sync::Arc;

use crate::game::{Game, GameNode};
use crate::poker::hands::{Board, Combo, Range, NUM_COMBOS};
use crate::poker::isomorphism::RiverIsomorphism;
use crate::poker::matchups::MatchupTable;
use crate::tree::{IndexedActionTree, IndexedNode, IndexedNodeType, Street, TreeConfig};

/// Configuration for a postflop game.
#[derive(Clone)]
pub struct PostflopConfig {
    /// Starting pot (chips already in the pot).
    pub pot: i32,
    /// Effective stack size per player.
    pub effective_stack: i32,
    /// The tree configuration.
    pub tree_config: TreeConfig,
}

/// A postflop poker game for CFR solving.
///
/// The game structure is:
/// 1. Root: Chance node that "deals" hands (enumerates matchups)
/// 2. Each matchup leads to the betting tree
///
/// Supports 2-6 players. For n players, ranges and stacks are indexed by
/// tree player ID (0 = first to act postflop, n-1 = last to act).
pub struct PostflopGame {
    /// The flattened action tree.
    pub tree: Arc<IndexedActionTree>,
    /// The board (3-5 cards).
    pub board: Board,
    /// Starting street (derived from board length).
    pub starting_street: Street,
    /// Precomputed matchup table (only for 5-card / river boards).
    pub matchups: Option<Arc<MatchupTable>>,
    /// Hand isomorphism for bucket mapping (only for river boards).
    pub isomorphism: Option<Arc<RiverIsomorphism>>,
    /// Ranges for each player (indexed by tree player ID).
    pub ranges: Vec<Range>,
    /// Starting stacks for each player (indexed by tree player ID).
    pub stacks: Vec<i32>,
    /// Number of players.
    num_players: usize,
    /// Starting pot.
    pub pot: i32,
    /// Number of canonical buckets.
    num_buckets: usize,
    /// Valid matchups for 2-player games: (oop_combo_idx, ip_combo_idx, weight).
    /// For n-player games, use `valid_matchups_multiway` instead.
    valid_matchups: Vec<(usize, usize, f32)>,
    /// Valid matchups for n-player games: (combo_indices, weight).
    /// Each entry is a tuple of combo indices for each player plus the weight.
    valid_matchups_multiway: Vec<(Vec<usize>, f32)>,
    /// Total weight of all matchups (for normalization).
    total_weight: f32,
}

impl PostflopGame {
    /// Create a new postflop game for 2 players (heads-up).
    ///
    /// This is the legacy API for backward compatibility. For n-player games,
    /// use `PostflopGame::new_multiway()`.
    ///
    /// Accepts boards with 3-5 cards:
    /// - 5 cards (river): precomputes matchup table and isomorphism
    /// - 3-4 cards (flop/turn): valid combos from board mask only
    pub fn new(
        tree: IndexedActionTree,
        board: Board,
        oop_range: Range,
        ip_range: Range,
        pot: i32,
        effective_stack: i32,
    ) -> Self {
        // Convert to n-player format (OOP = player 1, IP = player 0 in tree indexing)
        // Note: Player 0 is IP (acts second), Player 1 is OOP (acts first)
        let ranges = vec![ip_range, oop_range];
        let stacks = vec![effective_stack, effective_stack];

        Self::new_multiway(tree, board, ranges, stacks, pot)
    }

    /// Create a new postflop game for 2-6 players.
    ///
    /// Accepts boards with 3-5 cards:
    /// - 5 cards (river): precomputes matchup table and isomorphism
    /// - 3-4 cards (flop/turn): valid combos from board mask only
    ///
    /// # Arguments
    /// * `tree` - The indexed action tree
    /// * `board` - The board (3-5 cards)
    /// * `ranges` - Range for each player (indexed by tree player ID)
    /// * `stacks` - Starting stack for each player
    /// * `pot` - Starting pot size
    pub fn new_multiway(
        tree: IndexedActionTree,
        board: Board,
        ranges: Vec<Range>,
        stacks: Vec<i32>,
        pot: i32,
    ) -> Self {
        let num_players = ranges.len();
        assert!(
            num_players >= 2 && num_players <= 6,
            "Number of players must be 2-6, got {}",
            num_players
        );
        assert_eq!(
            stacks.len(),
            num_players,
            "Stacks length must match ranges length"
        );

        let starting_street = match board.len() {
            3 => Street::Flop,
            4 => Street::Turn,
            5 => Street::River,
            n => panic!("Board must have 3-5 cards, got {}", n),
        };

        let tree = Arc::new(tree);

        // For river boards, compute matchup table and isomorphism
        let (matchups, isomorphism, num_buckets) = if starting_street == Street::River {
            let m = Arc::new(MatchupTable::new(&board));
            let iso = Arc::new(RiverIsomorphism::new(&board));
            let nb = iso.num_buckets;
            (Some(m), Some(iso), nb)
        } else {
            (None, None, NUM_COMBOS)
        };

        // Precompute valid matchups with weights
        let (valid_matchups, valid_matchups_multiway, total_weight) = if num_players == 2 {
            // Use efficient 2-player matchup enumeration
            let (matchups_2p, weight) =
                enumerate_2player_matchups(&board, &ranges[0], &ranges[1]);
            (matchups_2p, Vec::new(), weight)
        } else {
            // Use n-player matchup enumeration
            let (matchups_np, weight) = enumerate_nplayer_matchups(&board, &ranges);
            (Vec::new(), matchups_np, weight)
        };

        PostflopGame {
            tree,
            board,
            starting_street,
            matchups,
            isomorphism,
            ranges,
            stacks,
            num_players,
            pot,
            num_buckets,
            valid_matchups,
            valid_matchups_multiway,
            total_weight,
        }
    }

    /// Get range for a specific player.
    pub fn range(&self, player: usize) -> &Range {
        &self.ranges[player]
    }

    /// Get the effective stack (for 2-player games, returns the first player's stack).
    pub fn effective_stack(&self) -> i32 {
        self.stacks[0]
    }

    /// Get the OOP range (player 1 in tree indexing). For backward compatibility.
    pub fn oop_range(&self) -> &Range {
        &self.ranges[1]
    }

    /// Get the IP range (player 0 in tree indexing). For backward compatibility.
    pub fn ip_range(&self) -> &Range {
        &self.ranges[0]
    }

    /// Get the number of valid matchup pairs (for 2-player games).
    pub fn num_matchups(&self) -> usize {
        if self.num_players == 2 {
            self.valid_matchups.len()
        } else {
            self.valid_matchups_multiway.len()
        }
    }

    /// Iterate over valid matchups (for 2-player games only).
    pub fn matchups(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        self.valid_matchups.iter().map(|&(o, i, _)| (o, i))
    }

    /// Iterate over valid multiway matchups.
    pub fn matchups_multiway(&self) -> impl Iterator<Item = &(Vec<usize>, f32)> {
        self.valid_matchups_multiway.iter()
    }

    /// Get the root node for a specific matchup (2-player only).
    ///
    /// Only supported for river boards (requires matchup table).
    pub fn root_for_matchup(&self, oop_combo: usize, ip_combo: usize) -> PostflopNode {
        PostflopNode {
            tree: Arc::clone(&self.tree),
            matchups: Arc::clone(self.matchups.as_ref().expect("root_for_matchup requires river board")),
            isomorphism: Arc::clone(self.isomorphism.as_ref().expect("root_for_matchup requires river board")),
            state: NodeState::Playing {
                node_idx: self.tree.root_idx,
                oop_combo,
                ip_combo,
            },
            pot: self.pot,
            stacks: [self.stacks[0], self.stacks[1]],
            num_buckets: self.num_buckets,
            num_matchups: self.valid_matchups.len(),
            valid_matchups: self.valid_matchups.clone(),
            total_weight: self.total_weight,
        }
    }

    /// Get the weight for a matchup (product of range weights).
    /// For 2-player games.
    pub fn matchup_weight(&self, oop_combo: usize, ip_combo: usize) -> f32 {
        self.ranges[1].weights[oop_combo] * self.ranges[0].weights[ip_combo]
    }

    /// Compute the total weight of all matchups.
    pub fn total_matchup_weight(&self) -> f32 {
        self.total_weight
    }

    /// Get the number of players.
    pub fn get_num_players(&self) -> usize {
        self.num_players
    }

    /// Check if this is a 2-player game.
    pub fn is_heads_up(&self) -> bool {
        self.num_players == 2
    }

    /// Get valid matchups for 2-player games (returns reference).
    pub fn valid_matchups_2p(&self) -> &[(usize, usize, f32)] {
        &self.valid_matchups
    }

    /// Get valid matchups for n-player games (returns reference).
    pub fn valid_matchups_np(&self) -> &[(Vec<usize>, f32)] {
        &self.valid_matchups_multiway
    }
}

impl Game for PostflopGame {
    type Node = PostflopNode;

    /// Get the root node.
    ///
    /// Only supported for 2-player river boards (the trait-based Game interface
    /// requires matchup tables). Use `PostflopSolver` for multi-street solving
    /// and multiway games.
    fn root(&self) -> PostflopNode {
        assert!(
            self.num_players == 2,
            "Game::root only supported for 2-player games"
        );
        // Root is a "dealing" chance node
        PostflopNode {
            tree: Arc::clone(&self.tree),
            matchups: Arc::clone(
                self.matchups
                    .as_ref()
                    .expect("Game::root requires river board"),
            ),
            isomorphism: Arc::clone(
                self.isomorphism
                    .as_ref()
                    .expect("Game::root requires river board"),
            ),
            state: NodeState::Dealing,
            pot: self.pot,
            stacks: [self.stacks[0], self.stacks[1]],
            num_buckets: self.num_buckets,
            num_matchups: self.valid_matchups.len(),
            valid_matchups: self.valid_matchups.clone(),
            total_weight: self.total_weight,
        }
    }

    fn num_players(&self) -> usize {
        self.num_players
    }

    fn num_info_sets(&self) -> usize {
        // num_players * tree_size * num_buckets
        self.num_players * self.tree.len() * self.num_buckets
    }
}

// === Helper functions for matchup enumeration ===

/// Enumerate valid matchups for 2-player games.
fn enumerate_2player_matchups(
    board: &Board,
    range_0: &Range,
    range_1: &Range,
) -> (Vec<(usize, usize, f32)>, f32) {
    let mut valid_matchups = Vec::new();
    let mut total_weight = 0.0f32;

    for idx_1 in 0..NUM_COMBOS {
        let weight_1 = range_1.weights[idx_1];
        if weight_1 == 0.0 {
            continue;
        }
        let combo_1 = Combo::from_index(idx_1);
        if combo_1.conflicts_with_mask(board.mask) {
            continue;
        }

        for idx_0 in 0..NUM_COMBOS {
            let weight_0 = range_0.weights[idx_0];
            if weight_0 == 0.0 {
                continue;
            }
            let combo_0 = Combo::from_index(idx_0);
            if combo_0.conflicts_with_mask(board.mask) {
                continue;
            }
            if combo_1.conflicts_with(&combo_0) {
                continue;
            }

            let weight = weight_1 * weight_0;
            valid_matchups.push((idx_1, idx_0, weight));
            total_weight += weight;
        }
    }

    (valid_matchups, total_weight)
}

/// Enumerate valid matchups for n-player games.
///
/// Uses recursive enumeration to find all valid combinations of hands
/// where no two players share any cards.
fn enumerate_nplayer_matchups(
    board: &Board,
    ranges: &[Range],
) -> (Vec<(Vec<usize>, f32)>, f32) {
    let num_players = ranges.len();

    // Pre-filter valid combos per player
    let valid_combos: Vec<Vec<(usize, f32)>> = ranges
        .iter()
        .map(|range| {
            (0..NUM_COMBOS)
                .filter_map(|idx| {
                    let weight = range.weights[idx];
                    if weight == 0.0 {
                        return None;
                    }
                    let combo = Combo::from_index(idx);
                    if combo.conflicts_with_mask(board.mask) {
                        return None;
                    }
                    Some((idx, weight))
                })
                .collect()
        })
        .collect();

    let mut valid_matchups = Vec::new();
    let mut total_weight = 0.0f32;

    // Recursive enumeration
    let mut current_combos = vec![0usize; num_players];
    let current_weight = 1.0f32;

    fn enumerate_recursive(
        player: usize,
        valid_combos: &[Vec<(usize, f32)>],
        current_combos: &mut Vec<usize>,
        current_weight: f32,
        used_cards: u64,
        valid_matchups: &mut Vec<(Vec<usize>, f32)>,
        total_weight: &mut f32,
    ) {
        if player == valid_combos.len() {
            // All players assigned - valid matchup
            valid_matchups.push((current_combos.clone(), current_weight));
            *total_weight += current_weight;
            return;
        }

        for &(idx, weight) in &valid_combos[player] {
            let combo = Combo::from_index(idx);
            let combo_mask = (1u64 << combo.c0) | (1u64 << combo.c1);

            // Check card conflicts with already assigned players
            if (combo_mask & used_cards) != 0 {
                continue;
            }

            current_combos[player] = idx;
            let new_weight = current_weight * weight;
            let new_used = used_cards | combo_mask;

            enumerate_recursive(
                player + 1,
                valid_combos,
                current_combos,
                new_weight,
                new_used,
                valid_matchups,
                total_weight,
            );
        }
    }

    enumerate_recursive(
        0,
        &valid_combos,
        &mut current_combos,
        current_weight,
        board.mask,
        &mut valid_matchups,
        &mut total_weight,
    );

    (valid_matchups, total_weight)
}

/// State of a postflop node.
#[derive(Clone)]
enum NodeState {
    /// Dealing phase - chance node that selects matchup
    Dealing,
    /// Playing phase - in the betting tree with specific hands
    Playing {
        node_idx: usize,
        oop_combo: usize,
        ip_combo: usize,
    },
}

/// A node in the postflop game tree.
#[derive(Clone)]
pub struct PostflopNode {
    /// Reference to the action tree.
    tree: Arc<IndexedActionTree>,
    /// Reference to the matchup table.
    matchups: Arc<MatchupTable>,
    /// Reference to isomorphism data.
    isomorphism: Arc<RiverIsomorphism>,
    /// Current node state.
    state: NodeState,
    /// Current pot size.
    pot: i32,
    /// Current stacks [OOP, IP].
    stacks: [i32; 2],
    /// Number of canonical buckets (cached).
    num_buckets: usize,
    /// Number of valid matchups.
    num_matchups: usize,
    /// Valid matchups (shared reference would be better, but Clone is required).
    valid_matchups: Vec<(usize, usize, f32)>,
    /// Total weight of all matchups (for normalization).
    total_weight: f32,
}

impl PostflopNode {
    /// Get the current tree node (only valid in Playing state).
    fn tree_node(&self) -> &IndexedNode {
        match &self.state {
            NodeState::Playing { node_idx, .. } => self.tree.get(*node_idx),
            NodeState::Dealing => panic!("tree_node called in Dealing state"),
        }
    }

    /// Get the combo for a player (only valid in Playing state).
    ///
    /// In heads-up postflop:
    /// - Player 0 = BTN/SB = IP (acts second)
    /// - Player 1 = BB = OOP (acts first)
    pub fn combo(&self, player: usize) -> Combo {
        match &self.state {
            NodeState::Playing {
                oop_combo,
                ip_combo,
                ..
            } => {
                // Player 0 is IP, Player 1 is OOP in heads-up postflop
                let idx = if player == 0 { *ip_combo } else { *oop_combo };
                Combo::from_index(idx)
            }
            NodeState::Dealing => panic!("combo called in Dealing state"),
        }
    }

    /// Get the bucket for a player's hand.
    ///
    /// Player 0 is IP, Player 1 is OOP in heads-up postflop.
    fn bucket(&self, player: usize) -> u16 {
        match &self.state {
            NodeState::Playing {
                oop_combo,
                ip_combo,
                ..
            } => {
                // Player 0 is IP, Player 1 is OOP in heads-up postflop
                let combo_idx = if player == 0 { *ip_combo } else { *oop_combo };
                self.isomorphism.bucket(combo_idx)
            }
            NodeState::Dealing => panic!("bucket called in Dealing state"),
        }
    }

    /// Get the current node index (for info set calculation).
    fn node_idx(&self) -> usize {
        match &self.state {
            NodeState::Playing { node_idx, .. } => *node_idx,
            NodeState::Dealing => panic!("node_idx called in Dealing state"),
        }
    }

    /// Get matchup info (for Playing state).
    fn matchup(&self) -> (usize, usize) {
        match &self.state {
            NodeState::Playing {
                oop_combo,
                ip_combo,
                ..
            } => (*oop_combo, *ip_combo),
            NodeState::Dealing => panic!("matchup called in Dealing state"),
        }
    }
}

impl GameNode for PostflopNode {
    fn is_terminal(&self) -> bool {
        match &self.state {
            NodeState::Dealing => false,
            NodeState::Playing { node_idx, .. } => self.tree.get(*node_idx).is_terminal(),
        }
    }

    fn is_chance(&self) -> bool {
        match &self.state {
            NodeState::Dealing => true, // Dealing is a chance node!
            NodeState::Playing { node_idx, .. } => self.tree.get(*node_idx).is_chance(),
        }
    }

    fn current_player(&self) -> usize {
        match &self.state {
            NodeState::Dealing => panic!("current_player called on chance node"),
            NodeState::Playing { node_idx, .. } => {
                match self.tree.get(*node_idx).node_type {
                    IndexedNodeType::Player { player } => player,
                    _ => panic!("current_player called on non-player node"),
                }
            }
        }
    }

    fn num_actions(&self) -> usize {
        match &self.state {
            NodeState::Dealing => self.num_matchups, // One "action" per matchup
            NodeState::Playing { node_idx, .. } => {
                let node = self.tree.get(*node_idx);
                if node.is_chance() {
                    node.children.len()
                } else {
                    node.actions.len()
                }
            }
        }
    }

    fn play(&self, action: usize) -> Self {
        match &self.state {
            NodeState::Dealing => {
                // Transition from dealing to playing with specific matchup
                let (oop_combo, ip_combo, _weight) = self.valid_matchups[action];
                PostflopNode {
                    tree: Arc::clone(&self.tree),
                    matchups: Arc::clone(&self.matchups),
                    isomorphism: Arc::clone(&self.isomorphism),
                    state: NodeState::Playing {
                        node_idx: self.tree.root_idx,
                        oop_combo,
                        ip_combo,
                    },
                    pot: self.pot,
                    stacks: self.stacks,
                    num_buckets: self.num_buckets,
                    num_matchups: self.num_matchups,
                    valid_matchups: self.valid_matchups.clone(),
                    total_weight: self.total_weight,
                }
            }
            NodeState::Playing {
                node_idx,
                oop_combo,
                ip_combo,
            } => {
                let node = self.tree.get(*node_idx);
                let child_idx = node.children[action];
                let child_node = self.tree.get(child_idx);

                // Update pot and stacks based on action
                let (new_pot, new_stacks) = if node.is_player() {
                    let player = node.player();
                    let action_taken = &node.actions[action];
                    let amount = action_taken.amount();

                    let mut stacks = self.stacks;
                    let current_bet = self.stacks[player] - stacks[player];
                    let to_put_in = (amount - current_bet).max(0);
                    let actual_bet = to_put_in.min(stacks[player]);
                    stacks[player] -= actual_bet;

                    let new_pot = self.pot + actual_bet;
                    (new_pot, stacks)
                } else {
                    (child_node.pot, self.stacks)
                };

                PostflopNode {
                    tree: Arc::clone(&self.tree),
                    matchups: Arc::clone(&self.matchups),
                    isomorphism: Arc::clone(&self.isomorphism),
                    state: NodeState::Playing {
                        node_idx: child_idx,
                        oop_combo: *oop_combo,
                        ip_combo: *ip_combo,
                    },
                    pot: new_pot,
                    stacks: new_stacks,
                    num_buckets: self.num_buckets,
                    num_matchups: self.num_matchups,
                    valid_matchups: self.valid_matchups.clone(),
                    total_weight: self.total_weight,
                }
            }
        }
    }

    fn payoff(&self, player: usize) -> f64 {
        let (oop_combo, ip_combo) = self.matchup();
        let node = self.tree_node();
        debug_assert!(node.is_terminal(), "payoff called on non-terminal node");

        match node.node_type {
            IndexedNodeType::TerminalFold { winner } => {
                // Winner takes the pot
                let pot = node.pot;
                if player == winner {
                    (pot / 2) as f64 // Win opponent's contribution
                } else {
                    -(pot / 2) as f64 // Lose our contribution
                }
            }
            IndexedNodeType::TerminalShowdown | IndexedNodeType::TerminalAllIn { .. } => {
                // Showdown - compare hands
                // result > 0 means OOP wins, result < 0 means IP wins
                let result = self.matchups.compare(oop_combo, ip_combo);
                let pot = node.pot;
                let half_pot = (pot / 2) as f64;

                if result == 0 {
                    // Tie - split pot
                    0.0
                } else if (result > 0 && player == 1) || (result < 0 && player == 0) {
                    // This player wins
                    // OOP (player 1) wins when result > 0
                    // IP (player 0) wins when result < 0
                    half_pot
                } else {
                    // This player loses
                    -half_pot
                }
            }
            _ => panic!("payoff called on non-terminal node"),
        }
    }

    fn info_set_id(&self) -> usize {
        let player = self.current_player();
        let bucket = self.bucket(player) as usize;
        let node_idx = self.node_idx();
        let tree_size = self.tree.len();

        // info_set_id = (player * tree_size + node_idx) * num_buckets + bucket
        (player * tree_size + node_idx) * self.num_buckets + bucket
    }

    fn chance_prob(&self, action: usize) -> f64 {
        match &self.state {
            NodeState::Dealing => {
                // Probability of matchup = (oop_weight * ip_weight) / total_weight
                let (_oop_combo, _ip_combo, weight) = self.valid_matchups[action];
                (weight / self.total_weight) as f64
            }
            NodeState::Playing { node_idx, .. } => {
                // For other chance nodes (e.g., future cards), use uniform
                let node = self.tree.get(*node_idx);
                1.0 / node.children.len() as f64
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::poker::board_parser::parse_board;
    use crate::poker::range_parser::parse_range;
    use crate::tree::{ActionTree, BetSizeOptions, Street, StreetConfig, TreeConfig};

    fn make_simple_tree() -> IndexedActionTree {
        let sizes = BetSizeOptions::try_from_strs("50%, 100%", "2x, a").unwrap();
        let config = TreeConfig::new(2)
            .with_stack(100)
            .with_starting_street(Street::River)
            .with_starting_pot(100)
            .with_river(StreetConfig::uniform(sizes));

        let tree = ActionTree::new(config).unwrap();
        tree.to_indexed()
    }

    #[test]
    fn test_postflop_game_creation() {
        let tree = make_simple_tree();
        let board = parse_board("KhQsJs2c3d").unwrap();
        let oop_range = parse_range("AA,KK,QQ").unwrap();
        let ip_range = parse_range("AA,KK,AKs").unwrap();

        let game = PostflopGame::new(tree, board, oop_range, ip_range, 100, 100);

        assert!(game.num_matchups() > 0);
        assert!(game.num_info_sets() > 0);
    }

    #[test]
    fn test_root_is_chance() {
        let tree = make_simple_tree();
        let board = parse_board("KhQsJs2c3d").unwrap();
        let oop_range = parse_range("AA").unwrap();
        let ip_range = parse_range("KK").unwrap();

        let game = PostflopGame::new(tree, board, oop_range, ip_range, 100, 100);
        let root = game.root();

        // Root should be a chance node (dealing)
        assert!(root.is_chance());
        assert!(!root.is_terminal());
        assert_eq!(root.num_actions(), game.num_matchups());
    }

    #[test]
    fn test_dealing_to_playing() {
        let tree = make_simple_tree();
        let board = parse_board("KhQsJs2c3d").unwrap();
        let oop_range = parse_range("AA").unwrap();
        let ip_range = parse_range("KK").unwrap();

        let game = PostflopGame::new(tree, board, oop_range, ip_range, 100, 100);
        let root = game.root();

        // After dealing, should be in playing state
        let after_deal = root.play(0);
        assert!(!after_deal.is_chance());
        assert!(!after_deal.is_terminal());
    }

    #[test]
    fn test_postflop_node_traversal() {
        let tree = make_simple_tree();
        let board = parse_board("KhQsJs2c3d").unwrap();
        let oop_range = parse_range("AA").unwrap();
        let ip_range = parse_range("KK").unwrap();

        let game = PostflopGame::new(tree, board, oop_range, ip_range, 100, 100);
        let root = game.root();

        // Deal hands
        let playing = root.play(0);
        assert!(!playing.is_terminal());
        assert!(!playing.is_chance());
        assert!(playing.num_actions() > 0);

        // Navigate to a child
        let child = playing.play(0);
        // Should be different state
        match (&playing.state, &child.state) {
            (NodeState::Playing { node_idx: idx1, .. }, NodeState::Playing { node_idx: idx2, .. }) => {
                assert_ne!(idx1, idx2);
            }
            _ => panic!("Expected Playing states"),
        }
    }

    #[test]
    fn test_info_set_encoding() {
        let tree = make_simple_tree();
        let board = parse_board("KhQsJs2c3d").unwrap();
        let oop_range = parse_range("AA,KK").unwrap();
        let ip_range = parse_range("QQ,JJ").unwrap();

        let game = PostflopGame::new(tree, board, oop_range, ip_range, 100, 100);

        // Different hands should have different info set IDs at the same node
        let mut info_sets = std::collections::HashSet::new();
        let root = game.root();

        for action in 0..root.num_actions() {
            let playing = root.play(action);
            if !playing.is_terminal() && !playing.is_chance() {
                let id = playing.info_set_id();
                info_sets.insert(id);
            }
        }

        // Should have multiple distinct info sets
        assert!(info_sets.len() >= 1);
    }

    #[test]
    fn test_showdown_payoff() {
        let tree = make_simple_tree();
        let board = parse_board("2c3d4h5s7c").unwrap(); // Rainbow low board
        let oop_range = Range::full(1.0);
        let ip_range = Range::full(1.0);

        let game = PostflopGame::new(tree, board, oop_range, ip_range, 100, 100);
        let root = game.root();

        // Test a few matchups
        for action in 0..root.num_actions().min(10) {
            let playing = root.play(action);

            // Navigate to a terminal (check-check line)
            let mut node = playing;
            while !node.is_terminal() {
                // Take first action (usually check)
                node = node.play(0);
            }

            // Verify payoff is valid
            let p0 = node.payoff(0);
            let p1 = node.payoff(1);

            // Zero-sum game
            assert!(
                (p0 + p1).abs() < 0.01,
                "Payoffs should sum to 0: {} + {} = {}",
                p0,
                p1,
                p0 + p1
            );
        }
    }
}
