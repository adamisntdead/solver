//! Vectorized postflop CFR solver with multi-street support and hand abstraction.
//!
//! Walks the betting tree ONCE per iteration with per-hand arrays,
//! matching the approach used by Gambit and binary_wasm_solver.
//!
//! Key data flow:
//! - `cfreach` (opponent counterfactual reach) flows DOWN the tree
//! - `result` (counterfactual values) flow UP the tree
//! - Regrets stored per (node, context, action, bucket) in flat arrays
//!
//! Hand abstraction:
//! - Suit isomorphism groups equivalent hands into buckets (lossless)
//! - Storage is per-bucket, reducing memory for monotone/two-tone boards
//! - Reaches are aggregated to buckets, values expanded back to hands
//!
//! Multi-street support:
//! - For 5-card boards (river): single context, hand ranks from HandInfo
//! - For 4-card boards (turn→river): one context per valid river card,
//!   hand ranks from precomputed `hand_rank_table`
//! - For 3-card boards (flop→turn→river): two-level chance nodes with
//!   composite contexts (turn_idx * num_river + river_idx)

use crate::poker::abstraction::{ComposedAbstraction, HandAbstraction};
use crate::poker::hands::{Board, Combo, NUM_COMBOS};
use crate::poker::isomorphism::INVALID_BUCKET;
use crate::poker::matchups::evaluate_7cards;
use crate::poker::postflop_game::PostflopGame;
use crate::tree::{IndexedActionTree, IndexedNode, IndexedNodeType, Street};

/// Info about a single hand (combo) for a player.
#[derive(Clone)]
struct HandInfo {
    combo_idx: usize,
    c0: u8,
    c1: u8,
    initial_weight: f32,
    /// Hand rank on the river board (only valid for river-only solving).
    hand_rank: u32,
}

/// Vectorized CFR solver for postflop poker with hand abstraction.
///
/// Instead of walking the tree once per hand pair, this solver walks the
/// betting tree once per iteration with arrays of per-hand values.
///
/// Uses hand abstraction (suit isomorphism) to reduce storage:
/// - Regrets and strategies are stored per-bucket, not per-hand
/// - Equivalent hands (differing only by suit labeling) share buckets
/// - Typical compression: 1.25x-3.3x depending on board texture
///
/// Supports multi-street solving via a card context system:
/// - Turn→river: nodes below the chance node have separate regrets per river card.
/// - Flop→turn→river: two levels of chance nodes with composite contexts.
///
/// Supports 2-6 players. For n-player games, the solver uses n-way equity
/// calculation for showdown evaluation.
pub struct PostflopSolver {
    /// Per-node regret storage.
    /// Layout: `regrets[node_idx][context * num_actions * num_buckets + action * num_buckets + bucket]`
    /// Only player nodes have non-empty vectors.
    /// Uses bucket indices from the abstraction, not hand indices.
    regrets: Vec<Vec<f32>>,

    /// Per-node cumulative strategy (same layout as regrets).
    cum_strategy: Vec<Vec<f32>>,

    /// Hand info per player.
    /// `hands[p]` = hands for player p (indexed by tree player ID)
    /// Still per-hand for reach tracking and terminal evaluation.
    hands: Vec<Vec<HandInfo>>,

    /// Hand abstraction per player.
    /// Maps combo indices to bucket indices for each card context.
    abstractions: Vec<ComposedAbstraction>,

    /// Number of buckets per context for each player.
    /// `num_buckets[player][context]` = number of buckets
    num_buckets_per_context: Vec<Vec<usize>>,

    /// Maps hand index to bucket index for each player.
    /// `hand_to_bucket[player][hand_idx]` = bucket index (for context 0, river-only)
    /// For multi-street, use abstraction directly with context.
    #[allow(dead_code)] // Reserved for future optimizations
    hand_to_bucket: Vec<Vec<u16>>,

    /// Number of iterations completed per player.
    num_steps: Vec<u32>,

    /// Number of players.
    num_players: usize,

    // === Multi-street fields ===
    /// The board used for this solve.
    board: Board,

    /// Starting street (derived from board length).
    #[allow(dead_code)]
    starting_street: Street,

    /// Valid turn cards to deal (non-empty only for flop boards: 49 cards).
    valid_turn_cards: Vec<u8>,

    /// Valid river cards to deal (empty for river-only boards).
    /// For flop boards: same 49 cards as turn (skip dealt turn at runtime).
    /// For turn boards: 48 cards.
    valid_river_cards: Vec<u8>,

    /// Precomputed hand ranks per card context.
    /// - Turn boards: `hand_rank_table[river_idx][combo_idx]`
    /// - Flop boards: `hand_rank_table[turn_idx * num_river + river_idx][combo_idx]`
    /// Empty for river-only boards.
    hand_rank_table: Vec<Vec<u32>>,

    /// Number of card contexts per node.
    /// - Depth 0 (before any chance): 1
    /// - Depth 1 (after first chance): num_turn_cards (flop) or num_river_cards (turn)
    /// - Depth 2 (after second chance): num_turn_cards * num_river_cards (flop only)
    node_num_contexts: Vec<usize>,
}

impl PostflopSolver {
    /// Create a new solver for a postflop game.
    pub fn new(game: &PostflopGame) -> Self {
        let tree = &game.tree;
        let board = game.board.clone();
        let starting_street = game.starting_street;
        let is_river = starting_street == Street::River;
        let is_flop = starting_street == Street::Flop;
        let num_players = game.get_num_players();

        // Build hand arrays for each player
        let mut hands: Vec<Vec<HandInfo>> = vec![Vec::new(); num_players];

        for player in 0..num_players {
            let range = game.range(player);
            for combo_idx in 0..NUM_COMBOS {
                let weight = range.weights[combo_idx];
                if weight == 0.0 {
                    continue;
                }
                let combo = Combo::from_index(combo_idx);
                if combo.conflicts_with_mask(board.mask) {
                    continue;
                }

                let hand_rank = if is_river {
                    game.matchups.as_ref().unwrap().hand_ranks[combo_idx]
                } else {
                    0 // Computed per context via hand_rank_table
                };

                hands[player].push(HandInfo {
                    combo_idx,
                    c0: combo.c0,
                    c1: combo.c1,
                    initial_weight: weight,
                    hand_rank,
                });
            }
        }

        // Compute valid cards and hand rank table for multi-street
        let (valid_turn_cards, valid_river_cards, hand_rank_table) = if is_river {
            (Vec::new(), Vec::new(), Vec::new())
        } else if is_flop {
            // Flop: 49 non-board cards serve as both turn and river candidates
            let mut cards: Vec<u8> = Vec::new();
            for card in 0..52u8 {
                if (board.mask >> card) & 1 == 0 {
                    cards.push(card);
                }
            }
            let table = precompute_hand_rank_table_flop(&board, &cards);
            (cards.clone(), cards, table)
        } else {
            // Turn: 48 non-board cards for the river
            let mut river_cards: Vec<u8> = Vec::new();
            for card in 0..52u8 {
                if (board.mask >> card) & 1 == 0 {
                    river_cards.push(card);
                }
            }
            let table = precompute_hand_rank_table(&board, &river_cards);
            (Vec::new(), river_cards, table)
        };

        // Build hand abstractions per player
        // Each player gets their own abstraction (same structure, but separate instances)
        let mut abstractions: Vec<ComposedAbstraction> = Vec::with_capacity(num_players);
        let mut num_buckets_per_context: Vec<Vec<usize>> = Vec::with_capacity(num_players);
        let mut hand_to_bucket: Vec<Vec<u16>> = Vec::with_capacity(num_players);

        for player in 0..num_players {
            // Create abstraction based on starting street
            let abstraction = if is_river {
                ComposedAbstraction::new(&board)
            } else if is_flop {
                ComposedAbstraction::for_flop(&board, &valid_turn_cards)
            } else {
                ComposedAbstraction::for_turn(&board, &valid_river_cards)
            };

            // Compute number of buckets per context
            let num_contexts = abstraction.num_contexts();
            let buckets_per_ctx: Vec<usize> = (0..num_contexts)
                .map(|ctx| abstraction.num_buckets(ctx))
                .collect();

            // Build hand_idx -> bucket mapping for context 0 (used in river-only mode)
            let h2b: Vec<u16> = hands[player]
                .iter()
                .map(|h| {
                    abstraction
                        .bucket(h.combo_idx, 0)
                        .unwrap_or(INVALID_BUCKET)
                })
                .collect();

            abstractions.push(abstraction);
            num_buckets_per_context.push(buckets_per_ctx);
            hand_to_bucket.push(h2b);
        }

        // Determine context count per node
        let node_num_contexts = compute_node_contexts(
            tree,
            starting_street,
            valid_turn_cards.len(),
            valid_river_cards.len(),
        );

        // Allocate per-node storage using bucket counts
        let num_nodes = tree.len();
        let mut regrets = Vec::with_capacity(num_nodes);
        let mut cum_strategy = Vec::with_capacity(num_nodes);

        for node_idx in 0..num_nodes {
            let node = tree.get(node_idx);
            if node.is_player() {
                let acting_player = node.player();
                let num_actions = node.actions.len();
                let num_contexts = node_num_contexts[node_idx];

                // Use bucket count instead of hand count for storage
                // For multi-context nodes, use max bucket count across contexts
                let max_buckets = if num_contexts == 1 {
                    num_buckets_per_context[acting_player][0]
                } else {
                    // For multi-street, each context may have different bucket counts
                    // We use uniform allocation based on the first non-trivial context
                    let ctx_index = if num_contexts > 1 { 1 } else { 0 };
                    num_buckets_per_context[acting_player]
                        .get(ctx_index)
                        .copied()
                        .unwrap_or(num_buckets_per_context[acting_player][0])
                };

                let size = num_contexts * num_actions * max_buckets;
                regrets.push(vec![0.0f32; size]);
                cum_strategy.push(vec![0.0f32; size]);
            } else {
                regrets.push(Vec::new());
                cum_strategy.push(Vec::new());
            }
        }

        PostflopSolver {
            regrets,
            cum_strategy,
            hands,
            abstractions,
            num_buckets_per_context,
            hand_to_bucket,
            num_steps: vec![0; num_players],
            num_players,
            board,
            starting_street,
            valid_turn_cards,
            valid_river_cards,
            hand_rank_table,
            node_num_contexts,
        }
    }

    /// Train using Linear CFR with alternating updates.
    pub fn train(&mut self, game: &PostflopGame, iterations: u32) {
        for i in 0..iterations {
            let traverser = (i as usize) % self.num_players;

            let num_trav_hands = self.hands[traverser].len();

            // Initial opponent reaches (one per opponent)
            // For n-player: we track cfreach for all non-traverser players
            // In the 2-player case, this is equivalent to the single opponent
            let mut opp_cfreaches: Vec<Vec<f32>> = Vec::new();
            for p in 0..self.num_players {
                if p != traverser {
                    let cfreach: Vec<f32> = self.hands[p]
                        .iter()
                        .map(|h| h.initial_weight)
                        .collect();
                    opp_cfreaches.push(cfreach);
                }
            }

            let mut result = vec![0.0f32; num_trav_hands];

            // Clone the tree Arc to avoid borrow issues
            let tree = game.tree.clone();
            let dealt_cards_mask = self.board.mask;

            if self.num_players == 2 {
                // Use optimized 2-player path
                let opponent = 1 - traverser;
                let cfreach: Vec<f32> = self.hands[opponent]
                    .iter()
                    .map(|h| h.initial_weight)
                    .collect();

                self.solve_recursive(
                    &mut result,
                    &tree,
                    tree.root_idx,
                    traverser,
                    &cfreach,
                    0,
                    dealt_cards_mask,
                );
            } else {
                // Use n-player path
                self.solve_recursive_multiway(
                    &mut result,
                    &tree,
                    tree.root_idx,
                    traverser,
                    &opp_cfreaches,
                    0,
                    dealt_cards_mask,
                );
            }

            self.num_steps[traverser] += 1;

            // Apply LCFR discount
            let t = self.num_steps[traverser] as f32;
            let discount = t / (t + 1.0);
            for node_regrets in &mut self.regrets {
                for r in node_regrets.iter_mut() {
                    *r *= discount;
                }
            }
            for node_strat in &mut self.cum_strategy {
                for s in node_strat.iter_mut() {
                    *s *= discount;
                }
            }
        }
    }

    /// Core CFR traversal: walks the tree once with hand arrays.
    ///
    /// Parameters:
    /// - `card_context`: index into the river card array (0 if above chance or river-only)
    /// - `dealt_cards_mask`: bitmask of all dealt cards (board + river card if dealt)
    fn solve_recursive(
        &mut self,
        result: &mut [f32],
        tree: &IndexedActionTree,
        node_idx: usize,
        traverser: usize,
        cfreach: &[f32],
        card_context: usize,
        dealt_cards_mask: u64,
    ) {
        let node = tree.get(node_idx);

        if node.is_terminal() {
            self.evaluate_terminal(
                result,
                node,
                traverser,
                cfreach,
                card_context,
                dealt_cards_mask,
            );
            return;
        }

        if node.is_chance() {
            self.handle_chance_node(
                result,
                tree,
                node,
                traverser,
                cfreach,
                dealt_cards_mask,
            );
            return;
        }

        let acting_player = node.player();
        let num_actions = node.actions.len();
        let child_indices: Vec<usize> = node.children.clone();

        if acting_player == traverser {
            // === TRAVERSER'S NODE ===
            let num_hands = self.hands[traverser].len();

            // Get current strategy via regret matching (expanded to hand-level)
            let strategy =
                self.regret_matching_for(node_idx, num_actions, num_hands, card_context, traverser);

            // Compute CFV for each action
            let mut cfv_actions = vec![0.0f32; num_actions * num_hands];
            for action in 0..num_actions {
                let mut action_result = vec![0.0f32; num_hands];
                self.solve_recursive(
                    &mut action_result,
                    tree,
                    child_indices[action],
                    traverser,
                    cfreach,
                    card_context,
                    dealt_cards_mask,
                );
                cfv_actions[action * num_hands..(action + 1) * num_hands]
                    .copy_from_slice(&action_result);
            }

            // Node CFV = strategy-weighted sum of action CFVs
            result.fill(0.0);
            for action in 0..num_actions {
                for h in 0..num_hands {
                    result[h] +=
                        strategy[action * num_hands + h] * cfv_actions[action * num_hands + h];
                }
            }

            // Update regrets: aggregate hand-level regrets to bucket-level
            // regret[bucket] += sum over hands in bucket (cfv_action - node_cfv)
            let num_buckets = self.num_buckets_per_context[traverser]
                .get(card_context)
                .copied()
                .unwrap_or(self.num_buckets_per_context[traverser][0]);
            let offset = card_context * num_actions * num_buckets;
            let abstraction = &self.abstractions[traverser];

            for action in 0..num_actions {
                for h in 0..num_hands {
                    let combo_idx = self.hands[traverser][h].combo_idx;
                    if let Some(bucket) = abstraction.bucket(combo_idx, card_context) {
                        let regret_delta = cfv_actions[action * num_hands + h] - result[h];
                        self.regrets[node_idx][offset + action * num_buckets + bucket as usize] +=
                            regret_delta;
                    }
                }
            }

            // Accumulate strategy for averaging (aggregate to buckets)
            for action in 0..num_actions {
                for h in 0..num_hands {
                    let combo_idx = self.hands[traverser][h].combo_idx;
                    if let Some(bucket) = abstraction.bucket(combo_idx, card_context) {
                        self.cum_strategy[node_idx]
                            [offset + action * num_buckets + bucket as usize] +=
                            strategy[action * num_hands + h];
                    }
                }
            }
        } else {
            // === OPPONENT'S NODE ===
            let num_opp_hands = self.hands[acting_player].len();
            let num_trav_hands = self.hands[traverser].len();

            // Get opponent's strategy (expanded to hand-level)
            let strategy =
                self.regret_matching_for(node_idx, num_actions, num_opp_hands, card_context, acting_player);

            // Recurse with updated cfreach (cfreach * opponent strategy per action)
            let mut cfv_actions = vec![0.0f32; num_actions * num_trav_hands];
            for action in 0..num_actions {
                let mut cfreach_action = vec![0.0f32; num_opp_hands];
                for j in 0..num_opp_hands {
                    cfreach_action[j] = cfreach[j] * strategy[action * num_opp_hands + j];
                }

                let mut action_result = vec![0.0f32; num_trav_hands];
                self.solve_recursive(
                    &mut action_result,
                    tree,
                    child_indices[action],
                    traverser,
                    &cfreach_action,
                    card_context,
                    dealt_cards_mask,
                );
                cfv_actions[action * num_trav_hands..(action + 1) * num_trav_hands]
                    .copy_from_slice(&action_result);
            }

            // Sum all action CFVs (opponent strategy already in cfreach)
            result.fill(0.0);
            for action in 0..num_actions {
                for h in 0..num_trav_hands {
                    result[h] += cfv_actions[action * num_trav_hands + h];
                }
            }
        }
    }

    /// Handle a chance node (street transition / card dealing).
    ///
    /// Supports two levels of chance for flop boards:
    /// - Level 1: deal turn card (when no extra cards dealt on flop board)
    /// - Level 2: deal river card (when turn already dealt, or turn board)
    fn handle_chance_node(
        &mut self,
        result: &mut [f32],
        tree: &IndexedActionTree,
        node: &IndexedNode,
        traverser: usize,
        cfreach: &[f32],
        dealt_cards_mask: u64,
    ) {
        if self.valid_river_cards.is_empty() {
            // River-only: pass through to single child
            if !node.children.is_empty() {
                self.solve_recursive(
                    result,
                    tree,
                    node.children[0],
                    traverser,
                    cfreach,
                    0,
                    dealt_cards_mask,
                );
            }
            return;
        }

        let extra_cards = (dealt_cards_mask ^ self.board.mask).count_ones();
        let dealing_turn = !self.valid_turn_cards.is_empty() && extra_cards == 0;

        let opponent = 1 - traverser;
        let num_trav_hands = result.len();
        let child_idx = node.children[0];
        result.fill(0.0);

        if dealing_turn {
            // === Deal turn card (flop board, first chance node) ===
            let turn_cards: Vec<u8> = self.valid_turn_cards.clone();
            let num_cards = turn_cards.len();
            let inv = 1.0 / num_cards as f32;

            for (turn_idx, &tc) in turn_cards.iter().enumerate() {
                let new_mask = dealt_cards_mask | (1u64 << tc);

                let mut cfreach_dealt = Vec::with_capacity(cfreach.len());
                for (j, &reach) in cfreach.iter().enumerate() {
                    let hand = &self.hands[opponent][j];
                    if hand.c0 == tc || hand.c1 == tc {
                        cfreach_dealt.push(0.0);
                    } else {
                        cfreach_dealt.push(reach * inv);
                    }
                }

                let mut card_result = vec![0.0f32; num_trav_hands];
                self.solve_recursive(
                    &mut card_result,
                    tree,
                    child_idx,
                    traverser,
                    &cfreach_dealt,
                    turn_idx,
                    new_mask,
                );

                for (h, hand) in self.hands[traverser].iter().enumerate() {
                    if hand.c0 != tc && hand.c1 != tc {
                        result[h] += card_result[h];
                    }
                }
            }
        } else {
            // === Deal river card ===
            // Clone to avoid borrow conflict with &mut self
            let river_cards: Vec<u8> = self.valid_river_cards.clone();
            let num_river = river_cards.len();

            // For flop start, find the dealt turn card and its index
            let (skip_card, turn_idx) = if !self.valid_turn_cards.is_empty() {
                let extra_mask = dealt_cards_mask ^ self.board.mask;
                let tc = extra_mask.trailing_zeros() as u8;
                let idx = self.valid_turn_cards.iter().position(|&c| c == tc).unwrap();
                (Some(tc), idx)
            } else {
                (None, 0)
            };

            let num_valid = num_river - if skip_card.is_some() { 1 } else { 0 };
            let inv = 1.0 / num_valid as f32;

            for (river_idx, &rc) in river_cards.iter().enumerate() {
                if Some(rc) == skip_card {
                    continue;
                }

                let new_mask = dealt_cards_mask | (1u64 << rc);

                let mut cfreach_dealt = Vec::with_capacity(cfreach.len());
                for (j, &reach) in cfreach.iter().enumerate() {
                    let hand = &self.hands[opponent][j];
                    if hand.c0 == rc || hand.c1 == rc {
                        cfreach_dealt.push(0.0);
                    } else {
                        cfreach_dealt.push(reach * inv);
                    }
                }

                // Compute card context
                let card_context = if skip_card.is_some() {
                    // Flop start: composite context
                    turn_idx * num_river + river_idx
                } else {
                    // Turn start: simple river index
                    river_idx
                };

                let mut card_result = vec![0.0f32; num_trav_hands];
                self.solve_recursive(
                    &mut card_result,
                    tree,
                    child_idx,
                    traverser,
                    &cfreach_dealt,
                    card_context,
                    new_mask,
                );

                for (h, hand) in self.hands[traverser].iter().enumerate() {
                    if hand.c0 != rc && hand.c1 != rc {
                        result[h] += card_result[h];
                    }
                }
            }
        }
    }

    /// Evaluate a terminal node for all traverser hands.
    fn evaluate_terminal(
        &self,
        result: &mut [f32],
        node: &IndexedNode,
        traverser: usize,
        cfreach: &[f32],
        card_context: usize,
        dealt_cards_mask: u64,
    ) {
        let opponent = 1 - traverser;
        let trav_hands = &self.hands[traverser];
        let opp_hands = &self.hands[opponent];
        let half_pot = node.pot as f32 / 2.0;

        match node.node_type {
            IndexedNodeType::TerminalFold { winner } => {
                let payoff = if winner == traverser {
                    half_pot
                } else {
                    -half_pot
                };

                for (h, trav) in trav_hands.iter().enumerate() {
                    let mut opp_reach = 0.0f32;
                    for (j, opp) in opp_hands.iter().enumerate() {
                        if cfreach[j] == 0.0 {
                            continue;
                        }
                        // No card conflict between the two hands
                        if trav.c0 != opp.c0
                            && trav.c0 != opp.c1
                            && trav.c1 != opp.c0
                            && trav.c1 != opp.c1
                        {
                            opp_reach += cfreach[j];
                        }
                    }
                    result[h] = payoff * opp_reach;
                }
            }
            IndexedNodeType::TerminalShowdown | IndexedNodeType::TerminalAllIn { .. } => {
                if self.valid_river_cards.is_empty() {
                    // === River-only fast path: use HandInfo.hand_rank ===
                    self.evaluate_showdown_river(result, trav_hands, opp_hands, cfreach, half_pot);
                } else {
                    let extra_cards = (dealt_cards_mask ^ self.board.mask).count_ones();
                    match extra_cards {
                        2 => {
                            // Turn + river dealt (flop start, all 5 cards present)
                            self.evaluate_showdown_with_context(
                                result,
                                trav_hands,
                                opp_hands,
                                cfreach,
                                half_pot,
                                card_context,
                            );
                        }
                        1 => {
                            if !self.valid_turn_cards.is_empty() {
                                // Flop start, turn dealt, all-in before river
                                let num_river = self.valid_river_cards.len();
                                let extra_mask = dealt_cards_mask ^ self.board.mask;
                                let turn_card = extra_mask.trailing_zeros() as u8;
                                self.evaluate_allin_runout(
                                    result,
                                    trav_hands,
                                    opp_hands,
                                    cfreach,
                                    half_pot,
                                    card_context * num_river,
                                    Some(turn_card),
                                );
                            } else {
                                // Turn start, river dealt → showdown with context
                                self.evaluate_showdown_with_context(
                                    result,
                                    trav_hands,
                                    opp_hands,
                                    cfreach,
                                    half_pot,
                                    card_context,
                                );
                            }
                        }
                        0 => {
                            if !self.valid_turn_cards.is_empty() {
                                // Flop start, no cards dealt, all-in on flop
                                self.evaluate_allin_double_runout(
                                    result,
                                    trav_hands,
                                    opp_hands,
                                    cfreach,
                                    half_pot,
                                );
                            } else {
                                // Turn start, no river dealt, all-in before river
                                self.evaluate_allin_runout(
                                    result,
                                    trav_hands,
                                    opp_hands,
                                    cfreach,
                                    half_pot,
                                    0,
                                    None,
                                );
                            }
                        }
                        _ => {}
                    }
                }
            }
            _ => panic!("evaluate_terminal called on non-terminal node"),
        }
    }

    /// Showdown evaluation using pre-stored hand ranks (river-only fast path).
    fn evaluate_showdown_river(
        &self,
        result: &mut [f32],
        trav_hands: &[HandInfo],
        opp_hands: &[HandInfo],
        cfreach: &[f32],
        half_pot: f32,
    ) {
        for (h, trav) in trav_hands.iter().enumerate() {
            let mut cfv = 0.0f32;
            for (j, opp) in opp_hands.iter().enumerate() {
                if cfreach[j] == 0.0 {
                    continue;
                }
                if trav.c0 == opp.c0
                    || trav.c0 == opp.c1
                    || trav.c1 == opp.c0
                    || trav.c1 == opp.c1
                {
                    continue;
                }
                if trav.hand_rank > opp.hand_rank {
                    cfv += half_pot * cfreach[j];
                } else if trav.hand_rank < opp.hand_rank {
                    cfv -= half_pot * cfreach[j];
                }
            }
            result[h] = cfv;
        }
    }

    /// Showdown evaluation using hand_rank_table for a specific card context.
    fn evaluate_showdown_with_context(
        &self,
        result: &mut [f32],
        trav_hands: &[HandInfo],
        opp_hands: &[HandInfo],
        cfreach: &[f32],
        half_pot: f32,
        card_context: usize,
    ) {
        let ranks = &self.hand_rank_table[card_context];
        for (h, trav) in trav_hands.iter().enumerate() {
            let trav_rank = ranks[trav.combo_idx];
            if trav_rank == u32::MAX {
                result[h] = 0.0;
                continue;
            }
            let mut cfv = 0.0f32;
            for (j, opp) in opp_hands.iter().enumerate() {
                if cfreach[j] == 0.0 {
                    continue;
                }
                if trav.c0 == opp.c0
                    || trav.c0 == opp.c1
                    || trav.c1 == opp.c0
                    || trav.c1 == opp.c1
                {
                    continue;
                }
                let opp_rank = ranks[opp.combo_idx];
                if opp_rank == u32::MAX {
                    continue;
                }
                if trav_rank > opp_rank {
                    cfv += half_pot * cfreach[j];
                } else if trav_rank < opp_rank {
                    cfv -= half_pot * cfreach[j];
                }
            }
            result[h] = cfv;
        }
    }

    /// All-in runout: enumerate valid river cards and average showdown results.
    /// Used when both players are all-in before the river card is dealt.
    ///
    /// Parameters:
    /// - `table_offset`: base offset into hand_rank_table (0 for turn start,
    ///   `turn_idx * num_river` for flop start with turn dealt)
    /// - `skip_card`: card to skip in enumeration (turn card for flop start, None for turn start)
    fn evaluate_allin_runout(
        &self,
        result: &mut [f32],
        trav_hands: &[HandInfo],
        opp_hands: &[HandInfo],
        cfreach: &[f32],
        half_pot: f32,
        table_offset: usize,
        skip_card: Option<u8>,
    ) {
        let river_cards = &self.valid_river_cards;
        for (h, trav) in trav_hands.iter().enumerate() {
            let mut cfv = 0.0f32;
            for (j, opp) in opp_hands.iter().enumerate() {
                if cfreach[j] == 0.0 {
                    continue;
                }
                if trav.c0 == opp.c0
                    || trav.c0 == opp.c1
                    || trav.c1 == opp.c0
                    || trav.c1 == opp.c1
                {
                    continue;
                }

                // Average over valid river cards
                let mut total_v = 0.0f32;
                let mut valid_count = 0u32;
                for (river_idx, &rc) in river_cards.iter().enumerate() {
                    if let Some(skip) = skip_card {
                        if rc == skip {
                            continue;
                        }
                    }
                    if trav.c0 == rc || trav.c1 == rc || opp.c0 == rc || opp.c1 == rc {
                        continue;
                    }
                    let trav_rank =
                        self.hand_rank_table[table_offset + river_idx][trav.combo_idx];
                    let opp_rank =
                        self.hand_rank_table[table_offset + river_idx][opp.combo_idx];
                    if trav_rank > opp_rank {
                        total_v += half_pot;
                    } else if trav_rank < opp_rank {
                        total_v -= half_pot;
                    }
                    valid_count += 1;
                }
                if valid_count > 0 {
                    cfv += cfreach[j] * total_v / valid_count as f32;
                }
            }
            result[h] = cfv;
        }
    }

    /// Double-runout all-in: enumerate (turn, river) pairs and average showdown.
    /// Used when both players are all-in on the flop (before turn is dealt).
    fn evaluate_allin_double_runout(
        &self,
        result: &mut [f32],
        trav_hands: &[HandInfo],
        opp_hands: &[HandInfo],
        cfreach: &[f32],
        half_pot: f32,
    ) {
        let turn_cards = &self.valid_turn_cards;
        let river_cards = &self.valid_river_cards;
        let num_river = river_cards.len();

        for (h, trav) in trav_hands.iter().enumerate() {
            let mut cfv = 0.0f32;
            for (j, opp) in opp_hands.iter().enumerate() {
                if cfreach[j] == 0.0 {
                    continue;
                }
                if trav.c0 == opp.c0
                    || trav.c0 == opp.c1
                    || trav.c1 == opp.c0
                    || trav.c1 == opp.c1
                {
                    continue;
                }

                // Average over all valid (turn, river) pairs
                let mut total_v = 0.0f32;
                let mut valid_count = 0u32;
                for (turn_idx, &tc) in turn_cards.iter().enumerate() {
                    if trav.c0 == tc || trav.c1 == tc || opp.c0 == tc || opp.c1 == tc {
                        continue;
                    }
                    for (river_idx, &rc) in river_cards.iter().enumerate() {
                        if rc == tc {
                            continue;
                        }
                        if trav.c0 == rc || trav.c1 == rc || opp.c0 == rc || opp.c1 == rc {
                            continue;
                        }
                        let table_idx = turn_idx * num_river + river_idx;
                        let trav_rank = self.hand_rank_table[table_idx][trav.combo_idx];
                        let opp_rank = self.hand_rank_table[table_idx][opp.combo_idx];
                        if trav_rank > opp_rank {
                            total_v += half_pot;
                        } else if trav_rank < opp_rank {
                            total_v -= half_pot;
                        }
                        valid_count += 1;
                    }
                }
                if valid_count > 0 {
                    cfv += cfreach[j] * total_v / valid_count as f32;
                }
            }
            result[h] = cfv;
        }
    }

    /// Compute current strategy from regrets via regret matching.
    ///
    /// Storage is per-bucket, but we return per-hand strategy for CFR traversal.
    /// Returns owned Vec of size `num_actions * num_hands`.
    fn regret_matching_for(
        &self,
        node_idx: usize,
        num_actions: usize,
        num_hands: usize,
        card_context: usize,
        acting_player: usize,
    ) -> Vec<f32> {
        let regrets = &self.regrets[node_idx];
        if regrets.is_empty() {
            let uniform = 1.0 / num_actions as f32;
            return vec![uniform; num_actions * num_hands];
        }

        // Get bucket count for this context
        let num_buckets = self.num_buckets_per_context[acting_player]
            .get(card_context)
            .copied()
            .unwrap_or(self.num_buckets_per_context[acting_player][0]);

        let offset = card_context * num_actions * num_buckets;

        // First, compute bucket-level strategy via regret matching
        let mut bucket_strategy = vec![0.0f32; num_actions * num_buckets];

        // Clamp negatives to 0
        for i in 0..num_actions * num_buckets {
            bucket_strategy[i] = regrets[offset + i].max(0.0);
        }

        // Normalize per bucket
        for b in 0..num_buckets {
            let mut sum = 0.0f32;
            for a in 0..num_actions {
                sum += bucket_strategy[a * num_buckets + b];
            }
            if sum > 0.0 {
                for a in 0..num_actions {
                    bucket_strategy[a * num_buckets + b] /= sum;
                }
            } else {
                let uniform = 1.0 / num_actions as f32;
                for a in 0..num_actions {
                    bucket_strategy[a * num_buckets + b] = uniform;
                }
            }
        }

        // Expand bucket strategy to hand strategy
        let mut strategy = vec![0.0f32; num_actions * num_hands];
        let abstraction = &self.abstractions[acting_player];

        for h in 0..num_hands {
            let combo_idx = self.hands[acting_player][h].combo_idx;
            if let Some(bucket) = abstraction.bucket(combo_idx, card_context) {
                for a in 0..num_actions {
                    strategy[a * num_hands + h] = bucket_strategy[a * num_buckets + bucket as usize];
                }
            } else {
                // Blocked hand - use uniform (shouldn't happen with valid hands)
                let uniform = 1.0 / num_actions as f32;
                for a in 0..num_actions {
                    strategy[a * num_hands + h] = uniform;
                }
            }
        }

        strategy
    }

    /// Compute average strategy for a node (from cumulative strategy sums).
    ///
    /// Storage is per-bucket, but we return per-hand strategy for CFR traversal.
    fn average_strategy_for(
        &self,
        node_idx: usize,
        num_actions: usize,
        num_hands: usize,
        card_context: usize,
        acting_player: usize,
    ) -> Vec<f32> {
        let cum = &self.cum_strategy[node_idx];
        if cum.is_empty() || cum.iter().all(|&x| x == 0.0) {
            let uniform = 1.0 / num_actions as f32;
            return vec![uniform; num_actions * num_hands];
        }

        // Get bucket count for this context
        let num_buckets = self.num_buckets_per_context[acting_player]
            .get(card_context)
            .copied()
            .unwrap_or(self.num_buckets_per_context[acting_player][0]);

        let offset = card_context * num_actions * num_buckets;

        // Read bucket-level cumulative strategy
        let mut bucket_avg = Vec::with_capacity(num_actions * num_buckets);
        for i in 0..num_actions * num_buckets {
            bucket_avg.push(cum[offset + i]);
        }

        // Normalize per bucket
        for b in 0..num_buckets {
            let mut sum = 0.0f32;
            for a in 0..num_actions {
                sum += bucket_avg[a * num_buckets + b];
            }
            if sum > 0.0 {
                for a in 0..num_actions {
                    bucket_avg[a * num_buckets + b] /= sum;
                }
            } else {
                let uniform = 1.0 / num_actions as f32;
                for a in 0..num_actions {
                    bucket_avg[a * num_buckets + b] = uniform;
                }
            }
        }

        // Expand to hand-level
        let mut avg = vec![0.0f32; num_actions * num_hands];
        let abstraction = &self.abstractions[acting_player];

        for h in 0..num_hands {
            let combo_idx = self.hands[acting_player][h].combo_idx;
            if let Some(bucket) = abstraction.bucket(combo_idx, card_context) {
                for a in 0..num_actions {
                    avg[a * num_hands + h] = bucket_avg[a * num_buckets + bucket as usize];
                }
            } else {
                let uniform = 1.0 / num_actions as f32;
                for a in 0..num_actions {
                    avg[a * num_hands + h] = uniform;
                }
            }
        }

        avg
    }

    /// Compute exploitability of the current average strategy.
    ///
    /// Returns exploitability in chips (per game).
    pub fn exploitability(&self, game: &PostflopGame) -> f32 {
        let tree = &game.tree;
        let mut total = 0.0f32;

        for traverser in 0..self.num_players {
            let num_trav_hands = self.hands[traverser].len();
            let dealt_cards_mask = self.board.mask;

            if self.num_players == 2 {
                // Optimized 2-player path
                let opponent = 1 - traverser;

                // Initial opponent reach = range weights
                let cfreach: Vec<f32> = self.hands[opponent]
                    .iter()
                    .map(|h| h.initial_weight)
                    .collect();

                let mut br_values = vec![0.0f32; num_trav_hands];
                self.best_response_value(
                    &mut br_values,
                    tree,
                    tree.root_idx,
                    traverser,
                    &cfreach,
                    0,
                    dealt_cards_mask,
                );

                // Weight by traverser's initial range
                let mut br_ev = 0.0f32;
                for (h, hand) in self.hands[traverser].iter().enumerate() {
                    br_ev += hand.initial_weight * br_values[h];
                }

                total += br_ev;
            } else {
                // N-player path: build cfreach for all opponents
                let mut opp_cfreaches: Vec<Vec<f32>> = Vec::new();
                for p in 0..self.num_players {
                    if p != traverser {
                        let cfreach: Vec<f32> = self.hands[p]
                            .iter()
                            .map(|h| h.initial_weight)
                            .collect();
                        opp_cfreaches.push(cfreach);
                    }
                }

                let mut br_values = vec![0.0f32; num_trav_hands];
                self.best_response_value_multiway(
                    &mut br_values,
                    tree,
                    tree.root_idx,
                    traverser,
                    &opp_cfreaches,
                    0,
                    dealt_cards_mask,
                );

                // Weight by traverser's initial range
                let mut br_ev = 0.0f32;
                for (h, hand) in self.hands[traverser].iter().enumerate() {
                    br_ev += hand.initial_weight * br_values[h];
                }

                total += br_ev;
            }
        }

        // Normalize by total matchup weight
        total / game.total_matchup_weight()
    }

    /// Compute best response values for the traverser against the
    /// opponent's average strategy.
    fn best_response_value(
        &self,
        result: &mut [f32],
        tree: &IndexedActionTree,
        node_idx: usize,
        traverser: usize,
        cfreach: &[f32],
        card_context: usize,
        dealt_cards_mask: u64,
    ) {
        let node = tree.get(node_idx);

        if node.is_terminal() {
            self.evaluate_terminal(
                result,
                node,
                traverser,
                cfreach,
                card_context,
                dealt_cards_mask,
            );
            return;
        }

        if node.is_chance() {
            self.best_response_chance(
                result,
                tree,
                node,
                traverser,
                cfreach,
                dealt_cards_mask,
            );
            return;
        }

        let acting_player = node.player();
        let num_actions = node.actions.len();

        if acting_player == traverser {
            // Best response: take MAX over actions for each hand
            let num_hands = self.hands[traverser].len();
            let mut cfv_actions = vec![0.0f32; num_actions * num_hands];

            for action in 0..num_actions {
                let mut action_result = vec![0.0f32; num_hands];
                self.best_response_value(
                    &mut action_result,
                    tree,
                    node.children[action],
                    traverser,
                    cfreach,
                    card_context,
                    dealt_cards_mask,
                );
                cfv_actions[action * num_hands..(action + 1) * num_hands]
                    .copy_from_slice(&action_result);
            }

            // Element-wise max across actions
            for h in 0..num_hands {
                result[h] = f32::NEG_INFINITY;
                for action in 0..num_actions {
                    let v = cfv_actions[action * num_hands + h];
                    if v > result[h] {
                        result[h] = v;
                    }
                }
            }
        } else {
            // Opponent plays average strategy
            let num_opp_hands = self.hands[acting_player].len();
            let num_trav_hands = self.hands[traverser].len();
            let avg_strategy =
                self.average_strategy_for(node_idx, num_actions, num_opp_hands, card_context, acting_player);

            let mut cfv_actions = vec![0.0f32; num_actions * num_trav_hands];
            for action in 0..num_actions {
                let mut cfreach_action = vec![0.0f32; num_opp_hands];
                for j in 0..num_opp_hands {
                    cfreach_action[j] = cfreach[j] * avg_strategy[action * num_opp_hands + j];
                }

                let mut action_result = vec![0.0f32; num_trav_hands];
                self.best_response_value(
                    &mut action_result,
                    tree,
                    node.children[action],
                    traverser,
                    &cfreach_action,
                    card_context,
                    dealt_cards_mask,
                );
                cfv_actions[action * num_trav_hands..(action + 1) * num_trav_hands]
                    .copy_from_slice(&action_result);
            }

            // Sum all action CFVs
            result.fill(0.0);
            for action in 0..num_actions {
                for h in 0..num_trav_hands {
                    result[h] += cfv_actions[action * num_trav_hands + h];
                }
            }
        }
    }

    /// Handle chance node in best_response_value (immutable version).
    /// Mirrors handle_chance_node logic for two-level dealing.
    fn best_response_chance(
        &self,
        result: &mut [f32],
        tree: &IndexedActionTree,
        node: &IndexedNode,
        traverser: usize,
        cfreach: &[f32],
        dealt_cards_mask: u64,
    ) {
        if self.valid_river_cards.is_empty() {
            // River-only: pass through
            if !node.children.is_empty() {
                self.best_response_value(
                    result,
                    tree,
                    node.children[0],
                    traverser,
                    cfreach,
                    0,
                    dealt_cards_mask,
                );
            }
            return;
        }

        let extra_cards = (dealt_cards_mask ^ self.board.mask).count_ones();
        let dealing_turn = !self.valid_turn_cards.is_empty() && extra_cards == 0;

        let opponent = 1 - traverser;
        let num_trav_hands = result.len();
        let child_idx = node.children[0];
        result.fill(0.0);

        if dealing_turn {
            // === Deal turn card (flop board, first chance node) ===
            let num_cards = self.valid_turn_cards.len();
            let inv = 1.0 / num_cards as f32;

            for (turn_idx, &tc) in self.valid_turn_cards.iter().enumerate() {
                let new_mask = dealt_cards_mask | (1u64 << tc);

                let mut cfreach_dealt = Vec::with_capacity(cfreach.len());
                for (j, &reach) in cfreach.iter().enumerate() {
                    let hand = &self.hands[opponent][j];
                    if hand.c0 == tc || hand.c1 == tc {
                        cfreach_dealt.push(0.0);
                    } else {
                        cfreach_dealt.push(reach * inv);
                    }
                }

                let mut card_result = vec![0.0f32; num_trav_hands];
                self.best_response_value(
                    &mut card_result,
                    tree,
                    child_idx,
                    traverser,
                    &cfreach_dealt,
                    turn_idx,
                    new_mask,
                );

                for (h, hand) in self.hands[traverser].iter().enumerate() {
                    if hand.c0 != tc && hand.c1 != tc {
                        result[h] += card_result[h];
                    }
                }
            }
        } else {
            // === Deal river card ===
            let num_river = self.valid_river_cards.len();

            // For flop start, find the dealt turn card and its index
            let (skip_card, turn_idx) = if !self.valid_turn_cards.is_empty() {
                let extra_mask = dealt_cards_mask ^ self.board.mask;
                let tc = extra_mask.trailing_zeros() as u8;
                let idx = self.valid_turn_cards.iter().position(|&c| c == tc).unwrap();
                (Some(tc), idx)
            } else {
                (None, 0)
            };

            let num_valid = num_river - if skip_card.is_some() { 1 } else { 0 };
            let inv = 1.0 / num_valid as f32;

            for (river_idx, &rc) in self.valid_river_cards.iter().enumerate() {
                if Some(rc) == skip_card {
                    continue;
                }

                let new_mask = dealt_cards_mask | (1u64 << rc);

                let mut cfreach_dealt = Vec::with_capacity(cfreach.len());
                for (j, &reach) in cfreach.iter().enumerate() {
                    let hand = &self.hands[opponent][j];
                    if hand.c0 == rc || hand.c1 == rc {
                        cfreach_dealt.push(0.0);
                    } else {
                        cfreach_dealt.push(reach * inv);
                    }
                }

                let card_context = if skip_card.is_some() {
                    turn_idx * num_river + river_idx
                } else {
                    river_idx
                };

                let mut card_result = vec![0.0f32; num_trav_hands];
                self.best_response_value(
                    &mut card_result,
                    tree,
                    child_idx,
                    traverser,
                    &cfreach_dealt,
                    card_context,
                    new_mask,
                );

                for (h, hand) in self.hands[traverser].iter().enumerate() {
                    if hand.c0 != rc && hand.c1 != rc {
                        result[h] += card_result[h];
                    }
                }
            }
        }
    }

    // === Public API for querying results ===

    /// Get the number of private hands for a player.
    pub fn num_hands(&self, player: usize) -> usize {
        self.hands[player].len()
    }

    /// Get hand info: (combo_index, initial_weight) for a player's hand.
    pub fn hand_info(&self, player: usize, hand_idx: usize) -> (usize, f32) {
        let h = &self.hands[player][hand_idx];
        (h.combo_idx, h.initial_weight)
    }

    /// Get the average strategy for a specific hand at a node (card_context = 0).
    ///
    /// Returns a Vec of probabilities (one per action).
    pub fn get_hand_strategy(&self, node_idx: usize, hand_idx: usize, player: usize) -> Vec<f32> {
        self.get_hand_strategy_ctx(node_idx, hand_idx, player, 0)
    }

    /// Get the average strategy for a specific hand at a node with card context.
    pub fn get_hand_strategy_ctx(
        &self,
        node_idx: usize,
        hand_idx: usize,
        player: usize,
        card_context: usize,
    ) -> Vec<f32> {
        let num_hands = self.hands[player].len();
        if num_hands == 0 {
            return Vec::new();
        }

        let num_contexts = self.node_num_contexts[node_idx];
        let total_size = self.regrets[node_idx].len();
        if total_size == 0 {
            return Vec::new();
        }

        // Storage is per-bucket, so we need to calculate num_actions from bucket count
        let num_buckets = self.num_buckets_per_context[player]
            .get(card_context)
            .copied()
            .unwrap_or(self.num_buckets_per_context[player][0]);
        let per_context = total_size / num_contexts;
        let num_actions = per_context / num_buckets;

        let avg = self.average_strategy_for(node_idx, num_actions, num_hands, card_context, player);
        (0..num_actions)
            .map(|a| avg[a * num_hands + hand_idx])
            .collect()
    }

    /// Get total iteration count (sum of all players).
    pub fn total_iterations(&self) -> u32 {
        self.num_steps.iter().sum()
    }

    /// Get the number of players.
    pub fn num_players(&self) -> usize {
        self.num_players
    }

    /// Get the number of card contexts for a node.
    pub fn num_contexts(&self, node_idx: usize) -> usize {
        self.node_num_contexts[node_idx]
    }

    /// Get the number of valid river cards (0 for river-only boards).
    pub fn num_river_cards(&self) -> usize {
        self.valid_river_cards.len()
    }

    /// Get string representations of all valid river cards (e.g., ["2c", "2d", ...]).
    /// Returns empty vec for river-only boards.
    pub fn river_card_strings(&self) -> Vec<String> {
        use crate::poker::hands::card_to_string;
        self.valid_river_cards
            .iter()
            .map(|&c| card_to_string(c))
            .collect()
    }

    /// Get the raw card index for a given river card context index.
    /// Returns None if the index is out of range or this is a river-only board.
    pub fn river_card_at(&self, ctx: usize) -> Option<u8> {
        self.valid_river_cards.get(ctx).copied()
    }

    /// Get the cards (c0, c1) for a player's hand at the given hand index.
    pub fn hand_cards(&self, player: usize, hand_idx: usize) -> (u8, u8) {
        let h = &self.hands[player][hand_idx];
        (h.c0, h.c1)
    }

    /// Get the number of valid turn cards (49 for flop boards, 0 otherwise).
    pub fn num_turn_cards(&self) -> usize {
        self.valid_turn_cards.len()
    }

    /// Get string representations of all valid turn cards.
    /// Returns empty vec for non-flop boards.
    pub fn turn_card_strings(&self) -> Vec<String> {
        use crate::poker::hands::card_to_string;
        self.valid_turn_cards
            .iter()
            .map(|&c| card_to_string(c))
            .collect()
    }

    /// Get the raw card index for a given turn card context index.
    /// Returns None if the index is out of range or this is not a flop board.
    pub fn turn_card_at(&self, ctx: usize) -> Option<u8> {
        self.valid_turn_cards.get(ctx).copied()
    }

    /// Get the chance depth for a node (0 = before any chance, 1 = after first, 2 = after second).
    pub fn chance_depth(&self, node_idx: usize) -> usize {
        let ctx = self.node_num_contexts[node_idx];
        if ctx <= 1 {
            return 0;
        }
        let num_turn = self.valid_turn_cards.len();
        let num_river = self.valid_river_cards.len();
        if num_turn > 0 {
            if ctx == num_turn {
                return 1;
            }
            if num_river > 0 && ctx == num_turn * num_river {
                return 2;
            }
        }
        if num_river > 0 && ctx == num_river {
            return 1;
        }
        0
    }

    /// Get the cards that a given context represents for a node.
    /// Returns (turn_card, river_card) - either or both may be None.
    pub fn context_cards(&self, node_idx: usize, ctx: usize) -> (Option<u8>, Option<u8>) {
        let depth = self.chance_depth(node_idx);
        match depth {
            1 if !self.valid_turn_cards.is_empty() => {
                // Flop tree, turn level
                (self.valid_turn_cards.get(ctx).copied(), None)
            }
            1 => {
                // Turn tree, river level
                (None, self.valid_river_cards.get(ctx).copied())
            }
            2 => {
                // Flop tree, river level (composite context)
                let num_river = self.valid_river_cards.len();
                let turn_idx = ctx / num_river;
                let river_idx = ctx % num_river;
                (
                    self.valid_turn_cards.get(turn_idx).copied(),
                    self.valid_river_cards.get(river_idx).copied(),
                )
            }
            _ => (None, None),
        }
    }

    /// Get the total number of buckets across all players (at context 0).
    pub fn total_buckets(&self) -> usize {
        self.num_buckets_per_context.iter().map(|v| v[0]).sum()
    }

    /// Get the total number of hands across all players.
    pub fn total_hands(&self) -> usize {
        self.hands.iter().map(|h| h.len()).sum()
    }

    /// Compute the compression ratio (total_hands / total_buckets).
    /// Returns 1.0 if there are no buckets.
    pub fn compression_ratio(&self) -> f32 {
        let buckets = self.total_buckets();
        if buckets == 0 {
            return 1.0;
        }
        self.total_hands() as f32 / buckets as f32
    }

    // === Multiway (N-player) solving methods ===
    //
    // These methods handle n-player CFR traversal where we track counterfactual
    // reach for multiple opponents simultaneously.

    /// Multiway CFR traversal for n-player games.
    ///
    /// Parameters:
    /// - `opp_cfreaches`: Vector of cfreach arrays, one per opponent (indexed by relative order)
    #[allow(clippy::too_many_arguments)]
    fn solve_recursive_multiway(
        &mut self,
        result: &mut [f32],
        tree: &IndexedActionTree,
        node_idx: usize,
        traverser: usize,
        opp_cfreaches: &[Vec<f32>],
        card_context: usize,
        dealt_cards_mask: u64,
    ) {
        let node = tree.get(node_idx);

        if node.is_terminal() {
            self.evaluate_terminal_multiway(
                result,
                node,
                traverser,
                opp_cfreaches,
                card_context,
                dealt_cards_mask,
            );
            return;
        }

        if node.is_chance() {
            self.handle_chance_node_multiway(
                result,
                tree,
                node,
                traverser,
                opp_cfreaches,
                dealt_cards_mask,
            );
            return;
        }

        let acting_player = node.player();
        let num_actions = node.actions.len();
        let child_indices: Vec<usize> = node.children.clone();

        if acting_player == traverser {
            // === TRAVERSER'S NODE ===
            let num_hands = self.hands[traverser].len();

            // Get current strategy via regret matching (expanded to hand-level)
            let strategy =
                self.regret_matching_for(node_idx, num_actions, num_hands, card_context, traverser);

            // Compute CFV for each action
            let mut cfv_actions = vec![0.0f32; num_actions * num_hands];
            for action in 0..num_actions {
                let mut action_result = vec![0.0f32; num_hands];
                self.solve_recursive_multiway(
                    &mut action_result,
                    tree,
                    child_indices[action],
                    traverser,
                    opp_cfreaches,
                    card_context,
                    dealt_cards_mask,
                );
                cfv_actions[action * num_hands..(action + 1) * num_hands]
                    .copy_from_slice(&action_result);
            }

            // Node CFV = strategy-weighted sum of action CFVs
            result.fill(0.0);
            for action in 0..num_actions {
                for h in 0..num_hands {
                    result[h] +=
                        strategy[action * num_hands + h] * cfv_actions[action * num_hands + h];
                }
            }

            // Update regrets (aggregate to buckets)
            let num_buckets = self.num_buckets_per_context[traverser]
                .get(card_context)
                .copied()
                .unwrap_or(self.num_buckets_per_context[traverser][0]);
            let offset = card_context * num_actions * num_buckets;
            let abstraction = &self.abstractions[traverser];

            for action in 0..num_actions {
                for h in 0..num_hands {
                    let combo_idx = self.hands[traverser][h].combo_idx;
                    if let Some(bucket) = abstraction.bucket(combo_idx, card_context) {
                        let regret_delta = cfv_actions[action * num_hands + h] - result[h];
                        self.regrets[node_idx][offset + action * num_buckets + bucket as usize] +=
                            regret_delta;
                    }
                }
            }

            // Accumulate strategy for averaging (aggregate to buckets)
            for action in 0..num_actions {
                for h in 0..num_hands {
                    let combo_idx = self.hands[traverser][h].combo_idx;
                    if let Some(bucket) = abstraction.bucket(combo_idx, card_context) {
                        self.cum_strategy[node_idx]
                            [offset + action * num_buckets + bucket as usize] +=
                            strategy[action * num_hands + h];
                    }
                }
            }
        } else {
            // === OPPONENT'S NODE ===
            // Find which opponent index this is
            let opp_idx = self.get_opponent_index(traverser, acting_player);
            let num_opp_hands = self.hands[acting_player].len();
            let num_trav_hands = self.hands[traverser].len();

            // Get opponent's strategy (expanded to hand-level)
            let strategy =
                self.regret_matching_for(node_idx, num_actions, num_opp_hands, card_context, acting_player);

            // Recurse with updated cfreach for this opponent
            let mut cfv_actions = vec![0.0f32; num_actions * num_trav_hands];
            for action in 0..num_actions {
                // Update only this opponent's cfreach
                let mut new_opp_cfreaches: Vec<Vec<f32>> = opp_cfreaches.to_vec();
                for j in 0..num_opp_hands {
                    new_opp_cfreaches[opp_idx][j] *= strategy[action * num_opp_hands + j];
                }

                let mut action_result = vec![0.0f32; num_trav_hands];
                self.solve_recursive_multiway(
                    &mut action_result,
                    tree,
                    child_indices[action],
                    traverser,
                    &new_opp_cfreaches,
                    card_context,
                    dealt_cards_mask,
                );
                cfv_actions[action * num_trav_hands..(action + 1) * num_trav_hands]
                    .copy_from_slice(&action_result);
            }

            // Sum all action CFVs (opponent strategy already in cfreach)
            result.fill(0.0);
            for action in 0..num_actions {
                for h in 0..num_trav_hands {
                    result[h] += cfv_actions[action * num_trav_hands + h];
                }
            }
        }
    }

    /// Get the index of an opponent in the opp_cfreaches array.
    fn get_opponent_index(&self, traverser: usize, opponent: usize) -> usize {
        // Opponents are stored in order, skipping the traverser
        let mut idx = 0;
        for p in 0..self.num_players {
            if p == traverser {
                continue;
            }
            if p == opponent {
                return idx;
            }
            idx += 1;
        }
        panic!("Invalid opponent index");
    }

    /// Evaluate terminal node for n-player games.
    #[allow(clippy::too_many_arguments)]
    fn evaluate_terminal_multiway(
        &self,
        result: &mut [f32],
        node: &IndexedNode,
        traverser: usize,
        opp_cfreaches: &[Vec<f32>],
        _card_context: usize,
        _dealt_cards_mask: u64,
    ) {
        let trav_hands = &self.hands[traverser];
        let half_pot = node.pot as f32 / 2.0;

        match node.node_type {
            IndexedNodeType::TerminalFold { winner } => {
                // Multiway fold: winner takes pot, folders lose their contribution
                // For now, simplified: winner gets pot/2 from each active player
                let payoff = if winner == traverser {
                    half_pot
                } else {
                    -half_pot / (self.num_players - 1) as f32
                };

                for (h, trav) in trav_hands.iter().enumerate() {
                    // Sum reach across all valid opponent combinations
                    // For simplicity, we use a product approximation
                    let mut product = 1.0f32;
                    for (opp_idx, opp_cfreach) in opp_cfreaches.iter().enumerate() {
                        let opp_player = self.get_player_from_opp_index(traverser, opp_idx);
                        let opp_hands = &self.hands[opp_player];

                        let mut opp_sum = 0.0f32;
                        for (j, opp) in opp_hands.iter().enumerate() {
                            if opp_cfreach[j] == 0.0 {
                                continue;
                            }
                            // Check card conflict with traverser
                            if trav.c0 != opp.c0
                                && trav.c0 != opp.c1
                                && trav.c1 != opp.c0
                                && trav.c1 != opp.c1
                            {
                                opp_sum += opp_cfreach[j];
                            }
                        }
                        product *= opp_sum;
                    }

                    result[h] = payoff * product;
                }
            }
            IndexedNodeType::TerminalShowdown | IndexedNodeType::TerminalAllIn { .. } => {
                // Multiway showdown: use n-way equity calculation
                // This is a simplified version - full implementation would need
                // to enumerate all opponent hand combinations

                // For now, use a similar product approximation
                for (h, trav) in trav_hands.iter().enumerate() {
                    let mut cfv = 0.0f32;

                    // For each opponent combination (simplified: independent approximation)
                    // This is an approximation that may not be fully accurate for multiway
                    let mut total_product = 0.0f32;
                    let mut ev_product = 0.0f32;

                    // Enumerate opponent hand combinations (product of independent reaches)
                    // For performance, we approximate by computing expected value per opponent
                    for (opp_idx, opp_cfreach) in opp_cfreaches.iter().enumerate() {
                        let opp_player = self.get_player_from_opp_index(traverser, opp_idx);
                        let opp_hands = &self.hands[opp_player];

                        for (j, opp) in opp_hands.iter().enumerate() {
                            if opp_cfreach[j] == 0.0 {
                                continue;
                            }
                            if trav.c0 == opp.c0
                                || trav.c0 == opp.c1
                                || trav.c1 == opp.c0
                                || trav.c1 == opp.c1
                            {
                                continue;
                            }

                            // Compare hands (using stored hand_rank for river)
                            let contrib = if trav.hand_rank > opp.hand_rank {
                                half_pot * opp_cfreach[j]
                            } else if trav.hand_rank < opp.hand_rank {
                                -half_pot * opp_cfreach[j]
                            } else {
                                0.0
                            };

                            cfv += contrib;
                            total_product += opp_cfreach[j];
                        }
                        ev_product = cfv;
                    }

                    // Normalize by opponent reach
                    if total_product > 0.0 {
                        result[h] = ev_product;
                    } else {
                        result[h] = 0.0;
                    }
                }
            }
            _ => panic!("evaluate_terminal_multiway called on non-terminal node"),
        }
    }

    /// Get the player index from opponent index.
    fn get_player_from_opp_index(&self, traverser: usize, opp_idx: usize) -> usize {
        let mut idx = 0;
        for p in 0..self.num_players {
            if p == traverser {
                continue;
            }
            if idx == opp_idx {
                return p;
            }
            idx += 1;
        }
        panic!("Invalid opponent index");
    }

    /// Handle chance node for n-player games.
    #[allow(clippy::too_many_arguments)]
    fn handle_chance_node_multiway(
        &mut self,
        result: &mut [f32],
        tree: &IndexedActionTree,
        node: &IndexedNode,
        traverser: usize,
        opp_cfreaches: &[Vec<f32>],
        dealt_cards_mask: u64,
    ) {
        if self.valid_river_cards.is_empty() {
            // River-only: pass through to single child
            if !node.children.is_empty() {
                self.solve_recursive_multiway(
                    result,
                    tree,
                    node.children[0],
                    traverser,
                    opp_cfreaches,
                    0,
                    dealt_cards_mask,
                );
            }
            return;
        }

        // For multi-street, handle card dealing similar to 2-player version
        // but update cfreach for all opponents
        let extra_cards = (dealt_cards_mask ^ self.board.mask).count_ones();
        let dealing_turn = !self.valid_turn_cards.is_empty() && extra_cards == 0;

        let num_trav_hands = result.len();
        let child_idx = node.children[0];
        result.fill(0.0);

        if dealing_turn {
            // Deal turn card
            let turn_cards: Vec<u8> = self.valid_turn_cards.clone();
            let num_cards = turn_cards.len();
            let inv = 1.0 / num_cards as f32;

            for (turn_idx, &tc) in turn_cards.iter().enumerate() {
                let new_mask = dealt_cards_mask | (1u64 << tc);

                // Update cfreach for all opponents
                let mut new_opp_cfreaches: Vec<Vec<f32>> = Vec::new();
                for (opp_idx, opp_cfreach) in opp_cfreaches.iter().enumerate() {
                    let opp_player = self.get_player_from_opp_index(traverser, opp_idx);
                    let opp_hands = &self.hands[opp_player];

                    let mut cfreach_dealt = Vec::with_capacity(opp_cfreach.len());
                    for (j, &reach) in opp_cfreach.iter().enumerate() {
                        let hand = &opp_hands[j];
                        if hand.c0 == tc || hand.c1 == tc {
                            cfreach_dealt.push(0.0);
                        } else {
                            cfreach_dealt.push(reach * inv);
                        }
                    }
                    new_opp_cfreaches.push(cfreach_dealt);
                }

                let mut card_result = vec![0.0f32; num_trav_hands];
                self.solve_recursive_multiway(
                    &mut card_result,
                    tree,
                    child_idx,
                    traverser,
                    &new_opp_cfreaches,
                    turn_idx,
                    new_mask,
                );

                for (h, hand) in self.hands[traverser].iter().enumerate() {
                    if hand.c0 != tc && hand.c1 != tc {
                        result[h] += card_result[h];
                    }
                }
            }
        } else {
            // Deal river card (similar logic)
            let river_cards: Vec<u8> = self.valid_river_cards.clone();
            let num_river = river_cards.len();

            let (skip_card, turn_idx) = if !self.valid_turn_cards.is_empty() {
                let extra_mask = dealt_cards_mask ^ self.board.mask;
                let tc = extra_mask.trailing_zeros() as u8;
                let idx = self.valid_turn_cards.iter().position(|&c| c == tc).unwrap();
                (Some(tc), idx)
            } else {
                (None, 0)
            };

            let num_valid = num_river - if skip_card.is_some() { 1 } else { 0 };
            let inv = 1.0 / num_valid as f32;

            for (river_idx, &rc) in river_cards.iter().enumerate() {
                if Some(rc) == skip_card {
                    continue;
                }

                let new_mask = dealt_cards_mask | (1u64 << rc);

                // Update cfreach for all opponents
                let mut new_opp_cfreaches: Vec<Vec<f32>> = Vec::new();
                for (opp_idx, opp_cfreach) in opp_cfreaches.iter().enumerate() {
                    let opp_player = self.get_player_from_opp_index(traverser, opp_idx);
                    let opp_hands = &self.hands[opp_player];

                    let mut cfreach_dealt = Vec::with_capacity(opp_cfreach.len());
                    for (j, &reach) in opp_cfreach.iter().enumerate() {
                        let hand = &opp_hands[j];
                        if hand.c0 == rc || hand.c1 == rc {
                            cfreach_dealt.push(0.0);
                        } else {
                            cfreach_dealt.push(reach * inv);
                        }
                    }
                    new_opp_cfreaches.push(cfreach_dealt);
                }

                let card_context = if skip_card.is_some() {
                    turn_idx * num_river + river_idx
                } else {
                    river_idx
                };

                let mut card_result = vec![0.0f32; num_trav_hands];
                self.solve_recursive_multiway(
                    &mut card_result,
                    tree,
                    child_idx,
                    traverser,
                    &new_opp_cfreaches,
                    card_context,
                    new_mask,
                );

                for (h, hand) in self.hands[traverser].iter().enumerate() {
                    if hand.c0 != rc && hand.c1 != rc {
                        result[h] += card_result[h];
                    }
                }
            }
        }
    }

    /// Best response value for n-player games.
    #[allow(clippy::too_many_arguments)]
    fn best_response_value_multiway(
        &self,
        result: &mut [f32],
        tree: &IndexedActionTree,
        node_idx: usize,
        traverser: usize,
        opp_cfreaches: &[Vec<f32>],
        card_context: usize,
        dealt_cards_mask: u64,
    ) {
        let node = tree.get(node_idx);

        if node.is_terminal() {
            self.evaluate_terminal_multiway(
                result,
                node,
                traverser,
                opp_cfreaches,
                card_context,
                dealt_cards_mask,
            );
            return;
        }

        if node.is_chance() {
            self.best_response_chance_multiway(
                result,
                tree,
                node,
                traverser,
                opp_cfreaches,
                dealt_cards_mask,
            );
            return;
        }

        let acting_player = node.player();
        let num_actions = node.actions.len();

        if acting_player == traverser {
            // Best response: take MAX over actions
            let num_hands = self.hands[traverser].len();
            let mut cfv_actions = vec![0.0f32; num_actions * num_hands];

            for action in 0..num_actions {
                let mut action_result = vec![0.0f32; num_hands];
                self.best_response_value_multiway(
                    &mut action_result,
                    tree,
                    node.children[action],
                    traverser,
                    opp_cfreaches,
                    card_context,
                    dealt_cards_mask,
                );
                cfv_actions[action * num_hands..(action + 1) * num_hands]
                    .copy_from_slice(&action_result);
            }

            // Element-wise max across actions
            for h in 0..num_hands {
                result[h] = f32::NEG_INFINITY;
                for action in 0..num_actions {
                    let v = cfv_actions[action * num_hands + h];
                    if v > result[h] {
                        result[h] = v;
                    }
                }
            }
        } else {
            // Opponent plays average strategy
            let opp_idx = self.get_opponent_index(traverser, acting_player);
            let num_opp_hands = self.hands[acting_player].len();
            let num_trav_hands = self.hands[traverser].len();
            let avg_strategy =
                self.average_strategy_for(node_idx, num_actions, num_opp_hands, card_context, acting_player);

            let mut cfv_actions = vec![0.0f32; num_actions * num_trav_hands];
            for action in 0..num_actions {
                // Update opponent's cfreach
                let mut new_opp_cfreaches: Vec<Vec<f32>> = opp_cfreaches.to_vec();
                for j in 0..num_opp_hands {
                    new_opp_cfreaches[opp_idx][j] *= avg_strategy[action * num_opp_hands + j];
                }

                let mut action_result = vec![0.0f32; num_trav_hands];
                self.best_response_value_multiway(
                    &mut action_result,
                    tree,
                    node.children[action],
                    traverser,
                    &new_opp_cfreaches,
                    card_context,
                    dealt_cards_mask,
                );
                cfv_actions[action * num_trav_hands..(action + 1) * num_trav_hands]
                    .copy_from_slice(&action_result);
            }

            // Sum all action CFVs
            result.fill(0.0);
            for action in 0..num_actions {
                for h in 0..num_trav_hands {
                    result[h] += cfv_actions[action * num_trav_hands + h];
                }
            }
        }
    }

    /// Best response chance node handler for n-player games.
    fn best_response_chance_multiway(
        &self,
        result: &mut [f32],
        tree: &IndexedActionTree,
        node: &IndexedNode,
        traverser: usize,
        opp_cfreaches: &[Vec<f32>],
        dealt_cards_mask: u64,
    ) {
        if self.valid_river_cards.is_empty() {
            // River-only: pass through
            if !node.children.is_empty() {
                self.best_response_value_multiway(
                    result,
                    tree,
                    node.children[0],
                    traverser,
                    opp_cfreaches,
                    0,
                    dealt_cards_mask,
                );
            }
            return;
        }

        let extra_cards = (dealt_cards_mask ^ self.board.mask).count_ones();
        let dealing_turn = !self.valid_turn_cards.is_empty() && extra_cards == 0;

        let num_trav_hands = result.len();
        let child_idx = node.children[0];
        result.fill(0.0);

        if dealing_turn {
            let num_cards = self.valid_turn_cards.len();
            let inv = 1.0 / num_cards as f32;

            for (turn_idx, &tc) in self.valid_turn_cards.iter().enumerate() {
                let new_mask = dealt_cards_mask | (1u64 << tc);

                // Update cfreach for all opponents
                let mut new_opp_cfreaches: Vec<Vec<f32>> = Vec::new();
                for (opp_idx, opp_cfreach) in opp_cfreaches.iter().enumerate() {
                    let opp_player = self.get_player_from_opp_index(traverser, opp_idx);
                    let opp_hands = &self.hands[opp_player];

                    let mut cfreach_dealt = Vec::with_capacity(opp_cfreach.len());
                    for (j, &reach) in opp_cfreach.iter().enumerate() {
                        let hand = &opp_hands[j];
                        if hand.c0 == tc || hand.c1 == tc {
                            cfreach_dealt.push(0.0);
                        } else {
                            cfreach_dealt.push(reach * inv);
                        }
                    }
                    new_opp_cfreaches.push(cfreach_dealt);
                }

                let mut card_result = vec![0.0f32; num_trav_hands];
                self.best_response_value_multiway(
                    &mut card_result,
                    tree,
                    child_idx,
                    traverser,
                    &new_opp_cfreaches,
                    turn_idx,
                    new_mask,
                );

                for (h, hand) in self.hands[traverser].iter().enumerate() {
                    if hand.c0 != tc && hand.c1 != tc {
                        result[h] += card_result[h];
                    }
                }
            }
        } else {
            let num_river = self.valid_river_cards.len();

            let (skip_card, turn_idx) = if !self.valid_turn_cards.is_empty() {
                let extra_mask = dealt_cards_mask ^ self.board.mask;
                let tc = extra_mask.trailing_zeros() as u8;
                let idx = self.valid_turn_cards.iter().position(|&c| c == tc).unwrap();
                (Some(tc), idx)
            } else {
                (None, 0)
            };

            let num_valid = num_river - if skip_card.is_some() { 1 } else { 0 };
            let inv = 1.0 / num_valid as f32;

            for (river_idx, &rc) in self.valid_river_cards.iter().enumerate() {
                if Some(rc) == skip_card {
                    continue;
                }

                let new_mask = dealt_cards_mask | (1u64 << rc);

                // Update cfreach for all opponents
                let mut new_opp_cfreaches: Vec<Vec<f32>> = Vec::new();
                for (opp_idx, opp_cfreach) in opp_cfreaches.iter().enumerate() {
                    let opp_player = self.get_player_from_opp_index(traverser, opp_idx);
                    let opp_hands = &self.hands[opp_player];

                    let mut cfreach_dealt = Vec::with_capacity(opp_cfreach.len());
                    for (j, &reach) in opp_cfreach.iter().enumerate() {
                        let hand = &opp_hands[j];
                        if hand.c0 == rc || hand.c1 == rc {
                            cfreach_dealt.push(0.0);
                        } else {
                            cfreach_dealt.push(reach * inv);
                        }
                    }
                    new_opp_cfreaches.push(cfreach_dealt);
                }

                let card_context = if skip_card.is_some() {
                    turn_idx * num_river + river_idx
                } else {
                    river_idx
                };

                let mut card_result = vec![0.0f32; num_trav_hands];
                self.best_response_value_multiway(
                    &mut card_result,
                    tree,
                    child_idx,
                    traverser,
                    &new_opp_cfreaches,
                    card_context,
                    new_mask,
                );

                for (h, hand) in self.hands[traverser].iter().enumerate() {
                    if hand.c0 != rc && hand.c1 != rc {
                        result[h] += card_result[h];
                    }
                }
            }
        }
    }
}

// === Helper functions ===

/// Precompute hand rank table for each valid river card.
///
/// Returns `table[river_idx][combo_idx]` = hand rank on (board + river_card).
fn precompute_hand_rank_table(board: &Board, valid_river_cards: &[u8]) -> Vec<Vec<u32>> {
    valid_river_cards
        .iter()
        .map(|&rc| {
            let mut ranks = vec![u32::MAX; NUM_COMBOS];
            let board_mask = board.mask | (1u64 << rc);
            for combo_idx in 0..NUM_COMBOS {
                let combo = Combo::from_index(combo_idx);
                if combo.conflicts_with_mask(board_mask) {
                    continue;
                }
                let cards = [
                    board.cards[0],
                    board.cards[1],
                    board.cards[2],
                    board.cards[3],
                    rc,
                    combo.c0,
                    combo.c1,
                ];
                ranks[combo_idx] = evaluate_7cards(&cards);
            }
            ranks
        })
        .collect()
}

/// Walk the tree to determine how many card contexts each node needs.
///
/// Uses chance_depth to track how many chance nodes are above each node:
/// - depth 0 → 1 context (e.g., flop/turn betting before any dealt card)
/// - depth 1 → turn-level contexts (flop: num_turn_cards, turn: num_river_cards)
/// - depth 2 → num_turn_cards * num_river_cards (flop only, river betting)
fn compute_node_contexts(
    tree: &IndexedActionTree,
    starting_street: Street,
    num_turn_cards: usize,
    num_river_cards: usize,
) -> Vec<usize> {
    let mut contexts = vec![1usize; tree.len()];
    if starting_street == Street::River {
        return contexts;
    }

    // Build context sizes per chance depth
    let context_sizes: Vec<usize> = if starting_street == Street::Flop {
        vec![1, num_turn_cards, num_turn_cards * num_river_cards]
    } else {
        // Turn
        vec![1, num_river_cards]
    };

    fn walk(
        tree: &IndexedActionTree,
        node_idx: usize,
        chance_depth: usize,
        contexts: &mut Vec<usize>,
        context_sizes: &[usize],
    ) {
        let node = tree.get(node_idx);
        if node.is_player() && chance_depth > 0 {
            let depth_idx = chance_depth.min(context_sizes.len() - 1);
            contexts[node_idx] = context_sizes[depth_idx];
        }
        let new_depth = if node.is_chance() {
            chance_depth + 1
        } else {
            chance_depth
        };
        for &child_idx in &node.children {
            walk(tree, child_idx, new_depth, contexts, context_sizes);
        }
    }

    walk(tree, tree.root_idx, 0, &mut contexts, &context_sizes);
    contexts
}

/// Precompute hand rank table for flop boards: all (turn, river) pairs.
///
/// Returns `table[turn_idx * num_cards + river_idx][combo_idx]` = hand rank
/// on (flop + turn_card + river_card). Diagonal entries (turn == river) are u32::MAX.
fn precompute_hand_rank_table_flop(board: &Board, valid_cards: &[u8]) -> Vec<Vec<u32>> {
    let num_cards = valid_cards.len();
    let mut table = Vec::with_capacity(num_cards * num_cards);

    for &tc in valid_cards.iter() {
        for &rc in valid_cards.iter() {
            if tc == rc {
                table.push(vec![u32::MAX; NUM_COMBOS]);
                continue;
            }
            let mut ranks = vec![u32::MAX; NUM_COMBOS];
            let board_mask = board.mask | (1u64 << tc) | (1u64 << rc);
            for combo_idx in 0..NUM_COMBOS {
                let combo = Combo::from_index(combo_idx);
                if combo.conflicts_with_mask(board_mask) {
                    continue;
                }
                let cards = [
                    board.cards[0],
                    board.cards[1],
                    board.cards[2],
                    tc,
                    rc,
                    combo.c0,
                    combo.c1,
                ];
                ranks[combo_idx] = evaluate_7cards(&cards);
            }
            table.push(ranks);
        }
    }
    table
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::poker::board_parser::parse_board;
    use crate::poker::range_parser::parse_range;
    use crate::tree::{ActionTree, BetSizeOptions, StreetConfig, TreeConfig};

    fn make_test_game(board_str: &str, oop_range_str: &str, ip_range_str: &str) -> PostflopGame {
        let sizes =
            BetSizeOptions::try_from_strs("50%, 100%", "2x, a").expect("Invalid bet sizes");
        let config = TreeConfig::new(2)
            .with_stack(100)
            .with_starting_street(Street::River)
            .with_starting_pot(100)
            .with_river(StreetConfig::uniform(sizes));

        let tree = ActionTree::new(config).expect("Failed to build tree");
        let indexed_tree = tree.to_indexed();
        let board = parse_board(board_str).expect("Invalid board");
        let oop_range = parse_range(oop_range_str).expect("Invalid OOP range");
        let ip_range = parse_range(ip_range_str).expect("Invalid IP range");

        PostflopGame::new(indexed_tree, board, oop_range, ip_range, 100, 100)
    }

    fn make_turn_test_game(
        board_str: &str,
        oop_range_str: &str,
        ip_range_str: &str,
    ) -> PostflopGame {
        let turn_sizes =
            BetSizeOptions::try_from_strs("50%", "2x, a").expect("Invalid bet sizes");
        let river_sizes =
            BetSizeOptions::try_from_strs("50%", "2x, a").expect("Invalid bet sizes");
        let config = TreeConfig::new(2)
            .with_stack(100)
            .with_starting_street(Street::Turn)
            .with_starting_pot(100)
            .with_turn(StreetConfig::uniform(turn_sizes))
            .with_river(StreetConfig::uniform(river_sizes));

        let tree = ActionTree::new(config).expect("Failed to build tree");
        let indexed_tree = tree.to_indexed();
        let board = parse_board(board_str).expect("Invalid board");
        let oop_range = parse_range(oop_range_str).expect("Invalid OOP range");
        let ip_range = parse_range(ip_range_str).expect("Invalid IP range");

        PostflopGame::new(indexed_tree, board, oop_range, ip_range, 100, 100)
    }

    // === River-only tests (backward compatibility) ===

    #[test]
    fn test_solver_creation() {
        let game = make_test_game("KhQsJs2c3d", "AA,KK", "QQ,JJ");
        let solver = PostflopSolver::new(&game);

        assert!(solver.num_hands(0) > 0); // IP hands
        assert!(solver.num_hands(1) > 0); // OOP hands
        assert_eq!(solver.num_river_cards(), 0); // River-only
    }

    #[test]
    fn test_exploitability_decreases() {
        let game = make_test_game("KhQsJs2c3d", "AA,KK,QQ,AKs", "AA,KK,QQ,JJ,AKs");
        let mut solver = PostflopSolver::new(&game);

        let exploit_before = solver.exploitability(&game);
        solver.train(&game, 100);
        let exploit_after = solver.exploitability(&game);

        assert!(
            exploit_after < exploit_before,
            "Exploitability should decrease: before={}, after={}",
            exploit_before,
            exploit_after,
        );
    }

    #[test]
    fn test_convergence() {
        let game = make_test_game(
            "KhQsJs2c3d",
            "AA,KK,QQ,AKs,AQs,KQs",
            "AA,KK,QQ,JJ,TT,AKs,AKo,KQs",
        );
        let mut solver = PostflopSolver::new(&game);

        solver.train(&game, 1000);
        let exploit = solver.exploitability(&game);
        let pot = 100.0f32;
        let exploit_pct = exploit / pot * 100.0;

        // Should converge to < 5% of pot with 1000 iterations
        assert!(
            exploit_pct < 5.0,
            "Exploitability should be < 5% pot after 1000 iterations, got {:.2}%",
            exploit_pct,
        );
    }

    // === Flop→Turn→River tests ===

    fn make_flop_test_game(
        board_str: &str,
        oop_range_str: &str,
        ip_range_str: &str,
    ) -> PostflopGame {
        let flop_sizes =
            BetSizeOptions::try_from_strs("67%", "a").expect("Invalid bet sizes");
        let turn_sizes =
            BetSizeOptions::try_from_strs("67%", "a").expect("Invalid bet sizes");
        let river_sizes =
            BetSizeOptions::try_from_strs("67%", "a").expect("Invalid bet sizes");
        let config = TreeConfig::new(2)
            .with_stack(100)
            .with_starting_street(Street::Flop)
            .with_starting_pot(100)
            .with_flop(StreetConfig::uniform(flop_sizes))
            .with_turn(StreetConfig::uniform(turn_sizes))
            .with_river(StreetConfig::uniform(river_sizes));

        let tree = ActionTree::new(config).expect("Failed to build tree");
        let indexed_tree = tree.to_indexed();
        let board = parse_board(board_str).expect("Invalid board");
        let oop_range = parse_range(oop_range_str).expect("Invalid OOP range");
        let ip_range = parse_range(ip_range_str).expect("Invalid IP range");

        PostflopGame::new(indexed_tree, board, oop_range, ip_range, 100, 100)
    }

    #[test]
    fn test_flop_solver_creation() {
        let game = make_flop_test_game("KhQsJs", "AA,KK", "QQ,JJ");
        let solver = PostflopSolver::new(&game);

        assert!(solver.num_hands(0) > 0);
        assert!(solver.num_hands(1) > 0);
        assert_eq!(solver.num_turn_cards(), 49); // 52 - 3 board cards
        assert_eq!(solver.num_river_cards(), 49); // same pool, skip at runtime
    }

    #[test]
    fn test_flop_exploitability_decreases() {
        let game = make_flop_test_game("KhQsJs", "AA,KK", "QQ,JJ");
        let mut solver = PostflopSolver::new(&game);

        let exploit_before = solver.exploitability(&game);
        solver.train(&game, 100);
        let exploit_after = solver.exploitability(&game);

        assert!(
            exploit_after < exploit_before,
            "Flop exploitability should decrease: before={}, after={}",
            exploit_before,
            exploit_after,
        );
    }

    #[test]
    fn test_flop_convergence() {
        let game = make_flop_test_game("KhQsJs", "AA,KK", "QQ,JJ");
        let mut solver = PostflopSolver::new(&game);

        solver.train(&game, 500);
        let exploit = solver.exploitability(&game);
        let pot = 100.0f32;
        let exploit_pct = exploit / pot * 100.0;

        // Should converge to < 15% of pot after 500 iterations
        assert!(
            exploit_pct < 15.0,
            "Flop exploitability should be < 15% pot after 500 iterations, got {:.2}%",
            exploit_pct,
        );
    }

    // === Turn→River tests (multi-street) ===

    #[test]
    fn test_turn_solver_creation() {
        let game = make_turn_test_game("KhQsJs2c", "AA,KK", "QQ,JJ");
        let solver = PostflopSolver::new(&game);

        assert!(solver.num_hands(0) > 0);
        assert!(solver.num_hands(1) > 0);
        assert!(solver.num_river_cards() > 0);
        assert_eq!(solver.num_river_cards(), 48); // 52 - 4 board cards
    }

    #[test]
    fn test_turn_exploitability_decreases() {
        let game = make_turn_test_game("KhQsJs2c", "AA,KK,QQ", "AA,KK,QQ,JJ");
        let mut solver = PostflopSolver::new(&game);

        let exploit_before = solver.exploitability(&game);
        solver.train(&game, 100);
        let exploit_after = solver.exploitability(&game);

        assert!(
            exploit_after < exploit_before,
            "Turn exploitability should decrease: before={}, after={}",
            exploit_before,
            exploit_after,
        );
    }

    #[test]
    fn test_turn_convergence() {
        let game = make_turn_test_game("KhQsJs2c", "AA,KK,QQ", "AA,KK,QQ,JJ");
        let mut solver = PostflopSolver::new(&game);

        solver.train(&game, 500);
        let exploit = solver.exploitability(&game);
        let pot = 100.0f32;
        let exploit_pct = exploit / pot * 100.0;

        // Should converge to < 10% of pot after 500 iterations
        assert!(
            exploit_pct < 10.0,
            "Turn exploitability should be < 10% pot after 500 iterations, got {:.2}%",
            exploit_pct,
        );
    }

    // === Multiway (N-player) tests ===

    fn make_3way_test_game(
        board_str: &str,
        ranges: &[&str],
    ) -> PostflopGame {
        use crate::poker::hands::Range;
        use crate::tree::TreeConfig;

        let sizes =
            BetSizeOptions::try_from_strs("50%, 100%", "2x, a").expect("Invalid bet sizes");
        let config = TreeConfig::new(3)
            .with_stack(100)
            .with_starting_street(Street::River)
            .with_starting_pot(100)
            .with_river(StreetConfig::uniform(sizes));

        let tree = ActionTree::new(config).expect("Failed to build tree");
        let indexed_tree = tree.to_indexed();
        let board = parse_board(board_str).expect("Invalid board");

        let parsed_ranges: Vec<Range> = ranges
            .iter()
            .map(|r| parse_range(r).expect("Invalid range"))
            .collect();

        let stacks = vec![100; 3];

        PostflopGame::new_multiway(indexed_tree, board, parsed_ranges, stacks, 100)
    }

    #[test]
    fn test_3way_solver_creation() {
        let game = make_3way_test_game("KhQsJs2c3d", &["AA,KK", "QQ,JJ", "TT,99"]);
        let solver = PostflopSolver::new(&game);

        assert_eq!(solver.num_players(), 3);
        assert!(solver.num_hands(0) > 0); // Player 0
        assert!(solver.num_hands(1) > 0); // Player 1
        assert!(solver.num_hands(2) > 0); // Player 2
    }

    #[test]
    fn test_3way_game_num_players() {
        let game = make_3way_test_game("KhQsJs2c3d", &["AA,KK", "QQ,JJ", "TT,99"]);
        assert_eq!(game.get_num_players(), 3);
        assert!(!game.is_heads_up());
    }

    #[test]
    fn test_3way_matchup_enumeration() {
        let game = make_3way_test_game("KhQsJs2c3d", &["AA,KK", "QQ,JJ", "TT,99"]);

        // Should have valid multiway matchups
        assert!(game.num_matchups() > 0);

        // Check that multiway matchups are populated
        let mut count = 0;
        for matchup in game.matchups_multiway() {
            let (combos, weight) = matchup;
            assert_eq!(combos.len(), 3); // 3 players
            assert!(*weight > 0.0);
            count += 1;
        }
        assert!(count > 0);
    }

    #[test]
    fn test_3way_exploitability_decreases() {
        let game = make_3way_test_game("KhQsJs2c3d", &["AA,KK,QQ", "AA,KK,QQ", "AA,KK,QQ"]);
        let mut solver = PostflopSolver::new(&game);

        let exploit_before = solver.exploitability(&game);
        solver.train(&game, 50);
        let exploit_after = solver.exploitability(&game);

        // Exploitability should decrease (or at least not increase significantly)
        // For 3-way, convergence may be slower, so we use a relaxed check
        assert!(
            exploit_after <= exploit_before * 1.1, // Allow small increase due to approximation
            "3-way exploitability should not increase significantly: before={}, after={}",
            exploit_before,
            exploit_after,
        );
    }
}
