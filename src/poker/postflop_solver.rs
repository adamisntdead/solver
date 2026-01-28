//! Vectorized postflop CFR solver with multi-street support.
//!
//! Walks the betting tree ONCE per iteration with per-hand arrays,
//! matching the approach used by Gambit and binary_wasm_solver.
//!
//! Key data flow:
//! - `cfreach` (opponent counterfactual reach) flows DOWN the tree
//! - `result` (counterfactual values) flow UP the tree
//! - Regrets stored per (node, context, action, hand) in flat arrays
//!
//! Multi-street support:
//! - For 5-card boards (river): single context, hand ranks from HandInfo
//! - For 4-card boards (turn→river): one context per valid river card,
//!   hand ranks from precomputed `hand_rank_table`

use crate::poker::hands::{Board, Combo, NUM_COMBOS};
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

/// Vectorized CFR solver for postflop poker.
///
/// Instead of walking the tree once per hand pair, this solver walks the
/// betting tree once per iteration with arrays of per-hand values.
///
/// Supports multi-street solving (turn→river) via a card context system:
/// nodes below chance nodes have separate regrets per dealt river card.
pub struct PostflopSolver {
    /// Per-node regret storage.
    /// Layout: `regrets[node_idx][context * num_actions * num_hands + action * num_hands + hand]`
    /// Only player nodes have non-empty vectors.
    regrets: Vec<Vec<f32>>,

    /// Per-node cumulative strategy (same layout as regrets).
    cum_strategy: Vec<Vec<f32>>,

    /// Hand info per player.
    /// `hands[0]` = tree player 0 (IP), `hands[1]` = tree player 1 (OOP)
    hands: [Vec<HandInfo>; 2],

    /// Number of iterations completed per player.
    num_steps: [u32; 2],

    // === Multi-street fields ===
    /// The board used for this solve.
    board: Board,

    /// Starting street (derived from board length).
    starting_street: Street,

    /// Valid river cards to deal (empty for river-only boards).
    valid_river_cards: Vec<u8>,

    /// Precomputed hand ranks per river card.
    /// `hand_rank_table[river_idx][combo_idx]` = hand rank on (board + river_card).
    /// Empty for river-only boards.
    hand_rank_table: Vec<Vec<u32>>,

    /// Number of card contexts per node.
    /// 1 for nodes above the chance node, `valid_river_cards.len()` for nodes below.
    node_num_contexts: Vec<usize>,
}

impl PostflopSolver {
    /// Create a new solver for a postflop game.
    pub fn new(game: &PostflopGame) -> Self {
        let tree = &game.tree;
        let board = game.board.clone();
        let starting_street = game.starting_street;
        let is_river = starting_street == Street::River;

        // Build hand arrays: player 0 = IP, player 1 = OOP
        let ranges = [&game.ip_range, &game.oop_range];
        let mut hands: [Vec<HandInfo>; 2] = [Vec::new(), Vec::new()];

        for player in 0..2 {
            let range = ranges[player];
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

        // Compute valid river cards and hand rank table for multi-street
        let (valid_river_cards, hand_rank_table) = if is_river {
            (Vec::new(), Vec::new())
        } else {
            let mut river_cards: Vec<u8> = Vec::new();
            for card in 0..52u8 {
                if (board.mask >> card) & 1 == 0 {
                    river_cards.push(card);
                }
            }
            let table = precompute_hand_rank_table(&board, &river_cards);
            (river_cards, table)
        };

        // Determine context count per node
        let node_num_contexts =
            compute_node_contexts(tree, is_river, valid_river_cards.len());

        // Allocate per-node storage
        let num_nodes = tree.len();
        let mut regrets = Vec::with_capacity(num_nodes);
        let mut cum_strategy = Vec::with_capacity(num_nodes);

        for node_idx in 0..num_nodes {
            let node = tree.get(node_idx);
            if node.is_player() {
                let acting_player = node.player();
                let num_actions = node.actions.len();
                let num_hands = hands[acting_player].len();
                let num_contexts = node_num_contexts[node_idx];
                let size = num_contexts * num_actions * num_hands;
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
            num_steps: [0, 0],
            board,
            starting_street,
            valid_river_cards,
            hand_rank_table,
            node_num_contexts,
        }
    }

    /// Train using Linear CFR with alternating updates.
    pub fn train(&mut self, game: &PostflopGame, iterations: u32) {
        for i in 0..iterations {
            let traverser = (i as usize) % 2;

            let num_trav_hands = self.hands[traverser].len();

            // Initial opponent reach = opponent's range weights
            let opponent = 1 - traverser;
            let cfreach: Vec<f32> = self.hands[opponent]
                .iter()
                .map(|h| h.initial_weight)
                .collect();

            let mut result = vec![0.0f32; num_trav_hands];

            // Clone the tree Arc to avoid borrow issues
            let tree = game.tree.clone();
            let dealt_cards_mask = self.board.mask;
            self.solve_recursive(
                &mut result,
                &tree,
                tree.root_idx,
                traverser,
                &cfreach,
                0,
                dealt_cards_mask,
            );

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

            // Get current strategy via regret matching
            let strategy =
                self.regret_matching_for(node_idx, num_actions, num_hands, card_context);

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

            // Update regrets: regret[a][h] += cfv_action[a][h] - node_cfv[h]
            let offset = card_context * num_actions * num_hands;
            for action in 0..num_actions {
                for h in 0..num_hands {
                    self.regrets[node_idx][offset + action * num_hands + h] +=
                        cfv_actions[action * num_hands + h] - result[h];
                }
            }

            // Accumulate strategy for averaging
            for i in 0..num_actions * num_hands {
                self.cum_strategy[node_idx][offset + i] += strategy[i];
            }
        } else {
            // === OPPONENT'S NODE ===
            let num_opp_hands = self.hands[acting_player].len();
            let num_trav_hands = self.hands[traverser].len();

            // Get opponent's strategy
            let strategy =
                self.regret_matching_for(node_idx, num_actions, num_opp_hands, card_context);

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

    /// Handle a chance node (street transition / river card dealing).
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

        // Deal river card: enumerate all valid river cards
        let opponent = 1 - traverser;
        let num_trav_hands = result.len();
        let num_cards = self.valid_river_cards.len();
        let inv_num_cards = 1.0 / num_cards as f32;
        let child_idx = node.children[0];
        result.fill(0.0);

        // Clone to avoid borrow conflict with &mut self
        let river_cards: Vec<u8> = self.valid_river_cards.clone();

        for (river_idx, &rc) in river_cards.iter().enumerate() {
            let new_mask = dealt_cards_mask | (1u64 << rc);

            // Scale cfreach by 1/num_cards, zero blocked opponent hands
            let mut cfreach_dealt = Vec::with_capacity(cfreach.len());
            for (j, &reach) in cfreach.iter().enumerate() {
                let hand = &self.hands[opponent][j];
                if hand.c0 == rc || hand.c1 == rc {
                    cfreach_dealt.push(0.0);
                } else {
                    cfreach_dealt.push(reach * inv_num_cards);
                }
            }

            let mut card_result = vec![0.0f32; num_trav_hands];
            self.solve_recursive(
                &mut card_result,
                tree,
                child_idx,
                traverser,
                &cfreach_dealt,
                river_idx,
                new_mask,
            );

            // Zero blocked traverser hands, accumulate
            for (h, hand) in self.hands[traverser].iter().enumerate() {
                if hand.c0 != rc && hand.c1 != rc {
                    result[h] += card_result[h];
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
                } else if dealt_cards_mask != self.board.mask {
                    // === River card dealt: use hand_rank_table[card_context] ===
                    self.evaluate_showdown_with_context(
                        result,
                        trav_hands,
                        opp_hands,
                        cfreach,
                        half_pot,
                        card_context,
                    );
                } else {
                    // === All-in runout: enumerate river cards, average showdown ===
                    self.evaluate_allin_runout(result, trav_hands, opp_hands, cfreach, half_pot);
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
    fn evaluate_allin_runout(
        &self,
        result: &mut [f32],
        trav_hands: &[HandInfo],
        opp_hands: &[HandInfo],
        cfreach: &[f32],
        half_pot: f32,
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
                    if trav.c0 == rc || trav.c1 == rc || opp.c0 == rc || opp.c1 == rc {
                        continue;
                    }
                    let trav_rank = self.hand_rank_table[river_idx][trav.combo_idx];
                    let opp_rank = self.hand_rank_table[river_idx][opp.combo_idx];
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

    /// Compute current strategy from regrets via regret matching.
    /// Returns owned Vec of size `num_actions * num_hands`.
    fn regret_matching_for(
        &self,
        node_idx: usize,
        num_actions: usize,
        num_hands: usize,
        card_context: usize,
    ) -> Vec<f32> {
        let regrets = &self.regrets[node_idx];
        if regrets.is_empty() {
            let uniform = 1.0 / num_actions as f32;
            return vec![uniform; num_actions * num_hands];
        }

        let offset = card_context * num_actions * num_hands;
        let mut strategy = vec![0.0f32; num_actions * num_hands];

        // Clamp negatives to 0
        for i in 0..num_actions * num_hands {
            strategy[i] = regrets[offset + i].max(0.0);
        }

        // Normalize per hand
        for h in 0..num_hands {
            let mut sum = 0.0f32;
            for a in 0..num_actions {
                sum += strategy[a * num_hands + h];
            }
            if sum > 0.0 {
                for a in 0..num_actions {
                    strategy[a * num_hands + h] /= sum;
                }
            } else {
                let uniform = 1.0 / num_actions as f32;
                for a in 0..num_actions {
                    strategy[a * num_hands + h] = uniform;
                }
            }
        }

        strategy
    }

    /// Compute average strategy for a node (from cumulative strategy sums).
    fn average_strategy_for(
        &self,
        node_idx: usize,
        num_actions: usize,
        num_hands: usize,
        card_context: usize,
    ) -> Vec<f32> {
        let cum = &self.cum_strategy[node_idx];
        if cum.is_empty() || cum.iter().all(|&x| x == 0.0) {
            let uniform = 1.0 / num_actions as f32;
            return vec![uniform; num_actions * num_hands];
        }

        let offset = card_context * num_actions * num_hands;
        let mut avg = Vec::with_capacity(num_actions * num_hands);
        for i in 0..num_actions * num_hands {
            avg.push(cum[offset + i]);
        }

        // Normalize per hand
        for h in 0..num_hands {
            let mut sum = 0.0f32;
            for a in 0..num_actions {
                sum += avg[a * num_hands + h];
            }
            if sum > 0.0 {
                for a in 0..num_actions {
                    avg[a * num_hands + h] /= sum;
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

        for traverser in 0..2 {
            let opponent = 1 - traverser;
            let num_trav_hands = self.hands[traverser].len();

            // Initial opponent reach = range weights
            let cfreach: Vec<f32> = self.hands[opponent]
                .iter()
                .map(|h| h.initial_weight)
                .collect();

            let mut br_values = vec![0.0f32; num_trav_hands];
            let dealt_cards_mask = self.board.mask;
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
                self.average_strategy_for(node_idx, num_actions, num_opp_hands, card_context);

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

        // Deal river card
        let opponent = 1 - traverser;
        let num_trav_hands = result.len();
        let num_cards = self.valid_river_cards.len();
        let inv_num_cards = 1.0 / num_cards as f32;
        let child_idx = node.children[0];
        result.fill(0.0);

        for (river_idx, &rc) in self.valid_river_cards.iter().enumerate() {
            let new_mask = dealt_cards_mask | (1u64 << rc);

            let mut cfreach_dealt = Vec::with_capacity(cfreach.len());
            for (j, &reach) in cfreach.iter().enumerate() {
                let hand = &self.hands[opponent][j];
                if hand.c0 == rc || hand.c1 == rc {
                    cfreach_dealt.push(0.0);
                } else {
                    cfreach_dealt.push(reach * inv_num_cards);
                }
            }

            let mut card_result = vec![0.0f32; num_trav_hands];
            self.best_response_value(
                &mut card_result,
                tree,
                child_idx,
                traverser,
                &cfreach_dealt,
                river_idx,
                new_mask,
            );

            // Zero blocked traverser hands, accumulate
            for (h, hand) in self.hands[traverser].iter().enumerate() {
                if hand.c0 != rc && hand.c1 != rc {
                    result[h] += card_result[h];
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

        let per_context = total_size / num_contexts;
        let num_actions = per_context / num_hands;

        let avg = self.average_strategy_for(node_idx, num_actions, num_hands, card_context);
        (0..num_actions)
            .map(|a| avg[a * num_hands + hand_idx])
            .collect()
    }

    /// Get total iteration count (sum of both players).
    pub fn total_iterations(&self) -> u32 {
        self.num_steps[0] + self.num_steps[1]
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
/// Nodes above the chance node (turn betting): 1 context.
/// Nodes below the chance node (river betting): `num_river_cards` contexts.
fn compute_node_contexts(
    tree: &IndexedActionTree,
    is_river: bool,
    num_river_cards: usize,
) -> Vec<usize> {
    let mut contexts = vec![1usize; tree.len()];
    if is_river || num_river_cards == 0 {
        return contexts;
    }

    fn walk(
        tree: &IndexedActionTree,
        node_idx: usize,
        below_chance: bool,
        contexts: &mut Vec<usize>,
        num_contexts: usize,
    ) {
        let node = tree.get(node_idx);
        if below_chance && node.is_player() {
            contexts[node_idx] = num_contexts;
        }
        let is_chance = node.is_chance();
        for &child_idx in &node.children {
            walk(
                tree,
                child_idx,
                below_chance || is_chance,
                contexts,
                num_contexts,
            );
        }
    }

    walk(tree, tree.root_idx, false, &mut contexts, num_river_cards);
    contexts
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
}
