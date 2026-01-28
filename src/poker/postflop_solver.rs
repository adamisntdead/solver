//! Vectorized postflop CFR solver.
//!
//! Walks the betting tree ONCE per iteration with per-hand arrays,
//! matching the approach used by Gambit and binary_wasm_solver.
//!
//! Key data flow:
//! - `cfreach` (opponent counterfactual reach) flows DOWN the tree
//! - `result` (counterfactual values) flow UP the tree
//! - Regrets stored per (node, action, hand) in flat arrays

use crate::poker::hands::{Combo, NUM_COMBOS};
use crate::poker::postflop_game::PostflopGame;
use crate::tree::{IndexedActionTree, IndexedNode, IndexedNodeType};

/// Info about a single hand (combo) for a player.
#[derive(Clone)]
struct HandInfo {
    combo_idx: usize,
    c0: u8,
    c1: u8,
    initial_weight: f32,
    hand_rank: u32,
}

/// Vectorized CFR solver for postflop poker.
///
/// Instead of walking the tree once per hand pair, this solver walks the
/// betting tree once per iteration with arrays of per-hand values.
pub struct PostflopSolver {
    /// Per-node regret storage.
    /// Layout: `regrets[node_idx][action * num_hands + hand]`
    /// Only player nodes have non-empty vectors.
    regrets: Vec<Vec<f32>>,

    /// Per-node cumulative strategy (same layout as regrets).
    cum_strategy: Vec<Vec<f32>>,

    /// Hand info per player.
    /// `hands[0]` = tree player 0 (IP), `hands[1]` = tree player 1 (OOP)
    hands: [Vec<HandInfo>; 2],

    /// Number of iterations completed per player.
    num_steps: [u32; 2],
}

impl PostflopSolver {
    /// Create a new solver for a postflop game.
    pub fn new(game: &PostflopGame) -> Self {
        let tree = &game.tree;
        let matchups = &game.matchups;

        // Build hand arrays: player 0 = IP, player 1 = OOP
        let ranges = [&game.ip_range, &game.oop_range];
        let mut hands: [Vec<HandInfo>; 2] = [Vec::new(), Vec::new()];

        for player in 0..2 {
            let range = ranges[player];
            for combo_idx in 0..NUM_COMBOS {
                let weight = range.weights[combo_idx];
                if weight == 0.0 || !matchups.is_valid_combo(combo_idx) {
                    continue;
                }
                let combo = Combo::from_index(combo_idx);
                hands[player].push(HandInfo {
                    combo_idx,
                    c0: combo.c0,
                    c1: combo.c1,
                    initial_weight: weight,
                    hand_rank: matchups.hand_ranks[combo_idx],
                });
            }
        }

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
                let size = num_actions * num_hands;
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
        }
    }

    /// Train using Linear CFR with alternating updates.
    pub fn train(&mut self, game: &PostflopGame, iterations: u32) {
        for i in 0..iterations {
            let traverser = (i as usize) % 2;
            let opponent = 1 - traverser;

            let num_trav_hands = self.hands[traverser].len();

            // Initial opponent reach = opponent's range weights
            let cfreach: Vec<f32> = self.hands[opponent]
                .iter()
                .map(|h| h.initial_weight)
                .collect();

            let mut result = vec![0.0f32; num_trav_hands];

            // Clone the tree Arc to avoid borrow issues
            let tree = game.tree.clone();
            self.solve_recursive(&mut result, &tree, tree.root_idx, traverser, &cfreach);

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
    fn solve_recursive(
        &mut self,
        result: &mut [f32],
        tree: &IndexedActionTree,
        node_idx: usize,
        traverser: usize,
        cfreach: &[f32],
    ) {
        let node = tree.get(node_idx);

        if node.is_terminal() {
            self.evaluate_terminal(result, node, traverser, cfreach);
            return;
        }

        if node.is_chance() {
            // River-only: pass through to single child
            if !node.children.is_empty() {
                self.solve_recursive(result, tree, node.children[0], traverser, cfreach);
            }
            return;
        }

        let acting_player = node.player();
        let num_actions = node.actions.len();
        let child_indices: Vec<usize> = node.children.clone();

        if acting_player == traverser {
            // === TRAVERSER'S NODE ===
            let num_hands = self.hands[traverser].len();

            // Get current strategy via regret matching
            let strategy = self.regret_matching_for(node_idx, num_actions, num_hands);

            // Compute CFV for each action
            let mut cfv_actions = vec![0.0f32; num_actions * num_hands];
            for action in 0..num_actions {
                let mut action_result = vec![0.0f32; num_hands];
                self.solve_recursive(&mut action_result, tree, child_indices[action], traverser, cfreach);
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
            for action in 0..num_actions {
                for h in 0..num_hands {
                    self.regrets[node_idx][action * num_hands + h] +=
                        cfv_actions[action * num_hands + h] - result[h];
                }
            }

            // Accumulate strategy for averaging
            for i in 0..num_actions * num_hands {
                self.cum_strategy[node_idx][i] += strategy[i];
            }
        } else {
            // === OPPONENT'S NODE ===
            let num_opp_hands = self.hands[acting_player].len();
            let num_trav_hands = self.hands[traverser].len();

            // Get opponent's strategy
            let strategy = self.regret_matching_for(node_idx, num_actions, num_opp_hands);

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

    /// Evaluate a terminal node for all traverser hands.
    fn evaluate_terminal(
        &self,
        result: &mut [f32],
        node: &IndexedNode,
        traverser: usize,
        cfreach: &[f32],
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
                for (h, trav) in trav_hands.iter().enumerate() {
                    let mut cfv = 0.0f32;
                    for (j, opp) in opp_hands.iter().enumerate() {
                        if cfreach[j] == 0.0 {
                            continue;
                        }
                        // Card conflict check
                        if trav.c0 == opp.c0
                            || trav.c0 == opp.c1
                            || trav.c1 == opp.c0
                            || trav.c1 == opp.c1
                        {
                            continue;
                        }
                        if trav.hand_rank > opp.hand_rank {
                            cfv += half_pot * cfreach[j]; // Win
                        } else if trav.hand_rank < opp.hand_rank {
                            cfv -= half_pot * cfreach[j]; // Lose
                        }
                        // Tie: cfv += 0
                    }
                    result[h] = cfv;
                }
            }
            _ => panic!("evaluate_terminal called on non-terminal node"),
        }
    }

    /// Compute current strategy from regrets via regret matching.
    /// Returns owned Vec of size `num_actions * num_hands`.
    fn regret_matching_for(
        &self,
        node_idx: usize,
        num_actions: usize,
        num_hands: usize,
    ) -> Vec<f32> {
        let regrets = &self.regrets[node_idx];
        if regrets.is_empty() {
            let uniform = 1.0 / num_actions as f32;
            return vec![uniform; num_actions * num_hands];
        }

        let mut strategy = vec![0.0f32; num_actions * num_hands];

        // Clamp negatives to 0
        for i in 0..regrets.len() {
            strategy[i] = regrets[i].max(0.0);
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
    ) -> Vec<f32> {
        let cum = &self.cum_strategy[node_idx];
        if cum.is_empty() || cum.iter().all(|&x| x == 0.0) {
            let uniform = 1.0 / num_actions as f32;
            return vec![uniform; num_actions * num_hands];
        }

        let mut avg = cum.clone();

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
            self.best_response_value(&mut br_values, tree, tree.root_idx, traverser, &cfreach);

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
    ) {
        let node = tree.get(node_idx);

        if node.is_terminal() {
            self.evaluate_terminal(result, node, traverser, cfreach);
            return;
        }

        if node.is_chance() {
            if !node.children.is_empty() {
                self.best_response_value(result, tree, node.children[0], traverser, cfreach);
            }
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
            let avg_strategy = self.average_strategy_for(node_idx, num_actions, num_opp_hands);

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

    /// Get the average strategy for a specific hand at a node.
    ///
    /// Returns a Vec of probabilities (one per action).
    pub fn get_hand_strategy(&self, node_idx: usize, hand_idx: usize, player: usize) -> Vec<f32> {
        let num_actions = self.regrets[node_idx].len() / self.hands[player].len();
        let num_hands = self.hands[player].len();

        if num_actions == 0 {
            return Vec::new();
        }

        let avg = self.average_strategy_for(node_idx, num_actions, num_hands);
        let mut result = Vec::with_capacity(num_actions);
        for a in 0..num_actions {
            result.push(avg[a * num_hands + hand_idx]);
        }
        result
    }

    /// Get total iteration count (sum of both players).
    pub fn total_iterations(&self) -> u32 {
        self.num_steps[0] + self.num_steps[1]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::poker::board_parser::parse_board;
    use crate::poker::range_parser::parse_range;
    use crate::tree::{ActionTree, BetSizeOptions, Street, StreetConfig, TreeConfig};

    fn make_test_game(
        board_str: &str,
        oop_range_str: &str,
        ip_range_str: &str,
    ) -> PostflopGame {
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

    #[test]
    fn test_solver_creation() {
        let game = make_test_game("KhQsJs2c3d", "AA,KK", "QQ,JJ");
        let solver = PostflopSolver::new(&game);

        assert!(solver.num_hands(0) > 0); // IP hands
        assert!(solver.num_hands(1) > 0); // OOP hands
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
        let game = make_test_game("KhQsJs2c3d", "AA,KK,QQ,AKs,AQs,KQs", "AA,KK,QQ,JJ,TT,AKs,AKo,KQs");
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
}
