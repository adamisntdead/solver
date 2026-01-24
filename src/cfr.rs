use std::collections::HashMap;

use crate::game::{Game, GameNode};

/// Information set data tracked during CFR.
#[derive(Clone)]
struct InfoSetData {
    /// Cumulative regrets for each action.
    regret_sum: Vec<f64>,
    /// Cumulative strategy weights for each action.
    strategy_sum: Vec<f64>,
    /// Number of actions available at this info set.
    num_actions: usize,
}

impl InfoSetData {
    fn new(num_actions: usize) -> Self {
        Self {
            regret_sum: vec![0.0; num_actions],
            strategy_sum: vec![0.0; num_actions],
            num_actions,
        }
    }

    /// Computes the current strategy using regret matching.
    /// Returns a probability distribution over actions.
    fn current_strategy(&self) -> Vec<f64> {
        let mut strategy = vec![0.0; self.num_actions];

        // Sum of positive regrets
        let normalizing_sum: f64 = self
            .regret_sum
            .iter()
            .map(|&r| r.max(0.0))
            .sum();

        if normalizing_sum > 0.0 {
            // Proportional to positive regrets
            for (i, &regret) in self.regret_sum.iter().enumerate() {
                strategy[i] = regret.max(0.0) / normalizing_sum;
            }
        } else {
            // Uniform strategy if no positive regrets
            let uniform_prob = 1.0 / self.num_actions as f64;
            for prob in &mut strategy {
                *prob = uniform_prob;
            }
        }

        strategy
    }

    /// Returns the average strategy (Nash equilibrium approximation).
    fn average_strategy(&self) -> Vec<f64> {
        let mut avg_strategy = vec![0.0; self.num_actions];
        let normalizing_sum: f64 = self.strategy_sum.iter().sum();

        if normalizing_sum > 0.0 {
            for (i, &sum) in self.strategy_sum.iter().enumerate() {
                avg_strategy[i] = sum / normalizing_sum;
            }
        } else {
            // Uniform if no data
            let uniform_prob = 1.0 / self.num_actions as f64;
            for prob in &mut avg_strategy {
                *prob = uniform_prob;
            }
        }

        avg_strategy
    }
}

/// Discount parameters for Discounted CFR.
///
/// Controls how past regrets and strategies are weighted relative to recent ones.
/// - `alpha`: Exponent for positive regret discounting (typically 1.5)
/// - `beta`: Exponent for negative regret discounting (typically 0.0)
/// - `gamma`: Exponent for strategy sum discounting (typically 2.0)
#[derive(Clone, Copy)]
pub struct DiscountParams {
    pub alpha: f64,
    pub beta: f64,
    pub gamma: f64,
}

impl Default for DiscountParams {
    fn default() -> Self {
        // Default values from the DCFR paper
        Self {
            alpha: 1.5,
            beta: 0.0,
            gamma: 2.0,
        }
    }
}

impl DiscountParams {
    /// Computes the discount factor for positive regrets at iteration t.
    fn positive_regret_discount(&self, t: u32) -> f64 {
        let t = t as f64;
        t.powf(self.alpha) / (t.powf(self.alpha) + 1.0)
    }

    /// Computes the discount factor for negative regrets at iteration t.
    fn negative_regret_discount(&self, t: u32) -> f64 {
        let t = t as f64;
        t.powf(self.beta) / (t.powf(self.beta) + 1.0)
    }

    /// Computes the discount factor for strategy sums at iteration t.
    fn strategy_discount(&self, t: u32) -> f64 {
        let t = t as f64;
        (t / (t + 1.0)).powf(self.gamma)
    }
}

/// Discounted CFR solver for n-player games.
pub struct CfrSolver {
    /// Data for each information set, keyed by info set string.
    info_sets: HashMap<String, InfoSetData>,
    /// Discount parameters.
    params: DiscountParams,
    /// Current iteration number.
    iteration: u32,
}

impl CfrSolver {
    /// Creates a new CFR solver with the given discount parameters.
    pub fn new(params: DiscountParams) -> Self {
        Self {
            info_sets: HashMap::new(),
            params,
            iteration: 0,
        }
    }

    /// Creates a new CFR solver with default discounted CFR parameters.
    pub fn discounted() -> Self {
        Self::new(DiscountParams::default())
    }

    /// Creates a new CFR solver with vanilla CFR parameters (no discounting).
    pub fn vanilla() -> Self {
        Self::new(DiscountParams {
            alpha: 1.0,
            beta: 1.0,
            gamma: 1.0,
        })
    }

    /// Runs CFR for the specified number of iterations.
    pub fn train<G: Game>(&mut self, game: &G, iterations: u32) {
        let num_players = game.num_players();

        for _ in 0..iterations {
            self.iteration += 1;

            // Run CFR for each player
            for player in 0..num_players {
                let root = game.root();
                // Initialize reach probabilities to 1.0 for all players
                let reach_probs = vec![1.0; num_players];
                self.cfr(&root, player, num_players, &reach_probs);
            }

            // Apply discounting
            self.apply_discounts();
        }
    }

    /// The core CFR recursive function.
    ///
    /// Returns the expected payoff for `player` at this node.
    ///
    /// # Arguments
    /// * `node` - Current game node
    /// * `player` - The player we're computing values for
    /// * `num_players` - Total number of players in the game
    /// * `reach_probs` - Reach probability for each player to this node
    fn cfr<N: GameNode>(
        &mut self,
        node: &N,
        player: usize,
        num_players: usize,
        reach_probs: &[f64],
    ) -> f64 {
        // Terminal node: return payoff
        if node.is_terminal() {
            return node.payoff(player);
        }

        let num_actions = node.num_actions();

        // Chance node: average over all outcomes (assuming uniform)
        if node.is_chance() {
            let mut expected_value = 0.0;
            for action in 0..num_actions {
                let child = node.play(action);
                expected_value += self.cfr(&child, player, num_players, reach_probs);
            }
            return expected_value / num_actions as f64;
        }

        let current_player = node.current_player();
        let info_key = node.info_set_key();

        // Get or create info set data and compute current strategy
        let strategy = {
            let info_set = self
                .info_sets
                .entry(info_key.clone())
                .or_insert_with(|| InfoSetData::new(num_actions));
            info_set.current_strategy()
        };

        // Compute counterfactual values for each action
        let mut action_values = vec![0.0; num_actions];
        let mut node_value = 0.0;

        for action in 0..num_actions {
            let child = node.play(action);
            let action_prob = strategy[action];

            // Update reach probability for the current player
            let mut child_reach_probs = reach_probs.to_vec();
            child_reach_probs[current_player] *= action_prob;

            let child_value = self.cfr(&child, player, num_players, &child_reach_probs);

            action_values[action] = child_value;
            node_value += action_prob * child_value;
        }

        // Update regrets and strategy sums only for the traversing player's info sets
        if current_player == player {
            // Counterfactual reach: product of all other players' reach probabilities
            let cf_reach: f64 = reach_probs
                .iter()
                .enumerate()
                .filter(|&(i, _)| i != player)
                .map(|(_, &p)| p)
                .product();

            let info_set = self.info_sets.get_mut(&info_key).unwrap();

            for action in 0..num_actions {
                // Regret is the counterfactual value of the action minus the node value
                let regret = action_values[action] - node_value;
                // Weight by counterfactual reach (all opponents' reach)
                info_set.regret_sum[action] += cf_reach * regret;
            }

            // Update strategy sum weighted by player's own reach probability
            for action in 0..num_actions {
                info_set.strategy_sum[action] += reach_probs[player] * strategy[action];
            }
        }

        node_value
    }

    /// Applies discount factors to all regrets and strategy sums.
    fn apply_discounts(&mut self) {
        let pos_discount = self.params.positive_regret_discount(self.iteration);
        let neg_discount = self.params.negative_regret_discount(self.iteration);
        let strat_discount = self.params.strategy_discount(self.iteration);

        for info_set in self.info_sets.values_mut() {
            // Discount regrets
            for regret in &mut info_set.regret_sum {
                if *regret > 0.0 {
                    *regret *= pos_discount;
                } else {
                    *regret *= neg_discount;
                }
            }

            // Discount strategy sums
            for sum in &mut info_set.strategy_sum {
                *sum *= strat_discount;
            }
        }
    }

    /// Returns the average strategy for a given information set.
    pub fn get_strategy(&self, info_set_key: &str) -> Option<Vec<f64>> {
        self.info_sets.get(info_set_key).map(|is| is.average_strategy())
    }

    /// Returns all computed strategies.
    pub fn all_strategies(&self) -> HashMap<String, Vec<f64>> {
        self.info_sets
            .iter()
            .map(|(k, v)| (k.clone(), v.average_strategy()))
            .collect()
    }

    /// Computes the exploitability of the current average strategy.
    /// Lower values indicate closer approximation to Nash equilibrium.
    ///
    /// For n-player games, this returns the average of each player's
    /// best response value (should converge to 0 for zero-sum games).
    pub fn exploitability<G: Game>(&self, game: &G) -> f64 {
        let num_players = game.num_players();
        let mut total = 0.0;

        for player in 0..num_players {
            // Two-pass best response for imperfect information games:
            // 1. Compute expected value of each action at each info set
            // 2. Use best action for each info set to compute final value
            let mut info_set_values: HashMap<String, Vec<f64>> = HashMap::new();
            let mut info_set_reach: HashMap<String, f64> = HashMap::new();

            let root = game.root();
            self.compute_br_action_values(
                &root,
                player,
                num_players,
                1.0,
                &mut info_set_values,
                &mut info_set_reach,
            );

            // Build best response strategy: pick best action at each info set
            let mut br_strategy: HashMap<String, usize> = HashMap::new();
            for (info_key, action_values) in &info_set_values {
                let best_action = action_values
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(i, _)| i)
                    .unwrap_or(0);
                br_strategy.insert(info_key.clone(), best_action);
            }

            // Compute value using best response strategy
            let br_value =
                self.compute_br_value(&root, player, num_players, &br_strategy);
            total += br_value;
        }

        total / num_players as f64
    }

    /// First pass: compute expected value of each action at each of the player's info sets.
    fn compute_br_action_values<N: GameNode>(
        &self,
        node: &N,
        player: usize,
        num_players: usize,
        reach_prob: f64,
        info_set_values: &mut HashMap<String, Vec<f64>>,
        info_set_reach: &mut HashMap<String, f64>,
    ) {
        if reach_prob < 1e-10 {
            return;
        }

        if node.is_terminal() {
            return;
        }

        let num_actions = node.num_actions();

        if node.is_chance() {
            let prob = 1.0 / num_actions as f64;
            for action in 0..num_actions {
                let child = node.play(action);
                self.compute_br_action_values(
                    &child,
                    player,
                    num_players,
                    reach_prob * prob,
                    info_set_values,
                    info_set_reach,
                );
            }
            return;
        }

        let current_player = node.current_player();

        if current_player == player {
            // Player's decision node: compute value of each action
            let info_key = node.info_set_key();

            // Initialize if needed
            if !info_set_values.contains_key(&info_key) {
                info_set_values.insert(info_key.clone(), vec![0.0; num_actions]);
                info_set_reach.insert(info_key.clone(), 0.0);
            }

            *info_set_reach.get_mut(&info_key).unwrap() += reach_prob;

            for action in 0..num_actions {
                let child = node.play(action);
                let action_value = self.compute_terminal_value(&child, player, num_players);
                info_set_values.get_mut(&info_key).unwrap()[action] += reach_prob * action_value;
            }
        } else {
            // Opponent's decision: follow their average strategy
            let info_key = node.info_set_key();
            let strategy = self
                .info_sets
                .get(&info_key)
                .map(|is| is.average_strategy())
                .unwrap_or_else(|| vec![1.0 / num_actions as f64; num_actions]);

            for action in 0..num_actions {
                let child = node.play(action);
                self.compute_br_action_values(
                    &child,
                    player,
                    num_players,
                    reach_prob * strategy[action],
                    info_set_values,
                    info_set_reach,
                );
            }
        }
    }

    /// Helper to compute expected terminal value from a node (opponents follow average strategy).
    fn compute_terminal_value<N: GameNode>(
        &self,
        node: &N,
        player: usize,
        num_players: usize,
    ) -> f64 {
        if node.is_terminal() {
            return node.payoff(player);
        }

        let num_actions = node.num_actions();

        if node.is_chance() {
            let mut expected_value = 0.0;
            for action in 0..num_actions {
                let child = node.play(action);
                expected_value += self.compute_terminal_value(&child, player, num_players);
            }
            return expected_value / num_actions as f64;
        }

        let info_key = node.info_set_key();

        // All players (including the best-responder) follow average strategy here
        // This is used to evaluate what happens after an action is taken
        let strategy = self
            .info_sets
            .get(&info_key)
            .map(|is| is.average_strategy())
            .unwrap_or_else(|| vec![1.0 / num_actions as f64; num_actions]);

        let mut expected_value = 0.0;
        for action in 0..num_actions {
            let child = node.play(action);
            expected_value +=
                strategy[action] * self.compute_terminal_value(&child, player, num_players);
        }
        expected_value
    }

    /// Second pass: compute value using the best response strategy.
    fn compute_br_value<N: GameNode>(
        &self,
        node: &N,
        player: usize,
        num_players: usize,
        br_strategy: &HashMap<String, usize>,
    ) -> f64 {
        if node.is_terminal() {
            return node.payoff(player);
        }

        let num_actions = node.num_actions();

        if node.is_chance() {
            let mut expected_value = 0.0;
            for action in 0..num_actions {
                let child = node.play(action);
                expected_value += self.compute_br_value(&child, player, num_players, br_strategy);
            }
            return expected_value / num_actions as f64;
        }

        let current_player = node.current_player();
        let info_key = node.info_set_key();

        if current_player == player {
            // Use best response action
            let action = *br_strategy.get(&info_key).unwrap_or(&0);
            let child = node.play(action);
            self.compute_br_value(&child, player, num_players, br_strategy)
        } else {
            // Opponent follows average strategy
            let strategy = self
                .info_sets
                .get(&info_key)
                .map(|is| is.average_strategy())
                .unwrap_or_else(|| vec![1.0 / num_actions as f64; num_actions]);

            let mut expected_value = 0.0;
            for action in 0..num_actions {
                let child = node.play(action);
                expected_value +=
                    strategy[action] * self.compute_br_value(&child, player, num_players, br_strategy);
            }
            expected_value
        }
    }

    /// Returns the current iteration count.
    pub fn iterations(&self) -> u32 {
        self.iteration
    }
}
