use crate::game::{Game, GameNode};
use rand::prelude::*;
use rand::rngs::SmallRng;

/// Information set data tracked during CFR.
#[derive(Clone, Default)]
struct InfoSetData {
    /// Cumulative regrets for each action.
    regret_sum: Vec<f64>,
    /// Cumulative strategy weights for each action.
    strategy_sum: Vec<f64>,
    /// Number of actions available at this info set.
    num_actions: usize,
    /// Whether this info set has been initialized.
    initialized: bool,
}

impl InfoSetData {
    fn init(&mut self, num_actions: usize) {
        if !self.initialized {
            self.regret_sum = vec![0.0; num_actions];
            self.strategy_sum = vec![0.0; num_actions];
            self.num_actions = num_actions;
            self.initialized = true;
        }
    }

    /// Computes the current strategy using regret matching.
    fn current_strategy(&self, output: &mut [f64]) {
        let num_actions = self.num_actions;

        // Sum of positive regrets
        let normalizing_sum: f64 = self.regret_sum.iter().map(|&r| r.max(0.0)).sum();

        if normalizing_sum > 0.0 {
            for (i, &regret) in self.regret_sum.iter().enumerate() {
                output[i] = regret.max(0.0) / normalizing_sum;
            }
        } else {
            let uniform_prob = 1.0 / num_actions as f64;
            for i in 0..num_actions {
                output[i] = uniform_prob;
            }
        }
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
            let uniform_prob = 1.0 / self.num_actions as f64;
            for prob in &mut avg_strategy {
                *prob = uniform_prob;
            }
        }

        avg_strategy
    }
}

/// CFR algorithm variant.
#[derive(Clone, Copy, Default)]
pub enum CfrVariant {
    /// CFR+ with regret matching+ and linear averaging.
    /// Fastest convergence for most games.
    #[default]
    CfrPlus,
    /// Linear CFR: both regrets and strategies use t/(t+1) discounting.
    /// Good balance of speed and stability.
    LinearCfr,
    /// Discounted CFR with configurable discount parameters.
    Discounted(DiscountParams),
}

/// Discount parameters for Discounted CFR.
#[derive(Clone, Copy)]
pub struct DiscountParams {
    pub alpha: f64,
    pub beta: f64,
    pub gamma: f64,
}

impl Default for DiscountParams {
    fn default() -> Self {
        Self {
            alpha: 1.5,
            beta: 0.0,
            gamma: 2.0,
        }
    }
}

impl DiscountParams {
    fn positive_regret_discount(&self, t: u32) -> f64 {
        let t = t as f64;
        t.powf(self.alpha) / (t.powf(self.alpha) + 1.0)
    }

    fn negative_regret_discount(&self, t: u32) -> f64 {
        let t = t as f64;
        t.powf(self.beta) / (t.powf(self.beta) + 1.0)
    }

    fn strategy_discount(&self, t: u32) -> f64 {
        let t = t as f64;
        (t / (t + 1.0)).powf(self.gamma)
    }
}

/// CFR solver for n-player games.
/// Supports CFR+ (default, fastest) and Discounted CFR variants.
pub struct CfrSolver {
    /// Data for each information set, indexed by info set ID.
    info_sets: Vec<InfoSetData>,
    /// CFR variant to use.
    variant: CfrVariant,
    /// Current iteration number.
    iteration: u32,
}

impl CfrSolver {
    /// Creates a new CFR solver with the specified variant.
    pub fn new<G: Game>(game: &G, variant: CfrVariant) -> Self {
        let num_info_sets = game.num_info_sets();
        Self {
            info_sets: vec![InfoSetData::default(); num_info_sets],
            variant,
            iteration: 0,
        }
    }

    /// Creates a new CFR+ solver (fastest convergence).
    pub fn new_with_defaults<G: Game>(game: &G) -> Self {
        Self::new(game, CfrVariant::CfrPlus)
    }

    /// Creates a new Discounted CFR solver.
    pub fn new_discounted<G: Game>(game: &G, params: DiscountParams) -> Self {
        Self::new(game, CfrVariant::Discounted(params))
    }

    /// Runs CFR for the specified number of iterations.
    ///
    /// This uses full enumeration of chance nodes, which can be slow for games
    /// with many chance outcomes. For such games, use `train_sampled` instead.
    ///
    /// Uses alternating updates: each iteration updates only one player.
    pub fn train<G: Game>(&mut self, game: &G, iterations: u32) {
        let num_players = game.num_players();
        let mut reach_probs = vec![1.0; num_players];
        // Scratch space for strategy computation (max 10 actions should cover most games)
        let mut strategy_buf = [0.0f64; 10];

        for i in 0..iterations {
            self.iteration += 1;

            // Alternating updates: only update one player per iteration
            let player = (i as usize) % num_players;
            let root = game.root();
            reach_probs.fill(1.0);
            self.cfr(&root, player, num_players, &reach_probs, &mut strategy_buf);

            self.post_iteration_update();
        }
    }

    /// Runs Monte Carlo CFR with chance sampling.
    ///
    /// Instead of enumerating all chance outcomes, this samples one outcome per
    /// iteration, making it much faster for games with many chance outcomes
    /// (like poker deals). Convergence is still guaranteed in expectation.
    ///
    /// Uses alternating updates: each iteration updates only one player,
    /// cycling through players. This matches Gambit's approach and improves
    /// convergence for zero-sum games.
    pub fn train_sampled<G: Game>(&mut self, game: &G, iterations: u32) {
        let num_players = game.num_players();
        let mut reach_probs = vec![1.0; num_players];
        let mut strategy_buf = [0.0f64; 10];
        let mut rng = SmallRng::from_entropy();

        for i in 0..iterations {
            self.iteration += 1;

            // Alternating updates: only update one player per iteration
            let player = (i as usize) % num_players;
            let root = game.root();
            reach_probs.fill(1.0);
            self.cfr_sampled(&root, player, num_players, &reach_probs, &mut strategy_buf, &mut rng);

            self.post_iteration_update();
        }
    }

    /// Applies post-iteration updates based on CFR variant.
    fn post_iteration_update(&mut self) {
        match self.variant {
            CfrVariant::CfrPlus => {
                // CFR+: Floor all regrets at 0 (regret matching+)
                // Strategy averaging is handled in cfr_sampled with linear weighting
                for info_set in &mut self.info_sets {
                    if !info_set.initialized {
                        continue;
                    }
                    for regret in &mut info_set.regret_sum {
                        if *regret < 0.0 {
                            *regret = 0.0;
                        }
                    }
                }
            }
            CfrVariant::LinearCfr => {
                // Linear CFR: Apply t/(t+1) discount to both regrets and strategies
                // This gives more weight to recent iterations
                let t = self.iteration as f64;
                let discount = t / (t + 1.0);

                for info_set in &mut self.info_sets {
                    if !info_set.initialized {
                        continue;
                    }
                    for regret in &mut info_set.regret_sum {
                        *regret *= discount;
                    }
                    for sum in &mut info_set.strategy_sum {
                        *sum *= discount;
                    }
                }
            }
            CfrVariant::Discounted(params) => {
                self.apply_discounts_with_params(&params);
            }
        }
    }

    fn apply_discounts_with_params(&mut self, params: &DiscountParams) {
        let pos_discount = params.positive_regret_discount(self.iteration);
        let neg_discount = params.negative_regret_discount(self.iteration);
        let strat_discount = params.strategy_discount(self.iteration);

        for info_set in &mut self.info_sets {
            if !info_set.initialized {
                continue;
            }
            for regret in &mut info_set.regret_sum {
                if *regret > 0.0 {
                    *regret *= pos_discount;
                } else {
                    *regret *= neg_discount;
                }
            }
            for sum in &mut info_set.strategy_sum {
                *sum *= strat_discount;
            }
        }
    }

    fn cfr<N: GameNode>(
        &mut self,
        node: &N,
        player: usize,
        num_players: usize,
        reach_probs: &[f64],
        strategy_buf: &mut [f64; 10],
    ) -> f64 {
        if node.is_terminal() {
            return node.payoff(player);
        }

        let num_actions = node.num_actions();

        if node.is_chance() {
            let mut expected_value = 0.0;
            for action in 0..num_actions {
                let child = node.play(action);
                expected_value += self.cfr(&child, player, num_players, reach_probs, strategy_buf);
            }
            return expected_value / num_actions as f64;
        }

        let current_player = node.current_player();
        let info_id = node.info_set_id();

        // Initialize info set if needed and get current strategy
        self.info_sets[info_id].init(num_actions);
        self.info_sets[info_id].current_strategy(&mut strategy_buf[..num_actions]);

        // Compute action values
        let mut action_values = [0.0f64; 10];
        let mut node_value = 0.0;

        let mut child_reach_probs = [0.0f64; 8];
        child_reach_probs[..num_players].copy_from_slice(&reach_probs[..num_players]);
        let original_reach = child_reach_probs[current_player];

        for action in 0..num_actions {
            let child = node.play(action);
            let action_prob = strategy_buf[action];

            child_reach_probs[current_player] = original_reach * action_prob;

            let child_value = self.cfr(
                &child,
                player,
                num_players,
                &child_reach_probs[..num_players],
                strategy_buf,
            );

            action_values[action] = child_value;
            node_value += action_prob * child_value;
        }

        if current_player == player {
            let cf_reach: f64 = reach_probs
                .iter()
                .enumerate()
                .filter(|&(i, _)| i != player)
                .map(|(_, &p)| p)
                .product();

            let info_set = &mut self.info_sets[info_id];

            for action in 0..num_actions {
                let regret = action_values[action] - node_value;
                info_set.regret_sum[action] += cf_reach * regret;
            }

            // Recompute strategy since buffer may have been overwritten by recursion
            info_set.current_strategy(&mut strategy_buf[..num_actions]);

            // CFR+ uses linear averaging (weight by iteration t)
            // Linear CFR and Discounted CFR use uniform weighting (discounting applied post-iteration)
            let weight = match self.variant {
                CfrVariant::CfrPlus => self.iteration as f64,
                CfrVariant::LinearCfr | CfrVariant::Discounted(_) => 1.0,
            };

            for action in 0..num_actions {
                info_set.strategy_sum[action] += weight * reach_probs[player] * strategy_buf[action];
            }
        }

        node_value
    }

    /// CFR traversal with chance sampling (External Sampling MCCFR).
    fn cfr_sampled<N: GameNode>(
        &mut self,
        node: &N,
        player: usize,
        num_players: usize,
        reach_probs: &[f64],
        strategy_buf: &mut [f64; 10],
        rng: &mut SmallRng,
    ) -> f64 {
        if node.is_terminal() {
            return node.payoff(player);
        }

        let num_actions = node.num_actions();

        if node.is_chance() {
            // Sample a single chance outcome instead of enumerating all
            let action = rng.gen_range(0..num_actions);
            let child = node.play(action);
            return self.cfr_sampled(&child, player, num_players, reach_probs, strategy_buf, rng);
        }

        let current_player = node.current_player();
        let info_id = node.info_set_id();

        // Initialize info set if needed and get current strategy
        self.info_sets[info_id].init(num_actions);
        self.info_sets[info_id].current_strategy(&mut strategy_buf[..num_actions]);

        // Compute action values
        let mut action_values = [0.0f64; 10];
        let mut node_value = 0.0;

        let mut child_reach_probs = [0.0f64; 8];
        child_reach_probs[..num_players].copy_from_slice(&reach_probs[..num_players]);
        let original_reach = child_reach_probs[current_player];

        for action in 0..num_actions {
            let child = node.play(action);
            let action_prob = strategy_buf[action];

            child_reach_probs[current_player] = original_reach * action_prob;

            let child_value = self.cfr_sampled(
                &child,
                player,
                num_players,
                &child_reach_probs[..num_players],
                strategy_buf,
                rng,
            );

            action_values[action] = child_value;
            node_value += action_prob * child_value;
        }

        if current_player == player {
            let cf_reach: f64 = reach_probs
                .iter()
                .enumerate()
                .filter(|&(i, _)| i != player)
                .map(|(_, &p)| p)
                .product();

            let info_set = &mut self.info_sets[info_id];

            for action in 0..num_actions {
                let regret = action_values[action] - node_value;
                info_set.regret_sum[action] += cf_reach * regret;
            }

            // Recompute strategy since buffer may have been overwritten by recursion
            info_set.current_strategy(&mut strategy_buf[..num_actions]);

            // CFR+ uses linear averaging (weight by iteration t)
            // Linear CFR and Discounted CFR use uniform weighting (discounting applied post-iteration)
            let weight = match self.variant {
                CfrVariant::CfrPlus => self.iteration as f64,
                CfrVariant::LinearCfr | CfrVariant::Discounted(_) => 1.0,
            };

            for action in 0..num_actions {
                info_set.strategy_sum[action] += weight * reach_probs[player] * strategy_buf[action];
            }
        }

        node_value
    }

    /// Returns the average strategy for a given information set ID.
    pub fn get_strategy(&self, info_set_id: usize) -> Option<Vec<f64>> {
        self.info_sets
            .get(info_set_id)
            .filter(|is| is.initialized)
            .map(|is| is.average_strategy())
    }

    /// Computes the exploitability of the current average strategy.
    pub fn exploitability<G: Game>(&self, game: &G) -> f64 {
        let num_players = game.num_players();
        let num_info_sets = game.num_info_sets();
        let mut total = 0.0;

        for player in 0..num_players {
            let mut info_set_values: Vec<Vec<f64>> = vec![Vec::new(); num_info_sets];
            let mut info_set_reach: Vec<f64> = vec![0.0; num_info_sets];

            let root = game.root();
            self.compute_br_action_values(
                &root,
                player,
                num_players,
                1.0,
                &mut info_set_values,
                &mut info_set_reach,
            );

            let mut br_strategy: Vec<usize> = vec![0; num_info_sets];
            for (info_id, action_values) in info_set_values.iter().enumerate() {
                if !action_values.is_empty() {
                    let best_action = action_values
                        .iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                        .map(|(i, _)| i)
                        .unwrap_or(0);
                    br_strategy[info_id] = best_action;
                }
            }

            let br_value = self.compute_br_value(&root, player, num_players, &br_strategy);
            total += br_value;
        }

        total / num_players as f64
    }

    fn compute_br_action_values<N: GameNode>(
        &self,
        node: &N,
        player: usize,
        num_players: usize,
        reach_prob: f64,
        info_set_values: &mut [Vec<f64>],
        info_set_reach: &mut [f64],
    ) {
        if reach_prob < 1e-10 || node.is_terminal() {
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
            let info_id = node.info_set_id();

            if info_set_values[info_id].is_empty() {
                info_set_values[info_id] = vec![0.0; num_actions];
            }

            info_set_reach[info_id] += reach_prob;

            for action in 0..num_actions {
                let child = node.play(action);
                let action_value = self.compute_terminal_value(&child, player, num_players);
                info_set_values[info_id][action] += reach_prob * action_value;
            }
        } else {
            let info_id = node.info_set_id();
            let strategy = self
                .info_sets
                .get(info_id)
                .filter(|is| is.initialized)
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

        let info_id = node.info_set_id();
        let strategy = self
            .info_sets
            .get(info_id)
            .filter(|is| is.initialized)
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

    fn compute_br_value<N: GameNode>(
        &self,
        node: &N,
        player: usize,
        num_players: usize,
        br_strategy: &[usize],
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
        let info_id = node.info_set_id();

        if current_player == player {
            let action = br_strategy[info_id];
            let child = node.play(action);
            self.compute_br_value(&child, player, num_players, br_strategy)
        } else {
            let strategy = self
                .info_sets
                .get(info_id)
                .filter(|is| is.initialized)
                .map(|is| is.average_strategy())
                .unwrap_or_else(|| vec![1.0 / num_actions as f64; num_actions]);

            let mut expected_value = 0.0;
            for action in 0..num_actions {
                let child = node.play(action);
                expected_value += strategy[action]
                    * self.compute_br_value(&child, player, num_players, br_strategy);
            }
            expected_value
        }
    }

    /// Returns the current iteration count.
    pub fn iterations(&self) -> u32 {
        self.iteration
    }
}
