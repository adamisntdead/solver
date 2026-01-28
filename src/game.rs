/// Trait representing a node in the game tree.
///
/// Each node corresponds to a game state where a decision must be made
/// (by a player or by chance), or the game has ended (terminal).
pub trait GameNode: Clone {
    /// Returns `true` if this is a terminal node (game over).
    fn is_terminal(&self) -> bool;

    /// Returns `true` if this is a chance node (nature/random action).
    fn is_chance(&self) -> bool;

    /// Returns the current player to act (0 or 1 for two-player games).
    /// Only valid when `!is_terminal() && !is_chance()`.
    fn current_player(&self) -> usize;

    /// Returns the number of available actions at this node.
    fn num_actions(&self) -> usize;

    /// Returns the node reached after taking the given action.
    fn play(&self, action: usize) -> Self;

    /// Returns the payoff for the given player at a terminal node.
    /// Only valid when `is_terminal()` returns `true`.
    fn payoff(&self, player: usize) -> f64;

    /// Returns the information set ID for the current player.
    /// Nodes in the same information set are indistinguishable to the player.
    /// IDs must be in range `0..Game::num_info_sets()`.
    fn info_set_id(&self) -> usize;

    /// Returns the probability of a chance action at a chance node.
    /// Only valid when `is_chance()` returns `true`.
    /// Default implementation returns uniform probability (1/num_actions).
    fn chance_prob(&self, _action: usize) -> f64 {
        1.0 / self.num_actions() as f64
    }
}

/// Trait representing a game (typically two-player zero-sum).
pub trait Game {
    /// The type of nodes in this game's tree.
    type Node: GameNode;

    /// Returns the root node of the game tree.
    fn root(&self) -> Self::Node;

    /// Returns the number of players (typically 2).
    fn num_players(&self) -> usize {
        2
    }

    /// Returns the total number of information sets in the game.
    /// Used to pre-allocate storage for strategies.
    fn num_info_sets(&self) -> usize;
}
