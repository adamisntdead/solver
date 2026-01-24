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

    /// Returns the information set key for the current player.
    /// Nodes in the same information set are indistinguishable to the player.
    fn info_set_key(&self) -> String;
}

/// Trait representing a two-player zero-sum game.
pub trait Game {
    /// The type of nodes in this game's tree.
    type Node: GameNode;

    /// Returns the root node of the game tree.
    fn root(&self) -> Self::Node;

    /// Returns the number of players (typically 2).
    fn num_players(&self) -> usize {
        2
    }
}
