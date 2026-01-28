//! Action types and betting state for poker game trees.

use crate::tree::config::{BetType, Street, TreeConfig};
use crate::tree::position::{all_active_mask, blind_seats, count_active, next_actor};

/// A poker action.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Action {
    /// Fold hand
    Fold,
    /// Check (pass, no bet)
    Check,
    /// Call the current bet
    Call(i32),
    /// Make an initial bet
    Bet(i32),
    /// Raise to a total amount
    Raise(i32),
    /// Go all-in
    AllIn(i32),
}

impl Action {
    /// Get the chip amount for this action.
    pub fn amount(&self) -> i32 {
        match self {
            Action::Fold | Action::Check => 0,
            Action::Call(a) | Action::Bet(a) | Action::Raise(a) | Action::AllIn(a) => *a,
        }
    }

    /// Check if this is an aggressive action (bet/raise/all-in).
    pub fn is_aggressive(&self) -> bool {
        matches!(self, Action::Bet(_) | Action::Raise(_) | Action::AllIn(_))
    }

    /// Format action for display.
    pub fn display(&self) -> String {
        match self {
            Action::Fold => "Fold".to_string(),
            Action::Check => "Check".to_string(),
            Action::Call(a) => format!("Call {}", a),
            Action::Bet(a) => format!("Bet {}", a),
            Action::Raise(a) => format!("Raise to {}", a),
            Action::AllIn(a) => format!("All-in {}", a),
        }
    }
}

/// Result of a terminal node.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum TerminalResult {
    /// Someone folded, winner is the specified player.
    Fold { winner: usize },
    /// Showdown (all remaining players show cards).
    Showdown,
    /// All-in runout (side pots may apply).
    AllInRunout { num_players: usize },
}

/// State of a betting round in progress.
#[derive(Debug, Clone)]
pub struct BettingState {
    /// Current street.
    pub street: Street,

    /// Bitmask of active (non-folded) players.
    pub active_mask: u8,

    /// Index of the player to act.
    pub current_actor: usize,

    /// Last player to make an aggressive action (bet/raise).
    pub last_aggressor: Option<usize>,

    /// Size of the last raise (for min-raise calculation).
    pub last_raise_size: i32,

    /// Number of raises in this betting round.
    pub num_raises: u8,

    /// Current bet amount per player this round (what each player has put in).
    pub bets: Vec<i32>,

    /// Remaining stack per player.
    pub stacks: Vec<i32>,

    /// Total pot (including current round bets).
    pub pot: i32,

    /// Total number of players in the game.
    pub num_players: usize,

    /// Whether the round is complete.
    pub round_complete: bool,

    /// Whether any player is all-in.
    pub has_all_in: bool,

    /// First actor of this betting round (for detecting round completion).
    first_actor: usize,

    /// Number of players who have acted this round.
    players_acted: u8,
}

impl BettingState {
    /// Create initial betting state for a hand.
    pub fn new(config: &TreeConfig) -> Self {
        let num_players = config.num_players;
        let stacks: Vec<i32> = (0..num_players)
            .map(|i| config.stack_for_player(i))
            .collect();

        // Determine first actor preflop
        let first_actor = if num_players == 2 { 0 } else { 0 }; // UTG or BTN for HU

        let mut state = Self {
            street: Street::Preflop,
            active_mask: all_active_mask(num_players),
            current_actor: first_actor,
            last_aggressor: None,
            last_raise_size: 0,
            num_raises: 0,
            bets: vec![0; num_players],
            stacks,
            pot: 0,
            num_players,
            round_complete: false,
            has_all_in: false,
            first_actor,
            players_acted: 0,
        };

        // Post blinds if preflop
        if let Some(ref pf) = config.preflop {
            let (sb_seat, bb_seat) = blind_seats(num_players);

            // Post antes
            let ante = pf.ante;
            if ante > 0 {
                for i in 0..num_players {
                    let ante_amount = ante.min(state.stacks[i]);
                    state.stacks[i] -= ante_amount;
                    state.pot += ante_amount;
                }
            }

            // Post BB ante
            if pf.bb_ante > 0 {
                let bb_ante = pf.bb_ante.min(state.stacks[bb_seat]);
                state.stacks[bb_seat] -= bb_ante;
                state.pot += bb_ante;
            }

            // Post small blind
            let sb = pf.blinds[0].min(state.stacks[sb_seat]);
            state.stacks[sb_seat] -= sb;
            state.bets[sb_seat] = sb;
            state.pot += sb;

            // Post big blind
            let bb = pf.blinds[1].min(state.stacks[bb_seat]);
            state.stacks[bb_seat] -= bb;
            state.bets[bb_seat] = bb;
            state.pot += bb;

            // Set up for first action
            state.last_raise_size = pf.blinds[1]; // BB is the "raise" size for min-raise calc
            state.last_aggressor = Some(bb_seat); // BB is technically the last aggressor

            // First to act preflop
            if num_players == 2 {
                state.current_actor = sb_seat; // BTN/SB acts first HU
                state.first_actor = sb_seat;
            } else {
                state.current_actor = 0; // UTG acts first
                state.first_actor = 0;
            }
        }

        state
    }

    /// Create a postflop betting state (for postflop-only trees).
    pub fn new_postflop(config: &TreeConfig, street: Street, pot: i32) -> Self {
        let num_players = config.num_players;
        let stacks: Vec<i32> = (0..num_players)
            .map(|i| config.stack_for_player(i))
            .collect();

        // First to act postflop: BB for HU, SB for multiway
        let first = if num_players == 2 { 1 } else { num_players - 2 };

        Self {
            street,
            active_mask: all_active_mask(num_players),
            current_actor: first,
            last_aggressor: None,
            last_raise_size: 0,
            num_raises: 0,
            bets: vec![0; num_players],
            stacks,
            pot,
            num_players,
            round_complete: false,
            has_all_in: false,
            first_actor: first,
            players_acted: 0,
        }
    }

    /// Get the current bet to call.
    pub fn to_call(&self) -> i32 {
        let max_bet = *self.bets.iter().max().unwrap_or(&0);
        max_bet - self.bets[self.current_actor]
    }

    /// Get the minimum raise amount (raise TO, not raise BY).
    pub fn min_raise(&self) -> i32 {
        let current_bet = *self.bets.iter().max().unwrap_or(&0);
        let min_raise_to = current_bet + self.last_raise_size.max(1);
        min_raise_to.min(self.stacks[self.current_actor] + self.bets[self.current_actor])
    }

    /// Get the maximum bet amount based on bet type.
    pub fn max_bet(&self, bet_type: BetType) -> i32 {
        let stack = self.stacks[self.current_actor];
        let current_bet = self.bets[self.current_actor];

        match bet_type {
            BetType::NoLimit => stack + current_bet,
            BetType::PotLimit => {
                // Pot-limit: max raise = pot after call + pot
                let to_call = self.to_call();
                let pot_after_call = self.pot + to_call;
                let max_raise = pot_after_call + pot_after_call;
                max_raise.min(stack + current_bet)
            }
        }
    }

    /// Get the number of active (non-folded) players.
    pub fn active_count(&self) -> usize {
        count_active(self.active_mask)
    }

    /// Check if a player is active.
    pub fn is_active(&self, player: usize) -> bool {
        (self.active_mask >> player) & 1 == 1
    }

    /// Check if the betting round is complete.
    pub fn is_round_complete(&self) -> bool {
        if self.round_complete {
            return true;
        }

        // Round complete if only one player left
        if self.active_count() <= 1 {
            return true;
        }

        // Round complete if everyone has acted and action returns to last aggressor
        // or if no aggressor and action returns to first actor
        false
    }

    /// Apply an action to the state.
    pub fn apply_action(&mut self, action: Action) {
        let player = self.current_actor;

        match action {
            Action::Fold => {
                self.active_mask &= !(1 << player);
            }
            Action::Check => {
                // No change to bets
            }
            Action::Call(amount) => {
                let to_pay = amount - self.bets[player];
                self.stacks[player] -= to_pay;
                self.bets[player] = amount;
                self.pot += to_pay;

                if self.stacks[player] == 0 {
                    self.has_all_in = true;
                }
            }
            Action::Bet(amount) | Action::Raise(amount) => {
                let prev_bet = *self.bets.iter().max().unwrap_or(&0);
                let raise_size = amount - prev_bet;
                let to_pay = amount - self.bets[player];

                self.stacks[player] -= to_pay;
                self.bets[player] = amount;
                self.pot += to_pay;
                self.last_raise_size = raise_size;
                self.last_aggressor = Some(player);
                self.num_raises += 1;

                if self.stacks[player] == 0 {
                    self.has_all_in = true;
                }
            }
            Action::AllIn(amount) => {
                let prev_bet = *self.bets.iter().max().unwrap_or(&0);
                let raise_size = (amount - prev_bet).max(0);
                let to_pay = amount - self.bets[player];

                self.stacks[player] -= to_pay;
                self.bets[player] = amount;
                self.pot += to_pay;
                self.has_all_in = true;

                // Only count as raise if it's a full raise
                if raise_size >= self.last_raise_size {
                    self.last_raise_size = raise_size;
                    self.last_aggressor = Some(player);
                    self.num_raises += 1;
                }
            }
        }

        // Advance to next actor
        self.advance_actor();
    }

    /// Advance to the next betting round.
    pub fn advance_street(&mut self) {
        if let Some(next) = self.street.next() {
            self.street = next;
            self.bets = vec![0; self.num_players];
            self.last_aggressor = None;
            self.last_raise_size = 0;
            self.num_raises = 0;
            self.round_complete = false;
            self.players_acted = 0;

            // First to act postflop
            if self.num_players == 2 {
                // HU: BB (seat 1) first
                if self.is_active(1) {
                    self.current_actor = 1;
                    self.first_actor = 1;
                } else {
                    self.current_actor = 0;
                    self.first_actor = 0;
                }
            } else {
                // Multi-way: SB first (or first active from SB)
                let sb_seat = self.num_players - 2;
                self.current_actor = sb_seat;
                while !self.is_active(self.current_actor) {
                    self.current_actor = (self.current_actor + 1) % self.num_players;
                }
                self.first_actor = self.current_actor;
            }
        }
    }

    /// Advance to next actor.
    fn advance_actor(&mut self) {
        self.players_acted += 1;

        if let Some(next) = next_actor(self.current_actor, self.num_players, self.active_mask) {
            // Check if we've returned to the last aggressor AND everyone has acted
            // (Preflop, BB is the "aggressor" but hasn't actually acted - they have the "option")
            if let Some(agg) = self.last_aggressor {
                if next == agg && self.players_acted >= self.active_count() as u8 {
                    self.round_complete = true;
                    return;
                }
            }

            // If no aggression, round is complete when everyone has acted once
            // and we return to the first actor
            if self.last_aggressor.is_none()
                && self.players_acted >= self.active_count() as u8
                && next == self.first_actor
            {
                self.round_complete = true;
                return;
            }

            self.current_actor = next;
        } else {
            self.round_complete = true;
        }
    }

    /// Get legal actions for the current player.
    pub fn legal_actions(&self, config: &TreeConfig) -> Vec<Action> {
        let mut actions = Vec::new();
        let player = self.current_actor;
        let stack = self.stacks[player];
        let current_bet = self.bets[player];
        let to_call = self.to_call();

        // Fold is always legal if there's a bet to call
        if to_call > 0 {
            actions.push(Action::Fold);
        }

        // Check or call
        if to_call == 0 {
            actions.push(Action::Check);
        } else if to_call <= stack {
            actions.push(Action::Call(current_bet + to_call));
        } else {
            // Partial call (all-in for less)
            actions.push(Action::AllIn(current_bet + stack));
        }

        // Bet/raise (if not capped and have chips)
        if self.num_raises < config.max_raises_per_round && stack > to_call {
            let min_raise = self.min_raise();
            let max_bet = self.max_bet(config.bet_type);

            if min_raise <= max_bet {
                // Can make a legal raise
                if to_call == 0 {
                    // Opening bet
                    if min_raise < stack + current_bet {
                        actions.push(Action::Bet(min_raise));
                    }
                } else {
                    // Raise
                    if min_raise < stack + current_bet {
                        actions.push(Action::Raise(min_raise));
                    }
                }

                // All-in is always an option if we have chips
                if stack + current_bet > *self.bets.iter().max().unwrap_or(&0) {
                    actions.push(Action::AllIn(stack + current_bet));
                }
            }
        } else if stack > 0 && to_call == 0 {
            // Can still go all-in as a bet even if raises capped
            actions.push(Action::AllIn(stack + current_bet));
        }

        actions
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tree::config::PreflopConfig;

    #[test]
    fn test_action_amount() {
        assert_eq!(Action::Fold.amount(), 0);
        assert_eq!(Action::Check.amount(), 0);
        assert_eq!(Action::Call(50).amount(), 50);
        assert_eq!(Action::Bet(100).amount(), 100);
        assert_eq!(Action::Raise(200).amount(), 200);
        assert_eq!(Action::AllIn(500).amount(), 500);
    }

    #[test]
    fn test_betting_state_preflop() {
        let config = TreeConfig::new(2)
            .with_stack(100)
            .with_preflop(PreflopConfig::new(1, 2));

        let state = BettingState::new(&config);

        assert_eq!(state.num_players, 2);
        assert_eq!(state.pot, 3); // SB + BB
        assert_eq!(state.bets[0], 1); // SB (BTN in HU)
        assert_eq!(state.bets[1], 2); // BB
        assert_eq!(state.stacks[0], 99); // 100 - 1 SB
        assert_eq!(state.stacks[1], 98); // 100 - 2 BB
        assert_eq!(state.current_actor, 0); // BTN acts first HU
    }

    #[test]
    fn test_to_call() {
        let config = TreeConfig::new(2)
            .with_stack(100)
            .with_preflop(PreflopConfig::new(1, 2));

        let state = BettingState::new(&config);
        assert_eq!(state.to_call(), 1); // BTN needs to call 1 more (has 1 in, BB has 2)
    }

    #[test]
    fn test_min_raise() {
        let config = TreeConfig::new(2)
            .with_stack(100)
            .with_preflop(PreflopConfig::new(1, 2));

        let state = BettingState::new(&config);
        // Min raise is current bet (2) + last raise size (2) = 4
        assert_eq!(state.min_raise(), 4);
    }

    #[test]
    fn test_apply_action_call() {
        let config = TreeConfig::new(2)
            .with_stack(100)
            .with_preflop(PreflopConfig::new(1, 2));

        let mut state = BettingState::new(&config);
        state.apply_action(Action::Call(2));

        assert_eq!(state.bets[0], 2);
        assert_eq!(state.stacks[0], 98);
        assert_eq!(state.pot, 4);
    }

    #[test]
    fn test_apply_action_raise() {
        let config = TreeConfig::new(2)
            .with_stack(100)
            .with_preflop(PreflopConfig::new(1, 2));

        let mut state = BettingState::new(&config);
        state.apply_action(Action::Raise(6)); // BTN raises to 6

        assert_eq!(state.bets[0], 6);
        assert_eq!(state.stacks[0], 94);
        assert_eq!(state.pot, 8); // 1 + 2 + 5 = 8
        assert_eq!(state.last_raise_size, 4); // Raised by 4 (from 2 to 6)
        assert_eq!(state.num_raises, 1);
    }

    #[test]
    fn test_apply_action_fold() {
        let config = TreeConfig::new(2)
            .with_stack(100)
            .with_preflop(PreflopConfig::new(1, 2));

        let mut state = BettingState::new(&config);
        state.apply_action(Action::Fold);

        assert!(!state.is_active(0));
        assert_eq!(state.active_count(), 1);
    }

    #[test]
    fn test_legal_actions_preflop() {
        let config = TreeConfig::new(2)
            .with_stack(100)
            .with_preflop(PreflopConfig::new(1, 2));

        let state = BettingState::new(&config);
        let actions = state.legal_actions(&config);

        // BTN can fold, call, raise, or all-in
        assert!(actions.contains(&Action::Fold));
        assert!(actions.contains(&Action::Call(2)));
        assert!(actions.iter().any(|a| matches!(a, Action::Raise(_))));
        assert!(actions.iter().any(|a| matches!(a, Action::AllIn(_))));
    }

    #[test]
    fn test_bb_min_raise_after_sb_call() {
        let config = TreeConfig::new(2)
            .with_stack(100)
            .with_preflop(PreflopConfig::new(1, 2));

        let mut state = BettingState::new(&config);

        // Initial: SB acts first, min raise is 4 (current_bet 2 + last_raise 2)
        assert_eq!(state.current_actor, 0); // SB/BTN
        assert_eq!(state.min_raise(), 4);

        // SB calls to 2
        state.apply_action(Action::Call(2));

        // Now BB acts
        assert_eq!(state.current_actor, 1); // BB
        assert_eq!(state.bets, vec![2, 2]);
        assert_eq!(state.last_raise_size, 2); // Unchanged from initial

        // BB's min raise should still be 4 (2 + 2), NOT 3
        assert_eq!(state.min_raise(), 4);

        // Legal actions for BB
        let actions = state.legal_actions(&config);

        // BB can check (bets are equal)
        assert!(actions.contains(&Action::Check));

        // Any raise must be at least to 4
        for action in &actions {
            if let Action::Raise(amount) = action {
                assert!(*amount >= 4, "Raise to {} is below min raise of 4", amount);
            }
            if let Action::Bet(amount) = action {
                assert!(*amount >= 4, "Bet of {} is below min raise of 4", amount);
            }
        }
    }
}
