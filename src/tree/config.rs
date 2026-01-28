//! Tree configuration types for poker game trees.

use crate::tree::bet_size::BetSizeOptions;

/// Betting structure type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum BetType {
    /// No limit - can bet any amount up to stack
    #[default]
    NoLimit,
    /// Pot limit - max bet is current pot size
    PotLimit,
}

/// Street (betting round) in poker.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Street {
    #[default]
    Preflop,
    Flop,
    Turn,
    River,
}

impl Street {
    /// Get the number of streets remaining after this one.
    pub fn streets_remaining(&self) -> u8 {
        match self {
            Street::Preflop => 4,
            Street::Flop => 3,
            Street::Turn => 2,
            Street::River => 1,
        }
    }

    /// Get the next street, if any.
    pub fn next(&self) -> Option<Street> {
        match self {
            Street::Preflop => Some(Street::Flop),
            Street::Flop => Some(Street::Turn),
            Street::Turn => Some(Street::River),
            Street::River => None,
        }
    }

    /// Get short name for display.
    pub fn short_name(&self) -> &'static str {
        match self {
            Street::Preflop => "Pre",
            Street::Flop => "Flop",
            Street::Turn => "Turn",
            Street::River => "River",
        }
    }
}

/// Configuration for a single street's bet sizes.
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct StreetConfig {
    /// Bet sizes per position (indexed by seat, 0 = first to act postflop).
    /// If fewer entries than players, the last entry is used for remaining positions.
    pub sizes: Vec<BetSizeOptions>,

    /// Donk bet sizes (optional, for OOP player after IP closed action).
    /// If None, donk bets are not allowed.
    pub donk_sizes: Option<Vec<BetSizeOptions>>,
}

impl StreetConfig {
    /// Create a config where all positions use the same bet sizes.
    pub fn uniform(sizes: BetSizeOptions) -> Self {
        Self {
            sizes: vec![sizes],
            donk_sizes: None,
        }
    }

    /// Create a config with donk bets enabled.
    pub fn with_donk(mut self, donk_sizes: Vec<BetSizeOptions>) -> Self {
        self.donk_sizes = Some(donk_sizes);
        self
    }

    /// Get bet sizes for a specific seat.
    pub fn sizes_for_seat(&self, seat: usize) -> &BetSizeOptions {
        if seat < self.sizes.len() {
            &self.sizes[seat]
        } else {
            self.sizes.last().unwrap_or(&DEFAULT_BET_SIZES)
        }
    }
}

static DEFAULT_BET_SIZES: BetSizeOptions = BetSizeOptions {
    bet: Vec::new(),
    raise: Vec::new(),
    reraise: Vec::new(),
    reraise_plus: Vec::new(),
};

/// Configuration for preflop betting.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PreflopConfig {
    /// Blind amounts [SB, BB] in chips.
    pub blinds: [i32; 2],

    /// Ante per player (0 if none).
    pub ante: i32,

    /// Big blind ante (single ante from BB position, 0 if none).
    pub bb_ante: i32,

    /// Allow limps (call BB without raise).
    /// Typically true in cash games, false in some tournament formats.
    pub allow_limps: bool,

    /// Open-raise sizes per position.
    /// Indexed by position: UTG=0, ..., BTN=n-3, SB=n-2, BB=n-1
    pub open_sizes: Vec<BetSizeOptions>,

    /// 3-bet sizes (vs open).
    pub three_bet_sizes: Vec<BetSizeOptions>,

    /// 4-bet+ sizes.
    pub four_bet_plus_sizes: Vec<BetSizeOptions>,

    /// Allow cold-calling (calling open when others already in pot).
    pub allow_cold_call: bool,
}

impl Default for PreflopConfig {
    fn default() -> Self {
        Self {
            blinds: [1, 2],
            ante: 0,
            bb_ante: 0,
            allow_limps: true,
            open_sizes: vec![BetSizeOptions::default()],
            three_bet_sizes: vec![BetSizeOptions::default()],
            four_bet_plus_sizes: vec![BetSizeOptions::default()],
            allow_cold_call: true,
        }
    }
}

impl PreflopConfig {
    /// Create a simple preflop config with given blinds.
    pub fn new(sb: i32, bb: i32) -> Self {
        Self {
            blinds: [sb, bb],
            ..Default::default()
        }
    }

    /// Set ante per player.
    pub fn with_ante(mut self, ante: i32) -> Self {
        self.ante = ante;
        self
    }

    /// Set big blind ante.
    pub fn with_bb_ante(mut self, bb_ante: i32) -> Self {
        self.bb_ante = bb_ante;
        self
    }

    /// Set open raise sizes.
    pub fn with_open_sizes(mut self, sizes: Vec<BetSizeOptions>) -> Self {
        self.open_sizes = sizes;
        self
    }

    /// Set 3-bet sizes.
    pub fn with_3bet_sizes(mut self, sizes: Vec<BetSizeOptions>) -> Self {
        self.three_bet_sizes = sizes;
        self
    }

    /// Set 4-bet+ sizes.
    pub fn with_4bet_sizes(mut self, sizes: Vec<BetSizeOptions>) -> Self {
        self.four_bet_plus_sizes = sizes;
        self
    }

    /// Get the total ante amount for the hand.
    pub fn total_ante(&self, num_players: usize) -> i32 {
        if self.bb_ante > 0 {
            self.bb_ante
        } else {
            self.ante * num_players as i32
        }
    }

    /// Get open sizes for a specific seat.
    pub fn open_sizes_for_seat(&self, seat: usize) -> &BetSizeOptions {
        if seat < self.open_sizes.len() {
            &self.open_sizes[seat]
        } else {
            self.open_sizes.last().unwrap_or(&DEFAULT_BET_SIZES)
        }
    }

    /// Get 3-bet sizes for a specific seat.
    pub fn three_bet_sizes_for_seat(&self, seat: usize) -> &BetSizeOptions {
        if seat < self.three_bet_sizes.len() {
            &self.three_bet_sizes[seat]
        } else {
            self.three_bet_sizes.last().unwrap_or(&DEFAULT_BET_SIZES)
        }
    }

    /// Get 4-bet+ sizes for a specific seat.
    pub fn four_bet_sizes_for_seat(&self, seat: usize) -> &BetSizeOptions {
        if seat < self.four_bet_plus_sizes.len() {
            &self.four_bet_plus_sizes[seat]
        } else {
            self.four_bet_plus_sizes.last().unwrap_or(&DEFAULT_BET_SIZES)
        }
    }
}

/// Complete tree configuration.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TreeConfig {
    // === Game Setup ===
    /// Number of players (2-6).
    pub num_players: usize,

    /// Starting stack per player (in chips).
    /// If a single value, all players start with the same stack.
    pub starting_stacks: Vec<i32>,

    /// Starting street for the tree. If None, starts at preflop if configured,
    /// otherwise the first configured postflop street.
    pub starting_street: Option<Street>,

    /// Starting pot for postflop trees. Ignored when starting from preflop.
    pub starting_pot: i32,

    // === Betting Structure ===
    /// Betting type (NoLimit or PotLimit).
    pub bet_type: BetType,

    // === Per-Street Configuration ===
    /// Preflop configuration. None for postflop-only trees.
    pub preflop: Option<PreflopConfig>,

    /// Flop bet sizes. None to skip flop.
    pub flop: Option<StreetConfig>,

    /// Turn bet sizes. None to skip turn.
    pub turn: Option<StreetConfig>,

    /// River bet sizes. None to skip river.
    pub river: Option<StreetConfig>,

    // === Tree Pruning ===
    /// Maximum number of raises per betting round (default 4).
    pub max_raises_per_round: u8,

    /// Force all-in if bet leaves less than this fraction of stack (default 0.15).
    /// E.g., 0.15 means force all-in if bet leaves < 15% of remaining stack.
    pub force_all_in_threshold: f64,

    /// Merge similar bet sizes using Pio algorithm (default 0.1).
    /// Higher values merge more aggressively.
    pub merge_threshold: f64,

    /// Add all-in if max bet size is less than this multiple of pot (default 1.5).
    /// E.g., 1.5 means add all-in if largest bet is < 150% pot.
    pub add_all_in_threshold: f64,
}

impl Default for TreeConfig {
    fn default() -> Self {
        Self {
            num_players: 2,
            starting_stacks: vec![100],
            starting_street: None,
            starting_pot: 0,
            bet_type: BetType::NoLimit,
            preflop: None,
            flop: None,
            turn: None,
            river: None,
            max_raises_per_round: 4,
            force_all_in_threshold: 0.15,
            merge_threshold: 0.1,
            add_all_in_threshold: 1.5,
        }
    }
}

impl TreeConfig {
    /// Create a new tree config for the given number of players.
    pub fn new(num_players: usize) -> Self {
        Self {
            num_players,
            ..Default::default()
        }
    }

    /// Set starting stacks (same for all players).
    pub fn with_stack(mut self, stack: i32) -> Self {
        self.starting_stacks = vec![stack];
        self
    }

    /// Set starting stacks per player.
    pub fn with_stacks(mut self, stacks: Vec<i32>) -> Self {
        self.starting_stacks = stacks;
        self
    }

    /// Set starting street (for postflop trees).
    pub fn with_starting_street(mut self, street: Street) -> Self {
        self.starting_street = Some(street);
        self
    }

    /// Set starting pot (for postflop trees).
    pub fn with_starting_pot(mut self, pot: i32) -> Self {
        self.starting_pot = pot;
        self
    }

    /// Set bet type.
    pub fn with_bet_type(mut self, bet_type: BetType) -> Self {
        self.bet_type = bet_type;
        self
    }

    /// Set preflop config.
    pub fn with_preflop(mut self, preflop: PreflopConfig) -> Self {
        self.preflop = Some(preflop);
        self
    }

    /// Set flop config.
    pub fn with_flop(mut self, flop: StreetConfig) -> Self {
        self.flop = Some(flop);
        self
    }

    /// Set turn config.
    pub fn with_turn(mut self, turn: StreetConfig) -> Self {
        self.turn = Some(turn);
        self
    }

    /// Set river config.
    pub fn with_river(mut self, river: StreetConfig) -> Self {
        self.river = Some(river);
        self
    }

    /// Get the starting stack for a player.
    pub fn stack_for_player(&self, player: usize) -> i32 {
        if player < self.starting_stacks.len() {
            self.starting_stacks[player]
        } else {
            *self.starting_stacks.last().unwrap_or(&100)
        }
    }

    /// Get the effective stack (minimum of all players).
    pub fn effective_stack(&self) -> i32 {
        if self.starting_stacks.len() == 1 {
            self.starting_stacks[0]
        } else {
            let mut stacks: Vec<i32> = (0..self.num_players)
                .map(|i| self.stack_for_player(i))
                .collect();
            stacks.sort();
            stacks[0] // Return smallest stack
        }
    }

    /// Get the starting pot size (blinds + antes).
    pub fn starting_pot(&self) -> i32 {
        match &self.preflop {
            Some(pf) => pf.blinds[0] + pf.blinds[1] + pf.total_ante(self.num_players),
            None => 0,
        }
    }

    /// Get config for a specific street.
    pub fn street_config(&self, street: Street) -> Option<&StreetConfig> {
        match street {
            Street::Preflop => None, // Preflop uses PreflopConfig
            Street::Flop => self.flop.as_ref(),
            Street::Turn => self.turn.as_ref(),
            Street::River => self.river.as_ref(),
        }
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<(), String> {
        if self.num_players < 2 || self.num_players > 6 {
            return Err(format!(
                "Number of players must be 2-6, got {}",
                self.num_players
            ));
        }

        if self.starting_stacks.is_empty() {
            return Err("Starting stacks cannot be empty".to_string());
        }

        for (i, &stack) in self.starting_stacks.iter().enumerate() {
            if stack <= 0 {
                return Err(format!("Stack for player {} must be positive: {}", i, stack));
            }
        }

        if let Some(ref pf) = self.preflop {
            if pf.blinds[0] <= 0 || pf.blinds[1] <= 0 {
                return Err(format!(
                    "Blinds must be positive: SB={}, BB={}",
                    pf.blinds[0], pf.blinds[1]
                ));
            }
            if pf.blinds[0] >= pf.blinds[1] {
                return Err(format!(
                    "Small blind must be less than big blind: SB={}, BB={}",
                    pf.blinds[0], pf.blinds[1]
                ));
            }
        }

        if self.max_raises_per_round == 0 {
            return Err("Max raises per round must be at least 1".to_string());
        }

        if self.force_all_in_threshold < 0.0 || self.force_all_in_threshold > 1.0 {
            return Err(format!(
                "Force all-in threshold must be 0-1: {}",
                self.force_all_in_threshold
            ));
        }

        if self.merge_threshold < 0.0 {
            return Err(format!(
                "Merge threshold must be non-negative: {}",
                self.merge_threshold
            ));
        }

        if self.add_all_in_threshold < 0.0 {
            return Err(format!(
                "Add all-in threshold must be non-negative: {}",
                self.add_all_in_threshold
            ));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_street_methods() {
        assert_eq!(Street::Preflop.streets_remaining(), 4);
        assert_eq!(Street::Flop.streets_remaining(), 3);
        assert_eq!(Street::River.streets_remaining(), 1);

        assert_eq!(Street::Preflop.next(), Some(Street::Flop));
        assert_eq!(Street::River.next(), None);
    }

    #[test]
    fn test_preflop_config() {
        let pf = PreflopConfig::new(1, 2).with_ante(1).with_bb_ante(0);
        assert_eq!(pf.blinds, [1, 2]);
        assert_eq!(pf.ante, 1);
        assert_eq!(pf.total_ante(6), 6);

        let pf_bb_ante = PreflopConfig::new(1, 2).with_bb_ante(6);
        assert_eq!(pf_bb_ante.total_ante(6), 6);
    }

    #[test]
    fn test_tree_config_validation() {
        let config = TreeConfig::new(2).with_stack(100);
        assert!(config.validate().is_ok());

        let bad_config = TreeConfig::new(7);
        assert!(bad_config.validate().is_err());

        let bad_config = TreeConfig::new(2).with_stack(0);
        assert!(bad_config.validate().is_err());
    }

    #[test]
    fn test_effective_stack() {
        let config = TreeConfig::new(2).with_stacks(vec![100, 150]);
        assert_eq!(config.effective_stack(), 100);

        let config = TreeConfig::new(3).with_stack(200);
        assert_eq!(config.effective_stack(), 200);
    }

    #[test]
    fn test_starting_pot() {
        let config = TreeConfig::new(6)
            .with_preflop(PreflopConfig::new(50, 100).with_ante(10));
        assert_eq!(config.starting_pot(), 50 + 100 + 60);
    }
}
