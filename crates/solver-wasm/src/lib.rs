//! WASM bindings for the poker tree builder and solver.

mod tree_view;

use std::cell::RefCell;

use serde::{Deserialize, Serialize};
use solver::poker::hands::{combo_to_string, Combo};
use solver::poker::{parse_board, parse_range, PostflopGame, PostflopSolver};
use solver::tree::bet_size::parse_bet_size;
use solver::{
    ActionTree, BetSizeOptions, BetType, IndexedActionTree, MemoryEstimate, PreflopConfig, Street,
    StreetConfig, TreeConfig, TreeStats,
};
use wasm_bindgen::prelude::*;

// === Solver State ===

struct SolverState {
    game: PostflopGame,
    solver: PostflopSolver,
    pot: i32,
}

thread_local! {
    static SOLVER_STATE: RefCell<Option<SolverState>> = RefCell::new(None);
}

/// Initialize panic hook for better error messages in browser console.
#[wasm_bindgen(start)]
pub fn init() {
    console_error_panic_hook::set_once();
}

/// JSON configuration for a poker spot.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpotConfig {
    /// Config name
    #[serde(default)]
    pub name: String,

    /// Number of players (2-6)
    pub num_players: usize,

    /// Starting stacks in chips (per player, or single value for all)
    pub starting_stacks: Vec<i32>,

    /// Starting street: "preflop", "flop", "turn", or "river"
    /// If not specified, uses preflop if preflop config exists, else first configured street
    #[serde(default)]
    pub starting_street: Option<String>,

    /// Starting pot for postflop trees (ignored for preflop start)
    #[serde(default = "default_starting_pot")]
    pub starting_pot: i32,

    /// Betting type: "NoLimit" or "PotLimit"
    #[serde(default)]
    pub bet_type: String,

    /// Preflop configuration
    pub preflop: Option<PreflopJsonConfig>,

    /// Flop bet sizes
    pub flop: Option<StreetJsonConfig>,

    /// Turn bet sizes
    pub turn: Option<StreetJsonConfig>,

    /// River bet sizes
    pub river: Option<StreetJsonConfig>,

    /// Max raises per betting round
    #[serde(default = "default_max_raises")]
    pub max_raises_per_round: u8,

    /// Force all-in threshold (0.0-1.0)
    #[serde(default = "default_force_all_in")]
    pub force_all_in_threshold: f64,

    /// Merge similar bet sizes threshold
    #[serde(default = "default_merge")]
    pub merge_threshold: f64,

    /// Add all-in if max bet < threshold * pot
    #[serde(default = "default_add_all_in")]
    pub add_all_in_threshold: f64,
}

fn default_max_raises() -> u8 {
    4
}
fn default_force_all_in() -> f64 {
    0.15
}
fn default_merge() -> f64 {
    0.1
}
fn default_add_all_in() -> f64 {
    1.5
}
fn default_starting_pot() -> i32 {
    0
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreflopJsonConfig {
    /// Blinds [SB, BB]
    pub blinds: [i32; 2],

    /// Ante per player
    #[serde(default)]
    pub ante: i32,

    /// Big blind ante
    #[serde(default)]
    pub bb_ante: i32,

    /// Open raise sizes (comma-separated, e.g., "2.5x, 3x, a")
    #[serde(default)]
    pub open_sizes: BetSizeStrings,

    /// 3-bet sizes
    #[serde(default)]
    pub three_bet_sizes: BetSizeStrings,

    /// 4-bet+ sizes
    #[serde(default)]
    pub four_bet_sizes: BetSizeStrings,

    /// Allow limps
    #[serde(default = "default_true")]
    pub allow_limps: bool,
}

fn default_true() -> bool {
    true
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BetSizeStrings {
    /// Bet sizes (e.g., "33%, 50%, 75%")
    #[serde(default)]
    pub bet: String,

    /// Raise sizes (e.g., "2.5x, 3x, a")
    #[serde(default)]
    pub raise: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreetJsonConfig {
    /// Bet/raise sizes
    pub sizes: BetSizeStrings,

    /// Donk bet sizes (optional)
    #[serde(default)]
    pub donk_sizes: Option<BetSizeStrings>,
}

/// Result of building a tree
#[derive(Debug, Serialize, Deserialize)]
pub struct BuildResult {
    pub success: bool,
    pub error: Option<String>,
    pub stats: Option<TreeStats>,
    pub memory: Option<MemoryEstimate>,
}

/// Build a tree from JSON config and return stats.
#[wasm_bindgen]
pub fn build_tree(config_json: &str) -> JsValue {
    let result = build_tree_internal(config_json);
    serde_wasm_bindgen::to_value(&result).unwrap()
}

fn build_tree_internal(config_json: &str) -> BuildResult {
    // Parse JSON config
    let spot_config: SpotConfig = match serde_json::from_str(config_json) {
        Ok(c) => c,
        Err(e) => {
            return BuildResult {
                success: false,
                error: Some(format!("Failed to parse config: {}", e)),
                stats: None,
                memory: None,
            }
        }
    };

    // Convert to TreeConfig
    let tree_config = match convert_config(&spot_config) {
        Ok(c) => c,
        Err(e) => {
            return BuildResult {
                success: false,
                error: Some(e),
                stats: None,
                memory: None,
            }
        }
    };

    // Build tree
    match ActionTree::new(tree_config) {
        Ok(tree) => {
            let stats = tree.stats();
            let memory = tree.memory_estimate_preflop();
            BuildResult {
                success: true,
                error: None,
                stats: Some(stats),
                memory: Some(memory),
            }
        }
        Err(e) => BuildResult {
            success: false,
            error: Some(e),
            stats: None,
            memory: None,
        },
    }
}

/// Validate a config without building the full tree.
#[wasm_bindgen]
pub fn validate_config(config_json: &str) -> JsValue {
    let result = validate_config_internal(config_json);
    serde_wasm_bindgen::to_value(&result).unwrap()
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ValidationResult {
    pub valid: bool,
    pub errors: Vec<String>,
}

fn validate_config_internal(config_json: &str) -> ValidationResult {
    let mut errors = Vec::new();

    // Parse JSON
    let spot_config: SpotConfig = match serde_json::from_str(config_json) {
        Ok(c) => c,
        Err(e) => {
            return ValidationResult {
                valid: false,
                errors: vec![format!("JSON parse error: {}", e)],
            }
        }
    };

    // Validate fields
    if spot_config.num_players < 2 || spot_config.num_players > 6 {
        errors.push(format!(
            "num_players must be 2-6, got {}",
            spot_config.num_players
        ));
    }

    if spot_config.starting_stacks.is_empty() {
        errors.push("starting_stacks cannot be empty".to_string());
    }

    for (i, &stack) in spot_config.starting_stacks.iter().enumerate() {
        if stack <= 0 {
            errors.push(format!("Stack for player {} must be positive: {}", i, stack));
        }
    }

    if let Some(ref pf) = spot_config.preflop {
        if pf.blinds[0] <= 0 || pf.blinds[1] <= 0 {
            errors.push(format!("Blinds must be positive: {:?}", pf.blinds));
        }
        if pf.blinds[0] >= pf.blinds[1] {
            errors.push("Small blind must be less than big blind".to_string());
        }

        // Validate bet size strings
        if let Err(e) = parse_bet_sizes_str(&pf.open_sizes.bet, false) {
            errors.push(format!("Invalid open bet sizes: {}", e));
        }
        if let Err(e) = parse_bet_sizes_str(&pf.open_sizes.raise, true) {
            errors.push(format!("Invalid open raise sizes: {}", e));
        }
    }

    // Validate street configs
    for (name, street) in [
        ("flop", &spot_config.flop),
        ("turn", &spot_config.turn),
        ("river", &spot_config.river),
    ] {
        if let Some(sc) = street {
            if let Err(e) = parse_bet_sizes_str(&sc.sizes.bet, false) {
                errors.push(format!("Invalid {} bet sizes: {}", name, e));
            }
            if let Err(e) = parse_bet_sizes_str(&sc.sizes.raise, true) {
                errors.push(format!("Invalid {} raise sizes: {}", name, e));
            }
        }
    }

    ValidationResult {
        valid: errors.is_empty(),
        errors,
    }
}

/// Preview a bet size resolution.
#[wasm_bindgen]
pub fn parse_bet_size_preview(size_str: &str, pot: i32, stack: i32) -> JsValue {
    let result = parse_bet_size_preview_internal(size_str, pot, stack);
    serde_wasm_bindgen::to_value(&result).unwrap()
}

#[derive(Debug, Serialize, Deserialize)]
pub struct BetSizePreview {
    pub valid: bool,
    pub error: Option<String>,
    pub chips: Option<i32>,
    pub pot_percentage: Option<f64>,
}

fn parse_bet_size_preview_internal(size_str: &str, pot: i32, stack: i32) -> BetSizePreview {
    match parse_bet_size(size_str, true) {
        Ok(bet_size) => {
            let chips = bet_size.resolve(pot, 0, 0, stack, 3);
            let pot_percentage = if pot > 0 {
                Some((chips as f64 / pot as f64) * 100.0)
            } else {
                None
            };
            BetSizePreview {
                valid: true,
                error: None,
                chips: Some(chips),
                pot_percentage,
            }
        }
        Err(e) => BetSizePreview {
            valid: false,
            error: Some(e),
            chips: None,
            pot_percentage: None,
        },
    }
}

/// Get the node state at a given path for tree navigation.
#[wasm_bindgen]
pub fn get_node_at_path(config_json: &str, path: &str) -> JsValue {
    let result = tree_view::get_node_at_path_internal(config_json, path);
    serde_wasm_bindgen::to_value(&result).unwrap()
}

/// Get the result of taking an action from the current path.
#[wasm_bindgen]
pub fn get_action_result(config_json: &str, current_path: &str, action_index: usize) -> JsValue {
    let result = tree_view::get_action_result_internal(config_json, current_path, action_index);
    serde_wasm_bindgen::to_value(&result).unwrap()
}

/// Get the tree root with its immediate children for the tree view.
#[wasm_bindgen]
pub fn get_tree_root(config_json: &str) -> JsValue {
    let result = tree_view::get_tree_root_internal(config_json);
    serde_wasm_bindgen::to_value(&result).unwrap()
}

/// Get children of a node at the given path for lazy loading.
#[wasm_bindgen]
pub fn get_tree_children(config_json: &str, path: &str) -> JsValue {
    let result = tree_view::get_tree_children_internal(config_json, path);
    serde_wasm_bindgen::to_value(&result).unwrap()
}

/// Get a default config as JSON.
#[wasm_bindgen]
pub fn get_default_config() -> String {
    let config = SpotConfig {
        name: "HU 100bb".to_string(),
        num_players: 2,
        starting_stacks: vec![200],
        starting_street: None, // None means auto-detect (preflop if config exists)
        starting_pot: 0,       // Ignored for preflop start
        bet_type: "NoLimit".to_string(),
        preflop: Some(PreflopJsonConfig {
            blinds: [1, 2],
            ante: 0,
            bb_ante: 0,
            open_sizes: BetSizeStrings {
                bet: "50%, 75%".to_string(),
                raise: "2.5x, 3x, a".to_string(),
            },
            three_bet_sizes: BetSizeStrings {
                bet: "".to_string(),
                raise: "2.5x, 3x, a".to_string(),
            },
            four_bet_sizes: BetSizeStrings {
                bet: "".to_string(),
                raise: "2.2x, a".to_string(),
            },
            allow_limps: true,
        }),
        flop: Some(StreetJsonConfig {
            sizes: BetSizeStrings {
                bet: "33%, 67%, 100%".to_string(),
                raise: "2.5x, a".to_string(),
            },
            donk_sizes: None,
        }),
        turn: Some(StreetJsonConfig {
            sizes: BetSizeStrings {
                bet: "33%, 67%, 100%".to_string(),
                raise: "2.5x, a".to_string(),
            },
            donk_sizes: None,
        }),
        river: Some(StreetJsonConfig {
            sizes: BetSizeStrings {
                bet: "33%, 67%, 100%".to_string(),
                raise: "2.5x, a".to_string(),
            },
            donk_sizes: None,
        }),
        max_raises_per_round: 4,
        force_all_in_threshold: 0.15,
        merge_threshold: 0.1,
        add_all_in_threshold: 1.5,
    };
    serde_json::to_string_pretty(&config).unwrap()
}

// === Solver WASM Functions ===

/// JSON config for creating a solver.
///
/// Supports two formats:
/// 1. Legacy 2-player: `oop_range`, `ip_range`, `effective_stack`
/// 2. N-player: `ranges`, `stacks`
///
/// The N-player format takes precedence if `ranges` is provided.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolverConfig {
    /// Board string (e.g., "KhQsJs2c3d" for river, "KhQsJs2c" for turn)
    pub board: String,

    /// Starting pot size
    pub pot: i32,

    /// Tree configuration (reuses SpotConfig)
    pub tree_config: SpotConfig,

    // === Legacy 2-player format ===
    /// OOP range string (e.g., "AA,KK,QQ,AKs") - for 2-player games
    #[serde(default)]
    pub oop_range: Option<String>,
    /// IP range string (e.g., "AA,KK,AKs,AQs") - for 2-player games
    #[serde(default)]
    pub ip_range: Option<String>,
    /// Effective stack size - for 2-player games
    #[serde(default)]
    pub effective_stack: Option<i32>,

    // === N-player format ===
    /// Range strings for each player (indexed by tree player ID)
    #[serde(default)]
    pub ranges: Option<Vec<String>>,
    /// Stack sizes for each player
    #[serde(default)]
    pub stacks: Option<Vec<i32>>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CreateSolverResult {
    pub success: bool,
    pub error: Option<String>,
    pub num_ip_hands: Option<usize>,
    pub num_oop_hands: Option<usize>,
}

/// Create a solver from JSON config.
///
/// Config includes board, ranges, pot, stack, and tree configuration.
#[wasm_bindgen]
pub fn create_solver(config_json: &str) -> JsValue {
    let result = create_solver_internal(config_json);
    serde_wasm_bindgen::to_value(&result).unwrap()
}

fn create_solver_internal(config_json: &str) -> CreateSolverResult {
    let solver_config: SolverConfig = match serde_json::from_str(config_json) {
        Ok(c) => c,
        Err(e) => {
            return CreateSolverResult {
                success: false,
                error: Some(format!("Failed to parse config: {}", e)),
                num_ip_hands: None,
                num_oop_hands: None,
            }
        }
    };

    // Build tree from tree_config
    let tree_config = match convert_config(&solver_config.tree_config) {
        Ok(c) => c,
        Err(e) => {
            return CreateSolverResult {
                success: false,
                error: Some(format!("Invalid tree config: {}", e)),
                num_ip_hands: None,
                num_oop_hands: None,
            }
        }
    };

    let tree = match ActionTree::new(tree_config) {
        Ok(t) => t,
        Err(e) => {
            return CreateSolverResult {
                success: false,
                error: Some(format!("Failed to build tree: {}", e)),
                num_ip_hands: None,
                num_oop_hands: None,
            }
        }
    };

    let indexed_tree = tree.to_indexed();

    // Parse board
    let board = match parse_board(&solver_config.board) {
        Ok(b) => b,
        Err(e) => {
            return CreateSolverResult {
                success: false,
                error: Some(format!("Invalid board: {}", e)),
                num_ip_hands: None,
                num_oop_hands: None,
            }
        }
    };

    let pot = solver_config.pot;

    // Determine if using n-player or legacy 2-player format
    let game = if let Some(ref range_strings) = solver_config.ranges {
        // N-player format
        let num_players = range_strings.len();
        if num_players < 2 || num_players > 6 {
            return CreateSolverResult {
                success: false,
                error: Some(format!("Number of players must be 2-6, got {}", num_players)),
                num_ip_hands: None,
                num_oop_hands: None,
            };
        }

        // Parse ranges
        let mut ranges = Vec::with_capacity(num_players);
        for (i, range_str) in range_strings.iter().enumerate() {
            match parse_range(range_str) {
                Ok(r) => ranges.push(r),
                Err(e) => {
                    return CreateSolverResult {
                        success: false,
                        error: Some(format!("Invalid range for player {}: {}", i, e)),
                        num_ip_hands: None,
                        num_oop_hands: None,
                    }
                }
            }
        }

        // Get stacks
        let stacks = match &solver_config.stacks {
            Some(s) => {
                if s.len() != num_players {
                    return CreateSolverResult {
                        success: false,
                        error: Some(format!(
                            "Stacks length ({}) must match ranges length ({})",
                            s.len(),
                            num_players
                        )),
                        num_ip_hands: None,
                        num_oop_hands: None,
                    };
                }
                s.clone()
            }
            None => {
                // Use effective_stack if provided, else default
                let default_stack = solver_config.effective_stack.unwrap_or(100);
                vec![default_stack; num_players]
            }
        };

        PostflopGame::new_multiway(indexed_tree, board, ranges, stacks, pot)
    } else {
        // Legacy 2-player format
        let oop_range_str = match &solver_config.oop_range {
            Some(s) => s.as_str(),
            None => {
                return CreateSolverResult {
                    success: false,
                    error: Some("Missing oop_range (or use 'ranges' for n-player format)".to_string()),
                    num_ip_hands: None,
                    num_oop_hands: None,
                }
            }
        };

        let ip_range_str = match &solver_config.ip_range {
            Some(s) => s.as_str(),
            None => {
                return CreateSolverResult {
                    success: false,
                    error: Some("Missing ip_range (or use 'ranges' for n-player format)".to_string()),
                    num_ip_hands: None,
                    num_oop_hands: None,
                }
            }
        };

        let oop_range = match parse_range(oop_range_str) {
            Ok(r) => r,
            Err(e) => {
                return CreateSolverResult {
                    success: false,
                    error: Some(format!("Invalid OOP range: {}", e)),
                    num_ip_hands: None,
                    num_oop_hands: None,
                }
            }
        };

        let ip_range = match parse_range(ip_range_str) {
            Ok(r) => r,
            Err(e) => {
                return CreateSolverResult {
                    success: false,
                    error: Some(format!("Invalid IP range: {}", e)),
                    num_ip_hands: None,
                    num_oop_hands: None,
                }
            }
        };

        let effective_stack = solver_config.effective_stack.unwrap_or(100);
        PostflopGame::new(indexed_tree, board, oop_range, ip_range, pot, effective_stack)
    };

    let solver = PostflopSolver::new(&game);
    let num_ip = solver.num_hands(0);
    let num_oop = solver.num_hands(1);

    SOLVER_STATE.with(|state| {
        *state.borrow_mut() = Some(SolverState { game, solver, pot });
    });

    CreateSolverResult {
        success: true,
        error: None,
        num_ip_hands: Some(num_ip),
        num_oop_hands: Some(num_oop),
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct RunIterationsResult {
    pub total_iterations: u32,
    pub exploitability: f32,
    pub exploitability_pct: f32,
}

/// Run solver iterations.
///
/// Returns total iteration count and current exploitability.
#[wasm_bindgen]
pub fn run_iterations(count: u32) -> JsValue {
    let result = SOLVER_STATE.with(|state| {
        let mut state = state.borrow_mut();
        let state = match state.as_mut() {
            Some(s) => s,
            None => {
                return RunIterationsResult {
                    total_iterations: 0,
                    exploitability: 0.0,
                    exploitability_pct: 0.0,
                }
            }
        };

        state.solver.train(&state.game, count);

        let exploit = state.solver.exploitability(&state.game);
        let pot = state.pot as f32;
        let exploit_pct = if pot > 0.0 {
            exploit / pot * 100.0
        } else {
            0.0
        };

        RunIterationsResult {
            total_iterations: state.solver.total_iterations(),
            exploitability: exploit,
            exploitability_pct: exploit_pct,
        }
    });

    serde_wasm_bindgen::to_value(&result).unwrap()
}

/// Get current exploitability.
#[wasm_bindgen]
pub fn get_exploitability() -> f32 {
    SOLVER_STATE.with(|state| {
        let state = state.borrow();
        match state.as_ref() {
            Some(s) => s.solver.exploitability(&s.game),
            None => 0.0,
        }
    })
}

#[derive(Debug, Serialize, Deserialize)]
pub struct NodeStrategyResult {
    pub success: bool,
    pub error: Option<String>,
    pub action_names: Vec<String>,
    pub hands: Vec<HandStrategy>,
    pub aggregate: Vec<f32>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct HandStrategy {
    pub combo: String,
    pub weight: f32,
    pub actions: Vec<f32>,
}

/// Get the strategy for a specific node after solving.
///
/// `path` is a dot-separated string of action indices (e.g., "0.1.2").
/// `player` is the acting player at this node (0=IP, 1=OOP).
#[wasm_bindgen]
pub fn get_node_strategy(path: &str, player: usize) -> JsValue {
    let result = SOLVER_STATE.with(|state| {
        let state = state.borrow();
        let state = match state.as_ref() {
            Some(s) => s,
            None => {
                return NodeStrategyResult {
                    success: false,
                    error: Some("No solver active".to_string()),
                    action_names: vec![],
                    hands: vec![],
                    aggregate: vec![],
                }
            }
        };

        get_node_strategy_internal(state, path, player)
    });

    serde_wasm_bindgen::to_value(&result).unwrap()
}

fn get_node_strategy_internal(
    state: &SolverState,
    path: &str,
    player: usize,
) -> NodeStrategyResult {
    let tree = &state.game.tree;

    // Navigate to the node
    let node_idx = match navigate_indexed_tree(tree, path) {
        Ok(idx) => idx,
        Err(e) => {
            return NodeStrategyResult {
                success: false,
                error: Some(e),
                action_names: vec![],
                hands: vec![],
                aggregate: vec![],
            }
        }
    };

    let node = tree.get(node_idx);

    // Get action names
    let action_names: Vec<String> = node.actions.iter().map(|a| a.display()).collect();
    let num_actions = action_names.len();

    if num_actions == 0 {
        return NodeStrategyResult {
            success: true,
            error: None,
            action_names: vec![],
            hands: vec![],
            aggregate: vec![],
        };
    }

    // Get per-hand strategies
    let num_hands = state.solver.num_hands(player);
    let num_contexts = state.solver.num_contexts(node_idx);

    let mut hands = Vec::with_capacity(num_hands);
    let mut aggregate = vec![0.0f32; num_actions];
    let mut total_weight = 0.0f32;

    for h in 0..num_hands {
        let (combo_idx, weight) = state.solver.hand_info(player, h);
        let combo = Combo::from_index(combo_idx);
        let combo_str = combo_to_string(combo);

        // Average strategy across card contexts
        let mut actions = vec![0.0f32; num_actions];
        for ctx in 0..num_contexts {
            let ctx_strategy = state.solver.get_hand_strategy_ctx(node_idx, h, player, ctx);
            for (a, &prob) in ctx_strategy.iter().enumerate() {
                actions[a] += prob;
            }
        }
        // Normalize by number of contexts
        if num_contexts > 1 {
            let inv = 1.0 / num_contexts as f32;
            for a in &mut actions {
                *a *= inv;
            }
        }

        // Accumulate weighted aggregate
        for (a, &prob) in actions.iter().enumerate() {
            aggregate[a] += weight * prob;
        }
        total_weight += weight;

        hands.push(HandStrategy {
            combo: combo_str,
            weight,
            actions,
        });
    }

    // Normalize aggregate
    if total_weight > 0.0 {
        for a in &mut aggregate {
            *a /= total_weight;
        }
    }

    NodeStrategyResult {
        success: true,
        error: None,
        action_names,
        hands,
        aggregate,
    }
}

// === River Card Context Functions ===

#[derive(Debug, Serialize, Deserialize)]
pub struct RiverCardsResult {
    pub has_river_cards: bool,
    pub cards: Vec<String>,
}

/// Get the list of valid river cards for the current solver.
///
/// Returns { has_river_cards: bool, cards: ["2c", "2d", ...] }.
/// `has_river_cards` is true when solving a turn spot (4-card board).
#[wasm_bindgen]
pub fn get_river_cards() -> JsValue {
    let result = SOLVER_STATE.with(|state| {
        let state = state.borrow();
        match state.as_ref() {
            Some(s) => {
                let cards = s.solver.river_card_strings();
                RiverCardsResult {
                    has_river_cards: !cards.is_empty(),
                    cards,
                }
            }
            None => RiverCardsResult {
                has_river_cards: false,
                cards: vec![],
            },
        }
    });
    serde_wasm_bindgen::to_value(&result).unwrap()
}

/// Check if a node at the given path is below a chance node (has multiple card contexts).
///
/// Returns true if the node has more than 1 card context (i.e., it's a river betting node
/// in a turn tree).
#[wasm_bindgen]
pub fn is_node_below_chance(path: &str) -> bool {
    SOLVER_STATE.with(|state| {
        let state = state.borrow();
        let state = match state.as_ref() {
            Some(s) => s,
            None => return false,
        };

        let tree = &state.game.tree;
        let node_idx = match navigate_indexed_tree(tree, path) {
            Ok(idx) => idx,
            Err(_) => return false,
        };

        state.solver.num_contexts(node_idx) > 1
    })
}

/// Get the chance depth for a node at the given path.
///
/// Returns 0 for nodes before any chance node, 1 for nodes after the first
/// chance node (turn betting in flop trees, river betting in turn trees),
/// and 2 for nodes after the second chance node (river betting in flop trees).
#[wasm_bindgen]
pub fn get_chance_depth(path: &str) -> i32 {
    SOLVER_STATE.with(|state| {
        let state = state.borrow();
        let state = match state.as_ref() {
            Some(s) => s,
            None => return 0,
        };

        let tree = &state.game.tree;
        let node_idx = match navigate_indexed_tree(tree, path) {
            Ok(idx) => idx,
            Err(_) => return 0,
        };

        state.solver.chance_depth(node_idx) as i32
    })
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TurnCardsResult {
    pub has_turn_cards: bool,
    pub cards: Vec<String>,
}

/// Get the list of valid turn cards for the current solver.
///
/// Returns { has_turn_cards: bool, cards: ["2c", "2d", ...] }.
/// `has_turn_cards` is true when solving a flop spot (3-card board).
#[wasm_bindgen]
pub fn get_turn_cards() -> JsValue {
    let result = SOLVER_STATE.with(|state| {
        let state = state.borrow();
        match state.as_ref() {
            Some(s) => {
                let cards = s.solver.turn_card_strings();
                TurnCardsResult {
                    has_turn_cards: !cards.is_empty(),
                    cards,
                }
            }
            None => TurnCardsResult {
                has_turn_cards: false,
                cards: vec![],
            },
        }
    });
    serde_wasm_bindgen::to_value(&result).unwrap()
}

/// Get the strategy for a specific node with a specific card context.
///
/// `card_context`: -1 for average across all river cards, 0+ for a specific river card index.
/// When a specific card context is selected, hands that conflict with the river card are excluded.
#[wasm_bindgen]
pub fn get_node_strategy_for_context(path: &str, player: usize, card_context: i32) -> JsValue {
    let result = SOLVER_STATE.with(|state| {
        let state = state.borrow();
        let state = match state.as_ref() {
            Some(s) => s,
            None => {
                return NodeStrategyResult {
                    success: false,
                    error: Some("No solver active".to_string()),
                    action_names: vec![],
                    hands: vec![],
                    aggregate: vec![],
                }
            }
        };

        if card_context < 0 {
            // Average mode: use existing function
            get_node_strategy_internal(state, path, player)
        } else {
            get_node_strategy_for_context_internal(state, path, player, card_context as usize)
        }
    });

    serde_wasm_bindgen::to_value(&result).unwrap()
}

fn get_node_strategy_for_context_internal(
    state: &SolverState,
    path: &str,
    player: usize,
    ctx: usize,
) -> NodeStrategyResult {
    let tree = &state.game.tree;

    // Navigate to the node
    let node_idx = match navigate_indexed_tree(tree, path) {
        Ok(idx) => idx,
        Err(e) => {
            return NodeStrategyResult {
                success: false,
                error: Some(e),
                action_names: vec![],
                hands: vec![],
                aggregate: vec![],
            }
        }
    };

    let node = tree.get(node_idx);
    let action_names: Vec<String> = node.actions.iter().map(|a| a.display()).collect();
    let num_actions = action_names.len();

    if num_actions == 0 {
        return NodeStrategyResult {
            success: true,
            error: None,
            action_names: vec![],
            hands: vec![],
            aggregate: vec![],
        };
    }

    // Get the cards for this context to filter blocked hands
    let (turn_card, river_card) = state.solver.context_cards(node_idx, ctx);

    let num_hands = state.solver.num_hands(player);
    let mut hands = Vec::with_capacity(num_hands);
    let mut aggregate = vec![0.0f32; num_actions];
    let mut total_weight = 0.0f32;

    for h in 0..num_hands {
        let (combo_idx, weight) = state.solver.hand_info(player, h);

        // Skip hands that conflict with dealt cards
        let (c0, c1) = state.solver.hand_cards(player, h);
        if let Some(tc) = turn_card {
            if c0 == tc || c1 == tc {
                continue;
            }
        }
        if let Some(rc) = river_card {
            if c0 == rc || c1 == rc {
                continue;
            }
        }

        let combo = Combo::from_index(combo_idx);
        let combo_str = combo_to_string(combo);

        // Get strategy for this specific context (no averaging)
        let actions = state.solver.get_hand_strategy_ctx(node_idx, h, player, ctx);

        // Accumulate weighted aggregate
        for (a, &prob) in actions.iter().enumerate() {
            aggregate[a] += weight * prob;
        }
        total_weight += weight;

        hands.push(HandStrategy {
            combo: combo_str,
            weight,
            actions,
        });
    }

    // Normalize aggregate
    if total_weight > 0.0 {
        for a in &mut aggregate {
            *a /= total_weight;
        }
    }

    NodeStrategyResult {
        success: true,
        error: None,
        action_names,
        hands,
        aggregate,
    }
}

/// Navigate an IndexedActionTree using a dot-separated action path.
/// Automatically skips chance nodes.
fn navigate_indexed_tree(tree: &IndexedActionTree, path: &str) -> Result<usize, String> {
    let mut node_idx = tree.root_idx;

    // Skip initial chance node
    while tree.get(node_idx).is_chance() {
        if tree.get(node_idx).children.is_empty() {
            return Err("Chance node has no children".to_string());
        }
        node_idx = tree.get(node_idx).children[0];
    }

    if path.is_empty() {
        return Ok(node_idx);
    }

    let indices: Vec<usize> = path
        .split('.')
        .map(|s| s.parse::<usize>())
        .collect::<Result<_, _>>()
        .map_err(|e| format!("Invalid path: {}", e))?;

    for &action_idx in &indices {
        let node = tree.get(node_idx);
        if action_idx >= node.children.len() {
            return Err(format!(
                "Invalid action index {} (node has {} actions)",
                action_idx,
                node.actions.len()
            ));
        }
        node_idx = node.children[action_idx];

        // Skip chance nodes
        while tree.get(node_idx).is_chance() {
            if tree.get(node_idx).children.is_empty() {
                break;
            }
            node_idx = tree.get(node_idx).children[0];
        }
    }

    Ok(node_idx)
}

// === Internal conversion functions ===

fn convert_config(spot: &SpotConfig) -> Result<TreeConfig, String> {
    let bet_type = match spot.bet_type.to_lowercase().as_str() {
        "potlimit" | "pot_limit" | "pot-limit" => BetType::PotLimit,
        _ => BetType::NoLimit,
    };

    // Parse starting street if provided
    let starting_street = spot.starting_street.as_ref().map(|s| {
        match s.to_lowercase().as_str() {
            "preflop" | "pre" => Street::Preflop,
            "flop" => Street::Flop,
            "turn" => Street::Turn,
            "river" => Street::River,
            _ => Street::Preflop,
        }
    });

    let mut config = TreeConfig {
        num_players: spot.num_players,
        starting_stacks: spot.starting_stacks.clone(),
        starting_street,
        starting_pot: spot.starting_pot,
        bet_type,
        preflop: None,
        flop: None,
        turn: None,
        river: None,
        max_raises_per_round: spot.max_raises_per_round,
        force_all_in_threshold: spot.force_all_in_threshold,
        merge_threshold: spot.merge_threshold,
        add_all_in_threshold: spot.add_all_in_threshold,
    };

    // Convert preflop
    if let Some(ref pf) = spot.preflop {
        config.preflop = Some(convert_preflop(pf)?);
    }

    // Convert streets
    if let Some(ref sc) = spot.flop {
        config.flop = Some(convert_street(sc)?);
    }
    if let Some(ref sc) = spot.turn {
        config.turn = Some(convert_street(sc)?);
    }
    if let Some(ref sc) = spot.river {
        config.river = Some(convert_street(sc)?);
    }

    Ok(config)
}

fn convert_preflop(pf: &PreflopJsonConfig) -> Result<PreflopConfig, String> {
    let open_sizes = convert_bet_size_strings(&pf.open_sizes)?;
    let three_bet_sizes = convert_bet_size_strings(&pf.three_bet_sizes)?;
    let four_bet_sizes = convert_bet_size_strings(&pf.four_bet_sizes)?;

    Ok(PreflopConfig {
        blinds: pf.blinds,
        ante: pf.ante,
        bb_ante: pf.bb_ante,
        allow_limps: pf.allow_limps,
        open_sizes: vec![open_sizes],
        three_bet_sizes: vec![three_bet_sizes],
        four_bet_plus_sizes: vec![four_bet_sizes],
        allow_cold_call: true,
    })
}

fn convert_street(sc: &StreetJsonConfig) -> Result<StreetConfig, String> {
    let sizes = convert_bet_size_strings(&sc.sizes)?;

    let donk_sizes = if let Some(ref ds) = sc.donk_sizes {
        Some(vec![convert_bet_size_strings(ds)?])
    } else {
        None
    };

    Ok(StreetConfig {
        sizes: vec![sizes],
        donk_sizes,
    })
}

fn convert_bet_size_strings(bss: &BetSizeStrings) -> Result<BetSizeOptions, String> {
    let bet = parse_bet_sizes_str(&bss.bet, false)?;
    let raise = parse_bet_sizes_str(&bss.raise, true)?;

    Ok(BetSizeOptions {
        bet: bet.clone(),
        raise: raise.clone(),
        reraise: raise.clone(),
        reraise_plus: raise,
    })
}

fn parse_bet_sizes_str(
    s: &str,
    allow_raise_rel: bool,
) -> Result<Vec<solver::tree::BetSize>, String> {
    if s.trim().is_empty() {
        return Ok(Vec::new());
    }

    s.split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| parse_bet_size(s, allow_raise_rel))
        .collect()
}
