//! Tauri commands that mirror the WASM API.

use serde::{Deserialize, Serialize};
use solver::poker::hands::{combo_to_string, Combo};
use solver::poker::{parse_board, parse_range, PostflopGame, PostflopSolver};
use solver::tree::bet_size::parse_bet_size;
use solver::tree::{Action, ActionTree, ActionTreeNode, Position, Street, TerminalResult};
use solver::{
    BetSizeOptions, BetType, IndexedActionTree, MemoryEstimate, PreflopConfig, StreetConfig,
    TreeConfig, TreeStats,
};
use tauri::State;

use crate::state::SolverState;

// === Config Types (mirrored from WASM) ===

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpotConfig {
    #[serde(default)]
    pub name: String,
    pub num_players: usize,
    pub starting_stacks: Vec<i32>,
    #[serde(default)]
    pub starting_street: Option<String>,
    #[serde(default = "default_starting_pot")]
    pub starting_pot: i32,
    #[serde(default)]
    pub bet_type: String,
    pub preflop: Option<PreflopJsonConfig>,
    pub flop: Option<StreetJsonConfig>,
    pub turn: Option<StreetJsonConfig>,
    pub river: Option<StreetJsonConfig>,
    #[serde(default = "default_max_raises")]
    pub max_raises_per_round: u8,
    #[serde(default = "default_force_all_in")]
    pub force_all_in_threshold: f64,
    #[serde(default = "default_merge")]
    pub merge_threshold: f64,
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
    pub blinds: [i32; 2],
    #[serde(default)]
    pub ante: i32,
    #[serde(default)]
    pub bb_ante: i32,
    #[serde(default)]
    pub open_sizes: BetSizeStrings,
    #[serde(default)]
    pub three_bet_sizes: BetSizeStrings,
    #[serde(default)]
    pub four_bet_sizes: BetSizeStrings,
    #[serde(default = "default_true")]
    pub allow_limps: bool,
}

fn default_true() -> bool {
    true
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BetSizeStrings {
    #[serde(default)]
    pub bet: String,
    #[serde(default)]
    pub raise: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreetJsonConfig {
    pub sizes: BetSizeStrings,
    #[serde(default)]
    pub donk_sizes: Option<BetSizeStrings>,
}

// === Result Types ===

#[derive(Debug, Serialize, Deserialize)]
pub struct BuildResult {
    pub success: bool,
    pub error: Option<String>,
    pub stats: Option<TreeStats>,
    pub memory: Option<MemoryEstimate>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ValidationResult {
    pub valid: bool,
    pub errors: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct BetSizePreview {
    pub valid: bool,
    pub error: Option<String>,
    pub chips: Option<i32>,
    pub pot_percentage: Option<f64>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SolverConfig {
    pub board: String,
    pub pot: i32,
    pub tree_config: SpotConfig,
    #[serde(default)]
    pub oop_range: Option<String>,
    #[serde(default)]
    pub ip_range: Option<String>,
    #[serde(default)]
    pub effective_stack: Option<i32>,
    #[serde(default)]
    pub ranges: Option<Vec<String>>,
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

#[derive(Debug, Serialize, Deserialize)]
pub struct RunIterationsResult {
    pub total_iterations: u32,
    pub exploitability: f32,
    pub exploitability_pct: f32,
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

#[derive(Debug, Serialize, Deserialize)]
pub struct RiverCardsResult {
    pub has_river_cards: bool,
    pub cards: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TurnCardsResult {
    pub has_turn_cards: bool,
    pub cards: Vec<String>,
}

// === Tree View Types ===

#[derive(Debug, Serialize, Deserialize)]
pub struct NodeState {
    pub node_type: String,
    pub street: String,
    pub player_to_act: Option<usize>,
    pub player_name: Option<String>,
    pub pot: i32,
    pub stacks: Vec<i32>,
    pub actions: Vec<ActionInfo>,
    pub action_history: Vec<ActionHistoryItem>,
    pub path: String,
    pub terminal_result: Option<String>,
    pub error: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ActionInfo {
    pub index: usize,
    pub name: String,
    pub short_name: String,
    pub action_type: String,
    pub amount: Option<i32>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ActionHistoryItem {
    pub player: usize,
    pub player_name: String,
    pub action: String,
    pub path: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TreeNodeInfo {
    pub path: String,
    pub node_type: String,
    pub label: String,
    pub action_type: Option<String>,
    pub player_name: Option<String>,
    pub terminal_result: Option<String>,
    pub street: String,
    pub has_children: bool,
    pub subtree_size: usize,
    pub children: Vec<TreeNodeInfo>,
}

// === Tree Building Commands ===

#[tauri::command]
pub fn build_tree(config_json: String) -> BuildResult {
    let spot_config: SpotConfig = match serde_json::from_str(&config_json) {
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

#[tauri::command]
pub fn validate_config(config_json: String) -> ValidationResult {
    let mut errors = Vec::new();

    let spot_config: SpotConfig = match serde_json::from_str(&config_json) {
        Ok(c) => c,
        Err(e) => {
            return ValidationResult {
                valid: false,
                errors: vec![format!("JSON parse error: {}", e)],
            }
        }
    };

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

        if let Err(e) = parse_bet_sizes_str(&pf.open_sizes.bet, false) {
            errors.push(format!("Invalid open bet sizes: {}", e));
        }
        if let Err(e) = parse_bet_sizes_str(&pf.open_sizes.raise, true) {
            errors.push(format!("Invalid open raise sizes: {}", e));
        }
    }

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

#[tauri::command]
pub fn parse_bet_size_preview(size_str: String, pot: i32, stack: i32) -> BetSizePreview {
    match parse_bet_size(&size_str, true) {
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

#[tauri::command]
pub fn get_default_config() -> String {
    let config = SpotConfig {
        name: "HU 100bb".to_string(),
        num_players: 2,
        starting_stacks: vec![200],
        starting_street: None,
        starting_pot: 0,
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

// === Tree Navigation Commands ===

#[tauri::command]
pub fn get_tree_root(config_json: String) -> TreeNodeInfo {
    let spot_config: SpotConfig = match serde_json::from_str(&config_json) {
        Ok(c) => c,
        Err(e) => {
            return error_tree_node(&format!("Failed to parse config: {}", e));
        }
    };

    let tree_config = match convert_config(&spot_config) {
        Ok(c) => c,
        Err(e) => {
            return error_tree_node(&e);
        }
    };

    let tree = match ActionTree::new(tree_config) {
        Ok(t) => t,
        Err(e) => {
            return error_tree_node(&e);
        }
    };

    let positions = Position::all_for_players(spot_config.num_players);
    let starting_street = determine_starting_street(&tree);
    build_tree_node(&tree.root, "", None, positions, starting_street, true)
}

#[tauri::command]
pub fn get_tree_children(config_json: String, path: String) -> Vec<TreeNodeInfo> {
    let spot_config: SpotConfig = match serde_json::from_str(&config_json) {
        Ok(c) => c,
        Err(_) => return vec![],
    };

    let tree_config = match convert_config(&spot_config) {
        Ok(c) => c,
        Err(_) => return vec![],
    };

    let tree = match ActionTree::new(tree_config) {
        Ok(t) => t,
        Err(_) => return vec![],
    };

    let positions = Position::all_for_players(spot_config.num_players);
    let starting_street = determine_starting_street(&tree);

    let node = navigate_to_node(&tree.root, &path, starting_street);
    let (node, street) = match node {
        Some((n, s)) => (n, s),
        None => return vec![],
    };

    get_children_info(node, &path, positions, street)
}

#[tauri::command]
pub fn get_node_at_path(config_json: String, path: String) -> NodeState {
    let spot_config: SpotConfig = match serde_json::from_str(&config_json) {
        Ok(c) => c,
        Err(e) => {
            return error_node_state(&format!("Failed to parse config: {}", e));
        }
    };

    let tree_config = match convert_config(&spot_config) {
        Ok(c) => c,
        Err(e) => {
            return error_node_state(&e);
        }
    };

    let tree = match ActionTree::new(tree_config.clone()) {
        Ok(t) => t,
        Err(e) => {
            return error_node_state(&e);
        }
    };

    navigate_to_path(&tree, &spot_config, &path)
}

#[tauri::command]
pub fn get_action_result(config_json: String, path: String, action_idx: usize) -> NodeState {
    let new_path = if path.is_empty() {
        action_idx.to_string()
    } else {
        format!("{}.{}", path, action_idx)
    };

    get_node_at_path(config_json, new_path)
}

// === Solver Commands ===

#[tauri::command]
pub fn create_solver(state: State<SolverState>, config_json: String) -> CreateSolverResult {
    let solver_config: SolverConfig = match serde_json::from_str(&config_json) {
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

    let game = if let Some(ref range_strings) = solver_config.ranges {
        let num_players = range_strings.len();
        if num_players < 2 || num_players > 6 {
            return CreateSolverResult {
                success: false,
                error: Some(format!("Number of players must be 2-6, got {}", num_players)),
                num_ip_hands: None,
                num_oop_hands: None,
            };
        }

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
                let default_stack = solver_config.effective_stack.unwrap_or(100);
                vec![default_stack; num_players]
            }
        };

        PostflopGame::new_multiway(indexed_tree, board, ranges, stacks, pot)
    } else {
        let oop_range_str = match &solver_config.oop_range {
            Some(s) => s.as_str(),
            None => {
                return CreateSolverResult {
                    success: false,
                    error: Some(
                        "Missing oop_range (or use 'ranges' for n-player format)".to_string(),
                    ),
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
                    error: Some(
                        "Missing ip_range (or use 'ranges' for n-player format)".to_string(),
                    ),
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

    *state.game.lock().unwrap() = Some(game);
    *state.solver.lock().unwrap() = Some(solver);
    *state.pot.lock().unwrap() = pot;

    CreateSolverResult {
        success: true,
        error: None,
        num_ip_hands: Some(num_ip),
        num_oop_hands: Some(num_oop),
    }
}

#[tauri::command]
pub fn run_iterations(state: State<SolverState>, count: u32) -> RunIterationsResult {
    let mut game_guard = state.game.lock().unwrap();
    let mut solver_guard = state.solver.lock().unwrap();
    let pot_guard = state.pot.lock().unwrap();

    let (game, solver) = match (game_guard.as_mut(), solver_guard.as_mut()) {
        (Some(g), Some(s)) => (g, s),
        _ => {
            return RunIterationsResult {
                total_iterations: 0,
                exploitability: 0.0,
                exploitability_pct: 0.0,
            }
        }
    };

    solver.train(game, count);

    let exploit = solver.exploitability(game);
    let pot = *pot_guard as f32;
    let exploit_pct = if pot > 0.0 {
        exploit / pot * 100.0
    } else {
        0.0
    };

    RunIterationsResult {
        total_iterations: solver.total_iterations(),
        exploitability: exploit,
        exploitability_pct: exploit_pct,
    }
}

#[tauri::command]
pub fn get_exploitability(state: State<SolverState>) -> f32 {
    let game_guard = state.game.lock().unwrap();
    let solver_guard = state.solver.lock().unwrap();

    match (game_guard.as_ref(), solver_guard.as_ref()) {
        (Some(game), Some(solver)) => solver.exploitability(game),
        _ => 0.0,
    }
}

// === Strategy Commands ===

#[tauri::command]
pub fn get_node_strategy(state: State<SolverState>, path: String, player: usize) -> NodeStrategyResult {
    let game_guard = state.game.lock().unwrap();
    let solver_guard = state.solver.lock().unwrap();

    let (game, solver) = match (game_guard.as_ref(), solver_guard.as_ref()) {
        (Some(g), Some(s)) => (g, s),
        _ => {
            return NodeStrategyResult {
                success: false,
                error: Some("No solver active".to_string()),
                action_names: vec![],
                hands: vec![],
                aggregate: vec![],
            }
        }
    };

    get_node_strategy_internal(game, solver, &path, player)
}

#[tauri::command]
pub fn get_node_strategy_for_context(
    state: State<SolverState>,
    path: String,
    player: usize,
    context: i32,
) -> NodeStrategyResult {
    let game_guard = state.game.lock().unwrap();
    let solver_guard = state.solver.lock().unwrap();

    let (game, solver) = match (game_guard.as_ref(), solver_guard.as_ref()) {
        (Some(g), Some(s)) => (g, s),
        _ => {
            return NodeStrategyResult {
                success: false,
                error: Some("No solver active".to_string()),
                action_names: vec![],
                hands: vec![],
                aggregate: vec![],
            }
        }
    };

    if context < 0 {
        get_node_strategy_internal(game, solver, &path, player)
    } else {
        get_node_strategy_for_context_internal(game, solver, &path, player, context as usize)
    }
}

#[tauri::command]
pub fn get_river_cards(state: State<SolverState>) -> RiverCardsResult {
    let solver_guard = state.solver.lock().unwrap();

    match solver_guard.as_ref() {
        Some(solver) => {
            let cards = solver.river_card_strings();
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
}

#[tauri::command]
pub fn get_turn_cards(state: State<SolverState>) -> TurnCardsResult {
    let solver_guard = state.solver.lock().unwrap();

    match solver_guard.as_ref() {
        Some(solver) => {
            let cards = solver.turn_card_strings();
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
}

#[tauri::command]
pub fn is_node_below_chance(state: State<SolverState>, path: String) -> bool {
    let game_guard = state.game.lock().unwrap();
    let solver_guard = state.solver.lock().unwrap();

    let (game, solver) = match (game_guard.as_ref(), solver_guard.as_ref()) {
        (Some(g), Some(s)) => (g, s),
        _ => return false,
    };

    let tree = &game.tree;
    let node_idx = match navigate_indexed_tree(tree, &path) {
        Ok(idx) => idx,
        Err(_) => return false,
    };

    solver.num_contexts(node_idx) > 1
}

#[tauri::command]
pub fn get_chance_depth(state: State<SolverState>, path: String) -> i32 {
    let game_guard = state.game.lock().unwrap();
    let solver_guard = state.solver.lock().unwrap();

    let (game, solver) = match (game_guard.as_ref(), solver_guard.as_ref()) {
        (Some(g), Some(s)) => (g, s),
        _ => return 0,
    };

    let tree = &game.tree;
    let node_idx = match navigate_indexed_tree(tree, &path) {
        Ok(idx) => idx,
        Err(_) => return 0,
    };

    solver.chance_depth(node_idx) as i32
}

// === Internal Helper Functions ===

fn get_node_strategy_internal(
    game: &PostflopGame,
    solver: &PostflopSolver,
    path: &str,
    player: usize,
) -> NodeStrategyResult {
    let tree = &game.tree;

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

    let num_hands = solver.num_hands(player);
    let num_contexts = solver.num_contexts(node_idx);

    let mut hands = Vec::with_capacity(num_hands);
    let mut aggregate = vec![0.0f32; num_actions];
    let mut total_weight = 0.0f32;

    for h in 0..num_hands {
        let (combo_idx, weight) = solver.hand_info(player, h);
        let combo = Combo::from_index(combo_idx);
        let combo_str = combo_to_string(combo);

        let mut actions = vec![0.0f32; num_actions];
        for ctx in 0..num_contexts {
            let ctx_strategy = solver.get_hand_strategy_ctx(node_idx, h, player, ctx);
            for (a, &prob) in ctx_strategy.iter().enumerate() {
                actions[a] += prob;
            }
        }
        if num_contexts > 1 {
            let inv = 1.0 / num_contexts as f32;
            for a in &mut actions {
                *a *= inv;
            }
        }

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

fn get_node_strategy_for_context_internal(
    game: &PostflopGame,
    solver: &PostflopSolver,
    path: &str,
    player: usize,
    ctx: usize,
) -> NodeStrategyResult {
    let tree = &game.tree;

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

    let (turn_card, river_card) = solver.context_cards(node_idx, ctx);

    let num_hands = solver.num_hands(player);
    let mut hands = Vec::with_capacity(num_hands);
    let mut aggregate = vec![0.0f32; num_actions];
    let mut total_weight = 0.0f32;

    for h in 0..num_hands {
        let (combo_idx, weight) = solver.hand_info(player, h);

        let (c0, c1) = solver.hand_cards(player, h);
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

        let actions = solver.get_hand_strategy_ctx(node_idx, h, player, ctx);

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

fn navigate_indexed_tree(tree: &IndexedActionTree, path: &str) -> Result<usize, String> {
    let mut node_idx = tree.root_idx;

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

        while tree.get(node_idx).is_chance() {
            if tree.get(node_idx).children.is_empty() {
                break;
            }
            node_idx = tree.get(node_idx).children[0];
        }
    }

    Ok(node_idx)
}

// === Config Conversion Functions ===

fn convert_config(spot: &SpotConfig) -> Result<TreeConfig, String> {
    let bet_type = match spot.bet_type.to_lowercase().as_str() {
        "potlimit" | "pot_limit" | "pot-limit" => BetType::PotLimit,
        _ => BetType::NoLimit,
    };

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

    if let Some(ref pf) = spot.preflop {
        config.preflop = Some(convert_preflop(pf)?);
    }

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

// === Tree Navigation Helper Functions ===

fn determine_starting_street(tree: &ActionTree) -> Street {
    tree.config.starting_street.unwrap_or_else(|| {
        if tree.config.preflop.is_some() {
            Street::Preflop
        } else if tree.config.flop.is_some() {
            Street::Flop
        } else if tree.config.turn.is_some() {
            Street::Turn
        } else {
            Street::River
        }
    })
}

fn navigate_to_path(tree: &ActionTree, spot_config: &SpotConfig, path: &str) -> NodeState {
    let num_players = spot_config.num_players;
    let positions = Position::all_for_players(num_players);

    let indices: Vec<usize> = if path.is_empty() {
        Vec::new()
    } else {
        match path
            .split('.')
            .map(|s| s.parse::<usize>())
            .collect::<Result<Vec<_>, _>>()
        {
            Ok(v) => v,
            Err(_) => return error_node_state(&format!("Invalid path: {}", path)),
        }
    };

    let mut current_node = &tree.root;
    let mut action_history = Vec::new();
    let mut current_path = String::new();

    let starting_street = determine_starting_street(tree);

    let mut pot = if starting_street == Street::Preflop {
        tree.config.starting_pot()
    } else {
        if tree.config.starting_pot > 0 {
            tree.config.starting_pot
        } else {
            tree.config.starting_pot()
        }
    };
    let mut stacks: Vec<i32> = (0..num_players)
        .map(|i| tree.config.stack_for_player(i))
        .collect();
    let mut street = starting_street;

    if starting_street == Street::Preflop {
        if let Some(ref pf) = tree.config.preflop {
            let (sb_seat, bb_seat) = solver::tree::position::blind_seats(num_players);
            stacks[sb_seat] -= pf.blinds[0];
            stacks[bb_seat] -= pf.blinds[1];
        }
    }

    for (i, &action_idx) in indices.iter().enumerate() {
        match current_node {
            ActionTreeNode::Player {
                player,
                actions,
                children,
            } => {
                if action_idx >= actions.len() {
                    return error_node_state(&format!(
                        "Invalid action index {} at path position {}",
                        action_idx, i
                    ));
                }

                let action = &actions[action_idx];
                let player_name = positions[*player].short_name().to_string();

                update_state_for_action(&mut pot, &mut stacks, *player, action);

                action_history.push(ActionHistoryItem {
                    player: *player,
                    player_name,
                    action: action.display(),
                    path: current_path.clone(),
                });

                if current_path.is_empty() {
                    current_path = action_idx.to_string();
                } else {
                    current_path = format!("{}.{}", current_path, action_idx);
                }

                current_node = &children[action_idx];
            }
            ActionTreeNode::Chance {
                street: next_street,
                child,
            } => {
                street = *next_street;
                current_node = child;
            }
            ActionTreeNode::Terminal { .. } => {
                return error_node_state("Cannot navigate past terminal node");
            }
        }
    }

    build_node_state_impl(
        current_node,
        positions,
        pot,
        stacks,
        street,
        action_history,
        path.to_string(),
    )
}

fn update_state_for_action(pot: &mut i32, stacks: &mut [i32], player: usize, action: &Action) {
    match action {
        Action::Fold | Action::Check => {}
        Action::Call(amount) | Action::Bet(amount) | Action::Raise(amount) | Action::AllIn(amount) => {
            let bet_amount = *amount;
            if stacks[player] >= bet_amount {
                stacks[player] -= bet_amount;
                *pot += bet_amount;
            }
        }
    }
}

fn build_node_state_impl(
    node: &ActionTreeNode,
    positions: &[Position],
    pot: i32,
    stacks: Vec<i32>,
    street: Street,
    action_history: Vec<ActionHistoryItem>,
    path: String,
) -> NodeState {
    match node {
        ActionTreeNode::Player {
            player,
            actions,
            ..
        } => {
            let player_name = positions[*player].short_name().to_string();
            let action_infos: Vec<ActionInfo> = actions
                .iter()
                .enumerate()
                .map(|(i, a)| ActionInfo {
                    index: i,
                    name: a.display(),
                    short_name: action_short_name(a),
                    action_type: action_type_name(a),
                    amount: action_amount(a),
                })
                .collect();

            NodeState {
                node_type: "player".to_string(),
                street: street.short_name().to_string(),
                player_to_act: Some(*player),
                player_name: Some(player_name),
                pot,
                stacks,
                actions: action_infos,
                action_history,
                path,
                terminal_result: None,
                error: None,
            }
        }
        ActionTreeNode::Chance {
            street: next_street,
            child,
        } => {
            build_node_state_impl(
                child,
                positions,
                pot,
                stacks,
                *next_street,
                action_history,
                path,
            )
        }
        ActionTreeNode::Terminal { result, pot: final_pot } => {
            let terminal_str = match result {
                TerminalResult::Fold { winner } => {
                    format!("Fold - {} wins", positions[*winner].short_name())
                }
                TerminalResult::Showdown => "Showdown".to_string(),
                TerminalResult::AllInRunout { num_players } => {
                    format!("All-in Runout ({} players)", num_players)
                }
            };

            NodeState {
                node_type: "terminal".to_string(),
                street: street.short_name().to_string(),
                player_to_act: None,
                player_name: None,
                pot: *final_pot,
                stacks,
                actions: vec![],
                action_history,
                path,
                terminal_result: Some(terminal_str),
                error: None,
            }
        }
    }
}

fn action_short_name(action: &Action) -> String {
    match action {
        Action::Fold => "F".to_string(),
        Action::Check => "X".to_string(),
        Action::Call(a) => format!("C{}", a),
        Action::Bet(a) => format!("B{}", a),
        Action::Raise(a) => format!("R{}", a),
        Action::AllIn(a) => format!("A{}", a),
    }
}

fn action_type_name(action: &Action) -> String {
    match action {
        Action::Fold => "fold".to_string(),
        Action::Check => "check".to_string(),
        Action::Call(_) => "call".to_string(),
        Action::Bet(_) => "bet".to_string(),
        Action::Raise(_) => "raise".to_string(),
        Action::AllIn(_) => "allin".to_string(),
    }
}

fn action_amount(action: &Action) -> Option<i32> {
    match action {
        Action::Fold | Action::Check => None,
        Action::Call(a) | Action::Bet(a) | Action::Raise(a) | Action::AllIn(a) => Some(*a),
    }
}

fn error_node_state(message: &str) -> NodeState {
    NodeState {
        node_type: "error".to_string(),
        street: "".to_string(),
        player_to_act: None,
        player_name: None,
        pot: 0,
        stacks: vec![],
        actions: vec![],
        action_history: vec![],
        path: "".to_string(),
        terminal_result: None,
        error: Some(message.to_string()),
    }
}

// === Tree View Helper Functions ===

fn navigate_to_node<'a>(
    root: &'a ActionTreeNode,
    path: &str,
    starting_street: Street,
) -> Option<(&'a ActionTreeNode, Street)> {
    if path.is_empty() {
        return Some((root, starting_street));
    }

    let indices: Vec<usize> = path
        .split('.')
        .map(|s| s.parse::<usize>())
        .collect::<Result<Vec<_>, _>>()
        .ok()?;

    let mut current = root;
    let mut street = starting_street;

    for &idx in &indices {
        match current {
            ActionTreeNode::Player { children, .. } => {
                current = children.get(idx)?;
            }
            ActionTreeNode::Chance {
                child,
                street: next_street,
            } => {
                street = *next_street;
                current = child;
            }
            ActionTreeNode::Terminal { .. } => return None,
        }

        while let ActionTreeNode::Chance {
            child,
            street: next_street,
        } = current
        {
            street = *next_street;
            current = child;
        }
    }

    Some((current, street))
}

fn get_children_info(
    node: &ActionTreeNode,
    parent_path: &str,
    positions: &[Position],
    street: Street,
) -> Vec<TreeNodeInfo> {
    match node {
        ActionTreeNode::Player {
            actions, children, ..
        } => actions
            .iter()
            .zip(children.iter())
            .enumerate()
            .map(|(i, (action, child))| {
                let child_path = if parent_path.is_empty() {
                    i.to_string()
                } else {
                    format!("{}.{}", parent_path, i)
                };

                let (actual_child, child_street) = skip_chance_nodes(child, street);

                build_tree_node(
                    actual_child,
                    &child_path,
                    Some(action),
                    positions,
                    child_street,
                    false,
                )
            })
            .collect(),
        _ => vec![],
    }
}

fn skip_chance_nodes(node: &ActionTreeNode, street: Street) -> (&ActionTreeNode, Street) {
    let mut current = node;
    let mut current_street = street;

    while let ActionTreeNode::Chance {
        child,
        street: next_street,
    } = current
    {
        current_street = *next_street;
        current = child;
    }

    (current, current_street)
}

fn build_tree_node(
    node: &ActionTreeNode,
    path: &str,
    action: Option<&Action>,
    positions: &[Position],
    street: Street,
    include_children: bool,
) -> TreeNodeInfo {
    let (node_type, label, action_type, player_name, terminal_result, has_children) = match node {
        ActionTreeNode::Player {
            player, actions, ..
        } => {
            let pos_name = positions[*player].short_name().to_string();
            let label = if let Some(a) = action {
                a.display()
            } else {
                format!("{} to act", pos_name)
            };
            let act_type = action.map(action_type_name);
            (
                "player".to_string(),
                label,
                act_type,
                Some(pos_name),
                None,
                !actions.is_empty(),
            )
        }
        ActionTreeNode::Chance { .. } => (
            "chance".to_string(),
            "Deal".to_string(),
            None,
            None,
            None,
            true,
        ),
        ActionTreeNode::Terminal { result, .. } => {
            let term_str = match result {
                TerminalResult::Fold { winner } => {
                    format!("{} wins", positions[*winner].short_name())
                }
                TerminalResult::Showdown => "Showdown".to_string(),
                TerminalResult::AllInRunout { num_players } => {
                    format!("All-in ({})", num_players)
                }
            };
            let label = if let Some(a) = action {
                format!("{} -> {}", a.display(), term_str)
            } else {
                term_str.clone()
            };
            let act_type = action.map(action_type_name);
            (
                "terminal".to_string(),
                label,
                act_type,
                None,
                Some(term_str),
                false,
            )
        }
    };

    let subtree_size = node.node_count();

    let children = if include_children && has_children {
        get_children_info(node, path, positions, street)
    } else {
        vec![]
    };

    TreeNodeInfo {
        path: path.to_string(),
        node_type,
        label,
        action_type,
        player_name,
        terminal_result,
        street: street.short_name().to_string(),
        has_children,
        subtree_size,
        children,
    }
}

fn error_tree_node(message: &str) -> TreeNodeInfo {
    TreeNodeInfo {
        path: "".to_string(),
        node_type: "error".to_string(),
        label: message.to_string(),
        action_type: None,
        player_name: None,
        terminal_result: None,
        street: "".to_string(),
        has_children: false,
        subtree_size: 0,
        children: vec![],
    }
}
