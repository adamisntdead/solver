//! WASM bindings for the poker tree builder.

mod tree_view;

use serde::{Deserialize, Serialize};
use solver::tree::bet_size::parse_bet_size;
use solver::{
    ActionTree, BetSizeOptions, BetType, MemoryEstimate, PreflopConfig,
    StreetConfig, TreeConfig, TreeStats,
};
use wasm_bindgen::prelude::*;

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

    /// Starting stacks in chips
    pub starting_stacks: Vec<i32>,

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

// === Internal conversion functions ===

fn convert_config(spot: &SpotConfig) -> Result<TreeConfig, String> {
    let bet_type = match spot.bet_type.to_lowercase().as_str() {
        "potlimit" | "pot_limit" | "pot-limit" => BetType::PotLimit,
        _ => BetType::NoLimit,
    };

    let mut config = TreeConfig {
        num_players: spot.num_players,
        starting_stacks: spot.starting_stacks.clone(),
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
