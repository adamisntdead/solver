//! PIO benchmark comparison example.
//!
//! Loads benchmark spots from PIO solver output and compares our solver's
//! strategies against the expected results.
//!
//! Note: The PIO benchmark files contain flop spots, but this solver currently
//! only supports river spots. This example demonstrates the framework for
//! comparison once multi-street support is added.
//!
//! Run with: `cargo run --example pio_benchmark --release`

#![allow(dead_code)]

use solver::poker::hands::{Board, Range};
use solver::poker::isomorphism::RiverIsomorphism;
use solver::poker::postflop_game::PostflopGame;
use solver::{
    ActionTree, BetSizeOptions, CfrSolver, CfrVariant, Game, GameNode, Street, StreetConfig,
    TreeConfig,
};
use std::fs;

/// A benchmark spot from PIO.
#[derive(Debug)]
struct BenchmarkSpot {
    oop_range: String,
    ip_range: String,
    board: String,
    stacks: i32,
    common_pot: i32,
    raises_str: Vec<String>,
    expected_oop_strat: Vec<f64>,
    expected_ip_strat: Vec<f64>,
}

fn main() {
    println!("=== PIO Benchmark Comparison ===\n");

    // For now, demonstrate with a manually-created river spot
    // since the PIO benchmarks are flop spots
    run_river_benchmark();
}

fn run_river_benchmark() {
    println!("Running river benchmark spot...\n");

    // Create a river spot with moderate ranges
    let board_str = "KhQsJs2c3d";
    let oop_range_str = "AA,KK,QQ,JJ,TT,AKs,AKo,AQs,KQs";
    let ip_range_str = "AA,KK,QQ,JJ,TT,99,AKs,AKo,AQs,KQs,KJs";
    let pot = 100;
    let stack = 100;

    // Parse inputs
    let board = Board::from_str(board_str).expect("Invalid board");
    let oop_range = Range::from_str(oop_range_str).expect("Invalid OOP range");
    let ip_range = Range::from_str(ip_range_str).expect("Invalid IP range");

    println!("Board: {}", board_str);
    println!("Pot: {}, Stack: {}", pot, stack);
    println!(
        "OOP range: {} combos",
        oop_range.count_combos()
    );
    println!("IP range: {} combos", ip_range.count_combos());

    // Isomorphism stats
    let iso = RiverIsomorphism::new(&board);
    let stats = iso.stats();
    println!(
        "Isomorphism: {} combos -> {} buckets ({:.1}x compression)",
        stats.valid_combos, stats.num_buckets, stats.compression_ratio
    );

    // Build tree with 50% and 100% pot bet sizes (similar to PIO)
    let sizes = BetSizeOptions::try_from_strs("50%, 100%", "2x, a").expect("Invalid bet sizes");
    let config = TreeConfig::new(2)
        .with_stack(stack)
        .with_starting_street(Street::River)
        .with_starting_pot(pot)
        .with_river(StreetConfig::uniform(sizes));

    let tree = ActionTree::new(config).expect("Failed to build tree");
    let indexed_tree = tree.to_indexed();

    println!("Tree: {} nodes", indexed_tree.len());

    // Create game
    let game = PostflopGame::new(indexed_tree, board, oop_range, ip_range, pot, stack);
    println!("Valid matchups: {}", game.num_matchups());
    println!("Info sets: {}", game.num_info_sets());

    // Run CFR with increasing iterations
    let iteration_counts = [1000, 5000];

    for &iters in &iteration_counts {
        let mut solver = CfrSolver::new(&game, CfrVariant::LinearCfr);

        // Check exploitability BEFORE training
        let exploit_before = solver.exploitability(&game);
        println!("\n--- {} iterations ---", iters);
        println!("Exploitability BEFORE training: {:.4} chips", exploit_before);

        solver.train(&game, iters);

        let exploitability = solver.exploitability(&game);
        let exploit_pct = exploitability / pot as f64 * 100.0;
        println!("Exploitability AFTER training: {:.4} chips ({:.2}% pot)", exploitability, exploit_pct);

        // Print strategies for all unique info sets at betting root
        println!("Strategies at betting root (OOP acts first, P1):");
        let mut seen_ids = std::collections::HashSet::new();
        for (oop_combo, ip_combo) in game.matchups() {
            let node = game.root_for_matchup(oop_combo, ip_combo);
            if node.is_terminal() || node.is_chance() {
                continue;
            }
            let player = node.current_player();
            let info_id = node.info_set_id();
            if seen_ids.contains(&info_id) {
                continue;
            }
            seen_ids.insert(info_id);

            let combo = node.combo(player);
            if let Some(strategy) = solver.get_strategy(info_id) {
                println!(
                    "  P{} {}: {:?}",
                    player,
                    solver::poker::hands::combo_to_string(combo),
                    strategy.iter().map(|p| format!("{:.0}%", p * 100.0)).collect::<Vec<_>>()
                );
            }
        }
    }

    println!("\n=== Benchmark Complete ===");
}

/// Compute aggregate strategy frequencies at the root for both players.
///
/// Returns (OOP strategy, IP strategy) as probability distributions.
fn compute_aggregate_strategies(
    game: &PostflopGame,
    solver: &CfrSolver,
) -> (Vec<f64>, Vec<f64>) {
    let root = game.root();

    if root.is_terminal() || root.is_chance() {
        return (vec![], vec![]);
    }

    let num_actions = root.num_actions();
    let mut oop_totals = vec![0.0; num_actions];
    let mut ip_totals = vec![0.0; num_actions];
    let mut oop_weight = 0.0;
    let mut ip_weight = 0.0;

    for (oop_combo, ip_combo) in game.matchups() {
        let matchup_weight = game.matchup_weight(oop_combo, ip_combo) as f64;

        // Get root node for this matchup
        let node = game.root_for_matchup(oop_combo, ip_combo);
        if node.is_terminal() || node.is_chance() {
            continue;
        }

        let player = node.current_player();
        let info_id = node.info_set_id();

        if let Some(strategy) = solver.get_strategy(info_id) {
            for (i, &prob) in strategy.iter().enumerate() {
                if player == 0 {
                    oop_totals[i] += matchup_weight * prob;
                } else {
                    ip_totals[i] += matchup_weight * prob;
                }
            }

            if player == 0 {
                oop_weight += matchup_weight;
            } else {
                ip_weight += matchup_weight;
            }
        }
    }

    // Normalize
    if oop_weight > 0.0 {
        for x in &mut oop_totals {
            *x /= oop_weight;
        }
    }
    if ip_weight > 0.0 {
        for x in &mut ip_totals {
            *x /= ip_weight;
        }
    }

    (oop_totals, ip_totals)
}

/// Load benchmark spots from JSON file (for future use).
#[allow(dead_code)]
fn load_benchmark_spots(path: &str) -> Result<Vec<BenchmarkSpot>, String> {
    let content = fs::read_to_string(path).map_err(|e| e.to_string())?;

    // Simple JSON parsing (would use serde in production)
    // For now, return empty
    let _ = content;
    Ok(vec![])
}
