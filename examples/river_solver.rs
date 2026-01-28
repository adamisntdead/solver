//! River solver example.
//!
//! Demonstrates solving a simple river spot using the poker CFR solver.
//!
//! Run with: `cargo run --example river_solver --release`

use solver::poker::hands::{combo_to_string, Board, Range};
use solver::poker::isomorphism::RiverIsomorphism;
use solver::poker::postflop_game::PostflopGame;
use solver::{
    ActionTree, BetSizeOptions, CfrSolver, CfrVariant, Game, GameNode, Street, StreetConfig,
    TreeConfig,
};

fn main() {
    println!("=== River Solver Example ===\n");

    // Setup: River board, ranges, and betting structure
    let board = Board::from_str("KhQsJs2c3d").expect("Invalid board");
    println!("Board: {}", board.to_string());

    // Simple ranges for demonstration
    let oop_range = Range::from_str("AA,KK,QQ,JJ,TT,AKs,AKo,AQs,KQs").expect("Invalid OOP range");
    let ip_range = Range::from_str("AA,KK,QQ,JJ,AKs,AKo,AQs,AQo,KQs,KJs").expect("Invalid IP range");

    println!("OOP range: {} combos", oop_range.count_combos());
    println!("IP range: {} combos", ip_range.count_combos());

    // Isomorphism statistics
    let iso = RiverIsomorphism::new(&board);
    let stats = iso.stats();
    println!(
        "\nIsomorphism: {} valid combos -> {} buckets (ratio: {:.2}x)",
        stats.valid_combos, stats.num_buckets, stats.compression_ratio
    );

    // Build action tree with simple bet sizes
    let sizes = BetSizeOptions::try_from_strs("50%, 100%", "2x, a").expect("Invalid bet sizes");
    let config = TreeConfig::new(2)
        .with_stack(100) // 100bb effective
        .with_starting_street(Street::River)
        .with_starting_pot(100) // Pot is 100bb
        .with_river(StreetConfig::uniform(sizes));

    let tree = ActionTree::new(config).expect("Failed to build tree");
    println!("\nAction tree: {} nodes", tree.node_count);

    // Convert to indexed tree
    let indexed_tree = tree.to_indexed();
    println!("Indexed tree: {} nodes", indexed_tree.len());

    // Create the postflop game
    let game = PostflopGame::new(indexed_tree, board, oop_range, ip_range, 100, 100);

    println!("\nGame setup:");
    println!("  Valid matchups: {}", game.num_matchups());
    println!("  Total weight: {:.2}", game.total_matchup_weight());
    println!("  Info sets: {}", game.num_info_sets());

    // Run CFR
    println!("\nRunning CFR...");
    let mut solver = CfrSolver::new(&game, CfrVariant::CfrPlus);

    let iterations = 1000;
    solver.train(&game, iterations);

    println!("Completed {} iterations", solver.iterations());

    // Compute exploitability
    let exploitability = solver.exploitability(&game);
    println!("\nExploitability: {:.4} (chips per game)", exploitability);
    println!(
        "Exploitability: {:.2}% of pot",
        exploitability / 100.0 * 100.0
    );

    // Display some strategies
    println!("\n=== Sample Strategies ===");
    display_root_strategies(&game, &solver);
}

fn display_root_strategies(game: &PostflopGame, solver: &CfrSolver) {
    // Get strategies at root for different hands
    let root = game.root();
    if root.is_terminal() || root.is_chance() {
        println!("Root is not a decision node");
        return;
    }

    // Collect strategies by hand type
    let mut seen_ids = std::collections::HashSet::new();

    for (oop_combo, ip_combo) in game.matchups().take(20) {
        let node = game.root_for_matchup(oop_combo, ip_combo);
        if node.is_terminal() || node.is_chance() {
            continue;
        }

        let info_id = node.info_set_id();
        if seen_ids.contains(&info_id) {
            continue;
        }
        seen_ids.insert(info_id);

        if let Some(strategy) = solver.get_strategy(info_id) {
            let combo = node.combo(node.current_player());
            println!(
                "  {}: {:?}",
                combo_to_string(combo),
                strategy
                    .iter()
                    .map(|&p| format!("{:.1}%", p * 100.0))
                    .collect::<Vec<_>>()
            );
        }
    }
}
