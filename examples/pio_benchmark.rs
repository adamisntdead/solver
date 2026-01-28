//! PIO benchmark comparison example.
//!
//! Tests our vectorized CFR solver on a river spot and measures
//! exploitability convergence.
//!
//! Run with: `cargo run --example pio_benchmark --release`

#![allow(dead_code)]

use solver::poker::hands::{combo_to_string, Board, Combo, Range};
use solver::poker::isomorphism::RiverIsomorphism;
use solver::poker::postflop_game::PostflopGame;
use solver::poker::postflop_solver::PostflopSolver;
use solver::{ActionTree, BetSizeOptions, Street, StreetConfig, TreeConfig};

fn main() {
    println!("=== Vectorized CFR Benchmark ===\n");
    run_river_benchmark();
}

fn run_river_benchmark() {
    println!("Running river benchmark spot...\n");

    let board_str = "KhQsJs2c3d";
    let oop_range_str = "AA,KK,QQ,JJ,TT,AKs,AKo,AQs,KQs";
    let ip_range_str = "AA,KK,QQ,JJ,TT,99,AKs,AKo,AQs,KQs,KJs";
    let pot = 100;
    let stack = 100;

    let board = Board::from_str(board_str).expect("Invalid board");
    let oop_range = Range::from_str(oop_range_str).expect("Invalid OOP range");
    let ip_range = Range::from_str(ip_range_str).expect("Invalid IP range");

    println!("Board: {}", board_str);
    println!("Pot: {}, Stack: {}", pot, stack);
    println!("OOP range: {} combos", oop_range.count_combos());
    println!("IP range: {} combos", ip_range.count_combos());

    let iso = RiverIsomorphism::new(&board);
    let stats = iso.stats();
    println!(
        "Isomorphism: {} combos -> {} buckets ({:.1}x compression)",
        stats.valid_combos, stats.num_buckets, stats.compression_ratio
    );

    let sizes = BetSizeOptions::try_from_strs("50%, 100%", "2x, a").expect("Invalid bet sizes");
    let config = TreeConfig::new(2)
        .with_stack(stack)
        .with_starting_street(Street::River)
        .with_starting_pot(pot)
        .with_river(StreetConfig::uniform(sizes));

    let tree = ActionTree::new(config).expect("Failed to build tree");
    let indexed_tree = tree.to_indexed();

    println!("Tree: {} nodes", indexed_tree.len());

    let game = PostflopGame::new(indexed_tree, board, oop_range, ip_range, pot, stack);
    println!("Valid matchups: {}", game.num_matchups());

    // Use the new vectorized solver
    let mut solver = PostflopSolver::new(&game);
    println!(
        "Solver: IP hands={}, OOP hands={}",
        solver.num_hands(0),
        solver.num_hands(1)
    );

    let exploit_before = solver.exploitability(&game);
    println!(
        "\nExploitability BEFORE training: {:.4} chips ({:.2}% pot)",
        exploit_before,
        exploit_before / pot as f32 * 100.0
    );

    // Train with increasing iterations and report convergence
    let checkpoints = [100, 500, 1000, 5000, 10000];

    let mut last_iters = 0u32;
    for &target in &checkpoints {
        let delta = target - last_iters;
        solver.train(&game, delta);
        last_iters = target;

        let exploit = solver.exploitability(&game);
        let exploit_pct = exploit / pot as f32 * 100.0;
        println!(
            "After {:>5} iterations: exploitability = {:.4} chips ({:.3}% pot)",
            target, exploit, exploit_pct
        );
    }

    // Print strategies at the root for OOP (player 1, acts first)
    println!("\n--- OOP strategies at betting root (player 1) ---");
    let root_idx = game.tree.root_idx;
    let root_node = game.tree.get(root_idx);
    let num_actions = root_node.actions.len();
    println!(
        "Actions: {:?}",
        root_node.actions.iter().map(|a| format!("{:?}", a)).collect::<Vec<_>>()
    );

    // OOP = player 1 in tree
    let oop_player = 1;
    let num_oop_hands = solver.num_hands(oop_player);
    for h in 0..num_oop_hands {
        let (combo_idx, _weight) = solver.hand_info(oop_player, h);
        let combo = Combo::from_index(combo_idx);
        let strat = solver.get_hand_strategy(root_idx, h, oop_player);
        if strat.is_empty() {
            continue;
        }
        println!(
            "  {}: [{}]",
            combo_to_string(combo),
            strat
                .iter()
                .map(|p| format!("{:.0}%", p * 100.0))
                .collect::<Vec<_>>()
                .join(", ")
        );
    }

    println!("\n=== Benchmark Complete ===");
}
