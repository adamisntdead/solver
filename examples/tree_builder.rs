//! Example demonstrating the poker game tree builder.
//!
//! Run with: cargo run --example tree_builder

use solver::{
    ActionTree, BetSize, BetSizeOptions, BetType, Position, PreflopConfig, StreetConfig,
    TreeConfig,
};
use solver::tree::memory::combos;

fn main() {
    println!("=== Poker Game Tree Builder Demo ===\n");

    // Example 1: Simple HU preflop tree
    hu_preflop_example();

    println!("\n{}\n", "=".repeat(60));

    // Example 2: HU postflop tree
    hu_postflop_example();

    println!("\n{}\n", "=".repeat(60));

    // Example 3: 6-max preflop tree
    sixmax_preflop_example();

    println!("\n{}\n", "=".repeat(60));

    // Example 4: Pot-limit tree
    pot_limit_example();
}

fn hu_preflop_example() {
    println!("=== HU 100BB Preflop Tree ===\n");

    // Create bet sizes
    let open_sizes = BetSizeOptions {
        bet: vec![
            BetSize::PotRelative(0.5), // 2.5x (roughly)
            BetSize::PotRelative(0.67),
        ],
        raise: vec![BetSize::PrevBetRelative(3.0), BetSize::AllIn],
        reraise: vec![BetSize::PrevBetRelative(2.5), BetSize::AllIn],
        reraise_plus: vec![BetSize::AllIn],
    };

    let preflop = PreflopConfig::new(1, 2)
        .with_open_sizes(vec![open_sizes.clone()])
        .with_3bet_sizes(vec![open_sizes.clone()])
        .with_4bet_sizes(vec![open_sizes]);

    let config = TreeConfig::new(2)
        .with_stack(200) // 100bb in chips (BB=2)
        .with_preflop(preflop);

    println!("Config:");
    println!("  Players: {}", config.num_players);
    println!("  Stack: {} chips ({}bb)", config.effective_stack(), config.effective_stack() / 2);
    println!("  Blinds: {}/{}",
        config.preflop.as_ref().unwrap().blinds[0],
        config.preflop.as_ref().unwrap().blinds[1]);
    println!("  Bet type: {:?}", config.bet_type);

    match ActionTree::new(config) {
        Ok(tree) => {
            let stats = tree.stats();
            let mem = tree.memory_estimate_preflop();

            println!("\nTree Statistics:");
            println!("  Total nodes: {}", stats.node_count);
            println!("  Terminal nodes: {}", stats.terminal_count);
            println!("  Player nodes: {}", stats.player_node_count);
            println!("  Max depth: {}", stats.max_depth);

            println!("\nMemory Estimate (169 canonical hands):");
            println!("  Info sets: {}", mem.info_set_count);
            println!("  Uncompressed: {}", mem.uncompressed_str());
            println!("  Compressed: {}", mem.compressed_str());
        }
        Err(e) => println!("Error building tree: {}", e),
    }
}

fn hu_postflop_example() {
    println!("=== HU Postflop Tree (Flop to River) ===\n");

    // Standard postflop sizings
    let flop_sizes = BetSizeOptions::try_from_strs("33%, 67%, 100%", "2.5x, a").unwrap();
    let turn_sizes = BetSizeOptions::try_from_strs("50%, 75%, 100%", "2.5x, a").unwrap();
    let river_sizes = BetSizeOptions::try_from_strs("50%, 75%, 100%, 150%", "2.5x, a").unwrap();

    let config = TreeConfig::new(2)
        .with_stack(500)
        .with_flop(StreetConfig::uniform(flop_sizes))
        .with_turn(StreetConfig::uniform(turn_sizes))
        .with_river(StreetConfig::uniform(river_sizes));

    println!("Config:");
    println!("  Players: {}", config.num_players);
    println!("  Effective stack: {}", config.effective_stack());
    println!("  Flop sizes: 33%, 67%, 100%");
    println!("  Turn sizes: 50%, 75%, 100%");
    println!("  River sizes: 50%, 75%, 100%, 150%");
    println!("  Max raises per round: {}", config.max_raises_per_round);

    match ActionTree::new(config) {
        Ok(tree) => {
            let stats = tree.stats();

            // Calculate memory for flop (3 board cards)
            let mem_flop = tree.memory_estimate(combos::FLOP, 4.0);

            println!("\nTree Statistics:");
            println!("  Total nodes: {}", stats.node_count);
            println!("  Terminal nodes: {}", stats.terminal_count);
            println!("  Player nodes: {}", stats.player_node_count);
            println!("  Max depth: {}", stats.max_depth);

            println!("\nMemory Estimate (flop, {} hand combos):", combos::FLOP);
            println!("  Info sets: {}", mem_flop.info_set_count);
            println!("  Uncompressed: {}", mem_flop.uncompressed_str());
            println!("  Compressed: {}", mem_flop.compressed_str());
        }
        Err(e) => println!("Error building tree: {}", e),
    }
}

fn sixmax_preflop_example() {
    println!("=== 6-Max Preflop Tree ===\n");

    // Show positions
    println!("Positions:");
    for (i, pos) in Position::all_for_players(6).iter().enumerate() {
        println!("  Seat {}: {}", i, pos.short_name());
    }

    let open_sizes = BetSizeOptions {
        bet: vec![BetSize::PotRelative(0.5)],
        raise: vec![BetSize::PrevBetRelative(3.0), BetSize::AllIn],
        reraise: vec![BetSize::AllIn],
        reraise_plus: vec![BetSize::AllIn],
    };

    let preflop = PreflopConfig::new(1, 2)
        .with_open_sizes(vec![open_sizes.clone()])
        .with_3bet_sizes(vec![open_sizes.clone()])
        .with_4bet_sizes(vec![open_sizes]);

    let config = TreeConfig::new(6)
        .with_stack(200) // 100bb
        .with_preflop(preflop);

    println!("\nConfig:");
    println!("  Players: {}", config.num_players);
    println!("  Stack: {} chips ({}bb)", config.effective_stack(), config.effective_stack() / 2);
    println!("  Starting pot: {} (blinds + antes)", config.starting_pot());

    match ActionTree::new(config) {
        Ok(tree) => {
            let stats = tree.stats();
            let mem = tree.memory_estimate_preflop();

            println!("\nTree Statistics:");
            println!("  Total nodes: {}", stats.node_count);
            println!("  Terminal nodes: {}", stats.terminal_count);
            println!("  Player nodes: {}", stats.player_node_count);
            println!("  Max depth: {}", stats.max_depth);

            println!("\nMemory Estimate (169 canonical hands):");
            println!("  Info sets: {}", mem.info_set_count);
            println!("  Uncompressed: {}", mem.uncompressed_str());
            println!("  Compressed: {}", mem.compressed_str());
        }
        Err(e) => println!("Error building tree: {}", e),
    }
}

fn pot_limit_example() {
    println!("=== Pot-Limit Omaha Postflop Tree ===\n");

    let sizes = BetSizeOptions::try_from_strs("50%, 75%, 100%", "100%, a").unwrap();

    let config = TreeConfig::new(2)
        .with_stack(500)
        .with_bet_type(BetType::PotLimit)
        .with_flop(StreetConfig::uniform(sizes.clone()))
        .with_turn(StreetConfig::uniform(sizes.clone()))
        .with_river(StreetConfig::uniform(sizes));

    println!("Config:");
    println!("  Players: {}", config.num_players);
    println!("  Effective stack: {}", config.effective_stack());
    println!("  Bet type: {:?}", config.bet_type);
    println!("  (Max bet capped at pot size)");

    match ActionTree::new(config) {
        Ok(tree) => {
            let stats = tree.stats();

            println!("\nTree Statistics:");
            println!("  Total nodes: {}", stats.node_count);
            println!("  Terminal nodes: {}", stats.terminal_count);
            println!("  Player nodes: {}", stats.player_node_count);
            println!("  Max depth: {}", stats.max_depth);
        }
        Err(e) => println!("Error building tree: {}", e),
    }
}
