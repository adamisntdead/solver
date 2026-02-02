//! Performance benchmarks for the CFR solver.
//!
//! Run with: cargo bench
//!
//! These benchmarks track solving speed across different game configurations
//! to detect performance regressions early.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use solver::poker::{
    board_parser::parse_board, postflop_game::PostflopGame, postflop_solver::PostflopSolver,
    range_parser::parse_range,
};
use solver::tree::{
    ActionTree, BetSizeOptions, Street, StreetConfig, TreeConfig,
};

/// Helper to create a river-only game (5 cards on board)
fn make_river_game(board_str: &str, oop_range: &str, ip_range: &str) -> PostflopGame {
    let sizes = BetSizeOptions::try_from_strs("50%, 100%", "2x, a").expect("Invalid bet sizes");
    let config = TreeConfig::new(2)
        .with_stack(100)
        .with_starting_street(Street::River)
        .with_starting_pot(100)
        .with_river(StreetConfig::uniform(sizes));

    let tree = ActionTree::new(config).expect("Failed to build tree");
    let indexed_tree = tree.to_indexed();
    let board = parse_board(board_str).expect("Invalid board");
    let oop_range = parse_range(oop_range).expect("Invalid OOP range");
    let ip_range = parse_range(ip_range).expect("Invalid IP range");

    PostflopGame::new(indexed_tree, board, oop_range, ip_range, 100, 100)
}

/// Helper to create a turn game (4 cards on board)
fn make_turn_game(board_str: &str, oop_range: &str, ip_range: &str) -> PostflopGame {
    let sizes = BetSizeOptions::try_from_strs("67%", "a").expect("Invalid bet sizes");
    let config = TreeConfig::new(2)
        .with_stack(100)
        .with_starting_street(Street::Turn)
        .with_starting_pot(100)
        .with_turn(StreetConfig::uniform(sizes.clone()))
        .with_river(StreetConfig::uniform(sizes));

    let tree = ActionTree::new(config).expect("Failed to build tree");
    let indexed_tree = tree.to_indexed();
    let board = parse_board(board_str).expect("Invalid board");
    let oop_range = parse_range(oop_range).expect("Invalid OOP range");
    let ip_range = parse_range(ip_range).expect("Invalid IP range");

    PostflopGame::new(indexed_tree, board, oop_range, ip_range, 100, 100)
}

/// Helper to create a flop game (3 cards on board)
fn make_flop_game(board_str: &str, oop_range: &str, ip_range: &str) -> PostflopGame {
    let sizes = BetSizeOptions::try_from_strs("67%", "a").expect("Invalid bet sizes");
    let config = TreeConfig::new(2)
        .with_stack(100)
        .with_starting_street(Street::Flop)
        .with_starting_pot(100)
        .with_flop(StreetConfig::uniform(sizes.clone()))
        .with_turn(StreetConfig::uniform(sizes.clone()))
        .with_river(StreetConfig::uniform(sizes));

    let tree = ActionTree::new(config).expect("Failed to build tree");
    let indexed_tree = tree.to_indexed();
    let board = parse_board(board_str).expect("Invalid board");
    let oop_range = parse_range(oop_range).expect("Invalid OOP range");
    let ip_range = parse_range(ip_range).expect("Invalid IP range");

    PostflopGame::new(indexed_tree, board, oop_range, ip_range, 100, 100)
}

/// Benchmark river solving with narrow ranges (fast baseline)
fn bench_river_narrow(c: &mut Criterion) {
    let game = make_river_game("KhQsJs2c3d", "AA,KK,QQ,AKs", "AA,KK,QQ,JJ,TT");
    let iterations = 100;

    c.bench_with_input(
        BenchmarkId::new("river_narrow", iterations),
        &iterations,
        |b, &iters| {
            b.iter(|| {
                let mut solver = PostflopSolver::new(&game);
                solver.train(&game, black_box(iters as u32));
                solver.exploitability(&game)
            });
        },
    );
}

/// Benchmark river solving with medium ranges
fn bench_river_medium(c: &mut Criterion) {
    let game = make_river_game(
        "Td9d6h5c2s",
        "AA,KK,QQ,JJ,TT,99,88,77,66,AKs,AQs,AJs,KQs",
        "AA,KK,QQ,JJ,TT,99,88,AKs,AKo,AQs,AQo,KQs,KQo",
    );
    let iterations = 500;

    c.bench_with_input(
        BenchmarkId::new("river_medium", iterations),
        &iterations,
        |b, &iters| {
            b.iter(|| {
                let mut solver = PostflopSolver::new(&game);
                solver.train(&game, black_box(iters as u32));
                solver.exploitability(&game)
            });
        },
    );
}

/// Benchmark turn solving (multi-street)
fn bench_turn_narrow(c: &mut Criterion) {
    let game = make_turn_game("KhQsJs2c", "AA,KK,QQ", "AA,KK,QQ,JJ");
    let iterations = 100;

    c.bench_with_input(
        BenchmarkId::new("turn_narrow", iterations),
        &iterations,
        |b, &iters| {
            b.iter(|| {
                let mut solver = PostflopSolver::new(&game);
                solver.train(&game, black_box(iters as u32));
                solver.exploitability(&game)
            });
        },
    );
}

/// Benchmark flop solving with narrow ranges
fn bench_flop_narrow(c: &mut Criterion) {
    let game = make_flop_game("KhQsJs", "AA,KK", "QQ,JJ");
    let iterations = 50;

    c.bench_with_input(
        BenchmarkId::new("flop_narrow", iterations),
        &iterations,
        |b, &iters| {
            b.iter(|| {
                let mut solver = PostflopSolver::new(&game);
                solver.train(&game, black_box(iters as u32));
                solver.exploitability(&game)
            });
        },
    );
}

/// Benchmark solver creation time (no training)
fn bench_solver_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("solver_creation");

    // River game creation
    let river_game = make_river_game("KhQsJs2c3d", "AA,KK,QQ,AKs", "AA,KK,QQ,JJ,TT");
    group.bench_function("river", |b| {
        b.iter(|| PostflopSolver::new(black_box(&river_game)));
    });

    // Turn game creation
    let turn_game = make_turn_game("KhQsJs2c", "AA,KK,QQ", "AA,KK,QQ,JJ");
    group.bench_function("turn", |b| {
        b.iter(|| PostflopSolver::new(black_box(&turn_game)));
    });

    // Flop game creation
    let flop_game = make_flop_game("KhQsJs", "AA,KK", "QQ,JJ");
    group.bench_function("flop", |b| {
        b.iter(|| PostflopSolver::new(black_box(&flop_game)));
    });

    group.finish();
}

/// Benchmark iteration throughput
fn bench_iteration_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("iteration_throughput");

    // River: measure iterations per second
    let river_game = make_river_game(
        "Td9d6h5c2s",
        "AA,KK,QQ,JJ,TT,99,88,77,66,AKs,AQs,AJs,KQs",
        "AA,KK,QQ,JJ,TT,99,88,AKs,AKo,AQs,AQo,KQs,KQo",
    );
    let river_iters = 100u64;
    group.throughput(Throughput::Elements(river_iters));
    group.bench_function("river_100_iters", |b| {
        let mut solver = PostflopSolver::new(&river_game);
        b.iter(|| {
            solver.train(&river_game, black_box(river_iters as u32));
        });
    });

    // Turn: measure iterations per second
    let turn_game = make_turn_game("KhQsJs2c", "AA,KK,QQ,JJ,TT", "AA,KK,QQ,JJ,TT,99");
    let turn_iters = 50u64;
    group.throughput(Throughput::Elements(turn_iters));
    group.bench_function("turn_50_iters", |b| {
        let mut solver = PostflopSolver::new(&turn_game);
        b.iter(|| {
            solver.train(&turn_game, black_box(turn_iters as u32));
        });
    });

    group.finish();
}

/// Benchmark exploitability calculation
fn bench_exploitability(c: &mut Criterion) {
    let mut group = c.benchmark_group("exploitability");

    // Pre-train a solver, then measure exploitability calculation time
    let game = make_river_game(
        "Td9d6h5c2s",
        "AA,KK,QQ,JJ,TT,99,88,77,66,AKs,AQs,AJs,KQs",
        "AA,KK,QQ,JJ,TT,99,88,AKs,AKo,AQs,AQo,KQs,KQo",
    );
    let mut solver = PostflopSolver::new(&game);
    solver.train(&game, 500);

    group.bench_function("river_after_500_iters", |b| {
        b.iter(|| solver.exploitability(black_box(&game)));
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_river_narrow,
    bench_river_medium,
    bench_turn_narrow,
    bench_flop_narrow,
    bench_solver_creation,
    bench_iteration_throughput,
    bench_exploitability,
);

criterion_main!(benches);
