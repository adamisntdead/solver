# Poker Solver

A high-performance Counterfactual Regret Minimization (CFR) solver for computing Nash equilibrium strategies in poker and other imperfect information games.

## Features

- **Multi-street postflop solver** - Supports flop, turn, and river starting streets
- **Vectorized CFR implementation** - Walks the betting tree once per iteration with per-hand arrays
- **Flexible tree building** - Configurable bet sizes, raise caps, and street transitions
- **WASM bindings** - Browser-compatible solver with interactive webapp
- **Fast hand evaluation** - 7-card evaluator with precomputed lookup tables

## Architecture

### Core Library (`src/`)

- **`cfr.rs`** - CFR solver with CfrPlus, LinearCfr, and Discounted variants
- **`game.rs`** - Game trait definitions for extensible game implementations
- **`tree/`** - Betting tree builder with configurable actions and streets
- **`poker/`** - Postflop game implementation with vectorized solver

### Poker Solver (`src/poker/`)

- **`postflop_solver.rs`** - Vectorized CFR for postflop spots (flop/turn/river)
- **`postflop_game.rs`** - Game state representation with indexed action trees
- **`matchups.rs`** - Hand evaluation and showdown logic
- **`hands.rs`** - Card/combo encoding with 1326 hole card combinations

### WASM Bindings (`crates/solver-wasm/`)

WebAssembly bindings for browser-based solving with:
- Tree building and validation
- Solver creation and iteration
- Strategy extraction with per-runout card contexts

### Webapp (`webapp/`)

Interactive web interface featuring:
- Tree configuration and visualization
- Range input for both players
- Real-time solving with progress tracking
- Strategy display with turn/river card selectors

## Build

```bash
# Build library
cargo build --release

# Run tests
cargo test

# Build WASM (requires wasm-pack)
cd crates/solver-wasm && wasm-pack build --target web
```

## Examples

```bash
cargo run --example rps                    # Rock-Paper-Scissors
cargo run --example kuhn13                 # Kuhn poker (13 cards)
cargo run --example push_fold --release    # Push-fold poker
```

## License

Copyright (c) 2024-2025 Adam Kelly (adamkelly2201@gmail.com)

All Rights Reserved. See [LICENSE](LICENSE) for details.
