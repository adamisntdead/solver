# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Run Commands

```bash
# Build
cargo build --release

# Run tests (use release mode for faster convergence tests)
cargo test --release

# Run examples
cargo run --example rps                    # Rock-Paper-Scissors (fast)
cargo run --example kuhn13                 # Kuhn poker with 13 cards
cargo run --example push_fold --release    # Push-fold poker (use release mode)

# Build WASM bindings
cd crates/solver-wasm && wasm-pack build --target web

# Check WASM compiles
cargo check -p solver-wasm
```

## Architecture

This is a **Counterfactual Regret Minimization (CFR) solver** for computing Nash equilibrium strategies in poker and other imperfect information games.

### Core Library (src/)

- **`cfr.rs`** - CFR solver implementation with three variants:
  - `CfrPlus` - Regret matching+ with linear averaging (default, fastest)
  - `LinearCfr` - t/(t+1) discounting on regrets and strategies
  - `Discounted` - Configurable discount parameters (alpha, beta, gamma)

- **`game.rs`** - Trait definitions for games:
  - `GameNode` - Single game state (terminal check, actions, payoffs, info set ID)
  - `Game` - Complete game (root node, player count, info set count)

- **`info_abstraction.rs`** - Groups similar private states into buckets to reduce computation

### Tree Building (src/tree/)

- **`config.rs`** - TreeConfig with street configs, bet sizes, stack depths
- **`builder.rs`** - Builds action trees from config with action merging
- **`action.rs`** - BettingState tracking pot, stacks, and legal actions

### Postflop Solver (src/poker/)

- **`postflop_solver.rs`** - Vectorized CFR solver:
  - Walks tree once per iteration with per-hand value arrays
  - Multi-street support: river (1 context), turn→river (48 contexts), flop→turn→river (2401 contexts)
  - Card contexts stored as `[turn_idx * num_river + river_idx]` for composite indexing

- **`postflop_game.rs`** - PostflopGame wrapping IndexedActionTree with ranges and board
- **`matchups.rs`** - 7-card hand evaluation via `evaluate_7cards()`
- **`hands.rs`** - Combo encoding (1326 combos), Board with card mask
- **`isomorphism.rs`** - Suit isomorphism for reduced strategy storage

### WASM Bindings (crates/solver-wasm/)

- **`lib.rs`** - WebAssembly exports:
  - `create_solver()` - Initialize solver from JSON config
  - `run_iterations()` - Run CFR iterations
  - `get_node_strategy_for_context()` - Get strategy with card context filtering
  - `get_turn_cards()` / `get_river_cards()` - Card info for selectors
  - `get_chance_depth()` - Node depth relative to chance nodes (0/1/2)

### Webapp (webapp/)

- **`js/main.js`** - Main app logic with solver integration
- **`js/tree-view.js`** - Tree visualization component
- **`index.html`** - UI with config panel, tree view, strategy display
- **`style.css`** - Dark theme styling

## Key Patterns

**Alternating player updates**: Each iteration updates only one player, cycling through. This matches Gambit's approach and improves convergence.

**Training methods**:
- `train()` - Full tree enumeration (small games)
- `train_sampled()` - Monte Carlo chance sampling (large games)

**CFR variant selection**:
- Use `LinearCfr` for `train_sampled()` (chance sampling)
- Use `CfrPlus` for `train()` (full enumeration)

**Multi-street context system**:
- River-only boards: 1 context per node
- Turn boards: `num_river_cards` contexts (48) for river betting nodes
- Flop boards: `num_turn_cards` (49) for turn nodes, `num_turn * num_river` (2401) for river nodes

## Canonical Hand Encoding

Pre-flop hands use 169 canonical indices (suit-isomorphic):
- 0-12: Pairs (22=0, ..., AA=12)
- 13-90: Suited hands
- 91-168: Offsuit hands

Postflop uses all 1326 combos indexed by `Combo::from_index(0..1326)`.

## License

Copyright (c) 2024-2025 Adam Kelly. All Rights Reserved.
