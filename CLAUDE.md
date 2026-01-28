# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Run Commands

```bash
# Build
cargo build --release

# Run examples
cargo run --example rps                    # Rock-Paper-Scissors (fast)
cargo run --example kuhn13                 # Kuhn poker with 13 cards
cargo run --example push_fold --release    # Push-fold poker (use release mode)

# Run tests (examples serve as integration tests)
cargo test
```

## Architecture

This is a **Counterfactual Regret Minimization (CFR) solver** for computing Nash equilibrium strategies in n-player imperfect information games.

### Core Library (src/)

- **`cfr.rs`** - CFR solver implementation with three variants:
  - `CfrPlus` - Regret matching+ with linear averaging (default, fastest)
  - `LinearCfr` - t/(t+1) discounting on regrets and strategies
  - `Discounted` - Configurable discount parameters (alpha, beta, gamma)

- **`game.rs`** - Trait definitions for games:
  - `GameNode` - Single game state (terminal check, actions, payoffs, info set ID)
  - `Game` - Complete game (root node, player count, info set count)

- **`info_abstraction.rs`** - Groups similar private states into buckets to reduce computation

### Key Patterns

**Alternating player updates**: Each iteration updates only one player, cycling through. This matches Gambit's approach and improves convergence.

**Training methods**:
- `train()` - Full tree enumeration (small games)
- `train_sampled()` - Monte Carlo chance sampling (large games like poker)

**CFR variant selection**:
- Use `LinearCfr` for `train_sampled()` (chance sampling)
- Use `CfrPlus` for `train()` (full enumeration)

### Examples

- **`rps.rs`** - Simple 2-player game demonstrating basic CFR
- **`kuhn13.rs`** - Poker game showing information abstraction (bucketing cards by strength)
- **`push_fold.rs`** - Multi-player push-fold poker with precomputed equity tables

### Poker Utilities (examples/poker/)

- **`cards.rs`** - Card encoding, 169 canonical pre-flop hands (pairs, suited, offsuit)
- **`hand_eval.rs`** - Fast 7-card hand evaluator returning comparable u32 ranks
- **`equity.rs`** - Monte Carlo equity calculation between hands

## Canonical Hand Encoding

Pre-flop hands use 169 canonical indices (suit-isomorphic):
- 0-12: Pairs (22=0, ..., AA=12)
- 13-90: Suited hands
- 91-168: Offsuit hands
