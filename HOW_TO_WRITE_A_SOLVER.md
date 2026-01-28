# How to Write a Poker CFR Solver

This document describes the architecture and key algorithms for building a range-based Counterfactual Regret Minimization (CFR) solver for poker, based on analysis of Gambit, binary_wasm_solver, and our own implementation.

## Overview

A poker CFR solver computes Nash equilibrium strategies for two-player zero-sum games. The key insight is processing all private hands simultaneously in a single tree traversal, rather than walking the tree once per hand combination.

## Architecture

### Wrong Approach: Dealing Chance Node

A naive approach is to model hand dealing as a chance node at the root of the game tree, with one child per (OOP hand, IP hand) pair. This creates a game tree where:
- The root has ~1362 children (one per valid matchup)
- Each child contains the full betting tree
- CFR walks through all 1362 × tree_size nodes per iteration

**This does not converge.** The problem is that each hand pair gets its own independent traversal, and the counterfactual values don't properly aggregate across hands.

### Correct Approach: Vectorized Hand Arrays

The correct approach walks the betting tree **once per iteration** with arrays indexed by private hand:

```
solve_recursive(result, tree, node_idx, traverser, cfreach)

  result:    &mut [f32]  -- output CFVs, length = num_hands[traverser]
  traverser: usize       -- which player we're solving for this iteration
  cfreach:   &[f32]      -- opponent's counterfactual reach, length = num_hands[opponent]
```

Two arrays flow through the recursion:
- **`cfreach`** (opponent reach) flows **DOWN** the tree
- **`result`** (counterfactual values) flows **UP** the tree

### Data Layout

Each tree node stores regrets and cumulative strategy in flat arrays:

```
regrets[node_idx]: Vec<f32>  // length = num_actions * num_hands[acting_player]
                             // layout: [action0_hand0, action0_hand1, ..., action1_hand0, ...]
```

Each hand has independent regrets and its own strategy at every decision point.

## Core Algorithm

### CFR Iteration (Alternating Updates)

```
for iteration in 0..num_iterations:
    traverser = iteration % 2  // alternate between players
    opponent = 1 - traverser

    cfreach = opponent's initial range weights
    result = [0.0; num_hands[traverser]]

    solve_recursive(result, tree, root, traverser, cfreach)

    apply_lcfr_discount()
```

### solve_recursive

At each node, behavior depends on the node type:

#### Terminal Nodes

Compute CFV for every traverser hand at once:

```
result[h] = sum_j(cfreach[j] * payoff(h, j) * no_card_conflict(h, j))
```

For **fold** terminals: payoff is +pot/2 (winner) or -pot/2 (loser) for all valid matchups.

For **showdown** terminals: payoff depends on hand comparison:
- Win: +pot/2 × cfreach[j]
- Lose: -pot/2 × cfreach[j]
- Tie: 0

An O(N×M) implementation works for small ranges. For wide ranges, use the sorted sweep with inclusion-exclusion (see "Efficient Terminal Evaluation" below).

#### Traverser's Decision Nodes

When the acting player is the traverser:

1. **Get strategy** via regret matching on this node's regrets
2. **Recurse** into each action (cfreach passes through unchanged)
3. **Compute node CFV**: `result[h] = sum_a(strategy[a][h] * cfv_action[a][h])`
4. **Update regrets**: `regret[a][h] += cfv_action[a][h] - result[h]`
5. **Accumulate strategy**: `cum_strategy[a][h] += strategy[a][h]`

Key: the opponent's reach (`cfreach`) is **not modified** at traverser nodes.

#### Opponent's Decision Nodes

When the acting player is the opponent:

1. **Get opponent's strategy** via regret matching
2. **Update cfreach**: for each action a, `cfreach_a[j] = cfreach[j] * strategy[a][j]`
3. **Recurse** into each action with the updated cfreach
4. **Sum all action CFVs**: `result[h] = sum_a(cfv_action[a][h])`

Key: the result is a simple **sum** (not strategy-weighted) because the opponent's strategy is already folded into the reach probabilities.

### Regret Matching

For each hand independently:

```
for h in 0..num_hands:
    sum = sum_a(max(0, regret[a][h]))
    if sum > 0:
        strategy[a][h] = max(0, regret[a][h]) / sum
    else:
        strategy[a][h] = 1 / num_actions  // uniform fallback
```

### LCFR Discounting

After each iteration, apply Linear CFR discount to all regrets and strategy sums:

```
discount = t / (t + 1)  // where t = iteration count for this player
for all nodes:
    regrets *= discount
    cum_strategy *= discount
```

This gives more weight to recent iterations, improving convergence.

## Best Response / Exploitability

Exploitability measures how far the current strategy is from Nash equilibrium.

### Best Response Value

Same tree walk as CFR, except at **traverser nodes**, take the **element-wise maximum** over actions:

```
// Instead of: result[h] = sum_a(strategy[a][h] * cfv[a][h])
// Best response: result[h] = max_a(cfv[a][h])
```

At opponent nodes, use the **average strategy** (not current strategy) with the same sum logic.

### Computing Exploitability

```
exploitability = 0
for each player p:
    cfreach = initial range weights of opponent
    br_values = best_response_value(tree, root, p, cfreach)
    br_ev = sum_h(weight[p][h] * br_values[h])
    exploitability += br_ev

exploitability /= total_matchup_weight
```

A converged solver should show exploitability approaching 0 (typically < 0.1% of pot after 1000 iterations).

## Efficient Terminal Evaluation

For wide ranges, the O(N×M) terminal evaluation becomes a bottleneck. Two optimizations:

### Fold Nodes: Inclusion-Exclusion (O(N + M))

Instead of checking card conflicts for every pair:

```
// Pass 1: accumulate opponent reach per card
cfreach_sum = sum(cfreach[j])
cfreach_per_card[card] = sum(cfreach[j] where opponent hand j contains card)

// Pass 2: for each traverser hand h with cards (c0, c1)
valid_reach = cfreach_sum - cfreach_per_card[c0] - cfreach_per_card[c1] + overlap_correction
result[h] = payoff * valid_reach
```

### Showdown Nodes: Sorted Sweep (O(N log N + M log M))

Pre-sort both players' hands by strength. Then use a two-pass sweep:

1. **Forward pass** (weak to strong): accumulate reach of all weaker opponent hands → wins
2. **Backward pass** (strong to weak): accumulate reach of all stronger opponent hands → losses

Apply card-removal correction via the inclusion-exclusion trick at each step.

## Player Mapping (Heads-Up Postflop)

In heads-up postflop poker:
- **Player 0** in the tree = SB/BTN = IP (acts second)
- **Player 1** in the tree = BB = OOP (acts first)

The solver's hand arrays must match:
- `hands[0]` = IP player's hands (from IP range)
- `hands[1]` = OOP player's hands (from OOP range)

The action tree builder sets `first_actor = 1` for HU postflop (BB acts first).

## Key Lessons Learned

1. **Never use a dealing chance node.** Both reference implementations (Gambit, binary_wasm_solver) process all hands simultaneously via vector operations, not via a chance node that enumerates matchups.

2. **cfreach flows down, CFV flows up.** This is the fundamental data flow. Getting this wrong (e.g., trying to track both players' reaches explicitly) leads to incorrect results.

3. **At opponent nodes, sum (don't weight) the CFVs.** The opponent's strategy is folded into cfreach, so the traverser's CFV is the simple sum across actions.

4. **Each hand has its own strategy.** Regrets and strategies are per (node, action, hand), not per (node, action). The regret matching produces an independent probability distribution for each hand.

5. **LCFR with alternating updates converges fast.** Linear CFR with `t/(t+1)` discounting applied after each iteration, combined with alternating player updates, converges rapidly (< 0.1% pot exploitability in ~500 iterations for typical spots).

6. **Payoffs are ±pot/2.** In zero-sum formulation, the winner gets +pot/2 and the loser gets -pot/2 relative to the "start" of the game (after antes are posted).

## Performance Numbers

For a river spot with:
- Board: KhQsJs2c3d
- OOP range: 54 combos (38 valid after board removal)
- IP range: 64 combos (46 valid)
- Tree: 21 nodes
- Bet sizes: 50% pot, 100% pot, plus raises

| Iterations | Exploitability | % of Pot |
|------------|---------------|----------|
| 100        | 0.087 chips   | 0.087%   |
| 500        | 0.012 chips   | 0.012%   |
| 1,000      | 0.003 chips   | 0.003%   |
| 5,000      | 0.000 chips   | 0.000%   |

## References

- **Gambit** (`/Users/adam/Desktop/gambit`): C++ implementation with PyTorch integration for multi-street solving. Uses belief-based reach probabilities with shared pointer optimization.

- **binary_wasm_solver** (`/Users/adam/Desktop/binary_wasm_solver`): Rust implementation targeting WebAssembly. Clean vectorized design with O(N+M) terminal evaluation.

- Zinkevich et al., "Regret Minimization in Games with Incomplete Information" (2007) - Original CFR paper.

- Brown & Sandholm, "Solving Imperfect-Information Games via Discounted Regret Minimization" (2019) - DCFR/LCFR.

- Johanson et al., "Accelerating Best Response Calculation in Large Extensive Games" (2011) - Efficient terminal evaluation with card removal.
