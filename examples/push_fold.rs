//! N-Player Push-Fold Poker CFR Solver
//!
//! Computes Nash equilibrium push/fold ranges for multi-player push-fold poker.
//!
//! Run with: `cargo run --example push_fold --release`

use solver::{CfrSolver, Game, GameNode};
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;
use std::sync::Arc;

#[path = "poker/mod.rs"]
mod poker;

use poker::{
    compute_equity, from_canonical, make_card, rank, CanonicalHand, HoleCards, NUM_CANONICAL_HANDS,
};

// ============================================================================
// Push-Fold Game Configuration
// ============================================================================

/// Maximum number of players supported.
const MAX_PLAYERS: usize = 6;

/// Configuration for push-fold poker.
#[derive(Debug, Clone)]
struct PushFoldConfig {
    /// Number of players.
    pub num_players: usize,
    /// Effective stack size in big blinds.
    pub stack_bb: f64,
    /// Number of Monte Carlo samples for equity calculation.
    pub equity_samples: usize,
}

impl Default for PushFoldConfig {
    fn default() -> Self {
        Self {
            num_players: 4,
            stack_bb: 10.0,
            equity_samples: 5000,
        }
    }
}

// ============================================================================
// Precomputed Equity Table (Heads-Up Only)
// ============================================================================

/// Precomputed heads-up equity matrix for all canonical hand pairs.
#[derive(Clone)]
struct EquityTable {
    data: Vec<f32>,
}

const EQUITY_TABLE_MAGIC: &[u8; 8] = b"EQTBL001";

impl EquityTable {
    fn load_or_compute(samples_per_matchup: usize, cache_path: &Path) -> Self {
        if let Some(table) = Self::load(cache_path, samples_per_matchup) {
            return table;
        }
        let table = Self::compute(samples_per_matchup);
        if let Err(e) = table.save(cache_path, samples_per_matchup) {
            eprintln!("Warning: Failed to save equity table cache: {}", e);
        }
        table
    }

    fn load(path: &Path, expected_samples: usize) -> Option<Self> {
        let file = File::open(path).ok()?;
        let mut reader = BufReader::new(file);

        let mut magic = [0u8; 8];
        reader.read_exact(&mut magic).ok()?;
        if &magic != EQUITY_TABLE_MAGIC {
            println!("Equity table cache has invalid format, recomputing...");
            return None;
        }

        let mut samples_bytes = [0u8; 8];
        reader.read_exact(&mut samples_bytes).ok()?;
        let samples = u64::from_le_bytes(samples_bytes) as usize;

        if samples != expected_samples {
            println!(
                "Equity table cache has {} samples, need {}, recomputing...",
                samples, expected_samples
            );
            return None;
        }

        let data_size = NUM_CANONICAL_HANDS * NUM_CANONICAL_HANDS;
        let mut data = vec![0.0f32; data_size];
        let data_bytes = unsafe {
            std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut u8, data_size * 4)
        };
        reader.read_exact(data_bytes).ok()?;

        println!("Loaded equity table from cache ({} samples)", samples);
        Some(Self { data })
    }

    fn save(&self, path: &Path, samples: usize) -> std::io::Result<()> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);
        writer.write_all(EQUITY_TABLE_MAGIC)?;
        writer.write_all(&(samples as u64).to_le_bytes())?;
        let data_bytes = unsafe {
            std::slice::from_raw_parts(self.data.as_ptr() as *const u8, self.data.len() * 4)
        };
        writer.write_all(data_bytes)?;
        println!("Saved equity table to cache");
        Ok(())
    }

    fn compute(samples_per_matchup: usize) -> Self {
        println!(
            "Computing equity table ({} samples per matchup)...",
            samples_per_matchup
        );
        let start = std::time::Instant::now();
        let mut data = vec![0.0f32; NUM_CANONICAL_HANDS * NUM_CANONICAL_HANDS];

        for hand_a in 0..NUM_CANONICAL_HANDS {
            if hand_a % 20 == 0 {
                print!("\r  Progress: {}/169 hands", hand_a);
                std::io::stdout().flush().ok();
            }
            let cards_a = from_canonical(hand_a as CanonicalHand);
            for hand_b in 0..NUM_CANONICAL_HANDS {
                let cards_b = get_non_conflicting_cards(hand_b as CanonicalHand, &[cards_a]);
                if let Some(cards_b) = cards_b {
                    let hands = vec![cards_a, cards_b];
                    let equity = compute_equity(&hands, 0, samples_per_matchup);
                    data[hand_a * NUM_CANONICAL_HANDS + hand_b] = equity[0] as f32;
                } else {
                    data[hand_a * NUM_CANONICAL_HANDS + hand_b] = 0.5;
                }
            }
        }
        println!("\r  Equity table computed in {:?}          ", start.elapsed());
        Self { data }
    }

    #[inline]
    fn get_headsup(&self, hand_a: CanonicalHand, hand_b: CanonicalHand) -> f64 {
        self.data[hand_a as usize * NUM_CANONICAL_HANDS + hand_b as usize] as f64
    }
}

/// Get hole cards for a canonical hand that don't conflict with existing hands.
fn get_non_conflicting_cards(canonical: CanonicalHand, existing: &[HoleCards]) -> Option<HoleCards> {
    let HoleCards(c1, c2) = from_canonical(canonical);
    let r1 = rank(c1);
    let r2 = rank(c2);

    let mut dead = 0u64;
    for h in existing {
        dead |= (1u64 << h.0) | (1u64 << h.1);
    }

    if r1 == r2 {
        let mut found = Vec::new();
        for s in 0..4 {
            let card = make_card(r1, s);
            if (dead & (1 << card)) == 0 {
                found.push(card);
            }
        }
        if found.len() >= 2 {
            Some(HoleCards(found[0], found[1]))
        } else {
            None
        }
    } else if canonical < 91 {
        for s in 0..4 {
            let c1 = make_card(r1, s);
            let c2 = make_card(r2, s);
            if (dead & (1 << c1)) == 0 && (dead & (1 << c2)) == 0 {
                return Some(HoleCards(c1, c2));
            }
        }
        None
    } else {
        for s1 in 0..4 {
            for s2 in 0..4 {
                if s1 == s2 {
                    continue;
                }
                let c1 = make_card(r1, s1);
                let c2 = make_card(r2, s2);
                if (dead & (1 << c1)) == 0 && (dead & (1 << c2)) == 0 {
                    return Some(HoleCards(c1, c2));
                }
            }
        }
        None
    }
}

// ============================================================================
// Decision Point Enumeration
// ============================================================================

/// Compute the decision point ID for a given action history.
/// Returns (decision_point_id, player_to_act) or None if terminal.
fn decision_point(actions: &[u8], num_players: usize) -> Option<(usize, usize)> {
    let acting_player = actions.len();
    if acting_player >= num_players {
        return None; // All players have acted
    }

    // Check if the hand is already over (everyone folded)
    let num_pushers = actions.iter().filter(|&&a| a == 1).count();

    // If everyone before BB has folded, BB wins automatically
    if acting_player == num_players - 1 && num_pushers == 0 {
        return None; // BB wins by default
    }

    // Encode the action history as a decision point ID
    // The history forms a binary number (action[i] at bit i)
    let history_id: usize = actions
        .iter()
        .enumerate()
        .map(|(i, &a)| (a as usize) << i)
        .sum();

    // For BB (last player), we skip the all-fold case (history_id=0)
    // So we subtract 1 from history_id for BB's decision points
    let adjusted_history = if acting_player == num_players - 1 {
        history_id - 1 // FFF case (id=0) is terminal, so FFP becomes id=0, etc.
    } else {
        history_id
    };

    // Base offset: sum of decision points for all prior players
    // P0: 1 point, P1: 2 points, P2: 4 points, ..., P(n-2): 2^(n-2) points
    // P(n-1): 2^(n-1) - 1 points (excluding all-fold)
    let mut base = 0usize;
    for p in 0..acting_player {
        base += 1 << p;
    }

    Some((base + adjusted_history, acting_player))
}

/// Count total decision points for a given number of players.
fn count_decision_points(num_players: usize) -> usize {
    let mut total = 0;
    for p in 0..num_players {
        if p == num_players - 1 {
            // BB doesn't act if everyone folded
            total += (1 << p) - 1; // Exclude all-fold case
        } else {
            total += 1 << p;
        }
    }
    total
}

// ============================================================================
// Push-Fold Game State
// ============================================================================

#[derive(Clone)]
struct PushFoldData {
    config: PushFoldConfig,
    equity_table: EquityTable,
    num_decision_points: usize,
}

/// Push-fold game node.
#[derive(Clone)]
enum PushFoldNode {
    /// Chance node: deal cards to all players.
    Deal { data: Arc<PushFoldData> },

    /// A player is acting.
    Acting {
        hands: Vec<CanonicalHand>,
        actions: Vec<u8>,
        data: Arc<PushFoldData>,
    },

    /// Terminal: showdown or fold-out.
    Terminal {
        hands: Vec<CanonicalHand>,
        actions: Vec<u8>,
        data: Arc<PushFoldData>,
    },
}

impl GameNode for PushFoldNode {
    fn is_terminal(&self) -> bool {
        matches!(self, PushFoldNode::Terminal { .. })
    }

    fn is_chance(&self) -> bool {
        matches!(self, PushFoldNode::Deal { .. })
    }

    fn current_player(&self) -> usize {
        match self {
            PushFoldNode::Acting { actions, .. } => actions.len(),
            _ => panic!("No current player at this node"),
        }
    }

    fn num_actions(&self) -> usize {
        match self {
            PushFoldNode::Deal { data } => {
                // 169^n canonical hand combinations
                NUM_CANONICAL_HANDS.pow(data.config.num_players as u32)
            }
            PushFoldNode::Acting { .. } => 2, // Fold or Push
            _ => 0,
        }
    }

    fn play(&self, action: usize) -> Self {
        match self {
            PushFoldNode::Deal { data } => {
                // Decode action to hand assignments
                let n = data.config.num_players;
                let mut hands = Vec::with_capacity(n);
                let mut remaining = action;
                for _ in 0..n {
                    hands.push((remaining % NUM_CANONICAL_HANDS) as CanonicalHand);
                    remaining /= NUM_CANONICAL_HANDS;
                }

                PushFoldNode::Acting {
                    hands,
                    actions: Vec::new(),
                    data: data.clone(),
                }
            }

            PushFoldNode::Acting {
                hands,
                actions,
                data,
            } => {
                let mut new_actions = actions.clone();
                new_actions.push(action as u8);

                let n = data.config.num_players;

                // Check if this leads to a terminal state
                if new_actions.len() == n {
                    // All players have acted
                    return PushFoldNode::Terminal {
                        hands: hands.clone(),
                        actions: new_actions,
                        data: data.clone(),
                    };
                }

                // Check if BB wins by everyone folding before them
                if new_actions.len() == n - 1 {
                    let num_pushers = new_actions.iter().filter(|&&a| a == 1).count();
                    if num_pushers == 0 {
                        // Everyone folded to BB
                        return PushFoldNode::Terminal {
                            hands: hands.clone(),
                            actions: new_actions,
                            data: data.clone(),
                        };
                    }
                }

                PushFoldNode::Acting {
                    hands: hands.clone(),
                    actions: new_actions,
                    data: data.clone(),
                }
            }

            _ => panic!("Cannot play at terminal node"),
        }
    }

    fn payoff(&self, player: usize) -> f64 {
        let (hands, actions, data) = match self {
            PushFoldNode::Terminal {
                hands,
                actions,
                data,
            } => (hands, actions, data),
            _ => panic!("Payoff only at terminal nodes"),
        };

        let n = data.config.num_players;
        let stack = data.config.stack_bb;

        // Identify pushers (players who went all-in)
        let pushers: Vec<usize> = actions
            .iter()
            .enumerate()
            .filter(|&(_, a)| *a == 1)
            .map(|(i, _)| i)
            .collect();

        // Calculate pot contributions
        // In push-fold: SB posts 0.5, BB posts 1.0, pushers commit full stack
        let sb_player = n - 2;
        let bb_player = n - 1;

        if pushers.is_empty() {
            // Everyone folded to BB - BB wins blinds
            if player == bb_player {
                return 0.5; // BB wins SB's 0.5bb (already has their own BB posted)
            } else if player == sb_player {
                return -0.5; // SB loses their blind
            } else {
                return 0.0; // Others lose nothing (no blind posted)
            }
        }

        // Calculate each player's contribution to the pot
        let mut contributions = vec![0.0; n];
        contributions[sb_player] = 0.5; // SB blind
        contributions[bb_player] = 1.0; // BB blind

        for &p in &pushers {
            contributions[p] = stack;
        }

        // Total pot
        let pot: f64 = contributions.iter().sum();

        // If only one pusher, they win everything
        if pushers.len() == 1 {
            let winner = pushers[0];
            return if player == winner {
                pot - contributions[player]
            } else {
                -contributions[player]
            };
        }

        // Multi-way showdown - compute equity
        let equity = compute_multiway_equity(hands, &pushers, &data.equity_table);

        // Player's EV = equity * pot - contribution
        equity[player] * pot - contributions[player]
    }

    fn info_set_id(&self) -> usize {
        match self {
            PushFoldNode::Acting {
                hands,
                actions,
                data,
            } => {
                let (dp_id, player) = decision_point(actions, data.config.num_players)
                    .expect("Acting node should have valid decision point");
                dp_id * NUM_CANONICAL_HANDS + hands[player] as usize
            }
            _ => panic!("No info set at this node"),
        }
    }
}

/// Approximate 3-way equity from heads-up table.
/// Uses product of pairwise win probabilities, normalized.
#[inline]
fn approximate_3way_equity(
    table: &EquityTable,
    a: CanonicalHand,
    b: CanonicalHand,
    c: CanonicalHand,
) -> [f64; 3] {
    // Get pairwise equities
    let ab = table.get_headsup(a, b); // P(A beats B)
    let ac = table.get_headsup(a, c); // P(A beats C)
    let bc = table.get_headsup(b, c); // P(B beats C)

    // Approximate: P(A wins 3-way) ≈ P(A beats B) × P(A beats C)
    // Then normalize
    let raw_a = ab * ac;
    let raw_b = (1.0 - ab) * bc;
    let raw_c = (1.0 - ac) * (1.0 - bc);

    let sum = raw_a + raw_b + raw_c;
    [raw_a / sum, raw_b / sum, raw_c / sum]
}

/// Approximate 4-way equity from heads-up table.
/// Uses product of pairwise win probabilities, normalized.
#[inline]
fn approximate_4way_equity(
    table: &EquityTable,
    a: CanonicalHand,
    b: CanonicalHand,
    c: CanonicalHand,
    d: CanonicalHand,
) -> [f64; 4] {
    // Get pairwise equities
    let ab = table.get_headsup(a, b);
    let ac = table.get_headsup(a, c);
    let ad = table.get_headsup(a, d);
    let bc = table.get_headsup(b, c);
    let bd = table.get_headsup(b, d);
    let cd = table.get_headsup(c, d);

    // P(A wins 4-way) ≈ P(A beats B) × P(A beats C) × P(A beats D)
    let raw_a = ab * ac * ad;
    let raw_b = (1.0 - ab) * bc * bd;
    let raw_c = (1.0 - ac) * (1.0 - bc) * cd;
    let raw_d = (1.0 - ad) * (1.0 - bd) * (1.0 - cd);

    let sum = raw_a + raw_b + raw_c + raw_d;
    [raw_a / sum, raw_b / sum, raw_c / sum, raw_d / sum]
}

/// Compute equity for a multi-way showdown using pairwise approximation.
fn compute_multiway_equity(
    hands: &[CanonicalHand],
    pushers: &[usize],
    equity_table: &EquityTable,
) -> Vec<f64> {
    let n = hands.len();
    let mut equity = vec![0.0; n];

    match pushers.len() {
        2 => {
            // Heads-up showdown - use precomputed table
            let p0 = pushers[0];
            let p1 = pushers[1];
            let eq0 = equity_table.get_headsup(hands[p0], hands[p1]);
            equity[p0] = eq0;
            equity[p1] = 1.0 - eq0;
        }
        3 => {
            // 3-way pot - approximate from pairwise
            let [p0, p1, p2] = [pushers[0], pushers[1], pushers[2]];
            let eq = approximate_3way_equity(equity_table, hands[p0], hands[p1], hands[p2]);
            equity[p0] = eq[0];
            equity[p1] = eq[1];
            equity[p2] = eq[2];
        }
        4 => {
            // 4-way pot - approximate from pairwise
            let [p0, p1, p2, p3] = [pushers[0], pushers[1], pushers[2], pushers[3]];
            let eq = approximate_4way_equity(
                equity_table,
                hands[p0],
                hands[p1],
                hands[p2],
                hands[p3],
            );
            equity[p0] = eq[0];
            equity[p1] = eq[1];
            equity[p2] = eq[2];
            equity[p3] = eq[3];
        }
        _ => panic!("Unsupported number of players in showdown: {}", pushers.len()),
    }

    equity
}

// ============================================================================
// Push-Fold Game
// ============================================================================

struct PushFoldGame {
    data: Arc<PushFoldData>,
}

impl PushFoldGame {
    fn new(config: PushFoldConfig) -> Self {
        let cache_path = Path::new("equity_table.bin");
        let equity_table = EquityTable::load_or_compute(config.equity_samples, cache_path);
        let num_decision_points = count_decision_points(config.num_players);

        println!(
            "Game: {} players, {} decision points, {} info sets",
            config.num_players,
            num_decision_points,
            num_decision_points * NUM_CANONICAL_HANDS
        );

        Self {
            data: Arc::new(PushFoldData {
                config,
                equity_table,
                num_decision_points,
            }),
        }
    }
}

impl Game for PushFoldGame {
    type Node = PushFoldNode;

    fn root(&self) -> Self::Node {
        PushFoldNode::Deal {
            data: self.data.clone(),
        }
    }

    fn num_players(&self) -> usize {
        self.data.config.num_players
    }

    fn num_info_sets(&self) -> usize {
        self.data.num_decision_points * NUM_CANONICAL_HANDS
    }
}

// ============================================================================
// Range Display
// ============================================================================

/// Print a push/call range as a 13x13 grid.
fn print_range(title: &str, frequencies: &[f64; NUM_CANONICAL_HANDS]) {
    const RANKS: &[char] = &['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2'];

    println!("{}", title);
    println!("     A     K     Q     J     T     9     8     7     6     5     4     3     2");

    for row in 0..13 {
        print!("{}  ", RANKS[row]);
        for col in 0..13 {
            let high_rank = (12 - row) as u8;
            let low_rank = (12 - col) as u8;

            let freq = if high_rank == low_rank {
                frequencies[high_rank as usize]
            } else if row < col {
                let (hi, lo) = if high_rank > low_rank {
                    (high_rank, low_rank)
                } else {
                    (low_rank, high_rank)
                };
                let tri_idx = (hi * (hi - 1)) / 2 + lo;
                frequencies[13 + tri_idx as usize]
            } else {
                let (hi, lo) = if high_rank > low_rank {
                    (high_rank, low_rank)
                } else {
                    (low_rank, high_rank)
                };
                let tri_idx = (hi * (hi - 1)) / 2 + lo;
                frequencies[91 + tri_idx as usize]
            };

            if freq >= 0.66 {
                print!("\x1b[42m{:5.1}\x1b[0m ", freq * 100.0);
            } else if freq >= 0.33 {
                print!("\x1b[43m{:5.1}\x1b[0m ", freq * 100.0);
            } else if freq >= 0.01 {
                print!("\x1b[41m{:5.1}\x1b[0m ", freq * 100.0);
            } else {
                print!("{:5.1} ", freq * 100.0);
            }
        }
        println!();
    }
    println!();
}

/// Extract push frequencies from solver for a specific decision point.
fn extract_push_frequencies(solver: &CfrSolver, base_info_set: usize) -> [f64; NUM_CANONICAL_HANDS] {
    let mut freqs = [0.0; NUM_CANONICAL_HANDS];
    for hand in 0..NUM_CANONICAL_HANDS {
        if let Some(strategy) = solver.get_strategy(base_info_set + hand) {
            freqs[hand] = strategy[1]; // action 1 = push
        }
    }
    freqs
}

/// Get position name for a player.
fn position_name(player: usize, num_players: usize) -> &'static str {
    match (num_players, player) {
        (2, 0) => "SB",
        (2, 1) => "BB",
        (3, 0) => "BTN",
        (3, 1) => "SB",
        (3, 2) => "BB",
        (4, 0) => "UTG",
        (4, 1) => "BTN",
        (4, 2) => "SB",
        (4, 3) => "BB",
        (5, 0) => "UTG",
        (5, 1) => "HJ",
        (5, 2) => "BTN",
        (5, 3) => "SB",
        (5, 4) => "BB",
        (6, 0) => "UTG",
        (6, 1) => "HJ",
        (6, 2) => "CO",
        (6, 3) => "BTN",
        (6, 4) => "SB",
        (6, 5) => "BB",
        _ => "??",
    }
}

// ============================================================================
// Main
// ============================================================================

fn main() {
    println!("Push-Fold Poker CFR Solver");
    println!("==========================");
    println!();

    // Configuration
    let num_players = 4;
    let stack_bb = 10.0;
    let iterations = 100_000; // ~25k updates per player with alternating
    let equity_samples = 5000;

    println!("Configuration:");
    println!("  Players: {}", num_players);
    println!("  Stack size: {} BB", stack_bb);
    println!("  MCCFR iterations: {}", iterations);
    println!();

    let config = PushFoldConfig {
        num_players,
        stack_bb,
        equity_samples,
    };

    let game = PushFoldGame::new(config);
    // Linear CFR with alternating updates (like Gambit)
    let mut solver = CfrSolver::new(&game, solver::CfrVariant::LinearCfr);

    println!("Training MCCFR (Linear CFR, alternating updates)...");
    let start = std::time::Instant::now();
    solver.train_sampled(&game, iterations);
    let train_time = start.elapsed();
    println!("Training completed in {:?}", train_time);
    println!();

    // Print push ranges for each position's first decision point
    // (i.e., when they're first to act or everyone folded before them)
    println!("=== Opening Push Ranges (first to act) ===\n");

    let mut dp_base = 0;
    for player in 0..num_players {
        let pos = position_name(player, num_players);
        let freqs = extract_push_frequencies(&solver, dp_base * NUM_CANONICAL_HANDS);
        let push_pct: f64 = freqs.iter().sum::<f64>() / 169.0 * 100.0;

        println!("{} Push Range ({:.1}% of hands):", pos, push_pct);
        print_range("", &freqs);

        // Move to next player's decision points
        if player < num_players - 1 {
            dp_base += 1 << player;
        }
    }

    // Print some specific ranges for BB facing different actions
    if num_players == 4 {
        println!("=== BB Call Ranges (facing push) ===\n");

        // BB decision points start after UTG(1) + BTN(2) + SB(4) = 7
        // But we need to subtract 1 for the FFF case where BB doesn't act
        // BB faces: FFP(dp=7), FPF(8), FPP(9), PFF(10), PFP(11), PPF(12), PPP(13)
        // But our encoding: base for BB = 1 + 2 + 4 = 7, then add history
        // FFP = 0b100 = 4, so dp = 7 + 4 - 1 = 10? Let me recalculate...

        // Actually, for BB (player 3), base = 1 + 2 + 4 = 7
        // History encoding: action[0]*1 + action[1]*2 + action[2]*4
        // FFP: [0,0,1] = 0 + 0 + 4 = 4, dp = 7 + 4 = 11
        // But we skip FFF (all fold), so actual indices are shifted

        // Let's just show a few key scenarios
        let scenarios = [
            ("UTG fold, BTN fold, SB push", vec![0, 0, 1]),
            ("UTG fold, BTN push, SB fold", vec![0, 1, 0]),
            ("UTG push, BTN fold, SB fold", vec![1, 0, 0]),
            ("UTG push, BTN push, SB push", vec![1, 1, 1]),
        ];

        for (desc, actions) in &scenarios {
            if let Some((dp_id, _)) = decision_point(actions, num_players) {
                let freqs = extract_push_frequencies(&solver, dp_id * NUM_CANONICAL_HANDS);
                let call_pct: f64 = freqs.iter().sum::<f64>() / 169.0 * 100.0;
                println!("BB vs {} ({:.1}%):", desc, call_pct);
                print_range("", &freqs);
            }
        }
    }
}
