//! Expected Hand Strength (EHS) and related feature computation.
//!
//! This module computes equity-based features for hand abstraction:
//!
//! - **EHS (Expected Hand Strength)**: Average equity against uniform opponent range
//! - **EHS²**: Expected squared equity (captures variance)
//! - **EMD Features**: 50-bin equity histogram over runouts
//! - **WinSplit Features**: Win and split frequencies (river only)
//!
//! # EHS Computation
//!
//! EHS(hand | board) = E_opponent[ equity(hand vs opponent | board) ]
//!
//! For non-river boards, we also average over remaining runout cards:
//! - Turn: average over 46 possible river cards
//! - Flop: average over 47×46 turn+river combinations
//!
//! # EMD Features
//!
//! For flop and turn, we compute a 50-bin histogram of equity values
//! over all possible runouts. This captures the "shape" of the equity
//! distribution, which is important for strategic similarity.

use crate::poker::hands::{Board, Card, Combo, DECK_SIZE, NUM_COMBOS};
use crate::poker::matchups::evaluate_7cards;

/// Number of bins for EMD histogram features.
pub const EMD_NUM_BINS: usize = 50;

/// Compute EHS for all 1326 combos on a board.
///
/// Returns an array where `ehs[combo_idx]` is the expected hand strength
/// for that combo, or 0.0 if the combo is blocked by the board.
///
/// # Arguments
/// * `board_cards` - The community cards (3-5 cards)
///
/// # Algorithm
/// For each hand, compute equity against all possible opponent hands,
/// averaging over runouts if not on the river.
pub fn compute_all_ehs(board_cards: &[Card]) -> [f32; NUM_COMBOS] {
    let mut ehs = [0.0f32; NUM_COMBOS];
    let board = Board::new(board_cards);

    match board_cards.len() {
        5 => compute_river_ehs(&board, &mut ehs),
        4 => compute_turn_ehs(&board, &mut ehs),
        3 => compute_flop_ehs(&board, &mut ehs),
        _ => panic!("EHS requires 3-5 board cards, got {}", board_cards.len()),
    }

    ehs
}

/// Compute EHS² (expected squared equity) for all combos.
///
/// EHS² = E[equity²] captures the variance of the equity distribution.
/// Hands with high EHS² have more "polarized" equity distributions.
pub fn compute_all_ehs_squared(board_cards: &[Card]) -> [f32; NUM_COMBOS] {
    let mut ehs_sq = [0.0f32; NUM_COMBOS];
    let board = Board::new(board_cards);

    match board_cards.len() {
        5 => compute_river_ehs_squared(&board, &mut ehs_sq),
        4 => compute_turn_ehs_squared(&board, &mut ehs_sq),
        3 => compute_flop_ehs_squared(&board, &mut ehs_sq),
        _ => panic!(
            "EHS² requires 3-5 board cards, got {}",
            board_cards.len()
        ),
    }

    ehs_sq
}

/// Compute EMD features (50-bin histogram) for a single hand.
///
/// Returns a 50-bin histogram where each bin represents the probability
/// that the hand has equity in that range over all runouts.
///
/// # Arguments
/// * `combo` - The hole cards
/// * `board_cards` - The community cards (3-4 cards, not river)
///
/// # Returns
/// 50-element array representing the equity distribution histogram.
/// Returns all zeros if the combo is blocked.
pub fn compute_emd_features(combo: Combo, board_cards: &[Card]) -> [f32; EMD_NUM_BINS] {
    let mut histogram = [0.0f32; EMD_NUM_BINS];
    let board = Board::new(board_cards);

    if combo.conflicts_with_mask(board.mask) {
        return histogram;
    }

    match board_cards.len() {
        4 => compute_turn_emd_features(combo, &board, &mut histogram),
        3 => compute_flop_emd_features(combo, &board, &mut histogram),
        5 => {
            // River: single equity value, put all mass in one bin
            let equity = compute_river_equity_single(combo, &board);
            let bin = equity_to_bin(equity);
            histogram[bin] = 1.0;
        }
        _ => panic!("EMD features require 3-5 board cards"),
    }

    histogram
}

/// Compute asymmetric EMD features with finer bins at decision-critical equities.
///
/// Bin distribution:
/// - [0.00, 0.20): 5 bins (0.04 each)
/// - [0.20, 0.50): 10 bins (0.03 each)
/// - [0.50, 0.80): 15 bins (0.02 each)
/// - [0.80, 1.00]: 20 bins (0.01 each)
pub fn compute_asymmetric_emd_features(combo: Combo, board_cards: &[Card]) -> [f32; EMD_NUM_BINS] {
    let mut histogram = [0.0f32; EMD_NUM_BINS];
    let board = Board::new(board_cards);

    if combo.conflicts_with_mask(board.mask) {
        return histogram;
    }

    match board_cards.len() {
        4 => compute_turn_asymemd_features(combo, &board, &mut histogram),
        3 => compute_flop_asymemd_features(combo, &board, &mut histogram),
        5 => {
            let equity = compute_river_equity_single(combo, &board);
            let bin = equity_to_asymmetric_bin(equity);
            histogram[bin] = 1.0;
        }
        _ => panic!("Asymmetric EMD features require 3-5 board cards"),
    }

    histogram
}

/// Compute WinSplit features for river only.
///
/// Returns [win_frequency, split_frequency] against uniform opponent range.
pub fn compute_winsplit_features(combo: Combo, board_cards: &[Card]) -> [f32; 2] {
    assert!(
        board_cards.len() == 5,
        "WinSplit features require 5 board cards"
    );
    let board = Board::new(board_cards);

    if combo.conflicts_with_mask(board.mask) {
        return [0.0, 0.0];
    }

    let mut wins = 0u32;
    let mut splits = 0u32;
    let mut total = 0u32;

    // Our hand strength
    let our_cards = [
        board_cards[0],
        board_cards[1],
        board_cards[2],
        board_cards[3],
        board_cards[4],
        combo.c0,
        combo.c1,
    ];
    let our_rank = evaluate_7cards(&our_cards);

    // Iterate over all opponent hands
    let combined_mask = board.mask | combo.to_mask();
    for opp_idx in 0..NUM_COMBOS {
        let opp = Combo::from_index(opp_idx);
        if opp.conflicts_with_mask(combined_mask) {
            continue;
        }

        let opp_cards = [
            board_cards[0],
            board_cards[1],
            board_cards[2],
            board_cards[3],
            board_cards[4],
            opp.c0,
            opp.c1,
        ];
        let opp_rank = evaluate_7cards(&opp_cards);

        if our_rank > opp_rank {
            wins += 1;
        } else if our_rank == opp_rank {
            splits += 1;
        }
        total += 1;
    }

    if total == 0 {
        return [0.0, 0.0];
    }

    [wins as f32 / total as f32, splits as f32 / total as f32]
}

// ============================================================================
// Internal computation functions
// ============================================================================

/// Compute river EHS for all combos.
fn compute_river_ehs(board: &Board, ehs: &mut [f32; NUM_COMBOS]) {
    for combo_idx in 0..NUM_COMBOS {
        let combo = Combo::from_index(combo_idx);
        if combo.conflicts_with_mask(board.mask) {
            continue;
        }
        ehs[combo_idx] = compute_river_equity_single(combo, board);
    }
}

/// Compute river equity for a single combo.
fn compute_river_equity_single(combo: Combo, board: &Board) -> f32 {
    let our_cards = [
        board.cards[0],
        board.cards[1],
        board.cards[2],
        board.cards[3],
        board.cards[4],
        combo.c0,
        combo.c1,
    ];
    let our_rank = evaluate_7cards(&our_cards);

    let mut equity_sum = 0.0f64;
    let mut total = 0u32;

    let combined_mask = board.mask | combo.to_mask();

    for opp_idx in 0..NUM_COMBOS {
        let opp = Combo::from_index(opp_idx);
        if opp.conflicts_with_mask(combined_mask) {
            continue;
        }

        let opp_cards = [
            board.cards[0],
            board.cards[1],
            board.cards[2],
            board.cards[3],
            board.cards[4],
            opp.c0,
            opp.c1,
        ];
        let opp_rank = evaluate_7cards(&opp_cards);

        if our_rank > opp_rank {
            equity_sum += 1.0;
        } else if our_rank == opp_rank {
            equity_sum += 0.5;
        }
        total += 1;
    }

    if total == 0 {
        return 0.0;
    }
    (equity_sum / total as f64) as f32
}

/// Compute turn EHS for all combos (average over river cards).
fn compute_turn_ehs(board: &Board, ehs: &mut [f32; NUM_COMBOS]) {
    for combo_idx in 0..NUM_COMBOS {
        let combo = Combo::from_index(combo_idx);
        if combo.conflicts_with_mask(board.mask) {
            continue;
        }

        let mut equity_sum = 0.0f64;
        let mut runout_count = 0u32;

        let combined_mask = board.mask | combo.to_mask();

        // Enumerate river cards
        for river in 0..DECK_SIZE as Card {
            if (combined_mask >> river) & 1 != 0 {
                continue;
            }

            // Create 5-card board
            let river_board = [
                board.cards[0],
                board.cards[1],
                board.cards[2],
                board.cards[3],
                river,
            ];

            let equity = compute_equity_on_board(&river_board, combo, combined_mask | (1 << river));
            equity_sum += equity as f64;
            runout_count += 1;
        }

        if runout_count > 0 {
            ehs[combo_idx] = (equity_sum / runout_count as f64) as f32;
        }
    }
}

/// Compute flop EHS for all combos (average over turn+river cards).
fn compute_flop_ehs(board: &Board, ehs: &mut [f32; NUM_COMBOS]) {
    for combo_idx in 0..NUM_COMBOS {
        let combo = Combo::from_index(combo_idx);
        if combo.conflicts_with_mask(board.mask) {
            continue;
        }

        let mut equity_sum = 0.0f64;
        let mut runout_count = 0u32;

        let combined_mask = board.mask | combo.to_mask();

        // Enumerate turn+river cards
        for turn in 0..DECK_SIZE as Card {
            if (combined_mask >> turn) & 1 != 0 {
                continue;
            }
            let turn_mask = combined_mask | (1 << turn);

            for river in (turn + 1)..DECK_SIZE as Card {
                if (turn_mask >> river) & 1 != 0 {
                    continue;
                }

                let full_board = [
                    board.cards[0],
                    board.cards[1],
                    board.cards[2],
                    turn,
                    river,
                ];

                let equity = compute_equity_on_board(&full_board, combo, turn_mask | (1 << river));
                equity_sum += equity as f64;
                runout_count += 1;
            }
        }

        if runout_count > 0 {
            ehs[combo_idx] = (equity_sum / runout_count as f64) as f32;
        }
    }
}

/// Compute river EHS² for all combos.
fn compute_river_ehs_squared(board: &Board, ehs_sq: &mut [f32; NUM_COMBOS]) {
    for combo_idx in 0..NUM_COMBOS {
        let combo = Combo::from_index(combo_idx);
        if combo.conflicts_with_mask(board.mask) {
            continue;
        }
        let equity = compute_river_equity_single(combo, board);
        ehs_sq[combo_idx] = equity * equity;
    }
}

/// Compute turn EHS² for all combos.
fn compute_turn_ehs_squared(board: &Board, ehs_sq: &mut [f32; NUM_COMBOS]) {
    for combo_idx in 0..NUM_COMBOS {
        let combo = Combo::from_index(combo_idx);
        if combo.conflicts_with_mask(board.mask) {
            continue;
        }

        let mut equity_sq_sum = 0.0f64;
        let mut runout_count = 0u32;

        let combined_mask = board.mask | combo.to_mask();

        for river in 0..DECK_SIZE as Card {
            if (combined_mask >> river) & 1 != 0 {
                continue;
            }

            let river_board = [
                board.cards[0],
                board.cards[1],
                board.cards[2],
                board.cards[3],
                river,
            ];

            let equity = compute_equity_on_board(&river_board, combo, combined_mask | (1 << river));
            equity_sq_sum += (equity * equity) as f64;
            runout_count += 1;
        }

        if runout_count > 0 {
            ehs_sq[combo_idx] = (equity_sq_sum / runout_count as f64) as f32;
        }
    }
}

/// Compute flop EHS² for all combos.
fn compute_flop_ehs_squared(board: &Board, ehs_sq: &mut [f32; NUM_COMBOS]) {
    for combo_idx in 0..NUM_COMBOS {
        let combo = Combo::from_index(combo_idx);
        if combo.conflicts_with_mask(board.mask) {
            continue;
        }

        let mut equity_sq_sum = 0.0f64;
        let mut runout_count = 0u32;

        let combined_mask = board.mask | combo.to_mask();

        for turn in 0..DECK_SIZE as Card {
            if (combined_mask >> turn) & 1 != 0 {
                continue;
            }
            let turn_mask = combined_mask | (1 << turn);

            for river in (turn + 1)..DECK_SIZE as Card {
                if (turn_mask >> river) & 1 != 0 {
                    continue;
                }

                let full_board = [
                    board.cards[0],
                    board.cards[1],
                    board.cards[2],
                    turn,
                    river,
                ];

                let equity = compute_equity_on_board(&full_board, combo, turn_mask | (1 << river));
                equity_sq_sum += (equity * equity) as f64;
                runout_count += 1;
            }
        }

        if runout_count > 0 {
            ehs_sq[combo_idx] = (equity_sq_sum / runout_count as f64) as f32;
        }
    }
}

/// Compute equity for a hand on a 5-card board against uniform opponent range.
fn compute_equity_on_board(board: &[Card; 5], combo: Combo, dead_mask: u64) -> f32 {
    let our_cards = [
        board[0], board[1], board[2], board[3], board[4], combo.c0, combo.c1,
    ];
    let our_rank = evaluate_7cards(&our_cards);

    let mut equity_sum = 0.0f32;
    let mut total = 0u32;

    for opp_idx in 0..NUM_COMBOS {
        let opp = Combo::from_index(opp_idx);
        if opp.conflicts_with_mask(dead_mask) {
            continue;
        }

        let opp_cards = [
            board[0], board[1], board[2], board[3], board[4], opp.c0, opp.c1,
        ];
        let opp_rank = evaluate_7cards(&opp_cards);

        if our_rank > opp_rank {
            equity_sum += 1.0;
        } else if our_rank == opp_rank {
            equity_sum += 0.5;
        }
        total += 1;
    }

    if total == 0 {
        return 0.0;
    }
    equity_sum / total as f32
}

/// Compute turn EMD features for a combo.
fn compute_turn_emd_features(combo: Combo, board: &Board, histogram: &mut [f32; EMD_NUM_BINS]) {
    let combined_mask = board.mask | combo.to_mask();
    let mut runout_count = 0u32;

    for river in 0..DECK_SIZE as Card {
        if (combined_mask >> river) & 1 != 0 {
            continue;
        }

        let river_board = [
            board.cards[0],
            board.cards[1],
            board.cards[2],
            board.cards[3],
            river,
        ];

        let equity = compute_equity_on_board(&river_board, combo, combined_mask | (1 << river));
        let bin = equity_to_bin(equity);
        histogram[bin] += 1.0;
        runout_count += 1;
    }

    // Normalize to probability distribution
    if runout_count > 0 {
        let scale = 1.0 / runout_count as f32;
        for h in histogram.iter_mut() {
            *h *= scale;
        }
    }
}

/// Compute flop EMD features for a combo.
fn compute_flop_emd_features(combo: Combo, board: &Board, histogram: &mut [f32; EMD_NUM_BINS]) {
    let combined_mask = board.mask | combo.to_mask();
    let mut runout_count = 0u32;

    for turn in 0..DECK_SIZE as Card {
        if (combined_mask >> turn) & 1 != 0 {
            continue;
        }
        let turn_mask = combined_mask | (1 << turn);

        for river in (turn + 1)..DECK_SIZE as Card {
            if (turn_mask >> river) & 1 != 0 {
                continue;
            }

            let full_board = [
                board.cards[0],
                board.cards[1],
                board.cards[2],
                turn,
                river,
            ];

            let equity = compute_equity_on_board(&full_board, combo, turn_mask | (1 << river));
            let bin = equity_to_bin(equity);
            histogram[bin] += 1.0;
            runout_count += 1;
        }
    }

    // Normalize
    if runout_count > 0 {
        let scale = 1.0 / runout_count as f32;
        for h in histogram.iter_mut() {
            *h *= scale;
        }
    }
}

/// Compute turn asymmetric EMD features.
fn compute_turn_asymemd_features(
    combo: Combo,
    board: &Board,
    histogram: &mut [f32; EMD_NUM_BINS],
) {
    let combined_mask = board.mask | combo.to_mask();
    let mut runout_count = 0u32;

    for river in 0..DECK_SIZE as Card {
        if (combined_mask >> river) & 1 != 0 {
            continue;
        }

        let river_board = [
            board.cards[0],
            board.cards[1],
            board.cards[2],
            board.cards[3],
            river,
        ];

        let equity = compute_equity_on_board(&river_board, combo, combined_mask | (1 << river));
        let bin = equity_to_asymmetric_bin(equity);
        histogram[bin] += 1.0;
        runout_count += 1;
    }

    if runout_count > 0 {
        let scale = 1.0 / runout_count as f32;
        for h in histogram.iter_mut() {
            *h *= scale;
        }
    }
}

/// Compute flop asymmetric EMD features.
fn compute_flop_asymemd_features(
    combo: Combo,
    board: &Board,
    histogram: &mut [f32; EMD_NUM_BINS],
) {
    let combined_mask = board.mask | combo.to_mask();
    let mut runout_count = 0u32;

    for turn in 0..DECK_SIZE as Card {
        if (combined_mask >> turn) & 1 != 0 {
            continue;
        }
        let turn_mask = combined_mask | (1 << turn);

        for river in (turn + 1)..DECK_SIZE as Card {
            if (turn_mask >> river) & 1 != 0 {
                continue;
            }

            let full_board = [
                board.cards[0],
                board.cards[1],
                board.cards[2],
                turn,
                river,
            ];

            let equity = compute_equity_on_board(&full_board, combo, turn_mask | (1 << river));
            let bin = equity_to_asymmetric_bin(equity);
            histogram[bin] += 1.0;
            runout_count += 1;
        }
    }

    if runout_count > 0 {
        let scale = 1.0 / runout_count as f32;
        for h in histogram.iter_mut() {
            *h *= scale;
        }
    }
}

// ============================================================================
// Bin conversion utilities
// ============================================================================

/// Convert equity [0, 1] to symmetric bin index [0, 49].
///
/// Each bin is 0.02 wide: bin 0 = [0.00, 0.02), bin 49 = [0.98, 1.00]
#[inline]
fn equity_to_bin(equity: f32) -> usize {
    let bin = (equity * EMD_NUM_BINS as f32) as usize;
    bin.min(EMD_NUM_BINS - 1)
}

/// Convert equity to asymmetric bin index.
///
/// Finer bins at high equities where decisions matter more:
/// - [0.00, 0.20): 5 bins
/// - [0.20, 0.50): 10 bins
/// - [0.50, 0.80): 15 bins
/// - [0.80, 1.00]: 20 bins
#[inline]
fn equity_to_asymmetric_bin(equity: f32) -> usize {
    if equity < 0.20 {
        // 5 bins for [0.00, 0.20), bin width = 0.04
        let bin = (equity / 0.04) as usize;
        bin.min(4)
    } else if equity < 0.50 {
        // 10 bins for [0.20, 0.50), bin width = 0.03
        let bin = ((equity - 0.20) / 0.03) as usize;
        5 + bin.min(9)
    } else if equity < 0.80 {
        // 15 bins for [0.50, 0.80), bin width = 0.02
        let bin = ((equity - 0.50) / 0.02) as usize;
        15 + bin.min(14)
    } else {
        // 20 bins for [0.80, 1.00], bin width = 0.01
        let bin = ((equity - 0.80) / 0.01) as usize;
        30 + bin.min(19)
    }
}

/// Compute EMD (Earth Mover's Distance) between two histograms.
///
/// For 1D histograms, EMD is the sum of absolute differences of cumulative
/// distributions.
pub fn emd_distance(hist1: &[f32; EMD_NUM_BINS], hist2: &[f32; EMD_NUM_BINS]) -> f32 {
    let mut cum1 = 0.0f32;
    let mut cum2 = 0.0f32;
    let mut distance = 0.0f32;

    for i in 0..EMD_NUM_BINS {
        cum1 += hist1[i];
        cum2 += hist2[i];
        distance += (cum1 - cum2).abs();
    }

    distance
}

/// Compute L2 (Euclidean) distance between two histograms.
pub fn l2_distance(hist1: &[f32; EMD_NUM_BINS], hist2: &[f32; EMD_NUM_BINS]) -> f32 {
    let mut sum_sq = 0.0f32;
    for i in 0..EMD_NUM_BINS {
        let diff = hist1[i] - hist2[i];
        sum_sq += diff * diff;
    }
    sum_sq.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::poker::board_parser::parse_board;
    use crate::poker::hands::make_card;

    #[test]
    fn test_equity_to_bin() {
        assert_eq!(equity_to_bin(0.0), 0);
        assert_eq!(equity_to_bin(0.01), 0);
        assert_eq!(equity_to_bin(0.02), 1);
        assert_eq!(equity_to_bin(0.5), 25);
        assert_eq!(equity_to_bin(0.99), 49);
        assert_eq!(equity_to_bin(1.0), 49);
    }

    #[test]
    fn test_asymmetric_bins() {
        // Low equity region
        assert_eq!(equity_to_asymmetric_bin(0.0), 0);
        assert_eq!(equity_to_asymmetric_bin(0.19), 4);

        // Mid-low region
        assert_eq!(equity_to_asymmetric_bin(0.20), 5);
        assert_eq!(equity_to_asymmetric_bin(0.49), 14);

        // Mid-high region
        assert_eq!(equity_to_asymmetric_bin(0.50), 15);
        assert_eq!(equity_to_asymmetric_bin(0.79), 29);

        // High equity region (finest bins)
        assert_eq!(equity_to_asymmetric_bin(0.80), 30);
        assert_eq!(equity_to_asymmetric_bin(0.99), 49);
    }

    #[test]
    fn test_river_ehs() {
        let board = parse_board("Kh Qs Js 2c 3d").unwrap();
        let ehs = compute_all_ehs(&board.cards);

        // AA should have high equity
        let ac = make_card(12, 0);
        let ad = make_card(12, 1);
        let aa = Combo::new(ac, ad);
        let aa_ehs = ehs[aa.to_index()];
        assert!(aa_ehs > 0.7, "AA should have high equity, got {}", aa_ehs);

        // 22 should have lower equity
        let c2 = make_card(0, 0);
        let d2 = make_card(0, 1);
        let pair_22 = Combo::new(c2, d2);
        let pair_22_ehs = ehs[pair_22.to_index()];
        assert!(
            pair_22_ehs < aa_ehs,
            "22 should have lower equity than AA"
        );

        // Blocked combo should have 0 EHS
        let kh = make_card(11, 2); // Kh is on board
        let ah = make_card(12, 2);
        let blocked = Combo::new(kh, ah);
        assert_eq!(
            ehs[blocked.to_index()],
            0.0,
            "Blocked combo should have 0 EHS"
        );
    }

    #[test]
    fn test_winsplit_features() {
        let board = parse_board("Kh Qs Js 2c 3d").unwrap();

        // AA should have high win rate
        let ac = make_card(12, 0);
        let ad = make_card(12, 1);
        let aa = Combo::new(ac, ad);
        let [win, split] = compute_winsplit_features(aa, &board.cards);
        assert!(win > 0.7, "AA should win often, got {}", win);
        assert!(split < 0.1, "AA should rarely split, got {}", split);
    }

    #[test]
    fn test_emd_distance() {
        let hist1 = [1.0 / EMD_NUM_BINS as f32; EMD_NUM_BINS];
        let hist2 = [1.0 / EMD_NUM_BINS as f32; EMD_NUM_BINS];

        // Same histograms should have 0 distance
        let dist = emd_distance(&hist1, &hist2);
        assert!(dist < 0.001, "Same histograms should have ~0 EMD");

        // Very different histograms should have high distance
        let mut hist3 = [0.0f32; EMD_NUM_BINS];
        hist3[0] = 1.0; // All mass in first bin
        let mut hist4 = [0.0f32; EMD_NUM_BINS];
        hist4[49] = 1.0; // All mass in last bin

        let dist2 = emd_distance(&hist3, &hist4);
        assert!(dist2 > 10.0, "Opposite histograms should have high EMD");
    }

    #[test]
    fn test_turn_ehs() {
        let board = parse_board("Kh Qs Js 2c").unwrap();
        let ehs = compute_all_ehs(&board.cards);

        // Check that we get reasonable values
        let mut valid_count = 0;
        let mut sum_ehs = 0.0;

        for &e in &ehs {
            if e > 0.0 {
                valid_count += 1;
                sum_ehs += e;
            }
        }

        assert!(valid_count > 1000, "Should have many valid combos");

        let avg_ehs = sum_ehs / valid_count as f32;
        // Average EHS should be around 0.5 (since we're vs uniform range)
        assert!(
            (avg_ehs - 0.5).abs() < 0.1,
            "Average EHS should be ~0.5, got {}",
            avg_ehs
        );
    }

    #[test]
    fn test_emd_features_turn() {
        let board = parse_board("Kh Qs Js 2c").unwrap();

        let ac = make_card(12, 0);
        let ad = make_card(12, 1);
        let aa = Combo::new(ac, ad);

        let histogram = compute_emd_features(aa, &board.cards);

        // Should sum to ~1.0 (probability distribution)
        let sum: f32 = histogram.iter().sum();
        assert!(
            (sum - 1.0).abs() < 0.01,
            "Histogram should sum to 1.0, got {}",
            sum
        );

        // AA should have most mass in high equity bins
        let high_equity_mass: f32 = histogram[35..].iter().sum();
        assert!(
            high_equity_mass > 0.5,
            "AA should have most equity in high bins"
        );
    }
}
