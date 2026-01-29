//! Matchup table: precomputed valid matchups and showdown results.
//!
//! For range-vs-range solving, we need to know:
//! 1. Which (OOP combo, IP combo) pairs are valid (no card conflicts)
//! 2. Who wins at showdown for each valid pair

use crate::poker::hands::{rank, suit, Board, Card, Combo, NUM_COMBOS};

/// Hand category constants (higher = better).
const HIGH_CARD: u32 = 0;
const PAIR: u32 = 1;
const TWO_PAIR: u32 = 2;
const TRIPS: u32 = 3;
const STRAIGHT: u32 = 4;
const FLUSH: u32 = 5;
const FULL_HOUSE: u32 = 6;
const QUADS: u32 = 7;
const STRAIGHT_FLUSH: u32 = 8;

/// Precomputed matchup table for a specific board.
pub struct MatchupTable {
    /// The board used to compute this table.
    pub board: Board,
    /// Hand rank for each combo (u32::MAX if blocked).
    pub hand_ranks: [u32; NUM_COMBOS],
    /// Valid matchup bitmap: bit at (i * NUM_COMBOS + j) % 64 of word (i * NUM_COMBOS + j) / 64.
    /// 1 = valid matchup (combos don't conflict), 0 = invalid.
    valid_matchups: Vec<u64>,
}

impl MatchupTable {
    /// Create a matchup table for a river board.
    pub fn new(board: &Board) -> Self {
        assert!(board.len() == 5, "MatchupTable requires a 5-card board");

        let mut hand_ranks = [u32::MAX; NUM_COMBOS];

        // Compute hand rank for each valid combo
        for combo_idx in 0..NUM_COMBOS {
            let combo = Combo::from_index(combo_idx);
            if !combo.conflicts_with_mask(board.mask) {
                let cards = board.with_combo(combo);
                hand_ranks[combo_idx] = evaluate_7cards(&cards);
            }
        }

        // Compute valid matchups bitmap
        let num_bits = NUM_COMBOS * NUM_COMBOS;
        let num_words = (num_bits + 63) / 64;
        let mut valid_matchups = vec![0u64; num_words];

        for i in 0..NUM_COMBOS {
            if hand_ranks[i] == u32::MAX {
                continue; // Combo i is blocked
            }
            let combo_i = Combo::from_index(i);

            for j in 0..NUM_COMBOS {
                if hand_ranks[j] == u32::MAX {
                    continue; // Combo j is blocked
                }
                let combo_j = Combo::from_index(j);

                if !combo_i.conflicts_with(&combo_j) {
                    let bit_idx = i * NUM_COMBOS + j;
                    valid_matchups[bit_idx / 64] |= 1u64 << (bit_idx % 64);
                }
            }
        }

        MatchupTable {
            board: board.clone(),
            hand_ranks,
            valid_matchups,
        }
    }

    /// Check if a combo is valid (not blocked by board).
    #[inline]
    pub fn is_valid_combo(&self, combo_idx: usize) -> bool {
        self.hand_ranks[combo_idx] != u32::MAX
    }

    /// Check if a matchup is valid (combos don't conflict with each other or board).
    #[inline]
    pub fn is_valid_matchup(&self, oop_idx: usize, ip_idx: usize) -> bool {
        let bit_idx = oop_idx * NUM_COMBOS + ip_idx;
        (self.valid_matchups[bit_idx / 64] >> (bit_idx % 64)) & 1 == 1
    }

    /// Compare two combos at showdown.
    ///
    /// Returns:
    /// - 1 if OOP wins
    /// - -1 if IP wins
    /// - 0 if tie
    #[inline]
    pub fn compare(&self, oop_idx: usize, ip_idx: usize) -> i8 {
        let oop_rank = self.hand_ranks[oop_idx];
        let ip_rank = self.hand_ranks[ip_idx];

        if oop_rank > ip_rank {
            1
        } else if ip_rank > oop_rank {
            -1
        } else {
            0
        }
    }

    /// Get hand rank for a combo (for debugging/display).
    #[inline]
    pub fn hand_rank(&self, combo_idx: usize) -> Option<u32> {
        let rank = self.hand_ranks[combo_idx];
        if rank == u32::MAX {
            None
        } else {
            Some(rank)
        }
    }

    /// Count valid combos.
    pub fn valid_combo_count(&self) -> usize {
        self.hand_ranks
            .iter()
            .filter(|&&r| r != u32::MAX)
            .count()
    }

    /// Count valid matchups.
    pub fn valid_matchup_count(&self) -> usize {
        self.valid_matchups
            .iter()
            .map(|w| w.count_ones() as usize)
            .sum()
    }

    /// Iterate over valid matchups.
    pub fn valid_matchups(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        (0..NUM_COMBOS).flat_map(move |i| {
            (0..NUM_COMBOS)
                .filter(move |&j| self.is_valid_matchup(i, j))
                .map(move |j| (i, j))
        })
    }
}

/// Evaluate a hand given a 5-card board and two hole cards.
///
/// Returns a comparable rank (higher = better), or `u32::MAX` if blocked.
pub fn evaluate_hand(board: &[Card], c0: Card, c1: Card) -> u32 {
    assert!(board.len() == 5, "Board must have 5 cards");
    let cards = [board[0], board[1], board[2], board[3], board[4], c0, c1];
    evaluate_7cards(&cards)
}

/// Evaluate a 7-card hand and return a comparable rank (higher = better).
pub fn evaluate_7cards(cards: &[Card; 7]) -> u32 {
    // Count cards per rank and per suit
    let mut rank_counts = [0u8; 13];
    let mut suit_counts = [0u8; 4];
    let mut suit_cards = [0u16; 4]; // Bitmask of ranks for each suit

    for &c in cards {
        let r = rank(c) as usize;
        let s = suit(c) as usize;
        rank_counts[r] += 1;
        suit_counts[s] += 1;
        suit_cards[s] |= 1 << r;
    }

    // Check for flush (5+ cards of same suit)
    let flush_suit = suit_counts.iter().position(|&c| c >= 5);

    // Rank bitmask for straight detection
    let rank_bits: u16 = rank_counts
        .iter()
        .enumerate()
        .filter(|&(_, c)| *c > 0)
        .fold(0u16, |acc, (r, _)| acc | (1 << r));

    // Check for straight
    let straight_high = find_straight(rank_bits);

    // Check for straight flush
    if let Some(fs) = flush_suit {
        let sf_high = find_straight(suit_cards[fs]);
        if let Some(sf_rank) = sf_high {
            return encode_hand(STRAIGHT_FLUSH, &[sf_rank]);
        }
    }

    // Count quads, trips, pairs
    let mut quads = Vec::new();
    let mut trips = Vec::new();
    let mut pairs = Vec::new();
    let mut singles = Vec::new();

    for r in (0..13).rev() {
        match rank_counts[r] {
            4 => quads.push(r as u8),
            3 => trips.push(r as u8),
            2 => pairs.push(r as u8),
            1 => singles.push(r as u8),
            _ => {}
        }
    }

    // Determine hand category and kickers
    if !quads.is_empty() {
        let kicker = find_best_kicker(&trips, &pairs, &singles, 1);
        return encode_hand(QUADS, &[quads[0], kicker[0]]);
    }

    if !trips.is_empty() && (!pairs.is_empty() || trips.len() >= 2) {
        let pair_rank = if trips.len() >= 2 {
            trips[1]
        } else {
            pairs[0]
        };
        return encode_hand(FULL_HOUSE, &[trips[0], pair_rank]);
    }

    if flush_suit.is_some() {
        let fs = flush_suit.unwrap();
        let flush_ranks = extract_top_n_bits(suit_cards[fs], 5);
        return encode_hand(FLUSH, &flush_ranks);
    }

    if let Some(high) = straight_high {
        return encode_hand(STRAIGHT, &[high]);
    }

    if !trips.is_empty() {
        let kickers = find_best_kicker(&[], &pairs, &singles, 2);
        return encode_hand(TRIPS, &[trips[0], kickers[0], kickers[1]]);
    }

    if pairs.len() >= 2 {
        let kicker = find_best_kicker(&[], &pairs[2..].to_vec(), &singles, 1);
        return encode_hand(TWO_PAIR, &[pairs[0], pairs[1], kicker[0]]);
    }

    if pairs.len() == 1 {
        let kickers = find_best_kicker(&[], &[], &singles, 3);
        return encode_hand(
            PAIR,
            &[pairs[0], kickers[0], kickers[1], kickers[2]],
        );
    }

    // High card: top 5 cards
    let kickers = find_best_kicker(&[], &[], &singles, 5);
    encode_hand(HIGH_CARD, &kickers)
}

/// Find the highest straight in a rank bitmask, returns high card rank.
fn find_straight(bits: u16) -> Option<u8> {
    const STRAIGHT_MASKS: [(u16, u8); 10] = [
        (0b1111100000000, 12), // AKQJT
        (0b0111110000000, 11), // KQJT9
        (0b0011111000000, 10), // QJT98
        (0b0001111100000, 9),  // JT987
        (0b0000111110000, 8),  // T9876
        (0b0000011111000, 7),  // 98765
        (0b0000001111100, 6),  // 87654
        (0b0000000111110, 5),  // 76543
        (0b0000000011111, 4),  // 65432
        (0b1000000001111, 3),  // 5432A (wheel)
    ];

    for &(mask, high) in &STRAIGHT_MASKS {
        if (bits & mask) == mask {
            return Some(high);
        }
    }
    None
}

/// Extract the top N set bits from a bitmask as ranks (descending order).
fn extract_top_n_bits(mut bits: u16, n: usize) -> Vec<u8> {
    let mut result = Vec::with_capacity(n);
    for r in (0..13).rev() {
        if result.len() >= n {
            break;
        }
        if (bits & (1 << r)) != 0 {
            result.push(r as u8);
            bits &= !(1 << r);
        }
    }
    result
}

/// Find the best N kickers from available cards.
fn find_best_kicker(trips: &[u8], pairs: &[u8], singles: &[u8], n: usize) -> Vec<u8> {
    let mut all: Vec<u8> = trips.iter().copied().collect();
    all.extend(pairs.iter().copied());
    all.extend(singles.iter().copied());
    all.sort_by(|a, b| b.cmp(a));
    all.truncate(n);
    while all.len() < n {
        all.push(0);
    }
    all
}

/// Encode a hand category and kickers into a single comparable u32.
fn encode_hand(category: u32, kickers: &[u8]) -> u32 {
    let mut result = category << 20;
    for (i, &k) in kickers.iter().take(5).enumerate() {
        result |= (k as u32) << (16 - i * 4);
    }
    result
}

/// Get the hand category name for a rank.
pub fn hand_category_name(rank: u32) -> &'static str {
    match rank >> 20 {
        0 => "High Card",
        1 => "Pair",
        2 => "Two Pair",
        3 => "Three of a Kind",
        4 => "Straight",
        5 => "Flush",
        6 => "Full House",
        7 => "Four of a Kind",
        8 => "Straight Flush",
        _ => "Unknown",
    }
}

/// Compute multiway equity using pairwise approximation.
///
/// Given hand ranks for N players and an active mask indicating which players are in the pot,
/// returns the equity (probability of winning) for each player.
///
/// The approximation uses: P(player i wins) ∝ ∏_{j≠i, j active} P(i beats j)
/// where P(i beats j) is derived from comparing hand ranks.
///
/// # Arguments
/// * `hand_ranks` - Hand rank for each player (higher = better, u32::MAX = folded/blocked)
/// * `active_mask` - Bitmask of active players (bit i set = player i is in the pot)
///
/// # Returns
/// A vector of equities, one per player. Folded/blocked players get 0.0.
pub fn compute_multiway_equity(hand_ranks: &[u32], active_mask: u64) -> Vec<f64> {
    let n = hand_ranks.len();
    let mut equity = vec![0.0; n];

    // Count active players and find their indices
    let active_players: Vec<usize> = (0..n)
        .filter(|&i| (active_mask >> i) & 1 == 1 && hand_ranks[i] != u32::MAX)
        .collect();

    let num_active = active_players.len();

    if num_active == 0 {
        return equity;
    }

    if num_active == 1 {
        // Single player wins automatically
        equity[active_players[0]] = 1.0;
        return equity;
    }

    if num_active == 2 {
        // Heads-up: direct comparison
        let p0 = active_players[0];
        let p1 = active_players[1];
        if hand_ranks[p0] > hand_ranks[p1] {
            equity[p0] = 1.0;
        } else if hand_ranks[p1] > hand_ranks[p0] {
            equity[p1] = 1.0;
        } else {
            // Tie
            equity[p0] = 0.5;
            equity[p1] = 0.5;
        }
        return equity;
    }

    // Multi-way: use pairwise approximation
    // P(player i wins) ≈ ∏_{j≠i} P(i beats j), then normalize
    let mut raw_equity = vec![0.0; n];

    for &pi in &active_players {
        let rank_i = hand_ranks[pi];
        let mut product = 1.0;

        for &pj in &active_players {
            if pi == pj {
                continue;
            }
            let rank_j = hand_ranks[pj];

            // P(i beats j): 1.0 if i > j, 0.5 if tie, 0.0 if j > i
            let p_win = if rank_i > rank_j {
                1.0
            } else if rank_i == rank_j {
                0.5
            } else {
                0.0
            };
            product *= p_win;
        }

        raw_equity[pi] = product;
    }

    // Normalize
    let sum: f64 = raw_equity.iter().sum();
    if sum > 0.0 {
        for i in 0..n {
            equity[i] = raw_equity[i] / sum;
        }
    } else {
        // All zero products (everyone loses to someone) - use fallback
        // Find the best hand(s) and split equally
        let best_rank = active_players.iter().map(|&i| hand_ranks[i]).max().unwrap();
        let winners: Vec<usize> = active_players
            .iter()
            .filter(|&&i| hand_ranks[i] == best_rank)
            .copied()
            .collect();
        let share = 1.0 / winners.len() as f64;
        for w in winners {
            equity[w] = share;
        }
    }

    equity
}

/// Compute multiway equity for a specific set of combo indices against a board.
///
/// This is a convenience wrapper that looks up hand ranks and calls `compute_multiway_equity`.
///
/// # Arguments
/// * `matchups` - The matchup table for the current board
/// * `combo_indices` - Combo index for each player (usize::MAX for folded players)
/// * `active_mask` - Bitmask of players in the showdown
pub fn compute_multiway_equity_from_matchups(
    matchups: &MatchupTable,
    combo_indices: &[usize],
    active_mask: u64,
) -> Vec<f64> {
    let hand_ranks: Vec<u32> = combo_indices
        .iter()
        .map(|&idx| {
            if idx == usize::MAX {
                u32::MAX
            } else {
                matchups.hand_ranks[idx]
            }
        })
        .collect();

    compute_multiway_equity(&hand_ranks, active_mask)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::poker::board_parser::parse_board;
    use crate::poker::hands::make_card;

    #[test]
    fn test_matchup_table_creation() {
        let board = parse_board("KhQsJs2c3d").unwrap();
        let table = MatchupTable::new(&board);

        // Should have many valid combos
        assert!(table.valid_combo_count() > 1000);

        // Should have even more valid matchups
        assert!(table.valid_matchup_count() > 0);
    }

    #[test]
    fn test_blocked_combos() {
        let board = parse_board("KhQsJs2c3d").unwrap();
        let table = MatchupTable::new(&board);

        // Kh combo should be blocked
        let kh = make_card(11, 2); // Kh
        let ah = make_card(12, 2); // Ah
        let blocked_combo = Combo::new(kh, ah);
        assert!(!table.is_valid_combo(blocked_combo.to_index()));

        // AsAd should be valid
        let as_ = make_card(12, 3); // As
        let ad = make_card(12, 1); // Ad
        let valid_combo = Combo::new(as_, ad);
        assert!(table.is_valid_combo(valid_combo.to_index()));
    }

    #[test]
    fn test_conflicting_combos() {
        let board = parse_board("KhQsJs2c3d").unwrap();
        let table = MatchupTable::new(&board);

        // AsAh vs AsKd - should conflict (both have As)
        let as_ = make_card(12, 3);
        let ah = make_card(12, 2);
        let kd = make_card(11, 1);

        let combo1 = Combo::new(as_, ah);
        let combo2 = Combo::new(as_, kd);

        // combo1 is blocked (Ah is on board as Kh? No, Kh != Ah)
        // Actually let me recalculate - board is KhQsJs2c3d
        // Kh = King of hearts, Ah = Ace of hearts
        // Both should be valid

        // But they conflict with each other because they share As
        if table.is_valid_combo(combo1.to_index()) && table.is_valid_combo(combo2.to_index()) {
            assert!(!table.is_valid_matchup(combo1.to_index(), combo2.to_index()));
        }
    }

    #[test]
    fn test_showdown_comparison() {
        let board = parse_board("Kh Qs Js 2c 3d").unwrap();
        let table = MatchupTable::new(&board);

        // AA vs KK - On this board with Kh, KK (KcKd) makes trips Kings
        // which beats AA (pair of Aces)
        let ac = make_card(12, 0); // Ac
        let ad = make_card(12, 1); // Ad
        let kc = make_card(11, 0); // Kc
        let kd = make_card(11, 1); // Kd

        // Note: Kh is on board, so KcKd makes three Kings
        let aa = Combo::new(ac, ad);
        let kk = Combo::new(kc, kd);

        // Both should be valid
        assert!(table.is_valid_combo(aa.to_index()));
        assert!(table.is_valid_combo(kk.to_index()));

        // KK (trips) beats AA (pair) on this board
        let result = table.compare(aa.to_index(), kk.to_index());
        assert_eq!(result, -1); // KK wins (trips vs pair)

        // Now test on a board where AA wins
        let board2 = parse_board("7h 5s 3s 2c 9d").unwrap();
        let table2 = MatchupTable::new(&board2);

        // On 7h5s3s2c9d, AA beats KK (both are pairs)
        let result2 = table2.compare(aa.to_index(), kk.to_index());
        assert_eq!(result2, 1); // AA wins (higher pair)
    }

    #[test]
    fn test_hand_evaluation() {
        let cards = [
            make_card(12, 0), // Ac
            make_card(12, 1), // Ad
            make_card(12, 2), // Ah
            make_card(12, 3), // As
            make_card(11, 0), // Kc
            make_card(10, 1), // Qd
            make_card(9, 2),  // Jh
        ];

        let rank = evaluate_7cards(&cards);
        assert_eq!(hand_category_name(rank), "Four of a Kind");
    }

    #[test]
    fn test_straight_flush() {
        let cards = [
            make_card(12, 0), // Ac
            make_card(11, 0), // Kc
            make_card(10, 0), // Qc
            make_card(9, 0),  // Jc
            make_card(8, 0),  // Tc
            make_card(2, 1),  // 4d
            make_card(0, 2),  // 2h
        ];

        let rank = evaluate_7cards(&cards);
        assert_eq!(hand_category_name(rank), "Straight Flush");
    }

    #[test]
    fn test_multiway_equity_headsup() {
        // Player 0 has better hand, player 1 has worse hand
        let hand_ranks = vec![100, 50];
        let active_mask = 0b11; // Both active

        let equity = compute_multiway_equity(&hand_ranks, active_mask);
        assert_eq!(equity[0], 1.0);
        assert_eq!(equity[1], 0.0);
    }

    #[test]
    fn test_multiway_equity_tie() {
        // Both players have same hand
        let hand_ranks = vec![100, 100];
        let active_mask = 0b11;

        let equity = compute_multiway_equity(&hand_ranks, active_mask);
        assert!((equity[0] - 0.5).abs() < 0.001);
        assert!((equity[1] - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_multiway_equity_3way() {
        // Player 0 has best hand, player 1 middle, player 2 worst
        let hand_ranks = vec![100, 80, 60];
        let active_mask = 0b111; // All three active

        let equity = compute_multiway_equity(&hand_ranks, active_mask);
        assert_eq!(equity[0], 1.0); // Best hand wins
        assert_eq!(equity[1], 0.0);
        assert_eq!(equity[2], 0.0);
    }

    #[test]
    fn test_multiway_equity_folded_player() {
        // 3 players, but player 1 folded
        let hand_ranks = vec![100, u32::MAX, 60];
        let active_mask = 0b101; // Players 0 and 2 active

        let equity = compute_multiway_equity(&hand_ranks, active_mask);
        assert_eq!(equity[0], 1.0);
        assert_eq!(equity[1], 0.0); // Folded
        assert_eq!(equity[2], 0.0);
    }

    #[test]
    fn test_multiway_equity_single_winner() {
        // Only one player left
        let hand_ranks = vec![100, u32::MAX, u32::MAX];
        let active_mask = 0b001;

        let equity = compute_multiway_equity(&hand_ranks, active_mask);
        assert_eq!(equity[0], 1.0);
        assert_eq!(equity[1], 0.0);
        assert_eq!(equity[2], 0.0);
    }
}
