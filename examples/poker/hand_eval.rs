//! Fast 7-card hand evaluator for Texas Hold'em.
//!
//! Returns a u32 hand rank where higher values beat lower values.

#![allow(dead_code)]

use super::cards::{rank, suit, Card};

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

/// Evaluate a 7-card hand and return a comparable rank (higher = better).
///
/// The returned value encodes the hand category in the high bits and
/// the relevant card ranks for comparison in the lower bits.
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
        // Four of a kind + best kicker
        let kicker = find_best_kicker(&trips, &pairs, &singles, 1);
        return encode_hand(QUADS, &[quads[0], kicker[0]]);
    }

    if !trips.is_empty() && (!pairs.is_empty() || trips.len() >= 2) {
        // Full house: best trips + best pair (or second trips)
        let pair_rank = if trips.len() >= 2 {
            trips[1]
        } else {
            pairs[0]
        };
        return encode_hand(FULL_HOUSE, &[trips[0], pair_rank]);
    }

    if flush_suit.is_some() {
        // Flush: top 5 cards of flush suit
        let fs = flush_suit.unwrap();
        let flush_ranks = extract_top_n_bits(suit_cards[fs], 5);
        return encode_hand(FLUSH, &flush_ranks);
    }

    if let Some(high) = straight_high {
        return encode_hand(STRAIGHT, &[high]);
    }

    if !trips.is_empty() {
        // Three of a kind + 2 kickers
        let kickers = find_best_kicker(&[], &pairs, &singles, 2);
        return encode_hand(TRIPS, &[trips[0], kickers[0], kickers[1]]);
    }

    if pairs.len() >= 2 {
        // Two pair + 1 kicker
        let kicker = find_best_kicker(&[], &pairs[2..].to_vec(), &singles, 1);
        return encode_hand(TWO_PAIR, &[pairs[0], pairs[1], kicker[0]]);
    }

    if pairs.len() == 1 {
        // One pair + 3 kickers
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
    // Check for A-high straight (AKQJT = bits 12,11,10,9,8)
    // down to wheel (5432A = bits 3,2,1,0,12)
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

/// Evaluate a 5-card hand (convenience function).
pub fn evaluate_5cards(cards: &[Card; 5]) -> u32 {
    let mut seven = [0u8; 7];
    seven[..5].copy_from_slice(cards);
    // Pad with impossible duplicate cards that won't affect ranking
    seven[5] = cards[0];
    seven[6] = cards[1];
    evaluate_7cards(&seven)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::poker::cards::make_card;

    #[test]
    fn test_high_card() {
        // A K Q J 9 different suits
        let cards = [
            make_card(12, 0), // Ac
            make_card(11, 1), // Kd
            make_card(10, 2), // Qh
            make_card(9, 3),  // Js
            make_card(7, 0),  // 9c
            make_card(5, 1),  // 7d
            make_card(3, 2),  // 5h
        ];
        let rank = evaluate_7cards(&cards);
        assert_eq!(rank >> 20, HIGH_CARD);
    }

    #[test]
    fn test_pair() {
        let cards = [
            make_card(12, 0), // Ac
            make_card(12, 1), // Ad
            make_card(10, 2), // Qh
            make_card(9, 3),  // Js
            make_card(7, 0),  // 9c
            make_card(5, 1),  // 7d
            make_card(3, 2),  // 5h
        ];
        let rank = evaluate_7cards(&cards);
        assert_eq!(rank >> 20, PAIR);
    }

    #[test]
    fn test_flush() {
        // All hearts
        let cards = [
            make_card(12, 2), // Ah
            make_card(10, 2), // Qh
            make_card(8, 2),  // Th
            make_card(6, 2),  // 8h
            make_card(4, 2),  // 6h
            make_card(2, 0),  // 4c
            make_card(0, 1),  // 2d
        ];
        let rank = evaluate_7cards(&cards);
        assert_eq!(rank >> 20, FLUSH);
    }

    #[test]
    fn test_straight() {
        let cards = [
            make_card(8, 0),  // Tc
            make_card(7, 1),  // 9d
            make_card(6, 2),  // 8h
            make_card(5, 3),  // 7s
            make_card(4, 0),  // 6c
            make_card(2, 1),  // 4d
            make_card(0, 2),  // 2h
        ];
        let rank = evaluate_7cards(&cards);
        assert_eq!(rank >> 20, STRAIGHT);
    }

    #[test]
    fn test_straight_flush() {
        // All spades A-high straight
        let cards = [
            make_card(12, 3), // As
            make_card(11, 3), // Ks
            make_card(10, 3), // Qs
            make_card(9, 3),  // Js
            make_card(8, 3),  // Ts
            make_card(2, 0),  // 4c
            make_card(0, 1),  // 2d
        ];
        let rank = evaluate_7cards(&cards);
        assert_eq!(rank >> 20, STRAIGHT_FLUSH);
    }

    #[test]
    fn test_quads() {
        let cards = [
            make_card(12, 0), // Ac
            make_card(12, 1), // Ad
            make_card(12, 2), // Ah
            make_card(12, 3), // As
            make_card(8, 0),  // Tc
            make_card(6, 1),  // 8d
            make_card(4, 2),  // 6h
        ];
        let rank = evaluate_7cards(&cards);
        assert_eq!(rank >> 20, QUADS);
    }

    #[test]
    fn test_full_house() {
        let cards = [
            make_card(12, 0), // Ac
            make_card(12, 1), // Ad
            make_card(12, 2), // Ah
            make_card(8, 0),  // Tc
            make_card(8, 1),  // Td
            make_card(6, 2),  // 8h
            make_card(4, 3),  // 6s
        ];
        let rank = evaluate_7cards(&cards);
        assert_eq!(rank >> 20, FULL_HOUSE);
    }

    #[test]
    fn test_hand_ordering() {
        // Verify that hand categories are properly ordered
        let high_card = [
            make_card(12, 0),
            make_card(10, 1),
            make_card(8, 2),
            make_card(6, 3),
            make_card(4, 0),
            make_card(2, 1),
            make_card(0, 2),
        ];
        let pair = [
            make_card(12, 0),
            make_card(12, 1),
            make_card(8, 2),
            make_card(6, 3),
            make_card(4, 0),
            make_card(2, 1),
            make_card(0, 2),
        ];
        let flush = [
            make_card(12, 0),
            make_card(10, 0),
            make_card(8, 0),
            make_card(6, 0),
            make_card(4, 0),
            make_card(2, 1),
            make_card(0, 2),
        ];

        assert!(evaluate_7cards(&high_card) < evaluate_7cards(&pair));
        assert!(evaluate_7cards(&pair) < evaluate_7cards(&flush));
    }
}
