//! Card representation and canonical hand conversion for poker.
//!
//! Cards are encoded as `card_id = 4 * rank + suit` where:
//! - rank: 0 (deuce) to 12 (ace)
//! - suit: 0-3 (clubs, diamonds, hearts, spades)

#![allow(dead_code)]

/// A card encoded as `4 * rank + suit` (0-51).
pub type Card = u8;

/// A pair of hole cards.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HoleCards(pub Card, pub Card);

/// Canonical hand index (0-168) representing suit-isomorphic equivalence classes.
///
/// Pre-flop, only the relative suit pattern matters:
/// - 13 pairs (AA, KK, ..., 22): indices 0-12
/// - 78 suited hands (AKs, AQs, ..., 32s): indices 13-90
/// - 78 offsuit hands (AKo, AQo, ..., 32o): indices 91-168
pub type CanonicalHand = u8;

/// Total number of canonical hands (suit-isomorphic equivalence classes).
pub const NUM_CANONICAL_HANDS: usize = 169;

/// Number of cards in a standard deck.
pub const DECK_SIZE: usize = 52;

/// Extract rank (0-12) from a card.
#[inline]
pub fn rank(card: Card) -> u8 {
    card / 4
}

/// Extract suit (0-3) from a card.
#[inline]
pub fn suit(card: Card) -> u8 {
    card % 4
}

/// Create a card from rank (0-12) and suit (0-3).
#[inline]
pub fn make_card(rank: u8, suit: u8) -> Card {
    rank * 4 + suit
}

/// Triangle number index for non-pair hands.
/// For hi > lo, returns the index into the 78 possible non-pair combinations.
#[inline]
fn triangle_index(hi: u8, lo: u8) -> u8 {
    (hi * (hi - 1)) / 2 + lo
}

/// Convert a pair of cards to their canonical hand representation.
///
/// Returns (canonical_hand_index, num_raw_combos).
/// - Pairs: 6 combos each
/// - Suited: 4 combos each
/// - Offsuit: 12 combos each
pub fn to_canonical(c1: Card, c2: Card) -> (CanonicalHand, u8) {
    let r1 = rank(c1);
    let r2 = rank(c2);
    let suited = suit(c1) == suit(c2);

    if r1 == r2 {
        // Pair: index = rank, 6 combos
        (r1, 6)
    } else {
        let (hi, lo) = if r1 > r2 { (r1, r2) } else { (r2, r1) };
        let tri_idx = triangle_index(hi, lo);
        if suited {
            // Suited: 13 (pairs) + tri_idx, 4 combos
            (13 + tri_idx, 4)
        } else {
            // Offsuit: 13 (pairs) + 78 (suited) + tri_idx, 12 combos
            (13 + 78 + tri_idx, 12)
        }
    }
}

/// Convert a canonical hand index back to a representative pair of cards.
/// Uses suit 0 and 1 for offsuit, suit 0 for suited/pairs.
pub fn from_canonical(canonical: CanonicalHand) -> HoleCards {
    if canonical < 13 {
        // Pair: rank = canonical, use suits 0 and 1
        let r = canonical;
        HoleCards(make_card(r, 0), make_card(r, 1))
    } else if canonical < 91 {
        // Suited: find hi/lo from triangle index
        let tri_idx = canonical - 13;
        let (hi, lo) = triangle_from_index(tri_idx);
        HoleCards(make_card(hi, 0), make_card(lo, 0))
    } else {
        // Offsuit: find hi/lo from triangle index
        let tri_idx = canonical - 91;
        let (hi, lo) = triangle_from_index(tri_idx);
        HoleCards(make_card(hi, 0), make_card(lo, 1))
    }
}

/// Reverse lookup for triangle index to (hi, lo) pair.
fn triangle_from_index(idx: u8) -> (u8, u8) {
    // Find hi such that hi*(hi-1)/2 <= idx < (hi+1)*hi/2
    let mut hi = 1u8;
    while (hi + 1) * hi / 2 <= idx {
        hi += 1;
    }
    let lo = idx - (hi * (hi - 1)) / 2;
    (hi, lo)
}

/// Returns a bitmask of cards that conflict with a canonical hand.
/// Used for blocker calculations.
pub fn canonical_blockers(canonical: CanonicalHand) -> u64 {
    let HoleCards(c1, c2) = from_canonical(canonical);
    let r1 = rank(c1);
    let r2 = rank(c2);

    if r1 == r2 {
        // Pair: all 4 cards of this rank block
        rank_mask(r1)
    } else if canonical < 91 {
        // Suited: we need same suit for both ranks
        // For blockers, any card of either rank blocks some combos
        rank_mask(r1) | rank_mask(r2)
    } else {
        // Offsuit: any card of either rank blocks some combos
        rank_mask(r1) | rank_mask(r2)
    }
}

/// Returns a 64-bit mask with bits set for all 4 cards of a given rank.
#[inline]
fn rank_mask(r: u8) -> u64 {
    0xF << (r * 4)
}

/// Count available combos for a canonical hand given dead cards (as bitmask).
///
/// Dead cards reduce the number of available combinations.
pub fn available_combos(canonical: CanonicalHand, dead: u64) -> u8 {
    let HoleCards(c1, c2) = from_canonical(canonical);
    let r1 = rank(c1);
    let r2 = rank(c2);

    // Count live cards for each rank
    let live1 = count_live_cards(r1, dead);
    let live2 = count_live_cards(r2, dead);

    if r1 == r2 {
        // Pair: C(live, 2) = live * (live - 1) / 2
        if live1 < 2 {
            0
        } else {
            (live1 * (live1 - 1)) / 2
        }
    } else if canonical < 91 {
        // Suited: need both cards in same suit
        // Count suits where both cards are available
        let mut count = 0u8;
        for s in 0..4 {
            let c1 = make_card(r1, s);
            let c2 = make_card(r2, s);
            if (dead & (1 << c1)) == 0 && (dead & (1 << c2)) == 0 {
                count += 1;
            }
        }
        count
    } else {
        // Offsuit: any two cards of different suits
        // Total = live1 * live2 - suited_combos
        let mut suited = 0u8;
        for s in 0..4 {
            let c1 = make_card(r1, s);
            let c2 = make_card(r2, s);
            if (dead & (1 << c1)) == 0 && (dead & (1 << c2)) == 0 {
                suited += 1;
            }
        }
        if live1 == 0 || live2 == 0 {
            0
        } else {
            live1 * live2 - suited
        }
    }
}

/// Count how many cards of a given rank are still live (not dead).
#[inline]
fn count_live_cards(r: u8, dead: u64) -> u8 {
    let mask = rank_mask(r);
    4 - ((dead & mask) >> (r * 4)).count_ones() as u8
}

/// Convert a card bitmask to a vector of cards.
pub fn mask_to_cards(mask: u64) -> Vec<Card> {
    let mut cards = Vec::new();
    for i in 0..52 {
        if (mask & (1 << i)) != 0 {
            cards.push(i as Card);
        }
    }
    cards
}

/// Convert a slice of cards to a bitmask.
pub fn cards_to_mask(cards: &[Card]) -> u64 {
    let mut mask = 0u64;
    for &c in cards {
        mask |= 1 << c;
    }
    mask
}

/// Format a card for display (e.g., "As", "Kh", "2c").
pub fn card_to_string(card: Card) -> String {
    const RANKS: &[char] = &['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A'];
    const SUITS: &[char] = &['c', 'd', 'h', 's'];
    let r = rank(card) as usize;
    let s = suit(card) as usize;
    format!("{}{}", RANKS[r], SUITS[s])
}

/// Format a canonical hand for display (e.g., "AA", "AKs", "AKo").
pub fn canonical_to_string(canonical: CanonicalHand) -> String {
    const RANKS: &[char] = &['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A'];

    if canonical < 13 {
        // Pair
        let r = canonical as usize;
        format!("{}{}", RANKS[r], RANKS[r])
    } else if canonical < 91 {
        // Suited
        let tri_idx = canonical - 13;
        let (hi, lo) = triangle_from_index(tri_idx);
        format!("{}{}s", RANKS[hi as usize], RANKS[lo as usize])
    } else {
        // Offsuit
        let tri_idx = canonical - 91;
        let (hi, lo) = triangle_from_index(tri_idx);
        format!("{}{}o", RANKS[hi as usize], RANKS[lo as usize])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_canonical_pairs() {
        // AA should be index 12
        let (idx, combos) = to_canonical(make_card(12, 0), make_card(12, 1));
        assert_eq!(idx, 12);
        assert_eq!(combos, 6);

        // 22 should be index 0
        let (idx, combos) = to_canonical(make_card(0, 2), make_card(0, 3));
        assert_eq!(idx, 0);
        assert_eq!(combos, 6);
    }

    #[test]
    fn test_canonical_suited() {
        // AKs should be 13 + triangle_index(12, 11) = 13 + 66 + 11 = 90
        let (idx, combos) = to_canonical(make_card(12, 0), make_card(11, 0));
        assert_eq!(idx, 13 + (12 * 11) / 2 + 11);
        assert_eq!(combos, 4);

        // 32s should be 13 + triangle_index(1, 0) = 13 + 0 = 13
        let (idx, combos) = to_canonical(make_card(1, 2), make_card(0, 2));
        assert_eq!(idx, 13);
        assert_eq!(combos, 4);
    }

    #[test]
    fn test_canonical_offsuit() {
        // AKo should be 91 + triangle_index(12, 11) = 91 + 77 = 168
        let (idx, combos) = to_canonical(make_card(12, 0), make_card(11, 1));
        assert_eq!(idx, 91 + (12 * 11) / 2 + 11);
        assert_eq!(combos, 12);
    }

    #[test]
    fn test_canonical_count() {
        // Verify we have exactly 169 canonical hands
        let mut seen = vec![false; NUM_CANONICAL_HANDS];
        for c1 in 0..52u8 {
            for c2 in (c1 + 1)..52u8 {
                let (idx, _) = to_canonical(c1, c2);
                seen[idx as usize] = true;
            }
        }
        assert!(seen.iter().all(|&x| x));
    }

    #[test]
    fn test_from_canonical_roundtrip() {
        for idx in 0..NUM_CANONICAL_HANDS {
            let hole = from_canonical(idx as CanonicalHand);
            let (back_idx, _) = to_canonical(hole.0, hole.1);
            assert_eq!(back_idx, idx as CanonicalHand);
        }
    }

    #[test]
    fn test_available_combos_no_blockers() {
        // With no dead cards, should get full combos
        assert_eq!(available_combos(0, 0), 6);  // Pair of 2s
        assert_eq!(available_combos(13, 0), 4); // 32s
        assert_eq!(available_combos(91, 0), 12); // 32o
    }
}
