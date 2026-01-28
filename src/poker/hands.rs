//! Core poker hand types: Combo, Range, and Board.
//!
//! Cards are encoded as `card_id = 4 * rank + suit` where:
//! - rank: 0 (deuce) to 12 (ace)
//! - suit: 0-3 (clubs, diamonds, hearts, spades)

/// A card encoded as `4 * rank + suit` (0-51).
pub type Card = u8;

/// Number of 2-card combinations from a 52-card deck: C(52,2) = 1326.
pub const NUM_COMBOS: usize = 1326;

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

/// A 2-card combination (hole cards).
///
/// Cards are stored in canonical order: `c0 < c1`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Combo {
    pub c0: Card,
    pub c1: Card,
}

impl Combo {
    /// Create a combo from two cards. Automatically orders them.
    #[inline]
    pub fn new(a: Card, b: Card) -> Self {
        if a < b {
            Combo { c0: a, c1: b }
        } else {
            Combo { c0: b, c1: a }
        }
    }

    /// Convert a combo index (0..1326) to a Combo.
    ///
    /// The index formula: For cards (c0, c1) with c0 < c1,
    /// index = c1 * (c1 - 1) / 2 + c0
    pub fn from_index(index: usize) -> Self {
        debug_assert!(index < NUM_COMBOS);
        // Find c1 such that c1*(c1-1)/2 <= index < (c1+1)*c1/2
        // Use usize to avoid overflow
        let mut c1: usize = 1;
        while (c1 + 1) * c1 / 2 <= index {
            c1 += 1;
        }
        let c0 = index - c1 * (c1 - 1) / 2;
        Combo {
            c0: c0 as u8,
            c1: c1 as u8,
        }
    }

    /// Convert a Combo to its index (0..1326).
    #[inline]
    pub fn to_index(&self) -> usize {
        (self.c1 as usize * (self.c1 as usize - 1) / 2) + self.c0 as usize
    }

    /// Check if this combo conflicts with a card (shares a card).
    #[inline]
    pub fn conflicts_with_card(&self, card: Card) -> bool {
        self.c0 == card || self.c1 == card
    }

    /// Check if this combo conflicts with a card mask.
    #[inline]
    pub fn conflicts_with_mask(&self, mask: u64) -> bool {
        ((mask >> self.c0) & 1) != 0 || ((mask >> self.c1) & 1) != 0
    }

    /// Check if this combo conflicts with another combo.
    #[inline]
    pub fn conflicts_with(&self, other: &Combo) -> bool {
        self.c0 == other.c0 || self.c0 == other.c1 || self.c1 == other.c0 || self.c1 == other.c1
    }

    /// Get the bitmask for this combo.
    #[inline]
    pub fn to_mask(&self) -> u64 {
        (1u64 << self.c0) | (1u64 << self.c1)
    }
}

/// A weighted poker range.
///
/// Stores a weight (0.0-1.0) for each of the 1326 possible 2-card combos.
#[derive(Clone)]
pub struct Range {
    /// Weight for each combo (0.0 = not in range, 1.0 = always in range).
    pub weights: [f32; NUM_COMBOS],
}

impl Default for Range {
    fn default() -> Self {
        Self {
            weights: [0.0; NUM_COMBOS],
        }
    }
}

impl Range {
    /// Create an empty range.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a range with uniform weight for all combos.
    pub fn full(weight: f32) -> Self {
        Self {
            weights: [weight; NUM_COMBOS],
        }
    }

    /// Set the weight for a specific combo.
    #[inline]
    pub fn set(&mut self, combo: Combo, weight: f32) {
        self.weights[combo.to_index()] = weight;
    }

    /// Get the weight for a specific combo.
    #[inline]
    pub fn get(&self, combo: Combo) -> f32 {
        self.weights[combo.to_index()]
    }

    /// Check if a combo is in the range (weight > 0).
    #[inline]
    pub fn contains(&self, combo: Combo) -> bool {
        self.weights[combo.to_index()] > 0.0
    }

    /// Count combos with non-zero weight.
    pub fn count_combos(&self) -> usize {
        self.weights.iter().filter(|&&w| w > 0.0).count()
    }

    /// Sum of all weights.
    pub fn total_weight(&self) -> f32 {
        self.weights.iter().sum()
    }

    /// Get valid combos that don't conflict with the board.
    pub fn valid_combos(&self, board_mask: u64) -> Vec<(usize, f32)> {
        let mut result = Vec::new();
        for (idx, &weight) in self.weights.iter().enumerate() {
            if weight > 0.0 {
                let combo = Combo::from_index(idx);
                if !combo.conflicts_with_mask(board_mask) {
                    result.push((idx, weight));
                }
            }
        }
        result
    }

    /// Normalize weights so they sum to 1.0.
    pub fn normalize(&mut self) {
        let total: f32 = self.weights.iter().sum();
        if total > 0.0 {
            for w in &mut self.weights {
                *w /= total;
            }
        }
    }
}

/// A poker board (community cards).
#[derive(Clone)]
pub struct Board {
    /// The community cards (3-5 cards).
    pub cards: Vec<Card>,
    /// Bitmask of cards on the board (dead cards).
    pub mask: u64,
}

impl Board {
    /// Create a board from a slice of cards.
    pub fn new(cards: &[Card]) -> Self {
        let mask = cards.iter().fold(0u64, |m, &c| m | (1u64 << c));
        Self {
            cards: cards.to_vec(),
            mask,
        }
    }

    /// Create an empty board.
    pub fn empty() -> Self {
        Self {
            cards: Vec::new(),
            mask: 0,
        }
    }

    /// Get the number of cards on the board.
    pub fn len(&self) -> usize {
        self.cards.len()
    }

    /// Check if the board is empty.
    pub fn is_empty(&self) -> bool {
        self.cards.is_empty()
    }

    /// Check if a combo conflicts with the board.
    #[inline]
    pub fn blocks(&self, combo: Combo) -> bool {
        combo.conflicts_with_mask(self.mask)
    }

    /// Add a card to the board.
    pub fn add_card(&mut self, card: Card) {
        self.cards.push(card);
        self.mask |= 1u64 << card;
    }

    /// Get board as 7-card array (for hand evaluation with hole cards).
    pub fn with_combo(&self, combo: Combo) -> [Card; 7] {
        debug_assert!(self.cards.len() == 5, "Board must have 5 cards for showdown");
        [
            self.cards[0],
            self.cards[1],
            self.cards[2],
            self.cards[3],
            self.cards[4],
            combo.c0,
            combo.c1,
        ]
    }
}

/// Format a card for display (e.g., "As", "Kh", "2c").
pub fn card_to_string(card: Card) -> String {
    const RANKS: &[char] = &['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A'];
    const SUITS: &[char] = &['c', 'd', 'h', 's'];
    let r = rank(card) as usize;
    let s = suit(card) as usize;
    format!("{}{}", RANKS[r], SUITS[s])
}

/// Format a combo for display (e.g., "AsKs").
pub fn combo_to_string(combo: Combo) -> String {
    format!("{}{}", card_to_string(combo.c1), card_to_string(combo.c0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_combo_index_roundtrip() {
        for i in 0..NUM_COMBOS {
            let combo = Combo::from_index(i);
            assert!(combo.c0 < combo.c1);
            assert_eq!(combo.to_index(), i);
        }
    }

    #[test]
    fn test_combo_index_bounds() {
        // First combo: (0, 1)
        let first = Combo::from_index(0);
        assert_eq!(first, Combo::new(0, 1));

        // Last combo: (50, 51)
        let last = Combo::from_index(NUM_COMBOS - 1);
        assert_eq!(last, Combo::new(50, 51));
    }

    #[test]
    fn test_combo_count() {
        // Verify we have exactly 1326 unique combos
        let mut count = 0;
        for c1 in 0..52u8 {
            for c0 in 0..c1 {
                let combo = Combo::new(c0, c1);
                assert!(combo.to_index() < NUM_COMBOS);
                count += 1;
            }
        }
        assert_eq!(count, NUM_COMBOS);
    }

    #[test]
    fn test_combo_conflicts() {
        let combo = Combo::new(0, 1);
        assert!(combo.conflicts_with_card(0));
        assert!(combo.conflicts_with_card(1));
        assert!(!combo.conflicts_with_card(2));

        let other = Combo::new(1, 2);
        assert!(combo.conflicts_with(&other)); // Share card 1

        let disjoint = Combo::new(2, 3);
        assert!(!combo.conflicts_with(&disjoint));
    }

    #[test]
    fn test_range_basic() {
        let mut range = Range::new();
        let combo = Combo::new(48, 49); // AsAd

        range.set(combo, 1.0);
        assert_eq!(range.get(combo), 1.0);
        assert!(range.contains(combo));
        assert_eq!(range.count_combos(), 1);
    }

    #[test]
    fn test_board_blocking() {
        let board = Board::new(&[0, 4, 8]); // 2c, 3c, 4c
        let blocked = Combo::new(0, 1); // 2c2d - blocked
        let valid = Combo::new(12, 13); // 5c5d - not blocked

        assert!(board.blocks(blocked));
        assert!(!board.blocks(valid));
    }
}
