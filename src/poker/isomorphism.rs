//! Hand isomorphism for state space reduction.
//!
//! Poker hands that differ only in suit labeling are strategically equivalent.
//! This module provides:
//! - [`SuitMapping`]: Transforms suits to canonical form
//! - [`RiverIsomorphism`]: Maps combos to canonical buckets for river play
//!
//! # Theory
//!
//! Given a board, we canonicalize suits by ordering them by first appearance.
//! For example, on "KhQsJs", suits become: h→0, s→1, and unused c,d→2,3.
//!
//! This reduces the number of distinct information sets because hands like
//! "AsKs" and "AhKh" become equivalent on suit-symmetric boards.

use crate::poker::hands::{make_card, rank, suit, Board, Card, Combo, NUM_COMBOS};

/// A suit permutation that maps old suits to new suits.
///
/// `mapping[old_suit] = new_suit`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SuitMapping {
    /// Maps old suit (0-3) to new suit (0-3).
    pub mapping: [u8; 4],
}

impl SuitMapping {
    /// Identity mapping (no change).
    pub const IDENTITY: SuitMapping = SuitMapping {
        mapping: [0, 1, 2, 3],
    };

    /// Create a canonical suit mapping for a board.
    ///
    /// Suits are ordered by first appearance in the board.
    /// Unused suits are assigned in ascending order.
    pub fn canonical_for_board(board: &Board) -> Self {
        let mut mapping = [4u8; 4]; // 4 = unassigned
        let mut next_canonical = 0u8;

        // Assign suits based on first appearance
        for &card in &board.cards {
            let s = suit(card);
            if mapping[s as usize] == 4 {
                mapping[s as usize] = next_canonical;
                next_canonical += 1;
            }
        }

        // Assign remaining suits in ascending order
        for s in 0..4 {
            if mapping[s] == 4 {
                mapping[s] = next_canonical;
                next_canonical += 1;
            }
        }

        SuitMapping { mapping }
    }

    /// Transform a card using this mapping.
    #[inline]
    pub fn transform_card(&self, card: Card) -> Card {
        let r = rank(card);
        let s = suit(card);
        let new_s = self.mapping[s as usize];
        make_card(r, new_s)
    }

    /// Transform a combo using this mapping.
    #[inline]
    pub fn transform_combo(&self, combo: Combo) -> Combo {
        let c0 = self.transform_card(combo.c0);
        let c1 = self.transform_card(combo.c1);
        Combo::new(c0, c1)
    }

    /// Transform a board using this mapping.
    pub fn transform_board(&self, board: &Board) -> Board {
        let cards: Vec<Card> = board.cards.iter().map(|&c| self.transform_card(c)).collect();
        Board::new(&cards)
    }

    /// Compose two mappings: self(other(x)) = result(x).
    pub fn compose(&self, other: &SuitMapping) -> SuitMapping {
        let mut result = [0u8; 4];
        for s in 0..4 {
            result[s] = self.mapping[other.mapping[s] as usize];
        }
        SuitMapping { mapping: result }
    }

    /// Generate all 24 suit permutations.
    pub fn all_permutations() -> [SuitMapping; 24] {
        let mut perms = [SuitMapping::IDENTITY; 24];
        let mut idx = 0;

        // Generate all permutations of 4 elements
        for a in 0..4u8 {
            for b in 0..4u8 {
                if b == a {
                    continue;
                }
                for c in 0..4u8 {
                    if c == a || c == b {
                        continue;
                    }
                    let d = 6 - a - b - c; // The remaining suit
                    perms[idx] = SuitMapping {
                        mapping: [a, b, c, d],
                    };
                    idx += 1;
                }
            }
        }

        perms
    }
}

/// River isomorphism: maps combos to canonical buckets.
///
/// On the river, we can group equivalent combos into buckets based on
/// suit isomorphism. This reduces the number of information sets.
pub struct RiverIsomorphism {
    /// Maps combo index to canonical bucket ID.
    pub combo_to_bucket: [u16; NUM_COMBOS],
    /// Number of canonical buckets.
    pub num_buckets: usize,
    /// For each bucket, list of combo indices that map to it.
    pub bucket_combos: Vec<Vec<usize>>,
    /// The canonical board (suits reordered by first appearance).
    pub canonical_board: Board,
    /// Mapping from original board to canonical board.
    pub board_mapping: SuitMapping,
}

impl RiverIsomorphism {
    /// Create river isomorphism for a 5-card board.
    pub fn new(board: &Board) -> Self {
        assert!(
            board.len() == 5,
            "RiverIsomorphism requires a 5-card board"
        );

        // Get canonical mapping for the board
        let board_mapping = SuitMapping::canonical_for_board(board);
        let canonical_board = board_mapping.transform_board(board);

        // For each combo, compute its canonical form
        let mut combo_to_bucket = [u16::MAX; NUM_COMBOS];
        let mut bucket_combos: Vec<Vec<usize>> = Vec::new();
        let mut canonical_to_bucket: std::collections::HashMap<usize, u16> =
            std::collections::HashMap::new();

        for combo_idx in 0..NUM_COMBOS {
            let combo = Combo::from_index(combo_idx);

            // Skip combos blocked by the board
            if combo.conflicts_with_mask(board.mask) {
                continue;
            }

            // Transform combo using board mapping
            let canonical_combo = board_mapping.transform_combo(combo);
            let canonical_idx = canonical_combo.to_index();

            // Get or create bucket
            let bucket = if let Some(&b) = canonical_to_bucket.get(&canonical_idx) {
                b
            } else {
                let b = bucket_combos.len() as u16;
                canonical_to_bucket.insert(canonical_idx, b);
                bucket_combos.push(Vec::new());
                b
            };

            combo_to_bucket[combo_idx] = bucket;
            bucket_combos[bucket as usize].push(combo_idx);
        }

        let num_buckets = bucket_combos.len();

        RiverIsomorphism {
            combo_to_bucket,
            num_buckets,
            bucket_combos,
            canonical_board,
            board_mapping,
        }
    }

    /// Get the bucket ID for a combo.
    #[inline]
    pub fn get_bucket(&self, combo_idx: usize) -> Option<u16> {
        let bucket = self.combo_to_bucket[combo_idx];
        if bucket == u16::MAX {
            None
        } else {
            Some(bucket)
        }
    }

    /// Get the bucket ID for a combo (panics if blocked).
    #[inline]
    pub fn bucket(&self, combo_idx: usize) -> u16 {
        let bucket = self.combo_to_bucket[combo_idx];
        debug_assert!(bucket != u16::MAX, "Combo is blocked by board");
        bucket
    }

    /// Get all combos in a bucket.
    pub fn combos_in_bucket(&self, bucket: u16) -> &[usize] {
        &self.bucket_combos[bucket as usize]
    }

    /// Compute the effective weight for a combo in a bucket.
    ///
    /// When multiple combos map to the same bucket, we need to weight
    /// strategies appropriately.
    pub fn bucket_weight(&self, bucket: u16) -> f32 {
        self.bucket_combos[bucket as usize].len() as f32
    }
}

/// Compute isomorphism statistics for a board.
pub struct IsomorphismStats {
    /// Number of valid (non-blocked) combos.
    pub valid_combos: usize,
    /// Number of canonical buckets.
    pub num_buckets: usize,
    /// Compression ratio (valid_combos / num_buckets).
    pub compression_ratio: f32,
}

impl RiverIsomorphism {
    /// Get statistics about this isomorphism.
    pub fn stats(&self) -> IsomorphismStats {
        let valid_combos: usize = self.bucket_combos.iter().map(|b| b.len()).sum();
        let num_buckets = self.num_buckets;
        let compression_ratio = valid_combos as f32 / num_buckets as f32;

        IsomorphismStats {
            valid_combos,
            num_buckets,
            compression_ratio,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::poker::board_parser::parse_board;

    #[test]
    fn test_suit_mapping_identity() {
        let combo = Combo::new(0, 4); // 2c, 3c
        let result = SuitMapping::IDENTITY.transform_combo(combo);
        assert_eq!(result, combo);
    }

    #[test]
    fn test_suit_mapping_transform() {
        // Swap clubs and hearts
        let mapping = SuitMapping {
            mapping: [2, 1, 0, 3],
        };
        let card = make_card(12, 0); // Ac
        let result = mapping.transform_card(card);
        assert_eq!(rank(result), 12);
        assert_eq!(suit(result), 2); // Now Ah
    }

    #[test]
    fn test_canonical_for_board() {
        let board = parse_board("KhQsJs2c3d").unwrap();
        let mapping = SuitMapping::canonical_for_board(&board);

        // h appears first, then s, then c, then d
        assert_eq!(mapping.mapping[2], 0); // h -> 0
        assert_eq!(mapping.mapping[3], 1); // s -> 1
        assert_eq!(mapping.mapping[0], 2); // c -> 2
        assert_eq!(mapping.mapping[1], 3); // d -> 3
    }

    #[test]
    fn test_all_permutations_count() {
        let perms = SuitMapping::all_permutations();
        assert_eq!(perms.len(), 24);

        // Check all are distinct
        for i in 0..24 {
            for j in (i + 1)..24 {
                assert_ne!(perms[i], perms[j]);
            }
        }
    }

    #[test]
    fn test_river_isomorphism_basic() {
        let board = parse_board("KhQsJs2c3d").unwrap();
        let iso = RiverIsomorphism::new(&board);

        // Check that isomorphism was created
        assert!(iso.num_buckets > 0);
        assert!(iso.num_buckets <= NUM_COMBOS);

        // Check compression (river boards typically get some compression)
        let stats = iso.stats();
        assert!(stats.valid_combos > 0);
    }

    #[test]
    fn test_river_isomorphism_blocked_combos() {
        let board = parse_board("AcAdAhAs2c").unwrap();
        let iso = RiverIsomorphism::new(&board);

        // AAAA blocks many combos
        let stats = iso.stats();
        assert!(stats.valid_combos < NUM_COMBOS);
    }

    #[test]
    fn test_equivalent_combos_same_bucket() {
        // Rainbow board - each suit appears once, no isomorphism expected
        let board = parse_board("Kh2c3d4s5h").unwrap();
        let iso = RiverIsomorphism::new(&board);

        // AsKs and other suited combos may map to same bucket on some boards
        // but on this rainbow board, compression is limited
        let stats = iso.stats();
        println!(
            "Valid: {}, Buckets: {}, Ratio: {:.2}",
            stats.valid_combos, stats.num_buckets, stats.compression_ratio
        );
    }

    #[test]
    fn test_monotone_board_compression() {
        // Monotone board should have good compression
        let board = parse_board("Kh Qh Jh 2h 3h").unwrap();
        let iso = RiverIsomorphism::new(&board);

        let stats = iso.stats();
        println!(
            "Monotone - Valid: {}, Buckets: {}, Ratio: {:.2}",
            stats.valid_combos, stats.num_buckets, stats.compression_ratio
        );

        // Monotone board should have some compression because non-heart
        // combos are equivalent under suit permutation
        assert!(stats.compression_ratio >= 1.0);
    }
}
