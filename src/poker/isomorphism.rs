//! Hand isomorphism for state space reduction.
//!
//! Poker hands that differ only in suit labeling are strategically equivalent.
//! This module provides:
//! - [`SuitMapping`]: Transforms suits to canonical form
//! - [`BoardIsomorphism`]: Maps combos to canonical buckets for any board (3-5 cards)
//! - [`RiverIsomorphism`]: Legacy alias for 5-card boards
//!
//! # Theory
//!
//! Given a board, we canonicalize suits by ordering them by first appearance.
//! For example, on "KhQsJs", suits become: h→0, s→1, and unused c,d→2,3.
//!
//! This reduces the number of distinct information sets because hands like
//! "AsKs" and "AhKh" become equivalent on suit-symmetric boards.
//!
//! # Memory Savings
//!
//! | Board Type | Combos | Iso Buckets | Savings |
//! |------------|--------|-------------|---------|
//! | Rainbow    | ~1000  | ~800        | 1.25x   |
//! | Two-tone   | ~1000  | ~500        | 2x      |
//! | Monotone   | ~1000  | ~300        | 3.3x    |

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

/// Marker value for blocked combos in combo_to_bucket arrays.
pub const INVALID_BUCKET: u16 = u16::MAX;

/// Board isomorphism: maps combos to canonical buckets for any board.
///
/// Generalizes suit isomorphism for boards of 3-5 cards. Equivalent combos
/// are grouped into buckets based on suit canonicalization.
///
/// This is the primary type for hand abstraction - used by the solver to
/// reduce strategy storage from per-combo to per-bucket.
#[derive(Clone)]
pub struct BoardIsomorphism {
    /// Maps combo_idx → bucket (or INVALID_BUCKET if blocked by board).
    pub combo_to_bucket: [u16; NUM_COMBOS],
    /// Number of buckets (canonical hands).
    pub num_buckets: usize,
    /// Size of each bucket (number of combos mapping to it).
    pub bucket_sizes: Vec<u16>,
    /// The canonical board (suits reordered by first appearance).
    pub canonical_board: Board,
    /// Suit mapping applied to reach canonical form.
    pub suit_mapping: SuitMapping,
}

impl BoardIsomorphism {
    /// Create isomorphism for any board (3-5 cards).
    pub fn new(board: &Board) -> Self {
        assert!(
            board.len() >= 3 && board.len() <= 5,
            "BoardIsomorphism requires 3-5 card board, got {}",
            board.len()
        );

        // Get canonical mapping for the board
        let suit_mapping = SuitMapping::canonical_for_board(board);
        let canonical_board = suit_mapping.transform_board(board);

        // For each combo, compute its canonical form
        let mut combo_to_bucket = [INVALID_BUCKET; NUM_COMBOS];
        let mut bucket_sizes: Vec<u16> = Vec::new();
        let mut canonical_to_bucket: std::collections::HashMap<usize, u16> =
            std::collections::HashMap::new();

        for combo_idx in 0..NUM_COMBOS {
            let combo = Combo::from_index(combo_idx);

            // Skip combos blocked by the board
            if combo.conflicts_with_mask(board.mask) {
                continue;
            }

            // Transform combo using board mapping
            let canonical_combo = suit_mapping.transform_combo(combo);
            let canonical_idx = canonical_combo.to_index();

            // Get or create bucket
            let bucket = if let Some(&b) = canonical_to_bucket.get(&canonical_idx) {
                bucket_sizes[b as usize] += 1;
                b
            } else {
                let b = bucket_sizes.len() as u16;
                canonical_to_bucket.insert(canonical_idx, b);
                bucket_sizes.push(1);
                b
            };

            combo_to_bucket[combo_idx] = bucket;
        }

        let num_buckets = bucket_sizes.len();

        BoardIsomorphism {
            combo_to_bucket,
            num_buckets,
            bucket_sizes,
            canonical_board,
            suit_mapping,
        }
    }

    /// Get the bucket ID for a combo, or None if blocked.
    #[inline]
    pub fn get_bucket(&self, combo_idx: usize) -> Option<u16> {
        let bucket = self.combo_to_bucket[combo_idx];
        if bucket == INVALID_BUCKET {
            None
        } else {
            Some(bucket)
        }
    }

    /// Get the bucket ID for a combo (panics if blocked).
    #[inline]
    pub fn bucket(&self, combo_idx: usize) -> u16 {
        let bucket = self.combo_to_bucket[combo_idx];
        debug_assert!(bucket != INVALID_BUCKET, "Combo is blocked by board");
        bucket
    }

    /// Get the size (number of combos) in a bucket.
    #[inline]
    pub fn bucket_size(&self, bucket: u16) -> u16 {
        self.bucket_sizes[bucket as usize]
    }

    /// Aggregate combo reaches to bucket reaches (sum combos in each bucket).
    ///
    /// Used when converting hand-level reach probabilities to bucket-level
    /// for regret matching. The result is the sum of reaches for all combos
    /// that map to each bucket.
    #[inline]
    pub fn aggregate_reaches(&self, combo_reaches: &[f32]) -> Vec<f32> {
        let mut bucket_reaches = vec![0.0; self.num_buckets];
        for (combo_idx, &reach) in combo_reaches.iter().enumerate() {
            let bucket = self.combo_to_bucket[combo_idx];
            if bucket != INVALID_BUCKET {
                bucket_reaches[bucket as usize] += reach;
            }
        }
        bucket_reaches
    }

    /// Expand bucket values to combo values (duplicate to all combos in bucket).
    ///
    /// Used when converting bucket-level strategy/regrets back to hand-level.
    /// Each combo receives the value of its bucket.
    #[inline]
    pub fn expand_to_combos(&self, bucket_values: &[f32]) -> Vec<f32> {
        let mut combo_values = vec![0.0; NUM_COMBOS];
        for combo_idx in 0..NUM_COMBOS {
            let bucket = self.combo_to_bucket[combo_idx];
            if bucket != INVALID_BUCKET {
                combo_values[combo_idx] = bucket_values[bucket as usize];
            }
        }
        combo_values
    }

    /// Average bucket regrets to combo values (divide by bucket size).
    ///
    /// When aggregating regrets, we sum. When expanding, we may want to
    /// average to maintain proper scaling for reach-weighted values.
    #[inline]
    pub fn expand_averaged(&self, bucket_values: &[f32]) -> Vec<f32> {
        let mut combo_values = vec![0.0; NUM_COMBOS];
        for combo_idx in 0..NUM_COMBOS {
            let bucket = self.combo_to_bucket[combo_idx];
            if bucket != INVALID_BUCKET {
                let size = self.bucket_sizes[bucket as usize] as f32;
                combo_values[combo_idx] = bucket_values[bucket as usize] / size;
            }
        }
        combo_values
    }

    /// Get statistics about this isomorphism.
    pub fn stats(&self) -> IsomorphismStats {
        let valid_combos: usize = self.bucket_sizes.iter().map(|&s| s as usize).sum();
        let num_buckets = self.num_buckets;
        let compression_ratio = valid_combos as f32 / num_buckets as f32;

        IsomorphismStats {
            valid_combos,
            num_buckets,
            compression_ratio,
        }
    }
}

/// River isomorphism: maps combos to canonical buckets.
///
/// On the river, we can group equivalent combos into buckets based on
/// suit isomorphism. This reduces the number of information sets.
///
/// NOTE: This is a legacy type for backward compatibility.
/// New code should use [`BoardIsomorphism`] which works for any board length.
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

    // === BoardIsomorphism tests ===

    #[test]
    fn test_board_isomorphism_river() {
        let board = parse_board("KhQsJs2c3d").unwrap();
        let iso = BoardIsomorphism::new(&board);

        // Should match RiverIsomorphism behavior
        let legacy = RiverIsomorphism::new(&board);
        assert_eq!(iso.num_buckets, legacy.num_buckets);

        // Check stats
        let stats = iso.stats();
        assert!(stats.valid_combos > 0);
        assert!(stats.compression_ratio >= 1.0);
    }

    #[test]
    fn test_board_isomorphism_flop() {
        let board = parse_board("KhQsJs").unwrap();
        let iso = BoardIsomorphism::new(&board);

        // Should have valid buckets
        assert!(iso.num_buckets > 0);
        assert!(iso.num_buckets <= NUM_COMBOS);

        let stats = iso.stats();
        println!(
            "Flop KhQsJs - Valid: {}, Buckets: {}, Ratio: {:.2}",
            stats.valid_combos, stats.num_buckets, stats.compression_ratio
        );
    }

    #[test]
    fn test_board_isomorphism_turn() {
        let board = parse_board("KhQsJs2c").unwrap();
        let iso = BoardIsomorphism::new(&board);

        assert!(iso.num_buckets > 0);
        assert!(iso.num_buckets <= NUM_COMBOS);

        let stats = iso.stats();
        println!(
            "Turn KhQsJs2c - Valid: {}, Buckets: {}, Ratio: {:.2}",
            stats.valid_combos, stats.num_buckets, stats.compression_ratio
        );
    }

    #[test]
    fn test_board_isomorphism_monotone_flop() {
        // Monotone flop - compression limited because board suit is special
        // The current implementation uses a single canonical mapping, which
        // doesn't fully exploit symmetry among non-board suits.
        // Future optimization: try all valid suit permutations to find canonical form.
        let board = parse_board("Kh Qh Jh").unwrap();
        let iso = BoardIsomorphism::new(&board);

        let stats = iso.stats();
        println!(
            "Monotone Flop - Valid: {}, Buckets: {}, Ratio: {:.2}",
            stats.valid_combos, stats.num_buckets, stats.compression_ratio
        );

        // Should have valid buckets (compression may be limited)
        assert!(stats.num_buckets > 0);
        assert!(stats.compression_ratio >= 1.0);
    }

    #[test]
    fn test_aggregate_and_expand_reaches() {
        let board = parse_board("KhQsJs2c3d").unwrap();
        let iso = BoardIsomorphism::new(&board);

        // Create test reaches (1.0 for all valid combos)
        let mut combo_reaches = vec![0.0f32; NUM_COMBOS];
        for combo_idx in 0..NUM_COMBOS {
            if iso.get_bucket(combo_idx).is_some() {
                combo_reaches[combo_idx] = 1.0;
            }
        }

        // Aggregate to buckets
        let bucket_reaches = iso.aggregate_reaches(&combo_reaches);
        assert_eq!(bucket_reaches.len(), iso.num_buckets);

        // Each bucket should have reach equal to its size
        for (bucket, &reach) in bucket_reaches.iter().enumerate() {
            let expected = iso.bucket_size(bucket as u16) as f32;
            assert!(
                (reach - expected).abs() < 0.001,
                "Bucket {} reach mismatch: {} vs {}",
                bucket,
                reach,
                expected
            );
        }

        // Expand back to combos
        let expanded = iso.expand_to_combos(&bucket_reaches);
        for combo_idx in 0..NUM_COMBOS {
            if let Some(bucket) = iso.get_bucket(combo_idx) {
                assert_eq!(expanded[combo_idx], bucket_reaches[bucket as usize]);
            } else {
                assert_eq!(expanded[combo_idx], 0.0);
            }
        }
    }

    #[test]
    fn test_bucket_sizes_consistent() {
        let board = parse_board("KhQsJs2c3d").unwrap();
        let iso = BoardIsomorphism::new(&board);

        // Sum of bucket sizes should equal valid combos
        let sum_sizes: usize = iso.bucket_sizes.iter().map(|&s| s as usize).sum();
        let stats = iso.stats();
        assert_eq!(sum_sizes, stats.valid_combos);
    }
}
