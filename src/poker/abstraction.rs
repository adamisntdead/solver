//! Hand abstraction for CFR solving.
//!
//! This module provides a two-layer abstraction system:
//!
//! 1. **Layer 1 - Suit Isomorphism** (lossless): Groups hands that are
//!    strategically equivalent due to suit symmetry.
//!
//! 2. **Layer 2 - Information Abstraction** (lossy, future): Groups hands
//!    by strategic similarity (EHS, EMD, etc.) - currently a passthrough.
//!
//! # Architecture
//!
//! ```text
//! combo_idx → [Layer 1: Suit Isomorphism] → iso_bucket
//!           → [Layer 2: Info Abstraction (passthrough)] → final_bucket
//! ```
//!
//! The [`HandAbstraction`] trait provides a unified interface for both layers.
//! The [`SuitIsomorphism`] struct implements only Layer 1 (lossless).
//! The [`ComposedAbstraction`] struct chains both layers.
//!
//! # Usage
//!
//! ```ignore
//! use solver::poker::abstraction::{SuitIsomorphism, HandAbstraction};
//! use solver::poker::Board;
//!
//! let board = Board::from_str("KhQsJs")?;
//! let abstraction = SuitIsomorphism::new(&board);
//!
//! // Get bucket for a combo in a given context
//! if let Some(bucket) = abstraction.bucket(combo_idx, context) {
//!     // Use bucket for regret storage
//! }
//! ```

use crate::poker::hands::Board;
use crate::poker::isomorphism::{BoardIsomorphism, INVALID_BUCKET};

/// Trait for hand abstraction (isomorphism + optional bucketing).
///
/// Provides a unified interface for mapping hands to buckets for CFR solving.
/// Implementations can be lossless (suit isomorphism only) or lossy
/// (with additional information abstraction).
pub trait HandAbstraction: Send + Sync {
    /// Map combo to final bucket for a given card context.
    ///
    /// Returns None if the combo is blocked by the board cards.
    fn bucket(&self, combo_idx: usize, context: usize) -> Option<u16>;

    /// Number of buckets for a given context.
    fn num_buckets(&self, context: usize) -> usize;

    /// Number of card contexts supported.
    fn num_contexts(&self) -> usize;

    /// Aggregate hand reaches to bucket reaches.
    ///
    /// Converts per-combo reach values to per-bucket by summing all combos
    /// that map to each bucket.
    fn hands_to_buckets(&self, reaches: &[f32], context: usize) -> Vec<f32>;

    /// Expand bucket values to hand values.
    ///
    /// Converts per-bucket values back to per-combo. Each combo receives
    /// the value of its bucket.
    fn buckets_to_hands(&self, values: &[f32], context: usize) -> Vec<f32>;

    /// Get the bucket size (number of combos) for a bucket in a context.
    fn bucket_size(&self, bucket: u16, context: usize) -> u16;
}

/// Suit isomorphism only (no further bucketing).
///
/// This is the default abstraction - lossless compression that groups hands
/// that differ only by suit labeling. Provides the [`HandAbstraction`] trait
/// interface over [`BoardIsomorphism`] tables.
///
/// # Multi-Street Support
///
/// For multi-street solving, maintains separate isomorphism tables per card
/// context (e.g., per river card for turn solving, per turn+river for flop).
pub struct SuitIsomorphism {
    /// Isomorphism table per card context.
    iso_tables: Vec<BoardIsomorphism>,
}

impl SuitIsomorphism {
    /// Create suit isomorphism for a single context (e.g., river-only).
    pub fn new(board: &Board) -> Self {
        let iso = BoardIsomorphism::new(board);
        SuitIsomorphism {
            iso_tables: vec![iso],
        }
    }

    /// Create suit isomorphism with multiple contexts.
    ///
    /// Each context gets its own isomorphism table computed from the
    /// corresponding board state.
    pub fn with_contexts(boards: Vec<Board>) -> Self {
        let iso_tables = boards.iter().map(BoardIsomorphism::new).collect();
        SuitIsomorphism { iso_tables }
    }

    /// Create suit isomorphism for turn → river solving.
    ///
    /// Creates one context per valid river card (48 contexts).
    pub fn for_turn_board(board: &Board, valid_river_cards: &[u8]) -> Self {
        let iso_tables = valid_river_cards
            .iter()
            .map(|&rc| {
                let mut cards = board.cards.clone();
                cards.push(rc);
                let river_board = Board::new(&cards);
                BoardIsomorphism::new(&river_board)
            })
            .collect();
        SuitIsomorphism { iso_tables }
    }

    /// Create suit isomorphism for flop → turn → river solving.
    ///
    /// Creates one context per (turn, river) pair with composite indexing:
    /// `context = turn_idx * num_river + river_idx`
    pub fn for_flop_board(board: &Board, valid_cards: &[u8]) -> Self {
        let num_cards = valid_cards.len();
        let mut iso_tables = Vec::with_capacity(num_cards * num_cards);

        for &tc in valid_cards.iter() {
            for &rc in valid_cards.iter() {
                if tc == rc {
                    // Skip diagonal - will be skipped at runtime
                    // Use the first valid river as placeholder
                    let first_valid = valid_cards.iter().find(|&&c| c != tc).unwrap();
                    let mut cards = board.cards.clone();
                    cards.push(tc);
                    cards.push(*first_valid);
                    let five_card = Board::new(&cards);
                    iso_tables.push(BoardIsomorphism::new(&five_card));
                } else {
                    let mut cards = board.cards.clone();
                    cards.push(tc);
                    cards.push(rc);
                    let five_card = Board::new(&cards);
                    iso_tables.push(BoardIsomorphism::new(&five_card));
                }
            }
        }

        SuitIsomorphism { iso_tables }
    }

    /// Get the underlying isomorphism for a context.
    pub fn get_iso(&self, context: usize) -> &BoardIsomorphism {
        &self.iso_tables[context]
    }
}

impl HandAbstraction for SuitIsomorphism {
    fn bucket(&self, combo_idx: usize, context: usize) -> Option<u16> {
        let iso = &self.iso_tables[context];
        let bucket = iso.combo_to_bucket[combo_idx];
        if bucket == INVALID_BUCKET {
            None
        } else {
            Some(bucket)
        }
    }

    fn num_buckets(&self, context: usize) -> usize {
        self.iso_tables[context].num_buckets
    }

    fn num_contexts(&self) -> usize {
        self.iso_tables.len()
    }

    fn hands_to_buckets(&self, reaches: &[f32], context: usize) -> Vec<f32> {
        self.iso_tables[context].aggregate_reaches(reaches)
    }

    fn buckets_to_hands(&self, values: &[f32], context: usize) -> Vec<f32> {
        self.iso_tables[context].expand_to_combos(values)
    }

    fn bucket_size(&self, bucket: u16, context: usize) -> u16 {
        self.iso_tables[context].bucket_size(bucket)
    }
}

/// Composed abstraction: suit isomorphism + optional info abstraction.
///
/// Chains two layers of abstraction:
/// 1. Suit isomorphism (lossless) - always applied
/// 2. Info abstraction (lossy) - optional, currently passthrough
///
/// # Future Work
///
/// When implementing lossy bucketing (EHS, EMD, etc.):
/// 1. Create [`InfoAbstraction`] implementations (EHSAbstraction, EMDAbstraction)
/// 2. Add iso_to_bucket mapping after suit isomorphism
/// 3. Update bucket() to chain: combo → iso_bucket → final_bucket
pub struct ComposedAbstraction {
    /// Layer 1: Suit isomorphism (always applied).
    isomorphism: SuitIsomorphism,

    /// Layer 2: Info abstraction (None = passthrough/identity).
    ///
    /// TODO: Replace with actual abstraction when implementing EHS/EMD.
    #[allow(dead_code)]
    info_abstraction: Option<Box<dyn InfoAbstraction>>,
}

impl ComposedAbstraction {
    /// Create with suit isomorphism only (no further bucketing).
    ///
    /// This is the lossless configuration - hands are only grouped by
    /// suit symmetry.
    pub fn suit_iso_only(iso: SuitIsomorphism) -> Self {
        Self {
            isomorphism: iso,
            info_abstraction: None, // Passthrough
        }
    }

    /// Create from a board with single context.
    pub fn new(board: &Board) -> Self {
        Self::suit_iso_only(SuitIsomorphism::new(board))
    }

    /// Create for turn → river solving.
    pub fn for_turn(board: &Board, valid_river_cards: &[u8]) -> Self {
        Self::suit_iso_only(SuitIsomorphism::for_turn_board(board, valid_river_cards))
    }

    /// Create for flop → turn → river solving.
    pub fn for_flop(board: &Board, valid_cards: &[u8]) -> Self {
        Self::suit_iso_only(SuitIsomorphism::for_flop_board(board, valid_cards))
    }

    /// Get the underlying suit isomorphism.
    pub fn isomorphism(&self) -> &SuitIsomorphism {
        &self.isomorphism
    }

    // TODO: Create with additional info abstraction
    // pub fn with_info_abstraction(
    //     iso: SuitIsomorphism,
    //     abs: impl InfoAbstraction + 'static,
    // ) -> Self {
    //     Self {
    //         isomorphism: iso,
    //         info_abstraction: Some(Box::new(abs)),
    //     }
    // }
}

impl HandAbstraction for ComposedAbstraction {
    fn bucket(&self, combo_idx: usize, context: usize) -> Option<u16> {
        // Layer 1: combo → isomorphic bucket
        let iso_bucket = self.isomorphism.bucket(combo_idx, context)?;

        // Layer 2: iso_bucket → final bucket (passthrough for now)
        //
        // TODO: When info_abstraction is Some, apply additional bucketing:
        // match &self.info_abstraction {
        //     Some(abs) => Some(abs.bucket(iso_bucket as usize, context) as u16),
        //     None => Some(iso_bucket),
        // }
        Some(iso_bucket)
    }

    fn num_buckets(&self, context: usize) -> usize {
        // With info abstraction, this would return abs.num_buckets()
        self.isomorphism.num_buckets(context)
    }

    fn num_contexts(&self) -> usize {
        self.isomorphism.num_contexts()
    }

    fn hands_to_buckets(&self, reaches: &[f32], context: usize) -> Vec<f32> {
        // With info abstraction, would need additional aggregation step
        self.isomorphism.hands_to_buckets(reaches, context)
    }

    fn buckets_to_hands(&self, values: &[f32], context: usize) -> Vec<f32> {
        // With info abstraction, would need additional expansion step
        self.isomorphism.buckets_to_hands(values, context)
    }

    fn bucket_size(&self, bucket: u16, context: usize) -> u16 {
        self.isomorphism.bucket_size(bucket, context)
    }
}

/// Placeholder trait for future info abstraction implementations.
///
/// Info abstractions group strategically similar hands (from suit isomorphism)
/// into a smaller number of buckets based on equity distributions.
///
/// # Planned Implementations
///
/// - **EHSAbstraction**: Group by Expected Hand Strength (single equity value)
/// - **EMDAbstraction**: Group by equity distribution histogram (Earth Mover's Distance)
/// - **AggSIAbstraction**: Flush-aware suit isomorphism
///
/// # File Format
///
/// When implementing file-based abstractions:
///
/// ```text
/// Header:
///   - magic: u32 (0x41425354 = "ABST")
///   - version: u32
///   - street: u8 (0=flop, 1=turn, 2=river)
///   - num_contexts: u32
///   - num_iso_hands: u32
///   - num_buckets: u32
///
/// Per context:
///   - iso_to_bucket: [u16; num_iso_hands]
///
/// Compression: zstd recommended
/// ```
pub trait InfoAbstraction: Send + Sync {
    /// Map isomorphic hand to abstraction bucket.
    fn bucket(&self, iso_hand: usize, context: usize) -> usize;

    /// Number of abstraction buckets.
    fn num_buckets(&self) -> usize;

    // TODO: Load from precomputed file
    // fn load(path: &Path) -> io::Result<Self> where Self: Sized;

    // TODO: Save to file
    // fn save(&self, path: &Path) -> io::Result<()>;
}

/// Identity abstraction (passthrough) - useful for testing.
///
/// Maps each isomorphic hand to itself (no additional bucketing).
pub struct IdentityAbstraction {
    num_hands: usize,
}

impl IdentityAbstraction {
    /// Create identity abstraction for the given number of isomorphic hands.
    pub fn new(num_hands: usize) -> Self {
        Self { num_hands }
    }
}

impl InfoAbstraction for IdentityAbstraction {
    fn bucket(&self, iso_hand: usize, _context: usize) -> usize {
        iso_hand
    }

    fn num_buckets(&self) -> usize {
        self.num_hands
    }
}

// ============================================================================
// Future EHS/EMD abstraction stubs
// ============================================================================

/// Expected Hand Strength abstraction (stub).
///
/// Groups hands by their expected equity against a random opponent hand.
/// Uses k-means clustering on EHS values.
///
/// TODO: Implement when adding lossy abstraction support.
#[allow(dead_code)]
pub struct EHSAbstraction {
    /// Number of buckets.
    num_buckets: usize,
    /// Maps iso_hand → bucket for each context.
    /// iso_to_bucket[context][iso_hand] = bucket
    iso_to_bucket: Vec<Vec<u16>>,
}

#[allow(dead_code)]
impl EHSAbstraction {
    /// Create EHS abstraction with the given number of buckets.
    ///
    /// TODO: Implement k-means clustering on EHS values.
    pub fn new(_num_buckets: usize, _contexts: usize, _iso_hands_per_context: &[usize]) -> Self {
        unimplemented!("EHS abstraction not yet implemented")
    }

    // TODO: Load from precomputed file
    // pub fn load(path: &Path) -> io::Result<Self> {
    //     unimplemented!("EHS abstraction loading not yet implemented")
    // }

    // TODO: Save to file
    // pub fn save(&self, path: &Path) -> io::Result<()> {
    //     unimplemented!("EHS abstraction saving not yet implemented")
    // }
}

impl InfoAbstraction for EHSAbstraction {
    fn bucket(&self, iso_hand: usize, context: usize) -> usize {
        self.iso_to_bucket[context][iso_hand] as usize
    }

    fn num_buckets(&self) -> usize {
        self.num_buckets
    }
}

/// Earth Mover's Distance abstraction (stub).
///
/// Groups hands by the shape of their equity distribution (histogram).
/// Uses k-means clustering with EMD metric on 50-bin histograms.
///
/// TODO: Implement when adding lossy abstraction support.
#[allow(dead_code)]
pub struct EMDAbstraction {
    /// Number of buckets.
    num_buckets: usize,
    /// Maps iso_hand → bucket for each context.
    iso_to_bucket: Vec<Vec<u16>>,
}

impl InfoAbstraction for EMDAbstraction {
    fn bucket(&self, iso_hand: usize, context: usize) -> usize {
        self.iso_to_bucket[context][iso_hand] as usize
    }

    fn num_buckets(&self) -> usize {
        self.num_buckets
    }
}

// ============================================================================
// Helper functions for abstraction construction
// ============================================================================

/// Create abstraction for the given board and starting street.
///
/// Automatically selects the appropriate abstraction type based on board length:
/// - 5 cards (river): Single context
/// - 4 cards (turn): 48 contexts (per river card)
/// - 3 cards (flop): 49×49 contexts (per turn×river)
pub fn create_abstraction(board: &Board, valid_cards: &[u8]) -> ComposedAbstraction {
    match board.len() {
        5 => ComposedAbstraction::new(board),
        4 => ComposedAbstraction::for_turn(board, valid_cards),
        3 => ComposedAbstraction::for_flop(board, valid_cards),
        n => panic!("Invalid board length: {}", n),
    }
}

/// Compute valid non-board cards.
pub fn compute_valid_cards(board: &Board) -> Vec<u8> {
    (0..52u8)
        .filter(|&card| (board.mask >> card) & 1 == 0)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::poker::board_parser::parse_board;
    use crate::poker::hands::NUM_COMBOS;

    #[test]
    fn test_suit_isomorphism_river() {
        let board = parse_board("KhQsJs2c3d").unwrap();
        let iso = SuitIsomorphism::new(&board);

        assert_eq!(iso.num_contexts(), 1);
        assert!(iso.num_buckets(0) > 0);
        assert!(iso.num_buckets(0) <= NUM_COMBOS);

        // Test bucket retrieval
        let mut valid_count = 0;
        for combo_idx in 0..NUM_COMBOS {
            if iso.bucket(combo_idx, 0).is_some() {
                valid_count += 1;
            }
        }
        assert!(valid_count > 0);
    }

    #[test]
    fn test_suit_isomorphism_turn() {
        let board = parse_board("KhQsJs2c").unwrap();
        let valid_cards = compute_valid_cards(&board);
        let iso = SuitIsomorphism::for_turn_board(&board, &valid_cards);

        assert_eq!(iso.num_contexts(), 48); // 52 - 4 board cards
        assert!(iso.num_buckets(0) > 0);
    }

    #[test]
    fn test_composed_abstraction_passthrough() {
        let board = parse_board("KhQsJs2c3d").unwrap();
        let composed = ComposedAbstraction::new(&board);

        // Should behave identically to SuitIsomorphism
        let iso = SuitIsomorphism::new(&board);

        for combo_idx in 0..NUM_COMBOS {
            assert_eq!(composed.bucket(combo_idx, 0), iso.bucket(combo_idx, 0));
        }
        assert_eq!(composed.num_buckets(0), iso.num_buckets(0));
    }

    #[test]
    fn test_hands_to_buckets_round_trip() {
        let board = parse_board("KhQsJs2c3d").unwrap();
        let iso = SuitIsomorphism::new(&board);

        // Create test reaches
        let mut reaches = vec![0.0f32; NUM_COMBOS];
        for combo_idx in 0..NUM_COMBOS {
            if iso.bucket(combo_idx, 0).is_some() {
                reaches[combo_idx] = 1.0;
            }
        }

        // Convert to buckets and back
        let bucket_reaches = iso.hands_to_buckets(&reaches, 0);
        let expanded = iso.buckets_to_hands(&bucket_reaches, 0);

        // Verify expanded values match bucket values
        for combo_idx in 0..NUM_COMBOS {
            if let Some(bucket) = iso.bucket(combo_idx, 0) {
                assert_eq!(
                    expanded[combo_idx],
                    bucket_reaches[bucket as usize],
                    "Mismatch at combo {}",
                    combo_idx
                );
            }
        }
    }

    #[test]
    fn test_identity_abstraction() {
        let identity = IdentityAbstraction::new(100);
        assert_eq!(identity.num_buckets(), 100);
        for i in 0..100 {
            assert_eq!(identity.bucket(i, 0), i);
        }
    }

    #[test]
    fn test_create_abstraction_helper() {
        let river_board = parse_board("KhQsJs2c3d").unwrap();
        let abs = create_abstraction(&river_board, &[]);
        assert_eq!(abs.num_contexts(), 1);

        let turn_board = parse_board("KhQsJs2c").unwrap();
        let valid_cards = compute_valid_cards(&turn_board);
        let abs = create_abstraction(&turn_board, &valid_cards);
        assert_eq!(abs.num_contexts(), 48);

        let flop_board = parse_board("KhQsJs").unwrap();
        let valid_cards = compute_valid_cards(&flop_board);
        let abs = create_abstraction(&flop_board, &valid_cards);
        assert_eq!(abs.num_contexts(), 49 * 49); // turn × river
    }
}
