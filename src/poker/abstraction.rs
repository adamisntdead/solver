//! Hand abstraction for CFR solving.
//!
//! This module provides a two-layer abstraction system:
//!
//! 1. **Layer 1 - Suit Isomorphism** (lossless): Groups hands that are
//!    strategically equivalent due to suit symmetry.
//!
//! 2. **Layer 2 - Information Abstraction** (lossy): Groups hands
//!    by strategic similarity (EHS, EMD, etc.).
//!
//! # Architecture
//!
//! ```text
//! combo_idx → [Layer 1: Suit Isomorphism] → iso_bucket
//!           → [Layer 2: Info Abstraction] → final_bucket
//! ```
//!
//! The [`HandAbstraction`] trait provides a unified interface for both layers.
//! The [`SuitIsomorphism`] struct implements only Layer 1 (lossless).
//! The [`ComposedAbstraction`] struct chains both layers.
//!
//! # Abstraction Types
//!
//! | Type | Features | Use Case |
//! |------|----------|----------|
//! | SuitIsomorph | Lossless, 0D | Base layer (always applied) |
//! | EHS | Single equity value, 1D | Simple clustering |
//! | EMD | 50-bin equity histogram, 50D | Best quality for flop/turn |
//! | AsymEMD | Non-uniform bins, 50D | Polarized ranges |
//! | WinSplit | Win + split frequencies, 2D | River only |
//! | AggSI | Flush potential features | Additional grouping |
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

use crate::poker::hands::{rank, suit, Board, Combo, NUM_COMBOS};
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
/// 2. Info abstraction (lossy) - optional, groups strategically similar hands
///
/// The info abstraction layer maps isomorphic hand buckets to a smaller number
/// of strategic buckets based on equity distributions (EHS, EMD, WinSplit, etc.).
pub struct ComposedAbstraction {
    /// Layer 1: Suit isomorphism (always applied).
    isomorphism: SuitIsomorphism,

    /// Layer 2: Info abstraction (None = passthrough/identity).
    info_abstraction: Option<Box<dyn InfoAbstraction>>,

    /// Number of buckets per context when info abstraction is active.
    /// Used for efficient bucket count lookup.
    info_num_buckets: Option<usize>,
}

impl ComposedAbstraction {
    /// Create with suit isomorphism only (no further bucketing).
    ///
    /// This is the lossless configuration - hands are only grouped by
    /// suit symmetry.
    pub fn suit_iso_only(iso: SuitIsomorphism) -> Self {
        Self {
            isomorphism: iso,
            info_abstraction: None,
            info_num_buckets: None,
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

    /// Create with suit isomorphism and an info abstraction layer.
    ///
    /// The info abstraction maps isomorphic hand indices to a smaller number
    /// of strategic buckets.
    pub fn with_info_abstraction(
        iso: SuitIsomorphism,
        abs: Box<dyn InfoAbstraction>,
    ) -> Self {
        let num_buckets = abs.num_buckets();
        Self {
            isomorphism: iso,
            info_abstraction: Some(abs),
            info_num_buckets: Some(num_buckets),
        }
    }

    /// Get the underlying suit isomorphism.
    pub fn isomorphism(&self) -> &SuitIsomorphism {
        &self.isomorphism
    }

    /// Check if this abstraction has an info abstraction layer.
    pub fn has_info_abstraction(&self) -> bool {
        self.info_abstraction.is_some()
    }

    /// Get the info abstraction layer (if any).
    pub fn info_abstraction(&self) -> Option<&dyn InfoAbstraction> {
        self.info_abstraction.as_ref().map(|a| a.as_ref())
    }
}

impl HandAbstraction for ComposedAbstraction {
    fn bucket(&self, combo_idx: usize, context: usize) -> Option<u16> {
        // Layer 1: combo → isomorphic bucket
        let iso_bucket = self.isomorphism.bucket(combo_idx, context)?;

        // Layer 2: iso_bucket → final bucket (if info abstraction is present)
        match &self.info_abstraction {
            Some(abs) => {
                // Map iso bucket to info abstraction bucket
                let final_bucket = abs.bucket(iso_bucket as usize, context);
                Some(final_bucket as u16)
            }
            None => Some(iso_bucket),
        }
    }

    fn num_buckets(&self, context: usize) -> usize {
        match &self.info_abstraction {
            Some(abs) => abs.num_buckets(),
            None => self.isomorphism.num_buckets(context),
        }
    }

    fn num_contexts(&self) -> usize {
        self.isomorphism.num_contexts()
    }

    fn hands_to_buckets(&self, reaches: &[f32], context: usize) -> Vec<f32> {
        match &self.info_abstraction {
            Some(abs) => {
                // First aggregate to iso buckets, then to info buckets
                let iso_reaches = self.isomorphism.hands_to_buckets(reaches, context);

                // Then aggregate iso reaches to info buckets
                let num_info_buckets = abs.num_buckets();
                let mut info_reaches = vec![0.0f32; num_info_buckets];

                for (iso_bucket, &reach) in iso_reaches.iter().enumerate() {
                    if reach > 0.0 {
                        let info_bucket = abs.bucket(iso_bucket, context);
                        info_reaches[info_bucket] += reach;
                    }
                }

                info_reaches
            }
            None => self.isomorphism.hands_to_buckets(reaches, context),
        }
    }

    fn buckets_to_hands(&self, values: &[f32], context: usize) -> Vec<f32> {
        match &self.info_abstraction {
            Some(abs) => {
                // Map info bucket values to iso bucket values
                let num_iso_buckets = self.isomorphism.num_buckets(context);
                let mut iso_values = vec![0.0f32; num_iso_buckets];

                for iso_bucket in 0..num_iso_buckets {
                    let info_bucket = abs.bucket(iso_bucket, context);
                    iso_values[iso_bucket] = values[info_bucket];
                }

                // Then expand iso values to hand values
                self.isomorphism.buckets_to_hands(&iso_values, context)
            }
            None => self.isomorphism.buckets_to_hands(values, context),
        }
    }

    fn bucket_size(&self, bucket: u16, context: usize) -> u16 {
        match &self.info_abstraction {
            Some(abs) => {
                // Count all iso buckets that map to this info bucket
                let num_iso_buckets = self.isomorphism.num_buckets(context);
                let mut size = 0u16;
                for iso_bucket in 0..num_iso_buckets {
                    if abs.bucket(iso_bucket, context) == bucket as usize {
                        size += self.isomorphism.bucket_size(iso_bucket as u16, context);
                    }
                }
                size
            }
            None => self.isomorphism.bucket_size(bucket, context),
        }
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
// Clustered abstractions (EHS, EMD, WinSplit)
// ============================================================================

/// Expected Hand Strength abstraction.
///
/// Groups hands by their expected equity against a random opponent hand.
/// Uses k-means clustering on EHS values.
pub struct EHSAbstraction {
    /// Number of buckets.
    num_buckets: usize,
    /// Maps iso_hand → bucket for each context.
    /// iso_to_bucket[context][iso_hand] = bucket
    iso_to_bucket: Vec<Vec<u16>>,
    /// Bucket centers (EHS values).
    pub centers: Option<Vec<f32>>,
}

impl EHSAbstraction {
    /// Create EHS abstraction from pre-computed bucket assignments.
    ///
    /// # Arguments
    /// * `num_buckets` - Number of abstraction buckets
    /// * `iso_to_bucket` - Per-context bucket assignments for each iso hand
    /// * `centers` - Optional cluster centers for visualization
    pub fn new(
        num_buckets: usize,
        iso_to_bucket: Vec<Vec<u16>>,
        centers: Option<Vec<f32>>,
    ) -> Self {
        Self {
            num_buckets,
            iso_to_bucket,
            centers,
        }
    }

    /// Create from a single context (e.g., single board).
    pub fn single_context(num_buckets: usize, assignments: Vec<u16>, centers: Option<Vec<f32>>) -> Self {
        Self {
            num_buckets,
            iso_to_bucket: vec![assignments],
            centers,
        }
    }

    /// Get the bucket centers (if available).
    pub fn get_centers(&self) -> Option<&[f32]> {
        self.centers.as_deref()
    }
}

impl InfoAbstraction for EHSAbstraction {
    fn bucket(&self, iso_hand: usize, context: usize) -> usize {
        self.iso_to_bucket[context][iso_hand] as usize
    }

    fn num_buckets(&self) -> usize {
        self.num_buckets
    }
}

/// Earth Mover's Distance abstraction.
///
/// Groups hands by the shape of their equity distribution (histogram).
/// Uses k-means clustering with EMD metric on 50-bin histograms.
pub struct EMDAbstraction {
    /// Number of buckets.
    num_buckets: usize,
    /// Maps iso_hand → bucket for each context.
    iso_to_bucket: Vec<Vec<u16>>,
    /// Bucket centers (50-bin histograms).
    pub centers: Option<Vec<[f32; 50]>>,
}

impl EMDAbstraction {
    /// Create EMD abstraction from pre-computed bucket assignments.
    pub fn new(
        num_buckets: usize,
        iso_to_bucket: Vec<Vec<u16>>,
        centers: Option<Vec<[f32; 50]>>,
    ) -> Self {
        Self {
            num_buckets,
            iso_to_bucket,
            centers,
        }
    }

    /// Create from a single context.
    pub fn single_context(
        num_buckets: usize,
        assignments: Vec<u16>,
        centers: Option<Vec<[f32; 50]>>,
    ) -> Self {
        Self {
            num_buckets,
            iso_to_bucket: vec![assignments],
            centers,
        }
    }
}

impl InfoAbstraction for EMDAbstraction {
    fn bucket(&self, iso_hand: usize, context: usize) -> usize {
        self.iso_to_bucket[context][iso_hand] as usize
    }

    fn num_buckets(&self) -> usize {
        self.num_buckets
    }
}

/// Win/Split frequency abstraction (river only).
///
/// Groups hands by their win and split frequencies against uniform opponent range.
/// Uses k-means clustering on 2D [win_freq, split_freq] features.
pub struct WinSplitAbstraction {
    /// Number of buckets.
    num_buckets: usize,
    /// Maps iso_hand → bucket.
    iso_to_bucket: Vec<u16>,
    /// Bucket centers ([win_freq, split_freq]).
    pub centers: Option<Vec<[f32; 2]>>,
}

impl WinSplitAbstraction {
    /// Create WinSplit abstraction from pre-computed bucket assignments.
    pub fn new(
        num_buckets: usize,
        assignments: Vec<u16>,
        centers: Option<Vec<[f32; 2]>>,
    ) -> Self {
        Self {
            num_buckets,
            iso_to_bucket: assignments,
            centers,
        }
    }
}

impl InfoAbstraction for WinSplitAbstraction {
    fn bucket(&self, iso_hand: usize, _context: usize) -> usize {
        self.iso_to_bucket[iso_hand] as usize
    }

    fn num_buckets(&self) -> usize {
        self.num_buckets
    }
}

// ============================================================================
// AggSI (Aggressive Suit Isomorphism)
// ============================================================================

/// Aggressive Suit Isomorphism abstraction.
///
/// Extends suit isomorphism with flush potential features:
/// - For each card, compute a "flush feature" based on board texture
/// - Group hands by (rank_high, rank_low, flush_high, flush_low)
///
/// This is deterministic (no clustering needed).
pub struct AggSIAbstraction {
    /// Number of buckets.
    num_buckets: usize,
    /// Maps combo → bucket.
    combo_to_bucket: [u16; NUM_COMBOS],
}

impl AggSIAbstraction {
    /// Create AggSI abstraction for a board.
    ///
    /// Flush feature: 1 if the card's suit has flush potential/blocker value.
    /// - River: 1 if board has 3+ cards of this suit
    /// - Flop/Turn: 1 if suit can make a flush with remaining cards
    pub fn new(board: &Board) -> Self {
        let mut combo_to_bucket = [INVALID_BUCKET; NUM_COMBOS];
        let mut bucket_map: std::collections::HashMap<u32, u16> = std::collections::HashMap::new();

        // Count suits on board
        let mut suit_counts = [0u8; 4];
        for &card in &board.cards {
            suit_counts[suit(card) as usize] += 1;
        }

        let remaining_cards = 5 - board.cards.len();

        for combo_idx in 0..NUM_COMBOS {
            let combo = Combo::from_index(combo_idx);
            if combo.conflicts_with_mask(board.mask) {
                continue;
            }

            let r0 = rank(combo.c0);
            let r1 = rank(combo.c1);
            let s0 = suit(combo.c0);
            let s1 = suit(combo.c1);

            // Compute flush features
            let flush0 = compute_flush_feature(s0, &suit_counts, remaining_cards);
            let flush1 = compute_flush_feature(s1, &suit_counts, remaining_cards);

            // Order by rank (high, low)
            let (rank_high, rank_low, flush_high, flush_low) = if r1 >= r0 {
                (r1, r0, flush1, flush0)
            } else {
                (r0, r1, flush0, flush1)
            };

            // Compute hash
            let is_pair = rank_high == rank_low;
            let hash = if is_pair {
                // Pairs: only need one flush feature (either both or neither relevant)
                let flush_any = if flush_high != 0 || flush_low != 0 { 1u32 } else { 0 };
                (rank_high as u32) * 2 + flush_any
            } else {
                // Non-pairs: 4 flush combinations
                1000 + (rank_high as u32) * 1000 + (rank_low as u32) * 10 +
                    (flush_high as u32) * 2 + flush_low as u32
            };

            // Get or create bucket
            let bucket = if let Some(&b) = bucket_map.get(&hash) {
                b
            } else {
                let b = bucket_map.len() as u16;
                bucket_map.insert(hash, b);
                b
            };

            combo_to_bucket[combo_idx] = bucket;
        }

        AggSIAbstraction {
            num_buckets: bucket_map.len(),
            combo_to_bucket,
        }
    }

    /// Get the bucket for a combo.
    pub fn get_bucket(&self, combo_idx: usize) -> Option<u16> {
        let b = self.combo_to_bucket[combo_idx];
        if b == INVALID_BUCKET {
            None
        } else {
            Some(b)
        }
    }
}

impl InfoAbstraction for AggSIAbstraction {
    fn bucket(&self, combo_idx: usize, _context: usize) -> usize {
        self.combo_to_bucket[combo_idx] as usize
    }

    fn num_buckets(&self) -> usize {
        self.num_buckets
    }
}

/// Compute flush feature for a card suit.
///
/// Returns 1 if the suit has flush relevance:
/// - River (remaining=0): 1 if board has 3+ of this suit (blocker)
/// - Turn (remaining=1): 1 if board has 2+ of this suit (potential)
/// - Flop (remaining=2): 1 if board has 2+ of this suit (potential)
fn compute_flush_feature(card_suit: u8, suit_counts: &[u8; 4], remaining_cards: usize) -> u8 {
    let board_count = suit_counts[card_suit as usize];

    match remaining_cards {
        0 => {
            // River: 3+ board cards = flush possible with hole cards
            if board_count >= 3 {
                1
            } else {
                0
            }
        }
        1 | 2 => {
            // Turn/Flop: 2+ board cards = flush draw possible
            if board_count >= 2 {
                1
            } else {
                0
            }
        }
        _ => 0,
    }
}

// ============================================================================
// SemiAggSI (Flop only, finer suit distinction)
// ============================================================================

/// Semi-Aggressive Suit Isomorphism (flop only).
///
/// More fine-grained than AggSI:
/// - Tracks exact suit count on board for each card
/// - Distinguishes suited vs offsuit hole cards
pub struct SemiAggSIAbstraction {
    /// Number of buckets.
    num_buckets: usize,
    /// Maps combo → bucket.
    combo_to_bucket: [u16; NUM_COMBOS],
}

impl SemiAggSIAbstraction {
    /// Create SemiAggSI abstraction for a flop board.
    pub fn new(board: &Board) -> Self {
        assert!(
            board.cards.len() == 3,
            "SemiAggSI is only for flop (3 cards)"
        );

        let mut combo_to_bucket = [INVALID_BUCKET; NUM_COMBOS];
        let mut bucket_map: std::collections::HashMap<u64, u16> = std::collections::HashMap::new();

        // Count suits on board
        let mut suit_counts = [0u8; 4];
        for &card in &board.cards {
            suit_counts[suit(card) as usize] += 1;
        }

        for combo_idx in 0..NUM_COMBOS {
            let combo = Combo::from_index(combo_idx);
            if combo.conflicts_with_mask(board.mask) {
                continue;
            }

            let r0 = rank(combo.c0);
            let r1 = rank(combo.c1);
            let s0 = suit(combo.c0);
            let s1 = suit(combo.c1);

            // Flush feature = count of board cards matching this suit (0-3)
            let flush0 = suit_counts[s0 as usize];
            let flush1 = suit_counts[s1 as usize];

            // Is suited?
            let suited = if s0 == s1 { 1u64 } else { 0 };

            // Order by rank
            let (rank_high, rank_low, flush_high, flush_low) = if r1 >= r0 {
                (r1, r0, flush1, flush0)
            } else {
                (r0, r1, flush0, flush1)
            };

            let is_pair = rank_high == rank_low;

            // Compute hash
            let hash = if is_pair {
                // Pairs: 6 flush combinations (0-3 for each, but symmetric)
                let flush_max = flush_high.max(flush_low);
                let flush_min = flush_high.min(flush_low);
                (rank_high as u64) * 100 + (flush_max as u64) * 10 + flush_min as u64
            } else if suited != 0 {
                // Suited non-pairs
                10000 + (rank_high as u64) * 1000 + (rank_low as u64) * 10 + flush_high as u64
            } else {
                // Offsuit non-pairs
                100000 + (rank_high as u64) * 10000 + (rank_low as u64) * 100 +
                    (flush_high as u64) * 10 + flush_low as u64
            };

            let bucket = if let Some(&b) = bucket_map.get(&hash) {
                b
            } else {
                let b = bucket_map.len() as u16;
                bucket_map.insert(hash, b);
                b
            };

            combo_to_bucket[combo_idx] = bucket;
        }

        SemiAggSIAbstraction {
            num_buckets: bucket_map.len(),
            combo_to_bucket,
        }
    }
}

impl InfoAbstraction for SemiAggSIAbstraction {
    fn bucket(&self, combo_idx: usize, _context: usize) -> usize {
        self.combo_to_bucket[combo_idx] as usize
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

    // ============================================================================
    // Validation tests ported from Gambit
    // ============================================================================

    #[test]
    fn test_sum_preservation() {
        // Test that sum(hand_range) == sum(bucket_range)
        let board = parse_board("KhQsJs2c3d").unwrap();
        let iso = SuitIsomorphism::new(&board);

        // Create random reaches
        let mut reaches = vec![0.0f32; NUM_COMBOS];
        let mut total_before = 0.0f32;
        for combo_idx in 0..NUM_COMBOS {
            if iso.bucket(combo_idx, 0).is_some() {
                let weight = (combo_idx as f32 + 1.0) / 1000.0;
                reaches[combo_idx] = weight;
                total_before += weight;
            }
        }

        // Convert to buckets
        let bucket_reaches = iso.hands_to_buckets(&reaches, 0);
        let total_buckets: f32 = bucket_reaches.iter().sum();

        // Sum should be preserved
        assert!(
            (total_before - total_buckets).abs() < 1e-5,
            "Sum not preserved: before={}, after={}",
            total_before,
            total_buckets
        );
    }

    #[test]
    fn test_suit_isomorphism_pairs() {
        // On a monotone board (all same suit), pairs without that suit should be isomorphic
        // On this board with 4 spades, AA combos without spades should be equivalent
        let board = parse_board("KsQsJs2s3h").unwrap();
        let iso = SuitIsomorphism::new(&board);

        // Find AA combos that don't use spades (c, d, h only)
        // AdAc, AdAh, AcAh
        let aa_combos: Vec<usize> = (0..NUM_COMBOS)
            .filter(|&idx| {
                let combo = crate::poker::hands::Combo::from_index(idx);
                let rank0 = combo.c0 / 4;
                let rank1 = combo.c1 / 4;
                let suit0 = combo.c0 % 4;
                let suit1 = combo.c1 % 4;
                // Both ranks are Aces
                if rank0 != 12 || rank1 != 12 {
                    return false;
                }
                // Neither suit is spades (3) or hearts (2) which is on board
                suit0 != 3 && suit1 != 3 && suit0 != 2 && suit1 != 2
            })
            .collect();

        // These AA combos (AdAc) should all be valid and map to same bucket
        let valid_aa: Vec<_> = aa_combos
            .iter()
            .filter_map(|&idx| iso.bucket(idx, 0).map(|b| (idx, b)))
            .collect();

        // Should have at least one valid combo
        assert!(!valid_aa.is_empty(), "Should have valid AA combo without spades or hearts");

        // All remaining AA combos should map to the same bucket
        if valid_aa.len() > 1 {
            let first_bucket = valid_aa[0].1;
            for (idx, bucket) in &valid_aa {
                assert_eq!(
                    *bucket, first_bucket,
                    "AA combo {} should map to same bucket as others",
                    idx
                );
            }
        }
    }

    #[test]
    fn test_composed_with_info_abstraction() {
        // Test ComposedAbstraction with a simple info abstraction layer
        let board = parse_board("KhQsJs2c3d").unwrap();
        let iso = SuitIsomorphism::new(&board);
        let num_iso_buckets = iso.num_buckets(0);

        // Create a simple info abstraction that halves the bucket count
        let num_info_buckets = (num_iso_buckets / 2).max(1);
        let assignments: Vec<u16> = (0..num_iso_buckets)
            .map(|i| (i % num_info_buckets) as u16)
            .collect();

        let info_abs = EHSAbstraction::single_context(num_info_buckets, assignments, None);
        let composed = ComposedAbstraction::with_info_abstraction(iso, Box::new(info_abs));

        // Verify bucket count is reduced
        assert_eq!(composed.num_buckets(0), num_info_buckets);

        // Verify bucketing works
        let mut has_valid = false;
        for combo_idx in 0..NUM_COMBOS {
            if let Some(bucket) = composed.bucket(combo_idx, 0) {
                has_valid = true;
                assert!(
                    (bucket as usize) < num_info_buckets,
                    "Bucket {} exceeds max {}",
                    bucket,
                    num_info_buckets
                );
            }
        }
        assert!(has_valid, "Should have at least one valid combo");
    }

    #[test]
    fn test_bucket_coverage() {
        // Test that all expected buckets are populated for a board
        let board = parse_board("KhQsJs2c3d").unwrap();
        let iso = SuitIsomorphism::new(&board);

        let num_buckets = iso.num_buckets(0);
        let mut bucket_counts = vec![0usize; num_buckets];

        for combo_idx in 0..NUM_COMBOS {
            if let Some(bucket) = iso.bucket(combo_idx, 0) {
                bucket_counts[bucket as usize] += 1;
            }
        }

        // Check that no bucket is empty (unless board blocks it)
        let non_empty: Vec<_> = bucket_counts.iter().filter(|&&c| c > 0).collect();
        assert!(!non_empty.is_empty(), "Should have at least one non-empty bucket");

        // All valid combos should be assigned
        let total_assigned: usize = bucket_counts.iter().sum();
        assert!(total_assigned > 0, "Should have assigned combos");
    }

    #[test]
    fn test_multi_context_bucket_independence() {
        // Test that different contexts have independent bucketing
        let board = parse_board("KhQsJs2c").unwrap();
        let valid_cards = compute_valid_cards(&board);
        let iso = SuitIsomorphism::for_turn_board(&board, &valid_cards);

        assert!(iso.num_contexts() > 1);

        // Different contexts may have different bucket counts
        let ctx0_buckets = iso.num_buckets(0);
        let ctx1_buckets = iso.num_buckets(1);

        // Both should be positive
        assert!(ctx0_buckets > 0);
        assert!(ctx1_buckets > 0);
    }
}
