//! Imperfect recall hand indexer for hand abstraction.
//!
//! Maps (board, hand) pairs to unique indices across all possible boards for a street.
//! This is essential for pre-computing hand abstractions that can be loaded at solve time.
//!
//! # Street Indexing
//!
//! Each street has a specific number of total combinations:
//! - **River**: For each canonical 5-card board, enumerate valid isomorphic hands
//! - **Turn**: For each canonical 4-card board, enumerate valid isomorphic hands
//! - **Flop**: For each canonical 3-card board, enumerate valid isomorphic hands
//!
//! # Board Canonicalization
//!
//! Boards are canonicalized using suit isomorphism to reduce the total space:
//! - Suits are reordered by first appearance
//! - This groups equivalent boards (e.g., Ah Kh Qh ≡ As Ks Qs)

use crate::poker::hands::{Board, Card, Combo, DECK_SIZE, NUM_COMBOS};
use crate::poker::isomorphism::{BoardIsomorphism, SuitMapping, INVALID_BUCKET};

/// Street identifier for indexing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Street {
    Preflop = 0,
    Flop = 1,
    Turn = 2,
    River = 3,
}

impl Street {
    /// Number of community cards for this street.
    pub fn num_board_cards(&self) -> usize {
        match self {
            Street::Preflop => 0,
            Street::Flop => 3,
            Street::Turn => 4,
            Street::River => 5,
        }
    }

    /// Street name for display.
    pub fn name(&self) -> &'static str {
        match self {
            Street::Preflop => "preflop",
            Street::Flop => "flop",
            Street::Turn => "turn",
            Street::River => "river",
        }
    }
}

/// Canonical board representation for indexing.
///
/// Boards are stored in a canonical form where suits are reordered
/// by first appearance. This allows grouping equivalent boards.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CanonicalBoard {
    /// Canonical card values (suits remapped).
    pub cards: Vec<Card>,
    /// Original board index before canonicalization.
    pub original_index: usize,
}

impl CanonicalBoard {
    /// Create a canonical board from raw cards.
    pub fn from_cards(cards: &[Card]) -> Self {
        let board = Board::new(cards);
        let mapping = SuitMapping::canonical_for_board(&board);
        let canonical_cards: Vec<Card> = cards.iter().map(|&c| mapping.transform_card(c)).collect();

        // Sort cards for consistent ordering
        let mut sorted = canonical_cards.clone();
        sorted.sort();

        CanonicalBoard {
            cards: sorted,
            original_index: 0,
        }
    }

    /// Convert to a Board.
    pub fn to_board(&self) -> Board {
        Board::new(&self.cards)
    }
}

/// Board enumerator: generates all canonical boards for a street.
pub struct BoardEnumerator {
    street: Street,
}

impl BoardEnumerator {
    /// Create a board enumerator for a street.
    pub fn new(street: Street) -> Self {
        BoardEnumerator { street }
    }

    /// Enumerate all canonical boards for the street.
    ///
    /// Returns boards in canonical form (suit-isomorphic boards grouped).
    pub fn enumerate_canonical(&self) -> Vec<CanonicalBoard> {
        let num_cards = self.street.num_board_cards();
        if num_cards == 0 {
            // Preflop: single "empty board"
            return vec![CanonicalBoard {
                cards: vec![],
                original_index: 0,
            }];
        }

        let mut canonical_boards = Vec::new();
        let mut seen: std::collections::HashSet<Vec<Card>> = std::collections::HashSet::new();

        // Generate all C(52, num_cards) boards
        self.enumerate_boards_recursive(
            &mut vec![],
            0,
            num_cards,
            &mut canonical_boards,
            &mut seen,
        );

        // Sort for deterministic ordering
        canonical_boards.sort_by(|a, b| a.cards.cmp(&b.cards));

        // Assign indices
        for (i, board) in canonical_boards.iter_mut().enumerate() {
            board.original_index = i;
        }

        canonical_boards
    }

    fn enumerate_boards_recursive(
        &self,
        current: &mut Vec<Card>,
        start: Card,
        remaining: usize,
        result: &mut Vec<CanonicalBoard>,
        seen: &mut std::collections::HashSet<Vec<Card>>,
    ) {
        if remaining == 0 {
            let canonical = CanonicalBoard::from_cards(current);
            if !seen.contains(&canonical.cards) {
                seen.insert(canonical.cards.clone());
                result.push(canonical);
            }
            return;
        }

        for card in start..(DECK_SIZE as Card - remaining as Card + 1) {
            current.push(card);
            self.enumerate_boards_recursive(current, card + 1, remaining - 1, result, seen);
            current.pop();
        }
    }

    /// Count total canonical boards for this street.
    pub fn count_canonical(&self) -> usize {
        // These are known values for standard suit isomorphism
        match self.street {
            Street::Preflop => 1,
            Street::Flop => 1_755, // C(52,3) / ~10 (suit symmetry)
            Street::Turn => 16_432, // Approximate
            Street::River => 134_459, // Approximate
        }
    }
}

/// Imperfect recall hand indexer.
///
/// Maps (canonical_board, isomorphic_hand) to a unique index.
/// This is used for storing and loading pre-computed abstractions.
pub struct ImperfectRecallIndexer {
    street: Street,
    /// For each canonical board: (board, start_index, num_iso_hands)
    board_info: Vec<(CanonicalBoard, usize, usize)>,
    /// Total number of (board, hand) combinations.
    total_size: usize,
}

impl ImperfectRecallIndexer {
    /// Create an indexer for a street.
    ///
    /// This enumerates all canonical boards and computes the index ranges
    /// for each board's isomorphic hands.
    pub fn new(street: Street) -> Self {
        let enumerator = BoardEnumerator::new(street);
        let canonical_boards = enumerator.enumerate_canonical();

        let mut board_info = Vec::with_capacity(canonical_boards.len());
        let mut current_index = 0;

        for board in canonical_boards {
            let b = board.to_board();
            let iso = BoardIsomorphism::new(&b);
            let num_hands = iso.num_buckets;

            board_info.push((board, current_index, num_hands));
            current_index += num_hands;
        }

        ImperfectRecallIndexer {
            street,
            board_info,
            total_size: current_index,
        }
    }

    /// Get the street this indexer is for.
    pub fn street(&self) -> Street {
        self.street
    }

    /// Total number of (board, hand) indices.
    pub fn round_size(&self) -> usize {
        self.total_size
    }

    /// Number of canonical boards.
    pub fn num_boards(&self) -> usize {
        self.board_info.len()
    }

    /// Get board info by board index.
    pub fn board_info(&self, board_idx: usize) -> &(CanonicalBoard, usize, usize) {
        &self.board_info[board_idx]
    }

    /// Find the board index for a given canonical board.
    pub fn find_board(&self, board: &Board) -> Option<usize> {
        let canonical = CanonicalBoard::from_cards(&board.cards);
        self.board_info
            .iter()
            .position(|(b, _, _)| b.cards == canonical.cards)
    }

    /// Get the index for a (board, isomorphic_hand) pair.
    ///
    /// Returns None if the board or hand is invalid.
    pub fn index(&self, board: &Board, combo: Combo) -> Option<usize> {
        // Find canonical board
        let board_idx = self.find_board(board)?;
        let (_, start_idx, _) = &self.board_info[board_idx];

        // Get isomorphic hand index
        let iso = BoardIsomorphism::new(board);
        let bucket = iso.get_bucket(combo.to_index())?;

        Some(start_idx + bucket as usize)
    }

    /// Reverse lookup: get (board_idx, iso_hand_idx) from global index.
    pub fn unindex(&self, global_idx: usize) -> Option<(usize, u16)> {
        if global_idx >= self.total_size {
            return None;
        }

        // Binary search for the board
        let board_idx = self
            .board_info
            .partition_point(|(_, start, _)| *start <= global_idx)
            .saturating_sub(1);

        let (_, start_idx, num_hands) = &self.board_info[board_idx];
        let iso_hand = (global_idx - start_idx) as u16;

        if iso_hand as usize >= *num_hands {
            return None;
        }

        Some((board_idx, iso_hand))
    }

    /// Iterate over all (board_idx, iso_hand_idx, global_idx) tuples.
    pub fn iter(&self) -> impl Iterator<Item = (usize, u16, usize)> + '_ {
        self.board_info
            .iter()
            .enumerate()
            .flat_map(|(board_idx, (_, start_idx, num_hands))| {
                (0..*num_hands).map(move |iso_hand| {
                    (board_idx, iso_hand as u16, start_idx + iso_hand)
                })
            })
    }
}

/// Compute the number of isomorphic hands for a board.
pub fn count_iso_hands(board: &Board) -> usize {
    BoardIsomorphism::new(board).num_buckets
}

/// Per-board indexer for faster lookups when solving a specific board.
///
/// Unlike ImperfectRecallIndexer which handles all boards, this is optimized
/// for repeated lookups on a single board.
pub struct SingleBoardIndexer {
    /// The board this indexer is for.
    pub board: Board,
    /// Isomorphism mapping for this board.
    pub iso: BoardIsomorphism,
    /// Base index for this board in global space (if applicable).
    pub base_index: usize,
}

impl SingleBoardIndexer {
    /// Create an indexer for a specific board.
    pub fn new(board: &Board) -> Self {
        SingleBoardIndexer {
            board: board.clone(),
            iso: BoardIsomorphism::new(board),
            base_index: 0,
        }
    }

    /// Create with a base index for global lookups.
    pub fn with_base_index(board: &Board, base_index: usize) -> Self {
        SingleBoardIndexer {
            board: board.clone(),
            iso: BoardIsomorphism::new(board),
            base_index,
        }
    }

    /// Get the isomorphic bucket for a combo.
    #[inline]
    pub fn get_bucket(&self, combo_idx: usize) -> Option<u16> {
        self.iso.get_bucket(combo_idx)
    }

    /// Get the global index for a combo.
    #[inline]
    pub fn global_index(&self, combo_idx: usize) -> Option<usize> {
        self.iso
            .get_bucket(combo_idx)
            .map(|b| self.base_index + b as usize)
    }

    /// Number of isomorphic hands for this board.
    pub fn num_iso_hands(&self) -> usize {
        self.iso.num_buckets
    }

    /// Check if a combo is valid (not blocked).
    #[inline]
    pub fn is_valid(&self, combo_idx: usize) -> bool {
        self.iso.combo_to_bucket[combo_idx] != INVALID_BUCKET
    }

    /// Iterate over valid (combo_idx, iso_bucket) pairs.
    pub fn valid_combos(&self) -> impl Iterator<Item = (usize, u16)> + '_ {
        (0..NUM_COMBOS).filter_map(move |combo_idx| {
            self.get_bucket(combo_idx).map(|bucket| (combo_idx, bucket))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::poker::board_parser::parse_board;

    #[test]
    fn test_street_num_cards() {
        assert_eq!(Street::Preflop.num_board_cards(), 0);
        assert_eq!(Street::Flop.num_board_cards(), 3);
        assert_eq!(Street::Turn.num_board_cards(), 4);
        assert_eq!(Street::River.num_board_cards(), 5);
    }

    #[test]
    fn test_canonical_board() {
        // Two suit-equivalent boards should have same canonical form
        let board1 = parse_board("AhKhQh").unwrap();
        let board2 = parse_board("AsKsQs").unwrap();

        let canonical1 = CanonicalBoard::from_cards(&board1.cards);
        let canonical2 = CanonicalBoard::from_cards(&board2.cards);

        assert_eq!(canonical1.cards, canonical2.cards);
    }

    #[test]
    fn test_board_enumerator_preflop() {
        let enumerator = BoardEnumerator::new(Street::Preflop);
        let boards = enumerator.enumerate_canonical();
        assert_eq!(boards.len(), 1);
    }

    #[test]
    fn test_board_enumerator_flop() {
        let enumerator = BoardEnumerator::new(Street::Flop);
        let boards = enumerator.enumerate_canonical();

        // Should have canonical flop boards
        assert!(boards.len() > 1000);
        assert!(boards.len() < 5000); // Much less than C(52,3) = 22100

        println!("Canonical flop boards: {}", boards.len());
    }

    #[test]
    fn test_single_board_indexer() {
        let board = parse_board("KhQsJs2c3d").unwrap();
        let indexer = SingleBoardIndexer::new(&board);

        // Should have valid iso hands
        assert!(indexer.num_iso_hands() > 0);
        assert!(indexer.num_iso_hands() <= NUM_COMBOS);

        // Count valid combos
        let valid_count = indexer.valid_combos().count();
        assert!(valid_count > 0);
        println!(
            "Valid combos: {}, Iso hands: {}",
            valid_count,
            indexer.num_iso_hands()
        );
    }

    #[test]
    fn test_single_board_indexer_global() {
        let board = parse_board("KhQsJs2c3d").unwrap();
        let indexer = SingleBoardIndexer::with_base_index(&board, 1000);

        // Global indices should start at base
        if let Some(global_idx) = indexer.global_index(0) {
            assert!(global_idx >= 1000);
        }
    }

    // Note: Full ImperfectRecallIndexer tests are expensive (enumerate all boards)
    // Run with --release for performance
    #[test]
    #[ignore] // Expensive test
    fn test_imperfect_recall_indexer_river() {
        let indexer = ImperfectRecallIndexer::new(Street::River);
        println!(
            "River indexer: {} boards, {} total indices",
            indexer.num_boards(),
            indexer.round_size()
        );

        // Should have many boards
        assert!(indexer.num_boards() > 100_000);

        // Test unindex
        let (board_idx, iso_hand) = indexer.unindex(0).unwrap();
        assert_eq!(board_idx, 0);
        assert_eq!(iso_hand, 0);
    }
}
