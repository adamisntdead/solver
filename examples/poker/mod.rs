//! Poker-specific modules for the CFR solver examples.
//!
//! This module provides:
//! - Card representation and canonical hand conversion
//! - 7-card hand evaluation
//! - Monte Carlo equity calculation

pub mod cards;
pub mod equity;
pub mod hand_eval;

pub use cards::{
    available_combos, from_canonical, make_card, rank, CanonicalHand, HoleCards,
    NUM_CANONICAL_HANDS,
};
pub use equity::compute_equity;

// Re-export items that may be useful but aren't used in the current example
#[allow(unused_imports)]
pub use cards::{card_to_string, canonical_to_string, cards_to_mask, mask_to_cards, to_canonical, Card, DECK_SIZE};
