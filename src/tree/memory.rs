//! Memory estimation for poker game trees.
//!
//! Calculates expected memory usage for CFR solver based on tree structure
//! and number of information sets.

use crate::tree::builder::ActionTree;

/// Memory usage estimate for a tree.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct MemoryEstimate {
    /// Memory for uncompressed regrets/strategies (32-bit floats).
    pub uncompressed_bytes: u64,

    /// Memory for compressed storage (16-bit integers).
    pub compressed_bytes: u64,

    /// Number of tree nodes.
    pub node_count: u64,

    /// Number of information sets (nodes × hands).
    pub info_set_count: u64,

    /// Number of player decision nodes.
    pub player_node_count: u64,
}

impl MemoryEstimate {
    /// Format bytes in human-readable form.
    pub fn format_bytes(bytes: u64) -> String {
        const KB: u64 = 1024;
        const MB: u64 = 1024 * KB;
        const GB: u64 = 1024 * MB;

        if bytes >= GB {
            format!("{:.2} GB", bytes as f64 / GB as f64)
        } else if bytes >= MB {
            format!("{:.2} MB", bytes as f64 / MB as f64)
        } else if bytes >= KB {
            format!("{:.2} KB", bytes as f64 / KB as f64)
        } else {
            format!("{} B", bytes)
        }
    }

    /// Get uncompressed memory as formatted string.
    pub fn uncompressed_str(&self) -> String {
        Self::format_bytes(self.uncompressed_bytes)
    }

    /// Get compressed memory as formatted string.
    pub fn compressed_str(&self) -> String {
        Self::format_bytes(self.compressed_bytes)
    }
}

impl ActionTree {
    /// Estimate memory usage for the tree.
    ///
    /// # Arguments
    /// * `num_hands` - Number of distinct hands (e.g., 169 for preflop, 1326 for full deck)
    /// * `num_actions_avg` - Average number of actions per node (for strategy storage)
    pub fn memory_estimate(&self, num_hands: u64, num_actions_avg: f64) -> MemoryEstimate {
        let player_nodes = self.player_node_count as u64;
        let info_sets = player_nodes * num_hands;

        // Storage per info set:
        // - Regret sum: num_actions × 4 bytes (f32)
        // - Strategy sum: num_actions × 4 bytes (f32)
        // - Current strategy: num_actions × 4 bytes (f32)
        // Total uncompressed: 3 × num_actions × 4 = 12 × num_actions bytes

        // Compressed storage uses 16-bit integers:
        // - Regret sum: num_actions × 2 bytes (i16)
        // - Strategy sum: num_actions × 2 bytes (i16)
        // Total compressed: 2 × num_actions × 2 = 4 × num_actions bytes

        let bytes_per_action_uncompressed = 12u64; // 3 arrays × 4 bytes
        let bytes_per_action_compressed = 4u64; // 2 arrays × 2 bytes

        let actions_per_info_set = num_actions_avg.max(1.0);
        let total_actions = (info_sets as f64 * actions_per_info_set) as u64;

        let uncompressed_bytes = total_actions * bytes_per_action_uncompressed;
        let compressed_bytes = total_actions * bytes_per_action_compressed;

        MemoryEstimate {
            uncompressed_bytes,
            compressed_bytes,
            node_count: self.node_count as u64,
            info_set_count: info_sets,
            player_node_count: player_nodes,
        }
    }

    /// Estimate memory for preflop-only tree (169 canonical hands).
    pub fn memory_estimate_preflop(&self) -> MemoryEstimate {
        // 169 canonical preflop hands (suit-isomorphic)
        // Average about 3 actions per node (fold, call, raise)
        self.memory_estimate(169, 3.0)
    }

    /// Estimate memory for postflop tree.
    ///
    /// # Arguments
    /// * `num_board_cards` - Number of board cards (3 for flop, 4 for turn, 5 for river)
    pub fn memory_estimate_postflop(&self, num_board_cards: usize) -> MemoryEstimate {
        // Number of hand combos depends on board cards
        // Full deck: 52 choose 2 = 1326 hands
        // But with board cards dealt, fewer hands available

        let remaining_cards = 52 - num_board_cards;
        let num_hands = (remaining_cards * (remaining_cards - 1) / 2) as u64;

        // Average about 4 actions per node (check/fold, call, bet sizes, all-in)
        self.memory_estimate(num_hands, 4.0)
    }
}

/// Calculate number of canonical preflop hand combinations.
pub const CANONICAL_PREFLOP_HANDS: u64 = 169;

/// Calculate number of hand combinations given dealt board cards.
pub const fn hand_combos(board_cards: usize) -> u64 {
    let remaining = 52 - board_cards;
    (remaining * (remaining - 1) / 2) as u64
}

/// Standard hand combo counts.
pub mod combos {
    /// Preflop: all 1326 combos
    pub const PREFLOP_ALL: u64 = 1326;

    /// Preflop: 169 suit-isomorphic combos
    pub const PREFLOP_CANONICAL: u64 = 169;

    /// Flop (3 cards): remaining combos
    pub const FLOP: u64 = super::hand_combos(3);

    /// Turn (4 cards): remaining combos
    pub const TURN: u64 = super::hand_combos(4);

    /// River (5 cards): remaining combos
    pub const RIVER: u64 = super::hand_combos(5);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tree::config::{PreflopConfig, TreeConfig};

    #[test]
    fn test_format_bytes() {
        assert_eq!(MemoryEstimate::format_bytes(500), "500 B");
        assert_eq!(MemoryEstimate::format_bytes(1024), "1.00 KB");
        assert_eq!(MemoryEstimate::format_bytes(1024 * 1024), "1.00 MB");
        assert_eq!(MemoryEstimate::format_bytes(1024 * 1024 * 1024), "1.00 GB");
    }

    #[test]
    fn test_memory_estimate() {
        let config = TreeConfig::new(2)
            .with_stack(100)
            .with_preflop(PreflopConfig::new(1, 2));

        let tree = ActionTree::new(config).unwrap();
        let estimate = tree.memory_estimate_preflop();

        assert!(estimate.info_set_count > 0);
        assert!(estimate.uncompressed_bytes > estimate.compressed_bytes);
    }

    #[test]
    fn test_hand_combos() {
        assert_eq!(hand_combos(0), 1326); // Full deck
        assert_eq!(hand_combos(3), 1176); // Flop
        assert_eq!(hand_combos(5), 1081); // River
    }

    #[test]
    fn test_combo_constants() {
        assert_eq!(combos::PREFLOP_ALL, 1326);
        assert_eq!(combos::PREFLOP_CANONICAL, 169);
        assert_eq!(combos::FLOP, hand_combos(3));
    }
}
