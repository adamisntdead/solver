//! Position handling for multi-way poker games.
//!
//! Standard poker positions and action ordering for 2-6 players.
//!
//! ## Position Names by Player Count
//!
//! | Seats | Positions |
//! |-------|-----------|
//! | 2 (HU) | BTN/SB, BB |
//! | 3 | BTN, SB, BB |
//! | 4 | CO, BTN, SB, BB |
//! | 5 | MP, CO, BTN, SB, BB |
//! | 6 | UTG, MP, CO, BTN, SB, BB |
//!
//! ## Action Order
//!
//! - **Preflop**: UTG → ... → BTN → SB → BB
//! - **Postflop**: SB → BB → UTG → ... → BTN

use crate::tree::Street;

/// Standard poker positions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Position {
    /// Under the gun (first preflop in 6-max)
    UTG,
    /// Middle position
    MP,
    /// Cutoff
    CO,
    /// Button (dealer)
    BTN,
    /// Small blind
    SB,
    /// Big blind
    BB,
}

impl Position {
    /// Get all positions for a given number of players.
    pub fn all_for_players(num_players: usize) -> &'static [Position] {
        use Position::*;
        match num_players {
            2 => &[BTN, BB], // HU: BTN posts SB
            3 => &[BTN, SB, BB],
            4 => &[CO, BTN, SB, BB],
            5 => &[MP, CO, BTN, SB, BB],
            6 => &[UTG, MP, CO, BTN, SB, BB],
            _ => panic!("Unsupported player count: {}", num_players),
        }
    }

    /// Get the position name as a short string.
    pub fn short_name(&self) -> &'static str {
        match self {
            Position::UTG => "UTG",
            Position::MP => "MP",
            Position::CO => "CO",
            Position::BTN => "BTN",
            Position::SB => "SB",
            Position::BB => "BB",
        }
    }
}

/// Convert a seat index to a position.
///
/// Seat indices are 0-based and follow the dealing order:
/// - For 6-max: 0=UTG, 1=MP, 2=CO, 3=BTN, 4=SB, 5=BB
/// - For HU: 0=BTN/SB, 1=BB
pub fn seat_to_position(seat: usize, num_players: usize) -> Position {
    Position::all_for_players(num_players)[seat]
}

/// Get the seat index of the first actor for a given street.
///
/// - Preflop: First active player after BB (UTG position)
/// - Postflop: First active player from SB position
pub fn first_actor(street: Street, num_players: usize, active_mask: u8) -> Option<usize> {
    if num_players < 2 || num_players > 6 {
        return None;
    }

    let start_seat = match street {
        Street::Preflop => {
            if num_players == 2 {
                // Heads-up: BTN/SB acts first preflop
                0
            } else {
                // Multi-way: UTG acts first (seat 0)
                0
            }
        }
        Street::Flop | Street::Turn | Street::River => {
            if num_players == 2 {
                // Heads-up: BB acts first postflop
                1
            } else {
                // Multi-way: SB acts first (or first active from SB)
                num_players - 2 // SB seat
            }
        }
    };

    // Find first active player starting from start_seat
    find_next_active(start_seat, num_players, active_mask, true)
}

/// Get the seat index of the next actor after the given seat.
///
/// Wraps around and skips folded players.
pub fn next_actor(current_seat: usize, num_players: usize, active_mask: u8) -> Option<usize> {
    if num_players < 2 || num_players > 6 {
        return None;
    }

    let next_seat = (current_seat + 1) % num_players;
    find_next_active(next_seat, num_players, active_mask, false)
}

/// Find the next active player starting from a given seat.
///
/// Returns None if no active players found.
fn find_next_active(
    start_seat: usize,
    num_players: usize,
    active_mask: u8,
    include_start: bool,
) -> Option<usize> {
    let initial_offset = if include_start { 0 } else { 0 };
    for i in initial_offset..num_players {
        let seat = (start_seat + i) % num_players;
        if (active_mask >> seat) & 1 == 1 {
            return Some(seat);
        }
    }
    None
}

/// Get the preflop action order for a given number of players.
///
/// Returns seat indices in the order they act preflop.
/// For multi-way: UTG → ... → BTN → SB → BB
/// For HU: BTN/SB → BB
pub fn preflop_action_order(num_players: usize) -> Vec<usize> {
    match num_players {
        2 => vec![0, 1], // BTN acts first, then BB
        3 => vec![0, 1, 2], // BTN, SB, BB
        4 => vec![0, 1, 2, 3], // CO, BTN, SB, BB
        5 => vec![0, 1, 2, 3, 4], // MP, CO, BTN, SB, BB
        6 => vec![0, 1, 2, 3, 4, 5], // UTG, MP, CO, BTN, SB, BB
        _ => panic!("Unsupported player count: {}", num_players),
    }
}

/// Get the postflop action order for a given number of players.
///
/// Returns seat indices in the order they act postflop (OOP first).
/// For multi-way: SB → BB → UTG → ... → BTN
/// For HU: BB → BTN/SB
pub fn postflop_action_order(num_players: usize) -> Vec<usize> {
    match num_players {
        2 => vec![1, 0], // BB acts first, then BTN
        3 => vec![1, 2, 0], // SB, BB, BTN
        4 => vec![2, 3, 0, 1], // SB, BB, CO, BTN
        5 => vec![3, 4, 0, 1, 2], // SB, BB, MP, CO, BTN
        6 => vec![4, 5, 0, 1, 2, 3], // SB, BB, UTG, MP, CO, BTN
        _ => panic!("Unsupported player count: {}", num_players),
    }
}

/// Get the blind seats (SB, BB) for a given number of players.
///
/// Returns (sb_seat, bb_seat).
pub fn blind_seats(num_players: usize) -> (usize, usize) {
    match num_players {
        2 => (0, 1), // HU: BTN posts SB
        _ => (num_players - 2, num_players - 1),
    }
}

/// Count the number of active players from a bitmask.
pub fn count_active(active_mask: u8) -> usize {
    active_mask.count_ones() as usize
}

/// Create an active mask with all players active.
pub fn all_active_mask(num_players: usize) -> u8 {
    (1u8 << num_players) - 1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_positions_for_players() {
        use Position::*;

        assert_eq!(Position::all_for_players(2), &[BTN, BB]);
        assert_eq!(Position::all_for_players(3), &[BTN, SB, BB]);
        assert_eq!(Position::all_for_players(6), &[UTG, MP, CO, BTN, SB, BB]);
    }

    #[test]
    fn test_seat_to_position() {
        use Position::*;

        // HU
        assert_eq!(seat_to_position(0, 2), BTN);
        assert_eq!(seat_to_position(1, 2), BB);

        // 6-max
        assert_eq!(seat_to_position(0, 6), UTG);
        assert_eq!(seat_to_position(3, 6), BTN);
        assert_eq!(seat_to_position(5, 6), BB);
    }

    #[test]
    fn test_first_actor_preflop() {
        // 6-max: UTG (seat 0) acts first
        let active = all_active_mask(6);
        assert_eq!(first_actor(Street::Preflop, 6, active), Some(0));

        // HU: BTN (seat 0) acts first
        let active = all_active_mask(2);
        assert_eq!(first_actor(Street::Preflop, 2, active), Some(0));
    }

    #[test]
    fn test_first_actor_postflop() {
        // 6-max: SB (seat 4) acts first
        let active = all_active_mask(6);
        assert_eq!(first_actor(Street::Flop, 6, active), Some(4));

        // HU: BB (seat 1) acts first
        let active = all_active_mask(2);
        assert_eq!(first_actor(Street::Flop, 2, active), Some(1));

        // 6-max with SB folded: BB (seat 5) acts first
        let active = 0b101111; // SB (seat 4) folded
        assert_eq!(first_actor(Street::Flop, 6, active), Some(5));
    }

    #[test]
    fn test_next_actor() {
        let active = all_active_mask(6);

        assert_eq!(next_actor(0, 6, active), Some(1));
        assert_eq!(next_actor(5, 6, active), Some(0)); // Wrap around

        // With folded player
        let active = 0b111011; // Seat 2 folded
        assert_eq!(next_actor(1, 6, active), Some(3)); // Skip seat 2
    }

    #[test]
    fn test_blind_seats() {
        assert_eq!(blind_seats(2), (0, 1)); // HU: BTN posts SB
        assert_eq!(blind_seats(6), (4, 5)); // 6-max: seats 4, 5
    }

    #[test]
    fn test_count_active() {
        assert_eq!(count_active(0b111111), 6);
        assert_eq!(count_active(0b111011), 5);
        assert_eq!(count_active(0b11), 2);
    }

    #[test]
    fn test_preflop_action_order() {
        assert_eq!(preflop_action_order(2), vec![0, 1]);
        assert_eq!(preflop_action_order(6), vec![0, 1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_postflop_action_order() {
        assert_eq!(postflop_action_order(2), vec![1, 0]);
        assert_eq!(postflop_action_order(6), vec![4, 5, 0, 1, 2, 3]);
    }
}
