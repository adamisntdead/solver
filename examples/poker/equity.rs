//! Monte Carlo equity calculator for multi-way all-in situations.

#![allow(dead_code)]

use super::cards::{Card, HoleCards, DECK_SIZE};
use super::hand_eval::evaluate_7cards;
use rand::prelude::*;
use rand::rngs::SmallRng;
use rayon::prelude::*;

/// Number of Monte Carlo samples per thread batch.
const BATCH_SIZE: usize = 1000;

/// Compute equity for each player in an all-in situation.
///
/// Returns a vector of equities (0.0 to 1.0) that sum to 1.0.
/// Uses Monte Carlo simulation with parallel sampling.
///
/// # Arguments
/// * `hands` - Hole cards for each player
/// * `dead_mask` - Bitmask of cards that cannot be dealt (already used)
/// * `num_samples` - Number of random board runouts to sample
pub fn compute_equity(hands: &[HoleCards], dead_mask: u64, num_samples: usize) -> Vec<f64> {
    let num_players = hands.len();
    if num_players == 0 {
        return vec![];
    }
    if num_players == 1 {
        return vec![1.0];
    }

    // Build the deck of available cards
    let mut dead = dead_mask;
    for h in hands {
        dead |= (1 << h.0) | (1 << h.1);
    }
    let deck: Vec<Card> = (0..DECK_SIZE as u8)
        .filter(|&c| (dead & (1 << c)) == 0)
        .collect();

    if deck.len() < 5 {
        // Not enough cards for a board - return equal equity
        return vec![1.0 / num_players as f64; num_players];
    }

    // Number of batches to run in parallel
    let num_batches = (num_samples + BATCH_SIZE - 1) / BATCH_SIZE;

    // Run batches in parallel
    let results: Vec<(Vec<f64>, usize)> = (0..num_batches)
        .into_par_iter()
        .map(|batch_idx| {
            let mut rng = SmallRng::seed_from_u64((batch_idx * 12345) as u64);
            let samples_this_batch = if batch_idx == num_batches - 1 {
                num_samples - batch_idx * BATCH_SIZE
            } else {
                BATCH_SIZE
            };

            let mut local_wins = vec![0.0f64; num_players];
            let mut deck_copy = deck.clone();

            for _ in 0..samples_this_batch {
                // Shuffle and take first 5 cards as board
                deck_copy.shuffle(&mut rng);
                let board: [Card; 5] = [
                    deck_copy[0],
                    deck_copy[1],
                    deck_copy[2],
                    deck_copy[3],
                    deck_copy[4],
                ];

                // Evaluate each player's hand
                let mut hand_ranks = Vec::with_capacity(num_players);
                for h in hands {
                    let seven = [h.0, h.1, board[0], board[1], board[2], board[3], board[4]];
                    hand_ranks.push(evaluate_7cards(&seven));
                }

                // Find winner(s)
                let max_rank = *hand_ranks.iter().max().unwrap();
                let num_winners = hand_ranks.iter().filter(|&&r| r == max_rank).count();
                let win_share = 1.0 / num_winners as f64;

                for (i, &rank) in hand_ranks.iter().enumerate() {
                    if rank == max_rank {
                        local_wins[i] += win_share;
                    }
                }
            }

            (local_wins, samples_this_batch)
        })
        .collect();

    // Aggregate results
    let mut total_wins = vec![0.0f64; num_players];
    let mut total_samples = 0usize;

    for (wins, samples) in results {
        for (i, w) in wins.iter().enumerate() {
            total_wins[i] += w;
        }
        total_samples += samples;
    }

    // Normalize to get equity
    let total = total_samples as f64;
    total_wins.iter().map(|&w| w / total).collect()
}

/// Compute equity with a specific partial board (flop, turn, or river).
///
/// # Arguments
/// * `hands` - Hole cards for each player
/// * `board` - Known community cards (0-5 cards)
/// * `dead_mask` - Additional dead cards
/// * `num_samples` - Number of samples for remaining cards
pub fn compute_equity_with_board(
    hands: &[HoleCards],
    board: &[Card],
    dead_mask: u64,
    num_samples: usize,
) -> Vec<f64> {
    let num_players = hands.len();
    if num_players == 0 {
        return vec![];
    }
    if num_players == 1 {
        return vec![1.0];
    }

    // Mark all known cards as dead
    let mut dead = dead_mask;
    for h in hands {
        dead |= (1 << h.0) | (1 << h.1);
    }
    for &c in board {
        dead |= 1 << c;
    }

    let cards_needed = 5 - board.len();
    if cards_needed == 0 {
        // Complete board - just evaluate once
        let mut hand_ranks = Vec::with_capacity(num_players);
        let board_arr: [Card; 5] = [board[0], board[1], board[2], board[3], board[4]];
        for h in hands {
            let seven = [
                h.0,
                h.1,
                board_arr[0],
                board_arr[1],
                board_arr[2],
                board_arr[3],
                board_arr[4],
            ];
            hand_ranks.push(evaluate_7cards(&seven));
        }

        let max_rank = *hand_ranks.iter().max().unwrap();
        let num_winners = hand_ranks.iter().filter(|&&r| r == max_rank).count();

        return hand_ranks
            .iter()
            .map(|&r| {
                if r == max_rank {
                    1.0 / num_winners as f64
                } else {
                    0.0
                }
            })
            .collect();
    }

    let deck: Vec<Card> = (0..DECK_SIZE as u8)
        .filter(|&c| (dead & (1 << c)) == 0)
        .collect();

    if deck.len() < cards_needed {
        return vec![1.0 / num_players as f64; num_players];
    }

    let num_batches = (num_samples + BATCH_SIZE - 1) / BATCH_SIZE;

    let results: Vec<(Vec<f64>, usize)> = (0..num_batches)
        .into_par_iter()
        .map(|batch_idx| {
            let mut rng = SmallRng::seed_from_u64((batch_idx * 12345 + 67890) as u64);
            let samples_this_batch = if batch_idx == num_batches - 1 {
                num_samples - batch_idx * BATCH_SIZE
            } else {
                BATCH_SIZE
            };

            let mut local_wins = vec![0.0f64; num_players];
            let mut deck_copy = deck.clone();

            for _ in 0..samples_this_batch {
                deck_copy.shuffle(&mut rng);

                // Complete the board
                let mut full_board = [0u8; 5];
                full_board[..board.len()].copy_from_slice(board);
                for i in 0..cards_needed {
                    full_board[board.len() + i] = deck_copy[i];
                }

                let mut hand_ranks = Vec::with_capacity(num_players);
                for h in hands {
                    let seven = [
                        h.0,
                        h.1,
                        full_board[0],
                        full_board[1],
                        full_board[2],
                        full_board[3],
                        full_board[4],
                    ];
                    hand_ranks.push(evaluate_7cards(&seven));
                }

                let max_rank = *hand_ranks.iter().max().unwrap();
                let num_winners = hand_ranks.iter().filter(|&&r| r == max_rank).count();
                let win_share = 1.0 / num_winners as f64;

                for (i, &rank) in hand_ranks.iter().enumerate() {
                    if rank == max_rank {
                        local_wins[i] += win_share;
                    }
                }
            }

            (local_wins, samples_this_batch)
        })
        .collect();

    let mut total_wins = vec![0.0f64; num_players];
    let mut total_samples = 0usize;

    for (wins, samples) in results {
        for (i, w) in wins.iter().enumerate() {
            total_wins[i] += w;
        }
        total_samples += samples;
    }

    let total = total_samples as f64;
    total_wins.iter().map(|&w| w / total).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::poker::cards::make_card;

    #[test]
    fn test_single_player() {
        let hands = vec![HoleCards(make_card(12, 0), make_card(11, 0))];
        let equity = compute_equity(&hands, 0, 100);
        assert_eq!(equity.len(), 1);
        assert!((equity[0] - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_two_player_equity_sums_to_one() {
        let hands = vec![
            HoleCards(make_card(12, 0), make_card(12, 1)), // AA
            HoleCards(make_card(0, 2), make_card(1, 3)),   // 23o
        ];
        let equity = compute_equity(&hands, 0, 1000);
        assert_eq!(equity.len(), 2);
        let sum: f64 = equity.iter().sum();
        assert!((sum - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_aa_vs_23o_aa_favored() {
        let hands = vec![
            HoleCards(make_card(12, 0), make_card(12, 1)), // AA
            HoleCards(make_card(0, 2), make_card(1, 3)),   // 23o
        ];
        let equity = compute_equity(&hands, 0, 10000);
        // AA should be heavily favored (usually ~85%)
        assert!(equity[0] > 0.80);
    }

    #[test]
    fn test_three_way_equity() {
        let hands = vec![
            HoleCards(make_card(12, 0), make_card(12, 1)), // AA
            HoleCards(make_card(11, 2), make_card(11, 3)), // KK
            HoleCards(make_card(10, 0), make_card(10, 1)), // QQ
        ];
        let equity = compute_equity(&hands, 0, 5000);
        assert_eq!(equity.len(), 3);
        let sum: f64 = equity.iter().sum();
        assert!((sum - 1.0).abs() < 0.01);
        // AA should have the most equity
        assert!(equity[0] > equity[1]);
        assert!(equity[1] > equity[2]);
    }
}
