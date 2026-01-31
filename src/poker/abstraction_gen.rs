//! Abstraction generation pipeline.
//!
//! This module generates hand abstractions by:
//! 1. Enumerating all boards for a street
//! 2. Computing features (EHS, EMD, etc.) for each (board, hand) pair
//! 3. Running k-means clustering to group similar hands
//! 4. Returning bucket assignments for serialization
//!
//! # Generation Process
//!
//! For each street, we:
//! 1. Enumerate all canonical boards (suit-isomorphic)
//! 2. For each board, compute features for all isomorphic hands
//! 3. Run k-means on the combined feature space
//! 4. Store bucket assignments indexed by imperfect recall index
//!
//! # Performance
//!
//! Generation is expensive (especially for flop with EMD features).
//! Use `--release` mode and consider parallelization with rayon.

#[cfg(feature = "rand")]
use crate::poker::abstraction::{
    AggSIAbstraction, EHSAbstraction, EMDAbstraction, InfoAbstraction, SemiAggSIAbstraction,
    WinSplitAbstraction,
};
#[cfg(feature = "rand")]
use crate::poker::clustering::{kmeans_1d, kmeans_2d, kmeans_emd, KMeansConfig};
#[cfg(feature = "rand")]
use crate::poker::ehs::{compute_all_ehs, compute_asymmetric_emd_features, compute_emd_features, compute_winsplit_features};
use crate::poker::ehs::EMD_NUM_BINS;
#[cfg(feature = "rand")]
use crate::poker::hands::{Board, Combo};
#[cfg(feature = "rand")]
use crate::poker::indexer::{BoardEnumerator, SingleBoardIndexer};
use crate::poker::indexer::Street;

/// Abstraction type to generate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AbstractionType {
    /// Expected Hand Strength (1D clustering).
    EHS,
    /// EHS squared (1D clustering, captures variance).
    EHSSquared,
    /// Earth Mover's Distance on equity histogram (50D).
    EMD,
    /// Asymmetric EMD with finer bins at high equity.
    AsymEMD,
    /// Win/split frequency (2D, river only).
    WinSplit,
    /// Aggressive suit isomorphism (deterministic).
    AggSI,
    /// Semi-aggressive suit isomorphism (flop only).
    SemiAggSI,
}

impl AbstractionType {
    /// Whether this type requires clustering.
    pub fn requires_clustering(&self) -> bool {
        matches!(
            self,
            AbstractionType::EHS
                | AbstractionType::EHSSquared
                | AbstractionType::EMD
                | AbstractionType::AsymEMD
                | AbstractionType::WinSplit
        )
    }

    /// Whether this type is valid for a given street.
    pub fn is_valid_for_street(&self, street: Street) -> bool {
        match self {
            AbstractionType::EHS | AbstractionType::EHSSquared => true, // All streets
            AbstractionType::EMD | AbstractionType::AsymEMD => {
                // Not river (no runouts to histogram)
                matches!(street, Street::Flop | Street::Turn)
            }
            AbstractionType::WinSplit => {
                // River only
                matches!(street, Street::River)
            }
            AbstractionType::AggSI => {
                // Postflop only (needs board)
                !matches!(street, Street::Preflop)
            }
            AbstractionType::SemiAggSI => {
                // Flop only
                matches!(street, Street::Flop)
            }
        }
    }

    /// Name for display.
    pub fn name(&self) -> &'static str {
        match self {
            AbstractionType::EHS => "EHS",
            AbstractionType::EHSSquared => "EHS²",
            AbstractionType::EMD => "EMD",
            AbstractionType::AsymEMD => "AsymEMD",
            AbstractionType::WinSplit => "WinSplit",
            AbstractionType::AggSI => "AggSI",
            AbstractionType::SemiAggSI => "SemiAggSI",
        }
    }
}

/// Configuration for abstraction generation.
pub struct AbstractionConfig {
    /// Street to generate for.
    pub street: Street,
    /// Abstraction type.
    pub abstraction_type: AbstractionType,
    /// Number of buckets (for clustering types).
    pub num_buckets: usize,
    /// Number of k-means restarts.
    pub num_restarts: usize,
    /// Maximum k-means iterations per restart.
    pub max_iterations: usize,
    /// Use EHS² instead of EHS for EHS type.
    pub ehs_squared: bool,
    /// Progress callback (called with percentage 0-100).
    pub progress_callback: Option<Box<dyn Fn(u32) + Send + Sync>>,
}

impl Default for AbstractionConfig {
    fn default() -> Self {
        Self {
            street: Street::River,
            abstraction_type: AbstractionType::EHS,
            num_buckets: 200,
            num_restarts: 5,
            max_iterations: 100,
            ehs_squared: false,
            progress_callback: None,
        }
    }
}

/// Result of abstraction generation.
#[derive(Debug)]
pub struct GeneratedAbstraction {
    /// Street this abstraction is for.
    pub street: Street,
    /// Abstraction type.
    pub abstraction_type: AbstractionType,
    /// Number of buckets.
    pub num_buckets: usize,
    /// Bucket assignments indexed by global index.
    /// assignments[global_idx] = bucket
    pub assignments: Vec<u16>,
    /// Optional cluster centers (for visualization).
    pub centers: Option<CenterData>,
    /// Number of canonical boards processed.
    pub num_boards: usize,
}

/// Cluster center data for different abstraction types.
#[derive(Debug, Clone)]
pub enum CenterData {
    /// 1D centers (EHS, EHS²).
    Scalar(Vec<f32>),
    /// 2D centers (WinSplit).
    TwoD(Vec<[f32; 2]>),
    /// 50D centers (EMD histograms).
    Histogram(Vec<[f32; EMD_NUM_BINS]>),
}

/// Generate abstraction for a single board.
///
/// This is useful for testing or for generating abstractions on-the-fly.
#[cfg(feature = "rand")]
pub fn generate_for_board(
    board: &Board,
    abstraction_type: AbstractionType,
    num_buckets: usize,
) -> Box<dyn InfoAbstraction> {
    match abstraction_type {
        AbstractionType::AggSI => Box::new(AggSIAbstraction::new(board)),
        AbstractionType::SemiAggSI => Box::new(SemiAggSIAbstraction::new(board)),
        AbstractionType::EHS | AbstractionType::EHSSquared => {
            let ehs = if abstraction_type == AbstractionType::EHSSquared {
                crate::poker::ehs::compute_all_ehs_squared(&board.cards)
            } else {
                compute_all_ehs(&board.cards)
            };

            // Get valid EHS values
            let indexer = SingleBoardIndexer::new(board);
            let mut valid_ehs = Vec::with_capacity(indexer.num_iso_hands());
            let mut iso_to_combo: Vec<usize> = Vec::with_capacity(indexer.num_iso_hands());

            for (combo_idx, bucket) in indexer.valid_combos() {
                if bucket as usize >= valid_ehs.len() {
                    valid_ehs.resize(bucket as usize + 1, 0.0);
                    iso_to_combo.resize(bucket as usize + 1, 0);
                }
                if valid_ehs[bucket as usize] == 0.0 {
                    valid_ehs[bucket as usize] = ehs[combo_idx];
                    iso_to_combo[bucket as usize] = combo_idx;
                }
            }

            // Cluster
            let result = kmeans_1d(&valid_ehs, num_buckets);

            Box::new(EHSAbstraction::single_context(
                num_buckets,
                result.assignments,
                Some(result.centers),
            ))
        }
        AbstractionType::WinSplit => {
            assert!(
                board.cards.len() == 5,
                "WinSplit requires 5-card board"
            );

            let indexer = SingleBoardIndexer::new(board);
            let mut features = Vec::with_capacity(indexer.num_iso_hands());

            for (combo_idx, _) in indexer.valid_combos() {
                let combo = Combo::from_index(combo_idx);
                features.push(compute_winsplit_features(combo, &board.cards));
            }

            let config = KMeansConfig {
                num_buckets,
                num_restarts: 5,
                max_iterations: 100,
                ..Default::default()
            };

            let result = kmeans_2d(&features, &config);

            Box::new(WinSplitAbstraction::new(
                num_buckets,
                result.assignments,
                Some(result.centers.to_vec()),
            ))
        }
        AbstractionType::EMD | AbstractionType::AsymEMD => {
            assert!(
                board.cards.len() <= 4,
                "EMD requires non-river board"
            );

            let indexer = SingleBoardIndexer::new(board);
            let mut features = Vec::with_capacity(indexer.num_iso_hands());

            for (combo_idx, _) in indexer.valid_combos() {
                let combo = Combo::from_index(combo_idx);
                let hist = if abstraction_type == AbstractionType::AsymEMD {
                    compute_asymmetric_emd_features(combo, &board.cards)
                } else {
                    compute_emd_features(combo, &board.cards)
                };
                features.push(hist);
            }

            let result = kmeans_emd(
                &features,
                &KMeansConfig {
                    num_buckets,
                    num_restarts: 5,
                    max_iterations: 100,
                    distance: crate::poker::clustering::DistanceMetric::EMD,
                    ..Default::default()
                },
            );

            Box::new(EMDAbstraction::single_context(
                num_buckets,
                result.assignments,
                Some(result.centers.to_vec()),
            ))
        }
    }
}

/// Generate abstraction for all boards on a street.
///
/// This is the main generation function for pre-building abstractions.
#[cfg(feature = "rand")]
pub fn generate_street_abstraction(config: &AbstractionConfig) -> GeneratedAbstraction {
    assert!(
        config.abstraction_type.is_valid_for_street(config.street),
        "{} is not valid for {}",
        config.abstraction_type.name(),
        config.street.name()
    );

    // Enumerate all canonical boards
    let enumerator = BoardEnumerator::new(config.street);
    let boards = enumerator.enumerate_canonical();

    // For deterministic types, process boards individually
    if !config.abstraction_type.requires_clustering() {
        return generate_deterministic_abstraction(config, &boards);
    }

    // For clustering types, collect all features then cluster
    match config.abstraction_type {
        AbstractionType::EHS | AbstractionType::EHSSquared => {
            generate_ehs_abstraction(config, &boards)
        }
        AbstractionType::WinSplit => generate_winsplit_abstraction(config, &boards),
        AbstractionType::EMD | AbstractionType::AsymEMD => {
            generate_emd_abstraction(config, &boards)
        }
        _ => unreachable!(),
    }
}

/// Generate deterministic abstraction (AggSI, SemiAggSI).
#[cfg(feature = "rand")]
fn generate_deterministic_abstraction(
    config: &AbstractionConfig,
    boards: &[crate::poker::indexer::CanonicalBoard],
) -> GeneratedAbstraction {
    let mut all_assignments = Vec::new();
    let mut max_bucket = 0u16;
    for (i, cb) in boards.iter().enumerate() {
        let board = cb.to_board();
        let indexer = SingleBoardIndexer::new(&board);

        let abs: Box<dyn InfoAbstraction> = match config.abstraction_type {
            AbstractionType::AggSI => Box::new(AggSIAbstraction::new(&board)),
            AbstractionType::SemiAggSI => Box::new(SemiAggSIAbstraction::new(&board)),
            _ => unreachable!(),
        };

        // Map iso hands to buckets
        for iso_hand in 0..indexer.num_iso_hands() {
            let bucket = abs.bucket(iso_hand, 0) as u16;
            all_assignments.push(bucket);
            max_bucket = max_bucket.max(bucket);
        }

        // Progress
        if let Some(ref callback) = config.progress_callback {
            callback((i * 100 / boards.len()) as u32);
        }
    }

    GeneratedAbstraction {
        street: config.street,
        abstraction_type: config.abstraction_type,
        num_buckets: max_bucket as usize + 1,
        assignments: all_assignments,
        centers: None,
        num_boards: boards.len(),
    }
}

/// Generate EHS-based abstraction.
#[cfg(feature = "rand")]
fn generate_ehs_abstraction(
    config: &AbstractionConfig,
    boards: &[crate::poker::indexer::CanonicalBoard],
) -> GeneratedAbstraction {
    // Collect all EHS values
    let mut all_ehs = Vec::new();
    let mut board_ranges: Vec<(usize, usize)> = Vec::new(); // (start, num_hands) per board

    for (i, cb) in boards.iter().enumerate() {
        let board = cb.to_board();
        let ehs_values = if config.ehs_squared || config.abstraction_type == AbstractionType::EHSSquared {
            crate::poker::ehs::compute_all_ehs_squared(&board.cards)
        } else {
            compute_all_ehs(&board.cards)
        };

        let indexer = SingleBoardIndexer::new(&board);
        let start = all_ehs.len();

        // Collect EHS for each isomorphic hand
        let mut iso_ehs = vec![0.0f32; indexer.num_iso_hands()];
        for (combo_idx, bucket) in indexer.valid_combos() {
            if iso_ehs[bucket as usize] == 0.0 {
                iso_ehs[bucket as usize] = ehs_values[combo_idx];
            }
        }

        all_ehs.extend(iso_ehs);
        board_ranges.push((start, indexer.num_iso_hands()));

        if let Some(ref cb) = config.progress_callback {
            cb((i * 50 / boards.len()) as u32);
        }
    }

    // Cluster all EHS values
    let result = kmeans_1d(&all_ehs, config.num_buckets);

    if let Some(ref cb) = config.progress_callback {
        cb(75);
    }

    GeneratedAbstraction {
        street: config.street,
        abstraction_type: config.abstraction_type,
        num_buckets: config.num_buckets,
        assignments: result.assignments,
        centers: Some(CenterData::Scalar(result.centers)),
        num_boards: boards.len(),
    }
}

/// Generate WinSplit abstraction (river only).
#[cfg(feature = "rand")]
fn generate_winsplit_abstraction(
    config: &AbstractionConfig,
    boards: &[crate::poker::indexer::CanonicalBoard],
) -> GeneratedAbstraction {
    let mut all_features: Vec<[f32; 2]> = Vec::new();

    for (i, cb) in boards.iter().enumerate() {
        let board = cb.to_board();
        let indexer = SingleBoardIndexer::new(&board);

        for (combo_idx, _) in indexer.valid_combos() {
            let combo = Combo::from_index(combo_idx);
            let features = compute_winsplit_features(combo, &board.cards);
            all_features.push(features);
        }

        if let Some(ref cb) = config.progress_callback {
            cb((i * 50 / boards.len()) as u32);
        }
    }

    let kmeans_config = KMeansConfig {
        num_buckets: config.num_buckets,
        num_restarts: config.num_restarts,
        max_iterations: config.max_iterations,
        ..Default::default()
    };

    let result = kmeans_2d(&all_features, &kmeans_config);

    GeneratedAbstraction {
        street: config.street,
        abstraction_type: config.abstraction_type,
        num_buckets: config.num_buckets,
        assignments: result.assignments,
        centers: Some(CenterData::TwoD(result.centers.to_vec())),
        num_boards: boards.len(),
    }
}

/// Generate EMD-based abstraction.
#[cfg(feature = "rand")]
fn generate_emd_abstraction(
    config: &AbstractionConfig,
    boards: &[crate::poker::indexer::CanonicalBoard],
) -> GeneratedAbstraction {
    let mut all_features: Vec<[f32; EMD_NUM_BINS]> = Vec::new();

    for (i, cb) in boards.iter().enumerate() {
        let board = cb.to_board();
        let indexer = SingleBoardIndexer::new(&board);

        for (combo_idx, _) in indexer.valid_combos() {
            let combo = Combo::from_index(combo_idx);
            let hist = if config.abstraction_type == AbstractionType::AsymEMD {
                compute_asymmetric_emd_features(combo, &board.cards)
            } else {
                compute_emd_features(combo, &board.cards)
            };
            all_features.push(hist);
        }

        if let Some(ref cb) = config.progress_callback {
            cb((i * 50 / boards.len()) as u32);
        }
    }

    let kmeans_config = KMeansConfig {
        num_buckets: config.num_buckets,
        num_restarts: config.num_restarts,
        max_iterations: config.max_iterations,
        distance: crate::poker::clustering::DistanceMetric::EMD,
        ..Default::default()
    };

    let result = kmeans_emd(&all_features, &kmeans_config);

    GeneratedAbstraction {
        street: config.street,
        abstraction_type: config.abstraction_type,
        num_buckets: config.num_buckets,
        assignments: result.assignments,
        centers: Some(CenterData::Histogram(result.centers.to_vec())),
        num_boards: boards.len(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::poker::board_parser::parse_board;

    #[test]
    fn test_abstraction_type_validity() {
        assert!(AbstractionType::EHS.is_valid_for_street(Street::River));
        assert!(AbstractionType::EHS.is_valid_for_street(Street::Flop));
        assert!(AbstractionType::WinSplit.is_valid_for_street(Street::River));
        assert!(!AbstractionType::WinSplit.is_valid_for_street(Street::Flop));
        assert!(AbstractionType::EMD.is_valid_for_street(Street::Turn));
        assert!(!AbstractionType::EMD.is_valid_for_street(Street::River));
    }

    #[test]
    #[cfg(feature = "rand")]
    fn test_generate_for_board_ehs() {
        let board = parse_board("KhQsJs2c3d").unwrap();
        let abs = generate_for_board(&board, AbstractionType::EHS, 50);

        assert!(abs.num_buckets() <= 50);
    }

    #[test]
    #[cfg(feature = "rand")]
    fn test_generate_for_board_aggsi() {
        let board = parse_board("KhQsJs2c3d").unwrap();
        let abs = generate_for_board(&board, AbstractionType::AggSI, 50);

        // AggSI ignores num_buckets (deterministic)
        assert!(abs.num_buckets() > 0);
    }

    #[test]
    #[cfg(feature = "rand")]
    fn test_generate_for_board_winsplit() {
        let board = parse_board("KhQsJs2c3d").unwrap();
        let abs = generate_for_board(&board, AbstractionType::WinSplit, 20);

        assert!(abs.num_buckets() <= 20);
    }
}
