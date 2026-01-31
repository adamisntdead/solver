//! K-means clustering for hand abstraction.
//!
//! This module provides k-means clustering with multiple distance metrics:
//! - L2 (Euclidean): For EHS (1D) and WinSplit (2D) features
//! - EMD (Earth Mover's Distance): For histogram features (50D)
//!
//! # Algorithm
//!
//! 1. **K-means++ initialization**: Smart seeding where probability of
//!    selecting next center ∝ distance² to nearest existing center
//! 2. **Lloyd's algorithm**: Iteratively assign points to nearest center,
//!    then update centers as mean of assigned points
//! 3. **Multiple restarts**: Run k-means N times, return lowest WCSS solution
//!
//! # Parallelism
//!
//! When the `rayon` feature is enabled, clustering operations are parallelized
//! across multiple cores for faster generation.

use crate::poker::ehs::{emd_distance, EMD_NUM_BINS};

#[cfg(feature = "rand")]
use rand::prelude::*;
#[cfg(feature = "rand")]
use rand::distributions::WeightedIndex;

/// Configuration for k-means clustering.
#[derive(Debug, Clone)]
pub struct KMeansConfig {
    /// Number of clusters (buckets).
    pub num_buckets: usize,
    /// Number of random restarts (best result is kept).
    pub num_restarts: usize,
    /// Maximum iterations per restart.
    pub max_iterations: usize,
    /// Distance metric to use.
    pub distance: DistanceMetric,
    /// Convergence threshold for center movement.
    pub epsilon: f32,
}

impl Default for KMeansConfig {
    fn default() -> Self {
        Self {
            num_buckets: 200,
            num_restarts: 5,
            max_iterations: 100,
            distance: DistanceMetric::L2,
            epsilon: 1e-6,
        }
    }
}

/// Distance metric for clustering.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistanceMetric {
    /// Euclidean distance (L2 norm).
    L2,
    /// Earth Mover's Distance (for histograms).
    EMD,
}

/// Result of k-means clustering.
#[derive(Debug, Clone)]
pub struct KMeansResult<const D: usize> {
    /// Cluster assignments for each point.
    pub assignments: Vec<u16>,
    /// Cluster centers.
    pub centers: Vec<[f32; D]>,
    /// Within-cluster sum of squares (lower is better).
    pub wcss: f64,
    /// Number of iterations until convergence.
    pub iterations: usize,
}

/// K-means clustering for fixed-dimension features.
///
/// # Arguments
/// * `features` - Feature vectors to cluster, one per data point
/// * `config` - Clustering configuration
///
/// # Returns
/// Best clustering result across all restarts.
#[cfg(feature = "rand")]
pub fn kmeans<const D: usize>(
    features: &[[f32; D]],
    config: &KMeansConfig,
) -> KMeansResult<D> {
    let mut best_result: Option<KMeansResult<D>> = None;
    let mut rng = rand::thread_rng();

    for _ in 0..config.num_restarts {
        let result = kmeans_single_run(features, config, &mut rng);

        if best_result.is_none() || result.wcss < best_result.as_ref().unwrap().wcss {
            best_result = Some(result);
        }
    }

    best_result.unwrap()
}

/// Single run of k-means clustering.
#[cfg(feature = "rand")]
fn kmeans_single_run<const D: usize, R: Rng>(
    features: &[[f32; D]],
    config: &KMeansConfig,
    rng: &mut R,
) -> KMeansResult<D> {
    let n = features.len();
    let k = config.num_buckets.min(n);

    // K-means++ initialization
    let mut centers = kmeans_plusplus_init(features, k, config.distance, rng);
    let mut assignments = vec![0u16; n];
    let mut iterations = 0;

    for iter in 0..config.max_iterations {
        iterations = iter + 1;

        // Assignment step: assign each point to nearest center
        let old_assignments = assignments.clone();
        for (i, feature) in features.iter().enumerate() {
            let mut best_cluster = 0;
            let mut best_dist = f32::INFINITY;

            for (c, center) in centers.iter().enumerate() {
                let dist = distance(feature, center, config.distance);
                if dist < best_dist {
                    best_dist = dist;
                    best_cluster = c;
                }
            }
            assignments[i] = best_cluster as u16;
        }

        // Check for convergence
        if assignments == old_assignments {
            break;
        }

        // Update step: recompute centers as mean of assigned points
        let new_centers = compute_centers(features, &assignments, k);

        // Check for center movement convergence
        let mut max_movement = 0.0f32;
        for (old, new) in centers.iter().zip(new_centers.iter()) {
            let movement = distance(old, new, DistanceMetric::L2);
            max_movement = max_movement.max(movement);
        }

        centers = new_centers;

        if max_movement < config.epsilon {
            break;
        }
    }

    // Compute final WCSS
    let wcss = compute_wcss(features, &assignments, &centers, config.distance);

    KMeansResult {
        assignments,
        centers,
        wcss,
        iterations,
    }
}

/// K-means++ initialization.
#[cfg(feature = "rand")]
fn kmeans_plusplus_init<const D: usize, R: Rng>(
    features: &[[f32; D]],
    k: usize,
    metric: DistanceMetric,
    rng: &mut R,
) -> Vec<[f32; D]> {
    let n = features.len();
    let mut centers = Vec::with_capacity(k);

    // Choose first center uniformly at random
    let first_idx = rng.gen_range(0..n);
    centers.push(features[first_idx]);

    // Distance to nearest center for each point
    let mut min_distances = vec![f32::INFINITY; n];

    // Choose remaining centers
    for _ in 1..k {
        // Update distances to nearest center
        for (i, feature) in features.iter().enumerate() {
            let dist = distance(feature, centers.last().unwrap(), metric);
            min_distances[i] = min_distances[i].min(dist);
        }

        // Weight by distance squared
        let weights: Vec<f64> = min_distances.iter().map(|&d| (d * d) as f64).collect();
        let total: f64 = weights.iter().sum();

        if total == 0.0 {
            // All points are centers already
            break;
        }

        // Sample next center with probability proportional to distance²
        let normalized: Vec<f64> = weights.iter().map(|w| w / total).collect();

        let dist = WeightedIndex::new(&normalized).unwrap();
        let next_idx = dist.sample(rng);
        centers.push(features[next_idx]);
    }

    centers
}

/// Compute cluster centers as mean of assigned points.
#[allow(dead_code)]
fn compute_centers<const D: usize>(
    features: &[[f32; D]],
    assignments: &[u16],
    k: usize,
) -> Vec<[f32; D]> {
    let mut centers = vec![[0.0f32; D]; k];
    let mut counts = vec![0usize; k];

    for (feature, &cluster) in features.iter().zip(assignments.iter()) {
        let c = cluster as usize;
        counts[c] += 1;
        for (j, &f) in feature.iter().enumerate() {
            centers[c][j] += f;
        }
    }

    // Divide by count to get mean
    for (center, &count) in centers.iter_mut().zip(counts.iter()) {
        if count > 0 {
            for val in center.iter_mut() {
                *val /= count as f32;
            }
        }
    }

    centers
}

/// Compute within-cluster sum of squares.
#[allow(dead_code)]
fn compute_wcss<const D: usize>(
    features: &[[f32; D]],
    assignments: &[u16],
    centers: &[[f32; D]],
    metric: DistanceMetric,
) -> f64 {
    let mut wcss = 0.0f64;

    for (feature, &cluster) in features.iter().zip(assignments.iter()) {
        let dist = distance(feature, &centers[cluster as usize], metric);
        wcss += (dist * dist) as f64;
    }

    wcss
}

/// Compute distance between two feature vectors.
#[allow(dead_code)]
fn distance<const D: usize>(a: &[f32; D], b: &[f32; D], metric: DistanceMetric) -> f32 {
    match metric {
        DistanceMetric::L2 => l2_distance_generic(a, b),
        DistanceMetric::EMD => {
            // EMD is defined for 50-bin histograms
            if D == EMD_NUM_BINS {
                // SAFETY: We've verified D == EMD_NUM_BINS
                let a_hist: &[f32; EMD_NUM_BINS] = unsafe {
                    &*(a as *const [f32; D] as *const [f32; EMD_NUM_BINS])
                };
                let b_hist: &[f32; EMD_NUM_BINS] = unsafe {
                    &*(b as *const [f32; D] as *const [f32; EMD_NUM_BINS])
                };
                emd_distance(a_hist, b_hist)
            } else {
                // Fall back to L2 for non-histogram data
                l2_distance_generic(a, b)
            }
        }
    }
}

/// L2 (Euclidean) distance.
#[allow(dead_code)]
fn l2_distance_generic<const D: usize>(a: &[f32; D], b: &[f32; D]) -> f32 {
    let mut sum_sq = 0.0f32;
    for i in 0..D {
        let diff = a[i] - b[i];
        sum_sq += diff * diff;
    }
    sum_sq.sqrt()
}

// ============================================================================
// Specialized clustering for 1D EHS
// ============================================================================

/// Optimized k-means for 1D EHS values.
///
/// For 1D data, k-means reduces to finding optimal bucket boundaries.
/// This uses dynamic programming for optimal solution.
#[cfg(feature = "rand")]
pub fn kmeans_1d(values: &[f32], num_buckets: usize) -> KMeans1DResult {
    let n = values.len();
    if n == 0 || num_buckets == 0 {
        return KMeans1DResult {
            assignments: vec![],
            boundaries: vec![],
            centers: vec![],
        };
    }

    // Sort values with original indices
    let mut indexed: Vec<(usize, f32)> = values.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    let k = num_buckets.min(n);

    // For small k, use DP for optimal solution
    if k <= 50 && n <= 10000 {
        return kmeans_1d_dp(&indexed, k);
    }

    // For larger k, use linear binning as approximation
    kmeans_1d_linear(&indexed, k)
}

/// Result of 1D k-means clustering.
#[derive(Debug, Clone)]
pub struct KMeans1DResult {
    /// Cluster assignment for each original point.
    pub assignments: Vec<u16>,
    /// Bucket boundaries (k-1 values).
    pub boundaries: Vec<f32>,
    /// Bucket centers (k values).
    pub centers: Vec<f32>,
}

/// Optimal 1D k-means using dynamic programming.
#[cfg(feature = "rand")]
fn kmeans_1d_dp(sorted: &[(usize, f32)], k: usize) -> KMeans1DResult {
    let n = sorted.len();

    // Prefix sums for computing SSE
    let mut prefix_sum = vec![0.0f64; n + 1];
    let mut prefix_sum_sq = vec![0.0f64; n + 1];

    for (i, &(_, val)) in sorted.iter().enumerate() {
        prefix_sum[i + 1] = prefix_sum[i] + val as f64;
        prefix_sum_sq[i + 1] = prefix_sum_sq[i] + (val as f64 * val as f64);
    }

    // SSE(i, j) = sum of squared deviations from mean for sorted[i..j]
    let sse = |i: usize, j: usize| -> f64 {
        let count = (j - i) as f64;
        if count <= 0.0 {
            return 0.0;
        }
        let sum = prefix_sum[j] - prefix_sum[i];
        let sum_sq = prefix_sum_sq[j] - prefix_sum_sq[i];
        sum_sq - (sum * sum) / count
    };

    // dp[c][i] = min SSE using c clusters for sorted[0..i]
    // backtrack[c][i] = start of last cluster
    let mut dp = vec![vec![f64::INFINITY; n + 1]; k + 1];
    let mut backtrack = vec![vec![0usize; n + 1]; k + 1];

    dp[0][0] = 0.0;

    for c in 1..=k {
        for i in c..=n {
            for j in (c - 1)..i {
                let cost = dp[c - 1][j] + sse(j, i);
                if cost < dp[c][i] {
                    dp[c][i] = cost;
                    backtrack[c][i] = j;
                }
            }
        }
    }

    // Backtrack to find cluster boundaries
    let mut boundaries_idx = Vec::with_capacity(k);
    let mut idx = n;
    for c in (1..=k).rev() {
        let start = backtrack[c][idx];
        if c > 1 {
            boundaries_idx.push(start);
        }
        idx = start;
    }
    boundaries_idx.reverse();

    // Convert to boundary values
    let boundaries: Vec<f32> = boundaries_idx
        .iter()
        .map(|&i| (sorted[i - 1].1 + sorted[i].1) / 2.0)
        .collect();

    // Compute cluster centers and assignments
    let mut centers = Vec::with_capacity(k);
    let mut assignments = vec![0u16; n];

    let mut cluster_bounds = vec![0];
    cluster_bounds.extend(&boundaries_idx);
    cluster_bounds.push(n);

    for c in 0..k {
        let start = cluster_bounds[c];
        let end = cluster_bounds[c + 1];

        // Center is mean of cluster
        let sum: f64 = sorted[start..end].iter().map(|&(_, v)| v as f64).sum();
        let count = (end - start) as f64;
        centers.push(if count > 0.0 {
            (sum / count) as f32
        } else {
            0.0
        });

        // Assign original indices
        for &(orig_idx, _) in &sorted[start..end] {
            assignments[orig_idx] = c as u16;
        }
    }

    KMeans1DResult {
        assignments,
        boundaries,
        centers,
    }
}

/// Linear binning for large datasets.
#[cfg(feature = "rand")]
fn kmeans_1d_linear(sorted: &[(usize, f32)], k: usize) -> KMeans1DResult {
    let n = sorted.len();
    let points_per_bucket = n / k;

    let mut boundaries = Vec::with_capacity(k - 1);
    let mut centers = Vec::with_capacity(k);
    let mut assignments = vec![0u16; n];

    for c in 0..k {
        let start = c * points_per_bucket;
        let end = if c == k - 1 {
            n
        } else {
            (c + 1) * points_per_bucket
        };

        // Boundary is midpoint between last of this and first of next
        if c < k - 1 {
            let boundary = (sorted[end - 1].1 + sorted[end].1) / 2.0;
            boundaries.push(boundary);
        }

        // Center is mean
        let sum: f64 = sorted[start..end].iter().map(|&(_, v)| v as f64).sum();
        let count = (end - start) as f64;
        centers.push((sum / count) as f32);

        // Assignments
        for &(orig_idx, _) in &sorted[start..end] {
            assignments[orig_idx] = c as u16;
        }
    }

    KMeans1DResult {
        assignments,
        boundaries,
        centers,
    }
}

// ============================================================================
// Clustering for 2D WinSplit features
// ============================================================================

/// K-means for 2D features (WinSplit: win freq, split freq).
#[cfg(feature = "rand")]
pub fn kmeans_2d(features: &[[f32; 2]], config: &KMeansConfig) -> KMeansResult<2> {
    kmeans::<2>(features, config)
}

// ============================================================================
// Clustering for EMD histograms
// ============================================================================

/// K-means for EMD histogram features.
#[cfg(feature = "rand")]
pub fn kmeans_emd(
    histograms: &[[f32; EMD_NUM_BINS]],
    config: &KMeansConfig,
) -> KMeansResult<EMD_NUM_BINS> {
    let mut emd_config = config.clone();
    emd_config.distance = DistanceMetric::EMD;
    kmeans::<EMD_NUM_BINS>(histograms, &emd_config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(feature = "rand")]
    fn test_kmeans_1d() {
        // Simple test: cluster values into 3 groups
        let values = vec![0.1, 0.15, 0.2, 0.5, 0.55, 0.6, 0.9, 0.92, 0.95];
        let result = kmeans_1d(&values, 3);

        assert_eq!(result.assignments.len(), 9);
        assert_eq!(result.centers.len(), 3);
        assert_eq!(result.boundaries.len(), 2);

        // Low values should be in cluster 0
        assert_eq!(result.assignments[0], result.assignments[1]);
        assert_eq!(result.assignments[1], result.assignments[2]);

        // Mid values should be in cluster 1
        assert_eq!(result.assignments[3], result.assignments[4]);
        assert_eq!(result.assignments[4], result.assignments[5]);

        // High values should be in cluster 2
        assert_eq!(result.assignments[6], result.assignments[7]);
        assert_eq!(result.assignments[7], result.assignments[8]);
    }

    #[test]
    #[cfg(feature = "rand")]
    fn test_kmeans_2d() {
        // Create two obvious clusters
        let mut features = Vec::new();
        for _ in 0..50 {
            features.push([0.1, 0.1]);
        }
        for _ in 0..50 {
            features.push([0.9, 0.9]);
        }

        let config = KMeansConfig {
            num_buckets: 2,
            num_restarts: 3,
            max_iterations: 50,
            distance: DistanceMetric::L2,
            epsilon: 1e-6,
        };

        let result = kmeans_2d(&features, &config);

        assert_eq!(result.assignments.len(), 100);
        assert_eq!(result.centers.len(), 2);

        // All points in first group should have same assignment
        let first_cluster = result.assignments[0];
        for i in 0..50 {
            assert_eq!(result.assignments[i], first_cluster);
        }

        // All points in second group should have same (different) assignment
        let second_cluster = result.assignments[50];
        assert_ne!(first_cluster, second_cluster);
        for i in 50..100 {
            assert_eq!(result.assignments[i], second_cluster);
        }
    }

    #[test]
    fn test_l2_distance() {
        let a = [1.0f32, 2.0, 3.0];
        let b = [4.0f32, 6.0, 3.0];

        let dist = l2_distance_generic(&a, &b);
        // sqrt((4-1)^2 + (6-2)^2 + (3-3)^2) = sqrt(9 + 16 + 0) = 5
        assert!((dist - 5.0).abs() < 0.001);
    }

    #[test]
    fn test_compute_centers() {
        let features = vec![
            [1.0f32, 1.0],
            [2.0, 2.0],
            [10.0, 10.0],
            [11.0, 11.0],
        ];
        let assignments = vec![0u16, 0, 1, 1];

        let centers = compute_centers(&features, &assignments, 2);

        // Cluster 0: mean of [1,1] and [2,2] = [1.5, 1.5]
        assert!((centers[0][0] - 1.5).abs() < 0.001);
        assert!((centers[0][1] - 1.5).abs() < 0.001);

        // Cluster 1: mean of [10,10] and [11,11] = [10.5, 10.5]
        assert!((centers[1][0] - 10.5).abs() < 0.001);
        assert!((centers[1][1] - 10.5).abs() < 0.001);
    }
}
