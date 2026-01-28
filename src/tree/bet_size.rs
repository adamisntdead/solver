//! Bet size specification and parsing.
//!
//! Supports Pio/Monker-style bet size formats:
//! - `X%` - Percentage of pot (e.g., "50%", "150%")
//! - `Xx` - Multiple of previous bet/raise (e.g., "2.5x", "3x")
//! - `Xc` - Fixed chip amount (e.g., "100c")
//! - `Xe` - Geometric sizing for X streets (e.g., "2e", "3e150%")
//! - `a` - All-in

use std::cmp::Ordering;
use std::str::FromStr;

/// A bet size specification.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum BetSize {
    /// Percentage of pot: "50%", "100%", "150%"
    PotRelative(f64),

    /// Multiple of previous bet/raise: "2x", "3.5x" (raises only)
    PrevBetRelative(f64),

    /// Fixed chip amount: "100c"
    Additive(i32),

    /// Geometric sizing for N streets: "2e", "3e", "2e200%"
    /// Calculates bet to reach all-in over N geometric bets.
    /// If num_streets is 0, it means "auto" (3 for flop, 2 for turn, 1 for river).
    /// max_pot_pct is the maximum pot percentage (f64::INFINITY if unlimited).
    Geometric {
        num_streets: u8,
        max_pot_pct: f64,
    },

    /// All-in
    AllIn,
}

impl PartialOrd for BetSize {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        // Define ordering for sorting bet sizes
        let rank = |bs: &BetSize| -> (u8, f64) {
            match bs {
                BetSize::PotRelative(r) => (0, *r),
                BetSize::PrevBetRelative(r) => (1, *r),
                BetSize::Additive(c) => (2, *c as f64),
                BetSize::Geometric { num_streets, .. } => (3, *num_streets as f64),
                BetSize::AllIn => (4, 0.0),
            }
        };
        let (r1, v1) = rank(self);
        let (r2, v2) = rank(other);
        match r1.cmp(&r2) {
            Ordering::Equal => v1.partial_cmp(&v2),
            other => Some(other),
        }
    }
}

impl BetSize {
    /// Resolve the bet size to a chip amount.
    ///
    /// # Arguments
    /// * `pot` - Current pot size
    /// * `to_call` - Amount to call (0 if opening bet)
    /// * `prev_bet` - Previous bet/raise amount (for PrevBetRelative)
    /// * `stack` - Remaining stack
    /// * `streets_remaining` - Streets remaining (for auto geometric)
    pub fn resolve(
        &self,
        pot: i32,
        to_call: i32,
        prev_bet: i32,
        stack: i32,
        streets_remaining: u8,
    ) -> i32 {
        match self {
            BetSize::PotRelative(ratio) => {
                // Bet = ratio * (pot + to_call)
                // The "pot" after we call
                let effective_pot = pot + to_call;
                ((effective_pot as f64) * ratio).round() as i32
            }
            BetSize::PrevBetRelative(ratio) => {
                // Raise TO prev_bet * ratio
                ((prev_bet as f64) * ratio).round() as i32
            }
            BetSize::Additive(chips) => *chips,
            BetSize::Geometric {
                num_streets,
                max_pot_pct,
            } => {
                let n = if *num_streets == 0 {
                    streets_remaining
                } else {
                    *num_streets
                };
                compute_geometric(pot, to_call, stack, n, *max_pot_pct)
            }
            BetSize::AllIn => stack,
        }
    }
}

/// Compute geometric bet size.
///
/// The geometric sizing calculates the bet amount that, if repeated n times
/// with the same pot-relative percentage, would result in all-in.
///
/// Formula: bet = pot * ((2 * SPR + 1)^(1/n) - 1) / 2
/// where SPR = stack / pot after call
fn compute_geometric(pot: i32, to_call: i32, stack: i32, n: u8, max_pot_pct: f64) -> i32 {
    if n == 0 {
        return stack; // All-in
    }

    let effective_pot = (pot + to_call) as f64;
    let effective_stack = (stack - to_call).max(0) as f64;

    if effective_stack <= 0.0 {
        return 0;
    }

    // SPR after calling
    let spr = effective_stack / effective_pot;

    // Geometric ratio
    let ratio = ((2.0 * spr + 1.0).powf(1.0 / n as f64) - 1.0) / 2.0;
    let capped_ratio = ratio.min(max_pot_pct);

    (effective_pot * capped_ratio).round() as i32
}

impl FromStr for BetSize {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        parse_bet_size(s, true)
    }
}

/// Parse a bet size string.
///
/// # Arguments
/// * `s` - The string to parse
/// * `allow_raise_rel` - Whether to allow PrevBetRelative (Xx format)
pub fn parse_bet_size(s: &str, allow_raise_rel: bool) -> Result<BetSize, String> {
    let s = s.trim().to_lowercase();
    let err_msg = || format!("Invalid bet size: '{}'", s);

    if s.is_empty() {
        return Err(err_msg());
    }

    // All-in: "a"
    if s == "a" {
        return Ok(BetSize::AllIn);
    }

    // Previous bet relative: "Xx" (e.g., "2.5x")
    if let Some(num_str) = s.strip_suffix('x') {
        if !allow_raise_rel {
            return Err(format!(
                "Relative size to previous bet not allowed for donk/open bets: '{}'",
                s
            ));
        }
        let value: f64 = parse_positive_float(num_str).ok_or_else(err_msg)?;
        if value <= 1.0 {
            return Err(format!("Multiplier must be greater than 1.0: '{}'", s));
        }
        return Ok(BetSize::PrevBetRelative(value));
    }

    // Additive (fixed chips): "Xc" (e.g., "100c")
    if let Some(num_str) = s.strip_suffix('c') {
        let value: f64 = parse_positive_float(num_str).ok_or_else(err_msg)?;
        if value.fract() != 0.0 {
            return Err(format!("Chip amount must be an integer: '{}'", s));
        }
        if value > i32::MAX as f64 {
            return Err(format!("Chip amount too large: '{}'", s));
        }
        return Ok(BetSize::Additive(value as i32));
    }

    // Geometric: "e", "Xe", "Xe%Y" (e.g., "2e", "3e150%")
    if s.contains('e') {
        let parts: Vec<&str> = s.split('e').collect();
        if parts.len() != 2 {
            return Err(err_msg());
        }

        let num_streets = if parts[0].is_empty() {
            0 // Auto
        } else {
            let n: f64 = parse_positive_float(parts[0]).ok_or_else(err_msg)?;
            if n.fract() != 0.0 || n < 1.0 || n > 100.0 {
                return Err(format!(
                    "Number of streets must be a positive integer (1-100): '{}'",
                    s
                ));
            }
            n as u8
        };

        let max_pot_pct = if parts[1].is_empty() {
            f64::INFINITY
        } else {
            let pct_str = parts[1]
                .strip_suffix('%')
                .ok_or_else(|| format!("Expected percentage after 'e': '{}'", s))?;
            let pct: f64 = parse_positive_float(pct_str).ok_or_else(err_msg)?;
            pct / 100.0
        };

        return Ok(BetSize::Geometric {
            num_streets,
            max_pot_pct,
        });
    }

    // Pot relative: "X%" (e.g., "50%", "150%")
    if let Some(num_str) = s.strip_suffix('%') {
        let value: f64 = parse_positive_float(num_str).ok_or_else(err_msg)?;
        return Ok(BetSize::PotRelative(value / 100.0));
    }

    Err(err_msg())
}

/// Parse a positive float, rejecting negative numbers and invalid formats.
fn parse_positive_float(s: &str) -> Option<f64> {
    if s.is_empty() || s.starts_with('-') || s.starts_with('+') {
        return None;
    }
    // Reject if contains letters (except for the format suffixes already stripped)
    if s.chars().any(|c| c.is_ascii_alphabetic()) {
        return None;
    }
    s.parse::<f64>().ok().filter(|&v| v >= 0.0 && v.is_finite())
}

/// Bet size options for different action types.
#[derive(Debug, Clone, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct BetSizeOptions {
    /// Sizes for initial bets (when facing check)
    pub bet: Vec<BetSize>,

    /// Sizes for first raise
    pub raise: Vec<BetSize>,

    /// Sizes for second raise (3-bet preflop)
    pub reraise: Vec<BetSize>,

    /// Sizes for third+ raises (4-bet+ preflop)
    pub reraise_plus: Vec<BetSize>,
}

impl BetSizeOptions {
    /// Create new bet size options with the same sizes for all action types.
    pub fn uniform(sizes: Vec<BetSize>) -> Self {
        Self {
            bet: sizes.clone(),
            raise: sizes.clone(),
            reraise: sizes.clone(),
            reraise_plus: sizes,
        }
    }

    /// Parse bet size options from comma-separated strings.
    ///
    /// # Arguments
    /// * `bet` - Bet sizes (e.g., "50%, 75%, 100%")
    /// * `raise` - Raise sizes (e.g., "2.5x, 3x, a")
    pub fn try_from_strs(bet: &str, raise: &str) -> Result<Self, String> {
        let bet_sizes = parse_size_list(bet, false)?;
        let raise_sizes = parse_size_list(raise, true)?;

        Ok(Self {
            bet: bet_sizes.clone(),
            raise: raise_sizes.clone(),
            reraise: raise_sizes.clone(),
            reraise_plus: raise_sizes,
        })
    }

    /// Get the appropriate sizes for the given raise count.
    pub fn sizes_for_raise_count(&self, raise_count: u8) -> &[BetSize] {
        match raise_count {
            0 => &self.bet,
            1 => &self.raise,
            2 => &self.reraise,
            _ => &self.reraise_plus,
        }
    }
}

/// Parse a comma-separated list of bet sizes.
fn parse_size_list(s: &str, allow_raise_rel: bool) -> Result<Vec<BetSize>, String> {
    let mut sizes: Vec<BetSize> = s
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| parse_bet_size(s, allow_raise_rel))
        .collect::<Result<Vec<_>, _>>()?;

    sizes.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    Ok(sizes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_pot_relative() {
        assert_eq!(
            parse_bet_size("50%", true),
            Ok(BetSize::PotRelative(0.5))
        );
        assert_eq!(
            parse_bet_size("100%", true),
            Ok(BetSize::PotRelative(1.0))
        );
        assert_eq!(
            parse_bet_size("150%", true),
            Ok(BetSize::PotRelative(1.5))
        );
        assert_eq!(
            parse_bet_size("0%", true),
            Ok(BetSize::PotRelative(0.0))
        );
        // Test that 33.3% parses (don't compare exact float)
        let result = parse_bet_size("33.3%", true);
        assert!(result.is_ok());
        if let Ok(BetSize::PotRelative(r)) = result {
            assert!((r - 0.333).abs() < 0.0001);
        }
    }

    #[test]
    fn test_parse_prev_bet_relative() {
        assert_eq!(
            parse_bet_size("2x", true),
            Ok(BetSize::PrevBetRelative(2.0))
        );
        assert_eq!(
            parse_bet_size("2.5x", true),
            Ok(BetSize::PrevBetRelative(2.5))
        );
        assert_eq!(
            parse_bet_size("3.0X", true),
            Ok(BetSize::PrevBetRelative(3.0))
        );

        // Not allowed for bets
        assert!(parse_bet_size("2x", false).is_err());

        // Must be > 1.0
        assert!(parse_bet_size("1x", true).is_err());
        assert!(parse_bet_size("0.5x", true).is_err());
    }

    #[test]
    fn test_parse_additive() {
        assert_eq!(parse_bet_size("100c", true), Ok(BetSize::Additive(100)));
        assert_eq!(parse_bet_size("0c", true), Ok(BetSize::Additive(0)));
        assert_eq!(parse_bet_size("500C", true), Ok(BetSize::Additive(500)));

        // Must be integer
        assert!(parse_bet_size("100.5c", true).is_err());
    }

    #[test]
    fn test_parse_geometric() {
        assert_eq!(
            parse_bet_size("e", true),
            Ok(BetSize::Geometric {
                num_streets: 0,
                max_pot_pct: f64::INFINITY
            })
        );
        assert_eq!(
            parse_bet_size("2e", true),
            Ok(BetSize::Geometric {
                num_streets: 2,
                max_pot_pct: f64::INFINITY
            })
        );
        assert_eq!(
            parse_bet_size("3e150%", true),
            Ok(BetSize::Geometric {
                num_streets: 3,
                max_pot_pct: 1.5
            })
        );
        assert_eq!(
            parse_bet_size("E", true),
            Ok(BetSize::Geometric {
                num_streets: 0,
                max_pot_pct: f64::INFINITY
            })
        );
    }

    #[test]
    fn test_parse_all_in() {
        assert_eq!(parse_bet_size("a", true), Ok(BetSize::AllIn));
        assert_eq!(parse_bet_size("A", true), Ok(BetSize::AllIn));
    }

    #[test]
    fn test_parse_errors() {
        assert!(parse_bet_size("", true).is_err());
        assert!(parse_bet_size("abc", true).is_err());
        assert!(parse_bet_size("50", true).is_err());
        assert!(parse_bet_size("-50%", true).is_err());
        assert!(parse_bet_size("x", true).is_err());
    }

    #[test]
    fn test_resolve_pot_relative() {
        let size = BetSize::PotRelative(0.5);
        // pot=100, to_call=0, stack=1000
        assert_eq!(size.resolve(100, 0, 0, 1000, 3), 50);
        // pot=100, to_call=50 -> effective pot = 150
        assert_eq!(size.resolve(100, 50, 0, 1000, 3), 75);
    }

    #[test]
    fn test_resolve_prev_bet_relative() {
        let size = BetSize::PrevBetRelative(2.5);
        // Previous bet was 100, raise to 250
        assert_eq!(size.resolve(200, 100, 100, 1000, 3), 250);
    }

    #[test]
    fn test_resolve_additive() {
        let size = BetSize::Additive(200);
        assert_eq!(size.resolve(100, 0, 0, 1000, 3), 200);
    }

    #[test]
    fn test_resolve_all_in() {
        let size = BetSize::AllIn;
        assert_eq!(size.resolve(100, 0, 0, 500, 3), 500);
    }

    #[test]
    fn test_bet_size_options() {
        let opts = BetSizeOptions::try_from_strs("50%, 75%, 100%", "2.5x, 3x, a").unwrap();
        assert_eq!(opts.bet.len(), 3);
        assert_eq!(opts.raise.len(), 3);
    }

    #[test]
    fn test_compute_geometric() {
        // pot=100, to_call=0, stack=100, n=1 -> should be all-in
        let result = compute_geometric(100, 0, 100, 1, f64::INFINITY);
        assert_eq!(result, 100);

        // pot=100, to_call=0, stack=200, n=2
        // SPR = 2, ratio = ((2*2+1)^0.5 - 1) / 2 = (sqrt(5) - 1) / 2 ≈ 0.618
        let result = compute_geometric(100, 0, 200, 2, f64::INFINITY);
        assert!((result - 62).abs() <= 1); // approximately 62
    }
}
