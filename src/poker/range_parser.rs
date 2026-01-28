//! Range string parsing (PIO format).
//!
//! Supports formats like:
//! - "AA,KK,QQ" - All combos of these hands at weight 1.0
//! - "AKs,AQs" - Suited hands only
//! - "AKo,AQo" - Offsuit hands only
//! - "AA:0.5,KK:0.75" - Weighted hands
//! - "AsKs,AhKh" - Specific combos

use crate::poker::hands::{make_card, Combo, Range};

/// Parse a range string in PIO format.
///
/// # Format
/// - Hand groups separated by commas: "AA,KK,QQ"
/// - Optional weight suffix: "QQ:0.5" (50% frequency)
/// - Hand types:
///   - "AA" - pair (6 combos)
///   - "AKs" - suited (4 combos)
///   - "AKo" - offsuit (12 combos)
///   - "AK" - both suited and offsuit (16 combos)
///   - "AsKs" - specific combo
///
/// # Examples
/// ```ignore
/// let range = parse_range("AA,KK,QQ:0.5,AKs")?;
/// ```
pub fn parse_range(s: &str) -> Result<Range, String> {
    let mut range = Range::new();

    // Handle empty string
    if s.trim().is_empty() {
        return Ok(range);
    }

    // Split by comma
    for part in s.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }

        // Check for weight suffix
        let (hand_str, weight) = if let Some(idx) = part.find(':') {
            let weight_str = &part[idx + 1..];
            let weight: f32 = weight_str
                .parse()
                .map_err(|_| format!("Invalid weight: {}", weight_str))?;
            if weight < 0.0 || weight > 1.0 {
                return Err(format!("Weight must be 0-1: {}", weight));
            }
            (&part[..idx], weight)
        } else {
            (part, 1.0)
        };

        // Parse the hand and add combos
        add_hand_to_range(&mut range, hand_str, weight)?;
    }

    Ok(range)
}

/// Add a hand or combo to the range.
fn add_hand_to_range(range: &mut Range, hand_str: &str, weight: f32) -> Result<(), String> {
    let chars: Vec<char> = hand_str.chars().collect();

    match chars.len() {
        // Pair: "AA"
        2 => {
            let rank1 = parse_rank_char(chars[0])?;
            let rank2 = parse_rank_char(chars[1])?;

            if rank1 == rank2 {
                // Pair: add all 6 combos
                add_pair_combos(range, rank1, weight);
            } else {
                // Unpaired without s/o suffix: add both suited and offsuit
                add_suited_combos(range, rank1, rank2, weight);
                add_offsuit_combos(range, rank1, rank2, weight);
            }
        }
        // Suited or offsuit: "AKs" or "AKo"
        3 => {
            let rank1 = parse_rank_char(chars[0])?;
            let rank2 = parse_rank_char(chars[1])?;
            let suffix = chars[2];

            match suffix {
                's' | 'S' => add_suited_combos(range, rank1, rank2, weight),
                'o' | 'O' => add_offsuit_combos(range, rank1, rank2, weight),
                _ => return Err(format!("Invalid hand suffix: {}", suffix)),
            }
        }
        // Specific combo: "AsKs"
        4 => {
            let combo = parse_specific_combo(hand_str)?;
            range.set(combo, weight);
        }
        _ => return Err(format!("Invalid hand format: {}", hand_str)),
    }

    Ok(())
}

/// Parse a rank character (2-9, T, J, Q, K, A) to rank index (0-12).
fn parse_rank_char(c: char) -> Result<u8, String> {
    match c {
        '2' => Ok(0),
        '3' => Ok(1),
        '4' => Ok(2),
        '5' => Ok(3),
        '6' => Ok(4),
        '7' => Ok(5),
        '8' => Ok(6),
        '9' => Ok(7),
        'T' | 't' => Ok(8),
        'J' | 'j' => Ok(9),
        'Q' | 'q' => Ok(10),
        'K' | 'k' => Ok(11),
        'A' | 'a' => Ok(12),
        _ => Err(format!("Invalid rank: {}", c)),
    }
}

/// Parse a suit character (c, d, h, s) to suit index (0-3).
fn parse_suit_char(c: char) -> Result<u8, String> {
    match c {
        'c' | 'C' => Ok(0),
        'd' | 'D' => Ok(1),
        'h' | 'H' => Ok(2),
        's' | 'S' => Ok(3),
        _ => Err(format!("Invalid suit: {}", c)),
    }
}

/// Add all 6 pair combos to the range.
fn add_pair_combos(range: &mut Range, rank: u8, weight: f32) {
    for s1 in 0..4u8 {
        for s2 in (s1 + 1)..4u8 {
            let c1 = make_card(rank, s1);
            let c2 = make_card(rank, s2);
            let combo = Combo::new(c1, c2);
            range.set(combo, weight);
        }
    }
}

/// Add all 4 suited combos to the range.
fn add_suited_combos(range: &mut Range, rank1: u8, rank2: u8, weight: f32) {
    for suit in 0..4u8 {
        let c1 = make_card(rank1, suit);
        let c2 = make_card(rank2, suit);
        let combo = Combo::new(c1, c2);
        range.set(combo, weight);
    }
}

/// Add all 12 offsuit combos to the range.
fn add_offsuit_combos(range: &mut Range, rank1: u8, rank2: u8, weight: f32) {
    for s1 in 0..4u8 {
        for s2 in 0..4u8 {
            if s1 != s2 {
                let c1 = make_card(rank1, s1);
                let c2 = make_card(rank2, s2);
                let combo = Combo::new(c1, c2);
                range.set(combo, weight);
            }
        }
    }
}

/// Parse a specific combo like "AsKs" or "AhKd".
fn parse_specific_combo(s: &str) -> Result<Combo, String> {
    let chars: Vec<char> = s.chars().collect();
    if chars.len() != 4 {
        return Err(format!("Invalid combo format: {}", s));
    }

    let rank1 = parse_rank_char(chars[0])?;
    let suit1 = parse_suit_char(chars[1])?;
    let rank2 = parse_rank_char(chars[2])?;
    let suit2 = parse_suit_char(chars[3])?;

    let c1 = make_card(rank1, suit1);
    let c2 = make_card(rank2, suit2);

    if c1 == c2 {
        return Err(format!("Cards cannot be the same: {}", s));
    }

    Ok(Combo::new(c1, c2))
}

impl Range {
    /// Parse a range from a string in PIO format.
    pub fn from_str(s: &str) -> Result<Self, String> {
        parse_range(s)
    }

    /// Format the range as a string (simplified, non-canonical).
    pub fn to_string_simple(&self) -> String {
        use crate::poker::hands::combo_to_string;
        let mut parts = Vec::new();
        for (idx, &weight) in self.weights.iter().enumerate() {
            if weight > 0.0 {
                let combo = Combo::from_index(idx);
                let s = if (weight - 1.0).abs() < 0.001 {
                    combo_to_string(combo)
                } else {
                    format!("{}:{:.2}", combo_to_string(combo), weight)
                };
                parts.push(s);
            }
        }
        parts.join(",")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_pair() {
        let range = parse_range("AA").unwrap();
        assert_eq!(range.count_combos(), 6);
    }

    #[test]
    fn test_parse_suited() {
        let range = parse_range("AKs").unwrap();
        assert_eq!(range.count_combos(), 4);
    }

    #[test]
    fn test_parse_offsuit() {
        let range = parse_range("AKo").unwrap();
        assert_eq!(range.count_combos(), 12);
    }

    #[test]
    fn test_parse_unpaired_no_suffix() {
        let range = parse_range("AK").unwrap();
        assert_eq!(range.count_combos(), 16); // 4 suited + 12 offsuit
    }

    #[test]
    fn test_parse_specific_combo() {
        let range = parse_range("AsKs").unwrap();
        assert_eq!(range.count_combos(), 1);
    }

    #[test]
    fn test_parse_weighted() {
        let range = parse_range("AA:0.5").unwrap();
        assert_eq!(range.count_combos(), 6);
        // All combos should have weight 0.5
        let combo = Combo::new(make_card(12, 0), make_card(12, 1)); // AcAd
        assert!((range.get(combo) - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_parse_multiple() {
        let range = parse_range("AA,KK,QQ").unwrap();
        assert_eq!(range.count_combos(), 18);
    }

    #[test]
    fn test_parse_complex() {
        let range = parse_range("AA,KK,QQ:0.5,AKs,AKo:0.75").unwrap();
        assert_eq!(range.count_combos(), 6 + 6 + 6 + 4 + 12);

        // Check QQ weight
        let qq = Combo::new(make_card(10, 0), make_card(10, 1));
        assert!((range.get(qq) - 0.5).abs() < 0.001);

        // Check AKo weight
        let ako = Combo::new(make_card(12, 0), make_card(11, 1));
        assert!((range.get(ako) - 0.75).abs() < 0.001);
    }

    #[test]
    fn test_parse_empty() {
        let range = parse_range("").unwrap();
        assert_eq!(range.count_combos(), 0);
    }

    #[test]
    fn test_invalid_weight() {
        assert!(parse_range("AA:1.5").is_err());
        assert!(parse_range("AA:-0.5").is_err());
        assert!(parse_range("AA:abc").is_err());
    }

    #[test]
    fn test_invalid_hand() {
        assert!(parse_range("XY").is_err());
        assert!(parse_range("AAx").is_err());
    }
}
