//! Board string parsing (e.g., "KhQsJs" -> Board).

use crate::poker::hands::{make_card, Board, Card};

/// Parse a board string like "KhQsJs" or "Kh Qs Js".
///
/// # Format
/// Each card is 2 characters: rank + suit
/// - Ranks: 2-9, T, J, Q, K, A
/// - Suits: c (clubs), d (diamonds), h (hearts), s (spades)
///
/// Whitespace is ignored.
pub fn parse_board(s: &str) -> Result<Board, String> {
    let cards = parse_cards(s)?;
    if cards.len() < 3 || cards.len() > 5 {
        return Err(format!(
            "Board must have 3-5 cards, got {}",
            cards.len()
        ));
    }
    Ok(Board::new(&cards))
}

/// Parse a string of cards into a vector.
pub fn parse_cards(s: &str) -> Result<Vec<Card>, String> {
    let s = s.replace(' ', "");
    if s.len() % 2 != 0 {
        return Err(format!("Invalid card string length: {}", s.len()));
    }

    let mut cards = Vec::new();
    let chars: Vec<char> = s.chars().collect();

    for chunk in chars.chunks(2) {
        let card = parse_card(chunk[0], chunk[1])?;
        cards.push(card);
    }

    // Check for duplicates
    for i in 0..cards.len() {
        for j in (i + 1)..cards.len() {
            if cards[i] == cards[j] {
                return Err(format!("Duplicate card in board"));
            }
        }
    }

    Ok(cards)
}

/// Parse a single card from rank and suit characters.
pub fn parse_card(rank_char: char, suit_char: char) -> Result<Card, String> {
    let rank = match rank_char {
        '2' => 0,
        '3' => 1,
        '4' => 2,
        '5' => 3,
        '6' => 4,
        '7' => 5,
        '8' => 6,
        '9' => 7,
        'T' | 't' => 8,
        'J' | 'j' => 9,
        'Q' | 'q' => 10,
        'K' | 'k' => 11,
        'A' | 'a' => 12,
        _ => return Err(format!("Invalid rank: {}", rank_char)),
    };

    let suit = match suit_char {
        'c' | 'C' => 0,
        'd' | 'D' => 1,
        'h' | 'H' => 2,
        's' | 'S' => 3,
        _ => return Err(format!("Invalid suit: {}", suit_char)),
    };

    Ok(make_card(rank, suit))
}

impl Board {
    /// Parse a board from a string like "KhQsJs".
    pub fn from_str(s: &str) -> Result<Self, String> {
        parse_board(s)
    }

    /// Format the board as a string.
    pub fn to_string(&self) -> String {
        use crate::poker::hands::card_to_string;
        self.cards.iter().map(|&c| card_to_string(c)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::poker::hands::{rank, suit};

    #[test]
    fn test_parse_card() {
        let card = parse_card('A', 's').unwrap();
        assert_eq!(rank(card), 12);
        assert_eq!(suit(card), 3);

        let card = parse_card('2', 'c').unwrap();
        assert_eq!(rank(card), 0);
        assert_eq!(suit(card), 0);
    }

    #[test]
    fn test_parse_board() {
        let board = parse_board("KhQsJs").unwrap();
        assert_eq!(board.len(), 3);

        let board = parse_board("Kh Qs Js 2c 3d").unwrap();
        assert_eq!(board.len(), 5);
    }

    #[test]
    fn test_board_roundtrip() {
        let original = "KhQsJs2c3d";
        let board = parse_board(original).unwrap();
        let formatted = board.to_string();
        assert_eq!(formatted, original);
    }

    #[test]
    fn test_invalid_board() {
        assert!(parse_board("Kh").is_err()); // Too few cards
        assert!(parse_board("KhQsJs2c3d4h").is_err()); // Too many cards
        assert!(parse_board("KhKh").is_err()); // Duplicate
        assert!(parse_board("Xh").is_err()); // Invalid rank
        assert!(parse_board("Kx").is_err()); // Invalid suit
    }
}
