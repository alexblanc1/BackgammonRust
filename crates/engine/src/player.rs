//! Le joueur : un simple enum à deux variantes.

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Player {
    White,
    Black,
}

impl Player {
    /// Renvoie le joueur adverse.
    pub fn other(self) -> Player {
        match self {
            Player::White => Player::Black,
            Player::Black => Player::White,
        }
    }
}
