//! Le trait `Agent` : tout ce qui peut jouer (humain, aléatoire, IA).

use crate::board::Board;
use crate::game::GameState;

/// Un joueur, quelle que soit sa nature.
pub trait Agent {
    /// Choisit un coup parmi les positions résultantes légales `legal`
    /// et renvoie l'indice de celle retenue.
    fn choose_play(&mut self, state: &GameState, legal: &[Board]) -> usize;

    /// Faut-il proposer de doubler (videau) ? Par défaut : non.
    fn should_double(&mut self, _state: &GameState) -> bool {
        false
    }

    /// Faut-il accepter un double proposé ? Par défaut : oui.
    fn should_accept_double(&mut self, _state: &GameState) -> bool {
        true
    }
}

pub mod human;
pub mod random;
