//! L'état d'une partie et la boucle de jeu.

use crate::agent::Agent;
use crate::board::Board;
use crate::dice::{Dice, Roll};
use crate::moves::legal_plays;
use crate::player::Player;

/// Phase courante de la partie.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Phase {
    /// On attend que le joueur lance les dés.
    AwaitingRoll,
    /// On attend que le joueur joue le lancer indiqué.
    AwaitingMove(Roll),
    /// La partie est terminée.
    GameOver { winner: Player, points: u8 },
}

/// L'état complet d'une partie.
#[derive(Clone, Debug)]
pub struct GameState {
    pub board: Board,
    pub to_move: Player,
    pub phase: Phase,
}

impl GameState {
    /// Crée une nouvelle partie dans la position de départ.
    pub fn new() -> GameState {
        GameState {
            board: Board::starting_position(),
            to_move: Player::White,
            phase: Phase::AwaitingRoll,
        }
    }
}

/// Joue une partie complète entre deux agents et renvoie le gagnant ainsi que
/// le nombre de points (1 = simple, 2 = gammon, 3 = backgammon).
///
/// `agents[0]` joue les Blancs, `agents[1]` les Noirs.
pub fn play(agents: &mut [Box<dyn Agent>; 2], dice: &mut Dice) -> (Player, u8) {
    let mut state = GameState::new();
    let mut idx = 0usize; // 0 = Blancs, 1 = Noirs

    loop {
        let roll = dice.roll();
        let plays = legal_plays(&state.board, &roll);

        // Si aucun coup n'est possible, le joueur passe simplement son tour.
        if !plays.is_empty() {
            let chosen = agents[idx].choose_play(&state, &plays);
            state.board = plays[chosen].clone();
        }

        if let Some(points) = state.board.win_check() {
            return (state.to_move, points);
        }

        // On passe la main : on retourne le plateau et on change de joueur.
        state.board = state.board.swap_perspective();
        state.to_move = state.to_move.other();
        idx = 1 - idx;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::random::RandomAgent;

    /// Joue beaucoup de parties aléatoires : si la génération de coups produit
    /// un coup illégal, on déclencherait souvent une panique (index hors borne
    /// ou soustraction `u8` négative). Ce test vérifie aussi que chaque partie
    /// se termine bien avec un score valide (1, 2 ou 3 points).
    #[test]
    fn parties_aleatoires_se_terminent_avec_un_score_valide() {
        for seed in 1..=300u64 {
            let mut agents: [Box<dyn Agent>; 2] = [
                Box::new(RandomAgent::new(seed)),
                Box::new(RandomAgent::new(seed.wrapping_mul(2_654_435_761).wrapping_add(1))),
            ];
            let mut dice = Dice::new(seed.wrapping_mul(0x9E37_79B9).wrapping_add(7));

            let (_winner, points) = play(&mut agents, &mut dice);
            assert!(
                (1..=3).contains(&points),
                "graine {seed} : score invalide {points}"
            );
        }
    }
}
