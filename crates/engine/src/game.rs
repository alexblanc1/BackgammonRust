//! L'état d'une partie et la boucle de jeu.

use crate::agent::Agent;
use crate::board::Board;
use crate::dice::{Dice, Roll};
use crate::moves::legal_plays;
use crate::player::Player;

/// Le videau (doubling cube) : la mise courante de la partie et son
/// propriétaire.
///
/// Au départ la partie vaut 1 point et le videau est « au milieu » (`owner =
/// None`) : chacun peut proposer de doubler. Quand un joueur double et que
/// l'autre accepte, la mise est multipliée par 2 et **l'accepteur** devient
/// propriétaire : lui seul pourra redoubler. Refuser un double concède la
/// partie à la mise courante.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Cube {
    pub value: u32,
    pub owner: Option<Player>,
}

impl Cube {
    pub fn new() -> Cube {
        Cube {
            value: 1,
            owner: None,
        }
    }

    /// Le joueur `p` a-t-il le droit de proposer un double ?
    /// (videau au milieu ou à lui, et mise plafonnée à 64).
    pub fn may_double(&self, p: Player) -> bool {
        self.value < 64 && self.owner.is_none_or(|o| o == p)
    }

    /// Enregistre un double accepté : mise ×2, le videau passe à l'accepteur.
    pub fn accept_double(&mut self, acceptor: Player) {
        self.value *= 2;
        self.owner = Some(acceptor);
    }
}

impl Default for Cube {
    fn default() -> Cube {
        Cube::new()
    }
}

/// Phase courante de la partie.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Phase {
    /// On attend que le joueur lance les dés.
    AwaitingRoll,
    /// On attend que le joueur joue le lancer indiqué.
    AwaitingMove(Roll),
    /// La partie est terminée. `points` inclut le videau (points de victoire
    /// × valeur du videau, ou mise concédée sur un double refusé).
    GameOver { winner: Player, points: u32 },
}

/// L'état complet d'une partie.
#[derive(Clone, Debug)]
pub struct GameState {
    pub board: Board,
    pub to_move: Player,
    pub phase: Phase,
    pub cube: Cube,
}

impl GameState {
    /// Crée une nouvelle partie dans la position de départ.
    pub fn new() -> GameState {
        GameState {
            board: Board::starting_position(),
            to_move: Player::White,
            phase: Phase::AwaitingRoll,
            cube: Cube::new(),
        }
    }
}

/// Joue une partie complète entre deux agents et renvoie le gagnant ainsi que
/// le nombre de points, videau compris (1/2/3 points × valeur du videau, ou la
/// mise courante si un double est refusé).
///
/// `agents[0]` joue les Blancs, `agents[1]` les Noirs.
pub fn play(agents: &mut [Box<dyn Agent>; 2], dice: &mut Dice) -> (Player, u32) {
    let mut state = GameState::new();
    let mut idx = 0usize; // 0 = Blancs, 1 = Noirs

    loop {
        // Videau : avant de lancer, le joueur au trait peut proposer un double.
        if state.cube.may_double(state.to_move) && agents[idx].should_double(&state) {
            if agents[1 - idx].should_accept_double(&state) {
                state.cube.accept_double(state.to_move.other());
            } else {
                // Refus : l'adversaire concède la mise courante.
                return (state.to_move, state.cube.value);
            }
        }

        let roll = dice.roll();
        let plays = legal_plays(&state.board, &roll);

        // Si aucun coup n'est possible, le joueur passe simplement son tour.
        if !plays.is_empty() {
            let chosen = agents[idx].choose_play(&state, &plays);
            state.board = plays[chosen].clone();
        }

        if let Some(points) = state.board.win_check() {
            return (state.to_move, points as u32 * state.cube.value);
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
    /// se termine bien avec un score valide (1, 2 ou 3 points : les agents
    /// aléatoires ne doublent jamais, le videau reste à 1).
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

    /// Les droits sur le videau : au milieu chacun peut doubler ; après un
    /// double accepté, seul l'accepteur peut redoubler ; plafond à 64.
    #[test]
    fn droits_du_videau() {
        let mut cube = Cube::new();
        assert!(cube.may_double(Player::White));
        assert!(cube.may_double(Player::Black));

        // Blancs doublent, Noirs acceptent : mise 2, videau aux Noirs.
        cube.accept_double(Player::Black);
        assert_eq!(cube.value, 2);
        assert!(!cube.may_double(Player::White));
        assert!(cube.may_double(Player::Black));

        // Montée jusqu'au plafond : 2 → 4 → … → 64, plus personne ne double.
        while cube.value < 64 {
            cube.accept_double(Player::White);
        }
        assert!(!cube.may_double(Player::White));
        assert!(!cube.may_double(Player::Black));
    }

    /// Un agent qui double à la première occasion contre un agent qui refuse :
    /// le doubleur gagne immédiatement la mise courante (1 point).
    #[test]
    fn double_refuse_concede_la_mise() {
        struct Doubler;
        impl Agent for Doubler {
            fn choose_play(&mut self, _: &GameState, _: &[Board]) -> usize {
                0
            }
            fn should_double(&mut self, _: &GameState) -> bool {
                true
            }
        }
        struct Refuser;
        impl Agent for Refuser {
            fn choose_play(&mut self, _: &GameState, _: &[Board]) -> usize {
                0
            }
            fn should_accept_double(&mut self, _: &GameState) -> bool {
                false
            }
        }

        let mut agents: [Box<dyn Agent>; 2] = [Box::new(Doubler), Box::new(Refuser)];
        let mut dice = Dice::new(1);
        let (winner, points) = play(&mut agents, &mut dice);
        assert_eq!(winner, Player::White);
        assert_eq!(points, 1);
    }

    /// Un double accepté multiplie bien les points de la victoire finale.
    #[test]
    fn double_accepte_multiplie_les_points() {
        // Blancs doublent une fois (puis plus jamais : le videau passe aux
        // Noirs qui ne redoublent pas) ; la partie se joue au hasard.
        struct DoubleOnce;
        impl Agent for DoubleOnce {
            fn choose_play(&mut self, _: &GameState, legal: &[Board]) -> usize {
                legal.len() / 2
            }
            fn should_double(&mut self, _: &GameState) -> bool {
                true
            }
        }

        let mut agents: [Box<dyn Agent>; 2] = [
            Box::new(DoubleOnce),
            Box::new(RandomAgent::new(99)),
        ];
        let mut dice = Dice::new(42);
        let (_winner, points) = play(&mut agents, &mut dice);
        // Mise finale = points de jeu (1..3) × videau (2) → toujours pair et ≥ 2.
        assert!(points >= 2 && points % 2 == 0, "points = {points}");
    }
}
