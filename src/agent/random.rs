//! Agent qui joue un coup légal au hasard.

use crate::agent::Agent;
use crate::board::Board;
use crate::dice::Dice;
use crate::game::GameState;

/// Joue un coup légal choisi au hasard.
///
/// Pratique comme adversaire d'entraînement et pour tester le moteur de bout
/// en bout dès que la génération de coups fonctionne.
pub struct RandomAgent {
    dice: Dice,
}

impl RandomAgent {
    pub fn new(seed: u64) -> RandomAgent {
        RandomAgent {
            dice: Dice::new(seed),
        }
    }
}

impl Agent for RandomAgent {
    fn choose_play(&mut self, _state: &GameState, legal: &[Board]) -> usize {
        self.dice.index(legal.len())
    }
}
