//! Agent piloté par un humain au clavier.

use crate::agent::Agent;
use crate::board::Board;
use crate::game::GameState;

/// Joueur humain : à toi d'afficher le plateau et de lire le choix au clavier.
pub struct HumanAgent;

impl HumanAgent {
    pub fn new() -> HumanAgent {
        HumanAgent
    }
}

impl Agent for HumanAgent {
    fn choose_play(&mut self, state: &GameState, legal: &[Board]) -> usize {
        todo!("afficher le plateau et les coups possibles, lire l'indice au clavier")
    }
}
