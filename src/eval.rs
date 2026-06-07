//! L'abstraction d'évaluation : noter une position, et jouer au mieux selon
//! cette note.
//!
//! Séparer « savoir noter » (`Evaluator`) de « savoir choisir » (`GreedyAgent`)
//! permet de réutiliser exactement la même logique de choix pour l'heuristique,
//! le futur réseau de neurones, et la base de l'expectiminimax.

use crate::agent::Agent;
use crate::board::Board;
use crate::game::GameState;

/// Tout ce qui sait attribuer un score à une position, **du point de vue du
/// joueur à jouer** : plus c'est haut, mieux c'est pour lui.
///
/// `&self` parce qu'un évaluateur peut porter un état (les poids d'un réseau,
/// par exemple) ; l'heuristique, elle, n'en a pas besoin.
pub trait Evaluator {
    fn evaluate(&self, board: &Board) -> f64;
}

/// Agent qui joue le coup menant à la position la mieux notée par son
/// évaluateur. `E` est un type générique : n'importe quel évaluateur convient.
pub struct GreedyAgent<E> {
    evaluator: E,
}

impl<E> GreedyAgent<E> {
    pub fn new(evaluator: E) -> GreedyAgent<E> {
        GreedyAgent { evaluator }
    }
}

// On n'implémente `Agent` que pour les `E` qui savent évaluer : c'est le rôle
// de la borne `E: Evaluator`. Un `GreedyAgent` autour d'un type quelconque
// existe, mais seul celui-ci est jouable.
impl<E: Evaluator> Agent for GreedyAgent<E> {
    fn choose_play(&mut self, _state: &GameState, legal: &[Board]) -> usize {
        // argmax : on garde l'indice du candidat au meilleur score.
        let mut best = 0usize;
        let mut best_score = f64::NEG_INFINITY;
        for (i, candidate) in legal.iter().enumerate() {
            let score = self.evaluator.evaluate(candidate);
            if score > best_score {
                best_score = score;
                best = i;
            }
        }
        best
    }
}
