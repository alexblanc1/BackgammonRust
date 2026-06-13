//! L'abstraction d'évaluation : noter une position, et jouer au mieux selon
//! cette note.
//!
//! Séparer « savoir noter » (`Evaluator`) de « savoir choisir » (`GreedyAgent`)
//! permet de réutiliser exactement la même logique de choix pour l'heuristique,
//! le futur réseau de neurones, et la base de l'expectiminimax.

use engine::agent::Agent;
use engine::board::Board;
use engine::game::GameState;

/// Tout ce qui sait attribuer un score à une position, **du point de vue du
/// joueur à jouer** : plus c'est haut, mieux c'est pour lui.
///
/// `&self` parce qu'un évaluateur peut porter un état (les poids d'un réseau,
/// par exemple) ; l'heuristique, elle, n'en a pas besoin.
pub trait Evaluator {
    fn evaluate(&self, board: &Board) -> f64;

    /// Valeur d'une position où le joueur à jouer **vient de gagner** `points`
    /// (1, 2 ou 3), sur la même échelle que `evaluate`. Sert aux recherches
    /// (expectiminimax, rollouts) pour noter les positions terminales.
    ///
    /// Par défaut : une valeur énorme, pour que gagner domine toujours les
    /// scores « ordinaires » (cas de l'heuristique, dont l'échelle est libre).
    /// Le réseau, lui, redéfinit cette valeur en points exacts, cohérents avec
    /// son équité.
    fn terminal_value(&self, points: u8) -> f64 {
        1e6 * points as f64
    }

    /// La probabilité que le joueur à jouer gagne, si l'évaluateur sait
    /// l'estimer (le réseau le sait, l'heuristique non). Sert aux décisions
    /// de videau (doubler / accepter).
    fn win_prob(&self, _board: &Board) -> Option<f64> {
        None
    }
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

    fn should_double(&mut self, state: &GameState) -> bool {
        default_should_double(&self.evaluator, state)
    }

    fn should_accept_double(&mut self, state: &GameState) -> bool {
        default_should_accept(&self.evaluator, state)
    }
}

// --- Décisions de videau par défaut -------------------------------------------
//
// Règles classiques fondées sur P(gain), pour tout évaluateur qui sait
// l'estimer (`win_prob`) ; les autres (heuristique, hasard) ne doublent jamais
// et acceptent toujours.

/// Doubler quand on est nettement favori (p ≥ 65 %) mais pas « trop bon »
/// (p < 95 % : à ce stade on préfère jouer pour le gammon plutôt que d'offrir
/// à l'adversaire la sortie à 1 × videau). `state.board` est vu du doubleur,
/// qui est le joueur au trait.
pub fn default_should_double<E: Evaluator>(eval: &E, state: &GameState) -> bool {
    match eval.win_prob(&state.board) {
        Some(p) => (0.65..0.95).contains(&p),
        None => false,
    }
}

/// Accepter un double tant qu'on garde ≈ 25 % de chances : refuser coûte
/// toujours 1 × videau, accepter coûte 2 × videau mais avec p de gagner —
/// rentable dès que p ≥ 0,25. `state.board` est vu du **doubleur** : on
/// accepte donc si SES chances ne dépassent pas 75 %.
pub fn default_should_accept<E: Evaluator>(eval: &E, state: &GameState) -> bool {
    match eval.win_prob(&state.board) {
        Some(p) => p <= 0.75,
        None => true,
    }
}
