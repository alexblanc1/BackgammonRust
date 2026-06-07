//! Le réseau de neurones évaluateur, façon TD-Gammon.
//!
//! Un perceptron à une seule couche cachée, codé à la main (pas de dépendance
//! lourde — le réseau est petit). Entrée : les `N_INPUTS` unités de
//! `encoding::encode`. Sortie : un nombre dans `(0, 1)`, la probabilité que le
//! joueur à jouer gagne.
//!
//! Tant qu'il n'est pas entraîné (poids aléatoires), il joue mal : l'objectif
//! de cette étape est seulement d'avoir un `Evaluator` qui marche. L'entraînement
//! TD(λ) viendra ensuite.

use crate::board::Board;
use crate::dice::Dice;
use crate::encoding::{encode, N_INPUTS};
use crate::eval::{Evaluator, GreedyAgent};

/// Fonction sigmoïde : écrase n'importe quel réel dans `(0, 1)`.
fn sigmoid(z: f64) -> f64 {
    1.0 / (1.0 + (-z).exp())
}

/// Réseau à une couche cachée : `N_INPUTS` entrées → `hidden` neurones cachés
/// (sigmoïde) → 1 sortie (sigmoïde).
#[derive(Clone)]
pub struct Net {
    hidden: usize,
    /// Poids entrée→caché, à plat : la ligne du neurone `j` est le tranche
    /// `w1[j * N_INPUTS .. (j + 1) * N_INPUTS]`.
    w1: Vec<f64>,
    /// Biais des neurones cachés (`hidden`).
    b1: Vec<f64>,
    /// Poids caché→sortie (`hidden`).
    w2: Vec<f64>,
    /// Biais de la sortie.
    b2: f64,
}

impl Net {
    /// Crée un réseau aux poids aléatoires (petits) et biais nuls.
    pub fn new_random(hidden: usize, seed: u64) -> Net {
        let mut d = Dice::new(seed);
        let scale = 0.1; // petits poids initiaux, autour de 0
        let rand_weight = |d: &mut Dice| (d.unit() * 2.0 - 1.0) * scale;

        let w1 = (0..hidden * N_INPUTS).map(|_| rand_weight(&mut d)).collect();
        let w2 = (0..hidden).map(|_| rand_weight(&mut d)).collect();

        Net {
            hidden,
            w1,
            b1: vec![0.0; hidden],
            w2,
            b2: 0.0,
        }
    }

    /// Passe avant. Renvoie les activations cachées (utiles à l'entraînement)
    /// et la sortie du réseau.
    pub fn forward(&self, x: &[f64; N_INPUTS]) -> (Vec<f64>, f64) {
        let mut h = vec![0.0; self.hidden];
        for j in 0..self.hidden {
            let row = &self.w1[j * N_INPUTS..(j + 1) * N_INPUTS];
            let mut z = self.b1[j];
            for i in 0..N_INPUTS {
                z += row[i] * x[i];
            }
            h[j] = sigmoid(z);
        }

        let mut z = self.b2;
        for j in 0..self.hidden {
            z += self.w2[j] * h[j];
        }
        (h, sigmoid(z))
    }

    /// La valeur d'une position : probabilité estimée que le joueur à jouer gagne.
    pub fn value(&self, board: &Board) -> f64 {
        let x = encode(board);
        self.forward(&x).1
    }
}

impl Evaluator for Net {
    fn evaluate(&self, board: &Board) -> f64 {
        self.value(board)
    }
}

/// Agent qui joue l'argmax de la valeur d'un réseau aux poids aléatoires.
/// Pratique pour tester le pipeline ; il faudra l'entraîner pour qu'il soit bon.
pub fn net_agent(hidden: usize, seed: u64) -> GreedyAgent<Net> {
    GreedyAgent::new(Net::new_random(hidden, seed))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::Agent;
    use crate::agent::random::RandomAgent;
    use crate::dice::Dice;
    use crate::game::play;

    #[test]
    fn tailles_des_poids() {
        let net = Net::new_random(40, 1);
        assert_eq!(net.w1.len(), 40 * N_INPUTS);
        assert_eq!(net.b1.len(), 40);
        assert_eq!(net.w2.len(), 40);
    }

    #[test]
    fn sortie_dans_zero_un_et_deterministe() {
        let net = Net::new_random(40, 42);
        let v1 = net.value(&Board::starting_position());
        let v2 = net.value(&Board::starting_position());
        assert!(v1 > 0.0 && v1 < 1.0, "valeur hors (0,1) : {v1}");
        assert_eq!(v1, v2, "le réseau doit être déterministe");
    }

    #[test]
    fn un_reseau_non_entraine_joue_une_partie_entiere() {
        // Même nul, le réseau doit produire des coups légaux jusqu'au bout.
        let mut agents: [Box<dyn Agent>; 2] = [
            Box::new(net_agent(40, 7)),
            Box::new(RandomAgent::new(99)),
        ];
        let mut dice = Dice::new(2024);
        let (_winner, points) = play(&mut agents, &mut dice);
        assert!((1..=3).contains(&points));
    }
}
