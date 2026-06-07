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

use std::path::Path;

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

/// Gradient de la sortie du réseau par rapport à chaque poids (même forme que
/// les poids du réseau). Sert à l'entraînement : descente de gradient ici, et
/// plus tard les traces d'éligibilité de TD(λ).
pub struct Gradients {
    pub w1: Vec<f64>,
    pub b1: Vec<f64>,
    pub w2: Vec<f64>,
    pub b2: f64,
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

    /// Rétropropagation : gradient de la sortie `y` par rapport à tous les
    /// poids, à partir de l'entrée `x` et des activations cachées `h` et de la
    /// sortie `y` déjà calculées par `forward`.
    pub fn output_gradient(&self, x: &[f64; N_INPUTS], h: &[f64], y: f64) -> Gradients {
        // Dérivée de la sigmoïde à la sortie : sigmoid'(z2) = y·(1 − y).
        let dy_dz2 = y * (1.0 - y);

        let mut g_w1 = vec![0.0; self.w1.len()];
        let mut g_b1 = vec![0.0; self.hidden];
        let mut g_w2 = vec![0.0; self.hidden];

        for j in 0..self.hidden {
            // Couche de sortie.
            g_w2[j] = dy_dz2 * h[j]; // ∂y/∂w2[j]

            // Rétropropagation vers la couche cachée.
            let dy_dh = dy_dz2 * self.w2[j]; // ∂y/∂h[j]
            let dy_dz1 = dy_dh * h[j] * (1.0 - h[j]); // × sigmoid'(z1[j])
            g_b1[j] = dy_dz1; // ∂y/∂b1[j]

            let base = j * N_INPUTS;
            for i in 0..N_INPUTS {
                g_w1[base + i] = dy_dz1 * x[i]; // ∂y/∂w1[j][i]
            }
        }

        Gradients {
            w1: g_w1,
            b1: g_b1,
            w2: g_w2,
            b2: dy_dz2, // ∂y/∂b2
        }
    }

    /// Un pas de descente de gradient pour rapprocher la sortie de `target`
    /// (erreur quadratique). Met à jour les poids et renvoie l'erreur `target − y`
    /// observée *avant* le pas.
    pub fn train_step(&mut self, x: &[f64; N_INPUTS], target: f64, lr: f64) -> f64 {
        let (h, y) = self.forward(x);
        let g = self.output_gradient(x, &h, y);
        let err = target - y;
        let step = lr * err; // Δw = lr · (target − y) · ∂y/∂w

        for k in 0..self.w1.len() {
            self.w1[k] += step * g.w1[k];
        }
        for j in 0..self.hidden {
            self.b1[j] += step * g.b1[j];
            self.w2[j] += step * g.w2[j];
        }
        self.b2 += step * g.b2;
        err
    }

    /// Un gradient/trace de la bonne forme, initialisé à zéro.
    pub fn zero_gradients(&self) -> Gradients {
        Gradients {
            w1: vec![0.0; self.w1.len()],
            b1: vec![0.0; self.hidden],
            w2: vec![0.0; self.hidden],
            b2: 0.0,
        }
    }

    /// Applique `Δw = facteur · trace` à tous les poids. Brique de la mise à
    /// jour TD(λ) : `facteur = α · erreur_TD`, `trace` = trace d'éligibilité.
    pub fn apply_update(&mut self, trace: &Gradients, factor: f64) {
        for k in 0..self.w1.len() {
            self.w1[k] += factor * trace.w1[k];
        }
        for j in 0..self.hidden {
            self.b1[j] += factor * trace.b1[j];
            self.w2[j] += factor * trace.w2[j];
        }
        self.b2 += factor * trace.b2;
    }

    /// Sauvegarde les poids dans un fichier texte (un nombre par ligne, précédé
    /// de la taille de la couche cachée). Le format `f64` de Rust est réécrit
    /// sans perte, donc le rechargement est exact.
    pub fn save(&self, path: impl AsRef<Path>) -> std::io::Result<()> {
        use std::fmt::Write as _;
        let mut s = String::new();
        let _ = writeln!(s, "{}", self.hidden);
        for v in self
            .w1
            .iter()
            .chain(self.b1.iter())
            .chain(self.w2.iter())
            .chain(std::iter::once(&self.b2))
        {
            let _ = writeln!(s, "{v}");
        }
        std::fs::write(path, s)
    }

    /// Recharge un réseau sauvegardé par [`Net::save`].
    pub fn load(path: impl AsRef<Path>) -> std::io::Result<Net> {
        use std::io::{Error, ErrorKind};
        let text = std::fs::read_to_string(path)?;
        let mut it = text.split_whitespace();
        let hidden: usize = it
            .next()
            .and_then(|s| s.parse().ok())
            .ok_or_else(|| Error::new(ErrorKind::InvalidData, "taille cachée manquante"))?;
        let nums: Vec<f64> = it.filter_map(|s| s.parse().ok()).collect();

        let n_w1 = hidden * N_INPUTS;
        let expected = n_w1 + hidden + hidden + 1;
        if nums.len() != expected {
            return Err(Error::new(
                ErrorKind::InvalidData,
                format!("{} poids lus, {expected} attendus", nums.len()),
            ));
        }
        Ok(Net {
            hidden,
            w1: nums[..n_w1].to_vec(),
            b1: nums[n_w1..n_w1 + hidden].to_vec(),
            w2: nums[n_w1 + hidden..n_w1 + 2 * hidden].to_vec(),
            b2: nums[n_w1 + 2 * hidden],
        })
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
    fn apprend_une_cible_fixe() {
        // Surapprentissage : sur une seule entrée, la sortie doit converger
        // vers la cible. Si la backprop est fausse, ça ne converge pas.
        let mut net = Net::new_random(10, 5);
        let x = encode(&Board::starting_position());
        for _ in 0..2000 {
            net.train_step(&x, 0.9, 0.1);
        }
        let y = net.forward(&x).1;
        assert!((y - 0.9).abs() < 0.01, "y={y} n'a pas convergé vers 0.9");
    }

    #[test]
    fn apprend_a_distinguer_deux_positions() {
        // Deux entrées distinctes vers deux cibles distinctes : seul un gradient
        // correct à travers la couche cachée permet de les séparer.
        let mut net = Net::new_random(20, 3);
        let xa = encode(&Board::starting_position());
        let xb = encode(&Board {
            points: {
                let mut p = [0i8; 24];
                p[0] = 2;
                p[23] = -2;
                p
            },
            bar: [0, 0],
            off: [0, 0],
        });
        for _ in 0..5000 {
            net.train_step(&xa, 0.8, 0.1);
            net.train_step(&xb, 0.2, 0.1);
        }
        let ya = net.forward(&xa).1;
        let yb = net.forward(&xb).1;
        assert!((ya - 0.8).abs() < 0.05, "ya={ya}");
        assert!((yb - 0.2).abs() < 0.05, "yb={yb}");
    }

    #[test]
    fn sauvegarde_et_rechargement_exacts() {
        let net = Net::new_random(20, 99);
        let path = std::env::temp_dir().join("backgammon_test_net.txt");
        net.save(&path).unwrap();
        let loaded = Net::load(&path).unwrap();

        // Les valeurs doivent être identiques sur plusieurs positions.
        let b1 = Board::starting_position();
        let b2 = b1.swap_perspective();
        assert_eq!(net.value(&b1), loaded.value(&b1));
        assert_eq!(net.value(&b2), loaded.value(&b2));
        let _ = std::fs::remove_file(&path);
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
