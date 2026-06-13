//! Le réseau de neurones évaluateur, façon TD-Gammon.
//!
//! Un perceptron à une seule couche cachée, codé à la main (pas de dépendance
//! lourde — le réseau est petit). Entrée : les `N_INPUTS` unités de
//! `encoding::encode`. Sortie : **six probabilités** (sigmoïdes), du point de
//! vue du joueur à jouer :
//!
//! | indice | signification                  |
//! |--------|--------------------------------|
//! | 0      | je gagne simple (1 point)      |
//! | 1      | je gagne gammon (2 points)     |
//! | 2      | je gagne backgammon (3 points) |
//! | 3      | je perds simple                |
//! | 4      | je perds gammon                |
//! | 5      | je perds backgammon            |
//!
//! La valeur d'une position est son **équité** : l'espérance de points
//! `(p0 + 2·p1 + 3·p2) − (p3 + 2·p4 + 3·p5)`, dans `[-3, +3]`. C'est elle que
//! l'agent glouton maximise — il distingue ainsi « gagner » de « gagner gros ».

use std::path::Path;

use engine::board::Board;
use engine::dice::Dice;
use crate::encoding::{encode, N_INPUTS};
use crate::eval::{Evaluator, GreedyAgent};

/// Nombre de sorties du réseau (voir le tableau ci-dessus).
pub const N_OUTPUTS: usize = 6;

/// Marqueur en tête du fichier de sauvegarde, pour reconnaître le format
/// multi-sorties (les anciens fichiers à 1 sortie commençaient par un nombre).
const SAVE_MAGIC: &str = "bgnet2";

/// Fonction sigmoïde : écrase n'importe quel réel dans `(0, 1)`.
fn sigmoid(z: f64) -> f64 {
    1.0 / (1.0 + (-z).exp())
}

/// Réseau à une couche cachée : `N_INPUTS` entrées → `hidden` neurones cachés
/// (sigmoïde) → `N_OUTPUTS` sorties (sigmoïdes).
#[derive(Clone)]
pub struct Net {
    hidden: usize,
    /// Poids entrée→caché, à plat : la ligne du neurone `j` est la tranche
    /// `w1[j * N_INPUTS .. (j + 1) * N_INPUTS]`.
    w1: Vec<f64>,
    /// Biais des neurones cachés (`hidden`).
    b1: Vec<f64>,
    /// Poids caché→sortie, à plat : la ligne de la sortie `k` est la tranche
    /// `w2[k * hidden .. (k + 1) * hidden]`.
    w2: Vec<f64>,
    /// Biais des sorties (`N_OUTPUTS`).
    b2: Vec<f64>,
}

/// Gradient d'**une** sortie du réseau par rapport à chaque poids (même forme
/// que les poids du réseau). Sert à l'entraînement : descente de gradient, et
/// les traces d'éligibilité de TD(λ) — une trace par sortie.
pub struct Gradients {
    pub w1: Vec<f64>,
    pub b1: Vec<f64>,
    pub w2: Vec<f64>,
    pub b2: Vec<f64>,
}

impl Net {
    /// Crée un réseau aux poids aléatoires (petits) et biais nuls.
    pub fn new_random(hidden: usize, seed: u64) -> Net {
        let mut d = Dice::new(seed);
        let scale = 0.1; // petits poids initiaux, autour de 0
        let rand_weight = |d: &mut Dice| (d.unit() * 2.0 - 1.0) * scale;

        let w1 = (0..hidden * N_INPUTS).map(|_| rand_weight(&mut d)).collect();
        let w2 = (0..N_OUTPUTS * hidden).map(|_| rand_weight(&mut d)).collect();

        Net {
            hidden,
            w1,
            b1: vec![0.0; hidden],
            w2,
            b2: vec![0.0; N_OUTPUTS],
        }
    }

    pub fn hidden(&self) -> usize {
        self.hidden
    }

    /// Passe avant. Renvoie les activations cachées (utiles à l'entraînement)
    /// et les six sorties du réseau.
    pub fn forward(&self, x: &[f64; N_INPUTS]) -> (Vec<f64>, [f64; N_OUTPUTS]) {
        let mut h = vec![0.0; self.hidden];
        for j in 0..self.hidden {
            let row = &self.w1[j * N_INPUTS..(j + 1) * N_INPUTS];
            let mut z = self.b1[j];
            for i in 0..N_INPUTS {
                z += row[i] * x[i];
            }
            h[j] = sigmoid(z);
        }

        let mut y = [0.0; N_OUTPUTS];
        for k in 0..N_OUTPUTS {
            let row = &self.w2[k * self.hidden..(k + 1) * self.hidden];
            let mut z = self.b2[k];
            for j in 0..self.hidden {
                z += row[j] * h[j];
            }
            y[k] = sigmoid(z);
        }
        (h, y)
    }

    /// Les six probabilités d'issue, **normalisées** pour sommer à 1 (les
    /// sigmoïdes brutes n'y sont pas contraintes, or les six issues sont
    /// exclusives et exhaustives).
    pub fn outcome_probs(&self, board: &Board) -> [f64; N_OUTPUTS] {
        let x = encode(board);
        let (_, mut y) = self.forward(&x);
        let total: f64 = y.iter().sum();
        if total > 1e-12 {
            for v in &mut y {
                *v /= total;
            }
        }
        y
    }

    /// L'équité : espérance de points du joueur à jouer, dans `[-3, +3]`.
    pub fn equity(&self, board: &Board) -> f64 {
        let p = self.outcome_probs(board);
        (p[0] + 2.0 * p[1] + 3.0 * p[2]) - (p[3] + 2.0 * p[4] + 3.0 * p[5])
    }

    /// Probabilité (normalisée) que le joueur à jouer gagne, tous scores
    /// confondus.
    pub fn p_win(&self, board: &Board) -> f64 {
        let p = self.outcome_probs(board);
        p[0] + p[1] + p[2]
    }

    /// Rétropropagation : gradient de **chaque** sortie par rapport à tous les
    /// poids, à partir de l'entrée `x`, des activations cachées `h` et des
    /// sorties `y` calculées par `forward`. Une entrée du vecteur résultat par
    /// sortie (TD(λ) entretient une trace par sortie).
    pub fn output_gradients(&self, x: &[f64; N_INPUTS], h: &[f64], y: &[f64; N_OUTPUTS]) -> Vec<Gradients> {
        let mut out = Vec::with_capacity(N_OUTPUTS);
        for k in 0..N_OUTPUTS {
            // Dérivée de la sigmoïde à la sortie k : sigmoid'(z2) = y·(1 − y).
            let dy_dz2 = y[k] * (1.0 - y[k]);

            let mut g_w1 = vec![0.0; self.w1.len()];
            let mut g_b1 = vec![0.0; self.hidden];
            let mut g_w2 = vec![0.0; self.w2.len()];
            let mut g_b2 = vec![0.0; N_OUTPUTS];

            let w2_row = &self.w2[k * self.hidden..(k + 1) * self.hidden];
            for j in 0..self.hidden {
                // Couche de sortie : seule la ligne k de w2 touche la sortie k.
                g_w2[k * self.hidden + j] = dy_dz2 * h[j]; // ∂y_k/∂w2[k][j]

                // Rétropropagation vers la couche cachée.
                let dy_dh = dy_dz2 * w2_row[j]; // ∂y_k/∂h[j]
                let dy_dz1 = dy_dh * h[j] * (1.0 - h[j]); // × sigmoid'(z1[j])
                g_b1[j] = dy_dz1; // ∂y_k/∂b1[j]

                let base = j * N_INPUTS;
                for i in 0..N_INPUTS {
                    g_w1[base + i] += dy_dz1 * x[i]; // ∂y_k/∂w1[j][i]
                }
            }
            g_b2[k] = dy_dz2; // ∂y_k/∂b2[k]

            out.push(Gradients {
                w1: g_w1,
                b1: g_b1,
                w2: g_w2,
                b2: g_b2,
            });
        }
        out
    }

    /// Un pas de descente de gradient pour rapprocher les sorties de `target`
    /// (erreur quadratique, sortie par sortie). Met à jour les poids et renvoie
    /// l'erreur quadratique moyenne observée *avant* le pas.
    pub fn train_step(&mut self, x: &[f64; N_INPUTS], target: &[f64; N_OUTPUTS], lr: f64) -> f64 {
        let (h, y) = self.forward(x);
        let grads = self.output_gradients(x, &h, &y);
        let mut sq_err = 0.0;
        for k in 0..N_OUTPUTS {
            let err = target[k] - y[k];
            sq_err += err * err;
            self.apply_update(&grads[k], lr * err); // Δw = lr · (t_k − y_k) · ∂y_k/∂w
        }
        sq_err / N_OUTPUTS as f64
    }

    /// Un jeu de gradients/traces de la bonne forme (une par sortie),
    /// initialisées à zéro.
    pub fn zero_traces(&self) -> Vec<Gradients> {
        (0..N_OUTPUTS)
            .map(|_| Gradients {
                w1: vec![0.0; self.w1.len()],
                b1: vec![0.0; self.hidden],
                w2: vec![0.0; self.w2.len()],
                b2: vec![0.0; N_OUTPUTS],
            })
            .collect()
    }

    /// Applique `Δw = facteur · trace` à tous les poids. Brique de la mise à
    /// jour TD(λ) : `facteur = α · erreur_TD_k`, `trace` = trace de la sortie k.
    pub fn apply_update(&mut self, trace: &Gradients, factor: f64) {
        for k in 0..self.w1.len() {
            self.w1[k] += factor * trace.w1[k];
        }
        for j in 0..self.hidden {
            self.b1[j] += factor * trace.b1[j];
        }
        for k in 0..self.w2.len() {
            self.w2[k] += factor * trace.w2[k];
        }
        for k in 0..N_OUTPUTS {
            self.b2[k] += factor * trace.b2[k];
        }
    }

    /// Sauvegarde les poids dans un fichier texte : un marqueur de format, la
    /// taille de la couche cachée, puis un nombre par ligne. Le format `f64` de
    /// Rust est réécrit sans perte, donc le rechargement est exact.
    pub fn save(&self, path: impl AsRef<Path>) -> std::io::Result<()> {
        use std::fmt::Write as _;
        let mut s = String::new();
        let _ = writeln!(s, "{SAVE_MAGIC}");
        let _ = writeln!(s, "{}", self.hidden);
        for v in self
            .w1
            .iter()
            .chain(self.b1.iter())
            .chain(self.w2.iter())
            .chain(self.b2.iter())
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

        match it.next() {
            Some(SAVE_MAGIC) => {}
            _ => {
                return Err(Error::new(
                    ErrorKind::InvalidData,
                    "ancien format de réseau (1 sortie) : ré-entraîne avec \
                     `cargo run --release --bin train`",
                ));
            }
        }
        let hidden: usize = it
            .next()
            .and_then(|s| s.parse().ok())
            .ok_or_else(|| Error::new(ErrorKind::InvalidData, "taille cachée manquante"))?;
        let nums: Vec<f64> = it.filter_map(|s| s.parse().ok()).collect();

        let n_w1 = hidden * N_INPUTS;
        let n_w2 = N_OUTPUTS * hidden;
        let expected = n_w1 + hidden + n_w2 + N_OUTPUTS;
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
            w2: nums[n_w1 + hidden..n_w1 + hidden + n_w2].to_vec(),
            b2: nums[n_w1 + hidden + n_w2..].to_vec(),
        })
    }
}

impl Evaluator for Net {
    /// La note d'une position est son équité (espérance de points).
    fn evaluate(&self, board: &Board) -> f64 {
        self.equity(board)
    }

    /// Gagner `points` vaut exactement `points` : cohérent avec l'échelle de
    /// l'équité, et une victoire sûre à 1 point reste battable par une
    /// position à fort espoir de gammon (équité proche de 2) — c'est voulu.
    fn terminal_value(&self, points: u8) -> f64 {
        points as f64
    }

    fn win_prob(&self, board: &Board) -> Option<f64> {
        Some(self.p_win(board))
    }
}

/// Agent qui joue l'argmax de l'équité d'un réseau aux poids aléatoires.
/// Pratique pour tester le pipeline ; il faudra l'entraîner pour qu'il soit bon.
pub fn net_agent(hidden: usize, seed: u64) -> GreedyAgent<Net> {
    GreedyAgent::new(Net::new_random(hidden, seed))
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine::agent::Agent;
    use engine::agent::random::RandomAgent;
    use engine::dice::Dice;
    use engine::game::play;

    #[test]
    fn tailles_des_poids() {
        let net = Net::new_random(40, 1);
        assert_eq!(net.w1.len(), 40 * N_INPUTS);
        assert_eq!(net.b1.len(), 40);
        assert_eq!(net.w2.len(), N_OUTPUTS * 40);
        assert_eq!(net.b2.len(), N_OUTPUTS);
    }

    #[test]
    fn sorties_normalisees_et_deterministes() {
        let net = Net::new_random(40, 42);
        let p1 = net.outcome_probs(&Board::starting_position());
        let p2 = net.outcome_probs(&Board::starting_position());
        assert_eq!(p1, p2, "le réseau doit être déterministe");
        let total: f64 = p1.iter().sum();
        assert!((total - 1.0).abs() < 1e-9, "probas non normalisées : {total}");
        let e = net.equity(&Board::starting_position());
        assert!((-3.0..=3.0).contains(&e), "équité hors bornes : {e}");
    }

    #[test]
    fn apprend_une_cible_fixe() {
        // Surapprentissage : sur une seule entrée, les sorties doivent converger
        // vers la cible. Si la backprop est fausse, ça ne converge pas.
        let mut net = Net::new_random(10, 5);
        let x = encode(&Board::starting_position());
        let target = [0.9, 0.05, 0.02, 0.2, 0.1, 0.05];
        for _ in 0..3000 {
            net.train_step(&x, &target, 0.1);
        }
        let (_, y) = net.forward(&x);
        for k in 0..N_OUTPUTS {
            assert!(
                (y[k] - target[k]).abs() < 0.02,
                "sortie {k} : y={} n'a pas convergé vers {}",
                y[k],
                target[k]
            );
        }
    }

    #[test]
    fn apprend_a_distinguer_deux_positions() {
        // Deux entrées distinctes vers deux cibles distinctes : seul un gradient
        // correct à travers la couche cachée permet de les séparer.
        let mut net = Net::new_random(20, 3);
        let xa = encode(&Board::starting_position());
        let xb = encode(&Board::from_parts({
                let mut p = [0i8; 24];
                p[0] = 2;
                p[23] = -2;
                p
            }, [0, 0], [0, 0]));
        let ta = [0.8, 0.1, 0.0, 0.1, 0.0, 0.0];
        let tb = [0.2, 0.0, 0.0, 0.7, 0.1, 0.0];
        for _ in 0..5000 {
            net.train_step(&xa, &ta, 0.1);
            net.train_step(&xb, &tb, 0.1);
        }
        let (_, ya) = net.forward(&xa);
        let (_, yb) = net.forward(&xb);
        assert!((ya[0] - 0.8).abs() < 0.05, "ya0={}", ya[0]);
        assert!((yb[3] - 0.7).abs() < 0.05, "yb3={}", yb[3]);
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
        assert_eq!(net.equity(&b1), loaded.equity(&b1));
        assert_eq!(net.outcome_probs(&b2), loaded.outcome_probs(&b2));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn l_ancien_format_est_rejete_proprement() {
        // Un fichier v1 commençait directement par la taille cachée : il doit
        // être refusé avec une erreur claire, pas un plantage ni des poids
        // silencieusement faux.
        let path = std::env::temp_dir().join("backgammon_test_net_v1.txt");
        std::fs::write(&path, "40\n0.1\n0.2\n").unwrap();
        // `unwrap_err()` exigerait `Net: Debug` (pour afficher le cas `Ok`) :
        // un `match` fait l'affaire sans dériver Debug sur des milliers de poids.
        let err = match Net::load(&path) {
            Err(e) => e,
            Ok(_) => panic!("le format v1 ne doit pas se charger"),
        };
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
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
        // Le réseau peut maintenant doubler en cours de partie (videau) :
        // points = 1..3 × valeur du videau.
        assert!(points >= 1);
    }
}
