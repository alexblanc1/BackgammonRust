//! Agent heuristique : évalue chaque position résultante et joue la meilleure.
//!
//! C'est la baseline de l'IA, sans apprentissage : une simple somme pondérée
//! de critères. Elle sert d'adversaire correct et de point de comparaison pour
//! les versions futures (le réseau de neurones devra la battre).

use crate::agent::Agent;
use crate::board::Board;
use crate::game::GameState;

/// Joue, parmi les positions légales, celle qui maximise `evaluate`.
pub struct HeuristicAgent;

impl HeuristicAgent {
    pub fn new() -> HeuristicAgent {
        HeuristicAgent
    }
}

impl Agent for HeuristicAgent {
    fn choose_play(&mut self, _state: &GameState, legal: &[Board]) -> usize {
        // argmax : on garde l'indice du candidat au meilleur score.
        let mut best = 0usize;
        let mut best_score = f64::NEG_INFINITY;
        for (i, candidate) in legal.iter().enumerate() {
            let score = evaluate(candidate);
            if score > best_score {
                best_score = score;
                best = i;
            }
        }
        best
    }
}

// --- L'évaluateur ------------------------------------------------------------

// Poids des critères. Réglables ; ces valeurs suffisent à dominer le hasard.
const W_PIP: f64 = 1.0; // par pip d'avance
const W_OFF: f64 = 4.0; // par pion sorti d'avance
const W_BAR: f64 = 6.0; // par pion adverse sur la barre (le mien : malus)
const W_BLOT: f64 = 3.0; // par blot exposé (le mien : malus)
const W_HOME: f64 = 4.0; // par point fait en jan intérieur d'avance
const W_MADE: f64 = 1.0; // par point fait d'avance (structure)

/// Évalue une position **du point de vue du joueur à jouer** : plus le score
/// est élevé, meilleure est la position pour lui. Antisymétrique : évaluer la
/// position retournée donne l'opposé (`evaluate(swap(b)) == -evaluate(b)`).
pub fn evaluate(b: &Board) -> f64 {
    // Les critères de l'adversaire = les mêmes fonctions vues de son côté.
    let opp = b.swap_perspective();

    let pip_diff = pip(&opp) as f64 - pip(b) as f64; // + = je suis devant dans la course
    let off_diff = b.off[0] as f64 - b.off[1] as f64;
    let bar_diff = b.bar[1] as f64 - b.bar[0] as f64; // + = l'adversaire est sur la barre
    let blot_diff = hittable_blots(&opp) as f64 - hittable_blots(b) as f64;
    let home_diff = home_points(b) as f64 - home_points(&opp) as f64;
    let made_diff = made_points(b) as f64 - made_points(&opp) as f64;

    W_PIP * pip_diff
        + W_OFF * off_diff
        + W_BAR * bar_diff
        + W_BLOT * blot_diff
        + W_HOME * home_diff
        + W_MADE * made_diff
}

/// Pip count du joueur à jouer : distance totale de ses pions à la sortie.
/// Un pion sur la case `p` est à `p + 1` pips ; un pion sur la barre à 25.
fn pip(b: &Board) -> u32 {
    let mut total = 0u32;
    for p in 0..24 {
        if b.points[p] > 0 {
            total += b.points[p] as u32 * (p as u32 + 1);
        }
    }
    total + b.bar[0] as u32 * 25
}

/// Nombre de tes blots (cases à exactement 1 pion à toi) exposés à une frappe
/// directe : un pion adverse à 6 cases ou moins derrière (l'adversaire avance
/// des petits index vers les grands), ou un adverse sur la barre pouvant entrer
/// dessus (blot dans ton jan, index 0..6).
fn hittable_blots(b: &Board) -> u32 {
    let mut n = 0;
    for p in 0..24usize {
        if b.points[p] != 1 {
            continue; // pas un blot à toi
        }
        let mut hittable = false;
        let lo = p.saturating_sub(6);
        for q in lo..p {
            if b.points[q] < 0 {
                hittable = true; // tir direct d'un pion adverse situé derrière
                break;
            }
        }
        if !hittable && b.bar[1] > 0 && p < 6 {
            hittable = true; // frappe possible à l'entrée depuis la barre adverse
        }
        if hittable {
            n += 1;
        }
    }
    n
}

/// Nombre de tes points faits (≥ 2 pions) dans ton jan intérieur (index 0..6).
fn home_points(b: &Board) -> u32 {
    (0..6).filter(|&p| b.points[p] >= 2).count() as u32
}

/// Nombre total de tes points faits (≥ 2 pions) sur le plateau.
fn made_points(b: &Board) -> u32 {
    (0..24).filter(|&p| b.points[p] >= 2).count() as u32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::random::RandomAgent;
    use crate::dice::Dice;
    use crate::game::play;
    use crate::player::Player;

    #[test]
    fn position_de_depart_neutre() {
        // La position de départ est symétrique → score exactement nul.
        assert_eq!(evaluate(&Board::starting_position()), 0.0);
    }

    #[test]
    fn evaluation_antisymetrique() {
        // Propriété somme nulle : ce qui est bon pour moi est mauvais pour l'autre.
        let mut points = [0i8; 24];
        points[23] = 2;
        points[12] = 5;
        points[7] = 3;
        points[5] = 5;
        points[0] = -3;
        points[11] = -5;
        points[16] = -3;
        points[18] = -4;
        let b = Board {
            points,
            bar: [0, 0],
            off: [0, 0],
        };
        let s = evaluate(&b);
        let s_opp = evaluate(&b.swap_perspective());
        assert!((s + s_opp).abs() < 1e-9, "s={s}, s_opp={s_opp}");
    }

    #[test]
    fn l_heuristique_bat_le_hasard() {
        let games = 200u64;
        let mut heur_wins = 0u32;
        for seed in 0..games {
            // On alterne les couleurs pour éviter tout biais de premier coup.
            let heuristic_is_white = seed % 2 == 0;
            let mut agents: [Box<dyn Agent>; 2] = if heuristic_is_white {
                [
                    Box::new(HeuristicAgent::new()),
                    Box::new(RandomAgent::new(seed * 7 + 1)),
                ]
            } else {
                [
                    Box::new(RandomAgent::new(seed * 7 + 1)),
                    Box::new(HeuristicAgent::new()),
                ]
            };
            let mut dice = Dice::new(seed.wrapping_mul(2_654_435_761).wrapping_add(12_345));

            let (winner, _pts) = play(&mut agents, &mut dice);
            let heuristic_won = (winner == Player::White) == heuristic_is_white;
            if heuristic_won {
                heur_wins += 1;
            }
        }
        let rate = heur_wins as f64 / games as f64;
        eprintln!("Heuristique vs hasard : {heur_wins}/{games} ({:.0}%)", rate * 100.0);
        assert!(rate > 0.65, "l'heuristique ne domine pas assez : {rate:.2}");
    }
}
