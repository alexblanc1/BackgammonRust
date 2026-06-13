//! Entraînement du réseau par self-play en TD(λ).
//!
//! Le réseau joue contre lui-même. Après chaque coup, on corrige la prédiction
//! précédente vers sa cible (différence temporelle), avec des traces
//! d'éligibilité pour propager le signal vers les coups passés. La cible finale
//! est le vrai résultat de la partie — désormais un vecteur **one-hot** sur les
//! six issues (gain/perte × simple/gammon/backgammon).
//!
//! Subtilité de perspective : les sorties du réseau sont du point de vue du
//! joueur à jouer. On travaille donc dans un vecteur **canonique** `U`, vu des
//! Blancs : `U = p` quand les Blancs jouent, et `U = swap(p)` quand les Noirs
//! jouent, où `swap` échange les triplets gain/perte (les gains des Noirs sont
//! les pertes des Blancs). Les traces accumulent les gradients **permutés** de
//! la même façon, pour rester cohérentes d'un bout à l'autre de la partie.

use engine::agent::Agent;
use crate::heuristic::heuristic_agent;
use engine::agent::random::RandomAgent;
use engine::board::Board;
use engine::dice::Dice;
use crate::encoding::encode;
use crate::eval::GreedyAgent;
use engine::game::play;
use engine::moves::legal_plays;
use crate::net::{Gradients, N_OUTPUTS, Net};
use engine::player::Player;

/// Garde-fou : abandonne une partie d'entraînement anormalement longue.
const MAX_PLIES: u32 = 10_000;

/// Indice du coup maximisant l'équité du réseau (politique gloutonne).
fn argmax_equity(net: &Net, plays: &[Board]) -> usize {
    let mut best = 0usize;
    let mut best_score = f64::NEG_INFINITY;
    for (i, b) in plays.iter().enumerate() {
        let v = net.equity(b);
        if v > best_score {
            best_score = v;
            best = i;
        }
    }
    best
}

/// L'indice canonique (vu des Blancs) de la sortie `k` prédite par le joueur
/// de signe `sign` : identité pour les Blancs, échange gain↔perte pour les
/// Noirs (sortie 0 « je gagne simple » des Noirs = « Blancs perdent simple »,
/// l'indice 3 en canonique).
fn canon(k: usize, sign: f64) -> usize {
    if sign > 0.0 { k } else { (k + 3) % N_OUTPUTS }
}

/// Pour chaque sortie canonique : trace ← γλ·trace + gradient permuté.
fn decay_accumulate(traces: &mut [Gradients], grads: &[Gradients], gl: f64, sign: f64) {
    for k in 0..N_OUTPUTS {
        let trace = &mut traces[k];
        // La sortie canonique k est prédite par la sortie canon⁻¹(k) du réseau ;
        // canon est sa propre inverse (échanger deux triplets deux fois = identité).
        let grad = &grads[canon(k, sign)];
        for i in 0..trace.w1.len() {
            trace.w1[i] = gl * trace.w1[i] + grad.w1[i];
        }
        for j in 0..trace.b1.len() {
            trace.b1[j] = gl * trace.b1[j] + grad.b1[j];
        }
        for i in 0..trace.w2.len() {
            trace.w2[i] = gl * trace.w2[i] + grad.w2[i];
        }
        for i in 0..trace.b2.len() {
            trace.b2[i] = gl * trace.b2[i] + grad.b2[i];
        }
    }
}

/// Le vecteur canonique (vu des Blancs) d'une prédiction `p` faite avec la
/// perspective de signe `sign`.
fn canon_vector(p: &[f64; N_OUTPUTS], sign: f64) -> [f64; N_OUTPUTS] {
    let mut u = [0.0; N_OUTPUTS];
    for k in 0..N_OUTPUTS {
        u[k] = p[canon(k, sign)];
    }
    u
}

/// Joue une partie en self-play et met à jour `net` par TD(λ).
fn train_one_game(net: &mut Net, dice: &mut Dice, alpha: f64, lambda: f64) {
    let gl = lambda; // γ = 1 (aucune récompense intermédiaire)
    let mut traces = net.zero_traces();
    let mut board = Board::starting_position();
    let mut sign = 1.0; // Blancs jouent en premier (référence de U)
    let mut prev_u: Option<[f64; N_OUTPUTS]> = None;

    for _ in 0..MAX_PLIES {
        let roll = dice.roll();
        let plays = legal_plays(&board, &roll);

        if plays.is_empty() {
            // Aucun coup : le joueur passe. Retournement sans nouvelle prédiction.
            board = board.swap_perspective();
            sign = -sign;
            continue;
        }

        // Politique gloutonne selon le réseau courant (on-policy).
        let idx = argmax_equity(net, &plays);
        let after = plays[idx].clone();
        let x = encode(&after);
        let (h, p) = net.forward(&x);

        // Vecteur canonique de cette prédiction.
        let u = canon_vector(&p, sign);

        // 1) Corrige la prédiction précédente vers la cible TD = U courant,
        //    sortie par sortie.
        if let Some(pu) = prev_u {
            for k in 0..N_OUTPUTS {
                net.apply_update(&traces[k], alpha * (u[k] - pu[k]));
            }
        }

        // 2) Accumule les gradients permutés dans les traces d'éligibilité.
        let grads = net.output_gradients(&x, &h, &p);
        decay_accumulate(&mut traces, &grads, gl, sign);

        // 3) Fin de partie : le joueur courant gagne `pts`. En canonique,
        //    l'issue est one-hot : « Blancs gagnent pts » si c'est lui le
        //    gagnant, « Blancs perdent pts » sinon.
        if let Some(pts) = after.win_check() {
            let mut u_terminal = [0.0; N_OUTPUTS];
            let win_idx = (pts - 1) as usize; // 1/2/3 points → indices 0/1/2
            u_terminal[canon(win_idx, sign)] = 1.0;
            for k in 0..N_OUTPUTS {
                net.apply_update(&traces[k], alpha * (u_terminal[k] - u[k]));
            }
            return;
        }

        board = after.swap_perspective();
        sign = -sign;
        prev_u = Some(u);
    }
}

/// Entraîne `net` par self-play TD(λ) sur `games` parties.
pub fn train_self_play(net: &mut Net, games: usize, alpha: f64, lambda: f64, seed: u64) {
    let mut dice = Dice::new(seed | 1);
    for _ in 0..games {
        train_one_game(net, &mut dice, alpha, lambda);
    }
}

// --- Évaluation (taux de victoire d'un réseau) -------------------------------

fn win_rate<F>(net: &Net, games: u64, seed: u64, mut opponent: F) -> f64
where
    F: FnMut(u64) -> Box<dyn Agent>,
{
    let mut wins = 0u64;
    for s in 0..games {
        let net_white = s % 2 == 0;
        let opp_seed = seed.wrapping_add(s).wrapping_mul(2_654_435_761).wrapping_add(1);
        let mut agents: [Box<dyn Agent>; 2] = if net_white {
            [Box::new(GreedyAgent::new(net.clone())), opponent(opp_seed)]
        } else {
            [opponent(opp_seed), Box::new(GreedyAgent::new(net.clone()))]
        };
        let mut dice = Dice::new(seed.wrapping_mul(40_503).wrapping_add(s).wrapping_add(1));
        let (winner, _) = play(&mut agents, &mut dice);
        if (winner == Player::White) == net_white {
            wins += 1;
        }
    }
    wins as f64 / games as f64
}

/// Taux de victoire du réseau (glouton) contre un agent aléatoire, couleurs
/// alternées pour éviter tout biais.
pub fn win_rate_vs_random(net: &Net, games: u64, seed: u64) -> f64 {
    win_rate(net, games, seed, |s| Box::new(RandomAgent::new(s)))
}

/// Taux de victoire du réseau (glouton) contre l'agent heuristique.
pub fn win_rate_vs_heuristic(net: &Net, games: u64, seed: u64) -> f64 {
    win_rate(net, games, seed, |_| Box::new(heuristic_agent()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn l_entrainement_tourne_sans_diverger() {
        // Quelques parties : on vérifie que rien ne panique et que les sorties
        // restent des nombres valides dans (0, 1) (pas de NaN/divergence).
        let mut net = Net::new_random(12, 1);
        train_self_play(&mut net, 40, 0.1, 0.7, 123);
        let p = net.outcome_probs(&Board::starting_position());
        for (k, v) in p.iter().enumerate() {
            assert!(v.is_finite() && *v >= 0.0 && *v <= 1.0, "sortie {k} invalide : {v}");
        }
        let e = net.equity(&Board::starting_position());
        assert!(e.is_finite(), "équité invalide : {e}");
    }

    #[test]
    #[ignore = "lent (quelques minutes) : entraînement réel ; lancer avec `cargo test -- --ignored`"]
    fn l_entrainement_bat_le_hasard() {
        // ~6000 parties pour dépasser le creux transitoire du self-play TD et
        // atteindre un réseau franchement supérieur au hasard.
        let mut net = Net::new_random(40, 7);
        let before = win_rate_vs_random(&net, 100, 555);
        train_self_play(&mut net, 6000, 0.1, 0.0, 42);
        let after = win_rate_vs_random(&net, 200, 555);
        assert!(after > before + 0.20, "pas d'amélioration : {before:.2} -> {after:.2}");
        assert!(after > 0.80, "le réseau entraîné devrait dominer le hasard : {after:.2}");
    }
}
