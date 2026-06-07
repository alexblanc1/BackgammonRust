//! Entraînement du réseau par self-play en TD(λ).
//!
//! Le réseau joue contre lui-même. Après chaque coup, on corrige la prédiction
//! précédente vers sa cible (différence temporelle), avec des traces
//! d'éligibilité pour propager le signal vers les coups passés. La cible finale
//! est le vrai résultat de la partie.
//!
//! Subtilité de perspective : `V(s)` estime la proba que le joueur à jouer
//! gagne. Après un coup et `swap_perspective`, la perspective bascule, donc la
//! cible de la prédiction précédente est `1 − V(suivant)`. On suit la parité
//! des retournements pour gérer aussi le cas où un joueur passe (aucun coup).

use crate::agent::Agent;
use crate::agent::heuristic::heuristic_agent;
use crate::agent::random::RandomAgent;
use crate::board::Board;
use crate::dice::Dice;
use crate::encoding::encode;
use crate::eval::GreedyAgent;
use crate::game::play;
use crate::moves::legal_plays;
use crate::net::{Gradients, Net};
use crate::player::Player;

/// Garde-fou : abandonne une partie d'entraînement anormalement longue.
const MAX_PLIES: u32 = 10_000;

/// Indice du coup maximisant la valeur du réseau (politique gloutonne).
fn argmax_value(net: &Net, plays: &[Board]) -> usize {
    let mut best = 0usize;
    let mut best_score = f64::NEG_INFINITY;
    for (i, b) in plays.iter().enumerate() {
        let v = net.value(b);
        if v > best_score {
            best_score = v;
            best = i;
        }
    }
    best
}

/// Trace ← γλ·trace + signe·gradient.
///
/// On accumule le gradient **signé** par la perspective courante : la trace
/// représente ainsi le gradient d'une valeur canonique unique `U = P(Blancs
/// gagnent)`, cohérente d'un bout à l'autre de la partie. Sans ce signe, la
/// trace mélangerait deux quantités opposées (P(Blancs) et P(Noirs)) et
/// l'entraînement diverge.
fn decay_accumulate(trace: &mut Gradients, grad: &Gradients, gl: f64, sign: f64) {
    for k in 0..trace.w1.len() {
        trace.w1[k] = gl * trace.w1[k] + sign * grad.w1[k];
    }
    for j in 0..trace.b1.len() {
        trace.b1[j] = gl * trace.b1[j] + sign * grad.b1[j];
        trace.w2[j] = gl * trace.w2[j] + sign * grad.w2[j];
    }
    trace.b2 = gl * trace.b2 + sign * grad.b2;
}

/// Joue une partie en self-play et met à jour `net` par TD(λ).
///
/// Tout est exprimé dans la valeur canonique `U = P(Blancs gagnent)`. Pour une
/// prédiction `p = V(after)` (proba que le joueur courant gagne) faite avec la
/// perspective de signe `sign` (`+1` = Blancs, `−1` = Noirs) :
/// `U = p` si Blancs jouent, `1 − p` sinon ; et `∇U = sign · ∇p`. La cible TD
/// devient simplement le `U` suivant — plus de « flip » explicite.
fn train_one_game(net: &mut Net, dice: &mut Dice, alpha: f64, lambda: f64) {
    let gl = lambda; // γ = 1 (aucune récompense intermédiaire)
    let mut trace = net.zero_gradients();
    let mut board = Board::starting_position();
    let mut sign = 1.0; // Blancs jouent en premier (référence de U)
    let mut prev_u: Option<f64> = None;

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
        let idx = argmax_value(net, &plays);
        let after = plays[idx].clone();
        let x = encode(&after);
        let (h, p) = net.forward(&x);

        // Valeur canonique de cette prédiction.
        let u = if sign > 0.0 { p } else { 1.0 - p };

        // 1) Corrige la prédiction précédente vers la cible TD = U courant.
        if let Some(pu) = prev_u {
            net.apply_update(&trace, alpha * (u - pu));
        }

        // 2) Accumule ∇U = signe · ∇p dans la trace d'éligibilité.
        let grad = net.output_gradient(&x, &h, p);
        decay_accumulate(&mut trace, &grad, gl, sign);

        // 3) Fin de partie : le joueur courant gagne, donc en canonique
        //    U_terminal = 1 si les Blancs gagnent, 0 sinon.
        if after.win_check().is_some() {
            let u_terminal = if sign > 0.0 { 1.0 } else { 0.0 };
            net.apply_update(&trace, alpha * (u_terminal - u));
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
        // Quelques parties : on vérifie que rien ne panique et que la sortie
        // reste un nombre valide dans (0, 1) (pas de NaN/divergence).
        let mut net = Net::new_random(12, 1);
        train_self_play(&mut net, 40, 0.1, 0.7, 123);
        let v = net.value(&Board::starting_position());
        assert!(v.is_finite() && v > 0.0 && v < 1.0, "valeur invalide après entraînement : {v}");
    }

    #[test]
    #[ignore = "lent (~45 s) : entraînement réel ; lancer avec `cargo test -- --ignored`"]
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
