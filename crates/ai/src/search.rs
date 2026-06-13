//! Recherche au moment de jouer : expectiminimax et rollouts Monte-Carlo.
//!
//! L'agent glouton (`GreedyAgent`) note chaque position résultante et joue la
//! meilleure : il regarde **0 coup** en avant. Ici, on regarde plus loin :
//!
//! - **Expectiminimax** : comme le minimax des échecs, mais avec des **nœuds de
//!   chance** entre les coups — on ne sait pas ce que l'adversaire va lancer,
//!   donc on fait la moyenne sur les 21 lancers possibles, pondérée par leur
//!   probabilité (1/36 pour un double, 2/36 sinon). À chaque lancer,
//!   l'adversaire répond par son meilleur coup. `plies = 1` regarde la réponse
//!   adverse, `plies = 2` ajoute notre réplique, etc.
//!
//! - **Rollouts Monte-Carlo** : pour départager les meilleurs candidats, on
//!   **joue réellement** la fin de la partie un certain nombre de fois (les
//!   deux camps suivant la politique gloutonne, dés aléatoires) et on moyenne
//!   les points obtenus. C'est une estimation sans biais de la vraie valeur,
//!   au prix de la variance (d'où plusieurs parties par candidat).
//!
//! Coût : chaque pli supplémentaire multiplie le travail par ~21 lancers ×
//! le nombre de coups. D'où deux approximations classiques :
//! - on n'explore en profondeur que les `top_k` meilleurs candidats au premier
//!   niveau (élagage par l'évaluation statique) ;
//! - aux niveaux suivants, le coup de chaque camp est choisi par l'évaluation
//!   statique, et seule la **valeur** est calculée récursivement (style GNUBG).

use engine::agent::Agent;
use engine::board::Board;
use engine::dice::{Dice, all_rolls};
use crate::eval::Evaluator;
use engine::game::GameState;
use engine::moves::legal_plays;

/// Indice du coup maximisant `eval.evaluate` (les positions gagnantes étant
/// notées par `terminal_value`, qui domine).
fn argmax_static<E: Evaluator>(eval: &E, plays: &[Board]) -> usize {
    let mut best = 0usize;
    let mut best_score = f64::NEG_INFINITY;
    for (i, b) in plays.iter().enumerate() {
        let score = match b.win_check() {
            Some(pts) => eval.terminal_value(pts),
            None => eval.evaluate(b),
        };
        if score > best_score {
            best_score = score;
            best = i;
        }
    }
    best
}

/// Valeur d'une position `after`, **du point de vue du joueur qui vient de
/// jouer** (le plateau est encore dans sa perspective), en regardant `plies`
/// demi-coups en avant.
pub fn move_value<E: Evaluator>(eval: &E, after: &Board, plies: u32) -> f64 {
    if let Some(pts) = after.win_check() {
        return eval.terminal_value(pts);
    }
    if plies == 0 {
        return eval.evaluate(after);
    }
    // Nœud de chance + nœud adverse : l'adversaire lance, joue au mieux, et sa
    // valeur est l'opposée de la nôtre (jeu à somme nulle, évaluateur
    // antisymétrique).
    -pre_roll_value(eval, &after.swap_perspective(), plies)
}

/// Valeur espérée d'une position **avant lancer**, du point de vue du joueur
/// qui va lancer : moyenne pondérée, sur les 21 lancers, de la valeur de son
/// meilleur coup.
fn pre_roll_value<E: Evaluator>(eval: &E, board: &Board, plies: u32) -> f64 {
    debug_assert!(plies >= 1);
    let mut total = 0.0;
    for (roll, prob) in all_rolls() {
        let plays = legal_plays(board, &roll);
        let v = if plays.is_empty() {
            // Aucun coup : le joueur passe, la position reste — équivaut à un
            // « coup nul » dont on évalue la suite.
            move_value(eval, board, plies - 1)
        } else {
            // Approximation GNUBG : le coup est choisi par l'évaluation
            // statique, seule sa valeur est affinée récursivement.
            let best = argmax_static(eval, &plays);
            move_value(eval, &plays[best], plies - 1)
        };
        total += prob * v;
    }
    total
}

/// Agent expectiminimax : note les `top_k` meilleurs candidats (au sens de
/// l'évaluation statique) en regardant `plies` demi-coups en avant, et joue le
/// meilleur.
///
/// `plies = 1` : on intègre la meilleure réponse adverse sur tous ses lancers
/// (≈ « 2-ply » au sens usuel). `plies = 2` ajoute notre propre réplique.
/// Chaque pli coûte ~21× plus cher : au-delà de 2, c'est lent.
pub struct ExpectiAgent<E> {
    evaluator: E,
    plies: u32,
    top_k: usize,
}

impl<E> ExpectiAgent<E> {
    pub fn new(evaluator: E, plies: u32, top_k: usize) -> ExpectiAgent<E> {
        ExpectiAgent {
            evaluator,
            plies,
            top_k: top_k.max(1),
        }
    }
}

/// Les indices des `k` meilleurs candidats selon l'évaluation statique.
fn top_k_indices<E: Evaluator>(eval: &E, plays: &[Board], k: usize) -> Vec<usize> {
    let mut scored: Vec<(usize, f64)> = plays
        .iter()
        .enumerate()
        .map(|(i, b)| {
            let s = match b.win_check() {
                Some(pts) => eval.terminal_value(pts),
                None => eval.evaluate(b),
            };
            (i, s)
        })
        .collect();
    // Tri décroissant par score. `partial_cmp` car les f64 ne sont que
    // partiellement ordonnés (NaN) ; nos scores n'en produisent pas.
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scored.into_iter().take(k).map(|(i, _)| i).collect()
}

impl<E: Evaluator> Agent for ExpectiAgent<E> {
    fn choose_play(&mut self, _state: &GameState, legal: &[Board]) -> usize {
        // Un coup gagnant immédiat à 3 points est imbattable : inutile de chercher.
        if let Some(i) = legal.iter().position(|b| b.win_check() == Some(3)) {
            return i;
        }
        let candidates = top_k_indices(&self.evaluator, legal, self.top_k);
        let mut best = candidates[0];
        let mut best_value = f64::NEG_INFINITY;
        for i in candidates {
            let v = move_value(&self.evaluator, &legal[i], self.plies);
            if v > best_value {
                best_value = v;
                best = i;
            }
        }
        best
    }

    fn should_double(&mut self, state: &GameState) -> bool {
        crate::eval::default_should_double(&self.evaluator, state)
    }

    fn should_accept_double(&mut self, state: &GameState) -> bool {
        crate::eval::default_should_accept(&self.evaluator, state)
    }
}

// --- Rollouts Monte-Carlo -----------------------------------------------------

/// Joue la fin de partie depuis `after` (le joueur au trait vient de jouer,
/// l'adversaire va lancer), les deux camps suivant la politique gloutonne de
/// `eval`. Renvoie les points du point de vue du joueur qui vient de jouer
/// (+pts s'il gagne, −pts sinon).
fn rollout_once<E: Evaluator>(eval: &E, after: &Board, dice: &mut Dice) -> f64 {
    let mut board = after.swap_perspective();
    let mut sign = -1.0; // c'est l'adversaire qui joue maintenant
    loop {
        let roll = dice.roll();
        let plays = legal_plays(&board, &roll);
        if !plays.is_empty() {
            let i = argmax_static(eval, &plays);
            board = plays[i].clone();
            if let Some(pts) = board.win_check() {
                return sign * pts as f64;
            }
        }
        board = board.swap_perspective();
        sign = -sign;
    }
}

/// Agent à rollouts : présélectionne les `top_k` candidats par évaluation
/// statique, joue `games` fins de partie complètes pour chacun, et retient le
/// candidat à la meilleure moyenne de points.
pub struct RolloutAgent<E> {
    evaluator: E,
    games: usize,
    top_k: usize,
    dice: Dice,
}

impl<E> RolloutAgent<E> {
    /// `games` parties simulées par candidat. Les dés du rollout sont semés par
    /// `seed` (reproductible) ; utiliser `Dice::random()` à la place via
    /// [`RolloutAgent::with_dice`] pour des simulations imprévisibles.
    pub fn new(evaluator: E, games: usize, top_k: usize, seed: u64) -> RolloutAgent<E> {
        RolloutAgent {
            evaluator,
            games: games.max(1),
            top_k: top_k.max(1),
            dice: Dice::new(seed),
        }
    }

    pub fn with_dice(evaluator: E, games: usize, top_k: usize, dice: Dice) -> RolloutAgent<E> {
        RolloutAgent {
            evaluator,
            games: games.max(1),
            top_k: top_k.max(1),
            dice,
        }
    }
}

impl<E: Evaluator> Agent for RolloutAgent<E> {
    fn choose_play(&mut self, _state: &GameState, legal: &[Board]) -> usize {
        if let Some(i) = legal.iter().position(|b| b.win_check() == Some(3)) {
            return i;
        }
        // Un coup qui gagne tout de suite : le rollout n'apprendrait rien.
        if let Some(i) = legal.iter().position(|b| b.win_check().is_some()) {
            return i;
        }
        let candidates = top_k_indices(&self.evaluator, legal, self.top_k);
        let mut best = candidates[0];
        let mut best_mean = f64::NEG_INFINITY;
        for i in candidates {
            let mut total = 0.0;
            for _ in 0..self.games {
                total += rollout_once(&self.evaluator, &legal[i], &mut self.dice);
            }
            let mean = total / self.games as f64;
            if mean > best_mean {
                best_mean = mean;
                best = i;
            }
        }
        best
    }

    fn should_double(&mut self, state: &GameState) -> bool {
        crate::eval::default_should_double(&self.evaluator, state)
    }

    fn should_accept_double(&mut self, state: &GameState) -> bool {
        crate::eval::default_should_accept(&self.evaluator, state)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::heuristic::{HeuristicEvaluator, heuristic_agent};
    use engine::game::play;
    use engine::player::Player;

    /// L'expectiminimax (1 pli) doit jouer une partie entière sans paniquer et
    /// battre nettement l'agent glouton de la même heuristique sur un petit
    /// échantillon — regarder la réponse adverse ne peut qu'aider.
    #[test]
    fn expectiminimax_joue_et_tient_tete_au_glouton() {
        let games = 30u64;
        let mut wins = 0u32;
        for s in 0..games {
            let deep_is_white = s % 2 == 0;
            let deep = || Box::new(ExpectiAgent::new(HeuristicEvaluator::new(), 1, 4));
            let greedy = || Box::new(heuristic_agent());
            let mut agents: [Box<dyn Agent>; 2] = if deep_is_white {
                [deep(), greedy()]
            } else {
                [greedy(), deep()]
            };
            let mut dice = Dice::new(s.wrapping_mul(977).wrapping_add(3));
            let (winner, _) = play(&mut agents, &mut dice);
            if (winner == Player::White) == deep_is_white {
                wins += 1;
            }
        }
        // Seuil volontairement bas (petit échantillon) : on veut surtout
        // vérifier que la recherche ne joue pas n'importe quoi.
        assert!(
            wins as f64 / games as f64 >= 0.4,
            "l'expectiminimax s'effondre face au glouton : {wins}/{games}"
        );
    }

    /// Le rollout doit lui aussi finir ses parties proprement.
    #[test]
    fn rollout_joue_une_partie_entiere() {
        let mut agents: [Box<dyn Agent>; 2] = [
            Box::new(RolloutAgent::new(HeuristicEvaluator::new(), 4, 3, 11)),
            Box::new(heuristic_agent()),
        ];
        let mut dice = Dice::new(2025);
        let (_winner, points) = play(&mut agents, &mut dice);
        assert!((1..=3).contains(&points));
    }

    /// Une victoire immédiate disponible doit être jouée sans hésiter.
    #[test]
    fn la_recherche_prend_la_victoire_immediate() {
        // Position artificielle : il me reste 1 pion en case 0, je sors et gagne.
        let mut points = [0i8; 24];
        points[0] = 1;
        points[23] = -2;
        let before = Board::from_parts(points, [0, 0], [14, 13]);
        let plays = legal_plays(&before, &engine::dice::Roll { d1: 1, d2: 2 });
        assert!(plays.iter().any(|b| b.win_check().is_some()));

        let state = GameState::new();
        let mut agent = ExpectiAgent::new(HeuristicEvaluator::new(), 1, 4);
        let i = agent.choose_play(&state, &plays);
        assert!(plays[i].win_check().is_some(), "la recherche doit gagner immédiatement");
    }
}
