//! Génération des coups légaux.

use crate::board::Board;
use crate::dice::Roll;

/// Toutes les positions distinctes atteignables en jouant légalement le
/// lancer `roll`.
///
/// Doit respecter les règles d'usage obligatoire des dés : priorité à la
/// barre, nombre maximum de dés joués, règle du plus grand dé, légalité de la
/// sortie, et déduplication des positions identiques. L'implémentation passe
/// typiquement par une recherche récursive (DFS) qui appelle
/// `board.single_die_moves(...)`.
pub fn legal_plays(board: &Board, roll: &Roll) -> Vec<Board> {
    if roll.d1 == roll.d2 {
        // --- Double : jusqu'à quatre coups du même dé, le maximum possible. ---
        let lines = play_doubles(board, roll.d1, 4);
        let max = lines.iter().map(|(_, used)| *used).max().unwrap_or(0);
        if max == 0 {
            return Vec::new(); // aucun coup possible : le joueur passe son tour
        }
        let boards: Vec<Board> = lines
            .into_iter()
            .filter(|(_, used)| *used == max) // on ne garde que les lignes les plus longues
            .map(|(b, _)| b) // on ne garde que la position
            .collect();
        dedup(boards)
    } else {
        // --- Non-double : on doit jouer les DEUX dés si c'est possible. ---
        let both = boards_using_both(board, roll.d1, roll.d2);
        if !both.is_empty() {
            return dedup(both);
        }
        // Sinon, un seul dé : on joue le plus grand s'il est jouable, sinon le petit.
        let large = roll.d1.max(roll.d2);
        let small = roll.d1.min(roll.d2);

        let large_moves = board.single_die_moves(large);
        if !large_moves.is_empty() {
            return dedup(large_moves);
        }
        let small_moves = board.single_die_moves(small);
        if !small_moves.is_empty() {
            return dedup(small_moves);
        }
        Vec::new() // vraiment aucun coup possible
    }
}

/// Toutes les positions atteignables en jouant les DEUX dés `a` et `b`,
/// dans n'importe quel ordre.
fn boards_using_both(board: &Board, a: u8, b: u8) -> Vec<Board> {
    let mut out = Vec::new();
    for after_a in board.single_die_moves(a) {
        // a, puis b
        for after_ab in after_a.single_die_moves(b) {
            out.push(after_ab);
        }
    }
    for after_b in board.single_die_moves(b) {
        // b, puis a
        for after_ba in after_b.single_die_moves(a) {
            out.push(after_ba);
        }
    }
    out
}

/// Explore récursivement les coups d'un double. Renvoie chaque position
/// atteinte avec le nombre de dés réellement joués pour y arriver.
fn play_doubles(board: &Board, die: u8, remaining: usize) -> Vec<(Board, usize)> {
    if remaining == 0 {
        return vec![(board.clone(), 0)]; // plus de dé à jouer
    }
    let moves = board.single_die_moves(die);
    if moves.is_empty() {
        return vec![(board.clone(), 0)]; // bloqué : 0 dé de plus depuis ici
    }
    let mut out = Vec::new();
    for next in moves {
        for (b, used) in play_doubles(&next, die, remaining - 1) {
            out.push((b, used + 1)); // +1 pour le dé qu'on vient de jouer
        }
    }
    out
}

/// Supprime les positions en double, en conservant l'ordre d'apparition.
fn dedup(boards: Vec<Board>) -> Vec<Board> {
    let mut out: Vec<Board> = Vec::new();
    for b in boards {
        if !out.contains(&b) {
            out.push(b);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::legal_plays;
    use crate::board::Board;
    use crate::dice::Roll;

    #[test]
    fn le_départ_a_des_coups() {
        let start = Board::starting_position();
        let plays = legal_plays(&start, &Roll { d1: 6, d2: 5 });
        assert!(!plays.is_empty());
    }
}
