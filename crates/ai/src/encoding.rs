//! Encodage d'une position en vecteur d'entrées pour le réseau de neurones.
//!
//! Suit l'encodage de Tesauro (TD-Gammon) : pour chaque case et chaque joueur,
//! 4 unités décrivent le nombre de pions ; s'ajoutent la barre et les pions
//! sortis. Le plateau étant toujours vu du joueur à jouer, on encode « moi »
//! (pions positifs) puis « l'adversaire » (pions négatifs).

use engine::board::Board;

/// Nombre d'entrées du réseau.
///
/// 24 cases × 4 unités × 2 joueurs = 192, plus la barre (2) et les pions sortis
/// (2). L'encodage original de Tesauro en a 198 : les 2 unités « à qui le tour »
/// sont inutiles ici, le plateau étant toujours normalisé du point de vue du
/// joueur à jouer.
pub const N_INPUTS: usize = 196;

/// Encode les `n` pions d'un joueur sur une case, dans `out` (4 unités) :
/// présence d'au moins 1, 2, 3 pions, puis le surplus au-delà de 3.
fn encode_point(n: u8, out: &mut [f64]) {
    out[0] = if n >= 1 { 1.0 } else { 0.0 };
    out[1] = if n >= 2 { 1.0 } else { 0.0 };
    out[2] = if n >= 3 { 1.0 } else { 0.0 };
    out[3] = if n > 3 { (n - 3) as f64 / 2.0 } else { 0.0 };
}

/// Transforme une position (vue du joueur à jouer) en vecteur d'entrées.
pub fn encode(board: &Board) -> [f64; N_INPUTS] {
    let mut x = [0.0f64; N_INPUTS];
    let mut i = 0;

    for p in 0..24 {
        let v = board.points()[p];
        let me = if v > 0 { v as u8 } else { 0 };
        let opp = if v < 0 { (-v) as u8 } else { 0 };

        encode_point(me, &mut x[i..i + 4]);
        i += 4;
        encode_point(opp, &mut x[i..i + 4]);
        i += 4;
    }

    // Barre : normalisée par 2 (moi, puis l'adversaire).
    x[i] = board.bar()[0] as f64 / 2.0;
    x[i + 1] = board.bar()[1] as f64 / 2.0;
    i += 2;

    // Pions sortis : normalisés par 15 (sur 15 pions au total).
    x[i] = board.off()[0] as f64 / 15.0;
    x[i + 1] = board.off()[1] as f64 / 15.0;
    i += 2;

    debug_assert_eq!(i, N_INPUTS);
    x
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn taille_et_position_de_depart() {
        let x = encode(&Board::starting_position());
        assert_eq!(x.len(), N_INPUTS);

        // Case 5 : +5 pions à moi → [≥1, ≥2, ≥3, (5-3)/2] = [1,1,1,1] ; adverse vide.
        let base = 5 * 8;
        assert_eq!(x[base..base + 4], [1.0, 1.0, 1.0, 1.0]);
        assert_eq!(x[base + 4..base + 8], [0.0, 0.0, 0.0, 0.0]);

        // Case 0 : -2 pions adverses → moi vide, adverse [1,1,0,0].
        assert_eq!(x[0..4], [0.0, 0.0, 0.0, 0.0]);
        assert_eq!(x[4..8], [1.0, 1.0, 0.0, 0.0]);

        // Barre et sorties vides au départ.
        assert_eq!(x[192..196], [0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn barre_et_sorties() {
        let mut points = [0i8; 24];
        points[0] = 3; // 3 pions à moi sur la case 0
        let b = Board::from_parts(points, [2, 1], [4, 15]);
        let x = encode(&b);

        // Case 0 : +3 → [1,1,1,0].
        assert_eq!(x[0..4], [1.0, 1.0, 1.0, 0.0]);
        // Barre : 2/2 et 1/2.
        assert_eq!(x[192], 1.0);
        assert_eq!(x[193], 0.5);
        // Sorties : 4/15 et 15/15.
        assert_eq!(x[194], 4.0 / 15.0);
        assert_eq!(x[195], 1.0);
    }

    #[test]
    fn surplus_au_dela_de_trois() {
        let mut points = [0i8; 24];
        points[10] = 6; // 6 pions → surplus (6-3)/2 = 1.5
        let b = Board::from_parts(points, [0, 0], [0, 0]);
        let x = encode(&b);
        let base = 10 * 8;
        assert_eq!(x[base..base + 4], [1.0, 1.0, 1.0, 1.5]);
    }
}
