//! Le plateau et les opérations qui s'appliquent dessus.

/// Le plateau, toujours vu du point de vue du joueur dont c'est le tour.
///
/// `points[i] > 0` : pions du joueur à jouer ; `points[i] < 0` : pions de
/// l'adversaire. Le joueur à jouer avance de l'index 23 vers l'index 0 ; son
/// jan intérieur (où il sort ses pions) est sur les index 0..6.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Board {
    // `pub(crate)` : visibles par les autres modules du moteur (ex. `moves.rs`)
    // mais pas depuis l'extérieur de la bibliothèque.
    pub(crate) points: [i8; 24],
    pub(crate) bar: [u8; 2],
    pub(crate) off: [u8; 2],
}

impl Board {
    /// La position de départ standard du backgammon.
    pub fn starting_position() -> Board {
        Board {
            points: [
                -2, 0, 0, 0, 0, 5, // index 0..5   (-2 = pions reculés adverses, +5 = ta pile de base)
                0, 3, 0, 0, 0, -5, // index 6..11  (+3 sur l'index 7, -5 sur l'index 11)
                5, 0, 0, 0, -3, 0, // index 12..17 (+5 = ton midpoint, -3 sur l'index 16)
                -5, 0, 0, 0, 0, 2, // index 18..23 (-5 sur l'index 18, +2 = tes pions reculés)
            ],
            bar: [0, 0],
            off: [0, 0],
        }
    }

    /// Renvoie le plateau vu du point de vue de l'autre joueur.
    pub fn swap_perspective(&self) -> Board {
        let mut points = [0i8; 24];
        for i in 0..24 {
            points[i] = -self.points[23 - i];
        }
        Board {
            points,
            bar: [self.bar[1], self.bar[0]],
            off: [self.off[1], self.off[0]],
        }
    }

    /// Tous tes pions sont-ils dans ton jan intérieur (index 0..6) ? C'est la
    /// condition pour avoir le droit de sortir des pions.
    fn can_bear_off(&self) -> bool {
        if self.bar[0] > 0 {
            return false; // un pion sur la barre interdit la sortie
        }
        for p in 6..24 {
            if self.points[p] > 0 {
                return false; // un pion à toi hors du jan
            }
        }
        true
    }

    /// Toutes les positions atteignables en jouant *un seul* dé de valeur
    /// `die`. C'est la primitive utilisée par `legal_plays` (dans `moves.rs`).
    ///
    /// Doit gérer : entrée depuis la barre (prioritaire), déplacement simple,
    /// frappe d'un pion adverse seul, et sortie quand c'est légal.
    pub fn single_die_moves(&self, die: u8) -> Vec<Board> {
        let d = die as i32;
        let mut result = Vec::new();

        // Règle 1 — la barre est prioritaire : tant qu'on a un pion dessus,
        // le seul coup permis est de le faire rentrer.
        if self.bar[0] > 0 {
            let entry = 24 - d; // case d'entrée pour ce dé (un index entre 18 et 23)
            let e = entry as usize;
            if self.points[e] >= -1 {
                // entrée possible : case libre, à toi, ou un seul pion adverse
                let mut b = self.clone();
                b.bar[0] -= 1;
                if b.points[e] == -1 {
                    b.points[e] = 1; // on frappe le pion adverse...
                    b.bar[1] += 1; // ...qui part sur la barre
                } else {
                    b.points[e] += 1;
                }
                result.push(b);
            }
            return result; // rien d'autre tant qu'un pion est sur la barre
        }

        // Règle 2 — sinon, on tente de déplacer un pion depuis chaque case occupée.
        let home = self.can_bear_off();
        for p in 0..24usize {
            if self.points[p] <= 0 {
                continue; // aucun pion à toi sur cette case
            }
            let target = p as i32 - d;

            if target >= 0 {
                // Déplacement classique (y compris à l'intérieur du jan).
                let t = target as usize;
                if self.points[t] >= -1 {
                    let mut b = self.clone();
                    b.points[p] -= 1;
                    if b.points[t] == -1 {
                        b.points[t] = 1; // frappe
                        b.bar[1] += 1;
                    } else {
                        b.points[t] += 1;
                    }
                    result.push(b);
                }
            } else if home {
                // Le coup sortirait du plateau → sortie d'un pion (bearing off).
                if target == -1 {
                    // Sortie exacte : le dé vaut pile ce qu'il faut.
                    let mut b = self.clone();
                    b.points[p] -= 1;
                    b.off[0] += 1;
                    result.push(b);
                } else {
                    // Dé trop grand : sortie autorisée seulement si aucun de tes
                    // pions n'est sur une case plus haute que p.
                    let mut higher = false;
                    for q in (p + 1)..6 {
                        if self.points[q] > 0 {
                            higher = true;
                            break;
                        }
                    }
                    if !higher {
                        let mut b = self.clone();
                        b.points[p] -= 1;
                        b.off[0] += 1;
                        result.push(b);
                    }
                }
            }
        }

        result
    }

    /// Renvoie `Some(points)` si le joueur à jouer vient de gagner
    /// (1 = simple, 2 = gammon, 3 = backgammon), sinon `None`.
    pub fn win_check(&self) -> Option<u8> {
        // Le joueur à jouer gagne quand il a sorti ses 15 pions.
        if self.off[0] < 15 {
            return None;
        }
        // L'adversaire a-t-il sorti au moins un pion ? Si oui → partie simple.
        if self.off[1] > 0 {
            return Some(1);
        }
        // L'adversaire n'a rien sorti : au moins un gammon. C'est un backgammon
        // s'il a encore un pion sur la barre ou dans ton jan intérieur (0..6).
        if self.bar[1] > 0 {
            return Some(3);
        }
        for p in 0..6 {
            if self.points[p] < 0 {
                return Some(3);
            }
        }
        Some(2) // gammon
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quinze_pions_chacun() {
        let b = Board::starting_position();
        let mut a_moi = 0i32;
        let mut adverse = 0i32;
        for &p in &b.points {
            if p > 0 {
                a_moi += p as i32;
            } else if p < 0 {
                adverse += -(p as i32);
            }
        }
        assert_eq!(a_moi, 15);
        assert_eq!(adverse, 15);
    }

    #[test]
    fn position_de_depart_symetrique() {
        let b = Board::starting_position();
        assert_eq!(b, b.swap_perspective());
    }

    #[test]
    fn un_dé_depuis_le_départ() {
        // Avec un 1, seuls 3 de tes pions peuvent bouger : celui de l'index 12
        // est bloqué par les 5 pions adverses en index 11.
        let b = Board::starting_position();
        assert_eq!(b.single_die_moves(1).len(), 3);
    }

    #[test]
    fn victoire_simple() {
        // Tu as sorti tes 15 pions, l'adversaire en a déjà sorti → 1 point.
        let b = Board {
            points: [0; 24],
            bar: [0, 0],
            off: [15, 3],
        };
        assert_eq!(b.win_check(), Some(1));
    }

    #[test]
    fn victoire_gammon() {
        // Adversaire n'a rien sorti, et aucun de ses pions dans ton jan/barre.
        let mut points = [0i8; 24];
        points[12] = -15;
        let b = Board {
            points,
            bar: [0, 0],
            off: [15, 0],
        };
        assert_eq!(b.win_check(), Some(2));
    }

    #[test]
    fn victoire_backgammon_dans_le_jan() {
        // Adversaire n'a rien sorti ET a un pion dans ton jan intérieur → 3 points.
        let mut points = [0i8; 24];
        points[3] = -1;
        points[12] = -14;
        let b = Board {
            points,
            bar: [0, 0],
            off: [15, 0],
        };
        assert_eq!(b.win_check(), Some(3));
    }

    #[test]
    fn victoire_backgammon_sur_la_barre() {
        // Un pion adverse sur la barre compte aussi comme backgammon.
        let mut points = [0i8; 24];
        points[12] = -14;
        let b = Board {
            points,
            bar: [0, 1],
            off: [15, 0],
        };
        assert_eq!(b.win_check(), Some(3));
    }

    #[test]
    fn pas_encore_gagné() {
        let b = Board::starting_position();
        assert_eq!(b.win_check(), None);
    }
}
