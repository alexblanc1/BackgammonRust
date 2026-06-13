//! Le tableau des scores : victoires/défaites et points marqués, conservés
//! d'une session à l'autre dans un petit fichier texte.
//!
//! Module du *binaire* (comme la TUI) : la persistance d'un score d'interface
//! n'a pas sa place dans le moteur.

use std::path::Path;

/// Fichier de sauvegarde, à côté de `net.txt` (git-ignoré lui aussi).
pub const SCORES_FILE: &str = "scores.txt";

/// Le bilan du joueur humain contre l'IA, toutes sessions confondues.
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub struct Scores {
    pub wins: u32,
    pub losses: u32,
    /// Points marqués par le joueur (gammons, backgammons et videau compris).
    pub points_for: u32,
    /// Points marqués par l'IA.
    pub points_against: u32,
}

impl Scores {
    /// Charge le tableau depuis `path`, ou repart de zéro si le fichier
    /// n'existe pas ou est illisible (un score ne vaut pas une erreur fatale).
    pub fn load(path: impl AsRef<Path>) -> Scores {
        let Ok(text) = std::fs::read_to_string(path) else {
            return Scores::default();
        };
        let mut nums = text.split_whitespace().filter_map(|s| s.parse().ok());
        // `let ... else` : si une des quatre valeurs manque, on repart de zéro.
        let (Some(wins), Some(losses), Some(points_for), Some(points_against)) =
            (nums.next(), nums.next(), nums.next(), nums.next())
        else {
            return Scores::default();
        };
        Scores {
            wins,
            losses,
            points_for,
            points_against,
        }
    }

    pub fn save(&self, path: impl AsRef<Path>) -> std::io::Result<()> {
        std::fs::write(
            path,
            format!(
                "{} {} {} {}\n",
                self.wins, self.losses, self.points_for, self.points_against
            ),
        )
    }

    /// Enregistre une victoire du joueur (`points` = points de la partie,
    /// videau compris).
    pub fn record_win(&mut self, points: u32) {
        self.wins += 1;
        self.points_for += points;
    }

    /// Enregistre une défaite du joueur.
    pub fn record_loss(&mut self, points: u32) {
        self.losses += 1;
        self.points_against += points;
    }

    /// Résumé court pour l'affichage : « 3 V · 2 D (+7/−5 pts) ».
    pub fn summary(&self) -> String {
        format!(
            "{} V · {} D  (+{}/−{} pts)",
            self.wins, self.losses, self.points_for, self.points_against
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aller_retour_sur_fichier() {
        let path = std::env::temp_dir().join("backgammon_test_scores.txt");
        let mut s = Scores::default();
        s.record_win(2);
        s.record_win(1);
        s.record_loss(4);
        s.save(&path).unwrap();
        assert_eq!(Scores::load(&path), s);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn fichier_absent_ou_corrompu_donne_zero() {
        assert_eq!(
            Scores::load("/chemin/qui/n/existe/pas.txt"),
            Scores::default()
        );
        let path = std::env::temp_dir().join("backgammon_test_scores_bad.txt");
        std::fs::write(&path, "n'importe quoi").unwrap();
        assert_eq!(Scores::load(&path), Scores::default());
        let _ = std::fs::remove_file(&path);
    }
}
