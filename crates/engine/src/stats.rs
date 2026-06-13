//! Vérification statistique des dés : les six faces sortent-elles bien
//! uniformément ?
//!
//! L'outil collecte un grand nombre de lancers puis applique le **test du χ²
//! d'adéquation** : on compare les effectifs observés de chaque face aux
//! effectifs attendus (1/6 chacun) par la statistique
//! `χ² = Σ (observé − attendu)² / attendu`. Si les dés sont équilibrés, cette
//! statistique suit une loi du χ² à 5 degrés de liberté (6 faces − 1) : elle
//! dépasse 11,07 dans seulement 5 % des cas, et 15,09 dans 1 % des cas. Une
//! valeur largement au-dessus trahirait un générateur biaisé.

use crate::dice::Dice;

/// Seuil de rejet à 5 % pour un χ² à 5 degrés de liberté.
pub const CHI2_THRESHOLD_5PCT: f64 = 11.07;
/// Seuil de rejet à 1 % pour un χ² à 5 degrés de liberté.
pub const CHI2_THRESHOLD_1PCT: f64 = 15.09;

/// Les effectifs observés sur une série de lancers de deux dés.
#[derive(Clone, Debug)]
pub struct DiceStats {
    /// Combien de fois chaque face (1..=6) est sortie, tous dés confondus.
    pub counts: [u64; 6],
    /// Nombre de lancers (chaque lancer compte deux dés).
    pub rolls: u64,
    /// Nombre de doubles observés (attendu : 1/6 des lancers).
    pub doubles: u64,
}

impl DiceStats {
    /// Lance `rolls` fois les deux dés et compte les faces.
    pub fn collect(dice: &mut Dice, rolls: u64) -> DiceStats {
        let mut counts = [0u64; 6];
        let mut doubles = 0u64;
        for _ in 0..rolls {
            let r = dice.roll();
            counts[(r.d1 - 1) as usize] += 1;
            counts[(r.d2 - 1) as usize] += 1;
            if r.d1 == r.d2 {
                doubles += 1;
            }
        }
        DiceStats {
            counts,
            rolls,
            doubles,
        }
    }

    /// Nombre total de dés observés (deux par lancer).
    pub fn faces(&self) -> u64 {
        2 * self.rolls
    }

    /// La statistique du χ² des six faces contre la loi uniforme.
    pub fn chi2(&self) -> f64 {
        let expected = self.faces() as f64 / 6.0;
        if expected == 0.0 {
            return 0.0;
        }
        self.counts
            .iter()
            .map(|&obs| {
                let d = obs as f64 - expected;
                d * d / expected
            })
            .sum()
    }

    /// Le test passe-t-il au seuil de 5 % ?
    pub fn uniform_at_5pct(&self) -> bool {
        self.chi2() < CHI2_THRESHOLD_5PCT
    }

    /// Fréquence observée d'une face (0.0 ≤ f ≤ 1.0 ; attendu : 1/6 ≈ 0,1667).
    pub fn frequency(&self, face: u8) -> f64 {
        debug_assert!((1..=6).contains(&face));
        if self.rolls == 0 {
            return 0.0;
        }
        self.counts[(face - 1) as usize] as f64 / self.faces() as f64
    }

    /// Fréquence observée des doubles (attendu : 1/6).
    pub fn doubles_frequency(&self) -> f64 {
        if self.rolls == 0 {
            return 0.0;
        }
        self.doubles as f64 / self.rolls as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Les dés `rand` doivent passer le test du χ² sur un grand échantillon.
    /// (Graine fixe : le résultat est déterministe, pas de test fragile.)
    #[test]
    fn les_des_sont_uniformes() {
        let mut dice = Dice::new(20_260_613);
        let stats = DiceStats::collect(&mut dice, 50_000);
        let chi2 = stats.chi2();
        assert!(
            chi2 < CHI2_THRESHOLD_5PCT,
            "χ² = {chi2:.2} ≥ {CHI2_THRESHOLD_5PCT} : distribution suspecte ({:?})",
            stats.counts
        );
        // Les doubles aussi doivent tourner autour de 1/6.
        let f = stats.doubles_frequency();
        assert!((f - 1.0 / 6.0).abs() < 0.01, "fréquence des doubles : {f:.4}");
    }

    /// Contre-exemple : un « dé » truqué doit être rejeté par le test.
    #[test]
    fn un_de_truque_est_rejete() {
        // 12 000 dés dont la face 6 sort une fois et demie trop souvent.
        let stats = DiceStats {
            counts: [1800, 1800, 1800, 1800, 1800, 3000],
            rolls: 6_000,
            doubles: 1_000,
        };
        assert!(
            stats.chi2() > CHI2_THRESHOLD_1PCT,
            "le biais aurait dû être détecté (χ² = {:.2})",
            stats.chi2()
        );
    }

    #[test]
    fn comptages_coherents() {
        let mut dice = Dice::new(7);
        let stats = DiceStats::collect(&mut dice, 1_000);
        assert_eq!(stats.counts.iter().sum::<u64>(), 2_000);
        assert_eq!(stats.faces(), 2_000);
        let total: f64 = (1..=6).map(|f| stats.frequency(f)).sum();
        assert!((total - 1.0).abs() < 1e-12);
    }
}
