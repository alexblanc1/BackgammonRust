//! Les dés : un lancer (`Roll`) et un lanceur (`Dice`).

use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

/// Un lancer de deux dés.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Roll {
    pub d1: u8,
    pub d2: u8,
}

impl Roll {
    /// Les valeurs de dés à jouer : quatre fois la même sur un double,
    /// les deux valeurs sinon.
    pub fn dice(&self) -> Vec<u8> {
        if self.d1 == self.d2 {
            vec![self.d1; 4]
        } else {
            vec![self.d1, self.d2]
        }
    }
}

/// Les 21 lancers distincts avec leur probabilité : 1/36 pour un double,
/// 2/36 pour les autres (les deux ordres du même lancer sont confondus).
/// C'est la distribution des nœuds de chance de l'expectiminimax.
pub fn all_rolls() -> Vec<(Roll, f64)> {
    let mut out = Vec::with_capacity(21);
    for d1 in 1..=6u8 {
        for d2 in d1..=6u8 {
            let p = if d1 == d2 { 1.0 / 36.0 } else { 2.0 / 36.0 };
            out.push((Roll { d1, d2 }, p));
        }
    }
    out
}

/// Lanceur de dés, et plus généralement la source d'aléa du projet.
///
/// Encapsule le générateur `StdRng` de la crate `rand` (un PRNG de qualité
/// cryptographique, bien plus robuste que l'ancien xorshift maison). Deux
/// constructeurs :
/// - [`Dice::new`] avec une graine → suite **reproductible** (tests,
///   entraînement comparable d'une exécution à l'autre) ;
/// - [`Dice::random`] → graine tirée auprès du système d'exploitation,
///   imprévisible (les vraies parties).
pub struct Dice {
    rng: StdRng,
}

impl Dice {
    /// Générateur déterministe : la même graine redonne la même suite.
    pub fn new(seed: u64) -> Dice {
        Dice {
            rng: StdRng::seed_from_u64(seed),
        }
    }

    /// Générateur vraiment imprévisible : graine tirée du générateur du
    /// système (via le RNG global de la crate `rand`).
    pub fn random() -> Dice {
        Dice {
            rng: StdRng::from_rng(&mut rand::rng()),
        }
    }

    /// Un entier aléatoire dans `0..bound` (suppose `bound > 0`).
    ///
    /// `random_range` est **uniforme** : contrairement au `% bound` de l'ancien
    /// xorshift, aucune valeur n'est légèrement favorisée (biais du modulo).
    pub fn index(&mut self, bound: usize) -> usize {
        self.rng.random_range(0..bound)
    }

    /// Un flottant aléatoire uniforme dans `[0, 1)`.
    pub fn unit(&mut self) -> f64 {
        self.rng.random()
    }

    /// Lance les deux dés.
    pub fn roll(&mut self) -> Roll {
        Roll {
            d1: self.rng.random_range(1..=6),
            d2: self.rng.random_range(1..=6),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn meme_graine_meme_suite() {
        let mut a = Dice::new(42);
        let mut b = Dice::new(42);
        for _ in 0..100 {
            assert_eq!(a.roll(), b.roll());
        }
    }

    #[test]
    fn valeurs_dans_les_bornes() {
        let mut d = Dice::new(7);
        for _ in 0..1000 {
            let r = d.roll();
            assert!((1..=6).contains(&r.d1) && (1..=6).contains(&r.d2));
            let i = d.index(13);
            assert!(i < 13);
            let u = d.unit();
            assert!((0.0..1.0).contains(&u));
        }
    }

    #[test]
    fn vingt_et_un_lancers_de_proba_totale_un() {
        let rolls = all_rolls();
        assert_eq!(rolls.len(), 21);
        let total: f64 = rolls.iter().map(|(_, p)| p).sum();
        assert!((total - 1.0).abs() < 1e-12);
    }
}
