//! Les dés : un lancer (`Roll`) et un lanceur (`Dice`).

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

/// Lanceur de dés.
///
/// Utilise un petit générateur pseudo-aléatoire (xorshift) afin de rester
/// sans dépendance externe. Tu pourras le remplacer par la crate `rand` plus
/// tard si tu veux un aléatoire de meilleure qualité.
pub struct Dice {
    state: u64,
}

impl Dice {
    pub fn new(seed: u64) -> Dice {
        // `| 1` pour garantir un état non nul (le xorshift reste bloqué à 0).
        Dice { state: seed | 1 }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    /// Un entier pseudo-aléatoire dans `0..bound` (suppose `bound > 0`).
    pub fn index(&mut self, bound: usize) -> usize {
        (self.next_u64() % bound as u64) as usize
    }

    /// Un flottant pseudo-aléatoire uniforme dans `[0, 1)`.
    /// (On garde les 53 bits de poids fort pour remplir la mantisse d'un `f64`.)
    pub fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Lance les deux dés.
    pub fn roll(&mut self) -> Roll {
        Roll {
            d1: (self.next_u64() % 6) as u8 + 1,
            d2: (self.next_u64() % 6) as u8 + 1,
        }
    }
}
