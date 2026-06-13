//! Le moteur de backgammon : plateau, dés, génération des coups légaux,
//! déroulement d'une partie (videau compris) et statistiques des dés.
//!
//! Tout est écrit du point de vue du **joueur à jouer** : `points[i] > 0` =
//! ses pions, il avance de l'index 23 vers 0, et `swap_perspective()` retourne
//! le plateau à chaque changement de tour. L'IA vit dans la crate `ai`, et
//! l'affichage dans `cli` : le moteur n'a aucune dépendance graphique.

pub mod agent;
pub mod board;
pub mod dice;
pub mod game;
pub mod moves;
pub mod player;
pub mod stats;

// Ré-exports : permet d'écrire `engine::Board` plutôt que
// `engine::board::Board`.
pub use agent::Agent;
pub use board::Board;
pub use dice::{Dice, Roll, all_rolls};
pub use game::{Cube, GameState, Phase};
pub use player::Player;
