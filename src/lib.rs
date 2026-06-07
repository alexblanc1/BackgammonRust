//! Moteur de backgammon — squelette généré.
//!
//! Les structures de données et les opérations simples sont déjà écrites.
//! Les parties algorithmiques difficiles sont marquées `todo!()` : tu les
//! implémentes une par une (voir le README pour l'ordre conseillé).

// Temporaire : évite les avertissements « code/variable jamais utilisé »
// tant que des corps de fonctions sont des `todo!()`. À retirer plus tard.
#![allow(dead_code, unused_variables)]

pub mod agent;
pub mod board;
pub mod dice;
pub mod encoding;
pub mod eval;
pub mod game;
pub mod moves;
pub mod net;
pub mod player;
pub mod train;

// Ré-exports : permet d'écrire `backgammon::Board` plutôt que
// `backgammon::board::Board`.
pub use agent::Agent;
pub use board::Board;
pub use dice::{Dice, Roll};
pub use encoding::{encode, N_INPUTS};
pub use eval::{Evaluator, GreedyAgent};
pub use net::Net;
pub use game::{GameState, Phase};
pub use player::Player;
