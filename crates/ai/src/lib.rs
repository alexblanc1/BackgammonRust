//! L'intelligence artificielle du backgammon, façon TD-Gammon :
//!
//! - [`encoding`] : la position devient un vecteur d'entrées (style Tesauro) ;
//! - [`net`] : le réseau de neurones (6 sorties : gain/perte ×
//!   simple/gammon/backgammon) ;
//! - [`eval`] : le trait [`Evaluator`] et l'agent glouton, partagés par
//!   l'heuristique, le réseau et la recherche ;
//! - [`heuristic`] : la baseline sans apprentissage ;
//! - [`search`] : expectiminimax (nœuds de chance) et rollouts Monte-Carlo ;
//! - [`train`] : le self-play TD(λ).

pub mod encoding;
pub mod eval;
pub mod heuristic;
pub mod net;
pub mod search;
pub mod train;

pub use encoding::{N_INPUTS, encode};
pub use eval::{Evaluator, GreedyAgent};
pub use net::{N_OUTPUTS, Net};
pub use search::{ExpectiAgent, RolloutAgent};
