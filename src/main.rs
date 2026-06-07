//! CLI du jeu de backgammon.

use std::time::{SystemTime, UNIX_EPOCH};

use backgammon::agent::Agent;
use backgammon::agent::heuristic::heuristic_agent;
use backgammon::agent::human::HumanAgent;
use backgammon::dice::Dice;
use backgammon::game::play;

fn main() {
    println!("Backgammon — tu joues les Blancs (X), l'ordinateur les Noirs (O).");

    // Graine variable : une partie différente à chaque lancement.
    let seed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0x1234_5678);

    // agents[0] = Blancs (toi), agents[1] = Noirs (l'IA heuristique).
    let mut agents: [Box<dyn Agent>; 2] = [
        Box::new(HumanAgent::new()),
        Box::new(heuristic_agent()),
    ];
    let mut dice = Dice::new(seed | 1);

    let (winner, points) = play(&mut agents, &mut dice);
    println!("\n{winner:?} l'emporte ({points} point(s)).");
}
