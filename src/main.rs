//! CLI du jeu de backgammon.

use backgammon::agent::random::RandomAgent;
use backgammon::agent::Agent;
use backgammon::dice::Dice;
use backgammon::game::play;

fn main() {
    println!("Squelette du moteur de backgammon.");
    println!(
        "Tant que les `todo!()` ne sont pas remplis, l'exécution panique sur \
         le premier rencontré (commence par board.rs, puis moves.rs)."
    );

    // Deux agents aléatoires s'affrontent : pratique pour tester le moteur de
    // bout en bout. Pour jouer toi-même, remplace l'un des deux par
    // `Box::new(HumanAgent::new())` (voir src/agent/human.rs).
    let mut agents: [Box<dyn Agent>; 2] = [
        Box::new(RandomAgent::new(1)),
        Box::new(RandomAgent::new(2)),
    ];
    let mut dice = Dice::new(12_345);

    let (winner, points) = play(&mut agents, &mut dice);
    println!("{winner:?} l'emporte ({points} point(s)).");
}
