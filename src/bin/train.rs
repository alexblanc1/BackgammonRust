//! Outil d'entraînement du réseau par self-play TD(λ).
//!
//! Usage : `cargo run --release --bin train -- [parties] [hidden]`
//! (compile en --release : l'entraînement est gourmand en calcul).

use std::time::Instant;

use backgammon::net::Net;
use backgammon::train::{train_self_play, win_rate_vs_heuristic, win_rate_vs_random};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let total: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(20_000);
    let hidden: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(40);
    let alpha: f64 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(0.1);
    // λ=0 (TD(0)) est le réglage stable par défaut. λ>0 apprend plus vite mais
    // amplifie le pas effectif : baisser alpha en conséquence.
    let lambda: f64 = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(0.0);
    // Palier auto : ~8 lignes de progression quel que soit le nombre de parties.
    let chunk = (total / 8).clamp(1, 2_000);

    println!(
        "Entraînement TD(λ) : {total} parties, hidden={hidden}, alpha={alpha}, lambda={lambda}\n"
    );

    let mut net = Net::new_random(hidden, 12_345);
    println!(
        "  {:>7} | vs hasard | vs heuristique",
        "parties"
    );
    println!("  {:->7}-+-----------+---------------", "");

    let start = Instant::now();
    let mut done = 0usize;
    let mut seed = 1u64;
    while done < total {
        let n = chunk.min(total - done);
        train_self_play(&mut net, n, alpha, lambda, seed);
        done += n;
        seed = seed.wrapping_add(1);

        let vr = win_rate_vs_random(&net, 200, 7_777);
        let vh = win_rate_vs_heuristic(&net, 200, 9_999);
        println!(
            "  {done:>7} |   {:>5.1}% |     {:>5.1}%",
            vr * 100.0,
            vh * 100.0
        );
    }

    println!("\nTerminé en {:.1?}.", start.elapsed());

    let path = "net.txt";
    match net.save(path) {
        Ok(()) => println!("Poids sauvegardés dans « {path} ». La TUI le chargera automatiquement."),
        Err(e) => eprintln!("Échec de la sauvegarde dans « {path} » : {e}"),
    }
}
