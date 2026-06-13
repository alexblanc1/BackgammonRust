//! Point d'entrée : lance l'interface graphique (TUI) du backgammon.
//!
//! Tu joues les Blancs contre l'IA (niveau au choix dans le menu). Lance avec
//! `cargo run` dans un vrai terminal — `--release` conseillé pour les niveaux
//! Expert et au-delà.

mod scores;
mod tui;

fn main() -> std::io::Result<()> {
    tui::run()
}
