//! Point d'entrée : lance l'interface graphique (TUI) du backgammon.
//!
//! Tu joues les Blancs (X) contre l'IA heuristique (O). Lance avec `cargo run`
//! dans un vrai terminal.

mod tui;

fn main() -> std::io::Result<()> {
    tui::run()
}
