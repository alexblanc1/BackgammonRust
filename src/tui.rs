//! Interface graphique du jeu dans le terminal, avec Ratatui.
//!
//! Module du *binaire* (pas de la bibliothèque) : le moteur reste ainsi sans
//! dépendance d'affichage. Le plateau est toujours montré du point de vue des
//! Blancs (le joueur humain) ; on retourne la position pour l'affichage quand
//! c'est l'IA qui vient de jouer.

use std::io;
use std::time::{SystemTime, UNIX_EPOCH};

use ratatui::DefaultTerminal;
use ratatui::Frame;
use ratatui::crossterm::event::{self, Event, KeyCode, KeyEventKind};
use ratatui::layout::{Constraint, Layout};
use ratatui::style::{Color, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Paragraph};

use backgammon::agent::heuristic::evaluate;
use backgammon::board::Board;
use backgammon::dice::{Dice, Roll};
use backgammon::moves::legal_plays;
use backgammon::net::Net;
use backgammon::player::Player;

/// Hauteur d'affichage d'une pile de pions.
const H: usize = 5;
const HUMAN: Color = Color::Cyan; // tes pions (X)
const AI: Color = Color::Red; // pions de l'IA (O)
const DIM: Color = Color::DarkGray;

fn dim() -> Style {
    Style::default().fg(DIM)
}

// --- Point d'entrée ----------------------------------------------------------

/// Prépare le terminal, joue la partie, puis restaure le terminal.
pub fn run() -> io::Result<()> {
    // Si un réseau entraîné a été sauvegardé (`cargo run --release --bin train`),
    // on joue contre lui ; sinon, contre l'heuristique.
    let ai = match Net::load("net.txt") {
        Ok(net) => Ai::Net(net),
        Err(_) => Ai::Heuristic,
    };

    let mut terminal = ratatui::init();
    let result = run_game(&mut terminal, &ai);
    ratatui::restore();
    result
}

fn run_game(terminal: &mut DefaultTerminal, ai: &Ai) -> io::Result<()> {
    let seed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0x1234_5678);
    let mut dice = Dice::new(seed | 1);

    let mut board = Board::starting_position();
    let mut to_move = Player::White; // les Blancs (toi) commencent

    loop {
        let roll = dice.roll();
        let plays = legal_plays(&board, &roll);
        let view = white_view(&board, to_move);

        if plays.is_empty() {
            let msg = vec![
                Line::from(format!("{} ne peut pas jouer ce lancer.", who(to_move))),
                Line::from("Tour passé."),
            ];
            if !wait_or_quit(terminal, &view, info_lines(to_move, roll, ai.label()), "Pas de coup", msg)? {
                return Ok(());
            }
        } else if to_move == Player::White {
            match human_select(terminal, &board, &plays, roll, ai.label())? {
                Some(i) => board = plays[i].clone(),
                None => return Ok(()), // l'utilisateur a quitté
            }
        } else {
            let i = ai.choose(&plays);
            board = plays[i].clone();
            let view_after = white_view(&board, to_move);
            let msg = vec![
                Line::from("L'IA a joué son tour."),
                Line::from("Observe le plateau."),
            ];
            if !wait_or_quit(terminal, &view_after, info_lines(to_move, roll, ai.label()), "IA", msg)? {
                return Ok(());
            }
        }

        if let Some(points) = board.win_check() {
            let view_final = white_view(&board, to_move);
            let label = match to_move {
                Player::White => format!("Tu gagnes ! ({points} point(s)) 🎉"),
                Player::Black => format!("L'IA gagne ({points} point(s))."),
            };
            wait_or_quit(
                terminal,
                &view_final,
                info_lines(to_move, roll, ai.label()),
                "Fin de la partie",
                vec![Line::from(label), Line::from("Une touche pour quitter.")],
            )?;
            return Ok(());
        }

        board = board.swap_perspective();
        to_move = to_move.other();
    }
}

// --- Boucle de saisie du joueur ---------------------------------------------

/// Laisse l'humain parcourir les coups possibles et en choisir un. Renvoie
/// `None` s'il quitte.
fn human_select(
    terminal: &mut DefaultTerminal,
    board: &Board,
    plays: &[Board],
    roll: Roll,
    opp: &str,
) -> io::Result<Option<usize>> {
    let mut sel = 0usize;
    loop {
        let panel: Vec<Line<'static>> = plays
            .iter()
            .enumerate()
            .map(|(i, cand)| {
                let text = format!(" {:>2}. {}", i + 1, describe(board, cand));
                if i == sel {
                    Line::from(Span::styled(
                        text,
                        Style::default().fg(Color::Black).bg(HUMAN),
                    ))
                } else {
                    Line::from(text)
                }
            })
            .collect();

        draw(
            terminal,
            board,
            info_lines(Player::White, roll, opp),
            "Coups possibles",
            panel,
            "↑/↓ choisir   Entrée valider   q quitter",
        )?;

        if let Event::Key(k) = event::read()? {
            if k.kind != KeyEventKind::Press {
                continue;
            }
            match k.code {
                KeyCode::Up | KeyCode::Char('k') => {
                    sel = if sel == 0 { plays.len() - 1 } else { sel - 1 };
                }
                KeyCode::Down | KeyCode::Char('j') => {
                    sel = (sel + 1) % plays.len();
                }
                KeyCode::Enter | KeyCode::Char(' ') => return Ok(Some(sel)),
                KeyCode::Char('q') | KeyCode::Esc => return Ok(None),
                _ => {}
            }
        }
    }
}

/// Affiche un écran et attend une touche. Renvoie `false` si l'utilisateur veut
/// quitter (`q`/Échap).
fn wait_or_quit(
    terminal: &mut DefaultTerminal,
    view: &Board,
    info: Vec<Line<'static>>,
    title: &str,
    panel: Vec<Line<'static>>,
) -> io::Result<bool> {
    draw(terminal, view, info, title, panel, "une touche pour continuer   q quitter")?;
    loop {
        if let Event::Key(k) = event::read()? {
            if k.kind == KeyEventKind::Press {
                return Ok(!matches!(k.code, KeyCode::Char('q') | KeyCode::Esc));
            }
        }
    }
}

/// L'adversaire : l'heuristique, ou un réseau entraîné chargé depuis un fichier.
enum Ai {
    Heuristic,
    Net(Net),
}

impl Ai {
    /// Indice du coup que l'IA préfère (argmax de sa valeur).
    fn choose(&self, plays: &[Board]) -> usize {
        let mut best = 0usize;
        let mut best_score = f64::NEG_INFINITY;
        for (i, b) in plays.iter().enumerate() {
            let s = match self {
                Ai::Heuristic => evaluate(b),
                Ai::Net(net) => net.value(b),
            };
            if s > best_score {
                best_score = s;
                best = i;
            }
        }
        best
    }

    fn label(&self) -> &'static str {
        match self {
            Ai::Heuristic => "heuristique",
            Ai::Net(_) => "réseau entraîné",
        }
    }
}

// --- Rendu -------------------------------------------------------------------

/// Dessine une frame complète : le plateau à gauche, un panneau à droite
/// (infos + contenu variable), et une ligne d'aide en bas.
fn render(
    f: &mut Frame,
    view: &Board,
    info: Vec<Line<'static>>,
    panel_title: &str,
    panel: Vec<Line<'static>>,
    help: &str,
) {
    let area = f.area();
    let [content, help_area] =
        Layout::vertical([Constraint::Min(0), Constraint::Length(1)]).areas(area);
    let [board_area, side] =
        Layout::horizontal([Constraint::Length(43), Constraint::Min(26)]).areas(content);

    f.render_widget(
        Paragraph::new(board_lines(view)).block(Block::bordered().title(" Backgammon ")),
        board_area,
    );

    let [info_area, panel_area] =
        Layout::vertical([Constraint::Length(5), Constraint::Min(3)]).areas(side);
    f.render_widget(
        Paragraph::new(info).block(Block::bordered().title(" Infos ")),
        info_area,
    );
    f.render_widget(
        Paragraph::new(panel).block(Block::bordered().title(format!(" {panel_title} "))),
        panel_area,
    );

    f.render_widget(Paragraph::new(format!(" {help} ")).style(dim()), help_area);
}

/// Pousse une frame vers le vrai terminal.
fn draw(
    terminal: &mut DefaultTerminal,
    view: &Board,
    info: Vec<Line<'static>>,
    panel_title: &str,
    panel: Vec<Line<'static>>,
    help: &str,
) -> io::Result<()> {
    terminal.draw(|f| render(f, view, info, panel_title, panel, help))?;
    Ok(())
}

fn info_lines(to_move: Player, roll: Roll, opp: &str) -> Vec<Line<'static>> {
    vec![
        Line::from(format!("Tour : {}", who(to_move))),
        Line::from(format!("Dés  : {} - {}", roll.d1, roll.d2)),
        Line::from(format!("IA   : {opp}")),
    ]
}

fn who(p: Player) -> &'static str {
    match p {
        Player::White => "Toi (X)",
        Player::Black => "IA (O)",
    }
}

/// La position vue des Blancs (toi) : telle quelle si c'est ton tour, retournée
/// si c'est celui de l'IA. Ainsi tes pions (X) restent toujours en bas.
fn white_view(board: &Board, to_move: Player) -> Board {
    match to_move {
        Player::White => board.clone(),
        Player::Black => board.swap_perspective(),
    }
}

/// Une colonne de pions (les `H` cellules, base près du bord du plateau).
struct Col {
    cells: [(char, Color); H],
}

fn make_col(v: i8) -> Col {
    let (sym, n, color) = if v > 0 {
        ('●', v as usize, HUMAN)
    } else if v < 0 {
        ('●', (-v) as usize, AI)
    } else {
        (' ', 0, DIM)
    };
    let mut cells = [(' ', DIM); H];
    let shown = n.min(H);
    for cell in cells.iter_mut().take(shown) {
        *cell = (sym, color);
    }
    if n > H {
        // Pile trop haute : on affiche le compte dans la cellule la plus interne.
        let digit = std::char::from_digit(n as u32, 10).unwrap_or('+');
        cells[H - 1] = (digit, color);
    }
    Col { cells }
}

/// Assemble 12 champs (3 caractères chacun) en une ligne, barre au milieu.
fn fields_to_line(fields: Vec<Span<'static>>) -> Line<'static> {
    let mut spans = Vec::with_capacity(fields.len() + 1);
    for (k, sp) in fields.into_iter().enumerate() {
        if k == 6 {
            spans.push(Span::styled(" │ ", dim()));
        }
        spans.push(sp);
    }
    Line::from(spans)
}

fn label_line(idx: &[usize; 12]) -> Line<'static> {
    let fields = idx
        .iter()
        .map(|i| Span::styled(format!("{i:>3}"), dim()))
        .collect();
    fields_to_line(fields)
}

fn cells_line(cols: &[Col], depth: usize) -> Line<'static> {
    let fields = cols
        .iter()
        .map(|c| {
            let (ch, color) = c.cells[depth];
            Span::styled(format!("{ch:>3}"), Style::default().fg(color))
        })
        .collect();
    fields_to_line(fields)
}

/// Rend le plateau (vu des Blancs) en lignes colorées.
fn board_lines(b: &Board) -> Vec<Line<'static>> {
    let top_idx: [usize; 12] = [23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12];
    let bot_idx: [usize; 12] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
    let pts = b.points();
    let top: Vec<Col> = top_idx.iter().map(|&i| make_col(pts[i])).collect();
    let bot: Vec<Col> = bot_idx.iter().map(|&i| make_col(pts[i])).collect();

    let mut lines = Vec::new();
    lines.push(label_line(&top_idx));
    for d in 0..H {
        lines.push(cells_line(&top, d));
    }
    lines.push(Line::from(Span::styled("─".repeat(39), dim())));
    for d in (0..H).rev() {
        lines.push(cells_line(&bot, d));
    }
    lines.push(label_line(&bot_idx));

    let bar = b.bar();
    let off = b.off();
    lines.push(Line::from(vec![
        Span::styled("Barre ", dim()),
        Span::styled(format!("toi:{} ", bar[0]), Style::default().fg(HUMAN)),
        Span::styled(format!("IA:{}", bar[1]), Style::default().fg(AI)),
        Span::styled("   Sortis ", dim()),
        Span::styled(format!("toi:{} ", off[0]), Style::default().fg(HUMAN)),
        Span::styled(format!("IA:{}", off[1]), Style::default().fg(AI)),
    ]));
    lines
}

/// Décrit le coup menant de `cur` à `cand` (bilan net : d'où partent tes pions,
/// où ils arrivent ; `✗` marque une frappe). `cur` et `cand` sont vus du même
/// côté que celui qui joue (pions positifs = à lui).
fn describe(cur: &Board, cand: &Board) -> String {
    let (cp, ap) = (cur.points(), cand.points());
    let (cb, ab) = (cur.bar(), cand.bar());
    let (co, ao) = (cur.off(), cand.off());

    let mut sources: Vec<String> = Vec::new();
    let mut dests: Vec<String> = Vec::new();

    let bar_in = cb[0] as i32 - ab[0] as i32;
    for _ in 0..bar_in.max(0) {
        sources.push("barre".to_string());
    }

    for p in (0..24).rev() {
        let cm = if cp[p] > 0 { cp[p] as i32 } else { 0 };
        let am = if ap[p] > 0 { ap[p] as i32 } else { 0 };
        let d = am - cm;
        if d < 0 {
            for _ in 0..(-d) {
                sources.push(p.to_string());
            }
        } else if d > 0 {
            for _ in 0..d {
                dests.push(p.to_string());
            }
        }
    }

    let off_out = ao[0] as i32 - co[0] as i32;
    for _ in 0..off_out.max(0) {
        dests.push("sortie".to_string());
    }

    let mut s = format!("{} → {}", sources.join(","), dests.join(","));
    if ab[1] > cb[1] {
        s.push_str(" ✗"); // frappe d'un pion adverse
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    #[test]
    fn le_rendu_ne_panique_pas_et_contient_le_plateau() {
        let board = Board::starting_position();
        let roll = Roll { d1: 3, d2: 1 };
        let plays = legal_plays(&board, &roll);
        let panel: Vec<Line<'static>> = plays
            .iter()
            .map(|c| Line::from(describe(&board, c)))
            .collect();

        let mut terminal = Terminal::new(TestBackend::new(80, 24)).unwrap();
        terminal
            .draw(|f| render(f, &board, info_lines(Player::White, roll, "heuristique"), "Coups", panel, "aide"))
            .unwrap();

        let screen = format!("{}", terminal.backend());
        assert!(screen.contains("Backgammon"), "le titre du plateau doit apparaître");
        assert!(screen.contains("Infos"), "le panneau d'infos doit apparaître");
    }
}
