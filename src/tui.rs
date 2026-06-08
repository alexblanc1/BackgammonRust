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
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Padding, Paragraph};

use backgammon::agent::heuristic::evaluate;
use backgammon::board::Board;
use backgammon::dice::{Dice, Roll};
use backgammon::moves::legal_plays;
use backgammon::net::Net;
use backgammon::player::Player;

/// Hauteur d'affichage d'une pile de pions.
const H: usize = 5;

const HUMAN: Color = Color::Cyan; // tes pions (●)
const AI: Color = Color::Red; // pions de l'IA (●)
const DIM: Color = Color::DarkGray;
/// Teintes des bandes (triangles) du plateau : une claire, une foncée.
const BAND_LIGHT: Color = Color::Rgb(222, 209, 184);
const BAND_DARK: Color = Color::Rgb(96, 70, 50);
/// La barre centrale, en bois.
const BAR_COLOR: Color = Color::Rgb(150, 111, 71);
/// Couleur d'accent pour les titres des cadres.
const ACCENT: Color = Color::Rgb(216, 190, 120);

fn dim() -> Style {
    Style::default().fg(DIM)
}

fn fg(c: Color) -> Style {
    Style::default().fg(c)
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
            if !wait(terminal, &view, to_move, roll, ai.label(), "Pas de coup", msg)? {
                return Ok(());
            }
        } else if to_move == Player::White {
            match human_select(terminal, &board, &plays, roll, ai.label())? {
                Some(i) => board = plays[i].clone(),
                None => return Ok(()), // l'utilisateur a quitté
            }
        } else {
            // On montre d'abord les dés de l'IA, *avant* qu'elle joue.
            let msg = vec![
                Line::from("L'IA a lancé les dés."),
                Line::from("Une touche : elle joue."),
            ];
            if !wait(terminal, &view, to_move, roll, ai.label(), "Au tour de l'IA", msg)? {
                return Ok(());
            }
            // Puis elle joue, et on révèle le plateau après son coup.
            let i = ai.choose(&plays);
            board = plays[i].clone();
            let view_after = white_view(&board, to_move);
            let msg = vec![
                Line::from("L'IA a joué son tour."),
                Line::from("Observe le plateau."),
            ];
            if !wait(terminal, &view_after, to_move, roll, ai.label(), "L'IA a joué", msg)? {
                return Ok(());
            }
        }

        if let Some(points) = board.win_check() {
            let view_final = white_view(&board, to_move);
            let label = match to_move {
                Player::White => format!("Tu gagnes ! ({points} point(s)) 🎉"),
                Player::Black => format!("L'IA gagne ({points} point(s))."),
            };
            wait(
                terminal,
                &view_final,
                to_move,
                roll,
                ai.label(),
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
                let text = format!(" {:>2}. {} ", i + 1, describe(board, cand));
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

        let screen = Screen {
            view: board,
            to_move: Player::White,
            roll,
            opp,
            title: "Coups possibles",
            panel,
            help: "↑/↓ choisir   Entrée valider   q quitter",
        };
        draw(terminal, &screen)?;

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
fn wait(
    terminal: &mut DefaultTerminal,
    view: &Board,
    to_move: Player,
    roll: Roll,
    opp: &str,
    title: &str,
    panel: Vec<Line<'static>>,
) -> io::Result<bool> {
    let screen = Screen {
        view,
        to_move,
        roll,
        opp,
        title,
        panel,
        help: "une touche pour continuer   q quitter",
    };
    draw(terminal, &screen)?;
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

/// Tout ce qu'il faut pour dessiner une frame. Regroupé pour alléger les appels.
struct Screen<'a> {
    view: &'a Board,
    to_move: Player,
    roll: Roll,
    opp: &'a str,
    title: &'a str,
    panel: Vec<Line<'static>>,
    help: &'a str,
}

/// Un cadre arrondi, à bord discret et titre coloré.
fn panel_block(title: &str, accent: Color) -> Block<'static> {
    Block::bordered()
        .border_type(BorderType::Rounded)
        .border_style(fg(DIM))
        .title(Span::styled(
            format!(" {title} "),
            Style::default().fg(accent).add_modifier(Modifier::BOLD),
        ))
}

/// Dessine une frame complète : le plateau à gauche ; à droite les infos, les
/// dés, puis le panneau variable (coups ou message) ; une ligne d'aide en bas.
fn render(f: &mut Frame, s: &Screen) {
    let area = f.area();
    let [content, help_area] =
        Layout::vertical([Constraint::Min(0), Constraint::Length(1)]).areas(area);
    let [board_area, side] =
        Layout::horizontal([Constraint::Length(43), Constraint::Min(28)]).areas(content);

    f.render_widget(
        Paragraph::new(board_lines(s.view)).block(panel_block("Backgammon", ACCENT)),
        board_area,
    );

    let [info_area, dice_area, panel_area] = Layout::vertical([
        Constraint::Length(5),
        Constraint::Length(7),
        Constraint::Min(3),
    ])
    .areas(side);

    f.render_widget(
        Paragraph::new(info_lines(s.to_move, s.opp, s.view))
            .block(panel_block("Infos", ACCENT).padding(Padding::horizontal(1))),
        info_area,
    );
    f.render_widget(
        Paragraph::new(dice_lines(s.roll, s.to_move))
            .block(panel_block(&dice_title(s.roll), ACCENT)),
        dice_area,
    );
    f.render_widget(
        Paragraph::new(s.panel.clone())
            .block(panel_block(s.title, ACCENT).padding(Padding::horizontal(1))),
        panel_area,
    );

    f.render_widget(Paragraph::new(format!("  {}", s.help)).style(dim()), help_area);
}

/// Pousse une frame vers le vrai terminal.
fn draw(terminal: &mut DefaultTerminal, s: &Screen) -> io::Result<()> {
    terminal.draw(|f| render(f, s))?;
    Ok(())
}

/// Les lignes du panneau « Infos » : tour courant, adversaire, pip counts.
fn info_lines(to_move: Player, opp: &str, view: &Board) -> Vec<Line<'static>> {
    let (human_pip, ai_pip) = pip_counts(view);
    vec![
        Line::from(vec![Span::styled("Tour   ", dim()), turn_span(to_move)]),
        Line::from(vec![
            Span::styled("Contre ", dim()),
            Span::raw(opp.to_string()),
        ]),
        Line::from(vec![
            Span::styled("Pips   ", dim()),
            Span::styled(format!("toi {human_pip}"), fg(HUMAN)),
            Span::styled("   ", dim()),
            Span::styled(format!("IA {ai_pip}"), fg(AI)),
        ]),
    ]
}

/// Le « pip count » de chacun : nombre de points de dé restants pour tout sortir.
/// Vue Blancs : tes pions (`> 0`) avancent 23→0 et coûtent `i+1` ; l'IA (`< 0`)
/// va 0→23 et coûte `24-i` ; un pion sur la barre vaut 25.
fn pip_counts(view: &Board) -> (u32, u32) {
    let pts = view.points();
    let bar = view.bar();
    let mut human = bar[0] as u32 * 25;
    let mut ai = bar[1] as u32 * 25;
    for (i, &v) in pts.iter().enumerate() {
        if v > 0 {
            human += v as u32 * (i as u32 + 1);
        } else if v < 0 {
            ai += (-v) as u32 * (24 - i as u32);
        }
    }
    (human, ai)
}

fn turn_span(p: Player) -> Span<'static> {
    let (txt, color) = match p {
        Player::White => ("Toi  ●", HUMAN),
        Player::Black => ("IA   ●", AI),
    };
    Span::styled(txt, Style::default().fg(color).add_modifier(Modifier::BOLD))
}

fn who(p: Player) -> &'static str {
    match p {
        Player::White => "Toi",
        Player::Black => "L'IA",
    }
}

// --- Les dés ----------------------------------------------------------------

/// Titre du cadre des dés : précise le double (joué 4 fois) le cas échéant.
fn dice_title(roll: Roll) -> String {
    if roll.d1 == roll.d2 {
        format!("Dés — double de {} (×4)", roll.d1)
    } else {
        format!("Dés — {} & {}", roll.d1, roll.d2)
    }
}

/// Les deux dés dessinés côte à côte, colorés selon le joueur courant.
fn dice_lines(roll: Roll, to_move: Player) -> Vec<Line<'static>> {
    let color = match to_move {
        Player::White => HUMAN,
        Player::Black => AI,
    };
    let left = die_face(roll.d1);
    let right = die_face(roll.d2);
    (0..5)
        .map(|r| {
            Line::from(vec![
                Span::raw("  "),
                Span::styled(left[r].clone(), fg(color)),
                Span::raw("   "),
                Span::styled(right[r].clone(), fg(color)),
            ])
        })
        .collect()
}

/// Un dé dessiné en 5 lignes (cadre + 3 rangées de points).
fn die_face(n: u8) -> [String; 5] {
    let g = pip_grid(n);
    let cell = |b: bool| if b { '●' } else { ' ' };
    let row = |r: usize| format!("│ {} {} {} │", cell(g[r][0]), cell(g[r][1]), cell(g[r][2]));
    [
        "╭───────╮".to_string(),
        row(0),
        row(1),
        row(2),
        "╰───────╯".to_string(),
    ]
}

/// Quels points (3×3) sont allumés pour la face `n`.
fn pip_grid(n: u8) -> [[bool; 3]; 3] {
    let mut g = [[false; 3]; 3];
    if matches!(n, 2 | 3 | 4 | 5 | 6) {
        g[0][0] = true; // coin haut-gauche
        g[2][2] = true; // coin bas-droit
    }
    if matches!(n, 4 | 5 | 6) {
        g[0][2] = true; // coin haut-droit
        g[2][0] = true; // coin bas-gauche
    }
    if matches!(n, 1 | 3 | 5) {
        g[1][1] = true; // centre
    }
    if n == 6 {
        g[1][0] = true; // milieux gauche/droit
        g[1][2] = true;
    }
    g
}

// --- Le plateau --------------------------------------------------------------

/// La position vue des Blancs (toi) : telle quelle si c'est ton tour, retournée
/// si c'est celui de l'IA. Ainsi tes pions restent toujours en bas.
fn white_view(board: &Board, to_move: Player) -> Board {
    match to_move {
        Player::White => board.clone(),
        Player::Black => board.swap_perspective(),
    }
}

/// Une colonne de pions : `H` cellules, `None` quand la cellule est vide (la
/// bande/triangle se dessinera à la place), base près du bord du plateau.
struct Col {
    cells: [Option<(char, Color)>; H],
}

fn make_col(v: i8) -> Col {
    let (n, color) = if v > 0 {
        (v as usize, HUMAN)
    } else if v < 0 {
        ((-v) as usize, AI)
    } else {
        (0, DIM)
    };
    let mut cells = [None; H];
    let shown = n.min(H);
    for cell in cells.iter_mut().take(shown) {
        *cell = Some(('●', color));
    }
    if n > H {
        // Pile trop haute : on affiche le compte dans la cellule la plus interne.
        let digit = std::char::from_digit(n as u32, 10).unwrap_or('+');
        cells[H - 1] = Some((digit, color));
    }
    Col { cells }
}

/// Assemble 12 champs (3 caractères chacun) en une ligne, barre au milieu.
fn fields_to_line(fields: Vec<Span<'static>>) -> Line<'static> {
    let mut spans = Vec::with_capacity(fields.len() + 1);
    for (k, sp) in fields.into_iter().enumerate() {
        if k == 6 {
            spans.push(Span::styled("┃", fg(BAR_COLOR)));
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

/// Une rangée de cellules. Les cases vides reçoivent le triangle de la bande
/// (`▼` en haut, `▲` en bas), en teinte alternée — claire/foncée — et opposée
/// entre le haut et le bas, comme sur un vrai plateau.
fn cells_line(cols: &[Col], depth: usize, top: bool) -> Line<'static> {
    let glyph = if top { '▼' } else { '▲' };
    let base = if top { 0 } else { 1 };
    let fields = cols
        .iter()
        .enumerate()
        .map(|(k, c)| match c.cells[depth] {
            Some((ch, color)) => Span::styled(format!(" {ch} "), fg(color)),
            None => {
                let light = (k + base) % 2 == 0;
                let band = if light { BAND_LIGHT } else { BAND_DARK };
                Span::styled(format!(" {glyph} "), fg(band))
            }
        })
        .collect();
    fields_to_line(fields)
}

/// Le trait horizontal qui sépare les deux moitiés, croisé par la barre.
fn mid_rule() -> Line<'static> {
    Line::from(vec![
        Span::styled("─".repeat(18), dim()),
        Span::styled("╋", fg(BAR_COLOR)),
        Span::styled("─".repeat(18), dim()),
    ])
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
        lines.push(cells_line(&top, d, true));
    }
    lines.push(mid_rule());
    for d in (0..H).rev() {
        lines.push(cells_line(&bot, d, false));
    }
    lines.push(label_line(&bot_idx));
    lines.push(Line::from(""));

    let bar = b.bar();
    let off = b.off();
    lines.push(Line::from(vec![
        Span::styled("Barre  ", dim()),
        Span::styled(format!("toi {} ", bar[0]), fg(HUMAN)),
        Span::styled(format!("IA {}", bar[1]), fg(AI)),
        Span::styled("    Sortis  ", dim()),
        Span::styled(format!("toi {} ", off[0]), fg(HUMAN)),
        Span::styled(format!("IA {}", off[1]), fg(AI)),
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

        let screen = Screen {
            view: &board,
            to_move: Player::White,
            roll,
            opp: "heuristique",
            title: "Coups",
            panel,
            help: "aide",
        };

        let mut terminal = Terminal::new(TestBackend::new(80, 24)).unwrap();
        terminal.draw(|f| render(f, &screen)).unwrap();

        let screen = format!("{}", terminal.backend());
        assert!(screen.contains("Backgammon"), "le titre du plateau doit apparaître");
        assert!(screen.contains("Infos"), "le panneau d'infos doit apparaître");
    }

    #[test]
    fn pip_count_position_de_depart() {
        // Position de départ : pip count standard = 167 pour chacun.
        let (human, ai) = pip_counts(&Board::starting_position());
        assert_eq!(human, 167);
        assert_eq!(ai, 167);
    }

    #[test]
    fn faces_de_des_ont_le_bon_nombre_de_points() {
        for n in 1..=6u8 {
            let count: usize = pip_grid(n).iter().flatten().filter(|&&b| b).count();
            assert_eq!(count, n as usize, "la face {n} doit avoir {n} points");
        }
    }
}
