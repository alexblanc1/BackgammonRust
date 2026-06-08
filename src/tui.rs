//! Interface graphique du jeu dans le terminal, avec Ratatui.
//!
//! Module du *binaire* (pas de la bibliothèque) : le moteur reste ainsi sans
//! dépendance d'affichage. Le plateau est toujours montré du point de vue des
//! Blancs (le joueur humain) ; on retourne la position pour l'affichage quand
//! c'est l'IA qui vient de jouer.
//!
//! Orientation : ta base (jan intérieur) est en **bas à droite**, l'adversaire
//! démarre en **haut à droite**, comme sur un vrai plateau.
//!
//! Saisie d'un coup : tu déplaces un curseur sur le plateau avec `hjkl`, les
//! coups possibles s'affichent en surbrillance, et tu joues pion par pion.

use std::io;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

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

/// Hauteur d'affichage d'une pile de pions (plus grand = plus lisible).
const H: usize = 6;

const HUMAN: Color = Color::Cyan; // tes pions (●)
const AI: Color = Color::Red; // pions de l'IA (●)
/// Gris « discret » mais en RGB explicite : contrairement à `DarkGray` (un
/// index de la palette du terminal, parfois confondu avec le fond selon le
/// thème), un RGB fixe reste visible quel que soit le thème.
const DIM: Color = Color::Rgb(125, 125, 125);
/// Couleur des numéros de cases (les « bandes ») : RGB fixe, toujours visible.
const LABEL: Color = Color::Rgb(165, 165, 165);
/// Teintes des bandes (triangles) du plateau : une claire, une foncée.
const BAND_LIGHT: Color = Color::Rgb(222, 209, 184);
const BAND_DARK: Color = Color::Rgb(96, 70, 50);
/// La barre centrale, en bois.
const BAR_COLOR: Color = Color::Rgb(150, 111, 71);
/// Couleur d'accent pour les titres des cadres.
const ACCENT: Color = Color::Rgb(216, 190, 120);

// Couleurs de surbrillance lors de la saisie d'un coup.
const SRC_BG: Color = Color::Rgb(45, 60, 82); // cases d'où tu peux jouer
const DST_BG: Color = Color::Rgb(34, 96, 52); // destinations possibles
const PICK_BG: Color = Color::Rgb(122, 96, 26); // case « prise en main »
const CURSOR_BG: Color = Color::Rgb(70, 70, 82); // position du curseur

/// Disposition visuelle des 12 cases du haut (de gauche à droite) : indices
/// moteur 12..17 à gauche de la barre, 18..23 à droite.
const TOP_IDX: [usize; 12] = [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23];
/// Disposition visuelle des 12 cases du bas : 11..6 à gauche, puis 5..0 à
/// droite. Ainsi l'index 0 (ta « case 1 », d'où tu sors) est en bas à droite.
const BOT_IDX: [usize; 12] = [11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0];

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

        if plays.is_empty() {
            // Personne ne peut jouer ce lancer : on l'annonce, puis on enchaîne.
            let view = white_view(&board, to_move);
            let screen = Screen {
                view: &view,
                to_move,
                roll,
                opp: ai.label(),
                title: "Pas de coup",
                panel: vec![
                    Line::from(format!("{} ne peut pas jouer ce lancer.", who(to_move))),
                    Line::from("Tour passé."),
                ],
                help: "Entrée : continuer   ·   q : quitter",
                hl: None,
            };
            if !pause(terminal, &screen, Duration::from_millis(1500))? {
                return Ok(());
            }
        } else if to_move == Player::White {
            // Ton tour : tu joues les pions au clavier sur le plateau.
            match human_turn(terminal, &board, &plays, roll, ai.label())? {
                Some(b) => board = b,
                None => return Ok(()), // tu as quitté
            }
        } else {
            // Tour de l'IA : elle choisit, joue, et on révèle la position. On
            // reprend automatiquement après un court délai (ou dès qu'une touche
            // est pressée), pour ne pas avoir à valider à la main.
            let i = ai.choose(&plays);
            board = plays[i].clone();
            let view = white_view(&board, to_move);
            let screen = Screen {
                view: &view,
                to_move,
                roll,
                opp: ai.label(),
                title: "L'IA a joué",
                panel: vec![
                    Line::from("L'IA a joué son coup."),
                    Line::from(""),
                    Line::from(Span::styled("Reprise automatique…", dim())),
                ],
                help: "Entrée : continuer tout de suite   ·   q : quitter",
                hl: None,
            };
            if !pause(terminal, &screen, Duration::from_millis(1500))? {
                return Ok(());
            }
        }

        if let Some(points) = board.win_check() {
            let view_final = white_view(&board, to_move);
            let label = match to_move {
                Player::White => format!("Tu gagnes ! ({points} point(s)) 🎉"),
                Player::Black => format!("L'IA gagne ({points} point(s))."),
            };
            let screen = Screen {
                view: &view_final,
                to_move,
                roll,
                opp: ai.label(),
                title: "Fin de la partie",
                panel: vec![Line::from(label), Line::from("Une touche pour quitter.")],
                help: "une touche pour quitter",
                hl: None,
            };
            wait_key(terminal, &screen)?;
            return Ok(());
        }

        board = board.swap_perspective();
        to_move = to_move.other();
    }
}

// --- Saisie d'un coup, pion par pion, au clavier -----------------------------

/// D'où part un pion qu'on déplace : une case du plateau, ou la barre.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Src {
    Point(usize),
    Bar,
}

/// Où arrive un pion : une case du plateau, ou la sortie (bear off).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Dst {
    Point(usize),
    Off,
}

/// Un déplacement d'un seul dé : sa valeur, sa source, sa destination, et la
/// position obtenue. `next` est une position **complète** du moteur.
#[derive(Clone)]
struct SubMove {
    die: u8,
    src: Src,
    dst: Dst,
    next: Board,
}

/// Laisse l'humain composer son coup au clavier. `finals` = `legal_plays`, la
/// liste (non vide) des positions complètes légales. Renvoie la position
/// choisie, ou `None` s'il quitte.
fn human_turn(
    terminal: &mut DefaultTerminal,
    start: &Board,
    finals: &[Board],
    roll: Roll,
    opp: &str,
) -> io::Result<Option<Board>> {
    let mut cur = start.clone();
    // Les valeurs de dés qu'on doit encore jouer (en respectant l'usage maximal).
    let initial_dice = maximal_dice(start, &roll);
    let mut dice_left = initial_dice.clone();

    // Curseur : on le place d'emblée sur une case jouable (ou une case d'entrée
    // si un pion est sur la barre).
    let init_moves = available_moves(&cur, &dice_left, finals);
    let mut cursor = init_moves
        .iter()
        .find_map(|m| match (m.src, m.dst) {
            (Src::Bar, Dst::Point(p)) => Some(p),     // entrée depuis la barre
            (Src::Point(p), _) => Some(p),            // case jouable
            _ => None,
        })
        .unwrap_or(0);

    // `picked` = la source qu'on a « prise en main » (None tant qu'on n'a rien
    // pris). Pile pour annuler un sous-coup (touche `u`).
    let mut picked: Option<Src> = None;
    let mut history: Vec<(Board, Vec<u8>, usize)> = Vec::new();

    loop {
        let moves = available_moves(&cur, &dice_left, finals);
        let complete = moves.is_empty(); // plus aucun dé jouable = coup terminé

        // Si le seul coup possible est de rentrer de la barre, on prend la barre
        // en main automatiquement (rien d'autre n'est permis).
        let must_bar = !complete && moves.iter().all(|m| matches!(m.src, Src::Bar));
        if must_bar && picked.is_none() {
            picked = Some(Src::Bar);
        }

        // Les coups « actifs » = ceux qu'on peut jouer là, tout de suite : depuis
        // la source prise en main, ou (sinon) depuis la case sous le curseur.
        let active: Vec<&SubMove> = match picked {
            Some(s) => moves.iter().filter(|m| m.src == s).collect(),
            None => moves
                .iter()
                .filter(|m| m.src == Src::Point(cursor))
                .collect(),
        };

        // Surbrillances à dessiner.
        let preview_dests: Vec<usize> = active
            .iter()
            .filter_map(|m| match m.dst {
                Dst::Point(p) => Some(p),
                Dst::Off => None,
            })
            .collect();
        let sources: Vec<usize> = if picked.is_some() {
            Vec::new()
        } else {
            let mut v = Vec::new();
            for m in &moves {
                if let Src::Point(p) = m.src
                    && !v.contains(&p)
                {
                    v.push(p);
                }
            }
            v
        };
        let picked_src = match picked {
            Some(Src::Point(p)) => Some(p),
            _ => None,
        };
        let hl = Hl {
            cursor,
            sources,
            picked_src,
            preview_dests,
        };

        let panel = turn_panel(&dice_left, &active, picked, must_bar, complete);
        let help = if complete {
            "Entrée : valider   ·   u : annuler   ·   q : quitter"
        } else {
            "hjkl : déplacer   ·   Entrée : jouer   ·   1-6 : choisir le dé   ·   u : annuler   ·   q : quitter"
        };
        let screen = Screen {
            view: &cur,
            to_move: Player::White,
            roll,
            opp,
            title: if complete { "Coup terminé" } else { "Ton coup" },
            panel,
            help,
            hl: Some(hl),
        };
        draw(terminal, &screen)?;

        // --- Lecture d'une touche ---
        let key = match event::read()? {
            Event::Key(k) if k.kind == KeyEventKind::Press => k.code,
            _ => continue,
        };

        if complete {
            // Le coup est entièrement joué : on valide, on annule, ou on quitte.
            match key {
                KeyCode::Enter | KeyCode::Char(' ') => return Ok(Some(cur)),
                KeyCode::Char('u') => {
                    if let Some((b, dl, c)) = history.pop() {
                        cur = b;
                        dice_left = dl;
                        cursor = c;
                        picked = None;
                    }
                }
                KeyCode::Char('q') | KeyCode::Esc => return Ok(None),
                _ => {}
            }
            continue;
        }

        match key {
            // Déplacement du curseur (hjkl ou flèches).
            KeyCode::Char('h') | KeyCode::Left => cursor = move_cursor(cursor, 'h'),
            KeyCode::Char('l') | KeyCode::Right => cursor = move_cursor(cursor, 'l'),
            KeyCode::Char('k') | KeyCode::Up => cursor = move_cursor(cursor, 'k'),
            KeyCode::Char('j') | KeyCode::Down => cursor = move_cursor(cursor, 'j'),

            // Entrée : prendre une case en main, ou jouer vers une destination.
            KeyCode::Enter | KeyCode::Char(' ') => match picked {
                Some(src) => {
                    // On joue vers la case sous le curseur si c'est une
                    // destination ; sinon, s'il n'y a qu'un coup, on le joue.
                    let target = cursor; // copie : la closure ne doit pas emprunter `cursor`
                    let played = apply_move(
                        &moves,
                        |m| m.src == src && m.dst == Dst::Point(target),
                        &mut cur,
                        &mut dice_left,
                        &mut cursor,
                        &mut picked,
                        &mut history,
                    );
                    if !played && active.len() == 1 {
                        apply_move(
                            &moves,
                            |m| m.src == src,
                            &mut cur,
                            &mut dice_left,
                            &mut cursor,
                            &mut picked,
                            &mut history,
                        );
                    }
                }
                None => {
                    let target = cursor; // copie : la closure ne doit pas emprunter `cursor`
                    let from: Vec<&SubMove> = moves
                        .iter()
                        .filter(|m| m.src == Src::Point(target))
                        .collect();
                    match from.len() {
                        0 => {} // pas une case jouable
                        1 => {
                            apply_move(
                                &moves,
                                |m| m.src == Src::Point(target),
                                &mut cur,
                                &mut dice_left,
                                &mut cursor,
                                &mut picked,
                                &mut history,
                            );
                        }
                        _ => {
                            // Plusieurs destinations : on prend la case en main et
                            // on amène le curseur sur la première destination.
                            picked = Some(Src::Point(cursor));
                            if let Some(p) = from.iter().find_map(|m| match m.dst {
                                Dst::Point(p) => Some(p),
                                Dst::Off => None,
                            }) {
                                cursor = p;
                            }
                        }
                    }
                }
            },

            // 1-6 : jouer directement avec ce dé (pratique pour la sortie).
            KeyCode::Char(c @ '1'..='6') => {
                let d = c as u8 - b'0';
                let src = picked.unwrap_or(if must_bar {
                    Src::Bar
                } else {
                    Src::Point(cursor)
                });
                apply_move(
                    &moves,
                    |m| m.src == src && m.die == d,
                    &mut cur,
                    &mut dice_left,
                    &mut cursor,
                    &mut picked,
                    &mut history,
                );
            }

            // Annuler la prise en main, ou le dernier sous-coup.
            KeyCode::Esc => {
                if picked.is_some() && !must_bar {
                    picked = None;
                } else {
                    return Ok(None);
                }
            }
            KeyCode::Char('u') => {
                if let Some((b, dl, c)) = history.pop() {
                    cur = b;
                    dice_left = dl;
                    cursor = c;
                    picked = None;
                }
            }
            KeyCode::Char('r') => {
                cur = start.clone();
                dice_left = initial_dice.clone();
                history.clear();
                picked = None;
            }
            KeyCode::Char('q') => return Ok(None),
            _ => {}
        }
    }
}

/// Joue le premier coup de `moves` satisfaisant `pred` : empile l'état pour
/// l'annulation, retire le dé utilisé, avance le plateau et place le curseur sur
/// la destination. Renvoie `true` si un coup a été joué.
#[allow(clippy::too_many_arguments)]
fn apply_move(
    moves: &[SubMove],
    pred: impl Fn(&SubMove) -> bool,
    cur: &mut Board,
    dice_left: &mut Vec<u8>,
    cursor: &mut usize,
    picked: &mut Option<Src>,
    history: &mut Vec<(Board, Vec<u8>, usize)>,
) -> bool {
    if let Some(m) = moves.iter().find(|m| pred(m)) {
        history.push((cur.clone(), dice_left.clone(), *cursor));
        *dice_left = without_one(dice_left, m.die);
        if let Dst::Point(p) = m.dst {
            *cursor = p;
        }
        *cur = m.next.clone();
        *picked = None;
        true
    } else {
        false
    }
}

/// Le contenu du panneau de droite pendant ton coup : dés restants, et la liste
/// (courte) des déplacements possibles depuis la source courante.
fn turn_panel(
    dice_left: &[u8],
    active: &[&SubMove],
    picked: Option<Src>,
    must_bar: bool,
    complete: bool,
) -> Vec<Line<'static>> {
    let mut lines = vec![Line::from(vec![
        Span::styled("Dés restants  ", dim()),
        Span::styled(fmt_dice_left(dice_left), fg(HUMAN)),
    ])];
    lines.push(Line::from(""));

    if complete {
        lines.push(Line::from("Coup terminé."));
        lines.push(Line::from("Entrée pour valider."));
        return lines;
    }

    if must_bar {
        lines.push(Line::from("Pion sur la barre :"));
        lines.push(Line::from("fais-le rentrer."));
    } else if picked.is_some() {
        lines.push(Line::from("Choisis la destination :"));
    } else if active.is_empty() {
        lines.push(Line::from("Place le curseur sur"));
        lines.push(Line::from("un pion ● jouable,"));
        lines.push(Line::from("puis Entrée."));
        return lines;
    } else {
        lines.push(Line::from("Coups possibles :"));
    }

    for m in active {
        let dst = match m.dst {
            Dst::Point(p) => format!("case {p}"),
            Dst::Off => "sortie".to_string(),
        };
        lines.push(Line::from(format!("  dé {} → {}", m.die, dst)));
    }
    lines
}

fn fmt_dice_left(dl: &[u8]) -> String {
    if dl.is_empty() {
        "—".to_string()
    } else {
        dl.iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join("  ")
    }
}

// --- Logique des sous-coups (dé par dé) --------------------------------------

/// Les valeurs de dés que le coup *maximal* doit jouer, dans l'ordre du lancer.
/// (Mêmes règles que `legal_plays` : on joue le plus de dés possible ; à défaut,
/// le plus grand dé pour un non-double.)
fn maximal_dice(board: &Board, roll: &Roll) -> Vec<u8> {
    if roll.d1 == roll.d2 {
        let n = longest_double_chain(board, roll.d1, 4);
        vec![roll.d1; n]
    } else if uses_both(board, roll.d1, roll.d2) {
        vec![roll.d1, roll.d2]
    } else {
        let large = roll.d1.max(roll.d2);
        let small = roll.d1.min(roll.d2);
        if !board.single_die_moves(large).is_empty() {
            vec![large]
        } else if !board.single_die_moves(small).is_empty() {
            vec![small]
        } else {
            Vec::new()
        }
    }
}

/// Longueur de la plus longue suite de coups d'un même dé (pour les doubles).
fn longest_double_chain(board: &Board, die: u8, remaining: usize) -> usize {
    if remaining == 0 {
        return 0;
    }
    let mut best = 0;
    for next in board.single_die_moves(die) {
        best = best.max(1 + longest_double_chain(&next, die, remaining - 1));
    }
    best
}

/// Peut-on jouer les deux dés `a` et `b` (dans un ordre ou l'autre) ?
fn uses_both(board: &Board, a: u8, b: u8) -> bool {
    board
        .single_die_moves(a)
        .iter()
        .any(|n| !n.single_die_moves(b).is_empty())
        || board
            .single_die_moves(b)
            .iter()
            .any(|n| !n.single_die_moves(a).is_empty())
}

/// Les sous-coups jouables depuis `cur` qui **mènent encore** à une position
/// finale légale (en jouant ensuite tous les dés restants). C'est ce filtre qui
/// garantit qu'on respecte l'usage maximal des dés tout au long de la saisie.
fn available_moves(cur: &Board, dice_left: &[u8], finals: &[Board]) -> Vec<SubMove> {
    let mut out = Vec::new();
    for d in distinct(dice_left) {
        let rem = without_one(dice_left, d);
        for next in cur.single_die_moves(d) {
            let ok = if rem.is_empty() {
                finals.contains(&next)
            } else {
                can_reach_final(&next, &rem, finals)
            };
            if ok {
                let (src, dst) = diff_move(cur, &next);
                out.push(SubMove {
                    die: d,
                    src,
                    dst,
                    next,
                });
            }
        }
    }
    out
}

/// Depuis `b`, en jouant exactement les dés `rem`, peut-on atteindre une
/// position de `finals` ?
fn can_reach_final(b: &Board, rem: &[u8], finals: &[Board]) -> bool {
    if rem.is_empty() {
        return finals.contains(b);
    }
    for d in distinct(rem) {
        let r2 = without_one(rem, d);
        for next in b.single_die_moves(d) {
            if can_reach_final(&next, &r2, finals) {
                return true;
            }
        }
    }
    false
}

/// Déduit (source, destination) d'un sous-coup en comparant `cur` et `next`.
/// Un seul de tes pions a bougé : on trouve la case qui en a perdu un (source)
/// et celle qui en a gagné un (destination).
fn diff_move(cur: &Board, next: &Board) -> (Src, Dst) {
    let (cp, np) = (cur.points(), next.points());

    let src = if next.bar()[0] < cur.bar()[0] {
        Src::Bar
    } else {
        let mut s = Src::Bar;
        for p in 0..24 {
            if np[p].max(0) == cp[p].max(0) - 1 {
                s = Src::Point(p);
                break;
            }
        }
        s
    };

    let dst = if next.off()[0] > cur.off()[0] {
        Dst::Off
    } else {
        let mut d = Dst::Off;
        for p in 0..24 {
            if np[p].max(0) == cp[p].max(0) + 1 {
                d = Dst::Point(p);
                break;
            }
        }
        d
    };

    (src, dst)
}

/// Les valeurs distinctes d'une liste de dés (pour ne pas explorer deux fois le
/// même dé).
fn distinct(v: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    for &d in v {
        if !out.contains(&d) {
            out.push(d);
        }
    }
    out
}

/// La même liste, privée d'une occurrence de `d`.
fn without_one(v: &[u8], d: u8) -> Vec<u8> {
    let mut out = v.to_vec();
    if let Some(pos) = out.iter().position(|&x| x == d) {
        out.remove(pos);
    }
    out
}

// --- Déplacement du curseur sur la grille ------------------------------------

/// Position (rangée, colonne) d'une case dans la grille visuelle.
/// rangée 0 = haut, 1 = bas ; colonne 0..11 de gauche à droite.
fn locate(p: usize) -> (usize, usize) {
    if let Some(c) = TOP_IDX.iter().position(|&x| x == p) {
        (0, c)
    } else {
        (1, BOT_IDX.iter().position(|&x| x == p).unwrap())
    }
}

/// L'index moteur à la position (rangée, colonne).
fn grid_index(row: usize, col: usize) -> usize {
    if row == 0 {
        TOP_IDX[col]
    } else {
        BOT_IDX[col]
    }
}

/// Déplace le curseur d'une case selon `h`/`j`/`k`/`l`. `k` monte vers la
/// rangée du haut, `j` descend vers celle du bas (en gardant la colonne).
fn move_cursor(cur: usize, key: char) -> usize {
    let (mut row, mut col) = locate(cur);
    match key {
        'h' => col = col.saturating_sub(1),
        'l' => col = (col + 1).min(11),
        'k' => row = 0,
        'j' => row = 1,
        _ => {}
    }
    grid_index(row, col)
}

// --- L'adversaire ------------------------------------------------------------

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

// --- Attente / temporisation -------------------------------------------------

/// Affiche un écran et attend : soit qu'une touche soit pressée, soit que
/// `dur` s'écoule (reprise automatique). Renvoie `false` si l'utilisateur veut
/// quitter (`q`/Échap), `true` sinon.
fn pause(terminal: &mut DefaultTerminal, s: &Screen, dur: Duration) -> io::Result<bool> {
    draw(terminal, s)?;
    let start = Instant::now();
    loop {
        let elapsed = start.elapsed();
        if elapsed >= dur {
            return Ok(true); // délai écoulé : on reprend
        }
        // `poll` attend un événement au plus le temps restant ; on ne lit
        // (`read`) que s'il y en a un, d'où le `&&` qui court-circuite.
        if event::poll(dur - elapsed)?
            && let Event::Key(k) = event::read()?
            && k.kind == KeyEventKind::Press
        {
            return Ok(!matches!(k.code, KeyCode::Char('q') | KeyCode::Esc));
        }
    }
}

/// Affiche un écran et attend (sans limite de temps) qu'une touche soit pressée.
fn wait_key(terminal: &mut DefaultTerminal, s: &Screen) -> io::Result<()> {
    draw(terminal, s)?;
    loop {
        if let Event::Key(k) = event::read()?
            && k.kind == KeyEventKind::Press
        {
            return Ok(());
        }
    }
}

// --- Rendu -------------------------------------------------------------------

/// Les surbrillances à dessiner sur le plateau pendant la saisie d'un coup.
struct Hl {
    cursor: usize,
    sources: Vec<usize>,
    picked_src: Option<usize>,
    preview_dests: Vec<usize>,
}

/// Tout ce qu'il faut pour dessiner une frame. Regroupé pour alléger les appels.
struct Screen<'a> {
    view: &'a Board,
    to_move: Player,
    roll: Roll,
    opp: &'a str,
    title: &'a str,
    panel: Vec<Line<'static>>,
    help: &'a str,
    hl: Option<Hl>,
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
/// dés, puis le panneau variable ; une ligne d'aide en bas.
fn render(f: &mut Frame, s: &Screen) {
    let area = f.area();
    let [content, help_area] =
        Layout::vertical([Constraint::Min(0), Constraint::Length(1)]).areas(area);
    let [board_area, side] =
        Layout::horizontal([Constraint::Length(51), Constraint::Min(28)]).areas(content);

    f.render_widget(
        Paragraph::new(board_lines(s.view, s.hl.as_ref()))
            .block(panel_block("Backgammon", ACCENT)),
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

/// Le fond (background) d'une case, selon les surbrillances. Ordre de priorité :
/// destination > case prise en main > source jouable > curseur.
fn point_bg(p: usize, hl: Option<&Hl>) -> Option<Color> {
    let hl = hl?;
    if hl.preview_dests.contains(&p) {
        Some(DST_BG)
    } else if hl.picked_src == Some(p) {
        Some(PICK_BG)
    } else if hl.sources.contains(&p) {
        Some(SRC_BG)
    } else if hl.cursor == p {
        Some(CURSOR_BG)
    } else {
        None
    }
}

/// Une cellule de 4 caractères de large (` x  `), avec couleur de pion et fond
/// éventuel.
fn styled_cell(ch: char, fg_color: Color, bg: Option<Color>) -> Span<'static> {
    let mut st = Style::default().fg(fg_color);
    if let Some(b) = bg {
        st = st.bg(b);
    }
    Span::styled(format!(" {ch}  "), st)
}

/// Assemble 12 champs (4 caractères chacun) en une ligne, barre au milieu.
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

/// La ligne des numéros de cases. La case sous le curseur est mise en évidence
/// (texte sombre sur fond doré), toujours visible quel que soit le thème.
fn label_line(idx: &[usize; 12], hl: Option<&Hl>) -> Line<'static> {
    let cursor = hl.map(|h| h.cursor);
    let fields = idx
        .iter()
        .map(|&i| {
            if cursor == Some(i) {
                Span::styled(
                    format!(" {i:>2} "),
                    Style::default()
                        .fg(Color::Black)
                        .bg(ACCENT)
                        .add_modifier(Modifier::BOLD),
                )
            } else {
                Span::styled(format!(" {i:>2} "), fg(LABEL))
            }
        })
        .collect();
    fields_to_line(fields)
}

/// Une rangée de cellules. Les cases vides reçoivent le triangle de la bande
/// (`▼` en haut, `▲` en bas), en teinte alternée — claire/foncée — et opposée
/// entre le haut et le bas, comme sur un vrai plateau.
fn cells_line(
    cols: &[Col],
    idx: &[usize; 12],
    depth: usize,
    top: bool,
    hl: Option<&Hl>,
) -> Line<'static> {
    let glyph = if top { '▼' } else { '▲' };
    let base = if top { 0 } else { 1 };
    let fields = cols
        .iter()
        .enumerate()
        .map(|(k, c)| {
            let bg = point_bg(idx[k], hl);
            match c.cells[depth] {
                Some((ch, color)) => styled_cell(ch, color, bg),
                None => {
                    let light = (k + base) % 2 == 0;
                    let band = if light { BAND_LIGHT } else { BAND_DARK };
                    styled_cell(glyph, band, bg)
                }
            }
        })
        .collect();
    fields_to_line(fields)
}

/// Le trait horizontal qui sépare les deux moitiés, croisé par la barre.
fn mid_rule() -> Line<'static> {
    Line::from(vec![
        Span::styled("─".repeat(24), dim()),
        Span::styled("╋", fg(BAR_COLOR)),
        Span::styled("─".repeat(24), dim()),
    ])
}

/// Rend le plateau (vu des Blancs) en lignes colorées, avec les surbrillances.
fn board_lines(b: &Board, hl: Option<&Hl>) -> Vec<Line<'static>> {
    let pts = b.points();
    let top: Vec<Col> = TOP_IDX.iter().map(|&i| make_col(pts[i])).collect();
    let bot: Vec<Col> = BOT_IDX.iter().map(|&i| make_col(pts[i])).collect();

    let mut lines = Vec::new();
    lines.push(label_line(&TOP_IDX, hl));
    for d in 0..H {
        lines.push(cells_line(&top, &TOP_IDX, d, true, hl));
    }
    lines.push(mid_rule());
    for d in (0..H).rev() {
        lines.push(cells_line(&bot, &BOT_IDX, d, false, hl));
    }
    lines.push(label_line(&BOT_IDX, hl));
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

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    #[test]
    fn le_rendu_ne_panique_pas_et_contient_le_plateau() {
        let board = Board::starting_position();
        let roll = Roll { d1: 3, d2: 1 };

        let screen = Screen {
            view: &board,
            to_move: Player::White,
            roll,
            opp: "heuristique",
            title: "Ton coup",
            panel: vec![Line::from("coups")],
            help: "aide",
            hl: None,
        };

        let mut terminal = Terminal::new(TestBackend::new(80, 24)).unwrap();
        terminal.draw(|f| render(f, &screen)).unwrap();

        let screen = format!("{}", terminal.backend());
        assert!(
            screen.contains("Backgammon"),
            "le titre du plateau doit apparaître"
        );
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

    /// Vérifie qu'en composant le coup dé par dé (via `available_moves`), on
    /// retombe **exactement** sur l'ensemble des positions de `legal_plays`.
    /// C'est l'invariant central de la saisie interactive.
    fn collect_finals(cur: &Board, dice_left: &[u8], finals: &[Board], out: &mut Vec<Board>) {
        if dice_left.is_empty() {
            if !out.contains(cur) {
                out.push(cur.clone());
            }
            return;
        }
        for m in available_moves(cur, dice_left, finals) {
            let rem = without_one(dice_left, m.die);
            collect_finals(&m.next, &rem, finals, out);
        }
    }

    fn assert_saisie_couvre_tout(board: &Board, roll: Roll) {
        let finals = legal_plays(board, &roll);
        let dice_left = maximal_dice(board, &roll);
        let mut reached = Vec::new();
        collect_finals(board, &dice_left, &finals, &mut reached);

        for f in &finals {
            assert!(
                reached.contains(f),
                "une position légale n'est pas atteignable pion par pion (roll {:?})",
                roll
            );
        }
        for r in &reached {
            assert!(
                finals.contains(r),
                "la saisie atteint une position illégale (roll {:?})",
                roll
            );
        }
    }

    #[test]
    fn saisie_incrementale_couvre_legal_plays() {
        let start = Board::starting_position();
        for roll in [
            Roll { d1: 3, d2: 1 },
            Roll { d1: 6, d2: 5 },
            Roll { d1: 2, d2: 4 },
            Roll { d1: 5, d2: 5 }, // double
            Roll { d1: 6, d2: 6 }, // double bloqué au départ
            Roll { d1: 1, d2: 1 },
        ] {
            assert_saisie_couvre_tout(&start, roll);
        }
    }
}
