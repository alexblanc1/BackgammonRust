//! Agent piloté par un humain au clavier.

use std::io::{self, Write};

use crate::agent::Agent;
use crate::board::Board;
use crate::game::GameState;

/// Joueur humain : affiche le plateau et les coups possibles, lit l'indice
/// du coup choisi au clavier.
pub struct HumanAgent;

impl HumanAgent {
    pub fn new() -> HumanAgent {
        HumanAgent
    }
}

impl Agent for HumanAgent {
    fn choose_play(&mut self, state: &GameState, legal: &[Board]) -> usize {
        println!("\n{}", render(&state.board));
        println!(
            "À toi de jouer ({:?}) — tu déplaces les X des grands index vers 0.",
            state.to_move
        );
        println!("{} coup(s) possible(s) :", legal.len());
        for (i, cand) in legal.iter().enumerate() {
            println!("  [{i}] {}", describe_move(&state.board, cand));
        }

        // Boucle de saisie : on insiste jusqu'à obtenir un indice valide.
        loop {
            print!("Ton choix (0..{}) : ", legal.len() - 1);
            // `print!` n'ajoute pas de retour à la ligne et n'affiche pas tout
            // de suite : il faut « vider » la sortie pour voir l'invite.
            io::stdout().flush().ok();

            let mut line = String::new();
            match io::stdin().read_line(&mut line) {
                // 0 octet lu = fin de l'entrée (Ctrl-D / flux épuisé) : on prend
                // le premier coup par défaut pour ne pas boucler à l'infini.
                Ok(0) => return 0,
                Ok(_) => {}
                Err(_) => continue,
            }

            // `.trim()` enlève les espaces et le retour à la ligne ; `.parse()`
            // tente de convertir le texte en `usize`. Le résultat est un
            // `Result`, qu'on filtre : on n'accepte qu'un nombre dans les bornes.
            match line.trim().parse::<usize>() {
                Ok(i) if i < legal.len() => return i,
                _ => println!("Entrée invalide, réessaie."),
            }
        }
    }
}

// --- Affichage ---------------------------------------------------------------

/// Hauteur d'affichage d'une pile de pions.
const H: usize = 5;

/// Construit la colonne d'un point : `H` caractères, l'indice 0 étant la
/// cellule la plus proche du bord du plateau. `X` = tes pions, `O` = adverse.
/// Si la pile dépasse `H`, la dernière cellule affiche le compte.
fn column(v: i8) -> [char; H] {
    let (sym, n) = if v > 0 {
        ('X', v as usize)
    } else if v < 0 {
        ('O', (-v) as usize)
    } else {
        (' ', 0)
    };

    let mut col = [' '; H];
    let shown = n.min(H);
    for cell in col.iter_mut().take(shown) {
        *cell = sym;
    }
    if n > H {
        col[H - 1] = std::char::from_digit(n as u32, 10).unwrap_or('+');
    }
    col
}

/// Assemble 12 champs de 2 caractères en une ligne, avec la barre au milieu.
fn join_fields(fields: &[String]) -> String {
    let mut s = String::new();
    for (k, f) in fields.iter().enumerate() {
        if k == 6 {
            s.push_str(" |");
        }
        if k != 0 {
            s.push(' ');
        }
        s.push_str(f);
    }
    s
}

/// Rend le plateau en texte, du point de vue du joueur à jouer.
fn render(b: &Board) -> String {
    // Ordre d'affichage : en haut les index 23→12, en bas 0→11.
    let top: [usize; 12] = [23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12];
    let bot: [usize; 12] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];

    let top_cols: Vec<[char; H]> = top.iter().map(|&i| column(b.points[i])).collect();
    let bot_cols: Vec<[char; H]> = bot.iter().map(|&i| column(b.points[i])).collect();

    let mut out = String::new();

    // Étiquettes du haut.
    let labels: Vec<String> = top.iter().map(|i| format!("{i:>2}")).collect();
    out.push_str(&join_fields(&labels));
    out.push('\n');

    // Pile du haut : la base est en haut, donc on descend de la cellule 0 à H-1.
    for depth in 0..H {
        let fields: Vec<String> = top_cols.iter().map(|c| format!(" {}", c[depth])).collect();
        out.push_str(&join_fields(&fields));
        out.push('\n');
    }

    out.push_str("  ──────────────────  |  ──────────────────\n");

    // Pile du bas : la base est en bas, donc on remonte de la cellule H-1 à 0.
    for depth in (0..H).rev() {
        let fields: Vec<String> = bot_cols.iter().map(|c| format!(" {}", c[depth])).collect();
        out.push_str(&join_fields(&fields));
        out.push('\n');
    }

    // Étiquettes du bas.
    let labels: Vec<String> = bot.iter().map(|i| format!("{i:>2}")).collect();
    out.push_str(&join_fields(&labels));
    out.push('\n');

    out.push_str(&format!(
        "Barre — toi: {} | adverse: {}    Sortis — toi: {} | adverse: {}",
        b.bar[0], b.bar[1], b.off[0], b.off[1]
    ));
    out
}

/// Nombre de tes pions sur le point `p` (0 si la case est vide ou adverse).
fn mover_at(b: &Board, p: usize) -> i32 {
    if b.points[p] > 0 { b.points[p] as i32 } else { 0 }
}

/// Décrit le coup menant de `cur` à `cand` par différence : d'où partent tes
/// pions, où ils arrivent, et les frappes éventuelles. Robuste (basé sur le
/// bilan net par case), même pour les doubles.
fn describe_move(cur: &Board, cand: &Board) -> String {
    let mut sources: Vec<String> = Vec::new();
    let mut dests: Vec<String> = Vec::new();

    // Pions entrés depuis la barre.
    let bar_in = cur.bar[0] as i32 - cand.bar[0] as i32;
    for _ in 0..bar_in.max(0) {
        sources.push("barre".to_string());
    }

    // Départs et arrivées sur le plateau (parcours des grands index vers 0).
    for p in (0..24).rev() {
        let d = mover_at(cand, p) - mover_at(cur, p);
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

    // Pions sortis.
    let off_out = cand.off[0] as i32 - cur.off[0] as i32;
    for _ in 0..off_out.max(0) {
        dests.push("sortie".to_string());
    }

    let mut s = format!("{} → {}", sources.join(", "), dests.join(", "));

    // Frappes : un pion adverse seul (-1) remplacé par un des tiens.
    if cand.bar[1] > cur.bar[1] {
        let hits: Vec<String> = (0..24)
            .rev()
            .filter(|&p| cur.points[p] == -1 && cand.points[p] > 0)
            .map(|p| p.to_string())
            .collect();
        if !hits.is_empty() {
            s.push_str(&format!("  (frappe en {})", hits.join(", ")));
        }
    }
    s
}
