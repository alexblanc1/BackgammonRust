# Project context for Claude

This file is auto-loaded by Claude Code at the start of each session. It carries
the project context across machines (e.g. when continuing on a different
computer). Keep it up to date as the project evolves.

---

## What this project is

A **backgammon engine + self-play AI** written from scratch in Rust, in the style
of **TD-Gammon** (Tesauro). A Cargo **workspace** of three crates:
`engine` (game, no deps on AI/display), `ai` (learning + search, depends on
engine), `cli` (TUI + training binary, depends on both). A `gui` crate may come
later.

The game is **complete and playable**: no `todo!()` left. You can play it in a
terminal UI (with a difficulty menu, a doubling cube and a persistent
scoreboard), and the neural network can be trained by self-play and then played
against.

See `README.md` for the public-facing overview.

## Build / run / test

```bash
cargo run                                  # play the TUI (menu → pick a level)
cargo run --release                        # recommended for Maître / Légende levels
cargo run --release --bin train -- 5000    # train the net by self-play, save to net.txt
cargo test --workspace                     # fast suite (currently 45 tests, 1 ignored)
cargo test --workspace -- --ignored        # + the real training test (~45 s)
```

The network-based levels load `net.txt` if present, otherwise fall back to the
heuristic. `net.txt` and `scores.txt` are git-ignored.

## Architecture & key design decisions

- **Workspace layout.** Crates import each other by short names: `engine::Board`,
  `ai::Net`, etc. (set via `[lib] name = …`). The engine exposes `Board::from_parts`
  so the other crates can build arbitrary positions (the fields stay private).
- **Board normalized from the point-of-view of the player to move.** In
  `engine/board.rs`: `points[i] > 0` = your checkers, `< 0` = opponent; you
  advance from index 23 → 0; your inner board is `0..6`. `swap_perspective()`
  flips the board at the end of each turn, so **all game logic is written for a
  single direction.**
- **Agents receive resulting positions, not move sequences.** The `Agent` trait
  (`engine/agent.rs`) gets `&[Board]` (the legal resulting positions from
  `moves::legal_plays`) and returns an index. It also has `should_double` /
  `should_accept_double` hooks (default: never double, always accept).
- **`Evaluator` + `GreedyAgent<E>`** (`ai/eval.rs`): "score a position" is
  separated from "pick the best move." The heuristic, the network, and the search
  agents all reuse the same argmax logic. `Evaluator` also has `terminal_value`
  (score of a won position) and `win_prob` (for cube decisions).
- **Search agents** (`ai/search.rs`): `ExpectiAgent` (expectiminimax with chance
  nodes, `plies` deep, only the `top_k` static-best candidates explored) and
  `RolloutAgent` (Monte-Carlo). They wrap any `Evaluator`.
- **Dice = the `rand` crate** (`engine/dice.rs`). `Dice::new(seed)` is
  deterministic (tests/training); `Dice::random()` is OS-seeded (real games).
  `all_rolls()` gives the 21 weighted rolls for the chance nodes. `engine/stats.rs`
  has the χ² dice-fairness test surfaced in the TUI.
- **CLI-only state lives in the cli crate**: `tui.rs` (menu, board, cube UI,
  dice-stats screen), `scores.rs` (persistent W/L scoreboard). Keeps the engine
  free of display/persistence concerns. Ratatui 0.30 + crossterm.

## AI roadmap & status (TD-Gammon)

1. ✅ Heuristic baseline — `ai/heuristic.rs`: weighted differential sum (pip
   count, off, bar, exposed blots, home points, made points). Antisymmetric.
   Beats random ~99.5%.
2. ✅ Network — `ai/encoding.rs` (`N_INPUTS = 196`, Tesauro-style) + `ai/net.rs`.
   Single hidden layer, **6 sigmoid outputs** (`N_OUTPUTS`): win/gammon/backgammon
   and the three symmetric losses, from the player-to-move's view. Position value
   = **equity** = `(p_w + 2·p_g + 3·p_bg) − (losses)`, in `[-3,+3]`. Hand-coded
   forward/backprop (one gradient per output), lossless save/load (format marker
   `bgnet2`; old 1-output files are rejected with a clear error).
3. ✅ Self-play TD(λ) training — `ai/train.rs` + `cli/bin/train.rs`. Key subtlety:
   work in a **canonical outcome vector seen from White**, and **permute the
   per-output gradients by perspective** (`canon`: a player's win indices are the
   opponent's loss indices). Without it, training mixes White's and Black's
   outcomes and diverges. `λ=0` (TD(0)) is the stable default; `λ>0` amplifies the
   effective step (~×1/(1−λ)), so lower `alpha`. Results (hidden=40, α=0.1, λ=0,
   ~10k games): ~100% vs random, ~60–69% vs heuristic.
4. ✅ Play-time look-ahead — expectiminimax (chance nodes) + Monte-Carlo rollouts
   in `ai/search.rs`; wired to the TUI's Expert / Maître / Légende levels.
5. ✅ Doubling cube — `engine/game.rs` (`Cube`), cube heuristics in `ai/eval.rs`,
   UI in the TUI.
6. ⬅️ **Next ideas**: a `gui` crate; cube-aware rollouts / learned doubling
   policy; a bigger network (maybe `tch`/`burn`/`candle`).

## Working with Alex

- **Converse in French.** Alex is a **Rust beginner** building this to learn both
  Rust and end-to-end reinforcement learning. Explain new Rust concepts the first
  time they appear, like to a beginner.
- **Write/modify the files directly** (Edit/Write), don't ask Alex to copy-paste
  code by hand. Then briefly explain what changed and why.
- **Repo docs (README, this file) are written in English**; the conversation and
  pedagogical explanations stay in French.
- Verify changes with `cargo build` / `cargo test --workspace`.
