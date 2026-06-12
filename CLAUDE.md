# Project context for Claude

This file is auto-loaded by Claude Code at the start of each session. It carries
the project context across machines (e.g. when continuing on a different
computer). Keep it up to date as the project evolves.

---

## What this project is

A **backgammon engine + self-play AI** written from scratch in Rust, in the style
of **TD-Gammon** (Tesauro). A single crate split into modules; a move to a
multi-crate workspace (engine / cli / ai / gui) is planned once the AI grows.

The game is **complete and playable**: no `todo!()` left. You can play it in a
terminal UI, and the neural network can be trained by self-play and then played
against.

See `README.md` for the public-facing overview.

## Build / run / test

```bash
cargo run                                  # play the TUI (you = White, vs the AI)
cargo run --release --bin train -- 5000    # train the net by self-play, save to net.txt
cargo test                                 # fast suite (currently 27 tests run, 1 ignored)
cargo test -- --ignored                    # + the real training test (~45 s)
```

The TUI loads `net.txt` if present (plays the trained network), otherwise falls
back to the heuristic. `net.txt` is git-ignored.

## Architecture & key design decisions

- **Board normalized from the point-of-view of the player to move.** In
  `board.rs`: `points[i] > 0` = your checkers, `< 0` = opponent; you advance from
  index 23 → 0; your inner board is `0..6`. `swap_perspective()` flips the board
  at the end of each turn, so **all game logic is written for a single direction.**
- **Agents receive resulting positions, not move sequences.** The `Agent` trait
  (`agent.rs`) gets `&[Board]` (the legal resulting positions from
  `moves::legal_plays`) and returns an index.
- **`Evaluator` + `GreedyAgent<E>`** (`eval.rs`): "score a position" is separated
  from "pick the best move." The heuristic, the network, and (later) the base of
  expectiminimax all reuse the same argmax logic.
- **TUI lives in the binary** (`tui.rs`, `main.rs`), not the library, to keep the
  engine free of display dependencies. Built with Ratatui 0.30 + crossterm.
  Board shown from White's side; move entry is checker-by-checker from the
  keyboard.

## AI roadmap & status (TD-Gammon)

1. ✅ Heuristic baseline — `agent/heuristic.rs`: weighted differential sum (pip
   count, off, bar, exposed blots, home points, made points). Antisymmetric.
   Beats random ~99.5%.
2. ✅ Network — `encoding.rs` (`N_INPUTS = 196`, Tesauro-style) + `net.rs`
   (single hidden layer, sigmoid → 1 sigmoid output = P(player to move wins),
   hand-coded forward/backprop, lossless save/load).
3. ✅ Self-play TD(λ) training — `train.rs` + `bin/train.rs`. Key subtlety: use a
   single **canonical value** `U = P(White wins)` and **sign the eligibility-trace
   gradient by perspective** (`+∇p` for White, `−∇p` for Black). Without the sign,
   training diverges (it mixes P(White) and P(Black)). `λ=0` (TD(0)) is the stable
   default; `λ>0` amplifies the effective step (~×1/(1−λ)), so lower `alpha`.
   Results (hidden=40, α=0.1, λ=0, a few thousand games): ~100% vs random,
   ~65–69% vs heuristic.
4. ⬅️ **Next**: play-time look-ahead — **expectiminimax** with chance nodes +
   Monte-Carlo **rollouts**. Also possible: multi-output net (win/gammon/
   backgammon probabilities), doubling-cube support, workspace split.

ML crates to reconsider if the net grows: burn, candle, or tch (libtorch).

## Working with Alex

- **Converse in French.** Alex is a **Rust beginner** building this to learn both
  Rust and end-to-end reinforcement learning. Explain new Rust concepts the first
  time they appear, like to a beginner.
- **Write/modify the files directly** (Edit/Write), don't ask Alex to copy-paste
  code by hand. Then briefly explain what changed and why.
- **Repo docs (README, this file) are written in English**; the conversation and
  pedagogical explanations stay in French.
- Verify changes with `cargo build` / `cargo test`.
