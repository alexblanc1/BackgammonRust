# 🎲 Backgammon in Rust — engine + TD-Gammon-style AI

A **backgammon** engine written from scratch in Rust, playable from the keyboard
in your terminal, paired with an **AI that learns to play entirely on its own**
through *self-play* — the same idea behind the famous **TD-Gammon** by Gerald
Tesauro (1992), the neural network that learned backgammon at world-champion
level just by playing against itself.

Everything is coded **by hand**, with no machine-learning library: the neural
network, the backpropagation and the temporal-difference learning TD(λ) all fit
in a few hundred lines of readable Rust.

```
        ┌─────────────────────────────────────────────────┐
        │  Human (●  cyan)   vs   trained AI (●  red)       │
        └─────────────────────────────────────────────────┘
           You move your checkers from the keyboard, one at
            a time; the network picks its moves by scoring
                        every possible position.
```

---

## ✨ What this project does

- **A complete, correct backgammon engine**: full legal-move generation (bar
  re-entry has priority, doubles, the "play the larger die" rule, bearing off,
  hits), and end-of-game detection (single / *gammon* / *backgammon*).
- **A terminal UI** (TUI, built with [Ratatui](https://ratatui.rs)): a board
  drawn with its wooden points, dice shown on screen, and move entry done
  **one checker at a time from the keyboard**, with the legal sources and
  destinations highlighted.
- **A three-tier AI**, from simplest to smartest:
  1. a **random** agent (training sparring partner and test harness);
  2. a **heuristic** agent (a weighted sum of positional features) that beats
     random play ~99% of the time;
  3. a **neural network** trained by self-play with **TD(λ)**, which beats the
     heuristic.
- **Weight persistence**: train the network once, save it, and the TUI plays
  against it.
- **28 automated tests** (`cargo test`) covering the rules, the evaluation, the
  backpropagation and the training loop.

---

## 🚀 Quick start

Requirements: [Rust](https://rustup.rs) (2024 edition, tested with `rustc 1.95`).

### Play a game

```bash
cargo run
```

You play **White** (cyan checkers) against the AI (red checkers). If no network
has been trained yet, the AI uses the heuristic; otherwise it loads the trained
network from `net.txt`.

#### Keyboard controls

| Key                     | Action                                                       |
| ----------------------- | ------------------------------------------------------------ |
| `h` `j` `k` `l` / arrows | Move the cursor around the board                            |
| `Enter`                 | Pick up the checker under the cursor, then choose its target |
| `1`–`6`                 | Play a die of that exact value directly (handy for bearing off) |
| `u`                     | Undo the last sub-move                                       |
| `r`                     | Restart the move from scratch                                |
| `Esc` / `q`             | Quit                                                         |

The orientation matches a real board: **your home (inner) board is bottom-right**,
and you move your checkers toward it to bear them off.

### Train the network

```bash
# 5,000 self-play games, hidden layer of 40 neurons (defaults)
cargo run --release --bin train -- 5000

# then play again: the TUI loads the trained network automatically
cargo run
```

The tool prints progress (win rate vs random and vs heuristic) as training goes,
then saves the weights to `net.txt`.

```
TD(λ) training: 5000 games, hidden=40, alpha=0.1, lambda=0

  games  | vs random | vs heuristic
  -------+-----------+--------------
     625 |    92.0%  |      41.5%
    1250 |    96.5%  |      48.0%
      …  |      …    |        …
    5000 |    99.5%  |      66.0%
```

Arguments: `train -- [games] [hidden_neurons] [alpha] [lambda]`.

### Run the tests

```bash
cargo test                 # the fast suite (23 tests + 4 TUI tests)
cargo test -- --ignored    # + the real training test (~45 s)
```

---

## 🧠 How it works

### The board, seen "from the player to move"

The core trick of the engine: the board is **always normalized from the point of
view of whoever is on move**.

- `points[i] > 0` → your checkers; `points[i] < 0` → the opponent's.
- You move **from index 23 toward index 0**; your inner board (where you bear off)
  is on indices `0..6`.
- At the end of each turn we call `swap_perspective()`, which flips the board and
  inverts the signs.

The payoff: **all the game logic is written for a single direction**. No need to
duplicate every rule for White and Black — there is only ever one "player to
move," and it's always you, seen from your side.

### Move generation

A single primitive, `single_die_moves(die)`, computes every position reachable by
playing **one die** (re-entering from the bar, a plain move, hitting a lone
*blot*, or bearing off). On top of it, `legal_plays(board, roll)` composes the two
dice (or the four moves of a double) via a small recursive search, and enforces
backgammon's mandatory-use rules:

- the **bar has priority**;
- you must play the **maximum number of dice possible**;
- on a non-double you can't play in full, you favor the **larger die**;
- identical positions are **deduplicated**.

Agents never receive "move sequences": they receive the **list of resulting
positions** and return the index of the one they choose. That's what makes the AI
so easy to plug in.

### The AI, step by step (the TD-Gammon roadmap)

```
RandomAgent  →  HeuristicAgent  →  Neural network (untrained)
                                          │
                                          ▼
                              Self-play TD(λ) training
                                          │
                                          ▼
                            Network that beats the heuristic
```

The key abstraction is the **`Evaluator`** trait: "know how to score a position."
A generic **`GreedyAgent<E>`** plays the `argmax` of any evaluator. The heuristic
**and** the network thus reuse the exact same choice logic.

**1. The heuristic** (`agent/heuristic.rs`) — a differential weighted sum: lead in
the race (*pip count*), checkers borne off, opponent checkers on the bar, *blots*
exposed to a hit, made points in the inner board, overall structure. It's
**antisymmetric** (`evaluate(swap(b)) == -evaluate(b)`) and beats random play
~99.5% of the time.

**2. The encoding** (`encoding.rs`) — each position becomes a vector of **196
inputs**, in the spirit of Tesauro's encoding: for each point and each player, 4
units describe the checker count (≥1, ≥2, ≥3, and the surplus), plus the bar and
the borne-off checkers. (The original uses 198; the 2 "whose turn" units are
useless here since the board is already normalized.)

**3. The network** (`net.rs`) — a hand-coded **single-hidden-layer perceptron**:
`196 inputs → sigmoid hidden layer → 1 sigmoid output`. The output is the
**estimated probability that the player to move wins**. It's all there: forward
pass, backpropagation (`output_gradient`), gradient step, and lossless
save/load of the weights.

**4. TD(λ) training** (`train.rs`) — the network plays against itself. After every
move, we nudge the previous prediction toward its target (the value of the next
move, then the actual outcome at game's end), using **eligibility traces** that
spread the signal back to earlier moves. This is pure reinforcement learning: no
human games, no database — just play against yourself.

> 💡 **The most instructive bug in the project.** The network's value depends on
> *who* is moving. Accumulating the gradient naively into the trace mixed
> *P(White wins)* and *P(Black wins)* — two opposite quantities — and training
> **diverged** (the network collapsed back below 50% vs random). The fix:
> express everything in a single **canonical value** `U = P(White wins)`, and
> **sign the gradient by the current perspective**. It's documented in detail in
> `train.rs`.

**Results** (hidden layer of 40 neurons, α = 0.1, λ = 0, a few thousand games):
**~100% wins against random** and **~65–69% against the heuristic**. So the
network surpasses the baseline it was meant to beat. The learning curve is typical
of self-play TD: a fast rise, a small transient dip around 2,000–4,000 games, then
a strong recovery.

---

## 🗂️ Code layout

A single crate, split into clear modules (a migration to a multi-crate
*workspace* is planned as the AI grows).

| File                    | Role                                                                |
| ----------------------- | ------------------------------------------------------------------- |
| `board.rs`              | The board, the starting position, `swap_perspective`, single-die moves, win detection |
| `dice.rs`               | The dice (dependency-free *xorshift* generator) and a roll          |
| `moves.rs`              | `legal_plays`: dice composition and the mandatory-use rules         |
| `player.rs`             | The `Player` enum (White / Black)                                   |
| `game.rs`               | A game's state and the play loop between two agents                 |
| `agent.rs`              | The `Agent` trait (human, random, AI…)                             |
| `agent/random.rs`       | Random agent                                                        |
| `agent/heuristic.rs`    | Heuristic evaluation + agent                                        |
| `agent/human.rs`        | Command-line human agent (+ ASCII board rendering)                  |
| `eval.rs`               | The `Evaluator` trait and the generic `GreedyAgent<E>`              |
| `encoding.rs`           | Encoding a position into 196 inputs (Tesauro-style)                |
| `net.rs`                | The neural network: forward, backprop, save/load                    |
| `train.rs`              | The self-play TD(λ) training and the win-rate measurements          |
| `bin/train.rs`          | The command-line training executable                                |
| `tui.rs`                | The terminal UI (Ratatui)                                           |

### Tech stack

- **Rust** (2024 edition), no `unsafe`.
- [**Ratatui**](https://ratatui.rs) + [**crossterm**](https://github.com/crossterm-rs/crossterm) for the TUI.
- Neural network, backprop and TD(λ) **coded by hand** — no `tch`, `burn` or
  `candle` (to be reconsidered if the network grows).

---

## 🛣️ What's next

- [ ] **Expectiminimax** with chance nodes + Monte-Carlo **rollouts** at play
      time (look one move deeper than the current greedy policy).
- [ ] **Multiple** network outputs: separate win / gammon / backgammon
      probabilities for each side.
- [ ] **Doubling cube** support.
- [ ] Migration to a multi-crate **workspace** (engine / cli / ai / gui).

---

## 🤖 About

This project — **engine, AI, interface and tests included** — was **entirely
built by Claude** (Anthropic's AI assistant), pairing with Alex: architecture
design, every line of code, getting the reinforcement learning to converge, and
writing this README. The goal was as much to build a real backgammon engine with
an AI that learns as it was to make it an educational walkthrough of Rust and
end-to-end reinforcement learning.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
