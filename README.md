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
  hits), end-of-game detection (single / *gammon* / *backgammon*), and full
  **doubling-cube** support.
- **A terminal UI** (TUI, built with [Ratatui](https://ratatui.rs)): a menu, a
  board drawn with its wooden points, dice shown on screen, and move entry done
  **one checker at a time from the keyboard**, with the legal sources and
  destinations highlighted. Pick the **difficulty level**, propose or answer a
  **double**, and watch a running **win/loss scoreboard** that persists between
  sessions.
- **A dice-fairness tool**: a built-in **χ² goodness-of-fit test** that checks
  the dice really are uniform (the RNG is the audited [`rand`](https://docs.rs/rand)
  crate, not a hand-rolled generator).
- **A six-tier AI**, from simplest to smartest:
  1. a **random** agent (training sparring partner and test harness);
  2. a **heuristic** agent (a weighted sum of positional features) that beats
     random play ~99% of the time;
  3. a **neural network** trained by self-play with **TD(λ)**, which beats the
     heuristic;
  4. **expectiminimax** with chance nodes (1 then 2 plies of look-ahead);
  5. **Monte-Carlo rollouts** that decide close calls by playing games out.
- **Weight persistence**: train the network once, save it, and the TUI plays
  against it.
- **45 automated tests** (`cargo test --workspace`) covering the rules, the
  doubling cube, the evaluation, the backpropagation, the training loop, the
  search and the dice statistics.

---

## 🚀 Quick start

Requirements: [Rust](https://rustup.rs) (2024 edition, tested with `rustc 1.95`).

### Play a game

```bash
cargo run            # debug build (fine up to the "Expert" level)
cargo run --release  # recommended for the "Maître" / "Légende" levels
```

A **menu** lets you choose the difficulty level (`←/→`), open the dice
statistics screen, or start playing. You play **White** (cyan checkers) against
the AI (red checkers). If no network has been trained yet, the network-based
levels fall back to the heuristic.

#### Keyboard controls

| Key                     | Action                                                       |
| ----------------------- | ------------------------------------------------------------ |
| `h` `j` `k` `l` / arrows | Move the cursor around the board                            |
| `Enter`                 | Pick up the checker under the cursor, then choose its target |
| `1`–`6`                 | Play a die of that exact value directly (handy for bearing off) |
| `d`                     | Propose a **double** (before rolling, when you may)         |
| `a` / `r`               | **Accept** / **refuse** a double the AI offers you          |
| `u`                     | Undo the last sub-move                                       |
| `r`                     | Restart the move from scratch                               |
| `Esc` / `q`             | Quit                                                         |

The orientation matches a real board: **your home (inner) board is bottom-right**,
and you move your checkers toward it to bear them off.

### Train the network

```bash
# 5,000 self-play games, hidden layer of 40 neurons (defaults)
cargo run --release --bin train -- 5000

# then play again: the TUI loads the trained network automatically
cargo run --release
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
cargo test --workspace          # the fast suite (45 tests)
cargo test --workspace -- --ignored   # + the real training test (~45 s)
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
                                          │
                              ┌───────────┴───────────┐
                              ▼                       ▼
                     Expectiminimax            Monte-Carlo
                     (chance nodes)              rollouts
```

The key abstraction is the **`Evaluator`** trait: "know how to score a position."
A generic **`GreedyAgent<E>`** plays the `argmax` of any evaluator; the search
agents (`ExpectiAgent`, `RolloutAgent`) wrap the same evaluator to look deeper.
The heuristic **and** the network thus reuse the exact same choice logic.

**1. The heuristic** (`ai/heuristic.rs`) — a differential weighted sum: lead in
the race (*pip count*), checkers borne off, opponent checkers on the bar, *blots*
exposed to a hit, made points in the inner board, overall structure. It's
**antisymmetric** (`evaluate(swap(b)) == -evaluate(b)`) and beats random play
~99.5% of the time.

**2. The encoding** (`ai/encoding.rs`) — each position becomes a vector of **196
inputs**, in the spirit of Tesauro's encoding: for each point and each player, 4
units describe the checker count (≥1, ≥2, ≥3, and the surplus), plus the bar and
the borne-off checkers. (The original uses 198; the 2 "whose turn" units are
useless here since the board is already normalized.)

**3. The network** (`ai/net.rs`) — a hand-coded **single-hidden-layer
perceptron**: `196 inputs → sigmoid hidden layer → 6 sigmoid outputs`. The six
outputs are the probabilities of each game outcome **from the player-to-move's
point of view**: win / gammon / backgammon, and the symmetric three losses. The
position's **equity** — its expected number of points, in `[-3, +3]` — is what
the agent maximizes, so it tells "winning" apart from "winning big." It's all
there: forward pass, backpropagation, gradient step, and lossless save/load.

**4. TD(λ) training** (`ai/train.rs`) — the network plays against itself. After
every move, we nudge the previous prediction toward its target (the value of the
next move, then the actual one-hot outcome at game's end), using **eligibility
traces** that spread the signal back to earlier moves. This is pure reinforcement
learning: no human games, no database — just play against yourself.

> 💡 **The most instructive bug in the project.** The network's outputs depend
> on *who* is moving. Accumulating the gradient naively into the traces mixed
> *White's* and *Black's* outcomes — opposite quantities — and training
> **diverged**. The fix: express everything in a single **canonical** outcome
> vector seen from White, and **permute the gradients by the current
> perspective** (a player's wins are the opponent's losses). It's documented in
> detail in `train.rs`.

**5. Look-ahead search** (`ai/search.rs`) — at play time the agent can look
deeper than the greedy policy:

- **Expectiminimax** adds **chance nodes**: it can't know the opponent's roll, so
  it averages the value of the opponent's best reply over all 21 dice rolls,
  weighted by probability (1/36 for a double, 2/36 otherwise). `plies = 1` looks
  at the reply, `plies = 2` adds our counter-reply. Each extra ply costs ~21×, so
  only the top few candidates (by static evaluation) are explored deeply.
- **Monte-Carlo rollouts** decide close calls by **actually playing** the rest of
  the game many times (both sides greedy, random dice) and averaging the points —
  an unbiased estimate of the true value.

**6. The doubling cube** (`engine/game.rs`) — the full cube is modelled: who owns
it, the current stake, and the legal right to double. The cube-aware agents use a
simple equity rule (double when you're a 65–95% favorite; take down to ~25%),
and the final score is multiplied by the cube. The TUI lets you propose and
answer doubles, and tallies the resulting points.

**Results** (hidden layer of 40 neurons, α = 0.1, λ = 0, a few thousand games):
**~100% wins against random** and **~60–69% against the heuristic**. So the
network surpasses the baseline it was meant to beat. The learning curve is typical
of self-play TD: a fast rise, a small transient dip around 2,000–4,000 games, then
a strong recovery.

---

## 🗂️ Code layout

A Cargo **workspace** split into three crates: the dependency-free **engine**,
the **ai** (which depends only on the engine), and the **cli** front-end (which
depends on both).

```
crates/
├── engine/   # the game, no display & no AI dependencies
│   ├── board.rs      # board, starting position, swap_perspective, single-die moves, win detection
│   ├── dice.rs       # the dice (rand-crate RNG), a roll, the 21 weighted rolls
│   ├── moves.rs      # legal_plays: dice composition and the mandatory-use rules
│   ├── player.rs     # the Player enum (White / Black)
│   ├── game.rs       # game state, the play loop, and the doubling Cube
│   ├── stats.rs      # the χ² dice-fairness test
│   ├── agent.rs      # the Agent trait (+ doubling hooks)
│   └── agent/        # random.rs, human.rs (console)
├── ai/       # learning & search; depends on engine
│   ├── encoding.rs   # a position → 196 inputs (Tesauro-style)
│   ├── eval.rs       # the Evaluator trait, GreedyAgent<E>, cube heuristics
│   ├── heuristic.rs  # the heuristic evaluation + agent
│   ├── net.rs        # the 6-output neural network: forward, backprop, save/load
│   ├── search.rs     # ExpectiAgent (expectiminimax) + RolloutAgent (Monte-Carlo)
│   └── train.rs      # the self-play TD(λ) training and win-rate measurements
└── cli/      # the front-end; depends on engine + ai
    ├── main.rs       # entry point (launches the TUI)
    ├── tui.rs        # the terminal UI: menu, board, dice, cube, dice-stats screen
    ├── scores.rs     # the persistent win/loss scoreboard
    └── bin/train.rs  # the command-line training executable
```

### Tech stack

- **Rust** (2024 edition), no `unsafe`.
- [**Ratatui**](https://ratatui.rs) + [**crossterm**](https://github.com/crossterm-rs/crossterm) for the TUI.
- [**rand**](https://docs.rs/rand) for the dice (a proper, audited RNG).
- Neural network, backprop and TD(λ) **coded by hand** — no `tch`, `burn` or
  `candle` (to be reconsidered if the network grows).

---

## 🛣️ What's next

- [x] **Expectiminimax** with chance nodes + Monte-Carlo **rollouts** at play time.
- [x] **Multiple** network outputs: win / gammon / backgammon probabilities.
- [x] **Doubling cube** support.
- [x] Migration to a multi-crate **workspace** (engine / ai / cli).
- [ ] A graphical front-end (`gui` crate).
- [ ] Cube-aware **rollouts** and a learned doubling policy.
- [ ] A larger / deeper network, possibly via `tch` or `burn`.

---

## 🤖 About

This project — **engine, AI, interface and tests included** — was **entirely
built by Claude** (Anthropic's AI assistant), pairing with Alex: architecture
design, every line of code, getting the reinforcement learning to converge, and
writing this README. The goal was as much to build a real backgammon engine with
an AI that learns as it was to make it an educational walkthrough of Rust and
end-to-end reinforcement learning.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
