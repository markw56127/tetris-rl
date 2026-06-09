# STATS 404 Final Project — Deep Reinforcement Learning for Tetris

**Author:** Mark Wang

This project trains agents to play Tetris from scratch using deep reinforcement
learning (RL). I implemented a custom, dependency-free Tetris engine and two
different learning agents on top of it — a policy-gradient method (Maskable PPO)
and a value-based method (afterstate DQN) — and compared them.

All code is in this repository. See [README.md](README.md) for setup and commands.

---

## 1. What dataset did you use, and what are the features?

Reinforcement learning does **not** use a fixed, pre-collected dataset the way a
supervised model does. Instead, the "data" is *generated on the fly* by the agent
interacting with an environment: the agent takes an action, the environment returns
the next state and a reward, and this stream of `(state, action, reward, next_state)`
transitions is what the model learns from. There is therefore no external dataset to
download — the data source is the Tetris environment I wrote, which lives in
[`tetris/`](tetris/) and is part of the submitted code.

Concretely, the environment generates an effectively unlimited stream of game states
using a **7-bag randomizer** (the standard modern-Tetris piece generator: each of the
7 tetrominoes appears exactly once per "bag" of 7, shuffled — guaranteeing a fair,
non-degenerate piece distribution). The DQN run above consumed ~150,000 such
transitions; PPO consumed ~10,000,000.

**The "features" are the state representation the network sees each step:**

| Feature | Shape | Description |
|---|---|---|
| `board` | 20×10 | Binary occupancy grid of the playfield (1 = filled cell, 0 = empty) |
| `current_piece` | 7 (one-hot) | Which tetromino is active |
| `hold` | 8 (one-hot) | Piece in the hold slot (7 pieces + "empty") |
| `queue` | 5 × 7 | The next 5 upcoming pieces |
| `action_mask` | 81 | Which of the 81 actions are currently legal |

For the **afterstate DQN**, the feature is even simpler: the single 20×10 binary board
that *results* from a candidate placement (after the piece locks and any full lines
clear). The network's only job is to score that resulting board.

---

## 2. What methods did you try, why this final choice, and the underlying principles?

I implemented and trained two methods that share the same environment, so the
comparison is apples-to-apples.

### Method A — Maskable PPO (policy gradient)

**Principle.** PPO (Proximal Policy Optimization) is an *actor–critic* policy-gradient
method. An **actor** network outputs a probability distribution over the 81 actions; a
**critic** network estimates the value (expected future reward) of the current state.
The agent plays, collects rollouts, and computes **advantages** with Generalized
Advantage Estimation (GAE) — a low-variance estimate of how much better an action was
than the critic expected. PPO then nudges the policy toward high-advantage actions
while a **clipped surrogate objective** prevents the policy from changing too fast in a
single update (the source of its stability). I used **action masking** (sb3-contrib's
`MaskablePPO`): illegal placements are assigned probability zero *before* the softmax,
so the agent never wastes capacity learning the rules of the game. The board is encoded
by a small **CNN** feature extractor; piece/hold/queue are embedded with linear layers
and concatenated (~339k parameters). Training used 8 parallel environments for 10M steps.

### Method B — Afterstate DQN (value-based) — *final choice*

**Principle.** DQN learns a value function rather than a policy. The key idea here is
the **afterstate**: instead of learning `Q(state, action)` for all 81 actions, I
enumerate every legal placement of the current piece, simulate it, and evaluate the
*resulting board* with a single-output CNN, `Q(afterstate) → scalar`. The agent simply
picks the placement whose resulting board has the highest predicted value. I trained it
with standard DQN machinery:

- **Experience replay:** transitions are stored in a buffer and sampled in random
  minibatches, breaking temporal correlation between consecutive samples.
- **Target network:** a periodically-frozen copy of the network provides stable
  regression targets (`r + γ·max Q_target(next afterstates)`).
- **Double DQN:** the online network *selects* the best next action and the target
  network *evaluates* it, which reduces the well-known overestimation bias of vanilla
  DQN.
- **ε-greedy exploration:** ε decays from 1.0 → 0.05 early, then trains at low ε.

(~892k parameters.)

### Why I chose the afterstate DQN

The afterstate formulation makes Tetris a dramatically easier learning problem. With
raw PPO, the network must map an observation to one of 81 action indices and *learn the
consequences of each index* — a hard credit-assignment problem with long delays between
a bad placement and the eventual game-over. With afterstates, the legal-move search
handles the combinatorics, and the network only has to answer one intuitive question:
*"how good does this resulting board look?"* This matches the structure of the game and,
empirically, trained far faster and far better (see §5). I therefore selected the
**afterstate DQN** as the final method.

---

## 3. Missing data, processing, and cleaning choices

There is no missing data in the supervised sense (no NaNs, no absent rows), but several
deliberate data-processing choices shaped what the network sees:

- **Board binarization.** The engine internally tracks piece *colors* (1–7) for
  rendering, but the agent receives a **binary occupancy grid** (filled / empty). Color
  identity is irrelevant to good play, so removing it shrinks the input and prevents the
  network from latching onto spurious color patterns.
- **Action masking instead of penalizing illegal moves.** Rather than feeding illegal
  actions to the agent and punishing them, I compute a legality mask each step and
  exclude illegal actions entirely. This is effectively cleaning the *action* space so
  the agent only ever considers valid moves.
- **Reward shaping (PPO).** Raw Tetris reward (line clears) is extremely sparse, so for
  PPO I added shaping terms: a survival bonus, a `lines² × 5` clear bonus, and small
  penalties for added holes, aggregate column height, and bumpiness. These hand-designed
  board-quality features guide early learning without dominating the true objective.
- **Reward design (DQN).** The DQN used a simpler reward: `[0, 1, 3, 5, 8]` for `[0–4]`
  lines cleared, a small per-step survival bonus, and a game-over penalty.
- **Fair data distribution.** The 7-bag randomizer guarantees a balanced piece
  distribution, avoiding pathological runs (e.g. long droughts of a needed piece) that
  would bias the experience the agent learns from.

---

## 4. Challenges faced during implementation

- **Credit assignment / sparse reward (PPO).** Line clears are rare early on, so the
  reward signal is weak and delayed. Much of the PPO effort went into reward shaping and
  entropy/learning-rate tuning, and it still underperformed (§5) — which is precisely
  what motivated the afterstate approach.
- **Afterstate enumeration cost.** Scoring every legal placement means cloning and
  simulating the game up to 40× *per step*, in pure Python. This makes DQN steps
  expensive (the 150k-step run took ~90 minutes on an Apple-silicon MPS device) and was
  the main wall-clock bottleneck.
- **Exploration schedule.** Getting the ε schedule right mattered: decay too slowly and
  the agent wastes most of training on random play; too fast and it converges
  prematurely. I settled on a fast decay to ε = 0.05 followed by a long low-ε phase.
- **Overestimation bias.** Early value-based experiments overestimated Q-values and
  produced unstable policies; adding Double DQN and a target network fixed this.
- **Tooling/environment.** Integrating action masking with Stable-Baselines3's vectorized
  envs (custom `MaskableEvalCallback`, one-hot handling in the feature extractor)
  required care, and one training run failed simply because it was launched with a Python
  interpreter that lacked the dependencies — a reminder to pin the environment.

---

## 5. Outcomes

**A note on metrics.** This is RL, not classification, so there is no "accuracy score."
The right measures of a Tetris agent are **lines cleared per game** (the true objective)
and **episode length / pieces placed** (how long it survives). I evaluate **greedily**
(no exploration) over 20 episodes, each on a distinct random seed.

| Agent | Training steps | Lines cleared / game | Pieces placed / game |
|---|---|---|---|
| Maskable PPO | 10,000,000 | 3.1 ± 1.6 | 81.8 ± 7.9 |
| **Afterstate DQN** | 150,000 | **54.6 ± 37.5** | **166.8 ± 93.5** |

**Did it work / was it a good fit?** Yes — the afterstate DQN learned a genuinely
competent policy, clearing ~18× more lines than PPO *with ~60× fewer training steps*.
During training its rolling average rose steadily from ~0 to ~19 lines/episode, with no
sign of divergence, indicating a healthy fit to the value-learning objective rather than
memorization (every game is a new random sequence, so there is no training set to overfit
to). The large standard deviation is expected and not a defect: Tetris outcomes are
genuinely high-variance — a lucky piece sequence yields a long game, an unlucky one ends
early — so spread across random seeds is inherent to the task.

**PPO, by contrast, underfit the problem:** despite vastly more experience it plateaued
at ~3 lines/game, consistent with the credit-assignment difficulty discussed above.

**Does it give good predictions?** In RL terms, the learned value function makes good
*decisions*: greedily following `argmax Q(afterstate)` produces long, multi-line-clearing
games. The result here comes from the **short** 150k-step schedule; the full 2M-step
configuration ([`dqn/configs/default.yaml`](dqn/configs/default.yaml)) clears
substantially more and is the recommended setting for a best-effort model.

### Reproducing these numbers

Trained weights are included in the repo (a few MB each), so the results can be
reproduced directly without retraining:

```bash
# Afterstate DQN
cd dqn && python evaluate.py --model models/dqn_final.pt --episodes 20

# Maskable PPO
python evaluate.py models/ppo_best.zip --episodes 20
```
