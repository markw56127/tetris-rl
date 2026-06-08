# Deep Reinforcement Learning for Tetris

STATS 404 final project. Two deep-RL agents learn to play Tetris from scratch on a
custom, dependency-free game engine:

1. **Maskable PPO** (top-level) — a policy-gradient agent that picks a `(rotation, column)`
   placement directly, using action masking to forbid illegal moves. Built on
   [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) /
   [sb3-contrib](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib).
2. **Afterstate DQN** (`dqn/`) — a value-based agent that enumerates every legal placement,
   scores the *resulting board* (the "afterstate") with a CNN, and picks the best. Uses
   Double DQN to reduce value overestimation.

Both share the same Tetris engine (`tetris/`), so the comparison is apples-to-apples.

## Repository layout

```
.
├── tetris/                # Tetris engine: 7-bag randomizer, SRS pieces, line clears, scoring
│   ├── game.py            #   Game state, place_piece(), action masking, clone() for search
│   ├── pieces.py          #   Piece rotations + SevenBag
│   ├── constants.py       #   Board dims, scoring table, colors
│   └── renderer.py        #   Optional pygame rendering
│
├── env/                   # PPO Gymnasium environment
│   ├── tetris_env.py      #   Dict obs (board/current/hold/queue/mask), reward shaping
│   └── wrappers.py        #   Episode-stat recording, etc.
├── agent/                 # PPO model
│   ├── ppo.py             #   build_model(): MaskablePPO + custom extractor
│   ├── networks.py        #   TetrisCNNExtractor (CNN over board + piece/queue embeddings)
│   └── rollout.py         #   Standalone GAE rollout buffer (reference implementation)
├── configs/default.yaml   # PPO hyperparameters
├── train.py               # PPO training entry point
├── evaluate.py            # PPO evaluation / rendering
│
└── dqn/                   # Afterstate DQN (self-contained; vendors a copy of tetris/)
    ├── env/afterstate.py  #   Enumerate legal placements -> candidate afterstate boards
    ├── agent/networks.py  #   QNetwork: board -> scalar Q
    ├── agent/replay_buffer.py
    ├── configs/default.yaml
    ├── train.py
    └── evaluate.py
```

## The Tetris environment

- **Board:** 20×10 visible grid (plus a 2-row hidden spawn buffer).
- **Pieces:** standard 7 tetrominoes dealt with a 7-bag randomizer (SRS rotation states).
- **Action space (PPO):** 81 discrete actions — 40 placements of the current piece
  (4 rotations × 10 columns), 40 placements of the held piece, and 1 hold-only action.
  Illegal actions are masked.
- **Afterstate actions (DQN):** the 40 current-piece placements; the agent compares the
  resulting boards rather than choosing an action index.
- **Reward shaping (PPO):** survival bonus, `lines² × 5` for clears, and penalties for added
  holes / aggregate height / bumpiness, with a terminal penalty on game over.
- **Reward (DQN):** `[0, 1, 3, 5, 8]` for `[0..4]` lines cleared, a small survival bonus, and
  a game-over penalty.

## Setup

The dependencies are not vendored. Create an environment and install requirements:

```bash
# top-level (PPO) needs gymnasium + stable-baselines3 + sb3-contrib + torch
pip install -r requirements.txt

# the DQN is lighter (torch + numpy + pyyaml; pygame only for rendering)
pip install -r dqn/requirements.txt
```

> **Note:** make sure you run with the interpreter that has these packages installed.
> Bare `python`/`python3` on a fresh machine will not have them.

## Running

### PPO

```bash
# train (config: configs/default.yaml — 8 parallel envs, action masking)
python train.py --config configs/default.yaml

# resume from a checkpoint
python train.py --resume checkpoints/best_model

# evaluate a saved model (add --render for a pygame window)
python evaluate.py checkpoints/best_model.zip --episodes 20
python evaluate.py checkpoints/best_model.zip --episodes 5 --render
```

Checkpoints are written to `checkpoints/`, TensorBoard logs to `runs/`
(`tensorboard --logdir runs`). Both are git-ignored.

### Afterstate DQN

```bash
cd dqn

# train (full run: 2M steps)
python train.py --config configs/default.yaml

# shorter run for a quick model
python train.py --config configs/short.yaml

# evaluate
python evaluate.py --model checkpoints/best_model.pt --episodes 20
```

DQN checkpoints are written to `dqn/checkpoints/`.

## Notes

- `checkpoints/` and `runs/` are git-ignored; trained weights and logs are not committed.
- The `dqn/tetris/` directory is a copy of the top-level `tetris/` engine so the DQN runs
  standalone from inside `dqn/`.
