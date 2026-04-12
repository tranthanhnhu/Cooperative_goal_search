# Cooperative Goal Search - Updated Workflow Package

This package follows the workflow you asked for:

1. **Train in the fast Python simulator**
2. **Export greedy paths / policy replay data**
3. **Open Webots and visually replay the robot motion**

The training and the Webots visualization are intentionally separated.
That is why Python training is much faster than running RL directly inside Webots.

---

## What this package now does

### Python simulator
- runs the cooperative goal search task from the paper in a lightweight 2D simulator
- trains 4 comparison methods
- saves paper-style learning curves
- exports `webots_paths.json` for later replay in Webots

### Webots
- reads the exported path file
- moves robots along the learned path so you can demonstrate the result visually
- acts as the **visual demo layer**, not the heavy RL training layer

---

## Important fixes made

Compared with the previous version, this package now fixes the following:

1. **Cluster incremental update corrected**
   - denominator now uses the new sample count, matching the paper equations

2. **Collision no longer ends the episode**
   - collision gives reward `-100`
   - episode only ends when the goal is reached or max steps is hit

3. **Methods are separated more clearly**
   - `dyna_no_sharing`: no sharing
   - `raw_sharing`: teammates directly consume the same real experiences
   - `request_sharing`: request-based sharing without T-test fusion
   - `proposed`: request-based sharing with T-statistic fusion

4. **Webots path export added**
   - after training, the script writes `results/webots_paths.json`
   - Webots controller reads that file automatically

5. **Plot smoothing added**
   - moving average is used so curves are easier to read like a paper figure

---

## Repository layout

```text
cooperative_goal_search/
├── README.md
├── requirements.txt
├── train_compare.py
├── src/
│   ├── config.py
│   ├── env.py
│   ├── agent.py
│   ├── training.py
│   └── plotting.py
└── webots/
    ├── controllers/
    │   └── cooperative_goal_search/
    │       └── cooperative_goal_search.py
    └── worlds/
        └── cooperative_goal_search.wbt
```

---

## Quick start

### 1) Create a Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Run training in the Python simulator

```bash
python train_compare.py --episodes 500 --trials 10 --save-dir results
```

This generates:

- `results/dyna_no_sharing_results.json`
- `results/raw_sharing_results.json`
- `results/request_sharing_results.json`
- `results/proposed_results.json`
- `results/cooperative_goal_search_curves.png`
- `results/webots_paths.json`

---

## Why training feels fast

That is expected.

You are **not training in Webots** here.
You are training in a lightweight Python simulator that only updates:

- agent positions
- rewards
- Q-values
- stochastic model clusters
- sharing / fusion logic

It does **not** render a 3D world or run Webots physics each step.

So the intended workflow is:

> fast RL training in Python → export paths → visualize in Webots

---

## How to use Webots

### 1) Install Webots
Open the world:

```text
webots/worlds/cooperative_goal_search.wbt
```

### 2) Make sure the controller is attached
Each robot should use:

```text
cooperative_goal_search
```

### 3) Keep `results/webots_paths.json`
The controller tries to read:

- `results/webots_paths.json`
- or `fixed_results/webots_paths.json`

relative to the package root.

### 4) Run the world
The robots will replay the path learned by the proposed method.

---

## Paper alignment notes

This code is **paper-aligned**, but still includes a few practical approximations:

### Exact or very close to the paper
- world size 300 x 300
- 3 robots with 3 start areas
- discrete actions up/down/left/right
- step length 5 with Gaussian noise 0.5
- reward setup
- cluster-based stochastic model
- request-based sharing
- T-statistic fusion in the proposed method

### Practical approximations
- obstacle geometry approximates Fig. 8 visually and functionally
- the learning environment is a fast 2D simulator instead of direct Webots training
- the baselines are simplified research-style baselines for comparison

---

## Recommended workflow for your homework

1. Train with Python
2. Save the plot as your learning-curve figure
3. Open Webots and show the robots replaying the learned path
4. Explain clearly:
   - Python simulator is used for efficient RL training
   - Webots is used for visual demonstration

---

## Good explanation sentence for your report

> We separated the implementation into two layers: a lightweight Python simulator for efficient reinforcement learning and a Webots world for visual replay of the learned navigation behavior. This design lets us train quickly while still demonstrating the cooperative robot motion in a realistic simulator.
