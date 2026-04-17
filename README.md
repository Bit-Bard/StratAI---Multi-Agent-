# 🚀 StratAI — Multi-Agent Battle Arena using Reinforcement Learning

## 🧠 Introduction

**StratAI** is a **multi-agent reinforcement learning project** where two AI agents learn to fight each other in a grid-based battle environment.

Instead of hardcoding behavior, both agents:

* Learn **how to move**
* Learn **when to attack**
* Learn **when to dodge**
* Develop **strategies over time**

👉 This project simulates **real decision-making AI**, similar to what is used in:

* Game AI (like NPCs)
* Robotics
* Autonomous systems

---

# 📘 Explanation Section (For Beginners)

## 🔹 What is Reinforcement Learning (RL)?

Reinforcement Learning is a type of machine learning where:

👉 An **agent learns by interacting with an environment**

* It takes actions
* Gets rewards or penalties
* Improves over time

### 🧠 Simple Example:

Imagine teaching a dog:

* Sit → reward ✅
* Jump randomly → no reward ❌

Over time, the dog learns what works.

---

## 🔹 Core RL Components

### 1. **Agent**

The learner (our AI player)

👉 Example: Player A in the game

---

### 2. **Environment**

Where the agent operates

👉 Example: Grid battlefield

---

### 3. **State**

Current situation of the agent

👉 Example:

* Position of agents
* Health
* Distance

---

### 4. **Action**

What the agent can do

👉 Example:

* Move (up, down, left, right)
* Attack
* Wait
* Dodge

---

### 5. **Reward**

Feedback from environment

👉 Example:

* Attack success → +reward
* Wrong action → penalty

---

### 6. **Policy**

Strategy used by agent

👉 Learned automatically

---

# ⚙️ Important Parameters (Simple Explanation)

## 🔹 Episodes

Number of games played during training

👉 More episodes = better learning

---

## 🔹 Gamma (γ)

Future reward importance

* High (0.99) → long-term thinking
* Low (0.5) → short-term

👉 Example:
Choosing between:

* Immediate attack
* Better position later

---

## 🔹 Epsilon (ε) *(DQN phase)*

Exploration vs exploitation

* High ε → random actions
* Low ε → learned behavior

---

## 🔹 Learning Rate

How fast model learns

* High → unstable
* Low → slow

---

## 🔹 PPO Clip (ε_clip)

Prevents drastic policy updates

👉 Keeps learning stable

---

# 🧩 Project Pipeline (Overview)

## Phase 1 → Basic RL Setup

* Created environment
* Added movement + attack
* Used simple reward system

---

## Phase 2 → DQN Training

* Trained agent using Q-learning
* Faced instability & oscillation
* Agents got stuck or behaved randomly

---

## Phase 3 → PPO (Advanced RL)

* Switched to PPO
* Stable learning
* Smooth behavior

---

## Phase 4 → Self-Play

* AI vs AI training
* No fixed opponent
* Real strategy emergence

---

## Phase 5 → Advanced Combat System

Added:

* Attack cooldown
* Dodge system
* Critical hits
* Sequential combat

---

## Phase 6 → Visualization

* Built real-time game UI
* Pygame-based rendering
* Health bars + attack effects

---

# 🔬 Step-by-Step Development

---

## 🔹 Phase 1: Environment Design

### 🎯 Goal:

Create basic world for agents

### 🧩 Features:

* Grid system
* Movement
* Basic attack

### ❗ Challenges:

* Agents not learning
* No meaningful reward

### ✅ Solution:

* Added reward shaping

---

## 🔹 Phase 2: DQN Training

### 🎯 Goal:

Train agent using Q-learning

### ❗ Problems:

* Oscillating rewards
* Agents stuck
* Unstable behavior

### 🧠 Insight:

DQN fails in multi-agent systems

---

## 🔹 Phase 3: PPO Upgrade

### 🎯 Why PPO?

Because:

* Stable learning
* Better for continuous decision making

### ✅ Result:

* Smooth behavior
* Better convergence

---

## 🔹 Phase 4: Self-Play

### 🎯 Goal:

AI learns from itself

### ❗ Problem:

* Collapse / imbalance

### ✅ Solution:

* Freeze opponent periodically

---

## 🔹 Phase 5: Combat System Fix

### Issues:

* Attack spam
* Draw matches

### Solutions:

* Attack cooldown
* Sequential attacks
* Random damage

---

## 🔹 Phase 6: Visualization

### Why important?

👉 RL has no clear metrics

Visualization helps:

* Understand behavior
* Debug learning

---

# 🧠 Key Challenges & Solutions

| Problem             | Solution                       |
| ------------------- | ------------------------------ |
| Agents stuck        | Use sampling instead of argmax |
| Always draw         | Sequential attack logic        |
| One agent dominates | Add randomness                 |
| No strategy         | Reward shaping                 |
| Hard to debug       | Visualization                  |

---

# 📁 Project Structure

```
StratAI/
│
├── env/
│   └── battle_env.py
│
├── model/
│   └── PPO / DQN
│
├── ppo_selfplay.py
├── visualize_game_pro.py
└── utils/
```

---

# 📦 Important Libraries

## 🔹 PyTorch

Used for:

* Neural networks
* Training model

---

## 🔹 Gymnasium

Used for:

* Environment structure

---

## 🔹 NumPy

Used for:

* Calculations
* State handling

---

## 🔹 Pygame

Used for:

* Visualization
* Game rendering

---

# 🛠️ How to Run

## 1. Clone repo

```bash
git clone <your-repo-link>
cd StratAI
```

---

## 2. Install dependencies

```bash
pip install torch pygame gymnasium numpy
```

---

## 3. Train model

```bash
python ppo_selfplay.py
```

---

## 4. Run visualization

```bash
python visualize_game_pro.py
```

---

# 📊 Final Result

* AI learns from scratch
* Develops strategies
* Shows real-time combat

👉 This is **multi-agent intelligence in action**

---

# 🧾 Summary

StratAI demonstrates:

* Reinforcement Learning fundamentals
* Transition from DQN → PPO
* Multi-agent self-play
* Real-world AI behavior

---

# ❤️ Connect With Me

📧 Email: [dhruvdevaliya@gmail.com](mailto:dhruvdevaliya@gmail.com)
📞 Phone: +91 8591216244
🔗 LinkedIn: https://www.linkedin.com/in/dhruv-devaliya/

---

# 💙 Built with passion by **Bit Bard**
