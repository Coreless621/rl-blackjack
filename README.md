# ♠️ Blackjack-v1 – Q-Learning Agent

This project implements a **Q-learning agent** for the `Blackjack-v1` environment in [Gymnasium](https://gymnasium.farama.org/), using a state-action value table (`Q(s, a)`) with dictionary-based function approximation.

The agent learns to play the simplified Blackjack card game using tabular Q-learning, epsilon-greedy exploration, and long-term reward optimization.

---

## 🃏 Environment Overview

- **Environment:** `Blackjack-v1` (with `sab=True`)
- **States:** Tuples representing player sum, dealer's visible card, and usable ace status
- **Actions:** `0 = stick`, `1 = hit`
- **Goal:** Maximize expected return over many episodes

---

## 🧠 Algorithm

- **Learning type:** Tabular Q-learning with `defaultdict`
- **Exploration:** Epsilon-greedy with exponential decay
- **Update rule:**  
  \( Q(s, a) \leftarrow Q(s, a) + \alpha [r + \gamma \cdot \max_a Q(s', a) - Q(s, a)] \)
- **Episode length:** Until terminal Blackjack game outcome

---

## ⚙️ Hyperparameters

- Learning rate   | `alpha = 0.01` 
- Discount factor | `gamma = 0.99` 
- Initial epsilon | `1.0` 
- Minimum epsilon | `0.1` 
- Epsilon decay   | adaptive over `1,000,000` episodes 
- Total episodes  | `1,000,000` 

---

## 📁 Project Structure

- `training.py`   | Trains the Q-learning agent on Blackjack-v1 and saves the Q-table as a `.pkl` file 
- `evaluation.py` | Loads the trained Q-table and evaluates the agent's performance over 10 episodes (rendered to screen) 
- `q_values.pkl`  | (Generated) Serialized Q-table using `pickle` 

---

## 📊 Visualization

The training script also includes a reward plot:
- Raw rewards (gray, transparent)
- Smoothed moving average over 100 episodes (blue)

This helps visualize the agent's learning progress over time.

---
