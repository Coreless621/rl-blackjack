import gymnasium as gym
import numpy as np
import pickle

env = gym.make("Blackjack-v1", sab = True, render_mode = "human")

def default_q_values():
    return np.zeros(2)

with open("q_values.pkl", "rb") as f:
    q_values = pickle.load(f)
print("Q-table loaded successfully!")

num_episodes = 10
num_wins = 0
for episode in range(num_episodes):
    done = False
    state, _ = env.reset()
    while not done:
        action = np.argmax(q_values[state])
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        if reward > 0:
            num_wins += reward


print(f"Won {num_wins} times out of {num_episodes} games.")