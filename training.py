import gymnasium as gym
import numpy as np
import pickle
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt

# Hyperparameters
alpha = 0.01
gamma = 0.99
epsilon = 1.0
min_epsilon = 0.1
num_episodes = 1_000_000
decay = (min_epsilon / epsilon) ** (1 / num_episodes)

# Default Q-values
def default_q_values():
    return np.zeros(env.action_space.n)

# Variables regarding environment
env = gym.make("Blackjack-v1", sab=True)
q_values = defaultdict(default_q_values)

episode_rewards = []

# training loop
for episode in tqdm(range(num_episodes)):
    done = False
    state, _ = env.reset()
    total_rewards = 0
    while not done:
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_values[state])

        next_state, reward, terminated, truncated, _ = env.step(action)
        # Q-value update
        next_q = (not terminated) * np.max(q_values[next_state])
        current_q = q_values[state][action]
        q_values[state][action] = current_q + alpha * (reward + gamma * next_q - current_q)

        state = next_state
        total_rewards += reward
        done = terminated or truncated

    # Decay epsilon
    epsilon = max(min_epsilon, epsilon * decay)
    episode_rewards.append(total_rewards)

print("Finished training")
print(f"Average reward over last 100 episodes: {np.mean(episode_rewards[-100:]):.2f}")

# Save the Q-table to a file
if np.mean(episode_rewards[-100:]) > 0.01:
    with open("q_values.pkl", "wb") as f:
        pickle.dump(q_values, f)
    print("Q-table saved successfully!")

# Moving average function for smoothing
def moving_average(data, window_size=100):
    return np.convolve(data, np.ones(window_size) / window_size, mode="valid")

# Plot raw rewards and smoothed rewards
plt.figure(figsize=(10, 6))
plt.plot(episode_rewards, color="lightgray", alpha=0.5, label="Raw Rewards")
plt.plot(
    moving_average(episode_rewards, window_size=100), color="blue", label="Smoothed Rewards"
)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Average Performance Over Time")
plt.legend()
plt.show()
