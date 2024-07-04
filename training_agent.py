# training_agent.py    
import gym
import numpy as np
from maze_env import SimpleMazeEnv
import json
import os

# Q-learning parameters
alpha = 0.1
gamma = 0.9
epsilon = 0.9

maze = [
    [0, 1, 0, 0, 0],
    [0, 1, 0, 1, 0],
    [0, 0, 0, 1, 0],
    [1, 1, 0, 1, 0],
    [0, 0, 0, 1, 0]
]
env = SimpleMazeEnv(maze)

# Q-table initialization
q_table = {}

# Training parameters
num_episodes = 1000
episode_iterations = []
q_table_history = []

# Training the agent
for episode in range(num_episodes):
    state = env.reset()
    done = False
    iterations = 0

    while not done:
        if state not in q_table:
            q_table[state] = np.zeros(env.action_space.n)

        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])

        next_state, reward, done, _ = env.step(action)

        if next_state not in q_table:
            q_table[next_state] = np.zeros(env.action_space.n)

        q_table[state][action] += alpha * (reward + gamma * np.max(q_table[next_state]) - q_table[state][action])
        state = next_state

        iterations += 1

    episode_iterations.append(iterations)
    q_table_history.append({str(k): v.tolist() for k, v in q_table.items()})
    print(f"Episode {episode + 1} completed. Iterations: {iterations}")

# Save Q-table history and episode iteration data
output_data = {
    'q_table_history': q_table_history,
    'episode_iterations': episode_iterations
}

output_folder = 'output'
os.makedirs(output_folder, exist_ok=True)
file_path = os.path.join(output_folder, 'q_learning_data.json')

with open(file_path, 'w') as f:
    json.dump(output_data, f)

print(f"Q-table history and episode iteration data saved to {file_path}")
print("Training complete.")

# Test the trained agent
state = env.reset()
env.render()

for _ in range(100):  # Maximum of 100 steps
    action = np.argmax(q_table[state])
    state, reward, done, _ = env.step(action)
    env.render()
    if done:
        break

env.close()