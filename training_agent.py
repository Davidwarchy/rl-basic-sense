import gym
import numpy as np
from maze_env import SimpleMazeEnv
import json
import os

# Q-learning parameters
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 0.9  # Exploration-exploitation trade-off

# Create a simple maze environment
maze = [
    [0, 1, 0, 0, 0],
    [0, 1, 0, 1, 0],
    [0, 0, 0, 1, 0],
    [1, 1, 0, 1, 0],
    [0, 0, 0, 1, 0]
]
env = SimpleMazeEnv(maze)

# Q-table initialization as a dictionary
q_table = {}

def state_to_index(state):
    # Convert local observation 3x3 grid to a single integer index
    return ''.join(map(str, state.flatten().astype(int)))

# Training parameters
num_episodes = 100
episode_iterations = []  # To store number of iterations per episode
q_table_history = []  # To store Q-table after each episode

# Training the agent
for episode in range(num_episodes):
    local_obs = env.reset()
    state_index = state_to_index(local_obs)
    done = False
    iterations = 0

    while not done:
        # Initialize Q-values for the state if not already present
        if state_index not in q_table:
            q_table[state_index] = np.zeros(env.action_space.n)

        # Choose action using epsilon-greedy policy
        randy = np.random.random()
        if randy < epsilon:
            action = env.action_space.sample()  # Exploration
        else:
            action = np.argmax(q_table[state_index])  # Exploitation

        next_local_obs, reward, done, _ = env.step(action)
        next_state_index = state_to_index(next_local_obs)

        # Initialize Q-values for the next state if not already present
        if next_state_index not in q_table:
            q_table[next_state_index] = np.zeros(env.action_space.n)

        # Q-table update using Bellman equation
        q_table[state_index][action] += alpha * (reward + gamma * np.max(q_table[next_state_index]) - q_table[state_index][action])
        state_index = next_state_index

        iterations += 1

    episode_iterations.append(iterations)
    q_table_history.append(q_table.copy())  # Save a copy of Q-table after each episode
    print(f"Episode {episode + 1} completed. Iterations: {iterations}")

# Convert dictionary Q-table history to JSON serializable format
q_table_history_serializable = [{k: v.tolist() for k, v in q.items()} for q in q_table_history]

# Save Q-table history and episode iteration data to a single JSON file
output_data = {
    'q_table_history': q_table_history_serializable,
    'episode_iterations': episode_iterations
}

output_folder = 'output'  # Output folder for saving data
os.makedirs(output_folder, exist_ok=True)
file_path = os.path.join(output_folder, 'q_learning_data.json')

with open(file_path, 'w') as f:
    json.dump(output_data, f)

print(f"Q-table history and episode iteration data saved to {file_path}")
print("Training complete.")
