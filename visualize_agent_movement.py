import json
import matplotlib.pyplot as plt
import numpy as np
from maze_env import SimpleMazeEnv

def visualize_agent_movement(q_table_file, maze):
    # Load Q-table history and episode iteration data from JSON file
    with open(q_table_file, 'r') as f:
        data = json.load(f)
    q_table_history = data['q_table_history']
    episode_iterations = data['episode_iterations']

    # Convert Q-table history back to original format
    q_table_history = [{k: np.array(v) for k, v in q.items()} for q in q_table_history]

    # Initialize environment
    env = SimpleMazeEnv(maze)

    def state_to_index(state):
        # Convert local observation 3x3 grid to a single integer index
        return ''.join(map(str, state.flatten().astype(int)))

    # Use the Q-table from the last episode for visualization
    q_table = q_table_history[-1]
    agent_positions = []

    # Reset environment to start visualization
    local_obs = env.reset()
    state_index = state_to_index(local_obs)
    done = False
    steps = 0

    while not done:
        agent_positions.append(env.agent_position)
        # Choose action with the highest Q-value for the current state
        if state_index in q_table:
            max_q_value = np.max(q_table[state_index])
            actions_with_max_q_value = [action for action, q_value in enumerate(q_table[state_index]) if q_value == max_q_value]
            action = np.random.choice(actions_with_max_q_value)
        else:
            action = env.action_space.sample()  # If state is not in Q-table, choose random action

        next_local_obs, reward, done, _ = env.step(action)
        state_index = state_to_index(next_local_obs)
        steps += 1

    # Plot the agent's movement
    plt.figure()
    env.render()

    # Plot agent's path
    for pos in agent_positions:
        plt.scatter(*env._maze_to_plot(pos), color='blue', marker='o', s=100)

    plt.title(f"Agent's Movement (Total steps: {steps})")
    plt.show()

    return steps

# Usage example:
maze = [
    [0, 1, 0, 0, 0],
    [0, 1, 0, 1, 0],
    [0, 0, 0, 1, 0],
    [1, 1, 0, 1, 0],
    [0, 0, 0, 1, 0]
]

q_table_file = 'output/q_learning_data.json'
steps_taken = visualize_agent_movement(q_table_file, maze)
print(f"Total steps taken by the agent: {steps_taken}")
