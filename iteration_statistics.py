import matplotlib.pyplot as plt
import numpy as np
import json 

# Load the saved Q-learning data
file_path = 'output/q_learning_data.json'
with open(file_path, 'r') as f:
    data = json.load(f)

episode_iterations = data['episode_iterations']

# Calculate variance in the number of iterations per episode
iterations_variance = [np.var(episode_iterations[:i+1]) for i in range(len(episode_iterations))]

# Plot the number of iterations per episode
plt.figure(figsize=(14, 7))

plt.subplot(2, 1, 1)
plt.plot(range(1, len(episode_iterations) + 1), episode_iterations, label='Iterations per Episode')
plt.xlabel('Episode')
plt.ylabel('Number of Iterations')
plt.title('Number of Iterations per Episode')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
