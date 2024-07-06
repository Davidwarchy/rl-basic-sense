import matplotlib.pyplot as plt
import numpy as np
import json

def plot_episode_iterations(data):
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

def count_unique_tuples(data):
    unique_counts = {}
    
    for item in set(data):
        unique_counts[item] = data.count(item)
    
    return unique_counts

if __name__ == "__main__":

    # Example usage:
    file_path = 'output/q_learning_data.json'
    with open(file_path, 'r') as f:
        data = json.load(f)

    # Plotting episodes iterations
    plot_episode_iterations(data)

    # Counting unique tuples
    data_tuples = [(0, 0), (0, 1), (0, 1), (0, 1), (0, 1), (0, 0), (0, 1), (0, 1), (0, 0), (0, 1)]
    unique_counts = count_unique_tuples(data_tuples)

    print("Unique Tuple Counts:")
    for item, count in unique_counts.items():
        print(f"{item}: {count}")
