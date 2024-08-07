# maze_env.py
import gym
import numpy as np
import matplotlib.pyplot as plt

class SimpleMazeEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, maze):
        super(SimpleMazeEnv, self).__init__()
        self.maze = maze
        self.goal_position = (len(maze)-1, len(maze[0])-1)  # Goal position is bottom-right corner
        self.agent_position = (0, 0)  # Agent starts at top-left corner
        self.action_space = gym.spaces.Discrete(4)  # Up, Down, Left, Right
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(3, 3), dtype=np.float32)  # 3x3 grid around the agent

        # Visualization setup
        self.fig, self.ax = plt.subplots()
        self.ax.set_xticks(np.arange(-.5, len(maze[0]), 1), minor=True)
        self.ax.set_yticks(np.arange(-.5, len(maze), 1), minor=True)
        self.ax.grid(which='minor', color='gray', linestyle='-', linewidth=1)
        self.ax.invert_yaxis()  # Flip the y-axis
        
        self.agent_marker = self.ax.scatter(*self._maze_to_plot(self.agent_position), color='red', marker='o', s=100)
        self.goal_marker = self.ax.scatter(*self._maze_to_plot(self.goal_position), color='green', marker='*', s=100)
        # self.ax.imshow(self.maze, cmap='binary', interpolation='nearest')

    def _maze_to_plot(self, position):
        """Convert maze coordinates to plot coordinates."""
        return (position[1], position[0])

    def _plot_to_maze(self, position):
        """Convert plot coordinates to maze coordinates."""
        return (position[1], position[0])

    def reset(self):
        self.agent_position = (0, 0)  # Reset agent to top-left corner
        self.agent_marker.set_offsets(self._maze_to_plot(self.agent_position))
        return self._get_local_observation()

    def step(self, action):
        x, y = self.agent_position

        if action == 0:  # Up
            next_position = (x - 1, y)
        elif action == 1:  # Down
            next_position = (x + 1, y)
        elif action == 2:  # Left
            next_position = (x, y - 1)
        elif action == 3:  # Right
            next_position = (x, y + 1)
        else:
            raise ValueError("Invalid action")

        # Check if the next position is within bounds and not a wall
        if (0 <= next_position[0] < len(self.maze) and
            0 <= next_position[1] < len(self.maze[0]) and
            self.maze[next_position[0]][next_position[1]] != 1):  # 1 represents a wall
            self.agent_position = next_position
            self.agent_marker.set_offsets(self._maze_to_plot(self.agent_position))
        else:
            # Penalize for hitting a wall or going out of bounds
            return self._get_local_observation(), -1.0, False, {}

        # Determine reward
        if self.agent_position == self.goal_position:
            reward = 1.0  # Reward for reaching the goal
            done = True
        else:
            # Reward only if the agent has moved
            if next_position != (x, y):
                reward = 0.1  # Small reward for moving
            else:
                reward = -0.1  # Penalize for not moving
            done = False
        
        return self._get_local_observation(), reward, done, {}

    # def _get_local_observation(self):
    #     """Get a 3x3 local observation around the agent."""
    #     x, y = self.agent_position
    #     local_obs = np.zeros((3, 3))
    #     for dx in range(-1, 2):
    #         for dy in range(-1, 2):
    #             nx, ny = x + dx, y + dy
    #             if 0 <= nx < len(self.maze) and 0 <= ny < len(self.maze[0]):
    #                 local_obs[dx+1, dy+1] = self.maze[nx][ny]
    #             else:
    #                 local_obs[dx+1, dy+1] = 1  # Treat out-of-bounds as walls
    #     return local_obs

    def _get_local_observation(self):
        x, y = self.agent_position
        
        # Pad the maze with 1's to handle out-of-bounds areas
        padded_maze = np.pad(self.maze, pad_width=1, mode='constant', constant_values=1)
        
        # Adjust position to account for padding
        x += 1
        y += 1
        
        # Extract the 3x3 sub-section centered on the agent's position
        sub_section = padded_maze[x-1:x+2, y-1:y+2]
        
        # Flatten the sub-section into a tuple
        flat_state = sub_section.flatten()
        
        return flat_state


    def render(self, mode='human'):
        self.fig.canvas.draw()
        plt.pause(0.1)  # Adjust the pause duration for visualization

    def close(self):
        plt.close(self.fig)

if __name__ == "__main__":
    
    # Example usage:
    maze = [
        [0, 1, 0, 0, 0],
        [0, 1, 0, 1, 0],
        [0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0],
        [0, 0, 0, 1, 0]
    ]

    env = SimpleMazeEnv(maze)
    obs = env.reset()
    env.render()

    # Example of changing coordinates
    new_positions = [(0, 1), (1, 1), (2, 1), (2, 2), (3, 2), (4, 2), (4, 3), (4, 4)]
    for pos in new_positions:
        env.agent_position = pos
        env.agent_marker.set_offsets(env._maze_to_plot(pos))
        plt.pause(.5)  # Pause to visualize the change

    input()
