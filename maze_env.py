# maze_env.py
import gym
import numpy as np
import matplotlib.pyplot as plt

class SimpleMazeEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, maze):
        super(SimpleMazeEnv, self).__init__()
        self.maze = maze
        self.goal_position = (len(maze)-1, len(maze[0])-1)
        self.agent_position = (0, 0)
        self.action_space = gym.spaces.Discrete(4)  # Up, Down, Left, Right
        self.observation_space = gym.spaces.MultiDiscrete([2, 2, 2, 2])  # 4 surrounding cells

        # Visualization setup
        self.fig, self.ax = plt.subplots()
        self.ax.set_xticks(np.arange(-.5, len(maze[0]), 1), minor=True)
        self.ax.set_yticks(np.arange(-.5, len(maze), 1), minor=True)
        self.ax.grid(which='minor', color='gray', linestyle='-', linewidth=1)
        self.ax.invert_yaxis()
        
        self.agent_marker = self.ax.scatter(*self._maze_to_plot(self.agent_position), color='red', marker='o', s=100)
        self.goal_marker = self.ax.scatter(*self._maze_to_plot(self.goal_position), color='green', marker='*', s=100)
        self.ax.imshow(self.maze, cmap='binary', interpolation='nearest')

    def _maze_to_plot(self, position):
        return (position[1], position[0])

    def reset(self):
        self.agent_position = (0, 0)
        self.agent_marker.set_offsets(self._maze_to_plot(self.agent_position))
        return self._get_observation()

    def _get_observation(self):
        x, y = self.agent_position
        obs = [
            1 if x > 0 and self.maze[x-1][y] == 0 else 0,  # Up
            1 if x < len(self.maze)-1 and self.maze[x+1][y] == 0 else 0,  # Down
            1 if y > 0 and self.maze[x][y-1] == 0 else 0,  # Left
            1 if y < len(self.maze[0])-1 and self.maze[x][y+1] == 0 else 0  # Right
        ]
        return tuple(obs)

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

        if (0 <= next_position[0] < len(self.maze) and
            0 <= next_position[1] < len(self.maze[0]) and
            self.maze[next_position[0]][next_position[1]] != 1):
            self.agent_position = next_position
            self.agent_marker.set_offsets(self._maze_to_plot(self.agent_position))
        else:
            return self._get_observation(), -1.0, False, {}

        if self.agent_position == self.goal_position:
            reward = 1.0
            done = True
        else:
            reward = -0.01  # Small penalty for each step
            done = False

        return self._get_observation(), reward, done, {}

    def render(self, mode='human'):
        self.fig.canvas.draw()
        plt.pause(0.1)

    def close(self):
        plt.close(self.fig)

