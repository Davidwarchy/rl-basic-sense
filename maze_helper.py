import numpy as np

class SubmazePositionExtractor:
    def __init__(self, maze):
        self.original_maze = maze
        self.maze_with_walls = self.add_walls(maze)
    
    def add_walls(self, maze):
        # Add a border of 1s around the maze
        n, m = maze.shape
        new_maze = np.ones((n + 2, m + 2), dtype=int)
        new_maze[1:-1, 1:-1] = maze
        return new_maze

    def extract_3x3_submatrices_with_positions(self, maze=None):
        if maze is None:
            maze = self.maze_with_walls
        submatrices_with_positions = []
        n, m = maze.shape
        for i in range(n - 2):
            for j in range(m - 2):
                submatrix = maze[i:i+3, j:j+3]
                position = (i - 1, j - 1)  # Adjust for the original maze coordinates
                submatrices_with_positions.append((submatrix, position))
        return submatrices_with_positions

    def compare_submatrices(self, submatrices_with_positions):
        unique_submatrices = []
        for submatrix, position in submatrices_with_positions:
            if not any(np.array_equal(submatrix, u[0]) for u in unique_submatrices):
                unique_submatrices.append((submatrix, position))
        return unique_submatrices

    def find_positions_of_submaze(self, contracted_submaze):
        submatrix = np.array([int(x) for x in contracted_submaze]).reshape(3, 3)
        submatrices_with_positions = self.extract_3x3_submatrices_with_positions()
        positions = [pos for sm, pos in submatrices_with_positions if np.array_equal(sm, submatrix)]
        return positions

    def print_unique_submatrices(self):
        submatrices_with_positions = self.extract_3x3_submatrices_with_positions()
        unique_submatrices = self.compare_submatrices(submatrices_with_positions)
        for idx, (submatrix, position) in enumerate(unique_submatrices):
            print(f"Unique Submatrix {idx + 1} at position {position}:\n{submatrix}\n")

if __name__ == "__main__":

    # Define the original maze
    original_maze = np.array([[0, 0, 1, 0, 0, 1, 0],
                            [0, 1, 1, 0, 1, 0, 0],
                            [0, 0, 0, 0, 0, 1, 0],
                            [1, 1, 1, 1, 0, 1, 0],
                            [0, 0, 0, 1, 0, 0, 0],
                            [0, 1, 0, 0, 1, 1, 0],
                            [0, 0, 0, 0, 0, 0, 0]])

    # Create an instance of the SubmazePositionExtractor
    submaze_extractor = SubmazePositionExtractor(original_maze)

    # Add walls to the maze and print unique submatrices
    # submaze_extractor.print_unique_submatrices()

    # Find positions of a specific submaze given its contracted representation
    contracted_submaze = "111100101"
    positions = submaze_extractor.find_positions_of_submaze(contracted_submaze)
    print(f"Positions of submaze '{contracted_submaze}': {positions}")
