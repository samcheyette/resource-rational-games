import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from grammar import *
from constraints import *
import numpy as np
from utils.utils import *
from typing import List, Tuple, Set
import json

#from make_stimuli_mousetrack import make_stimulus
#from depth_predictor import DepthPredictor

"""
NOTE: for the game boards, we use the following:
0-8:    number of mines nearby
-1:     unknown
-3:     flag placed
-4:     safe mark placed (we don't think there's a mine)

EXAMPLE: 
[[-1 -1  2 -1 -1  1 -1]
[ 1  1  3 -1 -1  4  3]
[ 0  0  1 -1 -1 -1 -1]
[ 0  0  1  3 -1  4  2]
[ 0  0  0  1 -1  1  0]
[ 0  1  1  3 -1  2  0]
[ 0  1 -1 -1 -1  1  0]]
"""


    
def board_to_constraints(board, n_mines=None, keep_irrelevant=False):
    constraints = []
    rows, cols = board.shape
    
    # Create variables for all unrevealed squares first
    variables = {}
    for r in range(rows):
        for c in range(cols):
            if board[r, c] == -1:
                variables[f"v_{r}_{c}"] = Variable(f"v_{r}_{c}", row=r, col=c)
    
    # Then create constraints from numbered squares
    for r in range(rows):
        for c in range(cols):
            cell_value = board[r, c]
            if 0 <= cell_value <= 8:
                adjacent = get_adjacent_coords(rows, cols, r, c)
                relevant_variables = set()
                for adj_r, adj_c in adjacent:
                    if board[adj_r, adj_c] == -1:
                        relevant_variables.add(variables[f"v_{adj_r}_{adj_c}"])
                
                if relevant_variables or keep_irrelevant:
                    constraint = EqualityConstraint(relevant_variables, cell_value, row=r, col=c)
                    if constraint not in constraints:
                        constraints.append(constraint)

    if n_mines is not None:
        constraints.append(EqualityConstraint(variables.values(), n_mines))
    
    return constraints


def print_board(board, symbols = {-1: '?', -3: 'F', -4: 'X'}):

    rows = []
    for row in board:
        row_str = [symbols.get(cell, str(cell)) for cell in row]
        rows.append(' '.join(row_str))
    print('\n'.join(rows) + "\n")

class Game:
    """Represents a Minesweeper game state"""
    def __init__(self, board: np.ndarray):
        self.initial_board = board.copy()
        self.board = board.copy()  # Make a copy to avoid modifying original
        self.rows, self.cols = board.shape
        
    def get_adjacent_coords(self, row: int, col: int) -> List[Tuple[int, int]]:
        """Get coordinates of all valid adjacent squares"""
        adjacent = []
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                new_r, new_c = row + dr, col + dc
                if 0 <= new_r < self.rows and 0 <= new_c < self.cols:
                    adjacent.append((new_r, new_c))
        return adjacent
        
    def get_unrevealed_squares(self) -> List[Tuple[int, int]]:
        """Get coordinates of all unrevealed squares (including flagged and marked safe)"""
        return [(r, c) for r in range(self.rows) for c in range(self.cols)
                if self.board[r, c] in [-1, -3, -4]]
                
    def get_unmarked_squares(self) -> List[Tuple[int, int]]:
        """Get coordinates of all unmarked squares (no flag or safe mark)"""
        return [(r, c) for r in range(self.rows) for c in range(self.cols)
                if self.board[r, c] == -1]
                
    def get_revealed_squares(self) -> List[Tuple[int, int]]:
        """Get coordinates of all revealed squares (with numbers)"""
        return [(r, c) for r in range(self.rows) for c in range(self.cols)
                if 0 <= self.board[r, c] <= 8]
                
    def get_neighboring_variables(self, row: int, col: int) -> Set[Variable]:
        """Get variables representing unrevealed squares adjacent to given position"""
        neighbors = self.get_adjacent_coords(row, col)
        return {Variable(f"v_{r}_{c}") for r, c in neighbors 
                if self.board[r, c] in [-1, -3, -4]}
                
    def get_revealed_neighbors(self, row: int, col: int) -> List[Tuple[int, int, int]]:
        """Get coordinates and values of revealed squares adjacent to given position.
        Returns list of (row, col, value) tuples."""
        neighbors = self.get_adjacent_coords(row, col)
        return [(r, c, self.board[r, c]) for r, c in neighbors 
                if 0 <= self.board[r, c] <= 8]
    
    def flag(self, row: int, col: int):
        """Place a flag at the specified position"""
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            raise ValueError(f"Position ({row}, {col}) is out of bounds")
        if self.board[row, col] not in [-1, -3, -4]:  # Can only flag unknown cells
            raise ValueError(f"Cannot flag revealed cell at ({row}, {col})")
        self.board[row, col] = -3
        
    def mark_safe(self, row: int, col: int):
        """Mark a cell as safe (no mine) at the specified position"""
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            raise ValueError(f"Position ({row}, {col}) is out of bounds")
        if self.board[row, col] not in [-1, -3, -4]:  # Can only mark unknown cells
            raise ValueError(f"Cannot mark revealed cell at ({row}, {col})")
        self.board[row, col] = -4

    def unmark(self, row: int, col: int):
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            raise ValueError(f"Position ({row}, {col}) is out of bounds")
        if self.board[row, col] not in [-3, -4]:  # Can only unmark flags and safe marks
            raise ValueError(f"Cannot unmark revealed cell at ({row}, {col})")
        self.board[row, col] = -1   
        
        
    def get_board_state(self):
        """Return current board state"""
        return self.board.copy()
    
    def is_solved(self):
        """Check if the game is solved"""
        return np.sum(self.board == -1) == 0
    
    def reset(self):
        self.board = self.initial_board.copy()
        
    def __str__(self) -> str:
        """Return string representation of the board"""
        # Create a mapping for special values
        symbols = {
            -1: '?',  # unknown
            -3: 'F',  # flag
            -4: 'X'   # safe mark
        }
        
        # Build the string representation row by row
        rows = []
        for row in self.board:
            # Convert each cell to its string representation
            row_str = [symbols.get(cell, str(cell)) for cell in row]
            rows.append(' '.join(row_str))
        
        return '\n'.join(rows)
        
    def __repr__(self) -> str:
        return f"Game(\n{str(self)}\n)"
    


def load_stimuli(path):
    with open(path, 'r') as file:
        stimuli = json.load(file)
    # Convert each stimulus to numpy array and save the conversion
    if stimuli:
        if isinstance(stimuli[0], list):
            stimuli = [np.array(stimulus) for stimulus in stimuli]
        else:
            stimuli = np.array([np.array(stimulus["game_state"]) for stimulus in stimuli])
        return stimuli



def get_board_state(agent, game_board):
    """Convert current agent state to board representation"""
    board = np.array(game_board, dtype=int)  # Convert to numpy array first
    
    # Then overlay agent's decisions
    for v in agent.variables:
        if v.is_assigned():
            if v.value == 1:  # Mine
                board[v.row][v.col] = -3  # Flag
            else:  # Safe
                board[v.row][v.col] = -4  # Safe mark
                
    return board

def randomly_complete_game(agent, game_state):
    """Randomly assign remaining unassigned variables"""
    unassigned = agent.get_unassigned_variables()
    for v in unassigned:
        # Random 0/1 assignment
        value = np.random.randint(2)
        v.assign(value)
        agent.solved_variables.add(v)
        # Update game state
        if value == 1:  # Mine
            game_state[v.row][v.col] = -3  # Flag
        else:  # Safe
            game_state[v.row][v.col] = -4  # Safe mark
    return game_state



def load_games(path):
    """Load games from JSON file and convert to numpy arrays
    
    Args:
        path: Path to JSON file containing games
        
    Returns:
        List of numpy arrays representing game boards
    """
    with open(path, 'r') as file:
        games = json.load(file)
    # Convert each game board back to numpy array
    return [np.array(game) for game in games]

def generate_random_minesweeper_game(rows: int, cols: int, n_mines: int, 
                                   require_unique: bool = True,
                                   n_revealed: int = 5,
                                   max_attempts: int = 1000) -> np.ndarray:
    """Generate a random valid Minesweeper puzzle."""
    
    def has_numbered_neighbor(board, x, y):
        """Check if cell has at least one numbered neighbor (0-8)"""
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < rows and 0 <= ny < cols:
                    if 0 <= board[nx, ny] <= 8:
                        return True
        return False

    def is_valid_puzzle(puzzle):
        """Check if puzzle has no isolated unknown cells"""
        for r in range(rows):
            for c in range(cols):
                if puzzle[r, c] == -1 and not has_numbered_neighbor(puzzle, r, c):
                    return False
        return True

    def create_complete_board():
        # Create mine vector and shuffle it
        mine_vector = np.array([1] * n_mines + [0] * (rows * cols - n_mines))
        np.random.shuffle(mine_vector)
        mine_matrix = mine_vector.reshape((rows, cols))
        
        # Calculate numbers for non-mine cells
        board = np.zeros(mine_matrix.shape, dtype=int)
        shifts = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        for dx, dy in shifts:
            shifted = np.roll(mine_matrix, shift=(dx, dy), axis=(0, 1))
            
            # Clear wrapped edges
            if dx == -1: shifted[-1, :] = 0
            elif dx == 1: shifted[0, :] = 0
            if dy == -1: shifted[:, -1] = 0
            elif dy == 1: shifted[:, 0] = 0
            
            board += shifted
            
        # Set mine locations to -1 (will be hidden in puzzle)
        board[mine_matrix == 1] = -1
        return board, mine_matrix
    
    def reveal_cell(board, revealed, x, y):
        """Recursively reveal cells starting from (x,y)"""
        if revealed[x, y] or board[x, y] == -1:
            return
            
        revealed[x, y] = True
        
        # If cell is empty, reveal adjacent cells
        if board[x, y] == 0:
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < rows and 0 <= ny < cols:
                        reveal_cell(board, revealed, nx, ny)
    
    for _ in range(max_attempts):
        # Generate complete board with mines and numbers
        complete_board, mine_matrix = create_complete_board()


        revealed = np.zeros((rows, cols), dtype=bool)
        
        # Get safe cells (non-mines)
        safe_cells = list(zip(*np.where(mine_matrix == 0)))
        if not safe_cells:
            continue
            
        # Reveal some random safe cells
        for _ in range(min(n_revealed, len(safe_cells))):
            if not safe_cells:
                break
            x, y = safe_cells.pop(np.random.randint(len(safe_cells)))
            reveal_cell(complete_board, revealed, x, y)
        
        # Create puzzle board with only revealed cells
        puzzle = np.full((rows, cols), -1)  # All cells start hidden
        puzzle[revealed] = complete_board[revealed]  # Show revealed cells

        # Check that all unknown cells have at least one numbered neighbor
        if not is_valid_puzzle(puzzle):
            continue
        
        # Check for unique solution if required
        if require_unique:
            constraints = board_to_constraints(puzzle)
            constraints = sort_constraints_by_relatedness(constraints)
            assignments = []

            for constraint in constraints:
                assignments, _ = integrate_new_constraint(assignments, constraint)
                if assignments is None:  # Hit contradiction
                    break
            if len(assignments) == 1:
                return puzzle
        else:
            return puzzle
            
    raise ValueError("Could not generate valid board after max attempts")

def save_boards(boards: List[np.ndarray], filename: str):
    """Save multiple minesweeper boards to a single JSON file."""
    save_dir = os.path.join(os.path.dirname(__file__), 'saved_boards', 'minesweeper')
    os.makedirs(save_dir, exist_ok=True)
    
    # Convert boards to lists and numpy integers to Python integers
    data = {
        'boards': [
            {
                'board': board.tolist(),
                'dimensions': tuple(int(x) for x in board.shape),  # Convert np.int64 to int
                'n_mines': int(np.sum(board == -1))  # Convert np.int64 to int
            }
            for board in boards
        ]
    }
    
    filepath = os.path.join(save_dir, f"{filename}.json")
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

def load_boards(filename: str) -> List[np.ndarray]:
    """Load multiple minesweeper boards from a JSON file.
    
    Args:
        filename: Name of file to load (without path or extension)
        
    Returns:
        List of board arrays
    """
    filepath = os.path.join(os.path.dirname(__file__), 'saved_boards', 'minesweeper', f"{filename}.json")
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    return [np.array(board_data['board']) for board_data in data['boards']]

if __name__ == "__main__":
    # Generate some test boards
    rows, cols = 7,7
    n_mines = 10
    n_boards = 25

    board = generate_random_minesweeper_game(rows, cols, n_mines, require_unique=True)
    
    print(f"Generating {n_boards} boards of size {rows}x{cols} with {n_mines} mines...")
    
    boards = []
    for i in range(n_boards):
        try:
            board = generate_random_minesweeper_game(rows, cols, n_mines, require_unique=True)

            constraints = board_to_constraints(board)
            assignments = []

            boards.append(board)
            print(f"\nGenerated board {i+1}/{n_boards}:")
            print_board(board)


        except ValueError as e:
            print(f"Failed to generate board {i+1}: {e}")
            continue
    
    # Save boards
    filename = f"minesweeper_{rows}_{cols}_{n_mines}"
    save_boards(boards, filename)
    print(f"\nSaved {len(boards)} boards to {filename}.json")


