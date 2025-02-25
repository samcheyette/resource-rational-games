import numpy as np
from typing import List, Tuple, Set
from constraints import *
import math
import random
import os
import json



def board_to_constraints(board: np.ndarray) -> List[Constraint]:
    """Convert board state to list of constraints.
    
    Creates variables for each empty cell and uniqueness constraints for:
    - Each row
    - Each column
    - Each box
    """
    constraints = []
    size = board.shape[0]
    box_size = int(math.sqrt(size))
    
    # Create variables for each empty cell
    variables = {}
    for r in range(size):
        for c in range(size):
            if board[r, c] == 0:  # Empty cell
                variables[f"v_{r}_{c}"] = Variable(f"v_{r}_{c}", domain=set(range(1, size+1)), row=r, col=c)
    
    # Add row constraints
    for r in range(size):
        # Get variables and constants in this row
        row_vars = [variables[f"v_{r}_{c}"] for c in range(size) 
                   if f"v_{r}_{c}" in variables]
        constants = {board[r, c] for c in range(size) 
                    if board[r, c] > 0}  # Given numbers
        if row_vars:  # Only create constraint if there are variables
            constraints.append(UniquenessConstraint(row_vars, constants=constants, row=r, col=None))
    
    # Add column constraints
    for c in range(size):
        # Get variables and constants in this column
        col_vars = [variables[f"v_{r}_{c}"] for r in range(size) 
                   if f"v_{r}_{c}" in variables]
        constants = {board[r, c] for r in range(size) 
                    if board[r, c] > 0}  # Given numbers
        if col_vars:
            constraints.append(UniquenessConstraint(col_vars, constants=constants, row=None, col=c))
    
    # Add box constraints
    for box_r in range(0, size, box_size):
        for box_c in range(0, size, box_size):
            # Get variables and constants in this box
            box_vars = []
            constants = set()
            for r in range(box_r, box_r + box_size):
                for c in range(box_c, box_c + box_size):
                    if f"v_{r}_{c}" in variables:
                        box_vars.append(variables[f"v_{r}_{c}"])
                    elif board[r, c] > 0:
                        constants.add(board[r, c])
            if box_vars:
                constraints.append(UniquenessConstraint(box_vars, constants=constants))
    
    return constraints


class Game:
    """Class representing a Sudoku game state."""
    
    def __init__(self, board: np.ndarray):
        """Initialize game from a board.
        
        Args:
            board: nxn numpy array with:
                0: empty cell
                1-n: filled cell
            where n must be a perfect square (4, 9, 16, etc)
        """
        rows, cols = board.shape
        if rows != cols:
            raise ValueError(f"Board must be square, got {rows}x{cols}")
            
        # Check if size is perfect square
        box_size = int(math.sqrt(rows))
        if box_size * box_size != rows:
            raise ValueError(f"Board size must be perfect square, got {rows}")
            
        self.size = rows
        self.box_size = box_size
        self.initial_board = board.copy()
        self.board = board.copy()


    def unmarked_squares_remaining(self):
        return np.sum(self.board == 0)
        
    def is_solved(self) -> bool:
        """Check if the game is solved."""
        valid_values = set(range(1, self.size + 1))
        
        # Convert board to absolute values for checking
        abs_board = np.abs(self.board)
        
        # Check all cells are filled
        if np.any(abs_board == 0):
            return False
            
        # Check rows
        for row in abs_board:
            if set(row) != valid_values:
                return False
                
        # Check columns
        for col in abs_board.T:
            if set(col) != valid_values:
                return False
                
        # Check boxes
        for i in range(0, self.size, self.box_size):
            for j in range(0, self.size, self.box_size):
                box = abs_board[i:i+self.box_size, j:j+self.box_size].flatten()
                if set(box) != valid_values:
                    return False
                    
        return True
    
    def reset(self):
        """Reset the board to initial state."""
        self.board = self.initial_board.copy()
        
    def __str__(self) -> str:
        """Return string representation of the board."""
        cell_width = len(str(self.size))  # Width needed for largest number
        box_width = self.box_size * (cell_width + 1) + 1
        
        # Add horizontal lines between boxes
        rows = []
        for i, row in enumerate(self.board):
            if i > 0 and i % self.box_size == 0:
                rows.append('-' * (box_width * self.box_size - 1))
            
            # Add vertical lines between boxes
            row_str = ''
            for j, cell in enumerate(row):
                if j > 0 and j % self.box_size == 0:
                    row_str += '|'
                # Convert negative numbers to positive and handle display
                value = abs(cell) if cell != 0 else 0
                display = str(value) if value != 0 else '.'
                if cell < 0:  # Player-placed numbers shown in parentheses
                    display = f"{value}"
                row_str += ' ' * (cell_width - len(display) + 1)
                row_str += display
            rows.append(row_str)
            
        return '\n'.join(rows) + '\n'
        
    def __repr__(self) -> str:
        return f"Game(\n{str(self)}\n)"

    def get_empty_cells(self) -> List[Tuple[int, int]]:
        """Get coordinates of all empty cells"""
        return [(r, c) for r in range(self.size) for c in range(self.size)
                if self.board[r, c] == 0]

    def get_player_cells(self) -> List[Tuple[int, int]]:
        """Get coordinates of all player-placed numbers"""
        return [(r, c) for r in range(self.size) for c in range(self.size)
                if self.board[r, c] < 0]

    def place_number(self, row: int, col: int, number: int):
        """Place a number at the specified position"""
        if not (0 <= row < self.size and 0 <= col < self.size):
            raise ValueError(f"Position ({row}, {col}) is out of bounds")
        if not (1 <= number <= self.size):
            raise ValueError(f"Number {number} must be between 1 and {self.size}")
        if self.initial_board[row, col] > 0:  # Can't modify given numbers
            raise ValueError(f"Cannot modify given number at ({row}, {col})")
        self.board[row, col] = -number  # Store as negative to indicate player-placed

    def clear_cell(self, row: int, col: int):
        """Clear a player-placed number from a cell"""
        if not (0 <= row < self.size and 0 <= col < self.size):
            raise ValueError(f"Position ({row}, {col}) is out of bounds")
        if self.initial_board[row, col] != 0:  # Can't modify given numbers
            raise ValueError(f"Cannot clear given number at ({row}, {col})")
        if self.board[row, col] >= 0:  # Can only clear player-placed numbers
            raise ValueError(f"No player-placed number to clear at ({row}, {col})")
        self.board[row, col] = 0

    def get_board_state(self):
        """Return current board state"""
        return self.board.copy()

    def get_value(self, row: int, col: int) -> int:
        """Get the value at the specified position
        
        Returns absolute value (positive) of the number in the cell.
        """
        if not (0 <= row < self.size and 0 <= col < self.size):
            raise ValueError(f"Position ({row}, {col}) is out of bounds")
        value = self.board[row, col]
        return abs(value) if value < 0 else value

    def is_given(self, row: int, col: int) -> bool:
        """Check if the number at the position was given in the initial board"""
        if not (0 <= row < self.size and 0 <= col < self.size):
            raise ValueError(f"Position ({row}, {col}) is out of bounds")
        return self.initial_board[row, col] != 0


def generate_random_solved_board(size=9):
    """Generate a random fully solved Sudoku board.
    
    Algorithm:
    1. Start with first row as 1..n
    2. For each subsequent row:
       - Find valid numbers for each position based on column constraints
       - Build valid permutation using only allowed numbers for each position
    """
    board = np.zeros((size, size), dtype=int)
    
    # First row is just 1..n
    board[0] = list(range(1, size + 1))
    random.shuffle(board[0])
    
    def get_valid_numbers(row_idx, col_idx):
        """Get numbers that could go in this position based on column and box constraints"""
        # Check column constraints
        used = {board[r, col_idx] for r in range(row_idx) if board[r, col_idx] != 0}
        
        # Check box constraints
        box_size = int(math.sqrt(size))
        box_row = (row_idx // box_size) * box_size
        box_col = (col_idx // box_size) * box_size
        
        # Get numbers already used in this box
        for r in range(box_row, min(box_row + box_size, row_idx)):
            for c in range(box_col, box_col + box_size):
                if board[r, c] != 0:
                    used.add(board[r, c])
        
        return set(range(1, size + 1)) - used
    
    def fill_next_row(row_idx):
        if row_idx >= size:
            return True
            
        # For each position, get valid numbers
        valid_numbers = [get_valid_numbers(row_idx, col) for col in range(size)]
        
        # Try to build valid permutation using only allowed numbers
        unused = set(range(1, size + 1))
        row = [0] * size
        
        def fill_position(col):
            if col >= size:
                return True
                
            # Get numbers that are both valid for this column and unused in this row
            available = valid_numbers[col] & unused
            if not available:
                return False
                
            # Try each available number
            numbers = list(available)
            random.shuffle(numbers)
            for num in numbers:
                row[col] = num
                unused.remove(num)
                
                if fill_position(col + 1):
                    return True
                    
                unused.add(num)
                row[col] = 0
            
            return False
        
        if fill_position(0):
            board[row_idx] = row
            return fill_next_row(row_idx + 1)
            
        return False
    
    # Try to fill the board
    while not fill_next_row(1): 
        random.shuffle(board[0])
    
    return board

def print_board(board):
    """Return string representation of the board."""
    cell_width = len(str(board.shape[0]))  # Width needed for largest number
    box_width = board.shape[0] * (cell_width + 1) + 1
    
    # Add horizontal lines between boxes
    rows = []
    for i, row in enumerate(board):
        if i > 0 and i % board.shape[0] == 0:
            rows.append('-' * (box_width * board.shape[0] - 1))
        
        # Add vertical lines between boxes
        row_str = ''
        for j, cell in enumerate(row):
            if j > 0 and j % board.shape[0] == 0:
                row_str += '|'
            # Convert negative numbers to positive and handle display
            value = abs(cell) if cell != 0 else 0
            display = str(value) if value != 0 else '.'
            if cell < 0:  # Player-placed numbers shown in parentheses
                display = f"{value}"
            row_str += ' ' * (cell_width - len(display) + 1)
            row_str += display
        rows.append(row_str)
        
    print('\n'.join(rows) + '\n')


def generate_puzzle(size=9, min_p_revealed=0.25, blowup_restart = 1000, attempts_remaining=100):
    """Generate a Sudoku puzzle with a unique solution.
    
    Algorithm:
    1. Start with solved board
    2. Hide all squares except size*size//4 random ones
    3. Add squares one by one until puzzle has unique solution
    """
    # Generate a solved board
    solution = generate_random_solved_board(size)
    puzzle = np.zeros((size, size), dtype=int)
    
    # Start with size*size//4 random squares revealed
    positions = [(r, c) for r in range(size) for c in range(size)]
    random.shuffle(positions)
    min_revealed = int(size * size * min_p_revealed)

    print(attempts_remaining,round(min_p_revealed,2), min_revealed)
    
    for r, c in positions[:min_revealed]:
        puzzle[r, c] = solution[r, c]
    
    # Add squares until we have unique solution
    for r, c in positions[min_revealed:]:
        # Check if current puzzle has unique solution
        constraints = board_to_constraints(puzzle)
        constraints = sort_constraints_by_relatedness(constraints)
        assignments = []
        for constraint in constraints:
            assignments, _ = integrate_new_constraint(assignments, constraint)
            if len(assignments) > blowup_restart:
                return generate_puzzle(size, min_p_revealed + (1/size**2)*(1-min_p_revealed), blowup_restart, attempts_remaining - 1)
            
        if len(assignments) == 1:  # Found unique solution
            break
            
        # Add another square
        puzzle[r, c] = solution[r, c]
    
    return puzzle

def save_boards(boards: List[np.ndarray], filename: str):
    """Save multiple Sudoku boards to a single JSON file."""
    save_dir = os.path.join(os.path.dirname(__file__), 'saved_boards', 'sudoku')
    os.makedirs(save_dir, exist_ok=True)
    
    # Convert boards to lists and numpy integers to Python integers
    data = {
        'boards': [
            {
                'board': board.tolist(),
                'dimensions': tuple(int(x) for x in board.shape),
                'size': int(math.sqrt(board.shape[0]))  # Box size
            }
            for board in boards
        ]
    }
    
    filepath = os.path.join(save_dir, f"{filename}.json")
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

def load_boards(filename: str) -> List[np.ndarray]:
    """Load multiple Sudoku boards from a JSON file."""
    filepath = os.path.join(os.path.dirname(__file__), 'saved_boards', 'sudoku', f"{filename}.json")
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    return [np.array(board_data['board']) for board_data in data['boards']]



if __name__ == "__main__":
    # Generate some test boards
    size = 4
    n_boards = 25
    
    print(f"Generating {n_boards} Sudoku boards of size {size}x{size}...")
    
    boards = []
    for i in range(n_boards):
        try:
            board = generate_puzzle(size)
            print(board)
            boards.append(board)
            print(f"\nGenerated board {i+1}/{n_boards}:")
            print(Game(board))
        except ValueError as e:
            print(f"Failed to generate board {i+1}: {e}")
            continue
    
    # Save boards
    filename = f"sudoku_{size}x{size}"
    save_boards(boards, filename)
    print(f"\nSaved {len(boards)} boards to {filename}.json")
