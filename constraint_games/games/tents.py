import numpy as np
from typing import List, Tuple, Set
from constraints import *
from utils.assignment_utils import *
import random
import os
import json

"""
NOTE: for the game boards, we use the following:
0:     empty cell
1:     tree
-1:    unknown
-3:    tent placed
-4:    marked as no-tent

EXAMPLE board with row/column clues:
board = 
[[ 0  1  0  0]  # 1
 [ 0  0  1  0]  # 1
 [ 0  0  0  1]  # 1
 [ 1  0  0  0]] # 0
 # 1  1  0  1   column clues

row_clues = [1, 1, 1, 0]
col_clues = [1, 1, 0, 1]
"""

def get_adjacent_coords(rows: int, cols: int, row: int, col: int, diagonal: bool = False) -> List[Tuple[int, int]]:
    """Get coordinates of adjacent squares"""
    adjacent = []
    # Add orthogonally adjacent squares
    for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
        new_r, new_c = row + dr, col + dc
        if 0 <= new_r < rows and 0 <= new_c < cols:
            adjacent.append((new_r, new_c))
            
    # Add diagonally adjacent squares if requested
    if diagonal:
        for dr, dc in [(1, 1), (-1, 1), (1, -1), (-1, -1)]:
            new_r, new_c = row + dr, col + dc
            if 0 <= new_r < rows and 0 <= new_c < cols:
                adjacent.append((new_r, new_c))
                
    return adjacent

def board_to_constraints(board: np.ndarray, row_clues: List[int], col_clues: List[int]) -> List[Constraint]:
    """Convert board state to list of constraints.
    
    Board values:
    1: tree
    -1: unknown/empty cell
    -3: tent placed
    -4: marked as no-tent
    """
    rows, cols = board.shape
    constraints = []
    
    # Create tent variables for all non-tree cells
    tent_vars = {}
    for r in range(rows):
        for c in range(cols):
            if board[r, c] != 1:  # Not a tree
                tent_vars[f"v_{r}_{c}"] = Variable(f"v_{r}_{c}", row=r, col=c)
                adjacent = get_adjacent_coords(rows, cols, r, c, diagonal=False)
                # If not adjacent to any tree, must be 0
                if not any(board[adj_r, adj_c] == 1 for (adj_r, adj_c) in adjacent):
                    constraints.append(EqualityConstraint([tent_vars[f"v_{r}_{c}"]], 0))

    # Add row/column sum constraints
    for r in range(rows):
        row_vars = [tent_vars[f"v_{r}_{c}"] for c in range(cols) if f"v_{r}_{c}" in tent_vars]
        if row_vars:
            constraints.append(EqualityConstraint(row_vars, row_clues[r], row=r, col=None))

    for c in range(cols):
        col_vars = [tent_vars[f"v_{r}_{c}"] for r in range(rows) if f"v_{r}_{c}" in tent_vars]
        if col_vars:
            constraints.append(EqualityConstraint(col_vars, col_clues[c], row=None, col=c))

    # Each tree must have an adjacent tent
    trees = [(r,c) for r in range(rows) for c in range(cols) if board[r,c] == 1]
    for r, c in trees:
        adjacent = get_adjacent_coords(rows, cols, r, c, diagonal=False)
        candidate_vars = [tent_vars[f"v_{adj_r}_{adj_c}"] for (adj_r, adj_c) in adjacent 
                         if f"v_{adj_r}_{adj_c}" in tent_vars]
        if candidate_vars:
            constraints.append(InequalityConstraint(candidate_vars, 0, greater_than=True, row=r, col=c))

    # No adjacent tents (including diagonally)
    for r in range(rows):
        for c in range(cols):
            key = f"v_{r}_{c}"
            if key in tent_vars:
                tent_var = tent_vars[key]
                adjacent = get_adjacent_coords(rows, cols, r, c, diagonal=True)
                adj_vars = [tent_vars[f"v_{adj_r}_{adj_c}"] for (adj_r, adj_c) in adjacent 
                           if f"v_{adj_r}_{adj_c}" in tent_vars]
                for other in adj_vars:
                    constraints.append(InequalityConstraint([tent_var, other], 2, greater_than=False, row=r, col=c))

    return constraints


class Game:
    """Represents a Tents puzzle game state"""
    def __init__(self, board: np.ndarray, row_clues: List[int], col_clues: List[int]):
        self.initial_board = board.copy()
        self.board = board.copy()
        self.row_clues = row_clues
        self.col_clues = col_clues
        self.rows, self.cols = board.shape
        
    def get_adjacent_coords(self, row: int, col: int, diagonal: bool = False) -> List[Tuple[int, int]]:
        """Get coordinates of all valid adjacent squares"""
        return get_adjacent_coords(self.rows, self.cols, row, col, diagonal)
        
    def get_unrevealed_squares(self) -> List[Tuple[int, int]]:
        """Get coordinates of all unrevealed squares"""
        return [(r, c) for r in range(self.rows) for c in range(self.cols)
                if self.board[r, c] in [-1, -3, -4]]
                
    def get_unmarked_squares(self) -> List[Tuple[int, int]]:
        """Get coordinates of all unmarked squares"""
        return [(r, c) for r in range(self.rows) for c in range(self.cols)
                if self.board[r, c] == -1]
                
    def place_tent(self, row: int, col: int):
        """Place a tent at the specified position"""
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            raise ValueError(f"Position ({row}, {col}) is out of bounds")
        if self.board[row, col] == 1:  # Can't place tent on a tree
            raise ValueError(f"Cannot place tent on a tree at ({row}, {col})")
        self.board[row, col] = -3
        
    def mark_no_tent(self, row: int, col: int):
        """Mark a cell as not containing a tent"""
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            raise ValueError(f"Position ({row}, {col}) is out of bounds")
        if self.board[row, col] == 1:  # Can't mark a tree
            raise ValueError(f"Cannot mark a tree at ({row}, {col})")
        self.board[row, col] = -4

    def unmark(self, row: int, col: int):
        """Remove tent or no-tent mark from a cell"""
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            raise ValueError(f"Position ({row}, {col}) is out of bounds")
        if self.board[row, col] not in [-3, -4]:  # Can only unmark tents and no-tent marks
            raise ValueError(f"Cannot unmark cell at ({row}, {col})")
        self.board[row, col] = -1
        
    def get_board_state(self):
        """Return current board state"""
        return self.board.copy()
    
    def is_solved(self):
        """Check if the game is solved"""
        # Check if all cells are marked
        if np.any(self.board == -1):
            return False
            
        # Check row clues
        for r in range(self.rows):
            if np.sum(self.board[r] == -3) != self.row_clues[r]:
                return False
                
        # Check column clues
        for c in range(self.cols):
            if np.sum(self.board[:, c] == -3) != self.col_clues[c]:
                return False
                
        # Check no adjacent tents
        for r in range(self.rows):
            for c in range(self.cols):
                if self.board[r, c] == -3:  # Tent
                    adjacent = self.get_adjacent_coords(r, c, diagonal=True)
                    if any(self.board[adj_r, adj_c] == -3 for adj_r, adj_c in adjacent):
                        return False
        
        # Find all trees and tents
        trees = [(r,c) for r in range(self.rows) for c in range(self.cols) if self.board[r,c] == 1]
        tents = [(r,c) for r in range(self.rows) for c in range(self.cols) if self.board[r,c] == -3]
        
        if len(trees) != len(tents):
            return False
        
        # Build adjacency graph between trees and tents
        adjacency = {}  # (tree_r,tree_c) -> list of adjacent tent positions
        for tree_r, tree_c in trees:
            adjacent = self.get_adjacent_coords(tree_r, tree_c, diagonal=False)
            adjacency[(tree_r,tree_c)] = [
                (r,c) for (r,c) in adjacent 
                if (r,c) in tents
            ]
        
        # Try to find a complete matching using DFS
        def find_matching(tree_idx, used_tents, matching):
            if tree_idx == len(trees):
                return True
            
            tree = trees[tree_idx]
            for tent in adjacency[tree]:
                if tent not in used_tents:
                    used_tents.add(tent)
                    matching[tree] = tent
                    if find_matching(tree_idx + 1, used_tents, matching):
                        return True
                    used_tents.remove(tent)
                    matching.pop(tree)
            return False
        
        matching = {}
        used_tents = set()
        return find_matching(0, used_tents, matching)
    
    def reset(self):
        """Reset the board to initial state"""
        self.board = self.initial_board.copy()
        
    def __str__(self) -> str:
        """Return string representation of the board"""
        symbols = {
            -1: '?',  # unknown/empty
            1: '↟',   # tree
            -3: '⧍',  # tent
            -4: 'X'   # no tent
        }
        
        # Convert board to string representation
        rows = []
        for r in range(self.rows):
            row = [symbols.get(cell, str(cell)) for cell in self.board[r]]
            rows.append(' '.join(row) + f" {self.row_clues[r]}")
        
        # Add column clues at bottom
        rows.append(' '.join(str(clue) for clue in self.col_clues))
        
        return  '\n'.join(rows) + '\n'
        
    def __repr__(self) -> str:
        return f"Game(\n{str(self)}\n)" 

def has_unique_solution(board: np.ndarray, row_clues: List[int], col_clues: List[int]) -> bool:
    """Check if the board has exactly one solution"""
    constraints = board_to_constraints(board, row_clues, col_clues)
    constraints = sort_constraints_by_relatedness(constraints)
    solutions, _ = integrate_constraints(constraints)
    if solutions is None:
        return False
    return len(solutions) == 1

def has_valid_solution(board: np.ndarray, row_clues: List[int], col_clues: List[int]) -> bool:
    """Check if the board has a valid solution"""
    constraints = board_to_constraints(board, row_clues, col_clues)
    constraints = sort_constraints_by_relatedness(constraints)
    solutions, _ = integrate_constraints(constraints)
    return solutions is not None

def try_generate_board(max_placement_attempts = 100):
    """Try to generate a valid board configuration."""
    board = np.full((rows, cols), -1, dtype=int)  # Initialize with -1 for unknown
    valid_tent_locations = np.ones((rows, cols), dtype=bool)
    tents = set()
    trees = set()
    
    # First place all tents (ensuring no adjacency)
    for _ in range(n_tents):
        # Find valid tent locations (not adjacent to other tents)
        valid_indices = np.argwhere(valid_tent_locations)
        if len(valid_indices) == 0:
            return None, None
            
        # Place tent
        tent_r, tent_c = valid_indices[random.randint(0, len(valid_indices) - 1)]
        tents.add((tent_r, tent_c))
        
        # Update validity mask - no tents can be adjacent to this tent
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                nr, nc = tent_r + dr, tent_c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    valid_tent_locations[nr, nc] = False
    
    # Then place trees next to tents
    for tent_r, tent_c in tents:
        # Find valid tree positions adjacent to this tent
        adjacent = [(tent_r + dr, tent_c + dc) for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]]
        valid_positions = [
            (r, c) for r, c in adjacent 
            if 0 <= r < rows and 0 <= c < cols and board[r, c] == -1  # Changed from 0 to -1
        ]
        
        if not valid_positions:
            return None, None
            
        # Place tree
        tree_r, tree_c = random.choice(valid_positions)
        board[tree_r, tree_c] = 1
        trees.add((tree_r, tree_c))
    
    return board, tents

def generate_random_tents_game(
    rows: int, 
    cols: int, 
    n_tents: int, 
    max_board_attempts: int = 5000,
    max_placement_attempts: int = 100,
    require_unique: bool = True):
    """Generate a random valid Tents puzzle."""
    
    # Quick sanity checks
    if n_tents > (rows * cols) / 2:  # Need space for both tents and trees
        raise ValueError("Too many tents for board size")
    
    uniqueness_attempts = 0
    max_uniqueness_attempts = 100  # Limit attempts to find unique solution
    
    for attempt in range(max_board_attempts):
        board, tents = try_generate_board(max_placement_attempts)
        if board is not None:
            row_clues = [sum(1 for c in range(cols) if (r, c) in tents) for r in range(rows)]
            col_clues = [sum(1 for r in range(rows) if (r, c) in tents) for c in range(cols)]
            
            if not require_unique and has_valid_solution(board, row_clues, col_clues):
                return board, row_clues, col_clues
            
            if has_unique_solution(board, row_clues, col_clues):
                return board, row_clues, col_clues
            
            uniqueness_attempts += 1
            if uniqueness_attempts >= max_uniqueness_attempts:
                raise ValueError("Could not find unique solution after max attempts")
    
    raise ValueError("Could not generate valid board after max attempts")

def print_board(board: np.ndarray, row_clues: List[int], col_clues: List[int]):
    """Helper function to print a board with clues"""
    symbols = {
            -1: '?',  # unknown/empty
            1: '↟',   # tree
            -3: '⧍',  # tent
            -4: 'X'   # no tent
        }    
    # Print board with row clues
    for r in range(len(board)):
        row = [symbols[cell] for cell in board[r]]
        print(' '.join(row), f" {row_clues[r]}")
    
    # Print column clues aligned with columns
    print(' '.join(str(clue) for clue in col_clues))
    print()

def save_boards(boards_data: List[Tuple[np.ndarray, List[int], List[int]]], filename: str):
    """Save multiple tents boards to a single JSON file.
    
    Args:
        boards_data: List of (board, row_clues, col_clues) tuples
        filename: Name of file to save (without path or extension)
    """
    # Create directory if it doesn't exist
    save_dir = os.path.join(os.path.dirname(__file__), 'saved_boards', 'tents')
    os.makedirs(save_dir, exist_ok=True)
    
    # Convert boards to lists for JSON serialization
    data = {
        'boards': [
            {
                'board': board.tolist(),
                'row_clues': row_clues,
                'col_clues': col_clues,
                'dimensions': board.shape,
                'n_tents': sum(row_clues)
            }
            for board, row_clues, col_clues in boards_data
        ]
    }
    
    filepath = os.path.join(save_dir, f"{filename}.json")
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

def load_boards(filename: str) -> List[Tuple[np.ndarray, List[int], List[int]]]:
    """Load multiple tents boards from a JSON file.
    
    Args:
        filename: Name of file to load (without path or extension)
        
    Returns:
        List of (board, row_clues, col_clues) tuples
    """
    filepath = os.path.join(os.path.dirname(__file__), 'saved_boards', 'tents', f"{filename}.json")
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    return [
        (
            np.array(board_data['board']),
            board_data['row_clues'],
            board_data['col_clues']
        )
        for board_data in data['boards']
    ]

# Test the generator
if __name__ == "__main__":
    # Configuration
    rows, cols = 5,5
    n_tents = 6



    board, row_clues, col_clues = generate_random_tents_game(rows, cols, n_tents, require_unique=True)
    constraints = board_to_constraints(board, row_clues, col_clues)
    assignments = []


    game = Game(board, row_clues, col_clues)

    print("\nSorting constraints by relatedness...")

    print(game)
    constraints = sort_constraints_by_relatedness(constraints.copy())

    for c in constraints:
        assignments, _ = integrate_new_constraint(assignments, c)
        solved_variables = get_solved_variables(assignments)
        for v in solved_variables:
            if solved_variables[v] == 0:
                game.mark_no_tent(v.row, v.col)
            elif solved_variables[v] == 1:
                game.place_tent(v.row, v.col)

        print(game)

        # print(c)
        # print(assignments)
        # print()

    print(len(constraints))
    print(len(assignments))
    print("")


    n_boards = 25  # Number of boards to generate
    
    print(f"Generating {n_boards} boards of size {rows}x{cols} with {n_tents} tents...")
    
    boards_data = []
    for board_num in range(n_boards):
        try:
            board, row_clues, col_clues = generate_random_tents_game(
                rows, cols, n_tents, require_unique=True
            )
            boards_data.append((board, row_clues, col_clues))
            
            print(f"\nGenerated board {board_num+1}/{n_boards}:")
            print_board(board, row_clues, col_clues)
            
        except ValueError as e:
            print(f"Failed to generate board {board_num+1}: {e}")
            continue
    
    # Save all boards to a single file
    filename = f"tents_{rows}_{cols}_{n_tents}"
    save_boards(boards_data, filename)
    print(f"\nSaved {len(boards_data)} boards to {filename}.json")