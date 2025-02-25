from itertools import combinations, product
from collections import defaultdict
from typing import Dict, Set, Iterator, Tuple, List, Any
from grammar import Expression, Variable, Sum, Number, GreaterThan, LessThan
from math import comb
from utils.utils import *
from utils.assignment_utils import *

import random
from collections import defaultdict
import numpy as np
import copy
import time

def get_overlapping_constraints(all_constraints, assignments=[], partial_path=[]):
    remaining_constraints = [c for c in all_constraints if c not in partial_path]
    if len(assignments) == 0 and len(partial_path) == 0:

        return [(c,1) for c in remaining_constraints if len(c.get_unassigned()) > 0]

    # Get variables from assignments and path
    assignment_vars = set().union(*[set(a.keys()) for a in assignments]) if assignments else set()
    path_vars = set().union(*[c.get_unassigned() for c in partial_path]) if partial_path else set()
    all_vars = assignment_vars | path_vars

    # Score remaining constraints by overlap
    overlap_scores = []
    for constraint in remaining_constraints:
        constraint_vars = constraint.get_unassigned()
        if not constraint_vars.isdisjoint(all_vars):
            shared = len(constraint_vars & all_vars)
            total = len(constraint_vars | all_vars)
            overlap_scores.append((constraint, shared/total))

    return sorted(overlap_scores, key=lambda x: x[1], reverse=True)


def sort_constraints_by_relatedness(constraints):
    """Sort constraints to minimize introduction of new variables at each step.
    
    At each step, chooses the constraint that adds the fewest new variables
    relative to all variables seen in previous constraints.
    """
    if not constraints:
        return []
    
    # Convert input to list to avoid modifying the original
    constraints = list(constraints)
    
    # Start with smallest constraint
    sorted_constraints = []
    seen_variables = set()
    
    while constraints:  # Continue until all constraints are used
        # Find constraint that introduces fewest new variables
        best_score = (float('inf'), float('inf'))  # (new_vars, total_size)
        best_next = None
        best_idx = None
        
        for i, c in enumerate(constraints):
            # Count new variables this constraint would add
            c_vars = c.get_variables()
            new_vars = len(c_vars - seen_variables)
            total_size = len(c_vars)
            score = (new_vars, total_size)
            
            if score < best_score:
                best_score = score
                best_next = c
                best_idx = i
        
        if best_next is None:  # Should never happen as constraints is not empty
            raise ValueError("Failed to find next constraint")
            
        # Add the chosen constraint and update seen variables
        sorted_constraints.append(best_next)
        seen_variables.update(best_next.get_variables())
        constraints.pop(best_idx)  # Remove the used constraint
    
    return sorted_constraints



class Constraint:
    """Base class for all constraints"""
    def __init__(self, variables: Set[Variable], target: int, **kwargs):
        self.variables = set(variables)
        self.target = target
        
        self.row, self.col = None, None
        if "row" in kwargs:
            self.row = kwargs["row"]
        if "col" in kwargs:
            self.col = kwargs["col"]

        for var in variables:
            var.add_constraint(self)

    def __del__(self):
        """Cleanup when constraint is destroyed"""
        for var in self.variables:
            var.remove_constraint(self)

    def get_variables(self) -> Set[Variable]:
        """Return set of variables in this constraint"""
        return self.variables

    def get_unassigned(self) -> Set[Variable]:
        """Return set of unassigned variables"""
        return {var for var in self.variables if var.value is None}

    def get_assigned(self) -> Set[Variable]:
        """Return set of assigned variables"""
        return {var for var in self.variables if var.value is not None}

    def is_active(self):
        """Check if constraint has any unassigned variables"""
        return len(self.get_unassigned()) > 0

    def get_neighbor_constraints(self, constraint_subset=None):
        """Get constraints that share variables with this constraint"""
        neighbors = set()
        for var in self.get_unassigned():
            relevant_constraints = (var.constraints if constraint_subset is None
                                  else [c for c in var.constraints if c in constraint_subset])
            for constraint in relevant_constraints:
                if (constraint != self and
                    not self.get_unassigned().isdisjoint(constraint.get_unassigned())):
                    neighbors.add(constraint)
        return neighbors

    def get_constraint_degree(self, constraint_subset=None) -> int:
        """Get number of other constraints this constraint shares variables with"""
        return len(self.get_neighbor_constraints(constraint_subset))

    def get_shared_variables(self, other: 'Constraint') -> Set[Variable]:
        """Get variables shared between this constraint and another"""
        return self.get_unassigned() & other.get_unassigned()

    def get_variable_overlap(self, other: 'Constraint') -> float:
        """Get fraction of variables shared with another constraint"""
        shared = len(self.get_shared_variables(other))
        total = len(self.get_unassigned() | other.get_unassigned())
        return shared / total if total > 0 else 0.0

    def get_effective_target(self, partial_assignment = None) -> int:
        """Return the effective target for this constraint"""
        effective_target = self.target - sum(var.value for var in self.get_assigned())
        unassigned = self.get_unassigned()
        if partial_assignment is not None:
            for key in partial_assignment:
                if key in unassigned:
                    effective_target -= partial_assignment[key]
        return effective_target

    def evaluate(self, assignment):
        """Evaluate if the constraint is satisfied by a complete assignment"""
        raise NotImplementedError()

    def is_consistent(self, partial_assignment):
        """Check if a partial assignment could lead to a valid solution"""
        raise NotImplementedError()

    def possible_solutions(self, partial_assignment=None):
        """Generate all possible solutions that satisfy the constraint"""
        raise NotImplementedError()

    def test_contradiction(self):
        """Test if current assignments make constraint impossible to satisfy"""
        raise NotImplementedError()

    def copy(self):
        """Create a copy of this constraint"""
        raise NotImplementedError()


class EqualityConstraint(Constraint):
    """Represents a constraint where sum of variables equals a target value"""
    def __init__(self, variables: Set[Variable], target: int, **kwargs):
        super().__init__(variables, target, **kwargs)
        self.sum_expr = Sum(*self.variables)

    def __str__(self) -> str:
        parts = []
        for v in self.variables:
            parts.append(str(v))
        return f"({' + '.join(parts)}) = {self.target}"

    def __repr__(self) -> str:
        return str(self)

    def __hash__(self):
        return hash(('equality', frozenset(self.variables), self.target))

    def __eq__(self, other):
        return (isinstance(other, EqualityConstraint) and
                self.variables == other.variables and
                self.target == other.target)

    def copy(self):
        return EqualityConstraint(self.variables.copy(), self.target)

    def evaluate(self, assignment):
        """Evaluate if the constraint is satisfied by a complete assignment"""
        return 1 if self.sum_expr.evaluate(assignment) == self.get_effective_target() else 0

    def is_consistent(self, partial_assignment):
        """Check if a partial assignment could lead to a valid solution"""
        initial_sum = 0
        for v in self.get_assigned():
            if v in partial_assignment and v.value != partial_assignment[v]:
                return 0
            initial_sum += v.value

        unassigned = self.get_unassigned()
        remaining = len(unassigned)
        for v in unassigned:
            if v in partial_assignment:
                initial_sum += partial_assignment[v]
                remaining -= 1

        return 1 if initial_sum <= self.target <= initial_sum + remaining else 0

    def test_contradiction(self):
        """Test if current assignments make constraint impossible to satisfy"""
        effective_target = self.get_effective_target()
        n_unassigned = len(self.get_unassigned())
        return effective_target < 0 or effective_target > n_unassigned

    def possible_solutions(self, partial_assignment=None):
        assignment = {} if partial_assignment is None else partial_assignment.copy()
        unassigned = self.get_unassigned()
        vars_to_assign = unassigned - set(assignment.keys())

        if not self.is_consistent(assignment):
            return

        if not vars_to_assign:
            if self.evaluate(assignment):
                yield assignment.copy()
            return

        target = self.get_effective_target(assignment)
        if target < 0 or target > len(vars_to_assign):
            return

        for ones_positions in combinations(sorted(vars_to_assign), target):
            solution = assignment.copy()
            for var in vars_to_assign:
                solution[var] = 1 if var in ones_positions else 0

            yield solution


class InequalityConstraint(Constraint):
    """Represents sum(variables) < target or sum(variables) > target"""
    def __init__(self, variables, target, greater_than=False, **kwargs):
        super().__init__(variables, target, **kwargs)
        self.greater_than = greater_than
        self.sum_expr = Sum(*variables)

    def __str__(self):
        parts = []
        for v in self.variables:
            parts.append(str(v))
        symbol = ">" if self.greater_than else "<"
        return f"({' + '.join(parts)}) {symbol} {self.target}"

    def __repr__(self):
        return str(self)

    def __hash__(self):
        return hash((frozenset(self.variables), self.target, self.greater_than))

    def __eq__(self, other):
        return (isinstance(other, InequalityConstraint) and
                self.variables == other.variables and
                self.target == other.target and
                self.greater_than == other.greater_than)

    def copy(self):
        return InequalityConstraint(self.variables.copy(), self.target, self.greater_than)

    def evaluate(self, assignment):
        total = self.sum_expr.evaluate(assignment)
        effective_target = self.get_effective_target()
        return 1 if (total > effective_target if self.greater_than else total < effective_target) else 0

    def is_consistent(self, assignment):
        current_sum = 0
        for v in self.variables:
            if v in assignment:
                current_sum += assignment[v]
            elif v.value is not None:
                current_sum += v.value

        remaining_vars = len(self.get_unassigned() - set(assignment.keys()))

        if self.greater_than:
            max_possible_sum = current_sum + remaining_vars
            return 1 if max_possible_sum > self.target else 0
        else:
            return 1 if current_sum < self.target else 0

    def test_contradiction(self):
        effective_target = self.get_effective_target()
        n_unassigned = len(self.get_unassigned())

        if self.greater_than:
            max_possible = n_unassigned
            return max_possible <= effective_target
        else:
            return effective_target <= 0

    def possible_solutions(self, starting_assignment=None):
        assignment = {} if starting_assignment is None else starting_assignment.copy()
        unassigned = self.get_unassigned()
        vars_to_assign = unassigned - set(assignment.keys())

        if not self.is_consistent(assignment):
            return

        if not vars_to_assign:
            if self.evaluate(assignment):
                yield assignment.copy()
            return

        current_sum = 0
        for v in self.variables:
            if v in assignment:
                current_sum += assignment[v]
            elif v.value is not None:
                current_sum += v.value

        if self.greater_than:
            # Need enough ones to make total > target
            remaining_needed = max(0, self.target + 1 - current_sum)
            if remaining_needed > len(vars_to_assign):
                return  # Impossible to satisfy
            min_ones = remaining_needed
            max_ones = len(vars_to_assign)
        else:
            # Need few enough ones to keep total < target
            min_ones = 0
            max_ones = min(self.target - current_sum, len(vars_to_assign))

        for n_ones in range(min_ones, max_ones + 1):
            for ones_positions in combinations(sorted(vars_to_assign), n_ones):
                solution = assignment.copy()
                for var in vars_to_assign:
                    solution[var] = 1 if var in ones_positions else 0
                if self.evaluate(solution):
                    yield solution



class UniquenessConstraint(Constraint):
    """Constraint that requires all variables to have different values, excluding given constants."""
    
    def __init__(self, variables, constants=None, **kwargs):
        """Initialize constraint.
        
        Args:
            variables: List of variables that must have unique values
            constants: List of values that are already used/fixed
        """
        # Call parent constructor with just the variables
        super().__init__(variables, target=None)
        
        # Store constants separately
        self.constants = set(constants) if constants is not None else set()
        if "row" in kwargs:
            self.row = kwargs["row"]
        if "col" in kwargs:
            self.col = kwargs["col"]
        
        # Verify all variables have same domain
        domains = {frozenset(v.domain) for v in variables}
        if len(domains) != 1:
            raise ValueError("All variables in UniquenessConstraint must have same domain")
        
        # Verify constants are valid
        if self.constants:
            domain = next(iter(domains))  # We know all domains are same
            invalid = self.constants - domain
            if invalid:
                raise ValueError(f"Constants {invalid} not in variable domain {domain}")
            

    def __eq__(self, other):
        return (isinstance(other, UniquenessConstraint) and
                self.variables == other.variables and
                self.constants == other.constants)
    
    def __hash__(self):
        return hash(('uniqueness', frozenset(self.variables), frozenset(self.constants)))
    
    def __repr__(self):
        return self.__str__()
    
    def __str__(self):
        """Return string representation using ≠ notation."""
        vars_str = " ≠ ".join(str(v) for v in sorted(self.variables, key=str))
        if self.constants:
            # Add constants at the end
            vars_str += f" ≠ {{{','.join(map(str, sorted(self.constants)))}}}"
        return vars_str

        
    def copy(self):
        """Return a copy of this constraint"""
        return UniquenessConstraint(self.variables.copy(), constants=self.constants.copy())
    
    def evaluate(self, assignment):
        """Check if all assigned variables have different values and don't use constants."""
        values = [assignment[v] for v in self.variables if v in assignment]
        # Check no duplicates and no overlap with constants
        return (len(values) == len(set(values)) and 
                not (set(values) & self.constants))
    
    def possible_solutions(self, starting_assignment=None):
        """Generate all possible assignments satisfying uniqueness."""
        assignment = {} if starting_assignment is None else starting_assignment.copy()
        
        # Don't add already assigned variables
        unassigned = self.get_unassigned() - set(assignment.keys())
        
        if not self.is_consistent(assignment):
            return
            
        if not unassigned:
            if self.evaluate(assignment):
                yield assignment.copy()
            return
        
        # Get available values (domain minus used values, constants, and assigned values)
        used_values = {assignment[v] for v in self.variables if v in assignment}
        used_values.update(v.value for v in self.variables if v.value is not None)  # Add assigned values
        
        first_var = next(iter(unassigned))
        available_values = first_var.domain - used_values - self.constants
        
        for value in available_values:
            new_assignment = assignment.copy()
            new_assignment[first_var] = value
            yield from self.possible_solutions(new_assignment)
    
    def is_consistent(self, assignment):
        """Check if current partial assignment could lead to solution."""
        # Get values from both assignment and assigned variables
        values = [assignment[v] for v in self.variables if v in assignment]
        values.extend(v.value for v in self.variables if v.value is not None)
        
        return (len(values) == len(set(values)) and  # No duplicates
                not (set(values) & self.constants))   # No overlap with constants

    def test_contradiction(self):
        """Test if current assignments make constraint impossible to satisfy"""
        # Check if any assigned variables have same value
        values = [v.value for v in self.get_assigned()]
        if len(values) != len(set(values)):
            return True
        # Check if any assigned values are in constants
        if set(values) & self.constants:
            return True
        # Check if we have enough values left for remaining variables
        remaining_vars = len(self.get_unassigned())
        if remaining_vars > 0:
            domain = next(iter(self.variables)).domain
            available_values = domain - set(values) - self.constants
            if len(available_values) < remaining_vars:
                return True
        return False




if __name__ == "__main__":
    # If x=1, then y+z must equal 1
    v0 = Variable("v0", domain={1, 2, 3, 4})
    v1 = Variable("v1", domain={1, 2, 3, 4})
    v2 = Variable("v2", domain={1, 2, 3, 4})
    v3 = Variable("v3", domain={1, 2, 3, 4})

    constraint = UniquenessConstraint([v0, v1, v2, v3])
    v0.assign(1)
    print(constraint)

    print(list(constraint.possible_solutions({v1:2})))



