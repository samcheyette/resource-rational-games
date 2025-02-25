from itertools import combinations, product
from typing import Dict, Set, Iterator, Tuple
from functools import reduce
from operator import mul
import numpy as np
class Expression:
    """Base class for all expressions in our algebra"""
    def evaluate(self, assignment: Dict['Variable', int]) -> int:
        """Evaluate the expression given an assignment of variables to values"""
        raise NotImplementedError()


    def get_variables(self) -> Set['Variable']:
        """Return set of variables in this expression"""
        raise NotImplementedError()
        
    def __add__(self, other):
        return Sum(self, other if isinstance(other, Expression) else Number(other))
        
    def __sub__(self, other):
        if isinstance(other, Sum):
            return Sum(self, *(-term for term in other.terms))
        return Sum(self, -other if isinstance(other, Expression) else Number(-other))
        
    def __mul__(self, other):
        return Product(self, other if isinstance(other, Expression) else Number(other))

        
    def __radd__(self, other):
        return Sum(Number(other), self)
    
        
    def __eq__(self, other):
        return Equals(self, other if isinstance(other, Expression) else Number(other))

    def __gt__(self, other):
        return GreaterThan(self, other if isinstance(other, Expression) else Number(other))
    
    def __lt__(self, other):
        return LessThan(self, other if isinstance(other, Expression) else Number(other))

    def apply(self, assignment: Dict['Variable', int]) -> 'Expression':
        """Apply variable assignments to the expression"""
        raise NotImplementedError()
    

    def structurally_equal(self, other) -> bool:
        """Compare if two expressions have the same structure"""
        return False  # Default implementation
        
    def __hash__(self):
        """Hash based on structural equality"""
        raise NotImplementedError()
    
    def is_consistent(self, partial_assignment: Dict['Variable', int]) -> int:
        """Check if a partial assignment could be part of a valid solution"""
        raise NotImplementedError()


class Number(Expression):
    """Represents a constant number"""
    def __init__(self, value: int):
        self.value = value
        
    def evaluate(self, assignment: Dict['Variable', int]) -> int:
        return self.value
        
    def get_variables(self) -> Set['Variable']:
        return set()
        
    def possible_solutions(self):
        yield {}, 1  # Empty assignment is always valid for a number
            
    def __str__(self):
        return str(self.value)
    
    def apply(self, assignment: Dict['Variable', int]) -> 'Number':
        return self  # Numbers are unchanged by assignment
    
    def structurally_equal(self, other):
        return isinstance(other, Number) and self.value == other.value
        
    def __hash__(self):
        return hash(self.value)
    
    def is_consistent(self, partial_assignment: Dict['Variable', int]) -> int:
        return 1

class Variable(Expression):
    """A variable that can take discrete values (default binary 0/1)"""
    def __init__(self, name, domain={0,1}, prior_fn=None,**kwargs):
        self.name = name
        if 'row' in kwargs and 'col' in kwargs:
            self.row = kwargs['row']
            self.col = kwargs['col']
        else:
            self.row = None
            self.col = None

        self.value = None
        self.domain = domain

        self.prior = {}
        for value in domain:
            if prior_fn is not None:
                self.prior[value] = prior_fn(value)
            else:
                self.prior[value] = 1
        self.prior = {k: v / sum(self.prior.values()) for k, v in self.prior.items()}

        self.constraints = []
        
    def evaluate(self, assignment: Dict['Variable', int]) -> int:
        return assignment.get(self, 0)
            
    def get_variables(self) -> Set['Variable']:
        return {self}
    
    def add_constraint(self, constraint):
        self.constraints.append(constraint)

    def remove_constraint(self, constraint):
        self.constraints.remove(constraint)
    
    def assign(self, value):
        """Assign a value to this variable"""
        if value is not None and value not in self.domain:
            raise ValueError(f"Value {value} is not in domain {self.domain}")
        self.value = value
            
    def unassign(self):
        """Unassign this variable's value"""
        self.value = None

    def is_assigned(self):
        return self.value is not None

    def possible_solutions(self) -> Iterator[Dict['Variable', int]]:
        """By default, yield both possible values for binary variable"""
        if self.value is not None:
            yield {self: self.value}
        else:
            for value in self.domain:
                yield {self: value}
        
    def __str__(self):
        if self.value is not None:
            return f"{self.value}"
        return self.name
        
    def __repr__(self):
        if self.value is not None:
            return f"{self.name}={self.value}"
        return self.name
        
    def __hash__(self):
        return hash(self.name)
        
    def __eq__(self, other):
        if isinstance(other, Variable):
            return self.name == other.name 

    def __lt__(self, other):
        if isinstance(other, Variable):
            return self.name < other.name
        return NotImplemented
        
    def __le__(self, other):
        if isinstance(other, Variable):
            return self.name <= other.name
        return NotImplemented
        
    def __gt__(self, other):
        if isinstance(other, Variable):
            return self.name > other.name
        return NotImplemented
        
    def __ge__(self, other):
        if isinstance(other, Variable):
            return self.name >= other.name
        return NotImplemented
    
    
    def apply(self, assignment: Dict['Variable', int]) -> 'Expression':
        if self in assignment:
            return Number(assignment[self])
        return self
    
    def structurally_equal(self, other):
        return isinstance(other, Variable) and self.name == other.name
        
    def __hash__(self):
        return hash(self.name)
    
    def is_consistent(self, partial_assignment: Dict['Variable', int]) -> int:
        if self in partial_assignment:
            if self.value is not None:
                return 1 if partial_assignment[self] == self.value else 0
            else:
                return 1 if partial_assignment[self] in self.domain else 0
        return 1
    
    def get_neighbor_variables(self, constraint_subset=None) -> Set['Variable']:
        """Get unassigned variables that share constraints with this variable"""
        if self.value is not None:  # If this variable is assigned, it has no neighbors
            return set()
            
        neighbors = set()
        relevant_constraints = (self.constraints if constraint_subset is None 
                              else [c for c in self.constraints if c in constraint_subset])
        for constraint in relevant_constraints:
            # Only add unassigned variables as neighbors
            neighbors.update(constraint.get_unassigned() - {self} )
        return neighbors
    
    def get_active_constraints(self):
        return [c for c in self.constraints if c.is_active()]
    
    
    def get_shared_constraints(self, other: 'Variable', constraint_subset=None):
        """Get constraints that contain both this variable and other"""
        my_constraints = set(self.constraints if constraint_subset is None 
                           else [c for c in self.constraints if c in constraint_subset])
        their_constraints = set(other.constraints if constraint_subset is None 
                              else [c for c in other.constraints if c in constraint_subset])
        return my_constraints & their_constraints
    

    def get_constraint_overlap(self, other: 'Variable') -> float:
        """Get fraction of constraints shared with another variable"""
        shared = len(self.get_shared_constraints(other))
        total = len(set(self.constraints) | set(other.constraints))
        return shared / total if total > 0 else 0.0
    
    def get_constraint_count(self) -> int:
        """Get number of constraints this variable appears in"""
        return len(self.constraints)
    
    def get_entropy(self) -> float:
        """Get entropy of this variable"""
        return -sum(p * np.log2(p) for p in self.prior.values())
    

class VariableState:
    """Class to save and restore the state of a set of variables"""
    def __init__(self, variables):
        """Save current state of variables"""
        self.saved_values = {var: var.value for var in variables}
        
    def restore(self):
        """Restore variables to their saved values"""
        for var, value in self.saved_values.items():
            var.assign(value)
            
    def __str__(self):
        return str({var.name: val for var, val in self.saved_values.items()})



class Sum(Expression):
    """Represents sum of expressions"""
    def __init__(self, *terms: Expression):
        # Flatten nested sums during construction
        flat_terms = []
        for term in terms:
            if isinstance(term, Sum):
                flat_terms.extend(term.terms)
            else:
                flat_terms.append(term)
        self.terms = tuple(flat_terms)
    
    def evaluate(self, assignment: Dict[Variable, int]) -> int:
        return sum(term.evaluate(assignment) for term in self.terms)
        
    def get_variables(self) -> Set[Variable]:
        return set().union(*(term.get_variables() for term in self.terms))
        
    def __str__(self):
        return f"({' + '.join(str(term) for term in self.terms)})"
    
    def apply(self, assignment: Dict['Variable', int]) -> 'Expression':
        new_terms = [term.apply(assignment) for term in self.terms]
        # If all terms are numbers, compute the sum
        if all(isinstance(term, Number) for term in new_terms):
            return Number(sum(term.value for term in new_terms))
        # Filter out zero terms
        new_terms = [term for term in new_terms if not (isinstance(term, Number) and term.value == 0)]
        if not new_terms:
            return Number(0)
        if len(new_terms) == 1:
            return new_terms[0]
        return Sum(*new_terms)
    
    def structurally_equal(self, other):
        return isinstance(other, Sum) and self.terms == other.terms
        
    def __hash__(self):
        return hash(self.terms)






class Equals(Expression):
    """Represents equality between two expressions"""
    def __init__(self, left: Expression, right: Expression):
        self.left = left
        self.right = right
        
    def evaluate(self, assignment: Dict[Variable, int]) -> int:
        left_val = self.left.evaluate(assignment)
        right_val = self.right.evaluate(assignment)
        return 1 if left_val == right_val else 0
            
    def get_variables(self) -> Set[Variable]:
        return self.left.get_variables() | self.right.get_variables()
        
    def __str__(self):
        return f"{self.left} = {self.right}"

    def __repr__(self) -> str:
        return f"{self.left} = {self.right}"
    
    def apply(self, assignment: Dict['Variable', int]) -> 'Expression':
        new_left = self.left.apply(assignment)
        new_right = self.right.apply(assignment)
        if isinstance(new_left, Number) and isinstance(new_right, Number):
            return Number(1 if new_left.value == new_right.value else 0)
        return Equals(new_left, new_right)
    
    def structurally_equal(self, other):
        return isinstance(other, Equals) and self.left.structurally_equal(other.left) and self.right.structurally_equal(other.right)
        
    def __hash__(self):
        return hash((self.left, self.right))

    def is_consistent(self, partial_assignment: Dict['Variable', int]) -> int:
        """Check if a partial assignment could lead to a valid solution"""
        left_vars = self.left.get_variables()
        right_vars = self.right.get_variables()
        
        # Calculate unassigned variables for each side separately
        left_unassigned = len([var for var in left_vars if var not in partial_assignment])
        right_unassigned = len([var for var in right_vars if var not in partial_assignment])
        
        # Evaluate current values
        left_val = self.left.evaluate(partial_assignment)
        right_val = self.right.evaluate(partial_assignment)
        
        # If left > right, we need enough unassigned variables on the right to catch up
        if left_val > right_val:
            if right_unassigned < left_val - right_val:
                return 0
        # If right > left, we need enough unassigned variables on the left to catch up
        elif right_val > left_val:
            if left_unassigned < right_val - left_val:
                return 0
                
        return 1

class GreaterThan(Expression):
    """Represents left > right"""
    def __init__(self, left: Expression, right: Expression):
        self.left = left
        self.right = right
        
    def evaluate(self, assignment: Dict[Variable, int]) -> int:
        left_val = self.left.evaluate(assignment)
        right_val = self.right.evaluate(assignment)
        return 1 if left_val > right_val else 0
            
    def get_variables(self) -> Set[Variable]:
        return self.left.get_variables() | self.right.get_variables()
        
    def __str__(self):
        return f"{self.left} > {self.right}"

    def __repr__(self) -> str:
        return f"{self.left} > {self.right}"
    
    def apply(self, assignment: Dict['Variable', int]) -> 'Expression':
        new_left = self.left.apply(assignment)
        new_right = self.right.apply(assignment)
        if isinstance(new_left, Number) and isinstance(new_right, Number):
            return Number(1 if new_left.value > new_right.value else 0)
        return GreaterThan(new_left, new_right)
    
    def structurally_equal(self, other):
        return isinstance(other, GreaterThan) and self.left.structurally_equal(other.left) and self.right.structurally_equal(other.right)
        
    def __hash__(self):
        return hash((self.left, self.right))

    def is_consistent(self, partial_assignment: Dict['Variable', int]) -> int:
        """Check if a partial assignment could lead to a valid solution"""
        left_vars = self.left.get_variables()
        right_vars = self.right.get_variables()
        
        # Calculate unassigned variables for each side separately
        left_unassigned = len([var for var in left_vars if var not in partial_assignment])
        right_unassigned = len([var for var in right_vars if var not in partial_assignment])
        
        # Evaluate current values
        left_val = self.left.evaluate(partial_assignment)
        right_val = self.right.evaluate(partial_assignment)
        
        # If left is already greater, it's consistent (can only increase)
        if left_val > right_val:
            return 1
            
        # If right is greater or equal, we need enough unassigned variables on the left
        # and few enough on the right to make left > right possible
        left_max = left_val + left_unassigned
        right_min = right_val
        
        return 1 if left_max > right_min else 0

class LessThan(Expression):
    """Represents left < right by wrapping GreaterThan"""
    def __init__(self, left: Expression, right: Expression):
        self.left = left
        self.right = right
        self._gt = GreaterThan(right, left)  # Flip the comparison
        
    def evaluate(self, assignment: Dict[Variable, int]) -> int:
        return self._gt.evaluate(assignment)
            
    def get_variables(self) -> Set[Variable]:
        return self.left.get_variables() | self.right.get_variables()
        
    def __str__(self):
        return f"{self.left} < {self.right}"

    def __repr__(self) -> str:
        return f"{self.left} < {self.right}"
    
    def apply(self, assignment: Dict['Variable', int]) -> 'Expression':
        new_left = self.left.apply(assignment)
        new_right = self.right.apply(assignment)
        if isinstance(new_left, Number) and isinstance(new_right, Number):
            return Number(1 if new_left.value < new_right.value else 0)
        return LessThan(new_left, new_right)
    
    def structurally_equal(self, other):
        return isinstance(other, LessThan) and self.left.structurally_equal(other.left) and self.right.structurally_equal(other.right)
        
    def __hash__(self):
        return hash((self.left, self.right))

    def is_consistent(self, partial_assignment: Dict['Variable', int]) -> int:
        return self._gt.is_consistent(partial_assignment)




if __name__ == "__main__":
    # Create some variables
    v0 = Variable("v0")
    v1 = Variable("v1")
    v2 = Variable("v2")
    v3 = Variable("v3")

    S = Sum(v0, v1, v2, v3)

    eq = Equals(S, Number(2))
    

    # Additional tests for other expressions
    # eq = And(v0 + v1 == 1, v0 + v2 == 1)


    # print(eq)

    # vars_list = [v0, v1, v2]

    # print("\nEquation Consistency Tests:")
    # for bits in product([0, 1], repeat=len(vars_list)):
    #     assignment = {var: bit for var, bit in zip(vars_list, bits)}
    #     print(f"Assignment {assignment}, eq is consistent: {eq.is_consistent(assignment)}")
