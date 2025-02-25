import pytest
from constraints import InequalityConstraint, Variable, EqualityConstraint, UniquenessConstraint
from utils.assignment_utils import integrate_constraints, integrate_new_constraint


def test_constraint_order_independence():
    """Test that order of constraints doesn't affect final solutions"""
    # Create some test variables
    v1 = Variable("v1")
    v2 = Variable("v2")
    v3 = Variable("v3")
    
    # Create some test constraints
    c1 = InequalityConstraint([v1, v2], 1, greater_than=False)  # v1 + v2 < 1
    c2 = InequalityConstraint([v2, v3], 1, greater_than=True)   # v2 + v3 > 1
    c3 = EqualityConstraint([v1, v3], 1)  # v1 + v3 = 1
    
    def try_constraints(constraints):
        assignments = []
        print(f"\nTrying constraints in order:")
        for c in constraints:
            print(f"\nApplying {c}:")
            print(f"Previous assignments: {assignments}")
            assignments, _ = integrate_new_constraint(assignments, c)
            print(f"New assignments: {assignments}")
        return assignments
    
    print("\nOrder 1:")
    solutions1 = try_constraints([c1, c2, c3])
    
    print("\nOrder 2:")
    solutions2 = try_constraints([c2, c3, c1])
    
    print("\nOrder 3:")
    solutions3 = try_constraints([c3, c1, c2])

def test_constraint_order_independence_complex():
    """Test order independence with more complex constraint combinations"""
    # Create a grid of variables
    vars = {}
    for i in range(2):
        for j in range(2):
            vars[f"v_{i}_{j}"] = Variable(f"v_{i}_{j}")
    
    # Create row sum constraints
    row0 = EqualityConstraint([vars["v_0_0"], vars["v_0_1"]], 1)
    row1 = EqualityConstraint([vars["v_1_0"], vars["v_1_1"]], 1)
    
    # Create column sum constraints
    col0 = EqualityConstraint([vars["v_0_0"], vars["v_1_0"]], 1)
    col1 = EqualityConstraint([vars["v_0_1"], vars["v_1_1"]], 1)
    
    # Create diagonal constraint
    diag = InequalityConstraint([vars["v_0_0"], vars["v_1_1"]], 1, greater_than=True)
    
    constraints = [row0, row1, col0, col1, diag]
    
    from utils.assignment_utils import integrate_constraints
    import itertools
    
    all_solutions = []
    for ordering in itertools.permutations(constraints):
        print(f"\nTrying order: {[str(c) for c in ordering]}")
        solutions, _ = integrate_constraints(ordering)
        solutions = sorted([frozenset(a.items()) for a in solutions])
        all_solutions.append(solutions)
        print(f"Solutions: {solutions}")
    
    # All orderings should give same solutions
    first_solution = all_solutions[0]
    for i, solution in enumerate(all_solutions[1:], 1):
        assert solution == first_solution, f"Different solutions for ordering {i}"

def test_constraint_order_with_contradictions():
    """Test that order of constraints gives same result with contradictions"""
    
    # Create variables
    v0 = Variable("v0")
    v1 = Variable("v1") 
    v2 = Variable("v2")
    v3 = Variable("v3")

    # Create contradictory constraints
    c1 = InequalityConstraint([v0, v1, v2], 1, greater_than=True)   # v0 + v1 + v2 > 1
    c2 = InequalityConstraint([v2, v3], 1, greater_than=True)       # v2 + v3 > 1
    c3 = InequalityConstraint([v0, v1, v2, v3], 2, greater_than=False)  # v0 + v1 + v2 + v3 < 2
    
    # Try different orderings
    orderings = [
        [c1, c2, c3],
        [c2, c3, c1],
        [c3, c1, c2],
        [c3, c2, c1]
    ]

    results = []
    for i, ordering in enumerate(orderings):
        print(f"\n\nTrying order {i+1}: {[str(c) for c in ordering]}")
        assignments = []
        for j, constraint in enumerate(ordering):
            print(f"\nStep {j+1}: Applying {constraint}")
            print(f"Current assignments: {assignments}")
            assignments, _ = integrate_new_constraint(assignments, constraint)
            print(f"After integration: {assignments}")
            if assignments is None:
                print("Hit contradiction!")
                break
        results.append(assignments)

    # Check all results are the same (either all None or all same solutions)
    first_result = results[0]
    for i, result in enumerate(results[1:], 1):
        assert result == first_result, f"Different results for ordering {i}: {result} vs {first_result}"

def test_constraint_order_non_contradictory():
    """Test different orderings of compatible constraints"""
    
    # Create variables
    v0 = Variable("v0")
    v1 = Variable("v1") 
    v2 = Variable("v2")
    v3 = Variable("v3")

    # Create compatible constraints
    c1 = InequalityConstraint([v0, v1], 0, greater_than=True)    # v0 + v1 > 0
    c2 = InequalityConstraint([v1, v2], 1, greater_than=False)   # v1 + v2 < 1
    c3 = EqualityConstraint([v2, v3], 1)                         # v2 + v3 = 1
    
    # Try different orderings
    orderings = [
        [c1, c2, c3],
        [c2, c3, c1],
        [c3, c1, c2],
        [c3, c2, c1]
    ]

    results = []
    for i, ordering in enumerate(orderings):
        print(f"\n\nTrying order {i+1}: {[str(c) for c in ordering]}")
        assignments = []
        for j, constraint in enumerate(ordering):
            print(f"\nStep {j+1}: Applying {constraint}")
            print(f"Current assignments: {assignments}")
            assignments, _ = integrate_new_constraint(assignments, constraint)
            print(f"After integration: {assignments}")
        results.append(assignments)

        # Print final variable values for this ordering
        if assignments:
            print("\nFinal variable values:")
            for assignment in assignments:
                print(f"Solution: v0={assignment.get(v0)}, v1={assignment.get(v1)}, "
                      f"v2={assignment.get(v2)}, v3={assignment.get(v3)}")

    # Check all results have same solutions (may be in different orders)
    def normalize_solutions(assignments):
        if not assignments:
            return assignments
        return sorted([frozenset(a.items()) for a in assignments])

    first_result = normalize_solutions(results[0])
    for i, result in enumerate(results[1:], 1):
        normalized = normalize_solutions(result)
        assert normalized == first_result, \
            f"Different results for ordering {i}: {normalized} vs {first_result}"

def test_uniqueness_constraint():
    # Create variables with domain {1,2,3,4}
    v1 = Variable("v1", domain={1,2,3,4})
    v2 = Variable("v2", domain={1,2,3,4})
    v3 = Variable("v3", domain={1,2,3,4})
    
    constraint = UniquenessConstraint([v1, v2, v3])
    
    print("\nTest 1: No assigned variables, no starting assignment")
    solutions = list(constraint.possible_solutions())
    print(f"Number of solutions: {len(solutions)}")
    print("First few solutions:", solutions[:3])
    
    print("\nTest 2: No assigned variables, with starting assignment v1=1")
    solutions = list(constraint.possible_solutions({v1: 1}))
    print(f"Number of solutions: {len(solutions)}")
    print("All solutions:", solutions)
    
    print("\nTest 3: v1 assigned 1, no starting assignment")
    v1.assign(1)
    solutions = list(constraint.possible_solutions())
    print(f"Number of solutions: {len(solutions)}")
    print("All solutions:", solutions)
    
    print("\nTest 4: v1 assigned 1, with starting assignment v2=2")
    solutions = list(constraint.possible_solutions({v2: 2}))
    print(f"Number of solutions: {len(solutions)}")
    print("All solutions:", solutions)
    
    print("\nTest 5: v1 assigned 1, with conflicting starting assignment v2=1")
    solutions = list(constraint.possible_solutions({v2: 1}))
    print(f"Number of solutions: {len(solutions)}")
    print("All solutions:", solutions)
    
    print("\nTest 6: v1 assigned 1, v2 assigned 2, no starting assignment")
    v2.assign(2)
    solutions = list(constraint.possible_solutions())
    print(f"Number of solutions: {len(solutions)}")
    print("All solutions:", solutions)
    
    print("\nTest 7: v1 assigned 1, v2 assigned 2, with starting assignment v3=3")
    solutions = list(constraint.possible_solutions({v3: 3}))
    print(f"Number of solutions: {len(solutions)}")
    print("All solutions:", solutions)
    
    print("\nTest 8: v1 assigned 1, v2 assigned 1 (conflict), no starting assignment")
    v2.assign(1)
    solutions = list(constraint.possible_solutions())
    print(f"Number of solutions: {len(solutions)}")
    print("All solutions:", solutions)

    # Verify all solutions are valid
    for test_solutions in [solutions]:
        for solution in test_solutions:
            values = [solution[v] for v in solution]
            assert len(values) == len(set(values)), f"Duplicate values in solution: {solution}"

if __name__ == "__main__":
    #test_constraint_order_with_contradictions()
    #test_constraint_order_non_contradictory()
    test_uniqueness_constraint()
    print("All tests passed!")