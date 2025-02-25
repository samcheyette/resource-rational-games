# Resource Rational Gameplay

This folder contains models of resource-bounded agents playing various games. The main code can be found in constraint_games.py.

## Constraint Games

Constraint games are a type of game where the players are trying to satisfy a system of constraints, e.g. minesweeper or sudoku. The constraints are of the form:

```
v0 + v1 + v2 = 2
v0 + v2 < 2
v0 +  v1+  v3 + v4 > 1
```

The players are trying to find an assignment to the variables that satisfies all the constraints. When the systems are large, people must be clever about how they structure their problem-solving strategies. The model is designed to capture 1) when  people make errors; 2) how people decide which constraints to tackle; 3) how people reason about the likely consequences of evaluating particular constraints in order to avoid making errors.

The Program-Based contains work in progress (don't judge the code) on a program that generates constraint games.






