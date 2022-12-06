# Bi-Directional-Search
This is the implementation of the paper 'Bidirectional Search That Is Guaranteed to Meet in the Middle' link:- https://webdocs.cs.ualberta.ca/~holte/Publications/MM-AAAI2016.pdf

***Team 12 CSE 571 AI ASU***

Requirements:- 
1. python == 3.6
2. scipy == 1.5.4
3. scikit-learn == 0.24.2



You can choose from the following arguments for search strategies:
- bfs --> breadthFirstSearch
- dfs --> depthFirstSearch
- astar --> aStarSearch
- ucs --> uniformCostSearch
- bid --> bidirectionalsearch for the MM version
- standard_bid --> standard_biderictional


You can choose from the following arguments for heuristics:
- nullHeuristic  --> no heurisitc involved
- euclideanHeuristic --> heuristic used is Euclidean distance
- manhattanHeuristic --> heuristic used is Manhattan distance
- mazeHeuristic --> hueristic used is the actual maze distance 

You can choose from the following arguments for layouts:
- tinyMaze
- smallMaze
- mediumMaze
- bigMaze
- customSmallMaze1
- customSmallMaze2
- customSmallMaze3
- customMediumMaze2
- customMediumMaze3
- customMediumMaze1
- customBigMaze1
- customBigMaze2
- customBigMaze3
- customBigMaze1_t
- customBigMaze2_t
- customMediumMaze1_t
- customMediumMaze2_t
- customMediumMaze3_t
- openMaze_t
- customMaze1
- customMaze2_t
- customMaze1_t
- Opposites
- openMaze
- openMaze_1

To run the code on a single layout use the following format:
- python pacman.py -l Layout_name -p SearchAgent -a fn=search_strategy,heuristic=heuristic_name -q

For example:
```
python pacman.py -l Opposites -p SearchAgent -a fn=bid,heuristic=nullHeuristic -q
```
To run T-test use the following format:
- python t_test.py [search_strategy_1,heuristic=heuristic_name:search_strategy_2,heuristic=heuristic_name]

For example:
```
python t_test.py [astar,heuristic=nullHeuristic:bid,heuristic=nullHeuristic]
```

Some notes:
- The 'bid' argument represents the bidirectional search strategy from the paper.
- For mm0 version put heuristic=nullHeuristic while calling the bid serach strategy and for mm version any of the euclidean,manhattan or maze heuristics would do.
- The maze heuristic may cause delay in execution time in some layouts.
