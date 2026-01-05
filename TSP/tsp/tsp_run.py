# tsp_run.py
# Exact TSP solver with DAG-style orchestration & per-node timing

import time
import math

def solve_one(dist_matrix, guide, time_limit=None, solver_order=None):
    """
    Solve one TSP instance (given distance matrix) using the provided guide(state) policy.
    Returns: sol, stats
      sol: list of city indices representing a tour (including return to start), or None if no solution.
      stats: {
          "visited": k, "n": N,
          "solver_order_resolved": [sequence of executed node names],
          "solver_node_times": {node_name: seconds, ...}
      }
    """
    N = len(dist_matrix)
    start_time = time.time()

    # Define default DAG execution order if not provided
    order = solver_order[:] if isinstance(solver_order, (list, tuple)) else ["init", "greedy", "search", "final"]
    node_times = {}
    executed = []

    # Shared state for search (these will be updated during DFS)
    current_path = []
    visited = set()
    best_path = None
    best_cost = math.inf

    # ---- Node: init ----
    if "init" in order:
        t0 = time.time()
        # Initialize starting city (fix city 0 as start to avoid symmetric duplicates)
        current_path = [0]
        visited = {0}
        # No initial cost yet
        current_length = 0.0
        node_times["init"] = time.time() - t0
        executed.append("init")
        # If no possible start (e.g., N == 0), return immediately
        if len(current_path) == 0:
            return None, {
                "visited": 0, "n": N,
                "solver_order_resolved": executed,
                "solver_node_times": node_times
            }

    # ---- Node: greedy ----
    if "greedy" in order:
        t0 = time.time()
        # Perform a greedy nearest-neighbor tour from the current state to get an initial upper bound
        greedy_path = list(current_path)
        greedy_visited = set(visited)
        greedy_cost = 0.0
        # Greedily add the nearest unvisited city until all are visited
        while len(greedy_visited) < N:
            u = greedy_path[-1]
            # Find nearest unvisited neighbor
            next_city = None
            min_dist = math.inf
            for v in range(N):
                if v in greedy_visited:
                    continue
                if dist_matrix[u][v] < min_dist:
                    min_dist = dist_matrix[u][v]
                    next_city = v
            if next_city is None:
                break  # no unvisited neighbor found (should not happen in a complete graph)
            greedy_path.append(next_city)
            greedy_visited.add(next_city)
            greedy_cost += min_dist
        # Complete the cycle by returning to start (0)
        if len(greedy_path) == N and 0 not in greedy_visited:
            # In case we didn't include 0 initially (should not happen since we started at 0)
            greedy_visited.add(0)
        if len(greedy_path) == N:
            greedy_cost += dist_matrix[greedy_path[-1]][0]
            greedy_path.append(0)
        # Update initial best solution
        if len(greedy_path) == N+1:  # full tour found (N cities + return to start)
            best_cost = greedy_cost
            best_path = greedy_path
        node_times["greedy"] = time.time() - t0
        executed.append("greedy")

    # ---- Node: search (DFS + guide) ----
    if "search" in order:
        t0 = time.time()
        # Depth-first search with branch-and-bound
        def dfs(current_city, current_length):
            nonlocal best_cost, best_path
            # Check time limit
            if time_limit is not None and time.time() - start_time > time_limit:
                return  # stop searching further

            # If all cities visited, complete the tour back to start and evaluate
            if len(visited) == N:
                total_cost = current_length + dist_matrix[current_city][0]
                if total_cost < best_cost:
                    best_cost = total_cost
                    best_path = list(current_path) + [0]
                return

            # Prepare state for the guide function
            state = {
                "dist": dist_matrix,
                "path": list(current_path),
                "visited": set(visited),
                "current_length": current_length,
                "cache": {"best": (None if best_cost == math.inf else best_cost)}
            }
            decision = guide(state)
            # Pivot (current city) from guide (not strictly needed for sequential TSP)
            u = decision.get("pivot", current_city)
            # Candidate order from guide, filter to ensure they are unvisited
            order_list = decision.get("order", [])
            cand_list = [v for v in order_list if v not in visited]
            if not cand_list:
                # Fallback: consider all unvisited if guide returned none or invalid
                cand_list = [v for v in range(N) if v not in visited]

            for next_city in cand_list:
                if next_city in visited:
                    continue
                # Branch bound: prune if adding next_city already exceeds best_cost
                new_length = current_length + dist_matrix[current_city][next_city]
                if new_length >= best_cost:
                    continue
                visited.add(next_city)
                current_path.append(next_city)
                dfs(next_city, new_length)
                # Backtrack
                current_path.pop()
                visited.remove(next_city)
        # Start DFS from the starting city (which is current_path[0])
        start_city = current_path[-1] if current_path else 0
        dfs(start_city, 0.0)
        node_times["search"] = time.time() - t0
        executed.append("search")

    # ---- Node: final ----
    if "final" in order:
        t0 = time.time()
        # Determine how many cities were visited in the found solution (N if full tour found)
        mapped = N if best_path is not None else len(current_path)
        node_times["final"] = time.time() - t0
        executed.append("final")
    else:
        mapped = N if best_path is not None else len(current_path)

    stats = {
        "visited": mapped,
        "n": N,
        "solver_order_resolved": executed,
        "solver_node_times": node_times
    }
    return best_path, stats
