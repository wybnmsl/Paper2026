# prob_tsp.py
# EoH problem wrapper for TSP with DAG meta feedback

import os, sys, time, types, warnings, pickle, glob
import numpy as np

from .tsp_run import solve_one

class TSPProblem():
    def __init__(self):
        # Use a small number of instances for quick evaluation
        self.n_inst_eva = 20
        self.time_limit = 60.0
        self.debug_mode = False

        # Locate training data for TSP instances
        base = os.path.dirname(os.path.abspath(__file__))
        root = os.path.normpath(os.path.join(base, ".."))  # root directory containing data
        self.train_root = os.path.join(root, "TrainingData", "TSP")
        self.test_root  = os.path.join(root, "TestData", "TSP")

        # Load a few TSP instances for evaluation
        self.train_instances = self._collect_instances(self.train_root)
        if len(self.train_instances) == 0:
            raise FileNotFoundError(
                f"No TSP instances found under {self.train_root}. "
                f"Please run tools/build_tsp_dataset.py first."
            )
        # Use only a limited number of instances for evaluation (at most n_inst_eva)
        self.train_instances = self.train_instances[:max(1, self.n_inst_eva)]

        from .prompts_tsp import GetPrompts
        self.prompts = GetPrompts()

    def _collect_instances(self, root):
        """Collects all TSP instance files (distance matrices) under the given root directory."""
        instances = []
        if not os.path.exists(root):
            return instances
        for d in sorted(glob.glob(os.path.join(root, "*"))):
            if os.path.isdir(d):
                # Look for data files inside subdirectory
                # e.g., dist matrix saved as pickle or numpy file
                dist_pkl = os.path.join(d, "dist.pkl")
                dist_npy = os.path.join(d, "dist.npy")
                if os.path.exists(dist_pkl):
                    instances.append(dist_pkl)
                elif os.path.exists(dist_npy):
                    instances.append(dist_npy)
            else:
                # Also allow instance files directly in the root
                if d.endswith(".pkl") or d.endswith(".npy"):
                    instances.append(d)
        return instances

    @staticmethod
    def _load_matrix(file_path):
        """Loads a distance matrix from file (pickle or numpy)."""
        if file_path.endswith(".pkl"):
            with open(file_path, "rb") as f:
                data = pickle.load(f)
            # If data is a dict or object containing distance matrix
            if isinstance(data, dict) and "dist" in data:
                dist_matrix = np.array(data["dist"], dtype=np.float64)
            else:
                dist_matrix = np.array(data, dtype=np.float64)
        elif file_path.endswith(".npy"):
            dist_matrix = np.load(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
        return dist_matrix

    def evaluateTSP(self, heuristic_module):
        """Evaluate the given heuristic by running it on sample TSP instances.
        Returns a dict with 'fitness' (average tour length) and 'meta' information.
        """
        guide = getattr(heuristic_module, "guide", None)
        if guide is None or not callable(guide):
            return {"fitness": 1e10, "meta": {"error": "no_guide"}}

        total_lengths = []
        agg_times = {}  # aggregate node times across instances
        order_first = None

        for i, inst_file in enumerate(self.train_instances):
            try:
                dist_matrix = self._load_matrix(inst_file)
                # Solve one TSP instance with time limit
                start_t = time.time()
                sol, stats = solve_one(dist_matrix, guide, time_limit=self.time_limit, solver_order=None)
                # Calculate tour length (objective value)
                if sol is not None:
                    # Calculate total distance of the returned tour (sol is list of cities in cycle)
                    tour_length = 0.0
                    for j in range(len(sol) - 1):
                        tour_length += dist_matrix[sol[j]][sol[j+1]]
                else:
                    tour_length = 1e9  # treat failure to find any tour as very high cost
                # If solver exceeded time limit by a margin, count as failure (very large objective)
                if time.time() - start_t > self.time_limit * 1.05:
                    tour_length = 1e9

                total_lengths.append(float(tour_length))

                # Collect DAG timing info
                node_times = stats.get("solver_node_times", {})
                for k, v in node_times.items():
                    agg_times.setdefault(k, []).append(float(v))
                if order_first is None:
                    order_first = list(stats.get("solver_order_resolved", []))
            except Exception:
                # On any exception, treat as failure with max cost
                total_lengths.append(1e9)

        if not total_lengths:
            return {"fitness": 1e10, "meta": {"error": "no_instances"}}

        # Compute average tour length across instances as fitness
        fitness = float(np.mean(total_lengths))
        # Average the DAG node times across instances
        avg_times = {k: (float(np.mean(v)) if v else None) for k, v in agg_times.items()}

        meta = {
            "solver_order_resolved": order_first,
            "solver_node_times": avg_times,
            "solver_params": {
                "time_limit": self.time_limit,
                "n_inst_eva": self.n_inst_eva
            }
        }
        return {"fitness": fitness, "meta": meta}

    def evaluate(self, code_string):
        """
        Compile the given code string and evaluate its guide(state) function on TSP instances.
        Returns dict: {"fitness": float, "meta": {...}}.
        """
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                heuristic_module = types.ModuleType("heuristic_module")
                exec(code_string, heuristic_module.__dict__)
                sys.modules[heuristic_module.__name__] = heuristic_module
                res = self.evaluateTSP(heuristic_module)
                # If evaluateTSP returns a raw float (legacy), wrap it into dict
                if isinstance(res, (int, float)):
                    return {"fitness": float(res), "meta": {}}
                return res
        except Exception:
            return {"fitness": 1e10, "meta": {"error": "evaluate_exception"}}
