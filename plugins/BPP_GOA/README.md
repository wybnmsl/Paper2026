# BPP_GOA plugin (for DASH)

**Where to place**
Copy this folder into your DASH project:

```
DASH/src/DASH/plugins/BPP_GOA/
```

**What it contains**
- `goa/`: GOA solver implementation for BPP (strict-online + hybrid/offline)
- `utils/`: dataset loader + utilities
- `plr.py`: Profiled Library Retrieval (PLR) group archive (fit / update / retrieve / save)
- `problem.py`: a task wrapper (`BPPGOAProblem`) that runs the solver and updates/retrieves PLR

**Quick test**
From your project root (with `src` on PYTHONPATH):

```bash
python -m DASH.plugins.BPP_GOA.examples.test_engine_grid --pkl /path/to/BPP5k_C500.pkl --case 0 --time 10
```
