# TSPGLS plugin (DASH)

This folder contains the TSP + Guided Local Search (GLS) task code as a DASH plugin.

## Import

```python
from DASH.plugins.tsp_gls import TSPGLS
```

## Layout

- `problem.py`: problem wrapper (`TSPGLS`) used by DASH.
- `gls/`: GLS skeleton and operators.
- `utils/`: TSPLIB loader and small utilities.
