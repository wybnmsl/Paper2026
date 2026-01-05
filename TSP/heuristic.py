import numpy as np

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Adaptive two-stage reweighting with node-propagated tour penalty and symmetric exploration.
    
    Inputs:
      - edge_distance: (N, N) numpy array of current edge distances
      - local_opt_tour: sequence/array of node IDs representing the current local-optimum tour
      - edge_n_used: (N, N) numpy array counting how often each edge was used
    
    Output:
      - updated_edge_distance: (N, N) numpy array of updated edge distances
    """
    edge_distance = np.array(edge_distance, dtype=float)
    edge_n_used = np.array(edge_n_used, dtype=float)
    tour = np.asarray(local_opt_tour, dtype=int)

    # ======= Sanity checks =======
    if edge_distance.ndim != 2 or edge_distance.shape[0] != edge_distance.shape[1]:
        raise ValueError("edge_distance must be a square matrix")

    N = edge_distance.shape[0]

    if edge_n_used.shape != (N, N):
        raise ValueError("edge_n_used must have the same shape as edge_distance")

    mask_offdiag = ~np.eye(N, dtype=bool)
    positive_mask = (edge_distance > 0) & mask_offdiag
    positive = edge_distance[positive_mask]
    mean_before = float(positive.mean()) if positive.size > 0 else 1.0
    eps = 1e-12

    # ======= Stage 1: Usage Normalization =======
    max_used = float(edge_n_used.max())
    denom = max_used if max_used > 0.0 else 1.0
    usage_norm = edge_n_used / (denom + eps)

    # ======= Build symmetric tour mask =======
    tour_mask = np.zeros((N, N), dtype=float)
    if tour.size > 0:
        L = tour.shape[0]
        for k in range(L):
            i = int(tour[k % L])
            j = int(tour[(k + 1) % L])
            if 0 <= i < N and 0 <= j < N:
                tour_mask[i, j] = 1.0
                tour_mask[j, i] = 1.0

    # ======= Parameters =======
    center = 0.45   # logistic shaping center
    steep = 6.0     # logistic steepness
    penal_amp = 1.2
    reward_amp = 0.35
    penal_pow = 1.8
    reward_pow = 1.3
    min_mult = 0.55
    max_mult = 5.0

    additive_scale = 1.05
    prop_smooth = 0.5
    lap_smooth = 0.35

    rng = np.random.default_rng()
    rand_frac = 0.14
    rand_sigma = 0.18
    strong_frac = 0.012
    strong_min = 0.28
    strong_max = 1.12

    jitter_rel = 1.5e-6
    min_offdiag = max(1e-12, mean_before * 1e-8)

    # ======= Stage 2: Multiplicative Penalty / Reward =======
    shaped = 1.0 / (1.0 + np.exp(-steep * (usage_norm - center)))
    penal_component = penal_amp * (shaped ** penal_pow)
    reward_component = reward_amp * ((1.0 - shaped) ** reward_pow)

    multiplier = 1.0 + penal_component - reward_component
    multiplier = np.clip(multiplier, min_mult, max_mult)
    np.fill_diagonal(multiplier, 1.0)

    # ======= Node-propagated Additive Penalty =======
    node_usage = usage_norm.mean(axis=1)
    pair_node_avg = (node_usage[:, None] + node_usage[None, :]) / 2.0

    base_add = tour_mask * (1.0 + pair_node_avg)
    additive = additive_scale * mean_before * (
        (1.0 - prop_smooth) * tour_mask + prop_smooth * base_add
    )

    row_mean = additive.mean(axis=1)
    col_mean = additive.mean(axis=0)
    propagated = (row_mean[:, None] + col_mean[None, :]) / 2.0
    smoothed_additive = (1.0 - lap_smooth) * additive + lap_smooth * propagated

    updated_edge_distance = edge_distance * multiplier + smoothed_additive

    # ======= Random Exploration =======
    sel = rng.random((N, N)) < rand_frac
    sel = np.logical_or(sel, sel.T)
    np.fill_diagonal(sel, False)
    sel = sel.astype(float)

    normal = rng.normal(loc=0.0, scale=rand_sigma, size=(N, N))
    rand_mult = np.exp(normal)
    rand_mult = np.sqrt(rand_mult * rand_mult.T)
    rand_apply = 1.0 + sel * (rand_mult - 1.0)
    updated_edge_distance *= rand_apply

    # ======= Strong Random Reductions =======
    strong_sel = rng.random((N, N)) < strong_frac
    strong_sel = np.logical_or(strong_sel, strong_sel.T)
    np.fill_diagonal(strong_sel, False)
    strong_sel = strong_sel.astype(float)

    strong_vals = strong_min + rng.random((N, N)) * (strong_max - strong_min)
    strong_vals = (strong_vals + strong_vals.T) / 2.0
    strong_mult = 1.0 + strong_sel * (strong_vals - 1.0)
    updated_edge_distance *= strong_mult

    # ======= Post-processing =======
    updated_edge_distance = (updated_edge_distance + updated_edge_distance.T) / 2.0
    np.fill_diagonal(updated_edge_distance, 0.0)

    offdiag = updated_edge_distance[mask_offdiag]
    if offdiag.size > 0:
        updated_edge_distance[mask_offdiag] = np.maximum(offdiag, min_offdiag)

    # Preserve mean
    positive_after_mask = (updated_edge_distance > 0) & mask_offdiag
    positive_after = updated_edge_distance[positive_after_mask]
    mean_after = float(positive_after.mean()) if positive_after.size > 0 else mean_before
    if mean_after > 0:
        scale_factor = mean_before / (mean_after + eps)
        updated_edge_distance *= scale_factor

    # ======= Tiny Symmetric Jitter =======
    jitter_scale = max(1e-16, jitter_rel * mean_before)
    jitter = rng.normal(loc=0.0, scale=jitter_scale, size=(N, N))
    jitter = (jitter + jitter.T) / 2.0
    updated_edge_distance += jitter

    # Final symmetry & floor
    updated_edge_distance = (updated_edge_distance + updated_edge_distance.T) / 2.0
    np.fill_diagonal(updated_edge_distance, 0.0)
    offdiag = updated_edge_distance[mask_offdiag]
    if offdiag.size > 0:
        updated_edge_distance[mask_offdiag] = np.maximum(offdiag, min_offdiag)

    return updated_edge_distance
