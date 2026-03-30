"""Shared HDBSCAN clustering with NearestCentroid assignment.

Canonical pipeline aligned with iggt_idmap.py:
  subsample → HDBSCAN(cluster_selection_epsilon) → NearestCentroid → contiguous relabel.

All callers (iggt_idmap, trace_instance_to_gaussians, instseg_infer) should
delegate to this module to keep the clustering logic in one place.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def hdbscan_assign(
    features: np.ndarray,
    *,
    cluster_selection_epsilon: float = 0.06,
    min_cluster_size: int = 500,
    min_samples: int = 100,
    max_points: int = 200_000,
    rng_seed: int = 0,
) -> np.ndarray:
    """Subsample → HDBSCAN → NearestCentroid assign **all** points.

    Args:
        features: ``(N, D)`` float32/64, expected to be already L2-normalised.
        cluster_selection_epsilon: HDBSCAN ``cluster_selection_epsilon``.
        min_cluster_size: HDBSCAN ``min_cluster_size``.
        min_samples: HDBSCAN ``min_samples``.
        max_points: Max points fed into HDBSCAN (random subsample when N > this).
        rng_seed: Seed for reproducible subsampling.

    Returns:
        ``(N,)`` int32 labels – **contiguous** IDs starting from 0.
        Every point is assigned (no noise label) thanks to NearestCentroid.
    """
    import hdbscan as hdb_lib
    from sklearn.neighbors import NearestCentroid

    N = features.shape[0]
    S = min(max_points, N)
    rng = np.random.default_rng(rng_seed)

    if S < N:
        idx_sub = rng.choice(N, S, replace=False)
        logger.info("Subsampled %d / %d points for HDBSCAN", S, N)
        sub_feat = features[idx_sub]
    else:
        idx_sub = None
        sub_feat = features

    logger.info(
        "Running HDBSCAN on %d points (D=%d) ...",
        sub_feat.shape[0], sub_feat.shape[1],
    )
    sub_labels = hdb_lib.HDBSCAN(
        cluster_selection_epsilon=cluster_selection_epsilon,
        min_samples=min_samples,
        min_cluster_size=min_cluster_size,
    ).fit_predict(sub_feat)

    mask_clustered = sub_labels >= 0
    n_clusters = int(sub_labels.max() + 1) if mask_clustered.any() else 0
    n_noise_sub = int((~mask_clustered).sum())
    logger.info(
        "HDBSCAN done: %d clusters, %d/%d noise in subsample",
        n_clusters, n_noise_sub, len(sub_labels),
    )

    if n_clusters >= 2:
        nc = NearestCentroid()
        nc.fit(sub_feat[mask_clustered], sub_labels[mask_clustered])
        labels = nc.predict(features)
    elif n_clusters == 1:
        labels = np.zeros(N, dtype=np.int32)
    else:
        labels = np.zeros(N, dtype=np.int32)

    unique = np.unique(labels)
    remap = {old: new for new, old in enumerate(unique)}
    labels = np.vectorize(remap.get)(labels).astype(np.int32)

    logger.info("Assigned %d unique IDs to %d points", len(unique), N)
    return labels
