"""Trajectory-similarity metrics used by :mod:`analysis.evaluate_traj`.

.. warning::
   **This module was reconstructed.** The original ``evaluation.py`` is absent
   from the repository, although ``analysis/evaluate_traj.py`` has always
   imported from it. The functions below were rebuilt from their call sites
   (signatures, argument names and the ``xy_only`` keyword) using standard
   definitions of each metric. They are *not* guaranteed to reproduce the exact
   constants of the original implementation -- in particular the normalisation
   choices in :func:`dtw_distance_normalized` and
   :func:`euclidean_distance_normalized`. Re-verify against published numbers
   before reusing them in a paper.

All metrics take two trajectories shaped ``(T, 3)`` and return a scalar, where
lower is more similar. ``xy_only=True`` drops the vertical axis, which is the
right choice on a map like de_dust2 where height is mostly stairs and ramps.
"""

import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial import procrustes as _procrustes

__all__ = [
    "interpolate_trajectory",
    "procrustes_disparity",
    "dtw_distance_normalized",
    "euclidean_distance_normalized",
    "rmse",
    "frechet_distance",
]


def _as_2d(trajectory, xy_only: bool) -> np.ndarray:
    """Coerce to a float (T, D) array, optionally dropping the z axis."""
    arr = np.asarray(trajectory, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr[:, :2] if xy_only else arr


def interpolate_trajectory(trajectory, num_points: int, method: str = "linear") -> np.ndarray:
    """Resample ``trajectory`` to exactly ``num_points`` samples.

    Resampling is done against a normalised parameter in ``[0, 1]`` so that two
    trajectories of different lengths become point-wise comparable. A
    single-sample trajectory is broadcast, since there is no curve to follow.
    """
    arr = np.asarray(trajectory, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if len(arr) == 0:
        raise ValueError("Cannot interpolate an empty trajectory")
    if len(arr) == 1:
        return np.repeat(arr, num_points, axis=0)

    src = np.linspace(0.0, 1.0, len(arr))
    dst = np.linspace(0.0, 1.0, num_points)
    return interp1d(src, arr, kind=method, axis=0, assume_sorted=True)(dst)


def procrustes_disparity(traj_a, traj_b, xy_only: bool = False) -> float:
    """Procrustes disparity: sum of squared differences after optimal alignment.

    Translation, uniform scale and rotation are factored out, so this measures
    *shape* agreement rather than absolute positional agreement. Returns ``nan``
    for degenerate (zero-variance) input, which ``scipy`` cannot standardise.
    """
    a = _as_2d(traj_a, xy_only)
    b = _as_2d(traj_b, xy_only)
    if len(a) != len(b):
        raise ValueError(f"Procrustes needs equal-length trajectories, got {len(a)} and {len(b)}")
    if np.allclose(a.std(axis=0), 0) or np.allclose(b.std(axis=0), 0):
        return float("nan")
    _, _, disparity = _procrustes(a, b)
    return float(disparity)


def dtw_distance_normalized(traj_a, traj_b, xy_only: bool = False) -> float:
    """Dynamic time warping distance, normalised by the warping-path length.

    Implemented as a dependency-free numpy DP over the full cost matrix, which
    is comfortable at the trajectory lengths used here (a few hundred samples).
    Dividing by the number of steps on the optimal path keeps the value
    comparable across trajectories of different duration.
    """
    a = _as_2d(traj_a, xy_only)
    b = _as_2d(traj_b, xy_only)
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return float("nan")

    # Pointwise euclidean cost matrix.
    cost = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)

    acc = np.full((n + 1, m + 1), np.inf)
    acc[0, 0] = 0.0
    steps = np.zeros((n + 1, m + 1), dtype=np.int64)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            prev = (acc[i - 1, j], acc[i, j - 1], acc[i - 1, j - 1])
            k = int(np.argmin(prev))
            acc[i, j] = cost[i - 1, j - 1] + prev[k]
            src = ((i - 1, j), (i, j - 1), (i - 1, j - 1))[k]
            steps[i, j] = steps[src] + 1

    path_len = max(int(steps[n, m]), 1)
    return float(acc[n, m] / path_len)


def euclidean_distance_normalized(traj_a, traj_b, xy_only: bool = False) -> float:
    """Mean point-wise euclidean distance between two aligned trajectories.

    Assumes the inputs were already resampled to a common length (which is what
    :func:`interpolate_trajectory` is for); mismatched lengths are truncated to
    the shorter of the two.
    """
    a = _as_2d(traj_a, xy_only)
    b = _as_2d(traj_b, xy_only)
    k = min(len(a), len(b))
    if k == 0:
        return float("nan")
    return float(np.mean(np.linalg.norm(a[:k] - b[:k], axis=1)))


def rmse(traj_a, traj_b, xy_only: bool = False) -> float:
    """Root mean squared error between two aligned trajectories."""
    a = _as_2d(traj_a, xy_only)
    b = _as_2d(traj_b, xy_only)
    k = min(len(a), len(b))
    if k == 0:
        return float("nan")
    return float(np.sqrt(np.mean(np.sum((a[:k] - b[:k]) ** 2, axis=1))))


def frechet_distance(traj_a, traj_b, xy_only: bool = True) -> float:
    """Discrete Fréchet distance (Eiter & Mannila), computed iteratively.

    Unlike the mean-based metrics this is a worst-case measure: it reports the
    shortest "leash" that lets two walkers traverse both curves without ever
    moving backwards, so a single large excursion dominates the score.
    """
    a = _as_2d(traj_a, xy_only)
    b = _as_2d(traj_b, xy_only)
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return float("nan")

    cost = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    ca = np.empty((n, m), dtype=np.float64)
    ca[0, 0] = cost[0, 0]
    for i in range(1, n):
        ca[i, 0] = max(ca[i - 1, 0], cost[i, 0])
    for j in range(1, m):
        ca[0, j] = max(ca[0, j - 1], cost[0, j])
    for i in range(1, n):
        for j in range(1, m):
            ca[i, j] = max(min(ca[i - 1, j], ca[i, j - 1], ca[i - 1, j - 1]), cost[i, j])
    return float(ca[n - 1, m - 1])
