import jax, jax.numpy as jnp
from jax import Array
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from matplotlib import gridspec
import seaborn as sns
import numpy as np
from typing import Optional, Any
from jax.scipy.special import logsumexp
import math
from collections import defaultdict, Counter
import numpy as np


def _subset_indices(n: int, max_points: int) -> Array:
    """
    Deterministic, evenly spaced indices in [0, n-1], length s = min(n, max_points).
    n must be a Python int (use theta.shape[0] inside jit), max_points is a Python int.
    """
    s = max(1, min(n, max_points))
    # evenly spaced positions, inclusive of 0 and n-1
    idx = jnp.floor(jnp.linspace(0, n - 1, s)).astype(jnp.int32)
    return idx  # (s,)

def _upper_tri_median(d2_sub: Array) -> Array:
    """
    Median over i<j of a (s,s) squared-distance matrix, without boolean indexing.
    Returns scalar jnp.array.
    """
    s = d2_sub.shape[0]
    # mask upper triangle by setting others to +inf
    I = jnp.arange(s)[:, None]
    J = jnp.arange(s)[None, :]
    v = jnp.where(I < J, d2_sub, jnp.inf).ravel()  # (s*s,)
    m = s * (s - 1) // 2
    k = m // 2
    part_k  = jnp.partition(v, k)
    if (m % 2) == 1:
        med = part_k[k]
    else:
        part_km1 = jnp.partition(v, k - 1)
        med = 0.5 * (part_k[k] + part_km1[k - 1])
    return med

def _median_lengthscale_subset(theta: Array, max_points: int) -> Array:
    """
    Approx median heuristic on a static-size subset:
    ell = sqrt( median_offdiag(||θ_i-θ_j||^2) / (2 log(s+1)) )
    """
    n = theta.shape[0]                     # static int inside jit
    idx = _subset_indices(n, max_points)   # (s,)
    sub = theta[idx]                       # (s,d)
    x2  = jnp.sum(sub * sub, axis=1, keepdims=True)
    d2  = x2 + x2.T - 2.0 * (sub @ sub.T)  # (s,s)
    med_d2 = _upper_tri_median(d2)
    s = sub.shape[0]                       # static int
    h2 = jnp.maximum(med_d2 / (2.0 * jnp.log(s + 1.0)), 1e-12)
    return jnp.sqrt(h2)