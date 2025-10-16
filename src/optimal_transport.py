"""
optimal_transport.py

This module implements the entropic regularized optimal transport (Sinkhorn) algorithm
and utilities for aligning topics across consecutive time slices. We use a simple
cost matrix based on the Euclidean distance between topic-word distributions.
"""
from typing import Tuple
import numpy as np
from scipy.optimize import linear_sum_assignment

def sinkhorn(a: np.ndarray, b: np.ndarray, C: np.ndarray, reg: float = 0.01,
             max_iter: int = 200, tol: float = 1e-9) -> np.ndarray:
    """Compute the Sinkhorn transport plan between two discrete distributions.

    Parameters
    ----------
    a : np.ndarray
        Source distribution of shape (n,). Must sum to 1.
    b : np.ndarray
        Target distribution of shape (m,). Must sum to 1.
    C : np.ndarray
        Cost matrix of shape (n, m). Entries should be non-negative.
    reg : float
        Entropic regularization strength. Higher values yield smoother couplings.
    max_iter : int
        Maximum number of iterations for the Sinkhorn updates.
    tol : float
        Convergence tolerance. Iterations stop when changes in u are below tol.

    Returns
    -------
    gamma : np.ndarray
        The optimal coupling matrix of shape (n, m).
    """
    # Ensure distributions sum to 1
    a = a / a.sum()
    b = b / b.sum()

    # Initialize kernel matrix
    K = np.exp(-C / reg)
    K += 1e-16  # Avoid division by zero

    u = np.ones_like(a)
    v = np.ones_like(b)
    for _ in range(max_iter):
        u_prev = u.copy()
        u = a / (K.dot(v))
        v = b / (K.T.dot(u))
        # Check convergence
        if np.linalg.norm(u - u_prev) < tol:
            break
    gamma = np.outer(u, v) * K
    return gamma

def compute_ot_distance(a: np.ndarray, b: np.ndarray, C: np.ndarray, reg: float = 0.01) -> float:
    """Compute the entropic-regularized OT distance between two distributions.

    This is the sum of the elementwise product of the optimal coupling and the cost matrix.
    """
    gamma = sinkhorn(a, b, C, reg=reg)
    return float((gamma * C).sum())

def align_topics_by_similarity(src_topics: np.ndarray, tgt_topics: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Align target topics to source topics based on cosine similarity.

    We compute the cost matrix as (1 - cosine similarity) between each pair of topics
    and solve an assignment problem to find the best matching. The target topics are
    reordered according to this matching.

    Parameters
    ----------
    src_topics : np.ndarray
        Source topic-word matrix of shape (n_topics, vocab_size).
    tgt_topics : np.ndarray
        Target topic-word matrix of shape (n_topics, vocab_size).

    Returns
    -------
    reordered_tgt : np.ndarray
        The target topic-word matrix with rows reordered to best match the source topics.
    assignment : np.ndarray
        Array of indices representing the matching from source to target topics.
    """
    # Normalize topics to unit vectors for cosine similarity
    src_norm = src_topics / (np.linalg.norm(src_topics, axis=1, keepdims=True) + 1e-16)
    tgt_norm = tgt_topics / (np.linalg.norm(tgt_topics, axis=1, keepdims=True) + 1e-16)
    # Compute cosine similarity matrix
    sim = src_norm @ tgt_norm.T
    cost = 1.0 - sim  # We minimize cost
    row_ind, col_ind = linear_sum_assignment(cost)
    reordered_tgt = tgt_topics[col_ind]
    return reordered_tgt, col_ind

if __name__ == '__main__':
    # Simple test: two distributions on a line
    a = np.array([0.5, 0.5])
    b = np.array([0.5, 0.5])
    C = np.array([[0.0, 1.0], [1.0, 0.0]])
    gamma = sinkhorn(a, b, C, reg=0.1)
    print("Coupling:\n", gamma)