"""
clustering.py

This module provides hierarchical clustering routines to group document or topic
representations. It uses scikit‑learn's AgglomerativeClustering.
"""
from typing import Optional, Union
import numpy as np
from sklearn.cluster import AgglomerativeClustering

def hierarchical_clustering(data: np.ndarray, n_clusters: Union[int, None] = None,
                            distance_threshold: Optional[float] = None, linkage: str = 'ward') -> np.ndarray:
    """Perform hierarchical clustering on the given data.

    Parameters
    ----------
    data : np.ndarray
        Data matrix of shape (n_samples, n_features) on which to perform clustering.
    n_clusters : int or None, optional
        The number of clusters to find. If None, distance_threshold must be specified.
    distance_threshold : float or None, optional
        The linkage distance threshold at which clustering stops. If specified, n_clusters
        must be None. See sklearn AgglomerativeClustering documentation for details.
    linkage : str, optional
        Which linkage criterion to use. Options include 'ward', 'complete', 'average', 'single'.

    Returns
    -------
    labels : np.ndarray
        An array of cluster labels for each sample.
    """
    if n_clusters is None and distance_threshold is None:
        raise ValueError("Either n_clusters or distance_threshold must be specified.")
    clusterer = AgglomerativeClustering(n_clusters=n_clusters,
                                        distance_threshold=distance_threshold,
                                        linkage=linkage)
    labels = clusterer.fit_predict(data)
    return labels

if __name__ == '__main__':
    # Example usage
    from sklearn.datasets import make_blobs
    X, _ = make_blobs(n_samples=100, centers=3, n_features=5, random_state=0)
    labels = hierarchical_clustering(X, n_clusters=3)
    print(np.bincount(labels))