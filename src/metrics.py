"""
metrics.py

This module implements a suite of evaluation metrics for topic models inspired by the
HCNTM paper. The metrics include:

* Topic Coherence (TC): computed via normalized PMI (NPMI) over the top words in each topic.
* Topic Diversity (TD): the proportion of unique words across all topics' top words.
* Evolution Smoothness (ES): the average Jensen–Shannon divergence between topics across
  consecutive time slices, accounting for the alignment of topics.
* Topic Alignment (TA): the average maximum cosine similarity between topics in
  consecutive time slices.
* Perplexity (PPL): the per-word perplexity of a fitted LDA model on a given corpus.
"""
from typing import List
import numpy as np
from scipy.spatial.distance import jensenshannon
from sklearn.metrics.pairwise import cosine_similarity


def _compute_npmi_matrix(doc_term_matrix: np.ndarray) -> np.ndarray:
    """Compute the normalized PMI matrix for all word pairs.

    Parameters
    ----------
    doc_term_matrix : np.ndarray
        Binary document-term matrix of shape (n_docs, n_words). Values should be 0/1 indicating
        word presence.

    Returns
    -------
    npmi : np.ndarray
        A symmetric matrix of shape (n_words, n_words) containing NPMI values for each pair.
    """
    doc_presence = (doc_term_matrix > 0).astype(float)
    n_docs, n_words = doc_presence.shape
    # Document frequencies for each word
    df = doc_presence.sum(axis=0) + 1e-12  # add small epsilon to avoid division by zero
    # Joint document frequencies for each pair (co-occurrence matrix)
    co_matrix = doc_presence.T @ doc_presence  # shape (n_words, n_words)
    # Compute probabilities
    p_i = df / n_docs
    # Outer product gives p_i * p_j for all pairs
    p_i_j = p_i[:, None] * p_i[None, :]
    # Joint probability
    p_ij = co_matrix / n_docs
    # PMI calculation: log(p_ij / (p_i * p_j))
    with np.errstate(divide='ignore', invalid='ignore'):
        pmi = np.log(np.where(p_ij > 0, p_ij / p_i_j, 1.0))
    # Normalize PMI by -log(p_ij)
    with np.errstate(divide='ignore', invalid='ignore'):
        npmi = pmi / (-np.log(p_ij + 1e-12))
    npmi[np.isnan(npmi)] = 0.0
    return npmi


def compute_topic_coherence(topic_word_dist: np.ndarray, doc_term_matrix: np.ndarray,
                            top_n: int = 10) -> float:
    """Compute topic coherence using NPMI for each topic.

    Parameters
    ----------
    topic_word_dist : np.ndarray
        Topic-word distribution matrix of shape (n_topics, vocab_size).
    doc_term_matrix : np.ndarray
        Document-term matrix used to compute word co-occurrence statistics.
    top_n : int
        Number of top words per topic to consider when computing NPMI.

    Returns
    -------
    coherence : float
        The mean NPMI score across all topic pairs and topics. Higher is better.
    """
    # Compute NPMI matrix once
    npmi_matrix = _compute_npmi_matrix(doc_term_matrix)
    n_topics, vocab_size = topic_word_dist.shape
    coherence_scores = []
    for k in range(n_topics):
        # Get indices of top_n words for this topic
        top_word_indices = np.argsort(topic_word_dist[k])[-top_n:]
        # Compute mean NPMI over all pairs of top words
        scores = []
        for i in range(top_n):
            for j in range(i + 1, top_n):
                w_i = top_word_indices[i]
                w_j = top_word_indices[j]
                scores.append(npmi_matrix[w_i, w_j])
        if scores:
            coherence_scores.append(float(np.mean(scores)))
    if coherence_scores:
        return float(np.mean(coherence_scores))
    else:
        return 0.0


def compute_topic_diversity(topic_word_dist: np.ndarray, top_n: int = 10) -> float:
    """Compute the topic diversity metric.

    Parameters
    ----------
    topic_word_dist : np.ndarray
        Topic-word distribution matrix of shape (n_topics, vocab_size).
    top_n : int
        Number of top words per topic to consider.

    Returns
    -------
    diversity : float
        The fraction of unique words among all selected top_n words from each topic.
    """
    n_topics, vocab_size = topic_word_dist.shape
    selected_words = []
    for k in range(n_topics):
        top_idx = np.argsort(topic_word_dist[k])[-top_n:]
        selected_words.extend(top_idx.tolist())
    unique_words = set(selected_words)
    total = n_topics * top_n
    return float(len(unique_words)) / float(total)


def compute_evolution_smoothness(topic_word_slices: List[np.ndarray]) -> float:
    """Compute the evolution smoothness (ES) metric across time slices.

    ES is defined as one minus the average Jensen–Shannon divergence between aligned
    topics in consecutive slices. We assume that the order of topics has been aligned
    using an alignment procedure such as Hungarian matching (see align_topics_by_similarity).

    Parameters
    ----------
    topic_word_slices : list of np.ndarray
        A list of topic-word distributions for each time slice.

    Returns
    -------
    es : float
        The evolution smoothness metric, in [0, 1], where higher values indicate
        smoother topic evolution.
    """
    if len(topic_word_slices) < 2:
        return 0.0
    jsd_list = []
    for t in range(len(topic_word_slices) - 1):
        topics_t = topic_word_slices[t]
        topics_t1 = topic_word_slices[t + 1]
        n_topics = topics_t.shape[0]
        # Ensure topics_t1 has been aligned; assume shapes match
        for k in range(n_topics):
            p = topics_t[k]
            q = topics_t1[k]
            jsd = jensenshannon(p, q, base=2.0)
            jsd_list.append(jsd)
    if jsd_list:
        avg_jsd = float(np.mean(jsd_list))
        es = 1.0 - avg_jsd  # Smoothness defined as 1 - JS divergence
        return es
    else:
        return 0.0


def compute_topic_alignment(topic_word_t: np.ndarray, topic_word_t1: np.ndarray) -> float:
    """Compute topic alignment (TA) between consecutive slices.

    TA is defined as the average maximum cosine similarity between each topic in slice t
    and all topics in slice t+1.

    Parameters
    ----------
    topic_word_t : np.ndarray
        Topic-word distribution for slice t of shape (n_topics, vocab_size).
    topic_word_t1 : np.ndarray
        Topic-word distribution for slice t+1 of shape (n_topics, vocab_size).

    Returns
    -------
    ta : float
        The topic alignment metric, in [0, 1]. Higher indicates better alignment.
    """
    # Compute cosine similarity matrix
    sim_matrix = cosine_similarity(topic_word_t, topic_word_t1)
    # For each topic in slice t, find the maximum similarity with topics in slice t+1
    max_similarities = sim_matrix.max(axis=1)
    return float(max_similarities.mean())


def compute_perplexity(lda_model, doc_term_matrix: np.ndarray) -> float:
    """Compute per-word perplexity of a document-term matrix using a fitted LDA model.

    Parameters
    ----------
    lda_model : sklearn.decomposition.LatentDirichletAllocation
        A fitted LDA model.
    doc_term_matrix : np.ndarray
        Document-term matrix on which to evaluate perplexity.

    Returns
    -------
    perplexity : float
        The per-word perplexity. Lower values indicate better generative performance.
    """
    return float(lda_model.perplexity(doc_term_matrix))
