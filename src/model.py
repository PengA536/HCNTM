"""
model.py
This module implements topic modeling using scikit‑learn's Latent Dirichlet Allocation (LDA).

Functions
---------
fit_lda(doc_term_matrix: np.ndarray, n_topics: int, max_iter: int) -> LatentDirichletAllocation
    Fit an LDA model on the given document-term matrix.

get_topic_word_distribution(lda_model) -> np.ndarray
    Extract the topic-word distributions from a fitted LDA model.

get_doc_topic_distribution(lda_model, doc_term_matrix: np.ndarray) -> np.ndarray
    Infer the document-topic distributions for a set of documents using a fitted LDA model.
"""
from typing import Optional
import numpy as np
from sklearn.decomposition import LatentDirichletAllocation

def fit_lda(doc_term_matrix: np.ndarray, n_topics: int = 10, max_iter: int = 10, random_state: Optional[int] = 0) -> LatentDirichletAllocation:
    """Fit an LDA model on a document-term matrix.

    Parameters
    ----------
    doc_term_matrix : np.ndarray
        A document-term matrix of shape (n_documents, n_terms).
    n_topics : int
        The number of latent topics to learn.
    max_iter : int
        The maximum number of iterations for the EM algorithm.
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    lda_model : LatentDirichletAllocation
        The fitted LDA model.
    """
    lda_model = LatentDirichletAllocation(n_components=n_topics, max_iter=max_iter,
                                           random_state=random_state, learning_method='batch',
                                           evaluate_every=-1, n_jobs=-1)
    lda_model.fit(doc_term_matrix)
    return lda_model

def get_topic_word_distribution(lda_model: LatentDirichletAllocation) -> np.ndarray:
    """Return the topic-word distribution matrix from an LDA model.

    Each row corresponds to a topic and each column to a word. Values are normalized
    to form probability distributions.

    Parameters
    ----------
    lda_model : LatentDirichletAllocation
        A fitted LDA model.

    Returns
    -------
    topic_word : np.ndarray
        A matrix of shape (n_topics, n_terms) containing word probabilities for each topic.
    """
    # LDA components_ is of shape (n_topics, n_terms); rows sum to 1 after normalization
    topic_word = lda_model.components_ / lda_model.components_.sum(axis=1, keepdims=True)
    return topic_word

def get_doc_topic_distribution(lda_model: LatentDirichletAllocation, doc_term_matrix: np.ndarray) -> np.ndarray:
    """Infer the document-topic distributions for a given document-term matrix.

    Parameters
    ----------
    lda_model : LatentDirichletAllocation
        A fitted LDA model.
    doc_term_matrix : np.ndarray
        Document-term matrix of shape (n_documents, n_terms).

    Returns
    -------
    doc_topic : np.ndarray
        A matrix of shape (n_documents, n_topics) containing topic probabilities for each document.
    """
    doc_topic = lda_model.transform(doc_term_matrix)
    return doc_topic

if __name__ == '__main__':
    # Smoke test
    import numpy as np
    from dataset import load_dataset
    slices, vocab, vec = load_dataset(n_slices=1, max_features=500)
    mat = slices[0]
    lda = fit_lda(mat, n_topics=5, max_iter=5)
    tw = get_topic_word_distribution(lda)
    dt = get_doc_topic_distribution(lda, mat)
    print("Topic-word shape", tw.shape, "Doc-topic shape", dt.shape)