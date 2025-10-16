"""
dataset.py
This module handles data loading and preprocessing for the HCNTM project. For reproducibility
we rely on the 20 Newsgroups dataset from scikit‑learn. The documents are split into
several artificial time slices by their original ordering to simulate a temporal collection.

Functions
---------
load_dataset(n_slices: int, max_features: int) -> Tuple[list, list, CountVectorizer]
    Load the 20 Newsgroups dataset, split into time slices, and return the preprocessed
    document-term matrices along with the fitted vectorizer.

Notes
-----
The 20 Newsgroups dataset is publicly available and thus satisfies the requirement to
use open data. You can modify the number of slices and the vocabulary size via the
function arguments. The preprocessing includes lowercasing and stop word removal.
"""
from typing import List, Tuple
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

def _generate_synthetic_corpus(n_docs: int, n_words: int, vocab_size: int, random_state: int = 0) -> list:
    """Generate a synthetic corpus of documents.

    Each document is created by sampling a document‑specific word distribution
    from a Dirichlet prior and then drawing word counts from a multinomial.

    Parameters
    ----------
    n_docs : int
        Number of documents to generate.
    n_words : int
        Number of words (tokens) per document.
    vocab_size : int
        Size of the vocabulary.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    documents : list of str
        Synthetic documents represented as space‑separated strings of word IDs.
    """
    rng = np.random.default_rng(random_state)
    alpha = np.ones(vocab_size)  # symmetric Dirichlet prior
    documents = []
    for _ in range(n_docs):
        topic_dist = rng.dirichlet(alpha)
        # Sample word counts for the document
        word_counts = rng.multinomial(n_words, topic_dist)
        # Convert counts to repeated word IDs (as strings)
        words = []
        for idx, count in enumerate(word_counts):
            words.extend([f"w{idx}" for _ in range(count)])
        documents.append(" ".join(words))
    return documents

def load_dataset(n_slices: int = 5, max_features: int = 2000) -> Tuple[List[np.ndarray], List[str], CountVectorizer]:
    """Load and preprocess the 20 Newsgroups dataset.

    Parameters
    ----------
    n_slices : int, optional
        The number of time slices to split the dataset into (default is 5). The documents are
        partitioned evenly in their original order.
    max_features : int, optional
        The maximum size of the vocabulary to learn (default is 2000). This limits the
        dimensionality of the document-term matrices.

    Returns
    -------
    slices : list of np.ndarray
        A list of document-term matrices (one per slice) of shape (n_docs_slice, vocab_size).
    feature_names : list of str
        The vocabulary terms corresponding to the columns of the matrices.
    vectorizer : CountVectorizer
        The fitted CountVectorizer instance used to transform the raw documents.
    """
    # Load the raw text data
    try:
        data = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
        documents = data.data
    except Exception:
        # If the dataset cannot be downloaded (e.g. offline), generate synthetic data
        # Generate 1000 synthetic documents with 100 words each
        print("Warning: Failed to download 20 Newsgroups dataset. Falling back to synthetic data.")
        n_docs = 1000
        n_words_per_doc = 100
        vocab_size = max_features
        documents = _generate_synthetic_corpus(n_docs, n_words_per_doc, vocab_size, random_state=42)
        # Create a fake list of newsgroup names for compatibility
        # Each synthetic document belongs to the same group; this information is unused
    if n_slices < 1:
        raise ValueError("Number of slices must be at least 1")

    # Fit a count vectorizer to the entire corpus
    vectorizer = CountVectorizer(max_features=max_features, stop_words='english', lowercase=True)
    full_doc_term = vectorizer.fit_transform(documents)
    feature_names = vectorizer.get_feature_names_out()

    # Compute sizes for slicing
    total_docs = full_doc_term.shape[0]
    slice_size = total_docs // n_slices

    slices = []
    for i in range(n_slices):
        start = i * slice_size
        # Last slice takes all remaining documents
        end = (i + 1) * slice_size if i < n_slices - 1 else total_docs
        slice_matrix = full_doc_term[start:end]
        slices.append(slice_matrix.toarray())

    return slices, list(feature_names), vectorizer

if __name__ == '__main__':
    # Simple smoke test when run directly
    slices, vocab, vec = load_dataset(n_slices=3, max_features=500)
    print(f"Loaded {len(slices)} slices. Vocabulary size: {len(vocab)}")