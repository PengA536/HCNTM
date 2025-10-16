"""
train.py
========

This script orchestrates a simple reproducible experiment for the Hierarchical
Contrastive Neural Topic Modeling (HCNTM) project. While the original paper
presents a sophisticated framework with neural ordinary differential equations,
optimal transport, and contrastive learning, re‑implementing the entire method
from scratch is beyond the scope of this repository. Instead, this training
script demonstrates the core ideas described in the paper on an accessible
public dataset (the 20 Newsgroups) using standard tools available in
scikit‑learn and SciPy. The workflow includes:

1. Loading and preprocessing text data into multiple artificial time slices.
2. Fitting an LDA model on each slice to obtain topic–word and document–topic
   distributions.
3. Aligning topics across consecutive slices via a similarity‑based assignment,
   inspired by optimal transport (we employ Hungarian matching on cosine
   similarities as a proxy for the OT alignment step).
4. Computing evaluation metrics such as topic coherence, topic diversity,
   evolution smoothness, topic alignment, and perplexity.
5. Exporting results to the console and optionally to a CSV file.

By following this script and adjusting parameters, researchers can reproduce
baseline experiments analogous to those presented in the HCNTM study. The
implementation is self‑contained and uses only widely available open‑source
libraries, satisfying reproducibility requirements.

Usage
-----
Execute the script from the repository root as follows:

```
python3 src/train.py \
    --n_slices 5 \
    --n_topics 10 \
    --max_features 2000 \
    --max_iter 50 \
    --output_csv results.csv
```

Arguments
~~~~~~~~~
* ``--n_slices``: number of artificial time slices into which the corpus is
  partitioned. Default is 5.
* ``--n_topics``: number of latent topics to learn per slice. Default is 10.
* ``--max_features``: size of the vocabulary (top‑``max_features`` words by
  frequency). Default is 2000.
* ``--max_iter``: maximum number of iterations for the LDA EM algorithm.
  Default is 50.
* ``--output_csv``: optional path to save the computed metrics as a CSV file.

Note
----
The script may take several minutes to run depending on the number of slices
and topics. It is recommended to start with small values to ensure
everything functions correctly before scaling up.
"""

import argparse
import csv
from pathlib import Path
from typing import List

import numpy as np

from dataset import load_dataset
from model import fit_lda, get_topic_word_distribution
from optimal_transport import align_topics_by_similarity
from metrics import (
    compute_topic_coherence,
    compute_topic_diversity,
    compute_evolution_smoothness,
    compute_topic_alignment,
    compute_perplexity,
)


def run_experiment(
    n_slices: int = 5,
    n_topics: int = 10,
    max_features: int = 2000,
    max_iter: int = 50,
    random_state: int = 0,
    output_csv: str = None,
) -> None:
    """Run the dynamic topic modeling experiment and report metrics.

    Parameters
    ----------
    n_slices : int
        Number of artificial time slices to split the corpus into.
    n_topics : int
        Number of latent topics to learn per slice.
    max_features : int
        Size of the vocabulary to consider (top words by frequency).
    max_iter : int
        Number of iterations for the LDA EM algorithm.
    random_state : int
        Random seed for reproducibility of LDA fitting.
    output_csv : str or None
        If provided, a path to write the metrics as a CSV file. The file
        will contain one row summarising the metrics across slices.

    Returns
    -------
    None
    """
    # Step 1: Load and preprocess the dataset
    print(f"Loading dataset with {n_slices} slices and vocabulary size {max_features}…")
    slices, vocab, _ = load_dataset(n_slices=n_slices, max_features=max_features)

    # Containers for learned models and topic‑word distributions
    lda_models: List = []
    topic_word_slices: List[np.ndarray] = []

    # Step 2: Train an LDA model on each slice
    for t, doc_matrix in enumerate(slices):
        print(f"Training LDA on slice {t + 1}/{n_slices} with {doc_matrix.shape[0]} documents…")
        lda = fit_lda(doc_matrix, n_topics=n_topics, max_iter=max_iter, random_state=random_state)
        lda_models.append(lda)
        topic_word = get_topic_word_distribution(lda)
        topic_word_slices.append(topic_word)

    # Step 3: Align topics across consecutive slices via similarity matching
    print("Aligning topics across slices…")
    for t in range(1, n_slices):
        src = topic_word_slices[t - 1]
        tgt = topic_word_slices[t]
        aligned_tgt, assignment = align_topics_by_similarity(src, tgt)
        topic_word_slices[t] = aligned_tgt
        print(f"Slice {t + 1} topics reordered according to assignment: {assignment}")

    # Step 4: Compute metrics
    print("Computing evaluation metrics…")
    # Topic coherence, diversity and perplexity per slice
    coherence_vals = []
    diversity_vals = []
    perplexity_vals = []
    for t in range(n_slices):
        doc_matrix = slices[t]
        topic_word = topic_word_slices[t]
        # Coherence uses binary presence/absence for NPMI; convert doc matrix accordingly
        binary_matrix = (doc_matrix > 0).astype(int)
        coherence = compute_topic_coherence(topic_word, binary_matrix)
        diversity = compute_topic_diversity(topic_word)
        perplexity = compute_perplexity(lda_models[t], doc_matrix)
        coherence_vals.append(coherence)
        diversity_vals.append(diversity)
        perplexity_vals.append(perplexity)
    # Aggregate per‑slice metrics by averaging
    avg_coherence = float(np.mean(coherence_vals))
    avg_diversity = float(np.mean(diversity_vals))
    avg_perplexity = float(np.mean(perplexity_vals))

    # Evolution smoothness (across aligned slices)
    es = compute_evolution_smoothness(topic_word_slices)
    # Topic alignment (average across consecutive slices)
    alignment_vals = []
    for t in range(n_slices - 1):
        ta = compute_topic_alignment(topic_word_slices[t], topic_word_slices[t + 1])
        alignment_vals.append(ta)
    avg_alignment = float(np.mean(alignment_vals))

    # Report metrics
    print("\n===== Experiment Summary =====")
    print(f"Average Topic Coherence (TC): {avg_coherence:.4f}")
    print(f"Average Topic Diversity (TD): {avg_diversity:.4f}")
    print(f"Evolution Smoothness (ES): {es:.4f}")
    print(f"Average Topic Alignment (TA): {avg_alignment:.4f}")
    print(f"Average Perplexity (PPL): {avg_perplexity:.2f}\n")

    # Optionally write metrics to CSV
    if output_csv:
        print(f"Saving metrics to {output_csv}…")
        fieldnames = [
            "n_slices",
            "n_topics",
            "max_features",
            "avg_coherence",
            "avg_diversity",
            "evolution_smoothness",
            "avg_alignment",
            "avg_perplexity",
        ]
        row = {
            "n_slices": n_slices,
            "n_topics": n_topics,
            "max_features": max_features,
            "avg_coherence": avg_coherence,
            "avg_diversity": avg_diversity,
            "evolution_smoothness": es,
            "avg_alignment": avg_alignment,
            "avg_perplexity": avg_perplexity,
        }
        with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run HCNTM baseline experiment on 20 Newsgroups.")
    parser.add_argument("--n_slices", type=int, default=5, help="Number of time slices to split the data into")
    parser.add_argument("--n_topics", type=int, default=10, help="Number of latent topics")
    parser.add_argument("--max_features", type=int, default=2000, help="Vocabulary size (top words)")
    parser.add_argument("--max_iter", type=int, default=50, help="Maximum iterations for LDA EM algorithm")
    parser.add_argument("--random_state", type=int, default=0, help="Random seed for reproducibility")
    parser.add_argument("--output_csv", type=str, default=None, help="Optional path to save metrics as CSV")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_experiment(
        n_slices=args.n_slices,
        n_topics=args.n_topics,
        max_features=args.max_features,
        max_iter=args.max_iter,
        random_state=args.random_state,
        output_csv=args.output_csv,
    )