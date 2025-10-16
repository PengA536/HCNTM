# HCNTM Baseline Experiment

This repository contains a self‑contained implementation of a **baseline
experiment** inspired by the paper *Hierarchical Contrastive Neural Topic
Modeling (HCNTM)*. The goal of the original study is to model dynamic topic
evolution in text corpora using optimal transport, hierarchical clustering,
contrastive learning and neural ordinary differential equations. Implementing
the full model from scratch is out of scope here. Instead, this project
provides a reproducible pipeline that captures the core ideas and evaluation
methodology on a readily available dataset.

## Overview

The project implements the following components:

* **Dataset loading and slicing**: We use the [20 Newsgroups](https://scikit-learn.org/stable/datasets/real_world.html#newsgroups-dataset)
  dataset from scikit‑learn as a proxy for a temporal corpus. The documents are
  partitioned into artificial time slices by their original ordering. The size
  of the vocabulary (number of most frequent words) and the number of slices
  are configurable.
* **Topic modeling**: Each slice is modeled independently using Latent
  Dirichlet Allocation (LDA) from scikit‑learn. This yields topic–word and
  document–topic distributions per slice.
* **Topic alignment**: To mimic the optimal transport alignment step from
  HCNTM, we match topics across consecutive slices using the Hungarian
  algorithm on a cosine similarity cost matrix. This reorders topics to
  maximize similarity across time.
* **Hierarchical clustering**: A clustering module is provided based on
  AgglomerativeClustering; although not used directly in the baseline pipeline,
  it illustrates how hierarchical structures could be incorporated.
* **Evaluation metrics**: We compute topic coherence (TC), topic diversity
  (TD), evolution smoothness (ES), topic alignment (TA) and perplexity
  (PPL). These metrics are inspired by the HCNTM paper and provide insight
  into topic quality and temporal dynamics.
* **Training script**: `src/train.py` orchestrates the entire process: it loads
  data, trains LDA models on each slice, aligns topics, computes metrics and
  optionally writes results to a CSV file.

The code is written in pure Python with a handful of mature dependencies
(`numpy`, `scipy`, `scikit-learn`). There are no proprietary software or large
pretrained models required, making the experiment straightforward to reproduce
on any machine.

## Directory Structure

```
HCNTM-Project/
├── README.md           # Project overview and instructions (this file)
├── requirements.txt    # Python dependencies
├── src/                # Source code for data loading, models, alignment and metrics
│   ├── dataset.py      # Load and preprocess the 20 Newsgroups dataset
│   ├── model.py        # Train and interface with LDA models
│   ├── optimal_transport.py # Sinkhorn algorithm and topic alignment utilities
│   ├── clustering.py   # Hierarchical clustering utilities
│   ├── metrics.py      # Implement evaluation metrics (TC, TD, ES, TA, PPL)
│   └── train.py        # End‑to‑end experiment script
└── data/               # Placeholder for storing custom data (unused for 20 Newsgroups)
```

## Installation

To set up the environment, clone this repository and install the required
dependencies. We recommend using a virtual environment (such as `venv` or
conda) to avoid conflicts with other projects.

```bash
# Clone the repository (if you haven't already)
git clone https://github.com/your-username/HCNTM-Project.git
cd HCNTM-Project

# Create and activate a virtual environment (optional but recommended)
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

Alternatively, if you use Conda, you can create an environment via
```
conda env create -f environment.yml
conda activate hcntm-env
```
(The `environment.yml` file is optional and provided only if you prefer Conda.)

## Running the Experiment

The main entry point is `src/train.py`. It accepts several command line
arguments to customise the experiment:

```bash
python3 src/train.py \
    --n_slices 5 \
    --n_topics 10 \
    --max_features 2000 \
    --max_iter 50 \
    --random_state 0 \
    --output_csv results.csv
```

* `--n_slices` specifies how many time slices to divide the corpus into. The
  default is 5. Larger values create finer temporal resolution but yield
  smaller slices.
* `--n_topics` determines the number of latent topics to learn per slice.
* `--max_features` controls the size of the vocabulary by selecting the top‑N
  most frequent words.
* `--max_iter` sets the number of EM iterations for training each LDA model.
* `--random_state` ensures reproducibility by seeding the LDA algorithm.
* `--output_csv` optionally writes the aggregated metrics to the specified file.

After running the script, you will see a summary of the computed metrics in the
terminal and, if requested, a CSV file containing the results.

If your environment cannot download the 20 Newsgroups dataset (for example, when
working offline), the script automatically falls back to generating a small
synthetic corpus. This ensures the experiment remains fully reproducible even
without network access, albeit with different numerical results.

Example output:

```
Loading dataset with 5 slices and vocabulary size 2000…
Training LDA on slice 1/5 with 11314 documents…
Training LDA on slice 2/5 with 11313 documents…
…
Slice 2 topics reordered according to assignment: [ 0  1  3  7  2  4  5  8  6  9]
…

===== Experiment Summary =====
Average Topic Coherence (TC): 0.4075
Average Topic Diversity (TD): 0.8450
Evolution Smoothness (ES): 0.6532
Average Topic Alignment (TA): 0.7316
Average Perplexity (PPL): 1340.59

Saving metrics to results.csv…
```

The exact numbers will differ depending on the random seed and parameter
choices.

## Reproducing Results from the Paper

While the baseline pipeline included here is simplified, it preserves key
evaluation steps from the HCNTM paper. To get closer to the reported results,
you may consider the following:

1. **Larger or different datasets**: The original paper evaluated on ACL
   Anthology, arXiv CS, NYTimes, UN Debates and Twitter Trending corpora. You
   can adapt `dataset.py` to load your own corpora split by time.
2. **Advanced alignment**: Implement entropic regularised optimal transport
   (Sinkhorn) using `optimal_transport.sinkhorn` instead of the simple
   Hungarian assignment. See `optimal_transport.py` for details.
3. **Contrastive learning**: Introduce contrastive objectives based on
   hierarchical clustering (see `clustering.py`) to further refine topic
   distributions.
4. **Neural ODE dynamics**: Replace the independent LDA models with a neural
   network that predicts continuous topic embeddings across time.

All these extensions require more substantial development. However, the
existing scaffolding in this repository should help you get started.

## Troubleshooting

* **Slow training**: LDA can be slow on high‑dimensional or large datasets.
  Reduce the number of topics, slices or vocabulary size to speed up
  experimentation. You may also set `n_jobs` in `model.fit_lda` to use multiple
  CPU cores (the default is to use all available cores).
* **Memory usage**: The 20 Newsgroups dataset is relatively small, but larger
  datasets might not fit into memory. Consider using sparse matrices or
  incremental learning.
* **Metric variability**: Coherence and diversity metrics depend on the
  sampling of documents and choice of top words. Try adjusting `top_n` in
  `metrics.compute_topic_coherence` and `compute_topic_diversity` to see how
  results change.

## License

This project is provided for educational purposes and is released under the
MIT License. See `LICENSE` for details.

