# Data Directory

This directory is intentionally left empty in this baseline implementation. The
training script `src/train.py` uses the 20 Newsgroups dataset built into
scikit‑learn and therefore does not require you to download any external
data.

If you wish to experiment with different corpora (such as ACL Anthology,
arXiv CS, NYTimes, UN Debates, or Twitter Trending), place your own
preprocessed data files here and modify `src/dataset.py` accordingly. The
project assumes that the data will be split into discrete time slices, with
each slice represented as a document–term matrix.