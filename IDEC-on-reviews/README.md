---
title: Review Intelligence IDEC
emoji: 🔍
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.32.0
app_file: app.py
pinned: false
---

# Review Intelligence Dashboard

Unsupervised pain point discovery across 15 productivity apps using **Improved Deep Embedded Clustering (IDEC)**.

## Pipeline
`Raw Reviews → NLP Preprocessing → Sentence Transformer Embeddings → Deep Autoencoder → IDEC Joint Training → 6 Interpretable Clusters`

## Results
| Method | Silhouette | Davies-Bouldin |
|---|---|---|
| TF-IDF + K-Means | 0.0059 | 10.51 |
| BERT + K-Means | 0.0283 | 3.97 |
| UMAP + HDBSCAN | 0.4039 | — |
| **IDEC (ours)** | **0.8420 ± 0.0025** | **0.20** |

## Dataset
7,824 Google Play Store reviews · 15 productivity applications

## Author
Adithya Damodara · Lovely Professional University