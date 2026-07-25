---
title: Multimodal Fashion Search
emoji: 🔍
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: true
license: mit
---

# Multimodal Fashion Search API

A production-grade multimodal product search engine for fashion e-commerce.

## Endpoints

- `POST /search/text` — Search by natural language text query
- `POST /search/image` — Search by uploading an image
- `GET /find_similar_products` — Find similar products by product ID
- `GET /health` — Health check
- `GET /docs` — Interactive Swagger documentation

## Architecture

- **Dense Retrieval**: FAISS HNSW graphs with Sentence-BERT (384-dim) and FashionCLIP (512-dim)
- **Sparse Retrieval**: BM25 (Okapi) for exact keyword matching
- **Fusion**: Reciprocal Rank Fusion (RRF)
- **Re-ranking**: Cross-Encoder (ms-marco-MiniLM-L-6-v2)
