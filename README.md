# RedisRAG

RAG (Retrieval-Augmented Generation) pipeline using Redis as a vector store, with FastAPI server for ingestion and semantic search.

## Stack

- **Vector Store**: Redis Stack with HNSW index
- **Embeddings**: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (384 dims, CPU, supports 50+ languages including Hindi)
- **Server**: FastAPI + Uvicorn
- **Client Library**: RedisVL

## Setup

```bash
make setup    # creates venv, installs deps, starts Redis, downloads model
make server   # starts FastAPI on http://localhost:8000
```

## API

### POST /ingest

Reads all `.txt` files from `data/`, chunks by paragraph, embeds, and stores in Redis.

```bash
curl -X POST http://localhost:8000/ingest
```

```json
{"files": 10, "chunks": 50}
```

### POST /search

Semantic search across all ingested chunks.

```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "How does gravity work?", "k": 3}'
```

```json
{
  "query": "How does gravity work?",
  "results": [
    {
      "title": "physics.txt",
      "content": "Einstein's general theory of relativity describes gravity not as a force but as the curvature of spacetime...",
      "similarity": 0.5952
    }
  ],
  "embed_ms": 22.04,
  "search_ms": 12.74
}
```

## Latency Results

Tested on Apple Silicon (CPU only), 50 chunks from 10 files.

### Cold start (first query after startup)

| Query | Embed (ms) | Search (ms) | Total (ms) | Top Hit | Similarity |
|-------|-----------|-------------|------------|---------|------------|
| Tell me about gravity | 1486.7 | 71.5 | 1558.2 | General Relativity | 0.5952 |
| How does machine learning work? | 364.2 | 12.0 | 376.2 | Machine Learning | 0.7061 |

### Warmed up (subsequent queries)

| Query | Embed (ms) | Search (ms) | Total (ms) | Top Hit | Similarity |
|-------|-----------|-------------|------------|---------|------------|
| Tell me about gravity | 15.7 | 1.9 | 17.5 | General Relativity | 0.5952 |
| How does machine learning work? | 25.1 | 4.0 | 29.0 | Machine Learning | 0.7061 |
| What is DNA? | 25.1 | 4.0 | 29.0 | DNA Replication | 0.5869 |
| Explain encryption and security | 14.1 | 9.9 | 24.0 | Cryptography | 0.5990 |
| How do stars produce energy? | 25.4 | 5.5 | 30.9 | Nuclear Fusion | 0.5556 |

**Notes:**
- First query after startup has ~1.5s latency due to model warm-up
- Warmed-up embedding latency: **~10-25ms**
- Redis vector search latency: **~2-10ms**
- Total warmed-up latency: **~15-30ms**

### Embedding Latency Report

PyTorch CPU inference has high variance due to OS thread scheduling. We benchmarked 20 runs of the same query (`"Tell me about gravity"`) after warm-up:

| Config | Min (ms) | Max (ms) | Median (ms) | Stdev (ms) |
|--------|---------|---------|-------------|------------|
| Multi-thread (default) | 16.9 | 176.7 | 59.6 | 307.9 |
| Single-thread (`OMP_NUM_THREADS=1`) | 12.9 | 76.6 | 29.5 | 25.2 |

**Root cause**: PyTorch spawns multiple OpenMP threads for CPU inference. On Apple Silicon, OS thread scheduling across efficiency/performance cores causes unpredictable latency spikes.

**Fix applied**: Server sets `OMP_NUM_THREADS=1` and `torch.set_num_threads(1)` at startup. This gives ~2x lower median latency and ~12x lower variance, since a small model like MiniLM-L6-v2 doesn't benefit from multi-threading at single-query batch size.

## Multilingual / Cross-lingual Retrieval

The embedding model was swapped from `all-MiniLM-L6-v2` (English-only) to `paraphrase-multilingual-MiniLM-L12-v2` so that Devanagari (Hindi) queries can retrieve from the English corpus. Both models are 384 dims, so the existing Redis schema is unchanged — only a re-ingest is needed after the swap.

### Benchmark

`benchmark_multilingual.py` runs 6 test cases (one per topic — physics, biology, AI, chemistry, space, CS). Each test case has:

- a **clean** Devanagari question, and
- a **noisy** version simulating ASR transcription errors (dropped matras, swapped consonants like ब/व and श/स, mixed Devanagari + Romanized fragments)

Run:

```bash
.venv/bin/python benchmark_multilingual.py
```

### Results

| Query type | Top-1 correct | Notes |
|---|---|---|
| Clean Devanagari | **6 / 6** | All Hindi questions retrieve the matching English chunk |
| Noisy / ASR-style | **4 / 6** | 2 misses (chemistry → math, space → physics) |

Mean top-1 similarity drop clean → noisy: ~0.08.

### Observations

1. **Embeddings tolerate surface-level noise** (dropped matras, swapped consonants, missing words) **as long as the clean query already has high margin to the right doc.** Physics clean=0.81 → noisy=0.77 stays correct. Chemistry clean=0.59 → noisy=0.40 drifted to the math file because there wasn't much margin to begin with.
2. **Mixed Hindi + Latin script works.** Queries with English words mid-sentence (`suraj`, `glucose`, `transformer`, `dark matter`) all retrieved correctly. In `cs.txt` the Latin token `embedding` actually *helped* — the noisy variant scored slightly higher than the clean one.
3. **Failure mode is "near-miss to a semantically adjacent topic"**, not random garbage. Space → physics (black holes) and chemistry → math (number theory) are reasonable confusions for a small (118M param) multilingual model.

### What this means for ASR pipelines

If you're feeding noisy ASR transcripts of Hindi questions into this RAG setup:

- Multilingual MiniLM is good enough for queries with clean similarity > ~0.6 — those tolerate moderate transcription errors.
- For borderline queries (~0.5–0.6), small ASR errors can knock the right answer out of rank 1. Possible mitigations (not implemented):
  - Retrieve top-k=5–10 instead of top-1 and let downstream code/LLM pick.
  - Upgrade to `intfloat/multilingual-e5-base` (~278M params) for stronger cross-lingual ranking.
  - Add a reranker pass (e.g., `BAAI/bge-reranker-v2-m3`) over the top-k.

Full per-query results are saved to `multilingual_benchmark.json`.

## Project Structure

```
RedisRAG/
  app.py                       # FastAPI server (model loaded once at startup)
  config.py                    # shared config (Redis URL, model, schema)
  ingest_data.py               # standalone ingestion script
  retrieval.py                 # standalone retrieval script
  benchmark_multilingual.py    # cross-lingual (Hindi → English) benchmark
  entrypoint.sh                # setup script (venv, Docker, model download)
  Makefile                     # make setup / server / clean
  requirements.txt
  data/               # 10 topic files chunked by paragraph
    ai.txt
    biology.txt
    chemistry.txt
    cs.txt
    earth.txt
    economics.txt
    math.txt
    physics.txt
    science.txt
    space.txt
```

## Cleanup

```bash
make clean   # removes venv and Redis container
```
