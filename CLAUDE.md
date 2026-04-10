# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
make setup    # creates .venv, installs deps, starts redis-stack in Docker (port 6379, password "redis"), pre-downloads embedding model
make server   # runs setup then `uvicorn app:app --reload --port 8000`
make clean    # removes .venv and the redis-stack container
```

There is no test suite, linter, or build step. Standalone scripts can be run directly against a running Redis: `.venv/bin/python ingest_data.py` and `.venv/bin/python retrieval.py`.

End-to-end smoke flow once the server is up:

```bash
curl -X POST http://localhost:8000/ingest
curl -X POST http://localhost:8000/search -H "Content-Type: application/json" -d '{"query":"...","k":3}'
```

## Architecture

Two parallel entry points share a single source of truth (`config.py`) for the Redis URL, model name, and index schema:

- **`app.py`** — FastAPI server. The `HFTextVectorizer` and `SearchIndex` are constructed once in the `startup` event and reused across requests. `/ingest` reads `data/*.txt`, splits each file on blank lines (paragraph chunking), embeds with `embed_many`, and bulk-loads documents with `title`, `content`, and a `float32` byte-encoded `embedding`. `/search` embeds the query, runs a `VectorQuery` against the HNSW index, and returns `1 - vector_distance` as similarity.
- **`ingest_data.py` / `retrieval.py`** — standalone CLI versions of the same flows; useful for benchmarking or working without the server. `retrieval.py` writes per-query latency to `latency.json`.
- **`benchmark_multilingual.py`** — cross-lingual benchmark. Embeds 6 hand-written Devanagari (Hindi) queries, each in a clean and ASR-noisy variant, against the same English-document index and writes per-query results to `multilingual_benchmark.json`. Use this as the template if you need to add other multilingual or robustness tests.
- **`benchmark_dirty_transcription.py`** — extends the multilingual benchmark with 3 escalating noise levels (clean / dirty / very dirty) per topic, simulating progressively bad ASR output. Writes results to `dirty_transcription_benchmark.json`. Findings (documented in `LATENCY_REPORT.md`): the system has a robustness floor — going from "dirty" to "very dirty" barely changed accuracy or mean similarity (5/6 top-1 vs 5/6, mean 0.5130 vs 0.5109), and several "very dirty" queries actually scored *higher* than their dirty counterparts because they happened to include more Latin-script keywords matching the source text. Don't be surprised by counter-intuitive numbers there; embed-time anchoring is keyword-driven, not grammar-driven.
- **`latency_report.py`** — multi-iteration (30 warm + 3 warmup) latency profiler with min/median/mean/p95/p99/stdev per query. Writes `latency_report.json`. Use this rather than ad-hoc timing if you change the model or add a reranker — it's the canonical latency baseline, and `LATENCY_REPORT.md` is the canonical written report covering both latency and robustness.

The index schema (`config.py`) uses `redisvl` with HNSW + cosine + 384 dims. The current model is `paraphrase-multilingual-MiniLM-L12-v2` (multilingual, 50+ languages). Both this and the original `all-MiniLM-L6-v2` are 384 dims, so the schema is unchanged — but **swapping models requires re-running `ingest_data.py`** to re-embed everything (the existing vectors in Redis are model-specific). The index name is `voice-index` and keys are prefixed `voice:`. Both code paths call `index.create(overwrite=True)` on ingest, which drops and rebuilds the index.

### Important: thread pinning

`app.py` sets `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, and `torch.set_num_threads(1)` **before any other imports**. This is load-bearing — without it, PyTorch CPU inference on Apple Silicon thrashes across efficiency/performance cores and median embed latency degrades ~2x with ~12x higher variance (see README "Embedding Latency Report"). If you add new modules that import torch, do not import them above the thread-pinning block in `app.py`.

### Redis assumptions

`entrypoint.sh` starts redis-stack via Docker with `--requirepass redis` and the URL in `config.py` (`redis://:redis@localhost:6379`) is hard-coded to match. If port 6379 is already in use the script skips the Docker start and assumes the existing Redis is compatible (i.e., redis-stack with the search module — plain Redis will not work).

### Gotcha: entrypoint.sh hardcodes the embedding model name

`entrypoint.sh`'s "pre-download the embedding model" step has the model name `sentence-transformers/all-MiniLM-L6-v2` hardcoded — it does **not** read from `config.py`. So if you change `MODEL_NAME` in `config.py`, `make setup` will silently pre-download the *wrong* model. Either fix the script to source from `config.py`, or manually run `.venv/bin/python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('<new-model>')"` after the swap. The first ingest/server start will download the correct model on demand anyway, but you lose the pre-download benefit.
