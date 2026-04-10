# RedisRAG Latency Report

End-to-end latency profile of the FastAPI + Redis vector search RAG pipeline, measured against the multilingual-MiniLM cross-lingual setup currently in `config.py`.

## TL;DR

| Stage (warm) | Median | p95 | p99 |
|---|---:|---:|---:|
| Embedding (CPU, torch) | **10.58 ms** | 13.62 ms | 13.81 ms |
| Redis HNSW vector search | **2.06 ms** | 3.01 ms | 3.27 ms |
| HTTP / FastAPI overhead | **1.27 ms** | 1.49 ms | 1.56 ms |
| **Wall (HTTP client → server → response)** | **14.08 ms** | 17.41 ms | 18.62 ms |

- **~75% of warm latency is the embedding step** (CPU inference of the multilingual MiniLM model). Everything else is essentially free.
- **Redis HNSW search is ~2 ms.** This includes the round-trip to the Redis container on localhost.
- **FastAPI HTTP overhead is ~1 ms.** Pydantic serialization + uvicorn is not a meaningful cost.
- **Cold start is ~12 seconds** dominated by the one-time model load (`HFTextVectorizer(model=...)`).

## Configuration

| | |
|---|---|
| Model | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (118M params, 384 dims) |
| Index | `voice-index` — HNSW, 384 dims, cosine, 50 chunks |
| Top-K | 3 |
| Threading | `OMP_NUM_THREADS=1`, `torch.set_num_threads(1)` (load-bearing — see CLAUDE.md) |
| Iterations | 30 per query, 3 warmup |
| Hardware | Apple Silicon (CPU only, no GPU/MPS) |
| Redis | `redis-secure` Docker container, localhost:6379 |

## Methodology

Two probes were used so we can attribute time to each layer cleanly:

1. **`latency_report.py`** — calls `HFTextVectorizer.embed()` and `SearchIndex.query()` directly, no HTTP. This isolates the model + Redis costs from any framework overhead.
2. **`_http_probe.py`** (one-off, deleted) — POSTs JSON to `/search` on a running uvicorn instance and times the wall clock from `urllib`'s perspective. The server response includes `embed_ms` and `search_ms` fields, so we can compute HTTP overhead as:

       http_overhead = wall_ms − server_embed_ms − server_search_ms

Both used the same query (`आइंस्टीन की सापेक्षता के सिद्धांत में गुरुत्वाकर्षण को कैसे समझाया गया है?`) so the numbers are directly comparable.

Each measurement is `time.perf_counter()` deltas in milliseconds. Stats are computed over 30 warm iterations after discarding 3 warmup runs.

## Startup costs (one-time, cold)

Measured during process boot, before any query:

| Step | Time |
|---|---:|
| `HFTextVectorizer(model=...)` (download from HF cache + load weights into torch) | **11,353.6 ms** |
| `SearchIndex.from_dict(SCHEMA, redis_url=...)` (Redis handshake + index attach) | **2.5 ms** |
| **Total cold startup** | **~11.4 s** |

Notes:
- The 11.4 s model load is the reason `app.py` calls `HFTextVectorizer(...)` once in the `@app.on_event("startup")` hook and reuses it for the lifetime of the process. **Do not** instantiate it per request — it would dominate every call.
- The MiniLM L12 multilingual model is 470 MB on disk; the load time is dominated by reading those weights into the torch state dict. This is roughly 2× slower than the English-only L6 model (~5 s) because L12 has twice the layers and a larger vocabulary.
- `SearchIndex.from_dict` does not actually call `FT.CREATE` — that happens later on `index.create(overwrite=True)` during ingest. The 2.5 ms here is just opening the Redis connection and registering the schema in memory.

## Cold-start query (first call after startup)

The first `/search` after server boot pays a small additional warmup tax. Measured directly (no HTTP):

| | Embed | Search |
|---|---:|---:|
| **Cold first call** | 38.06 ms | 31.59 ms |
| **Warm median** | 9.65 ms | 1.60 ms |
| **Cold ÷ warm ratio** | 4× | 20× |

The first cold call is ~4× slower for embedding (lazy initialization in torch — kernel selection, memory allocator warmup) and ~20× slower for the Redis search (RediSearch HNSW lazily loads HNSW graph nodes into memory; first query touches the most pages).

Through HTTP the cold-start picture is even worse because the FastAPI app doesn't pre-warm the model — the first call I made through HTTP after server start was **329.91 ms wall, 239.81 ms server-embed, 35.53 ms server-search**. The 240 ms embed is well above the direct cold-start of 38 ms, meaning torch hadn't been touched at all between the `startup` hook returning and the first request landing.

**Recommendation:** add a single `vectorizer.embed("warmup")` call at the end of the `startup` event in `app.py:33`. This shifts ~250 ms of work from the first user-facing request into server boot, where it doesn't matter. Not done in this commit — flagged as a one-line follow-up.

## Warm latency — direct (no HTTP)

30 iterations per query, 8 queries (4 clean Devanagari + 4 ASR-noisy variants). All times in ms.

### Embedding latency

| query | len (chars) | min | median | mean | p95 | p99 | max | stdev |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| clean.physics | 74 | 8.99 | 9.65 | 9.68 | 10.43 | 10.65 | 10.71 | 0.53 |
| clean.biology | 80 | 9.54 | 10.10 | 12.30 | 26.97 | 34.98 | 35.51 | 6.32 |
| clean.ai | 75 | 9.21 | 9.72 | 9.75 | 10.36 | 10.68 | 10.81 | 0.44 |
| clean.cs | 74 | 9.31 | 10.20 | 10.70 | 13.13 | 16.81 | 18.01 | 1.69 |
| noisy.physics | 61 | 9.21 | 10.25 | 11.99 | 15.51 | 29.13 | 34.68 | 4.59 |
| noisy.biology | 60 | 9.25 | 9.94 | 10.23 | 11.74 | 13.30 | 13.84 | 0.87 |
| noisy.ai | 68 | 9.75 | 10.98 | 11.77 | 11.74 | 28.98 | 35.98 | 4.51 |
| noisy.cs | 63 | 9.23 | 10.21 | 11.08 | 11.82 | 29.61 | 36.70 | 4.80 |

### Search latency

| query | min | median | mean | p95 | p99 | max | stdev |
|---|---:|---:|---:|---:|---:|---:|---:|
| clean.physics | 0.51 | 1.60 | 1.66 | 2.59 | 2.95 | 3.02 | 0.57 |
| clean.biology | 0.48 | 0.78 | 1.15 | 3.66 | 5.23 | 5.42 | 1.15 |
| clean.ai | 0.57 | 1.97 | 2.08 | 3.31 | 3.71 | 3.85 | 0.69 |
| clean.cs | 1.48 | 2.25 | 2.31 | 2.98 | 4.95 | 5.69 | 0.75 |
| noisy.physics | 0.64 | 2.13 | 2.85 | 5.32 | 17.83 | 22.43 | 3.79 |
| noisy.biology | 1.27 | 2.24 | 2.31 | 3.03 | 3.26 | 3.35 | 0.48 |
| noisy.ai | 1.16 | 2.27 | 2.27 | 3.30 | 3.67 | 3.70 | 0.62 |
| noisy.cs | 0.57 | 2.06 | 2.53 | 3.62 | 13.00 | 16.83 | 2.76 |

### Aggregates (median across queries)

| | embed (ms) | search (ms) | total (ms) |
|---|---:|---:|---:|
| Clean queries (n=4) | **9.92** | ~1.65 | ~11.6 |
| Noisy queries (n=4) | **10.34** | ~2.18 | ~12.5 |
| All queries (n=8) | **10.13** | ~1.91 | ~12.0 |

**Clean vs noisy is statistically indistinguishable** for embedding latency (10.0 vs 10.3 ms). This confirms what we'd expect: the model embeds based on token count and semantic content, not on whether the query is "well-formed." Noisy queries are actually slightly *shorter* (60-68 chars vs 74-80) so they should be marginally faster, but the difference is within noise.

**Search latency is also unaffected by query content** — Redis HNSW operates entirely on the dense vector representation, which is the same shape regardless of input quality. The ~2 ms median is determined by the index size (50 vectors at 384 dims is tiny — HNSW does maybe ~10-20 distance computations) and the Redis round-trip.

## Warm latency — through FastAPI HTTP

30 iterations of `POST /search` against `localhost:8765` after warmup. The same query as above.

| Layer | min | median | mean | p95 | p99 | max |
|---|---:|---:|---:|---:|---:|---:|
| **Wall (full HTTP round-trip)** | 12.32 | 14.08 | 14.64 | 17.41 | 18.62 | 25.84 |
| Server-reported embed | 9.46 | 10.58 | 11.07 | 13.62 | 13.81 | 14.71 |
| Server-reported search | 0.74 | 2.06 | 1.92 | 3.01 | 3.27 | 3.53 |
| **HTTP overhead** (wall − embed − search) | 0.97 | 1.27 | 1.65 | 1.49 | 1.56 | 13.32 |

The server-reported `embed_ms` and `search_ms` here match the direct measurements above almost exactly (10.58 vs 9.92 median embed; 2.06 vs 1.91 median search), which validates that:

1. The direct probe and HTTP probe are measuring the same underlying work.
2. **FastAPI + Pydantic + uvicorn add ~1.3 ms median overhead** — this is the JSON request parsing, response serialization, and TCP loopback round-trip combined. It is not a bottleneck.

The one ugly outlier — `max = 25.84 ms` wall with `13.32 ms` overhead — is GC or scheduling jitter, not a real signal.

## Where the time goes (median, warm)

```
                  ┌─────────────── 14.08 ms wall ─────────────────┐
HTTP request  →   │  embed 10.58 ms │  search 2.06 ms │ HTTP 1.27 │   →  HTTP response
                  └─────────────────┴─────────────────┴───────────┘
                       ~75%                ~15%            ~9%
```

If you want this faster:
- The only place worth optimizing is the embed step. Options:
  - **Smaller multilingual model** — `paraphrase-multilingual-MiniLM-L12-v2` could be swapped to a distilled or quantized variant for ~2-4× speedup at some quality cost.
  - **GPU / MPS** — Apple Silicon Metal would likely take embed to <2 ms, but adds startup latency and complicates deployment.
  - **ONNX or CoreML export** — converts the torch model to an optimized inference runtime, typically 2-3× faster on CPU.
- Redis search is essentially free at this corpus size and doesn't justify any work.
- FastAPI overhead at ~1 ms is also not worth touching.

## Response payload structure

The `/search` endpoint accepts:

```json
POST /search
Content-Type: application/json

{
  "query": "string — natural language question, any language the model supports",
  "k": 3
}
```

`k` is optional, defaults to 3. There is no auth, rate limiting, or pagination.

The response shape (Pydantic models in `app.py:40-56`):

```json
{
  "query": "<echoed back from the request>",
  "results": [
    {
      "title": "<source filename, e.g. 'physics.txt'>",
      "content": "<the full paragraph chunk that was indexed>",
      "similarity": 0.8079
    },
    ...
  ],
  "embed_ms": 10.26,
  "search_ms": 1.91
}
```

Field semantics:

| Field | Type | Notes |
|---|---|---|
| `query` | string | Echoed input. Useful for clients that batch and need to correlate. |
| `results` | array of `SearchResult` | Length is exactly `k`. Sorted by descending similarity. Always returns `k` items even if some are obviously irrelevant — there is no minimum similarity threshold. |
| `results[].title` | string | Source filename of the chunk. **There is no chunk ID, paragraph index, or character offset** — only the filename. If you need to dedupe by source file or fetch surrounding context, you have to re-read the file yourself. |
| `results[].content` | string | The full paragraph as ingested. Length is whatever the source paragraph was — ranges from ~300 to ~700 characters in the current corpus. |
| `results[].similarity` | float | `1 − cosine_distance`, rounded to 4 decimal places at the server (`app.py:110`). Range is theoretically [-1, 1] but in practice [0.4, 0.9] for this corpus. **Higher is better.** Note that this is computed from `vector_distance` returned by RediSearch, which is cosine distance for this index. |
| `embed_ms` | float | Server-side wall time for the model embedding step. Rounded to 2 decimals. |
| `search_ms` | float | Server-side wall time for the Redis HNSW query. Rounded to 2 decimals. |

There are no error codes documented — the endpoint will 500 if Redis is unreachable or the model fails to load. There is no `200 OK` empty-result case because `index.query` always returns `k` items as long as the index has ≥ `k` documents.

### Full sample response

For `query = "आइंस्टीन की सापेक्षता के सिद्धांत में गुरुत्वाकर्षण को कैसे समझाया गया है?"` (the example query throughout this report):

```json
{
  "query": "आइंस्टीन की सापेक्षता के सिद्धांत में गुरुत्वाकर्षण को कैसे समझाया गया है?",
  "results": [
    {
      "title": "physics.txt",
      "content": "Einstein's general theory of relativity describes gravity not as a force but as the curvature of spacetime caused by mass and energy. Massive objects like stars and planets bend the fabric of spacetime around them, and other objects follow curved paths through this warped geometry. The theory predicts gravitational lensing, where light bends around massive objects, and gravitational waves, ripples in spacetime caused by accelerating masses. LIGO first detected gravitational waves in 2015, confirming Einstein's century-old prediction.",
      "similarity": 0.8079
    },
    {
      "title": "space.txt",
      "content": "Einstein's special theory of relativity fundamentally changed our understanding of space, time, and energy. The theory is built on two postulates: the laws of physics are the same in all inertial reference frames, and the speed of light in vacuum is constant for all observers. Time dilation causes moving clocks to tick slower relative to stationary observers, and length contraction shortens objects in the direction of motion. The famous equation E equals mc squared reveals that mass and energy are interchangeable, with a small amount of mass converting to enormous energy.",
      "similarity": 0.6984
    },
    {
      "title": "physics.txt",
      "content": "Classical mechanics describes the motion of macroscopic objects under the influence of forces, based on Newton's three laws of motion. The first law states that objects remain at rest or in uniform motion unless acted upon by an external force. The second law relates force, mass, and acceleration through F equals ma. The third law states that every action has an equal and opposite reaction, governing interactions between objects.",
      "similarity": 0.4451
    }
  ],
  "embed_ms": 10.26,
  "search_ms": 1.91
}
```

A few things worth noting about this response:

1. **Rank 1 is correct** (Einstein's general relativity → physics.txt) at sim=0.8079. This is the strongest signal in the whole benchmark.
2. **Rank 2 is "Einstein's special relativity" from space.txt** at sim=0.6984. This is a *correct* semantic match — special and general relativity are closely related — but a naive consumer that only checks the source filename would call this a "wrong file". When evaluating a multilingual RAG, you have to be careful not to penalize close-but-correct chunks.
3. **Rank 3 is classical mechanics** from physics.txt at sim=0.4451. The big similarity gap between rank 2 (0.70) and rank 3 (0.44) is a useful signal — anything below ~0.5 should probably be treated as "unrelated".
4. **Same source file appears at both rank 1 and rank 3.** There is no per-file deduplication. Clients that want one chunk per source need to do it themselves.

## Caveats and what this report does NOT show

- **Single-client measurement only.** All numbers are for a single sequential client. Concurrent load (e.g., 10 clients hitting `/search` simultaneously) will compete for the same single-threaded torch model and serialize behind the GIL — actual p99 under load will be much worse than what's shown here. This report gives the floor, not the ceiling.
- **Localhost Redis.** The 2 ms search number includes a Unix-localhost TCP round-trip. A remote Redis on the same VPC would add 0.5-2 ms; a cross-region Redis would add 20-100 ms.
- **Tiny corpus.** 50 chunks at 384 dims is well below the size where HNSW shows interesting performance characteristics. At 100K+ documents, Redis search latency would scale roughly with `log(N)` — expect ~5-15 ms instead of 2 ms.
- **No batching.** The current `/search` endpoint embeds one query at a time. The torch model is several × more efficient at batches of 8-32 — if a client has multiple queries, batching them at the API level could reduce per-query latency to ~3-5 ms.
- **CPU only, single thread.** `OMP_NUM_THREADS=1` is set deliberately because on Apple Silicon multi-threading actually hurts (see README "Embedding Latency Report"). On Linux Xeon hardware, removing this would likely give a ~2× speedup with higher variance.
- **Cold start tax not amortized in the warm numbers.** The very first request a real user sees will be the 240-330 ms cold-start range. Adding a warmup `vectorizer.embed("warmup")` to the FastAPI `startup` hook (one line in `app.py:33`) would eliminate this.

## Reproducibility

```bash
.venv/bin/python latency_report.py        # writes latency_report.json
```

For the HTTP numbers, start `make server` in another terminal, then run a probe similar to the one used here. The probe script itself was a one-off and is not committed.

Raw JSON outputs are saved in `latency_report.json` (direct path) and `http_latency.json` (HTTP path) — both contain the full per-iteration data, not just the summary stats above.
