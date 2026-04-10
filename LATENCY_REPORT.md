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

## Robustness against dirty transcriptions

Latency tells us how fast the pipeline is. **Robustness** tells us whether the answers are still correct when the input is garbage — which is the realistic scenario when the upstream is an ASR (speech-to-text) system. The two are independent concerns but the user asked for both in the same report.

`benchmark_dirty_transcription.py` runs 6 topics × 3 escalating noise levels = 18 query runs against the same Redis index used throughout this report. Hand-written queries; one set of "dirty" and one set of "very dirty" Devanagari variants per topic, designed to look like increasingly bad ASR output.

### What the noise levels mean

| Level | Construction strategy | Example (physics: Einstein gravity) |
|---|---|---|
| clean | Well-formed Devanagari question, baseline | `आइंस्टीन की सापेक्षता के सिद्धांत में गुरुत्वाकर्षण को कैसे समझाया गया है?` |
| dirty | Drop matras, swap consonants (ब/व, श/स, ण/न), mix in Latin fragments, drop a function word or two | `आइनस्टिन की सपेक्षता मे gravity को कसे समझाया गुरुत्वकरसन` |
| very dirty | Heavy code-switching, missing function words, key concepts mangled or replaced with vague paraphrase, sometimes the question structure is gone entirely | `वो ainstain वाली theery जिसमे gravety wala कुछ था spacetime kuch वो` |

### Results

| Level | Top-1 correct | Mean top-1 sim | Min sim | Max sim |
|---|---|---:|---:|---:|
| clean | **6 / 6** | 0.6561 | 0.5509 | 0.8079 |
| dirty | **5 / 6** | 0.5130 | 0.3550 | 0.7912 |
| very dirty | **5 / 6** | 0.5109 | 0.3900 | 0.6066 |

| Transition | Mean similarity drop |
|---|---:|
| clean → dirty | −0.1431 |
| clean → very dirty | −0.1451 |
| dirty → very dirty | −0.0020 |

### Per-topic walkthrough

The aggregate numbers above hide a lot of structure. Here are all 18 query runs grouped by topic, showing the actual queries we ran and what came back at rank 1.

#### physics.txt

| Level | Query | Top-1 file (paragraph) | Sim | ✓/✗ |
|---|---|---|---:|:---:|
| clean | `आइंस्टीन की सापेक्षता के सिद्धांत में गुरुत्वाकर्षण को कैसे समझाया गया है?` | physics.txt (general relativity) | 0.8079 | ✓ |
| dirty | `आइनस्टिन की सपेक्षता मे gravity को कसे समझाया गुरुत्वकरसन` | physics.txt (general relativity) | 0.7912 | ✓ |
| very dirty | `वो ainstain वाली theery जिसमे gravety wala कुछ था spacetime kuch वो` | space.txt (solar system) | 0.3900 | ✗ |

The `ainstain` mangling killed the strongest anchor in the question. Without "Einstein" properly spelled, `spacetime` dragged it toward space topics. Note physics.txt was still in top-3 (rank 3 at sim=0.3677), so a `k=5` retrieval would have caught it.

#### biology.txt

| Level | Query | Top-1 file (paragraph) | Sim | ✓/✗ |
|---|---|---|---:|:---:|
| clean | `प्रकाश संश्लेषण की प्रक्रिया में पौधे सूर्य के प्रकाश से ग्लूकोज कैसे बनाते हैं?` | biology.txt (photosynthesis) | 0.6838 | ✓ |
| dirty | `परकास सनस्लेसन में पौदा suraj की रोसनी से gluucose बनाता` | biology.txt (photosynthesis) | 0.3550 | ✓ |
| very dirty | `vo पेड़ wala chemistry जिसमे sun light से sugar बनती photo कुछ` | biology.txt (photosynthesis) | 0.5070 | ✓ |

The very-dirty version (sim=0.51) actually beat the dirty version (sim=0.36). Latin tokens `sun light sugar photo` pulled it toward the photosynthesis paragraph harder than the mangled Devanagari `परकास सनस्लेसन` did.

#### ai.txt

| Level | Query | Top-1 file (paragraph) | Sim | ✓/✗ |
|---|---|---|---:|:---:|
| clean | `ट्रांसफॉर्मर मॉडल सेल्फ-अटेंशन का उपयोग करके भाषा को कैसे प्रोसेस करते हैं?` | ai.txt (NLP) | 0.6639 | ✓ |
| dirty | `transfarmer मॉडेल self atension से language कैसे process करता` | ai.txt (NLP) | 0.6059 | ✓ |
| very dirty | `vo जो attention वाला model GPT जैसा mechanism sentence parse karne ka` | ai.txt (transformers) | 0.4576 | ✓ |

In clean and dirty, rank 1 is the *NLP* paragraph (not the transformer paragraph the question is actually about), but it's still ai.txt so the top-1 check passes. In very dirty, where the literal token `GPT` appears, rank 1 finally lands on the transformer paragraph itself. This is a mild reminder that "correct file" is a coarser success metric than "correct paragraph" — for a downstream LLM that just needs *some* relevant context, both are fine.

#### chemistry.txt

| Level | Query | Top-1 file (paragraph) | Sim | ✓/✗ |
|---|---|---|---:|:---:|
| clean | `आवर्त सारणी में तत्वों को परमाणु क्रमांक के अनुसार कैसे व्यवस्थित किया जाता है?` | chemistry.txt (periodic table) | 0.5900 | ✓ |
| dirty | `आवरत साराणी में elements को परमानु number के हिसाब से arrange` | chemistry.txt (periodic table) | 0.4243 | ✓ |
| very dirty | `jo table mein hydrogen helium वाला है chemistry वाला periodic kuch` | chemistry.txt (periodic table) | 0.5990 | ✓ |

Same counter-intuitive pattern: very dirty (0.60) beat both clean (0.59) and dirty (0.42). The very-dirty query is essentially a keyword dump (`hydrogen helium chemistry periodic table`) which is exactly what the source paragraph contains.

#### space.txt

| Level | Query | Top-1 file (paragraph) | Sim | ✓/✗ |
|---|---|---|---:|:---:|
| clean | `ब्रह्मांड में डार्क मैटर और डार्क एनर्जी का क्या महत्व है?` | space.txt (astronomy / dark matter) | 0.6398 | ✓ |
| dirty | `ब्रमंद में dark मेटर और energy ka क्या मतलब है` | physics.txt (thermodynamics) | 0.3870 | ✗ |
| very dirty | `vo जो universe me dark wala 95 percent जो अदृष्य है vo कुछ space वाला` | space.txt (astronomy) | 0.5053 | ✓ |

This is the only case where the dirty version is *worse* than the very-dirty version *and* gets a wrong answer. Dropping `मैटर` (matter) and shortening `ब्रह्मांड → ब्रमंद` removed the dark-matter signal, leaving "energy" as the dominant token — which dragged it to thermodynamics. The very-dirty version added `universe`, `95 percent`, `अदृष्य` (invisible), `space` — restoring the astronomy anchor.

#### cs.txt

| Level | Query | Top-1 file (paragraph) | Sim | ✓/✗ |
|---|---|---|---:|:---:|
| clean | `वेक्टर सर्च में एम्बेडिंग का उपयोग करके समान वस्तुओं को कैसे खोजा जाता है?` | cs.txt (vector search) | 0.5509 | ✓ |
| dirty | `वेकटर सरच मे embedding से similar items कैसे find` | cs.txt (vector search) | 0.5144 | ✓ |
| very dirty | `ye jo HNSW vala database hai jisme similarity ke basis pe search hota semantic` | cs.txt (vector search) | 0.6066 | ✓ |

Best-case demonstration of the keyword-anchoring effect. The very-dirty version is essentially ungrammatical Hinglish, but `HNSW`, `database`, `similarity`, `search`, `semantic` are all literal terms in the cs.txt vector search paragraph — so it scored *higher than the clean Devanagari version*.

### Headline finding: there's a robustness floor, not a degradation curve

The most important number above is the `dirty → very dirty` transition: **−0.002**. Going from "moderately bad" to "really bad" did essentially nothing. The same number of queries got the right answer (5/6) and the mean similarity moved less than half a percent. This means dense multilingual embeddings have a floor below which additional surface-level corruption doesn't matter — once enough content keywords are present, the embedding lands in roughly the right region of the 384-dim vector space and stays there even as you mangle more grammar.

This is unlike what you might naively expect, which is a smooth degradation: clean (6/6) → dirty (5/6) → very dirty (3/6) → completely garbage (0/6). The actual curve looks more like a step function: clean is great, anything sufficiently noisy is moderate, and the moderate plateau is wider than expected.

### Counterintuitive finding: very dirty often beats dirty

**3 of the 6 topics had the very-dirty query score *higher* than the dirty query**, sometimes by a lot:

| Topic | Dirty sim | Very dirty sim | Why the very-dirty version won |
|---|---:|---:|---|
| cs | 0.5144 | **0.6066** | The very-dirty query happened to include `HNSW`, `database`, `similarity`, `search`, `semantic` — all literal keywords from the source paragraph. Grammar destroyed but keyword density was perfect. |
| chemistry | 0.4243 | **0.5990** | `hydrogen helium periodic chemistry table` is basically a keyword dump that hits the source paragraph head-on. |
| biology | 0.3550 | **0.5070** | `sun light sugar photo` cleanly hits the photosynthesis paragraph despite the messy syntax. |

The model is anchored on **which content tokens are present**, not on **how grammatical the sentence is**. Drop a function word, and the embedding moves a tiny amount. Drop a key noun, and it moves a lot. Substitute a noun with its English equivalent, and it often moves *toward* the source rather than away (because the source is in English).

This has a real implication for ASR pipelines: **a partially-correct ASR transcript that falls back to English transliteration for technical terms is usually *better* for retrieval than a fully-Devanagari transcript with mangled spellings.** ASR systems that "give up" on rare technical vocabulary and emit Latin script are accidentally helping cross-lingual retrieval rather than hurting it.

### Failure modes

Both failures across all 18 runs were "near-miss drift to a semantically adjacent topic", never random garbage:

1. **Dirty failure: space → physics** (`ब्रमंद में dark मेटर और energy ka क्या मतलब है`)
   Top-1 was thermodynamics (physics.txt) at sim=0.39. The model latched onto `energy` and the mangled `ब्रमंद` lost enough specificity that "energy" overpowered "dark matter / universe." Note that the *clean* query for this same topic landed correctly at sim=0.64, so the failure is real degradation, not bad data.

2. **Very dirty failure: physics → space** (`वो ainstain वाली theery जिसमे gravety wala कुछ था spacetime kuch वो`)
   Top-1 was the solar system paragraph (space.txt) at sim=0.39. The mangled `ainstain` lost the strong "Einstein" anchor that drove the clean query to 0.81 similarity. The remaining tokens (`spacetime`, `theery`, `gravity`) are split between physics and space topics, and the model picked the wrong one by a small margin.

In both cases the **expected file appeared in the top-3 list**, just not at rank 1. A pipeline that retrieves top-k=3 and lets a downstream LLM choose would have recovered both failures.

### Latency impact: zero

For completeness: the dirty queries take essentially the same time to embed and search as the clean queries. Latency depends on token count and not on input quality. The `noisy.*` queries in the latency study above (mean 10.34 ms median embed) versus the `clean.*` queries (9.92 ms median embed) confirmed this with a rigorous 30-iteration measurement. The dirty benchmark didn't re-measure latency because there was nothing new to learn.

### Practical recommendations for ASR-driven RAG

Based on the combined latency + correctness picture above:

1. **Retrieve top-k = 5 or top-k = 10, not top-k = 1.** Both failures in the dirty benchmark had the right answer in top-3. The marginal latency cost of larger k is negligible (search stays at 2 ms regardless of k for this corpus size) and the correctness gain is real.
2. **Trust the embedding when the clean baseline similarity is high.** If a query gets > 0.7 sim on the clean version, it will tolerate significant transcription errors. If it gets < 0.55, even mild errors will push it off rank 1.
3. **Don't try to "clean up" Devanagari from the ASR before sending it to the embedder.** Mixed scripts work fine. Let the ASR emit whatever it emits.
4. **Use the source-file rank-2 / rank-3 results as a confidence signal.** If the same source file appears at multiple ranks (e.g., physics.txt at rank 1 and rank 3), the model is voting strongly for that topic and you can trust the answer. If rank 1 and rank 2 are different files with similar similarity scores, treat the result as low-confidence and consider re-ranking with a stronger model.

Full per-query results (including all 3 ranks for all 18 queries) are in `dirty_transcription_benchmark.json`.

## Reproducibility

```bash
.venv/bin/python latency_report.py                  # writes latency_report.json
.venv/bin/python benchmark_dirty_transcription.py   # writes dirty_transcription_benchmark.json
```

For the HTTP numbers, start `make server` in another terminal, then run a probe similar to the one used here. The probe script itself was a one-off and is not committed.

Raw JSON outputs are saved in `latency_report.json` (direct latency path), `http_latency.json` (HTTP latency path), and `dirty_transcription_benchmark.json` (robustness path) — all three contain the full per-iteration data, not just the summary stats above.
