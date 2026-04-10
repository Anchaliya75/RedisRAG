#!/usr/bin/env python3
"""Detailed latency profile for RedisRAG.

Measures:
  - Model load time (one-time, cold)
  - Cold-start query latency (first embed after model load)
  - Warm latency stats over N iterations per query
      embed_ms, search_ms, total_ms — min / max / mean / median / p50 / p95 / p99 / stdev
  - Per-query breakdown across both clean Devanagari and noisy ASR-style queries
  - Full response payload shape for one example query

Run:  .venv/bin/python latency_report.py
Out:  prints a multi-table report and saves latency_report.json
"""

import json
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import time
from pathlib import Path
from statistics import mean, median, pstdev

import torch
torch.set_num_threads(1)

import numpy as np
from redisvl.index import SearchIndex
from redisvl.query import VectorQuery
from redisvl.utils.vectorize import HFTextVectorizer

from config import REDIS_URL, MODEL_NAME, SCHEMA

ITERATIONS = 30  # per query, after warmup
WARMUP_ITERATIONS = 3
K = 3
OUT_FILE = Path(__file__).parent / "latency_report.json"

# (label, query) — mix of clean Devanagari and ASR-noisy variants
QUERIES = [
    ("clean.physics",  "आइंस्टीन की सापेक्षता के सिद्धांत में गुरुत्वाकर्षण को कैसे समझाया गया है?"),
    ("clean.biology",  "प्रकाश संश्लेषण की प्रक्रिया में पौधे सूर्य के प्रकाश से ग्लूकोज कैसे बनाते हैं?"),
    ("clean.ai",       "ट्रांसफॉर्मर मॉडल सेल्फ-अटेंशन का उपयोग करके भाषा को कैसे प्रोसेस करते हैं?"),
    ("clean.cs",       "वेक्टर सर्च में एम्बेडिंग का उपयोग करके समान वस्तुओं को कैसे खोजा जाता है?"),
    ("noisy.physics",  "आइनस्टिन की सपेक्षता के सिधांत मे गुरुत्वाकरसन कसे समझाया गया"),
    ("noisy.biology",  "प्रकास सनसलेसन में पोधे suraj की रोसनी से glucose कैसे बनाते"),
    ("noisy.ai",       "transformer मोडल self attension का उपयोग करके लंगुएज को कैसे प्रोसेस"),
    ("noisy.cs",       "वेकटर सरच में embedding का उपयोग करके सामान वस्तुओ को कैसे खोजा"),
]


def percentile(sorted_samples, p):
    """Linear-interpolation percentile (works for any sample size)."""
    n = len(sorted_samples)
    if n == 0:
        return 0.0
    if n == 1:
        return sorted_samples[0]
    k = (n - 1) * p / 100
    f = int(k)
    c = min(f + 1, n - 1)
    if f == c:
        return sorted_samples[f]
    return sorted_samples[f] + (k - f) * (sorted_samples[c] - sorted_samples[f])


def stats(samples):
    s = sorted(samples)
    return {
        "n":      len(s),
        "min":    s[0],
        "max":    s[-1],
        "mean":   mean(s),
        "median": median(s),
        "p50":    percentile(s, 50),
        "p95":    percentile(s, 95),
        "p99":    percentile(s, 99),
        "stdev":  pstdev(s) if len(s) > 1 else 0.0,
    }


def fmt(d):
    return (f"{d['min']:7.2f} {d['median']:7.2f} {d['mean']:7.2f} "
            f"{d['p95']:7.2f} {d['p99']:7.2f} {d['max']:7.2f} {d['stdev']:7.2f}")


def time_query(vectorizer, index, text):
    t0 = time.perf_counter()
    emb = vectorizer.embed(text)
    embed_ms = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    results = index.query(VectorQuery(
        vector=np.array(emb, dtype=np.float32).tobytes(),
        vector_field_name="embedding",
        return_fields=["title", "content"],
        num_results=K,
    ))
    search_ms = (time.perf_counter() - t0) * 1000

    return embed_ms, search_ms, results


def main():
    print("=" * 96)
    print(f"  RedisRAG Latency Report")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Index: {SCHEMA['index']['name']}  (HNSW, 384 dims, cosine)")
    print(f"  Iterations: {ITERATIONS} per query (after {WARMUP_ITERATIONS} warmup)")
    print(f"  Top-K: {K}")
    print(f"  Threads: OMP={os.environ.get('OMP_NUM_THREADS')}, torch={torch.get_num_threads()}")
    print("=" * 96)

    # ---- Model load (one-time cold) ---------------------------------------
    t0 = time.perf_counter()
    vectorizer = HFTextVectorizer(model=MODEL_NAME)
    model_load_ms = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    index = SearchIndex.from_dict(SCHEMA, redis_url=REDIS_URL)
    index_connect_ms = (time.perf_counter() - t0) * 1000

    print(f"\n[startup]  model_load = {model_load_ms:8.1f} ms   "
          f"index_connect = {index_connect_ms:6.1f} ms")

    # ---- Cold-start query (first embed) -----------------------------------
    cold_embed, cold_search, _ = time_query(vectorizer, index, QUERIES[0][1])
    print(f"[cold]     first query embed = {cold_embed:7.2f} ms   "
          f"search = {cold_search:6.2f} ms   "
          f"(includes any lazy model warmup)")

    # ---- Warmup -----------------------------------------------------------
    for _ in range(WARMUP_ITERATIONS):
        time_query(vectorizer, index, QUERIES[0][1])

    # ---- Measurement loop -------------------------------------------------
    per_query = {}
    sample_response_doc = None  # capture one full response for the report

    print(f"\n{'query':<18} {'len':>4} | {'min':>7} {'med':>7} {'mean':>7} "
          f"{'p95':>7} {'p99':>7} {'max':>7} {'stdev':>7}  metric")
    print("-" * 96)

    for label, q in QUERIES:
        embeds, searches, totals = [], [], []
        for _ in range(ITERATIONS):
            e_ms, s_ms, results = time_query(vectorizer, index, q)
            embeds.append(e_ms)
            searches.append(s_ms)
            totals.append(e_ms + s_ms)
            if sample_response_doc is None and label == "clean.physics":
                sample_response_doc = {
                    "query": q,
                    "results": [
                        {
                            "title": doc["title"],
                            "similarity": round(1 - float(doc["vector_distance"]), 4),
                            "content": doc["content"],
                        }
                        for doc in results
                    ],
                }

        e = stats(embeds)
        s = stats(searches)
        t = stats(totals)
        per_query[label] = {"query": q, "len_chars": len(q),
                            "embed": e, "search": s, "total": t}

        print(f"{label:<18} {len(q):>4} | {fmt(e)}  embed_ms")
        print(f"{'':<18} {'':>4} | {fmt(s)}  search_ms")
        print(f"{'':<18} {'':>4} | {fmt(t)}  total_ms")
        print()

    # ---- Aggregate across all queries -------------------------------------
    all_embed = [v for q in per_query.values() for v in [q["embed"]["median"]]]
    all_search = [v for q in per_query.values() for v in [q["search"]["median"]]]
    clean_embed = [per_query[k]["embed"]["median"] for k in per_query if k.startswith("clean")]
    noisy_embed = [per_query[k]["embed"]["median"] for k in per_query if k.startswith("noisy")]

    print("=" * 96)
    print("AGGREGATE (median-of-medians across queries)")
    print("-" * 96)
    print(f"  embed   all:    {mean(all_embed):6.2f} ms")
    print(f"  embed   clean:  {mean(clean_embed):6.2f} ms   (mean over {len(clean_embed)} queries)")
    print(f"  embed   noisy:  {mean(noisy_embed):6.2f} ms   (mean over {len(noisy_embed)} queries)")
    print(f"  search  all:    {mean(all_search):6.2f} ms")
    print()

    # ---- Sample response payload ------------------------------------------
    print("=" * 96)
    print("SAMPLE RESPONSE PAYLOAD  (top-3 for clean.physics)")
    print("-" * 96)
    print(json.dumps(sample_response_doc, ensure_ascii=False, indent=2))
    print()

    # ---- Save -------------------------------------------------------------
    report = {
        "config": {
            "model": MODEL_NAME,
            "index": SCHEMA["index"]["name"],
            "dims": 384,
            "k": K,
            "iterations": ITERATIONS,
            "warmup_iterations": WARMUP_ITERATIONS,
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            "torch_num_threads": torch.get_num_threads(),
        },
        "startup": {
            "model_load_ms": round(model_load_ms, 2),
            "index_connect_ms": round(index_connect_ms, 2),
        },
        "cold_start_query": {
            "embed_ms": round(cold_embed, 2),
            "search_ms": round(cold_search, 2),
        },
        "per_query": per_query,
        "sample_response": sample_response_doc,
    }
    OUT_FILE.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=float))
    print(f"Full report saved to {OUT_FILE.name}")


if __name__ == "__main__":
    main()
