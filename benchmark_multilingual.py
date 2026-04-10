#!/usr/bin/env python3
"""Cross-lingual benchmark: Devanagari (Hindi) queries against English corpus.

For each topic we run two queries through the same Redis vector index:
  - clean    : a well-formed Hindi question
  - noisy    : the same question with realistic ASR-style transcription errors
               (dropped matras, swapped consonants like ब/व, श/स, ण/न,
                phonetic look-alikes, occasional Romanized fragments)

The point: dense multilingual embeddings encode meaning, not exact tokens, so
both should land on the same English chunk even when the noisy version would
not match anything by keyword.
"""

import json
import time
from pathlib import Path

import numpy as np
from redisvl.index import SearchIndex
from redisvl.query import VectorQuery
from redisvl.utils.vectorize import HFTextVectorizer

from config import REDIS_URL, MODEL_NAME, SCHEMA

# (expected_file, clean_devanagari, noisy_devanagari)
TEST_CASES = [
    (
        "physics.txt",
        "आइंस्टीन की सापेक्षता के सिद्धांत में गुरुत्वाकर्षण को कैसे समझाया गया है?",
        "आइनस्टिन की सपेक्षता के सिधांत मे गुरुत्वाकरसन कसे समझाया गया",
    ),
    (
        "biology.txt",
        "प्रकाश संश्लेषण की प्रक्रिया में पौधे सूर्य के प्रकाश से ग्लूकोज कैसे बनाते हैं?",
        "प्रकास सनसलेसन में पोधे suraj की रोसनी से glucose कैसे बनाते",
    ),
    (
        "ai.txt",
        "ट्रांसफॉर्मर मॉडल सेल्फ-अटेंशन का उपयोग करके भाषा को कैसे प्रोसेस करते हैं?",
        "transformer मोडल self attension का उपयोग करके लंगुएज को कैसे प्रोसेस",
    ),
    (
        "chemistry.txt",
        "आवर्त सारणी में तत्वों को परमाणु क्रमांक के अनुसार कैसे व्यवस्थित किया जाता है?",
        "आवरत सारनी में तत्वो को परमानु क्रमांक के अनुसार कैसे वेवस्थित",
    ),
    (
        "space.txt",
        "ब्रह्मांड में डार्क मैटर और डार्क एनर्जी का क्या महत्व है?",
        "ब्रमांड में dark matter और dark एनर्जी का क्या मेहत्व",
    ),
    (
        "cs.txt",
        "वेक्टर सर्च में एम्बेडिंग का उपयोग करके समान वस्तुओं को कैसे खोजा जाता है?",
        "वेकटर सरच में embedding का उपयोग करके सामान वस्तुओ को कैसे खोजा",
    ),
]

OUT_FILE = Path(__file__).parent / "multilingual_benchmark.json"


def run_query(vectorizer, index, text, k=3):
    t0 = time.time()
    emb = vectorizer.embed(text)
    embed_ms = (time.time() - t0) * 1000

    t0 = time.time()
    results = index.query(VectorQuery(
        vector=np.array(emb, dtype=np.float32).tobytes(),
        vector_field_name="embedding",
        return_fields=["title", "content"],
        num_results=k,
    ))
    search_ms = (time.time() - t0) * 1000

    hits = [
        {
            "title": doc["title"],
            "similarity": round(1 - float(doc["vector_distance"]), 4),
            "snippet": doc["content"][:80] + "...",
        }
        for doc in results
    ]
    return hits, embed_ms, search_ms


def fmt_hit(h, expected):
    mark = "✓" if h["title"] == expected else "✗"
    return f"{mark} {h['title']:<14} sim={h['similarity']:.4f}  {h['snippet']}"


def main():
    vectorizer = HFTextVectorizer(model=MODEL_NAME)
    index = SearchIndex.from_dict(SCHEMA, redis_url=REDIS_URL)

    # warm up the model so the first query doesn't dominate latency
    vectorizer.embed("warmup")

    log = []
    clean_correct = 0
    noisy_correct = 0
    sim_drops = []

    print(f"\nModel: {MODEL_NAME}")
    print(f"Index: {SCHEMA['index']['name']}  ({len(TEST_CASES)} test cases)\n")
    print("=" * 90)

    for expected, clean_q, noisy_q in TEST_CASES:
        print(f"\nTopic: {expected}")
        print("-" * 90)

        clean_hits, c_emb_ms, c_search_ms = run_query(vectorizer, index, clean_q)
        noisy_hits, n_emb_ms, n_search_ms = run_query(vectorizer, index, noisy_q)

        clean_top_correct = clean_hits[0]["title"] == expected
        noisy_top_correct = noisy_hits[0]["title"] == expected
        clean_correct += int(clean_top_correct)
        noisy_correct += int(noisy_top_correct)
        sim_drops.append(clean_hits[0]["similarity"] - noisy_hits[0]["similarity"])

        print(f"  Clean:  {clean_q}")
        print(f"          → {fmt_hit(clean_hits[0], expected)}")
        print(f"          embed={c_emb_ms:.1f}ms  search={c_search_ms:.1f}ms")
        print(f"  Noisy:  {noisy_q}")
        print(f"          → {fmt_hit(noisy_hits[0], expected)}")
        print(f"          embed={n_emb_ms:.1f}ms  search={n_search_ms:.1f}ms")

        log.append({
            "expected": expected,
            "clean": {"query": clean_q, "hits": clean_hits,
                      "embed_ms": round(c_emb_ms, 2), "search_ms": round(c_search_ms, 2)},
            "noisy": {"query": noisy_q, "hits": noisy_hits,
                      "embed_ms": round(n_emb_ms, 2), "search_ms": round(n_search_ms, 2)},
        })

    print("\n" + "=" * 90)
    print("SUMMARY")
    print("-" * 90)
    n = len(TEST_CASES)
    print(f"  Clean top-1 correct: {clean_correct}/{n}")
    print(f"  Noisy top-1 correct: {noisy_correct}/{n}")
    print(f"  Mean similarity drop (clean → noisy): {sum(sim_drops)/n:+.4f}")
    print()

    OUT_FILE.write_text(json.dumps(log, ensure_ascii=False, indent=2))
    print(f"Full results saved to {OUT_FILE.name}")


if __name__ == "__main__":
    main()
