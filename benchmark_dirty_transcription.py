#!/usr/bin/env python3
"""Dirty-transcription robustness benchmark.

For each of 6 topics in the corpus we hand-wrote three Devanagari query
variants representing what an ASR pipeline might output at increasing levels
of degradation:

  level 0  clean       — well-formed Hindi question (baseline)
  level 1  dirty       — typical ASR errors: dropped matras, swapped consonants
                         (श/स, ब/व, ण/न), some Latin-script fragments
  level 2  very dirty  — heavy code-switching, missing function words, wrong
                         word substitutions, sometimes the question structure
                         is gone entirely and only fragments remain

For each variant we run a top-3 vector search against the same Redis index
that was built from the English data files, and check whether the expected
source file shows up at rank 1.

This is the "kuch galat likha ho aur bol rha ho" test — does dense retrieval
still work when the surface text is garbage?
"""

import json
import time
from pathlib import Path

import numpy as np
from redisvl.index import SearchIndex
from redisvl.query import VectorQuery
from redisvl.utils.vectorize import HFTextVectorizer

from config import REDIS_URL, MODEL_NAME, SCHEMA

K = 3
OUT_FILE = Path(__file__).parent / "dirty_transcription_benchmark.json"

# (expected_file, [(level_label, query), ...])
TEST_CASES = [
    (
        "physics.txt",
        [
            ("clean",      "आइंस्टीन की सापेक्षता के सिद्धांत में गुरुत्वाकर्षण को कैसे समझाया गया है?"),
            ("dirty",      "आइनस्टिन की सपेक्षता मे gravity को कसे समझाया गुरुत्वकरसन"),
            ("very dirty", "वो ainstain वाली theery जिसमे gravety wala कुछ था spacetime kuch वो"),
        ],
    ),
    (
        "biology.txt",
        [
            ("clean",      "प्रकाश संश्लेषण की प्रक्रिया में पौधे सूर्य के प्रकाश से ग्लूकोज कैसे बनाते हैं?"),
            ("dirty",      "परकास सनस्लेसन में पौदा suraj की रोसनी से gluucose बनाता"),
            ("very dirty", "vo पेड़ wala chemistry जिसमे sun light से sugar बनती photo कुछ"),
        ],
    ),
    (
        "ai.txt",
        [
            ("clean",      "ट्रांसफॉर्मर मॉडल सेल्फ-अटेंशन का उपयोग करके भाषा को कैसे प्रोसेस करते हैं?"),
            ("dirty",      "transfarmer मॉडेल self atension से language कैसे process करता"),
            ("very dirty", "vo जो attention वाला model GPT जैसा mechanism sentence parse karne ka"),
        ],
    ),
    (
        "chemistry.txt",
        [
            ("clean",      "आवर्त सारणी में तत्वों को परमाणु क्रमांक के अनुसार कैसे व्यवस्थित किया जाता है?"),
            ("dirty",      "आवरत साराणी में elements को परमानु number के हिसाब से arrange"),
            ("very dirty", "jo table mein hydrogen helium वाला है chemistry वाला periodic kuch"),
        ],
    ),
    (
        "space.txt",
        [
            ("clean",      "ब्रह्मांड में डार्क मैटर और डार्क एनर्जी का क्या महत्व है?"),
            ("dirty",      "ब्रमंद में dark मेटर और energy ka क्या मतलब है"),
            ("very dirty", "vo जो universe me dark wala 95 percent जो अदृष्य है vo कुछ space वाला"),
        ],
    ),
    (
        "cs.txt",
        [
            ("clean",      "वेक्टर सर्च में एम्बेडिंग का उपयोग करके समान वस्तुओं को कैसे खोजा जाता है?"),
            ("dirty",      "वेकटर सरच मे embedding से similar items कैसे find"),
            ("very dirty", "ye jo HNSW vala database hai jisme similarity ke basis pe search hota semantic"),
        ],
    ),
]


def run_query(vectorizer, index, text):
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

    hits = [
        {
            "rank": i + 1,
            "title": doc["title"],
            "similarity": round(1 - float(doc["vector_distance"]), 4),
            "snippet": doc["content"][:90] + ("..." if len(doc["content"]) > 90 else ""),
        }
        for i, doc in enumerate(results)
    ]
    return hits, embed_ms, search_ms


def fmt_hit(h, expected):
    mark = "✓" if h["title"] == expected else "✗"
    return f"  {mark} #{h['rank']} {h['title']:<14} sim={h['similarity']:.4f}  {h['snippet']}"


def main():
    vectorizer = HFTextVectorizer(model=MODEL_NAME)
    index = SearchIndex.from_dict(SCHEMA, redis_url=REDIS_URL)
    vectorizer.embed("warmup")  # avoid first-call cold latency in the per-query numbers

    print("=" * 100)
    print(f"  Dirty Transcription Robustness Benchmark")
    print(f"  Model: {MODEL_NAME}")
    print(f"  6 topics × 3 levels (clean / dirty / very dirty) = 18 query runs")
    print("=" * 100)

    log = []
    per_level = {"clean": {"correct": 0, "sims": []},
                 "dirty": {"correct": 0, "sims": []},
                 "very dirty": {"correct": 0, "sims": []}}

    for expected, variants in TEST_CASES:
        print(f"\n── {expected}  (expected source) " + "─" * (66 - len(expected)))
        case_log = {"expected": expected, "variants": []}

        for level, query in variants:
            hits, e_ms, s_ms = run_query(vectorizer, index, query)
            top1 = hits[0]
            correct = top1["title"] == expected
            per_level[level]["correct"] += int(correct)
            per_level[level]["sims"].append(top1["similarity"])

            print(f"\n  [{level:>10}]  {query}")
            for h in hits:
                print(fmt_hit(h, expected))

            case_log["variants"].append({
                "level": level,
                "query": query,
                "top1_correct": correct,
                "embed_ms": round(e_ms, 2),
                "search_ms": round(s_ms, 2),
                "hits": hits,
            })
        log.append(case_log)

    # ---- Summary -----------------------------------------------------------
    n = len(TEST_CASES)
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("-" * 100)
    print(f"{'level':<12} {'top-1 correct':<16} {'mean top-1 sim':<18} {'min':<10} {'max':<10}")
    for level in ("clean", "dirty", "very dirty"):
        sims = per_level[level]["sims"]
        c = per_level[level]["correct"]
        mean_sim = sum(sims) / len(sims)
        print(f"{level:<12} {c}/{n:<14} {mean_sim:<18.4f} {min(sims):<10.4f} {max(sims):<10.4f}")

    clean_mean = sum(per_level["clean"]["sims"]) / n
    dirty_mean = sum(per_level["dirty"]["sims"]) / n
    very_dirty_mean = sum(per_level["very dirty"]["sims"]) / n
    print()
    print(f"  similarity drop  clean → dirty       : {clean_mean - dirty_mean:+.4f}")
    print(f"  similarity drop  clean → very dirty  : {clean_mean - very_dirty_mean:+.4f}")
    print(f"  similarity drop  dirty → very dirty  : {dirty_mean - very_dirty_mean:+.4f}")

    OUT_FILE.write_text(json.dumps({
        "config": {"model": MODEL_NAME, "k": K, "n_topics": n},
        "per_level_summary": {
            level: {
                "top1_correct": per_level[level]["correct"],
                "mean_top1_sim": round(sum(per_level[level]["sims"]) / n, 4),
                "min_top1_sim": round(min(per_level[level]["sims"]), 4),
                "max_top1_sim": round(max(per_level[level]["sims"]), 4),
            } for level in ("clean", "dirty", "very dirty")
        },
        "cases": log,
    }, ensure_ascii=False, indent=2))
    print(f"\nFull results saved to {OUT_FILE.name}")


if __name__ == "__main__":
    main()
