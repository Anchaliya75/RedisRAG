#!/usr/bin/env python3
"""Query the Redis vector index and save latency to latency.json."""

import json
import time
from pathlib import Path
import numpy as np
from redisvl.index import SearchIndex
from redisvl.query import VectorQuery
from redisvl.utils.vectorize import HFTextVectorizer
from config import REDIS_URL, MODEL_NAME, SCHEMA

QUERIES = [
    "How does gravity work?",
    "Tell me about deep learning",
]

LATENCY_FILE = Path(__file__).parent / "latency.json"


def main():
    vectorizer = HFTextVectorizer(model=MODEL_NAME)
    index = SearchIndex.from_dict(SCHEMA, redis_url=REDIS_URL)

    latency_log = []
    for q in QUERIES:
        t0 = time.time()
        emb = vectorizer.embed(q)
        embed_ms = (time.time() - t0) * 1000

        t0 = time.time()
        results = index.query(VectorQuery(
            vector=np.array(emb, dtype=np.float32).tobytes(),
            vector_field_name="embedding",
            return_fields=["title", "content"], num_results=3,
        ))
        search_ms = (time.time() - t0) * 1000

        print(f'Query: "{q}"')
        hits = []
        for doc in results:
            sim = 1 - float(doc["vector_distance"])
            print(f"  {sim:.4f}  {doc['title']}")
            hits.append({"title": doc["title"], "similarity": round(sim, 4)})
        print(f"  embed={embed_ms:.1f}ms  search={search_ms:.1f}ms\n")

        latency_log.append({
            "query": q,
            "embed_ms": round(embed_ms, 2),
            "search_ms": round(search_ms, 2),
            "hits": hits,
        })

    LATENCY_FILE.write_text(json.dumps(latency_log, indent=2))
    print(f"Saved to {LATENCY_FILE.name}")


if __name__ == "__main__":
    main()
