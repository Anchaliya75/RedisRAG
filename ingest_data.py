#!/usr/bin/env python3
"""Read files from data/, chunk by paragraph, embed, and store in Redis."""

from pathlib import Path
import numpy as np
from redisvl.index import SearchIndex
from redisvl.utils.vectorize import HFTextVectorizer
from config import REDIS_URL, MODEL_NAME, SCHEMA

DATA_DIR = Path(__file__).parent / "data"


def chunk_file(filepath):
    """Split a file into chunks by paragraph (blank line separated)."""
    text = filepath.read_text().strip()
    chunks = []
    for para in text.split("\n\n"):
        para = para.strip()
        if para:
            chunks.append({
                "source": filepath.name,
                "content": para,
            })
    return chunks


def main():
    all_chunks = []
    for f in sorted(DATA_DIR.glob("*.txt")):
        chunks = chunk_file(f)
        print(f"  {f.name}: {len(chunks)} chunks")
        all_chunks.extend(chunks)

    print(f"\nTotal: {len(all_chunks)} chunks from {len(list(DATA_DIR.glob('*.txt')))} files")

    vectorizer = HFTextVectorizer(model=MODEL_NAME)
    index = SearchIndex.from_dict(SCHEMA, redis_url=REDIS_URL)
    index.create(overwrite=True)

    embeddings = vectorizer.embed_many([c["content"] for c in all_chunks])
    index.load([
        {
            "title": c["source"],
            "content": c["content"],
            "embedding": np.array(e, dtype=np.float32).tobytes(),
        }
        for c, e in zip(all_chunks, embeddings)
    ])
    print(f"Indexed {len(all_chunks)} chunks")


if __name__ == "__main__":
    main()
