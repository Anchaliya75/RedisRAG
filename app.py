#!/usr/bin/env python3
"""FastAPI server for RedisRAG — model loaded once at startup."""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import torch
torch.set_num_threads(1)

import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from redisvl.index import SearchIndex
from redisvl.query import VectorQuery
from redisvl.utils.vectorize import HFTextVectorizer

from config import REDIS_URL, MODEL_NAME, SCHEMA

app = FastAPI(title="RedisRAG")

DATA_DIR = Path(__file__).parent / "data"

# Loaded once at startup
vectorizer = None
index = None


@app.on_event("startup")
def startup():
    global vectorizer, index
    vectorizer = HFTextVectorizer(model=MODEL_NAME)
    index = SearchIndex.from_dict(SCHEMA, redis_url=REDIS_URL)
    print(f"Model loaded, connected to Redis")


class SearchRequest(BaseModel):
    query: str
    k: int = 3


class SearchResult(BaseModel):
    title: str
    content: str
    similarity: float


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]
    embed_ms: float
    search_ms: float


class IngestResponse(BaseModel):
    files: int
    chunks: int


@app.post("/ingest", response_model=IngestResponse)
def ingest():
    """Read all txt files from data/, chunk by paragraph, embed, and store."""
    all_chunks = []
    for f in sorted(DATA_DIR.glob("*.txt")):
        text = f.read_text().strip()
        for para in text.split("\n\n"):
            para = para.strip()
            if para:
                all_chunks.append({"source": f.name, "content": para})

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

    file_count = len(list(DATA_DIR.glob("*.txt")))
    return IngestResponse(files=file_count, chunks=len(all_chunks))


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    """Embed query and search Redis for similar chunks."""
    t0 = time.time()
    emb = vectorizer.embed(req.query)
    embed_ms = (time.time() - t0) * 1000

    t0 = time.time()
    results = index.query(VectorQuery(
        vector=np.array(emb, dtype=np.float32).tobytes(),
        vector_field_name="embedding",
        return_fields=["title", "content"],
        num_results=req.k,
    ))
    search_ms = (time.time() - t0) * 1000

    hits = [
        SearchResult(
            title=doc["title"],
            content=doc["content"],
            similarity=round(1 - float(doc["vector_distance"]), 4),
        )
        for doc in results
    ]

    return SearchResponse(
        query=req.query,
        results=hits,
        embed_ms=round(embed_ms, 2),
        search_ms=round(search_ms, 2),
    )
