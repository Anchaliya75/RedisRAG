REDIS_URL = "redis://:redis@localhost:6379"
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

SCHEMA = {
    "index": {"name": "voice-index", "prefix": "voice:"},
    "fields": [
        {"name": "title", "type": "text"},
        {"name": "content", "type": "text"},
        {"name": "embedding", "type": "vector", "attrs": {
            "algorithm": "hnsw", "dims": 384, "distance_metric": "cosine", "datatype": "float32",
        }},
    ],
}
