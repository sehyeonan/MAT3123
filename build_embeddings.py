"""
Embed NAVER and IMDB review texts using Upstage embeddings and save compact files.

Usage (requires network access):
    export UPSTAGE_API_KEY="YOUR_KEY"
    pip install openai==1.52.2 numpy
    python data_pipeline/build_embeddings.py

Outputs (float16 compressed):
    output/embeddings/naver_embeddings.npz   # ids, embeddings
    output/embeddings/imdb_embeddings.npz    # ids, embeddings
"""

from __future__ import annotations

import os
import csv
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
from openai import OpenAI


MODEL_NAME = "embedding-query"
BATCH_SIZE = 32

BASE_URL = "https://api.upstage.ai/v1"
API_KEY_ENV = "UPSTAGE_API_KEY"

DATA_DIR = Path("output")
EMBED_DIR = DATA_DIR / "embeddings"


def load_texts(csv_path: Path, text_column: str) -> Tuple[List[str], List[str]]:
    ids: List[str] = []
    texts: List[str] = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = (row.get(text_column) or "").strip()
            if not text:
                continue
            ids.append(row["id"])
            texts.append(text)
    return ids, texts


def chunked(iterable: List[str], size: int) -> Iterable[List[str]]:
    for i in range(0, len(iterable), size):
        yield iterable[i : i + size]


def embed_texts(client: OpenAI, texts: List[str]) -> List[np.ndarray]:
    vectors: List[np.ndarray] = []
    total = len(texts)
    processed = 0
    for batch in chunked(texts, BATCH_SIZE):
        resp = client.embeddings.create(model=MODEL_NAME, input=batch)
        # Ensure order matches input
        resp_sorted = sorted(resp.data, key=lambda d: d.index)
        vectors.extend(np.asarray(item.embedding, dtype=np.float32) for item in resp_sorted)
        processed += len(batch)
        print(f"[embed] {processed}/{total} done", flush=True)
    return vectors


def save_embeddings(ids: List[str], vectors: List[np.ndarray], out_path: Path) -> None:
    if len(ids) != len(vectors):
        raise ValueError("ids and vectors lengths differ")
    EMBED_DIR.mkdir(parents=True, exist_ok=True)
    arr = np.stack([np.asarray(v, dtype=np.float16) for v in vectors])
    id_arr = np.array(ids, dtype="U")
    np.savez_compressed(out_path, ids=id_arr, embeddings=arr)


def main() -> None:
    api_key = os.getenv(API_KEY_ENV)
    if not api_key:
        raise RuntimeError(f"{API_KEY_ENV} is not set; export your Upstage API key first.")

    client = OpenAI(api_key=api_key, base_url=BASE_URL)

    # NAVER: use original Korean comment text
    naver_ids, naver_texts = load_texts(DATA_DIR / "reviews_naver_merged.csv", "original_text")
    naver_vecs = embed_texts(client, naver_texts)
    save_embeddings(naver_ids, naver_vecs, EMBED_DIR / "naver_embeddings.npz")

    # IMDB: use English trimmed content
    imdb_ids, imdb_texts = load_texts(DATA_DIR / "imdb_clean.csv", "content_trim")
    imdb_vecs = embed_texts(client, imdb_texts)
    save_embeddings(imdb_ids, imdb_vecs, EMBED_DIR / "imdb_embeddings.npz")


if __name__ == "__main__":
    main()
