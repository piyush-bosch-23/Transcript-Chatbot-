from __future__ import annotations

from typing import Any

import requests

from app.config import (
    SUBSCRIPTION_KEY,
    EMBEDDING_URL,
    PROXIES,
    EMBEDDING_BATCH_SIZE,
)


def _post_embeddings(input_value: str | list[str]) -> dict[str, Any]:
    headers = {
        "Content-Type": "application/json",
        "genaiplatform-farm-subscription-key": SUBSCRIPTION_KEY,
    }

    response = requests.post(
        EMBEDDING_URL,
        headers=headers,
        json={"input": input_value},
        proxies=PROXIES,
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def get_embeddings(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []

    all_vectors: list[list[float]] = []

    for batch_start in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        batch = texts[batch_start : batch_start + EMBEDDING_BATCH_SIZE]
        data = _post_embeddings(batch)

        rows = data.get("data", [])
        rows = sorted(rows, key=lambda row: row.get("index", 0))

        for row in rows:
            vector = row.get("embedding")
            if not isinstance(vector, list):
                raise ValueError("Embedding response did not include a valid vector list.")
            all_vectors.append(vector)

    if len(all_vectors) != len(texts):
        raise ValueError("Embedding response size mismatch.")

    return all_vectors


def get_embedding(text: str) -> list[float]:
    vectors = get_embeddings([text])
    return vectors[0]
