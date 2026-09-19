from pathlib import Path

import chromadb
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

openai_client = OpenAI()

chroma_client = chromadb.PersistentClient(
    path=str(BASE_DIR / "chroma_db")
)

collection = chroma_client.get_collection(
    name="asset_profiles"
)


def create_embedding(text: str) -> list[float]:

    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )

    return response.data[0].embedding


def vector_search(
    query: str,
    top_k: int = 3
) -> list[dict]:

    query_embedding = create_embedding(query)

    # Daha fazla vector record çekiyoruz.
    # Sonra canonical asset bazında tekilleştireceğiz.
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k * 5
    )

    candidates = []
    seen_assets = set()

    ids = results["ids"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for asset_id, metadata, distance in zip(
        ids,
        metadatas,
        distances
    ):

        canonical_asset = metadata["canonical_asset"]

        # Aynı asset'i ikinci kez ekleme
        if canonical_asset in seen_assets:
            continue

        seen_assets.add(canonical_asset)

        candidates.append({
            "id": asset_id,
            "canonical_asset": canonical_asset,
            "asset_type": metadata["asset_type"],
            "distance": distance
        })

        if len(candidates) == top_k:
            break

    return candidates


if __name__ == "__main__":

    tests = [
        "chip maker",
        "digital currency",
        "euv machine maker",
        "american crude",
        "nvidia"
    ]

    for test in tests:

        print()
        print("QUERY:", test)

        results = vector_search(
            test,
            top_k=3
        )

        for result in results:
            print(result)