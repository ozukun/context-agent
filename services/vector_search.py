from pathlib import Path

import chromadb
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()


PROJECT_ROOT = Path(__file__).resolve().parent.parent

CHROMA_PATH = PROJECT_ROOT / "chroma_db"
COLLECTION_NAME = "asset_profiles"


openai_client = OpenAI()


chroma_client = chromadb.PersistentClient(
    path=str(CHROMA_PATH)
)

collection = chroma_client.get_collection(
    name=COLLECTION_NAME
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

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    candidates = []

    ids = results["ids"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for asset_id, metadata, distance in zip(
        ids,
        metadatas,
        distances
    ):

        candidates.append({
            "id": asset_id,
            "canonical_asset": metadata["canonical_asset"],
            "asset_type": metadata["asset_type"],
            "distance": distance
        })

    return candidates


if __name__ == "__main__":

    results = vector_search(
        "american crude"
    )

    for result in results:
        print(result)