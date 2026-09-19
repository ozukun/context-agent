import json
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
COLLECTION_NAME = "asset_profiles"

openai_client = OpenAI()

chroma = chromadb.PersistentClient(
    path=str(BASE_DIR / "chroma_db")
)


# Eski index'i temizle
try:
    chroma.delete_collection(
        name=COLLECTION_NAME
    )
except Exception:
    pass


collection = chroma.create_collection(
    name=COLLECTION_NAME
)


with open(
    BASE_DIR / "context" / "asset_profiles.json",
    "r",
    encoding="utf-8"
) as f:
    profiles = json.load(f)


documents = []
ids = []
metadatas = []


for asset, data in profiles.items():

    # ----------------------------------
    # Base asset document
    # ----------------------------------

    base_text = (
        f"Asset: {asset}. "
        f"Asset type: {data['asset_type']}. "
        f"Aliases: {', '.join(data['aliases'])}."
    )

    documents.append(base_text)

    ids.append(
        f"asset:{asset}:base"
    )

    metadatas.append({
        "canonical_asset": asset,
        "asset_type": data["asset_type"],
        "record_type": "base"
    })


    # ----------------------------------
    # Semantic terms
    # ----------------------------------

    for index, term in enumerate(
        data.get("semantic_terms", [])
    ):

        documents.append(term)

        ids.append(
            f"asset:{asset}:semantic:{index}"
        )

        metadatas.append({
            "canonical_asset": asset,
            "asset_type": data["asset_type"],
            "record_type": "semantic",
            "semantic_term": term
        })


# ----------------------------------
# Create embeddings
# ----------------------------------

response = openai_client.embeddings.create(
    model="text-embedding-3-small",
    input=documents
)


embeddings = [
    item.embedding
    for item in response.data
]


# ----------------------------------
# Store in ChromaDB
# ----------------------------------

collection.upsert(
    ids=ids,
    documents=documents,
    metadatas=metadatas,
    embeddings=embeddings
)


print(
    f"Indexed {len(ids)} vector records."
)