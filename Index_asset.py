import json
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

client = OpenAI()

chroma = chromadb.PersistentClient(
    path=str(BASE_DIR / "chroma_db")
)

collection = chroma.get_or_create_collection(
    name="asset_profiles"
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

    text = (
        f"Asset: {asset}. "
        f"Asset type: {data['asset_type']}. "
        f"Aliases: {', '.join(data['aliases'])}. "
        f"Analysis dimensions: {', '.join(data['analysis_dimensions'])}."
    )

    documents.append(text)

    ids.append(f"asset:{asset}")

    metadatas.append({
        "canonical_asset": asset,
        "asset_type": data["asset_type"]
    })


response = client.embeddings.create(
    model="text-embedding-3-small",
    input=documents
)

embeddings = [
    item.embedding
    for item in response.data
]


collection.upsert(
    ids=ids,
    documents=documents,
    metadatas=metadatas,
    embeddings=embeddings
)


print(f"Indexed {len(ids)} assets.")