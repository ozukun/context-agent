import json
from pathlib import Path

from services.vector_search import vector_search


ASSET_PROFILE_PATH = (
    Path(__file__).resolve().parent.parent
    / "context"
    / "asset_profiles.json"
)

VECTOR_DISTANCE_THRESHOLD = 1.15
VECTOR_TOP_K = 3


def load_asset_profiles() -> dict:

    with open(
        ASSET_PROFILE_PATH,
        "r",
        encoding="utf-8"
    ) as f:
        return json.load(f)


def search_asset(query: str) -> dict:

    profiles = load_asset_profiles()

    normalized_query = query.strip().lower()

    # ------------------------------------------------
    # 1. Exact search
    # ------------------------------------------------

    for asset_name, profile in profiles.items():

        aliases = [
            alias.lower()
            for alias in profile.get("aliases", [])
        ]

        if (
            normalized_query == asset_name.lower()
            or normalized_query in aliases
        ):

            return {
                "found": True,
                "asset": asset_name,
                "asset_type": profile.get("asset_type"),
                "analysis_dimensions": profile.get(
                    "analysis_dimensions",
                    []
                ),
                "match_type": "exact",
                "requires_selection": False
            }

    # ------------------------------------------------
    # 2. Vector search
    # ------------------------------------------------

    vector_results = vector_search(
        query=query,
        top_k=VECTOR_TOP_K
    )

    # ------------------------------------------------
    # 3. Threshold filtering
    # ------------------------------------------------

    candidates = []

    for result in vector_results:

        if result["distance"] <= VECTOR_DISTANCE_THRESHOLD:

            canonical_asset = result["canonical_asset"]

            profile = profiles[canonical_asset]

            candidates.append({
                "asset": canonical_asset,
                "asset_type": profile.get("asset_type"),
                "distance": result["distance"]
            })

    # ------------------------------------------------
    # 4. No valid candidate
    # ------------------------------------------------

    if not candidates:

        return {
            "found": False,
            "asset": None,
            "match_type": "vector_rejected",
            "requires_selection": False,
            "candidates": []
        }

    # ------------------------------------------------
    # 5. Return candidates
    # ------------------------------------------------

    return {
        "found": True,
        "asset": None,
        "match_type": "vector_candidates",
        "requires_selection": True,
        "candidates": candidates
    }


if __name__ == "__main__":

    tests = [
        "wti",
        "american crude",
        "yellow precious metal",
        "digital currency",
        "chip maker",
        "banana",
        "gold"
    ]

    for test in tests:

        print()
        print("QUERY:", test)

        result = search_asset(test)

        print(result)