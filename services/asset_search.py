import json
from pathlib import Path


ASSET_PROFILE_PATH = (
    Path(__file__).resolve().parent.parent
    / "context"
    / "asset_profiles.json"
)


def load_asset_profiles() -> dict:
    with open(ASSET_PROFILE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def search_asset(query: str) -> dict:
    profiles = load_asset_profiles()

    normalized_query = query.strip().lower()

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
                    "analysis_dimensions", []
                )
            }

    return {
        "found": False,
        "asset": None
    }


if __name__ == "__main__":
    print(search_asset("gold"))
    print(search_asset("xau"))
    print(search_asset("wti"))
    print(search_asset("btc"))
    print(search_asset("nvda"))