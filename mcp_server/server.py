from fastmcp import FastMCP

from services.asset_search import search_asset as search_asset_service


mcp = FastMCP("context-agent")


@mcp.tool
def search_asset(query: str) -> dict:
    """
    Resolve a financial asset name, ticker, or alias
    to its canonical asset profile.
    """
    return search_asset_service(query)


if __name__ == "__main__":
    mcp.run()