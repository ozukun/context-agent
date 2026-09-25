import asyncio
import sys
from pathlib import Path

from fastmcp import Client
from fastmcp.client.transports import StdioTransport


PROJECT_ROOT = Path(__file__).resolve().parent.parent


async def main():

    transport = StdioTransport(
        command=sys.executable,
        args=["-m", "mcp_server.server"],
        cwd=str(PROJECT_ROOT)
    )

    client = Client(transport)

    async with client:

        queries = [
            "gold",
            "american crude",
            "digital currency",
            "chip maker",
            "banana"
        ]

        for query in queries:

            result = await client.call_tool(
                "search_asset",
                {"query": query}
            )

            print()
            print("QUERY:", query)
            print(result.data)


if __name__ == "__main__":
    asyncio.run(main())