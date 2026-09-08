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

        tools = await client.list_tools()

        for tool in tools:
            print(tool.name)
            print(tool.description)
            print(tool.input_schema)

        result = await client.call_tool(
            "search_asset",
            {"query": "wti"}
        )

        print("RESULT:")
        print(result.data)


if __name__ == "__main__":
    asyncio.run(main())