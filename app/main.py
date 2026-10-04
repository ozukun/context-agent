import json
import sys
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI

from fastmcp import Client
from fastmcp.client.transports import StdioTransport


app = FastAPI(
    title="Context Agent API",
    version="0.2.0"
)

openai_client = OpenAI()

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class AssetRequest(BaseModel):
    query: str


class AskRequest(BaseModel):
    question: str


# --------------------------------------------------
# Existing direct asset endpoint
# --------------------------------------------------

from services.asset_search import search_asset


@app.get("/health")
def health():
    return {
        "status": "ok"
    }


@app.post("/asset/search")
def asset_search(request: AssetRequest):
    return search_asset(request.query)


# --------------------------------------------------
# MCP call
# --------------------------------------------------

async def call_mcp_search_asset(query: str) -> dict:

    transport = StdioTransport(
        command=sys.executable,
        args=["-m", "mcp_server.server"],
        cwd=str(PROJECT_ROOT)
    )

    client = Client(transport)

    async with client:

        result = await client.call_tool(
            "search_asset",
            {
                "query": query
            }
        )

        return result.data


# --------------------------------------------------
# LLM Tool Definition
# --------------------------------------------------

SEARCH_ASSET_TOOL = {
    "type": "function",
    "name": "search_asset",
    "description": (
        "Resolve a financial asset, ticker, alias, "
        "or semantic asset description."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "The asset name, ticker, alias, "
                    "or description to resolve."
                )
            }
        },
        "required": ["query"],
        "additionalProperties": False
    },
    "strict": True
}


# --------------------------------------------------
# Main Agent Endpoint
# --------------------------------------------------

@app.post("/ask")
async def ask(request: AskRequest):

    response = openai_client.responses.create(
        model="gpt-5.6",
        instructions="""
You are a financial context agent.

Use the search_asset tool whenever the user refers to
a financial asset, ticker, commodity, cryptocurrency,
company, or semantic asset description.

Never choose between ambiguous asset candidates yourself.

If the tool returns requires_selection=true,
present the candidates to the user and ask them to choose.

If found=false, explain that the asset could not be resolved.
""",
        input=request.question,
        tools=[
            SEARCH_ASSET_TOOL
        ]
    )

    # Find tool call
    for item in response.output:

        if (
            item.type == "function_call"
            and item.name == "search_asset"
        ):

            arguments = json.loads(
                item.arguments
            )

            tool_result = await call_mcp_search_asset(
                arguments["query"]
            )

            # Send MCP result back to LLM
            final_response = openai_client.responses.create(
                model="gpt-5.6",
                previous_response_id=response.id,
                input=[
                    {
                        "type": "function_call_output",
                        "call_id": item.call_id,
                        "output": json.dumps(tool_result)
                    }
                ]
            )

            return {
                "answer": final_response.output_text,
                "tool_result": tool_result
            }

    # LLM decided no tool was necessary
    return {
        "answer": response.output_text,
        "tool_result": None
    }