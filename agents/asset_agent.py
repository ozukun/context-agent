import asyncio
import json
import sys
from pathlib import Path

from openai import OpenAI
from fastmcp import Client
from fastmcp.client.transports import StdioTransport
from dotenv import load_dotenv

from pydantic import BaseModel

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class AssetAgent:

    def __init__(self):

        transport = StdioTransport(
            command=sys.executable,
            args=["-m", "mcp_server.server"],
            cwd=str(PROJECT_ROOT)
        )

        self.mcp_client = Client(transport)


    def convert_tools(self, mcp_tools):

        result = []

        for tool in mcp_tools:

            result.append({
                "type": "function",
                "name": tool.name,
                "description": tool.description or "",
                "parameters": tool.input_schema
            })

        return result




    


async def main():

    agent = AssetAgent()

    # -----------------------------------------
    # 1. CONNECT TO MCP SERVER
    # -----------------------------------------

    async with agent.mcp_client:

        # -------------------------------------
        # 2. DISCOVER MCP TOOLS
        # -------------------------------------

        mcp_tools = await agent.mcp_client.list_tools()

        openai_tools = agent.convert_tools(mcp_tools)

        # -------------------------------------
        # 3. CREATE OPENAI CLIENT
        # -------------------------------------

        openai_client = OpenAI()

        # -------------------------------------
        # 4. SEND USER INPUT TO THE LLM
        # -------------------------------------

        response = openai_client.responses.create(
            model="gpt-5.6",
            instructions="""
            You are a financial context agent.

            Use only the tool result when resolving assets.

            If the tool result contains requires_selection=true:
            - Do not choose a candidate.
            - Do not add company descriptions.
            - Do not add candidates that are not present in the tool result.
            - Ask the user to choose one of the returned candidates.

            If found=false, explain that the asset could not be resolved.
            """,
            input="XAU prices", #"Tell me about chip maker", 'XAU prices'
            tools=openai_tools
        )

        # -------------------------------------
        # 5. CHECK WHETHER THE LLM REQUESTED A TOOL CALL
        # -------------------------------------

        tool_outputs = []

        for item in response.output:

            if item.type == "function_call":

                print("TOOL:", item.name)
                print("ARGS:", item.arguments)

                # Convert JSON string to Python dictionary
                arguments = json.loads(
                    item.arguments
                )

                # ---------------------------------
                # 6. EXECUTE THE SELECTED MCP TOOL
                # ---------------------------------

                result = await agent.mcp_client.call_tool(
                    item.name,
                    arguments
                )

                print("MCP RESULT:", result.data)

                # ---------------------------------
                # 7. PREPARE TOOL RESULT FOR THE LLM
                # ---------------------------------

                tool_outputs.append({
                    "type": "function_call_output",
                    "call_id": item.call_id,
                    "output": json.dumps(
                        result.data,
                        default=str
                    )
                })

        # -------------------------------------
        # IF NO TOOL WAS CALLED, RETURN THE LLM RESPONSE
        # -------------------------------------

        if not tool_outputs:

            print("LLM ANSWER:")
            print(response.output_text)

            return



        class AssetAgentState(BaseModel):
            result_count: int
            selected_asset: str | list[dict]  |None= None

        # LANGGRAPH involved

        from langgraph.graph import StateGraph, END


        if "candidates" not in result.data:
            state1 = AssetAgentState(result_count=len([(result.data.get("asset", []))]) , selected_asset=result.data.get("asset", None))
        else:
            state1 = AssetAgentState(result_count=len(result.data.get("candidates", [])) , selected_asset=result.data.get("candidates", None))

        print("Initial state:", result.data) 
        result_count_x = state1.result_count   

        g1 = StateGraph(state_schema=AssetAgentState)

        def step1(state_x: AssetAgentState):

            print("step1 is running")

            return {}

        def step2(state_x: AssetAgentState):

            print("Select an asset:")

            candidates = result.data.get("candidates", [])

            for candidate in candidates:
                print(candidate["asset"])

            selected_asset = input(
                "Enter asset: "
            )

            print("Selected asset:", selected_asset)

            return {
                "result_count": 1,
                "selected_asset": selected_asset
            }

        def step3(state_x: AssetAgentState):

            print("step3 is running")

            return {
                "result_count": result_count_x
            }

        def decide_next_step(state):

            if state.result_count > 1:
                return "select_asset"

            return "continue"

        g1.add_node("step1", step1)
        g1.add_node("select_asset", step2)
        g1.add_node("continue", step3)
   
        
        g1.set_entry_point("step1")
        g1.add_edge("select_asset", END)
        g1.add_edge("continue", END)

        g1.add_conditional_edges\
        (
            "step1",
            decide_next_step,
            {
                "select_asset": "select_asset",
                "continue": "continue"
            }
        )

        app=g1.compile()
        graph_result = app.invoke(state1)

        print(graph_result)

        
        # -------------------------------------
        # 8. SEND MCP TOOL RESULTS BACK TO THE LLM
        # -------------------------------------


        """
        final_response = openai_client.responses.create(
            model="gpt-5.6",
            previous_response_id=response.id,
            input=tool_outputs,
            tools=openai_tools
        )

        # -------------------------------------
        # 9. PRINT FINAL RESPONSE
        # -------------------------------------

        print("FINAL ANSWER:")
        print(final_response.output_text)
        """


if __name__ == "__main__":
    asyncio.run(main())