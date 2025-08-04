import asyncio
from langchain_ollama import OllamaLLM
from langchain_mcp_adapters.client import MultiServerMCPClient

async def main():
    # 1) Ollama LLM (not used for tool call in this example)
    ollama = OllamaLLM(model="llama3")

    # 2) MCP client points to your MCP server
    client = MultiServerMCPClient({
        "database_tool": {
            "transport": "streamable_http",
            "url": "http://localhost:8001/mcp/",  # trailing slash required
        }
    })

    # 3) Discover tools
    tools = await client.get_tools()
    print("Tools loaded:", [t.name for t in tools])  # e.g. ['list_tables_tables_get']

    # 4) Call the MCP tool directly (simulate a user calling the tool)
    #    Adjust to match the actual tool name as printed above!
    if tools:
        tool = tools[0]  # Pick the first (or find by name)
        # You can use the tool with its .ainvoke method
        result = await tool.ainvoke({})
        print("Tool response:", result)
    else:
        print("No MCP tools found!")

if __name__ == "__main__":
    asyncio.run(main())
