import asyncio
from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp.client.session import ClientSession

async def main():
    # Connect to the MCP server via stdio
    server_params = StdioServerParameters(
        command="uv",
        args=["run", "faq.py"]
    )
    
    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            # Initialize the session
            await session.initialize()
            
            # List tools
            tools = await session.list_tools()
            print("Available tools:")
            for tool in tools.tools:
                print(f"- {tool.name}: {tool.description}")
            
            # Call the search_faq tool
            result = await session.call_tool("search_faq", {"query": "What is the refund policy?"})
            print("\nSearch results:")
            for content in result.content:
                if hasattr(content, 'text'):
                    print(content.text)

if __name__ == "__main__":
    asyncio.run(main())
