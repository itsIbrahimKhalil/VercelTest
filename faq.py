from mcp.server import Server
from mcp.types import Tool, TextContent
import mcp.server.stdio
from pinecone import Pinecone
import cohere
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize clients
co = cohere.Client(os.getenv("COHERE_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("topnotch")

# Create MCP server
app = Server("faq-search-server")

@app.list_tools()
async def list_tools():
    return [
        Tool(
            name="search_faq",
            description="Search uploaded policy PDFs (via Cohere embeddings) for answers to questions.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Question or text to search for in company policies."
                    }
                },
                "required": ["query"]
            }
        )
    ]

@app.call_tool()
async def call_tool(name: str, arguments: dict):
    if name != "search_faq":
        raise ValueError(f"Unknown tool: {name}")
    
    query = arguments.get("query", "")
    if not query:
        return [TextContent(type="text", text="Error: Missing 'query' parameter")]

    try:
        # Embed query with Cohere
        response = co.embed(
            texts=[query],
            model="embed-english-v3.0",
            input_type="search_query"
        )
        query_embedding = response.embeddings[0]

        # Search Pinecone
        results = index.query(
            vector=query_embedding,
            top_k=3,
            include_metadata=True
        )

        # Format results
        formatted_results = []
        for match in results.matches:
            formatted_results.append(
                f"Score: {match.score:.4f}\n"
                f"Source: {match.metadata.get('source', 'Unknown')}\n"
                f"Preview: {match.metadata.get('content_preview', '')[:300]}\n"
            )
        
        return [TextContent(
            type="text",
            text="\n---\n".join(formatted_results) if formatted_results else "No results found."
        )]

    except Exception as e:
        return [TextContent(type="text", text=f"Error: {str(e)}")]

async def main():
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
