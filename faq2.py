# mcp_server.py
from mcp.server import Server
from mcp.types import Tool, TextContent
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn
import pinecone
import cohere
import os
from dotenv import load_dotenv

load_dotenv()

# === Initialize clients ===
co = cohere.Client(os.getenv("COHERE_API_KEY"))
pc = pinecone.Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("topnotch")

# === Create MCP Server ===
app = Server("faq-search-server")

# FastAPI app to wrap MCP
fastapi_app = FastAPI()

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
        # Embed with Cohere
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

        # Format
        formatted = []
        for match in results.matches:
            formatted.append(
                f"Score: {match.score:.4f}\n"
                f"Source: {match.metadata.get('source', 'Unknown')}\n"
                f"Preview: {match.metadata.get('content_preview', '')[:300]}\n"
            )
        
        return [TextContent(
            type="text",
            text="\n---\n".join(formatted) if formatted else "No results found."
        )]

    except Exception as e:
        return [TextContent(type="text", text=f"Error: {str(e)}")]

# === Mount MCP on /mcp ===
fastapi_app.mount("/mcp", app.create_app())

# === Vercel Entry Point ===
@fastapi_app.get("/")
async def root():
    return {"message": "MCP Server Running. Use /mcp"}

# === Run with Uvicorn (for local testing) ===
if __name__ == "__main__":
    uvicorn.run(fastapi_app, host="0.0.0.0", port=8000)
# Export for Vercel
app = fastapi_app
