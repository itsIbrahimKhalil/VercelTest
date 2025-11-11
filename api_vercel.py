# Simplified API for Vercel deployment
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pinecone
import cohere
import os
from dotenv import load_dotenv

load_dotenv()

# === Initialize clients ===
co = cohere.Client(os.getenv("COHERE_API_KEY"))
pc = pinecone.Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("topnotch")

# === FastAPI app ===
app = FastAPI(title="FAQ Search API")

class SearchQuery(BaseModel):
    query: str

@app.get("/")
async def root():
    return {"message": "FAQ Search API Running", "endpoints": ["/search"]}

@app.post("/search")
async def search_faq(query: SearchQuery):
    """Search uploaded policy PDFs for answers to questions."""
    if not query.query:
        raise HTTPException(status_code=400, detail="Missing 'query' parameter")

    try:
        # Embed with Cohere
        response = co.embed(
            texts=[query.query],
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
        formatted = []
        for match in results.matches:
            formatted.append({
                "score": float(match.score),
                "source": match.metadata.get('source', 'Unknown'),
                "preview": match.metadata.get('content_preview', '')[:300]
            })
        
        return {
            "success": True,
            "results": formatted if formatted else [],
            "message": "No results found." if not formatted else f"Found {len(formatted)} results."
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

# Export for Vercel
# app is already defined above
