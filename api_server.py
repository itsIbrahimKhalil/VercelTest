"""
FastAPI server for FAQ search - to be used with OpenAI Agent Builder
Run locally with: uv run uvicorn api_server:app --reload
Deploy to Vercel: vercel deploy
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pinecone import Pinecone
import cohere
import os

# Initialize clients (Vercel will use environment variables)
co = cohere.Client(os.getenv("COHERE_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("topnotch")

app = FastAPI(
    title="FAQ Search API",
    description="Search company policy documents for answers to customer questions",
    version="1.0.0"
)

# Add CORS middleware to allow OpenAI Agent Builder to access the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SearchQuery(BaseModel):
    query: str
    top_k: int = 3

class SearchResult(BaseModel):
    score: float
    source: str
    content: str

@app.get("/")
async def root():
    return {
        "message": "FAQ Search API",
        "endpoints": {
            "/search": "POST - Search policy documents",
            "/docs": "API documentation"
        }
    }

@app.post("/search", response_model=list[SearchResult])
async def search_faq(search_query: SearchQuery):
    """
    Search company policy PDFs for answers to questions.
    
    - **query**: The question or topic to search for
    - **top_k**: Number of results to return (default: 3)
    """
    try:
        # Embed query with Cohere
        response = co.embed(
            texts=[search_query.query],
            model="embed-english-v3.0",
            input_type="search_query"
        )
        query_embedding = response.embeddings[0]

        # Search Pinecone
        results = index.query(
            vector=query_embedding,
            top_k=search_query.top_k,
            include_metadata=True
        )

        # Format results
        formatted_results = []
        for match in results.matches:
            formatted_results.append(SearchResult(
                score=round(match.score, 4),
                source=match.metadata.get('source', 'Unknown'),
                content=match.metadata.get('content_preview', '')
            ))
        
        return formatted_results

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    print("Starting FAQ Search API server...")
    print("Access the API at: http://localhost:8000")
    print("API docs at: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
