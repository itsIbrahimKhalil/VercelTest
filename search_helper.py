"""
Simple helper to search FAQs without MCP server overhead.
Use this directly in your Python code.
"""
from pinecone import Pinecone
import cohere
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize clients
co = cohere.Client(os.getenv("COHERE_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("topnotch")

def search_faq(query: str, top_k: int = 3) -> list:
    """
    Search uploaded policy PDFs for answers to questions.
    
    Args:
        query: Question or text to search for in company policies
        top_k: Number of results to return (default: 3)
    
    Returns:
        List of search results with scores, sources, and content previews
    """
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
            top_k=top_k,
            include_metadata=True
        )

        # Format results
        formatted_results = []
        for match in results.matches:
            formatted_results.append({
                "score": match.score,
                "source": match.metadata.get('source', 'Unknown'),
                "content_preview": match.metadata.get('content_preview', '')[:300]
            })
        
        return formatted_results

    except Exception as e:
        return [{"error": str(e)}]


if __name__ == "__main__":
    # Example usage
    results = search_faq("What is the refund policy?")
    for i, result in enumerate(results, 1):
        if "error" in result:
            print(f"Error: {result['error']}")
        else:
            print(f"\nResult {i}:")
            print(f"Score: {result['score']:.4f}")
            print(f"Source: {result['source']}")
            print(f"Preview: {result['content_preview']}")
