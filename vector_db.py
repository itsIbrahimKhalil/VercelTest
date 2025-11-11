import fitz  # PyMuPDF
import glob
import os
from typing import List, Dict
import tiktoken
import cohere
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Cohere client
co = cohere.Client(os.getenv("COHERE_API_KEY"))

def chunk_text_by_tokens(text: str, max_tokens: int = 8000, overlap: int = 200) -> List[str]:
    """
    Split text into chunks by tokens with overlap for context preservation.
    """
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    chunks = []
    
    start = 0
    while start < len(tokens):
        end = start + max_tokens
        chunk_tokens = tokens[start:end]
        chunk_text = encoding.decode(chunk_tokens)
        chunks.append(chunk_text)
        start += max_tokens - overlap
    
    return chunks

def embed_text(text: str) -> List[float]:
    """
    Generate embeddings using Cohere.
    """
    try:
        response = co.embed(
            texts=[text],
            model="embed-english-v3.0",
            input_type="search_document"  # Use for indexing documents
        )
        return response.embeddings[0]
    except Exception as e:
        print(f"Error generating embedding: {e}")
        raise

def embed_query(text: str) -> List[float]:
    """
    Generate embeddings for search queries (different input_type).
    """
    try:
        response = co.embed(
            texts=[text],
            model="embed-english-v3.0",
            input_type="search_query"  # Use for search queries
        )
        return response.embeddings[0]
    except Exception as e:
        print(f"Error generating query embedding: {e}")
        raise

def extract_pdf_text(file_path: str) -> str:
    """Extract text from PDF using PyMuPDF."""
    try:
        doc = fitz.open(file_path)
        text = ""
        
        for page in doc:
            # Extract text with better formatting
            page_text = page.get_text("text")
            if page_text:
                text += page_text + "\n"
        
        doc.close()
        return text.strip()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return ""

def process_pdfs_to_vectors(pdf_pattern: str, max_chunk_tokens: int = 6000) -> List[Dict]:
    """
    Process PDFs and create vectors with metadata.
    
    Args:
        pdf_pattern: Glob pattern for PDF files (e.g., "policies/*.pdf")
        max_chunk_tokens: Maximum tokens per chunk
    
    Returns:
        List of vector dictionaries ready for Pinecone upsert
    """
    pdf_files = glob.glob(pdf_pattern)
    if not pdf_files:
        print(f"No PDFs found matching pattern: {pdf_pattern}")
        return []
    
    print(f"Found {len(pdf_files)} PDF files")
    policy_vectors = []
    
    for file_path in pdf_files:
        filename = os.path.basename(file_path)
        print(f"Processing: {filename}")
        
        # Extract text
        text = extract_pdf_text(file_path)
        if not text:
            print(f"Skipping {filename} - no text extracted")
            continue
        
        # Chunk text by tokens
        chunks = chunk_text_by_tokens(text, max_tokens=max_chunk_tokens)
        print(f"  Created {len(chunks)} chunks")
        
        # Create vectors for each chunk
        for idx, chunk in enumerate(chunks):
            try:
                embedding = embed_text(chunk)
                
                policy_vectors.append({
                    "id": f"{os.path.splitext(filename)[0]}-chunk-{idx}",
                    "values": embedding,
                    "metadata": {
                        "source": filename,
                        "chunk_index": idx,
                        "total_chunks": len(chunks),
                        "content_preview": chunk[:300],
                        "type": "policy",
                        "char_count": len(chunk)
                    }
                })
            except Exception as e:
                print(f"  Error embedding chunk {idx} of {filename}: {e}")
                continue
    
    return policy_vectors

def upsert_to_pinecone(index, vectors: List[Dict], batch_size: int = 100):
    """
    Upsert vectors to Pinecone in batches.
    
    Args:
        index: Pinecone index object
        vectors: List of vector dictionaries
        batch_size: Number of vectors per batch
    """
    if not vectors:
        print("No vectors to upsert")
        return
    
    print(f"Upserting {len(vectors)} vectors in batches of {batch_size}")
    
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i+batch_size]
        try:
            index.upsert(vectors=batch)
            print(f"  Upserted batch {i//batch_size + 1}/{(len(vectors)-1)//batch_size + 1}")
        except Exception as e:
            print(f"  Error upserting batch {i//batch_size + 1}: {e}")
    
    print(f"✓ Successfully inserted {len(vectors)} policy chunks")

def search_policies(index, query: str, top_k: int = 5) -> List[Dict]:
    """
    Search for relevant policy chunks.
    
    Args:
        index: Pinecone index object
        query: Search query text
        top_k: Number of results to return
    
    Returns:
        List of matching results with metadata
    """
    try:
        # Embed the query using search_query input type
        query_embedding = embed_query(query)
        
        # Search Pinecone
        results = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        
        return results.matches
    except Exception as e:
        print(f"Error searching: {e}")
        return []

# Usage Example
if __name__ == "__main__":
    from pinecone import Pinecone
    
    # Initialize Pinecone
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index("topnotch")
    
    # Process PDFs and upload
    print("=== Processing PDFs ===")
    policy_vectors = process_pdfs_to_vectors("policies/*.pdf")
    
    if policy_vectors:
        print("\n=== Uploading to Pinecone ===")
        upsert_to_pinecone(index, policy_vectors)
        
        # Test search
        print("\n=== Testing Search ===")
        test_query = "What is the refund policy?"
        results = search_policies(index, test_query, top_k=3)
        
        print(f"\nTop results for: '{test_query}'")
        for i, match in enumerate(results, 1):
            print(f"\n{i}. Score: {match.score:.4f}")
            print(f"   Source: {match.metadata['source']}")
            print(f"   Preview: {match.metadata['content_preview'][:150]}...")
    else:
        print("No vectors created. Check your PDF files.")