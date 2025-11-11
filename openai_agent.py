"""
OpenAI Agent with FAQ search capability
"""
from openai import OpenAI
from pinecone import Pinecone
import cohere
import os
import json
from dotenv import load_dotenv

load_dotenv()

# Initialize clients
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
co = cohere.Client(os.getenv("COHERE_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("topnotch")

# Define the FAQ search function
def search_faq(query: str, top_k: int = 3) -> str:
    """
    Search uploaded policy PDFs for answers to questions.
    
    Args:
        query: Question or text to search for in company policies
        top_k: Number of results to return
    
    Returns:
        JSON string with search results
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
                "score": round(match.score, 4),
                "source": match.metadata.get('source', 'Unknown'),
                "content": match.metadata.get('content_preview', '')
            })
        
        return json.dumps(formatted_results, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e)})

# Define the tool for OpenAI
tools = [
    {
        "type": "function",
        "function": {
            "name": "search_faq",
            "description": "Search company policy PDFs for answers to questions about refunds, warranties, delivery, returns, and other policies.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The question or topic to search for in the policy documents"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results to return (default: 3)",
                        "default": 3
                    }
                },
                "required": ["query"]
            }
        }
    }
]

def run_agent(user_message: str):
    """
    Run the OpenAI agent with FAQ search capability
    """
    messages = [
        {
            "role": "system",
            "content": "You are a helpful customer service assistant. You have access to company policy documents and can search them to answer customer questions accurately. Always use the search_faq function to find relevant information before answering policy-related questions."
        },
        {
            "role": "user",
            "content": user_message
        }
    ]
    
    print(f"\n{'='*60}")
    print(f"User: {user_message}")
    print(f"{'='*60}\n")
    
    # First API call
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        tools=tools,
        tool_choice="auto"
    )
    
    response_message = response.choices[0].message
    messages.append(response_message)
    
    # Handle tool calls
    if response_message.tool_calls:
        for tool_call in response_message.tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
            
            print(f"🔍 Searching policies for: {function_args.get('query')}")
            
            if function_name == "search_faq":
                function_response = search_faq(
                    query=function_args.get("query"),
                    top_k=function_args.get("top_k", 3)
                )
                
                print(f"📄 Found results from policy documents\n")
                
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": function_name,
                    "content": function_response
                })
        
        # Second API call with function results
        final_response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=messages
        )
        
        assistant_message = final_response.choices[0].message.content
    else:
        assistant_message = response_message.content
    
    print(f"🤖 Assistant: {assistant_message}\n")
    return assistant_message

if __name__ == "__main__":
    # Example usage
    print("OpenAI Agent with FAQ Search")
    print("=" * 60)
    
    # Test queries
    queries = [
        "What is the refund policy?",
        "Tell me about delivery options",
        "What warranties do you offer?"
    ]
    
    for query in queries:
        run_agent(query)
        print("\n")
