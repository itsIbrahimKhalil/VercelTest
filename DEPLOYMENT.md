# Deploying FAQ Search API to Vercel

## Prerequisites
1. Install Vercel CLI: `npm install -g vercel`
2. Have your Vercel account ready

## Deployment Steps

### 1. Login to Vercel
```powershell
vercel login
```

### 2. Deploy the API
```powershell
vercel deploy
```

### 3. Add Environment Variables
After deploying, go to your Vercel dashboard and add these environment variables:
- `COHERE_API_KEY` - Your Cohere API key
- `PINECONE_API_KEY` - Your Pinecone API key

Or add them via CLI:
```powershell
vercel env add COHERE_API_KEY
vercel env add PINECONE_API_KEY
```

### 4. Deploy to Production
```powershell
vercel --prod
```

## Using with OpenAI Agent Builder

1. Go to https://platform.openai.com/agent-builder/edit
2. Add a new action/tool
3. Use your Vercel URL: `https://your-project.vercel.app/search`
4. Method: POST
5. Add the OpenAPI schema:

```json
{
  "openapi": "3.0.0",
  "info": {
    "title": "FAQ Search API",
    "version": "1.0.0"
  },
  "servers": [
    {
      "url": "https://your-project.vercel.app"
    }
  ],
  "paths": {
    "/search": {
      "post": {
        "summary": "Search company policy documents",
        "operationId": "searchFAQ",
        "requestBody": {
          "required": true,
          "content": {
            "application/json": {
              "schema": {
                "type": "object",
                "properties": {
                  "query": {
                    "type": "string",
                    "description": "The question to search for"
                  },
                  "top_k": {
                    "type": "integer",
                    "default": 3,
                    "description": "Number of results"
                  }
                },
                "required": ["query"]
              }
            }
          }
        },
        "responses": {
          "200": {
            "description": "Search results",
            "content": {
              "application/json": {
                "schema": {
                  "type": "array",
                  "items": {
                    "type": "object",
                    "properties": {
                      "score": {"type": "number"},
                      "source": {"type": "string"},
                      "content": {"type": "string"}
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}
```

## Testing Locally

Start the server:
```powershell
$env:Path = "C:\Users\syedm\.local\bin;$env:Path"
uv run uvicorn api_server:app --reload
```

Test the endpoint:
```powershell
curl -X POST http://localhost:8000/search -H "Content-Type: application/json" -d '{"query": "refund policy"}'
```

## Alternative: Use ngrok for testing

If you want to test with OpenAI Agent Builder before deploying:

1. Install ngrok: https://ngrok.com/download
2. Start your local server
3. In another terminal: `ngrok http 8000`
4. Use the ngrok URL in OpenAI Agent Builder
