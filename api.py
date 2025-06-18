import json
from typing import List, Dict, Any
import os
from tqdm import tqdm
import chromadb
import google.generativeai as genai
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Discourse Search API",
    description="API for semantic search and answer generation using Gemini and ChromaDB",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyAWwXNH-7VJDWxQcb9vnT983Dox08QonWI")
genai.configure(api_key=GEMINI_API_KEY)
gemini = genai.GenerativeModel("gemini-2.0-flash")

# Initialize ChromaDB client
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(
    name="discourse_threads",
    metadata={"hnsw:space": "cosine"}
)

# Pydantic models for request/response
class SearchRequest(BaseModel):
    query: str
    top_k: int = 5

class GenerateAnswerRequest(BaseModel):
    query: str
    context_texts: List[str]

class SearchResponse(BaseModel):
    results: List[Dict[str, Any]]

class GenerateAnswerResponse(BaseModel):
    answer: str

class InitializeResponse(BaseModel):
    message: str

def get_gemini_embedding(text: str) -> List[float]:
    """Get embedding vector using Gemini embedding model"""
    try:
        response = genai.embed_content(
            model="models/embedding-001",
            content=text,
            task_type="RETRIEVAL_DOCUMENT"
        )
        return response["embedding"]
    except Exception as e:
        print(f"Error getting embedding: {str(e)}")
        return []

def process_posts(input_path: str) -> Dict[int, Dict[str, Any]]:
    """Load and group posts by topic from a single JSON file or folder of JSON files"""
    topics = {}
    
    try:
        # Check if input is a file or directory
        path = Path(input_path)
        if path.is_file():
            json_files = [path]
        else:
            json_files = list(path.glob("*.json"))
        
        if not json_files:
            raise FileNotFoundError(f"No JSON files found in {input_path}")
        
        print(f"📂 Found {len(json_files)} JSON file(s)")
        
        for json_file in tqdm(json_files, desc="📖 Loading JSON files"):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    
                # Handle both single file and folder structure
                if isinstance(data, list):
                    posts_data = data
                else:
                    posts_data = data.get("post_stream", {}).get("posts", [])
                
                for post in posts_data:
                    topic_id = post.get("topic_id")
                    if not topic_id:
                        continue
                        
                    if topic_id not in topics:
                        topics[topic_id] = {
                            "topic_title": post.get("topic_slug", "").replace("-", " ").title(),
                            "posts": []
                        }
                    topics[topic_id]["posts"].append(post)
                    
            except Exception as e:
                print(f"⚠️ Error loading {json_file}: {str(e)}")
                continue
        
        # Sort posts by post_number
        for topic in topics.values():
            topic["posts"].sort(key=lambda p: p.get("post_number", 0))
        
        return topics
    except Exception as e:
        print(f"Error processing posts: {str(e)}")
        return {}

def build_thread_map(posts: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    """Build reply tree structure"""
    thread_map = {}
    for post in posts:
        parent = post.get("reply_to_post_number")
        if parent not in thread_map:
            thread_map[parent] = []
        thread_map[parent].append(post)
    return thread_map

def extract_thread(root_num: int, posts: List[Dict[str, Any]], thread_map: Dict[int, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Extract full thread starting from root post"""
    thread = []
    
    def collect_replies(post_num):
        post = next(p for p in posts if p["post_number"] == post_num)
        thread.append(post)
        for reply in thread_map.get(post_num, []):
            collect_replies(reply["post_number"])
    
    collect_replies(root_num)
    return thread

def embed_and_index_threads(topics: Dict[int, Dict[str, Any]], batch_size: int = 100):
    """Embed threads using Gemini and index in ChromaDB"""
    try:
        for topic_id, topic_data in tqdm(topics.items(), desc="🔍 Embedding and indexing threads"):
            posts = topic_data["posts"]
            topic_title = topic_data["topic_title"]
            thread_map = build_thread_map(posts)
            
            # Process root posts (those without parents)
            root_posts = thread_map.get(None, [])
            for root_post in root_posts:
                thread = extract_thread(root_post["post_number"], posts, thread_map)
                
                # Combine thread text
                combined_text = f"Topic: {topic_title}\n\n"
                combined_text += "\n\n---\n\n".join(
                    post.get("content", post.get("cooked", "")).strip() for post in thread
                )
                
                # Get embedding from Gemini
                embedding = get_gemini_embedding(combined_text)
                
                # Convert post_numbers list to string for ChromaDB metadata
                post_numbers_str = ",".join(map(str, [p["post_number"] for p in thread]))
                
                collection.add(
                    documents=[combined_text],
                    embeddings=[embedding],
                    ids=[f"{topic_id}_{root_post['post_number']}"],
                    metadatas=[{
                        "topic_id": str(topic_id),
                        "topic_title": topic_title,
                        "root_post_number": str(root_post["post_number"]),
                        "post_numbers": post_numbers_str
                    }]
                )
    except Exception as e:
        print(f"Error during embedding and indexing: {str(e)}")

def semantic_search(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Search for relevant threads using Gemini embeddings"""
    try:
        query_embedding = get_gemini_embedding(query)
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        
        if not results["documents"] or not results["metadatas"] or not results["distances"]:
            return []
        
        hits = []
        for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
            post_numbers = [int(x) for x in meta["post_numbers"].split(",")]
            
            hits.append({
                "score": 1.0 - dist,
                "topic_id": int(meta["topic_id"]),
                "topic_title": meta["topic_title"],
                "root_post_number": int(meta["root_post_number"]),
                "post_numbers": post_numbers,
                "combined_text": doc
            })
        
        return hits
    except Exception as e:
        print(f"Error during semantic search: {str(e)}")
        return []

def generate_answer(query: str, context_texts: List[str]) -> str:
    """Generate answer using Gemini"""
    try:
        context = "\n\n---\n\n".join(context_texts)
        prompt = f"""Based on these forum excerpts:

{context}

Question: {query}

Answer:"""

        response = gemini.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Error generating answer: {str(e)}")
        return "Sorry, I encountered an error while generating the answer."

# FastAPI endpoints
@app.get("/")
async def root():
    return {
        "message": "Discourse Search API is running",
        "version": "1.0.0",
        "status": "healthy"
    }

@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    try:
        results = semantic_search(request.query, request.top_k)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-answer", response_model=GenerateAnswerResponse)
async def generate_answer_endpoint(request: GenerateAnswerRequest):
    try:
        if not request.query or not request.context_texts:
            raise HTTPException(status_code=400, detail="Query and context_texts are required")
        answer = generate_answer(request.query, request.context_texts)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/initialize", response_model=InitializeResponse)
async def initialize():
    try:
        # Check if file exists
        if not os.path.exists("discourse_posts.json"):
            raise HTTPException(
                status_code=404,
                detail="discourse_posts.json file not found. Please ensure the file exists in the root directory."
            )
            
        # Process posts and verify data
        topics = process_posts("discourse_posts.json")
        if not topics:
            raise HTTPException(
                status_code=500,
                detail="No topics were processed from the JSON file. Please check the file format."
            )
            
        # Embed and index threads
        embed_and_index_threads(topics)
        
        return {
            "message": f"Data initialization complete. Processed {len(topics)} topics.",
            "topics_processed": len(topics)
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid JSON format in discourse_posts.json: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)