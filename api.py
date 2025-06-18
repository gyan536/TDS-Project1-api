import json
from typing import List, Dict, Any
import os
from tqdm import tqdm
import chromadb
import google.generativeai as genai
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Initialize FastAPI app
app = FastAPI(title="Discourse Search API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure Gemini API
genai.configure(api_key="AIzaSyAWwXNH-7VJDWxQcb9vnT983Dox08QonWI")
gemini = genai.GenerativeModel("gemini-2.0-flash")

def get_gemini_embedding(text: str) -> List[float]:
    """Get embedding vector using Gemini embedding model"""
    response = genai.embed_content(
        model="models/embedding-001",
        content=text,
        task_type="RETRIEVAL_DOCUMENT"
    )
    return response["embedding"]

# Initialize ChromaDB client
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="discourse_threads")

def process_posts(input_path: str) -> Dict[int, Dict[str, Any]]:
    """Load and group posts by topic from a single JSON file or folder of JSON files"""
    topics = {}
    
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
                # Try 'content' first, fall back to 'cooked' if not found
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
                    "topic_id": str(topic_id),  # Convert to string to ensure compatibility
                    "topic_title": topic_title,
                    "root_post_number": str(root_post["post_number"]),  # Convert to string
                    "post_numbers": post_numbers_str  # Store as comma-separated string
                }]
            )

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
            print("⚠️ No results found for the query.")
            return []
        
        hits = []
        for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
            # Convert post_numbers string back to list
            post_numbers = [int(x) for x in meta["post_numbers"].split(",")]
            
            hits.append({
                "score": 1.0 - dist,  # similarity = 1 - distance
                "topic_id": int(meta["topic_id"]),  # Convert back to int
                "topic_title": meta["topic_title"],
                "root_post_number": int(meta["root_post_number"]),  # Convert back to int
                "post_numbers": post_numbers,
                "combined_text": doc
            })
        
        return hits
    except Exception as e:
        print(f"❌ Error during semantic search: {str(e)}")
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
        print(f"❌ Error generating answer: {str(e)}")
        return "Sorry, I encountered an error while generating the answer."

def test_queries(queries: List[str], top_k: int = 3):
    """Test multiple queries and display results"""
    for query in queries:
        print(f"\n🔍 Testing query: {query}")
        results = semantic_search(query, top_k=top_k)
        
        if results:
            print("\nTop search results:")
            for i, res in enumerate(results, 1):
                print(f"\n[{i}] Score: {res['score']:.4f}")
                print(f"Topic: {res['topic_title']}")
                print(f"Content snippet: {res['combined_text'][:500]}...\n")
            
            # Generate answer
            context_texts = [res["combined_text"] for res in results]
            answer = generate_answer(query, context_texts)
            print("\nGenerated Answer:\n", answer)
        else:
            print("No relevant results found.")

# API endpoints
@app.get("/")
async def root():
    return {"message": "Discourse Search API is running"}

@app.post("/search")
async def search(query: str, top_k: int = 5):
    try:
        results = semantic_search(query, top_k=top_k)
        if not results:
            raise HTTPException(status_code=404, detail="No results found")
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-answer")
async def generate_answer_endpoint(query: str, context_texts: List[str]):
    try:
        answer = generate_answer(query, context_texts)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Example usage
if __name__ == "__main__":
    try:
        # Load and process data
        print("🔄 Loading topics from discourse_posts.json...")
        topics = process_posts("discourse_posts.json")
        print(f"✅ Successfully loaded {len(topics)} topics")
        
        if not topics:
            print("❌ No topics found in the file. Please check if the file is empty or has the correct format.")
            exit(1)
            
        # Index data
        print("\n🔄 Starting to embed and index threads...")
        embed_and_index_threads(topics)
        print("✅ Indexing complete")
        
        # Start the FastAPI server
        print("\n🚀 Starting API server...")
        port = int(os.getenv("PORT", 8000))
        uvicorn.run(app, host="0.0.0.0", port=port)
            
    except FileNotFoundError:
        print("❌ Error: discourse_posts.json file not found!")
        print("Please make sure the file exists in the current directory.")
    except json.JSONDecodeError:
        print("❌ Error: Invalid JSON format in discourse_posts.json")
        print("Please check if the file contains valid JSON data.")
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print("An unexpected error occurred. Please check the error message above.")