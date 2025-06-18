import json
from typing import List, Dict, Any
import os
from tqdm import tqdm
import google.generativeai as genai
from pinecone import Pinecone, ServerlessSpec

# === Replace with your keys (or use environment variables) ===
GEMINI_API_KEY = "AIzaSyAWwXNH-7VJDWxQcb9vnT983Dox08QonWI"
PINECONE_API_KEY = "pcsk_2nKSrk_8mTJdLo5exiHrNcZPQKnDtx8t3BJLWNfvEGQQN9HkkdRq9PfB1yTAtR4hVCgQKH"

# === Initialize Gemini + Pinecone ===
genai.configure(api_key=GEMINI_API_KEY)
pinecone = Pinecone(api_key=PINECONE_API_KEY)

# === Pinecone Setup ===
index_name = "discourse-embeddings"
if index_name not in pinecone.list_indexes().names():
    pinecone.create_index(
        name=index_name,
        dimension=768,  # Gemini embedding dimension
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-west-2")
    )
index = pinecone.Index(index_name)

# === Rest of the logic ===
def process_posts(filename: str) -> Dict[int, Dict[str, Any]]:
    with open(filename, "r", encoding="utf-8") as f:
        posts_data = json.load(f)
    topics = {}
    for post in posts_data:
        topic_id = post["topic_id"]
        if topic_id not in topics:
            topics[topic_id] = {"topic_title": post.get("topic_title", ""), "posts": []}
        topics[topic_id]["posts"].append(post)
    for topic in topics.values():
        topic["posts"].sort(key=lambda p: p["post_number"])
    return topics

def build_thread_map(posts: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    thread_map = {}
    for post in posts:
        parent = post.get("reply_to_post_number")
        if parent not in thread_map:
            thread_map[parent] = []
        thread_map[parent].append(post)
    return thread_map

def extract_thread(root_num: int, posts: List[Dict[str, Any]], thread_map: Dict[int, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    thread = []
    def collect_replies(post_num):
        post = next(p for p in posts if p["post_number"] == post_num)
        thread.append(post)
        for reply in thread_map.get(post_num, []):
            collect_replies(reply["post_number"])
    collect_replies(root_num)
    return thread

def embed_text(text: str) -> List[float]:
    embed_model = genai.GenerativeModel("embedding-001")
    response = embed_model.embed_content(content=text, task_type="retrieval_document")
    return response["embedding"]

def embed_and_index_threads(topics: Dict[int, Dict[str, Any]], batch_size: int = 100):
    vectors = []
    for topic_id, topic_data in tqdm(topics.items()):
        posts = topic_data["posts"]
        topic_title = topic_data["topic_title"]
        thread_map = build_thread_map(posts)
        root_posts = thread_map.get(None, [])
        for root_post in root_posts:
            thread = extract_thread(root_post["post_number"], posts, thread_map)
            combined_text = f"Topic: {topic_title}\n\n" + "\n\n---\n\n".join(post["content"].strip() for post in thread)
            embedding = embed_text(combined_text)
            vector = {
                "id": f"{topic_id}_{root_post['post_number']}",
                "values": embedding,
                "metadata": {
                    "topic_id": topic_id,
                    "topic_title": topic_title,
                    "root_post_number": root_post["post_number"],
                    "post_numbers": [p["post_number"] for p in thread],
                    "combined_text": combined_text
                }
            }
            vectors.append(vector)
            if len(vectors) >= batch_size:
                index.upsert(vectors=vectors)
                vectors = []
    if vectors:
        index.upsert(vectors=vectors)

def semantic_search(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    query_embedding = embed_text(query)
    search_response = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
    return [{
        "score": match.score,
        "topic_id": match.metadata["topic_id"],
        "topic_title": match.metadata["topic_title"],
        "root_post_number": match.metadata["root_post_number"],
        "post_numbers": match.metadata["post_numbers"],
        "combined_text": match.metadata["combined_text"]
    } for match in search_response.matches]

def generate_answer(query: str, context_texts: List[str]) -> str:
    context = "\n\n---\n\n".join(context_texts)
    prompt = f"""
You are a helpful assistant that answers questions based on forum discussions.

Forum context:
{context}

Question: {query}
Answer:"""
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    return response.text.strip()

# === Main Execution ===
if __name__ == "__main__":
    topics = process_posts("discourse_posts.json")
    print(f"Loaded {len(topics)} topics")

    embed_and_index_threads(topics)  # Do once
    print("Indexing complete")

    query = "If a student scores 10/10 on GA4 as well as a bonus, how would it appear on the dashboard?"
    results = semantic_search(query, top_k=3)

    print("\nTop search results:")
    for i, res in enumerate(results, 1):
        print(f"\n[{i}] Score: {res['score']:.4f}")
        print(f"Topic: {res['topic_title']}")
        print(f"Content snippet: {res['combined_text'][:500]}...\n")

    context_texts = [res["combined_text"] for res in results]
    answer = generate_answer(query, context_texts)
    print("\nGenerated Answer:\n", answer)
