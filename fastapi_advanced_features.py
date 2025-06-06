
from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import List
import re
from sentence_transformers import SentenceTransformer
import openai
from elasticsearch import Elasticsearch

# Initialize
app = FastAPI()
es = Elasticsearch("http://localhost:9200")
model = SentenceTransformer("all-MiniLM-L6-v2")
INDEX_NAME = "timeseries_metadata_demo"

# ----- Semantic Parsing -----
def extract_filters(query: str):
    filters = []
    if "active" in query.lower():
        filters.append({"term": {"status": "Active"}})
    if "weather" in query.lower():
        filters.append({"term": {"industry": "Weather"}})
    if "phoenix" in query.lower():
        filters.append({"term": {"location": "Phoenix"}})
    if "humidity over" in query.lower():
        match = re.search(r"humidity over (\d+)", query.lower())
        if match:
            filters.append({"range": {"humidity": {"gt": int(match.group(1))}}})
    return filters

# ----- Semantic Hybrid Search -----
def semantic_hybrid_search(query_text, top_k=5, num_candidates=1000, min_score=0.7):
    vector = model.encode(query_text).tolist()
    filters = extract_filters(query_text)

    res = es.search(
        index=INDEX_NAME,
        knn={"field": "embedding", "query_vector": vector, "k": 50, "num_candidates": num_candidates},
        query={"bool": {"must": filters, "should": [{"match": {"combined_text": query_text}}]}},
        _source=True
    )
    hits = [hit["_source"] for hit in res["hits"]["hits"] if hit["_score"] >= min_score]
    return hits[:top_k] if hits else [{"warning": "No relevant matches found."}]

# ----- RAG Endpoint -----
@app.get("/rag")
def rag_answer(query: str):
    docs = semantic_hybrid_search(query, top_k=5)
    context = "\n".join([doc["notes"] for doc in docs if "notes" in doc])
    prompt = f"Answer the question based on the logs:\n{context}\n\nQ: {query}\nA:"

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return {"answer": response["choices"][0]["message"]["content"]}

# ----- Rerank Endpoint -----
@app.post("/rerank")
def rerank_results(query: str, docs: List[str]):
    combined_scores = [
        {
            "text": doc,
            "score": model.encode(f"{query} [SEP] {doc}", convert_to_tensor=True).mean().item()
        }
        for doc in docs
    ]
    reranked = sorted(combined_scores, key=lambda x: x["score"], reverse=True)
    return reranked
