from fastapi import FastAPI, Query
from search_engine import (
    keyword_search, full_text_search,
    fuzzy_search, semantic_search, hybrid_search
)

app = FastAPI(title="Hybrid Semantic Search API")

@app.get("/search/keyword")
def search_keyword(field: str, value: str):
    return keyword_search(field, value)

@app.get("/search/fulltext")
def search_full_text(field: str, text: str):
    return full_text_search(field, text)

@app.get("/search/fuzzy")
def search_fuzzy(field: str, text: str):
    return fuzzy_search(field, text)

@app.get("/search/semantic")
def search_semantic(text: str, top_k: int = Query(5, ge=1)):
    return semantic_search(text, top_k)

@app.get("/search/hybrid")
def search_hybrid(text: str, top_k: int = Query(5, ge=1)):
    return hybrid_search(text, top_k)
