from sentence_transformers import SentenceTransformer
from elasticsearch import Elasticsearch
import os
from dotenv import load_dotenv

load_dotenv()

es = Elasticsearch(
    cloud_id=os.getenv("ES_CLOUD_ID"),
    api_key=os.getenv("ES_API_KEY")
)

INDEX_NAME = "timeseries_metadata_v2"

model = SentenceTransformer("all-MiniLM-L6-v2")

FIELDS = [
    "record_id", "sensor_name", "location", "industry", "status",
    "temperature", "pressure", "humidity", "wind_speed",
    "is_critical", "timestamp", "notes"
]

def filter_fields(results):
    return [
        {k: hit["_source"].get(k) for k in FIELDS}
        for hit in results["hits"]["hits"]
    ]

def keyword_search(field, value):
    res = es.search(index=INDEX_NAME, query={"term": {field: {"value": value}}})
    return filter_fields(res)

def full_text_search(field, text):
    res = es.search(index=INDEX_NAME, query={"match": {field: text}})
    return filter_fields(res)

def fuzzy_search(field, text):
    res = es.search(index=INDEX_NAME, query={"match": {field: {"query": text, "fuzziness": "AUTO"}}})
    return filter_fields(res)

def semantic_search(query_text, top_k=5):
    vector = model.encode(query_text).tolist()
    res = es.search(
        index=INDEX_NAME,
        knn={"field": "embedding", "query_vector": vector, "k": top_k, "num_candidates": 100}
    )
    return filter_fields(res)

def hybrid_search(query_text, top_k=5):
    vector = model.encode(query_text).tolist()
    res = es.search(
        index=INDEX_NAME,
        knn={"field": "embedding", "query_vector": vector, "k": top_k, "num_candidates": 100},
        query={"bool": {"should": [{"match": {"combined_text": query_text}}]}}
    )
    return filter_fields(res)
