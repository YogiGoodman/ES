from faker import Faker
from tqdm import tqdm
from elasticsearch import Elasticsearch, helpers
from sentence_transformers import SentenceTransformer
import random
import os
from dotenv import load_dotenv

load_dotenv()

INDEX_NAME = "timeseries_metadata_v2"

def generate_and_index_data():
    fake = Faker()
    model = SentenceTransformer("all-MiniLM-L6-v2")
    es = Elasticsearch(
        cloud_id=os.getenv("ES_CLOUD_ID"),
        api_key=os.getenv("ES_API_KEY")
    )

    def generate_record(record_id):
        return {
            "record_id": record_id,
            "sensor_name": fake.domain_word(),
            "location": fake.city(),
            "industry": random.choice(["Energy", "Weather"]),
            "status": random.choice(["Active", "Inactive"]),
            "temperature": round(random.uniform(-20, 50), 2),
            "pressure": round(random.uniform(950, 1050), 2),
            "humidity": round(random.uniform(0, 100), 2),
            "wind_speed": round(random.uniform(0, 100), 2),
            "is_critical": random.choice([True, False]),
            "timestamp": fake.date_time_this_decade().isoformat(),
            "notes": fake.text(max_nb_chars=200)
        }

    def get_combined_text(doc):
        fields = [
            "sensor_name", "location", "industry", "status",
            "temperature", "pressure", "humidity", "wind_speed",
            "is_critical", "timestamp", "notes"
        ]
        return " ".join([str(doc[f]) for f in fields])

    # Generate data
    data = [generate_record(i) for i in range(50)]
    for doc in tqdm(data):
        doc["combined_text"] = get_combined_text(doc)
        doc["embedding"] = model.encode(doc["combined_text"]).tolist()

    # Define mapping
    mapping = {
        "mappings": {
            "properties": {
                "record_id": {"type": "long"},
                "sensor_name": {"type": "keyword"},
                "location": {"type": "keyword"},
                "industry": {"type": "keyword"},
                "status": {"type": "keyword"},
                "temperature": {"type": "float"},
                "pressure": {"type": "float"},
                "humidity": {"type": "float"},
                "wind_speed": {"type": "float"},
                "is_critical": {"type": "boolean"},
                "timestamp": {"type": "date"},
                "notes": {"type": "text"},
                "combined_text": {"type": "text"},
                "embedding": {
                    "type": "dense_vector",
                    "dims": 384,
                    "index": True,
                    "similarity": "cosine"
                }
            }
        }
    }

    # Create index
    if es.indices.exists(index=INDEX_NAME):
        es.indices.delete(index=INDEX_NAME)
    es.indices.create(index=INDEX_NAME, body=mapping)

    # Bulk index
    actions = [{"_index": INDEX_NAME, "_id": doc["record_id"], "_source": doc} for doc in data]
    helpers.bulk(es, actions)
