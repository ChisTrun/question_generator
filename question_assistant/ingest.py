import os
import pandas as pd
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer
from elasticsearch import Elasticsearch


MODEL_NAME = 'multi-qa-MiniLM-L6-cos-v1'
ELASTIC_URL = os.getenv("ELASTIC_URL", "http://elasticsearch:9200")
ELASTIC_URL_LOCAL = os.getenv("ELASTIC_URL", "http://localhost:9200")
DATA_PATH = os.getenv("DATA_PATH", "./data/data.csv")
INDEX_NAME = "question-list"

def fetch_documents():
    print("Fetching documents...")
    df = pd.read_csv(DATA_PATH)
    documents = df.to_dict(orient="records")
    print(f"Fetched {len(documents)} documents")
    return documents

def load_model():
    print(f"Loading model: {MODEL_NAME}")
    return SentenceTransformer(MODEL_NAME)


def setup_elasticsearch():
    print("Setting up Elasticsearch...")
    es_client = Elasticsearch(ELASTIC_URL_LOCAL)

    index_settings = {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0
        },
        "mappings": {
            "properties": {
                "question": {"type": "text"},
                "category": {"type": "text"},
                "type": {"type": "text"},
                "job_position": {"type": "text"},
                "level": {"type": "text"},
                "description": {"type": "text"},
                "question_vector": {"type": "dense_vector", "dims": 384, "index": True, "similarity": "cosine"},
                "category_vector": {"type": "dense_vector", "dims": 384, "index": True, "similarity": "cosine"},
                "type_vector": {"type": "dense_vector", "dims": 384, "index": True, "similarity": "cosine"},
                "job_position_vector": {"type": "dense_vector", "dims": 384, "index": True, "similarity": "cosine"},
                "level_vector": {"type": "dense_vector", "dims": 384, "index": True, "similarity": "cosine"},
                "description_vector": {"type": "dense_vector", "dims": 384, "index": True, "similarity": "cosine"},
            }
        }
    }

    es_client.indices.delete(index=INDEX_NAME, ignore_unavailable=True)
    es_client.indices.create(index=INDEX_NAME, body=index_settings)
    print(f"Elasticsearch index '{INDEX_NAME}' created")
    return es_client


def index():
    index_elastic = Elasticsearch(ELASTIC_URL)
    return index_elastic


def index_documents(es_client, documents, model):
    print("Indexing documents...")
    #question,category,type,job_position,level,description
    for doc in tqdm(documents):
        question = doc["question"]
        category = doc["category"]
        types = doc["type"]
        job_position = doc["job_position"]
        level = doc["level"]
        description = doc["description"]
        doc['question_vector'] = model.encode(question)
        doc['category_vector'] = model.encode(category)
        doc['type_vector'] = model.encode(types)
        doc['job_position_vector'] = model.encode(job_position)
        doc['level_vector'] = model.encode(level)
        doc['description_vector'] = model.encode(description)
        doc["text_vector"] = model.encode(question + " " + category + " " + types \
                                          + " " + job_position + " " + level + " " + description  \
                                         ).tolist()
        es_client.index(index=INDEX_NAME, document=doc)
    print(f"Indexed {len(documents)} documents")


def ingest_data():
    # you may consider to comment <start>
    # if you just want to init the db or didn't want to re-index
    print("Starting the indexing process...")

    documents = fetch_documents()
    model = load_model()
    es_client = setup_elasticsearch()
    index_documents(es_client, documents, model)
    
    print("Indexing completed")
