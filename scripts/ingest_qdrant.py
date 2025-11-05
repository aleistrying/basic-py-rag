import uuid
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from utils import read_texts, chunk_text, embed_e5
import sys

COLLECTION = "docs_qdrant"
DIM = 768  # multilingual-e5-base has 768 dimensions

client = QdrantClient(host="localhost", port=6333)

# Reset database: Always recreate collection to start fresh
print(f"🔄 Resetting Qdrant collection: {COLLECTION}")
try:
    client.delete_collection(collection_name=COLLECTION)
    print(f"✅ Deleted existing collection: {COLLECTION}")
except Exception as e:
    print(f"ℹ️  Collection {COLLECTION} doesn't exist (OK for first run)")

# Create fresh collection with cosine distance
client.create_collection(
    collection_name=COLLECTION,
    vectors_config=VectorParams(size=DIM, distance=Distance.COSINE)
)
print(f"✅ Created fresh collection: {COLLECTION} (768 dims, cosine)")

# Read dataset from command line or default
folder = sys.argv[1] if len(sys.argv) > 1 else "./data/raw"
items = read_texts(folder)

points = []
for item in items:
    doc_id = str(uuid.uuid4())
    chunks = chunk_text(item["text"]) or [item["text"]]
    # Use E5 embeddings with passage prefix
    embs = embed_e5(chunks, is_query=False)
    for idx, (chunk, emb) in enumerate(zip(chunks, embs)):
        payload = {
            "doc_id": doc_id,
            "chunk_id": f"{idx}",
            "text": chunk,  # Use 'text' field (not 'content')
            "path": item["path"],
        }
        # Add schedule if present
        if "schedule" in item:
            payload["schedule"] = item["schedule"]

        points.append(PointStruct(
            id=str(uuid.uuid4()),
            vector=emb,
            payload=payload,
        ))

# Upsert in batches
BATCH = 128
for i in range(0, len(points), BATCH):
    client.upsert(collection_name=COLLECTION, points=points[i:i+BATCH])

print(f"✅ Ingested {len(points)} chunks to {COLLECTION}")
print(f"✅ Using multilingual-e5-base (768 dimensions) with cosine similarity")
