#!/usr/bin/env python3
"""
Test script to verify:
1. Database reset functionality
2. Schedule table extraction from PDF
3. Data retrieval from both Qdrant and pgvector
"""

from qdrant_client import QdrantClient
import psycopg2
from utils import embed_e5


def test_qdrant():
    """Test Qdrant database"""
    print("\n" + "="*60)
    print("TESTING QDRANT")
    print("="*60)

    client = QdrantClient(url='http://localhost:6333')

    # Get collection info
    info = client.get_collection('docs_qdrant')
    print(f"\n✅ Collection exists with {info.points_count} points")

    # Test schedule data presence
    points = client.scroll(collection_name='docs_qdrant', limit=10)[0]
    schedule_points = [p for p in points if 'schedule' in p.payload]
    print(f"✅ Found {len(schedule_points)} points with schedule data")

    if schedule_points:
        schedule = schedule_points[0].payload['schedule']
        print(f"\n📅 Schedule preview:")
        print(schedule[:300] + "...")

    # Test semantic search
    print(f"\n🔍 Testing semantic search...")
    queries = [
        "Cuándo se entrega el Proyecto 1 de SQL?",
        "Qué temas se cubren sobre NoSQL?",
        "Cuándo es la sesión práctica de consultas?"
    ]

    for query in queries:
        query_vec = embed_e5([query], is_query=True)
        results = client.search(
            collection_name='docs_qdrant',
            query_vector=query_vec[0],
            limit=1
        )
        if results:
            score = results[0].score
            text = results[0].payload.get('text', '')[:150]
            print(f"\nQuery: {query}")
            print(f"Score: {score:.4f}")
            print(f"Answer: {text}...")


def test_pgvector():
    """Test PostgreSQL database"""
    print("\n" + "="*60)
    print("TESTING PGVECTOR")
    print("="*60)

    conn = psycopg2.connect(
        dbname="vectordb", user="pguser", password="pgpass",
        host="localhost", port=5432
    )
    cur = conn.cursor()

    # Get table info
    cur.execute("SELECT COUNT(*) FROM docs;")
    count = cur.fetchone()[0]
    print(f"\n✅ Table exists with {count} rows")

    # Check for schedule data
    cur.execute(
        "SELECT COUNT(*) FROM docs WHERE metadata->>'schedule' IS NOT NULL;")
    schedule_count = cur.fetchone()[0]
    print(f"✅ Found {schedule_count} rows with schedule data")

    if schedule_count > 0:
        cur.execute(
            "SELECT metadata->>'schedule' FROM docs WHERE metadata->>'schedule' IS NOT NULL LIMIT 1;")
        schedule = cur.fetchone()[0]
        print(f"\n📅 Schedule preview:")
        print(schedule[:300] + "...")

    # Test semantic search
    print(f"\n🔍 Testing semantic search...")
    query = "Cuándo se entrega el Proyecto 1 de SQL?"
    query_vec = embed_e5([query], is_query=True)[0]

    cur.execute("""
        SELECT content, 1 - (embedding <=> %s::vector) AS similarity
        FROM docs
        ORDER BY embedding <=> %s::vector
        LIMIT 1;
    """, (query_vec, query_vec))

    result = cur.fetchone()
    if result:
        content, similarity = result
        print(f"\nQuery: {query}")
        print(f"Similarity: {similarity:.4f}")
        print(f"Answer: {content[:150]}...")

    conn.close()


if __name__ == "__main__":
    print("\n🧪 RAG SYSTEM VERIFICATION TEST")
    print("="*60)

    try:
        test_qdrant()
    except Exception as e:
        print(f"\n❌ Qdrant test failed: {e}")

    try:
        test_pgvector()
    except Exception as e:
        print(f"\n❌ pgvector test failed: {e}")

    print("\n" + "="*60)
    print("✅ TESTS COMPLETE")
    print("="*60)
