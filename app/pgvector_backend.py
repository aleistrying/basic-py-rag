import psycopg2
import json

# Cosine similarity query using <=> operator (cosine distance)
# Distance is in [0, 2], so similarity = 1 - distance gives [-1, 1] range
# For normalized vectors, distance ∈ [0, 2] → similarity ∈ [-1, 1]
SQL = """ 
SELECT content, metadata->>'path' AS path, 
       doc_id, chunk_id, 
       (1 - (embedding <=> %s::vector)) AS similarity 
FROM docs 
ORDER BY embedding <=> %s::vector 
LIMIT %s; 
"""


def get_connection():
    """Get a fresh PostgreSQL connection"""
    try:
        conn = psycopg2.connect(
            dbname="vectordb",
            user="pguser",
            password="pgpass",
            host="localhost",
            port=5432
        )
        conn.autocommit = True
        return conn
    except Exception as e:
        print(f"Error connecting to PostgreSQL: {e}")
        return None


def search_pgvector(query_emb, k=5):
    """
    Search using PostgreSQL + pgvector with proper cosine similarity.

    Args:
        query_emb: Query embedding (768-dim for e5-base)
        k: Number of results to return

    Returns:
        List of results with cosine similarity scores in [-1, 1] range
    """
    conn = get_connection()
    if not conn:
        return []  # Return empty results if can't connect

    try:
        with conn.cursor() as cur:
            # Convert embedding to proper format
            emb_str = json.dumps(query_emb)
            cur.execute(SQL, (emb_str, emb_str, k))
            rows = cur.fetchall()

        results = []
        for r in rows:
            results.append({
                "content": r[0],           # content
                "path": r[1],              # path from metadata
                "doc_id": r[2],            # doc_id
                "chunk_id": r[3],          # chunk_id
                "score": float(r[4])       # cosine similarity [-1, 1]
            })
        return results

    except Exception as e:
        print(f"Error in pgvector search: {e}")
        return []
    finally:
        if conn:
            conn.close()
