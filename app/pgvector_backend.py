import psycopg2
import json

# SQL templates for different table formats
CLEAN_SQL = """ 
SELECT content, source_path, page, chunk_id, metadata,
       (1 - (embedding <=> %s::vector)) AS similarity 
FROM docs_clean 
ORDER BY embedding <=> %s::vector 
LIMIT %s; 
"""

LEGACY_SQL = """ 
SELECT content, metadata->>'path' AS path, 
       doc_id, chunk_id, metadata,
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


def get_available_table(conn):
    """Determine which table format to use"""
    try:
        with conn.cursor() as cur:
            # Check for clean table first
            cur.execute(
                "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'docs_clean');")
            if cur.fetchone()[0]:
                # Verify it has data
                cur.execute("SELECT COUNT(*) FROM docs_clean;")
                if cur.fetchone()[0] > 0:
                    return "clean"

            # Check for legacy table
            cur.execute(
                "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'docs');")
            if cur.fetchone()[0]:
                cur.execute("SELECT COUNT(*) FROM docs;")
                if cur.fetchone()[0] > 0:
                    return "legacy"

            # Default to clean (will be created by ingest)
            return "clean"

    except Exception as e:
        print(f"Error checking tables: {e}")
        return "legacy"  # Fallback


def search_pgvector(query_emb, k=5):
    """
    Search using PostgreSQL + pgvector with proper cosine similarity.
    Supports both clean pipeline and legacy table formats.

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
        table_format = get_available_table(conn)

        with conn.cursor() as cur:
            # Convert embedding to proper format
            emb_str = json.dumps(query_emb)

            if table_format == "clean":
                cur.execute(CLEAN_SQL, (emb_str, emb_str, k))
                rows = cur.fetchall()

                results = []
                for r in rows:
                    # Clean format: content, source_path, page, chunk_id, metadata, similarity
                    metadata = r[4] if r[4] else {}
                    results.append({
                        "content": r[0],
                        "path": r[1],           # source_path
                        "page": r[2],           # page number
                        "chunk_id": r[3],       # chunk_id
                        "score": float(r[5]),   # similarity
                        "metadata": metadata,
                        "doc_id": "",           # Legacy compatibility
                    })

            else:  # legacy format
                cur.execute(LEGACY_SQL, (emb_str, emb_str, k))
                rows = cur.fetchall()

                results = []
                for r in rows:
                    # Legacy format: content, path, doc_id, chunk_id, metadata, similarity
                    metadata = r[4] if r[4] else {}
                    results.append({
                        "content": r[0],
                        "path": r[1],           # path from metadata
                        "doc_id": r[2],         # doc_id
                        "chunk_id": r[3],       # chunk_id
                        "score": float(r[5]),   # similarity
                        "metadata": metadata,
                        "page": None,           # Not available in legacy
                    })

            return results

    except Exception as e:
        print(f"Error in pgvector search: {e}")
        return []
    finally:
        if conn:
            conn.close()
