import psycopg2
import json
import os

# SQL template for clean table format with filtering support
CLEAN_SQL_BASE = """ 
SELECT content, source_path, page, chunk_id, metadata,
       (1 - (embedding <=> %s::vector)) AS similarity 
FROM docs_clean 
WHERE 1=1 {where_clause}
ORDER BY embedding <=> %s::vector 
LIMIT %s; 
"""


def get_connection():
    """Get a fresh PostgreSQL connection"""
    # Use Docker service name when running in container, localhost otherwise
    pg_host = os.getenv("POSTGRES_HOST", "localhost")

    try:
        conn = psycopg2.connect(
            dbname="vectordb",
            user="pguser",
            password="pgpass",
            host=pg_host,
            port=5432
        )
        conn.autocommit = True
        return conn
    except Exception as e:
        print(f"Error connecting to PostgreSQL: {e}")
        return None


def get_available_table(conn):
    """Check if clean table exists and has data"""
    try:
        with conn.cursor() as cur:
            # Check for clean table
            cur.execute(
                "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'docs_clean');")
            if cur.fetchone()[0]:
                # Verify it has data
                cur.execute("SELECT COUNT(*) FROM docs_clean;")
                if cur.fetchone()[0] > 0:
                    return "clean"

            # If no clean table with data, return None to indicate missing data
            return None

    except Exception as e:
        print(f"Error checking tables: {e}")
        return None


def build_where_clause(where_dict):
    """Build SQL WHERE clause from filter dictionary"""
    if not where_dict:
        return "", []

    clauses = []
    params = []

    if 'document_type' in where_dict:
        clauses.append("source_path LIKE %s")
        params.append(f"%.{where_dict['document_type']}")

    if 'page' in where_dict:
        clauses.append("page = %s")
        params.append(where_dict['page'])

    if 'contains' in where_dict:
        clauses.append("content ILIKE %s")
        params.append(f"%{where_dict['contains']}%")

    if 'section' in where_dict:
        clauses.append("(content ILIKE %s OR metadata->>'section' ILIKE %s)")
        params.extend([f"%{where_dict['section']}%",
                      f"%{where_dict['section']}%"])

    if 'topic' in where_dict:
        clauses.append("(content ILIKE %s OR metadata->>'topic' ILIKE %s)")
        params.extend([f"%{where_dict['topic']}%", f"%{where_dict['topic']}%"])

    where_clause = " AND " + " AND ".join(clauses) if clauses else ""
    return where_clause, params


def search_pgvector(query_emb, k=5, where=None):
    """
    Search using PostgreSQL + pgvector with proper cosine similarity.
    Uses only the clean pipeline table format.

    Args:
        query_emb: Query embedding (768-dim for e5-base)
        k: Number of results to return
        where: Filter dict, supports:
            - document_type: str (e.g., "pdf", "txt", "md")
            - section: str (e.g., "objetivos", "cronograma", "evaluacion")
            - topic: str (e.g., "nosql", "vectorial", "sql")
            - page: int (for PDFs)
            - contains: str (text must contain this string)

    Returns:
        List of results with cosine similarity scores in [-1, 1] range
    """
    conn = get_connection()
    if not conn:
        return []  # Return empty results if can't connect

    try:
        table_format = get_available_table(conn)
        if table_format != "clean":
            print(
                "Warning: docs_clean table not found or empty. Please run the clean pipeline first.")
            return []

        where_clause, where_params = build_where_clause(where)

        with conn.cursor() as cur:
            # Convert embedding to proper format
            emb_str = json.dumps(query_emb)

            sql = CLEAN_SQL_BASE.format(where_clause=where_clause)
            params = [emb_str, emb_str] + where_params + [k]
            cur.execute(sql, params)
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
                })

            return results

    except Exception as e:
        print(f"Error in pgvector search: {e}")
        return []
    finally:
        if conn:
            conn.close()
