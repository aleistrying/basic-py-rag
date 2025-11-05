"""
Memory-efficient chunker that splits clean text into overlapping chunks.
Uses simple word-based chunking to avoid memory issues.
"""
import json
import re
from pathlib import Path
from typing import List, Dict

from ingest_config import CLEAN_DIR, CHUNK_TOKENS, CHUNK_OVERLAP


def simple_token_count(text: str) -> int:
    """Simple word-based token approximation - no heavy tokenizers!"""
    # Split on whitespace and punctuation
    tokens = re.findall(r'\w+', text)  # Just words
    return len(tokens)


def chunk_text_tokens(text: str, max_tokens: int = CHUNK_TOKENS, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Memory-efficient word-based chunking with overlap.
    No heavy tokenizers - just simple word counting!

    Args:
        text: Input text to chunk
        max_tokens: Maximum tokens per chunk (approximated as words)  
        overlap: Number of overlapping tokens between chunks

    Returns:
        List of text chunks
    """
    if not text or not text.strip():
        return []

    # Clean the text a bit
    text = re.sub(r'\s+', ' ', text).strip()

    # Split into words (simple and fast)
    words = text.split()

    # Simple approximation: 1 token ≈ 1 word (close enough for chunking)
    max_words = max_tokens
    overlap_words = overlap

    # If text is short enough, return as single chunk
    if len(words) <= max_words:
        return [text]

    chunks = []
    start = 0

    while start < len(words):
        # Get end position for this chunk
        end = min(len(words), start + max_words)

        # Extract words for this chunk
        chunk_words = words[start:end]
        chunk_text = " ".join(chunk_words).strip()

        # Only add non-empty chunks
        if chunk_text and len(chunk_text) > 20:  # Skip very short chunks
            chunks.append(chunk_text)

        # Move start with overlap (but ensure progress)
        next_start = end - overlap_words
        if next_start <= start:
            next_start = start + max(1, max_words // 2)  # Ensure progress
        start = next_start

    return chunks


def add_chunk_metadata(chunk: str, source_path: str, page: int, chunk_id: int, extractor: str) -> Dict:
    """Add metadata to chunk - fast and memory-efficient"""
    # Basic stats (no heavy processing)
    char_count = len(chunk)
    word_count = len(chunk.split())
    token_count = word_count  # Simple 1:1 approximation

    return {
        "source_path": source_path,
        "page": page,
        "chunk_id": chunk_id,
        "content": chunk,
        "metadata": {
            "extractor": extractor,
            "char_count": char_count,
            "word_count": word_count,
            "token_count": token_count,
            "source_name": Path(source_path).name
        }
    }


def chunk_jsonl(clean_jsonl_path: Path) -> Path:
    """
    Convert clean JSONL to chunked JSONL.

    Input format: {"source_path": str, "page": int, "text": str, "extractor": str}
    Output format: {"source_path": str, "page": int, "chunk_id": int, "content": str, "metadata": dict}
    """
    output_path = clean_jsonl_path.with_suffix(".chunks.jsonl")

    total_chunks = 0
    total_pages = 0

    try:
        with open(clean_jsonl_path, "r", encoding="utf-8") as input_file, \
                open(output_path, "w", encoding="utf-8") as output_file:

            for line in input_file:
                page_record = json.loads(line.strip())
                total_pages += 1

                source_path = page_record["source_path"]
                page_num = page_record["page"]
                page_text = page_record["text"]
                extractor = page_record["extractor"]

                # Chunk the page text
                chunks = chunk_text_tokens(
                    page_text, max_tokens=CHUNK_TOKENS, overlap=CHUNK_OVERLAP)

                # Write chunks with metadata
                for chunk_idx, chunk_text in enumerate(chunks):
                    chunk_record = add_chunk_metadata(
                        chunk=chunk_text,
                        source_path=source_path,
                        page=page_num,
                        chunk_id=chunk_idx,
                        extractor=extractor
                    )

                    output_file.write(json.dumps(
                        chunk_record, ensure_ascii=False) + "\n")
                    total_chunks += 1

        print(f"  📊 Created {total_chunks} chunks from {total_pages} pages")
        return output_path

    except Exception as e:
        print(f"  ❌ Chunking failed: {e}")
        raise


def chunk_all_clean_files():
    """Chunk all clean JSONL files"""
    clean_path = Path(CLEAN_DIR)
    if not clean_path.exists():
        print(f"❌ Clean directory {CLEAN_DIR} does not exist!")
        print("   Run pdf_cleaner.py first")
        return

    clean_files = list(clean_path.glob("*.jsonl"))
    # Filter out already chunked files
    clean_files = [
        f for f in clean_files if not f.name.endswith(".chunks.jsonl")]

    if not clean_files:
        print(f"❌ No clean JSONL files found in {CLEAN_DIR}")
        return

    print(f"✂️  Chunking {len(clean_files)} clean files...")

    total_chunks = 0
    for clean_file in clean_files:
        print(f"\n📄 Processing: {clean_file.name}")
        try:
            chunk_file = chunk_jsonl(clean_file)

            # Count chunks in output file
            with open(chunk_file, "r") as f:
                file_chunks = sum(1 for _ in f)
                total_chunks += file_chunks
                print(
                    f"  ✅ Generated {file_chunks} chunks → {chunk_file.name}")

        except Exception as e:
            print(f"  ❌ Failed: {e}")
            continue

    print(f"\n✅ Total chunks generated: {total_chunks}")
    print(f"📁 Chunked files saved to: {CLEAN_DIR}")


if __name__ == "__main__":
    chunk_all_clean_files()
