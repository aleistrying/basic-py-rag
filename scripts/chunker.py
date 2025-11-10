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


def smart_sentence_split(text: str) -> List[str]:
    """
    Smart sentence splitting that preserves schedule/time information
    """
    # Split on sentence boundaries but keep time/schedule info together
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)

    # Group related sentences (especially for schedules)
    grouped = []
    current_group = []

    for sentence in sentences:
        current_group.append(sentence)

        # Time/schedule indicators that should stay together
        has_time_info = any(pattern in sentence.lower() for pattern in [
            'hora', 'tiempo', 'clase', 'lunes', 'martes', 'miércoles', 'jueves', 'viernes',
            'sábado', 'domingo', 'am', 'pm', 'mañana', 'tarde', 'cronograma', 'fecha',
            'entrega', 'proyecto', 'evaluación', 'examen', ':', 'h'
        ])

        # End group if we have enough content or hit a natural break
        if (len(' '.join(current_group)) > 100 and not has_time_info) or len(' '.join(current_group)) > 300:
            grouped.append(' '.join(current_group))
            current_group = []

    # Add remaining sentences
    if current_group:
        grouped.append(' '.join(current_group))

    return grouped


def chunk_text_tokens(text: str, max_tokens: int = CHUNK_TOKENS, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Memory-efficient word-based chunking with overlap.
    Optimized for granular search queries like "what time are classes".
    Uses smart sentence splitting to keep related information together.

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

    # Try smart sentence-based chunking first for better semantic coherence
    sentences = smart_sentence_split(text)

    # If we have good sentence splits, use them
    if len(sentences) > 1:
        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            sentence_words = len(sentence.split())

            # If adding this sentence would exceed max_tokens, finalize current chunk
            if current_length + sentence_words > max_tokens and current_chunk:
                chunk_text = ' '.join(current_chunk).strip()
                if len(chunk_text) > 10:
                    chunks.append(chunk_text)

                # Start new chunk with overlap (keep last sentence if room)
                if overlap > 0 and current_chunk:
                    current_chunk = current_chunk[-1:] if len(
                        current_chunk[-1].split()) < overlap else []
                    current_length = len(' '.join(current_chunk).split())
                else:
                    current_chunk = []
                    current_length = 0

            current_chunk.append(sentence)
            current_length += sentence_words

        # Add remaining chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk).strip()
            if len(chunk_text) > 10:
                chunks.append(chunk_text)

        if chunks:
            return chunks

    # Fallback to word-based chunking if sentence splitting doesn't work well
    words = text.split()
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

        # Only add non-empty chunks (lowered threshold for granular search)
        # Allow very short chunks for specific details
        if chunk_text and len(chunk_text) > 10:
            chunks.append(chunk_text)

        # Move start with overlap (but ensure progress)
        next_start = end - overlap_words
        if next_start <= start:
            # Smaller steps for better coverage
            next_start = start + max(1, max_words // 3)
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

                # Check if this is already chunked content from enhanced_pdf_cleaner
                if 'chunk_id' in page_record and 'content' in page_record:
                    # Already chunked, just write it out but fix the page field for PostgreSQL
                    total_chunks += 1

                    # Fix page field to be integer
                    page_value = page_record.get("page", 1)
                    if isinstance(page_value, str) and '-' in page_value:
                        # Extract first page number from range like "1-25"
                        try:
                            page_value = int(page_value.split('-')[0])
                        except (ValueError, IndexError):
                            page_value = 1
                    elif not isinstance(page_value, int):
                        try:
                            page_value = int(page_value)
                        except (ValueError, TypeError):
                            page_value = 1

                    # Update the record with proper page number
                    page_record["page"] = page_value
                    output_file.write(json.dumps(
                        page_record, ensure_ascii=False) + '\n')
                    continue

                source_path = page_record["source_path"]
                page_num = page_record["page"]
                page_text = page_record.get(
                    "text", page_record.get("content", ""))
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
