#!/usr/bin/env python3
"""
Enhanced PDF cleaner that integrates advanced extraction capabilities
Replaces the basic pdf_cleaner.py with support for:
- Multiple extraction libraries with fallbacks
- Advanced table detection and extraction
- Memory-efficient processing for large documents
- Better text normalization and watermark removal
"""

from ingest_config import RAW_DIR, CLEAN_DIR, MIN_CHARS
import os
import sys
import json
import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Generator

# Add project to path
sys.path.append(str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import pdfplumber
except ImportError:
    pdfplumber = None

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

try:
    import PyPDF2
except ImportError:
    PyPDF2 = None

try:
    from unstructured.partition.pdf import partition_pdf
except ImportError:
    partition_pdf = None


# Ensure clean directory exists
Path(CLEAN_DIR).mkdir(parents=True, exist_ok=True)


class EnhancedPDFExtractor:
    """Enhanced PDF extractor with multiple libraries and fallbacks"""

    def __init__(self):
        self.available_extractors = self._check_available_extractors()
        logger.info(
            f"Available extractors: {list(self.available_extractors.keys())}")

    def _check_available_extractors(self) -> Dict[str, bool]:
        """Check which extraction libraries are available"""
        return {
            'pdfplumber': pdfplumber is not None,
            'pymupdf': fitz is not None,
            'pypdf2': PyPDF2 is not None,
            'unstructured': partition_pdf is not None
        }

    def extract_pdf_intelligently(self, pdf_path: Path, max_pages_per_chunk: int = 50) -> Generator[Dict[str, Any], None, None]:
        """
        Extract PDF with intelligent chunking for memory efficiency
        """
        logger.info(f"Starting intelligent extraction of {pdf_path}")

        # Get PDF info first
        pdf_info = self._get_pdf_info(pdf_path)
        total_pages = pdf_info['total_pages']

        logger.info(
            f"PDF has {total_pages} pages, processing in chunks of {max_pages_per_chunk}")

        # Process in chunks to avoid memory issues
        for start_page in range(1, total_pages + 1, max_pages_per_chunk):
            end_page = min(start_page + max_pages_per_chunk - 1, total_pages)

            logger.info(f"Processing pages {start_page}-{end_page}")

            # Try extraction methods in order of preference
            chunk_result = self._extract_page_range(
                pdf_path, start_page, end_page)

            if chunk_result['success']:
                yield chunk_result
            else:
                logger.warning(
                    f"Failed to extract pages {start_page}-{end_page}")

    def _get_pdf_info(self, pdf_path: Path) -> Dict[str, Any]:
        """Get basic PDF information"""
        try:
            if fitz:
                doc = fitz.open(pdf_path)
                info = {
                    'total_pages': doc.page_count,
                    'file_size': os.path.getsize(pdf_path),
                    'method': 'pymupdf'
                }
                doc.close()
                return info
        except:
            pass

        # Fallback to PyPDF2
        try:
            if PyPDF2:
                with open(pdf_path, 'rb') as file:
                    reader = PyPDF2.PdfReader(file)
                    return {
                        'total_pages': len(reader.pages),
                        'file_size': os.path.getsize(pdf_path),
                        'method': 'pypdf2'
                    }
        except:
            pass

        return {'total_pages': 0, 'file_size': 0, 'method': 'unknown'}

    def _extract_page_range(self, pdf_path: Path, start_page: int, end_page: int) -> Dict[str, Any]:
        """Extract a specific range of pages using the best available method"""

        # Try extraction methods in order of preference
        extractors = [
            ('pdfplumber', self._extract_with_pdfplumber),
            ('unstructured', self._extract_with_unstructured),
            ('pymupdf', self._extract_with_pymupdf),
            ('pypdf2', self._extract_with_pypdf2)
        ]

        for method_name, method_func in extractors:
            if self.available_extractors.get(method_name, False):
                try:
                    result = method_func(pdf_path, start_page, end_page)
                    if result['success'] and result['text'].strip():
                        logger.info(
                            f"Successfully extracted pages {start_page}-{end_page} with {method_name}")
                        return result
                except Exception as e:
                    logger.warning(
                        f"{method_name} failed for pages {start_page}-{end_page}: {e}")
                    continue

        return {
            'success': False,
            'text': '',
            'tables': [],
            'method': 'none',
            'page_range': f"{start_page}-{end_page}",
            'error': 'All extraction methods failed'
        }

    def _extract_with_pdfplumber(self, pdf_path: Path, start_page: int, end_page: int) -> Dict[str, Any]:
        """Extract using pdfplumber with enhanced table detection"""
        if not pdfplumber:
            return {'success': False, 'text': '', 'tables': [], 'method': 'pdfplumber', 'error': 'Not available'}

        try:
            text_parts = []
            all_tables = []

            with pdfplumber.open(pdf_path) as pdf:
                for page_num in range(start_page - 1, min(end_page, len(pdf.pages))):
                    page = pdf.pages[page_num]

                    # Extract text
                    page_text = page.extract_text() or ""
                    if page_text:
                        text_parts.append(f"\n--- Page {page_num + 1} ---\n")
                        text_parts.append(page_text)

                    # Extract tables
                    tables = page.extract_tables()
                    if tables:
                        for table in tables:
                            if table and len(table) > 1:
                                formatted_table = self._format_table_as_markdown(
                                    table)
                                all_tables.append(formatted_table)
                                text_parts.append(
                                    f"\n[TABLE]\n{formatted_table}\n[/TABLE]\n")

            combined_text = "\n".join(text_parts)
            normalized_text = self._normalize_text(combined_text)

            return {
                'success': True,
                'text': normalized_text,
                'tables': all_tables,
                'method': 'pdfplumber',
                'page_range': f"{start_page}-{end_page}",
                'pages_processed': end_page - start_page + 1
            }

        except Exception as e:
            return {
                'success': False,
                'text': '',
                'tables': [],
                'method': 'pdfplumber',
                'error': str(e)
            }

    def _extract_with_unstructured(self, pdf_path: Path, start_page: int, end_page: int) -> Dict[str, Any]:
        """Extract using unstructured library"""
        if not partition_pdf:
            return {'success': False, 'text': '', 'tables': [], 'method': 'unstructured', 'error': 'Not available'}

        try:
            # Note: unstructured doesn't support page ranges easily, so we process whole doc
            # and filter later (not ideal for huge docs, but works for medium ones)
            elements = partition_pdf(str(pdf_path))

            text_parts = []
            tables_found = []

            for element in elements:
                if hasattr(element, 'text') and element.text:
                    # Add semantic markup based on element type
                    element_type = type(element).__name__
                    if 'Title' in element_type:
                        text_parts.append(f"\n# {element.text}\n")
                    elif 'Header' in element_type:
                        text_parts.append(f"\n## {element.text}\n")
                    elif 'Table' in element_type:
                        tables_found.append(element.text)
                        text_parts.append(
                            f"\n[TABLE]\n{element.text}\n[/TABLE]\n")
                    else:
                        text_parts.append(element.text)

            combined_text = "\n".join(text_parts)
            normalized_text = self._normalize_text(combined_text)

            return {
                'success': True,
                'text': normalized_text,
                'tables': tables_found,
                'method': 'unstructured',
                'page_range': f"{start_page}-{end_page}",
                'semantic_structure': True
            }

        except Exception as e:
            return {
                'success': False,
                'text': '',
                'tables': [],
                'method': 'unstructured',
                'error': str(e)
            }

    def _extract_with_pymupdf(self, pdf_path: Path, start_page: int, end_page: int) -> Dict[str, Any]:
        """Extract using PyMuPDF"""
        if not fitz:
            return {'success': False, 'text': '', 'tables': [], 'method': 'pymupdf', 'error': 'Not available'}

        try:
            text_parts = []
            tables_found = []

            doc = fitz.open(pdf_path)

            for page_num in range(start_page - 1, min(end_page, doc.page_count)):
                page = doc[page_num]

                # Extract text
                page_text = page.get_text()
                if page_text:
                    text_parts.append(f"\n--- Page {page_num + 1} ---\n")
                    text_parts.append(page_text)

                    # Simple table detection
                    if self._looks_like_table(page_text):
                        tables_found.append(page_text)

            doc.close()

            combined_text = "\n".join(text_parts)
            normalized_text = self._normalize_text(combined_text)

            return {
                'success': True,
                'text': normalized_text,
                'tables': tables_found,
                'method': 'pymupdf',
                'page_range': f"{start_page}-{end_page}",
                'pages_processed': end_page - start_page + 1
            }

        except Exception as e:
            return {
                'success': False,
                'text': '',
                'tables': [],
                'method': 'pymupdf',
                'error': str(e)
            }

    def _extract_with_pypdf2(self, pdf_path: Path, start_page: int, end_page: int) -> Dict[str, Any]:
        """Extract using PyPDF2 as fallback"""
        if not PyPDF2:
            return {'success': False, 'text': '', 'tables': [], 'method': 'pypdf2', 'error': 'Not available'}

        try:
            text_parts = []

            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)

                for page_num in range(start_page - 1, min(end_page, len(reader.pages))):
                    page = reader.pages[page_num]
                    page_text = page.extract_text()

                    if page_text:
                        text_parts.append(f"\n--- Page {page_num + 1} ---\n")
                        text_parts.append(page_text)

            combined_text = "\n".join(text_parts)
            normalized_text = self._normalize_text(combined_text)

            return {
                'success': True,
                'text': normalized_text,
                'tables': [],
                'method': 'pypdf2',
                'page_range': f"{start_page}-{end_page}",
                'pages_processed': end_page - start_page + 1
            }

        except Exception as e:
            return {
                'success': False,
                'text': '',
                'tables': [],
                'method': 'pypdf2',
                'error': str(e)
            }

    def _format_table_as_markdown(self, table: List[List[str]]) -> str:
        """Format table as markdown"""
        if not table or len(table) < 2:
            return ""

        lines = []
        for row_idx, row in enumerate(table):
            if row and any(cell for cell in row if cell):
                clean_row = [str(cell).strip() if cell else "" for cell in row]
                lines.append(" | ".join(clean_row))
                if row_idx == 0:  # Add separator after header
                    lines.append(" | ".join(["---"] * len(clean_row)))

        return "\n".join(lines)

    def _looks_like_table(self, text: str) -> bool:
        """Heuristic to detect table-like content"""
        if not text or len(text) < 50:
            return False

        lines = text.strip().split('\n')
        if len(lines) < 3:
            return False

        # Check for patterns indicating tables
        separator_count = sum(1 for line in lines if any(
            sep in line for sep in ['|', '\t', '  ']))
        numeric_lines = sum(1 for line in lines if any(
            c.isdigit() for c in line))

        return (separator_count / len(lines) > 0.5) or (numeric_lines / len(lines) > 0.4)

    def _normalize_text(self, text: str) -> str:
        """Enhanced text normalization"""
        if not text:
            return ""

        # Remove watermarks and institutional marks
        text = self._remove_watermarks(text)

        # Basic normalization
        import re

        # Remove null bytes and control characters
        text = text.replace("\x00", " ")
        text = text.replace("\ufffd", "")
        text = text.replace("\u00ad", "")  # Soft hyphens

        # Merge hyphenated words across line breaks
        text = re.sub(r"-\s*\n\s*", "", text)

        # Normalize whitespace
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\s+\n", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()

    def _remove_watermarks(self, text: str) -> str:
        """Remove watermarks and PDF artifacts"""
        import re

        # Common watermark patterns
        watermarks = [
            r"UTP-FISC-2S2025",
            r"UTP[\s\-]*FISC[\s\-]*2S2025",
            r"Universidad\s+Tecnológica\s+de\s+Panamá",
            r"FISC[\s\-]*\d+S\d+",
            r"Facultad\s+de\s+Ingeniería\s+de\s+Sistemas\s+Computacionales",
            r"Maestría\s+en\s+Analítica\s+de\s+Datos",
            r"Sistemas\s+de\s+Bases\s+de\s+Datos\s+Avanzadas",
            r"GUIA\s+DE\s+CURSO.*SEMESTRE\s+2025"
        ]

        for pattern in watermarks:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE)

        # Remove scattered letter patterns (OCR artifacts)
        corrupted_patterns = [
            r"\b[A-Z]\s+[A-Z]\s+[-–]\s+[A-Z]\s+\d+\s+[A-Z]\b",
            r"\b\d+\s+[A-Z]\s+\d+\s+[A-Z]\s+\d+\s+[A-Z]\b",
            r"(?:[A-Z]\s+){3,}\d+",
            r"(?:\d+\s+){3,}[A-Z]"
        ]

        for pattern in corrupted_patterns:
            text = re.sub(pattern, "", text)

        return text


class ImprovedChunker:
    """Improved chunking system that properly handles large documents"""

    def __init__(self, target_chunk_size: int = 400, max_chunk_size: int = 800,
                 min_chunk_size: int = 100, overlap_size: int = 50):
        self.target_chunk_size = target_chunk_size
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.overlap_size = overlap_size

    def chunk_extracted_content(self, extraction_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Convert extraction result into properly sized chunks"""

        if not extraction_result['success'] or not extraction_result['text']:
            return []

        text = extraction_result['text']

        # Debug: Check text length
        logger.debug(f"Processing text of length {len(text)} for chunking")

        # If text is very short, just return as single chunk
        if len(text) <= self.max_chunk_size:
            if len(text) >= self.min_chunk_size:
                return [self._create_chunk(0, text, extraction_result)]
            else:
                return []  # Too short to be useful

        # Split by logical boundaries first (paragraphs, then sentences)
        chunks = self._chunk_by_paragraphs(text, extraction_result)

        # If we got very few chunks, try more aggressive sentence-based chunking
        if len(chunks) < 3 and len(text) > self.target_chunk_size * 2:
            logger.debug(
                f"Few chunks ({len(chunks)}) for large text, trying sentence chunking")
            chunks = self._chunk_by_sentences(text, extraction_result)

        return chunks

    def _chunk_by_paragraphs(self, text: str, extraction_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Chunk by paragraphs first, then by size"""
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]

        if not paragraphs:
            return self._chunk_by_sentences(text, extraction_result)

        chunks = []
        current_chunk = []
        current_size = 0
        chunk_id = 0

        for paragraph in paragraphs:
            paragraph_size = len(paragraph)

            # If this paragraph alone is too big, split it
            if paragraph_size > self.max_chunk_size:
                # Finalize current chunk if it has content
                if current_chunk:
                    chunk_text = "\n\n".join(current_chunk)
                    if len(chunk_text) >= self.min_chunk_size:
                        chunks.append(self._create_chunk(
                            chunk_id, chunk_text, extraction_result))
                        chunk_id += 1
                    current_chunk = []
                    current_size = 0

                # Split the large paragraph by sentences
                large_para_chunks = self._split_large_paragraph(
                    paragraph, extraction_result, chunk_id)
                chunks.extend(large_para_chunks)
                chunk_id += len(large_para_chunks)
                continue

            # If adding this paragraph would exceed max size, finalize current chunk
            # +2 for \n\n
            if current_chunk and (current_size + paragraph_size + 2 > self.max_chunk_size):
                chunk_text = "\n\n".join(current_chunk)
                if len(chunk_text) >= self.min_chunk_size:
                    chunks.append(self._create_chunk(
                        chunk_id, chunk_text, extraction_result))
                    chunk_id += 1

                # Start new chunk with some overlap if current chunk is large enough
                if current_size > self.target_chunk_size and current_chunk:
                    # Take last paragraph as overlap
                    overlap_text = current_chunk[-1]
                    if len(overlap_text) <= self.overlap_size:
                        current_chunk = [overlap_text, paragraph]
                        current_size = len(overlap_text) + paragraph_size + 2
                    else:
                        current_chunk = [paragraph]
                        current_size = paragraph_size
                else:
                    current_chunk = [paragraph]
                    current_size = paragraph_size
            else:
                current_chunk.append(paragraph)
                current_size += paragraph_size + \
                    (2 if current_chunk else 0)  # +2 for \n\n separator

            # If we've reached target size, consider finalizing
            if current_size >= self.target_chunk_size:
                chunk_text = "\n\n".join(current_chunk)
                if len(chunk_text) >= self.min_chunk_size:
                    chunks.append(self._create_chunk(
                        chunk_id, chunk_text, extraction_result))
                    chunk_id += 1

                # Start new chunk with overlap
                if len(current_chunk) > 1:
                    overlap_text = current_chunk[-1]
                    if len(overlap_text) <= self.overlap_size:
                        current_chunk = [overlap_text]
                        current_size = len(overlap_text)
                    else:
                        current_chunk = []
                        current_size = 0
                else:
                    current_chunk = []
                    current_size = 0

        # Add final chunk if there's content
        if current_chunk:
            chunk_text = "\n\n".join(current_chunk)
            if len(chunk_text) >= self.min_chunk_size:
                chunks.append(self._create_chunk(
                    chunk_id, chunk_text, extraction_result))

        return chunks

    def _split_large_paragraph(self, paragraph: str, extraction_result: Dict[str, Any], start_chunk_id: int) -> List[Dict[str, Any]]:
        """Split a large paragraph into smaller chunks"""
        sentences = self._split_into_sentences(paragraph)

        chunks = []
        current_chunk = []
        current_size = 0
        chunk_id = start_chunk_id

        for sentence in sentences:
            sentence_size = len(sentence)

            if current_chunk and (current_size + sentence_size + 1 > self.max_chunk_size):
                chunk_text = " ".join(current_chunk)
                if len(chunk_text) >= self.min_chunk_size:
                    chunks.append(self._create_chunk(
                        chunk_id, chunk_text, extraction_result))
                    chunk_id += 1

                # Start new chunk with overlap
                overlap_sentences = self._get_overlap_sentences(current_chunk)
                current_chunk = overlap_sentences + [sentence]
                current_size = sum(len(s) + 1 for s in current_chunk) - 1
            else:
                current_chunk.append(sentence)
                current_size += sentence_size + (1 if current_chunk else 0)

        # Add final chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            if len(chunk_text) >= self.min_chunk_size:
                chunks.append(self._create_chunk(
                    chunk_id, chunk_text, extraction_result))

        return chunks

    def _chunk_by_sentences(self, text: str, extraction_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Fallback: chunk by sentences when paragraph chunking doesn't work well"""
        sentences = self._split_into_sentences(text)

        if not sentences:
            # Very last resort: split by character count
            return self._chunk_by_character_count(text, extraction_result)

        chunks = []
        current_chunk = []
        current_size = 0
        chunk_id = 0

        for sentence in sentences:
            sentence_size = len(sentence)

            # If adding this sentence would exceed max size, finalize current chunk
            if current_chunk and (current_size + sentence_size + 1 > self.max_chunk_size):
                chunk_text = " ".join(current_chunk)
                if len(chunk_text) >= self.min_chunk_size:
                    chunks.append(self._create_chunk(
                        chunk_id, chunk_text, extraction_result))
                    chunk_id += 1

                # Start new chunk with overlap
                overlap_sentences = self._get_overlap_sentences(current_chunk)
                current_chunk = overlap_sentences + [sentence]
                current_size = sum(len(s) + 1 for s in current_chunk) - 1
            else:
                current_chunk.append(sentence)
                current_size += sentence_size + \
                    (1 if len(current_chunk) > 1 else 0)

            # If we've reached target size, consider finalizing
            if current_size >= self.target_chunk_size:
                chunk_text = " ".join(current_chunk)
                if len(chunk_text) >= self.min_chunk_size:
                    chunks.append(self._create_chunk(
                        chunk_id, chunk_text, extraction_result))
                    chunk_id += 1

                # Start new chunk with overlap
                overlap_sentences = self._get_overlap_sentences(current_chunk)
                current_chunk = overlap_sentences
                current_size = sum(
                    len(s) + 1 for s in current_chunk) - 1 if current_chunk else 0

        # Add final chunk if there's content
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            if len(chunk_text) >= self.min_chunk_size:
                chunks.append(self._create_chunk(
                    chunk_id, chunk_text, extraction_result))

        return chunks

    def _chunk_by_character_count(self, text: str, extraction_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Last resort: split by character count with word boundaries"""
        words = text.split()

        chunks = []
        current_chunk = []
        current_size = 0
        chunk_id = 0

        for word in words:
            word_size = len(word) + 1  # +1 for space

            if current_chunk and (current_size + word_size > self.max_chunk_size):
                chunk_text = " ".join(current_chunk)
                if len(chunk_text) >= self.min_chunk_size:
                    chunks.append(self._create_chunk(
                        chunk_id, chunk_text, extraction_result))
                    chunk_id += 1

                # Start new chunk with some overlap
                # Last 10 words as overlap
                overlap_words = current_chunk[-min(10, len(current_chunk)):]
                current_chunk = overlap_words + [word]
                current_size = sum(len(w) + 1 for w in current_chunk) - 1
            else:
                current_chunk.append(word)
                current_size += word_size

            # If we've reached target size, finalize
            if current_size >= self.target_chunk_size:
                chunk_text = " ".join(current_chunk)
                if len(chunk_text) >= self.min_chunk_size:
                    chunks.append(self._create_chunk(
                        chunk_id, chunk_text, extraction_result))
                    chunk_id += 1

                # Start new chunk with overlap
                overlap_words = current_chunk[-min(10, len(current_chunk)):]
                current_chunk = overlap_words
                current_size = sum(
                    len(w) + 1 for w in current_chunk) - 1 if current_chunk else 0

        # Add final chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            if len(chunk_text) >= self.min_chunk_size:
                chunks.append(self._create_chunk(
                    chunk_id, chunk_text, extraction_result))

        return chunks

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences while preserving structure"""
        import re

        # Split on sentence boundaries but be smart about abbreviations
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)

        # Clean and filter sentences
        clean_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 20:  # Skip very short fragments
                clean_sentences.append(sentence)

        return clean_sentences

    def _get_overlap_sentences(self, sentences: List[str]) -> List[str]:
        """Get sentences for overlap with next chunk"""
        if not sentences:
            return []

        # Take last 1-2 sentences for overlap, but don't exceed overlap_size
        overlap = []
        total_size = 0

        for sentence in reversed(sentences):
            if total_size + len(sentence) <= self.overlap_size:
                overlap.insert(0, sentence)
                total_size += len(sentence)
            else:
                break

        return overlap

    def _is_good_break_point(self, sentence: str) -> bool:
        """Determine if this is a good place to break a chunk"""
        # Look for structural indicators
        break_indicators = [
            'Chapter', 'Section', 'Part', 'Capítulo', 'Sección',
            'Introduction', 'Conclusion', 'Summary', 'Introducción', 'Conclusión'
        ]

        return any(indicator in sentence for indicator in break_indicators)

    def _create_chunk(self, chunk_id: int, text: str, extraction_result: Dict[str, Any]) -> Dict[str, Any]:
        """Create a chunk with metadata"""
        # Extract page number from page range for PostgreSQL compatibility
        page_range = extraction_result.get('page_range', 'unknown')
        try:
            if '-' in str(page_range):
                # Take first page number
                page_num = int(str(page_range).split('-')[0])
            else:
                page_num = int(page_range) if str(page_range).isdigit() else 1
        except (ValueError, AttributeError):
            page_num = 1

        return {
            "chunk_id": chunk_id,
            "content": text,
            "metadata": {
                "char_count": len(text),
                "word_count": len(text.split()),
                "source_method": extraction_result['method'],
                "page_range": page_range,
                "page_number": page_num,  # Integer page for PostgreSQL
                "has_tables": len(extraction_result.get('tables', [])) > 0,
                "tables_count": len(extraction_result.get('tables', []))
            }
        }


def process_pdf_enhanced(pdf_path: Path) -> List[Dict[str, Any]]:
    """Process a PDF with enhanced extraction and chunking"""

    logger.info(f"Processing PDF: {pdf_path}")

    extractor = EnhancedPDFExtractor()
    chunker = ImprovedChunker(
        target_chunk_size=400, max_chunk_size=800, min_chunk_size=100, overlap_size=50)

    all_chunks = []
    total_text_extracted = 0

    # Process PDF in chunks
    for extraction_result in extractor.extract_pdf_intelligently(pdf_path, max_pages_per_chunk=25):
        if extraction_result['success']:
            text_length = len(extraction_result['text'])
            total_text_extracted += text_length

            # Convert to properly sized chunks
            chunks = chunker.chunk_extracted_content(extraction_result)
            all_chunks.extend(chunks)

            logger.info(
                f"Processed pages {extraction_result['page_range']}: {text_length} chars -> {len(chunks)} chunks")
        else:
            logger.warning(
                f"Failed to process pages {extraction_result.get('page_range', 'unknown')}")

    logger.info(
        f"Total extraction: {total_text_extracted} characters -> {len(all_chunks)} chunks")
    return all_chunks


def pdf_to_clean_jsonl_enhanced(pdf_path: Path) -> Optional[Path]:
    """Enhanced PDF to JSONL conversion"""

    output_path = Path(CLEAN_DIR) / f"{pdf_path.stem}.jsonl"

    logger.info(f"Converting {pdf_path} to {output_path}")

    # Process PDF with enhanced extraction
    chunks = process_pdf_enhanced(pdf_path)

    if not chunks:
        logger.warning(f"No content extracted from {pdf_path}")
        return None

    # Save to JSONL
    with open(output_path, 'w', encoding='utf-8') as f:
        for chunk in chunks:
            # Create record compatible with existing pipeline
            record = {
                "source_path": str(pdf_path),
                # Use integer page for PostgreSQL
                "page": chunk["metadata"].get("page_number", 1),
                "chunk_id": chunk["chunk_id"],
                "text": chunk["content"],
                "extractor": chunk["metadata"]["source_method"],
                "metadata": chunk["metadata"]
            }
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    logger.info(f"Saved {len(chunks)} chunks to {output_path}")
    return output_path


def clean_all_pdfs_enhanced():
    """Process all PDFs with enhanced extraction"""

    raw_path = Path(RAW_DIR)
    if not raw_path.exists():
        logger.error(f"Raw directory does not exist: {RAW_DIR}")
        return

    pdf_files = list(raw_path.glob("*.pdf"))
    if not pdf_files:
        logger.warning(f"No PDF files found in {RAW_DIR}")
        return

    logger.info(f"Found {len(pdf_files)} PDF files to process")

    for pdf_file in pdf_files:
        try:
            result = pdf_to_clean_jsonl_enhanced(pdf_file)
            if result:
                logger.info(f"✅ Successfully processed {pdf_file.name}")
            else:
                logger.warning(f"❌ Failed to process {pdf_file.name}")
        except Exception as e:
            logger.error(f"❌ Error processing {pdf_file.name}: {e}")


if __name__ == "__main__":
    # Test with Williams PDF if it exists
    williams_path = Path("data/old/Williams Obstetricia 26a Edicion.pdf")
    if williams_path.exists():
        print(f"🧪 Testing enhanced extraction with Williams PDF")
        chunks = process_pdf_enhanced(williams_path)
        print(f"📊 Generated {len(chunks)} chunks from Williams PDF")

        if chunks:
            avg_size = sum(len(chunk['content'])
                           for chunk in chunks) / len(chunks)
            print(f"📏 Average chunk size: {avg_size:.0f} characters")

            # Save sample
            sample_path = Path("data/clean/williams_enhanced_sample.json")
            sample_path.parent.mkdir(exist_ok=True)
            with open(sample_path, 'w', encoding='utf-8') as f:
                json.dump(chunks[:10], f, ensure_ascii=False, indent=2)
            print(f"💾 Sample saved to {sample_path}")
    else:
        print("Williams PDF not found, running general PDF processing")
        clean_all_pdfs_enhanced()
