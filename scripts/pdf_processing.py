#!/usr/bin/env python3
"""
Unified PDF Processing Module
Consolidates all PDF extraction capabilities from pdf_cleaner.py, enhanced_pdf_cleaner.py, and pipeline.py
Supports multiple extraction libraries with intelligent fallbacks and memory-efficient processing
"""

from ingest_config import RAW_DIR, CLEAN_DIR, MIN_CHARS
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Generator, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial

# Add project to path
sys.path.append(str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import extraction libraries with fallbacks
try:
    import pdfplumber
except ImportError:
    pdfplumber = None
    logger.warning(
        "pdfplumber not available - table extraction will be limited")

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None
    logger.warning("PyMuPDF not available - fallback extraction limited")

try:
    import PyPDF2
except ImportError:
    PyPDF2 = None
    logger.warning("PyPDF2 not available - last resort extraction unavailable")

try:
    from unstructured.partition.pdf import partition_pdf
except ImportError:
    partition_pdf = None
    logger.warning(
        "unstructured not available - advanced semantic extraction disabled")

# OCR libraries
try:
    from pdf2image import convert_from_path
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    logger.warning(
        "OCR libraries not available - image-based PDFs will not be extracted")


# Ensure clean directory exists
Path(CLEAN_DIR).mkdir(parents=True, exist_ok=True)


class UnifiedPDFProcessor:
    """
    Unified PDF processor that combines all extraction capabilities:
    - Basic text extraction (PyPDF2, PyMuPDF, pdfplumber)
    - Advanced table detection and extraction
    - Memory-efficient streaming for large documents
    - Intelligent watermark removal and text normalization
    - Fallback strategies for robust processing
    """

    def __init__(self, max_pages_per_chunk: int = 25):
        self.max_pages_per_chunk = max_pages_per_chunk
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

    def process_pdf_file(self, pdf_path: Path, output_format: str = "jsonl") -> Optional[Path]:
        """
        Process a PDF file and output in specified format

        Args:
            pdf_path: Path to PDF file
            output_format: "jsonl" for clean JSONL, "chunks" for chunked content

        Returns:
            Path to output file or None if failed
        """
        logger.info(f"Processing PDF: {pdf_path.name}")

        if output_format == "jsonl":
            return self._pdf_to_clean_jsonl(pdf_path)
        elif output_format == "chunks":
            return self._pdf_to_chunks(pdf_path)
        else:
            raise ValueError(f"Unsupported output format: {output_format}")

    def _pdf_to_clean_jsonl(self, pdf_path: Path) -> Optional[Path]:
        """Convert PDF to clean JSONL format with page-by-page extraction"""
        output_path = Path(CLEAN_DIR) / (pdf_path.stem + ".jsonl")

        # Get PDF info for memory management
        pdf_info = self._get_pdf_info(pdf_path)
        total_pages = pdf_info.get('total_pages', 0)

        if total_pages == 0:
            logger.error(f"Could not determine page count for {pdf_path.name}")
            return None

        logger.info(
            f"Processing {total_pages} pages in chunks of {self.max_pages_per_chunk}")

        pages_processed = 0
        total_pages_written = 0

        with open(output_path, 'w', encoding='utf-8') as output_file:
            # Process in page chunks for memory efficiency
            for start_page in range(0, total_pages, self.max_pages_per_chunk):
                end_page = min(
                    start_page + self.max_pages_per_chunk, total_pages)

                logger.info(f"Processing pages {start_page + 1}-{end_page}")

                # Extract page range
                extraction_result = self._extract_page_range(
                    pdf_path, start_page, end_page)

                if extraction_result and extraction_result.get('pages'):
                    for page_data in extraction_result['pages']:
                        # Write page record to JSONL
                        page_record = {
                            "source_path": str(pdf_path),
                            "page": page_data['page'],
                            "text": page_data['text'],
                            "extractor": page_data['extractor']
                        }
                        output_file.write(json.dumps(
                            page_record, ensure_ascii=False) + '\n')
                        total_pages_written += 1

                pages_processed = end_page
                logger.info(f"Processed {pages_processed}/{total_pages} pages")

        logger.info(
            f"Completed: {total_pages_written} pages written to {output_path.name}")
        return output_path

    def _get_pdf_info(self, pdf_path: Path) -> Dict[str, Any]:
        """Get basic PDF information"""
        info = {"total_pages": 0, "file_size": 0}

        try:
            info["file_size"] = pdf_path.stat().st_size

            # Try to get page count with available libraries
            if fitz:
                doc = fitz.open(str(pdf_path))
                info["total_pages"] = doc.page_count
                doc.close()
            elif pdfplumber:
                with pdfplumber.open(pdf_path) as pdf:
                    info["total_pages"] = len(pdf.pages)
            elif PyPDF2:
                with open(pdf_path, 'rb') as file:
                    reader = PyPDF2.PdfReader(file)
                    info["total_pages"] = len(reader.pages)

        except Exception as e:
            logger.error(f"Failed to get PDF info for {pdf_path.name}: {e}")

        return info

    def _extract_page_range(self, pdf_path: Path, start_page: int, end_page: int) -> Dict[str, Any]:
        """Extract a range of pages using best available method"""

        # Try extractors in order of preference
        # PyMuPDF with OCR is now preferred for its speed and OCR fallback
        extractors = [
            ('pymupdf', self._extract_with_pymupdf),  # Fast with OCR fallback
            ('pdfplumber', self._extract_with_pdfplumber),  # Good for tables
            # Slow but comprehensive
            ('unstructured', self._extract_with_unstructured),
            ('pypdf2', self._extract_with_pypdf2)  # Last resort
        ]

        for extractor_name, extractor_func in extractors:
            if not self.available_extractors.get(extractor_name, False):
                continue

            try:
                result = extractor_func(pdf_path, start_page, end_page)
                if result and result.get('pages'):
                    logger.debug(
                        f"Successfully extracted with {extractor_name}")
                    return result
            except Exception as e:
                logger.warning(
                    f"{extractor_name} failed for pages {start_page}-{end_page}: {e}")
                continue

        logger.error(
            f"All extractors failed for pages {start_page}-{end_page}")
        return {"pages": [], "tables": [], "extractor": "failed"}

    def _extract_with_pdfplumber(self, pdf_path: Path, start_page: int, end_page: int) -> Dict[str, Any]:
        """Extract using pdfplumber (best for tables and complex layouts)"""
        if not pdfplumber:
            raise ImportError("pdfplumber not available")

        pages = []
        tables = []

        with pdfplumber.open(pdf_path) as pdf:
            for page_idx in range(start_page, end_page):
                if page_idx >= len(pdf.pages):
                    break

                page = pdf.pages[page_idx]

                # Extract text
                page_text = page.extract_text() or ""
                if page_text:
                    normalized = self._normalize_text(page_text)
                    if len(normalized) >= MIN_CHARS:
                        pages.append({
                            "page": page_idx + 1,
                            "text": f"Page {page_idx + 1}:\n{normalized}",
                            "extractor": "pdfplumber"
                        })

                # Extract tables
                page_tables = page.extract_tables()
                for table_idx, table in enumerate(page_tables or []):
                    if table and len(table) > 1:  # Has header + data
                        table_text = self._format_table_as_markdown(table)
                        if table_text:
                            tables.append({
                                "page": page_idx + 1,
                                "table_id": table_idx,
                                "content": table_text
                            })

        return {"pages": pages, "tables": tables, "extractor": "pdfplumber"}

    def _extract_with_unstructured(self, pdf_path: Path, start_page: int, end_page: int) -> Dict[str, Any]:
        """Extract using unstructured (semantic document understanding)"""
        if not partition_pdf:
            raise ImportError("unstructured not available")

        pages = []

        try:
            # Extract elements for the page range
            elements = partition_pdf(
                filename=str(pdf_path),
                strategy="hi_res",  # High resolution for better extraction
                infer_table_structure=True,
                extract_images_in_pdf=False  # Skip images for text focus
            )

            # Group elements by page
            page_elements = {}
            for element in elements:
                page_num = getattr(element.metadata, 'page_number', 1)
                if start_page < page_num <= end_page:
                    if page_num not in page_elements:
                        page_elements[page_num] = []
                    page_elements[page_num].append(element)

            # Convert elements to text per page
            for page_num in sorted(page_elements.keys()):
                page_text = "\n".join([str(elem)
                                      for elem in page_elements[page_num]])
                if page_text:
                    normalized = self._normalize_text(page_text)
                    if len(normalized) >= MIN_CHARS:
                        pages.append({
                            "page": page_num,
                            "text": f"Page {page_num}:\n{normalized}",
                            "extractor": "unstructured"
                        })

        except Exception as e:
            logger.warning(f"Unstructured extraction failed: {e}")
            raise

        return {"pages": pages, "tables": [], "extractor": "unstructured"}

    def _extract_single_page(self, pdf_path: Path, page_idx: int) -> Optional[Dict[str, Any]]:
        """Extract a single page (for parallel processing)"""
        try:
            doc = fitz.open(str(pdf_path))
            page = doc[page_idx]
            page_text = page.get_text()
            doc.close()

            # Normalize text and check if we have enough content
            normalized = self._normalize_text(page_text) if page_text else ""
            extractor_used = "pymupdf"

            # If regular extraction yields minimal content, try OCR
            if len(normalized) < 50 and OCR_AVAILABLE:
                ocr_text = self._extract_page_with_ocr(pdf_path, page_idx)

                if ocr_text:
                    normalized = self._normalize_text(ocr_text)
                    extractor_used = "pymupdf+ocr"

            # Return page data if we have enough content
            if len(normalized) >= MIN_CHARS:
                return {
                    "page": page_idx + 1,
                    "text": f"Page {page_idx + 1}:\n{normalized}",
                    "extractor": extractor_used
                }
            return None

        except Exception as e:
            logger.error(f"Error extracting page {page_idx + 1}: {e}")
            return None

    def _extract_with_pymupdf(self, pdf_path: Path, start_page: int, end_page: int) -> Dict[str, Any]:
        """Extract using PyMuPDF (fast and reliable) with parallel OCR fallback"""
        if not fitz:
            raise ImportError("PyMuPDF not available")

        pages = []
        ocr_used_count = 0

        # Get page range
        doc = fitz.open(str(pdf_path))
        actual_end = min(end_page, doc.page_count)
        doc.close()

        page_indices = list(range(start_page, actual_end))

        # Use ThreadPoolExecutor for parallel OCR (4 workers for safety)
        # ThreadPool is better than ProcessPool for I/O-bound OCR operations
        max_workers = min(12, len(page_indices))

        if max_workers > 1:
            logger.info(
                f"Processing {len(page_indices)} pages with {max_workers} parallel workers")

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all pages for processing
                future_to_idx = {
                    executor.submit(self._extract_single_page, pdf_path, page_idx): page_idx
                    for page_idx in page_indices
                }

                # Collect results as they complete
                for future in as_completed(future_to_idx):
                    page_idx = future_to_idx[future]
                    try:
                        result = future.result()
                        if result:
                            pages.append(result)
                            if 'ocr' in result['extractor']:
                                ocr_used_count += 1
                    except Exception as e:
                        logger.error(f"Page {page_idx + 1} failed: {e}")

            # Sort pages by page number
            pages.sort(key=lambda x: x['page'])
        else:
            # Sequential processing for small chunks
            for page_idx in page_indices:
                result = self._extract_single_page(pdf_path, page_idx)
                if result:
                    pages.append(result)
                    if 'ocr' in result['extractor']:
                        ocr_used_count += 1

        if ocr_used_count > 0:
            logger.info(
                f"OCR used for {ocr_used_count}/{len(page_indices)} pages in this chunk")

        return {"pages": pages, "tables": [], "extractor": "pymupdf"}

    def _extract_with_pypdf2(self, pdf_path: Path, start_page: int, end_page: int) -> Dict[str, Any]:
        """Extract using PyPDF2 (last resort)"""
        if not PyPDF2:
            raise ImportError("PyPDF2 not available")

        pages = []

        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)

            # Handle password-protected PDFs
            if reader.is_encrypted:
                common_passwords = ["", "password", "123456", "admin"]
                for password in common_passwords:
                    try:
                        if reader.decrypt(password):
                            break
                    except:
                        continue
                else:
                    logger.warning(f"Could not decrypt {pdf_path.name}")
                    raise ValueError("Encrypted PDF cannot be decrypted")

            for page_idx in range(start_page, min(end_page, len(reader.pages))):
                page = reader.pages[page_idx]
                page_text = page.extract_text()

                if page_text:
                    normalized = self._normalize_text(page_text)
                    if len(normalized) >= MIN_CHARS:
                        pages.append({
                            "page": page_idx + 1,
                            "text": f"Page {page_idx + 1}:\n{normalized}",
                            "extractor": "pypdf2"
                        })

        return {"pages": pages, "tables": [], "extractor": "pypdf2"}

    def _extract_page_with_ocr(self, pdf_path: Path, page_idx: int) -> Optional[str]:
        """Extract single page using OCR (for image-based PDFs)"""
        if not OCR_AVAILABLE:
            return None

        try:
            # Convert page to image
            images = convert_from_path(
                str(pdf_path),
                first_page=page_idx + 1,
                last_page=page_idx + 1,
                dpi=300
            )

            if not images:
                return None

            # OCR with Spanish + English language support
            text = pytesseract.image_to_string(
                images[0],
                lang='spa+eng',
                config='--psm 3'  # Fully automatic page segmentation
            )

            # Clean up image
            images[0].close()

            return text if text.strip() else None

        except Exception as e:
            logger.debug(f"OCR failed for page {page_idx + 1}: {e}")
            return None

    def _normalize_text(self, text: str) -> str:
        """Comprehensive text normalization with watermark removal"""
        if not text:
            return ""

        # Remove watermarks first
        text = self._remove_watermarks(text)

        # Remove null bytes and control characters
        text = text.replace("\x00", " ")
        text = text.replace("\ufffd", "")  # Replacement character
        text = text.replace("\u00ad", "")  # Soft hyphens

        # Merge hyphenated words across line breaks
        text = re.sub(r"-\s*\n\s*", "", text)

        # Normalize whitespace
        # Multiple spaces/tabs → single space
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\s+\n", "\n", text)  # Trailing spaces before newlines
        # Multiple newlines → double newline
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()

    def _remove_watermarks(self, text: str) -> str:
        """Aggressively remove watermarks and PDF corruption artifacts"""
        # Specific UTP watermark patterns
        watermarks = [
            r"UTP-FISC-2S2025",
            r"UTP[\s\-]*FISC[\s\-]*2S2025",
            r"Universidad\s+Tecnológica\s+de\s+Panamá",
            r"FISC[\s\-]*\d+S\d+",
            r"Facultad\s+de\s+Ingeniería\s+de\s+Sistemas\s+Computacionales",
            r"Maestría\s+en\s+Analítica\s+de\s+Datos",
            r"Sistemas\s+de\s+Bases\s+de\s+Datos\s+Avanzadas",
            r"GUIA\s+DE\s+CURSO.*SEMESTRE\s+2025",
        ]

        # Remove watermark patterns
        for pattern in watermarks:
            text = re.sub(pattern, "", text,
                          flags=re.IGNORECASE | re.MULTILINE)

        # Remove scattered letter patterns (OCR artifacts)
        corrupted_patterns = [
            r"\b[A-Z]\s+[A-Z]\s+[-–]\s+[A-Z]\s+\d+\s+[A-Z]\b",
            r"\b\d+\s+[A-Z]\s+\d+\s+[A-Z]\s+\d+\s+[A-Z]\b",
            r"\b[A-Z]\s+\d+\s+[A-Z]\s+\d+\s+[A-Z]\b",
            r"(?:[A-Z]\s+){3,}\d+",
            r"(?:\d+\s+){3,}[A-Z]",
        ]

        for pattern in corrupted_patterns:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE)

        # Remove lines with high corruption ratio
        lines = text.split('\n')
        clean_lines = []

        for line in lines:
            line = line.strip()
            if not line:
                clean_lines.append(line)
                continue

            # Count spaces vs characters
            space_ratio = line.count(' ') / len(line) if len(line) > 0 else 0

            # Count scattered characters (single chars with spaces)
            scattered_chars = len(re.findall(r'\b[A-Za-z]\b', line))
            scattered_ratio = scattered_chars / \
                len(line.split()) if line.split() else 0

            # Skip lines that look like corrupted watermarks
            if space_ratio > 0.6 or scattered_ratio > 0.5:
                continue

            # Skip very short lines with mostly punctuation
            if len(line) < 10 and len(re.sub(r'[^A-Za-z0-9]', '', line)) < 3:
                continue

            clean_lines.append(line)

        return '\n'.join(clean_lines)

    def _format_table_as_markdown(self, table: List[List[str]]) -> str:
        """Format table as markdown text"""
        if not table or len(table) < 2:
            return ""

        table_text = "\n=== TABLA ===\n"

        for row_idx, row in enumerate(table):
            if row and any(cell for cell in row if cell):  # Non-empty row
                clean_row = []
                for cell in row:
                    if cell:
                        clean_cell = self._normalize_text(str(cell))
                        if clean_cell:
                            clean_row.append(clean_cell)

                if clean_row:
                    if row_idx == 0:  # Header
                        table_text += " | ".join(clean_row) + "\n"
                        table_text += "-" * (len(" | ".join(clean_row))) + "\n"
                    else:  # Data
                        table_text += " | ".join(clean_row) + "\n"

        table_text += "=== FIN TABLA ===\n"
        return table_text

    def extract_tables_from_pdf(self, pdf_path: Path) -> List[str]:
        """Extract all tables from PDF as formatted text"""
        tables_text = []

        if not pdfplumber:
            logger.warning(
                "pdfplumber not available - table extraction disabled")
            return tables_text

        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    page_tables = page.extract_tables()

                    for table_idx, table in enumerate(page_tables or []):
                        if table and len(table) > 1:
                            table_text = f"\n=== TABLA {table_idx + 1} (Página {page_num}) ===\n"
                            table_text += self._format_table_as_markdown(table)
                            if table_text.strip():
                                tables_text.append(table_text)

        except Exception as e:
            logger.error(f"Table extraction failed for {pdf_path.name}: {e}")

        return tables_text


def process_all_pdfs():
    """Process all PDFs in the raw directory"""
    processor = UnifiedPDFProcessor()

    raw_path = Path(RAW_DIR)
    if not raw_path.exists():
        logger.error(f"Raw directory not found: {RAW_DIR}")
        return

    pdf_files = list(raw_path.glob("*.pdf"))
    if not pdf_files:
        logger.warning(f"No PDF files found in {RAW_DIR}")
        return

    logger.info(f"Processing {len(pdf_files)} PDF files...")

    for pdf_file in pdf_files:
        try:
            logger.info(f"Processing: {pdf_file.name}")
            output_path = processor.process_pdf_file(
                pdf_file, output_format="jsonl")
            if output_path:
                logger.info(f"✅ Completed: {output_path.name}")
            else:
                logger.error(f"❌ Failed: {pdf_file.name}")
        except Exception as e:
            logger.error(f"❌ Error processing {pdf_file.name}: {e}")


def process_text_files():
    """Process text and markdown files"""
    raw_path = Path(RAW_DIR)
    if not raw_path.exists():
        return

    text_files = list(raw_path.glob("*.txt")) + list(raw_path.glob("*.md"))

    for text_file in text_files:
        try:
            logger.info(f"Processing text file: {text_file.name}")

            with open(text_file, 'r', encoding='utf-8') as f:
                content = f.read()

            if not content.strip():
                continue

            # Create JSONL output
            output_path = Path(CLEAN_DIR) / (text_file.stem + ".jsonl")

            with open(output_path, 'w', encoding='utf-8') as output_file:
                record = {
                    "source_path": str(text_file),
                    "page": 1,
                    "text": content,
                    "extractor": "text_file"
                }
                output_file.write(json.dumps(
                    record, ensure_ascii=False) + '\n')

            logger.info(f"✅ Completed: {output_path.name}")

        except Exception as e:
            logger.error(f"❌ Error processing {text_file.name}: {e}")


def clean_all_files():
    """Process all files (PDFs and text files) in the raw directory"""
    logger.info("🧹 Starting unified file processing...")

    # Process PDFs
    process_all_pdfs()

    # Process text files
    process_text_files()

    logger.info("✅ Unified file processing complete")


if __name__ == "__main__":
    # Configure logging for direct execution
    logging.basicConfig(level=logging.INFO,
                        format='%(levelname)s: %(message)s')

    # Process all files
    clean_all_files()
