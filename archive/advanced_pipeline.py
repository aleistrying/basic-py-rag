#!/usr/bin/env python3
"""
Enhanced PDF Processing Pipeline with Advanced Extraction
Integrates multiple PDF extraction libraries for optimal text and table extraction.
"""

import os
import sys
import re
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.query_embed import EmbeddingService
from app.settings import get_settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedPDFExtractor:
    """Enhanced PDF extractor using multiple libraries for best results"""
    
    def __init__(self):
        self.available_extractors = self._check_available_extractors()
        logger.info(f"Available extractors: {list(self.available_extractors.keys())}")
    
    def _check_available_extractors(self) -> Dict[str, bool]:
        """Check which extraction libraries are available"""
        extractors = {}
        
        # PyPDF2
        try:
            import PyPDF2
            extractors['pypdf2'] = True
        except ImportError:
            extractors['pypdf2'] = False
        
        # PyMuPDF
        try:
            import fitz
            extractors['pymupdf'] = True
        except ImportError:
            extractors['pymupdf'] = False
        
        # pdfplumber
        try:
            import pdfplumber
            extractors['pdfplumber'] = True
        except ImportError:
            extractors['pdfplumber'] = False
        
        # Camelot
        try:
            import camelot
            extractors['camelot'] = True
        except ImportError:
            extractors['camelot'] = False
        
        # pymupdf4llm
        try:
            import pymupdf4llm
            extractors['pymupdf4llm'] = True
        except ImportError:
            extractors['pymupdf4llm'] = False
        
        # unstructured
        try:
            from unstructured.partition.pdf import partition_pdf
            extractors['unstructured'] = True
        except ImportError:
            extractors['unstructured'] = False
        
        return extractors
    
    def extract_with_fallback(self, pdf_path: str, strategy: str = "best_quality") -> Dict[str, Any]:
        """
        Extract text and tables using the best available method with fallbacks
        
        Strategies:
        - "fastest": Prioritize speed
        - "best_quality": Prioritize text quality and structure
        - "tables": Prioritize table extraction
        """
        
        if strategy == "fastest":
            extraction_order = ['pymupdf', 'pypdf2', 'pdfplumber', 'unstructured']
        elif strategy == "tables":
            extraction_order = ['camelot', 'pdfplumber', 'pymupdf4llm', 'pymupdf']
        else:  # best_quality
            extraction_order = ['pymupdf4llm', 'unstructured', 'pymupdf', 'pdfplumber', 'pypdf2']
        
        results = {
            "success": False,
            "method_used": None,
            "text": "",
            "tables": [],
            "metadata": {},
            "extraction_time": 0,
            "fallback_attempts": []
        }
        
        for method in extraction_order:
            if not self.available_extractors.get(method, False):
                results["fallback_attempts"].append(f"{method}: not available")
                continue
            
            try:
                logger.info(f"Trying extraction with {method}")
                start_time = time.time()
                
                if method == "pymupdf4llm":
                    result = self._extract_pymupdf4llm(pdf_path)
                elif method == "unstructured":
                    result = self._extract_unstructured(pdf_path)
                elif method == "pymupdf":
                    result = self._extract_pymupdf(pdf_path)
                elif method == "pdfplumber":
                    result = self._extract_pdfplumber(pdf_path)
                elif method == "camelot":
                    result = self._extract_camelot(pdf_path)
                elif method == "pypdf2":
                    result = self._extract_pypdf2(pdf_path)
                else:
                    continue
                
                extraction_time = time.time() - start_time
                
                if result and result.get("text") and len(result["text"].strip()) > 50:
                    results.update({
                        "success": True,
                        "method_used": method,
                        "text": result["text"],
                        "tables": result.get("tables", []),
                        "metadata": result.get("metadata", {}),
                        "extraction_time": extraction_time
                    })
                    logger.info(f"✅ Successfully extracted {len(result['text'])} chars with {method}")
                    break
                else:
                    results["fallback_attempts"].append(f"{method}: insufficient text extracted")
                    
            except Exception as e:
                logger.warning(f"❌ {method} failed: {e}")
                results["fallback_attempts"].append(f"{method}: {str(e)}")
        
        return results
    
    def _extract_pymupdf4llm(self, pdf_path: str) -> Dict[str, Any]:
        """Extract using pymupdf4llm for best markdown structure"""
        import pymupdf4llm
        
        markdown_text = pymupdf4llm.to_markdown(pdf_path)
        
        # Extract tables from markdown
        tables = self._extract_tables_from_markdown(markdown_text)
        
        return {
            "text": markdown_text,
            "tables": tables,
            "metadata": {
                "format": "markdown",
                "preserves_structure": True,
                "tables_as_markdown": True
            }
        }
    
    def _extract_unstructured(self, pdf_path: str) -> Dict[str, Any]:
        """Extract using unstructured for semantic parsing"""
        from unstructured.partition.pdf import partition_pdf
        
        elements = partition_pdf(filename=pdf_path)
        
        # Combine text elements
        text_parts = []
        tables = []
        
        for element in elements:
            if hasattr(element, 'text'):
                # Add element type as context
                element_type = str(type(element).__name__)
                if element_type == "Title":
                    text_parts.append(f"# {element.text}")
                elif element_type == "Header":
                    text_parts.append(f"## {element.text}")
                elif "Table" in element_type:
                    tables.append(element.text)
                    text_parts.append(f"**Table:** {element.text}")
                else:
                    text_parts.append(element.text)
        
        return {
            "text": "\n\n".join(text_parts),
            "tables": tables,
            "metadata": {
                "format": "semantic",
                "elements_processed": len(elements),
                "preserves_structure": True
            }
        }
    
    def _extract_pymupdf(self, pdf_path: str) -> Dict[str, Any]:
        """Extract using PyMuPDF for robust text extraction"""
        import fitz
        
        doc = fitz.open(pdf_path)
        text_parts = []
        tables = []
        
        for page in doc:
            page_text = page.get_text()
            text_parts.append(page_text)
            
            # Look for table-like structures
            blocks = page.get_text("blocks")
            for block in blocks:
                if len(block) >= 5:
                    block_text = block[4]
                    if self._looks_like_table(block_text):
                        tables.append(block_text)
        
        doc.close()
        
        return {
            "text": "\n".join(text_parts),
            "tables": tables,
            "metadata": {
                "format": "plain_text",
                "layout_preserved": False
            }
        }
    
    def _extract_pdfplumber(self, pdf_path: str) -> Dict[str, Any]:
        """Extract using pdfplumber for table extraction"""
        import pdfplumber
        
        text_parts = []
        tables = []
        
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                # Extract text
                page_text = page.extract_text() or ""
                text_parts.append(page_text)
                
                # Extract tables
                page_tables = page.extract_tables()
                for table in page_tables:
                    if table:
                        # Convert table to text format
                        table_text = self._table_to_text(table)
                        tables.append(table_text)
        
        return {
            "text": "\n".join(text_parts),
            "tables": tables,
            "metadata": {
                "format": "plain_text",
                "table_extraction": True
            }
        }
    
    def _extract_camelot(self, pdf_path: str) -> Dict[str, Any]:
        """Extract using Camelot specifically for tables"""
        import camelot
        
        tables_data = camelot.read_pdf(pdf_path)
        tables = []
        
        for table in tables_data:
            df = table.df
            if not df.empty:
                # Convert DataFrame to readable text
                table_text = df.to_string(index=False)
                tables.append(table_text)
        
        # For text, fall back to PyMuPDF if available
        text = ""
        if self.available_extractors.get('pymupdf'):
            pymupdf_result = self._extract_pymupdf(pdf_path)
            text = pymupdf_result.get("text", "")
        
        return {
            "text": text,
            "tables": tables,
            "metadata": {
                "format": "camelot_tables",
                "specialized_table_extraction": True
            }
        }
    
    def _extract_pypdf2(self, pdf_path: str) -> Dict[str, Any]:
        """Extract using PyPDF2 as fallback"""
        import PyPDF2
        
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text_parts = []
            
            for page in reader.pages:
                page_text = page.extract_text()
                text_parts.append(page_text)
        
        return {
            "text": "\n".join(text_parts),
            "tables": [],
            "metadata": {
                "format": "basic_text",
                "fallback_method": True
            }
        }
    
    def _extract_tables_from_markdown(self, markdown_text: str) -> List[str]:
        """Extract tables from markdown text"""
        tables = []
        lines = markdown_text.split('\n')
        
        current_table = []
        in_table = False
        
        for line in lines:
            if '|' in line and line.strip().startswith('|'):
                current_table.append(line)
                in_table = True
            elif in_table and line.strip() == '':
                if current_table:
                    tables.append('\n'.join(current_table))
                    current_table = []
                in_table = False
            elif in_table and '|' not in line:
                if current_table:
                    tables.append('\n'.join(current_table))
                    current_table = []
                in_table = False
        
        # Add last table if exists
        if current_table:
            tables.append('\n'.join(current_table))
        
        return tables
    
    def _table_to_text(self, table: List[List[str]]) -> str:
        """Convert table array to formatted text"""
        if not table:
            return ""
        
        # Find max width for each column
        max_widths = []
        for row in table:
            for i, cell in enumerate(row):
                cell_str = str(cell) if cell else ""
                if i >= len(max_widths):
                    max_widths.append(len(cell_str))
                else:
                    max_widths[i] = max(max_widths[i], len(cell_str))
        
        # Format table
        formatted_rows = []
        for row in table:
            formatted_cells = []
            for i, cell in enumerate(row):
                cell_str = str(cell) if cell else ""
                width = max_widths[i] if i < len(max_widths) else 10
                formatted_cells.append(cell_str.ljust(width))
            formatted_rows.append(" | ".join(formatted_cells))
        
        return "\n".join(formatted_rows)
    
    def _looks_like_table(self, text: str) -> bool:
        """Heuristic to detect table-like text"""
        if len(text) < 50:
            return False
        
        lines = text.strip().split('\n')
        if len(lines) < 2:
            return False
        
        # Check for consistent separators
        separators = ['|', '\t', '  ']
        for sep in separators:
            if sum(1 for line in lines if sep in line) >= len(lines) * 0.7:
                return True
        
        # Check for number patterns
        number_lines = sum(1 for line in lines if any(c.isdigit() for c in line))
        return number_lines / len(lines) > 0.6


class EnhancedPipelineProcessor:
    """Enhanced pipeline with advanced PDF extraction"""
    
    def __init__(self):
        self.settings = get_settings()
        self.extractor = AdvancedPDFExtractor()
        self.embedding_service = EmbeddingService()
        
    def process_pdf(self, pdf_path: str, strategy: str = "best_quality") -> Dict[str, Any]:
        """Process a PDF with enhanced extraction and upload to vector databases"""
        
        logger.info(f"🔄 Processing PDF: {pdf_path}")
        
        # Extract text and tables
        extraction_result = self.extractor.extract_with_fallback(pdf_path, strategy)
        
        if not extraction_result["success"]:
            logger.error(f"❌ Failed to extract text from {pdf_path}")
            return {"success": False, "error": "Text extraction failed"}
        
        # Clean and normalize text
        clean_text = self.normalize_text(extraction_result["text"])
        
        # Create enhanced document with metadata
        document = {
            "source": os.path.basename(pdf_path),
            "text": clean_text,
            "tables": extraction_result["tables"],
            "extraction_method": extraction_result["method_used"],
            "extraction_time": extraction_result["extraction_time"],
            "metadata": extraction_result["metadata"],
            "processed_at": time.time()
        }
        
        # Chunk the text
        chunks = self.chunk_text(clean_text)
        
        # Upload to vector databases
        results = {
            "success": True,
            "document": document,
            "chunks_created": len(chunks),
            "extraction_method": extraction_result["method_used"],
            "tables_found": len(extraction_result["tables"]),
            "upload_results": {}
        }
        
        try:
            # Upload to databases (if configured)
            if hasattr(self.embedding_service, 'upsert_to_qdrant'):
                qdrant_result = self.embedding_service.upsert_to_qdrant(chunks, document["source"])
                results["upload_results"]["qdrant"] = qdrant_result
                
            if hasattr(self.embedding_service, 'upsert_to_postgres'):
                postgres_result = self.embedding_service.upsert_to_postgres(chunks, document["source"])
                results["upload_results"]["postgres"] = postgres_result
                
            logger.info(f"✅ Successfully processed {pdf_path}")
            logger.info(f"   Method: {extraction_result['method_used']}")
            logger.info(f"   Text: {len(clean_text)} chars")
            logger.info(f"   Tables: {len(extraction_result['tables'])}")
            logger.info(f"   Chunks: {len(chunks)}")
            
        except Exception as e:
            logger.error(f"❌ Error uploading to databases: {e}")
            results["upload_results"]["error"] = str(e)
        
        return results
    
    def normalize_text(self, text: str) -> str:
        """Enhanced text normalization with watermark removal"""
        
        # Remove common watermarks and artifacts
        watermark_patterns = [
            r'ProyeFc',  # Replace with 'Proyecto'
            r'InvestigacMn',  # Replace with 'Investigación'
            r'EvaluacMn',  # Replace with 'Evaluación'
            r'DesarMllo',  # Replace with 'Desarrollo'
            r'AnBlisis',  # Replace with 'Análisis'
            r'MaestLra',  # Replace with 'Maestría'
            r'UniversMdad',  # Replace with 'Universidad'
        ]
        
        # Apply watermark fixes
        text = re.sub(r'ProyeFc', 'Proyecto', text, flags=re.IGNORECASE)
        text = re.sub(r'InvestigacMn', 'Investigación', text, flags=re.IGNORECASE)
        text = re.sub(r'EvaluacMn', 'Evaluación', text, flags=re.IGNORECASE)
        text = re.sub(r'DesarMllo', 'Desarrollo', text, flags=re.IGNORECASE)
        text = re.sub(r'AnBlisis', 'Análisis', text, flags=re.IGNORECASE)
        text = re.sub(r'MaestLra', 'Maestría', text, flags=re.IGNORECASE)
        text = re.sub(r'UniversMdad', 'Universidad', text, flags=re.IGNORECASE)
        
        # Fix common OCR/extraction issues
        text = re.sub(r'\s+', ' ', text)  # Multiple spaces
        text = re.sub(r'\n\s*\n', '\n\n', text)  # Multiple newlines
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # Missing spaces
        
        # Fix date patterns
        text = re.sub(r'(\d{1,2})\s+(\d{1,2})', r'\1/\2', text)  # "10 29" -> "10/29"
        
        return text.strip()
    
    def chunk_text(self, text: str, chunk_size: int = 200, overlap: int = 50) -> List[str]:
        """Intelligent text chunking with schedule awareness"""
        
        # Split into sentences first
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence_tokens = len(sentence.split())
            
            # Check if this sentence contains schedule information
            is_schedule = any(pattern in sentence.lower() for pattern in [
                'lunes', 'martes', 'miércoles', 'jueves', 'viernes',
                'horario', 'clase', 'entrega', 'proyecto',
                'enero', 'febrero', 'marzo', 'abril', 'mayo', 'junio',
                'julio', 'agosto', 'septiembre', 'octubre', 'noviembre', 'diciembre'
            ])
            
            # If adding this sentence would exceed chunk size
            if current_size + sentence_tokens > chunk_size and current_chunk:
                # Don't split schedule information
                if is_schedule and current_size < chunk_size * 0.8:
                    current_chunk.append(sentence)
                    current_size += sentence_tokens
                else:
                    # Finalize current chunk
                    chunks.append(' '.join(current_chunk))
                    
                    # Start new chunk with overlap
                    if overlap > 0 and len(current_chunk) > 1:
                        overlap_sentences = current_chunk[-overlap//10:]  # Rough overlap
                        current_chunk = overlap_sentences + [sentence]
                        current_size = sum(len(s.split()) for s in current_chunk)
                    else:
                        current_chunk = [sentence]
                        current_size = sentence_tokens
            else:
                current_chunk.append(sentence)
                current_size += sentence_tokens
        
        # Add final chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks


def main():
    """Main function to test the enhanced pipeline"""
    
    # Find PDF files
    raw_dir = Path(__file__).parent.parent / "data" / "raw"
    pdf_files = list(raw_dir.glob("*.pdf"))
    
    if not pdf_files:
        logger.error(f"No PDF files found in {raw_dir}")
        return
    
    # Process each PDF
    processor = EnhancedPipelineProcessor()
    
    for pdf_file in pdf_files:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing: {pdf_file.name}")
        logger.info(f"{'='*60}")
        
        # Test different strategies
        strategies = ["best_quality", "fastest", "tables"]
        
        for strategy in strategies:
            logger.info(f"\n--- Testing strategy: {strategy} ---")
            result = processor.process_pdf(str(pdf_file), strategy=strategy)
            
            if result["success"]:
                logger.info(f"✅ Strategy '{strategy}' completed successfully")
                logger.info(f"   Method used: {result['extraction_method']}")
                logger.info(f"   Chunks created: {result['chunks_created']}")
                logger.info(f"   Tables found: {result['tables_found']}")
                
                # Save detailed results
                output_file = raw_dir.parent / "clean" / f"{pdf_file.stem}_{strategy}_result.json"
                output_file.parent.mkdir(exist_ok=True)
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False, default=str)
                
                logger.info(f"   Results saved to: {output_file}")
                break  # Use first successful strategy
            else:
                logger.warning(f"❌ Strategy '{strategy}' failed")


if __name__ == "__main__":
    main()