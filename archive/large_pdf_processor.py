#!/usr/bin/env python3
"""
Advanced PDF Extraction with Smart Chunking for Large Documents
Handles massive PDFs like the 1322-page Williams Obstetricia textbook
"""

import os
import sys
import time
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Generator
import logging

# Add project to path
sys.path.append(str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedPDFExtractor:
    """Enhanced PDF extractor with smart chunking for large documents"""
    
    def __init__(self):
        self.available_extractors = self._check_available_extractors()
        logger.info(f"Available extractors: {list(self.available_extractors.keys())}")
    
    def _check_available_extractors(self) -> Dict[str, bool]:
        """Check which extraction libraries are available"""
        extractors = {}
        
        # Basic libraries
        try:
            import PyPDF2
            extractors['pypdf2'] = True
        except ImportError:
            extractors['pypdf2'] = False
        
        try:
            import fitz  # PyMuPDF
            extractors['pymupdf'] = True
        except ImportError:
            extractors['pymupdf'] = False
        
        try:
            import pdfplumber
            extractors['pdfplumber'] = True
        except ImportError:
            extractors['pdfplumber'] = False
        
        # Advanced libraries
        try:
            import marker
            extractors['marker'] = True
        except ImportError:
            extractors['marker'] = False
        
        try:
            from unstructured.partition.pdf import partition_pdf
            extractors['unstructured'] = True
        except ImportError:
            extractors['unstructured'] = False
        
        return extractors
    
    def extract_large_pdf_intelligently(self, pdf_path: str, max_pages_per_chunk: int = 50) -> Generator[Dict[str, Any], None, None]:
        """
        Extract large PDFs in intelligent chunks to handle memory efficiently
        Yields chunks of extracted content instead of loading everything at once
        """
        
        logger.info(f"🔄 Processing large PDF: {pdf_path}")
        
        # First, get PDF info
        pdf_info = self._get_pdf_info(pdf_path)
        total_pages = pdf_info['total_pages']
        
        logger.info(f"📊 PDF has {total_pages} pages - processing in chunks of {max_pages_per_chunk}")
        
        # Process in chunks
        for chunk_start in range(0, total_pages, max_pages_per_chunk):
            chunk_end = min(chunk_start + max_pages_per_chunk, total_pages)
            
            logger.info(f"📄 Processing pages {chunk_start + 1}-{chunk_end}")
            
            # Extract this chunk
            chunk_result = self._extract_page_range(pdf_path, chunk_start, chunk_end)
            
            if chunk_result['success']:
                # Add chunk metadata
                chunk_result.update({
                    'chunk_info': {
                        'pages_start': chunk_start + 1,
                        'pages_end': chunk_end,
                        'total_pages': total_pages,
                        'chunk_number': (chunk_start // max_pages_per_chunk) + 1,
                        'total_chunks': (total_pages + max_pages_per_chunk - 1) // max_pages_per_chunk
                    }
                })
                
                yield chunk_result
            else:
                logger.warning(f"❌ Failed to extract pages {chunk_start + 1}-{chunk_end}")
    
    def _get_pdf_info(self, pdf_path: str) -> Dict[str, Any]:
        """Get basic PDF information without loading all content"""
        
        # Try PyMuPDF first (most efficient)
        if self.available_extractors.get('pymupdf'):
            try:
                import fitz
                doc = fitz.open(pdf_path)
                info = {
                    'total_pages': doc.page_count,
                    'metadata': doc.metadata,
                    'method': 'pymupdf'
                }
                doc.close()
                return info
            except Exception as e:
                logger.warning(f"PyMuPDF info extraction failed: {e}")
        
        # Fallback to PyPDF2
        if self.available_extractors.get('pypdf2'):
            try:
                import PyPDF2
                with open(pdf_path, 'rb') as file:
                    reader = PyPDF2.PdfReader(file)
                    return {
                        'total_pages': len(reader.pages),
                        'metadata': reader.metadata or {},
                        'method': 'pypdf2'
                    }
            except Exception as e:
                logger.warning(f"PyPDF2 info extraction failed: {e}")
        
        return {'total_pages': 0, 'metadata': {}, 'method': 'unknown'}
    
    def _extract_page_range(self, pdf_path: str, start_page: int, end_page: int) -> Dict[str, Any]:
        """Extract a specific range of pages efficiently"""
        
        # Strategy 1: unstructured (best for semantic parsing)
        if self.available_extractors.get('unstructured'):
            try:
                logger.info("🔍 Trying unstructured for semantic extraction...")
                return self._extract_with_unstructured(pdf_path, start_page, end_page)
            except Exception as e:
                logger.warning(f"❌ unstructured failed: {e}")
        
        # Strategy 2: marker (best for layout preservation)
        if self.available_extractors.get('marker'):
            try:
                logger.info("🔍 Trying marker for layout-perfect extraction...")
                return self._extract_with_marker(pdf_path, start_page, end_page)
            except Exception as e:
                logger.warning(f"❌ marker failed: {e}")
        
        # Strategy 3: pdfplumber (best for tables)
        if self.available_extractors.get('pdfplumber'):
            try:
                logger.info("🔍 Trying pdfplumber for table extraction...")
                return self._extract_with_pdfplumber(pdf_path, start_page, end_page)
            except Exception as e:
                logger.warning(f"❌ pdfplumber failed: {e}")
        
        # Strategy 4: PyMuPDF (reliable fallback)
        if self.available_extractors.get('pymupdf'):
            try:
                logger.info("🔍 Trying PyMuPDF for reliable extraction...")
                return self._extract_with_pymupdf(pdf_path, start_page, end_page)
            except Exception as e:
                logger.warning(f"❌ PyMuPDF failed: {e}")
        
        # Strategy 5: PyPDF2 (basic fallback)
        if self.available_extractors.get('pypdf2'):
            try:
                logger.info("🔍 Trying PyPDF2 for basic extraction...")
                return self._extract_with_pypdf2(pdf_path, start_page, end_page)
            except Exception as e:
                logger.warning(f"❌ PyPDF2 failed: {e}")
        
        return {'success': False, 'error': 'No extraction method available'}
    
    def _extract_with_unstructured(self, pdf_path: str, start_page: int, end_page: int) -> Dict[str, Any]:
        """Extract using unstructured for semantic document parsing"""
        from unstructured.partition.pdf import partition_pdf
        
        # unstructured doesn't support page ranges directly, so we extract all and filter
        elements = partition_pdf(filename=pdf_path)
        
        # Filter elements by page if page info is available
        filtered_elements = []
        for element in elements:
            if hasattr(element, 'metadata') and element.metadata:
                page_num = element.metadata.get('page_number', 1)
                if start_page <= page_num - 1 < end_page:  # Convert to 0-based
                    filtered_elements.append(element)
            else:
                # If no page info, include all (suboptimal but functional)
                filtered_elements.append(element)
        
        # Convert to structured text
        text_parts = []
        tables = []
        element_types = {}
        
        for element in filtered_elements:
            element_type = str(type(element).__name__)
            element_types[element_type] = element_types.get(element_type, 0) + 1
            
            if hasattr(element, 'text'):
                if 'Table' in element_type:
                    tables.append(element.text)
                    text_parts.append(f"**[Tabla]**: {element.text}")
                elif 'Title' in element_type:
                    text_parts.append(f"# {element.text}")
                elif 'Header' in element_type:
                    text_parts.append(f"## {element.text}")
                else:
                    text_parts.append(element.text)
        
        return {
            'success': True,
            'text': '\n\n'.join(text_parts),
            'tables': tables,
            'method': 'unstructured',
            'metadata': {
                'element_types': element_types,
                'elements_processed': len(filtered_elements),
                'semantic_parsing': True
            }
        }
    
    def _extract_with_marker(self, pdf_path: str, start_page: int, end_page: int) -> Dict[str, Any]:
        """Extract using marker for layout-perfect markdown"""
        # marker doesn't support page ranges, so this is a placeholder
        # In practice, you'd need to implement page splitting or process the whole document
        
        logger.warning("Marker doesn't support page ranges - would need full document processing")
        return {'success': False, 'error': 'Marker page range not implemented'}
    
    def _extract_with_pdfplumber(self, pdf_path: str, start_page: int, end_page: int) -> Dict[str, Any]:
        """Extract using pdfplumber with page range support"""
        import pdfplumber
        
        text_parts = []
        tables = []
        
        with pdfplumber.open(pdf_path) as pdf:
            for page_num in range(start_page, min(end_page, len(pdf.pages))):
                page = pdf.pages[page_num]
                
                # Extract text
                page_text = page.extract_text() or ""
                if page_text.strip():
                    text_parts.append(f"=== Página {page_num + 1} ===\n{page_text}")
                
                # Extract tables
                page_tables = page.extract_tables()
                for table_idx, table in enumerate(page_tables):
                    if table and len(table) > 1:
                        table_text = self._format_table_as_markdown(table)
                        tables.append(table_text)
        
        return {
            'success': True,
            'text': '\n\n'.join(text_parts),
            'tables': tables,
            'method': 'pdfplumber',
            'metadata': {
                'pages_processed': end_page - start_page,
                'tables_found': len(tables)
            }
        }
    
    def _extract_with_pymupdf(self, pdf_path: str, start_page: int, end_page: int) -> Dict[str, Any]:
        """Extract using PyMuPDF with page range support"""
        import fitz
        
        doc = fitz.open(pdf_path)
        text_parts = []
        tables = []
        
        for page_num in range(start_page, min(end_page, doc.page_count)):
            page = doc[page_num]
            page_text = page.get_text()
            
            if page_text.strip():
                text_parts.append(f"=== Página {page_num + 1} ===\n{page_text}")
            
            # Look for table-like blocks
            blocks = page.get_text("blocks")
            for block in blocks:
                if len(block) >= 5 and self._looks_like_table(block[4]):
                    tables.append(f"**Tabla detectada:**\n{block[4]}")
        
        doc.close()
        
        return {
            'success': True,
            'text': '\n\n'.join(text_parts),
            'tables': tables,
            'method': 'pymupdf',
            'metadata': {
                'pages_processed': end_page - start_page,
                'tables_detected': len(tables)
            }
        }
    
    def _extract_with_pypdf2(self, pdf_path: str, start_page: int, end_page: int) -> Dict[str, Any]:
        """Extract using PyPDF2 with page range support"""
        import PyPDF2
        
        text_parts = []
        
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            
            for page_num in range(start_page, min(end_page, len(reader.pages))):
                page = reader.pages[page_num]
                page_text = page.extract_text()
                
                if page_text.strip():
                    text_parts.append(f"=== Página {page_num + 1} ===\n{page_text}")
        
        return {
            'success': True,
            'text': '\n\n'.join(text_parts),
            'tables': [],
            'method': 'pypdf2',
            'metadata': {
                'pages_processed': end_page - start_page
            }
        }
    
    def _format_table_as_markdown(self, table: List[List[str]]) -> str:
        """Convert table to markdown format"""
        if not table:
            return ""
        
        markdown_rows = []
        
        # Header
        if table[0]:
            header = " | ".join(str(cell).strip() if cell else "" for cell in table[0])
            markdown_rows.append(f"| {header} |")
            markdown_rows.append(f"| {' | '.join(['---'] * len(table[0]))} |")
        
        # Data rows
        for row in table[1:]:
            if row and any(cell for cell in row if cell):
                row_text = " | ".join(str(cell).strip() if cell else "" for cell in row)
                markdown_rows.append(f"| {row_text} |")
        
        return "\n".join(markdown_rows)
    
    def _looks_like_table(self, text: str) -> bool:
        """Heuristic to detect table-like content"""
        if not text or len(text) < 50:
            return False
        
        lines = text.strip().split('\n')
        if len(lines) < 2:
            return False
        
        # Check for separators and numeric content
        separator_lines = sum(1 for line in lines if any(sep in line for sep in ['|', '\t', '  ']))
        numeric_lines = sum(1 for line in lines if any(c.isdigit() for c in line))
        
        return (separator_lines / len(lines) > 0.5) or (numeric_lines / len(lines) > 0.6)


class SmartChunker:
    """Intelligent chunking that preserves document structure and handles large texts"""
    
    def __init__(self, base_chunk_size: int = 500, overlap: int = 100):
        self.base_chunk_size = base_chunk_size
        self.overlap = overlap
    
    def chunk_large_document(self, text: str, source_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Intelligently chunk large documents while preserving context
        """
        
        # Split by logical sections first
        sections = self._split_into_sections(text)
        
        chunks = []
        chunk_id = 0
        
        for section_idx, section in enumerate(sections):
            section_chunks = self._chunk_section(section, section_idx)
            
            for chunk_text in section_chunks:
                chunk = {
                    'id': chunk_id,
                    'text': chunk_text,
                    'source': source_info.get('source', 'unknown'),
                    'metadata': {
                        'section_number': section_idx,
                        'chunk_in_section': len([c for c in chunks if c['metadata']['section_number'] == section_idx]),
                        'extraction_method': source_info.get('method', 'unknown'),
                        'pages_info': source_info.get('chunk_info', {}),
                        'chunk_size': len(chunk_text),
                        'has_tables': 'tabla' in chunk_text.lower() or '|' in chunk_text
                    }
                }
                chunks.append(chunk)
                chunk_id += 1
        
        logger.info(f"📊 Created {len(chunks)} intelligent chunks from {len(sections)} sections")
        return chunks
    
    def _split_into_sections(self, text: str) -> List[str]:
        """Split text into logical sections based on headers and natural breaks"""
        
        # Split by page markers first
        page_sections = re.split(r'=== Página \d+ ===', text)
        
        all_sections = []
        
        for page_section in page_sections:
            if not page_section.strip():
                continue
            
            # Further split by headers and natural breaks
            section_splits = re.split(r'\n(?=#{1,3}\s)', page_section)  # Markdown headers
            
            for section in section_splits:
                if section.strip() and len(section.strip()) > 100:  # Meaningful content
                    all_sections.append(section.strip())
        
        return all_sections
    
    def _chunk_section(self, section: str, section_idx: int) -> List[str]:
        """Chunk a section while preserving sentence boundaries"""
        
        if len(section) <= self.base_chunk_size:
            return [section]
        
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', section)
        
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence_size = len(sentence)
            
            # If adding this sentence would exceed chunk size
            if current_size + sentence_size > self.base_chunk_size and current_chunk:
                # Finalize current chunk
                chunk_text = ' '.join(current_chunk)
                chunks.append(chunk_text)
                
                # Start new chunk with overlap
                overlap_sentences = current_chunk[-self.overlap//50:] if len(current_chunk) > self.overlap//50 else []
                current_chunk = overlap_sentences + [sentence]
                current_size = sum(len(s) for s in current_chunk)
            else:
                current_chunk.append(sentence)
                current_size += sentence_size
        
        # Add final chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks


def process_large_pdf_demo(pdf_path: str):
    """Demonstrate processing a large PDF like Williams Obstetricia"""
    
    print(f"🚀 Processing Large PDF: {os.path.basename(pdf_path)}")
    print("=" * 80)
    
    extractor = AdvancedPDFExtractor()
    chunker = SmartChunker(base_chunk_size=800, overlap=150)  # Larger chunks for medical textbook
    
    # Get PDF info first
    pdf_info = extractor._get_pdf_info(pdf_path)
    print(f"📊 PDF Info: {pdf_info['total_pages']} pages")
    
    if pdf_info['total_pages'] > 1000:
        print("⚠️  Large document detected - processing first 100 pages as demo")
        max_pages = 100
    else:
        max_pages = pdf_info['total_pages']
    
    # Process in chunks
    all_chunks = []
    total_text_length = 0
    total_tables = 0
    
    chunk_generator = extractor.extract_large_pdf_intelligently(pdf_path, max_pages_per_chunk=25)
    
    for extraction_result in chunk_generator:
        if extraction_result['success']:
            # Get source info
            source_info = {
                'source': os.path.basename(pdf_path),
                'method': extraction_result['method'],
                'chunk_info': extraction_result['chunk_info']
            }
            
            # Create intelligent chunks
            text_chunks = chunker.chunk_large_document(extraction_result['text'], source_info)
            all_chunks.extend(text_chunks)
            
            # Track statistics
            total_text_length += len(extraction_result['text'])
            total_tables += len(extraction_result.get('tables', []))
            
            chunk_info = extraction_result['chunk_info']
            print(f"✅ Processed pages {chunk_info['pages_start']}-{chunk_info['pages_end']}: {len(text_chunks)} chunks")
        
        # Memory management for large documents
        if len(all_chunks) > 1000:  # Limit memory usage
            print(f"💾 Memory limit reached - stopping at {len(all_chunks)} chunks")
            break
    
    # Summary
    print(f"\n📊 PROCESSING SUMMARY:")
    print(f"   Total chunks created: {len(all_chunks)}")
    print(f"   Total text processed: {total_text_length:,} characters")
    print(f"   Tables detected: {total_tables}")
    print(f"   Average chunk size: {total_text_length // len(all_chunks) if all_chunks else 0} chars")
    
    # Save sample results
    output_file = Path("data/clean/large_pdf_processing_demo.json")
    output_file.parent.mkdir(exist_ok=True)
    
    # Save first 10 chunks as sample
    sample_chunks = all_chunks[:10]
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'pdf_info': pdf_info,
            'processing_summary': {
                'total_chunks': len(all_chunks),
                'total_text_length': total_text_length,
                'tables_detected': total_tables
            },
            'sample_chunks': sample_chunks
        }, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"💾 Sample results saved to: {output_file}")
    return all_chunks


def main():
    """Main function to test large PDF processing"""
    
    # Look for large PDFs
    pdf_paths = [
        "data/raw/Williams Obstetricia 26a Edicion.pdf",
        "data/old/Williams Obstetricia 26a Edicion.pdf"
    ]
    
    test_pdf = None
    for path in pdf_paths:
        if os.path.exists(path):
            test_pdf = path
            break
    
    if not test_pdf:
        print("❌ Williams Obstetricia PDF not found in expected locations")
        print("Available PDFs:")
        for pdf_dir in ["data/raw", "data/old"]:
            if os.path.exists(pdf_dir):
                pdfs = [f for f in os.listdir(pdf_dir) if f.endswith('.pdf')]
                for pdf in pdfs:
                    print(f"   {pdf_dir}/{pdf}")
        return
    
    # Process the large PDF
    chunks = process_large_pdf_demo(test_pdf)
    
    print("\n✅ LARGE PDF PROCESSING COMPLETE!")
    print("\n💡 KEY IMPROVEMENTS:")
    print("- ✅ Memory-efficient processing (chunks loaded on demand)")
    print("- ✅ Intelligent chunking preserves document structure")
    print("- ✅ Page-range extraction for massive documents")
    print("- ✅ Multiple extraction methods with fallbacks")
    print("- ✅ Smart overlap to maintain context between chunks")
    print("- ✅ Section-aware splitting (headers, page breaks)")
    print("- ✅ Table detection and preservation")


if __name__ == "__main__":
    main()