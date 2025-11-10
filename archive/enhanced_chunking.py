#!/usr/bin/env python3
"""
Enhanced Chunking System for Large Documents
Addresses the chunking issues with massive PDFs like Williams Obstetricia (1322 pages)
"""

import os
import re
import logging
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ChunkMetadata:
    """Metadata for each chunk"""
    chunk_id: int
    source: str
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    section_type: str = "content"  # content, header, table, caption, reference
    chapter: Optional[str] = None
    topic: Optional[str] = None
    word_count: int = 0
    char_count: int = 0
    has_tables: bool = False
    has_images: bool = False
    language_detected: str = "es"
    quality_score: float = 1.0  # 0-1, based on text coherence

class IntelligentChunker:
    """Advanced chunking system that understands document structure"""
    
    def __init__(self, 
                 target_chunk_size: int = 600,
                 max_chunk_size: int = 1000,
                 min_chunk_size: int = 200,
                 overlap_size: int = 100):
        self.target_chunk_size = target_chunk_size
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.overlap_size = overlap_size
        
        # Patterns for different content types
        self.chapter_patterns = [
            r'^CAPÍTULO\s+\d+',
            r'^CHAPTER\s+\d+',
            r'^PARTE\s+\d+',
            r'^SECCIÓN\s+\d+',
            r'^\d+\.\s+[A-ZÁÉÍÓÚÑ]',
        ]
        
        self.header_patterns = [
            r'^[A-ZÁÉÍÓÚÑ\s]+$',  # All caps headers
            r'^\d+\.\d+\s+',       # Numbered sections
            r'^#{1,6}\s+',         # Markdown headers
        ]
        
        self.reference_patterns = [
            r'Referencias?',
            r'Bibliografía',
            r'Bibliography',
            r'REFERENCES?',
            r'\[\d+\]',
            r'\(\d{4}\)',
        ]
        
        self.table_patterns = [
            r'Tabla\s+\d+',
            r'Table\s+\d+',
            r'TABLA\s+\d+',
            r'\|.*\|',  # Markdown tables
        ]
        
        self.medical_terms = [
            'diagnóstico', 'tratamiento', 'síntoma', 'enfermedad', 'paciente',
            'terapia', 'medicamento', 'cirugía', 'obstetricia', 'ginecología',
            'embarazo', 'parto', 'fetal', 'maternal', 'prenatal', 'postnatal'
        ]
    
    def chunk_document(self, text: str, source: str = "unknown", 
                      pages_info: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Main chunking function that intelligently splits large documents
        """
        logger.info(f"📄 Chunking document: {source} ({len(text):,} chars)")
        
        # Preprocess text
        cleaned_text = self._preprocess_text(text)
        
        # Detect document structure
        sections = self._detect_document_structure(cleaned_text)
        
        # Create chunks preserving structure
        chunks = []
        chunk_id = 0
        
        for section in sections:
            section_chunks = self._chunk_section(section, chunk_id, source, pages_info)
            chunks.extend(section_chunks)
            chunk_id += len(section_chunks)
        
        # Post-process chunks
        final_chunks = self._post_process_chunks(chunks)
        
        logger.info(f"✅ Created {len(final_chunks)} intelligent chunks")
        return final_chunks
    
    def _preprocess_text(self, text: str) -> str:
        """Clean and normalize text before chunking"""
        
        # Remove excessive whitespace
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        text = re.sub(r' +', ' ', text)
        
        # Fix common OCR errors
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # Missing spaces
        text = re.sub(r'(\d)([a-zA-Z])', r'\1 \2', text)   # Number-letter combinations
        
        # Normalize Spanish characters
        replacements = {
            'á': 'á', 'é': 'é', 'í': 'í', 'ó': 'ó', 'ú': 'ú',
            'ñ': 'ñ', 'ü': 'ü'
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        return text.strip()
    
    def _detect_document_structure(self, text: str) -> List[Dict[str, Any]]:
        """Detect logical sections in the document"""
        
        lines = text.split('\n')
        sections = []
        current_section = {
            'type': 'content',
            'lines': [],
            'metadata': {}
        }
        
        for line_num, line in enumerate(lines):
            line = line.strip()
            
            if not line:
                continue
            
            # Detect section type
            section_type = self._classify_line(line)
            
            # If we hit a new section type, save current and start new
            if (section_type in ['chapter', 'header'] and 
                current_section['lines'] and 
                section_type != current_section['type']):
                
                sections.append(current_section)
                current_section = {
                    'type': section_type,
                    'lines': [line],
                    'metadata': {'start_line': line_num}
                }
            else:
                current_section['lines'].append(line)
                current_section['type'] = section_type
        
        # Add final section
        if current_section['lines']:
            sections.append(current_section)
        
        # Merge sections and add metadata
        merged_sections = self._merge_related_sections(sections)
        
        return merged_sections
    
    def _classify_line(self, line: str) -> str:
        """Classify a line by its content type"""
        
        # Check for chapters
        for pattern in self.chapter_patterns:
            if re.match(pattern, line, re.IGNORECASE):
                return 'chapter'
        
        # Check for headers
        for pattern in self.header_patterns:
            if re.match(pattern, line):
                return 'header'
        
        # Check for references
        for pattern in self.reference_patterns:
            if re.search(pattern, line, re.IGNORECASE):
                return 'reference'
        
        # Check for tables
        for pattern in self.table_patterns:
            if re.search(pattern, line, re.IGNORECASE):
                return 'table'
        
        # Check for medical content
        medical_terms_found = sum(1 for term in self.medical_terms 
                                if term in line.lower())
        if medical_terms_found >= 2:
            return 'medical_content'
        
        return 'content'
    
    def _merge_related_sections(self, sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge small related sections to avoid tiny chunks"""
        
        merged = []
        current_merged = None
        
        for section in sections:
            section_text = '\n'.join(section['lines'])
            section_length = len(section_text)
            
            # If section is too small, try to merge with previous
            if (section_length < self.min_chunk_size and 
                current_merged and 
                section['type'] == current_merged['type']):
                
                current_merged['lines'].extend(section['lines'])
                current_merged['metadata']['merged_sections'] = current_merged['metadata'].get('merged_sections', 1) + 1
            else:
                # Save previous merged section
                if current_merged:
                    merged.append(current_merged)
                
                # Start new merged section
                current_merged = {
                    'type': section['type'],
                    'lines': section['lines'].copy(),
                    'metadata': section['metadata'].copy()
                }
        
        # Add final merged section
        if current_merged:
            merged.append(current_merged)
        
        return merged
    
    def _chunk_section(self, section: Dict[str, Any], start_chunk_id: int, 
                      source: str, pages_info: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Chunk a single section while preserving its integrity"""
        
        section_text = '\n'.join(section['lines'])
        section_type = section['type']
        
        # If section is small enough, make it one chunk
        if len(section_text) <= self.max_chunk_size:
            return [self._create_chunk(
                chunk_id=start_chunk_id,
                text=section_text,
                source=source,
                section_type=section_type,
                pages_info=pages_info,
                section_metadata=section['metadata']
            )]
        
        # For larger sections, split intelligently
        chunks = []
        
        if section_type in ['chapter', 'header']:
            # For structured content, split by paragraphs
            chunks = self._chunk_by_paragraphs(section_text, start_chunk_id, source, 
                                             section_type, pages_info, section['metadata'])
        elif section_type == 'table':
            # Keep tables together when possible
            chunks = self._chunk_tables(section_text, start_chunk_id, source, 
                                      pages_info, section['metadata'])
        else:
            # For regular content, use sentence-based chunking
            chunks = self._chunk_by_sentences(section_text, start_chunk_id, source, 
                                            section_type, pages_info, section['metadata'])
        
        return chunks
    
    def _chunk_by_paragraphs(self, text: str, start_id: int, source: str,
                           section_type: str, pages_info: Dict[str, Any],
                           section_metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Chunk text by paragraphs with intelligent merging"""
        
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        chunks = []
        current_chunk = []
        current_size = 0
        chunk_id = start_id
        
        for para in paragraphs:
            para_size = len(para)
            
            # If adding this paragraph would exceed target size
            if current_size + para_size > self.target_chunk_size and current_chunk:
                # Create chunk from accumulated paragraphs
                chunk_text = '\n\n'.join(current_chunk)
                chunks.append(self._create_chunk(
                    chunk_id=chunk_id,
                    text=chunk_text,
                    source=source,
                    section_type=section_type,
                    pages_info=pages_info,
                    section_metadata=section_metadata
                ))
                chunk_id += 1
                
                # Start new chunk with overlap if beneficial
                if self.overlap_size > 0 and len(current_chunk) > 1:
                    current_chunk = [current_chunk[-1], para]  # Keep last paragraph
                    current_size = len(current_chunk[-2]) + para_size
                else:
                    current_chunk = [para]
                    current_size = para_size
            else:
                current_chunk.append(para)
                current_size += para_size
        
        # Add final chunk
        if current_chunk:
            chunk_text = '\n\n'.join(current_chunk)
            chunks.append(self._create_chunk(
                chunk_id=chunk_id,
                text=chunk_text,
                source=source,
                section_type=section_type,
                pages_info=pages_info,
                section_metadata=section_metadata
            ))
        
        return chunks
    
    def _chunk_by_sentences(self, text: str, start_id: int, source: str,
                           section_type: str, pages_info: Dict[str, Any],
                           section_metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Chunk text by sentences with smart boundaries"""
        
        # Split into sentences (Spanish-aware)
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-ZÁÉÍÓÚÑ])', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        chunks = []
        current_chunk = []
        current_size = 0
        chunk_id = start_id
        
        for sentence in sentences:
            sentence_size = len(sentence)
            
            # Check if this sentence contains important medical information
            is_important = any(term in sentence.lower() for term in self.medical_terms[:5])
            
            # If adding would exceed size and we have content
            if current_size + sentence_size > self.target_chunk_size and current_chunk:
                chunk_text = ' '.join(current_chunk)
                chunks.append(self._create_chunk(
                    chunk_id=chunk_id,
                    text=chunk_text,
                    source=source,
                    section_type=section_type,
                    pages_info=pages_info,
                    section_metadata=section_metadata
                ))
                chunk_id += 1
                
                # Smart overlap - keep important sentences
                if self.overlap_size > 0:
                    overlap_sentences = []
                    overlap_size = 0
                    
                    for prev_sentence in reversed(current_chunk):
                        if overlap_size + len(prev_sentence) <= self.overlap_size:
                            overlap_sentences.insert(0, prev_sentence)
                            overlap_size += len(prev_sentence)
                        else:
                            break
                    
                    current_chunk = overlap_sentences + [sentence]
                    current_size = sum(len(s) for s in current_chunk)
                else:
                    current_chunk = [sentence]
                    current_size = sentence_size
            else:
                current_chunk.append(sentence)
                current_size += sentence_size
        
        # Add final chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append(self._create_chunk(
                chunk_id=chunk_id,
                text=chunk_text,
                source=source,
                section_type=section_type,
                pages_info=pages_info,
                section_metadata=section_metadata
            ))
        
        return chunks
    
    def _chunk_tables(self, text: str, start_id: int, source: str,
                     pages_info: Dict[str, Any], section_metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Handle table content specially"""
        
        # Try to keep tables intact when possible
        if len(text) <= self.max_chunk_size:
            return [self._create_chunk(
                chunk_id=start_id,
                text=text,
                source=source,
                section_type='table',
                pages_info=pages_info,
                section_metadata=section_metadata
            )]
        
        # For very large tables, split by rows but keep headers
        lines = text.split('\n')
        header_lines = []
        data_lines = []
        
        # Identify header vs data
        for i, line in enumerate(lines):
            if i < 3 or 'Tabla' in line or '|' in line[:20]:  # Likely header
                header_lines.append(line)
            else:
                data_lines.append(line)
        
        chunks = []
        chunk_id = start_id
        
        # Create chunks with header repeated
        current_data = []
        current_size = sum(len(line) for line in header_lines)
        
        for line in data_lines:
            if current_size + len(line) > self.target_chunk_size and current_data:
                # Create chunk with header + current data
                chunk_lines = header_lines + current_data
                chunk_text = '\n'.join(chunk_lines)
                
                chunks.append(self._create_chunk(
                    chunk_id=chunk_id,
                    text=chunk_text,
                    source=source,
                    section_type='table',
                    pages_info=pages_info,
                    section_metadata=section_metadata
                ))
                chunk_id += 1
                
                current_data = [line]
                current_size = sum(len(line) for line in header_lines) + len(line)
            else:
                current_data.append(line)
                current_size += len(line)
        
        # Add final chunk
        if current_data:
            chunk_lines = header_lines + current_data
            chunk_text = '\n'.join(chunk_lines)
            chunks.append(self._create_chunk(
                chunk_id=chunk_id,
                text=chunk_text,
                source=source,
                section_type='table',
                pages_info=pages_info,
                section_metadata=section_metadata
            ))
        
        return chunks
    
    def _create_chunk(self, chunk_id: int, text: str, source: str,
                     section_type: str, pages_info: Dict[str, Any] = None,
                     section_metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Create a standardized chunk with metadata"""
        
        # Calculate basic metrics
        word_count = len(text.split())
        char_count = len(text)
        has_tables = any(pattern in text for pattern in ['tabla', 'Table', '|'])
        has_images = any(pattern in text.lower() for pattern in ['figura', 'image', 'gráfico'])
        
        # Estimate quality based on completeness and medical relevance
        quality_score = self._estimate_quality(text)
        
        # Detect chapter/topic if possible
        chapter = self._extract_chapter(text)
        topic = self._extract_topic(text)
        
        chunk = {
            'id': chunk_id,
            'text': text,
            'metadata': ChunkMetadata(
                chunk_id=chunk_id,
                source=source,
                section_type=section_type,
                chapter=chapter,
                topic=topic,
                word_count=word_count,
                char_count=char_count,
                has_tables=has_tables,
                has_images=has_images,
                quality_score=quality_score
            ).__dict__
        }
        
        # Add pages info if available
        if pages_info:
            chunk['metadata'].update(pages_info)
        
        # Add section metadata if available
        if section_metadata:
            chunk['metadata']['section_info'] = section_metadata
        
        return chunk
    
    def _estimate_quality(self, text: str) -> float:
        """Estimate the quality/completeness of a chunk"""
        
        score = 1.0
        
        # Penalize very short chunks
        if len(text) < self.min_chunk_size:
            score -= 0.3
        
        # Reward medical terminology
        medical_terms_found = sum(1 for term in self.medical_terms if term in text.lower())
        if medical_terms_found > 0:
            score += min(0.2, medical_terms_found * 0.05)
        
        # Penalize incomplete sentences
        if not text.rstrip().endswith(('.', '!', '?', ':')):
            score -= 0.1
        
        # Reward structured content
        if any(pattern in text for pattern in ['1.', '2.', '•', '-']):
            score += 0.1
        
        return max(0.0, min(1.0, score))
    
    def _extract_chapter(self, text: str) -> Optional[str]:
        """Try to extract chapter information"""
        
        for pattern in self.chapter_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(0)
        
        return None
    
    def _extract_topic(self, text: str) -> Optional[str]:
        """Try to extract main topic"""
        
        # Look for medical specialties and procedures
        medical_topics = [
            'obstetricia', 'ginecología', 'embarazo', 'parto', 'cesárea',
            'fetal', 'maternal', 'prenatal', 'postnatal', 'lactancia',
            'diagnóstico', 'tratamiento', 'cirugía', 'medicamento'
        ]
        
        found_topics = [topic for topic in medical_topics if topic in text.lower()]
        return found_topics[0] if found_topics else None
    
    def _post_process_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Final processing to optimize chunks"""
        
        if not chunks:
            return chunks
        
        # Remove empty or very low quality chunks
        filtered_chunks = [
            chunk for chunk in chunks 
            if (len(chunk['text'].strip()) >= self.min_chunk_size and 
                chunk['metadata']['quality_score'] > 0.3)
        ]
        
        # Renumber chunks
        for i, chunk in enumerate(filtered_chunks):
            chunk['id'] = i
            chunk['metadata']['chunk_id'] = i
        
        logger.info(f"📊 Post-processing: {len(chunks)} → {len(filtered_chunks)} chunks")
        return filtered_chunks


def demo_large_document_chunking():
    """Demonstrate the enhanced chunking system"""
    
    # Sample large medical text (simulating Williams Obstetricia content)
    sample_text = """
    CAPÍTULO 15. EVALUACIÓN FETAL

    La evaluación fetal durante el embarazo es fundamental para garantizar el bienestar materno-fetal. Los métodos modernos de evaluación incluyen ultrasonografía, monitoreo fetal y pruebas bioquímicas.

    ULTRASONOGRAFÍA OBSTÉTRICA

    La ultrasonografía es la herramienta diagnóstica más importante en obstetricia moderna. Permite evaluar el crecimiento fetal, detectar anomalías congénitas y monitorear el bienestar fetal.

    Tipos de ultrasonografía:
    1. Ultrasonografía transabdominal
    2. Ultrasonografía transvaginal
    3. Ultrasonografía Doppler
    4. Ultrasonografía tridimensional (3D)
    5. Ultrasonografía tetradimensional (4D)

    MONITOREO FETAL

    El monitoreo fetal anteparto incluye varias pruebas diseñadas para evaluar el bienestar fetal. Estas pruebas son especialmente importantes en embarazos de alto riesgo.

    Tabla 15-1. Indicaciones para monitoreo fetal anteparto
    | Indicación | Frecuencia | Método preferido |
    |------------|------------|------------------|
    | Diabetes gestacional | Semanal | NST + Perfil biofísico |
    | Hipertensión gestacional | Bisemanal | NST + Doppler |
    | Restricción crecimiento fetal | Diaria | Perfil biofísico |
    | Embarazo prolongado | Bisemanal | NST + ILA |

    La prueba sin estrés (NST) es la prueba de monitoreo fetal más comúnmente utilizada. Se considera reactiva cuando se observan al menos dos aceleraciones de la frecuencia cardíaca fetal de 15 latidos por minuto por encima de la línea basal, que duran al menos 15 segundos, en un período de 20 minutos.

    INTERPRETACIÓN DE RESULTADOS

    La interpretación adecuada de las pruebas de bienestar fetal requiere experiencia clínica y conocimiento de las limitaciones de cada prueba. Los resultados falsos positivos pueden llevar a intervenciones innecesarias, mientras que los falsos negativos pueden resultar en morbilidad fetal.

    La combinación de múltiples métodos de evaluación aumenta la precisión diagnóstica y reduce la tasa de falsos positivos y negativos.
    """
    
    print("🧪 Demonstrating Enhanced Chunking for Large Medical Documents")
    print("=" * 80)
    
    chunker = IntelligentChunker(target_chunk_size=400, overlap_size=80)
    
    chunks = chunker.chunk_document(
        text=sample_text,
        source="Williams Obstetricia - Capítulo 15",
        pages_info={'pages_start': 285, 'pages_end': 295}
    )
    
    print(f"📊 Created {len(chunks)} intelligent chunks:")
    print()
    
    for i, chunk in enumerate(chunks):
        metadata = chunk['metadata']
        print(f"Chunk {i+1}:")
        print(f"  Type: {metadata['section_type']}")
        print(f"  Size: {metadata['char_count']} chars, {metadata['word_count']} words")
        print(f"  Quality: {metadata['quality_score']:.2f}")
        print(f"  Has tables: {metadata['has_tables']}")
        print(f"  Chapter: {metadata.get('chapter', 'N/A')}")
        print(f"  Topic: {metadata.get('topic', 'N/A')}")
        print(f"  Preview: {chunk['text'][:100]}...")
        print("-" * 50)
    
    # Save demo results
    output_file = "data/clean/enhanced_chunking_demo.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    
    print(f"💾 Demo results saved to: {output_file}")
    
    return chunks


if __name__ == "__main__":
    demo_large_document_chunking()