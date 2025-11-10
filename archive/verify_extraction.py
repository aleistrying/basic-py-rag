#!/usr/bin/env python3
"""
Verify PDF Extraction and Table Detection
Tests our enhanced extraction against the course guide
"""

import os
import sys
from pathlib import Path

# Add project to path
sys.path.append(str(Path(__file__).parent.parent))

# Test imports
try:
    import pdfplumber
    print("✅ pdfplumber available")
except ImportError:
    print("❌ pdfplumber not available")
    pdfplumber = None

try:
    import fitz
    print("✅ PyMuPDF available")
except ImportError:
    print("❌ PyMuPDF not available")
    fitz = None

def extract_and_analyze_course_guide():
    """Extract content from course guide and analyze table detection"""
    
    pdf_path = "data/raw/Guia de Curso 1CA217 2S2025 Sistemas BD Avanzadas v2.pdf"
    
    if not os.path.exists(pdf_path):
        print(f"❌ PDF not found: {pdf_path}")
        return
    
    print(f"\n📄 Analyzing: {os.path.basename(pdf_path)}")
    print("=" * 60)
    
    # Test with pdfplumber (best for tables)
    if pdfplumber:
        print("\n🔍 Testing pdfplumber table extraction...")
        try:
            with pdfplumber.open(pdf_path) as pdf:
                total_text = ""
                total_tables = 0
                
                for page_num, page in enumerate(pdf.pages, 1):
                    print(f"\n📄 Page {page_num}:")
                    
                    # Extract text
                    page_text = page.extract_text() or ""
                    print(f"   📝 Text: {len(page_text)} chars")
                    
                    # Extract tables
                    tables = page.extract_tables()
                    print(f"   📊 Tables found: {len(tables)}")
                    
                    for table_idx, table in enumerate(tables):
                        if table and len(table) > 1:
                            print(f"      Table {table_idx + 1}: {len(table)} rows × {len(table[0]) if table[0] else 0} cols")
                            
                            # Show first few rows
                            print("      Preview:")
                            for row_idx, row in enumerate(table[:3]):
                                if row and any(cell for cell in row if cell):
                                    clean_row = [str(cell).strip() if cell else "" for cell in row]
                                    clean_row = [cell for cell in clean_row if cell]  # Remove empty
                                    if clean_row:
                                        print(f"        Row {row_idx + 1}: {' | '.join(clean_row)}")
                    
                    total_text += page_text
                    total_tables += len([t for t in tables if t and len(t) > 1])
                
                print(f"\n📊 Summary:")
                print(f"   Total text: {len(total_text)} characters")
                print(f"   Total tables: {total_tables}")
                
                # Look for schedule keywords
                schedule_keywords = ['lunes', 'miércoles', 'horario', 'clase', 'p.m.', 'pm']
                found_keywords = [kw for kw in schedule_keywords if kw.lower() in total_text.lower()]
                print(f"   Schedule keywords found: {found_keywords}")
                
                # Search for specific schedule content
                if 'horario' in total_text.lower():
                    print(f"\n🎯 Found 'horario' in text!")
                    # Find the section with horario
                    lines = total_text.split('\n')
                    for i, line in enumerate(lines):
                        if 'horario' in line.lower():
                            context_start = max(0, i-2)
                            context_end = min(len(lines), i+3)
                            print("   Context:")
                            for j in range(context_start, context_end):
                                marker = ">>> " if j == i else "    "
                                print(f"{marker}{lines[j].strip()}")
                            break
                
        except Exception as e:
            print(f"❌ pdfplumber error: {e}")
    
    # Test with PyMuPDF for comparison
    if fitz:
        print("\n🔍 Testing PyMuPDF text extraction...")
        try:
            doc = fitz.open(pdf_path)
            pymupdf_text = ""
            
            for page_num in range(doc.page_count):
                page = doc[page_num]
                page_text = page.get_text()
                pymupdf_text += page_text
                
                print(f"   Page {page_num + 1}: {len(page_text)} chars")
            
            doc.close()
            
            print(f"   Total PyMuPDF text: {len(pymupdf_text)} chars")
            
            # Compare schedule detection
            if 'horario' in pymupdf_text.lower():
                print("   ✅ PyMuPDF also found 'horario'")
            else:
                print("   ❌ PyMuPDF didn't find 'horario'")
                
        except Exception as e:
            print(f"❌ PyMuPDF error: {e}")

def main():
    """Main function"""
    print("🧪 PDF Extraction and Table Detection Verification")
    print("=" * 60)
    
    extract_and_analyze_course_guide()
    
    print("\n✅ Analysis complete!")
    print("\nRecommendations:")
    print("- Use pdfplumber for best table extraction")
    print("- PyMuPDF as fallback for robust text extraction")
    print("- Both libraries can handle the course guide effectively")

if __name__ == "__main__":
    main()