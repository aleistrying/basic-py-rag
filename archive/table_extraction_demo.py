#!/usr/bin/env python3
"""
Enhanced PDF Table Extraction Demo
Demonstrates advanced table extraction capabilities for your RAG pipeline
"""

import os
import sys
from pathlib import Path
import json

# Add project to path
sys.path.append(str(Path(__file__).parent.parent))

def demonstrate_table_extraction():
    """Demonstrate table extraction with available PDF"""
    
    # Find available PDFs
    pdf_dir = Path("data/raw")
    pdf_files = list(pdf_dir.glob("*.pdf"))
    
    if not pdf_files:
        print("❌ No PDF files found for demonstration")
        return
    
    test_pdf = pdf_files[0]
    print(f"📄 Demonstrating with: {test_pdf.name}")
    
    # Test table extraction methods
    methods = {
        "pdfplumber": test_pdfplumber_tables,
        "pymupdf": test_pymupdf_tables,
    }
    
    results = {}
    
    for method_name, method_func in methods.items():
        print(f"\n🔍 Testing {method_name}...")
        try:
            result = method_func(test_pdf)
            results[method_name] = result
            print(f"✅ {method_name}: {result['text_length']} chars, {result['tables_found']} tables")
        except Exception as e:
            print(f"❌ {method_name} failed: {e}")
            results[method_name] = {"error": str(e)}
    
    # Generate recommendation
    print("\n📊 EXTRACTION RESULTS SUMMARY")
    print("=" * 50)
    
    for method, result in results.items():
        if "error" not in result:
            print(f"{method}:")
            print(f"  - Text extracted: {result['text_length']} characters")
            print(f"  - Tables found: {result['tables_found']}")
            print(f"  - Processing time: {result.get('time', 'N/A'):.3f}s")
        else:
            print(f"{method}: ERROR - {result['error']}")
    
    # Save demo results
    output_file = Path("data/clean/table_extraction_demo.json")
    output_file.parent.mkdir(exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\n💾 Results saved to: {output_file}")
    
    return results

def test_pdfplumber_tables(pdf_path):
    """Test pdfplumber table extraction"""
    import pdfplumber
    import time
    
    start_time = time.time()
    
    text_content = ""
    tables_found = 0
    table_details = []
    
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, 1):
            # Extract text
            page_text = page.extract_text() or ""
            text_content += page_text + "\n"
            
            # Extract tables
            tables = page.extract_tables()
            
            for table_idx, table in enumerate(tables):
                if table and len(table) > 1:  # Valid table
                    tables_found += 1
                    
                    # Analyze table structure
                    table_info = {
                        "page": page_num,
                        "table_index": table_idx,
                        "rows": len(table),
                        "columns": len(table[0]) if table[0] else 0,
                        "preview": table[:2] if len(table) >= 2 else table  # First 2 rows
                    }
                    table_details.append(table_info)
    
    processing_time = time.time() - start_time
    
    return {
        "text_length": len(text_content),
        "tables_found": tables_found,
        "table_details": table_details,
        "time": processing_time,
        "method": "pdfplumber"
    }

def test_pymupdf_tables(pdf_path):
    """Test PyMuPDF table detection"""
    import fitz
    import time
    
    start_time = time.time()
    
    text_content = ""
    tables_found = 0
    table_details = []
    
    doc = fitz.open(pdf_path)
    
    for page_num in range(doc.page_count):
        page = doc[page_num]
        
        # Extract text
        page_text = page.get_text()
        text_content += page_text + "\n"
        
        # Look for table-like blocks
        blocks = page.get_text("blocks")
        
        for block_idx, block in enumerate(blocks):
            if len(block) >= 5:
                block_text = block[4]
                
                if looks_like_table_heuristic(block_text):
                    tables_found += 1
                    
                    table_info = {
                        "page": page_num + 1,
                        "block_index": block_idx,
                        "coordinates": block[:4],  # x0, y0, x1, y1
                        "text_preview": block_text[:200] + "..." if len(block_text) > 200 else block_text
                    }
                    table_details.append(table_info)
    
    doc.close()
    processing_time = time.time() - start_time
    
    return {
        "text_length": len(text_content),
        "tables_found": tables_found,
        "table_details": table_details,
        "time": processing_time,
        "method": "pymupdf_heuristic"
    }

def looks_like_table_heuristic(text):
    """Enhanced heuristic for table detection"""
    if not text or len(text) < 50:
        return False
    
    lines = text.strip().split('\n')
    if len(lines) < 2:
        return False
    
    score = 0
    
    # 1. Check for separators
    separator_count = sum(1 for line in lines if any(sep in line for sep in ['|', '\t', '  ']))
    if separator_count / len(lines) > 0.6:
        score += 3
    
    # 2. Check for numeric content
    numeric_lines = sum(1 for line in lines if any(c.isdigit() for c in line))
    if numeric_lines / len(lines) > 0.4:
        score += 2
    
    # 3. Check for time patterns
    time_patterns = [':', 'pm', 'am', 'p.m.', 'a.m.']
    time_count = sum(1 for line in lines if any(pattern in line.lower() for pattern in time_patterns))
    if time_count > 0:
        score += 2
    
    # 4. Check for consistent line lengths (formatted table)
    avg_length = sum(len(line) for line in lines) / len(lines)
    similar_length_lines = sum(1 for line in lines if abs(len(line) - avg_length) < avg_length * 0.3)
    if similar_length_lines / len(lines) > 0.7:
        score += 1
    
    return score >= 4

def create_enhanced_pipeline_integration():
    """Create code snippet for integrating advanced table extraction"""
    
    integration_code = '''
# ENHANCED TABLE EXTRACTION FOR PIPELINE.PY
# Add this to your existing pipeline for better table handling

def extract_tables_with_fallback(pdf_path):
    """Enhanced table extraction with multiple methods"""
    
    # Method 1: pdfplumber (best for structured tables)
    try:
        import pdfplumber
        
        tables = []
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                page_tables = page.extract_tables()
                
                for table_idx, table in enumerate(page_tables):
                    if table and len(table) > 1:
                        # Convert to formatted text
                        table_text = format_table_as_markdown(table, page_num, table_idx + 1)
                        tables.append(table_text)
        
        if tables:
            return tables
            
    except Exception as e:
        print(f"pdfplumber table extraction failed: {e}")
    
    # Method 2: PyMuPDF heuristic detection
    try:
        import fitz
        
        tables = []
        doc = fitz.open(pdf_path)
        
        for page_num in range(doc.page_count):
            page = doc[page_num]
            blocks = page.get_text("blocks")
            
            for block in blocks:
                if len(block) >= 5 and looks_like_table_heuristic(block[4]):
                    table_text = f"\\n--- TABLA DETECTADA (Página {page_num + 1}) ---\\n"
                    table_text += block[4]
                    table_text += "\\n--- FIN TABLA ---\\n"
                    tables.append(table_text)
        
        doc.close()
        return tables
        
    except Exception as e:
        print(f"PyMuPDF table detection failed: {e}")
    
    return []

def format_table_as_markdown(table, page_num, table_num):
    """Convert table to clean markdown format"""
    if not table:
        return ""
    
    # Filter out empty rows
    clean_table = [row for row in table if any(cell and str(cell).strip() for cell in row)]
    
    if not clean_table:
        return ""
    
    markdown = f"\\n## Tabla {table_num} (Página {page_num})\\n\\n"
    
    # Header row
    if clean_table[0]:
        header = " | ".join(str(cell).strip() if cell else "" for cell in clean_table[0])
        markdown += f"| {header} |\\n"
        markdown += f"| {' | '.join(['---'] * len(clean_table[0]))} |\\n"
    
    # Data rows
    for row in clean_table[1:]:
        if any(cell for cell in row if cell):
            row_text = " | ".join(str(cell).strip() if cell else "" for cell in row)
            markdown += f"| {row_text} |\\n"
    
    markdown += "\\n"
    return markdown
'''
    
    # Save the integration code
    output_file = Path("data/clean/enhanced_table_extraction_code.py")
    output_file.parent.mkdir(exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(integration_code)
    
    print(f"📝 Enhanced table extraction code saved to: {output_file}")
    return output_file

def main():
    """Main demo function"""
    print("🚀 Advanced PDF Table Extraction Demo")
    print("=" * 60)
    
    # Run demonstration
    results = demonstrate_table_extraction()
    
    # Create integration code
    integration_file = create_enhanced_pipeline_integration()
    
    print("\n✅ DEMO COMPLETE!")
    print("\n📋 SUMMARY:")
    print("- Tested multiple table extraction methods")
    print("- pdfplumber: Best for structured tables with clear boundaries")
    print("- PyMuPDF: Good for detecting table-like content heuristically")
    print("- Both methods can enhance your existing RAG pipeline")
    
    print(f"\n🔧 INTEGRATION:")
    print(f"- Enhanced extraction code: {integration_file}")
    print("- Copy the functions to your pipeline.py for better table handling")
    print("- The code includes markdown formatting for better readability")
    
    print("\n💡 RECOMMENDATION:")
    print("- Use pdfplumber as primary method for table extraction")
    print("- Fall back to PyMuPDF heuristics for edge cases")
    print("- Format tables as markdown for better LLM understanding")

if __name__ == "__main__":
    main()