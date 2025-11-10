#!/usr/bin/env python3
"""
Advanced PDF Extraction Testing Suite
Tests multiple PDF extraction libraries for text, table, and image extraction.
Integrates with existing RAG pipeline for real-world evaluation.
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import traceback

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def install_missing_packages():
    """Install required packages if not available"""
    import subprocess
    
    packages = [
        "PyPDF2",
        "pymupdf",
        "pdfminer.six", 
        "camelot-py[cv]",
        "pdfplumber",
        "unstructured[pdf]",
        "tabula-py",
        "pypdfium2",
        "pymupdf4llm",
        "textract",
        "opencv-python",
        "pillow"
    ]
    
    for package in packages:
        try:
            result = subprocess.run([sys.executable, "-m", "pip", "install", package], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✅ Installed {package}")
            else:
                print(f"⚠️ Failed to install {package}: {result.stderr}")
        except Exception as e:
            print(f"⚠️ Failed to install {package}: {e}")

class PDFExtractorTester:
    """Test suite for PDF extraction libraries"""
    
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.results = {}
        
    def test_pypdf2(self) -> Dict[str, Any]:
        """Test PyPDF2 - Basic text extraction"""
        try:
            import PyPDF2
            
            start_time = time.time()
            with open(self.pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                pages_info = []
                
                for page_num, page in enumerate(reader.pages[:3]):  # First 3 pages
                    page_text = page.extract_text()
                    text += page_text + "\n"
                    pages_info.append({
                        "page": page_num + 1,
                        "chars": len(page_text),
                        "preview": page_text[:200] + "..." if len(page_text) > 200 else page_text
                    })
                    
            extraction_time = time.time() - start_time
            
            return {
                "success": True,
                "time": extraction_time,
                "total_chars": len(text),
                "pages_processed": len(pages_info),
                "pages_info": pages_info,
                "full_text": text,
                "library_version": PyPDF2.__version__
            }
            
        except Exception as e:
            return {"success": False, "error": str(e), "traceback": traceback.format_exc()}
    
    def test_pymupdf(self) -> Dict[str, Any]:
        """Test PyMuPDF - Advanced text and layout extraction"""
        try:
            import fitz
            
            start_time = time.time()
            doc = fitz.open(self.pdf_path)
            
            text = ""
            pages_info = []
            tables_found = []
            images_found = []
            
            for page_num in range(min(3, doc.page_count)):
                page = doc[page_num]
                
                # Basic text extraction
                page_text = page.get_text()
                text += page_text + "\n"
                
                # Text blocks with coordinates
                blocks = page.get_text("blocks")
                text_blocks = [block for block in blocks if len(block) >= 5]
                
                # Look for tables (heuristic: blocks with many numbers/spaces)
                potential_tables = []
                for block in text_blocks:
                    block_text = block[4] if len(block) > 4 else ""
                    if self._looks_like_table(block_text):
                        potential_tables.append(block_text[:100])
                
                # Extract images
                image_list = page.get_images()
                page_images = []
                for img_index, img in enumerate(image_list):
                    try:
                        xref = img[0]
                        base_image = doc.extract_image(xref)
                        page_images.append({
                            "index": img_index,
                            "extension": base_image["ext"],
                            "size_bytes": len(base_image["image"])
                        })
                    except:
                        pass
                
                pages_info.append({
                    "page": page_num + 1,
                    "chars": len(page_text),
                    "text_blocks": len(text_blocks),
                    "potential_tables": len(potential_tables),
                    "images": len(page_images),
                    "preview": page_text[:200] + "..." if len(page_text) > 200 else page_text
                })
                
                tables_found.extend(potential_tables)
                images_found.extend(page_images)
            
            doc.close()
            extraction_time = time.time() - start_time
            
            return {
                "success": True,
                "time": extraction_time,
                "total_chars": len(text),
                "pages_processed": len(pages_info),
                "pages_info": pages_info,
                "tables_found": len(tables_found),
                "images_found": len(images_found),
                "full_text": text,
                "library_version": fitz.version[0]
            }
            
        except Exception as e:
            return {"success": False, "error": str(e), "traceback": traceback.format_exc()}
    
    def test_camelot(self) -> Dict[str, Any]:
        """Test Camelot - Specialized table extraction"""
        try:
            import camelot
            
            start_time = time.time()
            
            # Extract tables
            tables = camelot.read_pdf(self.pdf_path, pages='1-3')
            extraction_time = time.time() - start_time
            
            tables_data = []
            total_rows = 0
            
            for i, table in enumerate(tables):
                df = table.df
                tables_data.append({
                    "table_index": i,
                    "rows": len(df),
                    "columns": len(df.columns),
                    "accuracy": table.accuracy,
                    "whitespace": table.whitespace,
                    "preview": df.head(3).to_dict() if not df.empty else {}
                })
                total_rows += len(df)
            
            return {
                "success": True,
                "time": extraction_time,
                "tables_found": len(tables),
                "total_rows": total_rows,
                "tables_data": tables_data,
                "library_version": camelot.__version__
            }
            
        except Exception as e:
            return {"success": False, "error": str(e), "traceback": traceback.format_exc()}
    
    def test_pdfplumber(self) -> Dict[str, Any]:
        """Test pdfplumber - Table and coordinate extraction"""
        try:
            import pdfplumber
            
            start_time = time.time()
            
            text = ""
            pages_info = []
            all_tables = []
            
            with pdfplumber.open(self.pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages[:3]):
                    # Extract text
                    page_text = page.extract_text() or ""
                    text += page_text + "\n"
                    
                    # Extract tables
                    tables = page.extract_tables()
                    page_tables = []
                    
                    for table_idx, table in enumerate(tables):
                        if table:
                            table_info = {
                                "table_index": table_idx,
                                "rows": len(table),
                                "columns": len(table[0]) if table else 0,
                                "preview": table[:3] if len(table) > 3 else table
                            }
                            page_tables.append(table_info)
                            all_tables.append(table)
                    
                    pages_info.append({
                        "page": page_num + 1,
                        "chars": len(page_text),
                        "tables": len(page_tables),
                        "preview": page_text[:200] + "..." if len(page_text) > 200 else page_text
                    })
            
            extraction_time = time.time() - start_time
            
            return {
                "success": True,
                "time": extraction_time,
                "total_chars": len(text),
                "pages_processed": len(pages_info),
                "tables_found": len(all_tables),
                "pages_info": pages_info,
                "full_text": text,
                "library_version": pdfplumber.__version__
            }
            
        except Exception as e:
            return {"success": False, "error": str(e), "traceback": traceback.format_exc()}
    
    def test_unstructured(self) -> Dict[str, Any]:
        """Test unstructured - Semantic document processing"""
        try:
            from unstructured.partition.pdf import partition_pdf
            
            start_time = time.time()
            
            # Partition the PDF into semantic elements
            elements = partition_pdf(filename=self.pdf_path)
            extraction_time = time.time() - start_time
            
            # Analyze elements
            element_types = {}
            text_content = []
            tables_found = []
            
            for element in elements:
                element_type = str(type(element).__name__)
                element_types[element_type] = element_types.get(element_type, 0) + 1
                
                if hasattr(element, 'text'):
                    text_content.append(element.text)
                    
                # Check if it's a table
                if 'table' in element_type.lower():
                    tables_found.append(str(element))
            
            full_text = "\n".join(text_content)
            
            return {
                "success": True,
                "time": extraction_time,
                "total_elements": len(elements),
                "element_types": element_types,
                "total_chars": len(full_text),
                "tables_found": len(tables_found),
                "full_text": full_text,
                "library_version": "unstructured"
            }
            
        except Exception as e:
            return {"success": False, "error": str(e), "traceback": traceback.format_exc()}
    
    def test_pymupdf4llm(self) -> Dict[str, Any]:
        """Test pymupdf4llm - Markdown generation for LLMs"""
        try:
            import pymupdf4llm
            
            start_time = time.time()
            markdown_text = pymupdf4llm.to_markdown(self.pdf_path)
            extraction_time = time.time() - start_time
            
            # Analyze markdown structure
            lines = markdown_text.split('\n')
            headers = [line for line in lines if line.startswith('#')]
            tables = [line for line in lines if '|' in line and line.strip().startswith('|')]
            
            return {
                "success": True,
                "time": extraction_time,
                "total_chars": len(markdown_text),
                "total_lines": len(lines),
                "headers_found": len(headers),
                "table_rows_found": len(tables),
                "full_text": markdown_text,
                "library_version": "pymupdf4llm"
            }
            
        except Exception as e:
            return {"success": False, "error": str(e), "traceback": traceback.format_exc()}
    
    def test_pdfminer(self) -> Dict[str, Any]:
        """Test pdfminer.six - Detailed text extraction"""
        try:
            from pdfminer.high_level import extract_text
            
            start_time = time.time()
            text = extract_text(self.pdf_path)
            extraction_time = time.time() - start_time
            
            return {
                "success": True,
                "time": extraction_time,
                "total_chars": len(text),
                "full_text": text,
                "library_version": "pdfminer.six"
            }
            
        except Exception as e:
            return {"success": False, "error": str(e), "traceback": traceback.format_exc()}
    
    def _looks_like_table(self, text: str) -> bool:
        """Heuristic to detect if text looks like a table"""
        if len(text) < 50:
            return False
            
        lines = text.strip().split('\n')
        if len(lines) < 2:
            return False
            
        # Check for consistent spacing or separators
        separator_chars = ['|', '\t', '  ']
        for sep in separator_chars:
            if all(sep in line for line in lines[:3]):
                return True
                
        # Check for number patterns
        number_lines = sum(1 for line in lines if any(c.isdigit() for c in line))
        return number_lines / len(lines) > 0.5
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all extraction tests"""
        print(f"🧪 Testing PDF extraction with: {os.path.basename(self.pdf_path)}")
        print("=" * 80)
        
        tests = [
            ("PyPDF2", self.test_pypdf2),
            ("PyMuPDF", self.test_pymupdf),
            ("pdfplumber", self.test_pdfplumber),
            ("Camelot", self.test_camelot),
            ("unstructured", self.test_unstructured),
            ("pymupdf4llm", self.test_pymupdf4llm),
            ("pdfminer", self.test_pdfminer),
        ]
        
        results = {}
        
        for test_name, test_func in tests:
            print(f"\n🔍 Testing {test_name}...")
            try:
                result = test_func()
                results[test_name] = result
                
                if result["success"]:
                    print(f"✅ {test_name}: {result['time']:.3f}s - {result.get('total_chars', 0)} chars")
                    if 'tables_found' in result:
                        print(f"   📊 Tables found: {result['tables_found']}")
                    if 'images_found' in result:
                        print(f"   🖼️ Images found: {result['images_found']}")
                else:
                    print(f"❌ {test_name}: {result['error']}")
                    
            except Exception as e:
                print(f"💥 {test_name}: Critical error - {e}")
                results[test_name] = {"success": False, "error": str(e)}
        
        return results
    
    def generate_report(self, results: Dict[str, Any]) -> str:
        """Generate a comprehensive comparison report"""
        report = []
        report.append("# PDF Extraction Library Comparison Report")
        report.append(f"**Document:** {os.path.basename(self.pdf_path)}")
        report.append(f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Performance summary
        report.append("## Performance Summary")
        report.append("| Library | Status | Time (s) | Text (chars) | Tables | Special Features |")
        report.append("|---------|--------|----------|--------------|--------|------------------|")
        
        for lib_name, result in results.items():
            if result["success"]:
                status = "✅"
                time_str = f"{result['time']:.3f}"
                chars = result.get('total_chars', 0)
                tables = result.get('tables_found', 0)
                
                features = []
                if result.get('images_found', 0) > 0:
                    features.append(f"{result['images_found']} images")
                if result.get('element_types'):
                    features.append("semantic parsing")
                if result.get('headers_found', 0) > 0:
                    features.append("markdown structure")
                    
                special = ", ".join(features) if features else "-"
                
            else:
                status = "❌"
                time_str = "-"
                chars = 0
                tables = 0
                special = result.get('error', 'Unknown error')[:50]
            
            report.append(f"| {lib_name} | {status} | {time_str} | {chars} | {tables} | {special} |")
        
        # Detailed results
        report.append("\n## Detailed Results")
        
        for lib_name, result in results.items():
            report.append(f"\n### {lib_name}")
            
            if result["success"]:
                report.append(f"- ⏱️ **Extraction time:** {result['time']:.3f} seconds")
                report.append(f"- 📝 **Text extracted:** {result.get('total_chars', 0)} characters")
                
                if 'pages_processed' in result:
                    report.append(f"- 📄 **Pages processed:** {result['pages_processed']}")
                
                if 'tables_found' in result and result['tables_found'] > 0:
                    report.append(f"- 📊 **Tables found:** {result['tables_found']}")
                
                if 'images_found' in result and result['images_found'] > 0:
                    report.append(f"- 🖼️ **Images found:** {result['images_found']}")
                
                if 'element_types' in result:
                    report.append("- 🏷️ **Element types:** " + ", ".join(f"{k}({v})" for k, v in result['element_types'].items()))
                
                # Show text preview
                text_preview = result.get('full_text', '')[:300]
                if text_preview:
                    report.append(f"- 👀 **Text preview:** {text_preview}...")
            else:
                report.append(f"- ❌ **Error:** {result['error']}")
        
        # Recommendations
        report.append("\n## Recommendations")
        
        successful_results = {k: v for k, v in results.items() if v["success"]}
        
        if successful_results:
            # Fastest
            fastest = min(successful_results.items(), key=lambda x: x[1]["time"])
            report.append(f"- 🏃 **Fastest extraction:** {fastest[0]} ({fastest[1]['time']:.3f}s)")
            
            # Most text
            most_text = max(successful_results.items(), key=lambda x: x[1].get("total_chars", 0))
            report.append(f"- 📝 **Most text extracted:** {most_text[0]} ({most_text[1].get('total_chars', 0)} chars)")
            
            # Best for tables
            table_results = {k: v for k, v in successful_results.items() if v.get("tables_found", 0) > 0}
            if table_results:
                best_tables = max(table_results.items(), key=lambda x: x[1]["tables_found"])
                report.append(f"- 📊 **Best for tables:** {best_tables[0]} ({best_tables[1]['tables_found']} tables)")
        
        return "\n".join(report)


def main():
    """Main testing function"""
    # Install missing packages
    print("📦 Checking and installing required packages...")
    install_missing_packages()
    
    # Find PDF files to test
    pdf_dir = Path(__file__).parent.parent / "data" / "raw"
    pdf_files = list(pdf_dir.glob("*.pdf"))
    
    if not pdf_files:
        print(f"❌ No PDF files found in {pdf_dir}")
        print("Please add some PDF files to test with.")
        return
    
    # Test with first PDF found
    test_pdf = pdf_files[0]
    print(f"\n🎯 Testing with: {test_pdf.name}")
    
    # Run tests
    tester = PDFExtractorTester(str(test_pdf))
    results = tester.run_all_tests()
    
    # Generate and save report
    report = tester.generate_report(results)
    
    # Save results
    results_dir = Path(__file__).parent.parent / "data" / "clean"
    results_dir.mkdir(exist_ok=True)
    
    # Save JSON results
    json_file = results_dir / "pdf_extraction_results.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Save markdown report
    report_file = results_dir / "pdf_extraction_report.md"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n📊 Results saved to:")
    print(f"   JSON: {json_file}")
    print(f"   Report: {report_file}")
    
    print("\n" + "="*80)
    print("SUMMARY REPORT")
    print("="*80)
    print(report)


if __name__ == "__main__":
    main()