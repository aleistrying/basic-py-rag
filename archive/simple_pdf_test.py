#!/usr/bin/env python3
"""
Simple PDF Extraction Test with Available Libraries
Tests the PDF extraction libraries that are already installed.
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import traceback

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_available_extractors(pdf_path: str) -> Dict[str, Any]:
    """Test available PDF extraction libraries"""
    
    results = {}
    
    print(f"🧪 Testing PDF extraction with: {os.path.basename(pdf_path)}")
    print("=" * 80)
    
    # Test PyPDF2
    print("\n🔍 Testing PyPDF2...")
    try:
        import PyPDF2
        
        start_time = time.time()
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            
            for page_num, page in enumerate(reader.pages[:3]):  # First 3 pages
                page_text = page.extract_text()
                text += page_text + "\n"
                
        extraction_time = time.time() - start_time
        
        results["PyPDF2"] = {
            "success": True,
            "time": extraction_time,
            "text_length": len(text),
            "preview": text[:300] + "..." if len(text) > 300 else text,
            "version": PyPDF2.__version__
        }
        print(f"✅ PyPDF2: {extraction_time:.3f}s - {len(text)} chars")
        
    except ImportError:
        results["PyPDF2"] = {"success": False, "error": "Library not installed"}
        print("❌ PyPDF2: Not installed")
    except Exception as e:
        results["PyPDF2"] = {"success": False, "error": str(e)}
        print(f"❌ PyPDF2: {e}")
    
    # Test PyMuPDF
    print("\n🔍 Testing PyMuPDF...")
    try:
        import fitz
        
        start_time = time.time()
        doc = fitz.open(pdf_path)
        
        text = ""
        tables_found = 0
        images_found = 0
        
        for page_num in range(min(3, doc.page_count)):
            page = doc[page_num]
            page_text = page.get_text()
            text += page_text + "\n"
            
            # Count images
            image_list = page.get_images()
            images_found += len(image_list)
            
            # Look for table-like blocks
            blocks = page.get_text("blocks")
            for block in blocks:
                if len(block) >= 5:
                    block_text = block[4]
                    if looks_like_table(block_text):
                        tables_found += 1
        
        doc.close()
        extraction_time = time.time() - start_time
        
        results["PyMuPDF"] = {
            "success": True,
            "time": extraction_time,
            "text_length": len(text),
            "tables_found": tables_found,
            "images_found": images_found,
            "preview": text[:300] + "..." if len(text) > 300 else text,
            "version": fitz.version[0]
        }
        print(f"✅ PyMuPDF: {extraction_time:.3f}s - {len(text)} chars, {tables_found} tables, {images_found} images")
        
    except ImportError:
        results["PyMuPDF"] = {"success": False, "error": "Library not installed"}
        print("❌ PyMuPDF: Not installed")
    except Exception as e:
        results["PyMuPDF"] = {"success": False, "error": str(e)}
        print(f"❌ PyMuPDF: {e}")
    
    # Test pdfplumber
    print("\n🔍 Testing pdfplumber...")
    try:
        import pdfplumber
        
        start_time = time.time()
        text = ""
        tables_found = 0
        
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages[:3]):
                page_text = page.extract_text() or ""
                text += page_text + "\n"
                
                # Extract tables
                page_tables = page.extract_tables()
                tables_found += len([t for t in page_tables if t])
        
        extraction_time = time.time() - start_time
        
        results["pdfplumber"] = {
            "success": True,
            "time": extraction_time,
            "text_length": len(text),
            "tables_found": tables_found,
            "preview": text[:300] + "..." if len(text) > 300 else text,
            "version": pdfplumber.__version__
        }
        print(f"✅ pdfplumber: {extraction_time:.3f}s - {len(text)} chars, {tables_found} tables")
        
    except ImportError:
        results["pdfplumber"] = {"success": False, "error": "Library not installed"}
        print("❌ pdfplumber: Not installed")
    except Exception as e:
        results["pdfplumber"] = {"success": False, "error": str(e)}
        print(f"❌ pdfplumber: {e}")
    
    # Test pdfminer
    print("\n🔍 Testing pdfminer.six...")
    try:
        from pdfminer.high_level import extract_text
        
        start_time = time.time()
        text = extract_text(pdf_path)
        extraction_time = time.time() - start_time
        
        results["pdfminer"] = {
            "success": True,
            "time": extraction_time,
            "text_length": len(text),
            "preview": text[:300] + "..." if len(text) > 300 else text,
            "version": "pdfminer.six"
        }
        print(f"✅ pdfminer: {extraction_time:.3f}s - {len(text)} chars")
        
    except ImportError:
        results["pdfminer"] = {"success": False, "error": "Library not installed"}
        print("❌ pdfminer: Not installed")
    except Exception as e:
        results["pdfminer"] = {"success": False, "error": str(e)}
        print(f"❌ pdfminer: {e}")
    
    return results

def looks_like_table(text: str) -> bool:
    """Simple heuristic to detect table-like text"""
    if len(text) < 50:
        return False
        
    lines = text.strip().split('\n')
    if len(lines) < 2:
        return False
        
    # Check for separators or consistent spacing
    separator_lines = sum(1 for line in lines if any(sep in line for sep in ['|', '\t', '  ']))
    return separator_lines / len(lines) > 0.5

def analyze_watermark_handling(results: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze how well each library handles watermarks"""
    
    analysis = {}
    
    # Common watermark artifacts to look for
    watermark_artifacts = [
        'ProyeFc',
        'InvestigacMn', 
        'EvaluacMn',
        'DesarMllo',
        'AnBlisis',
        'MaestLra',
        'UniversMdad'
    ]
    
    for lib_name, result in results.items():
        if result.get("success") and "preview" in result:
            text = result["preview"]
            
            artifacts_found = []
            for artifact in watermark_artifacts:
                if artifact in text:
                    artifacts_found.append(artifact)
            
            analysis[lib_name] = {
                "watermark_artifacts": artifacts_found,
                "artifact_count": len(artifacts_found),
                "text_quality_score": max(0, 10 - len(artifacts_found))  # Simple scoring
            }
    
    return analysis

def generate_comparison_report(results: Dict[str, Any], watermark_analysis: Dict[str, Any]) -> str:
    """Generate a comparison report"""
    
    report = []
    report.append("# PDF Extraction Library Comparison")
    report.append(f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    # Performance table
    report.append("## Performance Comparison")
    report.append("| Library | Status | Time (s) | Text Length | Tables | Images | Quality Score |")
    report.append("|---------|--------|----------|-------------|--------|--------|---------------|")
    
    for lib_name, result in results.items():
        if result.get("success"):
            status = "✅"
            time_str = f"{result['time']:.3f}"
            text_len = result.get('text_length', 0)
            tables = result.get('tables_found', 0)
            images = result.get('images_found', 0)
            quality = watermark_analysis.get(lib_name, {}).get('text_quality_score', 0)
        else:
            status = "❌"
            time_str = "-"
            text_len = 0
            tables = 0
            images = 0
            quality = 0
        
        report.append(f"| {lib_name} | {status} | {time_str} | {text_len} | {tables} | {images} | {quality}/10 |")
    
    # Watermark analysis
    report.append("\n## Watermark Handling Analysis")
    
    for lib_name, analysis in watermark_analysis.items():
        report.append(f"\n### {lib_name}")
        if analysis['artifact_count'] == 0:
            report.append("✅ **Clean extraction** - No watermark artifacts detected")
        else:
            report.append(f"⚠️ **Artifacts found:** {analysis['artifact_count']}")
            for artifact in analysis['watermark_artifacts']:
                report.append(f"   - `{artifact}`")
    
    # Recommendations
    report.append("\n## Recommendations")
    
    successful_libs = {k: v for k, v in results.items() if v.get("success")}
    
    if successful_libs:
        # Best overall
        best_quality = max(watermark_analysis.items(), key=lambda x: x[1].get('text_quality_score', 0))
        report.append(f"- 🏆 **Best text quality:** {best_quality[0]} (Score: {best_quality[1]['text_quality_score']}/10)")
        
        # Fastest
        fastest = min(successful_libs.items(), key=lambda x: x[1]['time'])
        report.append(f"- ⚡ **Fastest extraction:** {fastest[0]} ({fastest[1]['time']:.3f}s)")
        
        # Best for tables
        table_libs = {k: v for k, v in successful_libs.items() if v.get('tables_found', 0) > 0}
        if table_libs:
            best_tables = max(table_libs.items(), key=lambda x: x[1]['tables_found'])
            report.append(f"- 📊 **Best for tables:** {best_tables[0]} ({best_tables[1]['tables_found']} tables)")
    
    # Text previews
    report.append("\n## Text Extraction Previews")
    
    for lib_name, result in results.items():
        if result.get("success") and "preview" in result:
            report.append(f"\n### {lib_name}")
            report.append("```")
            report.append(result["preview"])
            report.append("```")
    
    return "\n".join(report)

def main():
    """Main function"""
    
    # Find PDF files
    pdf_dir = Path(__file__).parent.parent / "data" / "raw"
    pdf_files = list(pdf_dir.glob("*.pdf"))
    
    if not pdf_files:
        print(f"❌ No PDF files found in {pdf_dir}")
        return
    
    # Test with first PDF
    test_pdf = pdf_files[0]
    print(f"📄 Testing with: {test_pdf.name}")
    
    # Run tests
    results = test_available_extractors(str(test_pdf))
    
    # Analyze watermark handling
    watermark_analysis = analyze_watermark_handling(results)
    
    # Generate report
    report = generate_comparison_report(results, watermark_analysis)
    
    # Save results
    output_dir = Path(__file__).parent.parent / "data" / "clean"
    output_dir.mkdir(exist_ok=True)
    
    # Save JSON
    json_file = output_dir / "pdf_extraction_simple_test.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump({
            "results": results,
            "watermark_analysis": watermark_analysis,
            "test_file": test_pdf.name
        }, f, indent=2, ensure_ascii=False)
    
    # Save report
    report_file = output_dir / "pdf_extraction_simple_report.md"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n📊 Results saved:")
    print(f"   JSON: {json_file}")
    print(f"   Report: {report_file}")
    
    print("\n" + "="*80)
    print("SUMMARY REPORT")
    print("="*80)
    print(report)

if __name__ == "__main__":
    main()