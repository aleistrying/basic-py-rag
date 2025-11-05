"""
Robust PDF cleaner that extracts clean text from PDFs into JSONL format.
This ensures consistent, page-aware text processing before chunking.
"""
import json
import re
from pathlib import Path
import logging
from typing import Optional

try:
    import pdfplumber
except ImportError:
    pdfplumber = None

try:
    import PyPDF2
except ImportError:
    PyPDF2 = None

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

from ingest_config import RAW_DIR, CLEAN_DIR, MIN_CHARS

# Ensure clean directory exists
Path(CLEAN_DIR).mkdir(parents=True, exist_ok=True)

def normalize_text(text: str) -> str:
    """
    Normalize extracted PDF text:
    - Remove control characters
    - Collapse excessive whitespace
    - Keep diacritics (important for Spanish)
    - Preserve paragraph structure
    """
    if not text:
        return ""
    
    # Remove null bytes and other control characters
    text = text.replace("\x00", " ")
    text = text.replace("\ufffd", "")  # Replacement character
    
    # Remove soft hyphens that cause word splitting
    text = text.replace("\u00ad", "")
    
    # Merge hyphenated words across line breaks
    text = re.sub(r"-\s*\n\s*", "", text)
    
    # Normalize whitespace
    text = re.sub(r"[ \t]+", " ", text)  # Multiple spaces/tabs → single space
    text = re.sub(r"\s+\n", "\n", text)  # Trailing spaces before newlines
    text = re.sub(r"\n{3,}", "\n\n", text)  # Multiple newlines → double newline
    
    return text.strip()


def extract_pdf_with_pdfplumber(pdf_path: Path) -> list:
    """Extract text using pdfplumber (best for complex layouts)"""
    pages = []
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                try:
                    text = page.extract_text()
                    if text:
                        normalized = normalize_text(text)
                        if len(normalized) >= MIN_CHARS:
                            pages.append({
                                "page": i,
                                "text": f"Page {i}:\n{normalized}",
                                "extractor": "pdfplumber"
                            })
                except Exception as e:
                    logging.warning(f"Failed to extract page {i} from {pdf_path.name}: {e}")
                    continue
    except Exception as e:
        logging.error(f"pdfplumber failed for {pdf_path.name}: {e}")
        return []
    
    return pages


def extract_pdf_with_pymupdf(pdf_path: Path) -> list:
    """Extract text using PyMuPDF (good fallback)"""
    pages = []
    
    try:
        doc = fitz.open(str(pdf_path))
        for i in range(len(doc)):
            try:
                page = doc.load_page(i)
                text = page.get_text()
                if text:
                    normalized = normalize_text(text)
                    if len(normalized) >= MIN_CHARS:
                        pages.append({
                            "page": i + 1,
                            "text": f"Page {i + 1}:\n{normalized}",
                            "extractor": "pymupdf"
                        })
            except Exception as e:
                logging.warning(f"Failed to extract page {i+1} from {pdf_path.name}: {e}")
                continue
        doc.close()
    except Exception as e:
        logging.error(f"PyMuPDF failed for {pdf_path.name}: {e}")
        return []
    
    return pages


def extract_pdf_with_pypdf2(pdf_path: Path) -> list:
    """Extract text using PyPDF2 (last resort)"""
    pages = []
    
    try:
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
                    logging.warning(f"Could not decrypt {pdf_path.name}")
                    return []
            
            for i, page in enumerate(reader.pages):
                try:
                    text = page.extract_text()
                    if text:
                        normalized = normalize_text(text)
                        if len(normalized) >= MIN_CHARS:
                            pages.append({
                                "page": i + 1,
                                "text": f"Page {i + 1}:\n{normalized}",
                                "extractor": "pypdf2"
                            })
                except Exception as e:
                    logging.warning(f"Failed to extract page {i+1} from {pdf_path.name}: {e}")
                    continue
                    
    except Exception as e:
        logging.error(f"PyPDF2 failed for {pdf_path.name}: {e}")
        return []
    
    return pages


def pdf_to_clean_jsonl(pdf_path: Path) -> Optional[Path]:
    """
    Convert PDF to clean JSONL format, trying multiple extractors.
    Each line in JSONL contains: {"source_path": str, "page": int, "text": str, "extractor": str}
    """
    output_path = Path(CLEAN_DIR) / (pdf_path.stem + ".jsonl")
    
    # Try extractors in order of preference
    extractors = []
    if pdfplumber:
        extractors.append(("pdfplumber", extract_pdf_with_pdfplumber))
    if fitz:
        extractors.append(("pymupdf", extract_pdf_with_pymupdf))
    if PyPDF2:
        extractors.append(("pypdf2", extract_pdf_with_pypdf2))
    
    if not extractors:
        logging.error("No PDF extraction libraries available!")
        return None
    
    pages = []
    for name, extractor_func in extractors:
        print(f"  Trying {name}...")
        pages = extractor_func(pdf_path)
        if pages:
            print(f"  ✅ Success with {name}: {len(pages)} pages")
            break
    else:
        print(f"  ❌ All extractors failed for {pdf_path.name}")
        return None
    
    # Write to JSONL
    try:
        with open(output_path, "w", encoding="utf-8") as out_file:
            for page_data in pages:
                record = {
                    "source_path": str(pdf_path),
                    "page": page_data["page"],
                    "text": page_data["text"],
                    "extractor": page_data["extractor"]
                }
                out_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        
        print(f"  💾 Saved to {output_path.name}")
        return output_path
        
    except Exception as e:
        logging.error(f"Failed to write JSONL for {pdf_path.name}: {e}")
        return None


def clean_all_pdfs():
    """Clean all PDFs in the raw directory"""
    raw_path = Path(RAW_DIR)
    if not raw_path.exists():
        print(f"❌ Raw directory {RAW_DIR} does not exist!")
        return
    
    pdf_files = list(raw_path.rglob("*.pdf"))
    if not pdf_files:
        print(f"❌ No PDF files found in {RAW_DIR}")
        return
    
    print(f"🧹 Cleaning {len(pdf_files)} PDF files...")
    
    success_count = 0
    for pdf_path in pdf_files:
        print(f"\n📄 Processing: {pdf_path.name}")
        result = pdf_to_clean_jsonl(pdf_path)
        if result:
            success_count += 1
    
    print(f"\n✅ Cleaned {success_count}/{len(pdf_files)} PDFs")
    print(f"📁 Clean files saved to: {CLEAN_DIR}")


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.WARNING)
    
    # Clean all PDFs
    clean_all_pdfs()