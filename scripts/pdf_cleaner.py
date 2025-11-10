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


def remove_watermarks(text: str) -> str:
    """
    Aggressively remove watermarks and PDF corruption artifacts.
    """
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
        text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.MULTILINE)

    # AGGRESSIVE: Remove scattered letter patterns like "U C - F 2 S 2 5 P"
    # These are OCR artifacts from watermarks
    corrupted_patterns = [
        r"\b[A-Z]\s+[A-Z]\s+[-–]\s+[A-Z]\s+\d+\s+[A-Z]\b",  # "U C - F 2 S"
        r"\b\d+\s+[A-Z]\s+\d+\s+[A-Z]\s+\d+\s+[A-Z]\b",     # "5 2 0 2 S 2"
        r"\b[A-Z]\s+\d+\s+[A-Z]\s+\d+\s+[A-Z]\b",           # "S 2 C 5 S"
        # Multiple spaced caps + number
        r"(?:[A-Z]\s+){3,}\d+",
        # Multiple spaced numbers + letter
        r"(?:\d+\s+){3,}[A-Z]",
    ]

    for pattern in corrupted_patterns:
        text = re.sub(pattern, " ", text, flags=re.IGNORECASE)

    # Remove lines with high corruption ratio
    lines = text.split('\n')
    clean_lines = []

    for line in lines:
        line = line.strip()

        # Skip if too short
        if len(line) < 10:
            continue

        # Count corruption indicators
        corruption_count = 0
        total_chars = len(line)

        # Count single letters with spaces: "A B C D"
        corruption_count += len(re.findall(r'\b[A-Z]\b', line))

        # Count isolated numbers: " 2 " " 5 "
        corruption_count += len(re.findall(r'\b\d\b', line))

        # Count excessive dashes/hyphens
        corruption_count += line.count('-') + line.count('–')

        # If more than 30% corruption, skip the line
        corruption_ratio = corruption_count / max(total_chars, 1)
        if corruption_ratio > 0.3:
            continue

        # Skip lines that are mostly caps and numbers with spaces
        if re.match(r'^[A-Z0-9\s\-–]+$', line) and len(line.split()) > 5:
            continue

        clean_lines.append(line)

    return '\n'.join(clean_lines)


def normalize_text(text: str) -> str:
    """
    Normalize extracted PDF text:
    - Remove watermarks and institutional marks
    - Remove control characters
    - Collapse excessive whitespace
    - Keep diacritics (important for Spanish)
    - Preserve paragraph structure
    """
    if not text:
        return ""

    # First remove watermarks
    text = remove_watermarks(text)

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
    # Multiple newlines → double newline
    text = re.sub(r"\n{3,}", "\n\n", text)

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
                    logging.warning(
                        f"Failed to extract page {i} from {pdf_path.name}: {e}")
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
                logging.warning(
                    f"Failed to extract page {i+1} from {pdf_path.name}: {e}")
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
                    logging.warning(
                        f"Failed to extract page {i+1} from {pdf_path.name}: {e}")
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


def text_file_to_clean_jsonl(text_path: Path) -> Optional[Path]:
    """Convert text/markdown file to clean JSONL format"""
    try:
        # Read the text file
        with open(text_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Basic text cleaning
        clean_content = normalize_text(content)

        if len(clean_content.strip()) < MIN_CHARS:
            print(
                f"  ⚠️  Skipping {text_path.name} - too short ({len(clean_content)} chars)")
            return None

        # Create output path
        output_path = Path(CLEAN_DIR) / f"{text_path.stem}.jsonl"

        # Write JSONL
        with open(output_path, 'w', encoding='utf-8') as f:
            record = {
                "source_path": str(text_path),
                "page": 1,  # Single page for text files
                "text": f"Text File: {text_path.name}\n\n{clean_content}",
                "extractor": "text_reader"
            }
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

        print(f"  ✅ Extracted {len(clean_content)} characters")
        return output_path

    except Exception as e:
        print(f"  ❌ Error processing {text_path.name}: {e}")
        return None


def clean_all_files():
    """Process all files (PDFs, txt, md) in the raw directory"""
    raw_path = Path(RAW_DIR)
    if not raw_path.exists():
        print(f"❌ Raw directory {RAW_DIR} does not exist!")
        return

    # Find all supported files
    pdf_files = list(raw_path.glob("*.pdf"))
    txt_files = list(raw_path.glob("*.txt"))
    md_files = list(raw_path.glob("*.md"))

    all_files = pdf_files + txt_files + md_files

    if not all_files:
        print(f"❌ No supported files (PDF, TXT, MD) found in {RAW_DIR}")
        return

    print(f"📋 Processing {len(all_files)} files:")
    print(f"   📚 PDFs: {len(pdf_files)}")
    print(f"   📝 Text files: {len(txt_files)}")
    print(f"   📖 Markdown files: {len(md_files)}")

    total_processed = 0

    # Process PDFs
    for pdf_file in pdf_files:
        print(f"\n📄 Processing PDF: {pdf_file.name}")
        try:
            output_file = pdf_to_clean_jsonl(pdf_file)
            if output_file:
                print(f"  ✅ Created: {output_file.name}")
                total_processed += 1
            else:
                print(f"  ❌ Failed to process {pdf_file.name}")
        except Exception as e:
            print(f"  ❌ Error: {e}")

    # Process text files
    for text_file in txt_files + md_files:
        print(f"\n📝 Processing text: {text_file.name}")
        try:
            output_file = text_file_to_clean_jsonl(text_file)
            if output_file:
                print(f"  ✅ Created: {output_file.name}")
                total_processed += 1
            else:
                print(f"  ❌ Failed to process {text_file.name}")
        except Exception as e:
            print(f"  ❌ Error: {e}")

    print(f"\n✅ File processing complete!")
    print(f"📁 {total_processed} clean files saved to: {CLEAN_DIR}")


def clean_all_pdfs():
    """Process all PDFs - now using enhanced extraction"""
    print("🔄 Using enhanced PDF extraction...")

    # Import the enhanced cleaner
    try:
        from enhanced_pdf_cleaner import clean_all_pdfs_enhanced
        clean_all_pdfs_enhanced()
    except ImportError as e:
        print(f"❌ Enhanced PDF cleaner not available: {e}")
        print("📁 Falling back to basic PDF cleaning...")
        clean_all_files()


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.WARNING)

    # Clean all PDFs
    clean_all_pdfs()
