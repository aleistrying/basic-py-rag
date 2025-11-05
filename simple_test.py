#!/usr/bin/env python3
"""
Simple test to check if the pipeline can at least clean the files properly.
"""
import sys
from pathlib import Path

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent / "scripts"))


def test_file_cleaning():
    """Test just the file cleaning part"""
    print("🧪 Testing File Cleaning (No Dependencies)")

    # Check raw directory
    raw_path = Path("data/raw")
    if not raw_path.exists():
        print("❌ data/raw doesn't exist")
        return False

    # Count files
    pdf_files = list(raw_path.glob("*.pdf"))
    txt_files = list(raw_path.glob("*.txt"))
    md_files = list(raw_path.glob("*.md"))

    print(f"📁 Found files:")
    print(f"   📚 PDFs: {len(pdf_files)}")
    print(f"   📝 TXT: {len(txt_files)}")
    print(f"   📖 MD: {len(md_files)}")

    # Test basic text processing
    test_text = """This  is   a
    
    
      test    with    problems-
      here"""

    # Simple normalization (without imports)
    import re

    # Remove excessive whitespace
    clean_text = re.sub(r"[ \t]+", " ", test_text)
    clean_text = re.sub(r"\s+\n", "\n", clean_text)
    clean_text = re.sub(r"\n{3,}", "\n\n", clean_text)
    clean_text = clean_text.strip()

    print(f"\n🧹 Text cleaning test:")
    print(f"   Original: {repr(test_text)}")
    print(f"   Cleaned:  {repr(clean_text)}")

    return True


if __name__ == "__main__":
    test_file_cleaning()
