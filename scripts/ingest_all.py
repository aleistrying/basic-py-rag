#!/usr/bin/env python3
"""
Complete ingest pipeline - does everything in one go.
Runs: PDF cleaning → Chunking → Embedding → Upsert to backends.
"""
from chunker import chunk_all_clean_files
from pdf_cleaner import clean_all_pdfs
from ingest_config import *
import gc  # For garbage collection
import sys
import time
from pathlib import Path

# Add current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))


def check_dependencies():
    """Check if all required dependencies are available"""
    print("🔍 Checking dependencies...")

    issues = []

    # Check directories
    if not Path(RAW_DIR).exists():
        issues.append(f"Raw directory missing: {RAW_DIR}")

    # Check for PDF files
    raw_path = Path(RAW_DIR)
    if raw_path.exists():
        pdf_files = list(raw_path.glob("*.pdf"))
        if not pdf_files:
            issues.append(f"No PDF files found in {RAW_DIR}")
        else:
            print(f"   📚 Found {len(pdf_files)} PDF files")

    # Check embedding model
    try:
        from sentence_transformers import SentenceTransformer
        print(f"   🤖 Embedding model: {EMBED_MODEL}")
    except ImportError:
        issues.append("sentence-transformers not installed")

    # Check backend availability
    if USE_QDRANT:
        try:
            from qdrant_client import QdrantClient
            print(f"   🗄️  Qdrant client available")
        except ImportError:
            issues.append("qdrant-client not installed but USE_QDRANT=True")

    if USE_PGVECTOR:
        try:
            import psycopg2
            print(f"   � PostgreSQL client available")
        except ImportError:
            issues.append("psycopg2 not installed but USE_PGVECTOR=True")

    if issues:
        print("❌ Dependency issues found:")
        for issue in issues:
            print(f"   • {issue}")
        return False

    print("✅ All dependencies satisfied")
    return True


def print_pipeline_config():
    """Show current pipeline configuration"""
    print("\n⚙️  Pipeline Configuration")
    print("=" * 50)
    print(f"📁 Directories:")
    print(f"   Raw:   {RAW_DIR}")
    print(f"   Clean: {CLEAN_DIR}")

    print(f"\n🤖 Embedding:")
    print(f"   Model:      {EMBED_MODEL}")
    print(f"   Dimensions: {QDRANT_VECTOR_SIZE}")
    print(f"   Distance:   {QDRANT_DISTANCE}")
    print(f"   Prefixes:   '{E5_QUERY_PREFIX}' / '{E5_PASSAGE_PREFIX}'")

    print(f"\n✂️  Chunking:")
    print(f"   Tokens:  {CHUNK_TOKENS}")
    print(f"   Overlap: {CHUNK_OVERLAP}")

    print(f"\n🗄️  Backends:")
    qdrant_status = "✅ Enabled" if USE_QDRANT else "❌ Disabled"
    pg_status = "✅ Enabled" if USE_PGVECTOR else "❌ Disabled"
    print(f"   Qdrant:     {qdrant_status} → {QDRANT_COLLECTION}")
    print(f"   PostgreSQL: {pg_status} → {PG_TABLE}")


def run_full_pipeline(skip_existing=True):
    """
    Run the complete ingest pipeline with memory optimization.

    Args:
        skip_existing: Skip steps if output files already exist
    """
    print("🚀 Starting Complete Ingest Pipeline (Memory Optimized)")
    print("=" * 60)

    start_time = time.time()

    try:
        # Step 1: Clean PDFs (force re-clean if chunks are empty)
        print("\n📋 STEP 1: PDF Cleaning")
        print("-" * 30)

        clean_path = Path(CLEAN_DIR)
        existing_clean = list(clean_path.glob("*.jsonl")
                              ) if clean_path.exists() else []
        existing_clean = [
            f for f in existing_clean if not f.name.endswith(".chunks.jsonl")]

        # Check if existing chunks are empty (corrupted)
        force_reclean = False
        if existing_clean:
            for clean_file in existing_clean:
                chunk_file = clean_file.with_suffix(".chunks.jsonl")
                if chunk_file.exists():
                    # Check if chunks are empty
                    try:
                        with open(chunk_file, 'r') as f:
                            if sum(1 for _ in f) == 0:
                                print(
                                    f"⚠️  Empty chunks detected in {chunk_file.name} - forcing re-clean")
                                force_reclean = True
                                break
                    except:
                        force_reclean = True
                        break

        if not skip_existing or not existing_clean or force_reclean:
            if force_reclean:
                # Remove corrupted files
                for clean_file in existing_clean:
                    chunk_file = clean_file.with_suffix(".chunks.jsonl")
                    if chunk_file.exists():
                        chunk_file.unlink()
                    clean_file.unlink()
                print("🧹 Removed corrupted files")
            clean_all_pdfs()
        else:
            print(
                f"⏩ Skipping PDF cleaning - {len(existing_clean)} clean files exist")

        # Step 2: Chunking
        print("\n✂️  STEP 2: Text Chunking")
        print("-" * 30)

        existing_chunks = list(clean_path.glob(
            "*.chunks.jsonl")) if clean_path.exists() else []

        if skip_existing and existing_chunks and not force_reclean:
            print(
                f"⏩ Skipping chunking - {len(existing_chunks)} chunk files exist")
        else:
            chunk_all_clean_files()

        # Step 3: Embedding & Upsert (memory safe version)
        print("\n🤖 STEP 3: Embedding & Backend Upsert (Memory Safe)")
        print("-" * 30)

        # Use memory-safe embedding instead
        from embed_safe import safe_embed_and_upsert
        if not safe_embed_and_upsert():
            print("❌ Memory-safe embedding failed")
            return False

        # Force garbage collection after embedding
        gc.collect()

        # Pipeline complete
        end_time = time.time()
        duration = end_time - start_time

        print("\n" + "=" * 60)
        print("🎉 PIPELINE COMPLETE!")
        print("=" * 60)
        print(f"⏱️  Total time: {duration:.1f} seconds")

        # Show final statistics
        show_pipeline_stats()

        return True

    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def show_pipeline_stats():
    """Show statistics about the completed pipeline"""
    print("\n� Pipeline Statistics")
    print("-" * 30)

    try:
        clean_path = Path(CLEAN_DIR)

        # Count files at each stage
        pdf_files = list(Path(RAW_DIR).glob("*.pdf")
                         ) if Path(RAW_DIR).exists() else []
        clean_files = list(clean_path.glob("*.jsonl")
                           ) if clean_path.exists() else []
        clean_files = [
            f for f in clean_files if not f.name.endswith(".chunks.jsonl")]
        chunk_files = list(clean_path.glob("*.chunks.jsonl")
                           ) if clean_path.exists() else []

        print(f"📚 Source PDFs:    {len(pdf_files)}")
        print(f"🧹 Clean files:    {len(clean_files)}")
        print(f"✂️  Chunk files:    {len(chunk_files)}")

        # Count total chunks
        total_chunks = 0
        for chunk_file in chunk_files:
            try:
                with open(chunk_file, "r") as f:
                    file_chunks = sum(1 for _ in f)
                    total_chunks += file_chunks
            except:
                pass

        print(f"� Total chunks:   {total_chunks}")

        # Backend status
        if USE_QDRANT:
            print(f"🗄️  Qdrant:        {QDRANT_COLLECTION}")
        if USE_PGVECTOR:
            print(f"🐘 PostgreSQL:    {PG_TABLE}")

    except Exception as e:
        print(f"❌ Stats error: {e}")


def clean_pipeline():
    """Clean all pipeline outputs - start fresh"""
    print("🧹 Cleaning Pipeline Outputs")
    print("-" * 30)

    import shutil

    # Clean directory
    clean_path = Path(CLEAN_DIR)
    if clean_path.exists():
        try:
            shutil.rmtree(clean_path)
            print(f"✅ Removed {CLEAN_DIR}")
        except Exception as e:
            print(f"❌ Could not remove {CLEAN_DIR}: {e}")

    print("🧹 Clean complete - ready for fresh pipeline run")


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Complete RAG Ingest Pipeline")
    parser.add_argument("--config", action="store_true",
                        help="Show configuration and exit")
    parser.add_argument("--stats", action="store_true",
                        help="Show statistics and exit")
    parser.add_argument("--clean", action="store_true",
                        help="Clean all outputs and exit")
    parser.add_argument("--force", action="store_true",
                        help="Force re-run all steps")
    parser.add_argument("--check", action="store_true",
                        help="Check dependencies and exit")

    args = parser.parse_args()

    if args.config:
        print_pipeline_config()
        return

    if args.stats:
        show_pipeline_stats()
        return

    if args.clean:
        clean_pipeline()
        return

    if args.check:
        if check_dependencies():
            print("✅ Ready to run pipeline")
            sys.exit(0)
        else:
            print("❌ Dependencies not satisfied")
            sys.exit(1)

    # Show configuration
    print_pipeline_config()

    # Check dependencies
    if not check_dependencies():
        print("\n❌ Cannot proceed - fix dependency issues first")
        sys.exit(1)

    # Run the pipeline
    skip_existing = not args.force
    success = run_full_pipeline(skip_existing=skip_existing)

    if success:
        print("\n🎯 Next Steps:")
        print("   • Test: python test_clean_pipeline.py")
        print("   • Demo: python demo_clean_pipeline.py")
        print("   • Start API: python -m app.main")
        print("   • Query: curl 'http://localhost:8080/ask?q=¿Cuáles son las nubes?'")
        sys.exit(0)
    else:
        print("\n❌ Pipeline failed - check errors above")
        sys.exit(1)


if __name__ == "__main__":
    main()
