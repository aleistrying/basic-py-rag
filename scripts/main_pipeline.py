#!/usr/bin/env python3
"""
Unified Main Pipeline - Complete RAG Processing
Consolidates ingest_all.py and pipeline.py into single entry point
Runs: PDF cleaning → Chunking → Embedding → Database upsert

Usage:
    python scripts/main_pipeline.py                    # Process all files with enhanced extraction
    python scripts/main_pipeline.py --clear            # Clear databases first  
    python scripts/main_pipeline.py --memory-safe      # Use memory-safe processing
    python scripts/main_pipeline.py --basic            # Use basic extraction (legacy)
    python scripts/main_pipeline.py --config           # Show configuration
    python scripts/main_pipeline.py --stats            # Show pipeline statistics
    python scripts/main_pipeline.py --force            # Force re-processing
    python scripts/main_pipeline.py --parallel         # Enable parallel processing
    python scripts/main_pipeline.py --parallel --workers 8  # Parallel with 8 workers
"""

from ingest_config import *
from embedding_database import UnifiedEmbeddingProcessor, embed_and_upsert_all
from chunker import chunk_all_clean_files
from pdf_processing import UnifiedPDFProcessor, process_all_pdfs, process_text_files
import argparse
import gc
import json
import logging
import os
import sys
import time
import asyncio
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Dict, Any, Optional
import threading

# Add current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

# Configure logging - REDUCED noise for pipeline processing
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Reduce noise from dependencies
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("qdrant_client").setLevel(logging.WARNING)

# Import our unified modules


class UnifiedMainPipeline:
    """
    Unified main pipeline that orchestrates the complete RAG processing workflow
    """

    def __init__(self,
                 use_enhanced_extraction: bool = True,
                 memory_safe_mode: bool = True,
                 batch_size: Optional[int] = None,
                 large_docs: bool = False,
                 distance_metric: str = "cosine",
                 index_algorithm: str = "hnsw",
                 process_all_combinations: bool = True,
                 parallel_mode: bool = False,
                 max_workers: Optional[int] = None):
        self.use_enhanced_extraction = use_enhanced_extraction
        self.memory_safe_mode = memory_safe_mode
        self.batch_size = batch_size
        self.large_docs = large_docs
        self.distance_metric = distance_metric
        self.index_algorithm = index_algorithm
        self.process_all_combinations = process_all_combinations
        self.parallel_mode = parallel_mode

        # Configure parallel processing
        if parallel_mode:
            self.max_workers = max_workers or max(1, mp.cpu_count() // 2)
            logger.info(
                f"🔥 Parallel mode enabled with {self.max_workers} workers")
        else:
            self.max_workers = 1

        # Initialize PDF processor
        if use_enhanced_extraction:
            self.pdf_processor = UnifiedPDFProcessor()
        else:
            self.pdf_processor = None

        self.embedding_processor = UnifiedEmbeddingProcessor(
            memory_safe_mode=memory_safe_mode,
            batch_size=batch_size,
            large_docs=large_docs,
            distance_metric=distance_metric,
            index_algorithm=index_algorithm,
            process_all_combinations=process_all_combinations
        )

    def check_dependencies(self) -> bool:
        """Check if all required dependencies are available"""
        logger.info("🔍 Checking dependencies...")

        issues = []

        # Check directories
        if not Path(RAW_DIR).exists():
            issues.append(f"Raw directory not found: {RAW_DIR}")

        # Check for files to process
        raw_path = Path(RAW_DIR)
        if raw_path.exists():
            files = list(raw_path.glob("*.pdf")) + \
                list(raw_path.glob("*.txt")) + list(raw_path.glob("*.md")) + \
                list(raw_path.glob("*.yaml")) + list(raw_path.glob("*.yml"))
            if not files:
                issues.append(f"No files to process in {RAW_DIR}")

        # Check embedding model
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            issues.append("sentence-transformers not installed")

        # Check backend availability
        if USE_QDRANT:
            try:
                from qdrant_client import QdrantClient
            except ImportError:
                issues.append("qdrant-client not installed")

        if USE_PGVECTOR:
            try:
                import psycopg2
            except ImportError:
                issues.append("psycopg2 not installed")

        if issues:
            logger.error("❌ Dependency issues found:")
            for issue in issues:
                logger.error(f"   - {issue}")
            return False

        logger.info("✅ All dependencies satisfied")
        return True

    def print_pipeline_config(self):
        """Show current pipeline configuration"""
        logger.info("\n⚙️  Pipeline Configuration")
        logger.info("=" * 50)
        logger.info(f"📁 Directories:")
        logger.info(f"   Raw:   {RAW_DIR}")
        logger.info(f"   Clean: {CLEAN_DIR}")

        logger.info(f"\n🔧 Processing:")
        logger.info(
            f"   Enhanced Extraction: {'✅ Enabled' if self.use_enhanced_extraction else '❌ Basic only'}")
        logger.info(
            f"   Memory Safe Mode:    {'✅ Enabled' if self.memory_safe_mode else '❌ Disabled'}")
        logger.info(f"   Batch Size:          {self.batch_size or 'Default'}")

        logger.info(f"\n🤖 Embedding:")
        logger.info(f"   Model:      {EMBED_MODEL}")
        logger.info(f"   Dimensions: {QDRANT_VECTOR_SIZE}")
        logger.info(f"   Distance:   {QDRANT_DISTANCE}")
        logger.info(f"   Algorithm:  {self.index_algorithm}")
        logger.info(
            f"   All Combos: {'✅ Enabled (16 combinations)' if self.process_all_combinations else '❌ Single only'}")
        logger.info(
            f"   Prefixes:   '{E5_QUERY_PREFIX}' / '{E5_PASSAGE_PREFIX}'")

        logger.info(f"\n✂️  Chunking:")
        logger.info(f"   Tokens:  {CHUNK_TOKENS}")
        logger.info(f"   Overlap: {CHUNK_OVERLAP}")

        logger.info(f"\n🗄️  Backends:")
        qdrant_status = "✅ Enabled" if USE_QDRANT else "❌ Disabled"
        pg_status = "✅ Enabled" if USE_PGVECTOR else "❌ Disabled"
        logger.info(f"   Qdrant:     {qdrant_status} → {QDRANT_COLLECTION}")
        logger.info(f"   PostgreSQL: {pg_status} → {PG_TABLE}")

    def process_pdf_parallel(self, pdf_files: List[Path]) -> bool:
        """
        Process PDF files in parallel using multiprocessing
        """
        if not pdf_files:
            return True

        logger.info(
            f"🔄 Processing {len(pdf_files)} PDFs with {self.max_workers} parallel workers...")
        start_time = time.time()

        # Use ProcessPoolExecutor for CPU-bound PDF processing
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all PDF processing jobs
            future_to_pdf = {}
            for pdf_file in pdf_files:
                future = executor.submit(self._process_single_pdf, pdf_file)
                future_to_pdf[future] = pdf_file

            # Collect results with progress tracking
            completed = 0
            failed = 0
            for future in as_completed(future_to_pdf):
                pdf_file = future_to_pdf[future]
                try:
                    success = future.result()
                    if success:
                        completed += 1
                        logger.info(
                            f"✅ Completed: {pdf_file.name} ({completed}/{len(pdf_files)})")
                    else:
                        failed += 1
                        logger.warning(f"⚠️  Failed: {pdf_file.name}")
                except Exception as e:
                    failed += 1
                    logger.error(f"❌ Error processing {pdf_file.name}: {e}")

        elapsed = time.time() - start_time
        logger.info(f"🎉 Parallel PDF processing complete!")
        logger.info(f"   ✅ Success: {completed}/{len(pdf_files)} files")
        logger.info(
            f"   ⏱️  Time: {elapsed:.1f}s (avg: {elapsed/len(pdf_files):.1f}s per file)")
        logger.info(f"   🔥 Speedup: ~{self.max_workers:.1f}x theoretical")

        return failed == 0

    def _process_single_pdf(self, pdf_file: Path) -> bool:
        """
        Process a single PDF file - used by parallel processing
        This runs in a separate process
        """
        try:
            # Import here to avoid pickling issues in multiprocessing
            from pdf_processing import UnifiedPDFProcessor

            # Create processor instance and process the PDF
            processor = UnifiedPDFProcessor()
            result = processor.process_pdf_file(
                pdf_file, output_format="jsonl")

            if result:
                logger.info(f"✅ Successfully processed: {pdf_file.name}")
                return True
            else:
                logger.warning(
                    f"⚠️  Processing returned None: {pdf_file.name}")
                return False

        except Exception as e:
            logger.error(f"Error processing {pdf_file}: {e}")
            return False

    def run_full_pipeline(self,
                          skip_existing: bool = True,
                          clear_databases: bool = True,
                          force_reprocess: bool = False) -> bool:
        """
        Run the complete ingest pipeline

        Args:
            skip_existing: Skip steps if output files already exist
            clear_databases: Clear databases before processing
            force_reprocess: Force re-processing even if files exist

        Returns:
            True if successful, False otherwise
        """
        logger.info("🚀 Starting Unified Main Pipeline")
        logger.info("=" * 60)

        start_time = time.time()

        try:
            # Step 1: PDF and Text Processing
            logger.info("\n📋 STEP 1: File Processing (PDFs & Text)")
            logger.info("-" * 40)

            clean_path = Path(CLEAN_DIR)
            existing_clean = list(clean_path.glob(
                "*.jsonl")) if clean_path.exists() else []
            existing_clean = [
                f for f in existing_clean if not f.name.endswith(".chunks.jsonl")]

            # Check if we need to force re-clean due to corrupted chunks
            force_reclean = force_reprocess
            if not force_reclean and existing_clean:
                for clean_file in existing_clean:
                    chunk_file = clean_file.with_suffix(".chunks.jsonl")
                    if chunk_file.exists():
                        try:
                            with open(chunk_file, 'r') as f:
                                if sum(1 for _ in f) == 0:
                                    logger.warning(
                                        f"⚠️  Empty chunks detected - forcing re-clean")
                                    force_reclean = True
                                    break
                        except:
                            force_reclean = True
                            break

            should_process_files = (
                not skip_existing or
                not existing_clean or
                force_reclean
            )

            if should_process_files:
                if force_reclean:
                    # Remove corrupted files
                    for clean_file in existing_clean:
                        chunk_file = clean_file.with_suffix(".chunks.jsonl")
                        if chunk_file.exists():
                            chunk_file.unlink()
                        clean_file.unlink()
                    logger.info("🧹 Removed corrupted files")

                if self.use_enhanced_extraction:
                    logger.info(
                        "🔬 Using enhanced extraction (multi-library + tables)")
                    if self.pdf_processor:
                        # Get PDF files to process
                        pdf_files = list(Path(RAW_DIR).glob(
                            "*.pdf")) if Path(RAW_DIR).exists() else []

                        if pdf_files:
                            if self.parallel_mode and len(pdf_files) > 1:
                                # Use parallel processing for multiple PDFs
                                logger.info(
                                    f"🚀 Using parallel processing for {len(pdf_files)} PDFs")
                                pdf_success = self.process_pdf_parallel(
                                    pdf_files)
                                if not pdf_success:
                                    logger.warning(
                                        "⚠️  Some PDFs failed parallel processing")
                            else:
                                # Use sequential processing
                                logger.info("🔄 Using sequential processing")
                                process_all_pdfs()

                    # Process text files (usually fewer, so sequential is fine)
                    process_text_files()
                else:
                    logger.info("📄 Using basic extraction (legacy mode)")
                    # Import legacy functions for basic mode
                    from pdf_cleaner import clean_all_pdfs
                    clean_all_pdfs()
            else:
                logger.info(
                    f"⏩ Skipping file processing - {len(existing_clean)} clean files exist")

            # Step 2: Chunking
            logger.info("\n✂️  STEP 2: Text Chunking")
            logger.info("-" * 30)

            existing_chunks = list(clean_path.glob(
                "*.chunks.jsonl")) if clean_path.exists() else []

            if skip_existing and existing_chunks and not force_reprocess:
                logger.info(
                    f"⏩ Skipping chunking - {len(existing_chunks)} chunk files exist")
            else:
                chunk_all_clean_files()

            # Step 3: Embedding & Database Upsert
            logger.info("\n🤖 STEP 3: Embedding & Database Upsert")
            logger.info("-" * 40)

            success = self.embedding_processor.process_all_chunks(
                clear_first=clear_databases)
            if not success:
                logger.error("❌ Embedding and database upsert failed")
                return False

            # Force garbage collection
            gc.collect()

            # Pipeline complete
            end_time = time.time()
            duration = end_time - start_time

            logger.info("\n" + "=" * 60)
            logger.info("🎉 UNIFIED PIPELINE COMPLETE!")
            logger.info("=" * 60)
            logger.info(f"⏱️  Total processing time: {duration:.1f} seconds")

            # Show final statistics
            self.show_pipeline_stats()

            return True

        except Exception as e:
            logger.error(f"❌ Pipeline failed: {e}")
            return False
        finally:
            # Cleanup
            self.embedding_processor.cleanup()

    def show_pipeline_stats(self):
        """Show statistics about the completed pipeline"""
        logger.info("\n📊 Pipeline Statistics")
        logger.info("-" * 30)

        try:
            clean_path = Path(CLEAN_DIR)

            # Count files at each stage
            pdf_files = list(Path(RAW_DIR).glob("*.pdf")
                             ) if Path(RAW_DIR).exists() else []
            text_files = list(Path(RAW_DIR).glob(
                "*.txt")) + list(Path(RAW_DIR).glob("*.md")) + \
                list(Path(RAW_DIR).glob("*.yaml")) + \
                list(Path(RAW_DIR).glob("*.yml")
                     ) if Path(RAW_DIR).exists() else []
            clean_files = list(clean_path.glob("*.jsonl")
                               ) if clean_path.exists() else []
            clean_files = [
                f for f in clean_files if not f.name.endswith(".chunks.jsonl")]
            chunk_files = list(clean_path.glob("*.chunks.jsonl")
                               ) if clean_path.exists() else []

            logger.info(f"📚 Source PDFs:     {len(pdf_files)}")
            logger.info(f"📄 Source Text:     {len(text_files)}")
            logger.info(f"🧹 Clean files:     {len(clean_files)}")
            logger.info(f"✂️  Chunk files:     {len(chunk_files)}")

            # Count total chunks
            total_chunks = 0
            for chunk_file in chunk_files:
                try:
                    with open(chunk_file, "r") as f:
                        file_chunks = sum(1 for _ in f)
                        total_chunks += file_chunks
                except:
                    continue

            logger.info(f"📊 Total chunks:    {total_chunks}")

            # Show file details
            if pdf_files:
                logger.info(f"\n📚 PDF Files processed:")
                for pdf_file in pdf_files:
                    logger.info(f"   - {pdf_file.name}")

            if text_files:
                logger.info(f"\n📄 Text Files processed:")
                for text_file in text_files:
                    logger.info(f"   - {text_file.name}")

        except Exception as e:
            logger.error(f"Error gathering statistics: {e}")

    def clean_pipeline_outputs(self):
        """Clean all pipeline outputs - start fresh"""
        logger.info("🧹 Cleaning Pipeline Outputs")
        logger.info("-" * 30)

        # Clean directory
        clean_path = Path(CLEAN_DIR)
        if clean_path.exists():
            for file in clean_path.glob("*.jsonl"):
                file.unlink()
                logger.info(f"🗑️  Removed: {file.name}")

        # Clear databases
        if hasattr(self, 'embedding_processor'):
            self.embedding_processor.clear_databases()

        logger.info("🧹 Clean complete - ready for fresh pipeline run")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Unified Main Pipeline - Complete RAG Processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/main_pipeline.py                    # Full pipeline (all 16 combinations)
  python scripts/main_pipeline.py --clear            # Clear databases first
  python scripts/main_pipeline.py --memory-safe      # Use memory-safe processing  
  python scripts/main_pipeline.py --basic            # Use basic extraction only
  python scripts/main_pipeline.py --force            # Force re-processing
  python scripts/main_pipeline.py --single-combination # Process only single combination
        """
    )

    parser.add_argument("--config", action="store_true",
                        help="Show configuration and exit")
    parser.add_argument("--stats", action="store_true",
                        help="Show statistics and exit")
    parser.add_argument("--clean", action="store_true",
                        help="Clean all outputs and exit")
    parser.add_argument("--clear", action="store_true",
                        help="Clear databases before processing")
    parser.add_argument("--force", action="store_true",
                        help="Force re-processing even if files exist")
    parser.add_argument("--basic", action="store_true",
                        help="Use basic extraction instead of enhanced")
    parser.add_argument("--memory-safe", action="store_true", default=True,
                        help="Use memory-safe processing (default)")
    parser.add_argument("--batch-size", type=int,
                        help="Override batch size for processing")
    parser.add_argument("--large-docs", action="store_true",
                        help="Optimize for large documents (medical textbooks, etc.)")
    parser.add_argument("--no-skip", action="store_true",
                        help="Don't skip existing files (re-process everything)")
    parser.add_argument("--distance-metric", type=str, default="cosine",
                        choices=["cosine", "euclidean",
                                 "dot_product", "manhattan"],
                        help="Distance metric for vector similarity (default: cosine)")
    parser.add_argument("--index-algorithm", type=str, default="hnsw",
                        choices=["hnsw", "ivfflat",
                                 "scalar_quantization", "exact"],
                        help="Index algorithm for vector search (default: hnsw)")
    parser.add_argument("--all-combinations", action="store_true", default=True,
                        help="Process all 16 combinations of backends and metrics (default)")
    parser.add_argument("--single-combination", action="store_true",
                        help="Process only single combination instead of all 16")
    parser.add_argument("--parallel", action="store_true",
                        help="Enable parallel processing for improved performance")
    parser.add_argument("--workers", type=int,
                        help="Number of parallel workers (default: auto-detect)")

    args = parser.parse_args()

    # Initialize pipeline
    pipeline = UnifiedMainPipeline(
        use_enhanced_extraction=not args.basic,
        memory_safe_mode=args.memory_safe,
        batch_size=args.batch_size,
        large_docs=args.large_docs,
        distance_metric=args.distance_metric,
        index_algorithm=args.index_algorithm,
        process_all_combinations=args.all_combinations and not args.single_combination,
        parallel_mode=args.parallel,
        max_workers=args.workers
    )

    # Handle special commands
    if args.config:
        pipeline.print_pipeline_config()
        return

    if args.clean:
        pipeline.clean_pipeline_outputs()
        return

    if args.stats:
        pipeline.show_pipeline_stats()
        return

    # Check dependencies
    if not pipeline.check_dependencies():
        logger.error("❌ Dependencies not satisfied")
        return

    # Show configuration
    pipeline.print_pipeline_config()

    # Run pipeline
    success = pipeline.run_full_pipeline(
        skip_existing=not args.no_skip,
        clear_databases=args.clear,
        force_reprocess=args.force
    )

    if success:
        logger.info("✅ Pipeline completed successfully")

        # Quick instructions for next steps
        logger.info("\n🎯 Next Steps:")
        logger.info(
            "   1. Start API server: python -m uvicorn app.main:app --host 0.0.0.0 --port 8080")
        logger.info(
            "   2. Test queries: curl \"http://localhost:8080/ask?q=your question&backend=qdrant\"")

    else:
        logger.error("❌ Pipeline failed")
        exit(1)


if __name__ == "__main__":
    main()
