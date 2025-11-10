#!/bin/bash
# Monitor pipeline progress

echo "📊 Pipeline Progress Monitor"
echo "=============================="

# Check if pipeline is running
if pgrep -f "main_pipeline.py" > /dev/null; then
    echo "✅ Pipeline is RUNNING"
else
    echo "❌ Pipeline is NOT running"
fi

echo ""
echo "📄 Recent log entries:"
echo "----------------------"
tail -n 20 pipeline_with_ocr.log | grep -E "(Processing pages|OCR used|Processed|chunks processed|STEP|✅|⏱️)"

echo ""
echo "📈 Statistics:"
echo "-------------"
echo -n "Total pages processed: "
grep -o "Processed [0-9]*/1322" pipeline_with_ocr.log | tail -1 || echo "0/1322"

echo -n "OCR operations: "
grep -o "OCR used for [0-9]*/[0-9]*" pipeline_with_ocr.log | awk -F'[/ ]' '{sum+=$4; total+=$5} END {print sum " pages (from " NR " chunks)"}'

echo -n "Total chunks: "
grep -c "Processing 25 pages with 4 parallel workers" pipeline_with_ocr.log || echo "0"

echo ""
echo "💾 Database status:"
echo "------------------"
if [ -f pipeline_with_ocr.log ]; then
    grep -E "(Qdrant:|PostgreSQL:).*points|rows" pipeline_with_ocr.log | tail -2
fi

echo ""
echo "⏰ To watch live: tail -f pipeline_with_ocr.log"
