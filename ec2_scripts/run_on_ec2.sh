#!/bin/bash
# ============================================
# Run Metadata Path Fix Directly on EC2
# ============================================
# Copy this script to EC2 and run it there
# scp this file to EC2: scp run_on_ec2.sh ubuntu@EC2_IP:~/run_on_ec2.sh
# Then run: chmod +x run_on_ec2.sh && ./run_on_ec2.sh

echo "======================================================================"
echo "FIXING METADATA PATHS ON EC2 (DIRECT EXECUTION)"
echo "======================================================================"

# Find all JSON files
JSON_FILES=$(find ~/ResponsibleAI/cardd_data/GenAI_Results -name "processing_*.json" -type f 2>/dev/null)

if [ -z "$JSON_FILES" ]; then
    echo "❌ No JSON metadata files found!"
    exit 1
fi

TOTAL_FILES=$(echo "$JSON_FILES" | wc -l)
echo "Found $TOTAL_FILES JSON files to process"
echo ""

UPDATED_COUNT=0
SKIPPED_COUNT=0
PROCESSED_COUNT=0

echo "Starting path replacement..."
echo ""

# Process each file
for json_file in $JSON_FILES; do
    MODIFIED=false

    # Check if file has already been processed (contains EC2 paths and NO local paths)
    if grep -q "/home/ubuntu/ResponsibleAI/CarDD_release" "$json_file" 2>/dev/null && \
       ! grep -q "/Users/wjs" "$json_file" 2>/dev/null && \
       ! grep -q "\.\./CarDD_release" "$json_file" 2>/dev/null && \
       ! grep -q "\./CarDD_release" "$json_file" 2>/dev/null; then
        # File already fully processed, skip
        SKIPPED_COUNT=$((SKIPPED_COUNT + 1))
        PROCESSED_COUNT=$((PROCESSED_COUNT + 1))
        continue
    fi

    # First, fix any corrupted paths (remove duplicate prefixes)
    if grep -q "/home/ubuntu/ResponsibleA/home/ubuntu/Responsible/home/ubuntu/ResponsibleAI/CarDD_release" "$json_file" 2>/dev/null; then
        sed -i 's|/home/ubuntu/ResponsibleA/home/ubuntu/Responsible/home/ubuntu/ResponsibleAI/CarDD_release|/home/ubuntu/ResponsibleAI/CarDD_release|g' "$json_file"
        MODIFIED=true
    fi

    # Apply path replacements
    if grep -q "/Users/wjs/Local Storage/CarDD_release" "$json_file" 2>/dev/null || \
       grep -q "/Users/wjs/CarDD_release" "$json_file" 2>/dev/null || \
       grep -q "\.\./CarDD_release" "$json_file" 2>/dev/null || \
       grep -q "\./CarDD_release" "$json_file" 2>/dev/null; then

        sed -i \
            -e 's|/Users/wjs/Local Storage/CarDD_release|/home/ubuntu/ResponsibleAI/CarDD_release|g' \
            -e 's|/Users/wjs/CarDD_release|/home/ubuntu/ResponsibleAI/CarDD_release|g' \
            -e 's|\.\./CarDD_release|/home/ubuntu/ResponsibleAI/CarDD_release|g' \
            -e 's|\./CarDD_release|/home/ubuntu/ResponsibleAI/CarDD_release|g' \
            "$json_file"
        MODIFIED=true
    fi

    if [ "$MODIFIED" = "true" ]; then
        UPDATED_COUNT=$((UPDATED_COUNT + 1))
    fi

    PROCESSED_COUNT=$((PROCESSED_COUNT + 1))

    # Show progress every 100 files
    if [ $((PROCESSED_COUNT % 100)) -eq 0 ] || [ $PROCESSED_COUNT -eq $TOTAL_FILES ]; then
        echo "Progress: $PROCESSED_COUNT/$TOTAL_FILES files processed ($UPDATED_COUNT updated, $SKIPPED_COUNT skipped)"
    fi
done

echo ""
echo "======================================================================"
echo "✅ PATH FIXING COMPLETE"
echo "======================================================================"
echo "📊 Final Summary:"
echo "  Total files processed: $TOTAL_FILES"
echo "  Files updated: $UPDATED_COUNT"
echo "  Files skipped (already processed): $SKIPPED_COUNT"
echo ""

# Verify a few files
echo "Verification - checking updated paths in sample files:"
echo "$JSON_FILES" | head -3 | while read file; do
    echo "=== $(basename "$file") ==="
    grep -E '"original_image_path"|"processed_image_path"' "$file" 2>/dev/null | head -2
    echo ""
done

echo "======================================================================"
