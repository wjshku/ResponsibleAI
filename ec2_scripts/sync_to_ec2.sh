#!/bin/bash
# ============================================
# Sync Code and Data to EC2
# ============================================
# This script transfers your code and data to the EC2 instance

set -e

# ============================================
# Load EC2 configuration
# ============================================
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
CONFIG_FILE="$SCRIPT_DIR/ec2_config.sh"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ EC2 configuration not found"
    echo "Run ./ec2_scripts/setup_ec2.sh first"
    exit 1
fi

source "$CONFIG_FILE"

echo "======================================================================"
echo "SYNCING CODE AND DATA TO EC2"
echo "======================================================================"
echo "Instance: $EC2_INSTANCE_ID"
echo "IP: $EC2_PUBLIC_IP"
echo ""

# ============================================
# Test connection
# ============================================
echo "Testing connection..."
if ! ssh -o ConnectTimeout=5 -i "$EC2_KEY_PATH" "$EC2_SSH_USER@$EC2_PUBLIC_IP" "exit" 2>/dev/null; then
    echo "❌ Cannot connect to EC2 instance"
    echo "Make sure the instance is running:"
    echo "  aws ec2 start-instances --region $EC2_REGION --instance-ids $EC2_INSTANCE_ID"
    exit 1
fi
echo "✓ Connection OK"
echo ""

# ============================================
# Sync code
# ============================================
echo "======================================================================"
echo "SYNCING CODE"
echo "======================================================================"

# Change to ResponsibleAI root directory
cd "$SCRIPT_DIR/.."

# Sync requirements.txt first
echo "Syncing requirements.txt..."
rsync -avz \
    -e "ssh -i \"$EC2_KEY_PATH\"" \
    requirements.txt \
    "$EC2_SSH_USER@$EC2_PUBLIC_IP":~/ResponsibleAI/
echo "✓ requirements.txt synced"
echo ""

echo "Syncing domain_adapt code..."
rsync -avz --progress \
    --exclude='*.pth' \
    --exclude='*.png' \
    --exclude='*.pdf' \
    --exclude='__pycache__' \
    --exclude='.git' \
    --exclude='models/' \
    -e "ssh -i \"$EC2_KEY_PATH\"" \
    domain_adapt/ \
    "$EC2_SSH_USER@$EC2_PUBLIC_IP":~/ResponsibleAI/domain_adapt/

echo "✓ Code synced"
echo ""

# Sync simple_detect_car (dependencies)
echo "Syncing simple_detect_car dependencies..."
rsync -avz --progress \
    --exclude='*.pth' \
    --exclude='__pycache__' \
    --exclude='.git' \
    --exclude='models/' \
    -e "ssh -i \"$EC2_KEY_PATH\"" \
    simple_detect_car/*.py \
    "$EC2_SSH_USER@$EC2_PUBLIC_IP":~/ResponsibleAI/simple_detect_car/

echo "✓ Dependencies synced"
echo ""

# ============================================
# Sync data
# ============================================
echo "======================================================================"
echo "SYNCING DATA"
echo "======================================================================"
echo ""

# Ensure directories exist on EC2
echo "Creating directory structure on EC2..."
ssh -i "$EC2_KEY_PATH" "$EC2_SSH_USER@$EC2_PUBLIC_IP" << 'EOFMKDIR'
mkdir -p ~/ResponsibleAI/cardd_data/GenAI_Results/SD2/CarDD-{TE,TR,VAL}
mkdir -p ~/ResponsibleAI/cardd_data/GenAI_Results/Kontext/CarDD-{TE,TR,VAL}
mkdir -p ~/ResponsibleAI/CarDD_release/CarDD_SOD/CarDD-TE/CarDD-TE-{Image,Mask}
mkdir -p ~/ResponsibleAI/CarDD_release/CarDD_SOD/CarDD-TR/CarDD-TR-{Image,Mask}
mkdir -p ~/ResponsibleAI/CarDD_release/CarDD_SOD/CarDD-VAL/CarDD-VAL-{Image,Mask}
EOFMKDIR
echo "✓ Directories created"
echo ""

# Check if data already synced
echo "Checking existing data on EC2..."
DATA_STATUS=$(ssh -i "$EC2_KEY_PATH" "$EC2_SSH_USER@$EC2_PUBLIC_IP" << 'EOFCHECK'
ORIG_IMGS=$(find ~/ResponsibleAI/CarDD_release/CarDD_SOD -type f \( -name "*.jpg" -o -name "*.png" \) -path "*/Image/*" 2>/dev/null | wc -l)
MASKS=$(find ~/ResponsibleAI/CarDD_release/CarDD_SOD -type f -path "*/Mask/*" 2>/dev/null | wc -l)
SD2_JSON=$(find ~/ResponsibleAI/cardd_data/GenAI_Results/SD2 -name "*.json" 2>/dev/null | wc -l)
SD2_IMG=$(find ~/ResponsibleAI/cardd_data/GenAI_Results/SD2 \( -name "*.png" -o -name "*.jpg" \) 2>/dev/null | wc -l)
KONTEXT_JSON=$(find ~/ResponsibleAI/cardd_data/GenAI_Results/Kontext -name "*.json" 2>/dev/null | wc -l)
KONTEXT_IMG=$(find ~/ResponsibleAI/cardd_data/GenAI_Results/Kontext \( -name "*.png" -o -name "*.jpg" \) 2>/dev/null | wc -l)
echo "$ORIG_IMGS $MASKS $SD2_JSON $SD2_IMG $KONTEXT_JSON $KONTEXT_IMG"
EOFCHECK
)

read -r ORIG_COUNT MASK_COUNT SD2_JSON_COUNT SD2_IMG_COUNT KONTEXT_JSON_COUNT KONTEXT_IMG_COUNT <<< "$DATA_STATUS"

echo "Current data on EC2:"
echo "  Original images: $ORIG_COUNT (expected: 4,010 from TE+TR+VAL)"
echo "  Masks: $MASK_COUNT (expected: 4,010)"
echo "  SD2 metadata: $SD2_JSON_COUNT, images: $SD2_IMG_COUNT (expected: ~4,375)"
echo "  Kontext metadata: $KONTEXT_JSON_COUNT, images: $KONTEXT_IMG_COUNT (expected: ~4,000)"
echo ""

if [ "$SD2_IMG_COUNT" -gt "4000" ] && [ "$KONTEXT_IMG_COUNT" -gt "3500" ]; then
    echo "Data appears to be synced"
    read -p "Re-sync all data? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Skipping data sync"
        DATA_SYNC_SKIPPED=true
    fi
fi

if [ "$DATA_SYNC_SKIPPED" != "true" ]; then
    echo "======================================================================"
    echo "STEP 1: Syncing metadata files"
    echo "======================================================================"

    # Check if metadata needs syncing
    METADATA_SKIP=false
    if [ "$SD2_JSON_COUNT" -gt "4000" ] && [ "$KONTEXT_JSON_COUNT" -gt "3500" ]; then
        echo "Metadata appears to be synced:"
        echo "  SD2: $SD2_JSON_COUNT files (expected ~4,383)"
        echo "  Kontext: $KONTEXT_JSON_COUNT files (expected ~4,003)"
        read -p "Re-sync metadata? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Skipping metadata sync"
            METADATA_SKIP=true
        fi
    fi

    if [ "$METADATA_SKIP" != "true" ]; then
        echo "Syncing SD2 metadata (~4,383 JSON files: 375 TE + 3,192 TR + 816 VAL)..."
        rsync -avz --progress \
            --include="*/" \
            --include="*.json" \
            --exclude="*" \
            -e "ssh -i \"$EC2_KEY_PATH\"" \
            cardd_data/GenAI_Results/SD2/ \
            "$EC2_SSH_USER@$EC2_PUBLIC_IP":~/ResponsibleAI/cardd_data/GenAI_Results/SD2/
        echo "✓ SD2 metadata synced"
        echo ""

        echo "Syncing Kontext metadata (~4,003 JSON files: 375 TE + 2,817 TR + 811 VAL)..."
        rsync -avz --progress \
            --include="*/" \
            --include="*.json" \
            --exclude="*" \
            -e "ssh -i \"$EC2_KEY_PATH\"" \
            cardd_data/GenAI_Results/Kontext/ \
            "$EC2_SSH_USER@$EC2_PUBLIC_IP":~/ResponsibleAI/cardd_data/GenAI_Results/Kontext/
        echo "✓ Kontext metadata synced"
        echo ""
    else
        echo "✓ Metadata sync skipped"
        echo ""
    fi

    echo "======================================================================"
    echo "STEP 2: Syncing original CarDD images & masks"
    echo "======================================================================"

    # Check if original data needs syncing
    ORIGINAL_SKIP=false
    if [ "$ORIG_COUNT" -gt "3800" ] && [ "$MASK_COUNT" -gt "3800" ]; then
        echo "Original images/masks appear to be synced:"
        echo "  Images: $ORIG_COUNT (expected 4,010)"
        echo "  Masks: $MASK_COUNT (expected 4,010)"
        read -p "Re-sync original data? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Skipping original data sync"
            ORIGINAL_SKIP=true
        fi
    fi

    if [ "$ORIGINAL_SKIP" != "true" ]; then
        echo "Syncing CarDD-TE images (374 JPG files)..."
        rsync -avz --progress \
            -e "ssh -i \"$EC2_KEY_PATH\"" \
            "/Users/wjs/Local Storage/CarDD_release/CarDD_SOD/CarDD-TE/CarDD-TE-Image/" \
            "$EC2_SSH_USER@$EC2_PUBLIC_IP":~/ResponsibleAI/CarDD_release/CarDD_SOD/CarDD-TE/CarDD-TE-Image/
        echo "✓ CarDD-TE images synced"
        echo ""

        echo "Syncing CarDD-TE masks (374 PNG files)..."
        rsync -avz --progress \
            -e "ssh -i \"$EC2_KEY_PATH\"" \
            "/Users/wjs/Local Storage/CarDD_release/CarDD_SOD/CarDD-TE/CarDD-TE-Mask/" \
            "$EC2_SSH_USER@$EC2_PUBLIC_IP":~/ResponsibleAI/CarDD_release/CarDD_SOD/CarDD-TE/CarDD-TE-Mask/
        echo "✓ CarDD-TE masks synced"
        echo ""

        echo "Syncing CarDD-TR images (2,826 JPG files)..."
        rsync -avz --progress \
            -e "ssh -i \"$EC2_KEY_PATH\"" \
            "/Users/wjs/Local Storage/CarDD_release/CarDD_SOD/CarDD-TR/CarDD-TR-Image/" \
            "$EC2_SSH_USER@$EC2_PUBLIC_IP":~/ResponsibleAI/CarDD_release/CarDD_SOD/CarDD-TR/CarDD-TR-Image/
        echo "✓ CarDD-TR images synced"
        echo ""

        echo "Syncing CarDD-TR masks (2,826 PNG files)..."
        rsync -avz --progress \
            -e "ssh -i \"$EC2_KEY_PATH\"" \
            "/Users/wjs/Local Storage/CarDD_release/CarDD_SOD/CarDD-TR/CarDD-TR-Mask/" \
            "$EC2_SSH_USER@$EC2_PUBLIC_IP":~/ResponsibleAI/CarDD_release/CarDD_SOD/CarDD-TR/CarDD-TR-Mask/
        echo "✓ CarDD-TR masks synced"
        echo ""

        echo "Syncing CarDD-VAL images (810 JPG files)..."
        rsync -avz --progress \
            -e "ssh -i \"$EC2_KEY_PATH\"" \
            "/Users/wjs/Local Storage/CarDD_release/CarDD_SOD/CarDD-VAL/CarDD-VAL-Image/" \
            "$EC2_SSH_USER@$EC2_PUBLIC_IP":~/ResponsibleAI/CarDD_release/CarDD_SOD/CarDD-VAL/CarDD-VAL-Image/
        echo "✓ CarDD-VAL images synced"
        echo ""

        echo "Syncing CarDD-VAL masks (810 PNG files)..."
        rsync -avz --progress \
            -e "ssh -i \"$EC2_KEY_PATH\"" \
            "/Users/wjs/Local Storage/CarDD_release/CarDD_SOD/CarDD-VAL/CarDD-VAL-Mask/" \
            "$EC2_SSH_USER@$EC2_PUBLIC_IP":~/ResponsibleAI/CarDD_release/CarDD_SOD/CarDD-VAL/CarDD-VAL-Mask/
        echo "✓ CarDD-VAL masks synced"
        echo ""
    else
        echo "✓ Original data sync skipped"
        echo ""
    fi

    echo "======================================================================"
    echo "STEP 3: Syncing processed/fake images"
    echo "======================================================================"

    # Check if processed images need syncing
    PROCESSED_SKIP=false
    if [ "$SD2_IMG_COUNT" -gt "4000" ] && [ "$KONTEXT_IMG_COUNT" -gt "3500" ]; then
        echo "Processed images appear to be synced:"
        echo "  SD2: $SD2_IMG_COUNT (expected ~4,375)"
        echo "  Kontext: $KONTEXT_IMG_COUNT (expected ~4,000)"
        read -p "Re-sync processed images? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Skipping processed images sync"
            PROCESSED_SKIP=true
        fi
    fi

    if [ "$PROCESSED_SKIP" != "true" ]; then
        echo "Syncing SD2 images (~4,375 PNG files: 374 TE + 3,187 TR + 814 VAL)..."
        rsync -avz --progress \
            --include="*/" \
            --include="*.png" \
            --include="*.jpg" \
            --exclude="*" \
            -e "ssh -i \"$EC2_KEY_PATH\"" \
            cardd_data/GenAI_Results/SD2/ \
            "$EC2_SSH_USER@$EC2_PUBLIC_IP":~/ResponsibleAI/cardd_data/GenAI_Results/SD2/
        echo "✓ SD2 images synced"
        echo ""

        echo "Syncing Kontext images (~4,000 PNG files: 374 TE + 2,816 TR + 810 VAL)..."
        rsync -avz --progress \
            --include="*/" \
            --include="*.png" \
            --include="*.jpg" \
            --exclude="*" \
            -e "ssh -i \"$EC2_KEY_PATH\"" \
            cardd_data/GenAI_Results/Kontext/ \
            "$EC2_SSH_USER@$EC2_PUBLIC_IP":~/ResponsibleAI/cardd_data/GenAI_Results/Kontext/
        echo "✓ Kontext images synced"
        echo ""
    else
        echo "✓ Processed images sync skipped"
        echo ""
    fi

    echo "✓ All data synced"
fi
echo ""

# ============================================
# Update paths in code and metadata
# ============================================
echo "======================================================================"
echo "UPDATING FILE PATHS"
echo "======================================================================"

ssh -i "$EC2_KEY_PATH" "$EC2_SSH_USER@$EC2_PUBLIC_IP" << 'EOF'
cd ~/ResponsibleAI/domain_adapt

echo "✓ Code uses relative paths, no path updates needed"

# Update paths in metadata files
echo "Updating metadata file paths..."
cd ~/ResponsibleAI/cardd_data/GenAI_Results

# Update SD2 metadata
find SD2 -name "processing_*.json" -type f | while read file; do
    sed -i 's|/Users/wjs/Local Storage/CarDD_release|/home/ubuntu/ResponsibleAI/CarDD_release|g' "$file"
done

# Update Kontext metadata
find Kontext -name "processing_*.json" -type f | while read file; do
    sed -i 's|/Users/wjs/Local Storage/CarDD_release|/home/ubuntu/ResponsibleAI/CarDD_release|g' "$file"
done

echo "✓ Metadata paths updated"
EOF

echo ""

# ============================================
# Verify sync
# ============================================
echo "======================================================================"
echo "VERIFYING SYNC"
echo "======================================================================"

ssh -i "$EC2_KEY_PATH" "$EC2_SSH_USER@$EC2_PUBLIC_IP" << 'EOF'
echo "Checking files..."

# Check code files
if [ ! -f ~/ResponsibleAI/domain_adapt/train_dann.py ]; then
    echo "❌ train_dann.py not found"
    exit 1
fi

if [ ! -f ~/ResponsibleAI/domain_adapt/model_dann.py ]; then
    echo "❌ model_dann.py not found"
    exit 1
fi

echo "✓ Code files present"
echo ""

# Check original CarDD data
TE_IMG=$(find ~/ResponsibleAI/CarDD_release/CarDD_SOD/CarDD-TE/CarDD-TE-Image -type f 2>/dev/null | wc -l)
TE_MASK=$(find ~/ResponsibleAI/CarDD_release/CarDD_SOD/CarDD-TE/CarDD-TE-Mask -type f 2>/dev/null | wc -l)
TR_IMG=$(find ~/ResponsibleAI/CarDD_release/CarDD_SOD/CarDD-TR/CarDD-TR-Image -type f 2>/dev/null | wc -l)
TR_MASK=$(find ~/ResponsibleAI/CarDD_release/CarDD_SOD/CarDD-TR/CarDD-TR-Mask -type f 2>/dev/null | wc -l)
VAL_IMG=$(find ~/ResponsibleAI/CarDD_release/CarDD_SOD/CarDD-VAL/CarDD-VAL-Image -type f 2>/dev/null | wc -l)
VAL_MASK=$(find ~/ResponsibleAI/CarDD_release/CarDD_SOD/CarDD-VAL/CarDD-VAL-Mask -type f 2>/dev/null | wc -l)

TOTAL_IMG=$((TE_IMG + TR_IMG + VAL_IMG))
TOTAL_MASK=$((TE_MASK + TR_MASK + VAL_MASK))

echo "Original CarDD data:"
echo "  TE:  $TE_IMG images, $TE_MASK masks (expected: 374 each)"
echo "  TR:  $TR_IMG images, $TR_MASK masks (expected: 2,826 each)"
echo "  VAL: $VAL_IMG images, $VAL_MASK masks (expected: 810 each)"
echo "  TOTAL: $TOTAL_IMG images, $TOTAL_MASK masks (expected: 4,010 each)"

if [ "$TOTAL_IMG" -lt "3800" ] || [ "$TOTAL_MASK" -lt "3800" ]; then
    echo "⚠️  Warning: Original data count seems low"
else
    echo "✓ Original data complete"
fi
echo ""

# Check SD2 data
SD2_JSON=$(find ~/ResponsibleAI/cardd_data/GenAI_Results/SD2 -name "*.json" 2>/dev/null | wc -l)
SD2_IMG=$(find ~/ResponsibleAI/cardd_data/GenAI_Results/SD2 \( -name "*.jpg" -o -name "*.png" \) 2>/dev/null | wc -l)
echo "SD2 processed data:"
echo "  Metadata: $SD2_JSON (expected: ~4,383)"
echo "  Images: $SD2_IMG (expected: ~4,375)"

if [ "$SD2_JSON" -lt "4000" ] || [ "$SD2_IMG" -lt "4000" ]; then
    echo "⚠️  Warning: SD2 data count seems low"
else
    echo "✓ SD2 data complete"
fi
echo ""

# Check Kontext data
KONTEXT_JSON=$(find ~/ResponsibleAI/cardd_data/GenAI_Results/Kontext -name "*.json" 2>/dev/null | wc -l)
KONTEXT_IMG=$(find ~/ResponsibleAI/cardd_data/GenAI_Results/Kontext \( -name "*.jpg" -o -name "*.png" \) 2>/dev/null | wc -l)
echo "Kontext processed data:"
echo "  Metadata: $KONTEXT_JSON (expected: ~4,003)"
echo "  Images: $KONTEXT_IMG (expected: ~4,000)"

if [ "$KONTEXT_JSON" -lt "3500" ] || [ "$KONTEXT_IMG" -lt "3500" ]; then
    echo "⚠️  Warning: Kontext data count seems low"
else
    echo "✓ Kontext data complete"
fi
EOF

echo ""

# ============================================
# Success message
# ============================================
echo "======================================================================"
echo "✅ SYNC COMPLETED"
echo "======================================================================"
echo ""
echo "Next step:"
echo "  ./ec2_scripts/train_on_ec2.sh"
echo ""
echo "Or SSH in to check manually:"
echo "  ssh -i \"$EC2_KEY_PATH\" \"$EC2_SSH_USER@$EC2_PUBLIC_IP\""
echo "======================================================================"
