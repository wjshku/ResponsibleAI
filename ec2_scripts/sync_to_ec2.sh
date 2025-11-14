#!/bin/bash
# ============================================
# Sync Code and Data to EC2
# ============================================
# This script transfers your code and data to the EC2 instance
# IMPORTANT: Model files (*.pth, *.h5, *.pkl, *.joblib) and model directories are excluded

set -e

# SSH retry function removed for debugging

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

# ============================================
# Load sync modules
# ============================================
source "$SCRIPT_DIR/sync_code.sh"
source "$SCRIPT_DIR/sync_cardd.sh"
source "$SCRIPT_DIR/sync_metadata.sh"
source "$SCRIPT_DIR/sync_fakeimages.sh"
source "$SCRIPT_DIR/sync_mnist.sh"

# ============================================
# Main Script
# ============================================

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
# Always sync code first
# ============================================
sync_code

# ============================================
# PART 1: Upload CarDD Dataset
# ============================================
echo "======================================================================"
echo "PART 1: SYNCING CarDD DATASET"
echo "======================================================================"
echo "This includes original CarDD-TE, CarDD-TR, and CarDD-VAL images and masks"
echo ""

# Ensure directories exist on EC2
echo "Creating directory structure on EC2..."
ssh -i "$EC2_KEY_PATH" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR "$EC2_SSH_USER@$EC2_PUBLIC_IP" << 'EOFMKDIR'
mkdir -p ~/ResponsibleAI/CarDD_release/CarDD_SOD/CarDD-TE/CarDD-TE-{Image,Mask}
mkdir -p ~/ResponsibleAI/CarDD_release/CarDD_SOD/CarDD-TR/CarDD-TR-{Image,Mask}
mkdir -p ~/ResponsibleAI/CarDD_release/CarDD_SOD/CarDD-VAL/CarDD-VAL-{Image,Mask}
EOFMKDIR
echo "✓ CarDD directories created"
echo ""

# Ask user if they want to upload CarDD dataset
read -p "Upload CarDD dataset (original TE/TR/VAL images & masks)? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    UPLOAD_CARDD=true
    echo "✓ Will upload CarDD dataset"
else
    UPLOAD_CARDD=false
    echo "⏭️  Skipping CarDD dataset upload"
fi
echo ""

# ============================================
# PART 2: Upload SD2 and Kontext Metadata
# ============================================
echo "======================================================================"
echo "PART 2: SYNCING SD2 & KONTEKT METADATA"
echo "======================================================================"
echo "This includes JSON metadata files with automatic path adjustments"
echo ""

# Ensure directories exist on EC2 for SD2/Kontext
echo "Creating SD2/Kontext directory structure on EC2..."
ssh -i "$EC2_KEY_PATH" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR "$EC2_SSH_USER@$EC2_PUBLIC_IP" << 'EOFMKDIR2'
mkdir -p ~/ResponsibleAI/cardd_data/GenAI_Results/SD2/CarDD-{TE,TR,VAL}
mkdir -p ~/ResponsibleAI/cardd_data/GenAI_Results/Kontext/CarDD-{TE,TR,VAL}
EOFMKDIR2
echo "✓ SD2/Kontext directories created"
echo ""

# Ask user if they want to upload SD2/Kontext metadata
read -p "Upload SD2 & Kontext metadata (JSON files + path fixes)? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    UPLOAD_SD2_KONTEXT_METADATA=true
    echo "✓ Will upload SD2 & Kontext metadata"
else
    UPLOAD_SD2_KONTEXT_METADATA=false
    echo "⏭️  Skipping SD2 & Kontext metadata upload"
fi
echo ""

# PART 3: Upload SD2 and Kontext Fake Images
# ============================================
echo "======================================================================"
echo "PART 3: SYNCING SD2 & KONTEKT FAKE IMAGES"
echo "======================================================================"
echo "This includes processed/fake images (PNG/JPG files) - ~15-20GB transfer"
echo ""

# Ask user if they want to upload SD2/Kontext fake images
read -p "Upload SD2 & Kontext fake images (large transfer)? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    UPLOAD_SD2_KONTEXT_IMAGES=true
    echo "✓ Will upload SD2 & Kontext fake images"
else
    UPLOAD_SD2_KONTEXT_IMAGES=false
    echo "⏭️  Skipping SD2 & Kontext fake images upload"
fi
echo ""

# ============================================
# PART 4: Upload MNIST-M Dataset
# ============================================
echo "======================================================================"
echo "PART 4: SYNCING MNIST-M DATASET"
echo "======================================================================"
echo "MNIST: downloaded automatically by torchvision"
echo "MNIST-M: needs to be uploaded from local data (~50MB)"
echo ""

# Ask user if they want to upload MNIST-M
read -p "Upload MNIST-M dataset? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    UPLOAD_MNIST_M=true
    echo "✓ Will upload MNIST-M dataset"
else
    UPLOAD_MNIST_M=false
    echo "⏭️  Skipping MNIST-M dataset upload"
fi
echo ""

# ============================================
# Note: MNIST dataset is downloaded automatically
# ============================================
echo "Note: MNIST dataset will be downloaded automatically by torchvision when training starts"
echo ""

# ============================================
# Execute uploads based on user choices
# ============================================

# Check existing data status if any uploads are enabled
if [ "$UPLOAD_CARDD" = "true" ] || [ "$UPLOAD_SD2_KONTEXT" = "true" ] || [ "$UPLOAD_MNIST_M" = "true" ]; then
    echo "Checking existing data on EC2..."
    DATA_STATUS=$(ssh -T -i "$EC2_KEY_PATH" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o PasswordAuthentication=no -o ConnectTimeout=10 "$EC2_SSH_USER@$EC2_PUBLIC_IP" "
    ORIG_IMGS=\$(find ~/ResponsibleAI/CarDD_release/CarDD_SOD -type f \\\( -name '*.jpg' -o -name '*.png' \\\) -path '*/Image/*' 2>/dev/null | wc -l)
    MASKS=\$(find ~/ResponsibleAI/CarDD_release/CarDD_SOD -type f -path '*/Mask/*' 2>/dev/null | wc -l)
    SD2_JSON=\$(find ~/ResponsibleAI/cardd_data/GenAI_Results/SD2 -name '*.json' 2>/dev/null | wc -l)
    SD2_IMG=\$(find ~/ResponsibleAI/cardd_data/GenAI_Results/SD2 \\\( -name '*.png' -o -name '*.jpg' \\\) 2>/dev/null | wc -l)
    KONTEXT_JSON=\$(find ~/ResponsibleAI/cardd_data/GenAI_Results/Kontext -name '*.json' 2>/dev/null | wc -l)
    KONTEXT_IMG=\$(find ~/ResponsibleAI/cardd_data/GenAI_Results/Kontext \\\( -name '*.png' -o -name '*.jpg' \\\) 2>/dev/null | wc -l)
    MNIST_M_FILES=\$(find ~/ResponsibleAI/domain_adapt/data/mnist_m -name '*.png' 2>/dev/null | wc -l)
    echo \"\$ORIG_IMGS \$MASKS \$SD2_JSON \$SD2_IMG \$KONTEXT_JSON \$KONTEXT_IMG \$MNIST_M_FILES\"
    " 2>/dev/null)

    read -r ORIG_COUNT MASK_COUNT SD2_JSON_COUNT SD2_IMG_COUNT KONTEXT_JSON_COUNT KONTEXT_IMG_COUNT MNIST_M_COUNT <<< "$DATA_STATUS"

    echo "Current data on EC2:"
    echo "  CarDD images: $ORIG_COUNT (expected: 4,010 from TE+TR+VAL)"
    echo "  CarDD masks: $MASK_COUNT (expected: 4,010)"
    echo "  SD2 metadata: $SD2_JSON_COUNT, images: $SD2_IMG_COUNT (expected: ~4,375)"
    echo "  Kontext metadata: $KONTEXT_JSON_COUNT, images: $KONTEXT_IMG_COUNT (expected: ~4,000)"
    echo "  MNIST-M files: $MNIST_M_COUNT (expected: ~60,000)"
    echo ""
fi

# ============================================
# PART 1 EXECUTION: CarDD Dataset
# ============================================
if [ "$UPLOAD_CARDD" = "true" ]; then
    sync_cardd
else
    echo "⏭️  CarDD dataset upload skipped"
    echo ""
fi

# ============================================
# PART 2 EXECUTION: SD2 & Kontext Metadata
# ============================================
if [ "$UPLOAD_SD2_KONTEXT_METADATA" = "true" ]; then
    sync_metadata
else
    echo "⏭️  SD2 & Kontext metadata upload skipped"
    echo ""
fi

# ============================================
# PART 3 EXECUTION: SD2 & Kontext Fake Images
# ============================================
if [ "$UPLOAD_SD2_KONTEXT_IMAGES" = "true" ]; then
    sync_fakeimages
else
    echo "⏭️  SD2 & Kontext fake images upload skipped"
    echo ""
fi

# ============================================
# PART 4 EXECUTION: MNIST-M Dataset
# ============================================
if [ "$UPLOAD_MNIST_M" = "true" ]; then
    sync_mnist
else
    echo "⏭️  MNIST-M dataset upload skipped"
    echo ""
fi

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
