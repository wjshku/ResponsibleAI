#!/bin/bash
# ============================================
# Sync Fake Images Module
# ============================================
# Syncs SD2, Kontext, and Qwen processed/fake images to EC2

sync_fakeimages() {
    echo "======================================================================"
    echo "SYNCING PROCESSED/FAKE IMAGES"
    echo "======================================================================"

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
    # Test connection
    # ============================================
    echo "Testing connection to EC2..."
    if ! ssh -o ConnectTimeout=5 -i "$EC2_KEY_PATH" "$EC2_SSH_USER@$EC2_PUBLIC_IP" "exit" 2>/dev/null; then
        echo "❌ Cannot connect to EC2 instance"
        exit 1
    fi
    echo "✓ Connection OK"
    echo ""

    # ============================================
    # Check current image status on EC2
    # ============================================
    echo "Checking current image status on EC2..."

    # Count existing image files
    SD2_IMG_COUNT=$(ssh -T -i "$EC2_KEY_PATH" -o LogLevel=ERROR -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "$EC2_SSH_USER@$EC2_PUBLIC_IP" "find ~/ResponsibleAI/cardd_data/GenAI_Results/SD2 -name '*.png' -o -name '*.jpg' | wc -l")
    KONTEXT_IMG_COUNT=$(ssh -T -i "$EC2_KEY_PATH" -o LogLevel=ERROR -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "$EC2_SSH_USER@$EC2_PUBLIC_IP" "find ~/ResponsibleAI/cardd_data/GenAI_Results/Kontext -name '*.png' -o -name '*.jpg' | wc -l")
    QWEN_IMG_COUNT=$(ssh -T -i "$EC2_KEY_PATH" -o LogLevel=ERROR -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "$EC2_SSH_USER@$EC2_PUBLIC_IP" "find ~/ResponsibleAI/cardd_data/GenAI_Results/Qwen\ Image\ Edit -name '*.png' -o -name '*.jpg' | wc -l")

    echo "Current image files on EC2:"
    echo "  SD2: $SD2_IMG_COUNT files"
    echo "  Kontext: $KONTEXT_IMG_COUNT files"
    echo "  Qwen Image Edit: $QWEN_IMG_COUNT files"
    echo ""

    # ============================================
    # Determine if sync is needed
    # ============================================
    PROCESSED_SKIP=false

    if [ "$SD2_IMG_COUNT" -gt "4000" ] && [ "$KONTEXT_IMG_COUNT" -gt "3500" ] && [ "$QWEN_IMG_COUNT" -gt "3000" ]; then
        echo "Images appear to be synced - skipping automatically"
        echo ""
        read -p "Force resync images anyway? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo "✓ Will force resync images"
            PROCESSED_SKIP=false
        else
            PROCESSED_SKIP=true
        fi
    else
        echo "Images appear incomplete:"
        echo "  SD2: $SD2_IMG_COUNT (expected ~4,375)"
        echo "  Kontext: $KONTEXT_IMG_COUNT (expected ~4,000)"
        echo "  Qwen Image Edit: $QWEN_IMG_COUNT (expected ~3,500)"
        echo ""
        read -p "Sync image files? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Skipping image sync"
            PROCESSED_SKIP=true
        fi
    fi

    # ============================================
    # Sync images if needed
    # ============================================
    if [ "$PROCESSED_SKIP" != "true" ]; then
        echo "======================================================================"
        echo "SYNCING IMAGE FILES"
        echo "======================================================================"

        echo "⚠️  WARNING: This will upload ~12,000 image files"
        echo "   Estimated time: 45-90 minutes depending on connection"
        echo "   Estimated size: ~22-30 GB"
        echo ""

        read -p "Continue with image upload? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Image upload cancelled"
            return
        fi

        echo ""
        echo "Syncing SD2 images (~4,375 PNG files: 374 TE + 3,187 TR + 814 VAL)..."
        rsync -avz --progress --stats --timeout=300 --compress --partial \
            --include="*/" \
            --include="*.png" \
            --include="*.jpg" \
            --exclude="*" \
            -e "ssh -i \"$EC2_KEY_PATH\" -o ServerAliveInterval=60 -o ServerAliveCountMax=10 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR" \
            cardd_data/GenAI_Results/SD2/ \
            "$EC2_SSH_USER@$EC2_PUBLIC_IP":~/ResponsibleAI/cardd_data/GenAI_Results/SD2/
        echo "✓ SD2 images synced"
        echo ""

        echo "Syncing Kontext images (~4,000 PNG files: 374 TE + 2,816 TR + 810 VAL)..."
        rsync -avz --progress --stats --timeout=300 --compress --partial \
            --include="*/" \
            --include="*.png" \
            --include="*.jpg" \
            --exclude="*" \
            -e "ssh -i \"$EC2_KEY_PATH\" -o ServerAliveInterval=60 -o ServerAliveCountMax=10 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR" \
            cardd_data/GenAI_Results/Kontext/ \
            "$EC2_SSH_USER@$EC2_PUBLIC_IP":~/ResponsibleAI/cardd_data/GenAI_Results/Kontext/
        echo "✓ Kontext images synced"
        echo ""

        echo "Syncing Qwen images (~3,500 PNG files: 374 TE + 2,816 TR + 810 VAL)..."
        rsync -avz --progress --stats --timeout=300 --compress --partial \
            --include="*/" \
            --include="*.png" \
            --include="*.jpg" \
            --exclude="*" \
            -e "ssh -i \"$EC2_KEY_PATH\" -o ServerAliveInterval=60 -o ServerAliveCountMax=10 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR" \
            'cardd_data/GenAI_Results/Qwen Image Edit/' \
            "$EC2_SSH_USER@$EC2_PUBLIC_IP":~/ResponsibleAI/cardd_data/GenAI_Results/'Qwen Image Edit'/
        echo "✓ Qwen images synced"
        echo ""
    fi

    echo "🎉 Image Sync Complete!"
    echo "Total expected: ~11,875 processed images"
    echo ""
}
