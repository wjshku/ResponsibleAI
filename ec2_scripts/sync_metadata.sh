#!/bin/bash
# ============================================
# Sync Metadata Module
# ============================================
# Syncs SD2 and Kontext metadata files to EC2 and fixes paths

sync_metadata() {
    echo "======================================================================"
    echo "SYNCING METADATA FILES & FIXING PATHS"
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
    # Check current metadata status on EC2
    # ============================================
    echo "Checking current metadata status on EC2..."

    # Count existing metadata files
    SD2_JSON_COUNT=$(ssh -T -i "$EC2_KEY_PATH" -o LogLevel=ERROR -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "$EC2_SSH_USER@$EC2_PUBLIC_IP" "find ~/ResponsibleAI/cardd_data/GenAI_Results/SD2 -name 'processing_*.json' -type f 2>/dev/null | wc -l")
    KONTEXT_JSON_COUNT=$(ssh -T -i "$EC2_KEY_PATH" -o LogLevel=ERROR -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "$EC2_SSH_USER@$EC2_PUBLIC_IP" "find ~/ResponsibleAI/cardd_data/GenAI_Results/Kontext -name 'processing_*.json' -type f 2>/dev/null | wc -l")

    echo "Current metadata files on EC2:"
    echo "  SD2: $SD2_JSON_COUNT files"
    echo "  Kontext: $KONTEXT_JSON_COUNT files"
    echo ""

    # ============================================
    # Determine if sync is needed
    # ============================================
    METADATA_SKIP=false
    UPLOAD_SD2_KONTEXT=false

    if [ "$SD2_JSON_COUNT" -gt "4000" ] && [ "$KONTEXT_JSON_COUNT" -gt "3500" ]; then
        echo "Metadata appears to be synced - skipping automatically"
        echo ""
        read -p "Force resync metadata anyway? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo "✓ Will force resync metadata"
            METADATA_SKIP=false
            UPLOAD_SD2_KONTEXT=true
        else
            METADATA_SKIP=true
        fi
    else
        echo "Metadata appears incomplete:"
        echo "  SD2: $SD2_JSON_COUNT files (expected ~4,383)"
        echo "  Kontext: $KONTEXT_JSON_COUNT files (expected ~4,003)"
        echo ""
        read -p "Sync metadata files? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            UPLOAD_SD2_KONTEXT=true
        else
            echo "Skipping metadata sync"
            METADATA_SKIP=true
        fi
    fi

    # ============================================
    # Sync metadata if needed
    # ============================================
    if [ "$METADATA_SKIP" != "true" ]; then
        echo "======================================================================"
        echo "SYNCING METADATA FILES"
        echo "======================================================================"

        echo "Syncing SD2 metadata (~4,383 JSON files: 375 TE + 3,192 TR + 816 VAL)..."
        rsync -avz --progress --stats \
            --include="*/" \
            --include="*.json" \
            --exclude="*" \
            -e "ssh -i \"$EC2_KEY_PATH\" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR" \
            cardd_data/GenAI_Results/SD2/ \
            "$EC2_SSH_USER@$EC2_PUBLIC_IP":~/ResponsibleAI/cardd_data/GenAI_Results/SD2/
        echo "✓ SD2 metadata synced"
        echo ""

        echo "Syncing Kontext metadata (~4,003 JSON files: 375 TE + 2,817 TR + 811 VAL)..."
        rsync -avz --progress --stats \
            --include="*/" \
            --include="*.json" \
            --exclude="*" \
            -e "ssh -i \"$EC2_KEY_PATH\" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR" \
            cardd_data/GenAI_Results/Kontext/ \
            "$EC2_SSH_USER@$EC2_PUBLIC_IP":~/ResponsibleAI/cardd_data/GenAI_Results/Kontext/
        echo "✓ Kontext metadata synced"
        echo ""

        UPLOAD_SD2_KONTEXT=true
    fi

    # ============================================
    # Fix metadata paths
    # ============================================
    if [ "$UPLOAD_SD2_KONTEXT" = "true" ] && [ "$METADATA_SKIP" != "true" ]; then
        echo "======================================================================"
        echo "FIXING METADATA PATHS"
        echo "======================================================================"

        # Copy run_on_ec2.sh to EC2 and execute it
        echo "Copying path fixing script to EC2..."
        scp -i "$EC2_KEY_PATH" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "$SCRIPT_DIR/run_on_ec2.sh" "$EC2_SSH_USER@$EC2_PUBLIC_IP:~/run_on_ec2.sh"

        echo "Running metadata path fixing on EC2..."
        ssh -T -i "$EC2_KEY_PATH" -o LogLevel=ERROR -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "$EC2_SSH_USER@$EC2_PUBLIC_IP" "chmod +x ~/run_on_ec2.sh && ~/run_on_ec2.sh"

        # Clean up the copied script
        ssh -T -i "$EC2_KEY_PATH" -o LogLevel=ERROR -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "$EC2_SSH_USER@$EC2_PUBLIC_IP" "rm -f ~/run_on_ec2.sh" 2>/dev/null || true
    else
        echo "✓ Metadata path fixing skipped (no new metadata uploaded)"
    fi

    echo ""
    echo "🎉 Metadata Sync & Path Fix Complete!"
    echo ""
}
