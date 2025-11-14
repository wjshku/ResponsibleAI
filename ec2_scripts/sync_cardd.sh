#!/bin/bash
# ============================================
# Sync CarDD Dataset Module
# ============================================
# Syncs CarDD dataset (original images and masks) to EC2

sync_cardd() {
    echo "======================================================================"
    echo "EXECUTING PART 1: SYNCING CarDD DATASET"
    echo "======================================================================"

    # Check if CarDD data needs syncing
    if [ "$ORIG_COUNT" -gt "3800" ] && [ "$MASK_COUNT" -gt "3800" ]; then
        echo "CarDD data appears to be synced - skipping automatically"
        CARDD_SKIP=true
    else
        echo "CarDD data appears incomplete:"
        echo "  Images: $ORIG_COUNT (expected 4,010)"
        echo "  Masks: $MASK_COUNT (expected 4,010)"
        read -p "Re-sync CarDD data? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Skipping CarDD data sync"
            CARDD_SKIP=true
        fi
    fi

    if [ "$CARDD_SKIP" != "true" ]; then
        # echo "Syncing CarDD-TE images (374 JPG files)..."
        # rsync -avz --progress --stats \
        #     -e "ssh -i \"$EC2_KEY_PATH\" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR" \
        #         "/Users/wjs/Local Storage/CarDD_release/CarDD_SOD/CarDD-TE/CarDD-TE-Image/" \
        #         "$EC2_SSH_USER@$EC2_PUBLIC_IP":~/ResponsibleAI/CarDD_release/CarDD_SOD/CarDD-TE/CarDD-TE-Image/
        # echo "✓ CarDD-TE images synced"
        # echo ""

        # echo "Syncing CarDD-TE masks (374 PNG files)..."
        # rsync -avz --progress --stats \
        #     -e "ssh -i \"$EC2_KEY_PATH\" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR" \
        #         "/Users/wjs/Local Storage/CarDD_release/CarDD_SOD/CarDD-TE/CarDD-TE-Mask/" \
        #         "$EC2_SSH_USER@$EC2_PUBLIC_IP":~/ResponsibleAI/CarDD_release/CarDD_SOD/CarDD-TE/CarDD-TE-Mask/
        # echo "✓ CarDD-TE masks synced"
        # echo ""

        echo "Syncing CarDD-TR images (2,826 JPG files)..."
        rsync -avz --progress --stats \
            -e "ssh -i \"$EC2_KEY_PATH\" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR" \
                "/Users/wjs/Local Storage/CarDD_release/CarDD_SOD/CarDD-TR/CarDD-TR-Image/" \
                "$EC2_SSH_USER@$EC2_PUBLIC_IP":~/ResponsibleAI/CarDD_release/CarDD_SOD/CarDD-TR/CarDD-TR-Image/
        echo "✓ CarDD-TR images synced"
        echo ""

        # echo "Syncing CarDD-TR masks (2,826 PNG files)..."
        # rsync -avz --progress --stats \
        #     -e "ssh -i \"$EC2_KEY_PATH\" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR" \
        #         "/Users/wjs/Local Storage/CarDD_release/CarDD_SOD/CarDD-TR/CarDD-TR-Mask/" \
        #         "$EC2_SSH_USER@$EC2_PUBLIC_IP":~/ResponsibleAI/CarDD_release/CarDD_SOD/CarDD-TR/CarDD-TR-Mask/
        # echo "✓ CarDD-TR masks synced"
        # echo ""

        echo "Syncing CarDD-VAL images (810 JPG files)..."
        rsync -avz --progress --stats \
            -e "ssh -i \"$EC2_KEY_PATH\" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR" \
                "/Users/wjs/Local Storage/CarDD_release/CarDD_SOD/CarDD-VAL/CarDD-VAL-Image/" \
                "$EC2_SSH_USER@$EC2_PUBLIC_IP":~/ResponsibleAI/CarDD_release/CarDD_SOD/CarDD-VAL/CarDD-VAL-Image/
        echo "✓ CarDD-VAL images synced"
        echo ""

        # echo "Syncing CarDD-VAL masks (810 PNG files)..."
        # rsync -avz --progress --stats \
        #     -e "ssh -i \"$EC2_KEY_PATH\" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR" \
        #         "/Users/wjs/Local Storage/CarDD_release/CarDD_SOD/CarDD-VAL/CarDD-VAL-Mask/" \
        #         "$EC2_SSH_USER@$EC2_PUBLIC_IP":~/ResponsibleAI/CarDD_release/CarDD_SOD/CarDD-VAL/CarDD-VAL-Mask/
        # echo "✓ CarDD-VAL masks synced"
        # echo ""

        echo "🎉 CarDD Dataset Sync Complete!"
        echo "Total expected: 4,010 images + 4,010 masks = 8,020 files"
        echo ""
    else
        echo "✓ CarDD data sync skipped"
        echo ""
    fi
}
