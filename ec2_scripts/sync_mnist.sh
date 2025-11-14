#!/bin/bash
# ============================================
# Sync MNIST Dataset Module
# ============================================
# MNIST: downloaded automatically by torchvision
# MNIST-M: needs to be uploaded from local data

sync_mnist() {
    echo "Syncing MNIST-M dataset..."

    # Check if local MNIST-M data exists
    if [ ! -d "domain_adapt/data/mnist_m" ]; then
        echo "❌ Local MNIST-M data not found at domain_adapt/data/mnist_m"
        echo "Expected path: domain_adapt/data/mnist_m"
        echo ""
        echo "Note: MNIST-M dataset needs to be created from MNIST + BSDS500 backgrounds"
        echo "You may need to download or generate the MNIST-M dataset first"
        exit 1
    fi

    # Create remote directory
    ssh -i "$EC2_KEY_PATH" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR "$EC2_SSH_USER@$EC2_PUBLIC_IP" \
        "mkdir -p ~/ResponsibleAI/domain_adapt/data"

    # Sync MNIST-M data
    echo "Uploading MNIST-M dataset (~68K files, this may take a while)..."
    rsync -avz --progress \
        -e "ssh -i \"$EC2_KEY_PATH\"" \
        ./domain_adapt/data/mnist_m/ \
        "$EC2_SSH_USER@$EC2_PUBLIC_IP":~/ResponsibleAI/domain_adapt/data/mnist_m/

    echo "✓ MNIST-M dataset uploaded"
    echo "Note: MNIST dataset will be downloaded automatically by torchvision when training starts"
    echo ""
}
