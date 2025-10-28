#!/bin/bash
# ============================================
# Download Training Results from EC2
# ============================================
# This script downloads trained models and logs from EC2

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
echo "DOWNLOADING RESULTS FROM EC2"
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
    exit 1
fi
echo "✓ Connection OK"
echo ""

# ============================================
# Check training status
# ============================================
echo "Checking training status..."

TRAINING_SESSIONS=$(ssh -q -o LogLevel=ERROR -i "$EC2_KEY_PATH" "$EC2_SSH_USER@$EC2_PUBLIC_IP" \
    "screen -list | grep -E '(dann_training|nn_training|sk_training)' || echo ''")

if [ -n "$TRAINING_SESSIONS" ]; then
    echo "⚠️  Training sessions still running:"
    echo "$TRAINING_SESSIONS"
    echo ""
    read -p "Download anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted"
        exit 0
    fi
fi

# ============================================
# Select training type
# ============================================
echo ""
echo "Select training type to download:"
echo "  [1] DANN (domain_adapt)"
echo "  [2] Simple Detect CNN/MLP (simple_detect_car)"
echo ""
read -p "Choose [1-2]: " -n 1 -r TRAINING_TYPE
echo ""

case $TRAINING_TYPE in
    1)
        TRAINING_DIR="domain_adapt"
        MODEL_PATTERN="model_dann_*"
        ;;
    2)
        TRAINING_DIR="simple_detect_car"
        MODEL_PATTERN="model_*"
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

# ============================================
# List available models
# ============================================
echo ""
echo "======================================================================"
echo "AVAILABLE MODELS ($TRAINING_DIR)"
echo "======================================================================"

MODEL_LIST=$(ssh -q -o LogLevel=ERROR -i "$EC2_KEY_PATH" "$EC2_SSH_USER@$EC2_PUBLIC_IP" << EOF
cd ~/ResponsibleAI/$TRAINING_DIR 2>/dev/null || { echo "ERROR: Directory not found"; exit 1; }

if [ ! -d "models" ]; then
    echo "ERROR: No models directory found"
    exit 1
fi

MODEL_DIRS=\$(find models -maxdepth 1 -type d -name "$MODEL_PATTERN" 2>/dev/null | sort -r)

if [ -z "\$MODEL_DIRS" ]; then
    echo "ERROR: No trained models found"
    exit 1
fi

echo "\$MODEL_DIRS"
EOF
)

# Filter to only show model paths (ignore SSH banner)
MODEL_LIST=$(echo "$MODEL_LIST" | grep "^models/" || echo "$MODEL_LIST" | grep "^ERROR:")

if [[ "$MODEL_LIST" == ERROR:* ]]; then
    echo "❌ ${MODEL_LIST#ERROR: }"
    exit 1
fi

if [ -z "$MODEL_LIST" ]; then
    echo "❌ No models found"
    exit 1
fi

echo "$MODEL_LIST" | nl -w 2 -s '. '
echo ""
read -p "Download which model? (number, 'all', or 'latest'): " MODEL_CHOICE

# ============================================
# Download models
# ============================================
echo ""
echo "======================================================================"
echo "DOWNLOADING MODELS"
echo "======================================================================"

# Determine local download directory
if [ "$TRAINING_TYPE" == "1" ]; then
    LOCAL_DIR="$SCRIPT_DIR/../domain_adapt"  # domain_adapt
else
    LOCAL_DIR="$SCRIPT_DIR/../simple_detect_car"  # simple_detect_car
fi

cd "$LOCAL_DIR"

case $MODEL_CHOICE in
    all)
        echo "Downloading all models..."
        rsync -avz --progress \
            -e "ssh -i \"$EC2_KEY_PATH\"" \
            "$EC2_SSH_USER@$EC2_PUBLIC_IP":~/ResponsibleAI/$TRAINING_DIR/models/ \
            ./models/
        ;;
    latest|1)
        echo "Downloading latest model..."
        LATEST_MODEL=$(ssh -q -o LogLevel=ERROR -i "$EC2_KEY_PATH" "$EC2_SSH_USER@$EC2_PUBLIC_IP" \
            "cd ~/ResponsibleAI/$TRAINING_DIR && find models -maxdepth 1 -type d -name '$MODEL_PATTERN' | sort -r | head -1")

        if [ -z "$LATEST_MODEL" ]; then
            echo "❌ No models found"
            exit 1
        fi

        echo "Downloading: $LATEST_MODEL"
        rsync -avz --progress \
            -e "ssh -i \"$EC2_KEY_PATH\"" \
            "$EC2_SSH_USER@$EC2_PUBLIC_IP":~/ResponsibleAI/$TRAINING_DIR/$LATEST_MODEL/ \
            ./$LATEST_MODEL/
        ;;
    [0-9]*)
        # Download specific model by number
        SELECTED_MODEL=$(ssh -q -o LogLevel=ERROR -i "$EC2_KEY_PATH" "$EC2_SSH_USER@$EC2_PUBLIC_IP" \
            "cd ~/ResponsibleAI/$TRAINING_DIR && find models -maxdepth 1 -type d -name '$MODEL_PATTERN' | sort -r | sed -n '${MODEL_CHOICE}p'")

        if [ -z "$SELECTED_MODEL" ]; then
            echo "❌ Invalid model number"
            exit 1
        fi

        echo "Downloading: $SELECTED_MODEL"
        rsync -avz --progress \
            -e "ssh -i \"$EC2_KEY_PATH\"" \
            "$EC2_SSH_USER@$EC2_PUBLIC_IP":~/ResponsibleAI/$TRAINING_DIR/$SELECTED_MODEL/ \
            ./$SELECTED_MODEL/
        ;;
    *)
        echo "❌ Invalid choice"
        exit 1
        ;;
esac

echo "✓ Models downloaded"
echo ""

# ============================================
# Skip downloading training logs (can monitor via SSH)
# ============================================
# Logs can be viewed via: ./ec2_scripts/monitor_training.sh

# ============================================
# Display summary
# ============================================
echo "======================================================================"
echo "TRAINING SUMMARY"
echo "======================================================================"

# Find the downloaded model
DOWNLOADED_MODEL=$(find ./models -maxdepth 1 -type d -name "$MODEL_PATTERN" | sort -r | head -1)

if [ -n "$DOWNLOADED_MODEL" ] && [ -f "$DOWNLOADED_MODEL/metadata.json" ]; then
    echo ""
    python3 << EOF
import json

with open('$DOWNLOADED_MODEL/metadata.json', 'r') as f:
    metadata = json.load(f)

config = metadata['config']
timing = metadata.get('timing', {})

print(f"Model: {metadata['timestamp']}")
print(f"")
print(f"Configuration:")
print(f"  Epochs: {config.get('num_epochs', 'N/A')}")
print(f"  Batch size: {config.get('batch_size', 'N/A')}")
print(f"  Sample size: {config.get('sample_size', 'N/A')}")
print(f"  Image size: {config.get('target_size', 'N/A')}")
print(f"")

# Display results based on model type
if '$TRAINING_TYPE' == '1':
    # DANN model
    best_acc = metadata.get('best_target_accuracy', 'N/A')
    best_epoch = metadata.get('best_epoch', 'N/A')
    print(f"Results:")
    print(f"  Best target accuracy: {best_acc:.2f}%" if isinstance(best_acc, (int, float)) else f"  Best target accuracy: {best_acc}")
    print(f"  Best epoch: {best_epoch}")
else:
    # Simple Detect model
    results = metadata.get('results', {})
    best_acc = results.get('best_test_acc', 'N/A')
    print(f"Results:")
    print(f"  Best test accuracy: {best_acc:.2f}%" if isinstance(best_acc, (int, float)) else f"  Best test accuracy: {best_acc}")

print(f"")

if timing:
    print(f"Timing:")
    print(f"  Training start: {timing.get('training_start', 'N/A')}")
    print(f"  Training end: {timing.get('training_end', 'N/A')}")
    total_hours = timing.get('total_training_time_hours', timing.get('total_time_hours', 0))
    avg_epoch = timing.get('average_epoch_time_seconds', timing.get('avg_epoch_time_seconds', 0))
    print(f"  Total time: {total_hours:.2f} hours" if isinstance(total_hours, (int, float)) else f"  Total time: {total_hours}")
    print(f"  Avg epoch time: {avg_epoch:.1f}s" if isinstance(avg_epoch, (int, float)) else f"  Avg epoch time: {avg_epoch}")
EOF
fi

echo ""
echo "======================================================================"
echo "✅ DOWNLOAD COMPLETED"
echo "======================================================================"
echo ""
echo "Downloaded to: $DOWNLOADED_MODEL"
echo ""

if [ -f "$DOWNLOADED_MODEL/training_curves.png" ] || [ -f "$DOWNLOADED_MODEL/loss_curve.png" ]; then
    echo "View training curves:"
    [ -f "$DOWNLOADED_MODEL/training_curves.png" ] && echo "  open $DOWNLOADED_MODEL/training_curves.png"
    [ -f "$DOWNLOADED_MODEL/loss_curve.png" ] && echo "  open $DOWNLOADED_MODEL/loss_curve.png"
    echo ""
fi

echo "View metadata:"
echo "  cat $DOWNLOADED_MODEL/metadata.json | python -m json.tool"
echo ""
echo "Monitor training logs on EC2:"
echo "  ./ec2_scripts/monitor_training.sh"
echo ""
echo "⚠️  Remember to stop/terminate EC2 instance!"
echo "  aws ec2 stop-instances --region $EC2_REGION --instance-ids $EC2_INSTANCE_ID"
echo "  aws ec2 terminate-instances --region $EC2_REGION --instance-ids $EC2_INSTANCE_ID"
echo "======================================================================"
