#!/bin/bash
# ============================================
# Train Models on EC2
# ============================================
# This script launches training jobs on EC2 for:
# - Neural Network models (train_nn.py)
# - Domain Adversarial Neural Networks (cardd.py)

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
echo "TRAINING ON EC2"
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
# Select training type
# ============================================
echo "Select training type:"
echo "  [1] Neural Network (train_nn.py) - Car damage detection"
echo "  [2] Domain Adversarial NN (cardd.py) - SD2→Kontext adaptation"
echo ""

read -p "Choose [1-2]: " -n 1 -r TRAINING_TYPE
echo ""

case $TRAINING_TYPE in
    1)
        SCRIPT_NAME="train_nn.py"
        SESSION_NAME="nn_training"
        TRAINING_DIR="simple_detect_car"
        SCRIPT_DESC="Neural Network Training"
        ;;
    2)
        SCRIPT_NAME="cardd.py"
        SESSION_NAME="dann_training"
        TRAINING_DIR="domain_adapt"
        SCRIPT_DESC="DANN Training"
        ;;
    *)
        echo "❌ Invalid choice"
        exit 1
        ;;
esac

echo "✓ Selected: $SCRIPT_DESC ($SCRIPT_NAME)"
echo ""

# ============================================
# Configure training parameters
# ============================================
echo "======================================================================"
echo "CONFIGURE $SCRIPT_DESC"
echo "======================================================================"

if [ "$TRAINING_TYPE" = "1" ]; then
    # Neural Network training parameters
    echo "Domain options:"
    echo "  sd2    - Stable Diffusion 2 generated images"
    echo "  kontext - Kontext generated images"
    echo "  qwen   - Qwen generated images"
    echo ""
    read -p "Domain [sd2]: " DOMAIN
    DOMAIN=${DOMAIN:-sd2}

    read -p "Sample size (leave empty for full dataset): " SAMPLE_SIZE
    SAMPLE_SIZE=${SAMPLE_SIZE:-""}

    read -p "Target size [224]: " TARGET_SIZE
    TARGET_SIZE=${TARGET_SIZE:-224}

    read -p "Batch size [32]: " BATCH_SIZE
    BATCH_SIZE=${BATCH_SIZE:-32}

    read -p "Learning rate [0.001]: " LEARNING_RATE
    LEARNING_RATE=${LEARNING_RATE:-0.001}

    read -p "Epochs [20]: " EPOCHS
    EPOCHS=${EPOCHS:-20}

    echo "Model options:"
    echo "  vanilla - MLP network"
    echo "  cnn     - Convolutional Neural Network"
    echo ""
    read -p "Model [cnn]: " MODEL
    MODEL=${MODEL:-cnn}

    read -p "Hidden size [256]: " HIDDEN_SIZE
    HIDDEN_SIZE=${HIDDEN_SIZE:-256}

    # Build command
    CMD="cd ~/ResponsibleAI/$TRAINING_DIR && python $SCRIPT_NAME"
    CMD="$CMD --domain $DOMAIN"
    [ -n "$SAMPLE_SIZE" ] && CMD="$CMD --sample_size $SAMPLE_SIZE"
    CMD="$CMD --target_size $TARGET_SIZE"
    CMD="$CMD --batch_size $BATCH_SIZE"
    CMD="$CMD --learning_rate $LEARNING_RATE"
    CMD="$CMD --epochs $EPOCHS"
    CMD="$CMD --model $MODEL"
    CMD="$CMD --hidden_size $HIDDEN_SIZE"

elif [ "$TRAINING_TYPE" = "2" ]; then
    # DANN training parameters
    read -p "Number of epochs [5]: " EPOCHS
    EPOCHS=${EPOCHS:-5}

    read -p "Batch size [64]: " BATCH_SIZE
    BATCH_SIZE=${BATCH_SIZE:-64}

    read -p "Learning rate [0.001]: " LEARNING_RATE
    LEARNING_RATE=${LEARNING_RATE:-0.001}

    read -p "Gamma (domain loss weight) [5.0]: " GAMMA
    GAMMA=${GAMMA:-5.0}

    read -p "Zeta (gradient reversal) [1.0]: " ZETA
    ZETA=${ZETA:-1.0}

    # Build command
    CMD="cd ~/ResponsibleAI/$TRAINING_DIR && python $SCRIPT_NAME"
    CMD="$CMD --n_epoch $EPOCHS"
    CMD="$CMD --batch_size $BATCH_SIZE"
    CMD="$CMD --lr $LEARNING_RATE"
    CMD="$CMD --gamma $GAMMA"
    CMD="$CMD --zeta $ZETA"
fi

echo ""
echo "Training command:"
echo "  $CMD"
echo ""

# ============================================
# Check for existing training sessions (informational only)
# ============================================
echo "Checking for existing training sessions..."
EXISTING_SESSIONS=$(ssh -T -i "$EC2_KEY_PATH" -o LogLevel=ERROR -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "$EC2_SSH_USER@$EC2_PUBLIC_IP" \
    "screen -list | grep -E '$SESSION_NAME' | head -5" 2>/dev/null || echo "")

if [ -n "$EXISTING_SESSIONS" ]; then
    echo "ℹ️  Existing $SCRIPT_DESC sessions found:"
    echo "$EXISTING_SESSIONS" | sed 's/^/  /'
    echo ""
    echo "✓ Will start new session with unique timestamp: $UNIQUE_SESSION"
else
    echo "✓ No existing sessions found"
fi
echo ""

# ============================================
# Launch training in screen session
# ============================================
echo "======================================================================"
echo "LAUNCHING TRAINING"
echo "======================================================================"

# Generate unique session name with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
UNIQUE_SESSION="${SESSION_NAME}_${TIMESTAMP}"

echo "Starting training in screen session: $UNIQUE_SESSION"
echo ""

# Setup virtual environment and launch training
ssh -T -i "$EC2_KEY_PATH" -o LogLevel=ERROR -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "$EC2_SSH_USER@$EC2_PUBLIC_IP" << EOF
# Install uv if not available
if ! command -v uv &> /dev/null; then
    echo "Installing uv package manager..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

# Source uv environment
source "$HOME/.local/bin/env"

# Check if virtual environment exists, create if needed
if [ ! -d ~/ResponsibleAI/.venv ]; then
    echo "Creating virtual environment..."
    cd ~/ResponsibleAI
    uv venv
    echo "Installing PyTorch with CUDA support..."
    uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    echo "Installing other dependencies from requirements.txt..."
    uv pip install -r ec2_scripts/requirements.txt
fi

# Create screen session and run training
screen -dmS $UNIQUE_SESSION bash --noprofile --norc -c "
# Set up environment for this session
source "$HOME/.local/bin/env"

echo '======================================================================' > ~/ResponsibleAI/$TRAINING_DIR/training_log_\${UNIQUE_SESSION}.txt
echo '$SCRIPT_DESC - Started \$(date)' >> ~/ResponsibleAI/$TRAINING_DIR/training_log_\${UNIQUE_SESSION}.txt
echo '======================================================================' >> ~/ResponsibleAI/$TRAINING_DIR/training_log_\${UNIQUE_SESSION}.txt
echo '' >> ~/ResponsibleAI/$TRAINING_DIR/training_log_\${UNIQUE_SESSION}.txt

# Activate virtual environment
source ~/ResponsibleAI/.venv/bin/activate

# Run the training command with output to both screen and log file
$CMD 2>&1 | tee -a ~/ResponsibleAI/$TRAINING_DIR/training_log_\${UNIQUE_SESSION}.txt

echo '' >> ~/ResponsibleAI/$TRAINING_DIR/training_log_\${UNIQUE_SESSION}.txt
echo '======================================================================' >> ~/ResponsibleAI/$TRAINING_DIR/training_log_\${UNIQUE_SESSION}.txt
echo '$SCRIPT_DESC - Completed \$(date)' >> ~/ResponsibleAI/$TRAINING_DIR/training_log_\${UNIQUE_SESSION}.txt
echo '======================================================================' >> ~/ResponsibleAI/$TRAINING_DIR/training_log_\${UNIQUE_SESSION}.txt

# Keep screen alive briefly so user can see completion
sleep 3
"
EOF

echo "✓ Training launched in screen session: $UNIQUE_SESSION"
echo ""

# ============================================
# Verify training started
# ============================================
echo "Verifying training startup..."
sleep 3

SESSION_CHECK=$(ssh -T -i "$EC2_KEY_PATH" -o LogLevel=ERROR -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "$EC2_SSH_USER@$EC2_PUBLIC_IP" \
    "screen -list | grep '$UNIQUE_SESSION'" 2>/dev/null || echo "")

if [ -z "$SESSION_CHECK" ]; then
    echo "❌ Failed to start training session"
    exit 1
fi

echo "✓ Training session active: $UNIQUE_SESSION"
echo ""

# ============================================
# Show monitoring options
# ============================================
echo "======================================================================"
echo "TRAINING STARTED SUCCESSFULLY"
echo "======================================================================"
echo "Session: $UNIQUE_SESSION"
echo "Log file: ~/ResponsibleAI/$TRAINING_DIR/training_log_${UNIQUE_SESSION}.txt"
echo ""
echo "Monitoring options:"
echo "  1. Real-time monitoring:"
echo "     ./ec2_scripts/monitor_training.sh"
echo ""
echo "  2. SSH and attach to session:"
echo "     ssh -i '$EC2_KEY_PATH' $EC2_SSH_USER@$EC2_PUBLIC_IP"
echo "     screen -r $UNIQUE_SESSION"
echo ""
echo "  3. Check GPU usage:"
echo "     ssh -i '$EC2_KEY_PATH' $EC2_SSH_USER@$EC2_PUBLIC_IP 'nvidia-smi'"
echo ""
echo "⚠️  Remember to download results when training completes!"
echo "   ./ec2_scripts/download_results.sh"
echo "======================================================================"
