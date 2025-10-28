#!/bin/bash
# ============================================
# Launch Training on EC2
# ============================================
# This script starts DANN training on the EC2 instance

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
echo "LAUNCHING TRAINING ON EC2"
echo "======================================================================"
echo "Instance: $EC2_INSTANCE_ID"
echo "IP: $EC2_PUBLIC_IP"
echo ""

# ============================================
# Select training type
# ============================================
echo "Select training type:"
echo "  [1] DANN - Domain Adversarial Neural Network (domain_adapt)"
echo "  [2] Simple Detect - CNN/MLP for binary classification (simple_detect_car)"
echo "  [3] Simple Detect - Scikit-learn models (simple_detect_car)"
echo ""
read -p "Choose [1-3]: " -n 1 -r TRAINING_TYPE
echo ""

case $TRAINING_TYPE in
    1)
        TRAINING_DIR="domain_adapt"
        TRAINING_SCRIPT="train_dann.py"
        SESSION_NAME="dann_training"
        ;;
    2)
        TRAINING_DIR="simple_detect_car"
        TRAINING_SCRIPT="train_nn.py"
        SESSION_NAME="nn_training"
        ;;
    3)
        TRAINING_DIR="simple_detect_car"
        TRAINING_SCRIPT="train_sk.py"
        SESSION_NAME="sk_training"
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

echo "Selected: $TRAINING_SCRIPT"
echo ""

# ============================================
# Training configuration
# ============================================
if [ "$TRAINING_TYPE" == "1" ]; then
    # DANN configuration
    echo "Select DANN training configuration:"
    echo "  [1] Quick test (sample_size=1000, epochs=10) - ~30 min"
    echo "  [2] Medium (sample_size=3000, epochs=30) - ~2 hours"
    echo "  [3] Full training (all data, epochs=50) - ~4-6 hours"
    echo "  [4] Production (all data, epochs=100) - ~8-10 hours"
    echo "  [5] Custom"
    echo ""
    read -p "Choose [1-5]: " -n 1 -r CONFIG_CHOICE
    echo ""
else
    # Simple Detect configuration
    echo "Select dataset configuration:"
    echo "  [1] SD2 CarDD-TR (sample_size=500, epochs=10) - ~30 min"
    echo "  [2] SD2 CarDD-TR (full dataset, epochs=20) - ~2 hours"
    echo "  [3] Kontext CarDD-TR (sample_size=500, epochs=10) - ~30 min"
    echo "  [4] Custom"
    echo ""
    read -p "Choose [1-4]: " -n 1 -r CONFIG_CHOICE
    echo ""
fi

if [ "$TRAINING_TYPE" == "1" ]; then
    # DANN configuration
    case $CONFIG_CHOICE in
        1)
            SAMPLE_SIZE=1000
            NUM_EPOCHS=10
            BATCH_SIZE=32
            FEATURE_HIDDEN_SIZE=256
            GAMMA=10.0
            ;;
        2)
            SAMPLE_SIZE=3000
            NUM_EPOCHS=30
            BATCH_SIZE=32
            FEATURE_HIDDEN_SIZE=256
            GAMMA=10.0
            ;;
        3)
            SAMPLE_SIZE="None"
            NUM_EPOCHS=50
            BATCH_SIZE=32
            FEATURE_HIDDEN_SIZE=256
            GAMMA=10.0
            ;;
        4)
            SAMPLE_SIZE="None"
            NUM_EPOCHS=100
            BATCH_SIZE=32
            FEATURE_HIDDEN_SIZE=256
            GAMMA=10.0
            ;;
        5)
            read -p "Sample size (or 'None' for all): " SAMPLE_SIZE
            read -p "Number of epochs: " NUM_EPOCHS
            read -p "Batch size: " BATCH_SIZE
            read -p "Feature hidden size: " FEATURE_HIDDEN_SIZE
            read -p "Gamma (lambda schedule sharpness, default 10.0): " GAMMA
            ;;
        *)
            echo "Invalid choice"
            exit 1
            ;;
    esac

    echo ""
    echo "DANN Training configuration:"
    echo "  Sample size: $SAMPLE_SIZE"
    echo "  Epochs: $NUM_EPOCHS"
    echo "  Batch size: $BATCH_SIZE"
    echo "  Feature hidden size: $FEATURE_HIDDEN_SIZE"
    echo "  Gamma (lambda schedule): $GAMMA"
    echo ""
else
    # Simple Detect configuration
    case $CONFIG_CHOICE in
        1)
            IMG2IMG_TYPE="SD2"
            DATA_TYPE="CarDD-TR"
            SAMPLE_SIZE=500
            NUM_EPOCHS=10
            HIDDEN_SIZE=256
            ;;
        2)
            IMG2IMG_TYPE="SD2"
            DATA_TYPE="CarDD-TR"
            SAMPLE_SIZE="None"
            NUM_EPOCHS=20
            HIDDEN_SIZE=256
            ;;
        3)
            IMG2IMG_TYPE="Kontext"
            DATA_TYPE="CarDD-TR"
            SAMPLE_SIZE=500
            NUM_EPOCHS=10
            HIDDEN_SIZE=256
            ;;
        4)
            read -p "Image type (SD2/Kontext): " IMG2IMG_TYPE
            read -p "Data type (CarDD-TR/CarDD-TE/CarDD-VAL): " DATA_TYPE
            read -p "Sample size (or 'None' for all): " SAMPLE_SIZE
            read -p "Number of epochs: " NUM_EPOCHS
            read -p "Hidden size: " HIDDEN_SIZE
            ;;
        *)
            echo "Invalid choice"
            exit 1
            ;;
    esac

    echo ""
    echo "Simple Detect Training configuration:"
    echo "  Model: $TRAINING_SCRIPT"
    echo "  Image type: $IMG2IMG_TYPE"
    echo "  Dataset: $DATA_TYPE"
    echo "  Sample size: $SAMPLE_SIZE"
    echo "  Epochs: $NUM_EPOCHS"
    echo "  Hidden size: $HIDDEN_SIZE"
    echo ""
fi

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
# Verify GPU
# ============================================
echo "Verifying GPU..."
GPU_INFO=$(ssh -i "$EC2_KEY_PATH" "$EC2_SSH_USER@$EC2_PUBLIC_IP" "nvidia-smi --query-gpu=name --format=csv,noheader" 2>/dev/null || echo "No GPU")
echo "✓ GPU: $GPU_INFO"
echo ""

# ============================================
# Update training script with config
# ============================================
echo "Updating training configuration..."

if [ "$TRAINING_TYPE" == "1" ]; then
    # Update DANN script
    ssh -i "$EC2_KEY_PATH" "$EC2_SSH_USER@$EC2_PUBLIC_IP" << EOF
cd ~/ResponsibleAI/domain_adapt

python3 << 'PYTHON_EOF'
import re

with open('train_dann.py', 'r') as f:
    content = f.read()

# Update config
content = re.sub(
    r"'batch_size': \d+,",
    "'batch_size': $BATCH_SIZE,",
    content
)
content = re.sub(
    r"'num_epochs': \d+,",
    "'num_epochs': $NUM_EPOCHS,",
    content
)
content = re.sub(
    r"'sample_size': .*,  # Use None for full dataset",
    "'sample_size': $SAMPLE_SIZE,  # Use None for full dataset",
    content
)
content = re.sub(
    r"'feature_hidden_size': \d+,",
    "'feature_hidden_size': $FEATURE_HIDDEN_SIZE,",
    content
)
content = re.sub(
    r"'gamma': [\d.]+,",
    "'gamma': $GAMMA,",
    content
)

with open('train_dann.py', 'w') as f:
    f.write(content)

print("✓ DANN configuration updated")
PYTHON_EOF
EOF
else
    # Update simple_detect_car script
    ssh -i "$EC2_KEY_PATH" "$EC2_SSH_USER@$EC2_PUBLIC_IP" << EOF
cd ~/ResponsibleAI/simple_detect_car

python3 << 'PYTHON_EOF'
import re

with open('$TRAINING_SCRIPT', 'r') as f:
    content = f.read()

# Update img2img_type and data_type
content = re.sub(
    r'img2img_type, data_type = .*',
    'img2img_type, data_type = "$IMG2IMG_TYPE", "$DATA_TYPE"',
    content
)

# Update sample_size
if "$SAMPLE_SIZE" == "None":
    content = re.sub(
        r"'sample_size': \d+,.*",
        "'sample_size': None,  # Use full dataset",
        content
    )
else:
    content = re.sub(
        r"'sample_size': \d+,.*",
        "'sample_size': $SAMPLE_SIZE,  # Sample size",
        content
    )

# Update num_epochs
content = re.sub(
    r"'num_epochs': \d+,",
    "'num_epochs': $NUM_EPOCHS,",
    content
)

# Update hidden_size
content = re.sub(
    r"'hidden_size': \d+,",
    "'hidden_size': $HIDDEN_SIZE,",
    content
)

with open('$TRAINING_SCRIPT', 'w') as f:
    f.write(content)

print("✓ $TRAINING_SCRIPT configuration updated")
PYTHON_EOF
EOF
fi

echo "✓ Configuration updated"
echo ""

# ============================================
# Install dependencies
# ============================================
echo "======================================================================"
echo "CHECKING DEPENDENCIES"
echo "======================================================================"

ssh -i "$EC2_KEY_PATH" "$EC2_SSH_USER@$EC2_PUBLIC_IP" << 'EOF'
# Check disk space
echo "Checking disk space..."
df -h / | tail -1
AVAILABLE=$(df / --output=avail | tail -1 | tr -dc '0-9')
if [ "$AVAILABLE" -lt "10000000" ]; then
    echo "⚠️  Low disk space (less than 10GB available)"
    echo "Cleaning up caches..."

    # Clean apt cache
    sudo apt-get clean

    # Clean pip/uv cache
    rm -rf ~/.cache/pip
    rm -rf ~/.cache/uv

    # Clean old logs
    sudo journalctl --vacuum-time=3d

    echo "Disk space after cleanup:"
    df -h / | tail -1
fi

# Install uv if not available
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Source the uv environment
    if [ -f "$HOME/.local/bin/env" ]; then
        source "$HOME/.local/bin/env"
    fi
    # Also try the cargo path
    export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
fi

# Verify uv is available
if ! command -v uv &> /dev/null; then
    echo "❌ uv installation failed"
    exit 1
fi

echo "✓ uv available"

# Create or activate virtual environment
cd ~/ResponsibleAI
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    uv venv .venv
fi

# Activate virtual environment
source .venv/bin/activate
echo "✓ Virtual environment activated"

# Check if PyTorch is installed
if ! python -c "import torch" 2>/dev/null; then
    echo "Installing PyTorch and dependencies..."

    # Clean cache before installing large packages
    uv cache clean 2>/dev/null || true

    # Install PyTorch with CUDA support
    echo "Installing PyTorch (this may take several minutes)..."
    uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

    # Install other dependencies
    echo "Installing other dependencies..."
    uv pip install -r requirements.txt

    echo "✓ Dependencies installed"
else
    echo "✓ PyTorch already installed"
    # Still install/update other dependencies
    uv pip install -r requirements.txt --quiet 2>/dev/null || true
fi

# Verify installation
python3 << 'PYTHON'
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Device: {torch.cuda.get_device_name(0)}")
PYTHON
EOF

echo ""

# ============================================
# Start training
# ============================================
echo "======================================================================"
echo "STARTING TRAINING"
echo "======================================================================"
echo ""
echo "Training will run in a detached screen session"
echo "You can close this terminal - training will continue"
echo ""
read -p "Start training? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted"
    exit 0
fi

# Start training in screen session
ssh -i "$EC2_KEY_PATH" "$EC2_SSH_USER@$EC2_PUBLIC_IP" << EOFSCREEN
cd ~/ResponsibleAI/$TRAINING_DIR

# Install screen if not available
if ! command -v screen &> /dev/null; then
    echo "Installing screen..."
    sudo apt-get update -qq
    sudo apt-get install -y screen
fi

# Kill any existing training sessions with this name
screen -X -S $SESSION_NAME quit 2>/dev/null || true

# Start new screen session with robust error handling
screen -dmS $SESSION_NAME bash -c "
    exec > >(tee training_log_$SESSION_NAME.txt) 2>&1  # Redirect all output to log file

    echo '========================================'
    echo 'Training session: $SESSION_NAME'
    echo 'Started: \$(date)'
    echo '========================================'

    cd ~/ResponsibleAI/$TRAINING_DIR || { echo 'ERROR: Failed to cd to $TRAINING_DIR'; exit 1; }

    if [ -f ~/ResponsibleAI/.venv/bin/activate ]; then
        source ~/ResponsibleAI/.venv/bin/activate || { echo 'ERROR: Failed to activate venv'; exit 1; }
    else
        echo 'ERROR: Virtual environment not found at ~/ResponsibleAI/.venv'
        exit 1
    fi

    if [ ! -f $TRAINING_SCRIPT ]; then
        echo 'ERROR: Training script not found: $TRAINING_SCRIPT'
        ls -la
        exit 1
    fi

    python3 -u $TRAINING_SCRIPT
    EXIT_CODE=\$?
    echo ''
    echo '========================================'
    echo 'Training completed with exit code: '\$EXIT_CODE
    echo 'Finished: \$(date)'
    echo '========================================'
"

# Wait a moment for screen to start
sleep 2

# Check if screen started
if screen -list | grep -q $SESSION_NAME; then
    echo "✓ Training started in screen session '$SESSION_NAME'"
else
    echo "❌ Failed to start screen session"
    echo "Checking for errors..."
    if [ -f training_log_$SESSION_NAME.txt ]; then
        echo "Last 10 lines of training log:"
        tail -10 training_log_$SESSION_NAME.txt
    fi
    exit 1
fi
EOFSCREEN

echo ""

# ============================================
# Monitor initial output
# ============================================
echo "======================================================================"
echo "INITIAL TRAINING OUTPUT"
echo "======================================================================"
echo "Waiting for training to start..."
sleep 3

ssh -i "$EC2_KEY_PATH" "$EC2_SSH_USER@$EC2_PUBLIC_IP" << EOFMONITOR
cd ~/ResponsibleAI/$TRAINING_DIR

# Wait for log file to appear (up to 10 seconds)
for i in {1..10}; do
    if [ -f training_log_$SESSION_NAME.txt ]; then
        echo "Log file found, showing initial output:"
        echo "──────────────────────────────────────────────────────────────────"
        head -50 training_log_$SESSION_NAME.txt
        break
    fi
    echo "Waiting for log file... (\$i/10)"
    sleep 1
done

if [ ! -f training_log_$SESSION_NAME.txt ]; then
    echo "❌ Log file not created after 10 seconds"
    echo "This usually means the screen session failed to start properly"
    echo ""
    echo "Checking screen sessions:"
    screen -list
    echo ""
    echo "Checking directory contents:"
    ls -la
fi
EOFMONITOR

echo ""

# ============================================
# Instructions
# ============================================
echo "======================================================================"
echo "✅ TRAINING LAUNCHED"
echo "======================================================================"
echo ""
echo "Training Details:"
echo "  Script: $TRAINING_SCRIPT"
echo "  Session: $SESSION_NAME"
echo "  Log file: training_log_$SESSION_NAME.txt"
echo ""
echo "Monitor training (SSH manually):"
echo "  ssh -i \"$EC2_KEY_PATH\" \"$EC2_SSH_USER@$EC2_PUBLIC_IP\""
echo "  cd ~/ResponsibleAI/$TRAINING_DIR"
echo "  tail -f training_log_$SESSION_NAME.txt    # Follow log"
echo "  screen -r $SESSION_NAME                    # Attach to session"
echo "  watch -n 1 nvidia-smi                      # Monitor GPU"
echo ""
echo "Download results when done:"
echo "  ./ec2_scripts/download_results.sh"
echo ""
if [ "$TRAINING_TYPE" == "1" ]; then
    echo "Estimated completion:"
    case $CONFIG_CHOICE in
        1) echo "  ~30 minutes" ;;
        2) echo "  ~2 hours" ;;
        3) echo "  ~4-6 hours" ;;
        4) echo "  ~8-10 hours" ;;
    esac
else
    echo "Estimated completion:"
    case $CONFIG_CHOICE in
        1) echo "  ~30 minutes" ;;
        2) echo "  ~2 hours" ;;
        3) echo "  ~30 minutes" ;;
    esac
fi
echo ""
echo "⚠️  Don't forget to stop the instance when done!"
echo "======================================================================"
