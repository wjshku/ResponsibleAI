#!/bin/bash
# ============================================
# Monitor Training on EC2
# ============================================
# This script shows real-time training progress

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
echo "MONITORING TRAINING ON EC2"
echo "======================================================================"
echo "Instance: $EC2_INSTANCE_ID"
echo "IP: $EC2_PUBLIC_IP"
echo ""

# ============================================
# Test connection
# ============================================
if ! ssh -o ConnectTimeout=5 -i "$EC2_KEY_PATH" "$EC2_SSH_USER@$EC2_PUBLIC_IP" "exit" 2>/dev/null; then
    echo "❌ Cannot connect to EC2 instance"
    exit 1
fi

# ============================================
# List available training sessions
# ============================================
echo "Checking available training sessions..."
SESSIONS=$(ssh -i "$EC2_KEY_PATH" "$EC2_SSH_USER@$EC2_PUBLIC_IP" \
    "screen -list | grep -E '(dann_training|nn_training|sk_training)' || echo ''")

if [ -z "$SESSIONS" ]; then
    echo "⚠️  No training sessions found"
    echo ""
    echo "Available screen sessions:"
    ssh -i "$EC2_KEY_PATH" "$EC2_SSH_USER@$EC2_PUBLIC_IP" "screen -list || echo 'No screens running'"
    exit 0
fi

echo "Available training sessions:"
echo "$SESSIONS"
echo ""

# Count sessions
SESSION_COUNT=$(echo "$SESSIONS" | wc -l)

if [ "$SESSION_COUNT" -eq "1" ]; then
    # Only one session, use it automatically
    SESSION_NAME=$(echo "$SESSIONS" | grep -oE '(dann_training|nn_training|sk_training)' | head -1)
    echo "Monitoring session: $SESSION_NAME"
else
    # Multiple sessions, let user choose
    echo "Select session to monitor:"
    echo "  [1] dann_training"
    echo "  [2] nn_training"
    echo "  [3] sk_training"
    echo ""
    read -p "Choose [1-3]: " -n 1 -r SESSION_CHOICE
    echo ""

    case $SESSION_CHOICE in
        1) SESSION_NAME="dann_training" ;;
        2) SESSION_NAME="nn_training" ;;
        3) SESSION_NAME="sk_training" ;;
        *)
            echo "Invalid choice"
            exit 1
            ;;
    esac
fi

# Determine directory and log file based on session
case $SESSION_NAME in
    dann_training)
        TRAINING_DIR="domain_adapt"
        LOG_FILE="training_log_dann_training.txt"
        ;;
    nn_training|sk_training)
        TRAINING_DIR="simple_detect_car"
        LOG_FILE="training_log_${SESSION_NAME}.txt"
        ;;
esac

echo "✓ Monitoring: $SESSION_NAME"
echo ""

# ============================================
# Show menu
# ============================================
echo "Select monitoring option:"
echo "  [1] Follow training log (tail -f)"
echo "  [2] Show last 50 lines"
echo "  [3] Show GPU status"
echo "  [4] Show training summary"
echo "  [5] Attach to screen session (Ctrl+A+D to detach)"
echo ""
read -p "Choose [1-5]: " -n 1 -r CHOICE
echo ""

case $CHOICE in
    1)
        echo "Following training log (Ctrl+C to exit)..."
        echo ""
        ssh -i "$EC2_KEY_PATH" "$EC2_SSH_USER@$EC2_PUBLIC_IP" \
            "cd ~/ResponsibleAI/$TRAINING_DIR && tail -f $LOG_FILE"
        ;;
    2)
        echo "Last 50 lines:"
        echo ""
        ssh -i "$EC2_KEY_PATH" "$EC2_SSH_USER@$EC2_PUBLIC_IP" \
            "cd ~/ResponsibleAI/$TRAINING_DIR && tail -50 $LOG_FILE"
        ;;
    3)
        echo "GPU Status:"
        echo ""
        ssh -i "$EC2_KEY_PATH" "$EC2_SSH_USER@$EC2_PUBLIC_IP" "nvidia-smi"
        ;;
    4)
        echo "Training Summary:"
        echo ""
        ssh -i "$EC2_KEY_PATH" "$EC2_SSH_USER@$EC2_PUBLIC_IP" << EOFSUM
cd ~/ResponsibleAI/$TRAINING_DIR

if [ ! -f $LOG_FILE ]; then
    echo "No training log found"
    exit 0
fi

echo "Training Progress:"
echo "─────────────────────────────────────────────────────────────────"
grep -E "Epoch [0-9]+/[0-9]+" $LOG_FILE | tail -5
echo ""

echo "Latest Metrics:"
echo "─────────────────────────────────────────────────────────────────"
grep -E "Label Accuracy|Lambda|Training completed|Accuracy|Loss" $LOG_FILE | tail -10
echo ""

echo "GPU Utilization:"
echo "─────────────────────────────────────────────────────────────────"
nvidia-smi --query-gpu=utilization.gpu,utilization.memory,temperature.gpu --format=csv,noheader
EOFSUM
        ;;
    5)
        echo "Attaching to screen session..."
        echo "Press Ctrl+A then D to detach without stopping training"
        echo ""
        sleep 2
        ssh -t -i "$EC2_KEY_PATH" "$EC2_SSH_USER@$EC2_PUBLIC_IP" \
            "screen -r $SESSION_NAME"
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac
