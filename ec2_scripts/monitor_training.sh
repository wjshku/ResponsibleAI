#!/bin/bash
# ============================================
# Monitor Training on EC2
# ============================================
# This script shows real-time training progress for multiple concurrent runs

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
    "screen -list | grep -E '(dann_training|nn_training|sk_training)' | head -20")

if [ -z "$SESSIONS" ]; then
    echo "⚠️  No training sessions found"
    echo ""
    echo "Available screen sessions:"
    ssh -i "$EC2_KEY_PATH" "$EC2_SSH_USER@$EC2_PUBLIC_IP" "screen -list || echo 'No screens running'"
    exit 0
fi

echo "Available training sessions:"
echo "$SESSIONS" | nl -v 1
echo ""

# Count sessions and create array
SESSION_COUNT=$(echo "$SESSIONS" | wc -l | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
SESSION_ARRAY=()
while IFS= read -r line; do
    SESSION_ARRAY+=("$line")
done <<< "$SESSIONS"

if [ "$SESSION_COUNT" -eq "1" ]; then
    # Only one session, use it automatically
    SESSION_NAME=$(echo "${SESSION_ARRAY[0]}" | grep -oE '(mnist_dann_training|dann_training|nn_training|sk_training)(_([0-9_]+))?')
    echo "Monitoring session: $SESSION_NAME"
else
    # Multiple sessions, let user choose
    echo "Select session to monitor:"
    for i in "${!SESSION_ARRAY[@]}"; do
        SESSION_ID=$(echo "${SESSION_ARRAY[$i]}" | grep -oE '(mnist_dann_training|dann_training|nn_training|sk_training)(_([0-9_]+))?')
        echo "  [$((i+1))] $SESSION_ID"
    done
    echo ""
    read -p "Choose [1-$SESSION_COUNT]: " SESSION_CHOICE
    echo ""

    if ! [[ "$SESSION_CHOICE" =~ ^[0-9]+$ ]] || [ "$SESSION_CHOICE" -lt 1 ] || [ "$SESSION_CHOICE" -gt "$SESSION_COUNT" ]; then
        echo "Invalid choice"
        exit 1
    fi

    SESSION_NAME=$(echo "${SESSION_ARRAY[$((SESSION_CHOICE-1))]}" | grep -oE '(mnist_dann_training|dann_training|nn_training|sk_training)(_([0-9_]+))?')
fi

# Determine directory and log file based on session
case $SESSION_NAME in
    mnist_dann_training_*)
        TRAINING_DIR="domain_adapt"
        LOG_FILE="training_log_${SESSION_NAME}.txt"
        ;;
    dann_training_*)
        TRAINING_DIR="domain_adapt"
        LOG_FILE="training_log_${SESSION_NAME}.txt"
        ;;
    dann_training)
        TRAINING_DIR="domain_adapt"
        LOG_FILE="training_log_dann_training.txt"
        ;;
    nn_training_*|sk_training_*)
        TRAINING_DIR="simple_detect_car"
        LOG_FILE="training_log_${SESSION_NAME}.txt"
        ;;
    nn_training|sk_training)
        TRAINING_DIR="simple_detect_car"
        LOG_FILE="training_log_${SESSION_NAME}.txt"
        ;;
esac

# Verify variables are set
if [ -z "$TRAINING_DIR" ] || [ -z "$LOG_FILE" ]; then
    echo "❌ Error: Could not determine directory/log file for session '$SESSION_NAME'"
    exit 1
fi

echo "✓ Monitoring: $SESSION_NAME (dir: $TRAINING_DIR, log: $LOG_FILE)"
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
echo "  [6] Terminate training session"
echo ""
read -p "Choose [1-6]: " -n 1 -r CHOICE
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
    6)
        echo "⚠️  TRAINING TERMINATION OPTIONS"
        echo "Session: $SESSION_NAME"
        echo "Directory: $TRAINING_DIR"
        echo ""
        echo "Select termination method:"
        echo "  [1] Interrupt training gracefully (Ctrl+C equivalent)"
        echo "  [2] Kill entire screen session (force quit)"
        echo "  [3] Cancel - go back to monitoring menu"
        echo ""
        read -p "Choose [1-3]: " -n 1 -r TERM_CHOICE
        echo ""

        case $TERM_CHOICE in
            1)
                echo "🛑 Sending interrupt signal to training process..."
                echo "This will attempt a graceful shutdown (may take a moment)"
                ssh -i "$EC2_KEY_PATH" "$EC2_SSH_USER@$EC2_PUBLIC_IP" << EOF
screen -S $SESSION_NAME -X stuff "^C"
sleep 2
echo "Checking if training stopped..."
ps aux | grep -E "(python|train)" | grep -v grep || echo "No training processes found"
EOF
                echo "✅ Interrupt signal sent"
                ;;
            2)
                echo "💀 Force killing screen session: $SESSION_NAME"
                echo "⚠️  This will immediately terminate all processes in the session"
                read -p "Are you sure? Type 'yes' to confirm: " CONFIRM
                if [ "$CONFIRM" = "yes" ]; then
                    ssh -i "$EC2_KEY_PATH" "$EC2_SSH_USER@$EC2_PUBLIC_IP" \
                        "screen -S $SESSION_NAME -X quit"
                    echo "✅ Screen session killed"
                    echo "💡 Tip: Check screen -list to confirm it's gone"
                else
                    echo "❌ Termination cancelled"
                fi
                ;;
            3)
                echo "↩️  Cancelled - returning to monitoring menu"
                ;;
            *)
                echo "❌ Invalid choice - termination cancelled"
                ;;
        esac
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac
