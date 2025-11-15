#!/bin/bash
# ============================================
# Sync Code Module
# ============================================
# Syncs all code and dependencies to EC2

sync_code() {
    echo "======================================================================"
    echo "SYNCING CODE"
    echo "======================================================================"

    # Create base directory structure on EC2
    echo "Creating base directory structure on EC2..."
    ssh -T -i "$EC2_KEY_PATH" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR "$EC2_SSH_USER@$EC2_PUBLIC_IP" "mkdir -p ~/ResponsibleAI/ec2_scripts ~/ResponsibleAI/domain_adapt ~/ResponsibleAI/simple_detect_car ~/ResponsibleAI/cardd_data/GenAI_Results/SD2 ~/ResponsibleAI/cardd_data/GenAI_Results/Kontext"
    echo "✓ Base directories created"
    echo ""

    # Change to ResponsibleAI root directory
    cd "$SCRIPT_DIR/.."

    # Sync requirements.txt first
    echo "Syncing requirements.txt..."
    rsync -avz \
        -e "ssh -i \"$EC2_KEY_PATH\" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR" \
        ec2_scripts/requirements.txt \
        "$EC2_SSH_USER@$EC2_PUBLIC_IP":~/ResponsibleAI/ec2_scripts/
    echo "✓ requirements.txt synced"
    echo ""

    echo "Syncing training scripts (always synced - will overwrite existing)..."
    rsync -avz \
        --include='train_dann.py' \
        --include='mnist_dann.py' \
        --include='model_dann.py' \
        --include='eval_dann.py' \
        --include='experiments.py' \
        --exclude='*' \
        -e "ssh -i \"$EC2_KEY_PATH\" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR" \
        domain_adapt/ \
        "$EC2_SSH_USER@$EC2_PUBLIC_IP":~/ResponsibleAI/domain_adapt/
    echo "✓ Training scripts synced"
    echo ""

    echo "Syncing remaining domain_adapt code..."
    rsync -avz --progress \
        --exclude='*.pth' \
        --exclude='*.h5' \
        --exclude='*.pkl' \
        --exclude='*.joblib' \
        --exclude='*.png' \
        --exclude='*.pdf' \
        --exclude='__pycache__' \
        --exclude='.git' \
        --exclude='models/' \
        --exclude='models_minst/' \
        --exclude='models*/' \
        --exclude='**/models/' \
        --exclude='**/models_minst/' \
        --exclude='**/models_mnist/' \
        -e "ssh -i \"$EC2_KEY_PATH\" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR" \
        domain_adapt/ \
        "$EC2_SSH_USER@$EC2_PUBLIC_IP":~/ResponsibleAI/domain_adapt/

    echo "✓ Code synced"
    echo ""

    # Sync ec2_scripts
    echo "Syncing EC2 scripts..."
    rsync -avz --progress \
        --exclude='.git' \
        -e "ssh -i \"$EC2_KEY_PATH\" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR" \
        ec2_scripts/ \
        "$EC2_SSH_USER@$EC2_PUBLIC_IP":~/ResponsibleAI/ec2_scripts/

    echo "✓ EC2 scripts synced"
    echo ""

    # Sync simple_detect_car (dependencies)
    echo "Syncing simple_detect_car dependencies..."
    rsync -avz --progress \
        --exclude='*.pth' \
        --exclude='*.h5' \
        --exclude='*.pkl' \
        --exclude='*.joblib' \
        --exclude='__pycache__' \
        --exclude='.git' \
        --exclude='.venv' \
        --exclude='models/' \
        --exclude='models_minst/' \
        --exclude='models_mnist/' \
        --exclude='models*/' \
        --exclude='**/models/' \
        --exclude='**/models_minst/' \
        --exclude='**/models_mnist/' \
        -e "ssh -i \"$EC2_KEY_PATH\" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR" \
        simple_detect_car/ \
        "$EC2_SSH_USER@$EC2_PUBLIC_IP":~/ResponsibleAI/simple_detect_car/

    echo "✓ Dependencies synced"
    echo ""
}
