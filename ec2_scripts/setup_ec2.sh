#!/bin/bash
# ============================================
# EC2 Instance Setup Script for DANN Training
# ============================================
# This script launches and configures an EC2 instance for GPU training
#
# Prerequisites:
#   - AWS CLI installed: brew install awscli
#   - AWS credentials configured: aws configure
#   - EC2 key pair created in AWS console

set -e  # Exit on error

# ============================================
# Configuration
# ============================================
INSTANCE_TYPE="g4dn.xlarge"  # T4 GPU, $0.526/hour
# Alternative options:
# - g5.xlarge: A10G GPU, $1.006/hour (faster)
# - p3.2xlarge: V100 GPU, $3.06/hour (fastest)

REGION="us-east-1"
AMI_ID="ami-0c7217cdde317cfec"  # Deep Learning AMI (Ubuntu 20.04)
KEY_NAME="dann-training-key"
SECURITY_GROUP_NAME="dann-training-sg"
INSTANCE_NAME="dann-training-instance"
STORAGE_SIZE=100  # GB

echo "======================================================================"
echo "EC2 INSTANCE SETUP FOR DANN TRAINING"
echo "======================================================================"
echo "Instance Type: $INSTANCE_TYPE"
echo "Region: $REGION"
echo "Storage: ${STORAGE_SIZE}GB"
echo ""

# ============================================
# Check AWS CLI
# ============================================
if ! command -v aws &> /dev/null; then
    echo "❌ AWS CLI not found. Installing..."
    brew install awscli
fi

echo "✓ AWS CLI found"
echo ""

# ============================================
# Check AWS credentials
# ============================================
if ! aws sts get-caller-identity &> /dev/null; then
    echo "❌ AWS credentials not configured"
    echo "Run: aws configure"
    exit 1
fi

echo "✓ AWS credentials configured"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
echo "  Account ID: $ACCOUNT_ID"
echo ""

# ============================================
# Create Security Group (if not exists)
# ============================================
echo "Checking security group..."
SG_ID=$(aws ec2 describe-security-groups \
    --region $REGION \
    --filters "Name=group-name,Values=$SECURITY_GROUP_NAME" \
    --query 'SecurityGroups[0].GroupId' \
    --output text 2>/dev/null || echo "None")

if [ "$SG_ID" == "None" ]; then
    echo "Creating security group..."
    SG_ID=$(aws ec2 create-security-group \
        --region $REGION \
        --group-name $SECURITY_GROUP_NAME \
        --description "Security group for DANN training" \
        --query 'GroupId' \
        --output text)

    # Allow SSH from your IP
    MY_IP=$(curl -s https://checkip.amazonaws.com)
    aws ec2 authorize-security-group-ingress \
        --region $REGION \
        --group-id $SG_ID \
        --protocol tcp \
        --port 22 \
        --cidr ${MY_IP}/32

    echo "✓ Security group created: $SG_ID"
    echo "  SSH allowed from: $MY_IP"
else
    echo "✓ Security group exists: $SG_ID"
fi
echo ""

# ============================================
# Check Key Pair
# ============================================
echo "Checking key pair..."
if ! aws ec2 describe-key-pairs \
    --region $REGION \
    --key-names $KEY_NAME &> /dev/null; then
    echo ""
    echo "❌ Key pair '$KEY_NAME' not found"
    echo ""
    echo "Please create a key pair:"
    echo "1. Go to AWS Console → EC2 → Key Pairs"
    echo "2. Create key pair named: $KEY_NAME"
    echo "3. Download the .pem file to ~/.ssh/"
    echo "4. Run: chmod 400 ~/.ssh/${KEY_NAME}.pem"
    echo ""
    echo "Or create via CLI:"
    echo "  aws ec2 create-key-pair --region $REGION --key-name $KEY_NAME --query 'KeyMaterial' --output text > ~/.ssh/${KEY_NAME}.pem"
    echo "  chmod 400 ~/.ssh/${KEY_NAME}.pem"
    exit 1
fi

echo "✓ Key pair exists: $KEY_NAME"
echo ""

# ============================================
# Launch Instance
# ============================================
echo "======================================================================"
echo "LAUNCHING EC2 INSTANCE"
echo "======================================================================"
echo "This will start billing at $0.526/hour for g4dn.xlarge"
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted"
    exit 0
fi

INSTANCE_ID=$(aws ec2 run-instances \
    --region $REGION \
    --image-id $AMI_ID \
    --instance-type $INSTANCE_TYPE \
    --key-name $KEY_NAME \
    --security-group-ids $SG_ID \
    --block-device-mappings "[{\"DeviceName\":\"/dev/sda1\",\"Ebs\":{\"VolumeSize\":$STORAGE_SIZE,\"VolumeType\":\"gp3\"}}]" \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$INSTANCE_NAME}]" \
    --query 'Instances[0].InstanceId' \
    --output text)

echo "✓ Instance launched: $INSTANCE_ID"
echo ""

# ============================================
# Wait for instance to start
# ============================================
echo "Waiting for instance to start..."
aws ec2 wait instance-running --region $REGION --instance-ids $INSTANCE_ID

PUBLIC_IP=$(aws ec2 describe-instances \
    --region $REGION \
    --instance-ids $INSTANCE_ID \
    --query 'Reservations[0].Instances[0].PublicIpAddress' \
    --output text)

echo "✓ Instance running!"
echo ""

# ============================================
# Wait for SSH to be ready
# ============================================
echo "Waiting for SSH to be ready (this takes ~2 minutes)..."
for i in {1..60}; do
    if ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 \
        -i ~/.ssh/${KEY_NAME}.pem ubuntu@$PUBLIC_IP "exit" 2>/dev/null; then
        break
    fi
    echo -n "."
    sleep 5
done
echo ""
echo "✓ SSH ready!"
echo ""

# ============================================
# Setup environment
# ============================================
echo "======================================================================"
echo "CONFIGURING ENVIRONMENT"
echo "======================================================================"

ssh -i ~/.ssh/${KEY_NAME}.pem ubuntu@$PUBLIC_IP << 'EOF'
# Update system
sudo apt-get update -qq

# Create project directory
mkdir -p ~/ResponsibleAI/domain_adapt
mkdir -p ~/ResponsibleAI/cardd_data

# Install uv if not present
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi

# Install PyTorch with CUDA + required packages
uv pip install --system \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 \
    tqdm matplotlib scikit-learn pillow

echo "✓ Environment configured with uv"
EOF

echo ""

# ============================================
# Save connection info
# ============================================
CONFIG_FILE="ec2_scripts/ec2_config.sh"
cat > $CONFIG_FILE << EOF
# EC2 Instance Configuration
# Generated by setup_ec2.sh on $(date)

export EC2_INSTANCE_ID="$INSTANCE_ID"
export EC2_PUBLIC_IP="$PUBLIC_IP"
export EC2_REGION="$REGION"
export EC2_KEY_PATH="~/.ssh/${KEY_NAME}.pem"
export EC2_SSH_USER="ubuntu"

# SSH command
alias ssh-ec2="ssh -i \$EC2_KEY_PATH \$EC2_SSH_USER@\$EC2_PUBLIC_IP"
EOF

echo "✓ Configuration saved to: $CONFIG_FILE"
echo ""

# ============================================
# Success message
# ============================================
echo "======================================================================"
echo "✅ EC2 INSTANCE READY FOR TRAINING"
echo "======================================================================"
echo "Instance ID: $INSTANCE_ID"
echo "Public IP:   $PUBLIC_IP"
echo "Key:         ~/.ssh/${KEY_NAME}.pem"
echo ""
echo "To connect:"
echo "  ssh -i ~/.ssh/${KEY_NAME}.pem ubuntu@$PUBLIC_IP"
echo ""
echo "Or source the config:"
echo "  source $CONFIG_FILE"
echo "  ssh-ec2"
echo ""
echo "Next steps:"
echo "  1. Run: ./ec2_scripts/sync_to_ec2.sh    # Upload code and data"
echo "  2. Run: ./ec2_scripts/train_on_ec2.sh   # Start training"
echo "  3. Run: ./ec2_scripts/download_results.sh  # Get results"
echo ""
echo "⚠️  Remember to stop the instance when done to avoid charges!"
echo "  aws ec2 stop-instances --region $REGION --instance-ids $INSTANCE_ID"
echo "======================================================================"
