# EC2 Training Scripts

Complete automation scripts for training models on AWS EC2 GPU instances.

**Supports:**
- DANN (Domain Adversarial Neural Networks) - `domain_adapt/`
- Simple Detect (CNN/MLP and sklearn models) - `simple_detect_car/`

## 📋 Two Ways to Launch EC2

### Method 1: AWS Console (Recommended for Beginners) ⭐

Launch instance via AWS Console web interface, then use scripts for everything else.

**👉 See [QUICKSTART_CONSOLE.md](QUICKSTART_CONSOLE.md) for step-by-step guide**

```bash
# 1. Launch instance in AWS Console (g4dn.xlarge, Deep Learning AMI)
# 2. Connect to existing instance
./ec2_scripts/connect_existing.sh

# 3. Upload code and data
./ec2_scripts/sync_to_ec2.sh

# 4. Start training
./ec2_scripts/train_on_ec2.sh

# 5. Download results
./ec2_scripts/download_results.sh
```

### Method 2: AWS CLI (Advanced - Fully Automated)

Launch everything via command line. Requires AWS CLI configured.

**Prerequisites:**
- AWS CLI installed: `brew install awscli`
- AWS credentials: `aws configure`

```bash
# 1. Launch and configure EC2 instance (~5 minutes)
./ec2_scripts/setup_ec2.sh

# 2. Upload code and data (~10-15 minutes)
./ec2_scripts/sync_to_ec2.sh

# 3. Start training (runs in background)
./ec2_scripts/train_on_ec2.sh

# 4. Download results when complete
./ec2_scripts/download_results.sh
```

---

## 📁 Scripts Overview

### 0. `connect_existing.sh` - Connect to Console-Launched Instance ⭐
**What it does:**
- Connects to an EC2 instance you launched via AWS Console
- Tests SSH connection
- Gathers instance details (ID, region, GPU)
- Configures PyTorch environment
- Saves connection info to `ec2_config.sh`

**Usage:**
```bash
./ec2_scripts/connect_existing.sh
# Enter instance IP: 3.123.45.67
# Enter key path: ~/.ssh/dann-training-key.pem
```

**Time:** ~2 minutes

**When to use:** After launching instance via AWS Console

---

### 1. `setup_ec2.sh` - Instance Setup (CLI Method)
**What it does:**
- Creates security group (SSH access)
- Launches EC2 GPU instance (g4dn.xlarge by default)
- Waits for instance to be ready
- Configures PyTorch environment
- Saves connection info to `ec2_config.sh`

**Usage:**
```bash
./ec2_scripts/setup_ec2.sh
```

**Time:** ~5 minutes
**Cost:** Starts billing at $0.526/hour

**Output:**
```
Instance ID: i-0123456789abcdef
Public IP:   3.123.45.67
Key:         ~/.ssh/dann-training-key.pem
```

---

### 2. `sync_to_ec2.sh` - Upload Code & Data
**What it does:**
- Syncs `domain_adapt/` code to EC2
- Syncs `simple_detect_car/` dependencies
- Uploads CarDD dataset (~6000 images)
- Updates file paths for EC2 environment

**Usage:**
```bash
./ec2_scripts/sync_to_ec2.sh
```

**Time:** ~10-15 minutes (first time), ~1 minute (subsequent)
**Notes:**
- Data sync is skipped if already present
- Only uploads necessary files (excludes models, PDFs, etc.)

---

### 3. `train_on_ec2.sh` - Launch Training
**What it does:**
- Verifies GPU availability
- Prompts for training configuration
- Updates training parameters
- Starts training in detached `screen` session
- Training continues even if you disconnect

**Usage:**
```bash
./ec2_scripts/train_on_ec2.sh
```

**Training Options:**
```
[1] Quick test     - 1000 samples, 10 epochs  (~30 min)
[2] Medium         - 3000 samples, 30 epochs  (~2 hours)
[3] Full training  - All data, 50 epochs      (~4-6 hours)
[4] Production     - All data, 100 epochs     (~8-10 hours)
[5] Custom         - Choose your own settings
```

**What happens:**
- Training runs in a `screen` session named `dann_training`
- Output saved to `training_log.txt`
- You can close terminal - training continues
- Monitor with `monitor_training.sh`

---

### 4. `monitor_training.sh` - Check Progress
**What it does:**
- Shows training status
- Real-time log following
- GPU utilization
- Training summary

**Usage:**
```bash
./ec2_scripts/monitor_training.sh
```

**Options:**
```
[1] Follow training log     - Live updates (tail -f)
[2] Show last 50 lines      - Recent output
[3] Show GPU status         - nvidia-smi
[4] Show training summary   - Progress overview
[5] Attach to session       - Full terminal access
```

---

### 5. `download_results.sh` - Get Models
**What it does:**
- Lists all trained models
- Downloads selected model(s)
- Downloads training logs
- Shows training summary

**Usage:**
```bash
./ec2_scripts/download_results.sh
```

**Options:**
- Download `latest` model
- Download `all` models
- Download specific model by number

**Downloads:**
- `models/model_dann_YYYYMMDD_HHMMSS/`
  - `dann_best.pth` - Best model weights
  - `dann_final.pth` - Final model weights
  - `metadata.json` - Complete training history
  - `training_curves.png` - Visualization
- `training_log_ec2.txt` - Full training log

---

## 💰 Cost Management

### Hourly Costs
| Instance Type | GPU | Cost/Hour | Best For |
|--------------|-----|-----------|----------|
| **g4dn.xlarge** | T4 (16GB) | $0.526 | Development, testing |
| g5.xlarge | A10G (24GB) | $1.006 | Production training |
| p3.2xlarge | V100 (16GB) | $3.06 | Large-scale experiments |

### Typical Training Costs
| Configuration | Time | Cost (g4dn.xlarge) |
|--------------|------|-------------------|
| Quick test | 30 min | $0.26 |
| Medium | 2 hours | $1.05 |
| Full (50 epochs) | 5 hours | $2.63 |
| Production (100 epochs) | 10 hours | $5.26 |

### Stop Instance When Done! ⚠️
```bash
# Stop instance (keeps data, stops billing)
aws ec2 stop-instances --region us-east-1 --instance-ids i-xxxxx

# Terminate instance (deletes everything, stops billing)
aws ec2 terminate-instances --region us-east-1 --instance-ids i-xxxxx
```

---

## 🔧 Configuration

### Change Instance Type
Edit `setup_ec2.sh`:
```bash
INSTANCE_TYPE="g5.xlarge"  # For faster training
```

### Change Training Parameters
Option 1: Interactive (recommended)
```bash
./ec2_scripts/train_on_ec2.sh
# Choose option [5] Custom
```

Option 2: Manual
Edit `train_dann.py` on EC2:
```bash
ssh -i ~/.ssh/dann-training-key.pem ubuntu@<IP>
cd ~/ResponsibleAI/domain_adapt
nano train_dann.py  # Edit config section
```

---

## 🐛 Troubleshooting

### Cannot Connect to EC2
```bash
# Check instance status
aws ec2 describe-instances --instance-ids i-xxxxx --query 'Reservations[0].Instances[0].State.Name'

# Start if stopped
aws ec2 start-instances --instance-ids i-xxxxx
```

### Training Not Starting
```bash
# SSH into instance
ssh -i ~/.ssh/dann-training-key.pem ubuntu@<IP>

# Check GPU
nvidia-smi

# Check Python environment
source activate pytorch
python --version
```

### Sync Fails
```bash
# Check disk space on EC2
ssh -i ~/.ssh/dann-training-key.pem ubuntu@<IP> "df -h"

# Clear old models if needed
ssh -i ~/.ssh/dann-training-key.pem ubuntu@<IP> "rm -rf ~/ResponsibleAI/domain_adapt/models/*"
```

### Training Crashed
```bash
# Check logs
ssh -i ~/.ssh/dann-training-key.pem ubuntu@<IP>
cd ~/ResponsibleAI/domain_adapt
tail -100 training_log.txt

# Check system logs
dmesg | tail -50
```

---

## 📊 Monitoring Training

### From Local Machine
```bash
# Quick check
./ec2_scripts/monitor_training.sh

# Or manually
ssh -i ~/.ssh/dann-training-key.pem ubuntu@<IP>
cd ~/ResponsibleAI/domain_adapt
tail -f training_log.txt
```

### Understanding Output
```
Epoch 25/50
──────────────────────────────────────────────────────────────────
Metric               Train        Target                   Info
──────────────────────────────────────────────────────────────────
Label Loss          0.4521        0.4892
Label Accuracy     87.30%        84.50%
Precision          0.8654        0.8321
Recall             0.8801        0.8542
F1 Score           0.8727        0.8430
──────────────────────────────────────────────────────────────────
Domain Accuracy     48.23%     (fakes only)          ✓ Good
──────────────────────────────────────────────────────────────────
Lambda: 0.9866 | Train: 245.3s | Eval: 12.7s | Total: 258.0s
```

**Good signs:**
- Label accuracy increasing
- Domain accuracy near 50% (✓ Good)
- Training progressing smoothly

---

## 🔐 Security Best Practices

1. **Key Management**
   - Keep `.pem` files secure (`chmod 400`)
   - Never commit keys to git
   - Use different keys for different projects

2. **Security Groups**
   - Only allows SSH from your IP
   - Automatically configured by `setup_ec2.sh`
   - Update if your IP changes:
     ```bash
     MY_IP=$(curl -s https://checkip.amazonaws.com)
     aws ec2 authorize-security-group-ingress \
       --group-id sg-xxxxx \
       --protocol tcp --port 22 --cidr ${MY_IP}/32
     ```

3. **Instance Access**
   - Use IAM roles instead of hardcoded credentials
   - Enable CloudWatch for monitoring
   - Set billing alerts

---

## 📚 Additional Resources

### AWS CLI Reference
- [EC2 Commands](https://docs.aws.amazon.com/cli/latest/reference/ec2/)
- [Instance Types](https://aws.amazon.com/ec2/instance-types/)
- [Pricing Calculator](https://calculator.aws/)

### Training Tips
- Start with quick test to verify everything works
- Monitor GPU utilization (`nvidia-smi`)
- Check logs regularly for errors
- Download models periodically as backup

---

## 🎯 Example Full Workflow

```bash
# Day 1: Setup and quick test
./ec2_scripts/setup_ec2.sh
./ec2_scripts/sync_to_ec2.sh
./ec2_scripts/train_on_ec2.sh  # Choose [1] Quick test
# Wait 30 minutes
./ec2_scripts/download_results.sh

# Day 2: Full training
./ec2_scripts/train_on_ec2.sh  # Choose [3] Full training
# Check progress
./ec2_scripts/monitor_training.sh
# Wait 5 hours (or check periodically)
./ec2_scripts/download_results.sh

# Stop instance to avoid charges
aws ec2 stop-instances --instance-ids $(source ec2_scripts/ec2_config.sh && echo $EC2_INSTANCE_ID)
```

---

## ❓ FAQ

**Q: Can I run multiple trainings?**
A: Yes, but you need multiple instances. Each script targets one instance at a time.

**Q: What if training fails mid-way?**
A: Check `training_log.txt`. Models are saved after each epoch, so you won't lose everything.

**Q: Can I pause training?**
A: Stop the instance with `aws ec2 stop-instances`. Restart with `aws ec2 start-instances`. Resume training manually.

**Q: How do I change regions?**
A: Edit `REGION` in `setup_ec2.sh` before running.

**Q: Can I use Spot instances?**
A: Yes, but add `--instance-market-options` to the `run-instances` command in `setup_ec2.sh`.

---

**Questions or issues?** Check the troubleshooting section or create an issue.
