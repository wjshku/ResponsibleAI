# Domain Adversarial Neural Network (DANN) for Fake Image Detection

Implementation of "Unsupervised Domain Adaptation by Backpropagation" (Ganin & Lempitsky, 2015) for detecting AI-generated fake images across different generative models.

## 📋 Overview

This implementation trains a model to:
1. **Classify real vs fake images** (main task)
2. **Adapt across domains** - Learn features invariant to which generative model created the fakes

**Problem Setup:**
- **Source Domain (SD2)**: Real images + SD2-generated fakes
- **Target Domain (Kontext)**: Real images + Kontext-generated fakes
- **Goal**: Train on SD2, generalize to Kontext without labeled Kontext data

## 📁 File Structure

```
domain_adapt/
├── model_dann.py         # DANN model architecture (GRL + CNN)
├── train_dann.py         # Training script
├── eval_dann.py          # Evaluation script
├── README.md             # This file
└── Unsupervised Domain Adaptation by Backpropagation.pdf
```

## 🏗️ Architecture

```
Input Image
    ↓
Feature Extractor (CNN)
    ↓
Features (512-dim)
    ↓
    ├─→ Label Predictor → Real/Fake classification
    │
    └─→ [GRL] → Domain Classifier → SD2/Kontext classification
```

**Key Innovation: Gradient Reversal Layer (GRL)**
- Forward pass: Identity (x → x)
- Backward pass: Reverses gradient (∂L/∂x → -λ * ∂L/∂x)
- Makes feature extractor learn domain-invariant features

## 🚀 Quick Start

### Step 1: Activate Environment
```bash
conda activate trisure
cd "/Users/wjs/Library/CloudStorage/OneDrive-Personal/Coding, ML & DL/ResponsibleAI/domain_adapt"
```

### Step 2: Train the Model
```bash
python train_dann.py
```

**Training takes ~1-2 hours** depending on dataset size.

### Step 3: Evaluate the Model
```bash
python eval_dann.py
```

Select the trained model and choose evaluation domain.

## ⚙️ Configuration

Edit `train_dann.py` (lines 100-140) to customize:

```python
config = {
    # Data sources
    'source_name': 'SD2',              # Source domain
    'source_data_type': 'CarDD-TR',    # Training split
    'target_name': 'Kontext',          # Target domain
    'target_data_type': 'CarDD-TR',    # Training split

    # Training parameters
    'batch_size': 32,                  # Adjust for GPU memory
    'num_epochs': 50,                  # More epochs = better adaptation
    'learning_rate': 1e-4,             # Adam learning rate
    'target_size': (512, 512),         # Image resolution
    'sample_size': None,               # None = full dataset, or e.g., 1000

    # Model architecture
    'feature_hidden_size': 256,        # Label predictor size
    'domain_hidden_size': 256,         # Domain classifier size
    'dropout': 0.3,                    # Dropout probability
}
```

## 📊 Training Details

### What Happens During Training

1. **Loads two domains:**
   - Source: Real + SD2-generated fakes (labeled)
   - Target: Real + Kontext-generated fakes (labeled but labels only used for evaluation)

2. **Training process:**
   - **Label loss**: Learns to classify real vs fake on source domain
   - **Domain loss**: Only on fake images (SD2-fakes vs Kontext-fakes)
   - **Lambda schedule**: Gradually increases from 0 → 1 (adaptation strength)

3. **Outputs saved to:** `models/model_dann_YYYYMMDD_HHMMSS/`
   - `dann_best.pth` - Best model weights
   - `dann_final.pth` - Final model weights
   - `metadata.json` - Configuration and training history
   - `training_curves.png` - Loss/accuracy plots

### Metrics to Monitor

| Metric | Training Progress | Good Sign |
|--------|------------------|-----------|
| **Label loss** | Should decrease | ✅ < 0.3 |
| **Label accuracy (source)** | Should increase | ✅ > 90% |
| **Domain loss** | May fluctuate | Competing objectives |
| **Domain accuracy** | Should approach 50% | ✅ 45-55% |
| **Target accuracy** | Should improve | ✅ Better than baseline |

**Domain accuracy → 50% = Good!** This means SD2-fakes and Kontext-fakes are indistinguishable.

## 📈 Evaluation

### Run Evaluation Script

```bash
python eval_dann.py
```

### Interactive Steps:
1. Select a trained model from list
2. Choose evaluation domain:
   - `[0]` Source domain (SD2)
   - `[1]` Target domain (Kontext)
   - `[2]` Both domains (recommended)

### Metrics Reported

**Label Classification (Real vs Fake):**
- Overall accuracy
- Real accuracy, Fake accuracy
- Precision, Recall, F1-score
- Confusion matrix

**Domain Classification:**
- Domain accuracy on fake images
  - **✅ Good:** 45-55% (domains indistinguishable)
  - **❌ Poor:** >60% (adaptation failed)

### Output Files

- Console: Detailed metrics
- `evaluation_results.json`: Saved metrics for later analysis

## 🎯 Expected Results

### Success Indicators

✅ **Target domain accuracy > Source-only CNN baseline**
- DANN should improve accuracy on Kontext fakes compared to training only on SD2

✅ **Domain accuracy ≈ 50%**
- Features are domain-invariant (can't tell SD2-fakes from Kontext-fakes)

✅ **Real image accuracy stays high**
- Since real images are identical across domains

### Typical Accuracy

| Model | Train on | Test on Kontext | Improvement |
|-------|----------|----------------|-------------|
| **Source-only CNN** | SD2 | ~70-80% | Baseline |
| **DANN (ours)** | SD2 + Kontext | **~85-90%** | +10-15% |

## 🔬 Testing

### Test Model Architecture
```bash
python model_dann.py
```

Runs built-in tests:
- Creates DANN model
- Tests forward pass with different lambda values
- Verifies gradient reversal works correctly

## 📝 Complete Workflow

```bash
# 1. Train DANN model
conda activate trisure
cd domain_adapt
python train_dann.py

# Wait for training (~1-2 hours)

# 2. Evaluate on target domain
python eval_dann.py
# Select most recent model
# Choose [2] for both domains

# 3. Compare with baseline (optional)
cd ../simple_detect_car
python train_nn.py  # Train regular CNN on SD2
python eval.py      # Evaluate on Kontext

# DANN should outperform regular CNN on Kontext
```

## 🎓 Key Concepts

### Domain Adaptation
Learning features that work across domains without labeled target data.

### Gradient Reversal Layer (GRL)
- Novel technique from the paper
- Reverses gradients during backpropagation
- Forces feature extractor to maximize domain confusion

### Why Only Use Fakes for Domain Classification?
- Real images are **identical** across domains
- Domain shift exists **only in fake images** (different generators)
- Domain classifier only sees fakes to learn generator-invariant features

## 🐛 Troubleshooting

### Training is Slow
```python
# In train_dann.py config:
'batch_size': 16,              # Reduce from 32
'target_size': (256, 256),     # Reduce from (512, 512)
'sample_size': 1000,           # Use subset for testing
```

### Out of Memory Error
- Reduce `batch_size` (try 8 or 16)
- Reduce `target_size` (try (256, 256) or (224, 224))
- Reduce model sizes: `feature_hidden_size`, `domain_hidden_size`

### Domain Accuracy Stays High (>70%)
- Increase `num_epochs` (try 100)
- Domains might be too different
- Check that real images are truly identical

### Data Not Found Error
Verify your data structure:
```
cardd_data/GenAI_Results/
├── SD2/
│   └── CarDD-TR/
│       ├── metadata/
│       └── processed images...
└── Kontext/
    └── CarDD-TR/
        ├── metadata/
        └── processed images...
```

### ColorJitter Hue Error
If you see `OverflowError: Python integer out of bounds for uint8`:
- This is fixed in the latest version
- The hue parameter in ColorJitter has been adjusted
- Update torchvision: `conda install torchvision -c pytorch`

### Pin Memory Warning on MPS
```
'pin_memory' argument is set as true but not supported on MPS
```
- This is just a warning, can be ignored
- Or set `pin_memory=False` in data loader (slight performance impact)

## 📚 References

**Paper:**
Ganin, Y., & Lempitsky, V. (2015). Unsupervised Domain Adaptation by Backpropagation. ICML 2015.

**Key Contributions:**
1. Gradient Reversal Layer for domain adaptation
2. End-to-end training with standard backpropagation
3. No need for labeled target domain data

## 🔧 Advanced Usage

### Custom Domains

To train on different domains, edit `train_dann.py`:

```python
config = {
    'source_name': 'YourSourceDomain',
    'target_name': 'YourTargetDomain',
    # Data paths will be constructed as:
    # cardd_data/GenAI_Results/{domain_name}/{data_type}/
}
```

### Lambda Schedule

The lambda parameter controls adaptation strength. Current schedule:

```python
lambda_p = 2.0 / (1.0 + exp(-10 * p)) - 1.0
# where p = current_epoch / total_epochs
```

To modify, edit `compute_lambda_schedule()` in `model_dann.py`.

### Model Architecture

To use a different backbone:
1. Edit `model_dann.py`
2. Replace `CNNClassifier.features` with your backbone
3. Adjust `feature_dim` accordingly

## 📄 License

This implementation is for research and educational purposes.

## 👥 Authors

Based on implementation for Responsible AI project at NUS.

---

**Questions?** Check the paper PDF in this directory or refer to the code comments.
