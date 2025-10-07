## Car Image Binary Classification (Real vs Processed)

Classify car images as real (original) or processed (fake) using two training paths:
- **Classical ML (scikit-learn)**: Logistic Regression or SVM on flattened images
- **Neural Networks (PyTorch)**: MLP or CNN

### Project layout
- `data_loader.py`: Loads metadata and images, provides binary dataset and DataLoader
- `models.py`: Defines MLP/CNN (PyTorch) and sklearn model factory (logreg, svm)
- `train_sk.py`: Train/evaluate sklearn models, saves to `models/…`
- `train_nn.py`: Train/evaluate PyTorch models, saves to `models/…`
- `model_utils.py`: Save/load utilities and model listing
- `list_models.py`: Inspect saved models

### Requirements
Python >= 3.11. Dependencies are declared in `pyproject.toml`.

Install with uv (recommended):
```bash
uv sync
```

Or with pip:
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

Note (macOS): PyTorch can use Metal (MPS) if available; the script auto-selects `cuda` → `mps` → `cpu`.

### Data configuration
Both training scripts expect the CARDD dataset artifacts:
- Images directory: `/Users/wjs/Library/CloudStorage/OneDrive-Personal/Coding, ML & DL/ResponsibleAI/cardd_data/manipulated_results`
- Metadata JSONs: `/Users/wjs/Library/CloudStorage/OneDrive-Personal/Coding, ML & DL/ResponsibleAI/cardd_data/manipulated_results/metadata`

If your paths differ, edit `data_dir` and `metadata_dir` in `train_sk.py` and `train_nn.py` configs at the top of `main()`.

### Quick start
Train a scikit-learn model (logistic regression or SVM):
```bash
python train_sk.py
```

Train a neural network (MLP or CNN):
```bash
python train_nn.py
```

List and inspect saved models:
```bash
python list_models.py
```

Artifacts are saved under `models/model_YYYYMMDD_HHMMSS/` with `metadata.json` and `results.txt`.

### Tuning
- In `train_sk.py`: set `model_name` to `logreg` or `svm`. For SVM on large data, consider `sample_size=500` to speed up.
- In `train_nn.py`: set `model_name` to `mlp` or `cnn`, adjust `learning_rate`, `num_epochs`, `batch_size`, and `sample_size`.


