#!/usr/bin/env python3
"""
Interactive CLI to evaluate a saved model on the car scratch dataset and
plot loss curves from metadata. You'll be prompted to:
  1) Choose a model directory from a recent list under the models folder
  2) Confirm or select model architecture (auto-detected when possible)
  3) Choose whether to evaluate the best or final weights

Run:
  python eval.py
"""

import json
from pathlib import Path
from datetime import datetime
import torch
import torch.nn as nn

from data_loader import CarDDDataset, create_dataloader, get_eval_transforms
from models import get_model
from utils import plot_losses

# Optional rich CLI support
try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.completion import Completer, Completion
    PROMPT_AVAILABLE = True
except Exception:
    PROMPT_AVAILABLE = False


def evaluate_model(model, data_loader, device):
    model.eval()
    criterion = nn.BCELoss()
    running_loss = 0.0
    correct = 0
    total = 0
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    true_negatives = 0
    
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images).squeeze()
            if outputs.dim() == 0:
                outputs = outputs.unsqueeze(0)
            if labels.dim() == 0:
                labels = labels.unsqueeze(0)
            loss = criterion(outputs, labels.float())
            running_loss += loss.item()
            preds = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (preds == labels).sum().item()
            
            # Calculate confusion matrix components
            tp = ((preds == 1) & (labels == 1)).sum().item()
            fp = ((preds == 1) & (labels == 0)).sum().item()
            fn = ((preds == 0) & (labels == 1)).sum().item()
            tn = ((preds == 0) & (labels == 0)).sum().item()
            
            true_positives += tp
            false_positives += fp
            false_negatives += fn
            true_negatives += tn
    
    avg_loss = running_loss / max(1, len(data_loader))
    accuracy = 100.0 * correct / max(1, total)
    
    # Calculate precision and recall
    precision = true_positives / max(1, true_positives + false_positives)
    recall = true_positives / max(1, true_positives + false_negatives)
    f1_score = 2 * (precision * recall) / max(1e-8, precision + recall)
    
    return avg_loss, accuracy, precision, recall, f1_score, true_positives, false_positives, true_negatives, false_negatives


def main():
    # Simple interactive prompts (press Enter to accept defaults)
    models_base = input("Models base directory [models]: ").strip() or "models"
    try:
        max_list = int(input("How many recent runs to list? [30]: ").strip() or "30")
    except Exception:
        max_list = 30
    # Choose test domains
    print("\nSelect test domains:")
    print("  [1] SD2 only")
    print("  [2] Kontext only")
    print("  [3] SD2 + Kontext (combined)")
    test_domain_choice = input("Enter choice [1-3] (default: 1): ").strip() or "1"
    
    if test_domain_choice == "1":
        test_domains = ["SD2"]
    elif test_domain_choice == "2":
        test_domains = ["Kontext"]
    elif test_domain_choice == "3":
        test_domains = ["SD2", "Kontext"]
    else:
        print("Invalid choice, defaulting to SD2 only")
        test_domains = ["SD2"]
    
    print(f"\nTesting on: {', '.join(test_domains)}")
    
    # Choose test split
    print("\nSelect test split:")
    print("  [1] CarDD-VAL")
    print("  [2] CarDD-TE")
    test_split_choice = input("Enter choice [1-2] (default: 2): ").strip() or "2"
    test_data_type = "CarDD-VAL" if test_split_choice == "1" else "CarDD-TE"
    
    # Set up paths
    script_dir = Path(__file__).parent
    genai_root = script_dir.parent / "cardd_data" / "GenAI_Results"
    try:
        batch_size = int(input("Batch size [32]: ").strip() or "32")
    except Exception:
        batch_size = 32

    def _list_model_dirs(base_dir: Path):
        if not base_dir.exists():
            return []
        dirs = [d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith("model_")]
        dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return dirs

    def _describe_model_dir(d: Path):
        meta = d / "metadata.json"
        if meta.exists():
            try:
                with open(meta, 'r') as f:
                    m = json.load(f)
                ts = m.get('timestamp', '')
                acc = m.get('results', {}).get('best_test_acc', None)
                acc_str = f", best_acc={acc:.2f}%" if isinstance(acc, (int, float)) else ""
                return f"{d.name} (ts={ts}{acc_str})"
            except Exception:
                pass
        try:
            mt = datetime.fromtimestamp(d.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')
        except Exception:
            mt = "?"
        return f"{d.name} (modified {mt})"

    # Always interactively select model directory
    base = Path(models_base)
    candidates = _list_model_dirs(base)
    if not candidates:
        raise FileNotFoundError(f"No model directories found under: {base}")
    to_show = candidates[: max(1, max_list)]
    print("Select a model directory:")
    for i, d in enumerate(to_show):
        print(f"  [{i}] {_describe_model_dir(d)}")

    model_dir = None
    # If prompt_toolkit is available, offer name-based selection with completion
    if PROMPT_AVAILABLE:
        class DirCompleter(Completer):
            def get_completions(self, document, complete_event):
                text = document.text.lower()
                for d in to_show:
                    name = d.name
                    if not text or text in name.lower():
                        yield Completion(name, start_position=-len(document.text))
        try:
            session = PromptSession()
            typed = session.prompt("Type model dir name (or press Enter to choose by index): ", completer=DirCompleter())
            typed = typed.strip()
            if typed:
                matches = [d for d in to_show if d.name == typed]
                if matches:
                    model_dir = matches[0]
        except Exception:
            pass

    if model_dir is None:
        sel = input(f"Enter index [0-{len(to_show)-1}]: ").strip()
        try:
            idx = int(sel)
        except Exception:
            raise ValueError("Invalid selection. Expected an integer index.")
        if not (0 <= idx < len(to_show)):
            raise ValueError("Selection index out of range.")
        model_dir = to_show[idx]

    # Choose device
    dev_in = input("Device [auto|cpu|cuda|mps]: ").strip().lower()
    if dev_in in ("cpu", "cuda", "mps"):
        device = torch.device(dev_in)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Read metadata for defaults
    metadata = None
    metadata_file = model_dir / "metadata.json"
    if metadata_file.exists():
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
        except Exception:
            metadata = None

    # Determine model_name (from metadata if possible)
    detected_model = None
    if metadata and isinstance(metadata.get('config', {}).get('model_name', None), str):
        detected_model = metadata['config']['model_name']
    prompt = f"Detected model: {detected_model!r}. Press Enter to accept or type 'vanilla'/'cnn': " if detected_model else "Type model architecture ('vanilla' or 'cnn'): "
    choice = input(prompt).strip().lower()
    model_name = detected_model if (choice == '' and detected_model in ('vanilla', 'cnn')) else (choice if choice in ('vanilla', 'cnn') else None)
    if model_name is None:
        raise ValueError("Invalid model selection. Please choose 'vanilla' or 'cnn'.")

    # Ask whether to use best weights
    use_best_input = input("Use best weights if available? [y/N]: ").strip().lower()
    use_best = use_best_input in ("y", "yes")

    # Pull default hyperparameters from metadata when available
    cfg = metadata.get('config', {}) if metadata else {}
    meta_target_size = tuple(cfg.get('target_size')) if cfg.get('target_size') else (224, 224)

    # Choose target size (default to training metadata)
    ts_in = input(f"Target size H W [{meta_target_size[0]} {meta_target_size[1]}]: ").strip()
    if ts_in:
        try:
            parts = [int(p) for p in ts_in.split()]
            chosen_target_size = (parts[0], parts[1])
        except Exception:
            chosen_target_size = meta_target_size
    else:
        chosen_target_size = meta_target_size

    # Build eval datasets/loader from selected domains
    eval_transform = get_eval_transforms(target_size=chosen_target_size)
    test_datasets = []
    
    # Map test_data_type to split parameter
    split_map = {
        'CarDD-VAL': 'VAL',
        'CarDD-TE': 'TE'
    }
    split_param = split_map.get(test_data_type, 'VAL')
    
    for domain in test_domains:
        domain_lower = domain.lower()
        print(f"  Loading {domain} {test_data_type}...")
        try:
            ds = CarDDDataset(
                domain=domain_lower,
                split=split_param,
                transform=eval_transform,
                sample_size=None,
                load_to_memory=False  # Don't load to memory for evaluation
            )
            test_datasets.append(ds)
            print(f"    Loaded {len(ds)} samples from {domain}")
        except RuntimeError as e:
            print(f"Warning: Could not load {domain} {test_data_type}: {e}")
            continue
    
    if not test_datasets:
        print("Error: No test datasets loaded!")
        return
    
    # Combine datasets if multiple domains
    if len(test_datasets) > 1:
        print(f"\nCombining {len(test_datasets)} test datasets...")
        # Use torch.utils.data.ConcatDataset to combine datasets
        from torch.utils.data import ConcatDataset
        dataset = ConcatDataset(test_datasets)
        print(f"Combined test dataset size: {len(dataset)}")
    else:
        dataset = test_datasets[0]
    
    loader = create_dataloader(dataset, batch_size=batch_size, shuffle=False)

    # Reconstruct model
    if model_name == "cnn":
        # cnn depends on classifier hidden_size and dropout
        model = get_model(model_name="cnn",)
        state_path = None
        if use_best:
            if metadata and metadata.get('best_model_file'):
                candidate = model_dir / metadata['best_model_file']
                if candidate.exists():
                    state_path = candidate
            if state_path is None:
                candidate = model_dir / "cnn_best.pth"
                if candidate.exists():
                    state_path = candidate
        if state_path is None:
            state_path = model_dir / "cnn_model.pth"

    if not state_path.exists():
        # try checkpoint (prefer best checkpoint if requested)
        ckpt_path = None
        if use_best:
            if metadata and metadata.get('best_checkpoint_file'):
                cand = model_dir / metadata['best_checkpoint_file']
                if cand.exists():
                    ckpt_path = cand
            if ckpt_path is None:
                cand = model_dir / f"{model_name}_best_checkpoint.pth"
                if cand.exists():
                    ckpt_path = cand
        if ckpt_path is None:
            ckpt_path = model_dir / f"{model_name}_checkpoint.pth"
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt['model_state_dict'])
            print(f"Loaded checkpoint weights: {ckpt_path.name}")
        else:
            # try generic .pth present
            pths = list(model_dir.glob("*.pth"))
            if not pths:
                raise FileNotFoundError(f"No model weights found in {model_dir}")
            print(f"Loaded fallback weights: {pths[0].name}")
            model.load_state_dict(torch.load(pths[0], map_location=device))
    else:
        model.load_state_dict(torch.load(state_path, map_location=device))
        print(f"Loaded state_dict weights: {state_path.name}")

    model.to(device)

    # Evaluate
    loss, acc, precision, recall, f1, tp, fp, tn, fn = evaluate_model(model, loader, device)
    print(f"Evaluation complete:")
    print(f"  Loss: {loss:.4f}")
    print(f"  Accuracy: {acc:.2f}%")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    print(f"  Confusion Matrix:")
    print(f"    TP: {tp}, FP: {fp}")
    print(f"    FN: {fn}, TN: {tn}")


if __name__ == "__main__":
    main()


