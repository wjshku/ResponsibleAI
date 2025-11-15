#!/usr/bin/env python3
"""
Batch evaluation script for multiple models on car scratch dataset.

Evaluates a predefined list of models on TE (test) sets for SD2, Kontext, and Qwen domains,
reporting accuracy, precision, recall, and F1-score for each model-domain combination.

Usage:
  python eval_all.py
  python eval_all.py --batch_size 64 --output results.json
  python eval_all.py --domains sd2 kontext --fallback_target_size 256
"""

import json
import argparse
from pathlib import Path
from datetime import datetime
import torch
import torch.nn as nn
from typing import List, Dict, Any

from data_loader import CarDDDataset, create_dataloader, get_eval_transforms
from models import get_model

# Predefined list of models to evaluate
# Add your model directories here - each should be a path relative to the project root
MODEL_LIST = [ f"models/{model}" for model in [
    # Shallow models
    # "model_20251114_032000",
    # "model_20251114_035120",
    # "model_20251114_040527",
    # "model_20251114_042913",

    # Deep models
    "model_20251114_065644",
    "model_20251114_072440",
    "model_20251114_071329",
    "model_20251114_074935",
    "model_20251114_090859"
]]


def evaluate_model(model, data_loader, device):
    """Evaluate model on a single dataset and return comprehensive metrics."""
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

    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1_score,
        'tp': true_positives,
        'fp': false_positives,
        'fn': false_negatives,
        'tn': true_negatives,
        'samples': total
    }


def load_model_from_path(model_path: Path, device):
    """Load model from path, trying different formats and architectures."""
    model_dir = model_path if model_path.is_dir() else model_path.parent
    metadata_file = model_dir / "metadata.json"

    # Try to load metadata for model configuration
    metadata = None
    if metadata_file.exists():
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
        except Exception:
            metadata = None

    # Determine model architecture
    detected_model = None
    if metadata and isinstance(metadata.get('config', {}).get('model_name', None), str):
        detected_model = metadata['config']['model_name']

    # Default to 'cnn' if not detected
    model_name = detected_model if detected_model in ('vanilla', 'cnn') else 'cnn'

    # Get target size from metadata
    target_size = (224, 224)  # Default
    if metadata and 'config' in metadata and 'target_size' in metadata['config']:
        try:
            ts = metadata['config']['target_size']
            if isinstance(ts, list) and len(ts) == 2:
                target_size = tuple(ts)
        except Exception:
            pass  # Use default if parsing fails

    # Create model
    model = get_model(model_name=model_name)

    # Try different weight file locations
    weight_files = []

    # Check for best model file from metadata
    if metadata and metadata.get('best_model_file'):
        candidate = model_dir / metadata['best_model_file']
        if candidate.exists():
            weight_files.append(candidate)

    # Check for standard model files
    for filename in [f"{model_name}_best.pth", f"{model_name}_model.pth", "model.pth"]:
        candidate = model_dir / filename
        if candidate.exists():
            weight_files.append(candidate)

    # Check for checkpoint files
    for filename in [f"{model_name}_best_checkpoint.pth", f"{model_name}_checkpoint.pth", "checkpoint.pth"]:
        candidate = model_dir / filename
        if candidate.exists():
            weight_files.append(candidate)

    # Try to load from any available weight file
    loaded = False
    for weight_file in weight_files:
        try:
            if weight_file.name.endswith('_checkpoint.pth'):
                # Load checkpoint
                ckpt = torch.load(weight_file, map_location=device)
                model.load_state_dict(ckpt['model_state_dict'])
                print(f"    Loaded checkpoint: {weight_file.name}")
            else:
                # Load state dict
                model.load_state_dict(torch.load(weight_file, map_location=device))
                print(f"    Loaded weights: {weight_file.name}")
            loaded = True
            break
        except Exception as e:
            print(f"    Failed to load {weight_file.name}: {e}")
            continue

    if not loaded:
        raise FileNotFoundError(f"Could not load any model weights from {model_dir}")

    model.to(device)
    return model, model_name, target_size


def evaluate_model_on_domains(model_path: Path, domains: List[str], batch_size: int = 32,
                             target_size: int = 224, device=None) -> Dict[str, Dict[str, Any]]:
    """
    Evaluate a single model on multiple domains.

    Args:
        model_path: Path to model directory or weight file
        domains: List of domains to evaluate on ('sd2', 'kontext', 'qwen')
        batch_size: Batch size for evaluation
        target_size: Default image size for evaluation (may be overridden by model metadata)
        device: Device to run evaluation on

    Returns:
        Dict mapping domain names to evaluation results
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\nEvaluating {model_path.name} on {', '.join(domains)} domains...")

    # Load model and get its target size from metadata
    try:
        model, model_arch, model_target_size = load_model_from_path(model_path, device)
        print(f"  Model architecture: {model_arch}")
        print(f"  Model target size: {model_target_size}")
    except Exception as e:
        print(f"  Failed to load model: {e}")
        return {}

    # Use model-specific target size, fall back to provided default
    if model_target_size and isinstance(model_target_size, tuple) and len(model_target_size) == 2:
        eval_target_size = model_target_size
    else:
        eval_target_size = (target_size, target_size)

    # Set up evaluation transform
    eval_transform = get_eval_transforms(target_size=eval_target_size)

    results = {}

    for domain in domains:
        print(f"  Evaluating on {domain.upper()} domain...")

        try:
            # Create dataset for this domain
            dataset = CarDDDataset(
                domain=domain,
                split='TE',  # Always use test set
                transform=eval_transform,
                sample_size=None,
                load_to_memory=False
            )

            # Create data loader
            loader = create_dataloader(dataset, batch_size=batch_size, shuffle=False)

            # Evaluate
            metrics = evaluate_model(model, loader, device)

            results[domain] = metrics
            print(f"    Completed: {metrics['accuracy']:.2f}% accuracy on {metrics['samples']} samples")

        except Exception as e:
            print(f"    Failed to evaluate on {domain}: {e}")
            results[domain] = None

    return results


def print_results_table(all_results: Dict[str, Dict[str, Dict[str, Any]]], domains: List[str]):
    """Print results in a formatted table."""
    print("\n" + "="*80)
    print("EVALUATION RESULTS SUMMARY")
    print("="*80)

    # Header
    header = "|20"
    print(f"{'Model':<25} | {'Domain':<10} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
    print("-" * 85)

    # Results
    for model_name, domain_results in all_results.items():
        for domain in domains:
            if domain in domain_results and domain_results[domain] is not None:
                metrics = domain_results[domain]
                print(f"{model_name:<25} | {domain:<10} | {metrics['accuracy']:<10.2f} | {metrics['precision']:<10.4f} | {metrics['recall']:<10.4f} | {metrics['f1']:<10.4f}")
            else:
                print(f"{model_name:<25} | {domain:<10} | {'N/A':<10} | {'N/A':<10} | {'N/A':<10} | {'N/A':<10}")

    print("="*80)


def main():
    parser = argparse.ArgumentParser(description='Batch evaluation of models on car scratch dataset')
    parser.add_argument('--domains', nargs='+', default=['sd2', 'kontext', 'qwen'],
                       choices=['sd2', 'kontext', 'qwen'],
                       help='Domains to evaluate on (default: all)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for evaluation (default: 32)')
    parser.add_argument('--fallback_target_size', type=int, default=224,
                       help='Fallback target image size if not found in model metadata (default: 224)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output JSON file to save results')

    args = parser.parse_args()

    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Use predefined model list
    model_paths = [Path(model) for model in MODEL_LIST]

    # Validate model paths exist
    valid_models = []
    for model_path in model_paths:
        if model_path.exists() and (model_path.is_dir() or model_path.is_file()):
            valid_models.append(model_path)
        else:
            print(f"Warning: Model path does not exist: {model_path}")

    if not valid_models:
        raise FileNotFoundError("No valid model paths found")

    # Evaluate all models
    all_results = {}
    domains = args.domains

    print(f"Evaluating {len(valid_models)} models on {len(domains)} domains: {', '.join(domains)}")
    print(f"Models: {', '.join([p.name for p in valid_models])}")

    for model_path in valid_models:
        model_name = model_path.name
        results = evaluate_model_on_domains(
            model_path=model_path,
            domains=domains,
            batch_size=args.batch_size,
            target_size=args.fallback_target_size,
            device=device
        )
        all_results[model_name] = results

    # Print summary table
    print_results_table(all_results, domains)

    # Save results to JSON if requested
    if args.output:
        with open(args.output, 'w') as f:
            # Convert Path objects to strings for JSON serialization
            json_results = {}
            for model_name, domain_results in all_results.items():
                json_results[model_name] = {}
                for domain, metrics in domain_results.items():
                    if metrics is not None:
                        json_results[model_name][domain] = metrics
                    else:
                        json_results[model_name][domain] = None

            json.dump(json_results, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
