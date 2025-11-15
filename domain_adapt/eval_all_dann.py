#!/usr/bin/env python3
"""
Batch evaluation script for DANN models on their source and target domains.

Evaluates DANN models specified in the MODEL_LIST on their respective
source and target domains using TE (test) split, reporting accuracy, precision,
recall, and F1-score for each model-domain combination.

Usage:
  python eval_all_dann.py
  python eval_all_dann.py --batch_size 64 --output results.json
"""

import json
import argparse
from pathlib import Path
from datetime import datetime
import torch
import torch.nn as nn
from typing import List, Dict, Any, Tuple

# Import DANN evaluation and model classes
from dann.eval_dann import evaluate_dann_model
from model_arch_lib.model_deep import CARDDModel
from cardd import CARDDLoader

# Predefined list of DANN models to evaluate
# Add your DANN model directories here - each should be a path relative to the models_cardd directory
MODEL_LIST = [
    # Add your model directory names here
    # Shallow models
    # "model_dann_20251114_145443",
    # "model_dann_20251114_145521",
    # "model_dann_20251115_014817"

    # Deep models
    "model_dann_20251114_135954",
    "model_dann_20251114_135942",
    "model_dann_20251115_070559",
]


def load_dann_model_from_path(model_path: Path, device) -> Tuple[torch.nn.Module, str, str, int]:
    """
    Load DANN model from path and extract source/target domain info and target_size.

    Args:
        model_path: Path to model directory
        device: Device to load model on

    Returns:
        Tuple of (model, source_domain, target_domain, target_size)
    """
    # Read training summary to get source and target domains and target_size
    summary_file = model_path / "training_summary.json"
    if not summary_file.exists():
        raise FileNotFoundError(f"Training summary not found: {summary_file}")

    with open(summary_file, 'r') as f:
        summary = json.load(f)

    source_domain = summary['config']['source_name']
    target_domain = summary['config']['target_name']
    target_size = summary['config'].get('target_size', 512)  # Default to 512 if not found

    # Load the best target model (the DANN-adapted model)
    model_file = model_path / "best_target_model.pth"
    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found: {model_file}")

    model = CARDDModel()
    model.load_state_dict(torch.load(model_file, map_location=device))
    model.to(device)
    model.eval()

    return model, source_domain, target_domain, target_size


def evaluate_model_on_domains(model_path: Path, batch_size: int = 32, device=None) -> Dict[str, Dict[str, Any]]:
    """
    Evaluate a single DANN model on its source and target domains.

    Args:
        model_path: Path to model directory
        batch_size: Batch size for evaluation
        device: Device to run evaluation on

    Returns:
        Dict mapping domain names to evaluation results
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\nEvaluating {model_path.name}...")

    # Load model and get domain info
    try:
        model, source_domain, target_domain, target_size = load_dann_model_from_path(model_path, device)
        print(f"  Source domain: {source_domain}")
        print(f"  Target domain: {target_domain}")
        print(f"  Target size: {target_size}x{target_size}")
    except Exception as e:
        print(f"  Failed to load model: {e}")
        return {}

    # Create loader functions for source and target domains
    src_loader = CARDDLoader(source_domain)
    tgt_loader = CARDDLoader(target_domain)

    def get_src_loaders(batch_size=batch_size, split='TE'):
        return src_loader.get_loaders(batch_size=batch_size, image_size=target_size, split=split, load_to_memory=False)

    def get_tgt_loaders(batch_size=batch_size, split='TE'):
        return tgt_loader.get_loaders(batch_size=batch_size, image_size=target_size, split=split, load_to_memory=False)

    # Evaluate model using the existing evaluate_dann_model function
    try:
        # We'll capture the results by temporarily redirecting output or parsing the printed output
        # For now, let's modify evaluate_dann_model to return results instead of printing

        # Set up test loaders
        _, src_test_loader = get_src_loaders(batch_size=batch_size, split='TE')
        _, tgt_test_loader = get_tgt_loaders(batch_size=batch_size, split='TE')

        results = {}

        # Evaluate source domain
        criterion = nn.NLLLoss()
        if src_test_loader is not None:
            source_loss, source_acc, precision, recall, f1, tp, fp, fn, tn = _evaluate_single_domain(
                model, src_test_loader, criterion, device, domain_type='source'
            )
            results[source_domain] = {
                'loss': source_loss, 'accuracy': source_acc, 'samples': len(src_test_loader.dataset),
                'precision': precision, 'recall': recall, 'f1': f1,
                'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn
            }

        # Evaluate target domain
        if tgt_test_loader is not None:
            target_loss, target_acc, precision, recall, f1, tp, fp, fn, tn = _evaluate_single_domain(
                model, tgt_test_loader, criterion, device, domain_type='target'
            )
            results[target_domain] = {
                'loss': target_loss, 'accuracy': target_acc, 'samples': len(tgt_test_loader.dataset),
                'precision': precision, 'recall': recall, 'f1': f1,
                'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn
            }

        print(f"  Completed evaluation on {len(results)} domains")
        return results

    except Exception as e:
        print(f"  Failed to evaluate model: {e}")
        return {}


def _evaluate_single_domain(model, data_loader, criterion, device, domain_type='source'):
    """
    Evaluate model on a single domain (copied from eval_dann.py).

    Args:
        model: DANN model
        data_loader: DataLoader for the domain
        criterion: Loss criterion
        device: Device
        domain_type: 'source' or 'target'

    Returns:
        Evaluation metrics
    """
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

            # Forward pass through label classifier only (alpha=0 for evaluation)
            outputs, _ = model(images, alpha=0.0)

            # Get predictions using argmax (same as training code)
            preds = outputs.data.max(1, keepdim=True)[1].squeeze()

            if preds.dim() == 0:
                preds = preds.unsqueeze(0)
            if labels.dim() == 0:
                labels = labels.unsqueeze(0)

            # For loss calculation, use the raw LogSoftmax outputs
            loss = criterion(outputs, labels.long())
            running_loss += loss.item()
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


def print_results_table(all_results: Dict[str, Dict[str, Dict[str, Any]]]):
    """Print results in a formatted table."""
    print("\n" + "="*100)
    print("DANN MODEL EVALUATION RESULTS SUMMARY")
    print("="*100)

    # Header
    print(f"{'Model':<35} | {'Domain':<10} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Samples':<8}")
    print("-" * 110)

    # Results
    for model_name, domain_results in all_results.items():
        for domain, metrics in domain_results.items():
            if metrics is not None:
                print(f"{model_name:<35} | {domain:<10} | {metrics['accuracy']:<10.2f} | {metrics['precision']:<10.4f} | {metrics['recall']:<10.4f} | {metrics['f1']:<10.4f} | {metrics['samples']:<8}")
            else:
                print(f"{model_name:<35} | {domain:<10} | {'N/A':<10} | {'N/A':<10} | {'N/A':<10} | {'N/A':<10} | {'N/A':<8}")

    print("="*100)


def main():
    parser = argparse.ArgumentParser(description='Batch evaluation of DANN models on their source and target domains')
    parser.add_argument('--models_dir', type=str, default='models_cardd',
                       help='Directory containing DANN model folders (default: models_cardd)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for evaluation (default: 32)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output JSON file to save results')

    args = parser.parse_args()

    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Use predefined model list
    models_dir = Path(args.models_dir)
    model_paths = [models_dir / model for model in MODEL_LIST]

    # Validate model paths exist
    valid_models = []
    for model_path in model_paths:
        if model_path.exists() and model_path.is_dir():
            valid_models.append(model_path)
        else:
            print(f"Warning: Model path does not exist: {model_path}")

    if not valid_models:
        raise FileNotFoundError("No valid model paths found. Please check the MODEL_LIST.")

    print(f"Evaluating {len(valid_models)} DANN models")
    print(f"Models: {', '.join([p.name for p in valid_models])}")

    # Evaluate all models
    all_results = {}

    for model_path in valid_models:
        model_name = model_path.name
        results = evaluate_model_on_domains(
            model_path=model_path,
            batch_size=args.batch_size,
            device=device
        )
        all_results[model_name] = results

    # Print summary table
    print_results_table(all_results)

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
