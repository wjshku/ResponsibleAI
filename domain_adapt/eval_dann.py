#!/usr/bin/env python3
"""
Evaluation script for Domain Adversarial Neural Network (DANN).

This script evaluates a trained DANN model on:
- Source domain (SD2)
- Target domain (Kontext)

It provides detailed metrics on:
- Real vs Fake classification accuracy
- Domain classification accuracy (should be ~50% if adaptation worked)
- Confusion matrices
"""

import sys
import json
from pathlib import Path
import torch
import torch.nn as nn
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent / "simple_detect_car"))

from data_loader import (
    CarScratchDataset,
    create_dataloader,
    get_eval_transforms,
)
from model_dann import get_dann_model

# Import the DualDomainDataset from train_dann
sys.path.append(str(Path(__file__).parent))
from train_dann import DualDomainDataset


def evaluate_dann_detailed(model, data_loader, device, domain_name="test"):
    """
    Evaluate DANN model with detailed metrics.

    Args:
        model: DomainAdversarialNN model
        data_loader: DataLoader for evaluation
        device: Device (CPU/CUDA/MPS)
        domain_name: Name for logging

    Returns:
        Dictionary with detailed evaluation metrics
    """
    model.eval()
    criterion_label = nn.BCELoss()

    # Confusion matrix for label classification
    tp_label = 0  # True Positive (correctly identified fake)
    tn_label = 0  # True Negative (correctly identified real)
    fp_label = 0  # False Positive (real classified as fake)
    fn_label = 0  # False Negative (fake classified as real)

    # Domain classification metrics (on fakes only)
    domain_correct = 0
    domain_total = 0
    domain_preds = []
    domain_labels = []

    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        pbar = tqdm(data_loader, desc=f"Evaluating {domain_name}", leave=True)
        for images, labels, domains in pbar:
            images = images.to(device)
            labels = labels.to(device)
            domains = domains.to(device)

            # Derive is_fake from labels
            is_fake = (labels == 1)

            # Forward pass (alpha=0 for inference)
            label_pred, domain_pred = model(images, alpha=0.0)

            # Loss
            loss = criterion_label(label_pred.squeeze(), labels.float())
            total_loss += loss.item()
            num_batches += 1

            # Label predictions (real=0, fake=1)
            predicted_labels = (label_pred.squeeze() > 0.5).float()

            # Update confusion matrix
            for pred, true in zip(predicted_labels, labels):
                if true == 0:  # Real image
                    if pred == 0:
                        tn_label += 1
                    else:
                        fp_label += 1
                else:  # Fake image
                    if pred == 1:
                        tp_label += 1
                    else:
                        fn_label += 1

            # Domain classification (only on fakes)
            fake_mask = is_fake.bool()
            if fake_mask.sum() > 0:
                predicted_domains = (domain_pred[fake_mask].squeeze() > 0.5).float()
                true_domains = domains[fake_mask].float()

                domain_correct += (predicted_domains == true_domains).sum().item()
                domain_total += fake_mask.sum().item()

                # Store for later analysis
                domain_preds.extend(predicted_domains.cpu().numpy().tolist())
                domain_labels.extend(true_domains.cpu().numpy().tolist())

    # Compute metrics
    avg_loss = total_loss / max(1, num_batches)

    # Label classification metrics
    total_samples = tp_label + tn_label + fp_label + fn_label
    accuracy = 100.0 * (tp_label + tn_label) / max(1, total_samples)

    precision = tp_label / max(1, tp_label + fp_label)
    recall = tp_label / max(1, tp_label + fn_label)
    f1_score = 2 * (precision * recall) / max(1e-8, precision + recall)

    specificity = tn_label / max(1, tn_label + fp_label)

    # Real and fake accuracies
    real_accuracy = 100.0 * tn_label / max(1, tn_label + fp_label)
    fake_accuracy = 100.0 * tp_label / max(1, tp_label + fn_label)

    # Domain classification metrics
    domain_accuracy = 100.0 * domain_correct / max(1, domain_total)

    return {
        'loss': avg_loss,
        'overall_accuracy': accuracy,
        'real_accuracy': real_accuracy,
        'fake_accuracy': fake_accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'specificity': specificity,
        'confusion_matrix': {
            'tp': tp_label,
            'tn': tn_label,
            'fp': fp_label,
            'fn': fn_label
        },
        'domain_accuracy': domain_accuracy,
        'domain_total': domain_total,
        'total_samples': total_samples
    }


def print_metrics(metrics, domain_name):
    """Pretty print evaluation metrics."""
    print(f"\n{'='*70}")
    print(f"{domain_name.upper()} DOMAIN EVALUATION")
    print(f"{'='*70}")

    print(f"\nLabel Classification (Real vs Fake):")
    print(f"  Overall Accuracy: {metrics['overall_accuracy']:.2f}%")
    print(f"  Real Accuracy:    {metrics['real_accuracy']:.2f}%")
    print(f"  Fake Accuracy:    {metrics['fake_accuracy']:.2f}%")
    print(f"  Precision:        {metrics['precision']:.4f}")
    print(f"  Recall:           {metrics['recall']:.4f}")
    print(f"  F1 Score:         {metrics['f1_score']:.4f}")
    print(f"  Specificity:      {metrics['specificity']:.4f}")

    print(f"\nConfusion Matrix:")
    cm = metrics['confusion_matrix']
    print(f"                Predicted")
    print(f"               Real  Fake")
    print(f"  Actual Real  {cm['tn']:4d}  {cm['fp']:4d}")
    print(f"         Fake  {cm['fn']:4d}  {cm['tp']:4d}")

    print(f"\nDomain Classification (SD2 vs Kontext):")
    print(f"  Domain Accuracy: {metrics['domain_accuracy']:.2f}%")
    print(f"  (On {metrics['domain_total']} fake images)")

    if metrics['domain_accuracy'] > 60:
        print(f"  ⚠️  High domain accuracy suggests poor adaptation")
    elif 40 <= metrics['domain_accuracy'] <= 60:
        print(f"  ✓ Domain accuracy near 50% indicates good adaptation")
    else:
        print(f"  ⚠️  Very low domain accuracy is unexpected")

    print(f"\nTotal samples: {metrics['total_samples']}")


def main():
    print("=" * 70)
    print("DOMAIN ADVERSARIAL NEURAL NETWORK (DANN) EVALUATION")
    print("=" * 70)

    # ============================================
    # Interactive model selection
    # ============================================
    models_base = Path("models")
    if not models_base.exists():
        print(f"Error: Models directory not found: {models_base}")
        return

    # List available DANN models
    dann_models = sorted([d for d in models_base.iterdir()
                          if d.is_dir() and d.name.startswith("model_dann_")],
                         key=lambda x: x.stat().st_mtime, reverse=True)

    if not dann_models:
        print("No DANN models found in models/ directory")
        return

    print("\nAvailable DANN models:")
    for i, model_dir in enumerate(dann_models[:10]):  # Show last 10
        metadata_file = model_dir / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            acc = metadata.get('best_target_accuracy', 0)
            epoch = metadata.get('best_epoch', 0)
            print(f"  [{i}] {model_dir.name} (Best: {acc:.2f}% at epoch {epoch})")
        else:
            print(f"  [{i}] {model_dir.name}")

    # Select model
    try:
        selection = input(f"\nSelect model [0-{len(dann_models)-1}] or press Enter for [0]: ").strip()
        idx = int(selection) if selection else 0
        model_dir = dann_models[idx]
    except (ValueError, IndexError):
        print("Invalid selection")
        return

    print(f"\nSelected: {model_dir.name}")

    # ============================================
    # Load metadata and configuration
    # ============================================
    metadata_file = model_dir / "metadata.json"
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        config = metadata.get('config', {})
    else:
        print("Warning: No metadata found, using default config")
        config = {}

    # ============================================
    # Configuration
    # ============================================
    eval_config = {
        'model_dir': model_dir,
        'batch_size': config.get('batch_size', 32),
        'target_size': tuple(config.get('target_size', (512, 512))),
        'input_channels': config.get('input_channels', 3),
        'num_classes': config.get('num_classes', 1),
        'feature_hidden_size': config.get('feature_hidden_size', 256),
        'domain_hidden_size': config.get('domain_hidden_size', 256),
        'dropout': config.get('dropout', 0.3),
    }

    # Data paths
    base_path = "/Users/wjs/Library/CloudStorage/OneDrive-Personal/Coding, ML & DL/ResponsibleAI/cardd_data/GenAI_Results"

    # Ask which domain to evaluate
    print("\nEvaluate on:")
    print("  [0] Source domain (SD2)")
    print("  [1] Target domain (Kontext)")
    print("  [2] Both domains")

    eval_choice = input("Select [0/1/2] or press Enter for [2]: ").strip()
    eval_choice = int(eval_choice) if eval_choice else 2

    # Ask which model to load
    print("\nChoose model to evaluate:")
    best_model_path = model_dir / "dann_best.pth"
    final_model_path = model_dir / "dann_final.pth"

    available_models = []
    if best_model_path.exists():
        available_models.append(("best", best_model_path))
    if final_model_path.exists():
        available_models.append(("final", final_model_path))

    if not available_models:
        print("Error: No model weights found")
        return

    print("Available models:")
    for i, (model_type, path) in enumerate(available_models):
        if model_type == "best":
            print(f"  [{i}] Best model (saved at best validation accuracy)")
        else:
            print(f"  [{i}] Final model (saved at end of training)")

    model_choice = input(f"Select model [0-{len(available_models)-1}] or press Enter for [0]: ").strip()
    model_choice = int(model_choice) if model_choice else 0

    if model_choice < 0 or model_choice >= len(available_models):
        print("Invalid model selection")
        return

    selected_model_type, selected_model_path = available_models[model_choice]

    # ============================================
    # Load model
    # ============================================
    print("\n" + "=" * 70)
    print("LOADING MODEL")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available()
                         else 'mps' if torch.backends.mps.is_available()
                         else 'cpu')
    print(f"Using device: {device}")

    model = get_dann_model(
        input_channels=eval_config['input_channels'],
        num_classes=eval_config['num_classes'],
        feature_hidden_size=eval_config['feature_hidden_size'],
        domain_hidden_size=eval_config['domain_hidden_size'],
        dropout=eval_config['dropout']
    ).to(device)

    # Load selected model weights
    model.load_state_dict(torch.load(selected_model_path, map_location=device))
    print(f"Loaded {selected_model_type} model from: {selected_model_path.name}")

    # ============================================
    # Prepare data loaders
    # ============================================
    print("\n" + "=" * 70)
    print("LOADING DATA")
    print("=" * 70)

    eval_transform = get_eval_transforms(target_size=eval_config['target_size'])

    data_loaders = {}

    if eval_choice in [0, 2]:  # Source domain
        source_data_dir = f"{base_path}/SD2/CarDD-TE"  # Use test split
        source_metadata_dir = f"{source_data_dir}/metadata"

        print(f"\nLoading source domain (SD2)...")
        source_dataset = DualDomainDataset(
            data_dir=source_data_dir,
            metadata_dir=source_metadata_dir,
            domain_id=0,
            transform=eval_transform,
            sample_size=None
        )
        print(f"  Source dataset size: {len(source_dataset)}")

        data_loaders['source'] = create_dataloader(
            source_dataset,
            batch_size=eval_config['batch_size'],
            shuffle=False,
            target_size=eval_config['target_size']
        )

    if eval_choice in [1, 2]:  # Target domain
        target_data_dir = f"{base_path}/Kontext/CarDD-TE"  # Use test split
        target_metadata_dir = f"{target_data_dir}/metadata"

        print(f"\nLoading target domain (Kontext)...")
        target_dataset = DualDomainDataset(
            data_dir=target_data_dir,
            metadata_dir=target_metadata_dir,
            domain_id=1,
            transform=eval_transform,
            sample_size=None
        )
        print(f"  Target dataset size: {len(target_dataset)}")

        data_loaders['target'] = create_dataloader(
            target_dataset,
            batch_size=eval_config['batch_size'],
            shuffle=False,
            target_size=eval_config['target_size']
        )

    # ============================================
    # Evaluate
    # ============================================
    print("\n" + "=" * 70)
    print("EVALUATING")
    print("=" * 70)

    results = {}

    if 'source' in data_loaders:
        print("\nEvaluating on source domain...")
        source_metrics = evaluate_dann_detailed(
            model, data_loaders['source'], device, domain_name="source"
        )
        results['source'] = source_metrics
        print_metrics(source_metrics, "source")

    if 'target' in data_loaders:
        print("\nEvaluating on target domain...")
        target_metrics = evaluate_dann_detailed(
            model, data_loaders['target'], device, domain_name="target"
        )
        results['target'] = target_metrics
        print_metrics(target_metrics, "target")

    # ============================================
    # Summary comparison
    # ============================================
    if len(results) == 2:
        print("\n" + "=" * 70)
        print("DOMAIN COMPARISON")
        print("=" * 70)

        source = results['source']
        target = results['target']

        print(f"\n{'Metric':<25} {'Source':>12} {'Target':>12} {'Difference':>12}")
        print("-" * 70)
        print(f"{'Overall Accuracy':<25} {source['overall_accuracy']:>11.2f}% {target['overall_accuracy']:>11.2f}% {target['overall_accuracy']-source['overall_accuracy']:>11.2f}%")
        print(f"{'Real Accuracy':<25} {source['real_accuracy']:>11.2f}% {target['real_accuracy']:>11.2f}% {target['real_accuracy']-source['real_accuracy']:>11.2f}%")
        print(f"{'Fake Accuracy':<25} {source['fake_accuracy']:>11.2f}% {target['fake_accuracy']:>11.2f}% {target['fake_accuracy']-source['fake_accuracy']:>11.2f}%")
        print(f"{'Precision':<25} {source['precision']:>12.4f} {target['precision']:>12.4f} {target['precision']-source['precision']:>12.4f}")
        print(f"{'Recall':<25} {source['recall']:>12.4f} {target['recall']:>12.4f} {target['recall']-source['recall']:>12.4f}")
        print(f"{'F1 Score':<25} {source['f1_score']:>12.4f} {target['f1_score']:>12.4f} {target['f1_score']-source['f1_score']:>12.4f}")
        print(f"{'Domain Accuracy':<25} {source['domain_accuracy']:>11.2f}% {target['domain_accuracy']:>11.2f}% {target['domain_accuracy']-source['domain_accuracy']:>11.2f}%")

    # ============================================
    # Save results
    # ============================================
    results_file = model_dir / "evaluation_results.json"
    with open(results_file, 'w') as f:
        # Convert numpy types to native Python types for JSON serialization
        json_results = {}
        for domain, metrics in results.items():
            json_results[domain] = {k: float(v) if isinstance(v, (float, int)) else v
                                   for k, v in metrics.items()}
        json.dump(json_results, f, indent=2)
    print(f"\nResults saved to: {results_file}")

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
