#!/usr/bin/env python3
import torch
import torch.nn as nn
from pathlib import Path
import sys
import os

def evaluate_dann_model(model_path=None, model=None,
                       get_src_loaders=None, get_tgt_loaders=None,
                       source_name='sd2', target_name='kontext',
                       batch_size=32, target_size=224, device=None, print_results=True):
    """
    Evaluate DANN model on source and target test data.

    Args:
        model_path: Path to trained model file (if model not provided)
        model: Pre-loaded model instance (if model_path not provided)
        get_src_loaders: Function to get source domain data loaders
        get_tgt_loaders: Function to get target domain data loaders
        source_name: Source domain name for display
        target_name: Target domain name for display
        batch_size: Batch size for evaluation
        target_size: Image size for evaluation
        device: Device to run evaluation on (auto-detected if None)
        print_results: Whether to print results to console
    """
    # Validate inputs
    if model is None:
        raise ValueError("model must be provided")
    if get_src_loaders is None or get_tgt_loaders is None:
        raise ValueError("get_src_loaders and get_tgt_loaders must be provided")

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set up data loaders
    if print_results:
        print("Setting up test data loaders...")

    # Get test loaders with TE split using provided loader functions
    _, src_test_loader = get_src_loaders(batch_size=batch_size, split='TE')
    _, tgt_test_loader = get_tgt_loaders(batch_size=batch_size, split='TE')

    if print_results:
        print("Evaluating model on source and target domains...")

    # Initialize criterion
    criterion = nn.NLLLoss()

    # Evaluate source domain
    if src_test_loader is not None:
        source_loss, source_acc, precision, recall, f1, tp, fp, fn, tn = _evaluate_single_domain(
            model, src_test_loader, criterion, device, domain_type='source'
        )
        src_results = {
            'loss': source_loss, 'accuracy': source_acc, 'samples': len(src_test_loader.dataset),
            'precision': precision, 'recall': recall, 'f1': f1,
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn
        }

    # Evaluate target domain
    if tgt_test_loader is not None:
        target_loss, target_acc, precision, recall, f1, tp, fp, fn, tn = _evaluate_single_domain(
            model, tgt_test_loader, criterion, device, domain_type='target'
        )
        tgt_results = {
            'loss': target_loss, 'accuracy': target_acc, 'samples': len(tgt_test_loader.dataset),
            'precision': precision, 'recall': recall, 'f1': f1,
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn
        }

    # Print results if requested
    if print_results:
        print("\n" + "=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)

        if src_test_loader is not None:
            print(f"Source Domain ({source_name.upper()}):")
            print(f"  Test Loss: {src_results['loss']:.4f}")
            print(f"  Accuracy: {src_results['accuracy']:.2f}%")
            print(f"  Precision: {src_results['precision']:.4f}")
            print(f"  Recall: {src_results['recall']:.4f}")
            print(f"  F1-Score: {src_results['f1']:.4f}")
            print(f"  Confusion Matrix:")
            print(f"    TP: {src_results['tp']}, FP: {src_results['fp']}")
            print(f"    FN: {src_results['fn']}, TN: {src_results['tn']}")
            print(f"  Test Samples: {src_results['samples']}")

        if tgt_test_loader is not None:
            print(f"\nTarget Domain ({target_name.upper()}):")
            print(f"  Test Loss: {tgt_results['loss']:.4f}")
            print(f"  Accuracy: {tgt_results['accuracy']:.2f}%")
            print(f"  Precision: {tgt_results['precision']:.4f}")
            print(f"  Recall: {tgt_results['recall']:.4f}")
            print(f"  F1-Score: {tgt_results['f1']:.4f}")
            print(f"  Confusion Matrix:")
            print(f"    TP: {tgt_results['tp']}, FP: {tgt_results['fp']}")
            print(f"    FN: {tgt_results['fn']}, TN: {tgt_results['tn']}")
            print(f"  Test Samples: {tgt_results['samples']}")

        print("=" * 60)


def _evaluate_single_domain(model, data_loader, criterion, device, domain_type='source'):
    """
    Evaluate model on a single domain.

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

            # Calculate confusion matrix
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

    precision = true_positives / max(1, true_positives + false_positives)
    recall = true_positives / max(1, true_positives + false_negatives)
    f1_score = 2 * (precision * recall) / max(1e-8, precision + recall)

    return avg_loss, accuracy, precision, recall, f1_score, true_positives, false_positives, true_negatives, false_negatives