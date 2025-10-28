#!/usr/bin/env python3
"""
Grad-CAM Analysis for Car Scratch Detection

This script loads images using the data loader, applies proper preprocessing,
and generates Grad-CAM overlays for both original and processed images.
"""

import sys
import os
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from simple_detect_car.data_loader import CarScratchDataset, get_eval_transforms
from simple_detect_car.models import get_model
from interpret_cnn.utils import load_model, calc_gradcam, create_activation_overlay

def get_model_prediction(model, input_tensor):
    """Get model prediction score for an input tensor."""
    model.eval()
    with torch.no_grad():
        output = model(input_tensor.unsqueeze(0))  # Add batch dimension
        score = output.item()  # Model already outputs probabilities (0-1)
    return score

def calculate_metrics(model, dataset_raw, eval_transform, output_dir):
    """
    Calculate precision, recall, and confusion matrix for the dataset.
    
    Args:
        model: Trained CNN model
        dataset_raw: Dataset without transforms (for original images)
        eval_transform: Transform pipeline for preprocessing
        output_dir: Directory to save results
        
    Returns:
        dict: Metrics including precision, recall, f1_score, confusion_matrix
    """
    model.eval()
    
    # Initialize counters
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0
    
    all_scores = []
    all_labels = []
    
    print("Calculating metrics for all samples...")
    
    with torch.no_grad():
        for idx in range(len(dataset_raw)):
            # Get raw data
            raw_data = dataset_raw[idx]
            original_image = raw_data['original_image']
            processed_image = raw_data['processed_image']
            
            # Process both images for model input
            def preprocess_image(img_array):
                return eval_transform(img_array)
            
            # Get predictions
            original_tensor = preprocess_image(original_image)
            processed_tensor = preprocess_image(processed_image)
            
            original_score = model(original_tensor.unsqueeze(0)).item()
            processed_score = model(processed_tensor.unsqueeze(0)).item()
            
            # Store scores and labels
            all_scores.extend([original_score, processed_score])
            all_labels.extend([0, 1])  # 0 for original, 1 for processed
            
            # Calculate confusion matrix components
            original_pred = 1 if original_score > 0.5 else 0
            processed_pred = 1 if processed_score > 0.5 else 0
            
            # Original image (label 0)
            if original_pred == 0:
                true_negatives += 1
            else:
                false_positives += 1
            
            # Processed image (label 1)
            if processed_pred == 1:
                true_positives += 1
            else:
                false_negatives += 1
    
    # Calculate metrics
    precision = true_positives / max(1, true_positives + false_positives)
    recall = true_positives / max(1, true_positives + false_negatives)
    f1_score = 2 * (precision * recall) / max(1e-8, precision + recall)
    accuracy = (true_positives + true_negatives) / max(1, true_positives + false_positives + true_negatives + false_negatives)
    
    metrics = {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'accuracy': accuracy,
        'confusion_matrix': {
            'tp': true_positives,
            'fp': false_positives,
            'tn': true_negatives,
            'fn': false_negatives
        }
    }
    
    # Save metrics to file
    import json
    with open(f"{output_dir}/metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    
    return metrics

def visualize_gradcam_comparison(model, dataset_raw, eval_transform, num_samples=5, output_dir="gradcam_results"):
    """
    Visualize Grad-CAM overlays for original and processed images.
    
    Args:
        model: Trained CNN model
        dataset_raw: Dataset without transforms (for original images)
        eval_transform: Transform pipeline for preprocessing
        num_samples: Number of samples to visualize
        output_dir: Directory to save results
    """
    
    # Sample random indices
    import random
    random.seed(42)
    sample_indices = random.sample(range(len(dataset_raw)), min(num_samples, len(dataset_raw)))
    
    for idx, sample_idx in enumerate(sample_indices):
        # Get raw data (both original and processed images)
        raw_data = dataset_raw[sample_idx]
        
        # Extract images
        original_image = raw_data['original_image']  # numpy array (H, W, C)
        processed_image = raw_data['processed_image']  # numpy array (H, W, C)
        metadata = raw_data['metadata']
        
        print(f"\nSample {idx + 1}/{num_samples}")
        print(f"Image ID: {metadata.get('image_id', 'Unknown')}")

        # Process both images for model input
        def preprocess_image(img_array):
            """Convert numpy array to preprocessed tensor."""
            # The eval_transform expects numpy array, not PIL image
            return eval_transform(img_array)  # Returns CHW tensor, normalized
        
        # Preprocess both images
        original_tensor = preprocess_image(original_image)
        processed_tensor = preprocess_image(processed_image)
        
        # Get model predictions
        original_score = get_model_prediction(model, original_tensor)
        processed_score = get_model_prediction(model, processed_tensor)
        
        print(f"Original image score: {original_score:.4f}")
        print(f"Processed image score: {processed_score:.4f}")
        
        # Resize both images to target size for consistent visualization
        target_size = eval_transform.transforms[1].size  # Get size from Resize transform
        original_image = cv2.resize(original_image, target_size)
        processed_image = cv2.resize(processed_image, target_size)
        
        # Calculate Grad-CAM for both images
        original_gradcam = calc_gradcam(model, original_tensor, target_class=1)
        processed_gradcam = calc_gradcam(model, processed_tensor, target_class=1)
        
        # Create overlays
        original_overlay = create_activation_overlay(original_image, original_gradcam, percent_threshold=10)
        processed_overlay = create_activation_overlay(processed_image, processed_gradcam, percent_threshold=10)
        

        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Grad-CAM Analysis - Sample {idx + 1}', fontsize=16)
        
        # Original image
        axes[0, 0].imshow(original_image)
        axes[0, 0].set_title(f'Original Image\nScore: {original_score:.4f}')
        axes[0, 0].axis('off')
        
        # Original image with Grad-CAM overlay
        axes[0, 1].imshow(original_overlay)
        axes[0, 1].set_title(f'Original + Grad-CAM\nScore: {original_score:.4f}')
        axes[0, 1].axis('off')
        
        # Processed image
        axes[1, 0].imshow(processed_image)
        axes[1, 0].set_title(f'Processed Image\nScore: {processed_score:.4f}')
        axes[1, 0].axis('off')
        
        # Processed image with Grad-CAM overlay
        axes[1, 1].imshow(processed_overlay)
        axes[1, 1].set_title(f'Processed + Grad-CAM\nScore: {processed_score:.4f}')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        output_path = f'{output_dir}/gradcam_analysis_sample_{idx + 1}.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()  # Close to free memory
        
        print(f"Saved visualization as '{output_path}'")

def main():
    """Main function to run Grad-CAM analysis."""
    
    # Interactive prompts (similar to eval.py)
    models_base = input("Models base directory [../simple_detect_car/models]: ").strip() or "../simple_detect_car/models"
    
    # List available models
    from pathlib import Path
    base = Path(models_base)
    if not base.exists():
        raise FileNotFoundError(f"Models directory not found: {base}")
    
    model_dirs = [d for d in base.iterdir() if d.is_dir() and d.name.startswith("model_")]
    model_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    
    if not model_dirs:
        raise FileNotFoundError(f"No model directories found in {base}")
    
    print("Available models:")
    for i, d in enumerate(model_dirs[:10]):  # Show last 10
        print(f"  [{i}] {d.name}")
    
    model_idx = int(input(f"Select model [0-{min(9, len(model_dirs)-1)}]: ").strip() or "0")
    model_dir = model_dirs[model_idx]
    
    # Choose model file
    model_files = list(model_dir.glob("*.pth"))
    if not model_files:
        raise FileNotFoundError(f"No .pth files found in {model_dir}")
    
    print("Available model files:")
    for i, f in enumerate(model_files):
        print(f"  [{i}] {f.name}")
    
    file_idx = int(input(f"Select model file [0-{len(model_files)-1}]: ").strip() or "0")
    model_path = model_files[file_idx]
    
    # Data directory selection
    base_data_dir = "/Users/wjs/Library/CloudStorage/OneDrive-Personal/Coding, ML & DL/ResponsibleAI/cardd_data/GenAI_Results"
    data_dir = input(f"Data directory [{base_data_dir}/SD2/CarDD-TE]: ").strip() or f"{base_data_dir}/SD2/CarDD-TE"
    metadata_dir = f"{data_dir}/metadata"
    
    # Output directory
    output_dir = input("Output directory [gradcam_results]: ").strip() or "gradcam_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Number of samples
    num_samples = int(input("Number of samples to visualize [5]: ").strip() or "5")
    
    # Calculate metrics option
    calc_metrics = input("Calculate metrics for all samples? [y/N]: ").strip().lower() in ('y', 'yes')
    
    print(f"\nLoading model from {model_path}...")
    model, metadata = load_model(str(model_path))
    print(f"Model loaded successfully")
    
    # Extract target size from model metadata (same as eval.py)
    cfg = metadata.get('config', {}) if metadata else {}
    meta_target_size = tuple(cfg.get('target_size')) if cfg.get('target_size') else (224, 224)
    chosen_target_size = meta_target_size
    print(f"Using target size from model metadata: {chosen_target_size}")
    
    print("Loading dataset...")
    # Load dataset without transforms to get original images
    dataset_raw = CarScratchDataset(
        data_dir=data_dir,
        metadata_dir=metadata_dir,
        sample_size=None,
        transform=None,  # No transforms - get raw images
        load_processed=True,  # Load both original and processed
        load_masks=False
    )
    print(f"Dataset loaded: {len(dataset_raw)} samples")
    
    # Get evaluation transforms with correct target size
    eval_transform = get_eval_transforms(target_size=chosen_target_size)
    
    # Calculate metrics for all samples (if requested)
    if calc_metrics:
        print("\nCalculating metrics...")
        metrics = calculate_metrics(model, dataset_raw, eval_transform, output_dir)
        
        print(f"\nMetrics:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1-Score: {metrics['f1_score']:.4f}")
        print(f"  Confusion Matrix:")
        print(f"    TP: {metrics['confusion_matrix']['tp']}, FP: {metrics['confusion_matrix']['fp']}")
        print(f"    FN: {metrics['confusion_matrix']['fn']}, TN: {metrics['confusion_matrix']['tn']}")
    else:
        print("\nSkipping metrics calculation...")
    
    # Run Grad-CAM analysis
    print(f"\nStarting Grad-CAM analysis for {num_samples} samples...")
    visualize_gradcam_comparison(model, dataset_raw, eval_transform, num_samples, output_dir)
    
    print(f"\nGrad-CAM analysis complete! Results saved to '{output_dir}'")

if __name__ == "__main__":
    main()