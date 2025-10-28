#!/usr/bin/env python3
"""
Neuron Activation Analysis for Car Scratch Detection

This script analyzes neuron activations by examining which areas of images
activate specific neurons in the CNN model. It visualizes activation maps
with overlays showing areas with activation above the top 1% percentile.
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
from interpret_cnn.utils import load_model, calc_activated_filters, create_activation_overlay

def get_model_prediction(model, input_tensor):
    """Get model prediction score for an input tensor."""
    model.eval()
    with torch.no_grad():
        output = model(input_tensor.unsqueeze(0))  # Add batch dimension
        score = output.item()  # Model already outputs probabilities (0-1)
    return score

def calculate_activation_threshold(model, dataset_raw, eval_transform, layer_name, num_samples=50):
    """
    Calculate the top 1% percentile threshold for activations across a sample of images.
    
    Args:
        model: Trained CNN model
        dataset_raw: Dataset without transforms (for original images)
        eval_transform: Transform pipeline for preprocessing
        layer_name: Name of the layer to analyze
        num_samples: Number of samples to use for threshold calculation
        
    Returns:
        float: Top 1% percentile threshold
    """
    model.eval()
    
    all_activations = []
    
    print(f"Calculating activation threshold using {num_samples} samples...")
    
    # Sample random indices
    import random
    random.seed(42)
    sample_indices = random.sample(range(len(dataset_raw)), min(num_samples, len(dataset_raw)))
    
    with torch.no_grad():
        for idx, sample_idx in enumerate(sample_indices):
            # Get raw data
            raw_data = dataset_raw[sample_idx]
            original_image = raw_data['original_image']
            processed_image = raw_data['processed_image']
            
            # Process both images for model input
            def preprocess_image(img_array):
                return eval_transform(img_array)
            
            # Get activations for both images
            original_tensor = preprocess_image(original_image)
            processed_tensor = preprocess_image(processed_image)
            
            # Calculate activations for both images
            original_activations = calc_activated_filters(model, original_tensor, layer_name)
            processed_activations = calc_activated_filters(model, processed_tensor, layer_name)
            
            # Convert multi-filter activations to 2D maps by taking the mean across filters
            original_activation_map = original_activations.mean(dim=0)
            processed_activation_map = processed_activations.mean(dim=0)
            
            # Flatten and collect all activation values
            all_activations.extend(original_activation_map.flatten().cpu().numpy())
            all_activations.extend(processed_activation_map.flatten().cpu().numpy())
            
            if (idx + 1) % 10 == 0:
                print(f"Processed {idx + 1}/{len(sample_indices)} samples...")
    
    # Calculate top 1% percentile
    threshold = np.percentile(all_activations, 99)
    print(f"Top 1% activation threshold: {threshold:.6f}")
    
    return threshold

def visualize_neuron_activations(model, dataset_raw, eval_transform, layer_name, 
                                num_samples=5, filter_idx=0, output_dir="neuron_results"):
    """
    Visualize neuron activation maps for the most activated images.
    
    Args:
        model: Trained CNN model
        dataset_raw: Dataset without transforms (for original images)
        eval_transform: Transform pipeline for preprocessing
        layer_name: Name of the layer to analyze
        num_samples: Number of samples to visualize
        filter_idx: Index of the specific filter to analyze (default: 0)
        output_dir: Directory to save results
    """
    
    # Create output directory with layer and filter information
    layer_clean = layer_name.replace('.', '_')  # Replace dots with underscores for folder names
    filter_output_dir = f"{output_dir}/{layer_clean}_filter_{filter_idx}"
    os.makedirs(filter_output_dir, exist_ok=True)
    
    # Calculate activation threshold
    threshold = calculate_activation_threshold(model, dataset_raw, eval_transform, layer_name)
    
    # Find the most activated images
    print(f"Finding the {num_samples} most activated images...")
    print(f"Results will be saved to: {filter_output_dir}")
    activation_scores = []
    
    model.eval()
    with torch.no_grad():
        for idx in range(len(dataset_raw)):
            # Get raw data
            raw_data = dataset_raw[idx]
            original_image = raw_data['original_image']
            processed_image = raw_data['processed_image']
            
            # Process both images for model input
            def preprocess_image(img_array):
                return eval_transform(img_array)
            
            # Get activations for both images
            original_tensor = preprocess_image(original_image)
            processed_tensor = preprocess_image(processed_image)
            
            # Calculate activations for both images
            original_activations = calc_activated_filters(model, original_tensor, layer_name)
            processed_activations = calc_activated_filters(model, processed_tensor, layer_name)
            
            # Extract specific filter activation maps
            original_activation_map = original_activations[filter_idx]  # Shape: (H', W')
            processed_activation_map = processed_activations[filter_idx]  # Shape: (H', W')
            
            # Calculate percentage of pixels with activation > threshold
            original_above_threshold = (original_activation_map > threshold).float().mean().item()
            processed_above_threshold = (processed_activation_map > threshold).float().mean().item()
            
            # Store scores with image indices and types
            activation_scores.append((original_above_threshold, idx, 'original'))
            activation_scores.append((processed_above_threshold, idx, 'processed'))
            
            if (idx + 1) % 50 == 0:
                print(f"Processed {idx + 1}/{len(dataset_raw)} samples...")
    
    # Sort by activation score (descending) and get top samples
    activation_scores.sort(key=lambda x: x[0], reverse=True)
    top_samples = activation_scores[:num_samples]
    
    print(f"Top {num_samples} activation scores:")
    for i, (score, idx, img_type) in enumerate(top_samples):
        print(f"  {i+1}. Score: {score:.6f}, Image: {idx}, Type: {img_type}")
    
    # Get unique sample indices to avoid duplicates
    sample_indices = list(set([idx for _, idx, _ in top_samples]))
    
    # Visualize the top activated samples
    for viz_idx, sample_idx in enumerate(sample_indices):
        # Get raw data (both original and processed images)
        raw_data = dataset_raw[sample_idx]
        
        # Extract images
        original_image = raw_data['original_image']  # numpy array (H, W, C)
        processed_image = raw_data['processed_image']  # numpy array (H, W, C)
        metadata = raw_data['metadata']
        
        print(f"\nVisualizing Sample {viz_idx + 1}/{len(sample_indices)} (Image {sample_idx})")
        print(f"Image ID: {metadata.get('image_id', 'Unknown')}")

        # Process both images for model input
        def preprocess_image(img_array):
            """Convert numpy array to preprocessed tensor."""
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
        
        # Calculate neuron activations for both images
        original_activations = calc_activated_filters(model, original_tensor, layer_name)
        processed_activations = calc_activated_filters(model, processed_tensor, layer_name)
        
        # Extract specific filter activation maps
        original_activation_map = original_activations[filter_idx]  # Shape: (H', W')
        processed_activation_map = processed_activations[filter_idx]  # Shape: (H', W')
        
        # Calculate percentage of pixels with activation > threshold
        original_above_threshold = (original_activation_map > threshold).float().mean().item()
        processed_above_threshold = (processed_activation_map > threshold).float().mean().item()
        
        print(f"Original pixels above threshold: {original_above_threshold:.4f} ({original_above_threshold*100:.2f}%)")
        print(f"Processed pixels above threshold: {processed_above_threshold:.4f} ({processed_above_threshold*100:.2f}%)")
        
        # Create activation overlays using the calculated absolute threshold
        original_overlay = create_activation_overlay(original_image, original_activation_map, 
                                                   threshold=threshold)
        processed_overlay = create_activation_overlay(processed_image, processed_activation_map, 
                                                    threshold=threshold)

        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Most Activated Images - Sample {viz_idx + 1}\nLayer: {layer_name}, Filter: {filter_idx}\nImage {sample_idx}', fontsize=16)
        
        # Original image
        axes[0, 0].imshow(original_image)
        axes[0, 0].set_title(f'Original Image\nScore: {original_score:.4f}\nPixels > threshold: {original_above_threshold*100:.2f}%')
        axes[0, 0].axis('off')
        
        # Original image with activation overlay
        axes[0, 1].imshow(original_overlay)
        axes[0, 1].set_title(f'Original + Activations\nScore: {original_score:.4f}\nPixels > threshold: {original_above_threshold*100:.2f}%')
        axes[0, 1].axis('off')
        
        # Processed image
        axes[1, 0].imshow(processed_image)
        axes[1, 0].set_title(f'Processed Image\nScore: {processed_score:.4f}\nPixels > threshold: {processed_above_threshold*100:.2f}%')
        axes[1, 0].axis('off')
        
        # Processed image with activation overlay
        axes[1, 1].imshow(processed_overlay)
        axes[1, 1].set_title(f'Processed + Activations\nScore: {processed_score:.4f}\nPixels > threshold: {processed_above_threshold*100:.2f}%')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        output_path = f'{filter_output_dir}/neuron_analysis_sample_{viz_idx + 1}.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()  # Close to free memory
        
        print(f"Saved visualization as '{output_path}'")

def main():
    """Main function to run neuron activation analysis."""
    
    # Interactive prompts (similar to gradcam_analyze.py)
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
    
    # Layer selection
    print("\nAvailable layers for analysis:")
    print("Common CNN layer names:")
    print("  - features.0 (first conv layer)")
    print("  - features.2 (second conv layer)")
    print("  - features.4 (third conv layer)")
    print("  - features.6 (fourth conv layer)")
    print("  - features.8 (fifth conv layer)")
    print("  - features.10 (sixth conv layer)")
    print("  - features.12 (seventh conv layer)")
    print("  - features.14 (eighth conv layer)")
    
    layer_name = input("Enter layer name [features.6]: ").strip() or "features.6"
    
    # Filter selection
    filter_idx = int(input("Enter filter index to analyze [0]: ").strip() or "0")
    
    # Data directory selection
    base_data_dir = "/Users/wjs/Library/CloudStorage/OneDrive-Personal/Coding, ML & DL/ResponsibleAI/cardd_data/GenAI_Results"
    data_dir = input(f"Data directory [{base_data_dir}/SD2/CarDD-TE]: ").strip() or f"{base_data_dir}/SD2/CarDD-TE"
    metadata_dir = f"{data_dir}/metadata"
    
    # Output directory
    output_dir = input("Output directory [neuron_results]: ").strip() or "neuron_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Number of samples
    num_samples = int(input("Number of samples to visualize [5]: ").strip() or "5")
    
    print(f"\nLoading model from {model_path}...")
    model, metadata = load_model(str(model_path))
    print(f"Model loaded successfully")
    
    # Extract target size from model metadata
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
    
    # Run neuron activation analysis
    print(f"\nStarting neuron activation analysis for {num_samples} samples...")
    print(f"Analyzing layer: {layer_name}, filter: {filter_idx}")
    visualize_neuron_activations(model, dataset_raw, eval_transform, layer_name, num_samples, filter_idx, output_dir)
    
    # Create clean layer name for final message
    layer_clean = layer_name.replace('.', '_')
    print(f"\nNeuron activation analysis complete! Results saved to '{output_dir}/{layer_clean}_filter_{filter_idx}'")

if __name__ == "__main__":
    main()
