#!/usr/bin/env python3
"""
Grad-CAM Web Application

A Flask web app for real-time Grad-CAM visualization of car scratch detection.
"""

import sys
import os
from pathlib import Path
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import json
import base64
import io
import random
from flask import Flask, render_template, request, jsonify

# Add parent directory to path (now two levels up)
sys.path.append(str(Path(__file__).parent.parent.parent))

from simple_detect_car.data_loader import CarScratchDataset, get_eval_transforms
from simple_detect_car.models import get_model
from interpret_cnn.utils import load_model, calc_gradcam, create_activation_overlay

app = Flask(__name__)

# Global variables for model and dataset
model = None
dataset_raw = None
eval_transform = None
metadata = None

def get_model_prediction(model, input_tensor):
    """Get model prediction score for an input tensor."""
    model.eval()
    with torch.no_grad():
        output = model(input_tensor.unsqueeze(0))
        score = output.item()
    return score

def preprocess_image(img_array):
    """Convert numpy array to preprocessed tensor."""
    return eval_transform(img_array)

def create_gradcam_visualization(sample_idx):
    """Create Grad-CAM visualization for a given sample."""
    global model, dataset_raw, eval_transform
    
    # Get raw data
    raw_data = dataset_raw[sample_idx]
    original_image = raw_data['original_image']
    processed_image = raw_data['processed_image']
    metadata = raw_data['metadata']
    
    # Preprocess both images
    original_tensor = preprocess_image(original_image)
    processed_tensor = preprocess_image(processed_image)
    
    # Get model predictions
    original_score = get_model_prediction(model, original_tensor)
    processed_score = get_model_prediction(model, processed_tensor)
    
    # Resize images for consistent visualization
    target_size = eval_transform.transforms[1].size
    original_image = cv2.resize(original_image, target_size)
    processed_image = cv2.resize(processed_image, target_size)
    
    # Calculate Grad-CAM
    original_gradcam = calc_gradcam(model, original_tensor, target_class=1)
    processed_gradcam = calc_gradcam(model, processed_tensor, target_class=1)
    
    # Create overlays
    original_overlay = create_activation_overlay(original_image, original_gradcam, percent_threshold=10)
    processed_overlay = create_activation_overlay(processed_image, processed_gradcam, percent_threshold=10)
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Grad-CAM Analysis - Sample {sample_idx + 1}', fontsize=14)
    
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
    
    # Convert plot to base64 string
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
    img_buffer.seek(0)
    img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
    plt.close()
    
    return {
        'image': img_base64,
        'original_score': original_score,
        'processed_score': processed_score,
        'image_id': metadata.get('image_id', 'Unknown'),
        'sample_idx': sample_idx + 1
    }

@app.route('/')
def index():
    """Main page."""
    return render_template('index.html')

@app.route('/load_model', methods=['POST'])
def load_model_endpoint():
    """Load model and dataset."""
    global model, dataset_raw, eval_transform, metadata
    
    try:
        data = request.json
        model_path = data.get('model_path', '../../simple_detect_car/models/model_20251006_172724/cnn_best.pth')
        data_dir = data.get('data_dir', '/Users/wjs/Library/CloudStorage/OneDrive-Personal/Coding, ML & DL/ResponsibleAI/cardd_data/GenAI_Results/SD2/CarDD-TE')
        metadata_dir = f"{data_dir}/metadata"
        
        # Load model
        model, metadata = load_model(model_path)
        
        # Extract target size from metadata
        cfg = metadata.get('config', {}) if metadata else {}
        meta_target_size = tuple(cfg.get('target_size')) if cfg.get('target_size') else (224, 224)
        chosen_target_size = meta_target_size
        
        # Load dataset
        dataset_raw = CarScratchDataset(
            data_dir=data_dir,
            metadata_dir=metadata_dir,
            sample_size=None,
            transform=None,
            load_processed=True,
            load_masks=False
        )
        
        # Get evaluation transforms
        eval_transform = get_eval_transforms(target_size=chosen_target_size)
        
        return jsonify({
            'success': True,
            'message': f'Model loaded successfully. Dataset has {len(dataset_raw)} samples.',
            'target_size': chosen_target_size,
            'num_samples': len(dataset_raw)
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/get_sample', methods=['POST'])
def get_sample():
    """Get a random sample for visualization."""
    global model, dataset_raw, eval_transform
    
    if model is None or dataset_raw is None:
        return jsonify({'success': False, 'error': 'Model not loaded'})
    
    try:
        data = request.json
        sample_idx = data.get('sample_idx')
        
        if sample_idx is None:
            # Random sample
            sample_idx = random.randint(0, len(dataset_raw) - 1)
        
        result = create_gradcam_visualization(sample_idx)
        result['success'] = True
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)