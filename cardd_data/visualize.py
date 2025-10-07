#!/usr/bin/env python3
"""
Visualization utilities for car damage manipulation results

This module provides functions to visualize the results of image manipulation,
showing original images, masks, and inpainted results with their corresponding prompts.

Author: AI Assistant
Date: 2024
"""

import os
import json
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
from typing import List, Dict, Any, Optional
from PIL import Image


def load_metadata_from_folder(folder_path: str) -> List[Dict[str, Any]]:
    """
    Load all metadata files from a results folder
    
    Args:
        folder_path: Path to the results folder containing metadata subfolder
        
    Returns:
        List of metadata dictionaries sorted by timestamp
    """
    metadata_dir = os.path.join(folder_path, "metadata")
    if not os.path.exists(metadata_dir):
        print(f"❌ Metadata directory not found: {metadata_dir}")
        return []
    
    metadata_files = glob.glob(os.path.join(metadata_dir, "*.json"))
    metadata_list = []
    
    for file_path in metadata_files:
        try:
            with open(file_path, 'r') as f:
                metadata = json.load(f)
                metadata_list.append(metadata)
        except Exception as e:
            print(f"⚠ Failed to load metadata from {file_path}: {e}")
    
    # Sort by timestamp
    metadata_list.sort(key=lambda x: x.get('timestamp', ''))
    return metadata_list


def filter_successful_inpaints(metadata_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Filter metadata to only include successful inpainted images
    
    Args:
        metadata_list: List of metadata dictionaries
        
    Returns:
        Filtered list containing only successful inpaints
    """
    filtered_list = []
    
    for metadata in metadata_list:
        # Check if it's a successful inpaint
        success = metadata.get('success', False)
        processed_path = metadata.get('processed_image_path', '')
        
        # Must be successful AND have 'inpainted' in the filename
        if success and processed_path and 'inpainted' in os.path.basename(processed_path):
            filtered_list.append(metadata)
    
    print(f"📊 Filtered {len(filtered_list)} successful inpaints from {len(metadata_list)} total metadata files")
    return filtered_list


def find_corresponding_files(metadata: Dict[str, Any], results_folder: str) -> Dict[str, str]:
    """
    Find the corresponding image files for a metadata entry
    
    Args:
        metadata: Metadata dictionary
        results_folder: Path to the results folder
        
    Returns:
        Dictionary with paths to original image, mask, and inpainted result
    """
    files = {}
    
    # Find original image and mask from metadata
    files['original'] = metadata.get('original_image_path', '')
    files['mask'] = metadata.get('mask_path', '')
    files['inpainted'] = metadata.get('processed_image_path', '')
    
    # If processed_image_path is relative, make it absolute
    if files['inpainted'] and not os.path.isabs(files['inpainted']):
        files['inpainted'] = os.path.join(results_folder, os.path.basename(files['inpainted']))
    
    return files


def load_and_resize_image(image_path: str, target_size: tuple = (256, 256)) -> Optional[np.ndarray]:
    """
    Load and resize an image
    
    Args:
        image_path: Path to the image file
        target_size: Target size for resizing (width, height)
        
    Returns:
        Resized image as numpy array, or None if loading failed
    """
    if not image_path or not os.path.exists(image_path):
        return None
    
    try:
        # Load image
        if image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            image = cv2.imread(image_path)
            if image is not None:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                # Try with PIL as fallback
                pil_image = Image.open(image_path)
                image = np.array(pil_image)
                if len(image.shape) == 3 and image.shape[2] == 4:  # RGBA
                    image = image[:, :, :3]  # Remove alpha channel
        else:
            return None
        
        # Resize image
        image = cv2.resize(image, target_size)
        return image
    except Exception as e:
        print(f"⚠ Failed to load image {image_path}: {e}")
        return None


def create_placeholder_image(size: tuple = (256, 256), text: str = "Not Found") -> np.ndarray:
    """
    Create a placeholder image when the actual image is not found
    
    Args:
        size: Size of the placeholder image
        text: Text to display on the placeholder
        
    Returns:
        Placeholder image as numpy array
    """
    placeholder = np.ones((size[1], size[0], 3), dtype=np.uint8) * 128  # Gray background
    # Add text (simplified - just a solid color with different intensity)
    placeholder[50:100, 50:200] = [255, 0, 0]  # Red rectangle as placeholder
    return placeholder


def visualize_results_folder(folder_path: str, 
                           max_images: int = 5, 
                           figsize: tuple = (15, 10),
                           save_path: Optional[str] = None,
                           random_seed: int = None,
                           filter_successful: bool = True) -> None:
    """
    Visualize results from a manipulation results folder
    
    Args:
        folder_path: Path to the results folder
        max_images: Maximum number of images to display
        figsize: Figure size for matplotlib
        save_path: Optional path to save the visualization
        random_seed: Random seed for reproducible random sampling
        filter_successful: Whether to filter only successful inpaints
    """
    print(f"🖼️  Visualizing results from: {folder_path}")
    
    # Load metadata
    metadata_list = load_metadata_from_folder(folder_path)
    if not metadata_list:
        print("❌ No metadata found in the folder")
        return
    
    # Filter for successful inpaints if requested
    if filter_successful:
        metadata_list = filter_successful_inpaints(metadata_list)
        if not metadata_list:
            print("❌ No successful inpainted images found")
            return
    
    # Set random seed for reproducible sampling
    if random_seed is not None:
        random.seed(random_seed)
    
    # Randomly sample images if we have more than max_images
    total_available = len(metadata_list)
    if total_available > max_images:
        metadata_list = random.sample(metadata_list, max_images)
        print(f"📊 Randomly selected {len(metadata_list)} out of {total_available} available results")
    else:
        print(f"📊 Found {len(metadata_list)} results to visualize")
    
    # Create figure
    fig, axes = plt.subplots(len(metadata_list), 3, figsize=figsize)
    if len(metadata_list) == 1:
        axes = axes.reshape(1, -1)
    
    for i, metadata in enumerate(metadata_list):
        # Get file paths
        files = find_corresponding_files(metadata, folder_path)
        
        # Get prompt
        prompt = "No prompt found"
        try:
            tool_params = metadata.get('tool_parameters', {})
            prompt = tool_params.get('prompt', 'No prompt found')
            # Don't truncate - show full prompt in console
        except:
            pass
        
        # Load images
        original_img = load_and_resize_image(files['original'])
        mask_img = load_and_resize_image(files['mask'])
        inpainted_img = load_and_resize_image(files['inpainted'])
        
        # Create placeholders for missing images
        if original_img is None:
            original_img = create_placeholder_image(text="Original")
        if mask_img is None:
            mask_img = create_placeholder_image(text="Mask")
        if inpainted_img is None:
            inpainted_img = create_placeholder_image(text="Inpainted")
        
        # Display images
        axes[i, 0].imshow(original_img)
        axes[i, 0].set_title(f"Original\n{os.path.basename(files['original'])}", 
                            fontsize=10, fontweight='bold', pad=10)
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(mask_img, cmap='gray')
        # Put prompt as the title of the mask column
        prompt_text = f"Prompt: {prompt}"
        axes[i, 1].set_title(prompt_text, fontsize=9, fontweight='bold', pad=10, wrap=True)
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(inpainted_img)
        axes[i, 2].set_title(f"Inpainted\n{os.path.basename(files['inpainted'])}", 
                            fontsize=10, fontweight='bold', pad=10)
        axes[i, 2].axis('off')
        
        # Print prompt to console as well
        print(f"Row {i+1}: {prompt}")
    
    # Set simple overall title
    fig.suptitle("Car Damage Manipulation Results", fontsize=16, fontweight='bold', y=0.95)
    
    # Adjust layout with better spacing to accommodate prompt text
    plt.tight_layout()
    plt.subplots_adjust(top=0.90, hspace=0.4, wspace=0.1, bottom=0.1)
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Visualization saved to: {save_path}")
    
    # Show the plot
    plt.show()


def visualize_single_result(metadata: Dict[str, Any], 
                          results_folder: str,
                          figsize: tuple = (12, 4)) -> None:
    """
    Visualize a single result with detailed information
    
    Args:
        metadata: Metadata dictionary for the result
        results_folder: Path to the results folder
        figsize: Figure size for matplotlib
    """
    print(f"🖼️  Visualizing single result...")
    
    # Check if this is a successful inpaint
    success = metadata.get('success', False)
    processed_path = metadata.get('processed_image_path', '')
    
    if not success or not processed_path or 'inpainted' not in os.path.basename(processed_path):
        print("❌ This is not a successful inpainted image - skipping visualization")
        return
    
    # Get file paths
    files = find_corresponding_files(metadata, results_folder)
    
    # Get detailed information
    prompt = metadata.get('tool_parameters', {}).get('prompt', 'No prompt found')
    processing_time = metadata.get('processing_time', 0)
    success = metadata.get('success', False)
    model_name = metadata.get('model_name', 'Unknown')
    model_version = metadata.get('model_version', 'Unknown')
    
    # Load images
    original_img = load_and_resize_image(files['original'])
    mask_img = load_and_resize_image(files['mask'])
    inpainted_img = load_and_resize_image(files['inpainted'])
    
    # Create placeholders for missing images
    if original_img is None:
        original_img = create_placeholder_image(text="Original")
    if mask_img is None:
        mask_img = create_placeholder_image(text="Mask")
    if inpainted_img is None:
        inpainted_img = create_placeholder_image(text="Inpainted")
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Display images
    axes[0].imshow(original_img)
    axes[0].set_title("Original Image", fontsize=12, fontweight='bold', pad=15)
    axes[0].axis('off')
    
    axes[1].imshow(mask_img, cmap='gray')
    axes[1].set_title("Mask", fontsize=12, fontweight='bold', pad=15)
    axes[1].axis('off')
    
    axes[2].imshow(inpainted_img)
    axes[2].set_title("Inpainted Result", fontsize=12, fontweight='bold', pad=15)
    axes[2].axis('off')
    
    # Add prompt text below the inpainted image
    prompt_text = f"Prompt: {prompt}"
    axes[2].text(0.5, -0.15, prompt_text, transform=axes[2].transAxes, 
                 ha='center', va='top', fontsize=10, wrap=True,
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8))
    
    # Print detailed information to console
    print(f"Prompt: {prompt}")
    print(f"Processing Time: {processing_time:.2f}s | Success: {'✅' if success else '❌'} | Model: {model_name} v{model_version}")
    print(f"Files: {os.path.basename(files['original'])} | {os.path.basename(files['mask'])} | {os.path.basename(files['inpainted'])}")
    
    # Add simple info text
    info_text = f"Processing Time: {processing_time:.2f}s | Success: {'✅' if success else '❌'} | Model: {model_name} v{model_version}"
    
    fig.text(0.5, 0.02, info_text, ha='center', fontsize=10, 
             bbox=dict(boxstyle="round,pad=0.4", facecolor="lightgray", alpha=0.8))
    
    # Set overall title
    fig.suptitle("Car Damage Manipulation Result", fontsize=16, fontweight='bold', y=0.95)
    
    # Adjust layout with better spacing to accommodate prompt text
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15, top=0.88, wspace=0.1)
    
    # Show the plot
    plt.show()


def main():
    """Test the visualization functions"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python visualize.py <results_folder> [max_images] [save_path] [random_seed] [filter_successful]")
        print("  results_folder: Path to the results folder (e.g., ./test_results_full)")
        print("  max_images: Maximum number of images to display (default: 5)")
        print("  save_path: Optional path to save the visualization")
        print("  random_seed: Random seed for reproducible sampling (default: None)")
        print("  filter_successful: Whether to filter only successful inpaints (default: True)")
        return
    
    folder_path = sys.argv[1]
    max_images = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    save_path = sys.argv[3] if len(sys.argv) > 3 else None
    random_seed = int(sys.argv[4]) if len(sys.argv) > 4 else None
    filter_successful = sys.argv[5].lower() == 'true' if len(sys.argv) > 5 else True
    
    if not os.path.exists(folder_path):
        print(f"❌ Folder not found: {folder_path}")
        return
    
    # Visualize results
    visualize_results_folder(folder_path, max_images=max_images, save_path=save_path, random_seed=random_seed, filter_successful=filter_successful)


if __name__ == "__main__":
    main()
