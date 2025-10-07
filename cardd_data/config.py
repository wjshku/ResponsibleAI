"""
Configuration file for Car Damage Inpainting

This file contains configuration settings and API credentials
for the car damage inpainting system.
"""

import os
from typing import Dict, Any

# Dataset paths
CARDD_ROOT = "/Users/wjs/Local Storage/CarDD_release"
COCO_PATH = os.path.join(CARDD_ROOT, "CarDD_COCO")
SOD_PATH = os.path.join(CARDD_ROOT, "CarDD_SOD")

# API Configuration
# Replace with your actual API key
STABILITY_API_KEY = os.getenv("STABILITY_API_KEY", "your_stability_api_key_here")

# Default settings
DEFAULT_SETTINGS = {
    "use_local": False,
    "num_samples": 4,
    "output_dir": "./inpainted_results",
    "prompt": "add car damage, minor scratch marks, light surface damage, subtle wear",
    "image_size": (1024, 1024),
    "quality": "high"
}

# Model configurations
STABILITY_CONFIG = {
    "model_name": "stable-diffusion-xl-1024-v1-0",
    "max_resolution": 1024,
    "cfg_scale": 7,
    "steps": 20
}

LOCAL_CONFIG = {
    "model_name": "local_fallback",
    "description": "Returns original image without processing"
}

def get_stability_api_key() -> str:
    """Get Stability AI API key"""
    return STABILITY_API_KEY
    
def get_stability_config() -> Dict[str, Any]:
    """Get Stability AI model configuration"""
    return STABILITY_CONFIG

def get_local_config() -> Dict[str, Any]:
    """Get local API configuration"""
    return LOCAL_CONFIG

# Validation functions
def validate_stability_api_key() -> bool:
    """Validate if Stability AI API key is available"""
    api_key = get_stability_api_key()
    return api_key and api_key != "your_stability_api_key_here"

def validate_dataset_paths() -> Dict[str, bool]:
    """Validate if dataset paths exist"""
    return {
        "coco": os.path.exists(COCO_PATH),
        "sod": os.path.exists(SOD_PATH)
    }

def print_config_status():
    """Print current configuration status"""
    print("=== Inpainting Configuration Status ===")
    
    # Check API keys
    print("\nAPI Keys:")
    stability_status = "✓" if validate_stability_api_key() else "❌"
    print(f"  stability: {stability_status}")
    print(f"  local: ✓ (always available)")
    
    # Check dataset paths
    print("\nDataset Paths:")
    dataset_status = validate_dataset_paths()
    for dataset, exists in dataset_status.items():
        status = "✓" if exists else "❌"
        print(f"  {dataset}: {status}")
    
    # Print settings
    print(f"\nDefault Settings:")
    for key, value in DEFAULT_SETTINGS.items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    print_config_status()
