#!/usr/bin/env python3
"""
Script to list and inspect saved models.
"""

import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from utils import list_saved_models, print_model_summary


def main():
    """List all saved models and show their details."""
    print("=" * 60)
    print("SAVED MODELS OVERVIEW")
    print("=" * 60)
    
    models = list_saved_models()
    
    if not models:
        print("No saved models found in the 'models' directory.")
        print("Run training scripts to create models.")
        return
    
    print(f"Found {len(models)} saved models:\n")
    
    for i, model in enumerate(models, 1):
        print(f"{i}. {model['name']} ({model['type']})")
        print(f"   Timestamp: {model['timestamp']}")
        print(f"   Accuracy: {model['accuracy']}")
        print(f"   Path: {model['path']}")
        print()
    
    # Show detailed summary for the most recent model
    if models:
        print("Most recent model details:")
        print("-" * 40)
        print_model_summary(models[0]['path'])


if __name__ == "__main__":
    main()

