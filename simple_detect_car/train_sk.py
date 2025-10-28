#!/usr/bin/env python3
"""
Training script for scikit-learn models (logistic regression, SVM).
"""

import os
import sys
import time
import numpy as np
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from data_loader import CarScratchDataset, create_dataloader, create_train_test_split
from models import get_sklearn_model
from utils import save_sklearn_model

def main():
    """Main training function for sklearn models."""
    print("=" * 60)
    print("CAR IMAGE BINARY CLASSIFICATION - SKLEARN MODELS")
    print("=" * 60)
    
    # Set up paths (relative to this script's location)
    script_dir = Path(__file__).parent
    data_base = script_dir.parent / "cardd_data" / "manipulated_results"

    # Configuration
    config = {
        'data_dir': data_base,
        'metadata_dir': data_base / "metadata",
        'sample_size': None,  # Use None for full dataset, or set to 500 for faster SVM training
        'shuffle': True,
        'target_size': (224, 224),
        'batch_size': 32,
        'test_size': 0.2,
        'model_name': 'svm',  # 'logreg' or 'svm'
        'random_state': 42
    }
    
    # For SVM with large datasets, suggest using a sample for faster training
    if config['model_name'] == 'svm' and config['sample_size'] is None:
        print("Note: SVM training on full dataset may take 10+ minutes.")
        print("Consider setting 'sample_size': 500 in config for faster training during development.")
        print("Continuing with full dataset...")
    
    print(f"Configuration: {config}")
    
    # Check if data directory exists
    if not config['data_dir'].exists():
        print(f"Error: Data directory {config['data_dir']} does not exist!")
        return
    
    # Create binary classification dataset
    print("\n1. Creating binary classification dataset...")
    binary_dataset = CarScratchDataset.load_binary_dataset(
        data_dir=config['data_dir'],
        metadata_dir=config['metadata_dir'],
        sample_size=config['sample_size'],
        shuffle=config['shuffle']
    )
    print(f"Loaded {len(binary_dataset)} samples")
    
    # Create train/test split
    print("\n2. Creating train/test split...")
    train_indices, test_indices = create_train_test_split(
        binary_dataset, 
        test_size=config['test_size'],
        random_state=config['random_state']
    )
    
    # Create train and test datasets
    train_binary_dataset = CarScratchDataset.load_binary_dataset(
        data_dir=config['data_dir'],
        metadata_dir=config['metadata_dir'],
        sample_size=config['sample_size'],
        shuffle=config['shuffle']
    )
    test_binary_dataset = CarScratchDataset.load_binary_dataset(
        data_dir=config['data_dir'],
        metadata_dir=config['metadata_dir'],
        sample_size=config['sample_size'],
        shuffle=config['shuffle']
    )
    
    # Override valid_entries with split indices
    train_binary_dataset.valid_entries = [binary_dataset.valid_entries[i] for i in train_indices]
    test_binary_dataset.valid_entries = [binary_dataset.valid_entries[i] for i in test_indices]
    
    # Re-initialize shuffled_indices for both datasets
    train_binary_dataset.shuffled_indices = list(range(len(train_binary_dataset.valid_entries)))
    test_binary_dataset.shuffled_indices = list(range(len(test_binary_dataset.valid_entries)))
    
    # Create data loaders
    print("\n3. Creating data loaders...")
    train_loader = create_dataloader(
        train_binary_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        target_size=config['target_size']
    )
    test_loader = create_dataloader(
        test_binary_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        target_size=config['target_size']
    )
    
    # Create sklearn model
    print(f"\n4. Creating {config['model_name']} model...")
    model = get_sklearn_model(config['model_name'])
    print("Using scikit-learn model")
    
    # Collect training data
    print("\n5. Collecting training data...")
    X_train, y_train = [], []
    for images, labels in train_loader:
        # images: (B, H, W, C)
        images = images.float().numpy() / 255.0
        images = images.reshape(images.shape[0], -1)
        X_train.append(images)
        y_train.append(labels.numpy())
    X_train = np.concatenate(X_train, axis=0)
    y_train = np.concatenate(y_train, axis=0)
    
    print(f"Training data shape: {X_train.shape}, Labels shape: {y_train.shape}")
    print(f"Training class distribution: {np.bincount(y_train)} (0=original, 1=processed)")
    
    # Collect test data
    print("\n6. Collecting test data...")
    X_test, y_test = [], []
    for images, labels in test_loader:
        images = images.float().numpy() / 255.0
        images = images.reshape(images.shape[0], -1)
        X_test.append(images)
        y_test.append(labels.numpy())
    X_test = np.concatenate(X_test, axis=0)
    y_test = np.concatenate(y_test, axis=0)
    
    print(f"Test data shape: {X_test.shape}, Labels shape: {y_test.shape}")
    print(f"Test class distribution: {np.bincount(y_test)} (0=original, 1=processed)")
    
    # Show training complexity info
    print(f"\nTraining complexity:")
    print(f"  - Training samples: {X_train.shape[0]:,}")
    print(f"  - Features per sample: {X_train.shape[1]:,}")
    print(f"  - Total features: {X_train.shape[0] * X_train.shape[1]:,}")
    if config['model_name'] == 'svm':
        print(f"  - SVM complexity: O(n²) - may take several minutes for large datasets")
    else:
        print(f"  - Logistic regression: Usually fast")
    
    # Train model
    print(f"\n7. Training {config['model_name']} model...")
    print("=" * 60)
    
    start_time = time.time()
    
    # Add progress bar for SVM training
    if config['model_name'] == 'svm':
        print("SVM training can take a while... This may take several minutes.")
        print("Training progress:")
        
        # Create a custom callback for progress tracking
        class ProgressCallback:
            def __init__(self, total_iterations):
                self.total_iterations = total_iterations
                self.current_iteration = 0
                self.start_time = time.time()
            
            def __call__(self, iteration):
                self.current_iteration = iteration
                if iteration % 10 == 0 or iteration == self.total_iterations - 1:
                    elapsed = time.time() - self.start_time
                    progress = (iteration + 1) / self.total_iterations
                    bar_length = 40
                    filled_length = int(bar_length * progress)
                    bar = '█' * filled_length + '░' * (bar_length - filled_length)
                    print(f'\rProgress: |{bar}| {progress:.1%} ({iteration + 1}/{self.total_iterations}) - {elapsed:.1f}s', end='', flush=True)
        
        # For SVM, we can't easily track progress, so we'll show a simple progress indicator
        print("Starting SVM training...")
        print("Note: SVM training doesn't provide iteration progress, but we'll show elapsed time.")
        
        # Start a background thread to show elapsed time
        import threading
        import sys
        
        def show_progress():
            while not hasattr(show_progress, 'stop'):
                elapsed = time.time() - start_time
                print(f'\rElapsed time: {elapsed:.1f}s', end='', flush=True)
                time.sleep(1)
        
        progress_thread = threading.Thread(target=show_progress)
        progress_thread.daemon = True
        progress_thread.start()
        
        try:
            model.fit(X_train, y_train)
        finally:
            show_progress.stop = True
            progress_thread.join(timeout=1)
            print()  # New line after progress display
    
    else:
        # For logistic regression, it's usually fast
        model.fit(X_train, y_train)
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Evaluate on test set
    print("\n8. Evaluating on test set...")
    test_pred = model.predict(X_test)
    test_accuracy = (test_pred == y_test).mean() * 100.0
    
    print(f"Test Accuracy: {test_accuracy:.2f}%")
    
    # Get prediction probabilities if available
    if hasattr(model, 'predict_proba'):
        test_proba = model.predict_proba(X_test)
        print(f"Test Probabilities shape: {test_proba.shape}")
    
    # Prepare results
    results = {
        'test_accuracy': test_accuracy,
        'training_time': training_time,
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'features': X_train.shape[1]
    }
    
    # Save model with organized structure
    model_dir = save_sklearn_model(
        model=model,
        model_name=config['model_name'],
        results=results,
        config=config
    )
    
    print("\nTraining completed successfully!")


if __name__ == "__main__":
    main()
