#!/usr/bin/env python3
"""
Clean up failed metadata files from the results directory

This script removes metadata files that correspond to failed operations,
keeping only those with successful inpainted images.
"""

import os
import json
import glob
from typing import List

def cleanup_failed_metadata(results_dir: str) -> None:
    """
    Remove metadata files for failed operations
    
    Args:
        results_dir: Path to the results directory containing metadata subfolder
    """
    metadata_dir = os.path.join(results_dir, "metadata")
    if not os.path.exists(metadata_dir):
        print(f"❌ Metadata directory not found: {metadata_dir}")
        return
    
    # Find all metadata files
    metadata_files = glob.glob(os.path.join(metadata_dir, "*.json"))
    print(f"📁 Found {len(metadata_files)} metadata files")
    
    removed_count = 0
    kept_count = 0
    
    for file_path in metadata_files:
        try:
            with open(file_path, 'r') as f:
                metadata = json.load(f)
            
            # Skip batch metadata files - only process individual processing files
            if 'batch_' in os.path.basename(file_path):
                kept_count += 1
                print(f"📋 Kept batch file: {os.path.basename(file_path)}")
                continue
            
            # Check if this is a failed individual processing operation
            success = metadata.get('success', False)
            processed_path = metadata.get('processed_image_path', '')
            
            # Remove if failed or no processed image path
            if not success or not processed_path or 'inpainted' not in os.path.basename(processed_path):
                os.remove(file_path)
                removed_count += 1
                print(f"🗑️  Removed: {os.path.basename(file_path)}")
            else:
                kept_count += 1
                print(f"✅ Kept: {os.path.basename(file_path)}")
                
        except Exception as e:
            print(f"⚠️  Error processing {file_path}: {e}")
    
    print(f"\n📊 Cleanup Summary:")
    print(f"   Removed: {removed_count} failed metadata files")
    print(f"   Kept: {kept_count} successful metadata files")
    print(f"   Total: {removed_count + kept_count} files processed")

def main():
    """Main function"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python cleanup_failed_metadata.py <results_directory>")
        print("  results_directory: Path to the results directory (e.g., ./manipulated_results)")
        return
    
    results_dir = sys.argv[1]
    if not os.path.exists(results_dir):
        print(f"❌ Results directory not found: {results_dir}")
        return
    
    print(f"🧹 Cleaning up failed metadata in: {results_dir}")
    cleanup_failed_metadata(results_dir)

if __name__ == "__main__":
    main()
