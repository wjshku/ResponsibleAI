#!/usr/bin/env python3
"""
Manipulator Module

Main orchestrator for image manipulation with metadata tracking.

Author: AI Assistant
Date: 2024
"""

import os
import json
import time
import shutil
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict

from dataloader import DataLoader
from manipulation_tools import InpainterTool, EditTool, ManipulationTool
from config import DEFAULT_SETTINGS
import random


def get_random_prompt(tool: str) -> str:
    """Get a random prompt from the damage prompts list with car context"""
    if tool == "inpainter":
        inpainter_prompts = {
            # Minor damage
            "minor_scratch": "minor scratch marks, light surface damage, subtle wear",
            "small_dent": "small dent, minor indentation, light impact damage",
            "light_rust": "light rust spots, beginning corrosion, minor oxidation",
            
            # Moderate damage
            "deep_scratch": "deep scratch marks, visible surface damage",
            "medium_dent": "medium-sized dent, noticeable indentation",
            
            # Severe damage
            "major_dent": "large dent, severe indentation, major impact damage",
            "deep_scratch_severe": "deep severe scratches, major surface damage, heavy wear",
            
            # Collision damage
            "collision_damage": "collision damage, impact marks, crash damage",
            "impact_dents": "impact dents, collision marks, crash damage",

            # Vandalism damage
            "graffiti": "graffiti, spray paint, vandalism marks",
            "key_scratch": "key scratches, intentional damage, vandalism",
        }
        base_prompt = random.choice(list(inpainter_prompts.values()))

        car_contexts = [
            "on this car, ",
            "on the vehicle, ",
            "on this automobile, ",
            "on the car surface, "
        ]
        car_context = random.choice(car_contexts)

        prompt = car_context + base_prompt
    elif tool == "editor":
        edit_prompts = {
            # Minor damage
            "minor_scratch": "minor scratch marks, light surface damage, subtle wear",
            "small_dent": "small dent, minor indentation, light impact damage",
            "light_rust": "light rust spots, beginning corrosion, minor oxidation",
            
            # Moderate damage
            "deep_scratch": "deep scratch marks, visible surface damage",
            "medium_dent": "medium-sized dent, noticeable indentation",
            
            # Severe damage
            "major_dent": "large dent, severe indentation, major impact damage",
            "deep_scratch_severe": "deep severe scratches, major surface damage, heavy wear",
            
            # Collision damage
            "collision_damage": "collision damage, impact marks, crash damage",
            "impact_dents": "impact dents, collision marks, crash damage",

            # Vandalism damage
            "graffiti": "graffiti, spray paint, vandalism marks",
            "key_scratch": "key scratches, intentional damage, vandalism",
        }
        base_prompt = random.choice(list(edit_prompts.values()))

        car_contexts = [
            "on this damaged car, ",
            "on the damaged vehicle, ",
            "on this damaged automobile, ",
            "on the damaged car surface, "
        ]
        car_context = random.choice(car_contexts)

        damage_contexts = [
            "find the damaged parts of the car and modify them to have, ",
            "modify the damaged parts of the car to have, ",
            "modify the damaged parts of the car and have, ",
        ]
        damage_context = random.choice(damage_contexts)

        prompt = car_context + damage_context + base_prompt
    else:
        raise ValueError(f"Invalid tool: {tool}")
    return prompt


@dataclass
class ProcessingMetadata:
    """Metadata for a single image processing operation"""
    image_id: str
    original_image_path: str
    mask_path: str
    processed_image_path: str
    processing_time: float
    timestamp: str
    tool_name: str
    tool_version: str
    tool_parameters: Dict[str, Any]
    model_name: str
    model_version: str
    model_provider: str
    image_dimensions: Tuple[int, int]
    mask_area_pixels: int
    success: bool
    error_message: Optional[str] = None
    additional_info: Optional[Dict[str, Any]] = None


@dataclass
class BatchMetadata:
    """Metadata for a batch processing operation"""
    batch_id: str
    start_time: str
    end_time: str
    total_processing_time: float
    total_images: int
    successful_images: int
    failed_images: int
    dataset_path: str
    output_directory: str
    tool_name: str
    tool_version: str
    model_name: str
    model_version: str
    model_provider: str
    tool_configuration: Dict[str, Any]
    processing_settings: Dict[str, Any]


class MetadataTracker:
    """Handles metadata collection and persistence"""
    
    def __init__(self, output_dir: str = "./manipulated_results"):
        """
        Initialize MetadataTracker
        
        Args:
            output_dir: Directory to save metadata files
        """
        self.output_dir = output_dir
        self.metadata_dir = os.path.join(output_dir, "metadata")
        os.makedirs(self.metadata_dir, exist_ok=True)
        
    def create_processing_metadata(self, 
                                 image_id: str,
                                 original_image_path: str,
                                 mask_path: str,
                                 processed_image_path: str,
                                 processing_time: float,
                                 tool_name: str,
                                 tool_version: str,
                                 tool_parameters: Dict[str, Any],
                                 model_name: str,
                                 model_version: str,
                                 model_provider: str,
                                 success: bool,
                                 error_message: Optional[str] = None,
                                 additional_info: Optional[Dict[str, Any]] = None) -> ProcessingMetadata:
        """Create metadata for a single processing operation"""
        
        # Get image dimensions
        image_dimensions = None
        mask_area_pixels = 0
        
        try:
            import cv2
            image = cv2.imread(original_image_path)
            if image is not None:
                image_dimensions = image.shape[:2]
        except:
            pass
            
        try:
            import cv2
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                import numpy as np
                mask_area_pixels = int(np.sum(mask > 0))
        except:
            pass
        
        return ProcessingMetadata(
            image_id=image_id,
            original_image_path=original_image_path,
            mask_path=mask_path,
            processed_image_path=processed_image_path,
            processing_time=processing_time,
            timestamp=datetime.now().isoformat(),
            tool_name=tool_name,
            tool_version=tool_version,
            tool_parameters=tool_parameters,
            model_name=model_name,
            model_version=model_version,
            model_provider=model_provider,
            image_dimensions=image_dimensions,
            mask_area_pixels=mask_area_pixels,
            success=success,
            error_message=error_message,
            additional_info=additional_info
        )
    
    def save_processing_metadata(self, metadata: ProcessingMetadata) -> str:
        """Save processing metadata to JSON file"""
        filename = f"processing_{metadata.image_id}_{metadata.timestamp.replace(':', '-')}.json"
        filepath = os.path.join(self.metadata_dir, filename)
        
        # Convert dataclass to dict and handle non-serializable types
        metadata_dict = asdict(metadata)
        if metadata_dict['image_dimensions'] is not None:
            metadata_dict['image_dimensions'] = list(metadata_dict['image_dimensions'])
        
        with open(filepath, 'w') as f:
            json.dump(metadata_dict, f, indent=2)
            
        return filepath
    
    def create_batch_metadata(self, 
                            batch_id: str,
                            start_time: str,
                            end_time: str,
                            total_processing_time: float,
                            total_images: int,
                            successful_images: int,
                            failed_images: int,
                            dataset_path: str,
                            output_directory: str,
                            tool_name: str,
                            tool_version: str,
                            model_name: str,
                            model_version: str,
                            model_provider: str,
                            tool_configuration: Dict[str, Any],
                            processing_settings: Dict[str, Any]) -> BatchMetadata:
        """Create metadata for a batch processing operation"""
        
        return BatchMetadata(
            batch_id=batch_id,
            start_time=start_time,
            end_time=end_time,
            total_processing_time=total_processing_time,
            total_images=total_images,
            successful_images=successful_images,
            failed_images=failed_images,
            dataset_path=dataset_path,
            output_directory=output_directory,
            tool_name=tool_name,
            tool_version=tool_version,
            model_name=model_name,
            model_version=model_version,
            model_provider=model_provider,
            tool_configuration=tool_configuration,
            processing_settings=processing_settings
        )
    
    def save_batch_metadata(self, metadata: BatchMetadata) -> str:
        """Save batch metadata to JSON file"""
        filename = f"batch_{metadata.batch_id}_{metadata.start_time.replace(':', '-')}.json"
        filepath = os.path.join(self.metadata_dir, filename)
        
        metadata_dict = asdict(metadata)
        
        with open(filepath, 'w') as f:
            json.dump(metadata_dict, f, indent=2)
            
        return filepath
    
    def load_metadata_summary(self) -> Dict[str, Any]:
        """Load and summarize all metadata files"""
        summary = {
            "total_processing_files": 0,
            "total_batch_files": 0,
            "total_images_processed": 0,
            "successful_images": 0,
            "failed_images": 0,
            "total_processing_time": 0.0,
            "recent_batches": []
        }
        
        # Process individual metadata files
        for filename in os.listdir(self.metadata_dir):
            if filename.startswith("processing_"):
                filepath = os.path.join(self.metadata_dir, filename)
                try:
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                    
                    summary["total_processing_files"] += 1
                    summary["total_images_processed"] += 1
                    
                    if data.get("success", False):
                        summary["successful_images"] += 1
                    else:
                        summary["failed_images"] += 1
                    
                    summary["total_processing_time"] += data.get("processing_time", 0.0)
                    
                except Exception as e:
                    print(f"Error loading metadata file {filename}: {e}")
        
        # Process batch metadata files
        for filename in os.listdir(self.metadata_dir):
            if filename.startswith("batch_"):
                filepath = os.path.join(self.metadata_dir, filename)
                try:
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                    
                    summary["total_batch_files"] += 1
                    summary["recent_batches"].append({
                        "batch_id": data.get("batch_id"),
                        "start_time": data.get("start_time"),
                        "total_images": data.get("total_images", 0),
                        "successful_images": data.get("successful_images", 0)
                    })
                    
                except Exception as e:
                    print(f"Error loading batch metadata file {filename}: {e}")
        
        # Sort recent batches by start time
        summary["recent_batches"].sort(key=lambda x: x["start_time"], reverse=True)
        summary["recent_batches"] = summary["recent_batches"][:10]  # Keep only 10 most recent
        
        return summary


class Manipulator:
    """Main orchestrator for image manipulation with metadata tracking"""
    
    def __init__(self, 
                 dataset_path: str,
                 output_dir: str = "./manipulated_results",
                 tool: ManipulationTool = None,
                 model_name: str = None,
                 model_version: str = None,
                 model_provider: str = None,
                 use_local: bool = True,
                 api_url: str = None,
                 api_key: str = None,
                 fallback: bool = False):
        """
        Initialize the Manipulator
        
        Args:
            dataset_path: Path to SOD dataset root directory
            output_dir: Directory to save results and metadata
            tool: Manipulation tool to use (defaults to InpainterTool with local mode)
            model_name: Name of the AI model to use (if creating default tool)
            model_version: Version of the AI model (if creating default tool)
            model_provider: Provider of the AI model (if creating default tool)
            use_local: Whether to use local mode (if creating default tool)
            api_url: API URL for the tool (if creating default tool)
            api_key: API key for the tool (if creating default tool)
            fallback: Whether to use fallback mode (if creating default tool)
        """
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize components
        self.dataloader = DataLoader(dataset_path)
        self.metadata_tracker = MetadataTracker(output_dir)
        
        # Initialize tool
        if tool == "inpainter":
            self.tool = InpainterTool(
                use_local=use_local,
                api_url=api_url,
                api_key=api_key,
                fallback=fallback,
                model_name=model_name,
                model_version=model_version,
                model_provider=model_provider,
                output_dir=output_dir
            )
        elif tool == "edit":
            self.tool = EditTool(
                use_local=use_local,
                api_url=api_url,
                api_key=api_key,
                fallback=fallback,
                model_name=model_name,
                model_version=model_version,
                model_provider=model_provider,
                output_dir=output_dir
            )
        else:
            self.tool = tool
        
        # Track processing state
        self.current_batch_id = None
        self.processing_metadata = []
    
    def process_sample(self, 
                      sample_index: int = None,
                      random_seed: int = None,
                      use_random_prompts: bool = False,
                      custom_prompt: str = None) -> Dict[str, Any]:
        """
        Process a single sample from the dataset
        
        Args:
            sample_index: Specific sample index (if None, uses random sample)
            random_seed: Random seed for random sampling
            use_random_prompts: Whether to use random prompts from the default list
            custom_prompt: Custom manipulation prompt (overrides random prompts)
            
        Returns:
            Dictionary with processing results and metadata
        """
        # Generate sample ID
        sample_id = f"sample_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        print(f"Processing sample: {sample_id}")
        print(f"Tool: {self.tool.tool_name} v{self.tool.tool_version}")
        
        # Get sample from dataset
        try:
            sample = self.dataloader.get_sample(index=sample_index, random_seed=random_seed)
            print(f"Sample: {sample['image_file']}")
        except Exception as e:
            print(f"Error loading sample: {e}")
            return {"success": False, "error": str(e)}
        
        # Prepare tool parameters
        tool_params = self.tool.get_default_parameters().copy()
        
        # Determine prompt/parameters
        if custom_prompt:
            tool_params['prompt'] = custom_prompt
        elif use_random_prompts:
            tool_params['prompt'] = get_random_prompt(self.tool.tool_name)
        
        # Process the image
        processing_start = time.time()
        success = False
        error_message = None
        processed_path = None
        
        try:
            # Process the image with selected tool
            result_url = self.tool.process_image(
                sample['image_path'],
                sample['mask_path'],
                **tool_params
            )
            
            # Handle result - could be URL or local file path
            if result_url.startswith('http'):
                # Download from URL
                processed_filename = f"processed_{sample_id}.jpg"
                processed_path = os.path.join(self.output_dir, processed_filename)
                
                import requests
                response = requests.get(result_url)
                response.raise_for_status()
                with open(processed_path, 'wb') as f:
                    f.write(response.content)
            else:
                # Local file path - check if it's a fallback (original image) or processed result
                if result_url == sample['image_path']:
                    # Fallback mode - copy original to output directory with processed name
                    processed_filename = f"processed_{sample_id}.jpg"
                    processed_path = os.path.join(self.output_dir, processed_filename)
                    shutil.copy2(result_url, processed_path)
                else:
                    # Processed result - use it directly (already saved by tool with unique name)
                    processed_path = result_url
                    processed_filename = os.path.basename(processed_path)
            
            success = True
            print(f"✓ Successfully processed sample -> {processed_filename}")
            
        except Exception as e:
            error_message = str(e)
            print(f"✗ Error processing sample: {error_message}")
        
        processing_time = time.time() - processing_start
        
        # Create and save processing metadata
        sample_info = self.dataloader.get_sample_info(sample)
        processing_meta = self.metadata_tracker.create_processing_metadata(
            image_id=sample_id,
            original_image_path=sample['image_path'],
            mask_path=sample['mask_path'],
            processed_image_path=processed_path or "",
            processing_time=processing_time,
            tool_name=self.tool.tool_name,
            tool_version=self.tool.tool_version,
            tool_parameters=tool_params,
            model_name=self.tool.model_name,
            model_version=self.tool.model_version,
            model_provider=self.tool.model_provider,
            success=success,
            error_message=error_message,
            additional_info=sample_info
        )
        
        # Save individual metadata
        metadata_file = self.metadata_tracker.save_processing_metadata(processing_meta)
        self.processing_metadata = [processing_meta]  # Store single sample metadata
        
        # Print summary
        print(f"\n=== Sample Processing Complete ===")
        print(f"Sample ID: {sample_id}")
        print(f"Success: {success}")
        print(f"Processing time: {processing_time:.2f} seconds")
        print(f"Result saved to: {processed_path}")
        print(f"Metadata saved to: {self.metadata_tracker.metadata_dir}")
        
        return {
            "success": success,
            "sample_id": sample_id,
            "processing_time": processing_time,
            "output_path": processed_path,
            "metadata_directory": self.metadata_tracker.metadata_dir,
            "processing_metadata": processing_meta,
            "error_message": error_message
        }
    
    def process_full_dataset(self, 
                           shuffle: bool = True,
                           random_seed: int = None,
                           use_random_prompts: bool = False,
                           custom_prompt: str = None,
                           max_samples: int = None,
                           start_index: int = 0) -> Dict[str, Any]:
        """
        Process the full dataset
        
        Args:
            shuffle: Whether to shuffle the samples
            random_seed: Random seed for reproducible shuffling
            use_random_prompts: Whether to use random prompts from the default list
            custom_prompt: Custom manipulation prompt (overrides random prompts)
            max_samples: Maximum number of samples to process (None for all samples)
            start_index: Index to start processing from (skip first N samples)
            
        Returns:
            Dictionary with processing results and metadata
        """
        # Generate batch ID
        self.current_batch_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        start_time = datetime.now().isoformat()
        
        print(f"Starting full dataset processing: {self.current_batch_id}")
        print(f"Dataset path: {self.dataset_path}")
        print(f"Dataset size: {self.dataloader.size}")
        print(f"Tool: {self.tool.tool_name} v{self.tool.tool_version}")
        
        # Load full dataset
        try:
            samples = self.dataloader.load_full_dataset(shuffle=shuffle, random_seed=random_seed)
            print(f"Loaded {len(samples)} samples")
        except Exception as e:
            print(f"Error loading full dataset: {e}")
            return {"success": False, "error": str(e)}
        
        if not samples:
            print("No samples found!")
            return {"success": False, "error": "No samples found"}
        
        # Process each sample
        self.processing_metadata = []
        successful_images = 0
        failed_images = 0
        
        # Apply start_index to skip samples at the beginning
        if start_index > 0:
            samples = samples[start_index:]
            print(f"Skipping first {start_index} samples, starting from index {start_index}")
        
        # Determine how many samples to process
        total_samples = len(samples)
        if max_samples is not None:
            total_samples = min(max_samples, len(samples))
            print(f"Processing {total_samples} samples (limited by max_samples={max_samples})")
        
        for i, sample in enumerate(samples):
            i = i + start_index # show the same index as the original dataset
            # Stop if we've reached max_samples
            if max_samples is not None and i - start_index >= max_samples:
                print(f"\nStopping at {i} samples (max_samples={max_samples} reached)")
                break
                
            print(f"\nProcessing sample {i+1}/{total_samples+start_index}...")
            print(f"Image: {os.path.basename(sample['image_path'])}")
            
            # Get sample info
            sample_info = self.dataloader.get_sample_info(sample)
            
            # Prepare tool parameters
            tool_params = self.tool.get_default_parameters().copy()
            
            # Determine prompt/parameters
            if custom_prompt:
                tool_params['prompt'] = custom_prompt
            elif use_random_prompts:
                tool_params['prompt'] = get_random_prompt(self.tool.tool_name)
            
            # Process the image
            processing_start = time.time()
            success = False
            error_message = None
            processed_path = None
            
            try:
                # Process the image with selected tool
                result_url = self.tool.process_image(
                    sample['image_path'],
                    sample['mask_path'],
                    **tool_params
                )
                
                # Handle result - could be URL or local file path
                if result_url.startswith('http'):
                    # Download from URL
                    processed_filename = f"processed_{self.current_batch_id}_{i+1:04d}.jpg"
                    processed_path = os.path.join(self.output_dir, processed_filename)
                    
                    import requests
                    response = requests.get(result_url)
                    response.raise_for_status()
                    with open(processed_path, 'wb') as f:
                        f.write(response.content)
                else:
                    # Local file path - check if it's a fallback (original image) or processed result
                    if result_url == sample['image_path']:
                        # Fallback mode - copy original to output directory with processed name
                        processed_filename = f"processed_{self.current_batch_id}_{i+1:04d}.jpg"
                        processed_path = os.path.join(self.output_dir, processed_filename)
                        shutil.copy2(result_url, processed_path)
                    else:
                        # Processed result - use it directly (already saved by tool with unique name)
                        processed_path = result_url
                        processed_filename = os.path.basename(processed_path)
                
                success = True
                successful_images += 1
                print(f"✓ Successfully processed sample {i+1} -> {processed_filename}")
                
            except Exception as e:
                success = False
                error_message = str(e)
                failed_images += 1
                print(f"✗ Error processing sample {i+1}: {error_message}")
            
            processing_time = time.time() - processing_start
            
            # Create and save processing metadata
            image_id = f"{self.current_batch_id}_{i+1:04d}"
            processing_meta = self.metadata_tracker.create_processing_metadata(
                image_id=image_id,
                original_image_path=sample['image_path'],
                mask_path=sample['mask_path'],
                processed_image_path=processed_path or "",
                processing_time=processing_time,
                tool_name=self.tool.tool_name,
                tool_version=self.tool.tool_version,
                tool_parameters=tool_params,
                model_name=self.tool.model_name,
                model_version=self.tool.model_version,
                model_provider=self.tool.model_provider,
                success=success,
                error_message=error_message,
                additional_info=sample_info
            )
            
            # Save individual metadata
            if success:
                metadata_file = self.metadata_tracker.save_processing_metadata(processing_meta)
            else:
                print(f"No metadata saved for failed operation")
            self.processing_metadata.append(processing_meta)
            
            # Add delay between requests to avoid rate limiting
            if hasattr(self.tool, 'use_local') and not self.tool.use_local:
                time.sleep(2)
        
        # Create and save batch metadata
        end_time = datetime.now().isoformat()
        total_processing_time = sum(meta.processing_time for meta in self.processing_metadata)
        
        batch_meta = self.metadata_tracker.create_batch_metadata(
            batch_id=self.current_batch_id,
            start_time=start_time,
            end_time=end_time,
            total_processing_time=total_processing_time,
            total_images=len(samples),
            successful_images=successful_images,
            failed_images=failed_images,
            dataset_path=self.dataset_path,
            output_directory=self.output_dir,
            tool_name=self.tool.tool_name,
            tool_version=self.tool.tool_version,
            model_name=self.tool.model_name,
            model_version=self.tool.model_version,
            model_provider=self.tool.model_provider,
            tool_configuration=self.tool.get_default_parameters(),
            processing_settings={
                "shuffle": shuffle,
                "random_seed": random_seed,
                "use_random_prompts": use_random_prompts,
                "custom_prompt": custom_prompt
            }
        )
        
        batch_metadata_file = self.metadata_tracker.save_batch_metadata(batch_meta)
        
        # Print summary
        print(f"\n=== Full Dataset Processing Complete ===")
        print(f"Batch ID: {self.current_batch_id}")
        print(f"Total images: {len(samples)}")
        print(f"Successful: {successful_images}")
        print(f"Failed: {failed_images}")
        print(f"Total processing time: {total_processing_time:.2f} seconds")
        print(f"Results saved to: {self.output_dir}")
        print(f"Metadata saved to: {self.metadata_tracker.metadata_dir}")
        
        return {
            "success": True,
            "batch_id": self.current_batch_id,
            "total_images": len(samples),
            "successful_images": successful_images,
            "failed_images": failed_images,
            "total_processing_time": total_processing_time,
            "output_directory": self.output_dir,
            "metadata_directory": self.metadata_tracker.metadata_dir,
            "processing_metadata": self.processing_metadata,
            "batch_metadata": batch_meta
        }
    
    def get_processing_summary(self) -> Dict[str, Any]:
        """Get summary of all processing operations"""
        return self.metadata_tracker.load_metadata_summary()
    
    def visualize_results(self, sample_index: int = 0, show_metadata: bool = True):
        """Visualize results for a specific sample"""
        if not self.processing_metadata:
            print("No processing metadata available. Run process_sample() or process_full_dataset() first.")
            return
        
        if sample_index >= len(self.processing_metadata):
            print(f"Sample index {sample_index} out of range. Available samples: {len(self.processing_metadata)}")
            return
        
        meta = self.processing_metadata[sample_index]
        
        # Import visualization function
        from dataloader import visualize_results
        
        # Visualize the results
        visualize_results(
            meta.original_image_path,
            meta.mask_path,
            meta.processed_image_path,
            f"Sample {sample_index + 1} - {meta.image_id}"
        )
        
        # Print metadata if requested
        if show_metadata:
            print(f"\n=== Sample {sample_index + 1} Metadata ===")
            print(f"Image ID: {meta.image_id}")
            print(f"Processing time: {meta.processing_time:.2f} seconds")
            print(f"Tool used: {meta.tool_name} v{meta.tool_version}")
            print(f"Model: {meta.model_name} v{meta.model_version} ({meta.model_provider})")
            print(f"Tool parameters: {meta.tool_parameters}")
            print(f"Success: {meta.success}")
            if meta.image_dimensions:
                print(f"Image dimensions: {meta.image_dimensions}")
            print(f"Mask area: {meta.mask_area_pixels} pixels")
            if meta.error_message:
                print(f"Error: {meta.error_message}")


def test_edit(fallback=False):
    """Test case for Manipulator - process one sample with EditTool"""
    print("=== Manipulator Test: Process One Sample with EditTool ===")
    print(f"Fallback mode: {'ON' if fallback else 'OFF'}\n")
    
    try:
        # Initialize Manipulator with EditTool
        from config import SOD_PATH
        dataset_path = os.path.join(SOD_PATH, "CarDD-TR")
        
        # Create EditTool
        edit_tool = EditTool(
            use_local=True,
            api_url="http://localhost:8000/edit",
            output_dir="./test_results",
            fallback=fallback
        )
        
        manipulator = Manipulator(
            dataset_path=dataset_path, 
            output_dir="./test_results",
            tool=edit_tool
        )
        print(f"✓ Manipulator with EditTool initialized")
        print(f"  Dataset size: {manipulator.dataloader.size}")
        print(f"  Tool: {manipulator.tool.tool_name} v{manipulator.tool.tool_version}")
        print(f"  Model: {manipulator.tool.model_name} v{manipulator.tool.model_version}")
        
        # Process one sample with edit tool
        print(f"\n🚀 Processing sample 0 with EditTool...")
        results = manipulator.process_sample(
            sample_index=0,
            use_random_prompts=True,
            # custom_prompt="change the car color to red",
        )
        
        if results["success"]:
            print(f"✅ Sample processed successfully with EditTool!")
            print(f"  Output: {results['output_path']}")
            print(f"  Time: {results['processing_time']:.2f}s")
            # Show prompt used
            try:
                used_prompt = results['processing_metadata'].tool_parameters.get('prompt', 'N/A')
                print(f"  Prompt: {used_prompt}")
            except Exception:
                pass
            
            # Visualize result with metadata
            try:
                print("\n🖼️  Visualizing result...")
                manipulator.visualize_results(0, show_metadata=True)
            except Exception as viz_e:
                print(f"⚠ Visualization failed: {viz_e}")
        else:
            print(f"❌ Sample processing failed: {results.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"❌ EditTool test failed: {e}")
    
    print("\n=== EditTool Test Complete ===")

def main():
    """Main function - choose which test to run"""
    import sys
    
    test_edit(fallback=False)


if __name__ == "__main__":
    main()