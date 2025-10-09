#!/usr/bin/env python3
"""
Main manipulation script for generating car damage images

This script provides a simple interface to use the Manipulator class
for generating various types and severities of car damage.

Author: AI Assistant
Date: 2024
"""

from manipulator import Manipulator
from config import SOD_PATH
from manipulation_tools import EditTool, InpainterTool


def inpaint_pipeline():
    """Main function to run the damage generation pipeline"""
    
    print("🚗 Car Damage Generation Pipeline")
    print("Starting full dataset processing with local API...")
    
    try:
        # Initialize manipulator - let DataLoader handle all path validation
        print("🔧 Initializing manipulator...")
        dataset_type = "CarDD-VAL"

        manipulator = Manipulator(
            dataset_path=f"{SOD_PATH}/{dataset_type}",
            output_dir=f"./GenAI_Results/SD2/{dataset_type}",
            tool="inpainter",
            model_name="stable diffusion 2 inpainting",
            model_version="1.0.0",
            model_provider="local-inpainter",
            use_local=True,
            api_url="http://3.115.116.28:8000/inpaint",  # Custom API endpoint
            # api_url="http://3.113.213.110:8000/inpaint",
            fallback=False  # Set to True to return original images without processing
        )
        
        print(f"✅ Manipulator initialized successfully")
        print(f"   Dataset size: {manipulator.dataloader.size}")
        print(f"   Model: {manipulator.tool.model_name} v{manipulator.tool.model_version}")
        print(f"   Provider: {manipulator.tool.model_provider}")
        
        # Process full dataset with random prompts
        print(f"\n🚀 Starting full dataset processing...")
        results = manipulator.process_full_dataset(
            shuffle=True,
            random_seed=6,
            use_random_prompts=True,
            custom_prompt=None,
            max_samples=None,
            start_index=0
        )
        
        if results["success"]:
            print(f"\n✅ Processing completed successfully!")
            print(f"   📊 Total images: {results['total_images']}")
            print(f"   ✅ Successful: {results['successful_images']}")
            print(f"   ❌ Failed: {results['failed_images']}")
            print(f"   ⏱️  Total time: {results['total_processing_time']:.2f} seconds")
            print(f"   📁 Results saved to: {results['output_directory']}")
            print(f"   📋 Metadata saved to: {results['metadata_directory']}")
            
            print("\n🎉 Pipeline completed successfully!")
            print(f"Check the '/GenAI_Results/SD2/{dataset_type}' directory for results.")
        else:
            print(f"❌ Processing failed: {results.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Please check your dataset path in config.py")


def edit_pipeline():
    """Main function to run the damage generation pipeline"""
    
    print("🚗 Car Damage Generation Pipeline")
    print("Starting full dataset processing with local API...")
    
    try:
        dataset_type = "CarDD-VAL"
        model_name = "Qwen Image Edit"
        manipulator = Manipulator(
            dataset_path=f"{SOD_PATH}/{dataset_type}", 
            output_dir=f"./GenAI_Results/{model_name}/{dataset_type}",
            tool="edit",
            model_name=model_name,
            model_version="1.0.0",
            model_provider="local-edit",
            use_local=True,
            api_url="http://localhost:8000/edit",
            fallback=False
        )

        print(f"✅ Manipulator initialized successfully")
        print(f"   Dataset size: {manipulator.dataloader.size}")
        print(f"   Model: {manipulator.tool.model_name} v{manipulator.tool.model_version}")
        print(f"   Provider: {manipulator.tool.model_provider}")
        
        # Process full dataset with random prompts
        print(f"\n🚀 Starting full dataset processing...")
        results = manipulator.process_full_dataset(
            shuffle=True,
            random_seed=6,
            use_random_prompts=True,
            custom_prompt=None,
            max_samples=None,
            start_index=24
        )
        
        if results["success"]:
            print(f"\n✅ Processing completed successfully!")
            print(f"   📊 Total images: {results['total_images']}")
            print(f"   ✅ Successful: {results['successful_images']}")
            print(f"   ❌ Failed: {results['failed_images']}")
            print(f"   ⏱️  Total time: {results['total_processing_time']:.2f} seconds")
            print(f"   📁 Results saved to: {results['output_directory']}")
            print(f"   📋 Metadata saved to: {results['metadata_directory']}")
            
            print("\n🎉 Pipeline completed successfully!")
            print(f"Check the '/GenAI_Results/SD2/{dataset_type}' directory for results.")
        else:
            print(f"❌ Processing failed: {results.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Please check your dataset path in config.py")


    
if __name__ == "__main__":
    # inpaint_pipeline()
    edit_pipeline()