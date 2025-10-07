# Car Damage Inpainting System

This system uses AI APIs to inpaint car damage areas in the CarDD dataset, creating realistic repairs where damage has been filled in.

## Features

- **Stability AI Support**: Uses Stability AI for high-quality inpainting
- **Local API Fallback**: Includes local API that returns original images for testing
- **Dual Dataset Support**: Handles both COCO and SOD format datasets
- **Automatic Mask Generation**: Creates masks from bounding boxes or uses existing SOD masks
- **Batch Processing**: Process multiple images with customizable settings
- **Comprehensive Metadata Tracking**: Tracks processing times, success rates, and detailed metadata
- **DataLoader Integration**: Unified interface for loading images and masks from different sources
- **Visualization**: Compare original, masked, and inpainted images
- **Error Handling**: Robust error handling and retry mechanisms
 - **Qwen Image Edit (New)**: Prompt-based whole-image editing via `EditTool`

## Installation

### Option 1: Using uv (Recommended)

1. Install uv if you haven't already:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Run the setup script:
```bash
./setup_uv.sh
```

3. Activate the virtual environment:
```bash
source .venv/bin/activate
```

### Option 2: Using pip

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### API Credentials (Optional)

Set up API credentials (optional):
```bash
# For Stability AI (optional - can use local fallback)
export STABILITY_API_KEY="your_stability_key"
```

### Optional: Enable Qwen Image Edit

Install extra dependencies:
```bash
uv add ".[edit]"  # if using uv
# or with pip
pip install diffusers  # install PyTorch separately as per your platform
```

Install PyTorch (follow official instructions for your platform and CUDA/MPS):
```bash
# See: https://pytorch.org/get-started/locally/
```

## Quick Start

### 1. Test the Setup
```bash
# Test without API calls (checks dataset availability)
python test.py

# Test with API calls
python test.py YOUR_API_KEY
python test.py --local
```

### 2. Run Inpainting on 4 Images
```bash
# Using Local API (returns original images - good for testing)
python inpaint.py --use-local --num-samples 4

# Using Stability AI
python inpaint.py --api-key YOUR_STABILITY_KEY --num-samples 4
```

### 3. Process Different Dataset Types
```bash
# Process SOD dataset (default)
python inpaint.py --use-local --dataset-type sod --num-samples 4

# Process COCO dataset
python inpaint.py --api-key YOUR_KEY --dataset-type coco --num-samples 4
```

### 4. Using the Manipulator (New!)
```bash
# Basic usage with metadata tracking
python manipulator.py --use-local --num-samples 4

# With specific damage type
python manipulator.py --use-local --damage-type dent --num-samples 2

# With custom prompt
python manipulator.py --use-local --custom-prompt "restore to perfect condition" --num-samples 2

# Run example script
python example_usage.py
```

### 5. Testing Individual Modules
```bash
# Test data loading functionality
python dataloader.py

# Test manipulation tools
python manipulation_tools.py

# Test main manipulator
python manipulator.py

# Run integration tests
python integration_test.py

# Test full dataset loading
python example_full_dataset.py
```

## API Providers

### Stability AI (Recommended)
- **Model**: Stable Diffusion XL
- **Pros**: High quality, customizable, good for car damage
- **Cost**: Pay per image
- **Setup**: Get key from [platform.stability.ai](https://platform.stability.ai)

### Local API (Fallback)
- **Model**: Local fallback
- **Pros**: No API costs, always available, good for testing
- **Cons**: Returns original image without inpainting
- **Setup**: No setup required

### Qwen Image Edit (Diffusers)
- **Model**: `Qwen/Qwen-Image-Edit`
- **Pros**: High-quality prompt-based image edits
- **Cons**: Requires GPU for speed; installs via diffusers + PyTorch
- **Setup**: Install extras and PyTorch, no API key needed

## Metadata Tracking

The Manipulator automatically tracks comprehensive metadata for each processing operation:

### Individual Processing Metadata
- Image ID and file paths
- Processing time and timestamp
- API used and prompt applied
- Image dimensions and mask area
- Success/failure status and error messages
- Additional sample information

### Batch Processing Metadata
- Batch ID and timing information
- Total processing statistics
- Dataset configuration
- API settings and processing parameters

### Metadata Storage
Metadata is saved as JSON files in the `metadata/` subdirectory:
- `processing_*.json`: Individual image processing metadata
- `batch_*.json`: Batch processing metadata
- Summary statistics available via `get_processing_summary()`

## Configuration

Edit `config.py` to customize:
- API keys
- Default prompts
- Model settings
- Output directories

## Module Structure

The system is now organized into three main modules:

### 1. `dataloader.py` - Data Loading
- **DataLoader**: Loads images and masks from COCO and SOD datasets
- **Data Loading Functions**: `load_coco_annotations()`, `get_sample_images_from_coco()`, `get_sample_images_from_sod()`
- **Full Dataset Loading**: `load_full_dataset()`, `load_dataset_batch()`, `get_dataset_size()`
- **Visualization**: `visualize_results()` for displaying original, mask, and processed images
- **Self-contained**: No external dependencies on other project files

### 2. `manipulation_tools.py` - Image Processing Tools
- **ManipulationTool**: Abstract base class for all manipulation tools
- **Built-in Tools**: InpainterTool, EditTool
- **ToolRegistry**: Manages available tools
- **Factory Functions**: Create tools by name with parameters

### 3. `manipulator.py` - Main Orchestrator
- **Manipulator**: Main class that coordinates data loading, tool execution, and metadata tracking
- **MetadataTracker**: Handles processing and batch metadata
- **ProcessingMetadata & BatchMetadata**: Data classes for metadata storage

## Usage Examples

### Basic Usage with Manipulator (Recommended)
```python
from manipulator import Manipulator
from manipulation_tools import InpainterTool

# Create tools
inpainter = InpainterTool(use_local=True)

# Initialize manipulator with metadata tracking
manipulator = Manipulator(
    dataset_type="sod",
    output_dir="./results",
    manipulate_tools=[inpainter]
)

# Process a batch with metadata tracking
results = manipulator.process_batch(
    num_samples=4,
    damage_type="dent",  # Use specific damage type prompt
    random_seed=42
)

# Get processing summary
summary = manipulator.get_processing_summary()
print(f"Processed {summary['total_images_processed']} images")

# Visualize results
manipulator.visualize_results(0)  # Show first result
```

### Direct Inpainter Usage
```python
from inpaint import CarDamageInpainter

# Initialize inpainter
inpainter = CarDamageInpainter("your_api_key", use_local=False)

# Inpaint a single image
result_url = inpainter.inpaint_image(
    "path/to/image.jpg",
    "path/to/mask.png",
    "repair car damage, make it look new"
)

# Download result
inpainter.download_result(result_url, "output.jpg")
```

### Direct EditTool Usage (Qwen Image Edit)
```python
from manipulation_tools import EditTool

editor = EditTool(device="cuda", dtype="bfloat16", output_dir="./edited_results")

result_path = editor.process_image(
    image_path="./input.png",
    mask_path="unused.png",  # kept for interface compatibility
    prompt="Change the rabbit's color to purple, with a flash light background.",
    num_inference_steps=50,
    true_cfg_scale=4.0,
    negative_prompt=" ",
    seed=0,
)
print("Saved to", result_path)
```

### Batch Processing with DataLoader
```python
from dataloader import DataLoader

# Initialize dataloader
dataloader = DataLoader("sod", "/path/to/dataset")

# Load samples
samples = dataloader.load_samples(num_samples=10, random_seed=42)

# Process each sample
for sample in samples:
    print(f"Image: {sample['image_path']}")
    print(f"Mask: {sample['mask_path']}")
```

### Full Dataset Loading
```python
from dataloader import DataLoader

# Initialize dataloader
dataloader = DataLoader("sod", "/path/to/dataset")

# Get dataset size
dataset_size = dataloader.get_dataset_size()
print(f"Dataset contains {dataset_size} samples")

# Load full dataset
full_samples = dataloader.load_full_dataset(shuffle=True, random_seed=42)
print(f"Loaded {len(full_samples)} samples")

# Load dataset in batches (memory efficient)
batch_size = 100
for start_idx in range(0, dataset_size, batch_size):
    batch = dataloader.load_dataset_batch(
        batch_size=batch_size, 
        start_index=start_idx, 
        shuffle=True, 
        random_seed=42
    )
    print(f"Processing batch {start_idx//batch_size + 1}: {len(batch)} samples")
    # Process batch...
```

### Local API (Fallback)
- **Model**: No processing (returns original image)
- **Quality**: N/A (testing only)
- **Cost**: Free
- **Setup**: No setup required

## Dataset Structure

The system expects the CarDD dataset in the following structure:
```
CarDD_release/
├── CarDD_COCO/
│   ├── annotations/
│   │   ├── instances_train2017.json
│   │   ├── instances_val2017.json
│   │   └── instances_test2017.json
│   ├── train2017/
│   ├── val2017/
│   └── test2017/
└── CarDD_SOD/
    ├── CarDD-TR/
    │   ├── CarDD-TR-Image/
    │   ├── CarDD-TR-Mask/
    │   └── CarDD-TR-Edge/
    ├── CarDD-VAL/
    └── CarDD-TE/
```