# Visualize the filters of the CNN model

import torch
import torch.nn as nn
import json
import sys
from pathlib import Path
import cv2

# Add the parent directory to the path to import from detection_car
sys.path.append(str(Path(__file__).parent.parent))
from detection_car.models import CNNClassifier

# Global model input size configuration
MODEL_INPUT_SIZE = (512, 512)

# ImageNet normalization constants (same as used in training)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def preprocess_image(tensor):
    """
    Apply Preprocessing to a tensor image.

    Args:
        tensor: Input tensor of shape (C, H, W) in range [0, 1]

    Returns:
        torch.Tensor: Normalized tensor with ImageNet statistics
    """
    import torchvision.transforms as T

    # Create normalization transform
    normalize = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    # Apply normalization
    return normalize(tensor)

def convert_to_numpy_image(input_image):
    """
    Convert input image (tensor or numpy array) to standardized numpy HWC format.

    Args:
        input_image: Input tensor of shape (C, H, W) or (1, C, H, W), or numpy array (H, W, C) or (C, H, W)

    Returns:
        numpy.ndarray: Image in HWC format (H, W, C) with values in original range
    """
    import numpy as np

    if isinstance(input_image, torch.Tensor):
        # Convert PyTorch tensor to numpy
        if input_image.dim() == 4:  # NCHW
            input_np = input_image.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)  # CHW -> HWC
        elif input_image.dim() == 3:
            # Check if it's CHW (3, H, W) or HWC (H, W, 3)
            if input_image.shape[0] == 3:  # CHW format
                input_np = input_image.detach().cpu().numpy().transpose(1, 2, 0)  # CHW -> HWC
            else:  # Assume HWC format (H, W, 3)
                input_np = input_image.detach().cpu().numpy()
        else:
            raise ValueError(f"Unsupported tensor shape: {input_image.shape}")
    elif isinstance(input_image, np.ndarray):
        # Already numpy array
        if input_image.ndim == 3 and input_image.shape[-1] == 3:  # HWC
            input_np = input_image
        elif input_image.ndim == 3 and input_image.shape[0] == 3:  # CHW
            input_np = input_image.transpose(1, 2, 0)  # CHW -> HWC
        else:
            raise ValueError(f"Unsupported numpy array shape: {input_image.shape}")
    else:
        raise ValueError(f"Unsupported input type: {type(input_image)}")

    return input_np

def load_model(model_path):
    """
    Load the CNN model from the model path.
    
    Args:
        model_path (str): Path to the model file (.pth)
        
    Returns:
        torch.nn.Module: Loaded CNN model in evaluation mode
    """
    # Get model directory to find metadata
    model_dir = Path(model_path).parent
    metadata_path = model_dir / "metadata.json"
    
    # Load metadata to get model configuration
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        config = metadata.get('config', {})
        hidden_size = config.get('hidden_size', 64)
        dropout = config.get('dropout', 0.2)
    else:
        # Default values if metadata not found
        hidden_size = 64
        dropout = 0.2
        print("Warning: metadata.json not found, using default model parameters")
    
    # Create model with same architecture
    model = CNNClassifier(
        input_channels=3,
        num_classes=1,
        dropout=dropout,
        hidden_size=hidden_size
    )
    
    # Load model weights
    device = torch.device('cpu')  # Use CPU for interpretation
    checkpoint = torch.load(model_path, map_location=device)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            # Full checkpoint with optimizer state
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            # Alternative checkpoint format
            model.load_state_dict(checkpoint['state_dict'])
        else:
            # Direct state dict
            model.load_state_dict(checkpoint)
    else:
        # Direct state dict
        model.load_state_dict(checkpoint)
    
    # Set to evaluation mode
    model.eval()
    
    print(f"Model loaded successfully from {model_path}")

    return model, metadata if metadata_path.exists() else None

def output_model_structure(model):
    """
    Output the structure of the CNN model.
    Namely, the number of filters/neurons in each layer.
    Size of the filters. 
    """
    import torch.nn as nn

    def summarize_sequential(prefix: str, seq: nn.Sequential):
        layers_summary = []
        for idx, layer in enumerate(seq):
            name = f"{prefix}[{idx}].{layer.__class__.__name__}"
            info: dict = {"name": name, "type": layer.__class__.__name__}
            if isinstance(layer, nn.Conv2d):
                info.update({
                    "in_channels": layer.in_channels,
                    "out_channels": layer.out_channels,
                    "kernel_size": tuple(layer.kernel_size),
                    "stride": tuple(layer.stride),
                    "padding": tuple(layer.padding),
                })
            elif isinstance(layer, nn.BatchNorm2d):
                info.update({"num_features": layer.num_features})
            elif isinstance(layer, nn.MaxPool2d):
                k = layer.kernel_size if isinstance(layer.kernel_size, tuple) else (layer.kernel_size, layer.kernel_size)
                s = layer.stride if isinstance(layer.stride, tuple) else (layer.stride, layer.stride)
                info.update({"kernel_size": tuple(k), "stride": tuple(s)})
            elif isinstance(layer, nn.AdaptiveAvgPool2d):
                info.update({"output_size": layer.output_size})
            elif isinstance(layer, nn.Linear):
                info.update({"in_features": layer.in_features, "out_features": layer.out_features})
            elif isinstance(layer, nn.Dropout):
                info.update({"p": layer.p})
            layers_summary.append(info)
        return layers_summary

    summary = {"layers": [], "total_parameters": int(sum(p.numel() for p in model.parameters()))}

    # Summarize feature extractor if present
    if hasattr(model, "features") and isinstance(model.features, nn.Sequential):
        summary["layers"].extend(summarize_sequential("features", model.features))
    # Summarize classifier if present
    if hasattr(model, "classifier") and isinstance(model.classifier, nn.Sequential):
        summary["layers"].extend(summarize_sequential("classifier", model.classifier))

    # Fallback: iterate named_modules for other models
    if not summary["layers"]:
        for name, layer in model.named_modules():
            if name == "":
                continue
            layer_type = layer.__class__.__name__
            info = {"name": name, "type": layer_type}
            if isinstance(layer, nn.Conv2d):
                info.update({
                    "in_channels": layer.in_channels,
                    "out_channels": layer.out_channels,
                    "kernel_size": tuple(layer.kernel_size),
                    "stride": tuple(layer.stride),
                    "padding": tuple(layer.padding),
                })
            elif isinstance(layer, nn.Linear):
                info.update({"in_features": layer.in_features, "out_features": layer.out_features})
            summary["layers"].append(info)

    # Pretty print
    print("=== Model Structure ===")
    print(f"Total parameters: {summary['total_parameters']:,}")
    for layer in summary["layers"]:
        name = layer["name"]
        ltype = layer["type"]
        details = {k: v for k, v in layer.items() if k not in ("name", "type")}
        print(f"- {name}: {ltype} {details}")

    return summary

def create_activation_overlay(input_image, activation_map, percent_threshold=None, threshold=None):
    """
    Create an activation overlay on an input image.
    Shows activation areas above threshold as original image, inactivated areas as darkened (transparent black).

    Args:
        input_image: Input tensor of shape (C, H, W) or (1, C, H, W), or numpy array (H, W, C)
        activation_map: Activation map as torch.Tensor or numpy array of shape (H', W')
        percent_threshold: Percentage of top activations to highlight (e.g., 1 for top 1%)
        threshold: Absolute threshold value for activations
        Note: Either percent_threshold or threshold must be provided, but not both

    Returns:
        numpy.ndarray: RGB image showing the activation overlay
    """
    import numpy as np

    # Convert input image to numpy HWC format
    input_np = convert_to_numpy_image(input_image)

    # Normalize to [0,1] if needed
    if input_np.dtype != np.float32:
        input_np = input_np.astype(np.float32)
    if input_np.max() > 1.0:
        input_np = (input_np - input_np.min()) / (input_np.max() - input_np.min() + 1e-8)

    # Handle activation map format
    if isinstance(activation_map, torch.Tensor):
        act_map = activation_map.detach().cpu().numpy()
    elif isinstance(activation_map, np.ndarray):
        act_map = activation_map
    else:
        raise ValueError(f"Unsupported activation_map type: {type(activation_map)}")

    # Ensure activation map is 2D
    if act_map.ndim == 3:
        act_map = act_map.squeeze(0) if act_map.shape[0] == 1 else act_map.mean(axis=0)
    elif act_map.ndim != 2:
        raise ValueError(f"Activation map must be 2D, got shape: {act_map.shape}")

    # Resize activation map to input image dimensions using bilinear interpolation
    act_map_resized = cv2.resize(act_map, (input_np.shape[1], input_np.shape[0]), interpolation=cv2.INTER_LINEAR)

    # Validate input parameters
    if percent_threshold is None and threshold is None:
        raise ValueError("Either percent_threshold or threshold must be provided")
    if percent_threshold is not None and threshold is not None:
        raise ValueError("Cannot specify both percent_threshold and threshold")

    # Calculate threshold based on input type
    if percent_threshold is not None:
        # Relative threshold: top X% of activations
        threshold_value = np.percentile(act_map_resized, 100-percent_threshold)
    else:
        # Absolute threshold: use provided value directly
        threshold_value = threshold

    # Create mask for high activation areas (activated = True, inactivated = False)
    activated_mask = act_map_resized > threshold_value
    inactivated_mask = ~activated_mask

    # Create overlay image starting with original
    overlay = input_np.copy()

    # Apply semi-transparent black mask to inactivated areas
    # Keep activated areas as original image
    overlay[inactivated_mask] = overlay[inactivated_mask] * 0.3  # Darken inactivated areas

    # Convert to uint8
    overlay = (overlay * 255).astype(np.uint8)

    return overlay


def calc_activated_filters(model, input_image, layer_name):
    """
    Get the activated filters of the CNN model.

    Args:
        model: CNN model
        input_image: Input tensor of shape (C, H, W) or (1, C, H, W), or numpy array (H, W, C)
        layer_name: Name of the layer to extract activations from (e.g., 'features.0')

    Returns:
        torch.Tensor: Activation maps of shape (num_filters, H', W')
    """
    import numpy as np

    # Convert numpy array to tensor if needed
    if isinstance(input_image, np.ndarray):
        # Convert HWC numpy array to CHW tensor
        if input_image.ndim == 3 and input_image.shape[-1] == 3:  # HWC
            input_image = torch.from_numpy(input_image).permute(2, 0, 1).float()  # HWC -> CHW
        elif input_image.ndim == 3 and input_image.shape[0] == 3:  # CHW
            input_image = torch.from_numpy(input_image).float()
        else:
            raise ValueError(f"Unsupported numpy array shape: {input_image.shape}")

    # Ensure input has batch dimension
    if input_image.dim() == 3:
        input_image = input_image.unsqueeze(0)

    activations = None

    def hook_fn(module, input, output):
        nonlocal activations
        activations = output.detach()

    # Register hook on the specified layer
    layer = dict(model.named_modules())[layer_name]
    hook = layer.register_forward_hook(hook_fn)

    # Forward pass
    with torch.no_grad():
        model(input_image)

    # Remove hook
    hook.remove()

    # Return activations (remove batch dimension if present)
    return activations.squeeze(0) if activations.dim() == 4 else activations

def visualize_activation_maximization(model, layer_name, filter_idx=None, neuron_idx=None,
                                   input_size=MODEL_INPUT_SIZE, iterations=100, lr=0.1):
    """
    Visualize activation maximization by generating and displaying the optimized image.

    Args:
        model: CNN model
        layer_name: Name of the layer to maximize (e.g., 'features.0')
        filter_idx: Index of filter to maximize (for conv layers)
        neuron_idx: Tuple (h, w) for specific neuron position (optional)
        input_size: Size of generated image (height, width)
        iterations: Number of optimization steps
        lr: Learning rate for gradient ascent

    Returns:
        numpy.ndarray: RGB image of the generated activation maximization result
    """
    import numpy as np

    # Generate the maximized activation image
    generated_tensor = calc_activation_maximization(
        model, layer_name, filter_idx, neuron_idx, input_size, iterations, lr
    )

    # Convert to numpy for visualization
    generated_np = generated_tensor.squeeze(0).numpy().transpose(1, 2, 0)  # CHW -> HWC

    # Normalize to [0, 1] for display
    generated_np = (generated_np - generated_np.min()) / (generated_np.max() - generated_np.min() + 1e-8)

    # Convert to RGB image (0-255)
    generated_rgb = (generated_np * 255).astype(np.uint8)

    return generated_rgb

def calc_activation_maximization(model, layer_name, filter_idx=None, neuron_idx=None,
                                input_size=MODEL_INPUT_SIZE, iterations=100, lr=0.1):
    """
    Calculate activation maximization for a specific filter or neuron.

    Args:
        model: CNN model
        layer_name: Name of the layer to maximize (e.g., 'features.0')
        filter_idx: Index of filter to maximize (for conv layers)
        neuron_idx: Tuple (h, w) for specific neuron position (optional)
        input_size: Size of generated image (height, width)
        iterations: Number of optimization steps
        lr: Learning rate for gradient ascent

    Returns:
        torch.Tensor: Generated image that maximizes the activation
    """
    import torch.nn.functional as F

    # Set model to eval mode
    model.eval()

    # Create random initial image
    generated = torch.randn(1, 3, input_size[0], input_size[1], requires_grad=True, device=next(model.parameters()).device)

    # Setup optimizer
    optimizer = torch.optim.Adam([generated], lr=lr)

    # Hook to capture activations
    activations = None
    def hook_fn(module, input, output):
        nonlocal activations
        activations = output

    # Register hook
    layer = dict(model.named_modules())[layer_name]
    hook = layer.register_forward_hook(hook_fn)

    for i in range(iterations):
        optimizer.zero_grad()

        # Forward pass
        _ = model(generated)

        # Get target activation
        if filter_idx is not None:
            if neuron_idx is not None:
                # Maximize specific neuron
                h, w = neuron_idx
                loss = -activations[0, filter_idx, h, w]
            else:
                # Maximize entire filter (sum of all activations)
                loss = -activations[0, filter_idx].sum()
        else:
            # Maximize all filters
            loss = -activations.sum()

        # Add regularization to make image look natural
        # L2 regularization to prevent large values
        reg_loss = 0.0001 * torch.norm(generated, 2)

        # Total variation regularization for smoothness
        tv_loss = 0.0001 * (
            torch.sum(torch.abs(generated[:, :, :, :-1] - generated[:, :, :, 1:])) +
            torch.sum(torch.abs(generated[:, :, :-1, :] - generated[:, :, 1:, :]))
        )

        total_loss = loss + reg_loss + tv_loss
        total_loss.backward()
        optimizer.step()

        # Clip values to reasonable range
        with torch.no_grad():
            generated.clamp_(-2, 2)

        # Blur occasionally to reduce noise
        if i % 20 == 0 and i > 0:
            with torch.no_grad():
                # Simple blur using average pooling
                generated.copy_(F.avg_pool2d(generated, 3, stride=1, padding=1))

        if (i + 1) % 50 == 0:
            print(".4f")

    # Remove hook
    hook.remove()

    # Return the generated image (detach and move to CPU)
    return generated.detach().cpu()


def calc_cam(model, input_image, layer_name=None, target_class=None):
    """
    Calculate the CAM of the CNN model.

    For CNN models with a classifier head, CAM is computed using the final convolutional
    layer and the weights of the first linear layer in the classifier.

    Args:
        model: CNN model (CNNClassifier)
        input_image: Input tensor of shape (C, H, W) or (1, C, H, W), or numpy array (H, W, C)
        layer_name: Name of the final convolutional layer (default: auto-detect last conv layer)
        target_class: Target class for CAM (default: 1 for positive class in binary classification)

    Returns:
        torch.Tensor: CAM heatmap of shape (H, W) where H, W are input image dimensions
    """
    import numpy as np

    # Convert numpy array to tensor if needed
    if isinstance(input_image, np.ndarray):
        # Convert HWC numpy array to CHW tensor
        if input_image.ndim == 3 and input_image.shape[-1] == 3:  # HWC
            input_image = torch.from_numpy(input_image).permute(2, 0, 1).float()  # HWC -> CHW
        elif input_image.ndim == 3 and input_image.shape[0] == 3:  # CHW
            input_image = torch.from_numpy(input_image).float()
        else:
            raise ValueError(f"Unsupported numpy array shape: {input_image.shape}")

    # Ensure input has batch dimension
    if input_image.dim() == 3:
        input_image = input_image.unsqueeze(0)

    # Auto-detect the last convolutional layer if not specified
    if layer_name is None:
        # Find the last Conv2d layer in features
        conv_layers = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                conv_layers.append(name)
        if conv_layers:
            layer_name = conv_layers[-1]  # Last conv layer
        else:
            raise ValueError("No convolutional layers found in model")

    # Get the final conv layer
    final_conv_layer = dict(model.named_modules())[layer_name]

    # Get the first linear layer weights after global average pooling
    # For this model, it's classifier.3 (Linear layer from 512 to hidden_size)
    first_linear_layer = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            first_linear_layer = module
            break

    if first_linear_layer is None:
        raise ValueError("No linear layer found in classifier")

    # For binary classification with single output
    if first_linear_layer.out_features == 1:
        # Single output binary classification (sigmoid)
        if target_class is None:
            target_class = 1  # Default to positive class

        if target_class == 1:
            # Positive class: use weights as-is (higher activation = more likely positive)
            weights = first_linear_layer.weight[0]  # Shape: (512,)
        elif target_class == 0:
            # Negative class: use negative weights (higher activation = more likely negative)
            weights = -first_linear_layer.weight[0]  # Shape: (512,)
        else:
            raise ValueError(f"For binary classification, target_class must be 0 or 1, got {target_class}")
    else:
        # Multi-class classification
        if target_class is None:
            target_class = 0
        if target_class >= first_linear_layer.out_features:
            raise ValueError(f"target_class {target_class} out of range for {first_linear_layer.out_features} classes")
        weights = first_linear_layer.weight[target_class]

    # Hook to capture feature maps
    feature_maps = None
    def hook_fn(module, input, output):
        nonlocal feature_maps
        feature_maps = output.detach()

    # Register hook
    hook = final_conv_layer.register_forward_hook(hook_fn)

    # Forward pass
    with torch.no_grad():
        model(input_image)

    # Remove hook
    hook.remove()

    # Squeeze batch dimension if present
    if feature_maps.dim() == 4:
        feature_maps = feature_maps.squeeze(0)  # Shape: (512, H', W')

    # Compute CAM: weighted sum of feature maps
    # weights shape: (512,), feature_maps shape: (512, H', W')
    cam = torch.sum(weights.unsqueeze(-1).unsqueeze(-1) * feature_maps, dim=0)  # Shape: (H', W')

    # For CAM, we don't apply ReLU here because negative weights are meaningful
    # (they show regions that contribute to negative class)
    # ReLU will be applied in visualization if needed

    return cam



def calc_gradcam(model, input_image, layer_name=None, target_class=None):
    """
    Calculate the Grad-CAM of the CNN model.
    Fixed version that works with sigmoid outputs by backpropagating on logits.

    Args:
        model: CNN model
        input_image: Input tensor of shape (C, H, W) or (1, C, H, W), or numpy array (H, W, C)
        layer_name: Name of the convolutional layer to get gradients from
        target_class: Target class index (default: 0 for binary classification)
    """
    import numpy as np

    # Convert numpy array to tensor if needed
    if isinstance(input_image, np.ndarray):
        # Convert HWC numpy array to CHW tensor
        if input_image.ndim == 3 and input_image.shape[-1] == 3:  # HWC
            input_image = torch.from_numpy(input_image).permute(2, 0, 1).float()  # HWC -> CHW
        elif input_image.ndim == 3 and input_image.shape[0] == 3:  # CHW
            input_image = torch.from_numpy(input_image).float()
        else:
            raise ValueError(f"Unsupported numpy array shape: {input_image.shape}")

    # Ensure input has batch dimension and requires gradients
    if input_image.dim() == 3:
        input_image = input_image.unsqueeze(0)

    # Enable gradient computation for input
    input_image.requires_grad_(True)

    # Auto-detect the last convolutional layer if not specified
    if layer_name is None:
        # Find the last Conv2d layer in features
        conv_layers = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                conv_layers.append(name)
        if conv_layers:
            layer_name = conv_layers[-1]  # Last conv layer
        else:
            raise ValueError("No convolutional layers found in model")

    # Use eval mode for consistency
    model.eval()

    # Ensure all layers are in eval mode for consistency
    for module in model.modules():
        module.eval()

    # Hook to capture feature maps and gradients
    feature_maps = None
    gradients = None

    def forward_hook(module, input, output):
        nonlocal feature_maps
        # Don't detach here - we need gradients to flow
        feature_maps = output

    def backward_hook(module, grad_input, grad_output):
        nonlocal gradients
        gradients = grad_output[0]

    # CRITICAL FIX: For sigmoid models, we need to backpropagate on logits, not sigmoid output
    # Hook the final Linear layer before Sigmoid to get logits
    logits = None
    def logits_hook(module, input, output):
        nonlocal logits
        logits = output

    # Register hooks
    target_layer = dict(model.named_modules())[layer_name]
    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_backward_hook(backward_hook)

    # Find the logits layer (last Linear before Sigmoid)
    logits_layer = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and 'classifier' in name:
            logits_layer = module  # This will be the last Linear layer

    if logits_layer is not None:
        logits_handle = logits_layer.register_forward_hook(logits_hook)

    # Forward pass
    output = model(input_image)

    # For binary classification, target class is usually 1 (positive class)
    if target_class is None:
        target_class = 1

    # CRITICAL FIX: Backpropagate on logits instead of sigmoid output
    model.zero_grad()
    if logits is not None:
        # For binary classification, logits has shape [batch, 1]
        # Use the single logit value regardless of target_class
        target = logits[0, 0]

        # For negative class, we want to maximize the negative logit (minimize positive logit)
        if target_class == 0:
            target = -target
    else:
        # Fallback to model output if logits hook failed
        target = output[0, 0]
        if target_class == 0:
            target = -target

    target.backward()

    # Remove hooks
    forward_handle.remove()
    backward_handle.remove()
    if logits_layer is not None and 'logits_handle' in locals():
        logits_handle.remove()

    # Squeeze batch dimension
    if feature_maps.dim() == 4:
        feature_maps = feature_maps.squeeze(0)  # Shape: (C, H, W)
    if gradients.dim() == 4:
        gradients = gradients.squeeze(0)  # Shape: (C, H, W)

    # Now we can detach for computation
    feature_maps = feature_maps.detach()
    gradients = gradients.detach()

    # Global average pool gradients across spatial dimensions to get weights
    weights = gradients.mean(dim=[1, 2])  # Shape: (C,)

    # Apply ReLU to weights (only positive gradients contribute)
    weights = torch.relu(weights)

    # Debug: Check if we have any non-zero weights
    if torch.sum(weights) == 0:
        print("Warning: All weights are zero after ReLU - Grad-CAM will be empty")
        print(f"Pre-ReLU weights stats: min={weights.min().item():.2e}, max={weights.max().item():.2e}")

    # Compute Grad-CAM: weighted sum of feature maps
    gradcam = torch.sum(weights.unsqueeze(-1).unsqueeze(-1) * feature_maps, dim=0)  # Shape: (H, W)

    # Apply ReLU to final result to focus on positive activations
    gradcam = torch.relu(gradcam)

    # Normalize to [0, 1] for visualization
    max_val = gradcam.max()
    if max_val > 0:
        gradcam = gradcam / max_val

    return gradcam

def calc_saliency_maps(model, input_image, target_class=None):
    """
    Calculate the Saliency Maps of the CNN model.
    Computes the gradient of the output w.r.t. input pixels.

    Args:
        model: CNN model
        input_image: Input tensor of shape (C, H, W) or (1, C, H, W), or numpy array (H, W, C)
        target_class: Target class for saliency (default: 0 for binary classification)

    Returns:
        torch.Tensor: Saliency map of shape (H, W) showing pixel importance
    """
    import numpy as np

    # Convert numpy array to tensor if needed
    if isinstance(input_image, np.ndarray):
        # Convert HWC numpy array to CHW tensor
        if input_image.ndim == 3 and input_image.shape[-1] == 3:  # HWC
            input_image = torch.from_numpy(input_image).permute(2, 0, 1).float()  # HWC -> CHW
        elif input_image.ndim == 3 and input_image.shape[0] == 3:  # CHW
            input_image = torch.from_numpy(input_image).float()
        else:
            raise ValueError(f"Unsupported numpy array shape: {input_image.shape}")
    elif isinstance(input_image, torch.Tensor):
        # Handle torch tensor input
        if input_image.dim() == 3:
            if input_image.shape[-1] == 3:  # HWC format
                input_image = input_image.permute(2, 0, 1)  # HWC -> CHW
            elif input_image.shape[0] == 3:  # Already CHW
                pass  # Keep as is
            else:
                raise ValueError(f"Unsupported torch tensor shape: {input_image.shape}")
        elif input_image.dim() == 4:  # Already has batch dimension
            if input_image.shape[1] == 3:  # NCHW
                pass  # Keep as is
            else:
                raise ValueError(f"Unsupported torch tensor shape: {input_image.shape}")
        else:
            raise ValueError(f"Unsupported torch tensor dimensions: {input_image.dim()}")

    # Ensure input has batch dimension and requires gradients
    if input_image.dim() == 3:
        input_image = input_image.unsqueeze(0)

    # Enable gradient computation for input
    input_image.requires_grad_(True)

    # Set model to eval mode
    model.eval()

    # Hook to capture logits before sigmoid (for gradient computation)
    logits = None
    def logits_hook(module, input, output):
        nonlocal logits
        logits = output

    # Find the logits layer (last Linear before Sigmoid)
    logits_layer = None
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and 'classifier' in name:
            logits_layer = module  # This will be the last Linear layer

    if logits_layer is not None:
        logits_handle = logits_layer.register_forward_hook(logits_hook)

    # Forward pass
    output = model(input_image)

    # For binary classification, target class is usually 0 (positive class)
    if target_class is None:
        target_class = 0

    # Backward pass to get gradients w.r.t. input
    # Use logits instead of sigmoid output to avoid gradient saturation
    model.zero_grad()
    if logits is not None:
        # Use logits for backpropagation (before sigmoid saturation)
        target = logits[0, 0]  # Single logit value for binary classification
    else:
        # Fallback to model output if logits hook failed
        target = output[0, target_class]

    target.backward()

    # Remove hooks
    if logits_layer is not None and 'logits_handle' in locals():
        logits_handle.remove()

    # Get the gradients w.r.t. input
    saliency = input_image.grad[0]  # Remove batch dimension

    # Take absolute value and sum across channels for visualization
    saliency = torch.abs(saliency).sum(dim=0)  # Shape: (H, W)

    return saliency

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from PIL import Image
    import numpy as np

    # Demonstrate Grad-CAM overlay workflow with real image
    model, _ = load_model("../detection_car/models/model_20251006_172724/cnn_best.pth")
    image = Image.open("../cardd_data/test_results/edited_1759739665412_6862.png").convert('RGB').resize((512, 512))
    # Preprocess image
    input_image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
    input_tensor = preprocess_image(input_image)
    # Calculate Grad-CAM
    gradcam_map = calc_gradcam(model, input_tensor, target_class=1)
    # Create overlay
    overlay = create_activation_overlay(input_image, gradcam_map, percent_threshold=10)
    
    # Save overlay
    plt.imshow(overlay); plt.title("Grad-CAM Overlay"); plt.axis('off')
    plt.savefig("gradcam_overlay_demo.png", bbox_inches='tight'); plt.close()
    print("Grad-CAM overlay saved as 'gradcam_overlay_demo.png'")