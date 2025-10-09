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
    
    return model

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

def activation_maximization_single_neuron(model, layer_name, neuron_index) -> dict:
    """
    Visualize the feature of the neuron in the layer.
    Input: model: torch.nn.Module, the CNN model.
    Input: layer_name: str, the name of the layer.
    Input: neuron_index: int, the index of the neuron.

    名称：Activation Maximization / Feature Visualization

    - 思路：找到能让某个 filter 激活最大的输入图像。
    - 方法：
        1. 随机初始化一张图像 x。
        2. 定义目标函数：最大化某个 filter 输出 F_i(x)。
        3. 通过梯度上升更新图像：
            
        x \leftarrow x + \eta \frac{\partial F_i(x)}{\partial x}
        
        4. 经过若干迭代后，得到一张激活该 filter 最强的“理想图像”。
    - 优点：能展示 filter 想要检测的模式，不局限于原始图像。
    - 论文示例：
        - Olah et al., “Feature Visualization” 系列（Distill.pub）
        - Zeiler & Fergus, 2014 (DeconvNet)
    """
    import torch
    import torch.nn.functional as F
    import numpy as np
    import matplotlib.pyplot as plt

    model.eval()
    device = next(model.parameters()).device

    # 1) 找到目标层，并注册前向hook以捕获激活
    target_activation = {"value": None}

    def forward_hook(module, inp, out):
        target_activation["value"] = out

    target_module = None
    for name, module in model.named_modules():
        if name == layer_name:
            target_module = module
            break
    if target_module is None:
        raise ValueError(f"Layer '{layer_name}' not found in model")

    handle = target_module.register_forward_hook(forward_hook)

    # 2) 初始化可学习的输入图像（从高斯噪声开始）
    img_size = 224
    x = torch.randn(1, 3, img_size, img_size, device=device, requires_grad=True)

    # 3) 定义超参数
    num_steps = 200
    step_size = 0.1
    l2_weight = 1e-4            # L2 正则，防止数值爆炸
    tv_weight = 1e-4            # Total Variation 正则，鼓励平滑

    def total_variation(image: torch.Tensor) -> torch.Tensor:
        # 简单的TV正则：相邻像素差的L1范数
        diff_x = image[:, :, 1:, :] - image[:, :, :-1, :]
        diff_y = image[:, :, :, 1:] - image[:, :, :, :-1]
        return (diff_x.abs().mean() + diff_y.abs().mean())

    optimizer = torch.optim.Adam([x], lr=step_size)

    history = []
    try:
        for t in range(num_steps):
            optimizer.zero_grad()
            out = model(x)
            act = target_activation["value"]  # Shape: (N, C, H, W)
            if act is None:
                raise RuntimeError("Activation hook did not capture output. Check layer_name.")
            if neuron_index < 0 or neuron_index >= act.shape[1]:
                raise ValueError(f"neuron_index {neuron_index} out of range for layer with {act.shape[1]} channels")

            # 目标：最大化指定通道的平均激活
            target = act[:, neuron_index, :, :].mean()

            # 正则项（负号是因为我们在做梯度上升，整体loss取负最大化目标）
            loss = -target + l2_weight * (x ** 2).mean() + tv_weight * total_variation(x)
            loss.backward()
            optimizer.step()

            # 约束到可视范围（不是必须，但更稳定）
            with torch.no_grad():
                x.clamp_(-2.5, 2.5)

            history.append({"step": t, "target": float(target.detach().cpu())})

    finally:
        handle.remove()

    # 将图像规范化到[0,1]以便可视化
    with torch.no_grad():
        img = x.detach().cpu()[0]
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)

    return {
        "optimized_image_chw": img.numpy(),
        "layer_name": layer_name,
        "neuron_index": neuron_index,
        "history": history,
        "params": {
            "num_steps": num_steps,
            "step_size": step_size,
            "l2_weight": l2_weight,
            "tv_weight": tv_weight,
            "init": "gaussian"
        }
    }

def visualize_activation_maximization_neurons(model, neuron_list: list) -> dict:
    """
    Visualize the activation maximization neurons.
    Input: neuron_list: list, the list of neurons.
    Output: visualization: dict, the visualization of the activation maximization neurons.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # Visualize in 2*2 grid
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))

    for i, (layer_name, neuron_index) in enumerate(neuron_list[:4]):
        result = activation_maximization_single_neuron(model, layer_name, neuron_index)
        img = result["optimized_image_chw"]
        axes[i // 2, i % 2].imshow(np.transpose(img, (1, 2, 0)))
        axes[i // 2, i % 2].axis('off')
        axes[i // 2, i % 2].set_title(f"{layer_name} [channel {neuron_index}]")
    plt.suptitle("Activation Maximization")
    plt.tight_layout()
    plt.show()

def visualize_activation_maps(model, input_image, layer_name, max_maps=16, save_path=None) -> dict:
    """
    Visualize the activation maps of the filters on a real input image.
    
    Args:
        model: Loaded CNN model
        input_image: Input image tensor (N, C, H, W) or numpy array
        layer_name: Name of the layer to visualize
        max_maps: Maximum number of activation maps to display
        save_path: Optional path to save the visualization
    
    Returns:
        dict: Contains activation maps and visualization data
    """
    import torch
    import torch.nn.functional as F
    import numpy as np
    import matplotlib.pyplot as plt
    
    model.eval()
    device = next(model.parameters()).device
    
    # Prepare input image
    if isinstance(input_image, np.ndarray):
        input_image = torch.from_numpy(input_image).float()
    
    if input_image.dim() == 3:
        input_image = input_image.unsqueeze(0)  # Add batch dimension
    
    input_image = input_image.to(device)
    
    # Hook to capture activations
    activations = {}
    def hook_fn(name):
        def hook(module, input, output):
            activations[name] = output.detach()
        return hook
    
    # Find target layer and register hook
    target_module = None
    for name, module in model.named_modules():
        if name == layer_name:
            target_module = module
            break
    
    if target_module is None:
        raise ValueError(f"Layer '{layer_name}' not found in model")
    
    handle = target_module.register_forward_hook(hook_fn(layer_name))
    
    # Forward pass
    with torch.no_grad():
        _ = model(input_image)
    
    # Get activation maps
    if layer_name not in activations:
        raise RuntimeError(f"Activation not captured for layer {layer_name}")
    
    activation_maps = activations[layer_name].cpu().numpy()  # Shape: (batch, channels, height, width)
    handle.remove()
    
    # Take first sample from batch
    maps = activation_maps[0]  # Shape: (channels, height, width)
    
    # Select maps to display
    num_maps = min(max_maps, maps.shape[0])
    
    # Create subplot grid
    cols = 4
    rows = (num_maps + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    # Visualize each activation map
    for i in range(num_maps):
        row = i // cols
        col = i % cols
        
        # Get activation map for this filter
        activation = maps[i]
        
        # Normalize to [0, 1] for visualization
        activation_norm = (activation - activation.min()) / (activation.max() - activation.min() + 1e-8)
        
        # Display as heatmap
        im = axes[row, col].imshow(activation_norm, cmap='hot', interpolation='nearest')
        axes[row, col].set_title(f'Filter {i}')
        axes[row, col].axis('off')
        
        # Add colorbar for reference
        plt.colorbar(im, ax=axes[row, col], fraction=0.046, pad=0.04)
    
    # Hide unused subplots
    for i in range(num_maps, rows * cols):
        row = i // cols
        col = i % cols
        axes[row, col].axis('off')
    
    plt.suptitle(f'Activation Maps from {layer_name}\n{len(maps)} activation maps')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Activation maps visualization saved to {save_path}")
    
    plt.show()
    
    return {
        "activation_maps": activation_maps,
        "layer_name": layer_name,
        "input_shape": input_image.shape,
        "output_shape": activation_maps.shape,
        "visualization_data": {
            "num_maps_displayed": num_maps,
            "maps_shape": maps.shape
        }
    }


def visualize_activation_maps_with_input(model, input_image, layer_name, max_maps=16, save_path=None) -> dict:
    """
    Visualize activation maps alongside the input image for better context.
    
    Args:
        model: Loaded CNN model
        input_image: Input image tensor or numpy array
        layer_name: Name of the layer to visualize
        max_maps: Maximum number of activation maps to display
        save_path: Optional path to save the visualization
    
    Returns:
        dict: Contains activation maps and visualization data
    """
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Get activation maps
    result = visualize_activation_maps(model, input_image, layer_name, max_maps, save_path)
    
    # Also show the input image for context
    if isinstance(input_image, torch.Tensor):
        input_img = input_image.cpu().numpy()
    else:
        input_img = input_image
    
    # Normalize input image for display
    if input_img.ndim == 4:
        input_img = input_img[0]  # Take first sample
    
    # Convert from CHW to HWC for display
    if input_img.shape[0] in [1, 3]:
        input_img = np.transpose(input_img, (1, 2, 0))
    
    # Normalize to [0, 1]
    input_img = (input_img - input_img.min()) / (input_img.max() - input_img.min() + 1e-8)
    
    # Create a combined visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Show input image
    axes[0].imshow(input_img)
    axes[0].set_title('Input Image')
    axes[0].axis('off')
    
    # Show activation maps summary (average across all maps)
    activation_maps = result["activation_maps"][0]  # Shape: (channels, height, width)
    avg_activation = np.mean(activation_maps, axis=0)
    avg_activation_norm = (avg_activation - avg_activation.min()) / (avg_activation.max() - avg_activation.min() + 1e-8)
    
    im = axes[1].imshow(avg_activation_norm, cmap='hot', interpolation='nearest')
    axes[1].set_title(f'Average Activation from {layer_name}')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    
    if save_path:
        combined_path = save_path.replace('.png', '_combined.png')
        plt.savefig(combined_path, dpi=150, bbox_inches='tight')
        print(f"Combined visualization saved to {combined_path}")
    
    plt.show()
    
    return result

if __name__ == "__main__":
    model_path = "../detection_car/models/model_20251006_172724/cnn_best.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path)

    # Test 1: Output the structure of the CNN model.
    output_model_structure(model)

    # Test 2: Get the filters and activation maps of the CNN model.
    # result = activation_maximization_single_neuron(model, "features.16", 2)
    # print(result)

    # Test 3: Visualize the activation maximization neurons.
    # neuron_list = [("features.16", 2), ("features.16", 3), ("features.16", 4), ("features.16", 5)]
    # visualize_activation_maximization_neurons(model, neuron_list)   

    # Test 4: Visualize the activation maps with the input image.
    input_image = cv2.imread("/Users/wjs/Library/CloudStorage/OneDrive-Personal/Coding, ML & DL/ResponsibleAI/cardd_data/test_results/edited_1759739665412_6862.png")
    input_image = cv2.resize(input_image, (224, 224))
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    input_image = torch.from_numpy(input_image).float()
    input_image = input_image.unsqueeze(0)
    input_image = input_image.to(device)
    visualize_activation_maps_with_input(model, input_image, "features.0")