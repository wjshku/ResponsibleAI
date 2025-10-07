#!/usr/bin/env python3
"""
Model interpretation tools for CNN models:
 - Activation maps (intermediate feature maps)
 - Grad-CAM and saliency maps
 - Fake vs Real visualization with predicted probabilities
 - Embedding visualization (PCA / t-SNE) from penultimate layer
 - (Optional) filter visualization

Usage:
  python interpret.py
Follow prompts to select a model directory and options.
"""

import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import cv2

from data_loader import CarScratchDataset, create_dataloader, get_eval_transforms
from models import get_model

# -------------------- Utilities --------------------

def _load_metadata(model_dir: Path):
    mp = model_dir / "metadata.json"
    if not mp.exists():
        raise FileNotFoundError(f"metadata.json not found in {model_dir}")
    with open(mp, 'r') as f:
        return json.load(f)

def _pick_model_dir(base: Path = Path("models")) -> Path:
    if not base.exists():
        raise FileNotFoundError(f"Models base not found: {base}")
    dirs = [d for d in base.iterdir() if d.is_dir() and d.name.startswith("model_")]
    dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    if not dirs:
        raise FileNotFoundError("No model_* dirs found")
    print("Select a model directory:")
    for i, d in enumerate(dirs[:30]):
        try:
            md = _load_metadata(d)
            ts = md.get('timestamp', '')
            acc = md.get('results', {}).get('best_test_acc', None)
            acc_str = f", best_acc={acc:.2f}%" if isinstance(acc, (int, float)) else ""
            print(f"  [{i}] {d.name} (ts={ts}{acc_str})")
        except Exception:
            print(f"  [{i}] {d.name}")
    sel = input(f"Enter index [0-{min(29, len(dirs)-1)}]: ").strip()
    idx = int(sel)
    return dirs[idx]

def _build_model_from_metadata(metadata: dict, device: torch.device):
    cfg = metadata.get('config', {})
    model_name = cfg.get('model_name', 'cnn')
    target_size = tuple(cfg.get('target_size', [224, 224]))
    hidden_size = int(cfg.get('hidden_size', 256))
    dropout = float(cfg.get('dropout', 0.3))
    if model_name == 'vanilla':
        input_size = target_size[0] * target_size[1] * 3
        model = get_model(model_name='vanilla', input_size=input_size, hidden_size=hidden_size, dropout=dropout, target_size=target_size)
    else:
        model = get_model(model_name='cnn', hidden_size=hidden_size, dropout=dropout)
    return model.to(device), model_name, target_size

def _load_weights(model_dir: Path, model: nn.Module, model_name: str, device: torch.device, prefer_best: bool = True):
    md = _load_metadata(model_dir)
    path = None
    if prefer_best and md.get('best_model_file'):
        cand = model_dir / md['best_model_file']
        if cand.exists():
            path = cand
    if path is None:
        default = model_dir / f"{model_name}_model.pth"
        if default.exists():
            path = default
    if path is None:
        # try checkpoints
        if prefer_best and md.get('best_checkpoint_file'):
            ck = model_dir / md['best_checkpoint_file']
            if ck.exists():
                state = torch.load(ck, map_location=device)
                model.load_state_dict(state['model_state_dict'])
                return
        ck = model_dir / f"{model_name}_checkpoint.pth"
        if ck.exists():
            state = torch.load(ck, map_location=device)
            model.load_state_dict(state['model_state_dict'])
            return
        # fallback any pth
        pths = list(model_dir.glob("*.pth"))
        if not pths:
            raise FileNotFoundError(f"No weights found in {model_dir}")
        path = pths[0]
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)


# -------------------- Activation Maps --------------------

def get_intermediate_activations(model: nn.Module, x: torch.Tensor, layers: List[str]) -> dict:
    hooks = {}
    outputs = {}
    def make_hook(name):
        def hook(_m, _i, o):
            outputs[name] = o.detach().cpu()
        return hook
    # Register on named modules
    for name, m in model.named_modules():
        if name in layers:
            h = m.register_forward_hook(make_hook(name))
            hooks[name] = h
    with torch.no_grad():
        _ = model(x)
    for h in hooks.values():
        h.remove()
    return outputs


# -------------------- Grad-CAM --------------------

class GradCAM:
    def __init__(self, model: nn.Module, target_module_name: str):
        self.model = model
        self.target_module_name = target_module_name
        self.activations = None
        self.gradients = None
        self._register()

    def _register(self):
        for name, m in self.model.named_modules():
            if name == self.target_module_name:
                m.register_forward_hook(self._forward_hook)
                m.register_full_backward_hook(self._backward_hook)
                break

    def _forward_hook(self, _m, _i, o):
        self.activations = o.detach()

    def _backward_hook(self, _m, gin, gout):
        self.gradients = gout[0].detach()

    def __call__(self, x: torch.Tensor, target_index: int = 0) -> torch.Tensor:
        self.model.zero_grad()
        y = self.model(x)
        if y.ndim > 1:
            y = y[:, target_index]
        y = y.sum()
        y.backward()
        # Global-average pool gradients
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (N,C,1,1)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (N,1,H,W)
        cam = F.relu(cam)
        # Normalize per image
        b, _, h, w = cam.shape
        cam = cam.view(b, -1)
        cam = cam / (cam.max(dim=1, keepdim=True).values + 1e-8)
        cam = cam.view(b, 1, h, w)
        return cam.detach().cpu()


# -------------------- Embeddings --------------------

def extract_embeddings(model: nn.Module, loader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    feats: List[np.ndarray] = []
    labels: List[np.ndarray] = []

    # Capture the input to the last Linear layer (penultimate representation)
    last_linear = None
    for m in model.modules():
        if isinstance(m, nn.Linear):
            last_linear = m
    captured = {'x': None}

    def hook_in(m, inputs, output):
        captured['x'] = inputs[0].detach()

    handle = last_linear.register_forward_hook(hook_in) if last_linear is not None else None

    with torch.no_grad():
        for images, y in loader:
            images = images.to(device)
            _ = model(images)
            rep = captured['x']
            if rep is not None:
                feats.append(rep.detach().cpu().numpy())
                labels.append(y.numpy())

    if handle is not None:
        handle.remove()

    if feats:
        X = np.concatenate(feats, axis=0)
        Y = np.concatenate(labels, axis=0)
    else:
        X = np.zeros((0, 2))
        Y = np.zeros((0,))
    return X, Y


# -------------------- Visualization helpers --------------------

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)

def _tensor_to_img(t: torch.Tensor) -> np.ndarray:
    # Expect (C,H,W) normalized
    a = t.detach().cpu().numpy()
    if a.ndim == 3 and a.shape[0] in (1, 3):
        if a.shape[0] == 3:
            a = a * IMAGENET_STD + IMAGENET_MEAN
        a = np.transpose(a, (1, 2, 0))
    a = np.clip(a, 0, 1)
    return (a * 255).astype(np.uint8)

def plot_gradcam_grid(images: torch.Tensor, cams: torch.Tensor, save_path: Path, max_items: int = 12, alpha: float = 0.45, cols: int = 4):
    n = min(max_items, images.shape[0])
    rows = int(np.ceil(n / cols))
    plt.figure(figsize=(3 * cols, 3 * rows))
    for i in range(n):
        img = _tensor_to_img(images[i])
        cam = cams[i, 0:1, :, :].detach().cpu().numpy()
        cam = cam.squeeze()
        # Resize cam to image size
        from PIL import Image as _PILImage
        cam_img = _PILImage.fromarray((cam * 255).astype(np.uint8)).resize((img.shape[1], img.shape[0]))
        cam_arr = np.array(cam_img).astype(np.float32) / 255.0
        # Apply colormap and blend over grayscale base to improve contrast
        import matplotlib.cm as cm
        heat = cm.jet(cam_arr)[..., :3]
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray3 = np.stack([gray, gray, gray], axis=2).astype(np.float32) / 255.0
        over = (1 - alpha) * gray3 + alpha * heat
        # Apply colormap and blend over grayscale base to improve contrast
        import matplotlib.cm as cm
        heat = cm.jet(cam_arr)[..., :3]
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray3 = np.stack([gray, gray, gray], axis=2).astype(np.float32) / 255.0
        # Apply heat only where CAM is strong
        over = gray3.copy()
        mask = cam_arr >= 0.4
        over[mask] = (1 - alpha) * gray3[mask] + alpha * heat[mask]
        heat = cm.jet(cam_arr)[..., :3]
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray3 = np.stack([gray, gray, gray], axis=2).astype(np.float32) / 255.0
        # Apply heat only where CAM is strong
        over = gray3.copy()
        mask = cam_arr >= 0.4
        over[mask] = (1 - alpha) * gray3[mask] + alpha * heat[mask]
        over = np.clip(over, 0, 1)
        r = i // cols
        c = i % cols
        plt.subplot(rows, cols, i + 1)
        plt.imshow(over)
        plt.axis('off')
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def plot_gradcam_with_mask_grid(images: torch.Tensor, cams: torch.Tensor, masks: List[np.ndarray], save_path: Path, max_items: int = 12, alpha: float = 0.45, cols: int = 4):
    n = min(max_items, images.shape[0], len(masks))
    rows = int(np.ceil(n / cols))
    plt.figure(figsize=(3 * cols, 3 * rows))
    for i in range(n):
        img = _tensor_to_img(images[i])
        cam = cams[i, 0:1, :, :].detach().cpu().numpy().squeeze()
        # Resize cam and mask to img size
        from PIL import Image as _PILImage
        cam_img = _PILImage.fromarray((cam * 255).astype(np.uint8)).resize((img.shape[1], img.shape[0]))
        cam_arr = np.array(cam_img).astype(np.float32) / 255.0
        m = masks[i]
        if m.ndim == 3:
            m = m.squeeze()
        mask_resized = cv2.resize(m.astype(np.uint8), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        # Build overlay
        import matplotlib.cm as cm
        heat = cm.jet(cam_arr)[..., :3]
        over = (1 - alpha) * (img.astype(np.float32) / 255.0) + alpha * heat
        over = np.clip(over, 0, 1)
        # Draw mask contour in red
        contours, _ = cv2.findContours((mask_resized > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        overlay = (over * 255).astype(np.uint8)
        cv2.drawContours(overlay, contours, -1, (255, 0, 0), thickness=2)
        plt.subplot(rows, cols, i + 1)
        plt.imshow(overlay)
        plt.axis('off')
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def plot_image_grid(images: torch.Tensor, save_path: Path, max_items: int = 12, cols: int = 4):
    n = min(max_items, images.shape[0])
    rows = int(np.ceil(n / cols))
    plt.figure(figsize=(3 * cols, 3 * rows))
    for i in range(n):
        img = _tensor_to_img(images[i])
        plt.subplot(rows, cols, i + 1)
        plt.imshow(img)
        plt.axis('off')
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def plot_tsne_embeddings(X: np.ndarray, Y: np.ndarray, save_path: Path, title: str = "t-SNE of embeddings"):
    n, d = X.shape
    if n < 5 or d < 2:
        print("Not enough data/dimensions for t-SNE; need at least 5 samples and 2D features.")
        return
    # Optional PCA to <=50 dims and also ensure d >= 2
    Xr = X
    n_components = min(50, d, n - 1)
    if n_components >= 2 and d > n_components:
        pca = PCA(n_components=n_components)
        Xr = pca.fit_transform(X)
    tsne = TSNE(n_components=2, perplexity=min(30, max(5, n // 5)), learning_rate='auto', init='pca', random_state=42)
    Z = tsne.fit_transform(Xr)
    # Plot
    plt.figure(figsize=(6, 5))
    Yb = (Y > 0.5).astype(int)
    colors = np.array([[0.2, 0.6, 1.0], [1.0, 0.4, 0.2]])
    for cls in (0, 1):
        mask = (Yb == cls)
        if np.any(mask):
            plt.scatter(Z[mask, 0], Z[mask, 1], s=18, c=[colors[cls]], label=f"{'REAL' if cls==0 else 'FAKE'}", alpha=0.8)
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    plt.close()


# -------------------- Main demo flow --------------------

def main():
    base = Path("models")
    model_dir = _pick_model_dir(base)
    md = _load_metadata(model_dir)

    # Device
    dev = input("Device [auto|cpu|cuda|mps]: ").strip().lower()
    if dev in ("cpu", "cuda", "mps"):
        device = torch.device(dev)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Build model and load weights
    model, model_name, target_size = _build_model_from_metadata(md, device)
    _load_weights(model_dir, model, model_name, device, prefer_best=True)
    model.eval()

    # Data (small batch preview loader)
    data_dir = md['config']['data_dir']
    metadata_dir = md['config']['metadata_dir']
    eval_tf = get_eval_transforms(target_size=tuple(target_size))
    ds = CarScratchDataset.load_binary_dataset(
        data_dir=data_dir,
        metadata_dir=metadata_dir,
        sample_size=64,
        shuffle=False,
        transform=eval_tf,
    )
    loader = create_dataloader(ds, batch_size=8, shuffle=False, target_size=tuple(target_size))

    # One batch from binary loader
    images, labels = next(iter(loader))
    images = images.to(device)

    # Activation maps (pick last conv block)
    target_layer = 'features.16'  # close to final conv (CNNClassifier)
    acts = get_intermediate_activations(model, images, layers=[target_layer])
    print(f"Captured activations: {list(acts.keys())}")

    # Grad-CAM
    cam = GradCAM(model, target_module_name=target_layer)
    cams = cam(images)  # (B,1,H,W) normalized
    print(f"Grad-CAM shape: {tuple(cams.shape)}")
    # Build masked Grad-CAM grids: separate REAL (original) and FAKE (processed), up to 12 each
    try:
        base_ds = CarScratchDataset(
            data_dir=data_dir,
            metadata_dir=metadata_dir,
            sample_size=None,
            transform=None,
            load_masks=True,
            load_processed=True,
        )
        real_imgs, real_masks = [], []
        fake_imgs, fake_masks = [], []
        for k in range(len(base_ds)):
            if len(real_imgs) >= 12 and len(fake_imgs) >= 12:
                break
            item = base_ds[k]
            m = item.get('mask')
            if m is None:
                continue
            if len(real_imgs) < 12 and item.get('original_image') is not None:
                real_imgs.append(eval_tf(item['original_image']))
                real_masks.append(m)
            if len(fake_imgs) < 12 and item.get('processed_image') is not None:
                fake_imgs.append(eval_tf(item['processed_image']))
                fake_masks.append(m)
        if real_imgs:
            r_tensor = torch.stack(real_imgs, dim=0).to(device)
            r_cams = cam(r_tensor)
            plot_gradcam_with_mask_grid(r_tensor, r_cams, real_masks, save_path=model_dir / "gradcam_mask_grid_real.png", max_items=12, cols=4)
            print(f"Saved Grad-CAM+Mask (REAL) grid to: {model_dir / 'gradcam_mask_grid_real.png'}")
            # Also plain real images grid
            plot_image_grid(r_tensor, save_path=model_dir / "real_grid.png", max_items=12, cols=4)
            print(f"Saved REAL images grid to: {model_dir / 'real_grid.png'}")
        else:
            print("No REAL images with masks found for Grad-CAM.")
        if fake_imgs:
            f_tensor = torch.stack(fake_imgs, dim=0).to(device)
            f_cams = cam(f_tensor)
            plot_gradcam_with_mask_grid(f_tensor, f_cams, fake_masks, save_path=model_dir / "gradcam_mask_grid_fake.png", max_items=12, cols=4)
            print(f"Saved Grad-CAM+Mask (FAKE) grid to: {model_dir / 'gradcam_mask_grid_fake.png'}")
            # Also plain fake images grid
            plot_image_grid(f_tensor, save_path=model_dir / "fake_grid.png", max_items=12, cols=4)
            print(f"Saved FAKE images grid to: {model_dir / 'fake_grid.png'}")
        else:
            print("No FAKE images with masks found for Grad-CAM.")
    except Exception as e:
        print(f"Failed to generate Grad-CAM+Mask grids: {e}")

    # Embeddings
    # Rebuild a full-dataset loader for embeddings (t-SNE over full set)
    full_ds = CarScratchDataset.load_binary_dataset(
        data_dir=data_dir,
        metadata_dir=metadata_dir,
        sample_size=None,
        shuffle=False,
        transform=eval_tf,
    )
    full_loader = create_dataloader(full_ds, batch_size=64, shuffle=False, target_size=tuple(target_size))
    X, Y = extract_embeddings(model, full_loader, device)
    print(f"Embeddings: {X.shape}, Labels: {Y.shape}")
    try:
        plot_tsne_embeddings(X, Y, save_path=model_dir / "embeddings_tsne.png")
        print(f"Saved t-SNE plot to: {model_dir / 'embeddings_tsne.png'}")
    except Exception as e:
        print(f"Failed to save t-SNE plot: {e}")

    print("Interpretation demo complete. Hook this file to your visualization routines (matplotlib) for plots.")


if __name__ == "__main__":
    main()


