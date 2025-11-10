#!/usr/bin/env python3
"""
Extract pooled CNN/DANN features from cardd_data GenAI_Results and run t-SNE.

- Sources models from:
  - simple_detect_car (CNN baseline)
  - domain_adapt (DANN)

- Uses the SAME eval preprocess as simple_detect_car.eval (get_eval_transforms)

Outputs:
  - .npz file with features, labels, domains, image paths
  - optional PNG scatter for t-SNE
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

try:
    from tqdm.auto import tqdm  # type: ignore
except Exception:
    tqdm = None  # Fallback when tqdm is unavailable

# Optional rich CLI support
try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.completion import Completer, Completion
    PROMPT_AVAILABLE = True
except Exception:
    PROMPT_AVAILABLE = False

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SIMPLE_DETECT_DIR = PROJECT_ROOT / "simple_detect_car"
DOMAIN_ADAPT_DIR = PROJECT_ROOT / "domain_adapt"

# Ensure imports work regardless of CWD
import sys  # noqa: E402
if str(SIMPLE_DETECT_DIR) not in sys.path:
    sys.path.insert(0, str(SIMPLE_DETECT_DIR))
if str(DOMAIN_ADAPT_DIR) not in sys.path:
    sys.path.insert(0, str(DOMAIN_ADAPT_DIR))

from data_loader import CarScratchDataset, create_dataloader, get_eval_transforms  # type: ignore  # noqa: E402
from models import CNNClassifier, get_model  # type: ignore  # noqa: E402
from model_dann import get_dann_model  # type: ignore  # noqa: E402


def find_latest_dir(base: Path, prefix: str) -> Optional[Path]:
    if not base.exists():
        return None
    dirs = [d for d in base.iterdir() if d.is_dir() and d.name.startswith(prefix)]
    if not dirs:
        return None
    dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return dirs[0]


def dir_has_cnn_weights(d: Path) -> bool:
    # direct weights
    direct = ["cnn_best.pth", "cnn_model.pth"]
    for name in direct:
        if (d / name).exists():
            return True
    # checkpoints
    ckpts = ["cnn_best_checkpoint.pth", "cnn_checkpoint.pth"]
    for name in ckpts:
        if (d / name).exists():
            return True
    # metadata-specified
    meta = d / "metadata.json"
    if meta.exists():
        try:
            m = json.loads(meta.read_text())
            for key in ("best_model_file", "best_checkpoint_file"):
                v = m.get(key)
                if v and (d / v).exists():
                    return True
        except Exception:
            pass
    # any .pth
    if list(d.glob("*.pth")):
        return True
    return False


def dir_has_dann_weights(d: Path) -> bool:
    for name in ["dann_best.pth", "dann_final.pth"]:
        if (d / name).exists():
            return True
    if list(d.glob("*.pth")):
        return True
    return False


def _extract_ts_from_name(name: str, prefix: str) -> Optional[int]:
    try:
        # Expect names like model_YYYYMMDD_HHMMSS or model_dann_YYYYMMDD_HHMMSS
        ts_part = name.replace(prefix, "")
        # Remove leading underscores
        while ts_part.startswith("_"):
            ts_part = ts_part[1:]
        # Split to date_time
        parts = ts_part.split("_")
        if len(parts) >= 2:
            dt = parts[0] + parts[1]  # YYYYMMDDHHMMSS
            return int(dt)
    except Exception:
        return None
    return None


def _list_model_dirs(base_dir: Path, prefix: str = "model_"):
    """List model directories with given prefix, sorted by modification time."""
    if not base_dir.exists():
        return []
    dirs = [d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith(prefix)]
    dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return dirs


def _describe_model_dir(d: Path):
    """Describe a model directory with metadata if available."""
    meta = d / "metadata.json"
    if meta.exists():
        try:
            with open(meta, 'r') as f:
                m = json.load(f)
            ts = m.get('timestamp', '')
            acc = m.get('results', {}).get('best_test_acc', None)
            acc_str = f", best_acc={acc:.2f}%" if isinstance(acc, (int, float)) else ""
            return f"{d.name} (ts={ts}{acc_str})"
        except Exception:
            pass
    try:
        from datetime import datetime
        mt = datetime.fromtimestamp(d.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')
    except Exception:
        mt = "?"
    return f"{d.name} (modified {mt})"


def _interactive_select_model_dir(base_dir: Path, prefix: str, model_type: str, has_weights_fn, max_list: int = 30) -> Optional[Path]:
    """Interactively select a model directory from a list."""
    candidates = _list_model_dirs(base_dir, prefix=prefix)
    if not candidates:
        print(f"No {model_type} model directories found under: {base_dir}")
        return None
    
    # Filter to only directories with weights
    candidates_with_weights = [d for d in candidates if has_weights_fn(d)]
    if not candidates_with_weights:
        print(f"No {model_type} models with weights found under: {base_dir}")
        return None
    
    to_show = candidates_with_weights[:max(1, max_list)]
    print(f"\nSelect a {model_type} model directory:")
    for i, d in enumerate(to_show):
        print(f"  [{i}] {_describe_model_dir(d)}")
    
    model_dir = None
    # If prompt_toolkit is available, offer name-based selection with completion
    if PROMPT_AVAILABLE:
        class DirCompleter(Completer):
            def get_completions(self, document, complete_event):
                text = document.text.lower()
                for d in to_show:
                    name = d.name
                    if not text or text in name.lower():
                        yield Completion(name, start_position=-len(document.text))
        try:
            session = PromptSession()
            typed = session.prompt(f"Type {model_type} model dir name (or press Enter to choose by index): ", completer=DirCompleter())
            typed = typed.strip()
            if typed:
                matches = [d for d in to_show if d.name == typed]
                if matches:
                    model_dir = matches[0]
        except Exception:
            pass
    
    if model_dir is None:
        sel = input(f"Enter index [0-{len(to_show)-1}]: ").strip()
        try:
            idx = int(sel)
        except Exception:
            raise ValueError("Invalid selection. Expected an integer index.")
        if not (0 <= idx < len(to_show)):
            raise ValueError("Selection index out of range.")
        model_dir = to_show[idx]
    
    return model_dir


def find_latest_with_weights(base: Path, prefix: str, has_weights_fn) -> Optional[Path]:
    if not base.exists():
        return None
    dirs = [d for d in base.iterdir() if d.is_dir() and d.name.startswith(prefix)]
    if not dirs:
        return None
    # Sort by timestamp parsed from name; fallback to mtime if parse fails
    def sort_key(p: Path):
        ts = _extract_ts_from_name(p.name, prefix)
        return (ts if ts is not None else 0, p.stat().st_mtime)
    dirs.sort(key=sort_key, reverse=True)
    for d in dirs:
        if has_weights_fn(d):
            return d
    return None


def load_cnn_model(model_dir: Path, device: torch.device) -> Tuple[nn.Module, Tuple[int, int], int, float, dict]:
    """Restore CNN model and return (model, target_size, hidden_size, dropout, weight_info)."""
    metadata_file = model_dir / "metadata.json"
    target_size = (224, 224)
    hidden_size = 256
    dropout = 0.3
    if metadata_file.exists():
        try:
            meta = json.loads(metadata_file.read_text())
            cfg = meta.get("config", {})
            if cfg.get("target_size"):
                ts = cfg.get("target_size")
                if isinstance(ts, (list, tuple)) and len(ts) == 2:
                    target_size = (int(ts[0]), int(ts[1]))
            hidden_size = int(cfg.get("hidden_size", hidden_size))
            dropout = float(cfg.get("dropout", dropout))
        except Exception:
            pass

    model = get_model(model_name="cnn", dropout=dropout, hidden_size=hidden_size)

    # Prefer best weights if present
    # Build ordered candidate list with preference: best -> best_checkpoint -> final -> checkpoint -> any .pth
    candidates: List[Tuple[str, Path]] = []
    for name in ["cnn_best.pth", "best_model_file", "cnn_best_checkpoint.pth", "cnn_model.pth", "cnn_checkpoint.pth"]:
        if name == "best_model_file":
            # try metadata-specified best file
            try:
                meta = json.loads((model_dir / "metadata.json").read_text())
                if meta.get("best_model_file"):
                    candidates.append(("best", model_dir / meta["best_model_file"]))
            except Exception:
                pass
        else:
            # map type by filename
            t = "best" if name == "cnn_best.pth" else (
                "best_checkpoint" if name == "cnn_best_checkpoint.pth" else (
                "final" if name == "cnn_model.pth" else (
                "checkpoint" if name == "cnn_checkpoint.pth" else "unknown")))
            candidates.append((t, model_dir / name))

    weight_path: Optional[Path] = None
    weight_type: str = "unknown"
    for t, p in candidates:
        if p.exists():
            weight_path = p
            weight_type = t
            break

    checkpoint_loaded = False
    if weight_path is None:
        # Fallback: any direct .pth state-dict
        pths = list(model_dir.glob("*.pth"))
        # Prefer explicit model files over checkpoints for direct load
        direct_state = [p for p in pths if "checkpoint" not in p.name]
        if direct_state:
            weight_path = direct_state[0]
            weight_type = "unknown"
        else:
            # Try checkpoints (prefer best)
            best_ckpt = model_dir / "cnn_best_checkpoint.pth"
            ckpt = model_dir / "cnn_checkpoint.pth"
            ckpt_path = best_ckpt if best_ckpt.exists() else (ckpt if ckpt.exists() else None)
            if ckpt_path is None:
                # metadata-specified best checkpoint
                try:
                    meta = json.loads((model_dir / "metadata.json").read_text())
                    if meta.get("best_checkpoint_file"):
                        cand = model_dir / meta["best_checkpoint_file"]
                        if cand.exists():
                            ckpt_path = cand
                except Exception:
                    pass
            if ckpt_path is None:
                raise FileNotFoundError(f"No weights or checkpoints found in {model_dir}")
            state = torch.load(ckpt_path, map_location=device)
            if isinstance(state, dict) and "model_state_dict" in state:
                model.load_state_dict(state["model_state_dict"])
                checkpoint_loaded = True
                weight_path = ckpt_path
                weight_type = "best_checkpoint" if ckpt_path.name == "cnn_best_checkpoint.pth" else "checkpoint"
            else:
                # Fallback: try to load as plain state dict
                model.load_state_dict(state)
                checkpoint_loaded = True
                weight_path = ckpt_path
                weight_type = "checkpoint"

    if not checkpoint_loaded:
        state = torch.load(weight_path, map_location=device)
        model.load_state_dict(state)
    
    model.to(device)
    model.eval()
    return model, target_size, hidden_size, dropout, {"file": str(weight_path) if weight_path else None, "type": weight_type}


def load_dann_model(model_dir: Path, device: torch.device) -> Tuple[nn.Module, dict]:
    """Restore DANN model and return (model, weight_info)."""
    metadata_file = model_dir / "metadata.json"
    feature_hidden_size = 256
    domain_hidden_size = 256
    dropout = 0.3
    if metadata_file.exists():
        try:
            meta = json.loads(metadata_file.read_text())
            cfg = meta.get("config", {})
            feature_hidden_size = int(cfg.get("feature_hidden_size", feature_hidden_size))
            domain_hidden_size = int(cfg.get("domain_hidden_size", domain_hidden_size))
            dropout = float(cfg.get("dropout", dropout))
        except Exception:
            pass

    model = get_dann_model(
        input_channels=3,
        num_classes=1,
        feature_hidden_size=feature_hidden_size,
        domain_hidden_size=domain_hidden_size,
        dropout=dropout,
    )

    # Prefer best weights
    candidates = [("best", model_dir / "dann_best.pth"), ("final", model_dir / "dann_final.pth")]
    weight_type = "unknown"
    weight_path = None
    for t, p in candidates:
        if p.exists():
            weight_path = p
            weight_type = t
            break
    if weight_path is None:
        # fallback: any .pth
        pths = list(model_dir.glob("*.pth"))
        if not pths:
            raise FileNotFoundError(f"No DANN weights found in {model_dir}")
        weight_path = pths[0]
        weight_type = "unknown"

    state = torch.load(weight_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, {"file": str(weight_path), "type": weight_type}


@torch.inference_mode()
def extract_features(
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    cnn: Optional[CNNClassifier] = None,
    dann: Optional[nn.Module] = None,
    cnn_hidden_size: Optional[int] = None,
    dann_feature_hidden_size: Optional[int] = None,
    take: Optional[int] = None,
    desc: Optional[str] = None,
    keep_label: Optional[int] = None,
):
    """Return dict of arrays with pooled features and labels/domains/paths."""
    # We will extract features at the hidden layer sizes from metadata:
    # - CNN: output of first Linear in classifier (dimension = hidden_size)
    # - DANN: output of first Linear in label_predictor (dimension = feature_hidden_size)

    features_cnn: List[np.ndarray] = []
    features_dann: List[np.ndarray] = []
    labels: List[int] = []
    domains: List[str] = []
    paths: List[str] = []

    num_seen = 0
    iterator = loader
    total_batches = None
    try:
        total_batches = len(loader)
    except Exception:
        total_batches = None
    if tqdm is not None:
        iterator = tqdm(loader, total=total_batches, desc=(desc or "Extracting"))

    for batch in iterator:
        batch_paths = None
        batch_domains = None
        if isinstance(batch, tuple):
            # (images, labels) or (images, labels, domain)
            if len(batch) == 2:
                images, y = batch
            elif len(batch) >= 3:
                images, y = batch[0], batch[1]
                # optional third is domain per sample; ignore batch-level domains here
            else:
                raise ValueError(f"Unexpected tuple batch length: {len(batch)}")
        elif isinstance(batch, list):
            # Handle cases:
            # 1) Already-collated style: [images_tensor, labels_tensor]
            # 2) Standard list of samples: use default_collate
            # 3) Fallback manual per-sample extraction
            if len(batch) == 2 and isinstance(batch[0], torch.Tensor) and isinstance(batch[1], torch.Tensor):
                images, y = batch[0], batch[1]
            else:
                try:
                    try:
                        from torch.utils.data import default_collate as _default_collate
                    except Exception:
                        from torch.utils.data._utils.collate import default_collate as _default_collate  # type: ignore
                    collated = _default_collate(batch)
                    if isinstance(collated, tuple) and len(collated) >= 2:
                        images, y = collated[0], collated[1]
                    elif isinstance(collated, dict):
                        images = collated.get("image") or collated.get("original_image") or collated.get("processed_image")
                        y = collated.get("label")
                    else:
                        raise TypeError("Unsupported collated batch structure from default_collate")
                except Exception:
                    # Manual per-sample extraction fallback
                    imgs_list = []
                    labels_list = []
                    for s in batch:
                        if isinstance(s, (tuple, list)) and len(s) >= 2:
                            img, lbl = s[0], s[1]
                        elif isinstance(s, dict):
                            img = s.get("image") or s.get("original_image") or s.get("processed_image")
                            lbl = s.get("label")
                        else:
                            raise TypeError("Unsupported sample structure inside list batch")
                        imgs_list.append(img)
                        labels_list.append(lbl)
                    if isinstance(imgs_list[0], torch.Tensor):
                        try:
                            images = torch.stack(imgs_list, dim=0)
                        except Exception:
                            images = torch.concat([i.unsqueeze(0) for i in imgs_list], dim=0)
                    else:
                        imgs_np = []
                        for i in imgs_list:
                            arr = np.asarray(i)
                            if arr.ndim == 3 and arr.shape[-1] in (1, 3):
                                arr = np.transpose(arr, (2, 0, 1))
                            imgs_np.append(arr)
                        images = torch.as_tensor(np.stack(imgs_np, axis=0))
                    y = torch.as_tensor([int(l) for l in labels_list])
        elif isinstance(batch, dict):
            # Rare path: dict with keys
            images = batch["image"]
            y = batch["label"]
            batch_paths = batch.get("path")
            batch_domains = batch.get("domain")
        else:
            raise TypeError(f"Unsupported batch type: {type(batch)}")

        images = images.to(device)
        y = (y if isinstance(y, torch.Tensor) else torch.as_tensor(y)).to(device)

        if cnn is not None:
            # Pass through conv features
            conv = cnn.features(images)
            # Pass through the beginning of classifier up to first Linear
            cls = cnn.classifier
            x = cls[0](conv)   # AdaptiveAvgPool2d
            x = cls[1](x)      # Flatten
            x = cls[2](x)      # Dropout
            x = cls[3](x)      # Linear(512 -> hidden_size)
            emb = x
            features_cnn.append(emb.detach().cpu().numpy())

        if dann is not None:
            conv = dann.feature_extractor(images)
            pooled = dann.feature_pooling(conv)  # (N, feature_dim)
            # Pass through label_predictor first Linear to get feature_hidden_size
            lp = dann.label_predictor
            x = lp[0](pooled)  # Dropout
            x = lp[1](x)       # Linear(feature_dim -> feature_hidden_size)
            emb = x
            features_dann.append(emb.detach().cpu().numpy())

        labels.extend(y.detach().cpu().long().tolist())

        # Attempt to recover paths/domains; fallback to placeholders
        n = images.shape[0]
        if isinstance(batch_paths, list) and len(batch_paths) == n:
            paths.extend([str(p) for p in batch_paths])
        else:
            paths.extend([""] * n)
        if isinstance(batch_domains, list) and len(batch_domains) == n:
            domains.extend([str(d) for d in batch_domains])
        else:
            domains.extend(["?"] * n)

        num_seen += n
        if take is not None and num_seen >= take:
            break

    # Convert to arrays
    labels_arr = np.asarray(labels, dtype=np.int64)
    domains_arr = np.asarray(domains)
    paths_arr = np.asarray(paths)
    feats_cnn = np.concatenate(features_cnn, axis=0) if features_cnn else None
    feats_dann = np.concatenate(features_dann, axis=0) if features_dann else None

    # Optional label filter to avoid duplicate REAL across domains
    if keep_label is not None:
        idx = np.where(labels_arr == int(keep_label))[0]
        labels_arr = labels_arr[idx]
        domains_arr = domains_arr[idx]
        paths_arr = paths_arr[idx]
        if feats_cnn is not None:
            feats_cnn = feats_cnn[idx]
        if feats_dann is not None:
            feats_dann = feats_dann[idx]

    out = {
        "labels": labels_arr,
        "domains": domains_arr,
        "paths": paths_arr,
    }
    if feats_cnn is not None:
        out["features_cnn"] = feats_cnn
        if cnn_hidden_size is not None:
            out["cnn_hidden_size"] = np.asarray(cnn_hidden_size, dtype=np.int64)
    if feats_dann is not None:
        out["features_dann"] = feats_dann
        if dann_feature_hidden_size is not None:
            out["dann_feature_hidden_size"] = np.asarray(dann_feature_hidden_size, dtype=np.int64)
    return out


def run_tsne_and_plot(
    features: np.ndarray,
    labels: np.ndarray,
    domains: np.ndarray,
    out_png: Path,
    title: str,
    n_components: int = 2,
    init: str = "pca",
    learning_rate: Union[str, float] = "auto",
    perplexity: float = 30,
    random_state: int = 42,
):
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    tsne = TSNE(
        n_components=n_components,
        init=init,
        learning_rate=learning_rate,
        perplexity=perplexity,
        random_state=random_state,
    )
    emb2d = tsne.fit_transform(features)

    plt.figure(figsize=(8, 6), dpi=150)
    # Color by category: real, fake(SD2), fake(Kontext)
    categories = np.empty_like(labels, dtype=object)
    categories[(labels == 0)] = "real"
    categories[(labels == 1) & (domains == "SD2")] = "fake_sd2"
    categories[(labels == 1) & (domains == "Kontext")] = "fake_kontext"

    cat_to_color = {"real": "tab:blue", "fake_sd2": "tab:red", "fake_kontext": "tab:green"}
    for cat, color in cat_to_color.items():
        idx = np.where(categories == cat)[0]
        if idx.size == 0:
            continue
        plt.scatter(emb2d[idx, 0], emb2d[idx, 1], c=color, label=cat, alpha=0.7, s=12)

    plt.legend(title="Category", fontsize=8)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def build_dataset_loader(data_root: Path, img2img: str, data_split: str, batch_size: int, target_size: Tuple[int, int], shuffle: bool, sample_size: Optional[int]) -> torch.utils.data.DataLoader:
    data_dir = data_root / img2img / data_split
    metadata_dir = data_dir / "metadata"
    transform = get_eval_transforms(target_size=target_size)
    dataset = CarScratchDataset.load_binary_dataset(
        data_dir=str(data_dir),
        metadata_dir=str(metadata_dir),
        sample_size=sample_size,
        shuffle=shuffle,
        transform=transform,
    )
    return create_dataloader(dataset, batch_size=batch_size, shuffle=shuffle, target_size=target_size)


def main():
    parser = argparse.ArgumentParser(description="Extract CNN/DANN features (hidden sizes from metadata) and run t-SNE")
    parser.add_argument("--genai_root", type=str, default=str(PROJECT_ROOT / "cardd_data/GenAI_Results"), help="Root of GenAI_Results")
    parser.add_argument("--img2img", type=str, nargs="+", default=["SD2", "Kontext"], help="List of generator domains to include")
    parser.add_argument("--split", type=str, default="CarDD-TR", help="Data split under each domain (e.g., CarDD-TR, CarDD-TE)")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--sample_size", type=int, default=None)
    parser.add_argument("--test", action="store_true", help="Test mode: use a small subset (default 100 per domain)")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--models", type=str, default="both", choices=["both", "cnn", "dann"], help="Which models to extract features from: 'both', 'cnn', or 'dann'")
    parser.add_argument("--cnn_dir", type=str, default=None, help="Base dir of CNN runs or a specific run dir. If not provided, will prompt interactively.")
    parser.add_argument("--dann_dir", type=str, default=None, help="Base dir of DANN runs or a specific run dir. If not provided, will prompt interactively.")
    parser.add_argument("--non_interactive", action="store_true", help="Disable interactive prompts, use default paths")
    parser.add_argument("--out_npz", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default=str(PROJECT_ROOT / "generalize/analysis"), help="Base directory to store timestamped analysis runs")
    parser.add_argument("--plot", action="store_true", help="Also generate a t-SNE plot")
    parser.add_argument("--tsne_perplexity", type=float, default=30, help="t-SNE perplexity parameter")
    parser.add_argument("--tsne_init", type=str, default="pca", choices=["pca", "random"], help="t-SNE initialization method")
    parser.add_argument("--tsne_learning_rate", type=str, default="auto", help="t-SNE learning rate ('auto' or a float)")
    parser.add_argument("--tsne_n_components", type=int, default=2, help="t-SNE number of components")
    parser.add_argument("--tsne_random_state", type=int, default=42, help="t-SNE random state for reproducibility")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    # If test mode and no explicit sample_size, limit to 100 per domain
    if args.test and (args.sample_size is None or args.sample_size > 100):
        args.sample_size = 100

    # Determine which models to use
    use_cnn = args.models in ["both", "cnn"]
    use_dann = args.models in ["both", "dann"]
    
    if not use_cnn and not use_dann:
        raise ValueError("At least one model must be selected. Use --models with 'both', 'cnn', or 'dann'")
    
    print(f"Model selection: {args.models}")
    print(f"  - CNN: {'enabled' if use_cnn else 'disabled'}")
    print(f"  - DANN: {'enabled' if use_dann else 'disabled'}")

    # Resolve CNN model directory (if needed)
    cnn_model = None
    cnn_hidden_size = None
    cnn_weight_info = None
    cnn_run_dir = None
    target_size = (224, 224)  # Default target size
    
    if use_cnn:
        if args.cnn_dir:
            cnn_dir = Path(args.cnn_dir)
            if cnn_dir.is_dir() and cnn_dir.name.startswith("model_"):
                cnn_run_dir = cnn_dir if dir_has_cnn_weights(cnn_dir) else None
            else:
                cnn_run_dir = find_latest_with_weights(cnn_dir, prefix="model_", has_weights_fn=dir_has_cnn_weights)
            if cnn_run_dir is None:
                raise FileNotFoundError(f"No CNN weights found under: {cnn_dir}. Please specify --cnn_dir to a run with weights or retrain.")
        else:
            # Interactive selection
            if not args.non_interactive:
                default_cnn_base = PROJECT_ROOT / "simple_detect_car" / "models"
                cnn_base_input = input(f"\nCNN models base directory [{default_cnn_base}]: ").strip()
                cnn_base = Path(cnn_base_input) if cnn_base_input else default_cnn_base
                cnn_run_dir = _interactive_select_model_dir(
                    cnn_base, 
                    prefix="model_", 
                    model_type="CNN",
                    has_weights_fn=dir_has_cnn_weights,
                    max_list=30
                )
                if cnn_run_dir is None:
                    raise FileNotFoundError("No CNN model selected.")
            else:
                # Non-interactive mode: try default path
                default_cnn_base = PROJECT_ROOT / "simple_detect_car" / "models"
                cnn_run_dir = find_latest_with_weights(default_cnn_base, prefix="model_", has_weights_fn=dir_has_cnn_weights)
                if cnn_run_dir is None:
                    raise FileNotFoundError(f"No CNN weights found under: {default_cnn_base}. Please specify --cnn_dir or disable --non_interactive.")
        
        cnn_model, target_size, cnn_hidden_size, _, cnn_weight_info = load_cnn_model(cnn_run_dir, device)
        print(f"Loaded CNN model from: {cnn_run_dir}")

    # Resolve DANN model directory (if needed)
    dann_model = None
    dann_feature_hidden_size = None
    dann_weight_info = None
    dann_run_dir = None
    
    if use_dann:
        if args.dann_dir:
            dann_dir = Path(args.dann_dir)
            if dann_dir.is_dir() and dann_dir.name.startswith("model_dann_"):
                dann_run_dir = dann_dir if dir_has_dann_weights(dann_dir) else None
            else:
                dann_run_dir = find_latest_with_weights(dann_dir, prefix="model_dann_", has_weights_fn=dir_has_dann_weights)
            if dann_run_dir is None:
                raise FileNotFoundError(f"No DANN weights found under: {dann_dir}. Please specify --dann_dir to a run with weights or retrain.")
        else:
            # Interactive selection
            if not args.non_interactive:
                default_dann_base = PROJECT_ROOT / "domain_adapt" / "models"
                dann_base_input = input(f"\nDANN models base directory [{default_dann_base}]: ").strip()
                dann_base = Path(dann_base_input) if dann_base_input else default_dann_base
                dann_run_dir = _interactive_select_model_dir(
                    dann_base,
                    prefix="model_dann_",
                    model_type="DANN",
                    has_weights_fn=dir_has_dann_weights,
                    max_list=30
                )
                if dann_run_dir is None:
                    raise FileNotFoundError("No DANN model selected.")
            else:
                # Non-interactive mode: try default path
                default_dann_base = PROJECT_ROOT / "domain_adapt" / "models"
                dann_run_dir = find_latest_with_weights(default_dann_base, prefix="model_dann_", has_weights_fn=dir_has_dann_weights)
                if dann_run_dir is None:
                    raise FileNotFoundError(f"No DANN weights found under: {default_dann_base}. Please specify --dann_dir or disable --non_interactive.")

        # Read DANN metadata for feature_hidden_size
        dann_meta_path = dann_run_dir / "metadata.json"
        dann_feature_hidden_size = 256
        if dann_meta_path.exists():
            try:
                dm = json.loads(dann_meta_path.read_text())
                cfg = dm.get("config", {})
                dann_feature_hidden_size = int(cfg.get("feature_hidden_size", dann_feature_hidden_size))
                # If target_size not set from CNN, try to get from DANN
                if not use_cnn and cfg.get("target_size"):
                    ts = cfg.get("target_size")
                    if isinstance(ts, (list, tuple)) and len(ts) == 2:
                        target_size = (int(ts[0]), int(ts[1]))
            except Exception:
                pass
        dann_model, dann_weight_info = load_dann_model(dann_run_dir, device)
        print(f"Loaded DANN model from: {dann_run_dir}")
    
    # Ensure we have a target_size even if neither model is used (shouldn't happen, but safety check)
    if target_size is None:
        target_size = (224, 224)

    # Build combined loader over requested domains by simple concatenation
    loaders = []
    data_root = Path(args.genai_root)
    for dom in args.img2img:
        loaders.append(
            build_dataset_loader(
                data_root=data_root,
                img2img=dom,
                data_split=args.split,
                batch_size=args.batch_size,
                target_size=target_size,
                shuffle=False,
                sample_size=args.sample_size,
            )
        )

    # Iterate each loader, collect features and annotate domain
    all_features: dict = {"labels": [], "domains": [], "paths": [], "features_cnn": [], "features_dann": []}
    for i, (dom, loader) in enumerate(zip(args.img2img, loaders)):
        # For the first domain, keep both real/fake; for subsequent domains, keep only fake to avoid duplicating real
        keep_label = None if i == 0 else 1
        out = extract_features(
            loader=loader,
            device=device,
            cnn=cnn_model if use_cnn else None,
            dann=dann_model if use_dann else None,
            cnn_hidden_size=cnn_hidden_size if use_cnn else None,
            dann_feature_hidden_size=dann_feature_hidden_size if use_dann else None,
            take=args.sample_size,
            desc=f"{dom} {args.split}",
            keep_label=keep_label,
        )
        labels = out["labels"]
        paths = out["paths"]
        doms = np.array([dom] * len(labels))

        all_features["labels"].append(labels)
        all_features["domains"].append(doms)
        all_features["paths"].append(paths)
        if use_cnn and "features_cnn" in out:
            all_features["features_cnn"].append(out["features_cnn"])
        if use_dann and "features_dann" in out:
            all_features["features_dann"].append(out["features_dann"]) 

    # Concatenate across domains
    result = {
        "labels": np.concatenate(all_features["labels"], axis=0) if all_features["labels"] else np.empty((0,), dtype=np.int64),
        "domains": np.concatenate(all_features["domains"], axis=0) if all_features["domains"] else np.empty((0,), dtype=str),
        "paths": np.concatenate(all_features["paths"], axis=0) if all_features["paths"] else np.empty((0,), dtype=str),
    }
    if use_cnn and all_features["features_cnn"]:
        result["features_cnn"] = np.concatenate(all_features["features_cnn"], axis=0)
        result["cnn_hidden_size"] = np.asarray(cnn_hidden_size, dtype=np.int64)
    if use_dann and all_features["features_dann"]:
        result["features_dann"] = np.concatenate(all_features["features_dann"], axis=0)
        result["dann_feature_hidden_size"] = np.asarray(dann_feature_hidden_size, dtype=np.int64)

    # Prepare timestamped run directory
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_out_dir = Path(args.out_dir)
    run_dir = base_out_dir / f"tsne_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Decide final npz path: use user-provided file if given, else default under run_dir
    if args.out_npz:
        out_npz = Path(args.out_npz)
        out_npz.parent.mkdir(parents=True, exist_ok=True)
    else:
        out_npz = run_dir / "features_tsne.npz"

    np.savez_compressed(out_npz, **result)
    print(f"Saved features to: {out_npz}")

    # Parse learning_rate: if it's "auto", keep as string; otherwise convert to float
    tsne_learning_rate = args.tsne_learning_rate
    if tsne_learning_rate != "auto":
        try:
            tsne_learning_rate = float(tsne_learning_rate)
        except ValueError:
            print(f"Warning: Invalid learning_rate '{args.tsne_learning_rate}', using 'auto'")
            tsne_learning_rate = "auto"

    if args.plot:
        if use_cnn and "features_cnn" in result:
            png_path = (run_dir / "cnn_tsne.png") if run_dir else out_npz.with_name(out_npz.stem + "_cnn_tsne.png")
            run_tsne_and_plot(
                result["features_cnn"],
                result["labels"],
                result["domains"],
                png_path,
                title=f"t-SNE (CNN hidden {int(result['cnn_hidden_size'])})",
                n_components=args.tsne_n_components,
                init=args.tsne_init,
                learning_rate=tsne_learning_rate,
                perplexity=args.tsne_perplexity,
                random_state=args.tsne_random_state,
            )
            print(f"Saved t-SNE plot (CNN): {png_path}")
        if use_dann and "features_dann" in result:
            png_path = (run_dir / "dann_tsne.png") if run_dir else out_npz.with_name(out_npz.stem + "_dann_tsne.png")
            run_tsne_and_plot(
                result["features_dann"],
                result["labels"],
                result["domains"],
                png_path,
                title=f"t-SNE (DANN hidden {int(result['dann_feature_hidden_size'])})",
                n_components=args.tsne_n_components,
                init=args.tsne_init,
                learning_rate=tsne_learning_rate,
                perplexity=args.tsne_perplexity,
                random_state=args.tsne_random_state,
            )
            print(f"Saved t-SNE plot (DANN): {png_path}")

    # Write metadata.json similar to models
    meta = {
        "timestamp": ts,
        "config": {
            "models": args.models,
            "img2img": args.img2img,
            "split": args.split,
            "batch_size": args.batch_size,
            "sample_size": args.sample_size,
            "device": str(device),
            "cnn_model_dir": str(cnn_run_dir) if use_cnn else None,
            "dann_model_dir": str(dann_run_dir) if use_dann else None,
            "cnn_hidden_size": int(result.get("cnn_hidden_size", -1)) if "cnn_hidden_size" in result else None,
            "dann_feature_hidden_size": int(result.get("dann_feature_hidden_size", -1)) if "dann_feature_hidden_size" in result else None,
            "cnn_weights": cnn_weight_info if use_cnn else None,
            "dann_weights": dann_weight_info if use_dann else None,
            "tsne": {
                "n_components": args.tsne_n_components,
                "init": args.tsne_init,
                "learning_rate": str(tsne_learning_rate),
                "perplexity": args.tsne_perplexity,
                "random_state": args.tsne_random_state,
            },
        },
        "sizes": {
            "total": int(result["labels"].shape[0]),
            "num_real": int((result["labels"] == 0).sum()),
            "num_fake": int((result["labels"] == 1).sum()),
            "num_fake_sd2": int(((result["labels"] == 1) & (result["domains"] == "SD2")).sum()),
            "num_fake_kontext": int(((result["labels"] == 1) & (result["domains"] == "Kontext")).sum()),
        },
        "artifacts": {
            "features_npz": str(out_npz),
            "cnn_tsne_png": str((run_dir / "cnn_tsne.png")) if use_cnn and (run_dir / "cnn_tsne.png").exists() else None,
            "dann_tsne_png": str((run_dir / "dann_tsne.png")) if use_dann and (run_dir / "dann_tsne.png").exists() else None,
        }
    }
    with open(run_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved analysis metadata: {run_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()


