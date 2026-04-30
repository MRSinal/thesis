"""Layer-wise probing for a pretrained GSViT (EfficientViT) encoder.

Loads a pretrained GSViT encoder (state_dict saved by `pretrain_model.py`),
freezes it, extracts globally average-pooled features after every encoder
sub-module for a labeled dataset, then trains a probe (linear or MLP,
controlled by `probe.type`) per layer and reports Accuracy, F1 (macro),
and Jaccard index (macro).

Layers probed:
    - patch_embed (the conv stem)
    - every top-level child of blocks1 / blocks2 / blocks3 (each
      EfficientViTBlock or downsample / patch-merging step)

GSViT was pretrained on cv2-loaded BGR frames in [0, 1] (no ImageNet
normalization). The reused Cholec80 dataset returns ImageNet-normalized
RGB tensors, so we undo that normalization and flip RGB->BGR before
feeding the encoder.

Run with:
    python probe.py --config-name cholec80 state_dict_path=/path/to/state_dict.pkl
"""

import csv
import importlib
import sys
from pathlib import Path

import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, jaccard_score
from torch.utils.data import DataLoader

# Make the project's dataset modules importable (they live in ../le-wm/).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "le-wm"))

from EfficientViT.classification.model.build import EfficientViT_M0

# ImageNet normalization stats used by the cholec80 dataset transform.
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def load_dataset(spec, kwargs):
    """spec: 'module.path:ClassName'"""
    module_path, class_name = spec.split(":")
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls(**dict(kwargs))


def build_probe(in_dim, num_classes, cfg):
    """Linear or MLP head, selected by cfg.probe.type."""
    t = cfg.probe.type
    if t == "linear":
        return nn.Linear(in_dim, num_classes)
    if t == "mlp":
        h = cfg.probe.get("hidden_dim") or in_dim * 2
        d = cfg.probe.get("dropout") or 0.0
        return nn.Sequential(
            nn.Linear(in_dim, h),
            nn.GELU(),
            nn.Dropout(d),
            nn.Linear(h, num_classes),
        )
    raise ValueError(f"Unknown probe.type: {t!r} (expected 'linear' or 'mlp')")


def load_encoder(state_dict_path, device):
    """Build an EfficientViT_M0 encoder and load weights from a saved state_dict.

    The training script saves `autoencoder.state_dict()` for an
    EfficientViTAutoEncoder, where the encoder lives under the `evit.*` prefix
    (the trailing classification head was already stripped at construction).
    We rebuild that same Sequential(patch_embed, blocks1, blocks2, blocks3)
    and load only the encoder portion.
    """
    state = torch.load(state_dict_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    # Strip a possible nn.DataParallel wrapper prefix.
    state = {
        (k[len("module."):] if k.startswith("module.") else k): v
        for k, v in state.items()
    }
    # Keep only the encoder weights.
    encoder_state = {
        k[len("evit."):]: v for k, v in state.items() if k.startswith("evit.")
    }
    if not encoder_state:
        raise RuntimeError(
            f"No 'evit.*' keys found in {state_dict_path}; "
            "expected an EfficientViTAutoEncoder state_dict."
        )

    base = EfficientViT_M0(pretrained=False)
    encoder = nn.Sequential(*list(base.children())[:-1])
    encoder.load_state_dict(encoder_state)
    return encoder.to(device)


def collect_layer_modules(encoder):
    """List (name, module) pairs for every sub-module we want to probe."""
    layers = [("patch_embed", encoder[0])]
    for stage_idx in range(1, 4):
        stage = encoder[stage_idx]
        for child_idx, child in enumerate(stage):
            layers.append((f"stage{stage_idx}.{child_idx}", child))
    return layers


@torch.no_grad()
def extract_features(encoder, loader, layer_modules, device):
    """Run the encoder once over the loader and stack per-layer GAP features.

    Returns:
        feats: list of (N, C_layer) tensors, one per probed layer (CPU)
        labels: (N,) long tensor
    """
    encoder.eval()

    captured = {}
    handles = []

    def make_hook(name):
        def hook(_mod, _inp, out):
            captured[name] = F.adaptive_avg_pool2d(out, 1).flatten(1).cpu()
        return hook

    for name, mod in layer_modules:
        handles.append(mod.register_forward_hook(make_hook(name)))

    per_layer = {name: [] for name, _ in layer_modules}
    all_labels = []

    mean = _IMAGENET_MEAN.to(device)
    std = _IMAGENET_STD.to(device)

    try:
        for batch in loader:
            pixels = batch["pixels"].to(device).float()
            labels = batch["label"]

            # If user provides (B, T, C, H, W), flatten T into B.
            if pixels.dim() == 5:
                B, T = pixels.shape[:2]
                pixels = pixels.reshape(B * T, *pixels.shape[2:])
                labels = labels.unsqueeze(1).expand(-1, T).reshape(-1)

            # Undo ImageNet normalization, then RGB -> BGR (GSViT training input).
            pixels = pixels * std + mean
            pixels = pixels[:, [2, 1, 0]]

            captured.clear()
            _ = encoder(pixels)

            for name, _ in layer_modules:
                per_layer[name].append(captured[name])
            all_labels.append(labels.cpu().long())
    finally:
        for h in handles:
            h.remove()

    feats = [torch.cat(per_layer[name], dim=0) for name, _ in layer_modules]
    labels = torch.cat(all_labels, dim=0)
    return feats, labels


def train_probe(train_x, train_y, val_x, val_y, num_classes, cfg, device):
    """Train a single probe (linear or MLP) and return val metrics dict."""
    in_dim = train_x.size(1)
    probe = build_probe(in_dim, num_classes, cfg).to(device)
    opt = torch.optim.AdamW(
        probe.parameters(), lr=cfg.probe.lr, weight_decay=cfg.probe.weight_decay
    )

    train_x = train_x.to(device)
    train_y = train_y.to(device)
    val_x = val_x.to(device)
    val_y_np = val_y.numpy()

    bs = cfg.probe.inner_batch_size
    n = train_x.size(0)

    for _ in range(cfg.probe.epochs):
        probe.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, bs):
            idx = perm[i : i + bs]
            logits = probe(train_x[idx])
            loss = F.cross_entropy(logits, train_y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()

    probe.eval()
    with torch.no_grad():
        preds = probe(val_x).argmax(dim=-1).cpu().numpy()

    return {
        "accuracy": float(accuracy_score(val_y_np, preds)),
        "f1": float(f1_score(val_y_np, preds, average="macro", zero_division=0)),
        "jaccard": float(
            jaccard_score(val_y_np, preds, average="macro", zero_division=0)
        ),
    }


@hydra.main(version_base=None, config_path="./config/eval", config_name="cholec80")
def run(cfg):
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    print(f"Loading state_dict: {cfg.state_dict_path}")
    encoder = load_encoder(cfg.state_dict_path, device)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    layer_modules = collect_layer_modules(encoder)
    layer_names = [n for n, _ in layer_modules]
    print(f"Probing {len(layer_names)} layers")

    ##########################
    ##   loading dataset    ##
    ##########################
    print(f"Loading dataset: {cfg.dataset.module}")
    ds_kwargs = {**cfg.dataset.kwargs, "seed": cfg.seed}
    train_set = load_dataset(cfg.dataset.module, {**ds_kwargs, "split": "train"})
    val_set = load_dataset(cfg.dataset.module, {**ds_kwargs, "split": "val"})
    print(f"Dataset: {len(train_set)} train / {len(val_set)} val")

    train_loader = DataLoader(train_set, shuffle=False, **cfg.loader)
    val_loader = DataLoader(val_set, shuffle=False, **cfg.loader)

    ##########################
    ##     loading cached   ##
    ##########################
    cache_path = Path(cfg.feature_cache) if cfg.feature_cache else None
    if cache_path and cache_path.exists():
        print(f"Loading cached features: {cache_path}")
        cache = torch.load(cache_path)
        train_feats, train_labels = cache["train_feats"], cache["train_labels"]
        val_feats, val_labels = cache["val_feats"], cache["val_labels"]
        layer_names = cache.get("layer_names", layer_names)
    else:
        train_feats, train_labels = extract_features(
            encoder, train_loader, layer_modules, device
        )
        val_feats, val_labels = extract_features(
            encoder, val_loader, layer_modules, device
        )
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "train_feats": train_feats,
                    "train_labels": train_labels,
                    "val_feats": val_feats,
                    "val_labels": val_labels,
                    "layer_names": layer_names,
                },
                cache_path,
            )
            print(f"Cached features to: {cache_path}")

    num_layers = len(train_feats)
    print(f"Features: {num_layers} layers")

    ##########################
    ##          eval        ##
    ##########################

    results = []
    print(f"\n{'layer':>6} | {'accuracy':>10} | {'f1':>10} | {'jaccard':>10}")
    print("-" * 46)
    for layer in range(num_layers):
        metrics = train_probe(
            train_feats[layer], train_labels,
            val_feats[layer], val_labels,
            num_classes=cfg.dataset.num_classes,
            cfg=cfg,
            device=device,
        )
        metrics["layer"] = layer
        results.append(metrics)
        print(
            f"{layer:>6} | {metrics['accuracy']:>10.4f} | "
            f"{metrics['f1']:>10.4f} | {metrics['jaccard']:>10.4f}"
        )

    ##########################
    ##   logging and dump   ##
    ##########################

    fieldnames = ["layer", "accuracy", "f1", "jaccard"]
    csv_path = Path(cfg.results_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            w.writerow({k: r[k] for k in fieldnames})
    print(f"\nResults saved to: {csv_path}")

    if cfg.wandb.enabled:
        import wandb
        run = wandb.init(**cfg.wandb.config)
        for r in results:
            wandb.log(
                {
                    "probe/accuracy": r["accuracy"],
                    "probe/f1": r["f1"],
                    "probe/jaccard": r["jaccard"],
                },
                step=r["layer"],
            )
        table = wandb.Table(
            columns=fieldnames,
            data=[[r[k] for k in fieldnames] for r in results],
        )
        wandb.log({"probe/results_table": table})
        wandb.save(str(csv_path))
        wandb.finish()


if __name__ == "__main__":
    run()
