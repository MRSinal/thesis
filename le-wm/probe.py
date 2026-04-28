"""Layer-wise probing for a pretrained JEPA encoder.

Loads a pretrained JEPA checkpoint, freezes the encoder, extracts CLS token
features from every transformer layer for a labeled dataset, then trains a
probe (linear or MLP, controlled by `probe.type`) per layer and reports
Accuracy, F1 (macro), and Jaccard index (macro).

Run with:
    python probe.py --config-name cholec80 ckpt_path=/path/to/ckpt
"""

import csv
import importlib
from pathlib import Path

import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, jaccard_score
from torch.utils.data import DataLoader


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


@torch.no_grad()
def extract_features(encoder, loader, device):
    """Run the encoder once over the loader and stack per-layer CLS features.

    Returns:
        feats: (N, L, D) tensor (L = num_hidden_layers + 1)
        labels: (N,) long tensor
    """
    encoder.eval()
    all_feats = []
    all_labels = []

    for batch in loader:
        pixels = batch["pixels"].to(device).float()
        labels = batch["label"]

        # ViT expects (B, C, H, W). If user provides (B, T, C, H, W), flatten T into B.
        if pixels.dim() == 5:
            B, T = pixels.shape[:2]
            pixels = pixels.reshape(B * T, *pixels.shape[2:])
            labels = labels.unsqueeze(1).expand(-1, T).reshape(-1)

        output = encoder(
            pixels,
            interpolate_pos_encoding=True,
            output_hidden_states=True,
        )
        # TODO: Do we do this in ./jepa.py?
        cls_per_layer = torch.stack(
            [h[:, 0].cpu() for h in output.hidden_states], dim=1
        )
        all_feats.append(cls_per_layer)
        all_labels.append(labels.cpu().long())

    feats = torch.cat(all_feats, dim=0)
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

    print(f"Loading checkpoint: {cfg.ckpt_path}")
    model = torch.load(cfg.ckpt_path, map_location=device, weights_only=False)
    encoder = model.encoder.to(device)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    
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
    else:
        train_feats, train_labels = extract_features(encoder, train_loader, device)
        val_feats, val_labels = extract_features(encoder, val_loader, device)
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "train_feats": train_feats,
                    "train_labels": train_labels,
                    "val_feats": val_feats,
                    "val_labels": val_labels,
                },
                cache_path,
            )
            print(f"Cached features to: {cache_path}")

    num_layers = train_feats.size(1)
    print(f"Features: {tuple(train_feats.shape)} (layers={num_layers})")

    ##########################
    ##          eval        ##
    ##########################

    results = []
    print(f"\n{'layer':>6} | {'accuracy':>10} | {'f1':>10} | {'jaccard':>10}")
    print("-" * 46)
    for layer in range(num_layers):
        metrics = train_probe(
            train_feats[:, layer], train_labels,
            val_feats[:, layer], val_labels,
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
