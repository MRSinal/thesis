"""Layer-wise probing for a pretrained JEPA encoder.

Loads a pretrained JEPA checkpoint, freezes the encoder, extracts
spatiotemporally-pooled features from every transformer layer for a labeled
dataset, then trains a probe (linear or MLP) per layer with a hyperparameter
sweep under 5-fold grouped cross-validation. Reports Accuracy, F1 (macro), and
Jaccard (macro) as mean ± std across folds.

Run with:
    python probe.py --config-name cholec80 ckpt_path=/path/to/ckpt
"""

import csv
import importlib
import itertools
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, jaccard_score
from sklearn.model_selection import GroupKFold
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
    """Run the encoder once over the loader and stack per-layer pooled features.

    Pooling: mean over patch tokens (excluding CLS) per layer; if pixels arrive
    as (B, T, C, H, W), also mean over T so each clip yields one feature vector.

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

        was_5d = pixels.dim() == 5
        if was_5d:
            B, T = pixels.shape[:2]
            pixels = pixels.reshape(B * T, *pixels.shape[2:])

        output = encoder(
            pixels,
            interpolate_pos_encoding=True,
            output_hidden_states=True,
        )
        # (B', L, D) — drop CLS at index 0, mean over patch tokens
        pooled_per_layer = torch.stack(
            [h[:, 1:].mean(dim=1).cpu() for h in output.hidden_states], dim=1
        )

        if was_5d:
            L, D = pooled_per_layer.shape[1], pooled_per_layer.shape[2]
            pooled_per_layer = pooled_per_layer.reshape(B, T, L, D).mean(dim=1)
        # else: pooled_per_layer is already (B, L, D)

        all_feats.append(pooled_per_layer)
        all_labels.append(labels.cpu().long())

    feats = torch.cat(all_feats, dim=0)
    labels = torch.cat(all_labels, dim=0)
    return feats, labels


def train_probe(train_x, train_y, val_x, val_y, num_classes, cfg, device,
                lr, weight_decay):
    """Train a single probe (linear or MLP) and return val metrics dict."""
    in_dim = train_x.size(1)
    probe = build_probe(in_dim, num_classes, cfg).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)

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

    # HF populates _CAN_RECORD_REGISTRY in PreTrainedModel.__init__, which
    # torch.load skips without this, output_hidden_states=True is ignored.
    from transformers.utils.output_capturing import _CAN_RECORD_REGISTRY
    _CAN_RECORD_REGISTRY[str(encoder.__class__)] = encoder._can_record_outputs

    ##########################
    ##   loading dataset    ##
    ##########################
    print(f"Loading dataset: {cfg.dataset.module}")
    ds_kwargs = {**cfg.dataset.kwargs, "seed": cfg.seed}
    dataset = load_dataset(cfg.dataset.module, ds_kwargs)
    groups = np.asarray(dataset.groups)
    print(f"Dataset: {len(dataset)} samples, {len(set(groups.tolist()))} groups")

    loader = DataLoader(dataset, shuffle=False, **cfg.loader)

    ##########################
    ##     loading cached   ##
    ##########################
    cache_path = Path(cfg.feature_cache) if cfg.feature_cache else None
    if cache_path and cache_path.exists():
        print(f"Loading cached features: {cache_path}")
        cache = torch.load(cache_path)
        feats, labels = cache["feats"], cache["labels"]
        cached_groups = np.asarray(cache["groups"])
        assert len(cached_groups) == len(groups), "cache size mismatch"
        groups = cached_groups
    else:
        feats, labels = extract_features(encoder, loader, device)
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {"feats": feats, "labels": labels, "groups": groups.tolist()},
                cache_path,
            )
            print(f"Cached features to: {cache_path}")

    num_layers = feats.size(1)
    print(f"Features: {tuple(feats.shape)} (layers={num_layers})")

    ##########################
    ##  fold + sweep grid   ##
    ##########################
    gkf = GroupKFold(n_splits=cfg.probe.cv_folds)
    fold_indices = list(gkf.split(np.arange(len(feats)), groups=groups))
    for f_idx, (tr, va) in enumerate(fold_indices):
        assert set(groups[tr]).isdisjoint(set(groups[va])), \
            f"group leakage in fold {f_idx}"

    sweep = list(itertools.product(cfg.probe.lr_grid, cfg.probe.weight_decay_grid))
    print(f"CV folds: {cfg.probe.cv_folds} | sweep configs: {len(sweep)}")

    ##########################
    ##          eval        ##
    ##########################
    sel_metric = cfg.probe.selection_metric
    results = []
    print(
        f"\n{'layer':>5} | {'accuracy (mean±std)':>22} | "
        f"{'f1 (mean±std)':>22} | {'jaccard (mean±std)':>22}"
    )
    print("-" * 84)
    for layer in range(num_layers):
        layer_x = feats[:, layer]
        per_fold_best = []  # list of dicts with metrics + chosen lr/wd
        for f_idx, (tr, va) in enumerate(fold_indices):
            tr_x, tr_y = layer_x[tr], labels[tr]
            va_x, va_y = layer_x[va], labels[va]

            best = None
            for lr, wd in sweep:
                m = train_probe(
                    tr_x, tr_y, va_x, va_y,
                    num_classes=cfg.dataset.num_classes,
                    cfg=cfg, device=device,
                    lr=float(lr), weight_decay=float(wd),
                )
                if best is None or m[sel_metric] > best[sel_metric]:
                    best = {**m, "lr": float(lr), "weight_decay": float(wd)}
            per_fold_best.append(best)

        accs = np.array([b["accuracy"] for b in per_fold_best])
        f1s = np.array([b["f1"] for b in per_fold_best])
        jacs = np.array([b["jaccard"] for b in per_fold_best])
        std_kw = dict(ddof=1) if len(per_fold_best) > 1 else dict(ddof=0)

        row = {
            "layer": layer,
            "accuracy_mean": float(accs.mean()),
            "accuracy_std": float(accs.std(**std_kw)),
            "f1_mean": float(f1s.mean()),
            "f1_std": float(f1s.std(**std_kw)),
            "jaccard_mean": float(jacs.mean()),
            "jaccard_std": float(jacs.std(**std_kw)),
            "best_lr_per_fold": ";".join(f"{b['lr']:g}" for b in per_fold_best),
            "best_wd_per_fold": ";".join(f"{b['weight_decay']:g}" for b in per_fold_best),
        }
        results.append(row)
        print(
            f"{layer:>5} | "
            f"{row['accuracy_mean']:>9.4f} ± {row['accuracy_std']:.4f}  | "
            f"{row['f1_mean']:>9.4f} ± {row['f1_std']:.4f}  | "
            f"{row['jaccard_mean']:>9.4f} ± {row['jaccard_std']:.4f}"
        )

    ##########################
    ##   logging and dump   ##
    ##########################
    fieldnames = [
        "layer",
        "accuracy_mean", "accuracy_std",
        "f1_mean", "f1_std",
        "jaccard_mean", "jaccard_std",
        "best_lr_per_fold", "best_wd_per_fold",
    ]
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
        wandb.init(**cfg.wandb.config)
        for r in results:
            wandb.log(
                {
                    "probe/accuracy_mean": r["accuracy_mean"],
                    "probe/accuracy_std": r["accuracy_std"],
                    "probe/f1_mean": r["f1_mean"],
                    "probe/f1_std": r["f1_std"],
                    "probe/jaccard_mean": r["jaccard_mean"],
                    "probe/jaccard_std": r["jaccard_std"],
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
