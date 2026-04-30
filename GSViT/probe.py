"""Layer-wise probing for the pretrained GSViT (EfficientViT-M5) encoder.

Run with:
    python probe.py --config-name cholec80 ckpt_path=/path/to/GSViT.pkl
"""

import csv
import importlib
import itertools
import sys
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, jaccard_score
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader

# Reuse Cholec80 dataset module from le-wm without copying it.
_LEWM = (Path(__file__).resolve().parent.parent / "le-wm").as_posix()
if _LEWM not in sys.path:
    sys.path.insert(0, _LEWM)

from load_gsvit import EfficientViT as GSViTWrapper  # noqa: E402


def load_dataset(spec, kwargs):
    """spec: 'module.path:ClassName'"""
    module_path, class_name = spec.split(":")
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls(**dict(kwargs))


def build_probe(in_dim, num_classes, cfg):
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
    raise ValueError(f"Unknown probe.type: {t!r}")


def bgr_flip(images):
    """In-place BGR<->RGB swap on (B, C, H, W) tensors. Matches load_gsvit.process_inputs."""
    out = images.clone()
    out[:, 0], out[:, 2] = images[:, 2], images[:, 0]
    return out


def collect_hook_targets(evit_seq):
    """Return ordered list of submodules to hook for per-layer features.

    `evit_seq` is the Sequential [patch_embed, blocks1, blocks2, blocks3].
    Layer 0 is the patch_embed output; subsequent layers are the direct
    children of each blocksN (mix of EfficientViTBlock and downsample modules).
    """
    targets = [("patch_embed", evit_seq[0])]
    for stage_idx in (1, 2, 3):
        stage = evit_seq[stage_idx]
        for child_idx, child in enumerate(stage):
            targets.append((f"blocks{stage_idx}.{child_idx}", child))
    return targets


@torch.no_grad()
def extract_features(evit_seq, hook_targets, loader, device):
    """Run the encoder once over the loader and collect per-layer pooled features.

    Returns a list of length L; entry ℓ is a (N, D_ℓ) tensor on CPU.
    Per-layer feature dim differs because EfficientViT widens with depth, so we
    keep a list-of-tensors instead of a stacked (N, L, D) tensor.

    Pooling: mean over (H', W'); if pixels arrive 5D, also mean over T.
    """
    evit_seq.eval()

    feat_buffer = {}
    handles = []
    for i, (_, mod) in enumerate(hook_targets):
        def make_hook(idx):
            def _hook(_m, _inp, out):
                feat_buffer[idx] = out
            return _hook
        handles.append(mod.register_forward_hook(make_hook(i)))

    L = len(hook_targets)
    per_layer = [[] for _ in range(L)]
    all_labels = []

    try:
        for batch in loader:
            pixels = batch["pixels"].to(device).float()
            labels = batch["label"]

            was_5d = pixels.dim() == 5
            if was_5d:
                B, T = pixels.shape[:2]
                pixels = pixels.reshape(B * T, *pixels.shape[2:])

            pixels = bgr_flip(pixels)
            feat_buffer.clear()
            evit_seq(pixels)

            for ell in range(L):
                fmap = feat_buffer[ell]            # (B', C_ℓ, H', W')
                pooled = fmap.mean(dim=(2, 3))     # (B', C_ℓ)
                if was_5d:
                    pooled = pooled.reshape(B, T, -1).mean(dim=1)  # (B, C_ℓ)
                per_layer[ell].append(pooled.cpu())

            all_labels.append(labels.cpu().long())
    finally:
        for h in handles:
            h.remove()

    feats_list = [torch.cat(x, dim=0) for x in per_layer]
    labels = torch.cat(all_labels, dim=0)
    return feats_list, labels


def train_probe(train_x, train_y, val_x, val_y, num_classes, cfg, device,
                lr, weight_decay):
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


def load_gsvit_encoder(ckpt_path, device):
    """Load GSViT.pkl into the EfficientViT wrapper and return the head-stripped Sequential."""
    wrapper = GSViTWrapper(in_size=1)
    state = torch.load(ckpt_path, map_location=device)
    wrapper.load_state_dict(state)
    evit_seq = wrapper.evit.to(device)
    evit_seq.eval()
    for p in evit_seq.parameters():
        p.requires_grad_(False)
    return evit_seq


@hydra.main(version_base=None, config_path="./config/eval", config_name="cholec80")
def run(cfg):
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    print(f"Loading checkpoint: {cfg.ckpt_path}")
    evit_seq = load_gsvit_encoder(cfg.ckpt_path, device)
    hook_targets = collect_hook_targets(evit_seq)
    print(f"Hookable layers: {len(hook_targets)}")
    for i, (name, _) in enumerate(hook_targets):
        print(f"  layer {i}: {name}")

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
        feats_list = cache["feats_list"]
        labels = cache["labels"]
        cached_groups = np.asarray(cache["groups"])
        assert len(cached_groups) == len(groups), "cache size mismatch"
        groups = cached_groups
    else:
        feats_list, labels = extract_features(evit_seq, hook_targets, loader, device)
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "feats_list": feats_list,
                    "labels": labels,
                    "groups": groups.tolist(),
                },
                cache_path,
            )
            print(f"Cached features to: {cache_path}")

    num_layers = len(feats_list)
    print(f"Per-layer feature dims: {[t.size(1) for t in feats_list]}")

    ##########################
    ##  fold + sweep grid   ##
    ##########################
    n = labels.size(0)
    gkf = GroupKFold(n_splits=cfg.probe.cv_folds)
    fold_indices = list(gkf.split(np.arange(n), groups=groups))
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
        layer_x = feats_list[layer]
        per_fold_best = []
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

