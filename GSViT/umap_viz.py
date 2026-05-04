"""Per-layer UMAP visualization of cached GSViT features (Cholec80).

Reads the same Hydra config as probe.py to locate the feature cache and
seed, draws a deterministic random subsample, fits UMAP(n_components=2)
on each layer, and saves one phase-coloured scatter per layer.

The cache must already exist (built by probe.py with feature_cache set).
GSViT layers have heterogeneous feature dims (the cache stores
`feats_list`, a list of (N, D_l) tensors), which is why this script
mirrors le-wm/umap_viz.py rather than sharing it.

Run with:
    python umap_viz.py --config-name cholec80 \\
        feature_cache=./cache/gsvit_cholec80_features.pt
"""

import sys
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import umap

# Reuse the Cholec80 phase labels defined in le-wm without copying them.
_LEWM = (Path(__file__).resolve().parent.parent / "le-wm").as_posix()
if _LEWM not in sys.path:
    sys.path.insert(0, _LEWM)

from dataset_cholec80 import PHASE_TO_IDX  # noqa: E402

PHASE_NAMES = [name for name, _ in sorted(PHASE_TO_IDX.items(), key=lambda kv: kv[1])]


def subsample(n, max_n, rng):
    if n <= max_n:
        return np.arange(n)
    return np.sort(rng.choice(n, size=max_n, replace=False))


def plot_layer(emb, labels, num_classes, out_path, title):
    cmap = plt.get_cmap("tab10", num_classes)
    fig, ax = plt.subplots(figsize=(6, 6))
    sc = ax.scatter(
        emb[:, 0], emb[:, 1],
        c=labels, cmap=cmap,
        vmin=-0.5, vmax=num_classes - 0.5,
        s=3, alpha=0.6, linewidths=0,
    )
    cbar = fig.colorbar(sc, ax=ax, ticks=range(num_classes))
    cbar.ax.set_yticklabels(PHASE_NAMES[:num_classes])
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


@hydra.main(version_base=None, config_path="./config/eval", config_name="cholec80")
def run(cfg):
    cache_path = Path(cfg.feature_cache) if cfg.feature_cache else None
    if cache_path is None or not cache_path.exists():
        raise FileNotFoundError(
            f"feature_cache not found ({cache_path}). "
            f"Run probe.py with feature_cache set first to build it."
        )

    print(f"Loading cached features: {cache_path}")
    cache = torch.load(cache_path)
    feats_list = cache["feats_list"]            # list of (N, D_l)
    labels_full = cache["labels"].numpy()
    n_total = labels_full.shape[0]
    num_layers = len(feats_list)

    rng = np.random.default_rng(cfg.seed)
    idx = subsample(n_total, cfg.umap.max_points, rng)
    labels = labels_full[idx]
    print(
        f"Subsampled to {len(idx)} / {n_total} frames; "
        f"layers={num_layers}; per-layer dims={[t.size(1) for t in feats_list]}"
    )

    figures_dir = Path(cfg.umap.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    for layer in range(num_layers):
        print(f"  layer {layer:>2}/{num_layers - 1}: fitting UMAP", flush=True)
        x = feats_list[layer][idx].numpy()
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=cfg.umap.n_neighbors,
            min_dist=cfg.umap.min_dist,
            metric=cfg.umap.metric,
            random_state=cfg.seed,
        )
        emb = reducer.fit_transform(x)
        out_path = figures_dir / f"umap_layer{layer:02d}.png"
        plot_layer(
            emb, labels,
            num_classes=cfg.dataset.num_classes,
            out_path=out_path,
            title=f"GSViT layer {layer}",
        )
        print(f"    saved {out_path}")


if __name__ == "__main__":
    run()
