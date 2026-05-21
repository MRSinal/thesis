# Latent Predictive Modeling vs. High-Fidelity Visual Reconstruction for Robust Surgical Phase Semantics

Bachelor thesis by Balint Tapai, Cognitive Science & Artificial Intelligence, Tilburg University (2026).
Supervisor: dr. Silvia-Laura Pintea.

This repository contains the code, configuration, and write-up for a comparison between a latent predictive
self-supervised objective (LeJEPA, adapted from `le-wm`) and a high-fidelity visual-reconstruction backbone
(GSViT) for learning representations that transfer to surgical phase recognition on Cholec80.

## Repository structure

```
thesis/
├── README.md                  this file
├── pyproject.toml             top-level uv workspace (members: le-wm, GSViT)
├── main.py                    placeholder entry point
├── split.py                   sample a frame-budgeted subset of videos
├── extract_frames.py          ffmpeg-based frame extraction to JPEGs
├── data_exploration.py        dataset stats / video metadata exploration
├── data_exploration.ipynb     notebook version of the above
├── results.ipynb              probe-results aggregation, tables, plots
├── assests/                   figures used by the paper
├── paper/                     LaTeX sources (main.tex, frontmatter.tex, bib, pdf)
│
├── TF-Cholec80/               vendored data-prep code (see "Third-party code")
│
├── le-wm/                     LeJEPA pre-training + linear probing
│   ├── train.py                  pre-train entry point
│   ├── jepa.py                   LeJEPA model / loss
│   ├── module.py                 Lightning wrapper
│   ├── dataset_surgical.py       my surgical-video dataset for pre-training
│   ├── dataset_cholec80.py       my Cholec80 phase-labelled dataset for probing
│   ├── probe.py                  my linear / MLP probe over frozen features
│   ├── umap_viz.py               my UMAP visualisation of frozen features
│   ├── config/train/             Hydra configs for pre-training
│   ├── config/eval/              Hydra configs for probing
│   └── scripts/                  SLURM submission scripts
│
└── GSViT/                     GSViT pre-training + linear probing
    ├── pretrain_model.py         pre-train entry point
    ├── load_gsvit.py             checkpoint loading helper
    ├── dataloader_surgical.py    my surgical-video dataloader
    ├── probe.py                  my linear / MLP probe over frozen features
    ├── umap_viz.py               my UMAP visualisation of frozen features
    ├── EfficientViT/             vendored GSViT/EfficientViT backbone (see below)
    ├── config/eval/              Hydra configs for probing
    └── scripts/                  SLURM submission scripts
```

## Reproducing the results

The project is managed with [uv](https://docs.astral.sh/uv/). `pyproject.toml` declares a uv workspace with
`le-wm/` and `GSViT/` as members, and pins a CUDA 12.1 PyTorch index.

### 1. Environment

```bash
# from the repo root
uv sync
```

This will create a `.venv` and install dependencies for the root project and both workspace members.

### 2. Data

The Cholec80 videos and phase annotations have to be obtained through the official channel described in the
`TF-Cholec80` repository (see "Third-party code" below). Once you have the raw videos and the
`phase_annotations/` folder, pre-extract frames:

```bash
# optional: sample a frame-budgeted subset of videos
uv run split.py /path/to/cholec80/videos --max-frames 1000000 --out subset.txt \
    --out-dir /path/to/cholec80/videos_subset --symlink

# extract JPEG frames at 224x224 (used everywhere downstream)
uv run extract_frames.py \
    --video-roots /path/to/cholec80/videos \
    --out /path/to/cholec80/frames \
    --img-size 224 --every 1 --jobs 16
```

Update the `frames_root` / `phase_root` paths in the eval configs
(`le-wm/config/eval/cholec80.yaml`, `GSViT/config/eval/cholec80.yaml`) and the training data config
(`le-wm/config/train/data/`) to point to your local copy.

### 3. Pre-training

LeJEPA on surgical frames:

```bash
cd le-wm
uv run train.py --config-name lewm_surgical
```

GSViT on surgical frames:

```bash
cd GSViT
uv run pretrain_model.py
```

SLURM submission wrappers for both are in `le-wm/scripts/jepa_pretrain_job.sh` and
`GSViT/scripts/gsvit_pretrain_job.sh`.

### 4. Linear / MLP probing on Cholec80

After pre-training, point the `ckpt_path` field in the eval YAML at the produced checkpoint and run:

```bash
# LeJEPA features
cd le-wm
uv run probe.py --config-name cholec80

# GSViT features
cd GSViT
uv run probe.py --config-name cholec80
```

Each probe sweeps `lr_grid x weight_decay_grid` with K-fold CV, caches frozen features under
`./cache/`, writes per-config metrics to `probe_results_*.csv`, and logs to W&B if enabled in the config.

### 5. UMAP visualisations

```bash
# from le-wm/ or GSViT/
uv run umap_viz.py --config-name cholec80
```

### 6. Aggregating results

Open `results.ipynb` at the repo root to load the `probe_results_*.csv` files and produce the tables and
figures used in the paper. The paper sources live under `paper/` and compile to `paper/main.pdf`.

## Third-party code

This work builds on three external repositories. Their roles and the modifications I made are documented
below.

- **[CAMMA-public/TF-Cholec80](https://github.com/CAMMA-public/TF-Cholec80)** — used to obtain the
  Cholec80 dataset (videos and phase annotations) and as a reference for the canonical Cholec80 splits.
  Vendored under `TF-Cholec80/`. Used as-is.

- **[lucas-maes/le-wm](https://github.com/lucas-maes/le-wm)** — starting point for the LeJEPA
  implementation under `le-wm/`. The original code is an action-conditioned latent world model; I removed
  the action-conditioning so the predictor operates purely on visual context, swapped in the surgical
  video dataloader (`dataset_surgical.py`), and added the Cholec80 probing pipeline. The pre-training
  loop (`train.py`, `module.py`, `jepa.py`) and configs were adapted accordingly.

- **[SamuelSchmidgall/GSViT](https://github.com/SamuelSchmidgall/GSViT)** — source of the GSViT /
  EfficientViT backbone under `GSViT/EfficientViT/` and the pre-training recipe. I fixed a bug in the
  residual connection of the backbone.

Everything else in this repository — the Cholec80 and surgical-video dataloaders, the probing pipelines
(`probe.py` in both `le-wm/` and `GSViT/`), the UMAP visualisations (`umap_viz.py`), the data exploration
notebooks/scripts, the frame-extraction and subsetting utilities, the Hydra configs, and the
results-aggregation notebook — was written by me as part of the thesis.

## Data

- [Cholec80](https://github.com/CAMMA-public/TF-Cholec80)
- [CATARACTS](https://ieee-dataport.org/open-access/cataracts)
- [AUTOLAPARO](https://autolaparo.github.io/)
- [JIGSAWS](https://cirl.lcsr.jhu.edu/research/hmm/datasets/jigsaws_release/)
- [PITVIS](https://rdr.ucl.ac.uk/articles/dataset/PitVis_Challenge_Endoscopic_Pituitary_Surgery_videos/26531686)
