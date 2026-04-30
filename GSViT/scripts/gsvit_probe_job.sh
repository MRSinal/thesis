#!/bin/bash
#SBATCH -p GPU
#SBATCH -N 1
#SBATCH -t 0-36:00
#SBATCH -o slurm.%N.%j.out
#SBATCH -e slurm.%N.%j.err
#SBATCH --gres=gpu:1
#SBATCH --mem=16G

set -euo pipefail

cd /home/u941663/thesis/GSViT/

echo "[host] $(hostname)  [date] $(date -Iseconds)"
nvidia-smi || true

# Abort with a loud error in slurm.*.err if torch can't see a CUDA device.
if ! uv run python - <<'PY'
import sys, torch
if not torch.cuda.is_available():
    print(f"torch={torch.__version__} built-for-cuda={torch.version.cuda}: no CUDA device", file=sys.stderr)
    sys.exit(1)
print(f"torch={torch.__version__} cuda={torch.version.cuda} devices={torch.cuda.device_count()} "
      f"name={torch.cuda.get_device_name(0)}")
PY
then
    echo "ABORT: torch.cuda.is_available() == False — driver/wheel mismatch. Skipping training." >&2
    exit 1
fi

uv run probe.py --config-name cholec80
