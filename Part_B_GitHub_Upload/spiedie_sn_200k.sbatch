#!/usr/bin/env bash
#SBATCH --job-name=sn-cifar10
#SBATCH --partition=gpucompute-h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=32G
#SBATCH --time=19:00:00
#SBATCH --output=sn-%j.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit with sbatch from the Part B project folder}"
if [[ ! -f run_part_b.py ]]; then
    echo "Submit from the extracted Part_B_Team_SN_200k directory." >&2
    exit 1
fi
export PYTHONUNBUFFERED=1
echo "Job $SLURM_JOB_ID on $(hostname)"
nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv

# Use the same interpreter for the preflight and the experiment.
python_bin="${PART_B_PYTHON:-python}"
"$python_bin" - <<'PY'
import sys
import torch, torchvision
print("Python:", sys.executable, sys.version)
print("PyTorch:", torch.__version__, "torchvision:", torchvision.__version__, "CUDA:", torch.version.cuda)
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable in this Python environment.")
layer = torch.nn.Conv2d(3, 8, 3).cuda()
layer(torch.ones(2, 3, 32, 32, device="cuda")).mean().backward()
torch.cuda.synchronize()
import train, eval_reproduction
print("GPU forward/backward and project imports passed.")
PY

# Optional argument: teammate's directory containing the two CIFAR-10 stats files.
args=(--out "$PWD/logs/SN-GAN_CIFAR10_CNN_200k")
if [[ $# -gt 0 ]]; then args+=(--stats-dir "$1"); fi
exec "$python_bin" -u run_part_b.py "${args[@]}"
