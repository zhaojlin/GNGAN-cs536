
import json
import os
import matplotlib

# Use a non-interactive backend for remote Linux servers.
matplotlib.use("Agg")

import matplotlib.pyplot as plt

# ============================================================
# Configuration
# ============================================================

eval_path = "logs/GN-GAN_CIFAR10_CNN_Reproduction_v2/eval.txt"
out_dir = "experiments/gn_baseline"

os.makedirs(out_dir, exist_ok=True)

# Consistent plotting settings
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 11,
    "figure.dpi": 120,
    "savefig.dpi": 300,
})

# ============================================================
# Load evaluation results
# ============================================================

records = []

with open(eval_path, "r") as f:
    for line in f:
        line = line.strip()

        if not line:
            continue

        records.append(json.loads(line))

# Sort records by training step.
records.sort(key=lambda r: r["step"])

steps = [r["step"] for r in records]
is_g = [r["IS"] for r in records]
is_ema = [r["IS_EMA"] for r in records]
fid_g = [r["FID"] for r in records]
fid_ema = [r["FID_EMA"] for r in records]

# Convert training steps to thousands.
steps_k = [s / 1000 for s in steps]

# ============================================================
# Figure 1: FID vs Training Steps
# ============================================================

fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(
    steps_k,
    fid_g,
    marker="o",
    linewidth=2,
    markersize=5,
    label="GN-GAN (G)"
)

ax.plot(
    steps_k,
    fid_ema,
    marker="o",
    linewidth=2,
    markersize=5,
    label="GN-GAN (EMA)"
)

ax.set_xlabel("Training Steps (k)")
ax.set_ylabel("FID (lower is better)")
ax.set_title("GN-GAN on CIFAR-10 (CNN): FID During Training")

# Logarithmic Y-axis
ax.set_yscale("log")

# Show readable FID values instead of scientific notation.
fid_ticks = [20, 30, 50, 100, 200, 400]

ax.set_yticks(fid_ticks)
ax.set_yticklabels([str(v) for v in fid_ticks])

# Training steps in thousands.
ax.set_xticks([0, 40, 80, 120, 160, 200])
ax.set_xlim(-5, 205)

ax.grid(
    True,
    which="major",
    alpha=0.3,
    linestyle="-"
)

ax.legend(loc="upper right")

# Explain the intermediate evaluation protocol.
fig.text(
    0.5,
    0.015,
    "Intermediate evaluation: 5,000 generated images per checkpoint",
    ha="center",
    fontsize=9,
    color="dimgray"
)

fig.tight_layout(rect=[0, 0.045, 1, 1])

fid_output = os.path.join(out_dir, "fid_curve.png")
fig.savefig(fid_output, dpi=300, bbox_inches="tight")
plt.close(fig)

# ============================================================
# Figure 2: IS vs Training Steps
# ============================================================

fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(
    steps_k,
    is_g,
    marker="o",
    linewidth=2,
    markersize=5,
    label="GN-GAN (G)"
)

ax.plot(
    steps_k,
    is_ema,
    marker="o",
    linewidth=2,
    markersize=5,
    label="GN-GAN (EMA)"
)

ax.set_xlabel("Training Steps (k)")
ax.set_ylabel("Inception Score (higher is better)")
ax.set_title("GN-GAN on CIFAR-10 (CNN): IS During Training")

# Keep IS on a linear Y-axis.
ax.set_yscale("linear")
ax.set_ylim(bottom=0)

ax.set_xticks([0, 40, 80, 120, 160, 200])
ax.set_xlim(-5, 205)

ax.grid(
    True,
    which="major",
    alpha=0.3,
    linestyle="-"
)

ax.legend(loc="lower right")

fig.text(
    0.5,
    0.015,
    "Intermediate evaluation: 5,000 generated images per checkpoint",
    ha="center",
    fontsize=9,
    color="dimgray"
)

fig.tight_layout(rect=[0, 0.045, 1, 1])

is_output = os.path.join(out_dir, "is_curve.png")
fig.savefig(is_output, dpi=300, bbox_inches="tight")
plt.close(fig)

# ============================================================
# Summary
# ============================================================

print("Training evaluation curves generated successfully.")
print(f"Number of evaluation checkpoints: {len(records)}")
print(f"Training steps: {min(steps)} to {max(steps)}")

print("\nSaved figures:")
print(fid_output)
print(is_output)
