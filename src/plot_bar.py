import matplotlib.pyplot as plt
import numpy as np
import os
import glob

# ---------------------------
# Paper-friendly styling (consistent with multi-column script)
# ---------------------------
plt.rcParams.update({
    "savefig.format": "pdf",   # vector
    "font.family": "serif",
    "font.serif": ["Times", "Times New Roman", "DejaVu Serif", "STIXGeneral"],
    "axes.titlesize": 8,
    "axes.labelsize": 4,
    "xtick.labelsize": 4.0,
    "ytick.labelsize": 4.0,
    "legend.fontsize": 5,   # slightly larger legend font
    "lines.linewidth": 0.9,
    "lines.markersize": 3,
})

# ---------------------------
# Data
# ---------------------------
controllers = ["2c-GravComp", "2c-DynComp", "4c-GravComp", "4c-DynComp"]

position_tracking = {
    "Joint 2": {"mean": [0.688, 0.470, 0.443, 0.205],
                "std":  [0.069, 0.053, 0.021, 0.006]},
    "Joint 4": {"mean": [0.920, 0.393, 0.230, 0.188],
                "std":  [0.122, 0.038, 0.023, 0.012]},
}
force_tracking = {
    "Joint 2": {"mean": [11.350, 5.380, 9.590, 7.570],
                "std":  [4.401, 2.237, 3.331, 2.539]},
    "Joint 4": {"mean": [9.573, 4.233, 7.817, 6.025],
                "std":  [4.809, 1.941, 2.089, 2.792]},
}
free_motion_impedance = {
    "Joint 2": {"mean": [31.738, 10.831, 28.156, 7.848],
                "std":  [6.294, 1.783, 3.291, 0.332]},
    "Joint 4": {"mean": [3.060, 1.351, 1.378, 1.220],
                "std":  [0.422, 0.148, 0.141, 0.045]},
}
max_impedance = {
    "Joint 2": {"mean": [1026.464, 1038.238, 2002.594, 2073.822],
                "std":  [64.222, 10.010, 21.640, 27.462]},
    "Joint 4": {"mean": [216.831, 219.590, 382.913, 446.919],
                "std":  [29.336, 3.291, 15.148, 15.480]},
}

metrics = [
    ("Position Tracking (NRMSE, %)", position_tracking),
    ("Force Tracking (NRMSE, %)", force_tracking),
    ("Leader’s Impedance (Nm/rad)", free_motion_impedance),
    ("Max. Transmittable Impedance (Nm/rad)", max_impedance),
]

# ---------------------------
# Plot helper
# ---------------------------
def plot_two_group_four_bar(ax, data, ylabel):
    groups = ["Joint 2", "Joint 4"]
    group_x = np.arange(len(groups))
    bar_width = 0.18
    offsets = (np.arange(len(controllers)) - (len(controllers) - 1) / 2) * (bar_width + 0.02)

    for i, ctrl in enumerate(controllers):
        means = [data["Joint 2"]["mean"][i], data["Joint 4"]["mean"][i]]
        errs = [data["Joint 2"]["std"][i], data["Joint 4"]["std"][i]]
        ax.bar(group_x + offsets[i], means, width=bar_width,
               label=ctrl, yerr=errs, capsize=2)

    ax.set_xticks(group_x)
    ax.set_xticklabels(groups)
    ax.set_ylabel(ylabel, fontsize=5)
    ax.grid(axis="y", linestyle="--", alpha=0.6)

# ---------------------------
# Build the 2×2 figure
# ---------------------------
fig, axes = plt.subplots(2, 2, figsize=(3.5, 4.0))  # one-column width
axes = axes.ravel()

for ax, (ylabel, data) in zip(axes, metrics):
    plot_two_group_four_bar(ax, data, ylabel)

# Align y-axis labels across subplots
fig.align_ylabels(axes)

# Shared legend above plots, in one row, slightly down
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False,
           bbox_to_anchor=(0.5, 0.95))  # ncol=4 = one line, closer to plots

plt.tight_layout(rect=[0, 0, 1, 0.93])

# ---------------------------
# Save only as PDF with auto-incremented name
# ---------------------------
save_dir = "../data"
os.makedirs(save_dir, exist_ok=True)

existing = glob.glob(os.path.join(save_dir, "bar_*.pdf"))
next_num = len(existing) + 1
out_path = os.path.join(save_dir, f"bar_{next_num}.pdf")

fig.savefig(out_path, bbox_inches="tight")
print(f"Saved figure as {out_path}")

plt.show()
