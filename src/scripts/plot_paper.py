#!/usr/bin/env python3
import matplotlib.pyplot as plt
import os
import sys
import numpy as np
from matplotlib.lines import Line2D

# ---------------------------
# Global font setting (Times-like fallback)
# ---------------------------
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times", "Times New Roman", "DejaVu Serif", "STIXGeneral"]

# ---------------------------
# Shaded time intervals (edit as needed)
# ---------------------------
SHADED_WINDOWS = [(13, 23)]  # list of (start, end) in seconds
SHADE_COLOR = "gray"
SHADE_ALPHA = 0.3

# ---------------------------
# I/O
# ---------------------------
def read_config(file_path):
    kinematics_vars = []
    dynamics_vars = []
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith("Kinematics data:"):
                kinematics_vars = line.split(":")[1].strip().split(", ")
            elif line.startswith("Dynamics data:"):
                dynamics_vars = line.split(":")[1].strip().split(", ")
    return kinematics_vars, dynamics_vars

def read_data(file_path, variable_names, dof=4):
    data_dict = {name: [] for name in variable_names}
    with open(file_path, 'r') as file:
        for line in file:
            values = list(map(float, line.strip().split(",")))
            data_dict[variable_names[0]].append(values[0])  # time
            idx = 1
            for var in variable_names[1:]:
                if idx + dof <= len(values):
                    data_dict[var].append(values[idx:idx + dof])
                    idx += dof
    for key in data_dict:
        data_dict[key] = np.array(data_dict[key])
    return data_dict

# ---------------------------
# Filtering (First-order RC LPF)
# ---------------------------
def _rc_alpha(dt, cutoff_hz):
    if cutoff_hz <= 0:
        return 1.0
    tau = 1.0 / (2.0 * np.pi * cutoff_hz)
    return dt / (tau + dt)

def lowpass_rc_signal(x, t, cutoff_hz):
    x = np.asarray(x, dtype=float)
    t = np.asarray(t, dtype=float)
    if x.size == 0:
        return x
    y = np.empty_like(x)
    y[0] = x[0]
    for i in range(1, len(x)):
        dt = max(t[i] - t[i-1], 1e-12)
        alpha = _rc_alpha(dt, cutoff_hz)
        y[i] = y[i-1] + alpha * (x[i] - y[i-1])
    return y

def smooth_external_torques(dynamics_data, cutoff_hz=5.0):
    if 'time' not in dynamics_data:
        raise ValueError("dynamics_data must contain 'time'")
    t = dynamics_data['time']
    for key in ['leader external torque', 'follower external torque']:
        if key in dynamics_data:
            raw = dynamics_data[key]
            filt = np.zeros_like(raw, dtype=float)
            for j in range(raw.shape[1]):
                filt[:, j] = lowpass_rc_signal(raw[:, j], t, cutoff_hz)
            dynamics_data[f"{key} (filtered)"] = filt
    return dynamics_data

# ---------------------------
# Utility: apply gray shading to an axis
# ---------------------------
def shade_intervals(ax, intervals):
    for start, end in intervals:
        ax.axvspan(start, end, color=SHADE_COLOR, alpha=SHADE_ALPHA)

# ---------------------------
# Plotting layout (1 column, 6 rows; legends outside)
# ---------------------------
def plot_layout(kinematics_data, dynamics_data, folder_name, joints=(1, 3)):
    """
    Rows (single column):
      1) Joint 2 position (Follower vs Leader)
      2) Joint 4 position (Follower vs Leader)
      3) Position errors (Follower−Leader)
      4) Joint 2 torque (Leader vs −Follower)
      5) Joint 4 torque (Leader vs −Follower)
      6) Torque mismatch errors (Leader − (−Follower))
    """
    j2, j4 = joints
    t_kin = kinematics_data['time']
    t_dyn = dynamics_data['time']

    leader_key = 'leader external torque (filtered)' if 'leader external torque (filtered)' in dynamics_data else 'leader external torque'
    follower_key = 'follower external torque (filtered)' if 'follower external torque (filtered)' in dynamics_data else 'follower external torque'
    has_torque = (leader_key in dynamics_data) and (follower_key in dynamics_data)

    fig, axes = plt.subplots(nrows=6, ncols=1, figsize=(10, 16), sharex=False)

    # Helper: legends outside above each axis
    def add_legend(ax, ncol=2):
        ax.legend(loc='upper center',
                  bbox_to_anchor=(0.5, 1.20),
                  ncol=ncol, frameon=False, fontsize=10,
                  handlelength=2, columnspacing=1)

    # --- Position tracking (Rows 1–3) ---
    ax = axes[0]
    # Follower = blue solid, Leader = red dashed
    ax.plot(t_kin, kinematics_data['desired joint pos'][:, j2], 'b-', label='Follower')
    ax.plot(t_kin, kinematics_data['feedback joint pos'][:, j2], 'r--', label='Leader')
    shade_intervals(ax, SHADED_WINDOWS)
    ax.set_ylabel('Joint 2 Position (rad)')
    add_legend(ax)

    ax = axes[1]
    ax.plot(t_kin, kinematics_data['desired joint pos'][:, j4], 'b-', label='Follower')
    ax.plot(t_kin, kinematics_data['feedback joint pos'][:, j4], 'r--', label='Leader')
    shade_intervals(ax, SHADED_WINDOWS)
    ax.set_ylabel('Joint 4 Position (rad)')
    add_legend(ax)

    pos_err_j2 = kinematics_data['desired joint pos'][:, j2] - kinematics_data['feedback joint pos'][:, j2]
    pos_err_j4 = kinematics_data['desired joint pos'][:, j4] - kinematics_data['feedback joint pos'][:, j4]
    ax = axes[2]
    # Errors: J2 = blue solid, J4 = red dashed
    ax.plot(t_kin, pos_err_j2, 'b-', label='Joint 2')
    ax.plot(t_kin, pos_err_j4, 'r--', label='Joint 4')
    shade_intervals(ax, SHADED_WINDOWS)
    ax.set_ylabel('Position Error (rad)')
    add_legend(ax)

    # --- Force tracking (Rows 4–6) ---
    if has_torque:
        ax = axes[3]
        # Keep role/style consistent: (-)Follower = blue solid, Leader = red dashed
        ax.plot(t_dyn, -dynamics_data[follower_key][:, j2], 'b-', label='(-)Follower')
        ax.plot(t_dyn,  dynamics_data[leader_key][:, j2],  'r--', label='Leader')
        shade_intervals(ax, SHADED_WINDOWS)
        ax.set_ylabel('Joint 2 Torque (Nm)')
        add_legend(ax)

        ax = axes[4]
        ax.plot(t_dyn, -dynamics_data[follower_key][:, j4], 'b-', label='(-)Follower')
        ax.plot(t_dyn,  dynamics_data[leader_key][:, j4],  'r--', label='Leader')
        shade_intervals(ax, SHADED_WINDOWS)
        ax.set_ylabel('Joint 4 Torque (Nm)')
        add_legend(ax)

        torque_err_j2 = dynamics_data[leader_key][:, j2] - (-dynamics_data[follower_key][:, j2])
        torque_err_j4 = dynamics_data[leader_key][:, j4] - (-dynamics_data[follower_key][:, j4])
        ax = axes[5]
        # Errors: J2 = blue solid, J4 = red dashed
        ax.plot(t_dyn, torque_err_j2, 'b-', label='Joint 2')
        ax.plot(t_dyn, torque_err_j4, 'r--', label='Joint 4')
        shade_intervals(ax, SHADED_WINDOWS)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Torque Error (Nm)')
        add_legend(ax)
    else:
        for idx in range(3, 6):
            axes[idx].axis('off')
            axes[idx].text(0.5, 0.5, 'No torque data', ha='center', va='center', transform=axes[idx].transAxes)

    # --- Shared group labels + vertical lines (kept at your positions) ---
    fig.text(0.05, 0.75, 'Position Tracking', va='center', rotation='vertical', fontsize=14)
    fig.add_artist(Line2D((0.085, 0.085), (0.50, 0.98), color='black', linewidth=1.2))
    fig.text(0.05, 0.25, 'External Torque Tracking', va='center', rotation='vertical', fontsize=14)
    fig.add_artist(Line2D((0.085, 0.085), (0.02, 0.48), color='black', linewidth=1.2))

    # Layout + save (tight, no whitespace)
    fig.tight_layout(rect=[0.08, 0.0, 1.0, 0.97])

    output_path = folder_name + "_plot.pdf"
    fig.savefig(output_path, format="pdf", bbox_inches="tight", pad_inches=0)
    print(f"Saved figure to {output_path}")

    plt.show()

# ---------------------------
# Main
# ---------------------------
def main(folder_name, cutoff_hz=5.0):
    base_folder = '../data'
    folder_path = os.path.join(base_folder, folder_name)

    config_file = os.path.join(folder_path, 'config.txt')
    kinematics_file = os.path.join(folder_path, 'kinematics.txt')
    dynamics_file = os.path.join(folder_path, 'dynamics.txt')

    kinematics_vars, dynamics_vars = read_config(config_file)
    kinematics_data = read_data(kinematics_file, kinematics_vars, dof=4)
    dynamics_data = read_data(dynamics_file, dynamics_vars, dof=4)

    # Filter external torques
    smooth_external_torques(dynamics_data, cutoff_hz=float(cutoff_hz))

    # Make the plot (joints 2 & 4 → indices 1 and 3)
    plot_layout(kinematics_data, dynamics_data, folder_name, joints=(1, 3))

if __name__ == "__main__":
    # Usage: python3 plot.py <folder_name> [cutoff_hz]
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: python3 plot.py <folder_name> [cutoff_hz]")
    else:
        folder_arg = sys.argv[1]
        cutoff_arg = sys.argv[2] if len(sys.argv) == 3 else 5.0
        main(folder_arg, cutoff_hz=cutoff_arg)
