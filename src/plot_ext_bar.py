import numpy as np
import matplotlib.pyplot as plt

# ---- Raw trial data (4 trials per controller) ----
controllers = ["2c-GravComp", "2c-DynComp", "4c-GravComp", "4c-DynComp"]

data = {
    "2c-GravComp": {
        "leader":   {"j2": [0.0912, 0.0913, 0.0833, 0.0810],
                    "j4": [0.1480, 0.1529, 0.1522, 0.1516]},
        "follower": {"j2": [0.2356, 0.2385, 0.1868, 0.2490],
                    "j4": [0.2633, 0.2578, 0.2586, 0.2481]},
    },
    "2c-DynComp": {
        "leader":   {"j2": [0.1198, 0.0871, 0.1103, 0.1041],
                    "j4": [0.2035, 0.1829, 0.2026, 0.2007]},
        "follower": {"j2": [0.2628, 0.2543, 0.2564, 0.2530],
                    "j4": [0.2546, 0.2499, 0.2220, 0.2231]},
    },
    "4c-GravComp": {
        "leader":   {"j2": [0.0981, 0.1027, 0.0934, 0.0984],
                    "j4": [0.1762, 0.2036, 0.1807, 0.2109]},
        "follower": {"j2": [0.2389, 0.2190, 0.2013, 0.2549],
                    "j4": [0.2426, 0.2396, 0.2499, 0.2539]},
    },
    "4c-DynComp": {
        "leader":   {"j2": [0.1177, 0.1255, 0.1094, 0.1118],
                    "j4": [0.1713, 0.1671, 0.1845, 0.1616]},
        "follower": {"j2": [0.2325, 0.2214, 0.1507, 0.2323],
                    "j4": [0.2098, 0.2268, 0.1907, 0.2029]},
    },
}

def mean_std(x):
    x = np.asarray(x, dtype=float)
    return float(np.mean(x)), float(np.std(x, ddof=1))  # sample std

def collect_stats(who):  # who in {"leader","follower"}
    m_j2, s_j2, m_j4, s_j4 = [], [], [], []
    for c in controllers:
        m2, s2 = mean_std(data[c][who]["j2"])
        m4, s4 = mean_std(data[c][who]["j4"])
        m_j2.append(m2); s_j2.append(s2)
        m_j4.append(m4); s_j4.append(s4)
    return np.array(m_j2), np.array(s_j2), np.array(m_j4), np.array(s_j4)

def plot_controller_errors(who, title):
    m_j2, s_j2, m_j4, s_j4 = collect_stats(who)

    x = np.arange(len(controllers))
    width = 0.38

    plt.figure(figsize=(8.5, 4.2))
    plt.bar(x - width/2, m_j2, width, yerr=s_j2, capsize=4, label="Joint 2")
    plt.bar(x + width/2, m_j4, width, yerr=s_j4, capsize=4, label="Joint 4")

    plt.xticks(x, controllers, rotation=20, ha="right")
    plt.ylabel("External torque estimation error (nRMSE)")
    plt.title(title)
    plt.grid(True, axis="y", linewidth=0.5, alpha=0.5)
    plt.legend()
    plt.tight_layout()

# ---- Separate figures: leader and follower ----
plot_controller_errors("leader",   "Leader external torque estimation error")
plot_controller_errors("follower", "Follower external torque estimation error")

plt.show()
