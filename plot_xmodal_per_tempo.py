"""
Plot the cross-modal ("urge to move") result across tempos for one checkpoint:
auditory-only vestibular prediction (red) vs the true vestibular triangle (blue
dashed), with the auditory beat input (grey) for reference. One row per tempo.

Usage:
    ~/miniconda3/envs/phd_codes_v2/bin/python plot_xmodal_per_tempo.py <checkpoint.pt> [out.png]
"""
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from network import SensorimotorPCRNN
from utils import generate_input_sequences

TEMPOS = [0.4, 0.5, 0.6, 0.7, 0.8]
TEST_DURATION = 20.0


def load_net(ckpt_path):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    n = ck["config"]["network"]
    net = SensorimotorPCRNN(
        higher_size=n["higher_size"], associative_size=n["associative_size"],
        vestibular_size=n["vestibular_size"], alpha_H=n["alpha_H"], alpha_x=n["alpha_x"],
        inference_learning_rate_H=n["inference_learning_rate_H"],
        inference_learning_rate_x=n["inference_learning_rate_x"],
        weight_learning_rate_H=n["weight_learning_rate_H"],
        weight_learning_rate_x=n["weight_learning_rate_x"],
        n_inference_steps=n["n_inference_steps"], random_seed=ck["config"]["experiment"]["random_seed"],
    )
    net.load_state_dict(ck["model_state_dict"])
    return net, ck["config"], ck.get("step", "?")


def main():
    ckpt = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else "xmodal_per_tempo.png"
    net, cfg, step = load_net(ckpt)
    dt = cfg["experiment"]["dt"]
    t = np.arange(0, TEST_DURATION, dt)

    fig, axes = plt.subplots(len(TEMPOS), 1, figsize=(11, 10), sharex=True)
    for ax, tempo in zip(axes, TEMPOS):
        ecfg = {"experiment": {"dt": dt, "duration": TEST_DURATION, "mode": "sensorimotor",
                               "random_phase": False, "zero_mean_beat": True,
                               "tempo": {"mode": "single", "value": tempo}}}
        vseq, bseq = generate_input_sequences(ecfg, tempo=tempo, rng=np.random.default_rng(0))
        vtrue = vseq.numpy()
        net.reset_states()
        vp = []
        for v, b in zip(vseq, bseq):
            r = net.timestep_inference(v, b, auditory_only=True, prediction_timing="before")
            vp.append(r["vest_pred"].item())
        vp = np.array(vp)
        corr = np.corrcoef(vp[int(3/dt):], vtrue[int(3/dt):])[0, 1]

        ax.plot(t, bseq.numpy(), color="0.7", lw=0.8, label="auditory input")
        ax.plot(t, vtrue, "b--", lw=1.0, alpha=0.7, label="true vestibular (not given)")
        ax.plot(t, vp, "r-", lw=1.4, label="predicted vestibular (auditory-only)")
        ax.set_ylabel(f"tempo {tempo}s\nr={corr:+.2f}", fontsize=9)
        ax.axhline(0, color="k", lw=0.3, alpha=0.3)
    axes[0].legend(fontsize=8, loc="upper right", ncol=3)
    axes[-1].set_xlabel("time (s)")
    fig.suptitle(f"Cross-modal 'urge to move' across tempos — combined model, checkpoint step {step}", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(out, dpi=130)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
