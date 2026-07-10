"""
Per-tempo evaluation for the Step-3 multi-tempo gate.

Reads predictions BEFORE the within-timestep inference loop (prediction_timing
="before") -- the genuine top-down anticipation. "after" is also reported for the
auditory-only cross-modal metric to expose how much fitting the concurrent
auditory input inflates it (that is postdiction, not anticipation).

Per checkpoint and tempo:
  - comb_vMSE   : combined vestibular prediction error, BEFORE inference
  - xmod_b      : auditory-only vest prediction corr vs true triangle, BEFORE  <-- the honest one
  - xmod_a      : same but AFTER inference (inflated; for contrast)
  - amp_b       : std of the BEFORE auditory-only vest prediction

Usage:
    ~/miniconda3/envs/phd_codes_v2/bin/python eval_gate_per_tempo.py <exp_dir>
"""
import sys, glob, re
import numpy as np
import torch
from network import SensorimotorPCRNN
from utils import generate_input_sequences

TEMPOS = [0.4, 0.5, 0.6, 0.7, 0.8]
TEST_DURATION = 20.0
TRANSIENT_S = 3.0


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
    return net, ck["config"]


def run(net, vseq, bseq, auditory_only, timing):
    net.reset_states()
    vp = []
    for v, b in zip(vseq, bseq):
        r = net.timestep_inference(v, b, auditory_only=auditory_only, prediction_timing=timing)
        vp.append(r["vest_pred"].item())
    return np.array(vp)


def eval_ckpt(ckpt_path):
    net, cfg = load_net(ckpt_path)
    dt = cfg["experiment"]["dt"]
    skip = int(TRANSIENT_S / dt)
    rows = []
    for tempo in TEMPOS:
        ecfg = {"experiment": {"dt": dt, "duration": TEST_DURATION, "mode": "sensorimotor",
                               "random_phase": False, "zero_mean_beat": True,
                               "tempo": {"mode": "single", "value": tempo}}}
        vseq, bseq = generate_input_sequences(ecfg, tempo=tempo, rng=np.random.default_rng(0))
        vtrue = vseq.numpy()

        vp_c = run(net, vseq, bseq, auditory_only=False, timing="before")
        vmse = np.mean((vp_c[skip:] - vtrue[skip:]) ** 2)

        vp_xb = run(net, vseq, bseq, auditory_only=True, timing="before")
        vp_xa = run(net, vseq, bseq, auditory_only=True, timing="after")

        def corr(x):
            return np.corrcoef(x[skip:], vtrue[skip:])[0, 1] if x[skip:].std() > 1e-9 else 0.0
        rows.append((tempo, vmse, corr(vp_xb), corr(vp_xa), vp_xb[skip:].std()))
    return rows


def main():
    exp_dir = sys.argv[1].rstrip("/")
    ckpts = sorted(glob.glob(f"{exp_dir}/model_step_*.pt"),
                   key=lambda p: int(re.search(r"step_(\d+)", p).group(1)))
    print(f"{'ckpt':>8} | {'tempo':>5} | {'comb_vMSE':>9} | {'xmod_BEFORE':>11} | {'xmod_after':>10} | {'amp_b':>6}")
    print("-" * 70)
    summary = []
    for c in ckpts:
        step = int(re.search(r"step_(\d+)", c).group(1))
        rows = eval_ckpt(c)
        cb = [r[2] for r in rows]; ca = [r[3] for r in rows]; vm = [r[1] for r in rows]
        for (tempo, vmse, corr_b, corr_a, amp) in rows:
            print(f"{step:>8} | {tempo:>5.2f} | {vmse:>9.4f} | {corr_b:>+11.3f} | {corr_a:>+10.3f} | {amp:>6.3f}")
        print(f"{'':>8} | {'MEAN':>5} | {np.mean(vm):>9.4f} | {np.mean(cb):>+11.3f} | {np.mean(ca):>+10.3f} | "
              f"min_before={min(cb):+.3f}")
        print("-" * 70)
        summary.append((step, np.mean(cb), min(cb), np.mean(ca)))
    print("\nSUMMARY: mean BEFORE-corr | worst-tempo BEFORE-corr | mean AFTER-corr (inflated)")
    for step, mb, mn, ma in summary:
        print(f"  step {step:>6}:  before mean={mb:+.3f}  worst={mn:+.3f}   |   after mean={ma:+.3f}")
    best = max(summary, key=lambda s: s[2])
    print(f"\nBest all-tempo checkpoint by worst-tempo BEFORE corr: step {best[0]} (worst={best[2]:+.3f})")


if __name__ == "__main__":
    main()
