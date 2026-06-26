"""
Step-1 verification for the revision-2 input machinery.

(1) Asserts the default path (auditory_envelope.shape == 'pulse', gait_variability
    'none') is numerically IDENTICAL to the original input generation, so the
    baseline reproduction is provably unchanged.
(2) Renders the new auditory envelopes and fractal gait timing to a PNG for a
    visual sanity check.

Run:
    ~/miniconda3/envs/phd_codes_v2/bin/python verify_revision_inputs.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils import generate_input_sequences


def base_config(mode="sensorimotor", tempo=0.5, zero_mean=True, random_phase=False,
                envelope=None, gait=None, dt=0.05, duration=10.0):
    exp = {
        "dt": dt, "duration": duration, "mode": mode,
        "random_phase": random_phase, "zero_mean_beat": zero_mean,
        "tempo": {"mode": "single", "value": tempo},
    }
    if envelope is not None:
        exp["auditory_envelope"] = envelope
    if gait is not None:
        exp["gait_variability"] = gait
    return {"experiment": exp}


def expected_legacy(tempo=0.5, zero_mean=True, dt=0.05, duration=10.0):
    """Reconstruct the ORIGINAL (pre-revision) sensorimotor output."""
    n_steps = int(duration / dt)
    t = np.arange(0, duration, dt)
    beat_times = np.arange(0.0, duration, tempo)
    beat_indices = np.round(beat_times / dt).astype(int)
    beat_indices = beat_indices[beat_indices < n_steps]
    beat = np.zeros(n_steps)
    beat[beat_indices] = 1.0
    if zero_mean:
        spp = round(tempo / dt)
        beat = np.where(beat > 0.5, (spp - 1.0) / spp, -1.0 / spp)
    freq = 1.0 / tempo
    saw = 2 * (t * freq - np.floor(0.5 + t * freq))
    vest = 1 - 2 * np.abs(saw)
    return vest, beat


def check_backward_compat():
    print("== backward-compatibility checks ==")
    for zero_mean in (True, False):
        for tempo in (0.5, 0.65):
            cfg = base_config(tempo=tempo, zero_mean=zero_mean, random_phase=False)
            vest, beat = generate_input_sequences(cfg, tempo=tempo, rng=np.random.default_rng(0))
            evest, ebeat = expected_legacy(tempo=tempo, zero_mean=zero_mean)
            assert np.allclose(vest.numpy(), evest, atol=1e-6), f"vest mismatch t={tempo} zm={zero_mean}"
            assert np.allclose(beat.numpy(), ebeat, atol=1e-6), f"beat mismatch t={tempo} zm={zero_mean}"
            print(f"  sensorimotor pulse/none  tempo={tempo} zero_mean={zero_mean}: OK")

    # doublebeat: both channels identical; beat: single channel
    cfg = base_config(mode="doublebeat", zero_mean=True, random_phase=False)
    a, b = generate_input_sequences(cfg, tempo=0.5, rng=np.random.default_rng(0))
    assert np.allclose(a.numpy(), b.numpy()), "doublebeat channels differ"
    print("  doublebeat: both channels identical: OK")

    cfg = base_config(mode="beat", zero_mean=True, random_phase=False)
    out = generate_input_sequences(cfg, tempo=0.5, rng=np.random.default_rng(0))
    assert out.ndim == 1, "beat mode should return a single sequence"
    print("  beat mode: single sequence returned: OK")
    print("  -> default path is unchanged.\n")


def make_figure(path="revision_inputs_check.png"):
    dt, duration, tempo = 0.05, 6.0, 0.5
    t = np.arange(0, duration, dt)
    rng = np.random.default_rng(7)

    envelopes = [
        ("pulse (discrete)", {"shape": "pulse"}),
        ("exp tau=0.05", {"shape": "exp", "width": 0.05}),
        ("exp tau=0.15", {"shape": "exp", "width": 0.15}),
        ("exp tau=0.30 (near-continuous)", {"shape": "exp", "width": 0.30}),
        ("alpha w=0.12 (shallow attack)", {"shape": "alpha", "width": 0.12}),
        ("exp tau=0.12 + jitter", {"shape": "exp", "width": 0.12,
                                   "amplitude_jitter": 0.4, "width_jitter": 0.4}),
    ]

    fig, axes = plt.subplots(len(envelopes) + 2, 1, figsize=(10, 13), sharex=True)

    for ax, (label, env) in zip(axes, envelopes):
        cfg = base_config(tempo=tempo, zero_mean=False, random_phase=False,
                          envelope=env, duration=duration)
        _, beat = generate_input_sequences(cfg, tempo=tempo, rng=np.random.default_rng(7))
        ax.plot(t, beat.numpy(), color="C3", lw=1.4)
        ax.set_ylabel(label, fontsize=8)
        ax.axhline(0, color="k", lw=0.4, alpha=0.3)

    # Fractal vs uniform timing, with vestibular overlaid to show phase-lock.
    for ax, gmode, title in (
        (axes[-2], "none", "uniform timing: vestibular (blue) + beats (red)"),
        (axes[-1], "fractal", "fractal gait timing: vestibular (blue) + beats (red)"),
    ):
        gait = None if gmode == "none" else {"mode": "fractal", "sigma": 0.06, "hurst": 0.85}
        cfg = base_config(tempo=tempo, zero_mean=False, random_phase=False,
                          envelope={"shape": "exp", "width": 0.08}, gait=gait, duration=duration)
        vest, beat = generate_input_sequences(cfg, tempo=tempo, rng=np.random.default_rng(3))
        ax.plot(t, vest.numpy(), color="C0", lw=1.2)
        ax.plot(t, beat.numpy(), color="C3", lw=1.2)
        ax.set_ylabel(title, fontsize=8)
        ax.axhline(0, color="k", lw=0.4, alpha=0.3)

    axes[-1].set_xlabel("time (s)")
    fig.suptitle("Revision-2 input machinery: auditory envelopes + gait timing", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(path, dpi=130)
    print(f"== figure saved -> {path} ==")


if __name__ == "__main__":
    check_backward_compat()
    make_figure()
    print("Step 1 verification complete.")
