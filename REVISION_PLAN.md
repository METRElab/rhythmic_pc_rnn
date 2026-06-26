# Revision 2 plan — "Maternal gait as a (potential) scaffold for rhythm processing"

Annals NYAS, manuscript 5574036. Revised version due **Aug 30, 2026**.
Status: R1 endorses publication. R2 still has major concerns. Editor wants Author
Contributions + Competing Interests sections.

This file is the working plan for the revision: the scientific reframe, the new
experiments, the code to build, and the text changes — each mapped to a reviewer point.

---

## 1. What the reviews actually demand

**Reviewer 1** — satisfied. One leftover (optional, helpful): add a comparison/discussion
vs Tichko et al. 2021 (auditory+vestibular Hebbian dynamical model). Plus small line items
(lowercase π in Fig 2, larger fonts, explain the auditory "two peaks," explain the Fig 4A
error bump ~3000–4000, fix "peaks precede pulse" wording, add refs for internal beat
maintenance). Most already drafted — verify.

**Reviewer 2** — one core objection with several facets, plus citation/clarity asks.

> **Core objection.** The model's whole premise is that auditory input is *discrete*
> (impulse-like footsteps) and that the *continuous* vestibular signal solves the resulting
> temporal-credit-assignment problem. R2 doesn't believe the auditory input is discrete:
> bone-conducted sound reaching the fetus through a fluid environment is heavily low-pass
> filtered → shallow, variable attacks, no sharp onsets. If the auditory input isn't
> discrete, the "vestibular rescues discreteness" story collapses.

Facets of that objection:
- **(A) Envelope shape.** Tested only one shape (sharp binary pulse). Demands we test a
  *range* of envelopes from discrete → continuous (low-pass, shallow/variable attack,
  decay-dominated), and show how much vestibular adds *over auditory alone* across that range.
- **(B) Fractal gait dynamics.** Real gait has well-documented 1/f (fractal) timing
  structure; we used perfectly periodic input. Wants a realistic quasi-steady gait pattern.
- **(C) Generalization.** A causally relevant rhythm model should generalize — across tempos
  and across rhythmic inputs — not work at a single tempo only.
- **(D) Acoustic prominence/systematicity.** Is a footstep loud/systematic enough to matter
  (esp. barefoot, evolutionarily "silent" locomotion)? → mostly a citation/argument matter.
- **(E) Framing.** Soften Abstract/Intro/Discussion (not just Limitations); restore the
  "potential" qualifier in the title; acknowledge social learning; fix beat-vs-surface-rhythm
  framing; add missing references.

---

## 2. The scientific reframe (Jon's steer — the key move)

Do **not** defend the claim that auditory input is discrete. Instead concede the envelope is
uncertain and turn the objection into a strength on two levels:

1. **Robustness.** Show the model's conclusions hold across a *whole family* of auditory
   envelopes (sharp → smeared/low-pass with shallow variable attacks). Vestibular helps
   regardless of envelope shape.

2. **Sharper mechanism — from "discreteness" to "tempo-invariant continuity."** The reason
   vestibular is special is *not* that the auditory signal is discrete. It's that the
   triangular vestibular wave **automatically fills every beat period at any tempo**, whereas
   any *fixed* auditory envelope (whatever its decay/attack) **cannot scale with tempo** — a
   physical footstep decay doesn't know how fast the mother is walking. So even a smeared
   auditory signal cannot replace vestibular *across the range of tempos a person actually
   experiences*.

Jon's three notes that shape this:
- A smeared/decaying auditory signal probably *still* won't let the network learn alone,
  because tanh is too weak a nonlinearity to exploit small differences in a decaying tail.
  *Contingency:* if smearing unexpectedly lets auditory-only succeed, **background noise**
  swamps those small differences and restores the need for vestibular.
- The decisive point is best shown with **tempo**: "the decay wouldn't scale appropriately
  with tempo." Hard to show with a one-tempo model, but doable.
- Fractal gait variation ≈ un-autocorrelated random variation for our purposes, but generate
  the maximally realistic version anyway.

**Decisive comparison (Jon, verbatim intent):** the model *can* do the combined condition
across multiple tempos, but *cannot* do the auditory-only condition across multiple tempos —
and will fail with damped auditory-only even when lightly damped. That contrast is the figure
that makes the point.

---

## 3. Experiments

Codebase facts that ground these:
- `utils.generate_input_sequences` currently emits **only** binary pulses (+ optional
  zero-mean transform, random phase) and a triangular vestibular wave. **No envelope / decay /
  smear / fractal generator exists yet** — that's the main thing to build.
- Modes: `sensorimotor` (triangle vestibular + correlated auditory), `doublebeat` (same signal
  on both channels = *no continuous vestibular carrier*), `beat` (auditory only, not wired into
  `train.py`), `uncorrelated` (triangle + random auditory).
- `train.py` already samples a tempo per round from `tempo.mode: range` → **multi-tempo
  training is already supported**; we just need a wider range and the right architecture.
- `period = tempo / dt`. dt=0.05 → tempo 0.5 s = 10 steps (matches the paper).
- "Auditory-only training" = `doublebeat` mode carrying the (possibly damped) envelope on both
  channels — both channels see the auditory signal, neither sees a tempo-scaled triangle. This
  is the existing "no-vestibular-scaffold" control, now with an envelope knob.

### E0 — Input machinery + baseline reproduction (prerequisite, code)
- Add an **auditory envelope generator** to `utils.py`. Convert the binary beat train into a
  continuous envelope via a per-beat kernel with one ordered "smear" knob plus shape options:
  - `pulse` (current sharp binary) — baseline.
  - `exp_decay(τ)` — sharp attack, exponential decay tail (footstrike + reverberation). τ in
    **absolute seconds**, *not* scaled to tempo (this is the point).
  - `lowpass`/`alpha`/`gammatone-like` — gradual *and* variable attack, peak lower than the
    decay tail, models bone-conduction-through-fluid low-pass (R2's "shallow variable attack").
  - per-beat **variability**: jitter amplitude (loudness) and attack slope.
  Single sweep axis: smear width (≈ τ) as a fraction of the period, from ~0 (sharp) to ~1
  (quasi-continuous).
- Add a **fractal/naturalistic gait-timing generator**: inter-beat intervals with long-range
  (1/f, fractional-Gaussian) correlations around the mean tempo; build **both** the auditory
  envelope and the vestibular triangle from the **same jittered beat times** so they stay
  phase-locked (correlation preserved; only the tempo wanders naturally).
- Reproduce the current headline sensorimotor result (single tempo 0.5) as the control before
  changing anything, so every new result has a baseline to compare against.

### E1 — Auditory envelope sweep, single tempo ("where is the threshold?")
- Sweep the smear knob (sharp → quasi-continuous), at one tempo (0.5 s).
- Two conditions per envelope: **(a) combined** (sensorimotor: envelope + vestibular triangle);
  **(b) auditory-only** (doublebeat with the same envelope on both channels, no triangle).
- Metric: did it learn? — final auditory + vestibular inference error, plus a cross-modal index
  (auditory-only test → power of the vestibular prediction at the beat frequency) and a
  continuation check.
- **Expected:** combined learns for *every* envelope; auditory-only fails for sharp envelopes
  and only starts to succeed once the envelope is smeared enough to be quasi-continuous — that
  crossover is the "threshold." Headline figure: x = envelope (discrete→continuous), two curves
  (combined vs auditory-only), y = learning quality. This is exactly R2's requested
  "vestibular adds over auditory-alone across a range of envelopes."
- **Contingency:** if auditory-only succeeds too readily at moderate smear, add background noise
  to the auditory channel and re-sweep (Jon's point 1).

### E2 — Multi-tempo decisive comparison (the figure that makes the point)
- Train across a **wide** tempo range (e.g., 0.4–0.8 s, step 0.1 → 5 tempos; sample per round).
- Conditions:
  - **(A) Combined, multi-tempo** (sensorimotor): expected to **succeed** — the triangle scales
    with tempo automatically.
  - **(B) Auditory-only, multi-tempo**, damped envelope with **fixed** τ (and a variant using
    the *best* single-tempo τ from E1): expected to **fail** — a fixed envelope can't fill the
    beat interval across a range of tempos.
- **Point:** vestibular is necessary specifically for *tempo-general* rhythm learning, because
  no fixed auditory envelope scales. This answers the generalization ask **and** the envelope
  ask together, and reframes "discreteness" → "tempo-invariant continuity."
- **Main risk / the real unknown:** does combined multi-tempo actually work with this
  architecture? Plan: (1) try the single-layer net across the range; (2) if it can't hold
  multiple tempos, enable the higher layer (`higher_size > 0`, the tempo-state unit — note the
  branch is literally `hierarchical_multi_tempo`); (3) Optuna-tune (`optuna_tune.py` exists) for
  the multi-tempo range. Jon bets it works; budget time to make it work.

### E3 — Robustness of the headline results to realistic envelopes
- Re-run the paper's two headline phenomena with 2–3 realistic envelopes (exp-decay, low-pass
  shallow-attack, amplitude/attack-jittered) instead of the sharp pulse:
  - **Fig-4 cross-modal:** auditory-only test still spontaneously generates the vestibular
    waveform (the "urge to move").
  - **Fig-5 continuation:** internal rhythm still continues after input stops.
- Deliverable: "we tested a variety of acoustic envelope shapes and the conclusions hold."

### E4 — Fractal / naturalistic gait timing
- Train combined with fractal-jittered beat times (auditory + correlated vestibular from the
  same times). Show learning + the headline phenomena survive naturalistic tempo wander.
- Optional contrast: fractal vs white IBI jitter (the Limitations already speculates that
  naturalistic microtiming might even *strengthen* learning — worth a sentence of evidence).
- Answers R2 concern (B). Cheap once E0's generator exists.

### Feasibility
All runs are small (≈200 steps/round × tens of inference steps × 60–64 units). Full sweep
(≈8 envelopes × 2 conditions × {single, multi}-tempo ≈ 30–40 short trainings) is hours of CPU,
not days. Reuse `experiments/paper_revision/`.

---

## 4. Code to build (in priority order)

1. `utils.py`: `generate_auditory_envelope(...)` + `fractal_beat_times(...)`; wire an
   `auditory_envelope` / `gait_variability` block into `generate_input_sequences` for all modes.
   Keep `pulse` as default so existing configs/results are unchanged.
2. Configs under `configs/`: `config_envelope_sweep.yaml`, wide-range multi-tempo configs for
   combined and doublebeat, a fractal config. Mirror the current best params
   (assoc 64, n_inference 5, 10k rounds) so results are comparable.
3. A small **driver/metric script** (`run_envelope_sweep.py`) that loops envelopes × conditions,
   trains, computes the "did it learn" + cross-modal metrics, and dumps a tidy table/`.npz`.
4. `visualization/`: one new figure — envelope sweep (combined vs auditory-only) and the
   multi-tempo combined-vs-damped-auditory comparison. The `make_figures` / `paper_figures`
   pipeline already has the plumbing to extend.
5. (Optional) wire `beat` mode into `train.py`, or just rely on `doublebeat`+envelope for the
   auditory-only condition.

Env: `~/miniconda3/envs/phd_codes_v2/bin/python` (torch 2.8.0, numpy 2.3.1, scipy, yaml).

---

## 5. Text / response-letter changes (mapped to reviewer points)

**Framing (R2-E):**
- Title: restore "**potential** scaffold."
- Abstract/Intro/Discussion: stop asserting auditory rhythm is discrete and that we *demonstrate*
  scaffolding. Recast around (i) robustness to envelope shape and (ii) tempo-invariant continuity.
  Reframe the credit-assignment sentence: it's not "music is discrete" but "the vestibular signal
  is a continuous, *tempo-scaled* carrier no fixed auditory envelope can match."
- Acknowledge social learning explicitly (Polak & Doumbia 2022) as the dominant postnatal force;
  position prenatal gait as an *initial* scaffold only.
- Beat vs surface rhythm: concede higher-order metrical representations are out of scope
  (London 2012; Lenc et al. 2021; London, Polak & Jacoby 2017).

**New results text (R2-A,B,C):** add E1/E2/E3/E4 to Results with the new figure(s); fold the
"discreteness is uncertain → but conclusion is robust and the real driver is tempo-scaling"
argument into Discussion, not just Limitations.

**Citations to add:** periodicity-in-humans (systematic musicology/ethnomusicology, Intro
35–36); Qirko 2024; Gelat et al. 2025 + Larsson & Falk 2025 (acoustic prominence, line 126);
internal beat maintenance refs in §4.1; metric-switching refs (Discussion ~416); Tichko et al.
2021 comparison (R1).

**Editor:** add Author Contributions + Competing Interests.

**R1 line items:** lowercase π in Fig 2; larger axis fonts on all figures; explain the auditory
double-peak (interaction of internal prediction + reactive response — already in §4.2, make
explicit); explain Fig 4A error bump; fix any remaining "prediction precedes pulse" wording;
report training-episode length and hidden-unit count; repo link.

---

## 6. Suggested order

1. **E0** — build envelope + fractal generators; reproduce baseline. *(unblocks everything)*
2. **E1** — single-tempo envelope sweep → threshold + the combined-vs-auditory-only figure.
3. **E2** — multi-tempo: get combined working (tune if needed), show damped-auditory-only fails.
4. **E3 + E4** — envelope robustness + fractal gait (fast once E0 exists).
5. Draft the new Results figure + caption; then the framing edits and response letter.

Risks: **E2 combined-multi-tempo** is the one genuine unknown — may need the hierarchical layer
and/or Optuna tuning. Everything else is low-risk given the existing infrastructure.
d