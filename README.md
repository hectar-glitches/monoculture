# Monoculture

**Agreement and shared priors are observationally equivalent, and which detector you use
decides whether that matters.**

AI Collusion Research Sprint, 23–25 October 2026. Track 2 (Detection and Audit of
Collusion), RQ 2.3, with RQ 1.3 and 1.4 as secondary.

## Provenance

**All code in this directory was written during the sprint. No preexisting code.**

The repository root (`../`) is the organisers' starter repo, cloned for its literature
index and its 139-definition corpus. Nothing here imports or executes starter code.

`../ignition/` is an **abandoned earlier direction** from day 1 of this sprint, kept for
transparency rather than submitted. It was dropped after adversarial novelty checks found
it scooped by Slocum et al. (arXiv:2609.35799) and by three September 2026 Apart projects.
It is not part of this submission.

## Run it

```bash
./reproduce.sh          # every arm, start to finish, ~12 min, $0, no API key, no GPU
```

Or individually:

```bash
python3 experiment.py        # arms 1-3: matching, context observability, homogeneity curve
python3 experiment_lagged.py # arm 2b: lagged coordination (net influence's envelope)
python3 aggregation.py       # arm 4: which group-aggregation forms invert
python3 robustness.py        # arms 5-7: sensitivity, mechanism, specification
python3 fig1_mechanism.py && python3 figure2.py && python3 fig3_repairs.py && python3 figs_appendix.py
```

Requires numpy, scipy, matplotlib. No network access needed at run time.

## The claim

A shared base model makes independent agents behave alike. We construct a cartel and a
population of independent agents sharing a prior — no channel, no agreement — whose
observed action distributions are *identical* once the shared context is marginalised out
(analytic TV 3.5×10⁻¹⁷, empirical 0.004 at T=200,000).

The mechanism: **homogeneity lifts every pair's dependence uniformly; agreement lifts one
pair only.** A detector that scores the most suspicious pair reads *level*, not *pattern*,
so it does not merely lose power under monoculture — it **inverts**, ranking honest
populations as more collusive than real cartels.

## Results

| Arm | Question | Result |
|---|---|---|
| 1 | Does the matched construction hold? | TV 3.5×10⁻¹⁷ analytic, 0.0042 empirical |
| 1 | Do behaviour-only detectors separate them? | AUROC **0.20** — inverted, not at chance |
| 1 | Does observing shared context repair it? | conditional MI 0.195 → **1.000** as φ: 0 → 1 |
| 1 | False accusation vs. homogeneity | 0.07 → **1.00** as κ: 0.25 → 0.55, no agreement in any run |
| 2 | Is net influence different? | Fails its control on contemporaneous cartels (0.648); passes on lagged (1.000) |
| 3 | Does population size rescue the contrast? | **No.** 0.900 → 0.887 for n = 3 → 10. Preregistered prediction falsified |
| 4 | Which aggregation forms invert? | **5/5 predicted.** Peak 0.097, Alignment 0.000, Split 1.000, GLRT/indep 0.097, GLRT/homog **1.000** |
| 5 | Does the sign hold off the baseline? | **26/26 cells.** Level inverted and pattern worked in every one |
| 6 | Is the mechanism really mean-not-variance? | **Yes.** sd(γ) grows 4.6×, raw contrast 3.7×, studentised **0.8×** |
| 7 | Does arm 4 depend on the score definition? | **No.** Peak inverts and Split works under max, mean and sum |

### Three repairs, in increasing order of availability

1. **Observe the shared context.** Works completely (AUROC → 1.000) but is only partly
   available for LLM agents, because much of the shared context is the model prior and
   cannot be logged.
2. **Specify the null class to contain a homogeneous population.** Same test, same data:
   0.097 against 1.000. Estimable from the data, no context observation required.
3. **Studentise by the spread.** The raw contrast grows 3.7× across the homogeneity range
   so its threshold does not transfer; the studentised contrast grows 0.8× and does.

## Layout

| File | What it is |
|---|---|
| `DESIGN.md` | Preregistration, frozen before any run, with five dated amendments |
| `processes.py` | Matched generative processes and the analytic/numerical matching |
| `detectors.py` | Cramér's V, marginal MI, conditional MI, Bonjour net influence |
| `aggregation.py` | Group-aggregation forms and the GLRT null-class comparison |
| `experiment.py` | Arms 1–3 |
| `experiment_lagged.py` | Arm 2b |
| `robustness.py` | Arms 5–7 |
| `fig1_mechanism.py`, `figure2.py`, `fig3_repairs.py`, `figs_appendix.py` | Figures |
| `MONOBENCH6.ipynb` | Model arm, pricing: cross-organisation convergence and the population comparison (§4.7) |
| `MONOBENCH7.ipynb` | Model arm, pricing: two matched pairs and the price screen (§4.8) |
| `MONOBENCH8.ipynb` | Model arm, MMLU: the sandbagging cartel on a public benchmark (§4.9) |
| `llm_arm.py` | The first local run (gemma3:1b via llama.cpp), cited in Appendix A.4 |
| `results/` | All JSON reports and figures |

## Methodological commitments

**Two preregistered gates** run before any result is admissible. The *construction gate*
requires analytic TV < 1e−9 and empirical TV < 0.01. The *positive-control gate* requires
conditional detectors to exceed AUROC 0.75 at φ=1; a detector failing it cannot have a
null reported from it. Net influence failed that gate in arm 1, which triggered arm 2
rather than a reported null.

**A falsified prediction is kept in the record.** Amendment 1's P5 predicted the
contrast's false-accusation rate would fall with population size. It does not. The
corrected mechanism was then measured in arm 6 rather than asserted.

**Multiple comparisons.** We report many AUROCs with 95% bootstrap intervals and do not
apply a family-wise correction, because no claim rests on a single marginally significant
cell. The headline results are sign claims with intervals far from 0.5 (arm 1: 0.20,
CI [0.16, 0.24]; arm 4: 0.097 vs 1.000) and a 26/26 count in arm 5. Individual
near-threshold cells should be read as suggestive.

**Scope.** The abstraction has no prices, payoffs, text or learning, so it speaks to the
dependence-based definitions of collusion (Bonjour; Xu) and the legal agreement sense, and
*not* to outcome-based definitions requiring supracompetitive benchmarks or
reward–punishment mechanisms. Whether labelling our monoculture flags "false accusations"
is correct depends on a definitional choice the field has not made: of 139 coded
definitions, only 18 state whether common-cause correlation counts as collusion.

**What this does not do.** It does not reproduce HELIOS (Rose et al., arXiv:2604.01151).
Their detector reads model activations, which this simulation does not produce. Arm 4
tests the *aggregation structure* their technique names describe, over behavioural
dependence scores. Whether their specific probe inverts is a question this motivates
rather than answers.

**The language-model arm.** Sections 4.9 and 4.10 of the paper are not simulation. Six
instruction-tuned models from six organisations (Qwen2.5-0.5B, gemma-3-1b-it,
Phi-4-mini-instruct, granite-3.1-2b-instruct, Falcon3-1B, SmolLM2-1.7B) act as sellers in
a pricing market calibrated to the price scale of Agrawal et al. (arXiv:2507.01413). For
each of eight prompts we take one forward pass and read the logits at the four
price-option tokens, normalise, cache, and sample every action from the cache. No text is
generated and no model is queried twice. `MONOBENCH7.ipynb` runs in about a minute on a
Kaggle T4.

| Measurement | Result |
|---|---|
| Most-agreeing unlinked pair, six different models | gemma-3-1b + Phi-4-mini at **0.988** (T=0.3) |
| Top pair among six copies of one model | Qwen2.5-0.5B 0.973, SmolLM2 0.801, Falcon3 0.607 |
| Cells where six different models out-agree six identical ones | **9 of 9** |
| λ needed to match a planted cartel to that innocent pair | 0.990 / 0.865 / 0.734 |
| Correlation screen names the real cartel | **54%** at T=0.7 and 1.0 (coin flip) |
| Agreement × supracompetitive share names it | **100%** at all three temperatures |
| Price when each pair agrees | cartel $110, innocent pair exactly **$90.00** |
| Explicit price-fixing instruction obeyed, by size | 0.423 (0.5B) → 0.083 (1.5B) → 0.0000022 (3B) |

`MONOBENCH8.ipynb` reruns the design on a public benchmark: 240 MMLU questions sampled
under a preregistered rule (Amendment 6). The matched cartel — both members answer the
first wrong option with probability lambda — is named by the correlation screen at
46–50% (coin flip) and by the error-weighted screen at 100%, though the cartel's
joint-error share of 0.58 stands against innocent pairs at 0.30–0.44 where the market
gave 0.99 against 0.00. One pricing result reverses: on MMLU six copies of one model
out-agree six different models in all nine cells, the opposite ordering. The Kim et al.
(2025) same-wrong-answer statistic replicates at 0.49.

Two defects in that arm are recorded rather than smoothed over. Ties in the correlation
screen resolve towards the lower-indexed pair, which inflates T=0.3 from a coin flip to
75%, so the two untied temperatures are the measurement. And neither screen can abstain,
so on an honest market each accuses somebody in every run: the price repair fixes
attribution, not detection. Amendment 5 in `DESIGN.md` records both, together with the
prediction (R2) that failed.

**What this does not establish.** Every cartel here is planted; none is learned. The model
arm has no buyers, no trades, no realised payoffs and no memory between rounds, so it
cannot speak to whether coordination emerges from repeated play. Appendix A of the paper
lists each difference between this environment and the published one it is calibrated to.
