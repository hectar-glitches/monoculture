# Monoculture: agreement and shared priors are observationally equivalent

**AI Collusion Research Sprint, 23–25 October 2026. Track 2 (Detection and Audit), RQ 2.3,
with RQ 1.3/1.4 as the detector baselines.**

Status: PREREGISTERED. Frozen before any run. Amendments appended and dated, never edited in.

---

## 1. Why this question and not the obvious ones

We ran adversarial novelty checks on four candidate projects before committing. All four were
already done, three of them by teams in this same sprint:

| Candidate | Killed by |
|---|---|
| Plus-factor battery + false-positive rates | `ayodejiades/collusion-plus-factors` (preregistered 14–17 Oct 2026); Tailor, *Audit the Whisper*, reports full ROC with 10,000 honest audits |
| Monitoring belief causes coordination | Alnuhait et al., *Covert Assistance* (arXiv:2609.39050, 30 Sep 2026) |
| Ignition / first unsolicited contact | Slocum et al. (arXiv:2609.35799, 11 Sep 2026), whose Step 2 is literally "reaching out"; three Sep-2026 Apart projects independently found discovery-without-ignition |
| Pairwise net influence on LLM agents | `Raj-Taware/collusion-net-influence`, complete with AUROCs |

Every one of those checks surfaced **the same unsolved residual**, independently:

- Raj-Taware's calibrated net influence still flags *innocent* same-model pairs in **7 of 10
  games**. His conclusion: *"Calibration removes chance correlation, not correlation from a
  shared model. Separating those needs evidence beyond the moves."*
- Xu et al. (arXiv:2607.26385) measure same-model residual correlation at **+0.053** versus
  **+0.0001** across models, and state that lawful multi-identity operation and conspiracy are
  *"behaviourally indistinguishable"* in their setting.
- Cartel-screening economics reports real-world false-positive rates of **6.9–14.6%**
  (Imhof & Wallimann), against the near-zero rates claimed in the LLM literature.

Nobody has asked whether that residual is a *calibration problem* or an *identifiability
problem*. We claim it is the second, and that this is provable rather than merely measurable.

## 2. The claim

**Behaviour-only collusion detection is identifiable only to the extent that the auditor
observes the context the agents share.** When that context is unobserved, there exists, for
every agreement process, a shared-prior process with no agreement whatsoever that induces an
*identical* distribution over observed actions. No statistic computed from actions alone can
separate them, at any sample size, however well calibrated.

For LLM agents this is not a corner case. The shared context includes the **model prior**,
which an auditor can never fully log. Two deployments of one base model are permanently in the
unobserved-context regime. So as agent deployments converge on fewer base models, false
accusation rates rise — detection gets harder exactly as markets get more concentrated.

## 3. Formal construction

`n` agents, `T` rounds, action set `A` with `|A| = k`. A context `s_t ∈ S`, `|S| = m`, follows
a Markov chain with persistence `p` (markets have memory). A focal map `f: S → A` assigns each
context a salient action, balanced so each action is focal equally often.

Shared-prior policy, concentration `κ`:

    π(a | s) = κ                 if a = f(s)
             = (1−κ)/(k−1)       otherwise

**Process I — independent agents, shared prior, no agreement.**
Every agent draws `aᵢ_t ~ π(·|s_t)` independently. Pairs are correlated purely through the
common context and the common policy. This is monoculture: lawful, and the null that matters.

**Process C — explicit agreement.**
A designated pair coordinates: with probability `λ` both play a jointly drawn `c_t ~ Unif(A)`;
otherwise both draw independently from `π(·|s_t)`. All other agents are independent.

**The matching.** Write `q = (1−κ)/(k−1)` and let `D(κ) = (1/k)[κ² + (k−1)q²]` be the
per-cell diagonal mass of the context-marginalised pair distribution. Then for any
`(κ_C, λ)` there is a `κ_I` solving

    D(κ_I) = λ/k + (1−λ)·D(κ_C)

and, because both distributions are exchangeable in `A` and normalised, matching the diagonal
forces the off-diagonal to match too. The two processes are then **exactly** equal in
distribution over observed actions with `s` marginalised out.

Conditional on `s`, they differ: under I the pair is independent given `s`; under C it is not.
That is the whole content of the result, and it is the reason the econometric
conditional-independence tests (Bajari–Ye) are the right family and marginal dependence
statistics are not.

**Verification before any claim.** We compute total variation distance between the empirical
matched distributions and require TV < 0.01. If the matching fails numerically, nothing else in
this design is reportable.

## 4. Observability sweep

The auditor observes `s_t` with probability `φ`, and `UNKNOWN` otherwise. `φ = 0` is the
LLM-monoculture regime (shared model prior, unloggable). `φ = 1` is a fully instrumented
market. Sweeping `φ ∈ {0, 0.25, 0.5, 0.75, 1}` turns the identifiability claim into an
engineering curve: **how much context logging buys how much detection power.**

## 5. Detectors

All pairwise, all ranking pairs, scored at run level by the maximum over pairs.

| Detector | Statistic | Family |
|---|---|---|
| D1 Cramér's V | contemporaneous association of `aⁱ_t, aʲ_t` | marginal |
| D2 Marginal MI | `I(aⁱ; aʲ)` | marginal |
| D3 Conditional MI | `I(aⁱ; aʲ | ŝ)` | conditional |
| D4 Net influence | `Γ(i;j) = γ(i→j) − max_k γ(k→j)`, `γ(i→j) = I(aⁱ_t; aʲ_{t+1} | ŝ_t)` | conditional, Bonjour et al. 2022 |

D4 is Bonjour's statistic. We include it because the field's one public LLM application of it
reports exactly the failure this design explains.

## 6. Preregistered predictions

**P1 (headline).** At `φ = 0` on matched pairs, every detector achieves AUROC statistically
indistinguishable from 0.5.

**P2 (the sharp version).** AUROC at `φ = 0` will fall *below* 0.5 — the honest monoculture
population will look **more** collusive than the real cartel. Reason: under I every one of the
`C(n,2)` pairs sits at the matched dependence level, whereas under C only the cartel pair does
and the rest sit lower, so a max-over-pairs score is larger in the honest population. If this
holds, "flag the most suspicious pair" is actively inverted under monoculture.

**P3.** Conditional detectors (D3, D4) recover AUROC monotonically in `φ`; marginal detectors
(D1, D2) do not recover at any `φ`, because they never condition.

**P4.** Holding `φ = 0` and sweeping `κ` with no agreement anywhere, false-accusation rate at a
fixed threshold rises monotonically with `κ`. This is the monoculture curve: the more alike the
agents, the more often honest populations are convicted.

## 7. What would falsify this

- A behaviour-only detector beating 0.5 significantly at `φ = 0` on data where TV < 0.01 →
  either the matching is wrong or the claim is wrong. Either way nothing is reportable until
  resolved.
- TV ≥ 0.01 → construction broken, stop.
- D3/D4 failing to recover at `φ = 1` → the detectors are mis-implemented, not the theory
  vindicated. We require the `φ = 1` positive control to pass before reporting any `φ = 0`
  null.

A null result here is **not** a safe default: the design is built so that the interesting
outcome and the broken-code outcome are distinguishable.

## 8. LLM arm (conditional on inference access)

The simulation establishes the geometry. The LLM arm tests whether real agents instantiate it:
same model + same prompt, same model + different personas, same family + different size,
different provider — **no communication channel in any cell**, so every alarm is a false
positive by construction. Prediction: measured pair dependence declines monotonically along
that axis, reproducing Xu et al.'s +0.053 / +0.0001 gap, and false-accusation rate tracks it.

Scripted spine runs at $0 and is the paper's backbone. The LLM arm is confirmatory.

## 9. Prior work this builds on, and what is new

Builds on: Bonjour et al. (2022) net influence; Xu et al. (2026) cross-agent residual
correlation; Tailor (2025) calibrated detectors; Rose et al. (2026) on AUROC collapse under
distribution shift; Bajari & Ye (2003) conditional-independence tests; Harrington & Imhof
(2022) cartel screening; Ortner, Chassang, Kawai & Nakabayashi on asymptotically safe tests;
Kovacic, Marshall, Marx & White (2011) on plus factors as likelihood ratios; and the two
concurrent sprint repos named in §1, cited as baselines rather than reimplemented.

**New here:** the identifiability result and its matched construction; the reframing of the
shared-model confound from a calibration nuisance to a structural limit; the
context-observability recovery curve as a design target for monitoring infrastructure; and the
prediction that model homogeneity inverts max-pair detectors rather than merely degrading them.

---

## AMENDMENT 1 — 23 October 2026, after arms 1–3

Appended, not edited in, per §0. Three preregistered predictions held, one was
falsified, and one detector needed a second arm before its null could be reported.

**P1 held in a stronger form than stated.** At `φ = 0` no detector was merely at chance.

**P2 held.** AUROC at `φ = 0` was 0.20 (D1), 0.20 (D2), 0.20 (D3) — far below 0.5. Honest
monoculture populations are ranked *more* suspicious than real cartels, because every pair
sits at the matched dependence level under I while only the cartel pair does under C, so a
max-over-pairs score is systematically larger in the innocent population.

**P3 held for D3, and required a second arm for D4.** D3 recovered monotonically in `φ`
(0.195 → 0.891 → 0.992 → 1.000 → 1.000). D1 and D2 were flat at ≈0.20 across all `φ`, as
predicted, since they never condition. D4 failed its `φ = 1` positive control at AUROC 0.648,
which §7 forbids reporting a null from. Diagnosis: net influence is a *lagged* statistic and
the arm-1 cartel coordinated within a round, so D4 was outside its design envelope rather
than broken. Arm 2 rebuilt the cartel to coordinate across time, with the matching solved
numerically on the lagged pair distribution (TV = 0.0051). D4 then passed at AUROC 1.000.

**Caveat on cross-arm comparison, which we did not anticipate and must state.** Each arm's
matching targets one statistic family. Arm 1 matches the contemporaneous pair distribution,
so D1/D2/D3 are validly tested there and D4 is not. Arm 2 matches the lagged pair
distribution, so D4 is validly tested there and D1/D2/D3 are not — their AUROC of 0.000 in
arm 2 is an artifact of a construction that did not target what they measure, and is **not**
reported as a finding.

**New finding, not preregistered.** D4's contrast term `− max_l γ(l→j)` cancels the *mean*
dependence lift that monoculture applies to every pair, which is why D4 discriminates at
AUROC 1.000 even at `φ = 0` in arm 2. This is the only detector tested that defeats the
confound without observing the context.

**P4 held, and generalises past what we predicted.** With no agreement in any run and
thresholds frozen at the 95th percentile of a diverse population, false-accusation rates rose
with prior similarity for every detector: D1/D2/D3 went 0.07 → 0.655 → 1.000 across
`κ = 0.25 → 0.45 → 0.55`, and D4 went 0.045 → 0.120 → 0.365, reaching 0.850 at `κ = 0.85`.
Within-regime calibration is fine (≈0.05 at the calibration point, as constructed); it is
*transfer across diversity regimes* that fails.

**P5 (added during arm 3) — FALSIFIED.** We predicted D4's false-accusation rate under
monoculture would fall steeply with population size, because its contrast is a maximum over
`n − 2` alternatives and `n = 3` leaves only one noisy draw. This was offered as an
explanation for the 7/10 same-model false-flag rate in the one public LLM application of net
influence. It is wrong. Measured rates were 0.900, 0.917, 0.890, 0.867, 0.893, 0.887 for
`n = 3, 4, 5, 6, 8, 10` — flat, not decreasing.

Corrected account: the contrast cancels the mean shift but not the *variance* shift. Higher
prior concentration inflates the spread of the `γ` estimates, so the gap between the largest
and second-largest grows, and a max-over-pairs score rises with `κ` regardless of `n`. The
published 7/10 figure is therefore a threshold-transfer failure, not a small-population
artifact — a fixed absolute threshold (Bonjour's α = 0.05 nats) carried onto same-model
agents.

**Revised conclusion.** The shared-model confound is not one problem. For marginal detectors
it is an identifiability problem, soluble only by observing the shared context. For net
influence it is a calibration problem: discrimination survives, threshold transfer does not.
Since model diversity is not observable in deployment, no fixed threshold is safe in the sense
of Ortner, Chassang, Kawai & Nakabayashi — passed by competitive agents regardless of
environment. That is the gap this work identifies and does not close.

---

## AMENDMENT 2 — 23 October 2026, arm 4 (group aggregation)

Added after a literature check showed that published multi-agent detectors aggregate
per-agent scores into a group verdict using forms the §AMENDMENT 1 mechanism makes direct
predictions about. Predictions were written into `aggregation.py`'s docstring before the
run.

**Predictions.** Level-reading forms (maximum over agents, mean pairwise) invert.
Pattern-reading forms (top-k against the rest, centred maximum) resist. A generalised
likelihood-ratio test inverts or resists according only to its null class: against an
independence null it is a level reader, against a null containing the homogeneous
population it is a pattern reader.

**Result: 5/5 correct**, at `φ = 0`, `n = 6`, 200 runs per class.

| Form | Reads | AUROC |
|---|---|---|
| Peak, max over agents | level | 0.097 [0.071, 0.127] |
| Alignment, mean pairwise | level | 0.000 [0.000, 0.000] |
| Split, top-k against rest | pattern | 1.000 [1.000, 1.000] |
| GLRT, independence null | level | 0.097 [0.071, 0.127] |
| GLRT, homogeneous null | pattern | 1.000 [0.999, 1.000] |

The last two rows are the controlled comparison: identical test on identical data,
differing only in whether the null class contains a homogeneous population. 0.097 against
1.000.

**Second repair, and a better one than §6's.** Amendment 1 concluded that observing the
shared context is the lever. For LLM agents that is only partly available, because much of
the shared context is the model prior and cannot be logged. Correctly specifying the null
can be done from the data alone, with no context observation at all. The paper should lead
with this.

**Scope limit, stated because it bounds the claim.** Rose et al. (arXiv:2604.01151) name
five aggregation techniques — Peak Suspicion, Deception Split, Sorted Concatenation,
Activation Alignment, Asymmetry Probe — with a GLRT reading of Peak in their Appendix R.
Their detector reads model activations, which this simulation does not produce, and their
exact formulas are in sections we could not retrieve. We therefore test the **aggregation
structure those names describe, over behavioural dependence scores**. We do not reproduce
HELIOS and do not claim to break it. The mechanism concerns how a group verdict is formed
from per-pair evidence, which is invariant to whether that evidence came from activations
or actions; whether their specific probe inverts is an open question this work motivates
rather than answers.

---

## AMENDMENT 3 — 24 October 2026, arms 5–7 (validation)

Added to close three gaps a reviewer would find: every number sat at one point in
parameter space, Amendment 1's corrected mechanism was asserted rather than measured, and
arm 4's per-agent score was our choice rather than Rose et al.'s.

**Arm 5, parameter sensitivity. 26/26.** Varying `k ∈ {3,4,6,8}`, `m ∈ {4,8,16}`,
persistence `∈ {0,0.3,0.7,0.9}`, `T ∈ {100,200,400,800}`, `n ∈ {3,4,6,8}`,
`λ ∈ {0.15,0.25,0.35,0.50}` and `κ_C ∈ {0.30,0.40,0.50}` one at a time, the level-reading
statistic's CI lay entirely below 0.5 in **all 26 cells** and the pattern-reading
statistic's entirely above it in **all 26**. The inversion is not knife-edge.

Two gradients worth reporting. The inversion *deepens* with population size (0.347 at
`n=3` to 0.079 at `n=8`), as the mechanism predicts: more pairs means a larger maximum
over the honest population's uniformly lifted pairs. And the pattern statistic weakens at
short horizons (0.899 at `T=100`) and weak cartels (0.870 at `λ=0.15`), which bounds the
repair rather than undermining it.

**Arm 6, mechanism verification — and a third repair.** Amendment 1 claimed the contrast
cancels the uniform lift's mean but not its variance. Measured, under pure monoculture
with no agreement anywhere: mean `γ` grows 22.5× from `κ=0.25` to `0.85` while sd(`γ`)
grows 4.6×. The raw contrast grows **3.7×**, so its threshold does not transfer. The
contrast studentised by the spread grows **0.8×**, so it does.

This makes the Amendment 1 explanation a measurement rather than a story, and converts the
falsified P5 into a constructive result: the fix for threshold transfer is not more agents
but division by the spread.

**Arm 7, specification sensitivity.** Arm 4's inversion does not depend on how per-agent
suspicion is defined. Under max, mean and sum aggregation of an agent's pairwise links,
Peak inverted in all three (0.142, 0.000, 0.000) and Split worked in all three (1.000,
0.897, 0.897).

**Reproducibility.** `reproduce.sh` runs every arm end to end and verifies determinism by
running arm 4 twice into separate files and comparing byte-for-byte. It passes. Total
runtime about 12 minutes, no API key, no GPU, no network at run time.

**Multiple comparisons.** No family-wise correction is applied, because no claim rests on
a single marginal cell: the headline results are sign claims with intervals far from 0.5
and a 26/26 count. Individual near-threshold cells are suggestive only. Stated so a reader
does not have to infer it.

---

## 10. Preexisting code

None. Everything under `monoculture/` is written during the sprint. The starter repo is cloned
for literature only and is not imported.

---

## Amendment 4 — 25 October 2026, before running MONOBENCH v6

Written and committed before the v6 run. Both the correction and the predictions are
recorded here so the outcome can be read against them whatever it turns out to be.

**The correction.** v5's matching gate compared the homogeneous group's top pair rate
against the *linked pair's* agreement rate in the diverse group, and certified a match in
all nine cells. The detector does not read the linked pair. It reads the maximum over all
fifteen pairs, and the two coincide only when the linked pair is the largest. v5's own
correct-pair column reported 0% in six of nine cells, which says it was not. The gate
therefore certified a match on a quantity no detector computes, and the AUROC values in
v5 section 7 are not a test of the matched claim. v6 compares maximum to maximum.

**What v5 implies underneath the error.** Holding the diverse group and the linked pair
fixed and varying only lambda, the linked pair was named in 100% of runs at an agreement
of 0.974 and in 0% of runs at 0.804, with the same bracket reappearing at each
temperature. Some unlinked pair in the diverse group therefore agrees at a rate inside
that bracket. v6 measures that pair directly rather than inferring it from the bracket.

**Predictions.**

- **Q1.** In the honest diverse group the top pair spans two organisations and agrees
  above 0.6 at temperature 0.3.
- **Q2.** At least one homogeneous base has a top pair rate below the honest diverse
  group's top pair rate, making the matched comparison infeasible in that cell. Being the
  same model is then not what drives agreement.
- **Q3.** Where the planted link sits below the most convergent innocent pair, the
  detector names that innocent pair in a majority of runs while the pattern statistic
  still separates the two populations.
- **Q4.** The probability mass falling outside the four price tokens rises with model size
  in the compliance ladder, so part of the largest model's apparent non-compliance in v5
  is refusal rather than competitive pricing. v5 renormalised over the four price tokens
  and so could not distinguish the two.

**Other changes.** Seeds raised from 12 to 24. AUROC reported with a bootstrap interval,
because a value of 1.000 from twelve runs without an interval says little. Each cell
records which pair the detector actually names, identified by model, and whether that pair
spans two organisations. Infeasible cells are run and reported rather than skipped, since
they are the cells in which the attribution question has a non-trivial answer. The config
block written to the results JSON in v5 referenced three variables left over from v4 and
recorded a stale lineup; it now records the lineup actually used.

---

## Amendment 5 — 25 October 2026, outcome of MONOBENCH v7

v6 planted the link on pair (0,1), and agent 1 (gemma-3-1b-it) is also half of the most
convergent innocent pair (1,2). Planting the link overwrote gemma's own draws, so the
convergence the cartel was meant to hide behind fell from 0.988 to 0.854 as a side effect
of creating the cartel. v7 removes the overlap by linking (0,4), Qwen2.5-0.5B and
Falcon3-1B, whose natural agreement is 0.011 to 0.142, and matches it to the untouched
rival pair (1,2) inside a single group. The cartel mechanism is a price-fixing agreement
at $110 rather than a uniformly drawn joint action.

**Matching gate.** Passes. Gaps between the two pairs' agreement rates are 0.000, 0.004
and 0.008 at temperatures 0.3, 0.7 and 1.0.

**R1 holds.** Matching required lambda of 0.990, 0.865 and 0.734, all above the
preregistered 0.7.

**R2 fails, at 61% against a predicted 40-60%.** The failure is one point outside the
interval and its direction is explained. At temperature 0.3 both pairs agree on about 198
of 200 rounds, so exact ties are frequent, and the argmax over pairs returns the first
maximum in `itertools.combinations` order, in which (0,4) precedes (1,2). Ties therefore
resolve in favour of the linked pair, and the figure at that temperature is 75%. At 0.7
and 1.0, where ties are rare, the screen names the cartel in 54% of runs. We report 54%
as the clean measurement, record 61% as the preregistered quantity, and note the
tie-breaking rule as a defect in the implementation rather than a property of the screen.

**R3 holds.** The price gap between the two pairs when each agrees is $20.00, $19.66 and
$18.97. The rival pair agrees at exactly $90.00 with 0.0% of rounds supracompetitive at
all three temperatures, so two models from two organisations converge on the competitive
price rather than above it.

**R4 holds, with a scope limit found in the negative control.** The price-augmented screen
names the cartel in 100% of runs at all three temperatures. On the honest market, however,
the correlation screen accuses gemma-and-Phi in 100% of runs and the price screen accuses
Falcon3-and-SmolLM2 in 100% of runs. Neither screen has an abstain option, because both
take an argmax over pairs with no threshold. The repair therefore improves attribution
conditional on a cartel being present and does nothing about deciding whether one is
present. This limit is recorded in the paper rather than left implicit in R4.

---

## Amendment 6 — 26 October 2026, before running MONOBENCH v8

Written and committed before the run.

**Why a new arm.** The pricing environment's eight prompts are our own construction, and
Appendix A.4 of the paper already concedes they are a probe and not a dataset. v8 replaces
them with a public benchmark: MMLU (Hendrycks et al.), the dataset family on which Kim,
Garg, Peng & Garg (2025) measure correlated errors across language models. Every MMLU item
is a question with four options A–D, so the logit-reading instrument carries over
unchanged, and every item has a ground-truth answer, which the pricing arm lacked.

**The framing.** On questions with known answers, the two faces of the paper's confound
are: a pair of agents sandbagging an evaluation must coordinate on wrong answers, while
two honest models that share a training distribution agree without coordinating, and Kim
et al. report that when two models are both wrong they frequently share the same wrong
answer. This is the AI-safety family's sense of collusion (coordination against an
overseer), where the pricing arm spoke to the economics family's sense.

**Dataset selection rule, fixed in advance.** `cais/mmlu`, configuration `all`, test
split, sampled down to 240 questions with `numpy.random.default_rng(2026)`, no other
filtering. The sample is drawn once and the identical questions are used in every
condition.

**Pair selection rule, fixed in advance.** No pairs are named beforehand, because no
honest agreement matrix exists yet for this task. RIVAL is the most-agreeing unlinked pair
in the honest matrix at each temperature. LINKED is the least-agreeing pair sharing no
agent with RIVAL. The cartel mechanism: with probability lambda both members of LINKED
answer the first incorrect option in A–D order, a focal rule computable from the question
alone, so the cartel needs no channel. lambda is solved as before from
lambda = (t − p)/(1 − p).

**Defect fixes carried in from Amendment 5.** Ties in every screen are broken uniformly at
random with a seeded generator, never by pair index. Both screens still lack an abstain
option, and that limit is restated rather than repaired.

**Screens.** S-corr ranks pairs by agreement rate. S-err ranks pairs by agreement
multiplied by the share of rounds on which both members answered incorrectly, the analogue
of the price screen. S-samewrong ranks pairs by the rate of giving the same wrong answer.

**Predictions.**

- **S1.** The most-agreeing unlinked pair among the six models spans two organisations and
  agrees above 0.5 at temperature 0.3.
- **S2.** Innocent agreement concentrates on correct answers: for the top unlinked pair,
  the majority of agreeing rounds are rounds where both are correct.
- **S3.** With LINKED matched to RIVAL on agreement rate, S-corr names the cartel in 40–60%
  of runs.
- **S4 (two-sided, both outcomes reportable).** In the pricing arm the innocent pair was
  never jointly supracompetitive, and the outcome screen reached 100%. On MMLU innocent
  models share wrong answers (Kim et al., 2025), so S-err and S-samewrong may degrade. If
  S-err stays above 90%, the outcome repair generalises to knowledge tasks. If it falls
  materially below, correlated innocent errors pollute the outcome dimension, and the
  fourth repair is task-dependent. We commit to reporting whichever occurs.
- **S5.** The Kim replication statistic: for cross-organisation unlinked pairs, the
  probability of the same wrong answer given both wrong is above 0.3 on this sample.

**Gates.** Matching gate: the two pairs' agreement rates within 0.05 of each other at
every temperature. Positive control: lambda = 0.9 cartel named in over 90% of runs by
S-corr against the honest market; if it fails, nothing else is interpretable.

### Amendment 6 — outcomes, recorded 26 October 2026 after the v8 run

The matching gate passed with gaps of 0.002, 0.005 and 0.006, and the positive control
named the blatant cartel in 100% of runs. The selection rules chose RIVAL =
Phi-4-mini-instruct + granite-3.1-2b-instruct (0.585 at T=0.3) and LINKED = gemma-3-1b-it
+ SmolLM2-1.7B-Instruct (0.401 natural). Matching needed lambda of only 0.29 to 0.31,
against 0.73 to 0.99 in the pricing arm.

All five predictions held. S1: the top unlinked pair is cross-organisation at 0.585.
S2: 70% of its agreement is both-correct. S3: S-corr named the cartel in 46%, 50% and
46% of runs. S4: S-err and S-samewrong named it in 100% of runs at every temperature, so
the outcome repair generalises — with its margin collapsed, since the cartel's joint-error
share of 0.58 stands against innocent pairs at 0.30 to 0.44 where the market gave 0.99
against 0.00. S5: the Kim statistic averaged 0.49 across cross-organisation pairs.

One unpredicted reversal is the run's most instructive result. On MMLU six copies of one
model out-agree the six different models in all nine cells (Falcon3 copies at 0.950
against 0.585 for the best unlinked pair), where the pricing task gave the opposite
ordering in all nine. Which innocent pair looks most guilty is a property of the task and
not of the models. The negative control's S-corr accused the innocent Phi + granite pair
in 100% of honest runs.
