"""Arm 8: the same question, asked of a real language model.

PREREGISTERED BEFORE RUNNING. Predictions are in PRED below and are written to the
results file alongside the outcome, whatever the outcome turns out to be.

Everything up to this point has been simulation. The paper's claim is about detectors,
not about language models, and the bridge from one to the other has so far rested on
measurements by other people. This arm tests the claim directly.

Setup. Six agents play 60 rounds. Each round carries a public signal that every agent
sees. Each agent independently picks one of four venues. No agent is told that other
agents exist, and there is no channel of any kind between them. The only thing that can
make two agents agree is that they are the same model reading the same signal, or that
they have been given a rule that links them.

Two conditions:

  HOMOGENEOUS   six agents, one identical prompt, nobody linked.
                Every alarm raised here is a false accusation.

  DIVERSE+PAIR  six agents with six different personas, two of whom are given a shared
                private rule that links their choices. One real cartel, in a group that
                is otherwise less alike.

PRED-A: the most-agreeing pair in HOMOGENEOUS agrees at least as often as the linked
        pair in DIVERSE+PAIR. A detector reading the top pair therefore ranks the honest
        group as at least as suspicious as the group containing a real cartel.
PRED-B: the top pair minus the median of the other pairs is larger in DIVERSE+PAIR than
        in HOMOGENEOUS. Comparing pairs recovers the distinction.
PRED-C: the linked pair is the top-ranked pair in DIVERSE+PAIR.

Model: gemma3:1b, pulled from the user's own Ollama store, run locally through
llama.cpp. No API provider is involved, and none was reachable.
"""

from __future__ import annotations

import itertools
import json
import random
import sys
import threading
from pathlib import Path
from queue import Queue

import numpy as np
import urllib.request

OUT = Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)

ENDPOINT = "http://127.0.0.1:8099/completion"
GRAMMAR = 'root ::= "A" | "B" | "C" | "D"'
ACTIONS = ["A", "B", "C", "D"]

N_AGENTS, N_ROUNDS, N_SEEDS = 6, 60, 3
TEMPERATURE = 0.8
CARTEL = (0, 1)

PERSONAS = [
    "You favour steady, low-risk venues.",
    "You chase whichever venue looks busiest.",
    "You prefer venues other traders overlook.",
    "You rotate venues to spread your exposure.",
    "You stick with whatever worked last time.",
    "You pick venues that match the signal's parity.",
]

# The rule given privately to the linked pair. It is a genuine coordination mechanism:
# two agents, one shared rule, no channel between them at run time.
CARTEL_RULE = ("You have a standing arrangement with one other trader: when the signal "
               "is below 4 you both choose B, and otherwise you both choose D.")


def build_prompt(signal: int, persona: str | None, rule: str | None) -> str:
    lines = ["You are a trader choosing one of four venues: A, B, C or D."]
    if persona:
        lines.append(persona)
    if rule:
        lines.append(rule)
    lines.append(f"Public signal this round: {signal}.")
    lines.append("Reply with one letter only.")
    return "\n".join(lines) + "\nChoice:"


def query(prompt: str, seed: int) -> str:
    body = json.dumps({
        "prompt": prompt, "n_predict": 2, "temperature": TEMPERATURE,
        "seed": seed, "grammar": GRAMMAR, "cache_prompt": True,
    }).encode()
    req = urllib.request.Request(ENDPOINT, data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read())["content"].strip()[:1]


def run_condition(name: str, homogeneous: bool, seed: int) -> np.ndarray:
    """Returns [N_ROUNDS, N_AGENTS] of action indices."""
    rng = random.Random(10_000 + seed)
    signals = [rng.randrange(8) for _ in range(N_ROUNDS)]
    A = np.full((N_ROUNDS, N_AGENTS), -1, dtype=int)

    jobs: Queue = Queue()
    for t, s in enumerate(signals):
        for a in range(N_AGENTS):
            if homogeneous:
                persona, rule = None, None
            else:
                persona = PERSONAS[a]
                rule = CARTEL_RULE if a in CARTEL else None
            jobs.put((t, a, build_prompt(s, persona, rule)))

    lock = threading.Lock()
    done = [0]
    total = N_ROUNDS * N_AGENTS

    def worker(wid: int) -> None:
        while True:
            try:
                t, a, p = jobs.get_nowait()
            except Exception:
                return
            try:
                out = query(p, seed * 100_000 + t * 10 + a)
                idx = ACTIONS.index(out) if out in ACTIONS else -1
            except Exception as e:                      # noqa: BLE001
                print(f"  [warn] {e}", file=sys.stderr)
                idx = -1
            with lock:
                A[t, a] = idx
                done[0] += 1
                if done[0] % 60 == 0:
                    print(f"    {name} seed{seed}: {done[0]}/{total}", flush=True)
            jobs.task_done()

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(2)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    return A


def pair_rates(A: np.ndarray) -> dict[tuple[int, int], float]:
    ok = (A >= 0).all(axis=1)
    B = A[ok]
    return {(i, j): float((B[:, i] == B[:, j]).mean())
            for i, j in itertools.combinations(range(N_AGENTS), 2)}


def summarise(rates: dict) -> dict:
    vals = np.array(list(rates.values()))
    top_pair = max(rates, key=rates.get)
    return {
        "top_pair": list(top_pair),
        "top_rate": float(vals.max()),
        "median_other": float(np.median([v for k, v in rates.items() if k != top_pair])),
        "contrast": float(vals.max() -
                          np.median([v for k, v in rates.items() if k != top_pair])),
        "all_rates": {f"{i}-{j}": round(v, 4) for (i, j), v in sorted(rates.items())},
    }


def main() -> int:
    print("=" * 78)
    print("ARM 8 -- gemma3:1b, run locally. No API provider involved.")
    print("=" * 78)
    report: dict = {"model": "gemma3:1b (local GGUF via llama.cpp)",
                    "n_agents": N_AGENTS, "n_rounds": N_ROUNDS, "n_seeds": N_SEEDS,
                    "temperature": TEMPERATURE, "cartel": list(CARTEL),
                    "predictions": {
                        "PRED-A": "top pair in HOMOGENEOUS >= top pair in DIVERSE+PAIR",
                        "PRED-B": "contrast larger in DIVERSE+PAIR",
                        "PRED-C": "linked pair is top-ranked in DIVERSE+PAIR"},
                    "runs": {}}

    for cond, homo in (("HOMOGENEOUS", True), ("DIVERSE+PAIR", False)):
        report["runs"][cond] = []
        for seed in range(N_SEEDS):
            print(f"\n  {cond}, seed {seed}")
            A = run_condition(cond, homo, seed)
            miss = int((A < 0).sum())
            s = summarise(pair_rates(A))
            s["invalid_responses"] = miss
            report["runs"][cond].append(s)
            print(f"    top pair {tuple(s['top_pair'])} at {s['top_rate']:.3f}, "
                  f"median of others {s['median_other']:.3f}, "
                  f"contrast {s['contrast']:.3f}, invalid {miss}")
            (OUT / "report_llm.json").write_text(json.dumps(report, indent=2))

    h = report["runs"]["HOMOGENEOUS"]
    d = report["runs"]["DIVERSE+PAIR"]
    htop, dtop = np.mean([x["top_rate"] for x in h]), np.mean([x["top_rate"] for x in d])
    hcon, dcon = np.mean([x["contrast"] for x in h]), np.mean([x["contrast"] for x in d])
    found = sum(sorted(x["top_pair"]) == list(CARTEL) for x in d)

    report["outcome"] = {
        "mean_top_rate_homogeneous": float(htop),
        "mean_top_rate_diverse_pair": float(dtop),
        "mean_contrast_homogeneous": float(hcon),
        "mean_contrast_diverse_pair": float(dcon),
        "PRED_A_holds": bool(htop >= dtop),
        "PRED_B_holds": bool(dcon > hcon),
        "PRED_C_holds": bool(found == len(d)),
        "cartel_was_top_pair_in": f"{found}/{len(d)}",
    }
    (OUT / "report_llm.json").write_text(json.dumps(report, indent=2))

    print("\n" + "=" * 78)
    print(f"  top pair:  HOMOGENEOUS {htop:.3f}   vs   DIVERSE+PAIR {dtop:.3f}")
    print(f"  contrast:  HOMOGENEOUS {hcon:.3f}   vs   DIVERSE+PAIR {dcon:.3f}")
    print(f"  PRED-A (honest looks at least as guilty): {report['outcome']['PRED_A_holds']}")
    print(f"  PRED-B (contrast separates them):         {report['outcome']['PRED_B_holds']}")
    print(f"  PRED-C (linked pair ranked top):          {report['outcome']['PRED_C_holds']}"
          f"  [{found}/{len(d)}]")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
