# Pushing this to GitHub

The repo is already initialised and committed. From the unzipped folder:

```bash
cd monoculture
git init                 # only if .git is missing (the zip excludes it)
git add -A
git commit -m "Guilty by Resemblance: model monoculture inverts collusion detection"

gh repo create monoculture --public --source=. --remote=origin --push
# without the gh CLI: create an empty public repo on github.com, then
#   git remote add origin https://github.com/<you>/monoculture.git
#   git branch -M main && git push -u origin main
```

Then paste the URL into the submission form's optional "Code" field.

## What a reviewer should look at first

| Order | File | Why |
|---|---|---|
| 1 | `DESIGN.md` | Preregistration frozen before any run, with five dated amendments. Amendments 1 and 5 record predictions (P5, R2) that were falsified, and both are reported as failures in the paper. |
| 2 | `reproduce.sh` | Runs every simulation arm end to end in ~12 minutes with no API key, no GPU and no network, then verifies determinism by running arm 4 twice and comparing byte-for-byte. |
| 3 | `processes.py` | The matched construction. `matched_kappa_I` solves for the shared-prior concentration making a pair identical in distribution to a linked pair; `verify_matching` checks it. |
| 4 | `MONOBENCH6.ipynb`, `MONOBENCH7.ipynb`, `MONOBENCH8.ipynb` | The three model-arm notebooks (§§4.7–4.9). Each runs in minutes on a Kaggle T4; v8 needs a Hugging Face read token for gemma. |
| 5 | `results/` | Every JSON report and figure behind the numbers in the paper. |

Only the notebooks whose numbers appear in the paper are kept. Versions 1–5 were development
iterations; what each got wrong, and how the design changed in response, is recorded in
`DESIGN.md` Amendments 4–6, and the files remain in the git history.
