#!/usr/bin/env bash
# Reproduce every result in the paper from scratch. ~12 min, $0, no API key, no GPU.
#
# Also verifies determinism: arm 4 is run twice into separate files and compared
# byte-for-byte, so a claim of reproducibility is checked rather than asserted.

set -euo pipefail
cd "$(dirname "$0")"

echo "=============================================================="
echo " MONOCULTURE -- full reproduction"
echo " python: $(python3 -V 2>&1)"
echo " numpy:  $(python3 -c 'import numpy; print(numpy.__version__)')"
echo " started: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "=============================================================="

mkdir -p results

echo
echo ">>> arms 1-3: matching gate, context observability, homogeneity curve"
python3 experiment.py

echo
echo ">>> arm 2b: lagged coordination"
python3 experiment_lagged.py

echo
echo ">>> arm 4: group-aggregation forms"
python3 aggregation.py

echo
echo ">>> arms 5-7: sensitivity, mechanism, specification"
python3 robustness.py

echo
echo ">>> figures"
python3 fig1_mechanism.py      # main  1: the mechanism, in raw agreement rates
python3 figure2.py             # main  2: which detector shapes invert
python3 fig3_repairs.py        # main  3: two of the three repairs
python3 figs_appendix.py       # appx A1, A2

echo
echo ">>> determinism check"
cp results/report_aggregation.json results/.determinism_first.json
python3 aggregation.py >/dev/null
if cmp -s results/.determinism_first.json results/report_aggregation.json; then
  echo "    PASS: arm 4 reproduced byte-for-byte on a second run"
else
  echo "    ** FAIL: output differs between identical runs -- seeds are not controlling"
  echo "       everything. Do not claim reproducibility until this is fixed."
  diff <(python3 -m json.tool results/.determinism_first.json) \
       <(python3 -m json.tool results/report_aggregation.json) | head -20 || true
  rm -f results/.determinism_first.json
  exit 1
fi
rm -f results/.determinism_first.json

echo
echo "=============================================================="
echo " DONE: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo " outputs in results/"
ls -1 results/
echo "=============================================================="
