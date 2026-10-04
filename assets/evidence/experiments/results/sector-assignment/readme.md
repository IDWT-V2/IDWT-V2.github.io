# Complete sector assignment: executable evidence

This bundle reproduces the complete assignment comparison for the stated
numerical mass prescription. It contains two independent Python scripts,
one shared input file, and the full scientific results of both calculations.
Python 3.11 or later is recommended. Only the Python standard library is used.

## Run the calculations

Keep all six files together. From this directory, run:

```sh
python3 exact-sector-check.py
python3 finite-sector-check.py
```

The order emphasizes the exact uniqueness result; neither script depends on
running the other. Each reads `sector-inputs.json` beside the script and writes
its corresponding result JSON beside the script. Running a checker replaces
that result file. The scripts also work when invoked by path from another
directory. Leave Python assertions enabled.

| File | Content |
| --- | --- |
| `sector-inputs.json` | Fixed observed masses in MeV, numerical sector coefficients, reference counts, and reference values for the finite consistency checks. |
| `exact-sector-check.py` | Rational fourth-power bracketing, exact nearest-rung and accuracy comparisons, and all 120 complete assignments. |
| `exact-sector-results.json` | All 60 particle/depth nearest-rung certificates, all 120 assignment gates, six fully admissible assignments, and every exclusion witness. |
| `finite-sector-check.py` | The separate finite assignment and reference-null calculation, including relaxed normalization controls. |
| `finite-sector-results.json` | Both complete finite score grids, all class summaries, selected predictions, null bounds, and 56 internal checks. |

Each result's `sources` mapping gives SHA-256 fingerprints of its public
checker and input JSON. These fingerprints identify the files actually read.

## Inputs and exact scope

The count is `S(n,d) = binomial(n+d-1,d)` for positive integer `n`, and the
original mass prescription is `m(d,n) = m_e a_d S(n,d)`. The scale-defined
depths are 2, 3, 4, 6, and 10. Their fixed factors are

```text
a_2 = sqrt(2890)
a_3 = sqrt(32 sqrt(7))
a_4 = sqrt(48 / sqrt(7)) / 15
a_6 = a_10 = 1 / 18564
```

The finite script computes these factors through
`a_d = sqrt(g_dd / g_66) / C_d`, using the numerical coefficients and counts
in the input JSON. Here `g_dd` names a sector coefficient; its status as a
physical coupling constant is unresolved. The exact script uses the
corresponding rational fourth powers directly:

```text
a_2^4 = 2890^2
a_3^4 = 7168
a_4^4 = 256 / 39375
a_6^4 = a_10^4 = 1 / 18564^4
```

The five intact blocks are W/Z/H, down/strange/bottom, up/charm/top,
electron/muon, and the separately assigned tau branch. They occupy distinct
depths. All observed masses, including the electron reference, stay fixed.
The tau block is separate because of its proposed sector placement; this does
not assert a separate empirical lepton family.

The exact checker enumerates all 120 permutations before applying the fixed
electron reference and common charged-lepton unit requirements. Its bracket
search doubles the index until it contains the target, then uses integer
binary search and exact rational comparisons. Positive integer indices have
no upper cutoff. Only the two adjacent rungs can minimize absolute logarithmic
error because the counts increase strictly. Comparison of their rational
fourth-power distortions makes both nearest-rung choices and comparisons
against the maintained per-particle errors exact. Decimal logarithms at
70-digit precision are used to display errors and their aggregate RMS.
The recorded decimal benchmark values are treated as rational inputs; exact
arithmetic does not make the empirical masses exact.

Twenty-four permutations pass the electron reference requirement; six also
preserve the common lepton unit. Exactly one meets every maintained
per-particle accuracy: block depths `(2, 3, 4, 6, 10)`. The closest fully
admissible alternative has 8.88588 times its RMS logarithmic error. The
reference count 18564 and the numerical scale recipe are inputs. Bottom's
nearest ordinary rung retains its approximately +8.9194% discrepancy and is
not established as a successful physical assignment.

## Separate finite statistical controls

The finite checker limits **every index to 1 through 100**. It compares 120
block permutations and 100 electron anchor indices, or 12000 candidates in
each of two declared windows. A single common normalization holds the measured
electron mass fixed at each trial; changing that normalization relaxes the
original absolute scale prescription. The four reported classes distinguish
which trials preserve the original absolute units, the shared lepton unit,
both, or all common normalizations (including trials that relax either
restriction). These classes are overlapping subsets.

For each window, the reference null uses eleven independent labelled
log-uniform mass ratios, conditioned on the electron. The primary interval
is `[1, 10^6]` electron masses; `[1, 10^7]` is a sensitivity check. An exact
clipped interval-union formula gives each nearest-rung distance CDF. The
negative sum of its eleven log probabilities has a Gamma(11,1) law for each
fixed complete candidate. The code reports a fixed-candidate lower bound and
a conservative candidate-count union upper bound for the searched tail.
The finite script evaluates these formulas with ordinary floating arithmetic.

For the six candidates preserving both original restrictions, the primary
finite upper bound is `3.0359970011803062e-9`. It conditions on the already
chosen numerical prescription, search domain, window, and reference-null
assumptions; historical selection of the prescription is not calibrated.
Repeated trial indices within comparison triples are allowed in this finite
control. The maintained winner and the closest strict alternative do not
require repeated indices.

These finite null bounds do not apply to the unbounded exact check. The exact
result is conditional uniqueness within the stated prescription and accuracy
criterion. It supplies no probability that the physical model is true, no
zero-probability conclusion, and no physical mass-operator derivation. Other
scale laws, undefined depth-five absolute scaling, neutrino absolute masses,
and the zero-rest-mass photon are outside this comparison.
