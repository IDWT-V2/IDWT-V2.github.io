"""Exact rational certificates for sector and mode-index uniqueness.

Keep sector-inputs.json beside this standard-library Python script. The output
replaces uniqueness-theorem-certificates.json in the same directory.
"""
from fractions import Fraction
from pathlib import Path
import hashlib
import json
import math

BUNDLE = Path(__file__).resolve().parent
INPUT = BUNDLE / 'sector-inputs.json'
MASS = {key: Fraction(str(value)) for key, value in json.loads(INPUT.read_text())['benchmark_mev'].items()}
FACTORS4 = {2: Fraction(2890 ** 2), 3: Fraction(7168),
            4: Fraction(256, 39375), 6: Fraction(1, 18564 ** 4),
            10: Fraction(1, 18564 ** 4)}
MAINTAINED = {'W': (2, 76), 'Z': (2, 81), 'H': (2, 95),
              'down': (3, 1), 'strange': (3, 4), 'bottom': (3, 17),
              'up': (4, 3), 'charm': (4, 20), 'top': (4, 72),
              'electron': (6, 13), 'muon': (6, 35), 'tau': (10, 23)}

def count(n, d):
    return math.comb(n + d - 1, d)

def prediction4(n, d):
    return MASS['electron'] ** 4 * FACTORS4[d] * count(n, d) ** 4

certificates = []

def less(name, left, right):
    left, right = Fraction(left), Fraction(right)
    difference = right - left
    assert difference > 0, name
    certificates.append(dict(name=name, left=str(left), right=str(right),
        positive_difference=str(difference), passed=True))

for d in [2, 3]:
    less(f'electron-first-rung-d{d}', 1, FACTORS4[d])
less('electron-d4-lower', FACTORS4[4], 1)
less('electron-d4-upper', 1, FACTORS4[4] * count(2, 4) ** 4)
assert count(13, 6) == 18564
assert FACTORS4[6] * count(13, 6) ** 4 == 1
less('electron-d10-lower', count(7, 10), 18564)
less('electron-d10-upper', 18564, count(8, 10))
assert {d for d in FACTORS4 if FACTORS4[d] == FACTORS4[6]} == {6, 10}

gap_rows = []
for d, low, high in [(3, 45, 46), (4, 58, 59)]:
    assert high == low + 1
    lower_ratio4 = prediction4(low, d) / MASS['W'] ** 4
    upper_ratio4 = prediction4(high, d) / MASS['W'] ** 4
    less(f'W-d{d}-below-half-percent-window', lower_ratio4, Fraction(199, 200) ** 4)
    less(f'W-d{d}-above-half-percent-window', Fraction(201, 200) ** 4, upper_ratio4)
    less(f'W-d{d}-simple-lower-certificate', lower_ratio4, Fraction(81, 100) if d == 3 else Fraction(4, 5))
    less(f'W-d{d}-simple-upper-certificate', Fraction(26, 25) if d == 3 else Fraction(41, 40), upper_ratio4)
    gap_rows.append(dict(depth=d, lower_index=low, upper_index=high,
        lower_count=count(low, d), upper_count=count(high, d),
        lower_prediction_ratio_fourth=str(lower_ratio4), upper_prediction_ratio_fourth=str(upper_ratio4)))
less('up-d3-first-rung-exceeds-double-target', 16, prediction4(1, 3) / MASS['up'] ** 4)
for name, upper in [('W', Fraction(201, 200)), ('up', Fraction(2))]:
    d, n = MAINTAINED[name]
    ratio4 = prediction4(n, d) / MASS[name] ** 4
    distortion4 = max(ratio4, 1 / ratio4)
    less(name + '-original-log-error-implies-outer-window', distortion4, upper ** 4)

nearest = []
for name, (d, n) in MAINTAINED.items():
    target8 = MASS[name] ** 8
    center4 = prediction4(n, d)
    lower_product4 = prediction4(n - 1, d) * center4 if n > 1 else None
    upper_product4 = center4 * prediction4(n + 1, d)
    if lower_product4 is not None:
        less(name + '-above-lower-log-midpoint', lower_product4, target8)
    less(name + '-below-upper-log-midpoint', target8, upper_product4)
    nearest.append(dict(particle=name, depth=d, index=n, count=count(n, d),
        target_eighth_power=str(target8),
        lower_neighbor_product_fourth=None if lower_product4 is None else str(lower_product4),
        upper_neighbor_product_fourth=str(upper_product4),
        strict_lower_boundary=n > 1, strict_upper_boundary=True))

report = dict(status='passed', exact_arithmetic='fractions.Fraction; no floating acceptance decisions',
    sector_uniqueness_tolerances={'W_relative_error': '1/200', 'up_relative_error': '1'},
    sector_uniqueness_requires_fixed_electron_and_shared_lepton_units=True,
    sector_factors_fourth={str(d): str(a) for d, a in FACTORS4.items()},
    electron_counts={'d6_n13': count(13, 6), 'd10_n7': count(7, 10), 'd10_n8': count(8, 10)},
    W_gap_certificates=gap_rows, unique_nearest_indices=nearest,
    strict_rational_inequalities=certificates, inequality_count=len(certificates),
    sources={file.name: hashlib.sha256(file.read_bytes()).hexdigest() for file in [Path(__file__), INPUT]},
    scope='Conditional sector uniqueness and unique integer indices at the original per-particle log errors; no probability or physical selection law.')
(BUNDLE / 'uniqueness-theorem-certificates.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({'status': report['status'], 'strict_rational_inequalities': len(certificates),
                  'unique_nearest_indices': len(nearest), 'gap_counts': gap_rows}, indent=2))
