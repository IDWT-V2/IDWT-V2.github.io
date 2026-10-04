"""Complete exclusive block assignments with the measured electron mass fixed."""
from itertools import permutations
from pathlib import Path
import hashlib
import json
import math

BUNDLE = Path(__file__).resolve().parent
INPUT_FILE = BUNDLE / 'sector-inputs.json'
INPUT = json.loads(INPUT_FILE.read_text())
MASS = INPUT['benchmark_mev']
assert MASS['electron'] == 0.51099895000
DEPTHS = (2, 3, 4, 6, 10)
BLOCKS = ('bosons', 'down_quarks', 'up_quarks', 'electron_muon', 'tau')
MEMBERS = (('W', 'Z', 'H'), ('down', 'strange', 'bottom'), ('up', 'charm', 'top'), ('muon',), ('tau',))
NAMES = tuple(name for members in MEMBERS for name in members)
COEFFICIENT = {int(k): v for k, v in INPUT['coefficients'].items()}
REFERENCE = {int(k): v for k, v in INPUT['reference_counts'].items()}
FACTOR = {d: math.sqrt(COEFFICIENT[d] / COEFFICIENT[6]) / REFERENCE[d] for d in DEPTHS}
N_MAX = 100
COUNTS = {d: [math.comb(n + d - 1, d) for n in range(1, N_MAX + 1)] for d in DEPTHS}
CENTERS = {d: [math.log(FACTOR[d] * count) for count in COUNTS[d]] for d in DEPTHS}
PERMUTATIONS = list(permutations(DEPTHS))
MAINTAINED = (2, 3, 4, 6, 10)
MAINTAINED_ID = PERMUTATIONS.index(MAINTAINED)
CHECKS = []


def check(name, condition):
    CHECKS.append({'name': name, 'passed': bool(condition)})
    if not condition:
        raise AssertionError(name)


def covered_fraction(centers, error, upper):
    total = 0.0
    last_low = last_high = None
    for center in centers:
        low, high = max(0.0, center - error), min(upper, center + error)
        if low >= high:
            continue
        if last_high is not None and low <= last_high:
            last_high = max(high, last_high)
        else:
            if last_high is not None:
                total += last_high - last_low
            last_low, last_high = low, high
    if last_high is not None:
        total += last_high - last_low
    return total / upper


def gamma_tail(x, shape=11):
    term = 1.0
    values = [term]
    for j in range(1, shape):
        term *= x / j
        values.append(term)
    return math.exp(-x) * math.fsum(values)


def make_row(name, d, anchor, upper):
    centers = [c - anchor for c in CENTERS[d]]
    observed = math.log(MASS[name] / MASS['electron'])
    index = min(range(N_MAX), key=lambda j: abs(centers[j] - observed))
    error = abs(centers[index] - observed)
    probability = covered_fraction(centers, error, upper)
    assert 0 < probability <= 1.0 + 1e-13
    probability = min(1.0, probability)
    return {
        'particle': name, 'depth': d, 'index': index + 1,
        'observed_mev': MASS[name],
        'predicted_mev': MASS['electron'] * math.exp(centers[index]),
        'signed_relative_residual': math.expm1(centers[index] - observed),
        'log_error': error, 'probability': probability,
    }


def details(candidate, upper):
    pid, ne, x, rms = candidate
    depths = PERMUTATIONS[pid]
    anchor = CENTERS[depths[3]][ne - 1]
    rows = [make_row(name, d, anchor, upper) for members, d in zip(MEMBERS, depths) for name in members]
    electron_factor = FACTOR[depths[3]] * COUNTS[depths[3]][ne - 1]
    check(f'details-{upper}-{pid}-{ne}:fixed-electron',
          math.isclose((MASS['electron'] / electron_factor) * electron_factor, MASS['electron'], rel_tol=2e-15))
    return {
        'permutation_id': pid, 'block_depths': dict(zip(BLOCKS, depths)),
        'electron_index': ne, 'electron_observed_mev': MASS['electron'],
        'electron_predicted_mev': MASS['electron'], 'electron_relative_residual': 0.0,
        'common_normalization_relative_to_original': math.exp(-anchor),
        'preserves_original_absolute_units': abs(anchor) <= 1e-14,
        'preserves_shared_lepton_unit': FACTOR[depths[3]] == FACTOR[depths[4]],
        'negative_log_product': x, 'fixed_candidate_tail': gamma_tail(x),
        'rms_log_error': rms, 'rows': rows,
        'repeated_indices_in_observed_triples': {
            block: len({row['index'] for row in rows if row['particle'] in members}) != len(members)
            for block, members in zip(BLOCKS[:3], MEMBERS[:3])
        },
    }


def run_window(power):
    upper = math.log(10 ** power)
    baseline_rows = [make_row(name, d, 0.0, upper)
                     for members, d in zip(MEMBERS, MAINTAINED) for name in members]
    baseline = {row['particle']: row['log_error'] for row in baseline_rows}
    records = []
    original_units = []
    shared_leptons = []
    both_gates = []
    no_worse_alternate = []
    grid_x = [[None] * N_MAX for _ in PERMUTATIONS]
    grid_rms = [[None] * N_MAX for _ in PERMUTATIONS]
    for electron_depth in DEPTHS:
        legal_ids = [pid for pid, ds in enumerate(PERMUTATIONS) if ds[3] == electron_depth]
        check(f'window{power}:d{electron_depth}:exclusive-permutation-count', len(legal_ids) == 24)
        for ne in range(1, N_MAX + 1):
            anchor = CENTERS[electron_depth][ne - 1]
            table = {}
            for name in NAMES:
                allowed = [electron_depth] if name == 'muon' else [d for d in DEPTHS if d != electron_depth]
                for d in allowed:
                    table[name, d] = make_row(name, d, anchor, upper)
            for pid in legal_ids:
                ds = PERMUTATIONS[pid]
                rows = [table[name, d] for members, d in zip(MEMBERS, ds) for name in members]
                x = -math.fsum(math.log(row['probability']) for row in rows)
                rms = math.sqrt(math.fsum(row['log_error'] ** 2 for row in rows) / 11)
                candidate = (pid, ne, x, rms)
                records.append(candidate)
                grid_x[pid][ne - 1] = x
                grid_rms[pid][ne - 1] = rms
                same_units = abs(anchor) <= 1e-14
                same_lepton_scale = FACTOR[ds[3]] == FACTOR[ds[4]]
                if same_units:
                    original_units.append(candidate)
                if same_lepton_scale:
                    shared_leptons.append(candidate)
                if same_units and same_lepton_scale:
                    both_gates.append(candidate)
                if pid != MAINTAINED_ID and all(row['log_error'] <= baseline[row['particle']] + 1e-14 for row in rows):
                    no_worse_alternate.append(candidate)
    check(f'window{power}:all-candidates', len(records) == 12000 and all(value is not None for row in grid_x for value in row))
    check(f'window{power}:shared-lepton-candidates', len(shared_leptons) == 1200)
    summaries = []
    for label, group in [('all_common_normalizations', records), ('shared_lepton_unit', shared_leptons),
                         ('original_absolute_units', original_units), ('both_preserved', both_gates)]:
        ordered = sorted(group, key=lambda row: row[2], reverse=True)
        raw = sorted(group, key=lambda row: row[3])
        best, raw_best = ordered[0], raw[0]
        best_other = next(row for row in ordered if row[0] != MAINTAINED_ID)
        raw_other = next(row for row in raw if row[0] != MAINTAINED_ID)
        q = gamma_tail(best[2])
        summaries.append({
            'class': label, 'complete_candidate_count': len(group),
            'distinct_permutation_count': len({row[0] for row in group}),
            'best_calibrated': details(best, upper),
            'best_raw': details(raw_best, upper),
            'best_alternate_permutation_calibrated': details(best_other, upper),
            'best_alternate_permutation_raw': details(raw_other, upper),
            'calibrated_winner_ties_1e12': sum(abs(row[2] - best[2]) <= 1e-12 for row in group),
            'raw_winner_ties_1e12': sum(abs(row[3] - raw_best[3]) <= 1e-12 for row in group),
            'searched_tail_lower_bound': q,
            'searched_tail_upper_bound': min(1.0, len(group) * q),
        })
    maintained = next(row for row in records if row[:2] == (MAINTAINED_ID, 13))
    old = next(w for w in INPUT['finite_reference_checks'] if w['ratio_window'][1] == 10 ** power)
    check(f'window{power}:maintained-tail-reproduced',
          math.isclose(gamma_tail(maintained[2]), old['searched_tail_lower_bound'], rel_tol=2e-12))
    check(f'window{power}:maintained-raw-reproduced',
          math.isclose(maintained[3], old['best_assignment']['rms_log_residual'], rel_tol=2e-12))
    return {
        'ratio_window': [1, 10 ** power], 'maintained': details(maintained, upper),
        'classes': summaries,
        'no_worse_alternate_permutations': [details(row, upper) for row in no_worse_alternate],
        'score_grid': grid_x, 'rms_grid': grid_rms,
        'best_permutation_summaries': [{
            'permutation_id': pid, 'block_depths': dict(zip(BLOCKS, ds)),
            'best_electron_index_calibrated': max(range(N_MAX), key=lambda j: grid_x[pid][j]) + 1,
            'best_electron_index_raw': min(range(N_MAX), key=lambda j: grid_rms[pid][j]) + 1,
            'max_score': max(grid_x[pid]), 'min_rms': min(grid_rms[pid]),
        } for pid, ds in enumerate(PERMUTATIONS)],
    }


check('permutations-exclusive-and-complete', len(PERMUTATIONS) == len(set(PERMUTATIONS)) == 120
      and all(len(set(ds)) == 5 for ds in PERMUTATIONS))
check('maintained-electron-normalization', CENTERS[6][12] == 0.0)
check('overlap-boundary-union', abs(covered_fraction([0.4, 0.48, 1.2, 1.98], 0.1, 2) - 0.3) < 1e-14)
check('outside-centers', abs(covered_fraction([-0.05, 1.05], 0.1, 1) - 0.1) < 1e-14)
windows = [run_window(power) for power in (6, 7)]
report = {
    'status': 'passed',
    'sources': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (Path(__file__), INPUT_FILE)},
    'benchmark_mev': MASS, 'dimensionless_sector_factors': FACTOR,
    'depth_domain': DEPTHS, 'index_range': [1, N_MAX], 'block_order': BLOCKS,
    'permutations': PERMUTATIONS, 'maintained_permutation_id': MAINTAINED_ID,
    'reference_mass_is_fixed': True,
    'electron_is_conditioned_not_an_independent_prediction': True,
    'normalization': 'One common value determined by the same fixed measured electron for each trial depth and index; relative sector factors fixed.',
    'windows': windows, 'checks': CHECKS,
    'scope': 'Finite mass-pattern search with exclusive sectors. Bounds condition on fixed relative numerical recipe; not historical discovery significance or a physical model derivation.',
}
(BUNDLE / 'finite-sector-results.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({'status': 'passed', 'checks': len(CHECKS),
    'primary': [{'class': c['class'], 'count': c['complete_candidate_count'],
      'best_calibrated': [c['best_calibrated']['block_depths'], c['best_calibrated']['electron_index']],
      'best_raw': [c['best_raw']['block_depths'], c['best_raw']['electron_index']],
      'upper_bound': c['searched_tail_upper_bound'],
      'best_other_raw': [c['best_alternate_permutation_raw']['block_depths'], c['best_alternate_permutation_raw']['electron_index'], c['best_alternate_permutation_raw']['rms_log_error']]}
      for c in windows[0]['classes']],
    'no_worse_alternate_count': len(windows[0]['no_worse_alternate_permutations'])},indent=2))
