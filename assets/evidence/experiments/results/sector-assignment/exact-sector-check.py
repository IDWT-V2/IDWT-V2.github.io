"""Exact complete-assignment check with no positive-integer mode cutoff."""
from decimal import Decimal, localcontext
from fractions import Fraction
from itertools import permutations
from pathlib import Path
import hashlib
import json
import math

BUNDLE = Path(__file__).resolve().parent
INPUT_FILE = BUNDLE / 'sector-inputs.json'
INPUT = json.loads(INPUT_FILE.read_text())
MASS = {name: Fraction(str(value)) for name, value in INPUT['benchmark_mev'].items()}
ELECTRON = MASS['electron']
DEPTHS = (2, 3, 4, 6, 10)
BLOCKS = ('bosons', 'down_quarks', 'up_quarks', 'electron_muon', 'tau')
MEMBERS = (('W', 'Z', 'H'), ('down', 'strange', 'bottom'), ('up', 'charm', 'top'), ('electron', 'muon'), ('tau',))
MAINTAINED = (2, 3, 4, 6, 10)
FACTORS4 = {2: Fraction(2890 ** 2), 3: Fraction(7168), 4: Fraction(256, 39375),
            6: Fraction(1, 18564 ** 4), 10: Fraction(1, 18564 ** 4)}


def decimal_value(value):
    return Decimal(value.numerator) / Decimal(value.denominator)


def count(n, depth):
    return math.comb(n + depth - 1, depth)


def predicted4(n, depth):
    return ELECTRON ** 4 * FACTORS4[depth] * count(n, depth) ** 4


def nearest(name, depth):
    target4 = MASS[name] ** 4
    if predicted4(1, depth) >= target4:
        lower = upper = selected = 1
    else:
        lower, upper = 1, 2
        while predicted4(upper, depth) < target4:
            lower, upper = upper, 2 * upper
        while upper - lower > 1:
            middle = (lower + upper) // 2
            if predicted4(middle, depth) < target4:
                lower = middle
            else:
                upper = middle
        assert predicted4(lower, depth) < target4 <= predicted4(upper, depth)
        threshold8 = predicted4(lower, depth) * predicted4(upper, depth)
        selected = lower if target4 ** 2 <= threshold8 else upper
    ratio4 = predicted4(selected, depth) / target4
    distortion = max(ratio4, 1 / ratio4)
    with localcontext() as context:
        context.prec = 70
        ratio = decimal_value(ratio4).sqrt().sqrt()
        error = decimal_value(distortion).ln() / 4
        predicted = decimal_value(MASS[name]) * ratio
        row = {
            'particle': name, 'depth': depth, 'index': selected,
            'bracket': [lower, upper],
            'observed_mev': str(decimal_value(MASS[name])),
            'predicted_mev': str(predicted),
            'signed_relative_residual': str(ratio - 1),
            'log_error': str(error),
            'ratio_fourth_power': str(ratio4),
            'distortion_fourth_power': str(distortion),
        }
    return row, distortion


TABLE = {(name, d): nearest(name, d) for name in MASS for d in DEPTHS}
BASELINE = {name: TABLE[name, d][1] for names, d in zip(MEMBERS, MAINTAINED) for name in names}
records = []
accepted = []
electron_admissible = []
all_pass = []
for ds in permutations(DEPTHS):
    rows = [TABLE[name, d][0] for names, d in zip(MEMBERS, ds) for name in names]
    exact_electron = TABLE['electron', ds[3]][1] == 1
    common_lepton_scale = FACTORS4[ds[3]] == FACTORS4[ds[4]]
    witnesses = [name for names, d in zip(MEMBERS, ds) for name in names if TABLE[name, d][1] > BASELINE[name]]
    candidate = {
        'block_depths': dict(zip(BLOCKS, ds)),
        'exact_electron_reference': exact_electron,
        'preserves_shared_lepton_unit': common_lepton_scale,
        'all_retained_constraints': exact_electron and common_lepton_scale,
        'no_worse_for_every_mass': not witnesses,
        'particles_worse_than_maintained': witnesses,
    }
    if exact_electron:
        electron_admissible.append(ds)
    if exact_electron and common_lepton_scale:
        candidate['rows'] = rows
        with localcontext() as context:
            context.prec = 70
            candidate['rms_log_error_eleven_nonanchor'] = str((sum(
                Decimal(row['log_error']) ** 2 for row in rows if row['particle'] != 'electron') / 11).sqrt())
        accepted.append(candidate)
        if not witnesses:
            all_pass.append(ds)
    records.append(candidate)

assert len(records) == 120
assert len(electron_admissible) == 24
assert len(accepted) == 6
assert all_pass == [MAINTAINED]
assert TABLE['top', 2][0]['index'] > 100
assert TABLE['electron', 6][0]['index'] == 13
assert all(TABLE['electron', d][1] != 1 for d in DEPTHS if d != 6)
maintained = next(c for c in accepted if tuple(c['block_depths'].values()) == MAINTAINED)
ordered = sorted(accepted, key=lambda c: Decimal(c['rms_log_error_eleven_nonanchor']))
assert ordered[0] == maintained
with localcontext() as context:
    context.prec = 70
    ratio = Decimal(ordered[1]['rms_log_error_eleven_nonanchor']) / Decimal(ordered[0]['rms_log_error_eleven_nonanchor'])
report = {
    'status': 'passed',
    'method': 'Exact rational fourth-power bracketing and distortion comparisons; positive integer n unbounded.',
    'mass_inputs_rational': {name: str(value) for name, value in MASS.items()},
    'sector_factor_fourth_powers': {str(d): str(value) for d, value in FACTORS4.items()},
    'permutations_tested': len(records), 'electron_reference_admissible': len(electron_admissible),
    'full_constraint_admissible': len(accepted),
    'assignments_meeting_every_maintained_accuracy': [dict(zip(BLOCKS, ds)) for ds in all_pass],
    'ordered_full_constraint_assignments': ordered,
    'nearest_alternative_rms_ratio': str(ratio),
    'all_permutation_gates': records,
    'all_depth_nearest_rungs': [row for row, distortion in TABLE.values()],
    'sources': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (Path(__file__), INPUT_FILE)},
    'statistical_scope': 'No odds computed from observed componentwise tolerances; finite-null bounds are a separate preserved result.',
}
(BUNDLE / 'exact-sector-results.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({'status': 'passed', 'permutations': len(records), 'full_constraints': len(accepted),
                 'only_assignment_meeting_all_current_accuracies': report['assignments_meeting_every_maintained_accuracy'],
                 'nearest_alternative_error_ratio': str(ratio),
                 'top_index_at_depth_two': TABLE['top', 2][0]['index']},indent=2))
