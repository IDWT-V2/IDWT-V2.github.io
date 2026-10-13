"""Replace arbitrary affine patches by the convex envelope of count tangents."""
from datetime import datetime, timezone
from fractions import Fraction
from hashlib import sha256
from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path
import json

import mpmath as mp
import numpy as np
import sympy as sp

ROOT = Path(__file__).resolve().parent
spec = spec_from_file_location('forward', ROOT/'forward-count-action.py')
forward = module_from_spec(spec)
spec.loader.exec_module(forward)
COUNTS, ERRORS = {}, {}


def check(value, label):
    if not bool(value):
        raise AssertionError(label)
    COUNTS[label] = COUNTS.get(label, 0)+1


def close(actual, expected, label, tolerance=3e-8):
    error = float(np.linalg.norm(np.asarray(actual)-np.asarray(expected))/max(1., np.linalg.norm(expected)))
    ERRORS[label] = max(ERRORS.get(label, 0.), error)
    check(error < tolerance, label)


def high(value):
    return mp.mpf(value.numerator)/value.denominator


def depth_carrier(depth):
    epsq = Fraction(3, 10)/max(forward.count_data(depth, n-1)[1]
                             for _, d, n in forward.radial.ASSIGNMENTS if d == depth)
    candidates = []
    level = 0
    while True:
        count, first, second = forward.count_data(depth, level)
        cq, hhq = epsq*first, Fraction(2*level+depth, 2)
        if cq >= 1:
            break
        # Strict convexity condition for the original f as a function of scalar ratio.
        curvature_numerator = hhq*epsq*second-cq*(1-cq*cq)
        if curvature_numerator > 0:
            c, x, h = high(cq), mp.sqrt(1-high(cq)**2), high(hhq)
            frequency = high(epsq*count)
            candidates.append(dict(level=level, c=c, x=x, h=h,
                                   m=h*x/c, eta=frequency-h/c, omega=frequency,
                                   c_exact=str(cq), count_exact=str(count)))
        level += 1
    return epsq, candidates, level


def main():
    mp.mp.dps = 100
    # An exact common convexity proof covers every depth >=3.
    for depth in [3, 4, 5, 6, 10]:
        hd = sum(Fraction(1, j) for j in range(1, depth+1))
        h2 = sum(Fraction(1, j*j) for j in range(1, depth+1))
        check(Fraction(depth, 2)*(hd*hd-h2)-hd > 0,
              'exact ground count derivatives start the all-shell strict-convexity proof at each maintained depth above two')
    data = {depth: depth_carrier(depth) for depth in forward.radial.DEPTHS}
    cases = []
    rng = np.random.default_rng(202610120823)
    for name, depth, index in forward.radial.ASSIGNMENTS:
        epsq, candidates, endpoint = data[depth]
        matches = [c for c in candidates if c['level'] == index-1]
        check(len(matches) == 1, 'every maintained assignment belongs to the common strictly convex count chart')
        selected = matches[0]
        x, c, h = selected['x'], selected['c'], selected['h']
        values = [candidate['m']*x+candidate['eta'] for candidate in candidates]
        winner = max(range(len(values)), key=values.__getitem__)
        check(candidates[winner]['level'] == index-1,
              'the maximum over every admissible geometric tangent uniquely selects the maintained branch')
        correct = selected['omega']-h*c
        check(abs(values[winner]-correct) < mp.mpf('1e-80'),
              'the selected envelope tangent preserves the complete stationary interaction value')
        gap = min(values[winner]-value for j, value in enumerate(values) if j != winner)
        check(gap > 0, 'the finite geometric envelope has a strict open winner gap at each maintained state')
        lower, upper = mp.mpf(-1), mp.mpf(1)
        for candidate in candidates:
            if candidate is selected:
                continue
            slope = selected['m']-candidate['m']
            crossing = (candidate['eta']-selected['eta'])/slope
            if slope > 0:
                lower = max(lower, crossing)
            else:
                upper = min(upper, crossing)
        check(lower < x < upper,
              'all competing tangent inequalities give a nonempty exact local scalar-ratio winner interval')
        for fraction in [mp.mpf('-.3'), mp.mpf('.2')]:
            ratio = x+fraction*min(x-lower, upper-x)
            trial = [candidate['m']*ratio+candidate['eta'] for candidate in candidates]
            check(max(range(len(trial)), key=trial.__getitem__) == winner,
                  'finite ratio displacements remain inside the exact affine winner chart')
        # Direct arbitrary matrix variation verifies the preferred-time local envelope source.
        generators = forward.radial.gammas(depth+1)
        alpha, beta = generators[:depth], generators[depth]
        rank, eye = len(beta), np.eye(len(beta))
        a, b = mp.sqrt((1+c)/2), mp.sqrt((1-c)/2)
        field = (float(a)*eye+float(b)*beta)/np.sqrt(rank)
        delta = rng.normal(size=field.shape)+1j*rng.normal(size=field.shape)
        delta /= np.linalg.norm(delta)
        # High-precision invariant differences avoid the tiny matter chart widths.
        scalar_variation = mp.mpf(float(2*np.vdot(delta, beta@field).real))
        density_variation = mp.mpf(float(2*np.vdot(delta, field).real))
        step = min(x-lower, upper-x)*mp.mpf('1e-3')
        def invariant_potential(density, scalar):
            return max(candidate['m']*scalar+candidate['eta']*density for candidate in candidates)
        fd = (invariant_potential(1+step*density_variation, x+step*scalar_variation)
              -invariant_potential(1-step*density_variation, x-step*scalar_variation))/(2*step)
        analytic = selected['m']*scalar_variation+selected['eta']*density_variation
        check(abs(fd-analytic)/max(1, abs(analytic)) < mp.mpf('1e-65'),
              'independent variation of the actual envelope gives the complete preferred-time source')
        for spatial_level in [0, index-1, index, index+5, 1000]:
            bare = spatial_level+depth/2
            operator = np.array([[bare, float(selected['m'])], [float(selected['m']), -bare]])-float(h/c)*np.eye(2)
            check(np.array_equal(operator, operator.T),
                  'every retained complete spatial block has an exactly Hermitian generator')
            close(np.linalg.eigvalsh(operator),
                  [-np.hypot(bare, float(selected['m']))-float(h/c), np.hypot(bare, float(selected['m']))-float(h/c)],
                  'independent massive spatial block diagonalization agrees with the full linear spectrum')
        cases.append(dict(particle=name, depth=depth, inferred_index=index,
                          epsilon_exact=str(epsq), candidate_count=len(candidates),
                          first_eligible_level=candidates[0]['level'], first_ineligible_c_one_level=endpoint,
                          canonical_c_exact=selected['c_exact'], count_exact=selected['count_exact'],
                          unique_winner_gap=str(gap), scalar_ratio=str(x),
                          scalar_ratio_winner_interval=[str(lower), str(upper)],
                          scalar_ratio_margin=str(min(x-lower, upper-x)),
                          classification='explicit finite envelope with an open exact quadratic field chart'))
        print(json.dumps(dict(completed=name, depth=depth, eligible_count=len(candidates))), flush=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    report = dict(experiment_key='convex-count-quadratic-envelope', created_utc=datetime.now(timezone.utc).isoformat(),
                  sources={'experiments/'+p.name: sha256(p.read_bytes()).hexdigest() for p in
                           [Path(__file__), ROOT/'forward-count-action.py', ROOT/'inverse-current-radial.py',
                            ROOT/'excited-shell-orientation-current.py']},
                  evaluations=sum(COUNTS.values()), conditions=COUNTS, maximum_errors=ERRORS, cases=cases,
                  limits=['the envelope is a specified effective constitutive action derived from geometric count tangents',
                          'the preferred-time realization has exact local quadratic dynamics and its existing Sobolev persistence argument applies',
                          'proper-current and responding-clock choices retain their current response and require separate persistence analysis',
                          'the strict convexity chart excludes the low depth-two tangents; every maintained massive depth-two assignment remains included',
                          'the global evolution at tied charts is unassigned; small-persistence neighborhoods stay inside their unique winner chart',
                          'no new sector-depth preference, localized particle, complete boost inertia or physical contact-source origin follows'])
    output = ROOT/('convex-count-envelope-results-'+stamp+'.json')
    output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(report=str(output), evaluations=report['evaluations'], maximum_errors=ERRORS)), flush=True)


if __name__ == '__main__':
    main()
