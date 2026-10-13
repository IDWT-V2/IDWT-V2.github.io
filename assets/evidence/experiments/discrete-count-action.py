"""A forward constitutive rule built only from integer spatial counts."""
from datetime import datetime, timezone
from fractions import Fraction
from hashlib import sha256
from importlib.util import spec_from_file_location, module_from_spec
from math import comb
from pathlib import Path
import json

import mpmath as mp
import numpy as np

ROOT = Path(__file__).resolve().parent
spec = spec_from_file_location('forward', ROOT/'forward-count-action.py')
forward = module_from_spec(spec)
spec.loader.exec_module(forward)
COUNTS, ERRORS = {}, {}


def check(value, label):
    if not bool(value):
        raise AssertionError(label)
    COUNTS[label] = COUNTS.get(label, 0)+1


def high(value):
    return mp.mpf(value.numerator)/value.denominator


def close(actual, expected, label, tolerance=3e-7):
    error = float(np.linalg.norm(np.asarray(actual)-np.asarray(expected))/max(1., np.linalg.norm(expected)))
    ERRORS[label] = max(ERRORS.get(label, 0.), error)
    check(error < tolerance, label)


def selector(c, epsilon, depth):
    # Forward adjacent differences are strictly increasing at every occupied depth.
    level = 0
    while epsilon*comb(level+depth, depth-1) < c:
        level += 1
    return level


def field_potential(field, epsilon, alpha, beta, depth):
    _, _, _, norm, support = forward.bilinears(field, alpha, beta)
    level = selector(support/norm, epsilon, depth)
    h = level+depth/2
    return epsilon*norm*comb(level+depth, depth)-h*support, level


def main():
    mp.mp.dps = 90
    rng = np.random.default_rng(202610120801)
    cases = []
    for name, depth, index in forward.radial.ASSIGNMENTS:
        level, hhq = index-1, Fraction(2*(index-1)+depth, 2)
        count, first, _ = forward.count_data(depth, level)
        epsq = Fraction(3, 10)/max(forward.count_data(depth, n-1)[1]
                                 for _, d, n in forward.radial.ASSIGNMENTS if d == depth)
        original_c = epsq*first
        lower = epsq*comb(level+depth-1, depth-1) if level else Fraction(0)
        upper = epsq*comb(level+depth, depth-1)
        check(lower < original_c < upper,
              'every reconstructed polarization lies strictly inside the corresponding integer-count selection interval')
        check(selector(original_c, epsq, depth) == level,
              'a forward search using only integer counts independently selects the inferred shell')
        # The adjacent-difference theorem certifies the global minimum on all nonnegative labels.
        last_allowed = selector(Fraction(1), epsq, depth)
        last = max(level+5, last_allowed+5)
        costs = [epsq*comb(k+depth, depth)-Fraction(2*k+depth, 2)*original_c for k in range(last+1)]
        check(min(range(len(costs)), key=costs.__getitem__) == level,
              'an independent exhaustive finite comparison agrees with the all-label convex-difference theorem')
        for k in range(last):
            check(costs[k+1]-costs[k] == epsq*comb(k+depth, depth-1)-original_c,
                  'exact adjacent count differences certify the minimum and its increasing tail')
        selected_intervals = []
        # The branch contains a family, rather than one retuned polarization.
        for weight in [Fraction(1, 4), Fraction(1, 2), Fraction(3, 4)]:
            cq = lower+(min(upper, Fraction(1))-lower)*weight
            c, hh, epsilon = high(cq), high(hhq), high(epsq)
            check(selector(cq, epsq, depth) == level,
                  'three independent polarizations occupy the same geometric branch under the identical action')
            x = mp.sqrt(1-c*c)
            a, b = mp.sqrt((1+c)/2), mp.sqrt((1-c)/2)
            mass = hh*x/c
            omega = epsilon*high(count)
            eta = omega-hh/c
            pair = mp.matrix([a, b])
            equation = mp.matrix([[hh+eta, mass], [mass, -hh+eta]])
            check(mp.norm(equation*pair-omega*pair)/omega < mp.mpf('1e-70'),
                  'the selected integer-count action gives the exact same clock across the whole tested polarization family')
            energy = hh*c+epsilon*high(count)-hh*c
            check(abs(energy-omega)/omega < mp.mpf('1e-75'),
                  'the complete family energy retains spatial kinetics and the exact count frequency')
            selected_intervals.append(str(cq))
        generators = forward.radial.gammas(depth+1)
        alpha, beta = generators[:depth], generators[depth]
        rank = len(beta)
        for sample in range(3):
            field = rng.normal(size=(rank, rank))+1j*rng.normal(size=(rank, rank))
            field /= np.linalg.norm(field)
            variation = rng.normal(size=field.shape)+1j*rng.normal(size=field.shape)
            variation /= np.linalg.norm(variation)
            epsilon = float(epsq)
            value, chosen = field_potential(field, epsilon, alpha, beta, depth)
            step = 2e-6
            plus, chosen_plus = field_potential(field+step*variation, epsilon, alpha, beta, depth)
            minus, chosen_minus = field_potential(field-step*variation, epsilon, alpha, beta, depth)
            check(chosen == chosen_plus == chosen_minus,
                  'the arbitrary full matrix source check remains in one smooth selected-label chart')
            hchosen = chosen+depth/2
            source = forward.source(field, hchosen, epsilon, alpha, beta, depth)
            close((plus-minus)/(2*step*max(1, hchosen)),
                  2*np.vdot(variation, source).real/max(1, hchosen),
                  'independent variation of the actual minimum-over-counts potential verifies its full field source')
        for sample in range(40):
            bloch = rng.normal(size=3); bloch /= np.linalg.norm(bloch)
            bx, by, bc = bloch
            radius = np.hypot(by, bc)
            chosen = selector(radius, float(epsq), depth)
            hchosen = chosen+depth/2
            energy = float(hhq)*bc+float(epsq)*comb(chosen+depth, depth)-hchosen*radius
            deficit = float(epsq*count)-energy
            distance = max(float(lower)-radius, radius-float(upper), 0.)
            lower_bound = float(hhq)*(radius-bc)+distance
            check(deficit >= lower_bound-2e-11*max(1., abs(energy)),
                  'the full Bloch-sphere energy deficit controls phase mismatch and distance to the stationary-family interval')
            check(by*by <= 2*deficit/float(hhq)+2e-11*max(1., deficit),
                  'the complete pair energy gives the quantitative transverse phase-persistence bound')
        cases.append(dict(particle=name, depth=depth, inferred_index=index,
                          count_exact=str(count), epsilon_exact=str(epsq),
                          selected_c_interval_exact=[str(lower), str(min(upper, Fraction(1)))],
                          original_polarization_exact=str(original_c),
                          original_polarization_margin_exact=str(min(original_c-lower, upper-original_c)),
                          same_clock_test_polarizations_exact=selected_intervals,
                          maximum_available_shell_at_c_one=last_allowed,
                          classification='conditional integer-count action with a geometric local label selector'))
        print(json.dumps(dict(completed=name, depth=depth, selected_shell=level)), flush=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    report = dict(experiment_key='forward-discrete-spatial-count-action', created_utc=datetime.now(timezone.utc).isoformat(),
                  sources={'experiments/'+p.name: sha256(p.read_bytes()).hexdigest() for p in
                           [Path(__file__), ROOT/'forward-count-action.py', ROOT/'inverse-current-radial.py',
                            ROOT/'excited-shell-orientation-current.py']},
                  evaluations=sum(COUNTS.values()), conditions=COUNTS, maximum_errors=ERRORS, cases=cases,
                  limits=['the minimum of count-contact cost minus spatial support is an explicit new constitutive premise',
                          'the discrete geometric rule does not select the sparse list of observed particles',
                          'the proof uses smooth open winner charts; action derivatives at ties and vacuum need a specified global evolution',
                          'the polarization family has a zero reduced shape curvature and potential secular drift',
                          'proper-current full spatial persistence is not inherited from other actions',
                          'no spatial localization, complete boost inertia or assigned-depth preference is established'])
    output = ROOT/('discrete-count-action-results-'+stamp+'.json')
    output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(report=str(output), evaluations=report['evaluations'], maximum_errors=ERRORS)), flush=True)


if __name__ == '__main__':
    main()
