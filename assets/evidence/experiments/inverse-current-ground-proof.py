"""Exact spatial ground-shell instability blocks, with rational certificates."""
from datetime import datetime, timezone
from hashlib import sha256
from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path
import json

import numpy as np
import sympy as sp

ROOT = Path(__file__).resolve().parent
spec = spec_from_file_location('radial', ROOT/'inverse-current-radial.py')
radial = module_from_spec(spec)
spec.loader.exec_module(radial)
CONDITIONS, ERRORS = {}, {}


def check(value, label):
    if not bool(value):
        raise AssertionError(label)
    CONDITIONS[label] = CONDITIONS.get(label, 0)+1


def exact_block(depth, harmonic, c, x):
    h = sp.Rational(depth, 2)
    first = sum(sp.Rational(1, j) for j in range(1, depth+1))
    second = first**2-sum(sp.Rational(1, j*j) for j in range(1, depth+1))
    eta = 1-(1-c*c)*first/(h*second)
    vector = 1-c*c/(h*first)
    u, v = sp.Rational(harmonic+depth-1, 2*harmonic+depth-1), sp.Rational(harmonic, 2*harmonic+depth-1)
    aa, bb, ab = (1+c)/2, (1-c)/2, x/2
    beta = sp.Matrix([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
    free = sp.diag(h+harmonic, -h-harmonic, -h-harmonic+1, h+harmonic-1)*c/h+x*beta-sp.eye(4)
    # Similarity by diag(sqrt(u),sqrt(u),sqrt(v),sqrt(v)) removes
    # square roots from all source dyads without changing the spectrum.
    scalar = sp.Matrix([[u*bb, -u*ab, v*bb, -v*ab],
                        [-u*ab, u*aa, -v*ab, v*aa],
                        [u*bb, -u*ab, v*bb, -v*ab],
                        [-u*ab, u*aa, -v*ab, v*aa]])
    radial_source = sp.Matrix([[v*aa, -v*ab, -v*aa, v*ab],
                               [-v*ab, v*bb, v*ab, -v*bb],
                               [-u*aa, u*ab, u*aa, -u*ab],
                               [u*ab, -u*bb, -u*ab, u*bb]])
    transverse = sp.Matrix([[v*bb, -v*ab, -v*bb, v*ab],
                            [-v*ab, v*aa, v*ab, -v*aa],
                            [-u*bb, u*ab, u*bb, -u*ab],
                            [u*ab, -u*aa, -u*ab, u*aa]])
    lx = free+2*eta*scalar+(2*vector*transverse if depth == 2 else sp.zeros(4))
    ly = free+2*vector*radial_source
    return lx, ly


def main():
    z = sp.Symbol('z')
    c, x = sp.symbols('c x', real=True)
    cases = []
    for depth in radial.DEPTHS:
        lx, ly = exact_block(depth, 1, c, x)
        raw = sp.Poly((ly*lx).charpoly(z).as_expr(), x)
        reduced = sp.rem(raw, sp.Poly(x*x-1+c*c, x)).as_expr()
        check(not reduced.has(x), 'exact harmonic polynomial depends on the physical mixing through c only')
        reduced = sp.factor(reduced)
        check(reduced.subs(z, 0) == 0, 'first spatial harmonic retains its zero symmetry frequency')
        cubic = sp.Poly(sp.cancel(reduced/z), z)
        discriminant = sp.factor(sp.discriminant(cubic.as_expr(), z))
        tested = sp.Poly(cubic.as_expr().subs(c, sp.Rational(3, 5)), z)
        disc = sp.discriminant(tested.as_expr(), z)
        check(disc < 0, 'one common rational mixing gives an exact complex spatial frequency pair at every occupied depth')
        check(tested.count_roots(-sp.oo, sp.oo) == 1, 'Sturm root count independently certifies the nonreal squared-frequency pair')
        check(tested.count_roots(-sp.oo, 0) == 0, 'the certified growing oscillation is distinct from a negative real squared frequency')
        numerical_x, numerical_y = radial.ground_block(depth, 1, sp.Rational(3, 5))
        numerical = np.poly(numerical_y@numerical_x)
        expected = np.array(sp.Poly(reduced.subs(c, sp.Rational(3, 5)), z).all_coeffs(), float)
        error = np.linalg.norm(numerical-expected)/np.linalg.norm(expected)
        check(error < 3e-13, 'exact certificate agrees with the independently normalized spatial harmonic block')
        ERRORS['independent normalized characteristic polynomial'] = max(ERRORS.get('independent normalized characteristic polynomial', 0.), float(error))
        roots = np.linalg.eigvals(numerical_y@numerical_x)
        growth = max(np.sqrt(-roots.astype(complex)).real)*depth/(2*.6)
        numerator, denominator = sp.fraction(discriminant)
        # The nonzero boundary roots delimit where the same exact
        # harmonic changes its root type; no floating sign is used.
        boundary = sp.Poly(numerator, c)
        intervals = boundary.intervals(eps=sp.Rational(1, 10**10))
        positive_intervals = [{'lower': str(a), 'upper': str(b), 'multiplicity': mult}
                              for (a, b), mult in intervals if b > 0 and a < 1]
        cases.append({'depth': depth, 'scalar_harmonic_degree': 1,
                      'c_exact': '3/5', 'x_exact': '4/5',
                      'squared_frequency_polynomial': str(reduced),
                      'cubic_discriminant': str(discriminant),
                      'cubic_discriminant_at_test': str(disc),
                      'real_cubic_roots': 1,
                      'discriminant_boundary_intervals_inside_physical_chart': positive_intervals,
                      'growth_in_inverse_radius_units': float(growth),
                      'classification': 'exact finite reducing spatial block, not a Galerkin truncation'})
        print(json.dumps(cases[-1]), flush=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    result = {'experiment_key': 'inverse-current-exact-ground-spatial-instability',
              'created_utc': datetime.now(timezone.utc).isoformat(),
              'sources': {'experiments/'+p.name: sha256(p.read_bytes()).hexdigest()
                          for p in [Path(__file__), ROOT/'inverse-current-radial.py']},
              'conditions': CONDITIONS, 'evaluations': sum(CONDITIONS.values()),
              'maximum_errors': ERRORS, 'cases': cases,
              'limits': ['supplied count action on a conditional ordinary sphere with its stipulated balanced matrix carrier',
                         'ground shells are controls, not the assigned excited electron, muon or tau',
                         'an exact reducing spatial growing mode disproves full linear stability of these controls',
                         'no observed decay, lifetime, nonlinear outcome or sector preference is established',
                         'restricted pair persistence remains valid inside its invariant family']}
    output = ROOT/('inverse-current-ground-proof-results-'+stamp+'.json')
    output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'report': str(output), 'evaluations': result['evaluations']}), flush=True)


if __name__ == '__main__':
    main()
