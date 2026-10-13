"""Exact necessary spatial stability conditions for an unrestricted current blend."""
from datetime import datetime, timezone
from hashlib import sha256
from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path
import json

import sympy as sp

ROOT = Path(__file__).resolve().parent
spec = spec_from_file_location('ground', ROOT/'inverse-current-ground-proof.py')
ground = module_from_spec(spec)
spec.loader.exec_module(ground)
K, Z, X = sp.symbols('kappa z x')


def polynomial(depth, degree, c):
    real, imaginary = ground.exact_block(depth, degree, c, X)
    if depth != 2:
        u = sp.Rational(degree+depth-1, 2*degree+depth-1)
        v = sp.Rational(degree, 2*degree+depth-1)
        aa, bb, ab = (1+c)/2, (1-c)/2, X/2
        tensor = sp.Matrix([[v*bb, -v*ab, -v*bb, v*ab],
                            [-v*ab, v*aa, v*ab, -v*aa],
                            [-u*bb, u*ab, u*bb, -u*ab],
                            [u*ab, -u*aa, -u*ab, u*aa]])
        harmonic_number = sum(sp.Rational(1, j) for j in range(1, depth+1))
        vector = 1-c*c/(sp.Rational(depth, 2)*harmonic_number)
        real += 2*K*vector*tensor
    raw = sp.Poly((imaginary*real).charpoly(Z).as_expr(), X)
    reduced = sp.factor(sp.rem(raw, sp.Poly(X*X-1+c*c, X)).as_expr())
    assert not reduced.has(X)
    if degree == 1:
        assert reduced.subs(Z, 0) == 0
        reduced = sp.cancel(reduced/Z)
    return sp.Poly(reduced, Z)


def necessary_conditions(poly):
    coefficients = poly.all_coeffs()
    signs = [sp.factor((-1)**j*a) >= 0 for j, a in enumerate(coefficients[1:], 1)]
    discriminant = sp.factor(sp.discriminant(poly.as_expr(), Z))
    return signs+[discriminant >= 0], discriminant


def main():
    cases = []
    for c in [sp.Rational(3, 10), sp.Rational(3, 5)]:
        for depth in ground.radial.DEPTHS:
            conditions, harmonics = [], []
            for degree in range(1, 7):
                poly = polynomial(depth, degree, c)
                added, discriminant = necessary_conditions(poly)
                conditions.extend(added)
                allowed = sp.reduce_inequalities(conditions, K)
                harmonics.append({'scalar_harmonic_degree': degree,
                                  'characteristic_polynomial': str(poly.as_expr()),
                                  'discriminant': str(discriminant),
                                  'remaining_necessary_blend_set': str(allowed)})
                print(json.dumps({'depth': depth, 'c': str(c), 'harmonic': degree,
                                  'empty': allowed == sp.false}), flush=True)
                if allowed == sp.false:
                    break
            cases.append({'depth': depth, 'c_exact': str(c), 'harmonics': harmonics,
                          'all_real_blends_excluded': allowed == sp.false,
                          'classification': 'exact necessary root inequalities; nonempty sets do not establish stability'})
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    report = {'experiment_key': 'ground-current-repair-harmonic-intersection',
              'created_utc': datetime.now(timezone.utc).isoformat(),
              'sources': {'experiments/'+p.name: sha256(p.read_bytes()).hexdigest()
                          for p in [Path(__file__), ROOT/'inverse-current-ground-proof.py', ROOT/'inverse-current-radial.py']},
              'cases': cases,
              'limits': ['unrestricted real blend coefficient in the stated two-action family',
                         'exact ground-shell reducing harmonic blocks, not assigned excited electron states',
                         'no theorem for every local Lorentz scalar potential is assumed',
                         'nonempty necessary sets do not prove real-rootedness of quartics or full spatial stability']}
    output = ROOT/('ground-harmonic-repair-results-'+stamp+'.json')
    output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'report': str(output)}), flush=True)


if __name__ == '__main__':
    main()
