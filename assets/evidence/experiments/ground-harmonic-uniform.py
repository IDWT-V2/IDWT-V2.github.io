"""Certify real nonnegative squared frequencies for every scalar harmonic."""
from datetime import datetime, timezone
from hashlib import sha256
from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path
import json

import sympy as sp

ROOT = Path(__file__).resolve().parent
spec = spec_from_file_location('repair', ROOT/'ground-harmonic-repair.py')
repair = module_from_spec(spec)
spec.loader.exec_module(repair)
L, T, Z, X = sp.symbols('L t z x')
C, KAPPA = sp.Rational(3, 10), sp.Rational(11, 10)


def symbolic_blocks(depth):
    h = sp.Rational(depth, 2)
    first = sum(sp.Rational(1, j) for j in range(1, depth+1))
    second = first**2-sum(sp.Rational(1, j*j) for j in range(1, depth+1))
    eta = 1-(1-C*C)*first/(h*second)
    vector = 1-C*C/(h*first)
    u, v = (L+depth-1)/(2*L+depth-1), L/(2*L+depth-1)
    aa, bb, ab = (1+C)/2, (1-C)/2, X/2
    beta = sp.Matrix([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
    free = sp.diag(h+L, -h-L, -h-L+1, h+L-1)*C/h+X*beta-sp.eye(4)
    scalar = sp.Matrix([[u*bb, -u*ab, v*bb, -v*ab],
                        [-u*ab, u*aa, -v*ab, v*aa],
                        [u*bb, -u*ab, v*bb, -v*ab],
                        [-u*ab, u*aa, -v*ab, v*aa]])
    radial = sp.Matrix([[v*aa, -v*ab, -v*aa, v*ab],
                        [-v*ab, v*bb, v*ab, -v*bb],
                        [-u*aa, u*ab, u*aa, -u*ab],
                        [u*ab, -u*bb, -u*ab, u*bb]])
    tensor = sp.Matrix([[v*bb, -v*ab, -v*bb, v*ab],
                        [-v*ab, v*aa, v*ab, -v*aa],
                        [-u*bb, u*ab, u*bb, -u*ab],
                        [u*ab, -u*aa, -u*ab, u*aa]])
    return free+2*eta*scalar+2*KAPPA*vector*tensor, free+2*vector*radial


def positive_certificate(expression):
    numerator, denominator = map(sp.factor, sp.fraction(sp.cancel(expression)))
    num = sp.Poly(sp.expand(numerator.subs(L, T+2)), T)
    den = sp.Poly(sp.expand(denominator.subs(L, T+2)), T)
    assert all(a >= 0 for a in num.all_coeffs()) and num.TC() > 0
    assert all(a >= 0 for a in den.all_coeffs()) and den.TC() > 0
    return {'expression': str(sp.factor(expression)),
            'shifted_numerator_coefficients': list(map(str, num.all_coeffs())),
            'shifted_denominator_coefficients': list(map(str, den.all_coeffs()))}


def main():
    cases = []
    for depth in [3, 4]:
        real, imaginary = symbolic_blocks(depth)
        raw = sp.Poly((imaginary*real).charpoly(Z).as_expr(), X)
        expression = sp.factor(sp.rem(raw, sp.Poly(X*X-1+C*C, X)).as_expr())
        assert not expression.has(X)
        polynomial = sp.Poly(expression, Z)
        assert polynomial.LC() == 1 and polynomial.degree() == 4
        for degree in range(2, 7):
            expected = repair.polynomial(depth, degree, C).as_expr().subs(repair.K, KAPPA)
            assert sp.cancel(expression.subs(L, degree)-expected) == 0
        coefficients = polynomial.all_coeffs()
        elementary = [positive_certificate((-1)**j*a) for j, a in enumerate(coefficients[1:], 1)]
        sums = [sp.Integer(4)]
        for k in range(1, 7):
            if k <= 4:
                value = -sum(coefficients[j]*sums[k-j] for j in range(1, k))-k*coefficients[k]
            else:
                value = -sum(coefficients[j]*sums[k-j] for j in range(1, 5))
            sums.append(sp.factor(value))
        hermite = sp.Matrix(4, 4, lambda i, j: sums[i+j])
        companion = sp.Matrix([[0, 0, 0, -coefficients[4]],
                               [1, 0, 0, -coefficients[3]],
                               [0, 1, 0, -coefficients[2]],
                               [0, 0, 1, -coefficients[1]]])
        assert (hermite*companion-companion.T*hermite).applyfunc(sp.cancel) == sp.zeros(4)
        minors = [positive_certificate(hermite[:j, :j].det()) for j in range(1, 5)]
        first = sp.Poly(sp.cancel(expression.subs(L, 1)/Z), Z)
        assert first.degree() == 3 and first.count_roots(0, sp.oo) == 3
        xr = sp.sqrt(1-C*C)
        xx, yy = real.subs({L: 1, X: xr}), imaginary.subs({L: 1, X: xr})
        generator = sp.BlockMatrix([[sp.zeros(4), yy], [-xx, sp.zeros(4)]]).as_explicit()
        nullities = [8-(generator**j).rank() for j in [1, 2, 3]]
        cases.append({'depth': depth, 'c_exact': str(C), 'kappa_exact': str(KAPPA),
                      'polynomial': str(expression), 'first_harmonic_cubic': str(first.as_expr()),
                      'elementary_symmetric_positive_certificates': elementary,
                      'hermite_leading_minor_positive_certificates': minors,
                      'positive_trace_metric_symmetrizes_companion': True,
                      'first_harmonic_generator_power_nullities': nullities,
                      'classification': 'all integer harmonics L>=1 have no exponentially growing eigenvalue in this reducing family'})
        print(json.dumps({'depth': depth, 'certified_all_harmonics': True, 'nullities': nullities}), flush=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    output = ROOT/('ground-harmonic-uniform-results-'+stamp+'.json')
    report = {'experiment_key': 'uniform-ground-harmonic-no-growth-window',
              'created_utc': datetime.now(timezone.utc).isoformat(),
              'sources': {'experiments/'+p.name: sha256(p.read_bytes()).hexdigest()
                          for p in [Path(__file__), ROOT/'ground-harmonic-repair.py', ROOT/'inverse-current-ground-proof.py', ROOT/'inverse-current-radial.py']},
              'cases': cases,
              'limits': ['ground states and exactly reducing rotational spatial family only',
                         'no assigned excited electron or full angular stability follows',
                         'kappa=11/10 changes the potential and lies outside positive blends',
                         'zero-frequency Jordan structure is reported separately from exponential spectral growth',
                         'no color selection or observed decay claim is made']}
    output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'report': str(output)}), flush=True)


if __name__ == '__main__':
    main()
