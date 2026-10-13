"""Seek exact all-harmonic certificates at one common small clock load."""
from datetime import datetime, timezone
from hashlib import sha256
from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path
import argparse
import json

import sympy as sp

ROOT = Path(__file__).resolve().parent
spec = spec_from_file_location('angular', ROOT/'covariant-clock-angular.py')
angular = module_from_spec(spec)
spec.loader.exec_module(angular)
L, Z = angular.L, angular.Z
T = sp.Symbol('t')
LOAD = sp.Rational(1, 100)


def certificate(expression):
    numerator, denominator = map(sp.expand, sp.fraction(sp.cancel(expression)))
    num = sp.Poly(numerator.subs(L, T+2), T)
    den = sp.Poly(denominator.subs(L, T+2), T)
    passed = all(a >= 0 for a in num.all_coeffs()) and num.TC() > 0 and all(a >= 0 for a in den.all_coeffs()) and den.TC() > 0
    roots = None
    exact_positive = bool(passed)
    if not passed:
        roots = [int(num.count_roots(0, sp.oo)), int(den.count_roots(0, sp.oo))]
        exact_positive = roots == [0, 0] and num.TC() > 0 and den.TC() > 0
    return dict(positive_coefficients=bool(passed), certified_positive=bool(exact_positive),
                shifted_nonnegative_sturm_roots=roots, expression=str(sp.factor(expression)),
                numerator_coefficients=list(map(str, num.all_coeffs())),
                denominator_coefficients=list(map(str, den.all_coeffs())))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--depths', type=int, nargs='+', default=[2, 3, 4, 5, 6, 10])
    args = parser.parse_args()
    cases = []
    for depth in args.depths:
        for family in (['scalar'] if depth == 2 else ['scalar', 'coexact-vector']):
            print(json.dumps(dict(depth=depth, family=family, status='calculating')), flush=True)
            real, imaginary = angular.form_blocks(depth, L, family, LOAD)
            # A constant field-block similarity keeps every entry rational
            # before the exact characteristic polynomial is calculated.
            scale = sp.sqrt(LOAD/5)
            for block in [real, imaginary]:
                block[:4, 4:] = (block[:4, 4:]/scale).applyfunc(sp.simplify)
                block[4:, :4] = (scale*block[4:, :4]).applyfunc(sp.simplify)
            expression = sp.factor((imaginary*real).charpoly(Z).as_expr())
            factors = sp.factor_list(expression, Z)[1]
            coupled = max((sp.Poly(f, Z).monic() for f, m in factors), key=lambda p: p.degree())
            n = coupled.degree()
            coefficients = coupled.all_coeffs()
            sums = [sp.Integer(n)]
            for k in range(1, 2*n-1):
                if k <= n:
                    value = -sum(coefficients[j]*sums[k-j] for j in range(1, k))-k*coefficients[k]
                else:
                    value = -sum(coefficients[j]*sums[k-j] for j in range(1, n+1))
                sums.append(sp.cancel(value))
            hermite = sp.Matrix(n, n, lambda i, j: sums[i+j])
            minors = []
            for order in range(1, n+1):
                domain_matrix = hermite[:order, :order].to_DM()
                determinant = domain_matrix.domain.to_sympy(domain_matrix.det())
                proof = certificate(determinant)
                minors.append(proof)
                print(json.dumps(dict(depth=depth, family=family, minor=order,
                                      positive_certificate=proof['certified_positive'])), flush=True)
            elementary = [certificate((-1)**j*a) for j, a in enumerate(coefficients[1:], 1)]
            companion = sp.zeros(n)
            for j in range(n-1):
                companion[j+1, j] = 1
            for j in range(n):
                companion[j, n-1] = -coefficients[n-j]
            assert (hermite*companion-companion.T*hermite).applyfunc(sp.cancel) == sp.zeros(n)
            first = sp.Poly(expression.subs(L, 1), Z)
            zero = 0
            while first.eval(0) == 0:
                first = sp.Poly(sp.cancel(first.as_expr()/Z), Z)
                zero += 1
            exact_first = int(first.count_roots(0, sp.oo))
            passed = all(p['certified_positive'] for p in minors+elementary) and exact_first == first.degree()
            cases.append(dict(depth=depth, family=family, c_exact='3/5', load_exact=str(LOAD),
                              polynomial=str(expression), coupled_degree=n,
                              hermite_leading_minor_certificates=minors,
                              elementary_symmetric_certificates=elementary,
                              first_harmonic_zero_multiplicity=zero,
                              first_harmonic_nonzero_real_nonnegative_roots=exact_first,
                              certified_all_harmonics=bool(passed)))
            print(json.dumps(dict(depth=depth, family=family, certified_all_harmonics=bool(passed))), flush=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    output = ROOT/('covariant-clock-uniform-results-'+stamp+'.json')
    report = dict(experiment_key='covariant-clock-all-harmonic-small-load',
                  created_utc=datetime.now(timezone.utc).isoformat(),
                  sources={'experiments/'+p.name: sha256(p.read_bytes()).hexdigest()
                           for p in [Path(__file__), ROOT/'covariant-clock-angular.py', ROOT/'covariant-clock-ground.py']},
                  cases=cases, limits=['ground-shell controls with an explicit physically uncalibrated density ratio',
                                      'failure of a positive coefficient certificate is not instability',
                                      'zero-frequency Jordan blocks are not bounded by a real-root certificate',
                                      'no assigned excited-state or nonlinear persistence claim'])
    output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(report=str(output))), flush=True)


if __name__ == '__main__':
    main()
