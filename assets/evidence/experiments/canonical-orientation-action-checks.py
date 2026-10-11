#!/usr/bin/env python3
"""Exact principal response of a first-derivative normal orientation action."""
import argparse
import hashlib
import json
from itertools import combinations
from pathlib import Path
import sympy as s

checks = 0


def check(ok):
    global checks
    if not ok:
        raise AssertionError('Exact orientation action condition failed')
    checks += 1


def anti(b):
    r, i = s.re(b), s.im(b)
    return r.row_join(i).col_join(i.row_join(-r))


def inner(a, b):
    return s.trace(a.T*b)


cases = []
for q in (2, 3, 4):
    basis = []
    for i, j in combinations(range(q), 2):
        e = s.zeros(q)
        e[i, j], e[j, i] = 1, -1
        basis.extend((anti(e)/2, anti(s.I*e)/2))
    gram = s.Matrix(len(basis), len(basis), lambda i, j: inner(basis[i], basis[j]))
    check(gram == s.eye(len(basis)))
    x = basis[0]
    brackets = [x*y-y*x for y in basis]
    c = s.Matrix(len(basis), len(basis), lambda i, j: inner(brackets[i], brackets[j]))
    check(c.T == c)
    eigenvalues = c.eigenvals()
    check(all(v >= 0 for v in eigenvalues))
    for f2, epsilon, k2 in ((1, 1, 1), (s.Rational(1, 3), 2, 3), (2, 0, 4)):
        kinetic = 4*f2*s.eye(len(basis))+epsilon*k2*c
        for cos2 in (0, s.Rational(1, 4), 1):
            gradient = 4*f2*s.eye(len(basis))+epsilon*k2*(1-cos2)*c
            for v in eigenvalues:
                speed2 = (4*f2+epsilon*k2*(1-cos2)*v)/(4*f2+epsilon*k2*v)
                check(0 < speed2 <= 1)
            check(kinetic-gradient == epsilon*k2*cos2*c)
    cases.append({'complex_normal_rank': q,
                  'orientation_real_dimension': len(basis),
                  'unit_geodesic_curvature_gram_eigenvalues': {str(v): m for v, m in eigenvalues.items()}})

q2 = cases[0]['unit_geodesic_curvature_gram_eigenvalues']
check(q2 == {'0': 1, '1': 1})
f2, epsilon, k2, cos2, c = s.symbols('f2 epsilon k2 cos2 c', nonnegative=True)
v2 = (4*f2+epsilon*k2*(1-cos2)*c)/(4*f2+epsilon*k2*c)
check(s.factor(1-v2) == epsilon*k2*cos2*c/(4*f2+epsilon*k2*c))

parser = argparse.ArgumentParser()
parser.add_argument('--report', type=Path, required=True)
args = parser.parse_args()
report = {'checks_passed': checks,
          'cases': cases,
          'occupied_cases': [{'real_depth': d, 'even_carrier': d if d % 2 == 0 else d+1,
                              'complex_normal_rank': (10-(d if d % 2 == 0 else d+1))//2,
                              'hodge_coefficient': int(d in (3, 4))}
                             for d in (2, 3, 4, 5, 6, 10)],
          'sources': {'experiments/canonical-orientation-action-checks.py': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
          'scope': 'Exact frozen principal symbol on a static single-geodesic orientation background; no general hyperbolicity, particle solution or fixed-flux guide existence claim'}
args.report.write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps({'checks_passed': checks, 'cases': cases}))
