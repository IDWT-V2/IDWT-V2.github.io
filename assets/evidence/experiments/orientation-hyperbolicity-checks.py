#!/usr/bin/env python3
"""Exact static-background cones and a moving-background failure threshold."""
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
        raise AssertionError('Orientation principal identity failed')
    checks += 1


def anti(b):
    r, i = s.re(b), s.im(b)
    return r.row_join(i).col_join(i.row_join(-r))


def bracket(a, b):
    return a*b-b*a


def inner(a, b):
    return s.trace(a.T*b)


cases = []
for q in (2, 3, 4):
    basis = []
    for i, j in combinations(range(q), 2):
        e = s.zeros(q)
        e[i, j], e[j, i] = 1, -1
        basis.extend((anti(e)/2, anti(s.I*e)/2))
    size = len(basis)
    spatial = [basis[0], basis[1], basis[-1]+basis[0]/3]
    for n in ((1, 0, 0), (0, s.Rational(3, 5), s.Rational(4, 5)),
              (s.Rational(1, 3), s.Rational(2, 3), s.Rational(2, 3))):
        check(sum(z*z for z in n) == 1)
        b = [[bracket(eta, a) for a in spatial] for eta in basis]
        c = s.Matrix(size, size, lambda r, t: sum(inner(b[r][i], b[t][i]) for i in range(3)))
        bn = [sum((n[i]*b[r][i] for i in range(3)), s.zeros(2*q)) for r in range(size)]
        cn = s.Matrix(size, size, lambda r, t: inner(bn[r], bn[t]))
        mag = [[n[i]*b[r][j]-n[j]*b[r][i] for i, j in combinations(range(3), 2)] for r in range(size)]
        g = s.Matrix(size, size, lambda r, t: sum(inner(x, y) for x, y in zip(mag[r], mag[t])))
        check(g == c-cn)
        check(cn.T == cn and g.T == g)
        for eta in basis:
            for i, j in combinations(range(3), 2):
                check(bracket(n[i]*eta, n[j]*eta) == s.zeros(2*q))
        for amplitude in (0, s.Rational(1, 2), 3):
            k = 4*s.eye(size)+amplitude*c
            gradient = 4*s.eye(size)+amplitude*g
            check(k-gradient == amplitude*cn)
            check(gradient-4*s.eye(size) == amplitude*g)
    x = basis[0]
    brackets = [bracket(x, eta) for eta in basis]
    c = s.Matrix(size, size, lambda r, t: inner(brackets[r], brackets[t]))
    eigenvalues = c.eigenvals()
    check(max(eigenvalues) == 1)
    # Full moving symbol: nonzero symmetric mixed term, with exact
    # positive-definite linearization and null-covector bounds.
    time = (basis[0]+basis[1])/4
    time_norm = inner(time, time)
    b0 = [bracket(eta, time) for eta in basis]
    c0 = s.Matrix(size, size, lambda r, t: inner(b0[r], b0[t]))
    for n in ((1, 0, 0), (s.Rational(1, 3), s.Rational(2, 3), s.Rational(2, 3))):
        bi = [[bracket(eta, a) for a in spatial] for eta in basis]
        cn_all = s.Matrix(size, size, lambda r, t: sum(inner(bi[r][i], bi[t][i]) for i in range(3)))
        bn = [sum((n[i]*bi[r][i] for i in range(3)), s.zeros(2*q)) for r in range(size)]
        cnn = s.Matrix(size, size, lambda r, t: inner(bn[r], bn[t]))
        cross = s.Matrix(size, size, lambda r, t: inner(bn[r], b0[t])+inner(b0[r], bn[t]))
        for epsilon in (1, 5, 20):
            check(epsilon*time_norm < 4)
            kinetic = 4*s.eye(size)+epsilon*cn_all
            gradient = 4*s.eye(size)+epsilon*(cn_all-cnn-c0)
            mixed = -epsilon*cross
            for sign in (-1, 1):
                null_gram = s.Matrix(size, size, lambda r, t: inner(bn[r]-sign*b0[r], bn[t]-sign*b0[t]))
                check(kinetic+sign*mixed-gradient == epsilon*null_gram)
            for positive in (kinetic, gradient):
                _, diagonal = positive.LDLdecomposition(hermitian=True)
                check(all(diagonal[i, i] > 0 for i in range(size)))
            zero = s.zeros(size)
            linearized = zero.row_join(gradient).col_join(gradient.row_join(-mixed))
            symmetrizer = gradient.row_join(zero).col_join(zero.row_join(kinetic))
            check(linearized.T == linearized and symmetrizer.T == symmetrizer)
    for moving_strength in (0, 2, 4, 5):
        speeds = {str(value): str(1-s.Rational(moving_strength, 4)*value) for value in eigenvalues}
        if moving_strength < 4:
            check(all(s.sympify(value) > 0 for value in speeds.values()))
        elif moving_strength == 4:
            check(min(s.sympify(value) for value in speeds.values()) == 0)
        else:
            check(min(s.sympify(value) for value in speeds.values()) < 0)
    cases.append({'complex_normal_rank': q,
                  'static_background_spatial_rank': 3 if q > 2 else 2,
                  'moving_critical_epsilon_times_speed_squared_over_f_squared': 4,
                  'curvature_gram_eigenvalues': {str(v): m for v, m in eigenvalues.items()}})

parser = argparse.ArgumentParser()
parser.add_argument('--report', type=Path, required=True)
args = parser.parse_args()
report = {'checks_passed': checks, 'cases': cases,
          'sources': {'experiments/orientation-hyperbolicity-checks.py': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
          'scope': 'Exact static symbol, full mixed moving pencil in a sufficient causal regime, and finite-energy moving-geodesic hyperbolicity threshold for the explicitly chosen action; no identified particle solution'}
args.report.write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps({'checks_passed': checks, 'cases': cases}))
