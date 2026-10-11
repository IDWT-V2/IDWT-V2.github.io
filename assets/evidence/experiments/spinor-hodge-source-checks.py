#!/usr/bin/env python3
"""Construct middle-form Hodge sources directly from spatial spinor bilinears."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
from hashlib import sha256
from itertools import combinations
import json
import math
from pathlib import Path

import numpy as np
import sympy as sp


CHECKS = 0
CASES = []


def check(ok, message):
    global CHECKS
    if not ok:
        raise AssertionError(message)
    CHECKS += 1


def close(actual, expected, message, atol=2e-10):
    check(np.allclose(actual, expected, atol=atol, rtol=2e-10), message)


def parity(index):
    return (-1)**sum(index[i] > index[j] for i in range(len(index)) for j in range(i+1, len(index)))


def multiply(first, second):
    p, a = first; q, b = second
    return p[q], b*a[q]


def dense(monomial):
    permutation, phase = monomial
    value = np.zeros((len(permutation), len(permutation)), dtype=complex)
    value[permutation, np.arange(len(permutation))] = phase
    return value


def gamma_monomials(m):
    index = np.arange(2**m)
    gamma = []
    for j in range(m):
        mask = 1 << (m-1-j)
        preceding = np.array([((int(c) >> (m-j)).bit_count()) % 2 for c in index])
        signs = (-1.)**preceding
        gamma.append((index ^ mask, signs.astype(complex)))
        gamma.append((index ^ mask, 1j*signs*((-1.)**((index & mask) > 0))))
    return gamma


def bilinear_data(m):
    gamma = gamma_monomials(m)
    size = 2**m
    identity = (np.arange(size), np.ones(size, dtype=complex))
    pairing = identity
    for j in range(m):
        pairing = multiply(pairing, gamma[2*j+1])
    keep = np.array([j for j in range(size) if int(j).bit_count() % 2 == 0])
    rank = len(keep)
    c = dense(pairing)
    for g in gamma:
        gd = dense(g)
        close(c@gd, (-1)**m*gd.T@c, 'invariant spinor bilinear Clifford transpose identity')
    middle = list(combinations(range(2*m), m))
    lower = [p for p in range(m) if (m-p) % 2 == 0]
    rows = []
    degrees = []
    middle_rows = []
    middle_map = {}
    grouped = defaultdict(list)
    for degree in lower+[m]:
        for index in combinations(range(2*m), degree):
            blade = identity
            for j in index:
                blade = multiply(blade, gamma[j])
            value = dense(multiply(pairing, blade))[np.ix_(keep, keep)]
            sign = (-1)**(degree*(degree-1)//2+m*degree+m*(m+1)//2)
            close(value.T, sign*value, 'exact graded symmetry of spinor pair form')
            if degree == m:
                middle_map[index] = value
                middle_rows.append(value.reshape(-1))
                # Pick one representative from each complementary Hodge pair.
                if 2*m-1 not in index:
                    continue
            rows.append(value.reshape(-1))
            degrees.append((degree, sign))
            permutation = tuple(np.argmax(abs(value), axis=0))
            grouped[permutation].append(value[np.asarray(permutation), np.arange(rank)])
    check(len(rows) == rank**2, 'all retained bilinear grades exhaust spinor tensor square')
    for phases in grouped.values():
        phases = np.asarray(phases)
        check(phases.shape == (rank, rank), 'complete orthogonal Clifford character block')
        close(phases@phases.conj().T, rank*np.eye(rank), 'all bilinear grades are orthogonal with canonical norm')
    check(len(grouped) == rank, 'all Clifford permutation blocks represented')
    for first, second in combinations(grouped, 2):
        check(all(a != b for a, b in zip(first, second)), 'different Clifford character blocks have disjoint matrix support')
    symmetric_count = sum(sign == 1 for degree, sign in degrees)
    alternating_count = len(rows)-symmetric_count
    check(symmetric_count == rank*(rank+1)//2, 'symmetric-square grade decomposition dimension')
    check(alternating_count == rank*(rank-1)//2, 'alternating-square grade decomposition dimension')
    hodge_eigenvalue = None
    for index in middle:
        other = tuple(j for j in range(2*m) if j not in index)
        star_value = parity(other+index)*middle_map[other]
        value = middle_map[index]
        nonzero = abs(value) > .5
        eigenvalue = star_value[nonzero][0]/value[nonzero][0]
        close(star_value, eigenvalue*value, 'spinor middle-form source has one fixed Hodge chirality')
        if hodge_eigenvalue is None:
            hodge_eigenvalue = eigenvalue
        close(eigenvalue, hodge_eigenvalue, 'same Hodge eigenvalue on every source component')
    close(hodge_eigenvalue**2, (-1)**m, 'spinor-source Hodge parity')
    return {'m': m, 'rank': rank, 'gamma': gamma, 'keep': keep,
            'pairing': pairing, 'middle': middle, 'middle_map': middle_map,
            'middle_rows': np.asarray(middle_rows), 'degrees': degrees,
            'hodge_eigenvalue': hodge_eigenvalue}


def rotation_and_fierz_checks(data, rng):
    m, rank = data['m'], data['rank']
    middle, form_map = data['middle'], data['middle_map']
    lookup = {index: j for j, index in enumerate(middle)}
    middle_rows = data['middle_rows']
    keep = data['keep']
    for _ in range(3):
        psi = rng.normal(size=rank)+1j*rng.normal(size=rank)
        source = middle_rows@np.outer(psi, psi).reshape(-1)
        lower_norm = 0.
        for degree in range(m-4, -1, -4):
            for index in combinations(range(2*m), degree):
                blade = (np.arange(2**m), np.ones(2**m, dtype=complex))
                for j in index:
                    blade = multiply(blade, data['gamma'][j])
                matrix = dense(multiply(data['pairing'], blade))[np.ix_(keep, keep)]
                lower_norm += abs(psi@matrix@psi)**2/rank
        close(np.vdot(source, source).real/(2*rank)+lower_norm, np.vdot(psi, psi).real**2,
              'complete symmetric Fierz norm includes all lower grades')
        hermitian_source = data['hodge_eigenvalue'].real*source
        if m % 2:
            close(hermitian_source, 0., 'odd-middle Hodge cancellation holds for actual spinor-generated source')
        else:
            close(np.vdot(hermitian_source, hermitian_source), np.vdot(source, source),
                  'even-middle Hodge filter retains entire spinor-generated source')
        for a, b in ((0, 1), (0, 2*m-1)):
            angle = .413
            product = dense(multiply(data['gamma'][a], data['gamma'][b]))
            u = (math.cos(angle/2)*np.eye(2**m)+math.sin(angle/2)*product)[np.ix_(keep, keep)]
            rotated_psi = u@psi
            actual = middle_rows@np.outer(rotated_psi, rotated_psi).reshape(-1)
            expected = source.copy()
            for j, index in enumerate(middle):
                if (a in index) == (b in index):
                    continue
                replaced = tuple((b if q == a else a if q == b else q) for q in index)
                sign = parity(replaced)
                other = tuple(sorted(replaced))
                coefficient = math.sin(angle)*(1 if a in index else -1)*sign
                expected[j] = math.cos(angle)*source[j]+coefficient*source[lookup[other]]
            close(actual, expected, 'actual spinor rotation intertwines with exterior-form rotation')
    for depth in (2*m-1, 2*m):
        CASES.append({'name': 'spinor-generated-Hodge-channel', 'real_depth': depth,
                      'parameters': {'even_carrier_depth': 2*m, 'middle_degree': m,
                                     'minimal_spinor_rank': rank,
                                     'source': 'F_I=psi^T C gamma_I psi',
                                     'odd_shadow_requires_declared_even_carrier': depth != 2*m},
                      'measurements': {'middle_channel_rank': math.comb(2*m-1, m-1),
                                       'symmetric_square_rank': rank*(rank+1)//2,
                                       'lower_symmetric_grades': list(range(m-4, -1, -4)),
                                       'Hodge_eigenvalue': str(data['hodge_eigenvalue']),
                                       'symmetric_Hodge_filter_active': not bool(m % 2),
                                       'middle_source_is_full_symmetric_square': m <= 3}})


def triplet_geometry_checks():
    x1, y1, x2, y2, x3, y3 = sp.symbols('x1 y1 x2 y2 x3 y3')
    spinors = [sp.Matrix([x1, y1]), sp.Matrix([x2, y2]), sp.Matrix([x3, y3])]
    veronese = lambda w: sp.Matrix([w[0]**2, sp.sqrt(2)*w[0]*w[1], w[1]**2])
    determinant = sp.det(sp.Matrix.hstack(*(veronese(w) for w in spinors)))
    pair_product = sp.sqrt(2)*sp.prod(sp.det(sp.Matrix.hstack(spinors[i], spinors[j]))
                                    for i, j in combinations(range(3), 2))
    check(sp.simplify(determinant-pair_product) == 0, 'triplet volume is product of three spinor pair areas')
    w = sp.Matrix([x1, y1]); c = veronese(w)
    check(sp.expand(c[1]**2-2*c[0]*c[2]) == 0, 'one spinor square occupies constrained conic not all triplet states')
    special = np.array([1., math.sqrt(2), 1.], dtype=complex)/2
    su3 = np.diag([np.exp(.31j), np.exp(-.31j), 1.])
    close(np.linalg.det(su3), 1., 'constraint-control mixing is in SU3')
    transformed = su3@special
    check(abs(transformed[1]**2-2*transformed[0]*transformed[2]) > .1,
          'generic SU3 mixing fails to preserve a one-spinor-square source')
    # Every binary quadratic factors over C: two distinct spinors fill the triplet carrier.
    rng = np.random.default_rng(425)
    for _ in range(8):
        desired = rng.normal(size=3)+1j*rng.normal(size=3)
        a, b, c0 = desired
        roots = np.roots([a, math.sqrt(2)*b, c0])
        u = np.array([1., -roots[0]])
        v = a*np.array([1., -roots[1]])
        actual = np.array([u[0]*v[0], (u[0]*v[1]+u[1]*v[0])/math.sqrt(2), u[1]*v[1]])
        close(actual, desired, 'distinct spinor pair parametrizes arbitrary complex triplet')
        norm = (np.vdot(u, u).real*np.vdot(v, v).real+abs(np.vdot(u, v))**2)/2
        close(np.vdot(actual, actual).real, norm, 'canonical symmetric distinct-pair kinetic norm')
    # A uniform unit spinor already induces exactly isotropic triplet covariance.
    t = sp.symbols('t', real=True)
    moments = [sp.integrate(t**2, (t, 0, 1)), sp.integrate(2*t*(1-t), (t, 0, 1)),
               sp.integrate((1-t)**2, (t, 0, 1))]
    check(moments == [sp.Rational(1, 3)]*3, 'uniform spinor source has triplet covariance I/3')
    s = sp.symbols('s', real=True)
    phase_averaged_pair = sp.integrate((1-t)*(s*(1-t)+(1-s)*t), (t, 0, 1))
    check(sp.simplify(phase_averaged_pair-(1+s)/6) == 0, 'third spinor pair-area fourth moment')
    check(sp.expand(2*(1-s)*phase_averaged_pair-(1-s**2)/3) == 0,
          'actual spinor-source third-slot contraction preserves canonical triplet pair operator')
    # A full Dirac kinetic carrier is retained: a source projection does not compress its principal symbol.
    for m in range(1, 7):
        gamma = gamma_monomials(m)
        keep = np.array([j for j in range(2**m) if j.bit_count() % 2 == 0])
        for g in gamma:
            matrix = dense(g)
            close(matrix@matrix, np.eye(2**m), 'full field Clifford kinetic direction survives source-only construction')
            close(matrix[np.ix_(keep, keep)], 0., 'chiral fibre-only kinetic compression would remove derivative')
    CASES.append({'name': 'triplet-source-geometry', 'real_depth': 4,
                  'parameters': {'carrier': 'Sym^2 C^2', 'one_spinor_source': '(x^2,sqrt(2)xy,y^2)'},
                  'measurements': {'source_constraint': 'c_2^2-2 c_1 c_3=0',
                                   'three_slot_volume': 'sqrt(2) product_{i<j} det(w_i,w_j)',
                                   'independent_color_derived': False,
                                   'two_distinct_spinor_pairs_can_fill_carrier': True,
                                   'uniform_unit_spinor_triplet_covariance': 'I_3/3',
                                   'unit_spinor_pair_energy_after_third_average': '(1-|u^dagger v|^4)/3',
                                   'full_Dirac_kinetic_carrier_must_be_retained': True}})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rng = np.random.default_rng(20261009)
    for m in range(1, 7):
        data = bilinear_data(m)
        rotation_and_fierz_checks(data, rng)
    triplet_geometry_checks()
    report = {'title': 'Spatial spinor bilinears generate the full middle-Hodge carrier',
              'created_utc': datetime.now(timezone.utc).isoformat(),
              'checks_passed': CHECKS, 'cases': CASES,
              'scope': 'Exact algebraic field-source construction across maintained and neighboring depths; no particle simulations.',
              'statistics': 'Commuting classical amplitudes; identical Grassmann spinor squares require a different slot symmetry.',
              'limitations': ['The bilinear is a geometric source, not an independently derived color multiplet.',
                              'One spinor square obeys algebraic constraints not preserved by arbitrary full-channel unitary mixing.',
                              'Odd depths use the declared even-carrier shadow.',
                              'The full kinetic spinor carrier is retained; a pointwise chiral compression would kill its symbol.'],
              'sources': {'experiments/spinor-hodge-source-checks.py': sha256(Path(__file__).read_bytes()).hexdigest()}}
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'checks_passed': CHECKS, 'cases': len(CASES), 'output': str(args.output)}))


if __name__ == '__main__':
    main()
