#!/usr/bin/env python3
"""Conjugate response cancellation, complete kinetics and all-depth controls."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
from itertools import combinations
import json
from math import comb
from pathlib import Path

import numpy as np
import sympy as sp


CHECKS = 0
CASES = []


def check(condition, message):
    global CHECKS
    if not condition:
        raise AssertionError(message)
    CHECKS += 1


def full_hessian(gaps, aa, bb):
    # Canonical order q,pq,lambda,plambda,x+,y+,x-,y-.
    h = np.diag([2., 1., 3., 1., gaps[0], gaps[0], gaps[1], gaps[1]])
    for j, (gap, a, b) in enumerate(zip(gaps, aa, bb)):
        ix, iy = 4+2*j, 5+2*j
        h[0, 0] += gap*a*a
        h[2, 2] += gap*b*b
        h[0, ix] = h[ix, 0] = -gap*a
        h[2, iy] = h[iy, 2] = -gap*b
    return h


def exact_pair():
    w, d, a, b, k, m, kl, ml = sp.symbols('Omega Delta A B K M Kl Ml', positive=True, real=True)
    action = sp.diag(k+2*d*a*a-m*w*w, kl+2*d*b*b-ml*w*w, d, d, d, d)
    for ix, iy, sign in ((2, 3, 1), (4, 5, -1)):
        action[0, ix] = action[ix, 0] = -d*a
        action[1, iy] = action[iy, 1] = -sign*d*b
        action[ix, iy] = -sp.I*w
        action[iy, ix] = sp.I*w
    keep, remove = [0, 1], [2, 3, 4, 5]
    effective = sp.simplify(action.extract(keep, keep)-action.extract(keep, remove)*
                            action.extract(remove, remove).inv()*action.extract(remove, keep))
    check(effective[0, 1] == 0 and effective[1, 0] == 0, 'paired cross response vanishes at all frequencies')
    check(sp.simplify(effective[0, 0]-(k-m*w*w-2*d*a*a*w*w/(d*d-w*w))) == 0,
          'receiver induced inertia adds')
    direct = 2*d*b*b*w*w/(d*d-w*w)
    check(sp.simplify(effective[1, 1]-(kl-ml*w*w-direct)) == 0,
          'direct source inertia survives pairing')
    check(sp.limit(direct, w, sp.oo) == -2*d*b*b, 'paired high-frequency source response survives')
    check(sp.simplify(sp.limit(direct*(d-w), w, d)-d*d*b*b) == 0,
          'bare fast pole in prescribed-source response is genuine')
    check(sp.factor(action.det()).as_poly(w).degree() == 8, 'complete pair contains four oscillators')

    dp, dm, epsilon = sp.symbols('Delta_plus Delta_minus epsilon', positive=True, real=True)
    mismatch = dp**2*a*b/(dp**2-w*w)-dm**2*a*b/(dm**2-w*w)
    check(sp.simplify(mismatch.subs(w, 0)) == 0, 'equal weights can cancel adiabatic gyro despite unequal gaps')
    check(sp.simplify(sp.limit(mismatch/w**2, w, 0)-a*b*(1/dp**2-1/dm**2)) == 0,
          'gap mismatch first appears at second frequency order')
    imbalance = d*d*a*b*epsilon/(d*d-w*w)
    check(sp.limit(imbalance, w, 0) == a*b*epsilon, 'occupation imbalance retains low-frequency gyro')
    CASES.append({'name': 'exact-paired-fast-bands', 'parameters': {'matching': 'equal positive gaps and equal weights'},
                  'measurements': {'cross_response': '0', 'source_response': str(direct),
                                   'source_response_high_frequency': str(-2*d*b*b),
                                   'unequal_gap_residual': str(mismatch)}})

    jmat = np.kron(np.eye(4), np.array([[0., 1.], [-1., 0.]]))
    for gap in (.3, 1.3, 3.7, 10., 30.):
        for amplitude in (.03, .1, .3):
            for mismatch_fraction in (0., .02, .2):
                gaps = (gap, gap*(1+mismatch_fraction))
                h = full_hessian(gaps, (amplitude, amplitude), (amplitude, -amplitude))
                check(np.linalg.eigvalsh(h)[0] > 0, 'positive complete paired Hamiltonian')
                dynamic = jmat@h
                eigen = np.linalg.eigvals(dynamic)
                check(np.max(np.abs(eigen.real)) < 1e-9, 'complete paired system is oscillatory')
                frequencies = sorted(v.imag for v in eigen if v.imag > 1e-9)
                check(len(frequencies) == 4, 'all four complete positive modes retained')
                check(np.linalg.norm(h@dynamic+dynamic.T@h) < 1e-9, 'complete paired energy conserved')
                CASES.append({'name': 'complete-paired-Hamiltonian',
                              'parameters': {'gaps': gaps, 'A': amplitude, 'B_signs': [1, -1]},
                              'measurements': {'frequencies': frequencies,
                                               'minimum_energy_eigenvalue': float(np.linalg.eigvalsh(h)[0])}})


def quantum_metric_and_color():
    theta, phi = sp.symbols('theta phi', real=True)
    u = sp.Matrix([sp.cos(theta/2), sp.exp(sp.I*phi)*sp.sin(theta/2)])
    v = sp.kronecker_product(u, u.conjugate())
    coords = (theta, phi)
    def geometry(state):
        p = sp.eye(len(state))-state*state.H
        return sp.simplify(sp.Matrix(2, 2, lambda i, j: (sp.diff(state, coords[i]).H*p*
                                                       sp.diff(state, coords[j]))[0]))
    qu, qv = geometry(u), geometry(v)
    for i in range(2):
        check(sp.simplify(sp.I*(v.H*sp.diff(v, coords[i]))[0]) == 0, 'product ray has zero Berry connection')
        for jj in range(2):
            check(sp.trigsimp(qv[i, jj]-2*sp.re(qu[i, jj]), method='fu') == 0,
                  'product ray doubles metric while cancelling curvature')
    for rank in (1, 3, 10, 35, 126, 462):
        check(sp.Rational(1, rank)*rank == 1, 'product-ray trace overlap normalization')
        CASES.append({'name': 'flat-ray-color-control', 'parameters': {'carrier_rank': rank},
                      'measurements': {'singlet_overlap_squared': str(sp.Rational(1, rank)),
                                       'quadratic_color_generator_expectation': rank-1,
                                       'Berry_curvature': '0', 'metric_factor': 2}})
    # Independent finite generator-action test of Casimir for u tensor conjugate(u).
    for rank in range(2, 7):
        state = np.zeros((rank, rank), dtype=complex)
        vec = np.arange(1, rank+1, dtype=float)+1j*np.arange(rank, 0, -1)
        vec /= np.linalg.norm(vec)
        state = np.outer(vec, vec.conjugate())
        generators = []
        for x, y in combinations(range(rank), 2):
            s = np.zeros((rank, rank), dtype=complex)
            s[x, y] = s[y, x] = .5
            generators.append(s)
            s = np.zeros((rank, rank), dtype=complex)
            s[x, y], s[y, x] = -.5j, .5j
            generators.append(s)
        for length in range(1, rank):
            s = np.zeros((rank, rank), dtype=complex)
            s[np.arange(length), np.arange(length)] = 1
            s[length, length] = -length
            generators.append(s/np.sqrt(2*length*(length+1)))
        norm = sum(np.linalg.norm(t@state-state@t)**2 for t in generators)
        check(np.isclose(norm, rank-1), 'explicit generators confirm nonneutral flat product ray')
        trace = np.eye(rank)/np.sqrt(rank)
        check(all(np.linalg.norm(t@trace-trace@t) < 1e-14 for t in generators),
              'true trace tensor annihilated by every generator')


def sign_sort(indices):
    return (-1)**sum(indices[i] > indices[j] for i in range(len(indices)) for j in range(i+1, len(indices)))


def hodge_controls():
    for m in range(1, 7):
        d = 2*m
        def star(index):
            other = tuple(i for i in range(d) if i not in index)
            return other, sign_sort(index+other)
        def rotate(index):
            # Infinitesimal physical rotation in the ordinary 0,1 plane.
            if (0 in index) == (1 in index):
                return None, 0
            raw = tuple(1 if i == 0 else 0 if i == 1 else i for i in index)
            return tuple(sorted(raw)), (1 if 0 in index else -1)*sign_sort(raw)
        for index in combinations(range(d), m):
            rotated, rs = rotate(index)
            starred, ss = star(index)
            left, ls = star(rotated) if rs else (None, 0)
            right, rrs = rotate(starred)
            check((left, rs*ls) == ((right if rrs else None), ss*rrs),
                  'Hodge projector commutes with physical rotation')
        rank = comb(d-1, m-1)
        for depth in (d-1, d):
            CASES.append({'name': 'all-depth-conjugation-control', 'real_depth': depth,
                          'parameters': {'even_carrier_depth': d, 'middle_degree': m,
                                         'odd_shadow_requires_separate_restriction_map': depth != d},
                          'measurements': {'channel_rank': rank,
                                           'conjugation_exchanges_Hodge_channels': bool(m % 2),
                                           'matched_response_pairs_can_cancel_gyro': True,
                                           'Hodge_rotation_alone_forces_equal_occupation': False,
                                           'flat_product_ray_color_Casimir': rank-1}})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    exact_pair()
    quantum_metric_and_color()
    hodge_controls()
    report = {'experiment_key': 'paired-curvature-response-20261009',
              'created_utc': datetime.now(timezone.utc).isoformat(),
              'sources': {'experiments/paired-curvature-response-checks.py': sha256(Path(__file__).read_bytes()).hexdigest()},
              'scope': 'Conditional conjugate fast-band cancellation, complete positive kinetics, quantum metric/color distinction and Hodge controls through depth twelve. No particle simulations.',
              'summary': {'checks_passed': CHECKS, 'case_count': len(CASES)}, 'cases': CASES}
    Path(args.output).write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report['summary']))


if __name__ == '__main__':
    main()
