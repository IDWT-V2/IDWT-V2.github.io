#!/usr/bin/env python3
"""Actual spherical monogenic bands, contact and quantum geometric tensors."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
from itertools import product
import json
from math import comb, factorial, prod, sqrt
from pathlib import Path

import numpy as np
import scipy.linalg as la
import sympy as sp


CHECKS = 0
CASES = []


def check(condition, message):
    global CHECKS
    if not condition:
        raise AssertionError(message)
    CHECKS += 1


def gammas(depth):
    m = depth//2
    identity = np.eye(2, dtype=complex)
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.diag([1., -1.]).astype(complex)
    def tensor(factors):
        result = np.ones((1, 1), dtype=complex)
        for factor in factors:
            result = np.kron(result, factor)
        return result
    result = []
    for index in range(m):
        for axis in (sx, sy):
            result.append(tensor([sz]*index+[axis]+[identity]*(m-index-1)))
    if depth % 2:
        result.append(tensor([sz]*m))
    return result


def compositions(total, length):
    if length == 1:
        yield (total,)
        return
    for first in range(total+1):
        for remaining in compositions(total-first, length-1):
            yield (first,)+remaining


def sphere_monomial(exponents):
    if any(a % 2 for a in exponents):
        return sp.Integer(0)
    numerator = prod(prod(range(1, a, 2)) for a in exponents)
    degree = sum(exponents)//2
    return sp.Rational(numerator, prod(len(exponents)+2*j for j in range(degree)))


def shell_data(depth, level, gamma):
    rank = len(gamma[0])
    ambient = depth+1
    symbol = gamma+[1j*np.eye(rank)]
    indices = list(compositions(level, ambient))
    lookup = {alpha: j for j, alpha in enumerate(indices)}
    dim = len(indices)*rank
    if level:
        lower = list(compositions(level-1, ambient))
        c = np.zeros((len(lower)*rank, dim), dtype=complex)
        for ib, beta in enumerate(lower):
            for axis in range(ambient):
                alpha = list(beta)
                alpha[axis] += 1
                ia = lookup[tuple(alpha)]
                c[ib*rank:(ib+1)*rank, ia*rank:(ia+1)*rank] = symbol[axis]*sqrt(alpha[axis])
        gram = c@c.conj().T
        cholesky = la.cho_factor(gram, lower=True)
        def project(z):
            return z-c.conj().T@la.cho_solve(cholesky, c@z)
        check(np.min(np.diag(cholesky[0]).real) > 0, 'surjective polynomial Dirac constraint')
        check(dim-len(lower)*rank == rank*comb(level+depth-1, depth-1), 'actual monogenic shell rank')
    else:
        c = np.zeros((0, dim), dtype=complex)
        def project(z):
            return z
    normalization = sqrt(prod(ambient+2*j for j in range(level)))
    evaluation = np.zeros((dim, rank), dtype=complex)
    north = (0,)*depth+(level,)
    j = lookup[north]
    evaluation[j*rank:(j+1)*rank] = np.eye(rank)*normalization/sqrt(factorial(level))
    derivatives = []
    for axis in range(depth):
        derivative = np.zeros_like(evaluation)
        if level:
            alpha = [0]*ambient
            alpha[-1] = level-1
            alpha[axis] = 1
            j = lookup[tuple(alpha)]
            derivative[j*rank:(j+1)*rank] = np.eye(rank)*normalization/sqrt(factorial(level-1))
        derivatives.append(derivative)
    projected = project(np.concatenate([evaluation]+derivatives, axis=1))
    w = projected[:, :rank]
    dd = [projected[:, (a+1)*rank:(a+2)*rank] for a in range(depth)]
    check(np.linalg.norm(c@w) < 1e-8, 'evaluation state is monogenic')
    multiplicity = comb(level+depth-1, depth-1)
    check(np.allclose(w.conj().T@w, multiplicity*np.eye(rank), atol=1e-9), 'physical sphere norm gives exact shell kernel')
    for a in range(depth):
        expected = -level/depth*symbol[-1].conj().T@symbol[a]
        check(np.allclose(w.conj().T@dd[a]/multiplicity, expected, atol=1e-9), 'actual shell connection moment')
        for b in range(depth):
            expected = (level*(level+depth-1)/depth*np.eye(rank) if a == b else
                        -level/depth*gamma[a]@gamma[b])
            check(np.allclose(dd[a].conj().T@dd[b]/multiplicity, expected, atol=1e-9),
                  'actual shell derivative moment')
    # At the north pole, ambient monogenicity plus Euler homogeneity is the intrinsic Dirac equation.
    if level:
        gradient = [np.zeros((rank, rank), dtype=complex) for _ in range(ambient)]
        for axis in range(depth):
            alpha = [0]*ambient
            alpha[-1], alpha[axis] = level-1, 1
            j = lookup[tuple(alpha)]
            gradient[axis] = w[j*rank:(j+1)*rank]/sqrt(factorial(level-1))
        j = lookup[north]
        gradient[-1] = level*w[j*rank:(j+1)*rank]/sqrt(factorial(level))
        value = w[j*rank:(j+1)*rank]/sqrt(factorial(level))
        equation = sum(1j*gamma[a]@gradient[a] for a in range(depth))
        check(np.allclose(equation, level*value, atol=1e-8), 'positive spherical Dirac branch at evaluation')
    return w, dd, multiplicity, dim


def spatial_checks():
    for depth in range(2, 11):
        gamma = gammas(depth)
        rank = len(gamma[0])
        max_level = (6 if depth == 2 else 4 if depth <= 4 else 3 if depth <= 6 else 2)
        shells = []
        for level in range(max_level+1):
            shells.append(shell_data(depth, level, gamma))
            count = comb(level+depth, depth)
            w = np.vstack([s[0] for s in shells])/sqrt(count)
            dd = [np.vstack([s[1][a] for s in shells])/sqrt(count) for a in range(depth)]
            check(np.allclose(w.conj().T@w, np.eye(rank), atol=1e-9), 'cumulative evaluation frame normalized')
            aa = [w.conj().T@derivative for derivative in dd]
            q = np.block([[dd[a].conj().T@dd[b]-aa[a].conj().T@aa[b]
                           for b in range(depth)] for a in range(depth)])
            cval = level*(level+depth+1)/(depth+1)**2
            metric = (depth-1+1/(depth+2))*cval
            for a in range(depth):
                for b in range(depth):
                    expected = metric*np.eye(rank) if a == b else -cval*gamma[a]@gamma[b]
                    check(np.allclose(q[a*rank:(a+1)*rank, b*rank:(b+1)*rank], expected, atol=1e-8),
                          'full spatial quantum geometric tensor')
            eigen = np.linalg.eigvalsh(q)
            target = sorted([cval/(depth+2)]*rank+[(depth+1)**2*cval/(depth+2)]*((depth-1)*rank))
            check(np.allclose(eigen, target, atol=1e-8), 'independent full tensor spectrum')
            check(np.min(eigen) > -1e-8, 'positive spatial quantum tensor')
            CASES.append({'name': 'actual-spatial-monogenic-band', 'real_depth': depth, 'mode_index': level+1,
                          'parameters': {'sphere_radius': 1, 'shell_cutoff': level,
                                         'spinor_rank': rank, 'physical_band_dimension': rank*count,
                                         'coefficient_dimension': sum(s[3] for s in shells)},
                          'measurements': {'reference_normalized_contact_eigenvalue': count,
                                           'curvature_coefficient': cval, 'metric_diagonal': metric,
                                           'Clifford_trace_eigenvalue': cval/(depth+2),
                                           'twistor_eigenvalue': (depth+1)**2*cval/(depth+2),
                                           'geometric_spectral_ratio': (depth+1)**2 if level else None,
                                           'maximum_spectrum_error': float(np.max(np.abs(eigen-target)))}})


def exact_moments_and_integration():
    for depth in range(2, 13):
        for level in range(41):
            total = comb(level+depth, depth)
            mean = sp.Rational(depth*level, depth+1)
            variance = sp.Rational(depth*level*(level+depth+1), (depth+1)**2*(depth+2))
            shell_mean = sp.Rational(sum(j*comb(j+depth-1, depth-1) for j in range(level+1)), total)
            shell_second = sp.Rational(sum(j*j*comb(j+depth-1, depth-1) for j in range(level+1)), total)
            check(shell_mean == mean and shell_second-mean**2 == variance, 'exact cumulative shell moments')
            cval = sp.Rational(level*(level+depth+1), (depth+1)**2)
            metric = (shell_second+(depth-1)*shell_mean)/depth-(shell_mean/depth)**2
            check(metric == (depth-1+sp.Rational(1, depth+2))*cval, 'count moments fix geometric metric')
            check(variance/depth == cval/(depth+2), 'kinetic variance fixes Clifford trace metric')
    # A harmonic test polynomial (x0+i*x1)^l independently checks Fischer/sphere normalization.
    for ambient in range(3, 12):
        for level in range(9):
            integral = sum(comb(level, j)*sphere_monomial((2*j, 2*(level-j))+(0,)*(ambient-2))
                           for j in range(level+1))
            fischer = 2**level*factorial(level)
            expected = sp.Rational(fischer, prod(ambient+2*j for j in range(level)))
            check(integral == expected, 'independent angular integral fixes polynomial normalization')
    for name, depth, n in [('W', 2, 76), ('Z', 2, 81), ('Higgs', 2, 95), ('down', 3, 1),
                            ('strange', 3, 4), ('up', 4, 3), ('charm', 4, 20), ('top', 4, 72),
                            ('nu1', 5, 10), ('nu2', 5, 15), ('nu3', 5, 22),
                            ('electron', 6, 13), ('muon', 6, 35), ('tau', 10, 23)]:
        level = n-1
        cval = sp.Rational(level*(level+depth+1), (depth+1)**2)
        CASES.append({'name': 'maintained-branch-geometric-moments', 'real_depth': depth, 'mode_index': n,
                      'parameters': {'particle': name, 'sphere_radius': 'R; eigenvalues listed in R^-2'},
                      'measurements': {'contact_count': comb(level+depth, depth),
                                       'curvature_coefficient_exact': str(cval),
                                       'Clifford_trace_eigenvalue_exact': str(cval/(depth+2)),
                                       'twistor_eigenvalue_exact': str(cval*(depth+1)**2/(depth+2)),
                                       'geometric_ratio': (depth+1)**2 if level else None}})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    exact_moments_and_integration()
    spatial_checks()
    report = {'experiment_key': 'actual-spatial-band-curvature-20261009',
              'created_utc': datetime.now(timezone.utc).isoformat(),
              'sources': {'experiments/spatial-band-curvature-checks.py': sha256(Path(__file__).read_bytes()).hexdigest()},
              'scope': 'Actual monogenic sphere Dirac bands, probability-normalized spatial integration, contact and quantum tensor; exact maintained-branch moments. A spatial control family, not an empirical identification or compactification.',
              'summary': {'checks_passed': CHECKS, 'case_count': len(CASES)}, 'cases': CASES}
    Path(args.output).write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report['summary']))


if __name__ == '__main__':
    main()
