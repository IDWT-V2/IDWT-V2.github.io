#!/usr/bin/env python3
"""Exact metric checks for the chiral-ray to ordinary-normal-rotation map."""
from pathlib import Path
from itertools import combinations
import argparse
import hashlib
import json
import sympy as s

CHECKS = 0


def check(condition, message):
    global CHECKS
    CHECKS += 1
    if not condition:
        raise AssertionError(message)


def clifford(q):
    dim = 2**q
    matrices = []
    for i in range(q):
        creation = s.zeros(dim)
        for mask in range(dim):
            if not (mask >> i) & 1:
                creation[mask | (1 << i), mask] = (-1)**((mask & ((1 << i)-1)).bit_count())
        matrices.extend((creation+creation.T, s.I*(creation-creation.T)))
    return matrices


def rotation(psi, gamma):
    n = len(gamma)
    return s.Matrix(n, n, lambda i, j: 0 if i == j else
                    s.simplify(s.I*(psi.conjugate().T*gamma[i]*gamma[j]*psi)[0]))


def variation(psi, tangent, gamma):
    n = len(gamma)
    return s.Matrix(n, n, lambda i, j: 0 if i == j else s.simplify(s.I*(
        tangent.conjugate().T*gamma[i]*gamma[j]*psi+
        psi.conjugate().T*gamma[i]*gamma[j]*tangent)[0]))


def inner(x, y):
    return s.simplify(s.trace(x.T*y))


def fs_inner(psi, x, y):
    return s.simplify(s.re((x.conjugate().T*y)[0]-
                          (x.conjugate().T*psi)[0]*(psi.conjugate().T*y)[0]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    cases = []
    for q in (2, 3, 4):
        gamma = clifford(q)
        dim = 2**q
        eye = s.eye(dim)
        vacuum = eye[:, 0]
        j0 = rotation(vacuum, gamma)
        check(j0*j0 == -s.eye(2*q), 'vacuum gives a full orthogonal normal structure')
        for i in range(2*q):
            for j in range(i, 2*q):
                check(gamma[i]*gamma[j]+gamma[j]*gamma[i] ==
                      (2*s.eye(dim) if i == j else s.zeros(dim)), 'exact Clifford relation')
        pair_masks = [(1 << i)+(1 << j) for i, j in combinations(range(q), 2)]
        tangents = [factor*eye[:, mask] for mask in pair_masks for factor in (1, s.I)]
        derivatives = [variation(vacuum, tangent, gamma) for tangent in tangents]
        gram = s.Matrix(len(tangents), len(tangents), lambda i, j: inner(derivatives[i], derivatives[j]))
        check(gram == 16*s.eye(len(tangents)), 'normal metric is sixteen times horizontal ray metric')
        for tangent, derivative in zip(tangents, derivatives):
            check((vacuum.conjugate().T*tangent)[0] == 0, 'ray tangent is horizontal')
            check(derivative.T == -derivative, 'normal variation remains skew')
            check(derivative*j0+j0*derivative == s.zeros(2*q), 'pure ray variation is an orientation tangent')

        # A nontrivial exact Spin rotation transports the metric certificate.
        generator = (gamma[0]*gamma[2]-gamma[1]*gamma[3])/2
        check(generator.conjugate().T == -generator, 'pair mixing is a Spin generator')
        check(generator**3 == -generator, 'pair generator exact minimal polynomial')
        unitary = s.eye(dim)+s.Rational(4, 5)*generator+s.Rational(2, 5)*generator**2
        check(unitary.conjugate().T*unitary == s.eye(dim), 'rational Spin transport is unitary')
        psi = unitary*vacuum
        check((psi.conjugate().T*psi)[0] == 1, 'transported ray is normalized')
        j = rotation(psi, gamma)
        check(j*j == -s.eye(2*q), 'transported ray remains pure')
        transported = [unitary*tangent for tangent in tangents]
        varied = [variation(psi, tangent, gamma) for tangent in transported]
        for i, x in enumerate(transported):
            for k, y in enumerate(transported):
                check(inner(varied[i], varied[k]) == 16*fs_inner(psi, x, y),
                      'metric homothety after a nontrivial Spin rotation')

        full_chiral_ray_dimension = 2*(2**(q-1)-1)
        pure_dimension = q*(q-1)
        rank = gram.rank()
        cases.append(dict(case=f'normal-ray-rank-{q}', complex_normal_rank=q,
                          ordinary_normal_directions=2*q,
                          full_chiral_projective_real_dimension=full_chiral_ray_dimension,
                          pure_orientation_dimension=pure_dimension,
                          differential_rank_on_pure_orbit=rank,
                          pure_metric_to_fs_ratio=16))
        if q == 4:
            top = eye[:, dim-1]
            for tangent in (top, s.I*top):
                check(variation(vacuum, tangent, gamma) == s.zeros(8),
                      'generic rank-four ray has an extra null metric direction')
                check(fs_inner(vacuum, tangent, tangent) == 1,
                      'extra direction has nonzero projective kinetic norm')
            psi_bad = (vacuum+top)/s.sqrt(2)
            check(rotation(psi_bad, gamma) == s.zeros(8),
                  'rank-four chiral polarization need not supply any normal complex structure')
            theta = s.symbols('theta', real=True)
            psi_theta = s.cos(theta)*vacuum+s.sin(theta)*top
            j_theta = rotation(psi_theta, gamma)
            check(s.simplify(j_theta-s.cos(2*theta)*j0) == s.zeros(8),
                  'rank-four extra ray direction has exact cosine normal bilinear')
            check(s.simplify(s.trace(s.eye(8)+j_theta*j_theta)-8*s.sin(2*theta)**2) == 0,
                  'unsquared purity deficit gives a quadratic restoring test')
            deficit_squared = inner(s.eye(8)+j_theta*j_theta, s.eye(8)+j_theta*j_theta)
            check(s.simplify(deficit_squared-8*s.sin(2*theta)**4) == 0,
                  'squared purity deficit begins quartically on the extra ray direction')
    occupied = []
    for d in (2, 3, 4, 5, 6, 10):
        even = d if d % 2 == 0 else d+1
        q = (10-even)//2
        occupied.append(dict(real_depth=d, even_carrier=even, complex_normal_rank=q,
                             has_nontrivial_normal_orientation=q > 1,
                             full_positive_chiral_ray_is_pure=q in (2, 3),
                             pure_metric_to_fs_ratio=16 if q > 1 else None,
                             caveat='empty normal carrier' if q == 0 else
                             'generic ray has two extra real directions' if q == 4 else None))
    report = dict(title='Chiral Ray and Ordinary Normal Rotation Kinetics',
                  checks_passed=CHECKS, cases=cases, occupied_depths=occupied,
                  sources={'experiments/normal-ray-metric-checks.py': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
                  scope='Exact local geometric kinetic map on pure rays; no physical charge selection, particle state or absolute amplitude calibration')
    args.report.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(checks_passed=CHECKS, cases=len(cases))))


if __name__ == '__main__':
    main()
