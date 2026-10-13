"""Actual rotational spatial Hessian of the inverse current action.

This is an exactly reducing matrix-field family, with a finite spectral
projection for its stability diagnostic. It is not full spatial stability.
"""
from datetime import datetime, timezone
from fractions import Fraction
from hashlib import sha256
from math import comb, factorial
from pathlib import Path
import argparse
import json

import mpmath as mp
import numpy as np
from scipy.linalg import eigvals, eigvalsh
from scipy.special import eval_gegenbauer, roots_jacobi
import sympy as sp

ROOT = Path(__file__).resolve().parent
DEPTHS = (2, 3, 4, 5, 6, 10)
ASSIGNMENTS = [('W', 2, 76), ('Z', 2, 81), ('Higgs', 2, 95),
               ('down', 3, 1), ('strange', 3, 4), ('up', 4, 3),
               ('charm', 4, 20), ('top', 4, 72), ('nu1', 5, 10),
               ('nu2', 5, 15), ('nu3', 5, 22), ('electron', 6, 13),
               ('muon', 6, 35), ('tau', 10, 23)]
CONDITIONS, ERRORS = {}, {}


def check(value, label):
    if not bool(value):
        raise AssertionError(label)
    CONDITIONS[label] = CONDITIONS.get(label, 0) + 1


def close(actual, expected, label, tolerance=2e-10):
    a, b = np.asarray(actual), np.asarray(expected)
    error = float(np.linalg.norm(a-b)/max(1., np.linalg.norm(b)))
    ERRORS[label] = max(ERRORS.get(label, 0.), error)
    check(error < tolerance, label)


def gammas(number):
    x = np.array([[0, 1], [1, 0]], complex)
    y = np.array([[0, -1j], [1j, 0]], complex)
    z = np.diag([1., -1.])
    result = []
    bits = number//2
    for k in range(bits):
        for matrix in (x, y):
            factors = [z]*k + [matrix] + [np.eye(2)]*(bits-k-1)
            out = np.ones((1, 1), complex)
            for factor in factors:
                out = np.kron(out, factor)
            result.append(out)
    if number % 2:
        out = np.ones((1, 1), complex)
        for _ in range(bits):
            out = np.kron(out, z)
        result.append(out)
    return result


def polynomial(depth, level):
    factors = [mp.mpf(level+j) for j in range(1, depth+1)]
    p = mp.fprod(factors)/factorial(depth)
    log = mp.fsum(1/a for a in factors)
    return p, p*log, p*(log*log-mp.fsum(1/a**2 for a in factors))


def local_operators(t):
    s = np.sqrt(1-t*t)
    beta = np.zeros((len(t), 4, 4))
    beta[:, 0, 2] = beta[:, 2, 0] = t
    beta[:, 0, 3] = beta[:, 3, 0] = -s
    beta[:, 1, 2] = beta[:, 2, 1] = -s
    beta[:, 1, 3] = beta[:, 3, 1] = -t
    radial = np.zeros((4, 4), complex)
    radial[1, 0] = radial[3, 2] = 1j
    radial[0, 1] = radial[2, 3] = -1j
    # At depth two only, the transverse Clifford vector stays in this
    # rotational matrix algebra. Its Killing-frame rotation is retained.
    beta0 = np.zeros((4, 4))
    beta0[0, 2] = beta0[2, 0] = 1
    beta0[1, 3] = beta0[3, 1] = -1
    transverse0 = np.zeros((4, 4))
    transverse0[0, 3] = transverse0[3, 0] = 1
    transverse0[1, 2] = transverse0[2, 1] = 1
    transverse = s[:, None, None]*beta0+t[:, None, None]*transverse0
    return beta, radial, transverse


def spatial_basis(depth, maximum, nodes):
    t, w = roots_jacobi(nodes, (depth-2)/2, (depth-2)/2)
    w /= w.sum()
    s = np.sqrt(1-t*t)
    values, frequencies, labels = [], [], []
    for block in (0, 1):
        for level in range(maximum+1):
            a = (level+depth-1)/(depth-1)*eval_gegenbauer(level, (depth-1)/2, t)
            b = eval_gegenbauer(level-1, (depth+1)/2, t) if level else np.zeros_like(t)
            g = comb(level+depth-1, depth-1)
            for sign in (1, -1):
                aa, bb = (a, b) if sign == 1 else (t*a-s*s*b, -(a+t*b))
                v = np.zeros((nodes, 4))
                v[:, block*2] = aa/np.sqrt(g)
                v[:, block*2+1] = s*bb/np.sqrt(g)
                values.append(v)
                frequencies.append(sign*(level+depth/2))
                labels.append((block, sign, level))
    basis = np.stack(values, axis=-1)
    gram = np.einsum('pai,paj,p->ij', basis, basis, w, optimize=True)
    close(gram, np.eye(len(labels)), 'complete rotational matrix basis has the physical Gram metric', 2e-9)
    beta, radial, transverse = local_operators(t)
    grading = np.einsum('pai,pab,pbj,p->ij', basis, beta, basis, w, optimize=True)
    expected = np.zeros_like(grading)
    for i, (block, sign, level) in enumerate(labels):
        expected[labels.index((1-block, -sign, level)), i] = 1.
    close(grading, expected, 'physical spatial grading maps every Dirac shell to its retained opposite partner', 2e-9)
    kinetic = np.diag(frequencies)
    close((kinetic@grading+grading@kinetic)/(np.linalg.norm(kinetic)*np.linalg.norm(grading)),
          0., 'full projected spatial Dirac operator anticommutes with parallel grading', 2e-9)
    return t, w, basis, np.array(frequencies), labels, beta, radial, transverse, expected


def exact_spatial_checks():
    t = sp.symbols('t', real=True)
    for depth in DEPTHS:
        h0 = sp.Rational(depth, 2)
        for level in range(5):
            a = sp.Rational(level+depth-1, depth-1)*sp.gegenbauer(level, sp.Rational(depth-1, 2), t)
            b = sp.gegenbauer(level-1, sp.Rational(depth+1, 2), t) if level else sp.Integer(0)
            for sign, aa, bb in [(1, a, b), (-1, t*a-(1-t*t)*b, -a-t*b)]:
                da = h0*aa+depth*t*bb-(1-t*t)*sp.diff(bb, t)
                db = sp.diff(aa, t)+(1-h0)*bb
                check(sp.expand(da-sign*(level+h0)*aa) == 0,
                      'exact radial Dirac scalar derivative retains the sphere connection')
                check(sp.expand(db-sign*(level+h0)*bb) == 0,
                      'exact radial Dirac vector derivative retains the sphere connection')
        alpha, beta0 = gammas(depth+1)[:depth], gammas(depth+1)[depth]
        rank = len(beta0)
        gamma = alpha[0]
        matrices = np.stack([np.eye(rank), -1j*gamma, beta0, -1j*gamma@beta0])/np.sqrt(rank)
        close(np.einsum('aij,bij->ab', matrices.conj(), matrices), np.eye(4),
              'actual Clifford matrices realize the normalized four-component algebra')
        for theta in (.21, 1.14, 2.63):
            tt, ss = np.cos(theta), np.sin(theta)
            frame = np.cos(theta/2)*np.eye(rank)-1j*np.sin(theta/2)*gamma
            physical_beta = frame.conj().T@beta0@frame
            bb, rr, vv = local_operators(np.array([tt]))
            project = lambda operator: np.einsum('aij,ik,bkj->ab', matrices.conj(), operator, matrices)
            close(project(physical_beta), bb[0], 'physical beta dependence agrees with the full spinor matrix projection')
            close(project(gamma), rr, 'radial current channel agrees with the full spinor matrix projection')
            for axis in range(1, depth):
                actual = project(frame.conj().T@alpha[axis]@frame)
                if depth == 2:
                    # The chosen gamma convention can reverse this axis.
                    expected = vv[0]
                    close(actual, expected if np.linalg.norm(actual-expected)<np.linalg.norm(actual+expected) else -expected,
                          'depth two retains its additional transverse current channel')
                else:
                    close(actual, 0., 'higher-depth transverse currents vanish within this exact rotational family')


def spatial_case(name, depth, index, epsilon, maximum, nodes, vector_weight=1.):
    level = index-1
    t, w, basis, frequencies, labels, beta, radial, transverse, grading = spatial_basis(depth, maximum, nodes)
    p, first, second = polynomial(depth, level)
    h = mp.mpf(level)+mp.mpf(depth)/2
    c = mp.mpf(epsilon)*first
    check(0 < c < 1, 'specified sector radius keeps every tested count branch in its open constitutive chart')
    x = mp.sqrt(1-c*c)
    a, b = mp.sqrt((1+c)/2), mp.sqrt((1-c)/2)
    lam = h/c
    omega = mp.mpf(epsilon)*p
    scalar_factor = 1-x*x*first/(h*second)
    vector_factor = vector_weight*(1-omega/lam)
    af, bf, xf = float(a), float(b), float(x)
    positive = basis[:, :, labels.index((0, 1, level))]
    negative = basis[:, :, labels.index((1, -1, level))]
    field = af*positive+bf*negative
    tangent = af*negative-bf*positive
    rho = np.sum(field*field, axis=1)
    close(np.einsum('pa,pab,pb->p', field, beta, field)/rho, xf,
          'inhomogeneous actual field retains the prescribed scalar-to-proper current ratio')
    scalar = np.einsum('pa,pai->pi', tangent, basis)/np.sqrt(rho[:, None])
    radial_source = (radial@field.T).T
    close(radial_source.real, 0., 'radial current response occupies the imaginary coefficient sector')
    vector = np.einsum('pa,pai->pi', radial_source.imag, basis)/np.sqrt(rho[:, None])
    free = np.diag(frequencies/float(lam))+xf*grading-np.eye(len(labels))
    lx = free+2*float(scalar_factor)*((scalar*w[:, None]).T@scalar)
    ly = free+2*float(vector_factor)*((vector*w[:, None]).T@vector)
    if depth == 2:
        transverse_source = np.einsum('pab,pb->pa', transverse, field)
        transverse_projection = np.einsum('pa,pai->pi', transverse_source, basis)/np.sqrt(rho[:, None])
        lx += 2*float(vector_factor)*((transverse_projection*w[:, None]).T@transverse_projection)
    close(lx, lx.T, 'real spatial Hessian includes all scalar and vector-current derivatives')
    close(ly, ly.T, 'imaginary spatial Hessian includes the reciprocal current derivatives')
    stationary = np.zeros(len(labels))
    stationary[labels.index((0, 1, level))] = af
    stationary[labels.index((1, -1, level))] = bf
    close(free@stationary, 0., 'actual projected stationary residual retains spatial kinetics and complete source')
    close(lx@stationary, 0., 'amplitude homogeneity gives its exact real stationary direction', 2e-8)
    close(ly@stationary, 0., 'phase symmetry gives its exact imaginary stationary direction', 2e-8)
    pair = np.eye(len(labels))[:, [labels.index((0, 1, level)), labels.index((1, -1, level))]]
    close((np.eye(len(labels))-pair@pair.T)@lx@pair, 0., 'old pair is exactly invariant inside the broader real spatial response', 2e-8)
    close((np.eye(len(labels))-pair@pair.T)@ly@pair, 0., 'old pair is exactly invariant inside the broader imaginary spatial response', 2e-8)
    if level == 0 and vector_weight == 1.:
        for harmonic in range(1, maximum+1):
            selected = [labels.index((0, 1, harmonic)), labels.index((1, -1, harmonic)),
                        labels.index((0, -1, harmonic-1)), labels.index((1, 1, harmonic-1))]
            lift = np.eye(len(labels))[:, selected]
            exact_x, exact_y = ground_block(depth, harmonic, c)
            close(lx@lift, lift@exact_x, 'ground scalar harmonic is an exact reducing real spatial block', 2e-8)
            close(ly@lift, lift@exact_y, 'ground scalar harmonic is an exact reducing imaginary spatial block', 2e-8)
    eigen = eigvals(ly@lx)
    growth = np.sqrt(-eigen.astype(complex)).real
    precision = 100*np.finfo(float).eps*len(labels)*max(1., np.linalg.norm(lx, 2)*np.linalg.norm(ly, 2))
    largest = np.argsort(growth)[-min(8, len(growth)):][::-1]
    record = {'name': name, 'depth': depth, 'mode_index': index,
              'cutoff': maximum, 'quadrature_nodes': nodes,
              'vector_weight': vector_weight,
              'complex_spatial_dimension': len(labels), 'epsilon': str(epsilon),
              'signed_shell_occupation': float(c), 'lambda': float(lam),
              'particle_frequency': float(omega),
              'maximum_growth_over_lambda': float(max(growth)),
              'maximum_growth_in_inverse_radius_units': float(max(growth)*float(lam)),
              'squared_frequency_resolution_over_lambda_squared': precision,
              'resolved_negative_squared_frequency_count': int(np.sum((eigen.real < -precision)&(abs(eigen.imag)<precision))),
              'resolved_complex_squared_frequency_count': int(np.sum(abs(eigen.imag)>precision)),
              'leading_growth_candidates': [dict(squared_frequency_real=float(eigen[i].real),
                                                squared_frequency_imaginary=float(eigen[i].imag),
                                                growth_in_inverse_radius_units=float(growth[i]*float(lam))) for i in largest],
              'full_spatial_stability_tested': False,
              'classification': 'finite Galerkin projection of an exactly reducing rotational spatial family'}
    return record


def ground_block(depth, harmonic, c):
    c = float(c)
    x = np.sqrt(1-c*c)
    a, b = np.sqrt((1+c)/2), np.sqrt((1-c)/2)
    u = np.sqrt((harmonic+depth-1)/(2*harmonic+depth-1))
    v = np.sqrt(harmonic/(2*harmonic+depth-1))
    h = depth/2
    lam = h/c
    _, first, second = polynomial(depth, 0)
    scalar_factor = 1-(1-c*c)*float(first)/(h*float(second))
    vector_factor = 1-c*c/(h*float(first))
    grading = np.zeros((4, 4))
    grading[0, 1] = grading[1, 0] = grading[2, 3] = grading[3, 2] = 1
    free = np.diag([harmonic+h, -harmonic-h, -harmonic+1-h, harmonic-1+h])/lam+x*grading-np.eye(4)
    scalar = np.array([-b*u, a*u, -b*v, a*v])
    radial = np.array([a*v, -b*v, -a*u, b*u])
    lx = free+2*scalar_factor*np.outer(scalar, scalar)
    ly = free+2*vector_factor*np.outer(radial, radial)
    if depth == 2:
        transverse = np.array([b*v, -a*v, -b*u, a*u])
        lx += 2*vector_factor*np.outer(transverse, transverse)
    return lx, ly


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pilot', action='store_true')
    parser.add_argument('--original-scale', action='store_true')
    parser.add_argument('--padding', type=int, default=6)
    parser.add_argument('--nodes-factor', type=int, default=4)
    parser.add_argument('--branch-scale', default='.3')
    parser.add_argument('--vector-weight', type=float, default=1.)
    args = parser.parse_args()
    mp.mp.dps = 60
    exact_spatial_checks()
    cases = []
    targets = [('ground', d, 1) for d in DEPTHS] if args.pilot else ASSIGNMENTS
    for name, depth, index in targets:
        last = max(n for _, d, n in targets if d == depth)
        epsilon = mp.mpf(1)/2**32 if args.original_scale else mp.mpf(args.branch_scale)/polynomial(depth, last-1)[1]
        maximum = index-1+args.padding
        nodes = max(48, args.nodes_factor*maximum+24)
        record = spatial_case(name, depth, index, epsilon, maximum, nodes, args.vector_weight)
        cases.append(record)
        print(json.dumps({'name': name, 'depth': depth, 'cutoff': maximum,
                          'growth': record['maximum_growth_in_inverse_radius_units'],
                          'resolved_negative': record['resolved_negative_squared_frequency_count'],
                          'resolved_complex': record['resolved_complex_squared_frequency_count']}), flush=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    report = {'experiment_key': 'inverse-current-actual-rotational-spatial-hessian',
              'created_utc': datetime.now(timezone.utc).isoformat(),
              'sources': {'experiments/inverse-current-radial.py': sha256(Path(__file__).read_bytes()).hexdigest()},
              'conditions': CONDITIONS, 'evaluations': sum(CONDITIONS.values()), 'maximum_errors': ERRORS,
              'cases': cases, 'arguments': vars(args),
              'limits': ['conditional ordinary spatial sphere, not compactification or deterministic electron trajectory replacement',
                         'matrix carrier and supplied count constitutive action retained from the inverse benchmark',
                         'finite Galerkin spectra are diagnostics, not a full continuum or nonlinear stability theorem',
                         'resolved finite-projection growth requires cutoff, quadrature and precision checks before physical interpretation',
                         'radius probes change epsilon explicitly and fit no physical field amplitude',
                         'vector_weight other than one is a changed potential with n=sqrt(rho squared-vector_weight times spatial current squared); it selects a preferred time direction',
                         'the rotational matrix family does not include all angular or column-polarization perturbations']}
    output = ROOT/('inverse-current-radial-results-'+stamp+'.json')
    output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'report': str(output), 'evaluations': report['evaluations'], 'conditions': len(CONDITIONS)}), flush=True)


if __name__ == '__main__':
    main()
