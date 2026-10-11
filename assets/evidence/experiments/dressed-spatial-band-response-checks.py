#!/usr/bin/env python3
"""Kinetic dressing of actual spatial-band geometry and gap-weighted response."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import importlib.util
import json
from math import comb, sqrt
from pathlib import Path

import mpmath as mp
import numpy as np
import scipy.linalg as la


CHECKS = 0
CASES = []


def check(condition, message):
    global CHECKS
    if not condition:
        raise AssertionError(message)
    CHECKS += 1


def dressed_data(depth, cutoff, strength):
    mp.mp.dps = 60
    mu = mp.mpf(str(strength))
    weights = [mp.mpf(comb(l+depth-1, depth-1)) for l in range(cutoff+1)]
    omega = [mp.mpf(l)+mp.mpf(depth)/2 for l in range(cutoff+1)]
    total = sum(weights)
    if cutoff == 0:
        z = omega[0]+mu
    else:
        def secular(value):
            return 1-mu*sum(g/(value-w) for g, w in zip(weights, omega))
        z = mp.findroot(secular, (omega[-1]+mu*weights[-1]/2, omega[-1]+mu*total),
                        solver='bisect', maxsteps=260, tol=mp.mpf('1e-53'))
    check(abs(1-mu*sum(g/(z-w) for g, w in zip(weights, omega))) < mp.mpf('1e-45'),
          'complete kinetic-contact secular root')
    raw = [g/(z-w)**2 for g, w in zip(weights, omega)]
    pp = [p/sum(raw) for p in raw]
    mean = sum(l*p for l, p in enumerate(pp))
    variance = sum((l-mean)**2*p for l, p in enumerate(pp))
    beta = mean/depth
    curvature = beta*(beta+1)
    metric = (depth-1)*curvature+variance/depth
    t0 = sum(p*mp.mpf(l*(l+depth))/(depth**2*(z-w))
             for l, (p, w) in enumerate(zip(pp, omega)))
    t1 = sum(p*(mp.mpf(l)/depth-beta)**2/(z-w)
             for l, (p, w) in enumerate(zip(pp, omega)))
    check(t0 >= 0 and t1 >= 0, 'positive gap-weighted tensors on top complement')
    return {'z': z, 'p': pp, 'mean': mean, 'variance': variance, 'curvature': curvature,
            'metric': metric, 't0': t0, 't1': t1, 'omega': omega, 'weights': weights}


def finite_spatial_comparison(source):
    spec = importlib.util.spec_from_file_location('frozen_spatial_band', source)
    spatial = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(spatial)
    for depth in range(2, 7):
        gamma = spatial.gammas(depth)
        rank = len(gamma[0])
        shell_basis, shell_u, shell_d, shell_g = [], [], [], []
        for level in range(3):
            w, dd, g, _ = spatial.shell_data(depth, level, gamma)
            basis = la.orth(np.concatenate([w]+dd, axis=1))
            shell_basis.append(basis)
            shell_u.append(basis.conj().T@w/sqrt(g))
            shell_d.append([basis.conj().T@v/sqrt(g) for v in dd])
            shell_g.append(g)
            total = sum(shell_g)
            raw_u = np.vstack([sqrt(g/total)*u for g, u in zip(shell_g, shell_u)])
            h0 = la.block_diag(*[(l+depth/2)*np.eye(len(b.T)) for l, b in enumerate(shell_basis)])
            for strength in (.1, 1., 10.):
                data = dressed_data(depth, level, strength)
                z = float(data['z'])
                aa = [sqrt(float(p)) for p in data['p']]
                dressed_u = np.vstack([a*u for a, u in zip(aa, shell_u)])
                derivatives = [np.vstack([a*dd[axis] for a, dd in zip(aa, shell_d)]) for axis in range(depth)]
                h = h0+strength*total*raw_u@raw_u.conj().T
                check(np.allclose(h@dressed_u, z*dressed_u, atol=1e-8), 'actual complete spatial dressed eigenspace')
                eigen, vectors = la.eigh(h)
                top = np.abs(eigen-z) < 1e-7
                check(int(sum(top)) == rank, 'dressed bright multiplet has unchanged spinor rank')
                inverse = np.zeros_like(eigen)
                inverse[~top] = 1/(z-eigen[~top])
                resolvent = (vectors*inverse)@vectors.conj().T
                gap_tensor = np.block([[da.conj().T@resolvent@db for db in derivatives] for da in derivatives])
                t0, t1 = float(data['t0']), float(data['t1'])
                for axis in range(depth):
                    for b in range(depth):
                        expected = ((depth-1)*t0+t1)*np.eye(rank) if axis == b else (t1-t0)*gamma[axis]@gamma[b]
                        check(np.allclose(gap_tensor[axis*rank:(axis+1)*rank, b*rank:(b+1)*rank],
                                          expected, rtol=1e-7, atol=1e-8), 'exact gap-weighted spatial response')
                target = sorted([depth*t1]*rank+[depth*t0]*((depth-1)*rank))
                check(np.allclose(np.linalg.eigvalsh(gap_tensor), target, rtol=1e-7, atol=1e-8),
                      'independent gap tensor Clifford spectrum')
                radial = np.diag([float(w) for w in data['omega']])+strength*np.outer(np.sqrt(shell_g), np.sqrt(shell_g))
                re, rv = la.eigh(radial)
                norm = np.array(aa)
                jet = norm*(np.arange(level+1)/depth-float(data['mean'])/depth)
                rmask = np.abs(re-z) > 1e-7
                radial_inverse = (rv[:, rmask]/(z-re[rmask]))@rv[:, rmask].T
                check(np.isclose(jet@radial_inverse@jet, t1, rtol=1e-7, atol=1e-9),
                      'Fredholm weighted-sum identity versus independent radial spectrum')
                q = np.block([[da.conj().T@db-(dressed_u.conj().T@da).conj().T@(dressed_u.conj().T@db)
                               for db in derivatives] for da in derivatives])
                curvature, metric = float(data['curvature']), float(data['metric'])
                for axis in range(depth):
                    for b in range(depth):
                        expected = metric*np.eye(rank) if axis == b else -curvature*gamma[axis]@gamma[b]
                        check(np.allclose(q[axis*rank:(axis+1)*rank, b*rank:(b+1)*rank], expected, atol=1e-8),
                              'dressed geometric moment law')
                CASES.append({'name': 'actual-spatial-dressed-response', 'real_depth': depth, 'mode_index': level+1,
                              'parameters': {'mu_times_radius': strength, 'sphere_radius': 1,
                                             'active_physical_dimension': len(h),
                                             'complete_physical_band_dimension': rank*total,
                                             'omitted_dark_modes': 'orthogonal to every evaluation/first-derivative source; exact zero contribution'},
                              'measurements': {'dressed_frequency': z, 'curvature': curvature,
                                               'metric_diagonal': metric, 'gap_tensor_Clifford_eigenvalue': depth*t1,
                                               'gap_tensor_twistor_eigenvalue': depth*t0,
                                               'top_occupation_inertia_sign': 'negative correction -2 I times this positive gap tensor'}})


def maintained_branches():
    assignments = [('W', 2, 76), ('Z', 2, 81), ('Higgs', 2, 95), ('down', 3, 1),
                   ('strange', 3, 4), ('up', 4, 3), ('charm', 4, 20), ('top', 4, 72),
                   ('nu1', 5, 10), ('nu2', 5, 15), ('nu3', 5, 22),
                   ('electron', 6, 13), ('muon', 6, 35), ('tau', 10, 23)]
    for strength in (.1, 1., 10.):
        for label, depth, n in assignments:
            data = dressed_data(depth, n-1, strength)
            uniform_ratio = (depth+1)**2
            ratio_q = (1+depth**2*data['curvature']/data['variance']) if n > 1 else None
            ratio_gap = data['t0']/data['t1'] if n > 1 else None
            CASES.append({'name': 'maintained-branch-dressed-geometry', 'real_depth': depth, 'mode_index': n,
                          'parameters': {'particle': label, 'mu_times_radius': strength,
                                         'one_common_coefficient_per_comparison': True,
                                         'sphere_radius': 'R; frequencies in R^-1, geometry in R^-2, gap tensor in R^-1'},
                          'measurements': {'contact_count': comb(n+depth-1, depth),
                                           'dressed_frequency': str(data['z']), 'mean_shell_degree': str(data['mean']),
                                           'shell_degree_variance': str(data['variance']),
                                           'quantum_tensor_ratio': str(ratio_q), 'gap_weighted_tensor_ratio': str(ratio_gap),
                                           'uniform_count_tensor_ratio': uniform_ratio if n > 1 else None,
                                           'gap_ratio_relative_change': str(ratio_gap/uniform_ratio-1) if n > 1 else None}})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--spatial-source', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    source = Path(args.spatial_source)
    finite_spatial_comparison(source)
    maintained_branches()
    report = {'experiment_key': 'dressed-spatial-band-response-20261009',
              'created_utc': datetime.now(timezone.utc).isoformat(),
              'sources': {'experiments/dressed-spatial-band-response-checks.py': sha256(Path(__file__).read_bytes()).hexdigest(),
                          'experiments/spatial-band-curvature-checks.py': sha256(source.read_bytes()).hexdigest()},
              'scope': 'Common kinetic-contact action, exact dressed spatial geometry and complement-gap tensor. Its occupied top-branch adiabatic inertia correction is negative; this is not complete particle inertia or a new empirical mass law.',
              'summary': {'checks_passed': CHECKS, 'case_count': len(CASES)}, 'cases': CASES}
    Path(args.output).write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report['summary']))


if __name__ == '__main__':
    main()
