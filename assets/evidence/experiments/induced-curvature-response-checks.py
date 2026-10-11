#!/usr/bin/env python3
"""Check squared-linear-curvature symbols and a retained-orientation comparison."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import math

import numpy as np
from scipy.integrate import quad
from scipy.linalg import eigh

CHECKS = 0
CASES = []


def check(condition, label):
    global CHECKS
    CHECKS += 1
    if not condition:
        raise AssertionError(label)


def close(a, b, label, tolerance=3e-8):
    check(np.allclose(a, b, atol=tolerance, rtol=tolerance), label)


def symbols(second, depth, direction):
    q = second.shape[0]
    matrices = []
    for i in range(depth):
        vector = second[:, i, :]@direction
        matrices.append(np.stack([(np.outer(np.eye(q)[a], vector)-np.outer(vector, np.eye(q)[a])).ravel()
                                  for a in range(q)], axis=1))
    kinetic = sum(x.T@x for x in matrices)
    along = sum(direction[i]*matrices[i] for i in range(depth))
    defect = along.T@along
    magnetic = kinetic-defect
    return kinetic, magnetic, defect, matrices


def bump(x):
    return math.exp(1-1/(1-x*x)) if abs(x) < 1 else 0.


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--geometry-source', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location('nested_geometry', args.geometry_source)
    geometry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(geometry)
    hessian = geometry.full_hessian()
    rng = np.random.default_rng(2026101052)
    expected = {2: ([4, 4, 2, 2, 4, 4, 4, 4], [2, 2, 2, 0, 2, 2, 2, 2]),
                3: ([4, 4, 4, 6, 6, 6], [4, 2, 2, 4, 4, 4]),
                4: ([6, 6, 6, 6, 8, 8], [6, 4, 4, 4, 6, 6]),
                5: ([2, 2, 4, 4], [2, 2, 4, 4]),
                6: ([2, 2, 4, 4], [2, 2, 4, 4])}
    for d in geometry.DEPTHS:
        even = d+d % 2
        q = 10-even
        m = even//2
        h = int(m % 2 == 0)
        if not q:
            CASES.append({'case': 'depth-10-no-normal-connection', 'real_depth': d,
                          'hodge_active': False, 'normal_curvature_response_present': False})
            continue
        second = hessian[even:, :even, :even]
        direction = np.zeros(even); direction[0] = 1
        kinetic, magnetic, defect, _ = symbols(second, d, direction)
        close(kinetic, np.diag(expected[d][0]), 'actual graph axial kinetic symbol')
        close(magnetic, np.diag(expected[d][1]), 'actual graph axial magnetic symbol')
        for _ in range(4):
            direction[:d] = rng.normal(size=d); direction /= np.linalg.norm(direction)
            a, b, c, matrices = symbols(second, d, direction)
            direct = sum((direction[j]*matrices[i]-direction[i]*matrices[j]).T@
                         (direction[j]*matrices[i]-direction[i]*matrices[j])
                         for i in range(d) for j in range(i+1, d))
            close(direct, b, 'exterior spatial curvature identity gives magnetic symbol')
            close(a-b, c, 'difference is a positive longitudinal curvature square')
            check(np.linalg.eigvalsh(b).min() > -1e-9, 'magnetic symbol is nonnegative')
            check(np.linalg.eigvalsh(c).min() > -1e-9, 'electric symbol bounds magnetic symbol')
            for k in (.1, 1., 10.):
                frequencies = eigh(k*k*np.eye(q)+h*k**4*b, np.eye(q)+h*k*k*a, eigvals_only=True)
                check(frequencies.min() > 0 and frequencies.max() <= k*k*(1+1e-9),
                      'positive frozen squared frequencies have phase speeds at most the base speed')
        CASES.append({'case': f'induced-curvature-depth-{d}', 'real_depth': d,
                      'hodge_active': bool(h), 'axial_electric_symbol_diagonal': expected[d][0],
                      'axial_magnetic_symbol_diagonal': expected[d][1],
                      'independent_gauss_source_constraint_survives_induced_reduction': False,
                      'frozen_symbol_is_complete_variable_geometry_solution': False,
                      'background_curvature_second_variation_included': False,
                      'arbitrary_real_graph_waves_preserve_unitary_normal_carrier': False})
        if not h:
            continue
        for a, b in sorted(set(zip(expected[d][0], expected[d][1]))):
            length = math.sqrt(a)
            tail = quad(lambda y: math.exp(-(2-y)/length)*bump(y)/(2*length), -1., 1., epsabs=1e-13)[0]
            check(tail > 0, 'a compact local probe has an exterior frozen acceleration response')
            unforced_tail = (1-b/a)*tail/a
            check((unforced_tail > 0) == (a > b), 'compact unforced displacement also has an exterior acceleration tail when a exceeds b')
            # A retained tilt mode replaces the exact derivative locking relation.
            lock = 1/3
            bare_speed_squared = 1-lock
            for k in (0., .01, .1, 1., 10., 100.):
                stiffness = np.array([[(bare_speed_squared+lock)*k*k, 1j*lock*k],
                                      [-1j*lock*k, lock+b*k*k]])
                inertia = np.diag([1., a])
                eigenvalues = eigh(stiffness, inertia, eigvals_only=True)
                check(eigenvalues.min() > -1e-9, 'finite-lock retained-mode energy is positive')
                if k:
                    for z in eigenvalues:
                        determinant = ((bare_speed_squared+lock)*k*k-z)*(lock+b*k*k-a*z)-lock*lock*k*k
                        close(determinant/(1+k**4+z*z), 0, 'full two-mode poles obey the exact Schur determinant')
                else:
                    close(eigenvalues, [0., lock/a], 'retained orientation has its own deformation clock')
            check(0 <= b/a <= 1 and bare_speed_squared+lock == 1,
                  'retained system has finite principal speeds bounded by the common unit speed')
            CASES.append({'case': f'active-depth-{d}-axial-symbol-{a}-{b}', 'real_depth': d,
                          'electric_symbol_a': a, 'magnetic_symbol_b': b,
                          'high_frequency_locked_speed_squared': b/a,
                          'compact_probe_acceleration_at_x_2': tail,
                          'compact_unforced_displacement_acceleration_at_x_2': unforced_tail,
                          'retained_lock_coefficient': lock,
                          'retained_bare_speed_squared': bare_speed_squared,
                          'retained_orientation_gap_squared': lock/a,
                          'independent_orientation_is_exact_integrable_graph': False})
    report = {'title': 'Induced Normal Curvature Gives Mixed Spatial Inertia and a Retained-Orientation Comparison',
              'sources': {'experiments/induced-curvature-response-checks.py': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                          'experiments/nested-normal-curvature-checks.py': hashlib.sha256(args.geometry_source.read_bytes()).hexdigest()},
              'checks_passed': CHECKS, 'case_count': len(CASES), 'occupied_depths': list(geometry.DEPTHS),
              'scope': 'actual squared-first-curvature-variation symbols, not the complete nonzero-background curvature Hessian; frozen real-normal response and finite-lock retained-orientation comparison',
              'cases': CASES}
    args.report.write_text(json.dumps(report, indent=2)+'\n')
    print(f'{CHECKS} induced-response checks passed across {len(CASES)} cases')


if __name__ == '__main__':
    main()
