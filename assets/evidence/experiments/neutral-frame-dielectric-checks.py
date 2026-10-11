#!/usr/bin/env python3
"""Test a neutral ordinary-spatial frame with a squared middle-form dielectric."""
from pathlib import Path
from itertools import combinations
from fractions import Fraction
import argparse
import hashlib
import importlib.util
import json
import math

import numpy as np
import sympy as sp
from scipy.integrate import simpson


DEPTHS = (2, 3, 4, 5, 6, 10)
CHECKS = 0
CASES = []


def check(condition, message):
    global CHECKS
    CHECKS += 1
    if not condition:
        raise AssertionError(message)


def close(actual, expected, message, atol=2e-9, rtol=2e-9):
    check(np.allclose(actual, expected, atol=atol, rtol=rtol), message)


def hodge(even):
    m = even // 2
    basis = list(combinations(range(even), m))
    lookup = {item: i for i, item in enumerate(basis)}
    star = np.zeros((len(basis), len(basis)))
    for j, item in enumerate(basis):
        complement = tuple(i for i in range(even) if i not in item)
        permutation = item + complement
        star[lookup[complement], j] = (-1)**sum(permutation[i] > permutation[k]
            for i in range(even) for k in range(i + 1, even))
    return basis, star, (star + star.T)/2


def exterior(frame, basis):
    return np.array([np.linalg.det(frame[list(item), :]) for item in basis])


def geometric_cases():
    rng = np.random.default_rng(2026101019)
    for d in DEPTHS:
        even = d if d % 2 == 0 else d + 1
        m = even // 2
        basis, star, symmetric = hodge(even)
        close(star.T@star, np.eye(len(basis)), 'ordinary middle-form metric')
        gate = float(np.trace(symmetric.T@symmetric)/len(basis))
        max_relative_area = 0.
        for _ in range(12):
            frame = rng.normal(size=(even, m))
            derivative = rng.normal(size=(even, m))
            form = exterior(frame, basis)
            gram = frame.T@frame
            u2 = np.sum(frame*frame)
            determinant = np.linalg.det(gram)
            bound = (u2/m)**m
            close(np.dot(form, form), determinant, 'Cauchy-Binet gives the actual geometric middle-form norm')
            close(np.dot(form, star@form), 0, 'a decomposable middle form has zero unweighted self-wedge')
            check(determinant <= bound*(1 + 1e-12), 'one balanced Gram bound at every occupied depth')
            close(np.dot(symmetric@form, symmetric@form), gate*determinant,
                  'the common Hodge action gates every actual frame preparation')
            if gate > 0:
                plus = (form + star@form)/2
                minus = (form - star@form)/2
                close(np.dot(plus, star@plus) - np.dot(minus, star@minus), determinant,
                      'both weighted chiral self-contacts reproduce the geometric area norm')
                close(np.dot(plus, minus), 0, 'the two real chiral area components are orthogonal')
            radial_derivative2 = np.sum(frame*derivative)**2/u2
            check(radial_derivative2 <= np.sum(derivative*derivative)*(1 + 1e-12),
                  'radial norm gradient never exceeds complete frame gradient')
            rotation, _ = np.linalg.qr(rng.normal(size=(even, even)))
            if np.linalg.det(rotation) < 0:
                rotation[:, 0] *= -1
            rotated = exterior(rotation@frame, basis)
            close(np.dot(symmetric@rotated, symmetric@rotated), gate*determinant,
                  'common ordinary spatial rotation preserves the geometric dielectric')
            max_relative_area = max(max_relative_area, determinant/bound)
        u = 1.7
        balanced = np.zeros((even, m))
        balanced[:m, :] = np.eye(m)*u/math.sqrt(m)
        form = exterior(balanced, basis)
        close(np.dot(form, form), u**(2*m)/m**m, 'equal orthogonal frame saturates the determinant bound')
        p = 2*m
        n = d - 1
        gradient_coefficient = 2*p - n*(p - 2)
        mass_coefficient = n*(p - 2)
        finite_profile = p > 2 and gradient_coefficient > 0
        check(finite_profile == (d in (3, 4)), 'common geometric degree only admits occupied three/four finite-width profiles')
        CASES.append({'case': f'geometric-frame-depth-{d}', 'real_depth': d,
                      'even_source_carrier': even, 'frame_vectors': m,
                      'balanced_dielectric_power': p, 'hodge_coefficient': gate,
                      'balanced_coefficient_fraction': str(Fraction(1, m**m)),
                      'pohozaev_gradient_coefficient': gradient_coefficient,
                      'pohozaev_mass_coefficient': mass_coefficient,
                      'nongated_regular_finite_width_profile_possible': finite_profile,
                      'active_color_source_permitted': gate > 0,
                      'random_frame_comparisons': 12,
                      'largest_sampled_area_fraction': max_relative_area})
    a, b, angle = sp.symbols('a b theta', positive=True, real=True)
    area = a*a*b*b*sp.sin(angle)**2
    check(sp.simplify((a*a + b*b)**2/4 - area -
                      ((a*a - b*b)**2/4 + a*a*b*b*sp.cos(angle)**2)) == 0,
          'exact imbalance and nonorthogonality area deficit')


def tube_frame_cases(n, solution, profile):
    # Raw frame coefficient kappa=4 makes the balanced scalar coefficient one.
    raw_kap = 4.
    q = 1.
    quartic = profile['quartic_integral']
    amplitude = (2*q*q/quartic**2)**(1/6)
    mass = amplitude*amplitude*profile['mass_integral']
    scalar = amplitude*amplitude*(profile['gradient_integral'] + profile['mass_integral'])/2
    electric = q*q/(2*amplitude**4*quartic)
    tension = scalar + electric
    close(electric/tension, 1/3, 'same scalar tube normalization in the full neutral-frame action')
    for imbalance in (.1, .3, .7):
        denominator = math.sqrt(2*math.cosh(2*imbalance))
        first = math.exp(imbalance)/denominator
        second = math.exp(-imbalance)/denominator
        close(first*first + second*second, 1, 'relative amplitude variation retains complete frame norm')
        dielectric_integral = raw_kap*first*first*second*second*amplitude**4*quartic
        energy = q*q/(2*dielectric_integral) + scalar
        close(energy - tension, electric*(math.cosh(2*imbalance)**2 - 1), 'exact positive imbalance energy')
        check(energy > tension, 'equal-length frame is preferred without an imposed amplitude fit')
        CASES.append({'case': f'frame-imbalance-n{n}-s{imbalance}', 'real_depth': n + 1,
                      'relative_log_amplitude': imbalance, 'reference_tension': tension,
                      'complete_energy': energy, 'positive_excess': energy - tension})
    for theta in (.4, .9, 1.2):
        area_fraction = math.sin(theta)**2
        energy = electric/area_fraction + scalar
        check(energy > tension, 'orthogonal physical deformation directions maximize flux support')
        CASES.append({'case': f'frame-angle-n{n}-theta{theta}', 'real_depth': n + 1,
                      'relative_angle': theta, 'area_fraction': area_fraction,
                      'complete_energy': energy, 'positive_excess': energy - tension})
    x = np.linspace(0, 20, 20001)
    v, dv = solution.sol(x)
    u, du = amplitude*v, amplitude*dv
    weight = x**(n - 1)
    area = 2*math.pi**(n/2)/math.gamma(n/2)
    def integrate(value):
        return area*simpson(weight*value, x=x)
    for twist in (.2, .6):
        theta = twist*(1 - np.exp(-x*x))
        derivative_theta = 2*twist*x*np.exp(-x*x)
        frame = np.zeros((len(x), 4, 2))
        derivative = np.zeros_like(frame)
        for slot in (0, 1):
            frame[:, slot, slot] = u*np.cos(theta)/math.sqrt(2)
            frame[:, slot + 2, slot] = u*np.sin(theta)/math.sqrt(2)
            derivative[:, slot, slot] = (du*np.cos(theta) - u*np.sin(theta)*derivative_theta)/math.sqrt(2)
            derivative[:, slot + 2, slot] = (du*np.sin(theta) + u*np.cos(theta)*derivative_theta)/math.sqrt(2)
        gram = np.einsum('nik,nil->nkl', frame, frame)
        determinant = np.linalg.det(gram)
        close(determinant, u**4/4, 'varying common physical frame retains the dielectric area')
        complete_gradient = np.sum(derivative*derivative, axis=(1, 2))
        close(complete_gradient, du*du + u*u*derivative_theta**2, 'complete frame twist gradient')
        excess = integrate(complete_gradient - du*du)/2
        close(excess, integrate(u*u*derivative_theta**2)/2, 'positive orientation gradient cost')
        check(excess > 0, 'spatial frame twist is not an omitted instability')
        CASES.append({'case': f'frame-twist-n{n}-theta{twist}', 'real_depth': n + 1,
                      'twist_amplitude': twist, 'positive_gradient_excess': excess,
                      'rigid_common_rotations_remain_free': True})
    close(mass, integrate(u*u), 'complete neutral norm agrees with the scalar branch')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--profile-code', type=Path, default=Path(__file__).with_name('flux-tube-checks.py'))
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    geometric_cases()
    spec = importlib.util.spec_from_file_location('profiles', args.profile_code)
    profiles = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(profiles)
    for n in (2, 3):
        solution, profile = profiles.solve_profile(n, 25.)
        tube_frame_cases(n, solution, profile)
    report = {'title': 'Neutral Geometric Frames Realize the Hodge-Gated Quartic Dielectric',
              'sources': {'experiments/neutral-frame-dielectric-checks.py': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                          'experiments/hodge-flux-tube-checks.py': hashlib.sha256(args.profile_code.read_bytes()).hexdigest()},
              'checks_passed': CHECKS, 'case_count': len(CASES), 'occupied_depths': list(DEPTHS),
              'status': 'explicit positive retained geometric-mode action; master-field reduction and vacuum Cauchy problem remain open',
              'conclusion': 'Squared middle-form volume gives a quartic dielectric on the active four-carrier. Exact Gram and gradient bounds reduce the entire neutral-frame fixed-flux energy to its balanced scalar branch. Amplitude imbalance, nonorthogonality and spatial frame twists cannot lower the tube minimum. All occupied-sector Hodge zeros and geometric-degree profile controls use the same law.',
              'cases': CASES}
    args.report.write_text(json.dumps(report, indent=2) + '\n')
    print(f'{CHECKS} checks passed across {len(CASES)} cases; {args.report}')


if __name__ == '__main__':
    main()
