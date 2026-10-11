#!/usr/bin/env python3
"""Check one positive neutral-mode dielectric tube at every occupied depth."""
from pathlib import Path
from fractions import Fraction
from itertools import combinations
import argparse
import hashlib
import json
import math
import platform

import numpy as np
import scipy
import sympy as sp
from scipy.integrate import quad, simpson, solve_bvp
from scipy.sparse import diags
from scipy.sparse.linalg import LinearOperator, eigsh


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


def hodge_coefficient(d):
    even = d if d % 2 == 0 else d + 1
    m = even // 2
    basis = list(combinations(range(even), m))
    lookup = {v: i for i, v in enumerate(basis)}
    star = np.zeros((len(basis), len(basis)))
    for j, item in enumerate(basis):
        complement = tuple(i for i in range(even) if i not in item)
        permutation = item + complement
        sign = (-1)**sum(permutation[i] > permutation[k]
                         for i in range(even) for k in range(i + 1, even))
        star[lookup[complement], j] = sign
    close(star @ star, (-1)**m * np.eye(len(basis)), 'middle Hodge square')
    symmetric = (star + star.T) / 2
    return float(np.trace(symmetric.T @ symmetric) / len(basis))


def algebraic_checks():
    n, k, mass, p = sp.symbols('n K M p', real=True)
    scale = (n - 2) * k / 2 + n * mass / 2 - n * (k + mass) / 4
    check(sp.simplify(4 * scale - ((n - 4) * k + n * mass)) == 0,
          'cubic Pohozaev identity follows from two independent variations')
    general = (n - 2) * k / 2 + n * mass / 2 - n * (k + mass) / p
    check(sp.simplify(-2 * p * general - ((2 * p - n * (p - 2)) * k - n * (p - 2) * mass)) == 0,
          'general dielectric exponent support identity')
    a, width, ck, cm, cp, q, kap = sp.symbols('a w K M P Q kappa', positive=True)
    energy = q**2 / (2 * kap * a**4 * width**n * cp) + a**2 * (width**(n - 2) * ck + width**n * cm) / 2
    optimal_a6 = 2 * q**2 / (kap * width**n * cp * (width**(n - 2) * ck + width**n * cm))
    check(sp.simplify(sp.diff(energy, a) * a**5 - (width**(n - 2) * ck + width**n * cm) * (a**6 - optimal_a6)) == 0,
          'amplitude is varied in the complete positive energy')
    reduced_width_log_derivative = (n - 4) / width + 4 * width * cm / (ck + width**2 * cm)
    check(sp.simplify(reduced_width_log_derivative -
                      ((n - 4) * ck + n * width**2 * cm) / (width * (ck + width**2 * cm))) == 0,
          'positive width requires n below four')
    for d in DEPTHS:
        transverse = d - 1
        gate = hodge_coefficient(d)
        support = transverse < 4
        check((gate > 0 and support) == (d in (3, 4)), 'one common law leaves only occupied three/four')
        CASES.append({'case': f'occupied-depth-{d}', 'real_depth': d,
                      'transverse_dimension': transverse, 'hodge_coefficient': gate,
                      'pohozaev_gradient_coefficient': 4 - transverse,
                      'pohozaev_mass_coefficient': transverse,
                      'nonzero_quartic_profile_possible': support,
                      'active_finite_width_tube_possible': bool(gate > 0 and support),
                      'exclusion_reason': None if gate > 0 and support else
                      'zero geometric dielectric' if not gate else 'positive integral identity',
                      'nongated_profile_exclusion': transverse >= 4})
    for exponent in (Fraction(3), Fraction(4), Fraction(5), Fraction(6)):
        by_depth = []
        for d in DEPTHS:
            coefficient = 2 * exponent - (d - 1) * (exponent - 2)
            by_depth.append({'real_depth': d, 'gradient_coefficient': str(coefficient),
                             'nongated_support_not_excluded': coefficient > 0})
        CASES.append({'case': f'dielectric-exponent-{exponent}', 'dielectric_power': str(exponent),
                      'classification': 'necessary Pohozaev support test; no extra profiles solved',
                      'occupied_depths': by_depth})


def solve_profile(n, radius):
    x = np.unique(np.concatenate((np.linspace(0, 4, 241), np.linspace(4, radius, 241))))
    central = {1: math.sqrt(2), 2: 2.2062, 3: 4.3374}[n]
    width = {1: 1., 2: .8, 3: .5}[n]
    v = central / np.cosh(x / width)
    derivative = -v * np.tanh(x / width) / width
    def equation(r, y):
        return np.vstack((y[1], y[0] - y[0]**3))
    def boundary(left, right):
        return np.array([left[1], right[1] + (1 + (n - 1) / (2 * radius)) * right[0]])
    solution = solve_bvp(equation, boundary, x, np.vstack((v, derivative)),
                         S=np.array([[0., 0.], [0., -(n - 1)]]), tol=2e-8, max_nodes=8000)
    check(solution.status == 0, 'nonlinear profile converged')
    check(central / 2 < solution.sol(0)[0] < 2 * central, 'profile is not the zero branch')
    sample = solution.sol(np.linspace(0, radius, 2001))[0]
    check(np.min(sample) >= -1e-9, 'profile has no radial node')
    check(np.max(np.diff(sample)) < 1e-8, 'positive ground profile is radially decreasing')
    area = 2 * math.pi**(n / 2) / math.gamma(n / 2)
    def integral(kind):
        def integrand(r):
            value, gradient = solution.sol(r)
            return r**(n - 1) * (value**2 if kind == 0 else gradient**2 if kind == 1 else value**4)
        return area * quad(integrand, 0, radius, epsabs=2e-8, epsrel=2e-8, limit=1000)[0]
    mass, gradient, quartic = (integral(i) for i in range(3))
    close(gradient + mass, quartic, 'integrated profile equation')
    close((4 - n) * gradient, n * mass, 'independent spatial scaling identity')
    case = {'case': f'profile-n{n}-radius-{radius}', 'transverse_dimension': n,
            'radius': radius, 'central_value': float(solution.sol(0)[0]),
            'mass_integral': mass, 'gradient_integral': gradient, 'quartic_integral': quartic,
            'equation_relative_error': abs(gradient + mass - quartic) / quartic,
            'pohozaev_relative_error': abs((4 - n) * gradient - n * mass) / (n * mass),
            'nodes': len(solution.x), 'maximum_rms_residual': float(max(solution.rms_residuals))}
    if n == 1:
        close([mass, gradient, quartic], [4, 4/3, 16/3], 'exact one-dimensional sech integrals')
        close(sample, math.sqrt(2) / np.cosh(np.linspace(0, radius, 2001)), 'exact sech profile')
    CASES.append(case)
    return solution, case


def energy_checks(n, solution, profile):
    mass, gradient, quartic = (profile[k] for k in ('mass_integral', 'gradient_integral', 'quartic_integral'))
    for q, kap, omega in ((1., 1., 1.), (2.1, .7, 1.4), (.4, 2.3, .6)):
        amplitude = (2 * q*q * omega**(2*n - 2) / (kap * quartic**2))**(1/6)
        i4 = amplitude**4 * omega**(-n) * quartic
        electric = q*q / (2 * kap * i4)
        kinetic = amplitude**2 * omega**(2 - n) * gradient / 2
        restoring = amplitude**2 * omega**(2 - n) * mass / 2
        tension = electric + kinetic + restoring
        formula = .75 * (2 * q*q * quartic / kap)**(1/3) * omega**((4 - n)/3)
        close(tension, formula, 'tension formula uses complete scalar and electric energy')
        close([electric/tension, kinetic/tension, restoring/tension],
              [1/3, n/6, (4 - n)/6], 'exact energy fractions')
        transverse_stress = 2 * kinetic / n - kinetic - restoring + electric
        close(transverse_stress, 0, 'integrated stress in every transverse direction vanishes')
        for beta in (.2, .65):
            gamma = 1 / math.sqrt(1 - beta*beta)
            # Boosted fields have E_z'=gamma E_z and B_iz'=gamma beta E_z.
            boosted_electric = gamma * (1 + beta*beta) * electric
            boosted_scalar = (kinetic + restoring + (gamma*gamma - 1) * 2 * kinetic / n) / gamma
            momentum = gamma * beta * (2 * electric + 2 * kinetic / n)
            close(boosted_electric + boosted_scalar, gamma * tension, 'complete transverse boost energy')
            close(momentum, gamma * beta * tension, 'complete transverse boost momentum')
        CASES.append({'case': f'tension-n{n}-Q{q}', 'transverse_dimension': n,
                      'classical_commuting_flux': q, 'dielectric_coefficient': kap,
                      'neutral_restoring_frequency': omega, 'amplitude': amplitude, 'tension': tension,
                      'energy_fractions_exact': {'electric': '1/3', 'gradient': str(Fraction(n, 6)),
                                                 'restoring': str(Fraction(4 - n, 6))},
                      'integrated_transverse_stress': transverse_stress,
                      'transverse_inertia_per_length_in_c1_units': tension,
                      'quantum_Casimir_scaling_derived': False})
    # Independent discrete flux minimization at a fixed nonzero profile.
    r = (np.arange(800) + .5) * (5/800)
    weight = r**(n - 1) * (5/800)
    epsilon = solution.sol(r)[0]**4
    total = np.dot(weight, epsilon)
    optimum = epsilon / total
    disturbance = np.sin(1.3 * r)
    disturbance -= np.dot(weight, epsilon * disturbance) / total
    trial = optimum + .2 * epsilon * disturbance
    close(np.dot(weight, trial), 1, 'varied electric displacement retains total flux')
    direct = np.dot(weight, trial*trial / (2 * epsilon))
    remainder = np.dot(weight, (trial - optimum)**2 / (2 * epsilon))
    close(direct, 1/(2*total) + remainder, 'exact positive constrained electric-energy remainder')
    check(remainder > 0, 'nonoptimal flux redistribution costs positive energy')
    # Direct energy second differences independently check the nonlocal Hessian.
    x = np.linspace(0, 20, 20001)
    v, dv = solution.sol(x)
    measure = x**(n - 1)
    p_radial = simpson(measure * v**4, x=x)
    q2 = p_radial**2 / 2
    def energy(value, derivative):
        return q2 / (2 * simpson(measure * value**4, x=x)) + \
            simpson(measure * (derivative**2 + value**2), x=x) / 2
    for rate in (1., 1.7, 2.4):
        direction = np.exp(-rate*x*x)
        derivative = -2*rate*x*direction
        b = simpson(measure * v**3 * direction, x=x)
        hessian = simpson(measure * (derivative**2 + (1 - 3*v*v)*direction**2), x=x) + 8*b*b/p_radial
        delta = 2e-4
        difference = (energy(v + delta*direction, dv + delta*derivative) -
                      2*energy(v, dv) + energy(v - delta*direction, dv - delta*derivative)) / delta**2
        close(difference, hessian, 'finite-difference variation of complete fixed-flux energy', atol=3e-6, rtol=2e-6)
        check(hessian > 0, 'tested radial shape direction has positive fixed-flux second variation')
    CASES.append({'case': f'flux-and-hessian-variation-n{n}', 'transverse_dimension': n,
                  'positive_flux_remainder': remainder, 'shape_directions_checked': 3})


def spectrum(n, solution, cells, angular):
    radius = 20.
    step = radius / cells
    edges = np.arange(cells + 1) * step
    r = (edges[1:] + edges[:-1]) / 2
    weight = (edges[1:]**n - edges[:-1]**n) / n
    conductance = edges[1:-1]**(n - 1) / step
    diagonal = np.zeros(cells)
    diagonal[:-1] += conductance
    diagonal[1:] += conductance
    diagonal[-1] += radius**(n - 1) * 2 / step
    v = solution.sol(r)[0]
    off = -conductance / np.sqrt(weight[:-1] * weight[1:])
    local = diags((off, diagonal/weight + 1 - 3*v*v + angular*(angular + n - 2)/r**2, off),
                  (-1, 0, 1)).tocsr()
    low = np.sort(eigsh(local, k=3, which='SA', tol=1e-9, return_eigenvectors=False,
                        v0=np.exp(-r)*np.sqrt(weight)))
    q = np.sqrt(weight) * v**3
    p = np.dot(weight, v**4)
    operator = LinearOperator((cells, cells), matvec=lambda value: local@value + (8/p)*q*np.dot(q, value), dtype=float)
    fixed = np.sort(eigsh(operator, k=3, which='SA', tol=1e-9, return_eigenvectors=False,
                          v0=np.exp(-r)*np.sqrt(weight))) if angular == 0 else low
    if angular == 0:
        check(low[0] < -1 and low[1] > 0, 'local cubic radial operator has one sampled negative mode')
        check(fixed[0] > .9, 'fixed-flux rank-one term removes the sampled negative radial mode')
    elif angular == 1:
        check(abs(low[0]) < .003, 'translation eigenvalue approaches zero with mesh refinement')
        check(low[1] > 1, 'next sampled angular-one mode is positive')
    else:
        check(low[0] > 1, 'angular-two channel is positive')
    case = {'case': f'hessian-n{n}-ell{angular}-cells{cells}', 'transverse_dimension': n,
            'angular_degree': angular, 'cells': cells, 'radius': radius,
            'local_eigenvalues': low.tolist(), 'fixed_flux_eigenvalues': fixed.tolist(),
            'scope': 'transverse scalar fixed-flux variations only'}
    CASES.append(case)
    return case


def floor_checks():
    for d in DEPTHS:
        n = d - 1
        energy = []
        for width in (2., 4., 8.):
            # u=0 and a Gaussian displacement carry fixed flux in a positive floor.
            area_factor = (2 * math.pi)**(n / 2)
            cost = 1 / (2 * .1 * area_factor * width**n)
            energy.append(cost)
            check(cost > 0, 'ordinary positive floor permits finite flux energy')
        close(energy[1]/energy[0], 2**(-n), 'flux can spread in the ordinary floor')
        close(energy[2]/energy[1], 2**(-n), 'spread flux energy tends to zero')
        CASES.append({'case': f'floor-depth-{d}', 'real_depth': d, 'transverse_dimension': n,
                      'floor': .1, 'widths': [2., 4., 8.], 'trial_energies': energy,
                      'infinite_plane_energy_infimum': 0,
                      'exact_vacuum_zero_required_for_absolute_tube_minimum': True})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    algebraic_checks()
    for n in (1, 2, 3):
        solutions = []
        for radius in (15., 20., 25.):
            solutions.append(solve_profile(n, radius))
        for _, profile in solutions[:-1]:
            for key in ('central_value', 'mass_integral', 'gradient_integral', 'quartic_integral'):
                close(profile[key], solutions[-1][1][key], 'independent radial cutoff convergence')
        solution, profile = solutions[-1]
        energy_checks(n, solution, profile)
        if n > 1:
            for angular in (0, 1, 2):
                results = [spectrum(n, solution, cells, angular) for cells in (400, 800, 1600)]
                close(results[-1]['fixed_flux_eigenvalues'], results[-2]['fixed_flux_eigenvalues'],
                      'scalar Hessian spectral mesh convergence', atol=7e-4, rtol=7e-4)
        print(f'transverse n={n}: profile, complete energy and scalar variations checked', flush=True)
    floor_checks()
    report = {'title': 'Positive Hodge-Gated Neutral Dielectric Flux Tubes',
              'sources': {'experiments/hodge-flux-tube-checks.py': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
              'checks_passed': CHECKS, 'case_count': len(CASES), 'occupied_depths': list(DEPTHS),
              'runtime': {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__, 'sympy': sp.__version__},
              'status': 'conditional neutral-mode same-field action; finite endpoints, vacuum Cauchy problem and full non-Abelian stability unresolved',
              'conclusion': 'One positive quartic dielectric action supports finite-width commuting color tubes only at occupied three/four after the common Hodge gate. Exact support, positive tension, complete transverse stress and boost response, and restricted scalar stability are checked. A positive ordinary dielectric floor removes absolute flux confinement.',
              'cases': CASES}
    args.report.write_text(json.dumps(report, indent=2) + '\n')
    print(f'{CHECKS} checks passed across {len(CASES)} cases; {args.report}', flush=True)


if __name__ == '__main__':
    main()
