#!/usr/bin/env python3
"""Construct finite-energy non-Abelian Gauss data for compact classical sources."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import math

import numpy as np
import sympy as sp
from scipy.integrate import quad


DEPTHS = (2, 3, 4, 5, 6, 10)
CHECKS = 0
CASES = []
S = 4
L = 24


def check(condition, message):
    global CHECKS
    CHECKS += 1
    if not condition:
        raise AssertionError(message)


def close(actual, expected, message, atol=3e-9):
    check(np.allclose(actual, expected, atol=atol, rtol=3e-9), message)


def norm2(matrix):
    return float((2*np.trace(matrix.conj().T @ matrix)).real)


def color_matrices():
    source = np.diag([1/3, -1/6, -1/6])
    index = np.arange(3)
    fourier = np.exp(2j*math.pi*np.outer(index, index)/3)/math.sqrt(3)
    rotated = fourier @ source @ fourier.conj().T
    close(np.diag(rotated), 0, 'pure fundamental mean source has a zero-diagonal unitary representative')
    eigenvalues = np.array([-1., 0., 1.])
    first = np.diag(eigenvalues)
    second = np.zeros((3, 3), complex)
    for i in range(3):
        for j in range(3):
            if i != j:
                second[i, j] = 1j*rotated[i, j]/(eigenvalues[i]-eigenvalues[j])
    first = fourier.conj().T @ first @ fourier
    second = fourier.conj().T @ second @ fourier
    for matrix in (first, second, source):
        close(matrix, matrix.conj().T, 'ordinary SU3 connection and displacement matrices are Hermitian')
        close(np.trace(matrix), 0, 'all matrices lie in su3')
    close(-1j*(first @ second-second @ first), source, 'one actual commutator supplies a pure fundamental mean-color source')
    close(norm2(first), 4, 'connection generator metric')
    close(norm2(second), .25, 'displacement generator metric')
    close(norm2(source), 1/3, 'pure fundamental source Casimir expectation norm')
    z = np.exp(2j*math.pi/3)*np.eye(3)
    close(z @ first @ z.conj().T, first, 'connection remains center neutral')
    close(z @ second @ z.conj().T, second, 'displacement remains center neutral')
    check(np.linalg.norm(z-np.eye(3)) > 1, 'fundamental center action remains nontrivial')
    return first, second, source


def radial_moment(d, power, exponent):
    # Integral over the unit d-ball of r^(2*power)*(1-r^2)^exponent.
    result = (sp.pi**sp.Rational(d, 2)*sp.gamma(sp.Rational(d+2*power, 2))*sp.gamma(exponent+1)
              /(sp.gamma(sp.Rational(d, 2))*sp.gamma(sp.Rational(d+2*power, 2)+exponent+1)))
    return sp.simplify(result)


def integral_data(d):
    source = sp.Rational(8*L*L, d)*radial_moment(d, 1, 2*L-2)
    electric = sp.Rational(8*L*L, d)*radial_moment(d, 1, 2*L-2-4*S)
    magnetic = 16*L*L*(radial_moment(d, 0, 2*L-2+4*S)
                          - sp.Rational(4*(L-1), d)*radial_moment(d, 1, 2*L-3+4*S)
                          + sp.Rational(2*(L-1)**2, d)*radial_moment(d, 2, 2*L-4+4*S))
    gradient = 4*S*S*radial_moment(d, 1, 2*S-2)
    restoring = radial_moment(d, 0, 2*S)
    values = [sp.simplify(x) for x in (source, electric, magnetic, gradient, restoring)]
    for value in values:
        check(value > 0, 'every complete compact-core energy integral is positive and finite')
    area = 2*math.pi**(d/2)/math.gamma(d/2)
    functions = [lambda r: 8*L*L/d*r*r*(1-r*r)**(2*L-2),
                 lambda r: 8*L*L/d*r*r*(1-r*r)**(2*L-2-4*S),
                 lambda r: 16*L*L*((1-r*r)**(2*L-2+4*S)
                                   - 4*(L-1)/d*r*r*(1-r*r)**(2*L-3+4*S)
                                   + 2*(L-1)**2/d*r**4*(1-r*r)**(2*L-4+4*S)),
                 lambda r: 4*S*S*r*r*(1-r*r)**(2*S-2),
                 lambda r: (1-r*r)**(2*S)]
    numeric = []
    for exact, function in zip(values, functions):
        value, error = quad(lambda r: area*r**(d-1)*function(r), 0, 1, epsabs=1e-11, epsrel=1e-11)
        close(value, float(exact), 'independent radial quadrature agrees with exact beta integral')
        numeric.append(value)
    return values, numeric


def core_fields(position):
    d = len(position)
    q = 1-position @ position
    if q <= 0:
        return 0., np.zeros(d), np.zeros((d, d)), np.zeros(d)
    rotation = np.zeros((d, d))
    rotation[0, 1] = -1
    rotation[1, 0] = 1
    gradient_f = -2*L*q**(L-1)*position
    hessian = -2*L*q**(L-1)*np.eye(d) + 4*L*(L-1)*q**(L-2)*np.outer(position, position)
    vector = rotation @ gradient_f
    derivative = rotation @ hessian
    gradient_w = -2*S*q**(S-1)*position
    return q**S, vector, derivative, gradient_w


def gauss_core_checks(d, matrices, rng):
    first, second, source = matrices
    exact, numeric = integral_data(d)
    i0, ie, im, k, mass = (float(x) for x in exact)
    charge = 1.1
    coupling = .7
    kappa = 1.3
    omega = .9
    product = charge/(coupling*i0)
    electric_coefficient = norm2(second)*ie/(2*kappa)
    magnetic_coefficient = kappa*norm2(first)*im/2
    ratio = product*math.sqrt(magnetic_coefficient/electric_coefficient)
    connection_minimum = 2*product*math.sqrt(electric_coefficient*magnetic_coefficient)
    close(connection_minimum, charge/coupling*math.sqrt(ie*im)/i0,
          'optimized electric and magnetic trial energy has the exact normalization')
    core_energies = []
    largest_gauss_error = 0.
    for amplitude in (.3, 1., 2.):
        beta = amplitude**2*math.sqrt(ratio)
        alpha = product/beta
        close(alpha*beta, product, 'source is held fixed as constitutive amplitude varies')
        electric_energy = electric_coefficient*beta*beta/amplitude**4
        magnetic_energy = magnetic_coefficient*alpha*alpha*amplitude**4
        scalar_energy = amplitude*amplitude*(k+omega*omega*mass)/2
        close(electric_energy, magnetic_energy, 'balanced positive electric and magnetic energies')
        close(electric_energy+magnetic_energy, connection_minimum, 'full connection energy remains finite')
        core_energies.append(electric_energy+magnetic_energy+scalar_energy)
        for sample in range(12):
            position = rng.normal(size=d)
            position *= rng.uniform(.04, .96)/np.linalg.norm(position)
            w, vector, derivative, gradient_w = core_fields(position)
            close(np.trace(derivative), 0, 'compact displacement vector has exactly zero ordinary divergence')
            a = [alpha*v*first for v in vector]
            displacement = [beta*v*second for v in vector]
            divergence = beta*np.trace(derivative)*second
            commutator = sum((-1j*coupling*(ai @ di-di @ ai) for ai, di in zip(a, displacement)),
                             np.zeros((3, 3), complex))
            rho = charge/i0*(vector @ vector)*source
            error = float(np.max(np.abs(divergence+commutator-rho)))
            largest_gauss_error = max(largest_gauss_error, error)
            close(divergence+commutator, rho, 'full non-Abelian Gauss law supports fixed one-signed compact mean-color density')
            epsilon = kappa*(amplitude*w)**4
            electric = [di/epsilon for di in displacement]
            electric_density = sum(norm2(di) for di in displacement)/(2*epsilon)
            close(electric_density, sum(epsilon*norm2(ei)/2 for ei in electric), 'positive complete electric constitutive Hamiltonian')
            curl_squared = 0.
            for i in range(d):
                for j in range(i+1, d):
                    curvature = alpha*(derivative[j, i]-derivative[i, j])*first -1j*coupling*(a[i] @ a[j]-a[j] @ a[i])
                    expected = alpha*(derivative[j, i]-derivative[i, j])*first
                    close(curvature, expected, 'same-color connection has genuine derivative curvature and no self commutator')
                    curl_squared += (derivative[j, i]-derivative[i, j])**2
            r2 = position @ position
            q = 1-r2
            s2 = position[0]**2+position[1]**2
            closed_curl = 16*L*L*q**(2*L-2) -32*L*L*(L-1)*s2*q**(2*L-3) +16*L*L*(L-1)**2*r2*s2*q**(2*L-4)
            close(curl_squared, closed_curl, 'independent tensor curl agrees with the exact radial-angular polynomial')
            check(electric_density >= 0 and curl_squared >= 0, 'every local electric and magnetic energy is nonnegative')
        # Fields vanish outside the fixed source ball, with no exterior electric or magnetic flux.
        w, vector, derivative, _ = core_fields(np.array([1.2]+[0.]*(d-1)))
        close(w, 0, 'neutral scalar has compact support')
        close(vector, 0, 'displacement and connection have compact support')
        close(derivative, 0, 'curvature also vanishes outside source ball')
    # Two cores with opposite source signs have disjoint support for any separation above two radii.
    for separation in (3., 10., 100.):
        check(separation > 2, 'fixed unit source balls have disjoint supports')
        for _ in range(4):
            offset = rng.normal(size=d)
            offset *= .83/np.linalg.norm(offset)
            shifted = offset - np.array([separation]+[0.]*(d-1))
            w1, v1, derivative1, gradient1 = core_fields(offset)
            w2, v2, derivative2, gradient2 = core_fields(shifted)
            close((w1+w2)**4, w1**4+w2**4, 'combined dielectric has no cross energy on disjoint cores')
            close(np.sum((gradient1+gradient2)**2), np.sum(gradient1**2)+np.sum(gradient2**2),
                  'complete scalar gradient has no intercore cross energy')
            a1 = [product/math.sqrt(ratio)*v*first for v in v1]
            a2 = [-product/math.sqrt(ratio)*v*first for v in v2]
            d1 = [math.sqrt(ratio)*v*second for v in v1]
            d2 = [math.sqrt(ratio)*v*second for v in v2]
            gauss = sum((-1j*coupling*((ai+aj) @ (di+dj)-(di+dj) @ (ai+aj))
                         for ai, aj, di, dj in zip(a1, a2, d1, d2)), np.zeros((3, 3), complex))
            close(gauss, charge/i0*(v1 @ v1-v2 @ v2)*source,
                  'opposite compact cores satisfy full combined Gauss without intercore displacement')
        midpoint = np.array([separation/2]+[0.]*(d-1))
        close(core_fields(midpoint)[1], 0, 'separating plane has no compulsory electric displacement flux')
    CASES.append({'case': f'compact-nonabelian-gauss-depth-{d}', 'real_depth': d,
                  'source_core_radius': 1, 'source_charge_scale': charge, 'gauge_coupling': coupling,
                  'dielectric_coefficient': kappa, 'neutral_restoring_frequency': omega,
                  'scalar_amplitudes': [.3, 1., 2.], 'complete_single_core_trial_energies': core_energies,
                  'exact_integrals': dict(zip(('source', 'electric', 'magnetic', 'gradient', 'restoring'), map(str, exact))),
                  'independent_integrals': numeric, 'connection_energy_after_balance': connection_minimum,
                  'maximum_gauss_error': largest_gauss_error,
                  'pair_energy_upper_bound_for_all_separations_above_two': 2*core_energies[1],
                  'exterior_field_strength': '0', 'ordinary_source_integral': 'Q H != 0 in the chosen source gauge',
                  'static_solution_constructed': False, 'localized_master_field_source_derived': False,
                  'quantum_fundamental_triality_screened': False})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--geometry-source', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location('gauss_geometry', args.geometry_source)
    geometry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(geometry)
    matrices = color_matrices()
    rng = np.random.default_rng(2026101022)
    for d in DEPTHS:
        hodge = geometry.hodge(d if d % 2 == 0 else d+1)
        active = bool(np.linalg.norm(hodge) > 1e-10)
        check(active == (d in (3, 4)), 'same Hodge dielectric tested across all occupied sectors')
        if active:
            gauss_core_checks(d, matrices, rng)
        CASES.append({'case': f'compact-source-gate-depth-{d}', 'real_depth': d,
                      'hodge_coefficient_active': active,
                      'nonzero_finite_energy_displacement_permitted': active,
                      'independent_normal_algebra_dimension': 8 if d <= 4 else 3 if d <= 6 else 0,
                      'exact_inactive_source_constraint_retained': not active and d != 10})
    report = {'title': 'Compact Non-Abelian Gauss Data Limit the Commuting Flux-Confinement Claim',
              'sources': {'experiments/nonabelian-gauss-compact-checks.py': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                          'experiments/hodge-gated-gauss-checks.py': hashlib.sha256(args.geometry_source.read_bytes()).hexdigest()},
              'checks_passed': CHECKS, 'dependency_checks_passed': geometry.CHECKS,
              'case_count': len(CASES), 'occupied_depths': list(DEPTHS),
              'scope': 'admissible full classical Gauss phase-space data for fixed extended sources; not a static solution, quantum center screening or particle construction',
              'conclusion': 'The full proposed classical connection Hamiltonian permits finite compact Gauss data for a one-signed mean-color source. Two opposite fixed cores give a separation-independent energy upper bound. Thus the commuting-channel linear lower bound cannot be transferred to this full classical source problem. Inactive Hodge source constraints remain unchanged.',
              'cases': CASES}
    args.report.write_text(json.dumps(report, indent=2)+'\n')
    print(f'{CHECKS} compact Gauss and energy checks plus {geometry.CHECKS} geometry checks passed across {len(CASES)} cases')


if __name__ == '__main__':
    main()
