"""A local homogeneous scalar-current source on actual opposite Dirac shells."""
from datetime import datetime, timezone
from fractions import Fraction
from hashlib import sha256
from importlib.util import module_from_spec, spec_from_file_location
from math import comb, sqrt
from pathlib import Path
import json

import numpy as np
from scipy.linalg import eigvals
import sympy as sp

ROOT = Path(__file__).resolve().parent
spec = spec_from_file_location('scalar_spatial', ROOT/'scalar-current-band-checks.py')
scalar = module_from_spec(spec)
spec.loader.exec_module(scalar)
CONDITIONS, ERRORS = {}, {}
COEFFICIENT = 256.
SX = np.array([[0, 1], [1, 0]], complex)
SZ = np.diag([1., -1.])


def check(value, label):
    if not bool(value):
        raise AssertionError(label)
    CONDITIONS[label] = CONDITIONS.get(label, 0)+1


def close(actual, expected, label, tolerance=3e-8):
    a, e = np.asarray(actual), np.asarray(expected)
    error = float(np.max(abs(a-e))/max(1., float(np.max(abs(e)))))
    ERRORS[label] = max(ERRORS.get(label, 0.), error)
    check(error < tolerance, label)


def local(field, alpha, beta, coefficient=COEFFICIENT):
    density = float(np.vdot(field, field).real)
    current = np.array([np.vdot(field, g@field).real for g in alpha])
    signed = float(np.vdot(field, beta@field).real)
    proper = sqrt(density*density-np.dot(current, current))
    check(proper > 0, 'actual tested field remains in the timelike-current domain')
    potential = coefficient*signed*signed/(2*proper)
    operator = coefficient*signed/proper*beta
    operator -= coefficient*signed*signed/(2*proper**3)*(density*np.eye(len(beta))-
                                                        sum(j*g for j, g in zip(current, alpha)))
    return potential, operator@field, density, current, signed, proper


def field_variations():
    rng = np.random.default_rng(1732)
    cases = []
    for depth in scalar.sphere.DEPTHS:
        alpha, beta, _ = scalar.clifford(depth)
        rank = len(beta)
        field = rng.normal(size=(rank, rank))+1j*rng.normal(size=(rank, rank))
        field /= np.linalg.norm(field)
        potential, gradient, density, current, signed, proper = local(field, alpha, beta)
        check(0 <= potential <= COEFFICIENT*density/2*(1+1e-12),
              'current-ratio potential is nonnegative and bounded by its local intensity coefficient')
        for _ in range(4):
            variation = rng.normal(size=field.shape)+1j*rng.normal(size=field.shape)
            variation /= np.linalg.norm(variation)
            step = 2e-5
            def finite(h):
                return (local(field+h*variation, alpha, beta)[0]-
                        local(field-h*variation, alpha, beta)[0])/(2*h)
            derivative = (4*finite(step/2)-finite(step))/3
            close(derivative, 2*np.vdot(gradient, variation).real,
                  'full complex variation retains scalar and time-spatial-current derivatives', 2e-7)
        for scale in (.3, 2+3j):
            p, g, *_ = local(scale*field, alpha, beta)
            close(p, abs(scale)**2*potential, 'local ratio potential has exact amplitude degree two')
            close(g, scale*gradient, 'complete local source has exact amplitude degree one')
        direction = rng.normal(size=depth)
        direction /= np.linalg.norm(direction)
        axis = sum(x*g for x, g in zip(direction, alpha))
        rapidity = .71
        boost = np.cosh(rapidity/2)*np.eye(rank)+np.sinh(rapidity/2)*axis
        close(boost.conj().T@beta@boost, beta, 'physical spinor boost preserves the scalar grading form')
        p, _, rho, vector, j, n = local(boost@field, alpha, beta)
        close(rho, np.cosh(rapidity)*density+np.sinh(rapidity)*np.dot(direction, current),
              'summed actual Dirac current obeys the local Lorentz boost law')
        close(j, signed, 'summed scalar current is Lorentz invariant')
        close(n, proper, 'proper summed Dirac-current norm is Lorentz invariant')
        close(p, potential, 'the changed current-ratio potential is a local Lorentz scalar')
        cases.append({'depth': depth, 'graded_spinor_rank': rank,
                      'proper_current_over_intensity': proper/density})
    return cases


def null_current_limit():
    cases = []
    for depth in scalar.sphere.DEPTHS:
        alpha, beta, _ = scalar.clifford(depth)
        values, vectors = np.linalg.eigh(alpha[0])
        u = vectors[:, np.argmax(values)]
        for angle in (-.05, -.01, .01, .05):
            field = np.cos(angle)*u+np.sin(angle)*beta@u
            potential, gradient, rho, vector, signed, proper = local(field, alpha, beta)
            close(signed, np.sin(2*angle), 'null-limit scalar current is the exact paired moment')
            close(vector[0], np.cos(2*angle), 'null-limit spatial current is the exact paired moment')
            close(vector[1:], np.zeros(depth-1), 'null-limit transverse spatial currents vanish')
            close(proper, abs(signed), 'nonnull path reaches the null-current set with proper norm equal to absolute scalar current')
            close(potential, COEFFICIENT*abs(signed)/2,
                  'potential has a continuous but nondifferentiable absolute-value null limit')
            close(gradient, COEFFICIENT*np.sign(signed)/2*beta@field,
                  'complete gradient has unequal one-sided null-current limits')
        cases.append({'depth': depth, 'null_field_norm': float(np.vdot(u, u).real),
                      'one_sided_gradient_jump_norm': COEFFICIENT,
                      'potential_limit': 0, 'differentiable_at_nonzero_null_current': False})
    return cases


def pair_jacobian(h, coefficient, c):
    x = float(np.vdot(c, SX@c).real/np.vdot(c, c).real)
    omega = coefficient/2+h*h/(2*coefficient)
    operator = h*SZ+coefficient*x*SX-coefficient*x*x/2*np.eye(2)
    columns = []
    for variation in (np.array([1, 0]), np.array([0, 1]), np.array([1j, 0]), np.array([0, 1j])):
        dx = 2*np.vdot(variation, SX@c).real-2*x*np.vdot(variation, c).real
        derivative = -1j*((operator-omega*np.eye(2))@variation+
                           coefficient*dx*(SX-x*np.eye(2))@c)
        columns.append(np.concatenate((derivative.real, derivative.imag)))
    jacobian = np.array(columns).T
    def rhs(y):
        z = y[:2]+1j*y[2:]
        sx = np.vdot(z, SX@z).real/np.vdot(z, z).real
        dz = -1j*(h*SZ@z+coefficient*sx*SX@z-coefficient*sx*sx/2*z-omega*z)
        return np.concatenate((dz.real, dz.imag))
    y = np.concatenate((c.real, c.imag))
    step = 1e-5
    numerical = np.column_stack([(rhs(y+step*e)-rhs(y-step*e))/(2*step) for e in np.eye(4)])
    close(jacobian, numerical, 'paired source Jacobian retains phase and norm variations')
    values = eigvals(jacobian)
    target = 2*sqrt(coefficient*coefficient-h*h)
    close(sorted(values.imag), [-target, 0, 0, target], 'complete paired linearization has its exact elliptic shape frequency')
    close(values.real, np.zeros(4), 'retained pair has no resolved exponential growth')
    return target


def spatial_solutions():
    cases = []
    for name, depth, index in scalar.sphere.ASSIGNMENTS:
        level = index-1
        alpha, beta0, _ = scalar.clifford(depth)
        rank = len(beta0)
        h = level+depth/2
        check(COEFFICIENT > h, 'one common current coefficient admits every tested mixed-sign branch')
        c = h/COEFFICIENT
        amplitudes = np.array([sqrt((1+c)/2), sqrt((1-c)/2)])
        x = 2*np.prod(amplitudes)
        omega = COEFFICIENT/2+h*h/(2*COEFFICIENT)
        for theta in (.13, .41, .87, 1.39, 2.07, 2.83):
            g = alpha[0]
            t, s = np.cos(theta), np.sin(theta)
            frame = np.cos(theta/2)*np.eye(rank)-1j*np.sin(theta/2)*g
            beta = frame.conj().T@beta0@frame
            actual_alpha = [frame.conj().T@a@frame for a in alpha]
            aa, bb = scalar.sphere.shells(depth, level, np.array([t]))
            degeneracy = comb(level+depth-1, depth-1)
            positive = (aa[level, 0]*np.eye(rank)-1j*s*bb[level, 0]*g)/sqrt(rank*degeneracy)
            negative = beta@positive
            field = amplitudes[0]*positive+amplitudes[1]*negative
            kinetic = h*(amplitudes[0]*positive-amplitudes[1]*negative)
            potential, gradient, rho, current, signed, proper = local(field, actual_alpha, beta)
            close(current/rho, np.zeros(depth), 'complete spatial multiplet has zero summed spatial current')
            close(signed/rho, x, 'nonconstant spatial density cancels from the scalar-current ratio')
            close(proper/rho, 1, 'balanced spatial current is strictly timelike')
            close((kinetic+gradient-omega*field)/((COEFFICIENT+h)*np.linalg.norm(field)),
                  np.zeros_like(field), 'full nonlinear spatial equation closes on its actual opposite Dirac-shell pair')
        energy = h*np.vdot(amplitudes, SZ@amplitudes).real+COEFFICIENT*x*x/2
        close(energy, omega, 'complete homogeneous stationary energy per canonical norm equals its clock')
        shape = pair_jacobian(h, COEFFICIENT, amplitudes)
        cases.append({'name': name, 'depth': depth, 'mode_index': index, 'spatial_shell_level': level,
                      'bare_shell_frequency_in_inverse_radius_units': h,
                      'common_current_coefficient_in_inverse_radius_units': COEFFICIENT,
                      'stationary_frequency_in_inverse_radius_units': omega,
                      'scalar_current_over_proper_current': float(x),
                      'retained_pair_shape_frequency_in_inverse_radius_units': shape,
                      'full_spatial_stability_tested': False})
    return cases


def exact_and_family_checks(cases):
    h, k = sp.symbols('h k', positive=True)
    c, dc, potential_value = sp.symbols('c dc potential_value', real=True)
    x = sp.sqrt(1-c*c)
    potential_derivative = h*x/c
    energy_derivative = c+h*dc+potential_derivative*(-c*dc/x)
    check(sp.simplify(energy_derivative-c) == 0,
          'every smooth connected ratio-law stationary pair has envelope derivative equal to its signed shell occupation')
    ratio_derivative = (h*c-(h*c+potential_value))/(h*h)
    check(sp.simplify(ratio_derivative+potential_value/(h*h)) == 0,
          'every nonnegative ratio potential compresses bare-shell ratios on a connected positive-clock pair branch')
    omega = k/2+h*h/(2*k)
    check(sp.simplify(sp.diff(omega, h)-h/k) == 0, 'stationary envelope derivative is the positive-minus-negative shell occupation')
    check(sp.simplify(sp.diff(omega/h, h)-(h*h-k*k)/(2*k*h*h)) == 0,
          'common branch compresses positive bare-shell frequency ratios')
    records = []
    for depth in (2, 3, 4, 5, 6):
        family = [a for a in cases if a['depth'] == depth]
        low, high = family[0], family[-1]
        count_ratio = Fraction(comb(high['spatial_shell_level']+depth, depth),
                               comb(low['spatial_shell_level']+depth, depth))
        bare_ratio = Fraction(2*high['spatial_shell_level']+depth, 2*low['spatial_shell_level']+depth)
        actual_ratio = high['stationary_frequency_in_inverse_radius_units']/low['stationary_frequency_in_inverse_radius_units']
        check(float(count_ratio) > float(bare_ratio) > actual_ratio,
              'all maintained multi-particle families exceed the exact common-branch ratio bound')
        records.append({'depth': depth, 'low': low['name'], 'high': high['name'],
                        'target_count_ratio_exact': str(count_ratio), 'bare_shell_ratio_exact': str(bare_ratio),
                        'constructed_branch_ratio': actual_ratio})
    return records


def main():
    variations = field_variations()
    null_limits = null_current_limit()
    cases = spatial_solutions()
    families = exact_and_family_checks(cases)
    dependencies = ['local-current-ratio-checks.py', 'scalar-current-band-checks.py',
                    'spatial-source-saturation-checks.py', 'spatial-band-curvature-checks.py']
    report = {'experiment_key': 'local-homogeneous-current-ratio-spatial-closure',
              'created_utc': datetime.now(timezone.utc).isoformat(),
              'sources': {'experiments/'+p: sha256((ROOT/p).read_bytes()).hexdigest() for p in dependencies},
              'classification': 'changed local covariant ratio action and actual full-multiplet spatial eigenfields',
              'assumptions': ['same-spinor matrix carrier with complete balanced columns is stipulated',
                              'ordinary round-sphere geometry and graded Dirac carrier are declared controls',
                              'local potential kappa J_scalar^2/(2 sqrt(j_vector squared)) on the timelike-current domain',
                              'one common kappa R=256 for every diagnostic depth and assigned comparison index',
                              'mapping assigned cumulative cutoff N to an individual shell l=N is only a test hypothesis'],
              'limits': ['the ratio potential extends continuously by zero at null-current fields but its full source has a nondifferentiable cusp there',
                         'a balanced matrix carrier does not derive observed spin, charge or a master-field reduction',
                         'retained-pair ellipticity is not full spatial or nonlinear particle stability',
                         'compact control eigenfields do not establish localization or complete flat-space particle inertia',
                         'every maintained multi-particle family fails the exact common-branch count-ratio bound',
                         'the generalized nonnegative-potential ratio bound assumes the compared stationary pair states lie on one smooth connected branch'],
              'condition_count': len(CONDITIONS), 'evaluation_count': sum(CONDITIONS.values()),
              'conditions': CONDITIONS, 'maximum_errors': ERRORS,
              'arbitrary_complex_variations': variations, 'null_current_limits': null_limits,
              'spatial_solutions': cases, 'family_comparison': families}
    path = ROOT/('local-current-ratio-results-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')+'.json')
    path.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'report': str(path), 'conditions': len(CONDITIONS), 'evaluations': sum(CONDITIONS.values()),
                      'maximum_errors': ERRORS, 'family_comparison': families}, indent=2), flush=True)


if __name__ == '__main__':
    main()
