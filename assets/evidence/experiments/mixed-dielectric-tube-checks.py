#!/usr/bin/env python3
"""Test flux-guide robustness under a quadratic plus quartic dielectric."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import math

import numpy as np
import sympy as sp
from scipy.optimize import brentq, minimize

CHECKS = 0
CASES = []


def check(condition, label):
    global CHECKS
    CHECKS += 1
    if not condition:
        raise AssertionError(label)


def close(a, b, label, atol=2e-7, rtol=2e-7):
    check(np.allclose(a, b, atol=atol, rtol=rtol), label)


def qshape(n, t):
    return (1-t)**(1-n/2)*t**(-1.5)*(2+(2-n)*t)


def shape_energy(n, t):
    return math.sqrt(t)*(3+(1-n)*t)/(2+(2-n)*t)


def reduced_energy(log_moments, n, constant, q, k2, k4, omega):
    kinetic, mass = np.exp(log_moments)
    quartic = constant*kinetic**(n/2)*mass**((4-n)/2)
    return q*q/(2*(k2*mass+k4*quartic))+(kinetic+omega*omega*mass)/2


def moment_derivatives(kinetic, mass, quartic, n, q, k2, k4, omega):
    integral = k2*mass+k4*quartic
    derivative = np.array([k4*n*quartic/2, k2*mass+k4*(4-n)*quartic/2])
    second = np.array([[k4*n*n*quartic/4, k4*n*(4-n)*quartic/4],
                       [k4*n*(4-n)*quartic/4, k2*mass+k4*(4-n)**2*quartic/4]])
    gradient = -q*q*derivative/(2*integral**2)+np.array([kinetic, omega*omega*mass])/2
    hessian = q*q*np.outer(derivative, derivative)/integral**3-q*q*second/(2*integral**2)
    hessian += np.diag([kinetic, omega*omega*mass])/2
    return gradient, hessian


def algebra_checks():
    t = sp.symbols('t', positive=True)
    t_fold = (9-sp.sqrt(33))/4
    t_bind = (9-sp.sqrt(17))/8
    q3 = (2-t)/(sp.sqrt(1-t)*t**sp.Rational(3, 2))
    derivative = sp.diff(q3, t)
    check(sp.simplify(derivative.subs(t, t_fold)) == 0, 'exact three-transverse branch fold')
    difference = sp.expand(t*(3-2*t)**2-(2-t)**2)
    check(sp.expand(difference-(t-1)*(4*t*t-9*t+4)) == 0, 'binding threshold factorization')
    check(sp.simplify(difference.subs(t, t_bind)) == 0, 'exact three-transverse global binding threshold')
    n = sp.symbols('n', integer=True)
    k, mass, quartic, e2, k2, k4, om2 = sp.symbols('K M P E2 k2 k4 Omega2', positive=True)
    # Combine the ordinary equation with its independent spatial scaling.
    equation = k+(om2-k2*e2)*mass-2*k4*e2*quartic
    scaling = (n-2)*k/2+n*(om2-k2*e2)*mass/2-n*k4*e2*quartic/2
    check(sp.expand(4*scaling-n*equation-((n-4)*k+n*(om2-k2*e2)*mass)) == 0,
          'mixed dielectric retains the subcritical cubic support identity')
    return float(t_fold), float(t_bind), str(t_fold), str(t_bind)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tube-source', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location('tube_geometry', args.tube_source)
    tube = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tube)
    t_fold, t_bind, fold_exact, bind_exact = algebra_checks()
    rng = np.random.default_rng(2026101060)
    for d in tube.DEPTHS:
        gate = tube.hodge_coefficient(d)
        check(bool(gate) == (d in (3, 4)), 'actual mixed dielectric has the same occupied-sector zeros')
        CASES.append({'case': f'mixed-dielectric-occupied-depth-{d}', 'real_depth': d,
                      'transverse_dimension': d-1, 'hodge_coefficient': gate,
                      'quadratic_dielectric_is_an_ordinary_positive_vacuum_floor': False,
                      'independent_connection_zero_source_constraint_retained': gate == 0 and d != 10,
                      'independent_connection_present': d != 10})
    for n in (1, 2, 3):
        _, profile = tube.solve_profile(n, 25.)
        mn, kn, pn = (profile[k] for k in ('mass_integral', 'gradient_integral', 'quartic_integral'))
        constant = pn/(kn**(n/2)*mn**((4-n)/2))
        close(constant, 4*(4-n)**(n/2-1)/(n**(n/2)*mn), 'sharp interpolation normalization')
        for k2, k4, omega in ((1., 1., 1.), (1.2, .7, .9), (.4, 2.3, 1.6)):
            q0 = k2**1.5*omega**(1-n)*mn/(2*k4*(4-n))
            for t in (.2, .5, .8, .95):
                electric = omega*math.sqrt(t/k2)
                inverse_width = omega*math.sqrt(1-t)
                amplitude = math.sqrt(k2*(1-t)/(2*k4*t))
                mass = amplitude**2*inverse_width**(-n)*mn
                kinetic = amplitude**2*inverse_width**(2-n)*kn
                quartic = amplitude**4*inverse_width**(-n)*pn
                q = electric*(k2*mass+k4*quartic)
                close(q, q0*qshape(n, t), 'flux from direct profile moments agrees with the universal branch curve')
                electric_energy = q*electric/2
                tension = electric_energy+(kinetic+omega*omega*mass)/2
                diffuse = q*omega/math.sqrt(k2)
                close(tension/diffuse, shape_energy(n, t), 'complete tension against diffuse-flux threshold')
                close(kinetic/n-(kinetic+omega*omega*mass)/2+electric_energy, 0,
                      'mixed tube retains vanishing complete transverse stress')
                close(2*electric_energy+kinetic/n, tension, 'complete transverse boost inertia stays positive')
                gradient, hessian = moment_derivatives(kinetic, mass, quartic, n, q, k2, k4, omega)
                close(gradient, 0, 'profile is stationary under independent amplitude and width variations')
                eigenvalues = np.linalg.eigvalsh(hessian)
                stable = n < 3 or t < t_fold
                check((eigenvalues.min() > 0) == stable, 'exact moment Hessian distinguishes the folded unstable branch')
                step = 2e-4
                derivative_q = (qshape(n, t+step)-qshape(n, t-step))/(2*step)
                def tension_from_t(value):
                    qq = q0*qshape(n, value)
                    return qq*omega/math.sqrt(k2)*shape_energy(n, value)
                derivative_tension = (tension_from_t(t+step)-tension_from_t(t-step))/(2*step)
                close(derivative_tension/(q0*derivative_q), electric,
                      'envelope derivative of tension is the actual electric field', atol=2e-5, rtol=2e-5)
                log_pair = np.log([kinetic, mass])
                for _ in range(2):
                    direction = rng.normal(size=2); direction /= np.linalg.norm(direction)
                    finite = (reduced_energy(log_pair+step*direction, n, constant, q, k2, k4, omega)
                              -2*tension+reduced_energy(log_pair-step*direction, n, constant, q, k2, k4, omega))/step**2
                    close(finite, direction@hessian@direction, 'direct energy differences verify the fixed-flux moment Hessian', atol=2e-5, rtol=2e-5)
                CASES.append({'case': f'mixed-profile-n{n}-k2{k2}-t{t}', 'transverse_dimension': n,
                              'real_depth': n+1, 'actual_hodge_active': n in (2, 3),
                              'quadratic_coefficient': k2, 'quartic_coefficient': k4,
                              'neutral_restoring_frequency': omega, 't': t,
                              'commuting_flux_Q': q, 'width': 1/inverse_width, 'amplitude': amplitude,
                              'complete_tension': tension, 'diffuse_threshold': diffuse,
                              'tension_to_diffuse_ratio': tension/diffuse,
                              'bound_below_diffuse_threshold': tension < diffuse,
                              'moment_hessian_eigenvalues': eigenvalues.tolist(),
                              'stable_moment_branch': stable})
        if n in (2, 3):
            q0 = mn/(2*(4-n))
            if n == 2:
                critical = 2*q0
                trials = (.7*critical, 1.2*critical, 3*critical)
                cuts = (0., 1.)
            else:
                critical = q0*qshape(n, t_bind)
                fold = q0*qshape(n, t_fold)
                trials = (.9*fold, (fold+critical)/2, 1.3*critical)
                cuts = (0., t_fold, 1.)
            for q in trials:
                roots = []
                for left, right in zip(cuts[:-1], cuts[1:]):
                    lo, hi = max(left, 1e-9), min(right, 1-1e-9)
                    fun = lambda t: q0*qshape(n, t)-q
                    if fun(lo)*fun(hi) < 0:
                        roots.append(brentq(fun, lo, hi, xtol=1e-13))
                candidate = min([q]+[q*shape_energy(n, t) for t in roots])
                finite_minimum = q > critical
                check((candidate < q*(1-1e-9)) == finite_minimum, 'exact branches cross the global diffuse threshold at the predicted flux')
                observed = []
                for start in ((-5., 0.), (0., 0.), (3., 2.)):
                    result = minimize(reduced_energy, np.array(start), args=(n, constant, q, 1., 1., 1.),
                                      method='L-BFGS-B', bounds=((-30., 20.), (-30., 20.)),
                                      options={'ftol': 1e-13, 'gtol': 1e-8, 'maxiter': 500})
                    observed.append(result.fun)
                close(min(observed), candidate, 'independent two-moment minimization agrees with analytic global branch selection', atol=2e-5)
                CASES.append({'case': f'global-selection-n{n}-Q{q:.8g}', 'real_depth': n+1,
                              'commuting_flux_Q': q, 'critical_global_binding_flux': critical,
                              'stationary_t_values': roots, 'global_energy_infimum': candidate,
                              'finite_profile_global_minimum': finite_minimum,
                              'independent_optimizer_energies': observed})
        if n == 3:
            close(shape_energy(n, t_bind), 1., 'the finite three-transverse binding profile exactly ties the diffuse threshold')
            CASES.append({'case': 'exact-three-transverse-thresholds', 'real_depth': 4,
                          'fold_t_exact': fold_exact, 'binding_t_exact': bind_exact,
                          'fold_t': t_fold, 'binding_t': t_bind,
                          'dimensionless_fold_flux': qshape(n, t_fold),
                          'dimensionless_binding_flux': qshape(n, t_bind),
                          'binding_width_times_Omega': 1/math.sqrt(1-t_bind)})
    for _ in range(8):
        first, other = rng.normal(size=(2, 4))
        u2 = first@first+other@other
        area2 = (first@first)*(other@other)-(first@other)**2
        close(u2*u2/4-area2, ((first@first-other@other)**2)/4+(first@other)**2,
              'neutral-frame area deficit survives the added quadratic invariant')
        for k2, k4 in ((1., 1.), (.4, 2.3)):
            check(k2*u2+k4*area2 <= k2*u2+k4*u2*u2/4+1e-9,
                  'mixed frame dielectric is bounded by the scalar law with quartic coefficient divided by four')
    CASES.append({'case': 'mixed-neutral-frame-completion',
                  'active_depths': [3, 4], 'effective_quartic_coefficient': 'kappa4/4',
                  'effective_quadratic_coefficient': 'kappa2',
                  'balanced_orthogonal_frame_attains_scalar_energy_bound': True})
    report = {'title': 'Quadratic Dielectric Corrections Preserve Conditional Three/Four Flux Guides Above Derived Thresholds',
              'sources': {'experiments/mixed-dielectric-tube-checks.py': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                          'experiments/hodge-flux-tube-checks.py': hashlib.sha256(args.tube_source.read_bytes()).hexdigest()},
              'checks_passed': CHECKS, 'dependency_checks_passed': tube.CHECKS, 'case_count': len(CASES),
              'occupied_depths': list(tube.DEPTHS),
              'scope': 'complete fixed commuting flux energy, sharp interpolation reduction, global threshold and local branch stability; not full non-Abelian confinement',
              'cases': CASES}
    args.report.write_text(json.dumps(report, indent=2)+'\n')
    print(f'{CHECKS} mixed-dielectric checks plus {tube.CHECKS} dependency checks passed across {len(CASES)} cases')


if __name__ == '__main__':
    main()
