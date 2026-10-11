#!/usr/bin/env python3
"""Check finite-source bounds and complete commuting-channel tube variations."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import math

import numpy as np
import sympy as sp
from scipy.integrate import quad, simpson


CHECKS = 0
CASES = []


def check(condition, message):
    global CHECKS
    CHECKS += 1
    if not condition:
        raise AssertionError(message)


def close(actual, expected, message, atol=2e-9, rtol=2e-9):
    check(np.allclose(actual, expected, atol=atol, rtol=rtol), message)


def exact_checks():
    u, displacement, field, kap, h, p = sp.symbols('u D E kappa h p', positive=True)
    electric = displacement**2 / (2 * kap * u**4)
    quadratic = (sp.diff(electric, u, 2)*h*h/2 + sp.diff(electric, u, displacement)*h*p +
                 sp.diff(electric, displacement, 2)*p*p/2).subs(displacement, kap*u**4*field)
    completed = (p - 4*kap*field*u**3*h)**2 / (2*kap*u**4) - 3*kap*field**2*u*u*h*h
    check(sp.simplify(quadratic - completed) == 0, 'complete electric Hamiltonian quadratic square')
    s = sp.symbols('s', real=True)
    w = 10*s**3 - 15*s**4 + 6*s**5
    for order in (0, 1, 2):
        derivative = sp.diff(w, s, order)
        close(float(derivative.subs(s, 0)), 0, 'cap joins the zero exterior smoothly')
        close(float(derivative.subs(s, 1)), 1 if order == 0 else 0, 'cap joins the tube smoothly')
    alpha2 = sp.integrate(w**2, (s, 0, 1))
    alpha4 = sp.integrate(w**4, (s, 0, 1))
    alpha_gradient = sp.integrate(sp.diff(w, s)**2, (s, 0, 1))
    charge = sp.integrate(sp.diff(w**4, s), (s, 0, 1))
    check(charge == 1, 'each cap carries exactly one signed flux unit')
    check(alpha2 > 0 and alpha4 > 0 and alpha_gradient > 0, 'all cap contributions are positive')
    CASES.append({'case': 'exact-cap-and-hamiltonian-identities',
                  'cap_polynomial': '10s^3-15s^4+6s^5',
                  'alpha2_exact': str(alpha2), 'alpha4_exact': str(alpha4),
                  'alpha_gradient_exact': str(alpha_gradient), 'cap_charge_exact': str(charge),
                  'electric_second_variation': '(p-4 kappa E u^3 h)^2/(2 kappa u^4)-3 kappa E^2 u^2 h^2'})
    return tuple(float(a) for a in (alpha2, alpha4, alpha_gradient))


def profile_cases(n, solution, profile, cap):
    quartic = profile['quartic_integral']
    mass = profile['mass_integral']
    alpha2, alpha4, alpha_gradient = cap
    for q, kap, omega in ((1., 1., 1.), (1.7, .8, 1.3), (.6, 2., .7)):
        tension = .75 * (2*q*q*quartic/kap)**(1/3) * omega**((4 - n)/3)
        amplitude = (2*q*q*omega**(2*n - 2)/(kap*quartic**2))**(1/6)
        scalar_norm = amplitude**2 * omega**(-n) * mass
        cap_width = .9 / omega
        cap_cost = tension*cap_width*(alpha4/3 + 2*alpha2/3) + scalar_norm*alpha_gradient/(2*cap_width)
        # The transverse sharp interpolation bound makes each slice at least tension.
        optimal_s = (2*q*q*quartic*omega**(4 - n)/kap)**(1/3)
        for factor in (.4, .8, 1., 1.6, 3.):
            s = factor*optimal_s
            lower_function = q*q*quartic*omega**(4 - n)/(2*kap*s*s) + s/2
            check(lower_function >= tension*(1 - 2e-10), 'sharp slice bound minimized at the checked tube')
        # Independent Gaussian trial profiles test the complete sharp inequality.
        for width in (.25/omega, .6/omega, 1.2/omega, 3./omega):
            for magnitude in (.4, 1., 2.):
                trial_mass = magnitude**2 * math.pi**(n/2) * width**n
                trial_gradient = n*trial_mass/(2*width*width)
                trial_quartic = magnitude**4 * (math.pi/2)**(n/2) * width**n
                s = trial_gradient + omega*omega*trial_mass
                check(trial_quartic <= omega**(n - 4)*s*s/quartic, 'sharp interpolation bound on independent Gaussian shapes')
                trial_energy = q*q/(2*kap*trial_quartic) + s/2
                check(trial_energy >= tension, 'independent Gaussian energy is above the sharp transverse minimum')
        def w(s):
            return 10*s**3 - 15*s**4 + 6*s**5
        def dw(s):
            return 30*s*s - 60*s**3 + 30*s**4
        # Integrate the cap directly rather than reusing its exact polynomial factors.
        cap_numeric = quad(lambda z: tension*w(z/cap_width)**4/3 +
                           2*tension*w(z/cap_width)**2/3 +
                           scalar_norm*dw(z/cap_width)**2/(2*cap_width**2), 0, cap_width,
                           epsabs=1e-11, epsrel=1e-11)[0]
        close(cap_numeric, cap_cost, 'independently integrated complete finite cap energy')
        lengths = [2./omega, 5./omega, 11./omega]
        energies = [tension*length + 2*cap_numeric for length in lengths]
        for i in range(2):
            close((energies[i + 1] - energies[i])/(lengths[i + 1] - lengths[i]), tension,
                  'fixed-source caps retain the same asymptotic linear separation coefficient')
        check(all(energy >= tension*length for energy, length in zip(energies, lengths)),
              'finite cap trials respect the source-free interval lower bound')
        CASES.append({'case': f'finite-source-separation-n{n}-Q{q}', 'real_depth': n + 1,
                      'classical_commuting_flux': q, 'dielectric_coefficient': kap,
                      'neutral_restoring_frequency': omega, 'tension': tension,
                      'fixed_cap_width': cap_width, 'one_cap_energy': cap_cost,
                      'lengths': lengths, 'upper_bound_energies': energies,
                      'lower_bound_energies': [tension*length for length in lengths],
                      'source_total_charges': [q, -q], 'exact_endpoint_stationary_solution_claimed': False,
                      'scope': 'prescribed rapidly decaying opposite commuting sources; no non-Abelian screening claim'})
    x = np.linspace(0, 20, 20001)
    v, dv = solution.sol(x)
    measure = x**(n - 1)
    def integrate(value):
        return simpson(measure*value, x=x)
    p_radial = integrate(v**4)
    epsilon = v**4
    coupling = 2*math.sqrt(2)*v**3
    for rate in (.6, 1., 2.):
        h = np.exp(-rate*x*x)
        dh = -2*rate*x*h
        b = integrate(v**3*h)
        optimum = coupling*h - epsilon*integrate(coupling*h)/p_radial
        disturbance = np.sin(1.9*x)
        disturbance -= integrate(epsilon*disturbance)/p_radial
        trial = optimum + .4*epsilon*disturbance
        close(integrate(trial), 0, 'complete displacement variation preserves plane flux')
        local = integrate(dh*dh + (1 - 3*v*v)*h*h)/2
        complete = local + integrate((trial - coupling*h)**2/(2*epsilon))
        scalar_minimum = local + 4*b*b/p_radial
        remainder = integrate((trial - optimum)**2/(2*epsilon))
        close(complete, scalar_minimum + remainder, 'full commuting electric variation bounds the scalar fixed-flux Hessian')
        check(complete >= scalar_minimum > 0, 'tested complete commuting-channel second variation is positive')
    CASES.append({'case': f'complete-electric-variation-n{n}', 'real_depth': n + 1,
                  'directions_checked': 3, 'full_Abelian_quadratic_bound': True,
                  'non_Abelian_variations_checked': False})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--profile-code', type=Path, default=Path(__file__).with_name('flux-tube-checks.py'))
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location('flux_profile_checks', args.profile_code)
    profiles = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(profiles)
    cap = exact_checks()
    for depth in profiles.DEPTHS:
        gate = profiles.hodge_coefficient(depth)
        transverse = depth - 1
        check((gate > 0 and transverse < 4) == (depth in (3, 4)),
              'finite-source flux bound uses the same occupied-sector gate and profile law')
        CASES.append({'case': f'occupied-sector-source-control-{depth}', 'real_depth': depth,
                      'hodge_coefficient': gate, 'transverse_dimension': transverse,
                      'nonzero_color_source_allowed_in_this_action': gate > 0,
                      'regular_quartic_transverse_profile_possible': transverse < 4})
    for n in (2, 3):
        solution, profile = profiles.solve_profile(n, 25.)
        profile_cases(n, solution, profile, cap)
    report = {'title': 'Finite Commuting-Source Separation Bounds and Complete Electric Variations',
              'sources': {'experiments/hodge-flux-separation-checks.py': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                          'experiments/hodge-flux-tube-checks.py': hashlib.sha256(args.profile_code.read_bytes()).hexdigest()},
              'checks_passed': CHECKS, 'case_count': len(CASES),
              'occupied_depths': list(profiles.DEPTHS),
              'scope': 'finite-source bounds at active three/four with the same explicit occupied-sector Hodge and profile controls',
              'conclusion': 'A sharp plane energy bound and fixed-shape finite cap trials share the same positive linear separation coefficient. Completing the entire Abelian electric quadratic energy retains the fixed-flux stability bound. Prescribed sources, exact vacuum zero and commuting flux remain premises.',
              'cases': CASES}
    args.report.write_text(json.dumps(report, indent=2) + '\n')
    print(f'{CHECKS} checks passed across {len(CASES)} cases; {args.report}')


if __name__ == '__main__':
    main()
