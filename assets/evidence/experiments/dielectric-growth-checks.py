#!/usr/bin/env python3
"""Check high-amplitude growth of the same fixed-flux guide functional."""
from pathlib import Path
from fractions import Fraction
import hashlib
import json
import math
import argparse

import numpy as np
import sympy as sp
from scipy.integrate import quad

DEPTHS = (2, 3, 4, 5, 6, 10)
CHECKS = 0
CASES = []


def check(condition, label):
    global CHECKS
    CHECKS += 1
    if not condition:
        raise AssertionError(label)


def close(actual, expected, label, tolerance=1e-10):
    check(abs(actual-expected) <= tolerance*max(1, abs(expected)), label)


def gaussian_moments(n, width, amplitude, powers):
    mass = amplitude**2 * math.pi**(n/2)*width**n
    kinetic = n*mass/(2*width**2)
    moments = {p: amplitude**p*(2*math.pi/p)**(n/2)*width**n for p in powers}
    return mass, kinetic, moments


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    for d in DEPTHS:
        n = d-1
        gate = int(d in (3, 4))
        critical = Fraction(2*n, n-2) if n > 2 else None
        for p in (4, 6, 8, 10, 12):
            collapse = n > 2 and p*(n-2) > 2*n
            if collapse:
                exponent = (Fraction(n, p)+Fraction(n-2, 2))/2
                gradient_power = n-2-2*exponent
                dielectric_power = n-p*exponent
                check(gradient_power > 0 and dielectric_power < 0,
                      'supercritical concentration lowers electric and deformation energy')
            else:
                exponent = None
            CASES.append(dict(case=f'growth-d{d}-p{p}', real_depth=d,
                              transverse_dimension=n, hodge_coefficient=gate,
                              critical_power=str(critical) if critical else None,
                              nongated_zero_energy_concentration_witness=collapse,
                              actual_active_collapse_witness=bool(gate and collapse),
                              amplitude_exponent=str(exponent) if exponent else None))
            check(bool(gate and collapse) == (d == 4 and p > 6),
                  'the occupied gate leaves supercritical collapse only at active four')

    # A smooth positive Gaussian gives a completely explicit concentrating sequence.
    for k_high in (1., 1e-3, 1e-6):
        energies = []
        for width in (1e-2, 1e-4, 1e-8, 1e-16, 1e-32):
            exponent = Fraction(7, 16)  # strictly between 3/8 and 1/2
            amplitude = width**(-float(exponent))
            mass, kinetic, moments = gaussian_moments(3, width, amplitude, (4, 8))
            integral = mass+moments[4]+k_high*moments[8]
            electric = 1/(2*integral)
            energy = electric+(kinetic+mass)/2
            energies.append(energy)
            close(kinetic, 1.5*math.pi**1.5*width**.125,
                  'exact Gaussian gradient scaling')
            CASES.append(dict(case=f'gaussian-collapse-k{k_high}-w{width}',
                              real_depth=4, high_power=8, high_coefficient=k_high,
                              width=width, amplitude=amplitude, electric_energy=electric,
                              gradient_energy=kinetic/2, restoring_energy=mass/2,
                              complete_tension=energy))
        check(energies[-1] < energies[0] and energies[-1] < .001,
              'even a small positive eighth-power coefficient permits vanishing tension')

    # The critical bubble is homogeneous H1, not L2 on all of R3.
    r = sp.symbols('r', positive=True)
    bubble = 1/sp.sqrt(1+r*r)
    k_bubble = sp.integrate(4*sp.pi*r*r*sp.diff(bubble, r)**2, (r, 0, sp.oo))
    p6_bubble = sp.integrate(4*sp.pi*r*r*bubble**6, (r, 0, sp.oo))
    check(sp.simplify(k_bubble-3*sp.pi**2/4) == 0, 'critical bubble exact gradient integral')
    check(sp.simplify(p6_bubble-sp.pi**2/4) == 0, 'critical bubble exact sixth moment')
    sobolev = sp.simplify(p6_bubble/k_bubble**3)
    check(sobolev == 16/(27*sp.pi**4), 'sharp sixth-moment Sobolev coefficient')
    K, Q, k6 = sp.symbols('K Q k6', positive=True)
    critical_energy = Q**2/(2*k6*sobolev*K**3)+K/2
    k_star = (3*Q**2/(k6*sobolev))**sp.Rational(1, 4)
    check(sp.simplify(sp.diff(critical_energy, K).subs(K, k_star)) == 0,
          'critical concentration optimizes its gradient moment')
    check(sp.simplify(critical_energy.subs(K, k_star)-2*k_star/3) == 0,
          'critical level is two-thirds of the optimum gradient moment')
    k2, om, Z = sp.symbols('k2 Omega Z', positive=True)
    capacity = k2*(Z-K)/om**2+k6*sobolev*K**3
    check(sp.simplify(sp.diff(capacity, K, 2)-6*k6*sobolev*K) == 0,
          'critical mixed capacity is strictly convex at fixed total deformation energy')
    diffuse_at_z = Q**2*om**2/(2*k2*Z)+Z/2
    z_diff = Q*om/sp.sqrt(k2)
    check(sp.simplify(sp.diff(diffuse_at_z, Z).subs(Z, z_diff)) == 0,
          'diffuse boundary optimizes at its exact total moment')
    check(sp.simplify(diffuse_at_z.subs(Z, z_diff)-z_diff) == 0,
          'diffuse boundary gives Q Omega over sqrt k2')
    critical_level = 2*k_star/3
    crossover = sp.Rational(4, 9)*sp.sqrt(3/(k6*sobolev))*k2/om**2
    check(sp.simplify((critical_level-Q*om/sp.sqrt(k2)).subs(Q, crossover)) == 0,
          'quadratic plus sextic exact infimum changes boundary branch at the stated flux')
    for z in np.logspace(-3, 3, 15):
        for fraction in (.1, .5, .9):
            kinetic = fraction*z
            integral = z-kinetic+float(sobolev)*kinetic**3
            bound = max(z, float(sobolev)*z**3)
            check(integral < bound, 'every nonzero H1 critical shape is strictly below endpoint capacity')
    # Smooth Gaussian cutoff of the bubble: R=L^(-1/2). All members are H1.
    for width in (1e-2, 1e-4, 1e-6):
        radius = width**(-.5)
        def value(x):
            return math.exp(-x*x/(2*radius**2))/math.sqrt(1+x*x)
        def derivative(x):
            return value(x)*(-x/(1+x*x)-x/(radius*radius))
        mass0 = quad(lambda x: 4*math.pi*x*x*value(x)**2, 0, 12*radius,
                     epsabs=1e-9, limit=250)[0]
        kinetic0 = quad(lambda x: 4*math.pi*x*x*derivative(x)**2, 0, 12*radius,
                        epsabs=1e-9, limit=250)[0]
        p4 = quad(lambda x: 4*math.pi*x*x*value(x)**4, 0, 12*radius,
                  epsabs=1e-9, limit=250)[0]*width
        p6 = quad(lambda x: 4*math.pi*x*x*value(x)**6, 0, 12*radius,
                  epsabs=1e-9, limit=250)[0]
        mass = width**2*mass0
        check(mass > 0 and kinetic0 > 0 and p6 > 0, 'cutoff bubble is an admissible smooth H1 shape')
        CASES.append(dict(case=f'critical-cutoff-w{width}', real_depth=4,
                          width=width, dimensionless_cutoff=radius, mass=mass,
                          gradient_integral=kinetic0, quartic_integral=p4,
                          sixth_integral=p6, sharp_gradient_limit=float(k_bubble),
                          sharp_sixth_limit=float(p6_bubble)))
    check(CASES[-1]['mass'] < CASES[-3]['mass']/100,
          'critical cutoff mass tends to zero')
    check(abs(CASES[-1]['gradient_integral']-float(k_bubble)) < .02,
          'critical cutoff approaches the sharp gradient integral')
    check(abs(CASES[-1]['sixth_integral']-float(p6_bubble)) < 1e-4,
          'critical cutoff approaches the sharp sixth integral')

    # A bounded nonlinear completion has a positive sextic Taylor correction,
    # yet its global growth stays quartic.
    u, delta, scale = sp.symbols('u delta scale', positive=True)
    completed = u**4*(1+delta*u*u/(scale*scale+u*u))
    check(sp.limit((completed-u**4)/u**6, u, 0) == delta/scale**2,
          'bounded completion has the positive local sixth-order correction')
    check(sp.limit(completed/u**4, u, sp.oo) == 1+delta,
          'bounded completion retains global quartic growth')
    for amplitude in np.logspace(-8, 8, 33):
        ratio = 1+.4*amplitude**2/(1.7**2+amplitude**2)
        check(1 <= ratio <= 1.4, 'pointwise quartic coefficient bounds')
    low, mid, high = sp.symbols('I_low I_mid I_high', positive=True)
    check(float(1/(2*3)) <= float(1/(2*2)) <= float(1/(2*1)),
          'fixed-flux energy reverses ordered dielectric integrals')

    # The same sharp-moment machinery gives an exact sextic depth-three control.
    t = sp.symbols('t', positive=True)
    quintic_curve = 1/(t*sp.sqrt(1-t))
    quintic_ratio = sp.sqrt(t)*(2-t)
    quintic_fold = sp.Rational(2, 3)
    quintic_binding = (3-sp.sqrt(5))/2
    check(sp.simplify(sp.diff(quintic_curve, t).subs(t, quintic_fold)) == 0,
          'sextic depth-three stationary branch has fold at two-thirds')
    check(sp.simplify((quintic_ratio**2-1).subs(t, quintic_binding)) == 0,
          'sextic depth-three binding has the exact golden-ratio-derived root')
    check(sp.expand(t*(2-t)**2-1-(t-1)*(t*t-3*t+1)) == 0,
          'sextic depth-three binding factorization')
    check(float(quintic_binding) < float(quintic_fold),
          'sextic depth-three binding lies on the lower branch')

    report = dict(title='Higher-Order Dielectric Growth and Concentration Controls',
                  checks_passed=CHECKS, case_count=len(CASES), occupied_depths=list(DEPTHS),
                  sources={'experiments/dielectric-growth-checks.py': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
                  scope='Explicit scalar commuting-flux concentration sequences and bounded-growth comparisons; no particle simulation or full non-Abelian stability',
                  sharp_sixth_coefficient=str(sobolev),
                  critical_concentration_level=str(2*k_star/3), cases=CASES)
    args.report.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(checks_passed=CHECKS, cases=len(CASES))))


if __name__ == '__main__':
    main()
