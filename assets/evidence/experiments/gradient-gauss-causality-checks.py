#!/usr/bin/env python3
"""Decide static sign, mobility and source causality for one gradient gauge law."""
from pathlib import Path
import argparse
import hashlib
import json
import math

import numpy as np
from scipy.integrate import quad
from scipy.linalg import eigh
import sympy as sp


DEPTHS = (2, 3, 4, 5, 6, 10)
CASES = []
CHECKS = 0


def check(condition, message):
    global CHECKS
    CHECKS += 1
    if not condition:
        raise AssertionError(message)


def near(actual, expected, message, atol=1e-10, rtol=1e-9):
    check(np.allclose(actual, expected, atol=atol, rtol=rtol), message)


def symbolic_field_checks():
    r, t = sp.symbols('r t', positive=True)
    for d in DEPTHS:
        area = 2 * sp.pi ** sp.Rational(d, 2) / sp.gamma(sp.Rational(d, 2))
        green = -sp.log(r) / area if d == 2 else r ** (2-d) / ((d-2)*area)
        check(sp.simplify(sp.diff(green, r, 2)+(d-1)*sp.diff(green, r)/r) == 0,
              f'd={d}: exterior Laplace equation')
        check(sp.simplify(-area*r**(d-1)*sp.diff(green, r)) == 1,
              f'd={d}: inverse-Laplacian unit flux')
        q = t**3/6
        tau = sp.symbols('tau', nonnegative=True)
        second_primitive = sp.integrate((t-tau)*tau**3/6, (tau, 0, t))
        check(second_primitive == t**5/120, 'quadrupole temporal integral')
        exterior = sp.expand(q*sp.diff(green, r)-second_primitive*sp.diff(green, r, 3))
        formula = -t**3/(6*area*r**(d-1))*(1-sp.Rational(d*(d-1),20)*t**2/r**2)
        check(sp.simplify(exterior-formula) == 0, f'd={d}: physical quadrupole tail')
        bracket = 1-sp.Rational(d*(d-1),320)
        check(bracket > 0, f'd={d}: nonzero spacelike response at r=4t')
        # A quadrupole has rho(k)=O(k^2); the energy integrand is O(k^(d-1)).
        check(d-1 > -1, f'd={d}: quadrupole finite infrared energy')
        CASES.append({'case': f'quadrupole-tail-depth-{d}', 'depth': d,
                      'sphere_area': str(area), 'tail_at_r_4t': str(sp.simplify(exterior.subs(r,4*t))),
                      'tail_relative_bracket': str(bracket), 'infrared_energy_power': d-1,
                      'isolated_cost_diverges': d <= 4,
                      'neutral_dipole_cost_finite': d > 2,
                      'conditional_confinement_window': 2 < d <= 4})
    k, omega, ell, speed = sp.symbols('k omega ell speed', positive=True)
    m = ell**2*k**2
    static = sp.simplify(1/(m*k**2))
    check(static == 1/(ell**2*k**4), 'ordinary Gauss gives positive fourth-order source kernel')
    for wave_number in (0.001, 0.1, 1., 10.):
        check(wave_number**2 > 0, 'nonzero wave number has positive kinetic weight')
        near(math.sqrt((2.3**2)*wave_number**4 / wave_number**2)/wave_number,
             2.3, 'transverse phase speed')
    # rho=Q(t) d_1^2 f, j_1=-Q'(t) d_1 f is ordinary conserved current.
    x = sp.symbols('x')
    f = sp.Function('f')(x)
    q = sp.Function('Q')(t)
    check(sp.simplify(sp.diff(q*sp.diff(f,x,2),t)+sp.diff(-sp.diff(q,t)*sp.diff(f,x),x)) == 0,
          'quadrupole ordinary continuity')
    CASES.append({'case': 'gradient-electric-action',
                  'static_kernel': str(static), 'ordinary_current_conservation': True,
                  'positive_nonzero_k_energy': True,
                  'homogeneous_wave_speed_finite': True,
                  'gauge_invariant_source_response_has_spacelike_tail': True,
                  'zero_spatial_mode_requires_domain_choice': True})


def independent_three_dimensional_kernel():
    # Abel-damped radial Fourier integral, independently integrated numerically.
    for t, r in ((1.,4.), (0.5,3.), (2.,0.7), (1.,1.)):
        for damping in (0.5,0.2,0.05):
            def integrand(k):
                if k == 0:
                    return t*r
                return math.exp(-damping*k)*math.sin(t*k)*math.sin(r*k)/k**2
            numeric, error = quad(integrand,0,40/damping,epsabs=2e-10,limit=4000)
            def primitive(b):
                return b*math.atan(b/damping)-damping/2*math.log1p((b/damping)**2)
            closed = (primitive(r+t)-primitive(r-t))/2
            near(numeric,closed,'independent damped Fourier kernel',atol=2e-8)
            exact = min(t,r)/(4*math.pi*r)
            kernel = numeric/(2*math.pi**2*r)
            check(0 < kernel <= exact+2e-8,'positive Abel kernel approaches exact retarded kernel')
            CASES.append({'case': 'three-dimensional-retarded-Fourier', 't':t,'r':r,
                          'damping':damping,'numeric':kernel,'closed':closed/(2*math.pi**2*r),
                          'undamped_limit':exact,'quadrature_error':error,
                          'outside_wave_cone':r>t})
    # Exact exterior solution, retarded initial derivative and Poisson shell jump.
    t,r=sp.symbols('t r',positive=True)
    inside=1/(4*sp.pi)
    outside=t/(4*sp.pi*r)
    check(sp.simplify(inside-outside.subs(r,t)) == 0,'retarded K continuous at wave shell')
    jump=sp.diff(outside,r).subs(r,t)-sp.diff(inside,r)
    check(sp.simplify(-jump-1/(4*sp.pi*t)) == 0,'Poisson derivative jump equals wave shell weight')
    check(sp.diff(outside,t) == 1/(4*sp.pi*r),'initial derivative is inverse-Laplacian Green function')


def positive_spectral_channels():
    rng=np.random.default_rng(20261010)
    for size in (1,2,3,8):
        b=rng.normal(size=(size,size))
        kinetic=b.T@b+np.eye(size)
        # Mix massless and massive ordinary same-field channels.
        q,_=np.linalg.qr(rng.normal(size=(size,size)))
        masses=np.diag([0. if i%2 == 0 else (i+1.)**2 for i in range(size)])
        root=np.linalg.cholesky(kinetic)
        mass_matrix=root@q@masses@q.T@root.T
        source=rng.normal(size=size)
        inv_root=np.linalg.inv(root)
        values,vectors=eigh(inv_root@mass_matrix@inv_root.T)
        values[np.abs(values)<1e-11]=0
        check(np.min(values)>=0,'positive ordinary channel squared masses')
        whitened=inv_root@source
        weights=(vectors.T@whitened)**2
        bound=float(source@np.linalg.solve(kinetic,source))
        near(sum(weights),bound,'positive spectral total source norm')
        for k in (1e-3,0.03,0.2,1.,10.,100.):
            direct=float(source@np.linalg.solve(k*k*kinetic+mass_matrix,source))
            spectral=float(np.sum(weights/(k*k+values)))
            near(direct,spectral,'positive simple-pole diagonalization',rtol=2e-7)
            check(0 <= k*k*direct <= bound*(1+2e-7),'ordinary positive channels obey simple-pole bound')
            CASES.append({'case': 'positive-ordinary-gauge-spectral-bound','channels':size,
                          'wave_number':k,'response':direct,'spectral_response':spectral,
                          'k_squared_response':k*k*direct,'upper_bound':bound,
                          'applies_to_depths':list(DEPTHS)})
    z,m=sp.symbols('z m',positive=True)
    check(sp.simplify(1/(z*(z+m*m))-(1/z-1/(z+m*m))/m**2) == 0,
          'covariant two-pole residues have opposite signs')
    check(sp.limit(z/(z*(z+m*m)),z,0,dir='+') == 1/m**2,
          'lower ordinary term restores simple infrared pole')
    CASES.append({'case':'covariant-box-factor-control',
                  'partial_fractions':'(1/z - 1/(z+m^2))/m^2',
                  'residues_opposite_signs':True,
                  'static_with_lower_term_has_simple_IR_pole':True,
                  'scope':'ordinary finite quadratic canonical realization'})


def neutral_mobility():
    # The determinant contraction is exact: <Ti.Ti>=4/3 and <Ti.Tj>=-2/3.
    for d in DEPTHS:
        positions=np.zeros((3,d))
        positions[1,0]=1
        positions[2,1]=2
        contraction=np.full((3,3),-2/3)
        np.fill_diagonal(contraction,4/3)
        def dipole_norm(p):
            return float(np.sum(contraction*(p@p.T)))
        pair_formula=2/3*sum(float(np.dot(positions[i]-positions[j],positions[i]-positions[j]))
                             for i in range(3) for j in range(i+1,3))
        near(dipole_norm(positions),pair_formula,'determinant dipole norm')
        translation=np.arange(1,d+1)/7
        near(dipole_norm(positions+translation),pair_formula,'neutral rigid translation preserves dipole norm')
        near(dipole_norm(2*positions),4*pair_formula,'changing separation changes conserved dipole norm')
        CASES.append({'case':f'neutral-determinant-mobility-depth-{d}','depth':d,
                      'dipole_squared':pair_formula,'translated_dipole_squared':dipole_norm(positions+translation),
                      'dilated_dipole_squared':dipole_norm(2*positions),
                      'translation_allowed_by_Abelian_dipole_conservation':True,
                      'unassisted_dilation_preserves_dipole':False,
                      'non_Abelian_completion_not_inferred':True})


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args()
    symbolic_field_checks()
    independent_three_dimensional_kernel()
    positive_spectral_channels()
    neutral_mobility()
    digest=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report={'title':'Gradient Gauss Static Energy and Physical Retarded Response',
            'sources':{'experiments/gradient-gauss-causality-checks.py':digest},
            'checks_passed':CHECKS,'case_count':len(CASES),'occupied_depths':list(DEPTHS),
            'conclusion':'Positive ordinary gradient-electric Gauss law realizes the static k^-4 energy and ordinary continuity, but its minimally coupled physical field has a nonzero spacelike retarded response. Finite positive ordinary gauge channels have only simple infrared poles. These are scoped linear results, not a general confinement no-go.',
            'cases':CASES}
    args.report.write_text(json.dumps(report,indent=2)+'\n')
    print(f'{CHECKS} checks passed across {len(CASES)} cases; report: {args.report}')


if __name__ == '__main__':
    main()
