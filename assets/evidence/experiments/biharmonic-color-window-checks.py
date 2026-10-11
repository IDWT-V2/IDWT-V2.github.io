#!/usr/bin/env python3
"""Common positive static response and its occupied-depth infrared window."""
import argparse
from datetime import datetime, timezone
from hashlib import sha256
from itertools import permutations
import json
import math
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.special import gamma, jv, exp1, erf
import sympy as sp

CHECKS = 0
CASES = []


def check(condition,message):
    global CHECKS
    if not condition:
        raise AssertionError(message)
    CHECKS += 1


def close(actual,expected,message,tolerance=2e-9):
    check(np.allclose(actual,expected,atol=tolerance*1e-3,rtol=tolerance),message)


def angular_loss(dimension,z):
    if abs(z)<.02:
        return z*z/(2*dimension)-z**4/(8*dimension*(dimension+2))+z**6/(48*dimension*(dimension+2)*(dimension+4))
    return 1-gamma(dimension/2)*(2/z)**(dimension/2-1)*jv(dimension/2-1,z)


def fourier_energy(dimension,radius,width,lower=0.):
    sphere = 2*math.pi**(dimension/2)/gamma(dimension/2)
    def integrand(k):
        if k == 0:
            return radius*radius/(2*dimension) if dimension == 3 else 0.
        return k**(dimension-5)*math.exp(-width*width*k*k)*angular_loss(dimension,k*radius)
    result,error = quad(integrand,lower,np.inf,epsabs=1e-10,epsrel=2e-10,limit=300)
    return sphere*result/(2*math.pi)**dimension


def schwinger_energy(dimension,radius,width):
    ratio = radius*radius/(4*width*width)
    def integrand(z):
        if z == 0:
            return ratio if dimension == 3 else 0.
        return z**(dimension-5)*(1-z*z)*(-math.expm1(-ratio*z*z))
    result,error = quad(integrand,0,1,epsabs=1e-11,epsrel=2e-11)
    return 2*width**(4-dimension)*result/(4*math.pi)**(dimension/2)


def closed_energy(dimension,radius,width):
    if dimension == 3:
        average = (radius+2*width*width/radius)*erf(radius/(2*width))+2*width/math.sqrt(math.pi)*math.exp(-radius*radius/(4*width*width))
        return (average-4*width/math.sqrt(math.pi))/(8*math.pi)
    if dimension == 4:
        x = radius*radius/(4*width*width)
        return (np.euler_gamma+math.log(x)+exp1(x)-1-math.expm1(-x)/x)/(16*math.pi**2)
    raise ValueError(dimension)


def ambient_energy(dimension,radius,width,normal_width,isolated=False):
    """Ambient ten-dimensional response, multiplied by normal effective volume."""
    codimension=10-dimension
    ratio=radius*radius/(4*width*width)
    def integrand(z):
        if z==0:
            return 0.
        normal=(normal_width*normal_width*z*z/
                (width*width+(normal_width*normal_width-width*width)*z*z))**(codimension/2)
        loss=1. if isolated else -math.expm1(-ratio*z*z)
        return z**(dimension-5)*(1-z*z)*normal*loss
    transition=min(.5,width/normal_width)
    result,error=quad(integrand,0,1,points=[transition],epsabs=1e-11,epsrel=2e-10,limit=250)
    return 2*width**(4-dimension)*result/(4*math.pi)**(dimension/2)


def quadrupole_energy(dimension,radius,width,schwinger=False):
    if schwinger:
        ratio=radius*radius/(4*width*width)
        def integrand(z):
            if z==0:
                return 0.
            x=ratio*z*z
            loss=(6*x*x-10*x**3+10.5*x**4 if x<.001 else
                  -4*math.expm1(-x)+math.expm1(-4*x))
            return z**(dimension-5)*(1-z*z)*loss
        value,error=quad(integrand,0,1,epsabs=1e-11,epsrel=2e-10,limit=250)
        return 2*width**(4-dimension)*value/(4*math.pi)**(dimension/2)
    sphere=2*math.pi**(dimension/2)/gamma(dimension/2)
    def integrand(k):
        if k==0:
            return 0.
        z=k*radius
        loss=(3*z**4/(2*dimension*(dimension+2))-5*z**6/(4*dimension*(dimension+2)*(dimension+4))
              if abs(z)<.02 else 4*angular_loss(dimension,z)-angular_loss(dimension,2*z))
        return k**(dimension-5)*math.exp(-width*width*k*k)*loss
    value,error=quad(integrand,0,np.inf,epsabs=1e-10,epsrel=2e-10,limit=300)
    return sphere*value/(2*math.pi)**dimension


def color_generators():
    result = []
    for a,b in ((0,1),(0,2),(1,2)):
        symmetric = sp.zeros(3)
        antisymmetric = sp.zeros(3)
        symmetric[a,b] = symmetric[b,a] = sp.Rational(1,2)
        antisymmetric[a,b],antisymmetric[b,a] = -sp.I/2,sp.I/2
        result.extend((symmetric,antisymmetric))
    result.extend((sp.diag(1,-1,0)/2,sp.diag(1,1,-2)/(2*sp.sqrt(3))))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    radius = sp.symbols('r',positive=True)
    for dimension in (2,3,4,5,6,10):
        if dimension == 2:
            green = radius**2*sp.log(radius)/(8*sp.pi)
        elif dimension == 4:
            green = -sp.log(radius)/(8*sp.pi**2)
        else:
            green = sp.gamma(sp.Rational(dimension,2)-2)*radius**(4-dimension)/(16*sp.pi**sp.Rational(dimension,2))
        def laplacian(value):
            return sp.diff(value,radius,2)+(dimension-1)*sp.diff(value,radius)/radius
        first = sp.simplify(laplacian(green))
        second = sp.simplify(laplacian(first))
        sphere = 2*sp.pi**sp.Rational(dimension,2)/sp.gamma(sp.Rational(dimension,2))
        flux = sp.simplify(sphere*radius**(dimension-1)*sp.diff(first,radius))
        check(second == 0,'the biharmonic spatial kernel solves the equation away from the source')
        check(flux == 1,'the biharmonic spatial kernel has exact positive unit distributional source')
        monopole_exponent = dimension-5
        dipole_exponent = dimension-3
        isolated_diverges = monopole_exponent <= -1
        neutral_finite = dipole_exponent > -1
        check((isolated_diverges and neutral_finite) == (dimension in (3,4)),
              'the common fourth-order response selects exactly occupied depths three and four for this infrared window')
        CASES.append({'real_depth':dimension,'geometry':'common positive inverse-biharmonic static response',
                      'parameters':{'inverse_response':'1/|k|^4','source_core_width':'finite and unspecified'},
                      'measurements':{'green_function_exact':str(green),'source_flux_exact':str(flux),
                                      'isolated_charge_radial_power':monopole_exponent,
                                      'neutral_dipole_radial_power':dipole_exponent,
                                      'isolated_charge_energy_infrared_divergent':isolated_diverges,
                                      'generic_neutral_pair_energy_infrared_finite':neutral_finite,
                                      'isolated_cost_and_finite_pair_window':isolated_diverges and neutral_finite,
                                      'all_higher_forces_excluded':False}})
    for dimension in (3,4,5,6,10):
        for width in (.4,1.,1.7):
            for separation in (.2,1.,4.,20.):
                direct = fourier_energy(dimension,separation,width)
                transformed = schwinger_energy(dimension,separation,width)
                close(direct,transformed,'independent Fourier/Bessel and Schwinger Gaussian pair energies agree')
                check(direct>0,'the complete neutral-source field energy remains positive')
                if dimension in (3,4):
                    close(direct,closed_energy(dimension,separation,width),
                          'independent closed Gaussian pair formula agrees with both integral representations')
                else:
                    ceiling = width**(4-dimension)/((4*math.pi)**(dimension/2)*(dimension/2-1)*(dimension/2-2))
                    check(direct<ceiling,'higher-depth neutral pair energy saturates below its finite isolated-source ceiling')
                CASES.append({'real_depth':dimension,'geometry':'Gaussian-regularized neutral pair static energy',
                              'parameters':{'separation':separation,'source_width':width,'common_source_strength_squared':1},
                              'measurements':{'fourier_energy':direct,'schwinger_energy':transformed,
                                              'particle_shape_or_amplitude_fitted':False}})
    # The depth-two dipole divergence is independently visible with a physical infrared cutoff.
    separation,width = 1.3,.7
    coefficient = separation**2/(8*math.pi)
    small = fourier_energy(2,separation,width,1e-5)
    large = fourier_energy(2,separation,width,1e-4)
    close(small-large,coefficient*math.log(10),'depth-two neutral dipoles retain the predicted logarithmic infrared divergence',1e-7)
    CASES.append({'real_depth':2,'geometry':'neutral pair infrared cutoff control',
                  'parameters':{'separation':separation,'source_width':width,'cutoffs':[1e-5,1e-4]},
                  'measurements':{'energy_increment':small-large,'predicted_logarithmic_increment':coefficient*math.log(10)}})
    # Finite-width normal localization restores the ambient infrared law.
    for dimension in (2,3,4,5,6,10):
        width,separation=.7,2.
        codimension=10-dimension
        normal_volume=(4*math.pi)**(codimension/2)*width**codimension
        isotropic=ambient_energy(dimension,separation,width,width)/normal_volume
        close(isotropic,schwinger_energy(10,separation,width),
              'finite normal localization gives the ambient ten-dimensional isotropic response')
        previous=0.
        for normal_width in (.3,1.,3.,10.,100.,10000.):
            scaled=ambient_energy(dimension,separation,width,normal_width)
            ceiling=ambient_energy(dimension,separation,width,normal_width,isolated=True)
            check(0<scaled<ceiling,'finite ambient normal width gives finite isolated cost and pair ceiling')
            if codimension:
                check(scaled>previous,'per-normal-volume cylindrical limit is monotone')
            previous=scaled
            if dimension>2:
                lower=schwinger_energy(dimension,separation,width)
                check(scaled<=lower*(1+1e-10),'ambient scaled neutral energy is bounded by the sector-only response')
                if normal_width==10000.:
                    check(scaled/lower>.998,'broad normal continuation approaches sector response per normal volume')
            CASES.append({'real_depth':dimension,'geometry':'finite-width normal support in ambient ten-dimensional response',
                          'parameters':{'separation':separation,'tangent_width':width,'normal_width':normal_width,
                                        'ambient_depth':10,'energy_normalization':'multiplied by effective normal volume'},
                          'measurements':{'scaled_pair_energy':scaled,'finite_scaled_isolated_ceiling':ceiling,
                                          'isolated_infrared_divergence':False,'sector_only_limit_requires_infinite_normal_continuation':bool(codimension)}})
    # Vertices change the pole before any dimensional inference.
    for dimension in (2,3,4,5,6,10):
        for derivatives in (0,1,2):
            pole=4-2*derivatives
            isolated_diverges=dimension<=pole
            neutral_finite=dimension+2>pole
            selected=isolated_diverges and neutral_finite
            check(selected==(dimension in (3,4) if derivatives==0 else dimension==2 if derivatives==1 else False),
                  'derivative vertices change the occupied-sector infrared window')
            CASES.append({'real_depth':dimension,'geometry':'common fourth-order stiffness with derivative source vertices',
                          'parameters':{'derivatives_per_vertex':derivatives,'effective_pole_order':pole},
                          'measurements':{'isolated_cost_and_finite_pair_window':selected}})
    # Neutral quadrupoles show that the depth-two dipole obstruction does
    # not exclude every conceivable neutral source.
    for dimension in (2,3,4,5,6,10):
        direct=quadrupole_energy(dimension,1.3,.7)
        transformed=quadrupole_energy(dimension,1.3,.7,True)
        close(direct,transformed,'independent Gaussian quadrupole energy representations agree',2e-8)
        check(direct>0,'neutral quadrupole energy is finite and positive at every occupied depth')
        CASES.append({'real_depth':dimension,'geometry':'neutral zero-dipole three-site source',
                      'parameters':{'source_weights':[1,-2,1],'separation':1.3,'width':.7},
                      'measurements':{'positive_finite_field_energy':direct,
                                      'generic_dipole_exclusion_does_not_exclude_this_source':dimension==2}})
    # A lower second-order stiffness cuts off the three-dimensional linear
    # law even if the fourth-order term dominates at intermediate scales.
    r,m=sp.symbols('r m',positive=True)
    screened=(1-sp.exp(-m*r))/(4*sp.pi*m*m*r)
    positive_pair=1/(4*sp.pi*m)-screened
    check(sp.simplify(sp.limit(positive_pair/r,r,0)-1/(8*sp.pi))==0,
          'second-order stiffness retains a short-distance linear start')
    check(sp.limit(positive_pair,r,sp.oo)==1/(4*sp.pi*m),
          'second-order stiffness makes the neutral pair energy saturate')
    CASES.append({'real_depth':3,'geometry':'allowed second-order stiffness infrared control',
                  'parameters':{'inverse_response':'1/[k^2(k^2+m^2)]','m':'positive'},
                  'measurements':{'neutral_pair_energy_exact':str(positive_pair),
                                  'short_distance_linear_slope_exact':'1/(8*pi)',
                                  'infinite_distance_energy_ceiling_exact':'1/(4*pi*m)'}})
    # Exact color singlet source contractions determine the neutral infrared order.
    generators = color_generators()
    casimir = sum((t*t for t in generators),sp.zeros(3))
    check(casimir == sp.Rational(4,3)*sp.eye(3),'the ordinary normal triplet has the exact standard quadratic Casimir')
    meson = sp.zeros(9,1)
    for a in range(3):
        meson[3*a+a] = 1/sp.sqrt(3)
    baryon = sp.zeros(27,1)
    for indices in permutations(range(3)):
        sign = (-1)**sum(indices[i]>indices[j] for i in range(3) for j in range(i+1,3))
        baryon[9*indices[0]+3*indices[1]+indices[2]] = sign/sp.sqrt(6)
    for t in generators:
        anti = -t.T
        total2 = sp.kronecker_product(t,sp.eye(3))+sp.kronecker_product(sp.eye(3),anti)
        check(total2*meson == sp.zeros(9,1),'conjugate normal triplets have an exactly neutral invariant source')
        total3 = sum((sp.kronecker_product(t if slot == 0 else sp.eye(3),
                                         t if slot == 1 else sp.eye(3),
                                         t if slot == 2 else sp.eye(3)) for slot in range(3)),sp.zeros(27))
        check(total3*baryon == sp.zeros(27,1),'three normal triplets have an exactly neutral determinant source')
    cmeson = sum((sp.kronecker_product(t,-t.T) for t in generators),sp.zeros(9))
    check(cmeson*meson == -sp.Rational(4,3)*meson,'the meson pair contraction fixes its positive neutral energy coefficient')
    for a,b in ((0,1),(0,2),(1,2)):
        operator = sum((sp.kronecker_product(t if 0 in (a,b) else sp.eye(3),
                                            t if 1 in (a,b) else sp.eye(3),
                                            t if 2 in (a,b) else sp.eye(3)) for t in generators),sp.zeros(27))
        check(operator*baryon == -sp.Rational(2,3)*baryon,'each baryon pair has the same invariant color contraction')
    x,y,z = sp.symbols('x y z',real=True)
    dipole = sp.Rational(4,3)*(x*x+y*y+z*z)-sp.Rational(4,3)*(x*y+x*z+y*z)
    check(sp.expand(dipole-sp.Rational(2,3)*((x-y)**2+(x-z)**2+(y-z)**2)) == 0,
          'the determinant source has a nonzero dipole norm whenever its three positions are separated')
    CASES.append({'real_depth':3,'geometry':'exact normal color meson and determinant source carriers',
                  'parameters':{'triplet_casimir_exact':'4/3'},
                  'measurements':{'meson_neutral_energy_coefficient_exact':'4/3',
                                  'baryon_sum_of_neutral_pair_energies_coefficient_exact':'2/3',
                                  'baryon_dipole_norm_exact':'(2/3) sum_pairs |r_a-r_b|^2'}})
    # A subtracted four-dimensional logarithmic kernel has the three-dimensional linear marginal.
    u = sp.symbols('u',real=True)
    derivative_integral = sp.integrate(-radius/(8*sp.pi**2*(radius**2+u**2)),(u,-sp.oo,sp.oo))
    check(sp.simplify(derivative_integral+1/(8*sp.pi)) == 0,
          'the renormalized logarithmic four-kernel has the exact linear three-kernel marginal')
    report = {'experiment':'biharmonic-color-window','created_utc':datetime.now(timezone.utc).isoformat(),
              'status':'passed','checks':CHECKS,'cases':CASES,
              'scope':'Conditional common static 1/k^4 response. Exact infrared window and normal-color contractions do not derive its same-field kinetic action, non-Abelian completion, mobility or observed confinement.',
              'sources':{'experiments/biharmonic-color-window-checks.py':sha256(Path(__file__).read_bytes()).hexdigest()}}
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'checks':CHECKS,'cases':len(CASES),'report':str(args.output)}))


if __name__ == '__main__':
    main()
