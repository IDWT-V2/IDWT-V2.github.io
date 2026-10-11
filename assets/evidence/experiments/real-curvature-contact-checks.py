#!/usr/bin/env python3
"""Exact real normal-curvature descent and its occupied-sector controls."""
import argparse
from datetime import datetime, timezone
from hashlib import sha256
from itertools import combinations
import json
import math
from pathlib import Path

import numpy as np
import sympy as sp

CHECKS = 0
CASES = []


def check(condition, message):
    global CHECKS
    if not condition:
        raise AssertionError(message)
    CHECKS += 1


def parity(index):
    return (-1)**sum(index[i]>index[j] for i in range(len(index)) for j in range(i+1,len(index)))


def realify(real, imaginary):
    return np.block([[real,-imaginary],[imaginary,real]]).astype(object)


def algebra():
    basis = []
    for a,b in combinations(range(3),2):
        real = np.zeros((3,3),dtype=int)
        imaginary = np.zeros((3,3),dtype=int)
        real[a,b],real[b,a] = 1,-1
        imaginary[a,b] = imaginary[b,a] = 1
        basis.extend((realify(real,np.zeros((3,3),dtype=int)),
                      realify(np.zeros((3,3),dtype=int),imaginary)))
    for diagonal in ((1,-1,0),(1,1,-2)):
        basis.append(realify(np.zeros((3,3),dtype=int),np.diag(diagonal)))
    return basis


def wedge(left,right):
    result = {}
    for a,va in left.items():
        for b,vb in right.items():
            if set(a).intersection(b):
                continue
            index = tuple(sorted(a+b))
            value = parity(a+b)*(va@vb)
            result[index] = result.get(index,np.zeros_like(value))+value
    return {index:value for index,value in result.items() if any(item != 0 for item in value.flat)}


def power(form, exponent, size=6):
    value = {():np.eye(size,dtype=object)}
    for _ in range(exponent):
        value = wedge(value,form)
    return value


def squared_norm(form):
    return sum(sum(item*item for item in value.flat) for value in form.values())


def project_su3(form,basis):
    result = {}
    for index,value in form.items():
        projected = np.zeros((6,6),dtype=object)
        for generator in basis:
            coefficient = sp.Rational(np.trace(generator.T@value),np.trace(generator.T@generator))
            projected += coefficient*generator
        result[index] = projected
    return result


def cs_constant_gradient(form, degree, basis):
    # The real-trace transgression has delta CS = degree tr(delta A F^(degree-1)).
    return project_su3(power(form,degree-1),basis)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    rng = np.random.default_rng(202610092359)
    basis = algebra()
    identity = np.eye(3,dtype=int)
    complex_structure = realify(np.zeros((3,3),dtype=int),identity)
    for a in basis:
        check(np.array_equal(a.T,-a),'actual ordinary normal rotations are real skew matrices')
        check(np.array_equal(a@complex_structure,complex_structure@a),'normal SU3 preserves the selected ordinary complex structure')
        check(np.trace(complex_structure.T@a) == 0,'normal SU3 excludes its central U1 rotation')
    for depth in (2,3,4,5,6,10):
        degree = (depth+1)//2
        exponent = degree-1
        form = {pair:sum((int(rng.integers(-2,3))*g for g in basis),np.zeros((6,6),dtype=object))
                for pair in combinations(range(depth),2)}
        value = power(form,exponent)
        gradient = cs_constant_gradient(form,degree,basis)
        even_power = exponent%2 == 0
        for coefficient in value.values():
            check(np.array_equal(coefficient.T,(-1)**exponent*coefficient),
                  'exterior curvature powers retain real transpose parity exactly')
        contact = squared_norm(gradient)
        check((contact != 0) == (depth in (3,4)),
              'maximal fitting real transgression gradient has the occupied-sector activation rule')
        if depth in (3,4):
            check(squared_norm({index:gradient[index]-value[index] for index in value}) == 0,
                  'the active response is exactly ordinary normal-curvature squared')
        else:
            check(contact == 0,'the odd real invariant gradient vanishes without a sector coefficient')
        # A parallel J is already available in the Hermitian normal geometry.
        weighted = {index:complex_structure@coefficient for index,coefficient in value.items()}
        alternative = project_su3(weighted,basis)
        alternative_norm = squared_norm(alternative)
        if depth in (5,6,10):
            check(alternative_norm != 0,'the allowed complex-structure-weighted curvature channel survives at higher occupied depths')
        if depth == 2:
            check(alternative_norm == 0,'the central trace alone does not generate depth-two SU3')
        CASES.append({'real_depth':depth,'geometry':'ordinary real normal SU3 curvature at one physical spatial point',
                      'parameters':{'character_degree':degree,'curvature_power':exponent,
                                    'source_form_degree':2*exponent,
                                    'trace_prescription':'unweighted real metric trace'},
                      'measurements':{'gradient_norm_squared_exact':str(contact),
                                      'normal_complex_structure_weighted_gradient_norm_squared_exact':str(alternative_norm),
                                      'active_unweighted_channel':contact != 0,
                                      'all_higher_color_forces_excluded':False}})
    # A single Cartan generator gives an independent exact character test.
    cartan = realify(np.zeros((3,3),dtype=int),np.diag((1,2,-3)))
    for dimension in (2,4,6,10):
        degree = dimension//2
        form = {(2*k,2*k+1):cartan for k in range(degree)}
        top = power(form,degree)[tuple(range(dimension))]
        trace = np.trace(top)
        weighted_trace = np.trace(complex_structure@top)
        expected = math.factorial(degree)*2*sp.re((sp.I**degree)*(1+2**degree+(-3)**degree))
        check(trace == expected,'independent Cartan power agrees with the conjugate-pair character')
        if degree%2:
            check(trace == 0,'conjugate triples cancel every tested odd real curvature trace')
            if degree > 1:
                check(weighted_trace != 0,'a complex-structure insertion distinguishes the conjugate contributions')
        else:
            check(trace != 0,'quadratic real curvature traces add rather than cancel')
        CASES.append({'real_depth':dimension,'geometry':'independent Cartan curvature character',
                      'parameters':{'normal_rotation_weights':[1,2,-3]},
                      'measurements':{'real_top_trace_exact':str(trace),
                                      'complex_structure_weighted_top_trace_exact':str(weighted_trace)}})
    # Matrix polynomial differentiation checks the Lie-projected gradient.
    t = sp.symbols('t',real=True)
    a = sp.Matrix(sum((int(rng.integers(-2,3))*g for g in basis),np.zeros((6,6),dtype=object)))
    b = sp.Matrix(basis[2])
    for degree in (1,2,3,5):
        direct = sp.diff(sp.trace((a+t*b)**degree),t).subs(t,0)
        expected = degree*sp.trace(b*a**(degree-1))
        check(sp.simplify(direct-expected) == 0,'real invariant first variation has the stated common polynomial gradient')
        if degree%2:
            check(direct == 0,'odd real trace has no hidden first variation along a physical skew connection')
    # Newton identities show that the fifth complex character reuses the cubic one.
    x,y = sp.symbols('x y',real=True)
    diagonal = (x,y,-x-y)
    traces = {degree:sum(item**degree for item in diagonal) for degree in (2,3,5)}
    check(sp.expand(traces[5]-sp.Rational(5,6)*traces[2]*traces[3]) == 0,
          'the fifth SU3 trace is the product of the quadratic and cubic characters')
    CASES.append({'real_depth':10,'geometry':'traceless complex normal rank-three invariant',
                  'parameters':{'matrix_eigenvalues':['x','y','-x-y']},
                  'measurements':{'fifth_trace_identity':'tr F^5 = (5/6)(tr F^2)(tr F^3)',
                                  'independent_fifth_character':False}})
    # Positive spatial field energy can leave a negative pair energy after self subtraction.
    laplacian = sp.Matrix([[2,-1],[-1,2]])
    green = laplacian.inv()
    swap = sp.zeros(9)
    for i in range(3):
        for j in range(3):
            swap[3*j+i,3*i+j] = 1
    casimir = (swap-sp.eye(9)/3)/2
    full_energy = sp.Rational(8,9)*sp.eye(9)+casimir/3
    pair_energy = full_energy-sp.Rational(8,9)*sp.eye(9)
    check(green == sp.Matrix([[2,1],[1,2]])/3,'the spatial inverse kernel is positive and reciprocal')
    check(laplacian.det()>0 and laplacian[0,0]>0,'the complete spatial stiffness is positive')
    check(casimir.eigenvals() == {sp.Rational(-2,3):3,sp.Rational(1,3):6},
          'normal triplet pair channels retain the standard Casimir eigenvalues')
    check(full_energy.eigenvals() == {sp.Rational(2,3):3,sp.Integer(1):6},
          'the complete quadratic field energy is positive in both color pair channels')
    check(pair_energy.eigenvals() == {sp.Rational(-2,9):3,sp.Rational(1,9):6},
          'self-energy subtraction leaves a negative antisymmetric and positive symmetric pair term')
    CASES.append({'real_depth':3,'geometry':'two-site positive normal-connection Gauss control',
                  'parameters':{'spatial_stiffness':[[2,-1],[-1,2]],'source_coupling':1},
                  'measurements':{'full_antisymmetric_energy_exact':'2/3','full_symmetric_energy_exact':'1',
                                  'pair_antisymmetric_energy_exact':'-2/9','pair_symmetric_energy_exact':'1/9',
                                  'same_field_reduction_derived':False}})
    # The four-dimensional Green function has the ordinary three-dimensional marginal.
    radius,u = sp.symbols('r u',positive=True,real=True)
    g3 = 1/(4*sp.pi*radius)
    g4 = 1/(4*sp.pi**2*radius**2)
    marginal = sp.integrate(1/(4*sp.pi**2*(radius**2+u**2)),(u,-sp.oo,sp.oo))
    check(sp.simplify(marginal-g3) == 0,'the noncompact four-dimensional Green marginal equals the three-dimensional kernel')
    for depth,kernel,sphere_area in ((3,g3,4*sp.pi),(4,g4,2*sp.pi**2)):
        lap = sp.diff(kernel,radius,2)+(depth-1)*sp.diff(kernel,radius)/radius
        flux = -sphere_area*radius**(depth-1)*sp.diff(kernel,radius)
        check(sp.simplify(lap) == 0,'the spatial Green function is harmonic away from the source')
        check(sp.simplify(flux-1) == 0,'the Green function has the canonical positive unit source')
        CASES.append({'real_depth':depth,'geometry':'ordinary noncompact radial Laplace Green function',
                      'parameters':{'radial_kernel':str(kernel)},
                      'measurements':{'away_source_laplacian_exact':'0','source_flux_exact':'1',
                                      'four_to_three_marginal_exact':str(marginal),
                                      'localized_particle_profile_derived':False}})
    stiffness,source,coordinate = sp.symbols('k rho q',positive=True,real=True)
    constraint = sp.diff(stiffness*coordinate**2/2-source*coordinate,coordinate)
    check(constraint == stiffness*coordinate-source,'removing stiffness does not remove the connection source vertex')
    check(sp.limit(source/stiffness,stiffness,0,dir='+') == sp.oo,
          'zero stiffness with retained nonzero source is singular rather than inert')
    CASES.append({'real_depth':2,'geometry':'inactive-curvature connection-source control',
                  'parameters':{'constraint':'k q - rho = 0'},
                  'measurements':{'zero_stiffness_requires_zero_source':True,
                                  'zero_stiffness_implies_zero_coupling':False}})
    report = {'experiment':'real-curvature-contact','created_utc':datetime.now(timezone.utc).isoformat(),
              'status':'passed','checks':CHECKS,'cases':CASES,
              'scope':'A candidate maximal-degree real-transgression-gradient energy has the occupied-sector zeros. Lower curvature norms and J-weighted terms remain allowed, so no complete force exclusion follows.',
              'sources':{'experiments/real-curvature-contact-checks.py':sha256(Path(__file__).read_bytes()).hexdigest()}}
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'checks':CHECKS,'cases':len(CASES),'report':str(args.output)}))


if __name__ == '__main__':
    main()
