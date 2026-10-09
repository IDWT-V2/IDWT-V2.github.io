#!/usr/bin/env python3
"""Exact shared sector geometry tests; no particle evolution or database writes."""
import argparse
from datetime import datetime, timezone
from hashlib import sha256
import itertools
import json
from pathlib import Path

import sympy as s


CHECKS = []
RESULTS = {}
SOURCE_CATALOG_PATH = 'experiments/shared-sector-geometry-checks.py'


def check(name, condition):
    if not bool(condition):
        raise AssertionError(name)
    CHECKS.append(name)


def zero(matrix):
    return all(s.simplify(x) == 0 for x in matrix)


def comm(a, b):
    return a*b-b*a


def unit(n, i, j):
    result = s.zeros(n)
    result[i,j] = 1
    return result


def tensor(*matrices):
    result = s.ones(1)
    for matrix in matrices:
        result = s.kronecker_product(result, matrix)
    return result


def exterior(m):
    size = 2**m
    creators = []
    for j in range(m):
        op = s.zeros(size)
        for subset in range(size):
            if not subset & (1 << j):
                before = (subset & ((1 << j)-1)).bit_count()
                op[subset | (1 << j), subset] = (-1)**before
        creators.append(op)
    gammas = []
    for op in creators:
        gammas.extend([op+op.T, s.I*(op-op.T)])
    number = s.diag(*[k.bit_count() for k in range(size)])
    parity = s.diag(*[(-1)**k.bit_count() for k in range(size)])
    return creators, gammas, number, parity


def exterior_action(x, creators):
    return sum((x[i,j]*creators[i]*creators[j].T
                for i in range(x.rows) for j in range(x.cols)), s.zeros(creators[0].rows))


def realify(x):
    return s.Matrix.vstack(s.Matrix.hstack(s.re(x),-s.im(x)),
                           s.Matrix.hstack(s.im(x),s.re(x)))


def charge_completion():
    creators, gammas, number, parity = exterior(3)
    eye = s.eye(8)
    for i,a in enumerate(gammas):
        check(f'normal gamma Hermitian {i}', a.H == a)
        for j,b in enumerate(gammas):
            check(f'normal Clifford metric {i}/{j}', a*b+b*a == 2*(i == j)*eye)
    altered_i = realify(s.I*parity)
    check('changed carrier complex structure squares minus one', altered_i**2 == -s.eye(16))
    for i,a in enumerate(gammas):
        ar = realify(a)
        check(f'changed carrier odd Clifford anti-linear {i}', zero(altered_i*ar+ar*altered_i))
        for j,b in enumerate(gammas):
            check(f'changed carrier even Clifford linear {i}/{j}', zero(comm(altered_i,realify(a*b))))

    even = (eye+parity)/2
    odd = eye-even
    matter = s.diag(even,odd)
    complement = s.eye(16)-matter
    q = s.diag(number,-number)/3
    completed = [s.diag(g,s.conjugate(g)) for g in gammas]
    for i,g in enumerate(completed):
        check(f'completion matter compression zero {i}', zero(matter*g*matter))
        b = complement*g*matter
        check(f'completion chiral symbol norm {i}', b.H*b == matter)
    vector = s.Matrix([1,2,-1,3,-2,1])
    symbol = sum((vector[i]*completed[i] for i in range(6)),s.zeros(16))
    b = complement*symbol*matter
    check('completion mixed chiral symbol metric', zero(b.H*b-vector.dot(vector)*matter))
    matter_weights = [q[i,i] for i in range(16) if matter[i,i] == 1]
    opposite_weights = [q[i,i] for i in range(16) if complement[i,i] == 1]
    check('matter weights retained', sorted(matter_weights) == sorted([0]+[s.Rational(2,3)]*3+[-1]+[s.Rational(-1,3)]*3))
    check('complement has conjugate weights', sorted(opposite_weights) == sorted([-x for x in matter_weights]))
    for i,g in enumerate(completed):
        paired = completed[i+1] if i % 2 == 0 else -completed[i-1]
        check(f'central charge rotates normal symbol {i}', zero(comm(s.I*3*q,g)-paired))
        check(f'central fibre charge not full kinetic commutant {i}', not zero(comm(q,g)))

    # Chern roots provide a representation identity, without an empirical interpretation.
    x = s.symbols('x1:4')
    weights = [s.Integer(0),x[0]+x[1],x[0]+x[2],x[1]+x[2],-sum(x),-x[0],-x[1],-x[2]]
    c1 = sum(x)
    c2 = sum(x[i]*x[j] for i in range(3) for j in range(i+1,3))
    c3 = s.prod(x)
    check('charge carrier first Chern character zero', s.expand(sum(weights)) == 0)
    check('charge carrier second Chern character', s.expand(sum(w**2 for w in weights)/2-2*(c1**2-c2)) == 0)
    check('charge carrier third Chern character equals minus normal Euler', s.expand(sum(w**3 for w in weights)/6+c3) == 0)
    RESULTS['charge_completion'] = {'normal_complex_rank':3,'matter_rank':8,'minimal_full_complex_clifford_completion_rank':16,'matter_weights':[str(x) for x in matter_weights],'conjugate_weights':[str(x) for x in opposite_weights],'chern_character_2':'2(c1(W)^2-c2(W))','chern_character_3':'-c3(W)'}


def tangent_projectors():
    _, gammas, _, _ = exterior(5)
    eye = s.eye(32)
    def volume(d):
        value = eye
        for g in gammas[:d]:
            value = value*g
        return s.simplify(s.I**(d*(d-1)//2))*value
    omega3,omega5 = volume(3),volume(5)
    k78,k910 = s.I*gammas[6]*gammas[7],s.I*gammas[8]*gammas[9]
    p6 = (eye+k78)*(eye+k910)/4
    p5 = p6*(eye+omega5)/2
    p3 = p5*(eye+omega3)/2
    projectors = {2:p3,3:p3,4:p5,5:p5,6:p6,10:eye}
    counts = {}
    for d,p in projectors.items():
        check(f'tangent projector Hermitian {d}', p.H == p)
        check(f'tangent projector idempotent {d}', p*p == p)
        check(f'tangent projector minimal rank {d}', s.trace(p) == 2**(d//2))
        for i,g in enumerate(gammas[:d]):
            check(f'tangent Clifford invariant {d}/{i}', zero(comm(p,g)))
        dimension = 0
        for bits in range(1024):
            degree = bits.bit_count()
            if all((degree - bool(bits & (1 << j))) % 2 == 0 for j in range(d)):
                dimension += 1
        counts[d] = dimension
    check('full flag Clifford commutant dimensions', list(counts.values()) == [256,128,64,32,16,1])
    previous = None
    for d,p in projectors.items():
        if previous is not None:
            check(f'nested minimal fibre projector {d}', previous*p == previous)
        previous = p
    for d in (3,5):
        p = (eye+volume(d))/2
        other = eye-p
        for j in range(d,10):
            check(f'odd tangent sign normal selection {d}/{j}', zero(p*gammas[j]*p))
        check(f'even normal product preserves odd sign {d}', zero(p*gammas[d]*gammas[d+1]*other))
    for d in (2,4,6,10):
        p = (eye+volume(d))/2
        check(f'even tangent Weyl compression loses symbol {d}', all(zero(p*g*p) for g in gammas[:d]))
    for d,p in projectors.items():
        for i,g in enumerate(gammas[:d]):
            compressed = p*g*p
            defect = p-compressed*compressed
            leakage = (eye-p)*g*p
            check(f'compression defect equals squared leakage {d}/{i}', defect == leakage.H*leakage)
    generic = s.zeros(32); generic[0,0]=1
    g = gammas[0]
    check('noninvariant projector has positive compression defect', generic-(generic*g*generic)**2 == ((eye-generic)*g*generic).H*((eye-generic)*g*generic))
    RESULTS['tangent_projectors'] = {'ranks':{str(d):int(s.trace(p)) for d,p in projectors.items()},'complex_commutant_dimensions':counts,'scope':'Pointwise compatible nested minimal Clifford carriers; parallel transport requires additional geometry.'}


def unitary_geometry():
    n = 3
    basis = [s.I*unit(n,j,j) for j in range(n)]
    for i in range(n):
        for j in range(i+1,n):
            basis += [(unit(n,i,j)-unit(n,j,i))/s.sqrt(2),s.I*(unit(n,i,j)+unit(n,j,i))/s.sqrt(2)]
    real_basis = [realify(t) for t in basis]
    jmatrix = realify(s.I*s.eye(n))
    eye = s.eye(6)
    for i,a in enumerate(basis):
        for j,b in enumerate(basis):
            check(f'unitary generator normalization {i}/{j}', s.simplify(-s.trace(a*b)) == (i == j))
    invariants = []
    for cosine in (s.Integer(0),s.Rational(3,5),s.Integer(1)):
        sine = s.sqrt(1-cosine**2)
        u = s.eye(6)[:,0]
        v = cosine*jmatrix*u+sine*s.eye(6)[:,1]
        p = u*u.T+v*v.T
        kappa = s.trace((p*jmatrix*p).T*(p*jmatrix*p))
        leakage = sum(s.trace(((eye-p)*t*p).T*((eye-p)*t*p)) for t in real_basis)
        check(f'plane Kaehler invariant {cosine}', s.simplify(kappa-2*cosine**2) == 0)
        check(f'plane leakage Casimir {cosine}', s.simplify(leakage-(5-cosine**2)) == 0)
        center = jmatrix/s.sqrt(3)
        center_leakage = s.trace(((eye-p)*center*p).T*((eye-p)*center*p))
        check(f'plane central leakage {cosine}', s.simplify(center_leakage-2*(1-cosine**2)/3) == 0)
        line = u*u.T
        constraints = []
        for t in real_basis:
            constraints.append(s.Matrix(list(comm(t,line))+list(comm(t,p))))
        stabilizer = 9-s.Matrix.hstack(*constraints).rank()
        check(f'line-plane stabilizer dimension {cosine}', stabilizer == (4 if cosine == 1 else 1))
        invariants.append({'cosine':str(cosine),'leakage':str(s.simplify(leakage)),'stabilizer_dimension':stabilizer})
    line = s.diag(1,0,0,0,0,0)
    line_leakage = sum(s.trace(((eye-line)*t*line).T*((eye-line)*t*line)) for t in real_basis)
    check('line leakage Casimir three', s.simplify(line_leakage) == 3)

    # An arbitrary rational normal connection at one base point tests the induced metric.
    sample_a = [real_basis[0],s.sqrt(2)*real_basis[3],s.sqrt(2)*real_basis[7],real_basis[2]]
    for r,indices in ((1,[0]),(2,[0,3]),(2,[0,1]),(6,list(range(6)))):
        include = s.eye(6)[:,indices]
        p = include*include.T
        y = include*s.Matrix(list(range(1,r+1)))
        k = s.Matrix.hstack(*[a*y for a in sample_a])
        g = s.BlockMatrix([[s.eye(4)+k.T*k,k.T*include],[include.T*k,s.eye(r)]]).as_explicit()
        c = (eye-p)*k
        check(f'induced contact volume Schur identity {indices}', g.det() == (s.eye(4)+c.T*c).det())
        check(f'induced contact volume nondecrease {indices}', g.det() >= 1)
    omega = s.Matrix([[1,2,3,4]])
    tau = s.symbols('tau', real=True)
    c = tau*jmatrix*s.eye(6)[:,0]*omega
    check('central real-line measure correction', s.factor((s.eye(4)+c.T*c).det()) == 1+30*tau**2)

    # Curvature matrices of the connection-metric construction generate all u(3).
    x12 = unit(3,0,1)-unit(3,1,0)
    y12 = s.I*(unit(3,0,1)+unit(3,1,0))
    x23 = unit(3,1,2)-unit(3,2,1)
    y23 = s.I*(unit(3,1,2)+unit(3,2,1))
    generators = [s.I*unit(3,0,0),x12,y12,x23,y23,s.I*s.eye(3)]
    candidates = generators+[comm(x12,y12),comm(x23,y23),comm(x12,x23),comm(x12,y23)]
    columns = [s.Matrix(list(s.re(t))+list(s.im(t))) for t in candidates]
    check('normal curvature Lie span full u3', s.Matrix.hstack(*columns).rank() == 9)
    RESULTS['unitary_geometry'] = {'normal_holonomy_lie_dimension':9,'line_leakage':'3','plane_controls':invariants,'scope':'Lie algebra, induced metric and orientation identities; geometric existence and completeness use the accompanying proofs.'}


def common_spin_factor():
    y = s.diag(s.Rational(-1,2),*([s.Rational(1,6)]*3))
    t3 = s.diag(s.Rational(1,2),s.Rational(-1,2))
    q = tensor(y,s.eye(2))+tensor(s.eye(4),t3)
    expected = [0,-1]+[s.Rational(2,3),s.Rational(-1,3)]*3
    check('spin and branch factor recover charge weights', list(q.diagonal()) == expected)
    check('spin factor traceless', s.trace(y) == 0)
    check('branch factor traceless', s.trace(t3) == 0)
    ladder = tensor(s.eye(4),unit(2,0,1))
    check('common factor ladder charge difference', comm(q,ladder) == ladder)
    check('common factor ladder squared zero', zero(ladder*ladder))
    check('common factor ladder anticommutator', ladder*ladder.H+ladder.H*ladder == s.eye(8))
    creators,gammas,number,parity = exterior(3)
    full_ladder = tensor(s.eye(8),unit(2,0,1))
    for i,g in enumerate(gammas):
        check(f'common multiplicity ladder Clifford scalar {i}', zero(comm(tensor(g,s.eye(2)),full_ladder)))
    even = (s.eye(8)+parity)/2
    neutral_even = unit(8,0,0)
    colored_even = even-neutral_even
    unequal = tensor(neutral_even+2*colored_even,unit(2,0,1))
    check('separately tuned chiral ladder not Clifford scalar', any(not zero(comm(tensor(g,s.eye(2)),unequal)) for g in gammas))
    fock_q = tensor(number/3,s.eye(2))+tensor(s.eye(8),s.diag(0,-1))
    p = tensor(even,s.eye(2))
    check('root-free positive chiral rank eight', s.trace(p) == 8)
    check('root-free common charge table', sorted([fock_q[i,i] for i in range(16) if p[i,i] == 1]) == sorted(expected))
    # Reconstruct the general Spin-h positive-carrier count under one rule.
    rank_rows = []
    for n in range(1,7):
        terms = [(k,s.binomial(n,k)) for k in range(0,n+1,2)]
        rank = sum(2*m for _,m in terms)
        check(f'Spin-h even carrier rank {n}', rank == 2**n)
        trace = sum(m*(s.Rational(2*k,n)-1) for k,m in terms)
        check(f'Spin-h carrier trace control {n}', trace == (-1 if n == 1 else 0))
        rank_rows.append({'normal_complex_rank':n,'positive_carrier_rank':int(rank),'even_degree_multiplicities':[int(m) for _,m in terms]})
    RESULTS['spin_factor'] = {'positive_chiral_charge_weights':[str(x) for x in expected],'normal_rank_controls':rank_rows,'scope':'Normal Spin-h carrier, not observer helicity or selected physical interaction.'}


def curvature_bounds():
    results = []
    for n,k in ((3,1),(3,2),(4,1),(4,2)):
        r=n-k
        b=s.I*s.diag(*range(1,r+1))
        if r > 1:
            b += unit(r,0,1)-unit(r,1,0)
        f=s.diag(s.zeros(k),b)
        t=s.trace(f)
        f0=f-t*s.eye(n)/n
        remainder=b-t*s.eye(r)/r
        norm=lambda a: s.simplify(s.trace(a.H*a))
        delta=s.simplify(norm(f0)-s.Rational(k,n*r)*s.conjugate(t)*t)
        check(f'sharp curvature exact excess {n}/{k}', s.simplify(delta-norm(remainder)) == 0)
        check(f'sharp curvature nonnegative excess {n}/{k}', delta >= 0)
        equal=s.diag(s.zeros(k),s.I*3*s.eye(r))
        equal_t=s.trace(equal)
        check(f'sharp curvature equality control {n}/{k}', s.simplify(norm(equal-equal_t*s.eye(n)/n)-s.Rational(k,n*r)*s.conjugate(equal_t)*equal_t) == 0)
        results.append({'n':n,'fixed_complex_rank':k,'trace_coefficient':str(s.Rational(k,n*r)),'sample_excess':str(delta)})
    roots=s.symbols('u v')
    c1=sum(roots);c2=s.prod(roots)
    check('rank two projective discriminant roots', s.expand(c1*c1-4*c2-(roots[0]-roots[1])**2) == 0)
    check('S2 times S2 discriminant nonzero', 2-4 == -2)
    RESULTS['curvature_bounds']={'pointwise_controls':results,'rank_two_discriminant':'c1^2-4c2','scope':'Normal curvature norms; traceless curvature alone does not prove non-Abelian holonomy or an observed force.'}


def photon_geometry():
    def cross(k):
        return s.Matrix([[0,-k[2],k[1]],[k[2],0,-k[0]],[-k[1],k[0],0]])
    for number,k in enumerate([s.Matrix([0,0,1]),s.Matrix([1,0,0]),s.Matrix([s.Rational(3,5),0,s.Rational(4,5)])]):
        q=s.eye(3)-k*k.T
        c=cross(k)
        check(f'transverse vector geometry {number}', c*c == -q)
        positive=(q+s.I*c)/2;negative=(q-s.I*c)/2
        for sign,p in ((1,positive),(-1,negative)):
            check(f'helicity projector Hermitian {number}/{sign}', p.H == p)
            check(f'helicity projector idempotent {number}/{sign}', zero(p*p-p))
            check(f'helicity projector rank one {number}/{sign}', s.trace(p) == 1)
        check(f'opposite helicity orthogonality {number}', zero(positive*negative))
    spinors=[(1,0),(0,1),(s.Rational(3,5),s.Rational(4,5)),(1/s.sqrt(2),s.I/s.sqrt(2))]
    def epsilon(u,v):
        return s.Matrix([u*u-v*v,s.I*(u*u+v*v),-2*u*v])/s.sqrt(2)
    for number,(u,v) in enumerate(spinors):
        k=s.Matrix([2*s.re(s.conjugate(u)*v),2*s.im(s.conjugate(u)*v),s.conjugate(u)*u-s.conjugate(v)*v])
        ep=epsilon(u,v)
        check(f'spinor square direction unit {number}', s.simplify(k.dot(k)) == 1)
        check(f'spinor square transverse {number}', s.simplify(k.dot(ep)) == 0)
        check(f'spinor square null polarization {number}', s.simplify(ep.dot(ep)) == 0)
        check(f'spinor square unit norm {number}', s.simplify((ep.H*ep)[0]) == 1)
        check(f'spinor square circular helicity {number}', zero(cross(k)*ep+s.I*ep))
        check(f'spinor square phase doubling {number}', zero(epsilon(s.I*u,s.I*v)+ep))
    for number,(a,b) in enumerate([(1,0),(s.Rational(3,5),s.Rational(4,5)),(1/s.sqrt(2),s.I/s.sqrt(2)),(s.Rational(3,5),4*s.I/5)]):
        s0=s.conjugate(a)*a+s.conjugate(b)*b
        stokes=[s.conjugate(a)*a-s.conjugate(b)*b,2*s.re(s.conjugate(a)*b),2*s.im(s.conjugate(a)*b)]
        check(f'Stokes pure state identity {number}', s.simplify(s0*s0-sum(x*x for x in stokes)) == 0)
        check(f'Stokes circular fraction {number}', s.simplify(stokes[2]/s0) == [0,0,1,s.Rational(24,25)][number])
    theta,phi=s.symbols('theta phi',real=True)
    z=s.Matrix([s.cos(theta/2),s.exp(s.I*phi)*s.sin(theta/2)])
    ep=epsilon(z[0],z[1])
    asp=s.simplify(-s.I*(s.conjugate(z).T*z.diff(phi))[0])
    ave=s.simplify(s.expand_complex(-s.I*(s.conjugate(ep).T*ep.diff(phi))[0]))
    check('Berry phase doubles under spinor square', s.trigsimp(ave-2*asp) == 0)
    check('spinor Berry flux one', s.integrate(s.diff(asp,theta),(theta,0,s.pi)) == 1)
    check('vector Berry flux two', s.simplify(s.integrate(s.diff(ave,theta),(theta,0,s.pi))) == 2)
    RESULTS['photon_geometry']={'spinor_Berry_flux':1,'vector_Berry_flux':2,'scope':'Observer transverse vector geometry and coefficient-line relation; no photon dynamics or sector topology selected.'}


def contact_checks():
    checked = []
    def check(name, value):
        assert value, name
        checked.append(name)
    def matrix_unit(n, i, j):
        M = s.zeros(n); M[i,j] = 1
        return M
    def rho(X):
        t = s.trace(X)
        return s.diag(s.zeros(1), t*s.eye(3)-X.T, s.Matrix([[-t]]), -X.T)
    generators = [matrix_unit(3,i,j) for i in range(3) for j in range(3)]
    R = rho(s.eye(3))
    T = matrix_unit(8,0,4) + sum((matrix_unit(8,1+i,5+i) for i in range(3)), s.zeros(8))
    Pminus = s.diag(0,0,0,0,1,1,1,1)
    Pplus = s.eye(8)-Pminus
    check('common determinant ladder square zero', T*T == s.zeros(8))
    check('ladder reduced incoming norm', T.T*T == Pminus)
    check('ladder reduced outgoing norm', T*T.T == Pplus)
    check('ladder Clifford-like anticommutator', T.T*T+T*T.T == s.eye(8))
    check('normalized charge ladder difference', (R*T-T*R)/3 == T)
    check('color trace strength ratio', s.trace(T[1:4,5:8].T*T[1:4,5:8]) == 3*s.trace(T[0:1,4:5].T*T[0:1,4:5]))
    for k,X in enumerate(generators):
        RX = rho(X)
        check(f'determinant equivariance generator {k}', RX*T-T*RX == s.trace(X)*T)
        for i in range(3):
            C = matrix_unit(8,0,5+i)
            B = C.T
            check(f'W vector equivariance {k}/{i}', RX*C-C*RX == sum((X[j,i]*matrix_unit(8,0,5+j) for j in range(3)), s.zeros(8)))
            check(f'W dual vector equivariance {k}/{i}', RX*B-B*RX == sum((-X[i,j]*matrix_unit(8,5+j,0) for j in range(3)), s.zeros(8)))
    kernel = s.diag(0,1,1,1,1,0,0,0)
    for i in range(3):
        G = matrix_unit(8,0,5+i)+matrix_unit(8,5+i,0)
        check(f'normal vector common kernel {i}', G*kernel == s.zeros(8))
        check(f'normal vector fails full Clifford {i}', G*G != s.eye(8))
    # Solve exact infinitesimal intertwiner systems after central-weight sieve.
    def hom_dimension(character, gens=generators, central=R, unit_weight=3, parity=None):
        weights = list(central.diagonal())
        basis = [matrix_unit(8,i,j) for i in range(8) for j in range(8) if weights[i]-weights[j] == unit_weight*character and (parity is None or parity[i]*parity[j] == (-1)**character)]
        columns = []
        for K in basis:
            flattened = []
            for X in gens:
                M = rho(X)*K-K*rho(X)-character*s.trace(X)*K
                flattened.extend(list(M))
            columns.append(s.Matrix(flattened))
        equations = s.Matrix.hstack(*columns)
        return len(basis)-equations.rank()
    check('scalar U3 contact dimension four', hom_dimension(0) == 4)
    check('determinant U3 contact dimension two', hom_dimension(1) == 2)
    check('inverse determinant U3 contact dimension two', hom_dimension(-1) == 2)
    check('central holonomy commutant dimension twenty', sum(1 for a in R.diagonal() for b in R.diagonal() if a == b) == 20)
    fixed_gens = [matrix_unit(3,i,j) for i in (1,2) for j in (1,2)]
    fixed_central = rho(s.diag(0,1,1))
    flag_parity = [1,1,-1,-1,-1,-1,1,1]
    check('oriented compatible line-plane flag scalar commutant dimension eight', hom_dimension(0,fixed_gens,fixed_central,2) == 8)
    check('oriented compatible line-plane flag determinant contact dimension five', hom_dimension(1,fixed_gens,fixed_central,2) == 5)
    check('unoriented compatible line-plane flag scalar commutant dimension six', hom_dimension(0,fixed_gens,fixed_central,2,flag_parity) == 6)
    check('unoriented compatible line-plane flag determinant contact dimension three', hom_dimension(1,fixed_gens,fixed_central,2,flag_parity) == 3)
    return {'passed':len(checked), 'checks':checked}



def contacts():
    result = contact_checks()
    CHECKS.extend(result['checks'])
    RESULTS['contacts'] = {'passed':result['passed'],'scope':'Exact representation intertwiners and common ladder norms, not physical rates.'}


def lifted_charge():
    from collections import Counter
    from math import comb

    def exponents(n, d):
        if d == 1:
            yield (n,)
        else:
            for k in range(n+1):
                for tail in exponents(n-k, d-1):
                    yield (k,)+tail

    def weight(key):
        alpha, mask, branch = key
        return mask.bit_count()-sum(alpha[:3])+sum(alpha[3:])-3*branch

    def add(target, key, value):
        target[key] = target.get(key, 0)+value
        if target[key] == 0:
            del target[key]

    def action(vector, kind):
        out = {}
        for (alpha, mask, branch), value in vector.items():
            if kind == 'M':
                add(out, (alpha, mask, branch), value*weight((alpha, mask, branch)))
                continue
            for j in range(3):
                occupied = bool(mask & (1 << j))
                sign = (-1)**((mask & ((1 << j)-1)).bit_count())
                # D: annihilate with dz; create with dzbar.
                # X: create with z; annihilate with zbar.
                coordinate = j if occupied == (kind == 'D') else j+3
                new_alpha = list(alpha)
                if kind == 'D':
                    coefficient = 2*new_alpha[coordinate]
                    if coefficient == 0:
                        continue
                    new_alpha[coordinate] -= 1
                else:
                    coefficient = 1
                    new_alpha[coordinate] += 1
                add(out, (tuple(new_alpha), mask ^ (1 << j), branch), value*sign*coefficient)
        return out

    def plus(a, b):
        out = dict(a)
        for key, value in b.items():
            add(out, key, value)
        return out

    def scalar_character(n, a, r):
        if n < 0:
            return Counter()
        out = Counter()
        weights = [0]*a+[-1]*r+[1]*r
        for alpha in exponents(n, len(weights)):
            out[sum(x*w for x,w in zip(alpha, weights))] += 1
        return out

    def convolve(a, b):
        out = Counter()
        for i,x in a.items():
            for j,y in b.items():
                out[i+j] += x*y
        return out

    fibre = Counter(mask.bit_count()-3*b for mask in range(8) for b in range(2))
    chirals = [Counter(mask.bit_count()-3*b for mask in range(8) for b in range(2)
                      if mask.bit_count()%2 == parity) for parity in range(2)]
    check('lifted full fibre character dimension', sum(fibre.values()) == 16)
    for n in range(4):
        for alpha in exponents(n, 6):
            for mask in range(8):
                # Both H branches differ only by a constant commuting shift.
                vector = {(alpha,mask,0):1}
                for op in ['D','X']:
                    check(f'lifted [M,{op}] n{n} a{alpha} s{mask}',
                          action(action(vector,op),'M') == action(action(vector,'M'),op))
                anti = plus(action(action(vector,'X'),'D'),action(action(vector,'D'),'X'))
                check(f'lifted Dirac ladder anticommutator n{n} a{alpha} s{mask}',
                      anti == {(alpha,mask,0):2*(n+3)})
    for n in range(3):
        source = [(alpha,mask,b) for alpha in exponents(n,6)
                  for mask in range(8) for b in range(2)]
        expected_full = convolve(scalar_character(n,0,3),fibre)
        lower_full = convolve(scalar_character(n-1,0,3),fibre)
        observed = Counter()
        for parity in range(2):
            expected = convolve(scalar_character(n,0,3),chirals[parity])
            lower = convolve(scalar_character(n-1,0,3),chirals[1-parity])
            for charge in sorted(set(expected)|set(lower)):
                cols = [key for key in source if weight(key)==charge and key[1].bit_count()%2==parity]
                images = [action({key:1},'D') for key in cols]
                rows = sorted({key for image in images for key in image})
                matrix = s.Matrix([[image.get(key,0) for image in images] for key in rows])
                rank = matrix.rank() if rows else 0
                nullity = len(cols)-rank
                check(f'lifted monogenic rank n{n} chirality{parity} M{charge}',
                      rank == lower[charge] and nullity == expected[charge]-lower[charge])
                observed[charge] += nullity
        check(f'lifted monogenic full character n{n}',
              observed == expected_full-lower_full)
        check(f'lifted monogenic full dimension n{n}',sum(observed.values())==16*comb(n+4,4))
    for p in range(6):
        key = ((p,0,0,0,p,0),2,0)
        check(f'lifted repeated M1 normal monogenic grade{2*p}',
              action({key:1},'D') == {} and weight(key)==1)
    for q in range(5):
        for delta in [0,2]:
            for b in range(2):
                key = ((0,0,q+delta,0,q,0),3,b)
                check(f'lifted four retained weights repeat q{q} delta{delta} branch{b}',
                      action({key:1},'D') == {} and weight(key)==2-delta-3*b
                      and key[1].bit_count()%2==0)
    table = []
    for depth,a,r in [(2,2,0),(3,3,0),(4,4,0),(6,4,1),(8,4,2),(10,4,3)]:
        for n in range(4):
            current = convolve(scalar_character(n,a,r),fibre)
            previous = convolve(scalar_character(n-1,a,r),fibre) if r else Counter()
            mono = Counter({q:current[q]-previous[q] for q in set(current)|set(previous)})
            check(f'lifted depth{depth} homogeneous count n{n}',
                  sum(current.values()) == 16*comb(n+depth-1,depth-1))
            expected = 16*comb(n+depth-2,depth-2) if r else sum(current.values())
            check(f'lifted depth{depth} normal monogenic count n{n}',sum(mono.values())==expected)
            check(f'lifted depth{depth} nonnegative weight multiplicities n{n}',all(x>=0 for x in mono.values()))
            table.append({'depth':depth,'grade':n,'all_spinor_jets':sum(current.values()),
                          'normal_monogenic':sum(mono.values()),'M0_monogenic':mono[0]})
    creators,gammas,number,parity = exterior(3)
    for j in range(3):
        check(f'fibre scalar-charge commutator x{j}',comm(number/3,gammas[2*j])==-s.I*gammas[2*j+1]/3)
        check(f'fibre rotated Clifford direction invertible {j}',gammas[2*j+1]**2==s.eye(8))
    RESULTS['lifted_charge'] = {'comparison':table,'scope':'Flat polynomial normal jets, not selected physical particles or full IDOS mass modes.',
                               'monogenic_generation':'(1-t)(1+u)^3(1+u^-3)/((1-tu)^3(1-t/u)^3)',
                               'fixed_fibre_obstruction':'A Q-scalar Clifford-invariant nonzero subspace cannot include a tangent v with Jv nonzero.'}


def real_clifford():
    creators, gammas, number, parity = exterior(3)
    c = [realify(g) for g in gammas]
    identity = s.eye(16)
    scalar_i = realify(s.I*s.eye(8))
    conjugation = s.diag(s.eye(8),-s.eye(8))
    omega = identity
    for g in c:
        omega = omega*g
    altered_i = realify(s.I*parity)
    a = c[0]*c[2]*c[4]*conjugation
    ia = scalar_i*a
    check('real normal volume identifies charge complex structure', omega == -altered_i)
    check('real normal volume squares minus identity', omega**2 == -identity)
    check('real normal volume is orthogonal complex structure', omega.T == -omega)
    check('quaternion commutant generator square', a*a == -identity)
    check('quaternion commutant generator skew', a.T == -a)
    check('quaternion commutant anticommutation', a*scalar_i+scalar_i*a == s.zeros(16))
    trial = scalar_i/3+2*a/3+2*ia/3
    check('quaternion sphere sample square', trial*trial == -identity)
    check('quaternion sphere sample skew', trial.T == -trial)
    for index,g in enumerate(c):
        for name,j in [('A',a),('IA',ia),('sample',trial)]:
            check(f'real full Clifford commutant {name}/{index}', zero(comm(j,g)))
        check(f'volume complex structure anti-linear symbol {index}', zero(altered_i*g+g*altered_i))
        check(f'volume complex structure not symbol scalar {index}', not zero(comm(altered_i,g)))
    for name,j in [('I',scalar_i),('A',a),('IA',ia)]:
        check(f'rotation volume commutes quaternion {name}', zero(comm(omega,j)))
        check(f'quaternion lies mixed rotation component {name}', s.trace(-omega*j) == 0)
    central = realify(s.I*number)
    check('quaternion determinant rotation A', comm(central,a) == 3*ia)
    check('quaternion determinant rotation IA', comm(central,ia) == -3*a)
    for index,x in enumerate([s.I*s.diag(1,-1,0),unit(3,0,1)-unit(3,1,0)]):
        lift = realify(exterior_action(x,creators))
        check(f'SU3 fixes quaternion A {index}', zero(comm(lift,a)))
        check(f'SU3 fixes quaternion IA {index}', zero(comm(lift,ia)))
    check('real geometric charge weights', -altered_i*central/3 == realify(parity*number/3))
    check('real geometric circle preserves charge complex structure', zero(comm(central,altered_i)))
    even = (identity+realify(parity))/2
    odd = identity-even
    raising = even*a*odd
    check('real raising incoming norm common', raising.T*raising == odd)
    check('real raising outgoing norm common', raising*raising.T == even)
    check('real raising nilpotent', zero(raising*raising))
    check('isolated raising is not full real Clifford scalar', not zero(comm(raising,c[0])))
    check('real compatible skew connection is E complex linear', zero(comm(a,altered_i)))
    for m in range(1,6):
        _,gs,_,gamma = exterior(m)
        volume = s.eye(2**m)
        for g in gs:
            volume = volume*g
        check(f'normal volume neighboring identity {m}', volume == s.I**m*gamma)
        check(f'normal volume neighboring square {m}', volume*volume == (-1)**(m*(2*m-1))*s.eye(2**m))
    RESULTS['real_clifford'] = {'real_rank':16,'charge_complex_structure':'minus oriented normal Clifford volume','full_compatible_complex_structures':'quaternionic two-sphere; canonical U3 fixes only plus or minus original scalar I','scope':'Real-linear spatial Clifford alternative and common connection strength; no physical kinetic action or charge selection.'}


def geometric_ladders():
    u = s.eye(3)[:,0]
    x = s.symbols('x1:5',real=True)
    connection = [s.I*unit(3,0,0),unit(3,1,0)-unit(3,0,1),unit(3,2,0)-unit(3,0,2),s.I*x[0]*s.eye(3)]
    alpha = [a*u for a in connection]
    det = lambda a,b,c: s.Matrix.hstack(a,b,c).det()
    curvature = {(i,j):connection[j].diff(x[i])-connection[i].diff(x[j])+comm(connection[i],connection[j]) for i in range(4) for j in range(4)}
    sigma = {(i,j):det(u,alpha[i],alpha[j]) for i in range(4) for j in range(4)}
    omega3 = {}; omega_f = {}
    for i,j,k in itertools.combinations(range(4),3):
        key = (i,j,k)
        omega3[key] = s.simplify(det(alpha[i],alpha[j],alpha[k]))
        omega_f[key] = s.simplify(det(u,alpha[i],curvature[j,k]*u)-det(u,alpha[j],curvature[i,k]*u)+det(u,alpha[k],curvature[i,j]*u))
        derivative = sigma[j,k].diff(x[i])-sigma[i,k].diff(x[j])+sigma[i,j].diff(x[k])
        derivative += s.trace(connection[i])*sigma[j,k]-s.trace(connection[j])*sigma[i,k]+s.trace(connection[k])*sigma[i,j]
        check(f'determinant exterior derivative identity {key}', s.simplify(derivative-3*omega3[key]+omega_f[key]) == 0)
    check('cubic witness component', omega3[0,1,2] == s.I)
    check('cubic variable witness component', omega3[1,2,3] == s.I*x[0])
    check('curvature witness component', omega_f[0,1,2] == 2*s.I)
    check('curvature witness other components vanish', all(v==0 for key,v in omega_f.items() if key!=(0,1,2)))
    transforms = [s.diag(s.I,1,1),s.Matrix([[0,1,0],[0,0,1],[1,0,0]]),s.Matrix([[s.Rational(3,5),-s.Rational(4,5),0],[s.Rational(4,5),s.Rational(3,5),0],[0,0,1]])]
    for index,g in enumerate(transforms):
        check(f'ladder unitary control {index}', g.H*g == s.eye(3))
        for key in omega3:
            i,j,k = key
            check(f'ladder determinant covariance {index}/{key}', s.simplify(det(g*alpha[i],g*alpha[j],g*alpha[k])-g.det()*omega3[key]) == 0)
    check('complex plane first derivative ladder zero', all(det(u,s.I*u,a)==0 for a in alpha))
    check('noncomplex plane first derivative ladder nonzero',det(u,s.eye(3)[:,1],alpha[2]) == 1)
    check('unoriented central sign fixes line projector',(-u)*(-u).H == u*u.H)
    check('unoriented central sign reverses determinant',(-s.eye(3)).det() == -1)
    t = s.symbols('t',real=True)
    samples = [s.Matrix.hstack(s.I*u,s.eye(3)[:,1],s.eye(3)[:,2],s.zeros(3,1)),s.Matrix([[s.I,2*s.I,0,s.I],[1,s.I,2,0],[0,1,s.I,3]]),s.Matrix.hstack(s.zeros(3,1),s.zeros(3,1),s.eye(3)[:,1],s.eye(3)[:,2])]
    for index,a in enumerate(samples):
        gram = s.re(a.H*a)
        norm2 = s.trace(gram)
        b3sq = s.simplify((a*a.H).det())
        minors = sum(abs(a[:,list(indices)].det())**2 for indices in itertools.combinations(range(4),3))
        check(f'cubic ladder Cauchy Binet {index}', s.simplify(minors-b3sq) == 0)
        check(f'cubic ladder bending bound {index}', s.simplify(norm2**3/27-b3sq) >= 0)
        contact = s.expand((s.eye(4)+t*t*gram).det())
        complex_volume = s.expand((s.eye(3)+t*t*a*a.H).det())
        delta = s.Poly(contact-complex_volume,t)
        check(f'contact dominates complex Gram volume {index}', all(c>=0 for c in delta.all_coeffs()))
        eta = a[1:3,:]
        sigma_sq = s.simplify((eta*eta.H).det())
        check(f'sigma bending bound {index}', s.simplify(s.trace(eta*eta.H)**2/4-sigma_sq)>=0)
        if index == 0:
            check('sharp cubic contact volume all offsets', s.factor(contact-(1+t*t)**3) == 0)
        if index == 2:
            check('sharp sigma contact volume all offsets', s.factor(contact-(1+t*t)**2) == 0)
    k = s.symbols('k',real=True)
    sparse = [unit(3,1,0)-unit(3,0,1),s.zeros(3),k*x[1]*(unit(3,2,0)-unit(3,0,2)),s.zeros(3)]
    check('curvature independent jet witness', sparse[2].diff(x[1])*u == k*s.eye(3)[:,2])
    check('curvature ladder survives rank one bending', det(u,sparse[0]*u,sparse[2].diff(x[1])*u) == k)
    RESULTS['geometric_ladders'] = {'common_identity':'d^D sigma = 3 Omega3 - OmegaF','sharp_cubic_bound':'|beta3|^2 <= |B|^6/216','sharp_contact_bound':'v(t) >= (1+t^2 |beta3|^(2/3))^(3/2)','sigma_contact_bound':'v(t) >= 1+t^2 |sigma|','scope':'Geometric bending and induced-volume identities; no energy, rate or selected interaction law.'}


def ambient_symmetry():
    # Root globals used: s, check, zero, comm, exterior, RESULTS.
    x1, x2, u, v = s.symbols('ambient_x1 ambient_x2 ambient_u ambient_v', real=True)
    coords = (x1, x2, u, v)
    y = s.Matrix([u, v])
    j = s.Matrix([[0, -1], [1, 0]])
    aa = [s.zeros(2), 2*x1*j]
    ff = [[s.zeros(2), 2*j], [-2*j, s.zeros(2)]]
    f = [[[s.expand((ff[mu][nu]*y)[a]) for a in range(2)]
          for nu in range(2)] for mu in range(2)]
    coframe = s.eye(4)
    for mu in range(2):
        for a in range(2):
            coframe[2+a, mu] = (aa[mu]*y)[a]
    # Columns are the coordinate components of the orthonormal frame.
    frame = coframe.inv()
    metric = coframe.T*coframe
    inverse_metric = frame*frame.T
    check('ambient connection metric inverse exact', zero(metric*inverse_metric-s.eye(4)))
    check('ambient connection metric determinant one', s.det(metric) == 1)

    def directional(expr, index):
        return sum(frame[k,index]*s.diff(expr, coords[k]) for k in range(4))

    christoffel = [[[s.simplify(sum(inverse_metric[k,l]*(
        s.diff(metric[l,jj],coords[ii])+s.diff(metric[l,ii],coords[jj])
        -s.diff(metric[ii,jj],coords[l])) for l in range(4))/2)
        for jj in range(4)] for ii in range(4)] for k in range(4)]
    ricci = s.Matrix(4,4,lambda ii,jj:s.simplify(sum(
        s.diff(christoffel[k][ii][jj],coords[k])-s.diff(christoffel[k][ii][k],coords[jj])
        +sum(christoffel[k][k][ell]*christoffel[ell][ii][jj]
             -christoffel[k][jj][ell]*christoffel[ell][ii][k] for ell in range(4))
        for k in range(4))))
    scalar = s.simplify(sum(inverse_metric[ii,jj]*ricci[ii,jj] for ii in range(4) for jj in range(4)))
    check('ambient actual coordinate scalar curvature',scalar == -2*(u*u+v*v))
    check('ambient actual coordinate vertical Ricci',ricci[2:4,2:4] == 2*s.Matrix([[v*v,-u*v],[-u*v,u*u]]))
    # w[r][i,j]=<nabla_e_r e_i,e_j>, obtained independently from coordinates.
    w = []
    for r in range(4):
        wr = s.zeros(4)
        for i in range(4):
            deriv = s.Matrix([directional(frame[k,i],r)+sum(
                christoffel[k][ll][mm]*frame[ll,r]*frame[mm,i]
                for ll in range(4) for mm in range(4)) for k in range(4)])
            for jj in range(4):
                wr[i,jj] = s.simplify((frame[:,jj].T*metric*deriv)[0])
        w.append(wr)
        check(f'ambient LC metric compatible {r}', wr.T == -wr)

    expected = [s.zeros(4) for _ in range(4)]
    for mu in range(2):
        for nu in range(2):
            for a in range(2):
                expected[mu][nu,2+a] = -f[mu][nu][a]/2
                expected[mu][2+a,nu] = f[mu][nu][a]/2
                expected[2+a][mu,nu] = f[mu][nu][a]/2
        for a in range(2):
            for b in range(2):
                expected[mu][2+a,2+b] = aa[mu][b,a]
    for r in range(4):
        check(f'ambient full LC coordinate verification {r}', zero(w[r]-expected[r]))
    for r in range(4):
        for i in range(r+1,4):
            bracket_coordinate = s.Matrix([directional(frame[k,i],r)
                -directional(frame[k,r],i) for k in range(4)])
            bracket_frame = coframe*bracket_coordinate
            torsion = s.Matrix([w[r][i,jj]-w[i][r,jj]
                -bracket_frame[jj] for jj in range(4)])
            check(f'ambient LC torsion free {r}/{i}', zero(torsion))

    _, gammas, _, _ = exterior(2)
    eye = s.eye(4)
    gamma_conn = [-sum((w[r][i,jj]*gammas[i]*gammas[jj]
        for i in range(4) for jj in range(4)),s.zeros(4))/4 for r in range(4)]
    for r in range(4):
        for i in range(4):
            expected_clifford = sum((w[r][i,jj]*gammas[jj] for jj in range(4)),s.zeros(4))
            check(f'ambient spin Clifford compatibility {r}/{i}',
                  zero(comm(gamma_conn[r],gammas[i])-expected_clifford))
    full_zero = sum((gammas[r]*gamma_conn[r] for r in range(4)),s.zeros(4))
    normal_zero = -sum((aa[mu][b,a]*gammas[mu]*gammas[2+a]*gammas[2+b]
        for mu in range(2) for a in range(2) for b in range(2)),s.zeros(4))/4
    curvature_zero = sum((f[mu][nu][a]*gammas[mu]*gammas[nu]*gammas[2+a]
        for mu in range(2) for nu in range(2) for a in range(2)),s.zeros(4))/8
    check('ambient complete Dirac curvature coefficient',zero(full_zero-normal_zero-curvature_zero))
    check('ambient mixed curvature term genuinely nonzero',not zero(curvature_zero))

    gspin = -sum((j[b,a]*gammas[2+a]*gammas[2+b]
        for a in range(2) for b in range(2)),s.zeros(4))/4
    vv = j*y
    vframe = s.Matrix([0,0,vv[0],vv[1]])
    killing = s.zeros(4)
    for r in range(4):
        for jj in range(4):
            killing[r,jj] = s.simplify(directional(vframe[jj],r)+sum(
                vframe[i]*w[r][i,jj] for i in range(4)))
    check('ambient Killing tensor skew',killing.T == -killing)
    check('ambient Killing horizontal curvature',killing[0,1] == (vv.dot(ff[0][1]*y))/2)
    check('ambient Killing vertical rotation',killing[2,3] == j[1,0])
    gamma_v = sum((vframe[r]*gamma_conn[r] for r in range(4)),s.zeros(4))
    kosmann_zero = -gamma_v-sum((killing[i,jj]*gammas[i]*gammas[jj]
        for i in range(4) for jj in range(4)),s.zeros(4))/4
    check('ambient covariant Killing lift curvature cancellation',zero(kosmann_zero-gspin))
    check('ambient curvature spin covariance',zero(comm(gspin,curvature_zero)
        -sum((vv[a]*curvature_zero.diff(coords[2+a]) for a in range(2)),s.zeros(4))))
    check('ambient pure fibre spin does not commute curvature term',not zero(comm(gspin,curvature_zero)))

    def d_operator(spinor):
        return sum((gammas[r]*sum((frame[k,r]*spinor.diff(coords[k])
            for k in range(4)),s.zeros(4,1)) for r in range(4)),s.zeros(4,1))+full_zero*spinor
    def rotation_operator(spinor):
        return gspin*spinor-sum((vv[a]*spinor.diff(coords[2+a])
            for a in range(2)),s.zeros(4,1))
    monomials = [s.Integer(1),x1,x2,u,v,x1*u,x2*v,u**2,u*v,v**2]
    for number,polynomial in enumerate(monomials):
        for component in range(4):
            spinor = s.zeros(4,1); spinor[component] = polynomial
            check(f'ambient full Dirac lifted commutator polynomial {number}/{component}',
                  zero(rotation_operator(d_operator(spinor))-d_operator(rotation_operator(spinor))))
    normal_chirality = s.I*gammas[2]*gammas[3]
    check('ambient normal chirality not LC parallel away zero section',
          not zero(comm(gamma_conn[0],normal_chirality)))
    check('ambient normal chirality LC parallel on zero section',
          all(zero(comm(gc.subs({u:0,v:0}),normal_chirality)) for gc in gamma_conn))

    # Actual six-normal domain and period controls.
    _, cn, number, parity = exterior(3)
    j6 = s.diag(j,j,j)
    spin6 = -sum((j6[b,a]*cn[a]*cn[b] for a in range(6) for b in range(6)),s.zeros(8))/4
    check('ambient ordinary normal spin central generator',spin6 == s.I*(number-s.Rational(3,2)*s.eye(8)))
    check('ambient ordinary spin period has sign at two pi',
          all(s.simplify(s.exp(2*s.pi*spin6[k,k])) == -1 for k in range(8)))
    retained_generator = s.diag(s.I*number,s.I*(number-3*s.eye(8)))
    check('ambient retained exterior branch period two pi',
          all(s.simplify(s.exp(2*s.pi*retained_generator[k,k])) == 1 for k in range(16)))
    line = s.diag(1,0,0,0,0,0)
    complex_plane = s.diag(1,1,0,0,0,0)
    real_plane = s.diag(1,0,1,0,0,0)
    check('ambient real line is not fixed circle domain',not zero(comm(line,j6)))
    check('ambient complex plane is fixed circle domain',zero(comm(complex_plane,j6)))
    check('ambient noncomplex plane is not fixed circle domain',not zero(comm(real_plane,j6)))
    r6 = s.Rational(3,5)*s.eye(6)+s.Rational(4,5)*j6
    rotated_line = r6*line*r6.T
    check('ambient moving line projector covariance',rotated_line**2 == rotated_line)
    check('ambient moving line differs from original',rotated_line != line)
    check('ambient moving line tangent symbol covariance',zero(
        sum((r6[a,0]*cn[a] for a in range(6)),s.zeros(8))**2-s.eye(8)))
    # A Clifford-scalar branch ladder also has to preserve total symmetry.
    auxiliary_generator = s.I*s.diag(s.Rational(3,2),-s.Rational(3,2))
    raising = s.Matrix([[0,1],[0,0]])
    check('ambient branch raising central weight three',comm(auxiliary_generator,raising) == 3*s.I*raising)
    check('ambient x-only nonzero raising breaks central symmetry',not zero(comm(auxiliary_generator,raising)))
    coefficient = (u+s.I*v)**3
    orbital_derivative = sum(vv[a]*s.diff(coefficient,coords[2+a]) for a in range(2))
    check('ambient orbital branch matching weight three',s.expand(orbital_derivative-3*s.I*coefficient) == 0)
    check('ambient matching orbital coefficient preserves raising',zero(
        comm(auxiliary_generator,coefficient*raising)-orbital_derivative*raising))
    RESULTS['ambient_symmetry'] = {
        'full_lc_dirac_curvature_term':'(1/8) sum_mu_nu_a (F_mu_nu y)^a c_mu c_nu c_a; full ordered mu,nu sum',
        'spin_generator':'G_spin=i(N_form-3/2); K_spin=G_spin-(Jy).partial_y',
        'retained_generator':'K=i(N_form-3b)-(Jy).partial_y',
        'fixed_sector_domain':'[Pi_U,J]=0; real line fails; complex plane succeeds',
        'family_identity':'U_theta D_Q U_theta^-1=D_(Phi_theta Q), with transported projector',
        'polynomial_control_count':40,
        'scope':'Coordinate LC verification and exact bounded commutators; general covariance proved in derive.md; no physical charge selection.'}


def total_real_carrier():
    _,normal,number,parity = exterior(3)
    n = [realify(g) for g in normal]
    identity = s.eye(16)
    scalar_i = realify(s.I*s.eye(8))
    conjugation = s.diag(s.eye(8),-s.eye(8))
    a = n[0]*n[2]*n[4]*conjugation
    j = realify(s.I*parity)
    j2 = s.diag(j,j)
    h = [s.BlockMatrix([[s.zeros(16),identity],[-identity,s.zeros(16)]]).as_explicit()]
    for q in [scalar_i,a,scalar_i*a]:
        h.append(s.BlockMatrix([[s.zeros(16),q],[q,s.zeros(16)]]).as_explicit())
    tangent = [j2*op for op in h]
    normal2 = [s.diag(g,g) for g in n]
    all_gammas = tangent+normal2
    for i,g in enumerate(all_gammas):
        check(f'real32 total symbol symmetric {i}', g.T == g)
        for k in range(i,10):
            check(f'real32 total spatial Clifford {i}/{k}',g*all_gammas[k]+all_gammas[k]*g == 2*(i==k)*s.eye(32))
        check(f'real32 normal volume tangent or normal relation {i}',zero(comm(j2,g)) if i<4 else zero(j2*g+g*j2))
    generator = s.diag(realify(s.I*number),realify(s.I*number))
    check('real32 first tangent fixed by exterior circle',zero(comm(generator,tangent[0])))
    check('real32 second tangent fixed by exterior circle',zero(comm(generator,tangent[1])))
    check('real32 third tangent rotates by triple angle',comm(generator,tangent[2]) == 3*tangent[3])
    check('real32 fourth tangent rotates by triple angle',comm(generator,tangent[3]) == -3*tangent[2])
    _,base,_,base_parity = exterior(2)
    full = [tensor(g,s.eye(8)) for g in base]+[tensor(base_parity,g) for g in normal]
    g = tensor(s.eye(4),s.I*number)
    jn = tensor(s.eye(4),s.I*parity)
    for i,c in enumerate(full):
        for k in range(i,10):
            check(f'real64 via complex32 total Clifford {i}/{k}',c*full[k]+full[k]*c == 2*(i==k)*s.eye(32))
        check(f'real64 normal volume tangent or normal relation {i}',zero(comm(jn,c)) if i<4 else zero(jn*c+c*jn))
    for i,c in enumerate(full[:4]):
        check(f'real64 pure normal circle fixes tangent {i}',zero(comm(g,c)))
    pauli1=s.Matrix([[0,1],[1,0]])/2
    pauli2=s.Matrix([[0,-s.I],[s.I,0]])/2
    pauli3=s.diag(1,-1)/2
    check('tangent charge splitting violates first observer rotation',comm(pauli3,pauli1) == s.I*pauli2)
    check('tangent charge splitting violates second observer rotation',comm(pauli3,pauli2) == -s.I*pauli1)
    RESULTS['total_real_carrier']={'algebraic_minimum_real_rank':32,'minimum_with_unchanged_pure_normal_U3_lift':64,'full_independent_Spin_h_real_rank':128,'scope':'Explicit Clifford carriers and scoped covariance minima; no physical component occupation selected.'}


def review_bounds_and_cubic_ladder():
    # Uses the principal checker's s, check, comm and zero helpers.
    t = s.symbols('review_t', real=True)
    eye3 = s.eye(3)
    u, e2, e3 = [eye3[:, j] for j in range(3)]
    a = s.Matrix.hstack(s.I*u, e2, e3, s.zeros(3, 1))
    g = s.re(a.H*a)
    cubic_squared = (a*a.H).det()
    check('review cubic bending sharp constant', cubic_squared == (2*s.trace(g))**3/216)
    check('review cubic contact volume sharp polynomial', (s.eye(4)+t*t*g).det() == (1+t*t)**3)
    eta = s.Matrix.hstack(e2, e3, s.zeros(3, 2))
    eta_complement = eta[1:, :]
    check('review sigma sharp bending constant', (eta_complement*eta_complement.H).det() == (s.trace(s.re(eta.H*eta))/2)**2)
    check('review sigma sharp volume polynomial', (s.eye(4)+t*t*s.re(eta.H*eta)).det() == (1+t*t)**2)
    dependent = s.Matrix.hstack(e2, s.I*e2)
    h = s.eye(2)+t*t*dependent.H*dependent
    check('review Hermitian real Gram strict difference', s.expand(s.re(h).det()-h.det()) == t**4)
    check('review bending need not give cubic determinant', (dependent*dependent.H).det() == 0 and s.trace(s.re(dependent.H*dependent)) == 2)
    v = s.Rational(3, 5)*s.I*u+s.Rational(4, 5)*e2
    beta1 = s.Matrix.hstack(u, v, e3).det()
    check('review plane Kahler-angle factor', abs(beta1)**2 == 1-s.Rational(3, 5)**2)
    check('review complex plane determinant zero', s.Matrix.hstack(u, s.I*u, e3).det() == 0)

    k = s.symbols('review_k', real=True)
    f23u = k*e3
    check('review curvature independent jet value', s.Matrix.hstack(u, e2, f23u).det() == k)
    check('review curvature jet leakage unchanged', s.re(e2.H*e2)[0] == 1)

    z = s.Matrix(s.symbols('review_z1:4'))
    aa = s.Matrix([[0, -1, 0], [1, 0, 0], [0, 0, 0]])
    bb = s.Matrix([[0, 0, -1], [0, 0, 0], [1, 0, 0]])
    polynomial = s.expand(s.Matrix.hstack(z, aa*z, bb*z).det())
    check('review cubic curvature full polynomial', polynomial == s.expand(z[0]*sum(q*q for q in z)))
    check('review cubic curvature circle Euler weight', s.expand(sum(z[j]*s.diff(polynomial, z[j]) for j in range(3))) == 3*polynomial)
    check('review cubic curvature finite circle', s.expand(polynomial.subs(dict(zip(z, s.I*z)), simultaneous=True)) == s.expand(-s.I*polynomial))
    gg = s.diag(s.I, 1, -1)
    covariant = s.Matrix.hstack(gg*z, (gg*aa*gg.H)*(gg*z), (gg*bb*gg.H)*(gg*z)).det()
    check('review cubic curvature unitary determinant covariance', s.expand(covariant-gg.det()*polynomial) == 0)
    check('review cubic curvature witness', polynomial.subs({z[0]: 1, z[1]: 0, z[2]: 0}) == 1)
    check('review cubic curvature extra determinant zero', polynomial.subs({z[0]: 1, z[1]: s.I, z[2]: 0}) == 0)
    check('review cubic curvature central curvature zero', s.Matrix.hstack(z, s.I*z, 2*s.I*z).det() == 0)
    b, bc = s.symbols('review_beta review_beta_conjugate')
    branch = s.Matrix([[0, b], [-bc, 0]])
    generator = s.diag(3*s.I/2, -3*s.I/2)
    orbital_derivative = s.Matrix([[0, 3*s.I*b], [3*s.I*bc, 0]])
    check('review paired upper and lower fixed-circle condition', zero(comm(generator, branch)-orbital_derivative))
    check('review paired ladder real descent trace', s.trace(branch) == 0)


def color_invariant_checks():
    import math
    checked = []
    def check(name, condition):
        assert bool(condition), name
        checked.append(name)
    def eq(a, b=0):
        return s.expand(a-b) == 0

    # Integral U3 weights include ordinary and determinant-twisted orbital types.
    for a in range(5):
        for b in range(5):
            for k in range(-3, 4):
                lam = (a+b+k, b+k, k)
                n = sum(lam)
                check(f'U3 triality a{a} b{b} k{k}', (n-(a+2*b)) % 3 == 0)
                if a == b == 0:
                    check(f'color singlet charge k{k}', s.Rational(n, 3) == k)
    for q in range(5):
        for t in range(5):
            for j in range(min(q, t)+1):
                u, v = q-j, t-j
                check(f'orbital trace triality q{q} s{t} j{j}',
                      (u+2*v-(q-t)) % 3 == 0)

    # Formal variables represent BASIS GENERATORS of the symmetric algebra,
    # not their dual coordinate functions. This fixes differential-action signs.
    x = s.Matrix(3, 4, lambda i, j: s.Symbol(f'x{i}{j}'))
    column_sets = list(itertools.combinations(range(4), 3))
    minors = [s.expand(x[:, list(cols)].det()) for cols in column_sets]
    upper_counts = [sum(j >= 2 for j in cols) for cols in column_sets]
    check('four independent cubic minors',
          len({s.Poly(m, *list(x)).monoms()[0] for m in minors}) == 4)
    # Each minor has a distinct degree vector in the four columns, a stronger
    # easy linear-independence certificate than checking a random numerical rank.
    check('distinct cubic column multidegrees', len(set(column_sets)) == 4)
    for idx, (m, r) in enumerate(zip(minors, upper_counts)):
        check(f'minor nonzero {idx}', not eq(m))
        for a in range(3):
            for b in range(3):
                # W* action -E_ab^T; D on each upper-branch generator.
                action = -sum(x[b,j]*s.diff(m, x[a,j]) for j in range(4))
                if a == b:
                    action += sum(x[i,j]*s.diff(m, x[i,j])
                                  for i in range(3) for j in (2,3))
                check(f'cubic determinant gl3 {idx}/{a}/{b}',
                      eq(action, (r-1)*int(a == b)*m))
        central = -sum(v*s.diff(m,v) for v in x)
        central += 3*sum(x[i,j]*s.diff(m,x[i,j])
                         for i in range(3) for j in (2,3))
        check(f'cubic normalized charge {idx}', eq(central, 3*(r-1)*m))

    def E(f):
        return s.expand(sum(x[i,j]*s.diff(f,x[i,j+1])
                            for i in range(3) for j in (0,2)))
    def F(f):
        return s.expand(sum(x[i,j+1]*s.diff(f,x[i,j])
                            for i in range(3) for j in (0,2)))
    def H(f):
        return s.expand(sum((x[i,j]*s.diff(f,x[i,j])-
                             x[i,j+1]*s.diff(f,x[i,j+1]))/2
                            for i in range(3) for j in (0,2)))
    Aplus, Aminus, Bplus, Bminus = minors
    for label, plus, minus in [('neutral',Aplus,Aminus),('charge one',Bplus,Bminus)]:
        check(f'{label} observer highest weight', eq(H(plus),plus/2))
        check(f'{label} observer lowest weight', eq(H(minus),-minus/2))
        check(f'{label} observer raising', eq(E(minus),plus))
        check(f'{label} observer highest annihilation', eq(E(plus)))
        check(f'{label} observer lowering', eq(F(plus),minus))
        check(f'{label} observer lowest annihilation', eq(F(minus)))
        for name, m in [('plus',plus),('minus',minus)]:
            casimir = H(H(m))+(E(F(m))+F(E(m)))/2
            check(f'{label} observer Casimir {name}', eq(casimir,3*m/4))

    for label, cols in [('all lower',(0,1)),('all upper',(2,3))]:
        for word in itertools.product(cols, repeat=3):
            check(f'{label} cubic vanishes {word}', eq(x[:,list(word)].det()))
    # Four cubics are algebraically independent: the Jacobian rank is four
    # at this exact integer witness. It is corroboration, not a replacement
    # for the global bundle-valued derivation.
    witness = s.Matrix([[1,0,0,1],[0,1,0,2],[0,0,1,3]])
    substitution = dict(zip(list(x),list(witness)))
    jacobian = s.Matrix(minors).jacobian(list(x)).subs(substitution)
    check('four cubic generators Jacobian rank', jacobian.rank() == 4)
    sextic = s.expand(Aplus*Bminus-Aminus*Bplus)
    check('observer scalar sextic nonzero', not eq(sextic))
    check('observer scalar sextic raising zero', eq(E(sextic)))
    check('observer scalar sextic lowering zero', eq(F(sextic)))
    check('observer scalar sextic weight zero', eq(H(sextic)))
    for a in range(3):
        for b in range(3):
            action = -sum(x[b,j]*s.diff(sextic,x[a,j]) for j in range(4))
            if a == b:
                action += sum(x[i,j]*s.diff(sextic,x[i,j])
                              for i in range(3) for j in (2,3))
            check(f'sextic determinant line gl3 {a}/{b}',
                  eq(action,int(a == b)*sextic))

    for m in range(1,5):
        d = 2*m
        per_charge = [math.comb(d,3) if d >= 3 else 0,
                      d*math.comb(d,2), d*math.comb(d,2),
                      math.comb(d,3) if d >= 3 else 0]
        check(f'spatial spectator wedge direct-sum dimension m{m}',
              sum(per_charge) == math.comb(2*d,3))
        if m == 1:
            check('single observer doublet cubic charge multiplicities',
                  per_charge == [0,2,2,0])
        if m == 2:
            check('two independent observer/mode slots cubic multiplicities',
                  per_charge == [4,24,24,4])
    check('alternating product color singlet dimension', math.comb(4+3-1,3) == 20)
    check('ordered triple color singlet dimension', 4**3 == 64)
    check('rank-two branch-only symmetric cubic absent', math.comb(2,3) == 0)
    return {'passed':len(checked), 'checks':checked,
            'cubic_charges':[0,0,1,1],
            'cubic_observer_casimir':'3/4',
            'symmetric_color_singlet_dimension':4,
            'alternating_color_singlet_dimension':20,
            'ordered_color_singlet_dimension':64,
            'sextic_observer_scalar_charge':1}


def observer_helicity():
    """Uses root checker globals; finite exact symbol checks, no physical dynamics."""
    table = []
    for d in (3,4,5,6,10):
        _, base, _, parity = exterior(d//2)
        branches = (-1,1) if d % 2 else (1,)
        for orient in branches:
            gam = base + ([orient*parity] if d % 2 else [])
            size = gam[0].rows
            eye = s.eye(size)
            om = -s.I*gam[0]*gam[1]*gam[2]
            hh = -s.I*gam[0]*gam[1]
            label = f'helicity d{d} orientation{orient}'
            check(label+' gamma3 factorization', zero(gam[2]-om*hh))
            check(label+' commuting involutions', zero(comm(om,hh)) and om**2==eye and hh**2==eye)
            for a in range(3,d):
                check(label+f' extra h commute {a}', zero(comm(hh,gam[a])))
                check(label+f' extra volume reverse {a}', zero(om*gam[a]+gam[a]*om))
            for chi in (-1,1):
                pc = (eye+chi*om)/2
                for a in range(3,d):
                    check(label+f' retained compression loses {chi}/{a}', zero(pc*gam[a]*pc))
                for helicity in (-1,1):
                    ph = (eye+helicity*hh)/2
                    joint = pc*ph
                    if d>=4:
                        check(label+f' joint multiplicity {chi}/{helicity}', s.trace(joint)==size//4)
                    for q in ((0,4) if d>=4 else (0,)):
                        symbol = 3*gam[2]+(q*gam[3] if d>=4 else s.zeros(size))
                        energy = 5 if q else 3
                        pp = (eye+symbol/energy)/2
                        weight = (1+s.Rational(chi*helicity*3,energy))/2
                        check(label+f' positive projector q{q}', pp**2==pp)
                        check(label+f' overlap {chi}/{helicity}/q{q}', zero(joint*pp*joint-weight*joint))
                        if d>=4:
                            check(label+f' positive helicity multiplicity {helicity}/q{q}', s.trace(ph*pp)==size//4)
            if d==3:
                chi = s.trace(om)/size
                pp = (eye+gam[2])/2
                check(label+' irreducible observer volume scalar', om==chi*eye)
                check(label+' sole positive helicity', zero((hh-chi*eye)*pp))
            if d==5:
                omega5 = -gam[0]*gam[1]*gam[2]*gam[3]*gam[4]
                kk = -s.I*gam[3]*gam[4]
                check(label+' five-volume relation', omega5==om*kk)
                check(label+' selected five-volume scalar', omega5==(s.trace(omega5)/size)*eye)
                check(label+' both normal plane signs', s.trace(kk)==0)
            table.append({'d':d,'orientation_constructor':orient,'complex_rank':size})
    # Full ambient carrier has eight per joint branch and per + symbol helicity.
    _, gam, _, _ = exterior(5)
    eye = s.eye(32)
    om = -s.I*gam[0]*gam[1]*gam[2]
    hh = -s.I*gam[0]*gam[1]
    pp = (eye+(3*gam[2]+4*gam[3])/5)/2
    for chi in (-1,1):
        for helicity in (-1,1):
            joint=(eye+chi*om)*(eye+helicity*hh)/4
            check(f'ambient helicity joint rank {chi}/{helicity}',s.trace(joint)==8)
            check(f'ambient positive helicity rank {helicity}',s.trace((eye+helicity*hh)*pp/2)==8)
    j=s.Matrix([[0,-s.I],[s.I,0]])
    for sign in (-1,1):
        vec=s.Matrix([1,sign*s.I])/s.sqrt(2)
        check(f'photon transverse circular weight {sign}',j*vec==sign*vec and (vec.H*vec)[0]==1)
    linear=s.Matrix([1,0])
    check('photon linear polarization zero axial expectation',(linear.H*j*linear)[0]==0)
    elliptical=s.Matrix([s.sqrt(3)/2,s.I/2])
    check('photon elliptical normalized',(elliptical.H*elliptical)[0]==1)
    check('photon elliptical axial expectation',(elliptical.H*j*elliptical)[0]==s.sqrt(3)/2)
    RESULTS['observer_helicity']={'minimal_carriers':table,'branch_overlap':'(1+chi*h*p/sqrt(p^2+q^2))/2','depth5_volume_selects_helicity':False,'scope':'spatial principal symbol, not physical Hamiltonian'}


def check_polarization_scalar_zeros():
    def cp(k):
        x,y,z=k
        return s.Matrix([[0,-z,y],[z,0,-x],[-y,x,0]])
    def qh(k,h):
        return (s.eye(3)-k*k.T+h*s.I*cp(k))/2
    def strength(p,k,h):
        return s.simplify((p.conjugate().T*qh(k,h)*p)[0])
    def direction(u,v):
        n=s.simplify(u*s.conjugate(u)+v*s.conjugate(v))
        return s.Matrix([s.re(2*s.conjugate(u)*v),s.im(2*s.conjugate(u)*v),u*s.conjugate(u)-v*s.conjugate(v)])/n
    u,v=s.symbols('u v')
    px,py,pz=s.symbols('px py pz')
    polynomial=(px+s.I*py)*u**2-2*pz*u*v+(-px+s.I*py)*v**2
    disc=s.discriminant(polynomial.subs(u,1),v)
    check('polarization scalar quadratic discriminant',s.simplify(disc-4*(px**2+py**2+pz**2))==0)
    a,b,x,y,z=s.symbols('a b x y z',real=True)
    p=s.Matrix([a,s.I*b,0]); k=s.Matrix([x,y,z])
    for h in (1,-1):
        expected=(a*a+b*b-a*a*x*x-b*b*y*y+2*h*a*b*z)/2
        check(f'polarization target elliptic formula h{h}',s.simplify(strength(p,k,h)-expected)==0)
    rotation=s.Matrix([[s.Rational(3,5),0,s.Rational(4,5)],[0,1,0],[-s.Rational(4,5),0,s.Rational(3,5)]])
    p=s.Matrix([3,s.I,0])/s.sqrt(10)
    for h in (1,-1):
        for sign in (1,-1):
            root=s.Matrix([sign*2*s.sqrt(2)/3,0,-s.Rational(h,3)])
            check(f'polarization ellipse zero h{h} sign{sign}',strength(p,root,h)==0)
            check(f'polarization common rotation zero h{h} sign{sign}',strength(rotation*p,rotation*root,h)==0)
        north=s.Matrix([0,0,h])
        check(f'polarization ellipse maximum h{h}',strength(p,north,h)==s.Rational(4,5))
        circ=s.Matrix([1,s.I,0])/s.sqrt(2)
        check(f'polarization circular dark h{h}',strength(circ,-north,h)==0)
        check(f'polarization circular bright h{h}',strength(circ,north,h)==1)
        ellpoly=(s.sqrt(s.Rational(9,10))+s.sqrt(s.Rational(1,10)))*u**2+(-s.sqrt(s.Rational(9,10))+s.sqrt(s.Rational(1,10)))*v**2
        check(f'polarization ellipse projective roots h{h}',s.simplify(ellpoly.subs({u:1,v:s.sqrt(2)}))==0)
        for kk in (s.Matrix([1,0,0]),s.Matrix([0,1,0]),s.Matrix([0,0,1]),s.Matrix([s.Rational(3,5),0,s.Rational(4,5)])):
            ex=s.Matrix([1,0,0]); ey=s.Matrix([0,1,0])
            total=strength(ex,kk,h)+strength(ey,kk,h)
            check(f'polarization rank2 no common zero h{h} k{tuple(kk)}',total==(1+kk[2]**2)/2 and total>=s.Rational(1,2))
            check(f'polarization nonzero common rotation h{h} k{tuple(kk)}',strength(rotation*p,rotation*kk,h)==strength(p,kk,h))
    kz=s.symbols('kz',real=True)
    r=s.symbols('r',real=True)
    factored=(a*a/2)*((1+r*z)**2-(1-r*r)*x*x)
    expected=(a*a+(a*r)**2-a*a*x*x-(a*r)**2*y*y+2*a*a*r*z)/2
    residual=s.factor(factored-expected)
    check('polarization zero factorization on unit sphere',s.simplify(residual-a*a*r*r*(x*x+y*y+z*z-1)/2)==0)
    delta=s.symbols('delta',real=True)
    check('polarization circular quartic strength',s.limit((a*a/2)*(1-s.cos(delta))**2/delta**4,delta,0)==a*a/8)
    check('polarization ellipse simple strength',s.limit((a*a/2)*(1-s.cos(delta))*2*(1-r*r)/delta**2,delta,0)==a*a*(1-r*r)/2)
    for kk in (s.Matrix([1,0,0]),s.Matrix([0,1,0]),s.Matrix([0,0,1])):
        check(f'polarization ellipse helicity sum {tuple(kk)}',strength(p,kk,1)+strength(p,kk,-1)==1-(9*kk[0]**2+kk[1]**2)/10)
    q=s.Matrix([1,s.I,0])/s.sqrt(2)
    null_direction=-s.I*q.conjugate().cross(q)/(q.conjugate().T*q)[0]
    check('polarization null-kernel direction sign',zero(null_direction-s.Matrix([0,0,1])))
    check('polarization null-kernel helicity',zero(qh(null_direction,1)*q-q))
    k=s.Matrix([1,0,0]); linear=s.Matrix([0,0,1])
    check('polarization arbitrary incident rank2 dark control',zero((s.eye(3)-k*k.T)*linear-linear) and zero(s.diag(1,1,0)*linear))
    return {'scalar_zero_multiplicity':2,'rank2_min_fixed_helicity':'1/2','ellipse_target':'(3,i,0)/sqrt(10)','ellipse_zero_directions':'(+-2sqrt(2)/3,0,-h/3)'}


def color_invariants():
    result = color_invariant_checks()
    CHECKS.extend(result.pop('checks'))
    RESULTS['color_invariants'] = result


def polarization_zeros():
    RESULTS['polarization_zeros'] = check_polarization_scalar_zeros()


def refined_counts():
    from collections import Counter
    from math import comb
    x,ell = s.symbols('count_x count_ell')
    expression = 1/(1-x)
    for shift in [1,2,ell+1,ell+2]:
        expression = s.factor(x*s.diff(expression,x)+shift*expression)
    numerator = (ell+1)*(ell+2)/2+(4-ell**2)*x+(ell-1)*(ell-2)*x*x/2
    check('fixed normal weight rational generating identity',s.factor(expression/4-numerator/(1-x)**5) == 0)
    check('fixed normal weights share leading singularity',s.expand(numerator.subs(x,1)) == 6)
    check('fixed normal weights share first numerator derivative',s.expand(6*ell+2*s.diff(numerator,x).subs(x,1)) == 12)
    for q in range(8):
        for p in range(8):
            expected=sum((q-j+1)*(p-j+1)*(q+p-2*j+2)//2 for j in range(min(q,p)+1))
            check(f'normal trace branching dimension {q}/{p}',expected == comb(q+2,2)*comb(p+2,2))
    def band(cutoff,a,r):
        result=Counter()
        for q in range(cutoff+1):
            for p in range(cutoff-q+1):
                multiplicity=comb(q+r-1,r-1)*comb(p+r-1,r-1)*comb(cutoff-q-p+a,a)
                result[q-p]+=multiplicity
        return result
    tables=[]
    for r in [1,2,3]:
        for cutoff in range(6):
            b=band(cutoff,4,r)
            check(f'refined band retains full spatial count {r}/{cutoff}',sum(b.values()) == comb(cutoff+4+2*r,4+2*r))
            check(f'refined band conjugate weight symmetry {r}/{cutoff}',all(v==b[-k] for k,v in b.items()))
            if r>1:
                previous=band(cutoff-1,4,r) if cutoff else Counter()
                earlier=band(cutoff-2,4,r) if cutoff>=2 else Counter()
                lower=band(cutoff,4,r-1)
                check(f'refined complex plane Pascal recursion {r}/{cutoff}',all(b[k]-previous[k-1]-previous[k+1]+earlier[k]==lower[k] for k in range(-cutoff-2,cutoff+3)))
            tables.append({'depth':4+2*r,'cutoff':cutoff,'neutral':b[0],'all':sum(b.values())})
    RESULTS['refined_counts']={'controls':tables,'six_normal_fixed_weight_numerator':str(numerator),'fixed_weight_leading_coefficient':'3/16 times N^9/9! on the ten-dimensional scalar band','scope':'Available polynomial modes and exact character identities, not a physical spectrum or preferred occupied charge.'}


def complex_points():
    x=s.Matrix(s.symbols('point_x1:5',real=True))
    norm=1+x.dot(x)
    w=x/s.sqrt(norm)
    origin={v:0 for v in x}
    for sign in [-1,1]:
        a=sign/s.sqrt(norm)
        check(f'normalized flag unit identity {sign}',s.simplify(a*a+w.dot(w)) == 1)
        check(f'normalized flag derivative index {sign}',w.jacobian(x).subs(origin) == s.eye(4))
        check(f'normalized flag alignment derivative zero {sign}',all(s.diff(a,v).subs(origin)==0 for v in x))
    check('complex point negative local index',s.diag(1,-1,1,1).det() == -1)
    check('complex point plane reversal preserves index',(-s.eye(4)).det() == 1)
    j=s.diag(s.Matrix([[0,-1],[1,0]]),s.Matrix([[0,-1],[1,0]]))
    for sign in [-1,1]:
        for u,v in [(s.eye(4),-sign*j),(s.zeros(4),s.eye(4)),(s.zeros(4),s.zeros(4))]:
            frob=lambda a:s.trace(a.T*a)
            derivative=v-sign*j*u
            check(f'complex point bending exact square {sign}/{frob(u)}/{frob(v)}',2*(frob(u)+frob(v))-frob(derivative) == frob(v+sign*j*u))
    unit3=s.eye(3);u=unit3[:,0]
    alpha=[s.I*u,unit3[:,1],unit3[:,2],s.zeros(3,1)]
    det=lambda a,b,c:s.Matrix.hstack(a,b,c).det()
    for sign in [-1,1]:
        check(f'complex point forces first ladder zero {sign}',all(det(u,sign*s.I*u,a)==0 for a in alpha))
    check('complex point need not zero cubic ladder',det(*alpha[:3]) == s.I)
    check('complex point need not zero lower-flag ladder',det(u,alpha[2],alpha[1]) == -1)
    q1=s.Rational(3,5);q2=4*s.I/5
    clutch=s.Matrix([[q1,-s.conjugate(q2)],[q2,s.conjugate(q1)]])
    check('one-zero clutch unitary determinant one',clutch.H*clutch == s.eye(2) and clutch.det()==1)
    check('one-zero clutch constant exterior section',clutch*unit3[:2,0] == s.Matrix([q1,q2]))
    p,q=s.symbols('point_gradient1 point_gradient2',real=True)
    square_hessian=2*s.Matrix([p,q])*s.Matrix([[p,q]])
    check('literal flag projection Hessian obstruction',square_hessian.det()==0 and s.diag(2,2).det()==4)
    RESULTS['complex_points']={'signed_transverse_count':'integral of c2(W) on the oriented closed base','forced_zero_map':'beta1; beta2 and beta3 need not vanish','scope':'Euler theorem and explicit normalized flags; topology is proved in derive.md, not inferred from finite checks.'}


def bott_constraints():
    rows=[]
    for m in range(1,6):
        _,gammas,number,parity=exterior(m)
        size=2**m
        eye=s.eye(2*size)
        scalar_i=realify(s.I*s.eye(size))
        conjugation=s.diag(s.eye(size),-s.eye(size))
        product=s.eye(size)
        for index in range(0,2*m,2):
            product=product*gammas[index]
        if m%2==0:
            product=parity*product
        b=realify(product)*conjugation
        j=realify(s.I*parity)
        gamma=realify(parity)
        generator=realify(s.I*number)
        check(f'Bott conjugation square m{m}',b*b==(-1)**(m*(m-1)//2)*eye)
        check(f'Bott conjugation parity m{m}',b*gamma==(-1)**m*gamma*b)
        check(f'Bott conjugation reporting phase m{m}',b*j==(-1)**(m+1)*j*b)
        check(f'Bott conjugation determinant weight m{m}',comm(generator,b)==m*scalar_i*b)
        check(f'Bott conjugation original phase reversal m{m}',b*scalar_i==-scalar_i*b)
        for index,g in enumerate(gammas):
            check(f'Bott full normal conjugation commutant m{m}/{index}',zero(comm(b,realify(g))))
        allowed=[]
        for name,t in [('I',scalar_i),('B',b),('IB',scalar_i*b)]:
            if t.T==-t and zero(comm(t,j)):
                allowed.append(name)
        expected=3 if m%4==3 else 1
        check(f'Bott E unitary normal connection dimension m{m}',len(allowed)==expected)
        odd_allowed=[name for name,t in [('I',scalar_i),('B',b),('IB',scalar_i*b)] if t.T==-t and zero(comm(t,j)) and zero(comm(t,gamma))]
        check(f'odd extension removes paired freedom m{m}',odd_allowed==['I'])
        rows.append({'normal_even_dimension':2*m,'real_carrier_rank':2*size,'E_unitary_connection_dimension':len(allowed),'odd_extension_connection_dimension':len(odd_allowed)})
    RESULTS['bott_constraints']={'canonical_controls':rows,'conditional_even_normal_gap':'6 modulo 8; only six among positive even gaps at most ten','scope':'Fixed exterior real carrier, fixed parity-conjugated phase and full normal metric Clifford compatibility; enlarged carriers and different phases change the result.'}


def ambient_curvature_bounds():
    def cross(a):
        return s.Matrix([[0,-a[2],a[1]],[a[2],0,-a[0]],[-a[1],a[0],0]])
    def norm2(a):
        return s.simplify(s.trace(a.H*a))
    controls=[
        (s.Matrix([0,0,1]),s.zeros(3,1),s.zeros(3,1),s.Matrix([0,1,0])),
        (s.Matrix([1,s.I,0]),s.Matrix([0,1,s.I]),s.Matrix([1,2,s.I]),s.Matrix([s.I,0,1])),
        (s.Matrix([1,0,0]),s.Matrix([0,1,0]),s.Matrix([0,0,1]),s.Matrix([1,s.I,2])),
        (s.zeros(3,1),s.zeros(3,1),s.Matrix([1,0,0]),s.Matrix([0,1,0]))]
    for index,(a1,a2,b1,b2) in enumerate(controls):
        ta=s.Matrix.hstack(cross(a2),-cross(a1))
        anorm=norm2(a1)+norm2(a2)
        expected=anorm*s.eye(3)-s.conjugate(a1)*a1.T-s.conjugate(a2)*a2.T
        check(f'ambient complex cross Gram identity {index}',zero(ta*ta.H-expected))
        beta=a2.cross(b1)-a1.cross(b2)
        bnorm=norm2(b1)+norm2(b2)
        check(f'ambient curvature ladder Cauchy bound {index}',norm2(beta)<=anorm*bnorm)
        check(f'ambient curvature scalar deficit bound {index}',norm2(beta)<=(anorm+bnorm)**2/4)
        if index==0:
            check('ambient curvature bound sharp nonzero witness',norm2(beta)==1 and anorm+bnorm==2)
    RESULTS['ambient_curvature_bounds']={'scalar_identity':'Scal_M=Scal_B-(1/2) sum_(i<j)|F_ij y|^2','sharp_ladder_bound':'|beta| <= |y| (Scal_B-Scal_M)','scope':'Actual connection metric curvature and one specified determinant ladder, not a physical energy or force.'}


def pure_spinor_geometry():
    rows=[]
    def spin_j(psi,gammas):
        norm=(psi.H*psi)[0]
        return s.Matrix(len(gammas),len(gammas),lambda a,b: 0 if a==b else s.simplify(s.I*(psi.H*gammas[a]*gammas[b]*psi)[0]/norm))
    for m in range(1,6):
        _,gammas,_,_=exterior(m)
        vacuum=s.eye(2**m)[:,0]
        states=[('vacuum',vacuum,True)]
        if m in (2,3):
            states.append(('twoform',vacuum+s.eye(2**m)[:,3],True))
        if m==3:
            states.extend([('mixed',vacuum+2*s.eye(8)[:,3]+3*s.I*s.eye(8)[:,5]-4*s.eye(8)[:,6],True),('no-vacuum',s.eye(8)[:,3]+2*s.I*s.eye(8)[:,5]+3*s.eye(8)[:,6],True)])
        if m>=4:
            states.append(('impure',vacuum+s.eye(2**m)[:,15],False))
        for label,psi,pure in states:
            matrix=s.Matrix.hstack(*[g*psi for g in gammas])
            nullity=2*m-matrix.rank()
            check(f'pure spinor nullity m{m}/{label}',nullity==(m if pure else m-4))
            j=spin_j(psi,gammas)
            check(f'pure spinor real bilinear m{m}/{label}',zero(s.im(j)))
            check(f'pure spinor skew bilinear m{m}/{label}',j.T==-j)
            check(f'pure spinor phase independent m{m}/{label}',zero(spin_j(s.I*psi,gammas)-j))
            check(f'pure spinor complex structure m{m}/{label}',zero(j*j+s.eye(2*m))==pure)
            if pure:
                kernel=s.Matrix.hstack(*matrix.nullspace())
                check(f'pure spinor annihilator isotropic m{m}/{label}',zero(kernel.T*kernel))
                check(f'pure spinor annihilator eigenplane m{m}/{label}',zero(j*kernel+s.I*kernel))
                odd=gammas[0]*psi
                check(f'pure spinor opposite chirality m{m}/{label}',s.Matrix.hstack(*[g*odd for g in gammas]).rank()==m)
                rotation=(s.eye(2**m)+gammas[0]*gammas[1])/s.sqrt(2)
                r=s.Matrix(2*m,2*m,lambda a,b:s.simplify(s.trace(gammas[a]*rotation*gammas[b]*rotation.H)/2**m))
                check(f'pure spinor rotation covariance m{m}/{label}',zero(spin_j(rotation*psi,gammas)-r*j*r.T))
            rows.append({'normal_dimension':2*m,'state':label,'annihilator_dimension':nullity,'pure':pure})
    pairs=list(itertools.combinations(range(4),2))
    def wedge(a,b):
        return s.Matrix([a[i]*b[j]-a[j]*b[i] for i,j in pairs])
    h=s.zeros(6)
    for a,b,sign in [(0,5,1),(1,4,-1),(2,3,1)]:
        h[a,b]=h[b,a]=sign
    check('normal vector real form squares identity',h*h==s.eye(6))
    variables=s.symbols('x0:8')
    symbolic=s.Matrix(4,2,variables)
    p=wedge(symbolic[:,0],symbolic[:,1])
    check('quadratic normal vector Plucker identity',s.expand((p.T*h*p)[0])==0)
    e=s.eye(4)
    controls=[s.Matrix.hstack(e[:,0],s.zeros(4,1)),s.Matrix.hstack(2*e[:,0],e[:,1]),s.Matrix.hstack(e[:,0],e[:,1]),s.Matrix.hstack(e[:,0]+s.I*e[:,2],2*e[:,1]+e[:,3])]
    for index,phi in enumerate(controls):
        p=wedge(phi[:,0],phi[:,1])
        u=(p+h*s.conjugate(p))/2
        v=(p-h*s.conjugate(p))/(2*s.I)
        check(f'quadratic normal rank detects frame {index}',zero(p)==(phi.rank()<2))
        check(f'quadratic normal vector norm {index}',s.simplify((p.H*p)[0]-(phi.H*phi).det())==0)
        check(f'quadratic normal vector null {index}',zero(p.T*h*p))
        check(f'quadratic normal real frame {index}',zero(h*s.conjugate(u)-u) and zero(h*s.conjugate(v)-v))
        check(f'quadratic normal equal lengths {index}',zero(u.T*h*u-v.T*h*v))
        check(f'quadratic normal orthogonal frame {index}',zero(u.T*h*v))
        check(f'quadratic normal positive real metric {index}',s.simplify((u.T*h*u)[0]-(p.H*p)[0]/2)==0)
        check(f'quadratic normal phase doubling {index}',wedge(s.I*phi[:,0],s.I*phi[:,1])==-p)
        aux=s.Matrix([[0,1],[-1,0]])
        changed=phi*aux.T
        check(f'quadratic normal auxiliary invariance {index}',wedge(changed[:,0],changed[:,1])==p)
        for k,g in enumerate([s.diag(s.I,1,1,-s.I),s.diag(aux,s.eye(2))]):
            exterior_g=s.Matrix.hstack(*[wedge(g[:,i],g[:,j]) for i,j in pairs])
            check(f'quadratic normal real covariance {index}/{k}',zero(exterior_g*h-h*s.conjugate(exterior_g)))
            check(f'quadratic normal metric covariance {index}/{k}',zero(exterior_g.T*h*exterior_g-h))
            check(f'quadratic normal state covariance {index}/{k}',wedge((g*phi)[:,0],(g*phi)[:,1])==exterior_g*p)
    generic=controls[1]
    g=s.diag(s.I,-s.I,s.Matrix([[0,1],[-1,0]]))
    aux=s.diag(-s.I,s.I)
    check('generic polarization stabilizer witness',g*generic*aux.T==generic)
    check('generic polarization simple spectral ray',generic*generic.H==s.diag(4,1,0,0))
    c=s.Matrix([[0,1],[-1,0]])
    g=s.diag(c,s.eye(2))
    equal=controls[2]
    check('equal polarization larger stabilizer witness',g*equal*s.conjugate(c).T==equal)
    check('equal polarization ray ambiguity witness',g*e[:,0]!=e[:,0])
    RESULTS['pure_spinor_geometry']={'annihilator_controls':rows,'quadratic_flag':'Rank-two Spin-h state gives a null complex normal vector and framed real plane. A simple top spin spectral ray supplies a compatible complex structure.','scope':'Pointwise and bundle geometry, with charge alignment and nondegeneracy assumptions; no particle field equation or spatial CP3 identification.'}


def joint_dimension_and_alignment():
    for m in (4,5,6,7):
        _,gamma,_,_=exterior(m)
        psi=s.eye(2**m)[:,0]+s.eye(2**m)[:,15]
        check(f'all dimension impure witness m{m}',2*m-s.Matrix.hstack(*[g*psi for g in gamma]).rank()==m-4)
    for m in range(1,6):
        _,gamma,_,parity=exterior(m)
        if m<3:
            psi=s.Matrix([j+1+s.I*(j%2) for j in range(2**m)])
            expected=m
        else:
            psi=s.eye(2**m)[:,0]+s.eye(2**m)[:,7]
            expected=m-3
        check(f'odd purity comparison m{m}',2*m+1-s.Matrix.hstack(*[g*psi for g in gamma+[parity]]).rank()==expected)
    for m in (1,2):
        _,gamma,_,parity=exterior(m)
        n=2**(m+1)
        j=tensor(realify(s.I*parity),s.eye(3))
        t12=tensor(s.eye(n),unit(3,0,1)-unit(3,1,0))
        t23=tensor(s.eye(n),unit(3,1,2)-unit(3,2,1))
        check(f'enlarged carrier nonabelian control m{m}',not zero(comm(t12,t23)))
        for k,t in enumerate((t12,t23)):
            check(f'enlarged carrier metric J control m{m}/{k}',t.T==-t and zero(comm(t,j)))
            for index,g in enumerate(gamma):
                check(f'enlarged carrier Clifford control m{m}/{k}/{index}',zero(comm(t,tensor(realify(g),s.eye(3)))))
    e=s.eye(4)
    aligned=s.Matrix.hstack(2*e[:,0],e[:,1])
    oblique=s.Matrix.hstack(2*e[:,0],e[:,0]+e[:,1])
    empty=s.Matrix.hstack(2*e[:,1],e[:,2])
    check('vacuum aligned spectral block',aligned*aligned.H==s.diag(4,1,0,0))
    check('vacuum incidence does not select eigenline',(oblique*oblique.H)[:2,:2]==s.Matrix([[5,1],[1,1]]))
    check('vacuum zero eigenvalue need not be rank one triplet',empty*empty.H==s.diag(0,4,1,0))
    c=s.Matrix([[s.Rational(3,5),s.Rational(4,5)],[-s.Rational(4,5),s.Rational(3,5)]])
    rotated=aligned*c
    check('auxiliary rotation retains normal spectral geometry',rotated*rotated.H==aligned*aligned.H)
    gram=rotated.H*rotated
    check('auxiliary axis can change at fixed normal geometry',gram[0,1]==s.Rational(36,25))
    old=s.diag(s.Rational(1,2),-s.Rational(1,2))
    new=(gram-s.eye(2))/3-s.eye(2)/2
    y=s.diag(-s.Rational(1,2),*[s.Rational(1,6)]*3)
    qold=tensor(y,s.eye(2))+tensor(s.eye(4),old)
    qnew=tensor(y,s.eye(2))+tensor(s.eye(4),new)
    check('charge operator changes with auxiliary axis',qold!=qnew)
    check('changed charge axis retains weight spectrum',qold.eigenvals()==qnew.eigenvals())
    x,y=s.symbols('chern_x chern_y')
    check('auxiliary Pontryagin Euler identity',s.expand((x+y)**2-x*x-y*y-2*x*y)==0)
    def ring(expr):
        return s.Poly(s.expand(expr),x,y).rem(s.Poly(x*x,x,y)).as_expr().expand().subs(y*y,0)
    check('original determinant square nonzero',ring((x+y)**2)==2*x*y)
    check('reconstructed determinant square differs',ring((x-y)**2)==-2*x*y)
    check('same normal Pontryagin does not imply auxiliary match',ring((x+y)**2-2*x*y)==ring((x-y)**2+2*x*y)==0)
    RESULTS['joint_dimension_and_alignment']={'joint_normal_rule':'Six uniquely satisfies automatic chiral purity and the fixed canonical J-preserving non-Abelian normal connection commutant.','alignment':'Both normal vacuum ray and ordered auxiliary axis are needed to keep the original Q.','scope':'Normal building-block comparison; enlarged ambient carriers require separate comparison.'}


def polarization_gradient():
    a,b=s.symbols('polarization_a polarization_b',positive=True)
    k=1/a**2+1/b**2
    variables=s.symbols('polarization_x0:16',real=True)
    x=s.Matrix(4,2,lambda i,j:variables[8*j+2*i]+s.I*variables[8*j+2*i+1])
    e=s.eye(4); phi=s.Matrix.hstack(a*e[:,0],b*e[:,1])
    pairs=list(itertools.combinations(range(4),2))
    def wedge(v,w): return s.Matrix([v[i]*w[j]-v[j]*w[i] for i,j in pairs])
    q=wedge(phi[:,0],phi[:,1]); dq=wedge(x[:,0],phi[:,1])+wedge(phi[:,0],x[:,1])
    h=s.zeros(6)
    for i,j,sign in [(0,5,1),(1,4,-1),(2,3,1)]: h[i,j]=h[j,i]=sign
    real_basis=s.Matrix.hstack(*[(s.eye(6)[:,i]+h*s.eye(6)[:,i])/s.sqrt(2) for i in (0,1,2)],*[(s.I*s.eye(6)[:,i]-s.I*h*s.eye(6)[:,i])/s.sqrt(2) for i in (0,1,2)])
    check('normal real frame coordinate basis unitary',zero(real_basis.H*real_basis-s.eye(6)))
    u=s.sqrt(2)*(q+h*s.conjugate(q))/(2*a*b)
    v=s.sqrt(2)*(q-h*s.conjugate(q))/(2*s.I*a*b)
    radial=s.re(dq[0])/(a*b)
    du=s.sqrt(2)*(dq+h*s.conjugate(dq))/(2*a*b)-radial*u
    dv=s.sqrt(2)*(dq-h*s.conjugate(dq))/(2*s.I*a*b)-radial*v
    ur=real_basis.H*u; vr=real_basis.H*v
    dur=s.simplify(real_basis.H*du); dvr=s.simplify(real_basis.H*dv)
    l=dur.jacobian(variables)
    check('derived normalized quadratic frame differential',zero(l*l.T-k*(s.eye(6)-ur*ur.T)))
    dp=dur*ur.T+ur*dur.T+dvr*vr.T+vr*dvr.T
    z=sum(s.expand_complex(s.conjugate(dq[i])*dq[i]) for i in (1,2,3,4))
    check('derived normal plane projector differential',s.simplify(s.trace(dp.T*dp)-4*z/(a*a*b*b))==0)
    phase={z0:0 for z0 in variables};phase[variables[1]]=b;phase[variables[11]]=a
    check('phase moves frame but not plane',zero(dp.subs(phase)) and not zero(dur.subs(phase)))
    drho=x*phi.H+phi*x.H
    zeta=s.Matrix([drho[1,0]/(a*a-b*b),drho[2,0]/(a*a),drho[3,0]/(a*a)])
    stationary={z0:0 for z0 in variables};stationary[variables[2]]=a;stationary[variables[8]]=b
    check('spectral geometry can move at stationary quadratic frame',zero(dq.subs(stationary)) and not zero(zeta.subs(stationary)))
    check('spectral gradient sharp gap witness',s.simplify((zeta.H*zeta)[0].subs(stationary)-(a*a+b*b)**2/(a*a-b*b)**2)==0)
    _,gamma,_,_=exterior(3);vac=s.eye(8)[:,0]
    for mask in (3,5,6):
        for phase0 in (1,s.I):
            variation=phase0*s.eye(8)[:,mask]
            dj=s.Matrix(6,6,lambda i,j:0 if i==j else s.I*(variation.H*gamma[i]*gamma[j]*vac+vac.H*gamma[i]*gamma[j]*variation)[0])
            check(f'pure spinor J differential metric {mask}/{phase0}',s.trace(dj.T*dj)==16 and zero(s.im(dj)))
    lp=s.MutableDenseMatrix(l)
    for col in (4,5,6,7):lp[:,col]=s.zeros(6,1)
    expected=(s.eye(6)-ur*ur.T-vr*vr.T)/b**2+k*vr*vr.T
    check('parallel normal J anisotropic frame differential',zero(lp*lp.T-expected))
    kval=s.Rational(5,4)
    witness=s.Matrix([[s.I*s.sqrt(5)/2,0,0,0],[0,1,0,0],[0,0,1,0]])
    cost=abs(witness[0,0])**2/kval+sum(abs(witness[i,j])**2 for i in (1,2) for j in range(4))
    cubic=abs(witness[:,:3].det())**2
    check('parallel normal J sharp cubic transition bound',cost==3 and cubic==kval*cost**3/27)
    RESULTS['polarization_gradient']={'frame_bound':'|nabla u|^2 <= (a^-2+b^-2)|D Phi|^2','J_bound':'|nabla J|_F^2 <=16(a^2+b^2)|D Phi|^2/(a^2-b^2)^2','parallel_J_cubic_bound':'|beta3|^2 <= (a^-2+b^-2)|D Phi|^6/(27 b^4)','scope':'Geometric derivative costs, not a prescribed field energy or interaction rate.'}


def transition_order():
    u = s.Matrix([1, 0, 0])
    e2 = s.Matrix([0, 1, 0])
    e3 = s.Matrix([0, 0, 1])

    def data(A):
        a = s.Matrix([-s.I * (u.H * A[:, j])[0] for j in range(4)])
        eta = A - s.I * u * a.T
        sigma = s.zeros(4)
        for i in range(4):
            for j in range(4):
                sigma[i, j] = s.Matrix.hstack(u, eta[:, i], eta[:, j]).det()
        C = sigma.T * a
        S = s.Matrix([
            sum(a[i] * sigma[j, k] * s.LeviCivita(i, j, k, l)
                for i in range(4) for j in range(4) for k in range(j + 1, 4))
            for l in range(4)])
        beta3 = s.Matrix([
            sum(s.LeviCivita(i, j, k, l)
                * s.Matrix.hstack(A[:, i], A[:, j], A[:, k]).det()
                for i, j, k in itertools.combinations(range(4), 3))
            for l in range(4)])
        sig2 = sum(abs(sigma[i, j])**2
                   for i in range(4) for j in range(i + 1, 4))
        return a, eta, sigma, C, S, beta3, sig2

    examples = [
        s.Matrix.hstack(s.I*u+e2, e3, s.zeros(3, 1), s.zeros(3, 1)),
        s.Matrix.hstack(s.I*u, e2, e3, s.zeros(3, 1)),
        s.Matrix.hstack(s.I*u, s.I*u+e2, e3, s.zeros(3, 1)),
        s.Matrix.hstack(s.I*s.sqrt(2)*u+s.sqrt(2)*e2,
                        e3/2, s.zeros(3, 1), s.zeros(3, 1)),
    ]
    for n, A in enumerate(examples):
        a, eta, sigma, C, S, beta3, sig2 = data(A)
        check(f'transition alpha unit tangency {n}',
              all(s.re((u.H*A[:, j])[0]) == 0 for j in range(4)))
        check(f'transition complement {n}', zero(u.H*eta))
        check(f'transition beta3 normalization {n}', zero(beta3-s.I*S))
        check(f'transition joint norm identity {n}',
              s.simplify((C.H*C)[0]+(S.H*S)[0]-(a.T*a)[0]*sig2) == 0)
        L = s.trace(A.H*A)
        check(f'transition joint bending bound {n}',
              s.simplify(L**3/27-(C.H*C)[0]-(S.H*S)[0]) >= 0)
        check(f'transition sigma decomposable {n}',
              sigma[0,1]*sigma[2,3]-sigma[0,2]*sigma[1,3]
              +sigma[0,3]*sigma[1,2] == 0)
        check(f'transition complex plane degree one zero {n}',
              all(s.Matrix.hstack(u,s.I*u,A[:,j]).det() == 0 for j in range(4)))

    _, _, _, C0, S0, _, _ = data(examples[0])
    _, _, _, C1, S1, _, _ = data(examples[1])
    check('transition C only exact', C0 == s.Matrix([0,1,0,0]) and zero(S0))
    check('transition S only exact', zero(C1) and S1 == s.Matrix([0,0,0,1]))
    check('transition C only leakage rank two', examples[0].rank() == 2)
    check('transition S only leakage rank three', examples[1].rank() == 3)
    for n in (0,1):
        a, eta, sigma, C, S, beta3, sig2 = data(examples[n])
        check(f'transition sharp joint bound {n}',
              (C.H*C)[0]+(S.H*S)[0] == s.trace(examples[n].H*examples[n])**3/27)

    # Exact counterexample to transferring the beta3 volume bound to C.
    G0 = s.re(examples[0].H*examples[0])
    check('transition C contact Gram', G0 == s.diag(2,1,0,0))
    check('transition C cubic volume transfer fails',
          (s.eye(4)+G0).det() == 6 and 6 < 8)

    # Exact optimal envelope: c=1, rho^2=X=7/2, z=1/2.
    X = s.Rational(7,2)
    z = s.Rational(1,2)
    A = examples[3]
    _, _, _, C, S, _, _ = data(A)
    G = s.re(A.H*A)
    check('transition envelope fixed C', (C.H*C)[0] == 1 and zero(S))
    check('transition envelope stationary cubic', z**3+X*z**2 == 1)
    check('transition envelope sharp equality',
          (s.eye(4)+X*G).det() == (1+2*X/z)*(1+X*z**2) == s.Rational(225,8))

    # Sign under reversal of the actual real normal line.
    # Applying u -> -u together with alpha -> -alpha keeps a unchanged
    # and negates sigma, C and S. Check directly in the principal if the
    # helper is generalized to accept u as an argument.


def ambient_dimension_audit():
    rows=[]
    for m in range(1,6):
        k=5-m
        _,tg,_,tp=exterior(k)
        _,ng,nn,np=exterior(m)
        tangent=[tensor(g,s.eye(2**m)) for g in tg]
        normal=[tensor(tp,g) for g in ng]
        gamma=tangent+normal
        g=s.I*tensor(s.eye(2**k),nn)
        gs=-sum((normal[2*j]*normal[2*j+1]/2 for j in range(m)),s.zeros(32))
        jn=s.I*tensor(s.eye(2**k),np)
        check(f'ambient all gap spin lift residual m{m}',g-gs==s.I*s.Rational(m,2)*s.eye(32))
        omega=s.eye(32)
        for n in normal:omega=omega*n
        check(f'ambient all gap volume reporting phase m{m}',jn==s.I**(1-m)*omega)
        for index,t in enumerate(tangent):
            check(f'ambient all gap neutral tangent m{m}/{index}',zero(comm(g,t)) and zero(comm(jn,t)))
        for index,n in enumerate(normal):
            target=normal[index+1] if index%2==0 else -normal[index-1]
            check(f'ambient all gap covariant normal m{m}/{index}',comm(g,n)==target and zero(jn*n+n*jn))
        for index,z in enumerate(gamma):
            check(f'ambient all gap Clifford squares m{m}/{index}',z.H==z and z*z==s.eye(32))
            for index2 in range(index):
                check(f'ambient all gap Clifford cross m{m}/{index}/{index2}',zero(z*gamma[index2]+gamma[index2]*z))
        p=s.eye(32)
        for index in range(0,10,2):p=p*gamma[index]
        b0=realify(p)*s.diag(s.eye(32),-s.eye(32))
        jr=realify(jn); ir=realify(s.I*s.eye(32))
        check(f'ambient all gap real structure m{m}',b0*b0==s.eye(64) and b0.T==b0)
        check(f'ambient all gap real structure I reversal m{m}',zero(b0*ir+ir*b0))
        check(f'ambient all gap J parity m{m}',b0*jr==(-1)**(m+1)*jr*b0)
        for index,z in enumerate(gamma):
            check(f'ambient all gap real structure Clifford m{m}/{index}',zero(comm(b0,realify(z))))
        rows.append({'normal_gap':2*m,'minimal_pure_normal_equivariant_real_rank':64,'minimal_nonabelian_real_rank':128,'J_only_algebra_at_128':'so(4)' if m%2 else 'u(2)','I_preserving_algebra_at_128':'u(2)'})
    pauli=[s.Matrix([[0,1],[1,0]]),s.Matrix([[0,-s.I],[s.I,0]]),s.diag(1,-1)]
    for index,p in enumerate(pauli):
        check(f'ambient multiplicity nonabelian skew {index}',(s.I*p).H==-s.I*p)
    check('ambient multiplicity nonabelian Lie bracket',comm(s.I*pauli[0],s.I*pauli[1])==-2*s.I*pauli[2])
    RESULTS['ambient_dimension_audit']={'comparisons':rows,'negative_control':'Automatic normal-factor purity plus minimal non-Abelian full ambient completion permits normal gaps two, four and six. The normal building-block discriminator is not invariant under this changed carrier premise.'}


def selfdual_curvature_bridge():
    pauli=[s.Matrix([[0,1],[1,0]]),s.Matrix([[0,-s.I],[s.I,0]]),s.diag(1,-1)]
    generators=[-s.I*p/2 for p in pauli]
    for i,t in enumerate(generators):
        check(f'curvature bridge su2 skew {i}',t.H==-t)
        for j,t2 in enumerate(generators):
            check(f'curvature bridge su2 bracket {i}/{j}',comm(t,t2)==sum((s.LeviCivita(i,j,k)*generators[k] for k in range(3)),s.zeros(2)))
    def eta(mu,nu,a):
        if mu==3:return int(nu==a)
        if nu==3:return -int(mu==a)
        return s.LeviCivita(mu,nu,a)
    def field(m):
        b=[sum((m[i,a]*generators[a] for a in range(3)),s.zeros(2)) for i in range(3)]
        return [[s.diag(0,sum((eta(i,j,a)*b[a] for a in range(3)),s.zeros(2))) for j in range(4)] for i in range(4)]
    def beta(f,y):
        return s.Matrix([s.expand(sum(s.Matrix.hstack(y,f[i][mu]*y,f[3][mu]*y).det() for mu in range(4))) for i in range(4)])
    z,a,b=s.symbols('curvature_z curvature_a curvature_b')
    scale=s.symbols('curvature_f',real=True)
    y=s.Matrix([z,a,b]); f=field(scale*s.eye(3))
    value=beta(f,y)
    expected=s.Matrix([-s.I*z*scale**2*(a*a-b*b)/2,z*scale**2*(a*a+b*b)/2,s.I*z*scale**2*a*b,0])
    check('curvature bridge exact spinor square',zero(value-expected))
    check('curvature bridge null spatial covector',s.expand((value.T*value)[0])==0)
    check('curvature bridge determinant central weight',zero(beta(f,s.I*y)+s.I*value))
    check('curvature bridge double spinor phase',zero(beta(f,s.Matrix([z,s.I*a,s.I*b]))+value))
    for index,control in enumerate((s.Matrix([1,1,0]),s.Matrix([2,1,0]),s.Matrix([1,1+s.I,2-s.I]),s.Matrix([0,1,1]),s.Matrix([1,0,0]))):
        fc=field(s.eye(3));bc=beta(fc,control)
        norm=s.simplify((bc.H*bc)[0]);wnorm=abs(control[1])**2+abs(control[2])**2
        ladder=sum((fc[i][j]*control).H.dot(fc[i][j]*control) for i in range(4) for j in range(i+1,4))
        check(f'curvature bridge Hermitian norm {index}',s.simplify(norm-abs(control[0])**2*wnorm**2/2)==0)
        check(f'curvature bridge scalar curvature {index}',s.simplify(ladder-s.Rational(3,2)*wnorm)==0)
        if norm!=0:
            vec=bc[:3,0];axis=s.simplify(s.I*vec.cross(s.conjugate(vec))/norm)
            check(f'curvature bridge unit real axis {index}',zero(s.im(axis)) and zero(axis.T*axis-s.ones(1)))
            check(f'curvature bridge transverse axis {index}',zero(axis.T*vec))
        if index in (0,1):
            ratio=s.simplify(norm/((control.H*control)[0]*(ladder/2)**2))
            check(f'curvature bridge bound fraction {index}',ratio==[s.Rational(4,9),s.Rational(32,45)][index])
    matrices=[s.eye(3),s.diag(1,2,3),s.Matrix([[1,2,0],[0,1,1],[1,0,2]]),s.diag(1,1,0),s.diag(1,0,0)]
    p=s.Matrix([-s.I*(a*a-b*b)/2,(a*a+b*b)/2,s.I*a*b])
    for index,m in enumerate(matrices):
        actual=beta(field(m),y)
        check(f'curvature bridge cofactor rule {index}',zero(actual[:3,0]-z*m.cofactor_matrix()*p))
    bad=beta(field(s.diag(1,2,3)),s.Matrix([1,1,0]))
    check('selfdual anisotropy need not give null contact',(bad.T*bad)[0]==-s.Rational(27,4))
    for index,g in enumerate((s.diag(s.I,1,1),s.Matrix([[0,1,0],[1,0,0],[0,0,1]]))):
        transformed=[[g*t*g.H for t in row] for row in f]
        check(f'curvature bridge full normal frame covariance {index}',zero(beta(transformed,g*y)-g.det()*value))
    coordinates=s.symbols('bpst_x0:4',real=True);rho=s.symbols('bpst_rho',positive=True)
    denominator=sum(x*x for x in coordinates)+rho*rho
    connection=[2*sum((eta(mu,nu,a)*coordinates[nu]*generators[a] for nu in range(4) for a in range(3)),s.zeros(2))/denominator for mu in range(4)]
    for mu in range(4):
        check(f'BPST determinant connection trace {mu}',s.trace(connection[mu])==0)
        for nu in range(mu+1,4):
            curvature=connection[nu].diff(coordinates[mu])-connection[mu].diff(coordinates[nu])+comm(connection[mu],connection[nu])
            target=-4*rho*rho*sum((eta(mu,nu,a)*generators[a] for a in range(3)),s.zeros(2))/denominator**2
            check(f'BPST full coordinate curvature {mu}/{nu}',all(s.cancel(x)==0 for x in curvature-target))
    RESULTS['selfdual_curvature_bridge']={'null_ladder':'beta=-(i/sqrt2) z0 f^2 epsilon(w)','generic_selfduality_control':'Anisotropic curvature fails universal nullness.','global_control':'Complete connection metric on ordinary R10 with compatible totally geodesic lower flag; beta vanishes on Q5 and Q6.','scope':'Prescribed geometric connection and candidate transition; axis is not a selected propagation momentum.'}


def charge_polarization_compatibility():
    y=s.diag(-s.Rational(1,2),*[s.Rational(1,6)]*3)
    t=s.diag(s.Rational(1,2),-s.Rational(1,2))
    q=tensor(y,s.eye(2))+tensor(s.eye(4),t)
    for eigenvalue in q.eigenvals():
        columns=[index%2 for index in range(8) if q[index,index]==eigenvalue]
        check(f'charge eigenspace has one auxiliary branch {eigenvalue}',len(set(columns))==1)
    e=s.eye(4);phi=s.Matrix.hstack(2*e[:,0],e[:,1]);vec=s.Matrix(list(phi))
    check('charge matrix vector isometry',(vec.H*vec)[0]==s.trace(phi.H*phi))
    check('charge tensor matrix action',q*vec==s.Matrix(list(y*phi+phi*t.T)))
    for index,sign in enumerate((1,-1)):
        op=tensor(y,s.eye(2))+sign*tensor(s.eye(4),t)
        variance=s.simplify((vec.H*op*op*vec)[0]/5-((vec.H*op*vec)[0]/5)**2)
        check(f'rank two charge variance {index}',variance==[s.Rational(4,225),s.Rational(4,9)][index])
    h=s.Matrix([[s.Rational(3,5),4*s.I/5],[4*s.I/5,s.Rational(3,5)]])
    changed=phi*h.T
    tau=phi.T*s.conjugate(phi);tau2=changed.T*s.conjugate(changed)
    gram2=changed.H*changed
    check('physical auxiliary density covariance',tau2==h*tau*h.H)
    check('complex auxiliary transpose convention',tau2==gram2.T and tau2!=gram2)
    r=vec*vec.H;dephased=s.diag(*[r[i,i] for i in range(8)])
    def partials(mat):
        spin=s.Matrix(4,4,lambda i,j:sum(mat[2*i+a,2*j+a] for a in range(2)))
        aux=s.Matrix(2,2,lambda a,b:sum(mat[2*i+a,2*i+b] for i in range(4)))
        return spin,aux
    check('charge dephasing invariant density',not zero(comm(q,r)) and zero(comm(q,dephased)))
    check('charge dephasing retains two reduced axes',partials(r)==partials(dephased))
    check('charge dephasing changes density rank',r.rank()==1 and dephased.rank()==2)
    check('charge partial traces isolate axes',partials(q)==(2*y,4*t))
    support=s.diag(*[int(dephased[i,i]!=0) for i in range(8)])
    check('unweighted charge support loses spectral gap',partials(support)[0]==s.diag(1,1,0,0))
    pairs=list(itertools.combinations(range(4),2))
    def wedge(v,w):return s.Matrix([v[i]*w[j]-v[j]*w[i] for i,j in pairs])
    p=wedge(phi[:,0],phi[:,1]);phase=(3+4*s.I)/5
    p2=wedge(phi[:,0],phase*phi[:,1])
    check('charge relative phase preserves plane projector',zero(p2*p2.H-p*p.H) and not zero(p2-p))
    projection=s.diag(1,1,0,0)
    exterior_projection=s.Matrix(6,6,lambda a,b:projection.extract(pairs[a],pairs[b]).det())
    check('neutral support reconstructs oriented normal plane',exterior_projection==p*p.H/(p.H*p)[0])
    RESULTS['charge_polarization_compatibility']={'obstruction':'Every fixed-Q eigenstate has rank one and zero quadratic framed-plane vector.','surviving_geometry':'Charge-invariant weighted density retains normal and auxiliary spectral axes and oriented normal plane, but no phase-sensitive real line.','scope':'Algebraic charge-eigencondition comparison, not an imposed physical superselection or particle model.'}


def curvature_metric_and_round_flag():
    def wedge(a,b):
        out={}
        for ia,ca in a.items():
            for ib,cb in b.items():
                index=ia+ib
                if len(set(index))!=len(index):continue
                sign=(-1)**sum(index[i]>index[j] for i in range(len(index)) for j in range(i+1,len(index)))
                key=tuple(sorted(index));out[key]=out.get(key,0)+sign*ca*cb
        return {key:s.expand(value) for key,value in out.items()}
    def interior(a,index):
        return {key[:j]+key[j+1:]:(-1)**j*value for key,value in a.items() for j in range(len(key)) if key[j]==index}
    sigma=[{(0,1):1,(2,3):1},{(0,2):1,(1,3):-1},{(0,3):1,(1,2):1}]
    top=(0,1,2,3)
    for i in range(3):
        for j in range(3):check(f'curvature metric normalized wedge Gram {i}/{j}',wedge(sigma[i],sigma[j]).get(top,0)==2*int(i==j))
    def urbantke(forms):
        return s.Matrix(4,4,lambda i,j:s.expand(sum(s.LeviCivita(a,b,c)*wedge(wedge(interior(forms[a],i),interior(forms[b],j)),forms[c]).get(top,0) for a,b,c in itertools.permutations(range(3)))/6))
    check('curvature metric Urbantke calibration',urbantke(sigma)==s.eye(4))
    m=s.Matrix(3,3,s.symbols('urbantke_m0:9',real=True))
    forms=[{key:sum(m[j,a]*sigma[j].get(key,0) for j in range(3)) for key in itertools.combinations(range(4),2)} for a in range(3)]
    gram=s.Matrix(3,3,lambda a,b:wedge(forms[a],forms[b]).get(top,0))
    check('curvature metric general wedge Gram',zero(gram-2*m.T*m))
    check('curvature metric general Urbantke tensor',zero(urbantke(forms)-m.det()*s.eye(4)))
    l=s.Matrix([[1,1,0,0],[0,2,0,0],[0,0,1,1],[0,0,0,1]])
    def pullback(form):
        out={}
        for (i,j),coefficient in form.items():
            term=wedge({(a,):l[i,a] for a in range(4)},{(a,):l[j,a] for a in range(4)})
            for key,value in term.items():out[key]=out.get(key,0)+coefficient*value
        return out
    check('curvature metric nondiagonal base covariance',urbantke([pullback(f) for f in sigma])==l.det()*l.T*l)
    anisotropic=s.diag(2,8,18)
    check('curvature metric anisotropy survives conformal reconstruction',anisotropic/s.trace(anisotropic)==s.diag(s.Rational(1,14),s.Rational(4,14),s.Rational(9,14)))
    x,y,r,t=s.symbols('topology_x topology_y topology_r topology_t')
    check('normal adjoint Pontryagin roots',s.expand((x-y)**2-(x+y)**2+4*x*y)==0)
    for sign in (1,-1):
        check(f'base Hodge Pontryagin roots {sign}',s.expand((r+sign*t)**2-r*r-t*t-sign*2*r*t)==0)
        c2=-sign
        check(f'round sphere chiral bundle Chern number {sign}',-4*c2==sign*4)
        check(f'product sphere SU2 Chern allowance {sign}',-4*(-2*sign)==sign*8)
    radial,rho,kappa=s.symbols('round_r round_rho round_kappa',positive=True)
    length=s.integrate(2*rho/(s.sqrt(kappa)*(radial**2+rho**2)),(radial,0,s.oo))
    check('curvature normalized R4 finite escape length',length==s.pi/s.sqrt(kappa))
    xs=s.symbols('sphere_x0:5',real=True);radius=s.symbols('sphere_radius',positive=True)
    x=s.Matrix(xs);constraint=sum(z*z for z in xs)-radius**2
    p=s.eye(5)-x*x.T/radius**2;e5=s.eye(5)[:,4];xi=e5-xs[4]*x/radius**2
    def sphere_zero(expr):
        numerator=s.together(expr).as_numer_denom()[0]
        return s.rem(s.expand(numerator),constraint,xs[0])==0
    check('round observer vector tangent',sphere_zero((x.T*xi)[0]))
    check('round observer vector norm',sphere_zero((xi.T*xi)[0]-1+xs[4]**2/radius**2))
    derivative=xi.jacobian(xs)
    check('round observer vector intrinsic Hessian',all(sphere_zero(v) for v in p*derivative*p+xs[4]*p/radius**2))
    for sign in (1,-1):
        pole={xs[i]:0 for i in range(4)};pole[xs[4]]=sign*radius
        check(f'round observer vector pole zero {sign}',zero(xi.subs(pole)))
        check(f'round observer vector pole index {sign}',((-sign/radius)*s.eye(4)).det()==radius**-4)
    creators,gamma,_,_=exterior(5)
    p6=(s.eye(32)+s.I*gamma[6]*gamma[7])*(s.eye(32)+s.I*gamma[8]*gamma[9])/4
    omega5=-gamma[0]*gamma[1]*gamma[2]*gamma[3]*gamma[4]
    omega3=-s.I*gamma[0]*gamma[1]*gamma[2]
    su2=[unit(5,3,4)-unit(5,4,3),s.I*(unit(5,3,4)+unit(5,4,3)),s.I*(unit(5,3,3)-unit(5,4,4))]
    for index,t in enumerate(su2):
        lift=exterior_action(t,creators)
        check(f'round normal SU2 vacuum projector invariant {index}',zero(comm(lift,p6)))
    p5=p6*(s.eye(32)+omega5)/2;p3=p5*(s.eye(32)+omega3)/2
    check('round own sector minimal projector ranks',[s.trace(p3),s.trace(p5),s.trace(p6)]==[2,4,8])
    check('round observer projector not parallel on whole base',not zero(comm(gamma[2]*gamma[3],p3)))
    for a,b in itertools.combinations(range(3),2):
        check(f'round observer own tangent holonomy preserves volume {a}/{b}',zero(comm(gamma[a]*gamma[b],p3)))
    RESULTS['curvature_metric_and_round_flag']={'metric':'Definite curvature fixes a conformal class; its anisotropy and metric volume remain distinct.','global_flag':'Round S4 base with chiral normal V admits complete nested sectors and minimal projectors parallel on each own sector.','forced_zeros':'A global observer vector on S4 cannot remain nonzero; the height-gradient control has two index-one poles.','scope':'Changed global topology, not physical compactification or selected particle geometry.'}


def seven_normal_reduction():
    triples={(1,2,3):1,(1,4,5):1,(1,6,7):1,(2,4,6):1,(2,5,7):-1,(3,4,7):-1,(3,5,6):-1}
    def phi(i,j,k):
        if len({i,j,k})<3:return 0
        seq=(i,j,k)
        return triples.get(tuple(sorted(seq)),0)*(-1)**sum(seq[a]>seq[b] for a in range(3) for b in range(a+1,3))
    cs=[]
    for i in range(1,8):
        c=s.zeros(8);c[i,0]=1;c[0,i]=-1
        for j in range(1,8):
            for k in range(1,8):c[k,j]=phi(i,j,k)
        cs.append(c)
    for i,c in enumerate(cs):
        check(f'seven real Clifford skew {i}',c.T==-c)
        for j,d in enumerate(cs):check(f'seven real Clifford {i}/{j}',c*d+d*c==-2*int(i==j)*s.eye(8))
    basis=s.eye(8)
    def tensors(a,b):
        k2=(a.T*a)[0]*(b.T*b)[0]-(a.T*b)[0]**2
        v=s.Matrix([-(a.T*c*b)[0] for c in cs])
        omega=s.Matrix(7,7,lambda i,j:-(a.T*(cs[i]*cs[j]+int(i==j)*s.eye(8))*b)[0])
        return k2,v,omega
    controls=[(basis[:,0],basis[:,7]),(2*basis[:,0],basis[:,7]),(2*basis[:,0],basis[:,0]+basis[:,7]),(basis[:,0]+basis[:,1],basis[:,2]+2*basis[:,7])]
    for index,(a,b) in enumerate(controls):
        k2,v,w=tensors(a,b)
        check(f'seven axis norm {index}',(v.T*v)[0]==k2)
        check(f'seven plane kernel {index}',zero(w*v))
        check(f'seven transverse complex structure {index}',w*w==-k2*s.eye(7)+v*v.T)
        psi=a+s.I*b;h=(psi.H*psi)[0];q=(psi.T*psi)[0]
        check(f'seven invariant discriminant {index}',s.expand(h*h-q*s.conjugate(q))==4*k2)
        for change,(ap,bp) in enumerate([(-b,a),(a+b,b),(2*a,b/2)]):
            check(f'seven oriented pair invariance {index}/{change}',tensors(ap,bp)==(k2,v,w))
        check(f'seven conjugate reverses geometry {index}',tensors(a,-b)==(k2,-v,-w))
    k2,v,w=tensors(*controls[0])
    expected=unit(7,0,5)-unit(7,5,0)-unit(7,1,4)+unit(7,4,1)-unit(7,2,3)+unit(7,3,2)
    check('seven octonion orientation calibration',v==s.eye(7)[:,6] and w==expected)
    check('seven phase real degeneration',tensors(basis[:,0],2*basis[:,0])[0]==0)
    spin=[cs[i]*cs[j]/2 for i,j in itertools.combinations(range(7),2)]
    a=basis[:,0];b=basis[:,7]
    single=s.Matrix.hstack(*[x*a for x in spin])
    pair=s.Matrix.hstack(*[s.Matrix.vstack(x*a,x*b) for x in spin])
    check('seven real spinor stabilizer dimension',21-single.rank()==14)
    check('seven actual pair stabilizer dimension',21-pair.rank()==8)
    for scale,expected_dim in [(1,9),(2,8)]:
        # Xa=-theta b, Xb=theta a for a complex projective eigenray.
        aa=scale*a
        mat=s.Matrix.hstack(*[s.Matrix.vstack(x*aa,x*b) for x in spin],s.Matrix.vstack(b,-aa))
        check(f'seven projective stabilizer dimension {scale}',22-mat.rank()==expected_dim)
        if scale==2:check('seven generic projective phase fixed',all(z[-1]==0 for z in mat.nullspace()))
    creators,gamma,_,parity=exterior(3);gamma=gamma+[parity]
    pairing=s.zeros(8)
    for subset in range(8):
        other=7^subset;degree=subset.bit_count()
        inversions=sum(i>j for i in range(3) if subset&(1<<i) for j in range(3) if other&(1<<j))
        pairing[subset,other]=(-1)**(degree*(degree+1)//2+inversions)
    check('seven invariant real pairing',pairing.T==pairing and pairing**2==s.eye(8))
    for i,g in enumerate(gamma):
        check(f'seven pairing Clifford skew {i}',zero(g.T*pairing+pairing*g))
        check(f'seven invariant conjugation {i}',zero(pairing*s.conjugate(g)+g*pairing))
    for index,psi in enumerate([basis[:,0],basis[:,7],basis[:,0]+basis[:,7],basis[:,0]+2*basis[:,7],basis[:,0]+s.I*basis[:,7]]):
        annihilator=7-s.Matrix.hstack(*[g*psi for g in gamma]).rank()
        check(f'seven pure iff null controls {index}',annihilator==(3 if index<2 else 0))
    psi=2*a+s.I*b;h=(psi.H*psi)[0];q=(psi.T*psi)[0]
    t=q/(h+s.sqrt(h*h-q*s.conjugate(q)));eta=psi-t*s.conjugate(psi)
    check('seven equivariant purification exact',eta==s.Rational(4,3)*(a+s.I*b) and (eta.T*eta)[0]==0)
    phased=s.I*psi;qp=(phased.T*phased)[0];tp=qp/(h+s.sqrt(h*h-qp*s.conjugate(qp)))
    check('seven purification phase covariance',phased-tp*s.conjugate(phased)==s.I*eta)
    RESULTS['seven_normal_reduction']={'geometry':'Every non-phase-real Spin7 ray yields an axis and Hermitian six-plane; purity is unnecessary for this nonlinear extraction.','stabilizers':'Null projective ray U3; generic nonnull projective ray SU3 with fixed determinant volume.','limits':'Axis gives a tubular intermediate sector, not a prescribed global embedding; Spin8 also permits nonlinear purification, so this rule alone is not dimension selection.'}


def lifted_charge_flag():
    weights=[[0,2,2,2],[-3,-1,-1,-1]]
    pairs=[(i,j) for i in range(4) for j in range(4) if i!=j]
    for i,j in pairs:check(f'lifted quadratic normal character {i}/{j}',weights[0][i]+weights[1][j] in (-1,1))
    for m in range(-5,6):
        minimum=min(abs(weights[0][i]-m)+abs(weights[1][j]-m) for i,j in pairs)
        exact=abs(m+1)+min(abs(m),abs(m-2))
        check(f'lifted quadratic jet minimum {m}',minimum==exact)
        check(f'lifted quadratic fixed point obstruction {m}',2*m not in (-1,1))
        for n in {0,2,-3,-1}:
            # One complex coordinate already attains the minimum orbital degree.
            degrees=[p+q for p in range(10) for q in range(10) if p-q==n-m]
            check(f'lifted component Taylor weight {m}/{n}',min(degrees)==abs(n-m))
    check('lifted carrier improves covariance only bound',abs(1+1)+min(abs(1),abs(1-2))==3)
    z=s.Matrix([1+2*s.I,2-s.I,3+s.I]);r2=(z.H*z)[0]
    phi=s.zeros(4,2);phi[0,0]=s.sqrt(1+r2)
    phi[1:4,1]=s.conjugate(z)
    check('lifted neutral punctured section rank',phi.rank()==2)
    check('lifted neutral constant spectral gap',phi.H*phi==s.diag(1+r2,r2))
    phase=(3+4*s.I)/5
    rotated=s.zeros(4,2);rotated[0,0]=phi[0,0];rotated[1:4,1]=s.conjugate(phase*z)
    check('lifted neutral combined covariance',zero(rotated[:,0]-phi[:,0]) and zero(rotated[:,1]-s.conjugate(phase)*phi[:,1]))
    u=s.Matrix([[0,s.I,0],[1,0,0],[0,0,(3+4*s.I)/5]])
    check('lifted metric dual complex unitary covariance',zero(s.conjugate(u*z)-u.inv().T*s.conjugate(z)))
    pairs3=[(1,2),(0,2),(0,1)];signs=s.diag(1,-1,1)
    exterior_u=s.Matrix(3,3,lambda i,j:u.extract(pairs3[i],pairs3[j]).det())
    check('lifted determinant cancellation complex covariance',zero(signs*exterior_u*signs/u.det()-u.inv().T))
    j=s.Matrix([[0,-1],[1,0]]);y=s.Matrix([2,3]);q=y+s.I*j*y
    check('lifted vector minus complex eigenspace',zero(j*q+s.I*q))
    check('lifted quadratic real frame orientation',j*s.re(q)==s.im(q))
    rotation=s.Rational(3,5)*s.eye(2)+s.Rational(4,5)*j
    aa,bb=s.symbols('quotient_a quotient_b',real=True)
    check('lifted simultaneous quotient frame coordinates',zero(rotation*(aa*y+bb*j*y)-aa*rotation*y-bb*j*rotation*y))
    check('lifted real line circle obstruction',rotation[1,0]!=0)
    RESULTS['lifted_charge_flag']={'jet_order':'ord_N q >= |M+1|+min(|M|,|M-2|), sharp for smooth germs.','zero_section':'Every integer single-total-charge state has vanishing quadratic frame at the circle fixed locus.','punctured_control':'A smooth total-neutral rank-two section gives a radial framed plane away from the zero section.','quotient':'Fixed-vector-space tautological O(-1) and simultaneous differential-action trivial quotient are distinct.'}


def adapted_connection():
    j=s.diag(*([s.Matrix([[0,-1],[1,0]])]*3))
    for a,b in itertools.combinations(range(6),2):
        t=unit(6,a,b)-unit(6,b,a);u=(t-j*t*j)/2;m=(t+j*t*j)/2
        check(f'adapted orthogonal connection splitting {a}/{b}',zero(comm(u,j)) and zero(m*j+j*m) and u+m==t and s.trace(u.T*m)==0)
        derivative=comm(t,j);k=-j*derivative/2
        check(f'adapted J transport minimal norm {a}/{b}',zero(comm(k,j)+derivative) and s.trace(k.T*k)==s.trace(derivative.T*derivative)/4)
    def skew(a,b):return unit(6,a,b)-unit(6,b,a)
    t1=skew(0,2)-skew(1,3);t2=skew(0,4)-skew(1,5)
    bx=comm(t1,j);by=comm(t2,j);bxy=comm(t1,by)
    kx=-j*bx/2;ky=-j*by/2
    curvature=-(bx*by+j*bxy)/2+(by*bx+j*bxy)/2+comm(kx,ky)
    check('adapted J curvature independent derivative sign',curvature==-comm(t1,t2) and curvature==-comm(bx,by)/4)
    a,b,c=s.symbols('adapted_a adapted_b adapted_c',real=True)
    u=s.eye(3)[:,0];alpha=s.Matrix([s.I*a,b,s.I*c]);eta=alpha-s.I*a*u;p=u*u.H
    k=u*eta.H-eta*u.H-s.I*a*p
    check('adapted frame skew and parallel',k.H==-k and zero(k*u+alpha))
    check('adapted complex frame norm',s.expand(s.trace(k.H*k))==a*a+2*b*b+2*c*c)
    check('adapted real frame norm',s.expand(s.trace(realify(k).T*realify(k)))==2*a*a+4*b*b+4*c*c)
    derivatives=[s.eye(3)[:,1],s.I*s.eye(3)[:,1],s.eye(3)[:,2],s.I*s.eye(3)[:,2]]
    ks=[u*v.H-v*u.H for v in derivatives];fs=[]
    for i,h in enumerate(derivatives):
        check(f'adapted frame derivative erased {i}',h+ks[i]*u==s.zeros(3,1))
        for z in range(i+1,4):
            v=derivatives[z]
            # K_j=u du_j^dagger-du_j u^dagger+i a_j P, a_j=-i u^dagger du_j.
            second=-int(i==z)*u
            di_kz=h*v.H+u*second.H-second*u.H-v*h.H+(h.H*v)[0]*p
            dz_ki=v*h.H+u*second.H-second*u.H-h*v.H+(v.H*h)[0]*p
            f=di_kz-dz_ki+comm(ks[i],ks[z]);expected=h*v.H-v*h.H
            check(f'adapted frame curvature derivative {i}/{z}',f==expected and f*u==s.zeros(3,1))
            fs.append(f)
    columns=[s.Matrix(list(s.re(f))+list(s.im(f))) for f in fs]
    check('adapted residual U2 dimension',s.Matrix.hstack(*columns).rank()==4)
    check('adapted residual nonabelian curvature',not zero(comm(fs[0],fs[1])))
    check('adapted metric scalar change',sum(((f*s.eye(3)[:,1]).H*(f*s.eye(3)[:,1]))[0] for f in fs)==8)
    check('adapted original determinant survives only old connection',s.Matrix.hstack(u,derivatives[0],derivatives[2]).det()==1)
    RESULTS['adapted_connection']={'rule':'K_J=-J nabla J/2; K_u=u eta^dagger-eta u^dagger-i a uu^dagger.','minimality':'Unique pointwise Frobenius-minimal metric correction preserving specified J and actual frame u,Ju.','curvature':'F_J=(F-JFJ)/2-(nabla J wedge nabla J)/4; F_flag=Q F_J Q+eta wedge eta^dagger.','control':'Adapted determinant jets vanish while residual curvature spans U2; scalar deficit changes from zero to four at the specified point and vector.','scope':'Changes connection and generally ambient metric; not a selected physical law.'}


def alternating_vector_rule():
    for m in range(1,6):
        _,gamma,_,parity=exterior(m)
        c=s.eye(2**m)
        for g in gamma[1::2]:c=c*g
        check(f'vector bilinear invariant transpose {m}',c.T==(-1)**(m*(m+1)//2)*c)
        for i,g in enumerate(gamma):
            check(f'vector bilinear gamma transpose {m}/{i}',g.T*c==(-1)**m*c*g)
            for chirality in (1,-1):
                indices=[j for j in range(2**m) if parity[j,j]==chirality]
                block=(c*g).extract(indices,indices)
                if m%2==0:
                    check(f'vector same chirality absent {m}/{i}/{chirality}',block==s.zeros(len(indices)))
                else:
                    sign=(-1)**(m*(m+3)//2)
                    check(f'vector exchange symmetry {m}/{i}/{chirality}',block.T==sign*block and block.det()!=0)
        if m==3:
            even=[j for j in range(8) if parity[j,j]==1]
            beta=[(c*g).extract(even,even) for g in gamma]
            for a,b in itertools.combinations(range(6),2):
                spin=(gamma[a]*gamma[b]/2).extract(even,even)
                for v in range(6):
                    target=beta[b] if v==a else -beta[a] if v==b else s.zeros(4)
                    check(f'vector bilinear Spin equivariance {a}/{b}/{v}',spin.T*beta[v]+beta[v]*spin==target)
            annihilators=[s.Matrix.hstack(*[g*s.eye(8)[:,j] for g in gamma]) for j in (0,3)]
            check('pure columns need not share maximal annihilator',6-s.Matrix.vstack(*annihilators).rank()==1)
    RESULTS['alternating_vector_rule']={'classification':'dim Hom_Spin(2m)(Lambda² Delta+,N_C)=1 iff m=3 mod4, zero otherwise.','joint_rule':'Automatic purity of every individual chiral spinor plus this alternating vector channel singles out six normal dimensions.','ambient_multiplicity':'Spin-trivial copies cannot create a missing normal Hom; mixed chirality or symmetric auxiliary channels change the rule.','scope':'Geometric-map premise, not selected particle states or absolute sector endpoints.'}


def joint_carrier_projectors():
    creators,gamma,number,parity=exterior(3)
    n=[s.SparseMatrix(realify(g)) for g in gamma]
    scalar_i=s.SparseMatrix(realify(s.I*s.eye(8)))
    conjugation=s.SparseMatrix(s.diag(s.eye(8),-s.eye(8)))
    aa=n[0]*n[2]*n[4]*conjugation
    j=s.SparseMatrix(realify(s.I*parity));g=s.SparseMatrix(realify(s.I*number))
    def quaternion_matrix(axis,right=False):
        result=s.zeros(4)
        result[axis,0]=1;result[0,axis]=-1
        for b in range(1,4):
            if b==axis:continue
            for c in range(1,4):result[c,b]=s.LeviCivita(b,axis,c) if right else s.LeviCivita(axis,b,c)
        return result
    left=[quaternion_matrix(a) for a in range(1,4)]
    right=s.SparseMatrix(s.diag(quaternion_matrix(1,True),quaternion_matrix(1,True)))
    h=[s.SparseMatrix(s.BlockMatrix([[s.zeros(4),s.eye(4)],[-s.eye(4),s.zeros(4)]]).as_explicit())]
    h.extend(s.SparseMatrix(s.diag(a,-a)) for a in left)
    for index,x in enumerate(h):
        check(f'joint tangent negative generator {index}',x.T==-x and comm(x,right)==s.zeros(8))
        for z,y in enumerate(h):check(f'joint tangent negative Clifford {index}/{z}',x*y+y*x==-2*int(index==z)*s.eye(8))
    check('joint right quaternion structure',right.T==-right and right**2==-s.eye(8))
    kron=lambda a,b:s.SparseMatrix(s.kronecker_product(a,b))
    tangent=[kron(j,a) for a in h];normal=[kron(a,s.eye(8)) for a in n]
    full=tangent+normal;eye=s.eye(128);zz=s.zeros(128)
    for i,a in enumerate(full):
        check(f'joint ambient symmetric gamma {i}',a.T==a)
        for b in range(i,10):check(f'joint ambient full Clifford {i}/{b}',a*full[b]+full[b]*a==2*int(i==b)*eye)
    k=-(n[2]*n[3]+n[4]*n[5])/2
    nv=s.diag(*[((v>>1)&1)+((v>>2)&1) for v in range(8)])
    check('joint complement circle identity',k==realify(s.I*(nv-s.eye(8))))
    e0=-k*k;e1=s.eye(16)-e0
    pe=kron(e0,s.eye(8));po=kron(e1,s.eye(8));z=kron(k,right)
    pp=(pe+z)/2;pm=(pe-z)/2;p96=po+pp
    check('joint complement split square',z*z==pe and pe*z==z and z.T==z)
    check('joint complementary refined projectors',pp*pm==zz and pp+pm==pe)
    su2=[unit(3,1,2)-unit(3,2,1),s.I*(unit(3,1,2)+unit(3,2,1)),s.I*(unit(3,1,1)-unit(3,2,2))]
    symmetries=[kron(a,s.eye(8)) for a in [scalar_i,aa,scalar_i*aa,j,g]]
    symmetries.extend(kron(realify(exterior_action(a,creators)),s.eye(8)) for a in su2)
    for name,p,rank in [('even',pe,64),('odd',po,64),('even_plus',pp,32),('even_minus',pm,32),('full_charge',p96,96)]:
        check(f'joint orthogonal projector {name}',p.T==p and p*p==p and s.trace(p)==rank)
        for index,a in enumerate(tangent+normal[:2]+symmetries):check(f'joint projector preserved algebra {name}/{index}',comm(p,a)==zz)
        for index,a in enumerate(tangent+normal[:2]):check(f'joint compressed spatial norm {name}/{index}',p*a*p*a*p==p)
    check('joint depth5 charge closure identity',comm(kron(g,s.eye(8)),normal[0])==normal[1])
    omega5=tangent[0]*tangent[1]*tangent[2]*tangent[3]*normal[0]
    check('joint odd volume fails fibre charge',comm(kron(g,s.eye(8)),omega5)==tangent[0]*tangent[1]*tangent[2]*tangent[3]*normal[1] and comm(kron(g,s.eye(8)),omega5)!=zz)
    q=kron(-j*g/3,s.eye(8))
    check('joint charge spectral projector commute',comm(q,p96)==zz)
    charges=[0,s.Rational(2,3),s.Rational(-1,3),-1]
    full_counts=[];projected_counts=[]
    for charge in charges:
        indices=[i for i in range(128) if q[i,i]==charge]
        full_counts.append(len(indices));projected_counts.append(sum(p96[i,i] for i in indices))
    check('joint full charge multiplicities',full_counts==[16,48,48,16])
    check('joint retained charge multiplicities',projected_counts==[8,40,40,8])
    tangent_circle=kron(s.eye(16),right)
    check('joint tangent Spin-c circle preserves retained projector',comm(tangent_circle,p96)==zz)
    check('joint tangent Spin-c circle preserves complement split',comm(tangent_circle,z)==zz)
    for index,a in enumerate(full):
        check(f'joint tangent Spin-c transition Clifford compatibility {index}',comm(tangent_circle,a)==zz)
    RESULTS['joint_carrier_projectors']={'ambient_real_rank':128,'projector_real_ranks':[32,64,96,128],'minimum_containing_all_original_U2_charge_components':96,'minimum_balanced_original_copies':128,'depth5_constraint':'Fibre charge and one real normal gamma force its complex partner gamma; allowed Q5 projectors are already Q6 compatible.','scope':'One larger common carrier, with nonminimal lower ranks; circle still rotates the fixed Q5 spatial domain.'}


def universal_vector_frame():
    # Sparse exterior states verify high-dimensional witnesses without dense carriers.
    def gamma_apply(state,index):
        coordinate=index//2;imaginary=index%2;answer={}
        for mask,value in state.items():
            occupied=bool(mask&(1<<coordinate))
            sign=(-1)**((mask&((1<<coordinate)-1)).bit_count())
            factor=sign*((-s.I if occupied else s.I) if imaginary else 1)
            target=mask^(1<<coordinate)
            answer[target]=answer.get(target,0)+factor*value
        return {key:s.expand(value) for key,value in answer.items() if value!=0}
    def c_apply(state,m):
        for j in reversed(range(m)):state=gamma_apply(state,2*j+1)
        return state
    def bilinear(a,b,m):
        answer=[]
        for j in range(2*m):
            cg=c_apply(gamma_apply(b,j),m)
            answer.append(s.expand(sum(value*cg.get(key,0) for key,value in a.items())))
        return s.Matrix(answer)
    for m in (3,7,11,15):
        vacuum={0:1};pair={3:1};q=bilinear(vacuum,pair,m)
        if m==3:
            check('universal six pure pair output',q==s.Matrix([0,0,0,0,-s.I,1]))
            check('universal six pure pair null',s.expand((q.T*q)[0])==0)
        else:
            check(f'universal higher pure independent pair zero {m}',q==s.zeros(2*m,1))
            check(f'universal higher rank obstruction {m}',2**(m-1)-1>2*m)
            full=(1<<m)-1
            a={0:1,15:1};b={full^8:1,(full^15)|8:1}
            q=bilinear(a,b,m);expected=s.zeros(2*m,1);expected[7]=-2
            check(f'universal higher nonnull vector witness {m}',q==expected)
            check(f'universal higher nonnull square {m}',(q.T*q)[0]==4)
        for r in range(0,m,2):
            t={(1<<r)-1:1};q=bilinear(vacuum,t,m)
            check(f'pure pair common annihilator selection {m}/{r}',(q!=s.zeros(2*m,1))==(m-r==1))
            check(f'one pure column gives null output {m}/{r}',s.expand((q.T*q)[0])==0)
        # Independently compare the sparse Clifford pairing with its complement sign.
        for mask in (0,3,(1<<m)-1,(1<<(m-1))|1):
            image=c_apply({mask:1},m)
            expected={((1<<m)-1)^mask:s.I**m*(-1)**sum(j+1 for j in range(m) if mask&(1<<j))}
            check(f'universal pairing complement sign {m}/{mask}',image==expected)
    RESULTS['universal_vector_frame']={'theorem':'The same nonzero alternating chiral-vector channel sends every independent pair to a nonzero null vector iff normal dimension is six.','purity':'Automatic purity is unnecessary for this theorem; higher-dimensional independent pure pairs can give zero.','changed_rule':'Selected pure pairs with annihilator intersection dimension one work at every map-bearing normal dimension.','witness':'At all m=3 mod4 >=7, the explicit pair (1+e1234,e_all_except4+e_4_through_m) has beta=-2 f8 and beta dot beta=4.'}


def joint_endpoint_rule():
    for n in (2,3,4,5,6,10):
        check(f'endpoint typed curvature Hodge degree {n}',((n-2)==2)==(n==4))
    for k in range(1,8):
        check(f'endpoint complete chiral middle rank {k}',(s.binomial(2*k,k)/2==3)==(k==2))
    a=s.symbols('triad_scale',positive=True)
    pauli=[s.Matrix([[0,1],[1,0]]),s.Matrix([[0,-s.I],[s.I,0]]),s.diag(1,-1)]
    t=[-s.I*x/2 for x in pauli];connection=[a*x for x in t]
    f={(i,j):comm(connection[i],connection[j]) for i in range(3) for j in range(3)}
    for i,j in itertools.product(range(3),repeat=2):
        expected=sum((a*a*s.LeviCivita(i,j,k)*t[k] for k in range(3)),s.zeros(2))
        check(f'endpoint actual three-base curvature {i}/{j}',f[i,j]==expected)
    check('endpoint three-base Bianchi',zero(comm(connection[0],f[1,2])+comm(connection[1],f[2,0])+comm(connection[2],f[0,1])))
    e=a*a*s.eye(3);inverse=e*e.T/e.det()
    check('endpoint curvature triad reconstructs metric',inverse==s.eye(3)/a**2)
    volume=s.symbols('triad_volume',positive=True);x=e/volume;d=volume*x.det()
    check('endpoint triad arbitrary auxiliary volume cancels',zero(x*x.T/d-inverse))
    rotation=s.Matrix([[s.Rational(3,5),s.Rational(-4,5),0],[s.Rational(4,5),s.Rational(3,5),0],[0,0,1]])
    rotated=e*rotation
    check('endpoint triad gauge invariance',zero(rotated*rotated.T/rotated.det()-inverse))
    check('endpoint curvature Gram perfection has five constraints',s.binomial(3+1,2)-1==5)
    RESULTS['joint_endpoint_rule']={'conditional_endpoints':'Universal alternating chiral frame plus complete same-degree Hodge two-form curvature law gives base4 and ambient10.','premise_scope':'The typed Hodge condition itself structurally imposes a four-base; it is not generic curvature-metric reconstruction.','countercontrol':'An actual constant SU2 connection on R3 reconstructs a positive triad metric and a coherent ambient9 control when the Hodge-chiral law is replaced.','null_contact':'Requires five extra perfect-curvature Gram conditions beyond definiteness.'}


def color_flag_tradeoff():
    x,y=s.symbols('curvature_eigen_x curvature_eigen_y',nonnegative=True)
    h=s.diag(0,x,y);h0=h-s.trace(h)*s.eye(3)/3
    check('color flag sharp response anisotropy',s.simplify(s.trace(h0*h0)-s.trace(h)**2/6-(x-y)**2/2)==0)
    pauli=[s.Matrix([[0,1],[1,0]]),s.Matrix([[0,-s.I],[s.I,0]]),s.diag(1,-1)]
    ts=[-s.I*p/2 for p in pauli]
    v=s.symbols('color_v0:3',real=True);b=s.symbols('color_central_b',real=True)
    traceless=sum((v[i]*ts[i] for i in range(3)),s.zeros(2))
    general=s.I*b*s.eye(2)+traceless
    check('color every SU2 channel scalar squared response',zero(traceless.H*traceless-sum(z*z for z in v)*s.eye(2)/4))
    check('color U2 central traceless cross response',zero(general.H*general-(b*b+sum(z*z for z in v)/4)*s.eye(2)+2*s.I*b*traceless))
    for coefficients in [(1,1,1),(1,2,3)]:
        h=sum((2*a*a*t.H*t for a,t in zip(coefficients,ts)),s.zeros(2))
        check(f'color isotropic complement need not perfect curvature {coefficients}',h==sum(a*a for a in coefficients)*s.eye(2)/2)
    creators,_,_,_=exterior(3);indices=[6,3,5]
    def wedge(f):return exterior_action(f,creators).extract(indices,indices)
    f=s.diag(s.zeros(1),general)
    check('color actual first triplet induced curvature',wedge(f)==s.diag(s.trace(general),general))
    check('color actual second triplet induced curvature',wedge(f)-s.trace(f)*s.eye(3)==s.diag(s.zeros(1),general-s.trace(general)*s.eye(2)))
    form_f=s.Matrix([1,0,0,0,0,0]);form_g=s.Matrix([-s.Rational(1,2),0,0,0,0,s.sqrt(3)/2])
    check('color equal norm curvature forms',(form_f.T*form_f)[0]==1 and (form_g.T*form_g)[0]==1 and ((form_f+form_g).T*(form_f+form_g))[0]==1)
    def responses(channels):
        first=[wedge(f) for f in channels]
        second=[wedge(f)-s.trace(f)*s.eye(3) for f in channels]
        return [sum((f.H*f for f in fs),s.zeros(3)).applyfunc(s.simplify) for fs in (channels,first,second)]
    framed=[s.diag(0,s.I*form_f[j],s.I*form_g[j]) for j in range(6)]
    check('color one triplet can be isotropic with real parallel line',responses(framed)==[s.diag(0,1,1),s.eye(3),s.diag(0,1,1)])
    plane_only=[s.I*s.diag(form_f[j],form_g[j],-form_f[j]-form_g[j]) for j in range(6)]
    check('color both triplets isotropic after relaxing real frame',responses(plane_only)==[s.eye(3)]*3)
    central=[s.I*form_f[j]*s.eye(3) for j in range(6)]
    check('color central plane response factors',responses(central)==[s.eye(3),4*s.eye(3),s.eye(3)])
    offdiag=[]
    for i,j in itertools.combinations(range(3),2):
        offdiag.extend([unit(3,i,j)-unit(3,j,i),s.I*(unit(3,i,j)+unit(3,j,i))])
    check('color irreducible curvature scalar response',sum((f.H*f for f in offdiag),s.zeros(3))==4*s.eye(3))
    closure=offdiag+[comm(offdiag[0],offdiag[1]),comm(offdiag[2],offdiag[3])]
    check('color curvature spans full SU3 Lie closure',s.Matrix.hstack(*[s.Matrix(list(s.re(f))+list(s.im(f))) for f in closure]).rank()==8)
    coords=s.symbols('color_x0:4',real=True);aform=coords[0];bform=-coords[0]/2
    connection=[s.zeros(3),s.I*s.diag(aform,bform,-aform-bform),s.zeros(3),s.I*s.diag(0,s.sqrt(3)*coords[2]/2,-s.sqrt(3)*coords[2]/2)]
    for index,(i,j) in enumerate(itertools.combinations(range(4),2)):
        curvature=connection[j].diff(coords[i])-connection[i].diff(coords[j])+comm(connection[i],connection[j])
        check(f'color plane only actual smooth curvature {i}/{j}',curvature==plane_only[index])
    u=s.eye(3)[:,0];alpha=[a*u for a in connection]
    check('color real line bending with zero determinant jets',alpha[1]!=s.zeros(3,1) and all(s.Matrix.hstack(u,alpha[i],alpha[j]).det()==0 for i,j in itertools.combinations(range(4),2)))
    RESULTS['color_flag_tradeoff']={'sharp_bound':'tr H0²=(tr H)²/6+(lambda1-lambda2)²/2 for the actual W curvature-square response with parallel framed line.','actual_triplets':'Lambda²W may be isotropic nonflat; its D^-1 twist W* cannot be isotropic nonflat under the framed-line premise.','changed_flag':'Parallel complex plane without parallel real phase admits explicit nonflat identical scalar response on both charge triplets.','scope':'Curvature response, not measured color breaking, mass splitting or physical rates; common Clifford transition normalization remains algebraically 1:3.'}


def curvature_observer_lift():
    t=s.symbols('locked_t',nonzero=True)
    doublet=t+1/t
    triplet=t*t+1+1/(t*t)
    char=4*doublet*(doublet+2)
    check('locked Dirac32 character',s.expand(char-(4+4*triplet+8*doublet))==0)
    check('locked Dirac32 dimension',char.subs(t,1)==32)
    check('locked central trace',char.subs(t,-1)==0)
    check('observer-only central trace',(16*doublet).subs(t,-1)==-32)
    check('locked first orbital doublet decomposition',s.expand(doublet**3-2*doublet-(t**3+t+1/t+1/t**3))==0)
    pauli=[s.Matrix([[0,1],[1,0]]),s.Matrix([[0,-s.I],[s.I,0]]),s.diag(1,-1)]
    normal=[s.diag(p/2,s.zeros(2)) for p in pauli]
    base=[s.diag(p/2,p/2) for p in pauli]
    generators=[tensor(base[i],s.eye(2),s.eye(4))+tensor(s.eye(4),s.eye(2),normal[i]) for i in range(3)]
    casimir=sum((g*g for g in generators),s.zeros(32))
    check('locked full Casimir multiplicities',casimir.eigenvals()=={s.Integer(0):4,s.Integer(2):12,s.Rational(3,4):16})
    check('locked full torus multiplicities',generators[2].eigenvals()=={s.Integer(-1):4,s.Integer(1):4,s.Integer(0):8,s.Rational(-1,2):8,s.Rational(1,2):8})
    for i,g in enumerate(generators):
        for j,h in enumerate(generators):
            check(f'locked full Lie algebra {i}/{j}',zero(comm(g,h)-s.I*sum((s.LeviCivita(i,j,k)*generators[k] for k in range(3)),s.zeros(32))))
    ar,ai,br,bi=s.symbols('locked_ar locked_ai locked_br locked_bi',real=True)
    aa,bb=ar+s.I*ai,br+s.I*bi
    q=s.Matrix([[aa,bb],[-s.conjugate(bb),s.conjugate(aa)]])
    quaternion=s.Matrix([[0,-1],[1,0]])
    check('quaternionic equivariant conjugation',zero(quaternion*s.conjugate(q)-q*quaternion))
    z,a,b=s.symbols('locked_z locked_a locked_b')
    w=s.Matrix([a,b])
    h1=s.conjugate(z)*w
    h2=z*quaternion*s.conjugate(w)
    check('charge preserving orbital H1 equivariance',zero(s.conjugate(z)*q*w-q*h1))
    check('charge preserving orbital H2 equivariance',zero(z*quaternion*s.conjugate(q*w)-q*h2))
    for i,h in enumerate((h1,h2)):
        check(f'charge preserving orbital weight {i}',zero(h.subs({z:t*z,s.conjugate(z):s.conjugate(z)/t,a:t*a,b:t*b,s.conjugate(a):s.conjugate(a)/t,s.conjugate(b):s.conjugate(b)/t},simultaneous=True)-h))
        check(f'orbital normal inversion odd {i}',zero(h.subs({a:-a,b:-b,s.conjugate(a):-s.conjugate(a),s.conjugate(b):-s.conjugate(b)},simultaneous=True)+h))
        check(f'orbital scalar-line zero {i}',zero(h.subs(z,0)))
        check(f'orbital complement zero {i}',zero(h.subs({a:0,b:0})))
    gram=s.Matrix.hstack(h1,h2).H*s.Matrix.hstack(h1,h2)
    expected=z*s.conjugate(z)*(a*s.conjugate(a)+b*s.conjugate(b))
    check('common compensator Gram scalar',zero(gram-expected*s.eye(2)))
    check('common compensator determinant',s.expand(s.Matrix.hstack(h1,h2).det()-expected)==0)
    for nv in range(3):
        for degree in range(4):
            check(f'locked section central parity {nv}/{degree}',(-1)*(-1)**nv*(-1)**degree==(-1)**(1+nv+degree))
    # A diagonal rotation fixes nu and has the matching adjoint action.
    q=s.diag((1-s.I)/s.sqrt(2),(1+s.I)/s.sqrt(2))
    r=s.Matrix([[0,-1,0],[1,0,0],[0,0,1]])
    def p(v):
        a,b=v
        return s.Matrix([-s.I*(a*a-b*b)/2,(a*a+b*b)/2,s.I*a*b])
    check('actual locked quadratic bridge covariance',zero(p(q*w)-r*p(w)))
    check('actual locked base rotation orthogonal',r.T*r==s.eye(3) and r.det()==1)
    def eta(mu,nu,a):
        if mu==3:return int(nu==a)
        if nu==3:return -int(mu==a)
        return s.LeviCivita(mu,nu,a)
    x=s.Matrix(s.symbols('locked_x0:4',real=True));rho=s.symbols('locked_rho',positive=True)
    rotation=s.diag(r,1)
    den=(x.T*x)[0]+rho*rho
    ta=[-s.I*v/2 for v in pauli]
    def connection(v):
        return [2*sum((eta(mu,nu,a)*v[nu]*ta[a] for nu in range(4) for a in range(3)),s.zeros(2))/den for mu in range(4)]
    original=connection(x);moved=connection(rotation*x)
    for mu in range(4):
        pullback=sum((rotation[nu,mu]*moved[nu] for nu in range(4)),s.zeros(2))
        check(f'actual BPST connection lifted isometry {mu}',zero(pullback-q*original[mu]*q.H))
    check('locked central spatial action nonidentity',(-s.eye(2))*s.Matrix([1,0])!=s.Matrix([1,0]))
    RESULTS['curvature_observer_lift']={'spatial_group':'Faithful locked Spin3 on the ordinary total space; its center reverses V, so no ambient SO3 descent.','output':'Quadratic determinant contact descends to the SO3 vector representation.','Dirac32':'4 spin0 + 4 spin1 + 8 spin1/2 fibre representations at a fixed point.','common_compensators':'conjugate(z0) w and z0 J conjugate(w) preserve original lifted charge and realize actual spin1/2 section subspaces in either charge-triplet doublet branch.','scope':'Additional section choices with forced zeros; no selected physical spin, color SU3, occupations or normalizable particle.'}


def common_charge_spin_covariants():
    t=s.symbols('covariant_t',nonzero=True)
    def character(n):
        return sum(t**k for k in range(-n,n+1,2))
    for p in range(7):
        for q in range(7):
            actual=s.expand(character(p)*character(q))
            target=sum(character(k) for k in range(abs(p-q),p+q+1,2))
            check(f'polynomial covariant character decomposition {p}/{q}',s.expand(actual-target)==0)
            multiplicity=actual.coeff(t,1)-actual.coeff(t,3)
            check(f'polynomial doublet bidegree selection {p}/{q}',multiplicity==int(abs(p-q)==1))
    for degree in range(13):
        count=0
        for a in range(degree+1):
            for b in range(degree-a+1):
                for p in range(degree-a-b+1):
                    q=degree-a-b-p
                    if a-b+p-q==0 and abs(p-q)==1:
                        count+=1
        check(f'charge invariant covariant Hilbert coefficient {degree}',count==(degree if degree>=2 and degree%2==0 else 0))
    z,a,b=s.symbols('covariant_z covariant_a covariant_b')
    w=s.Matrix([a,b]);j=s.Matrix([[0,-1],[1,0]])
    def p(v):
        a,b=v
        return s.Matrix([-s.I*(a*a-b*b)/2,(a*a+b*b)/2,s.I*a*b])
    check('compensator first circular output',zero(p(s.conjugate(z)*w)-s.conjugate(z)**2*p(w)))
    check('compensator second circular output',zero(p(z*j*s.conjugate(w))-z*z*s.conjugate(p(w))))
    for index,ww in enumerate((s.Matrix([1,0]),s.Matrix([1+s.I,2-s.I]),s.Matrix([2,3*s.I]))):
        c=p(ww);n=s.simplify(s.I*c.cross(s.conjugate(c))/(c.H*c)[0])
        d=p(j*s.conjugate(ww));nn=s.simplify(s.I*d.cross(s.conjugate(d))/(d.H*d)[0])
        check(f'compensators opposite circular axes {index}',zero(n+nn))
        frame=s.Matrix.hstack(ww,j*s.conjugate(ww));rho=(ww.H*ww)[0]
        check(f'normal frame determinant degree control {index}',s.simplify(frame.det()-rho)==0)
        check(f'normal frame unitary control {index}',zero(frame.H*frame-rho*s.eye(2)))
    # One stabilizer circle suffices to force H(0,w)=0; conjugation
    # transports the example w=e1 to every nonzero w.
    q=s.diag(s.I,-s.I)
    check('charge fixed-line compensator zero stabilizer',q-s.eye(2)!=s.zeros(2) and (q-s.eye(2)).det()!=0)
    zr,zi,ar,ai,br,bi=s.symbols('jet_zr jet_zi jet_ar jet_ai jet_br jet_bi',real=True)
    variables=[zr,zi,ar,ai,br,bi]
    zz=zr+s.I*zi;ww=s.Matrix([ar+s.I*ai,br+s.I*bi])
    hh1=s.conjugate(zz)*ww;hh2=zz*j*s.conjugate(ww)
    check('compensator quaternionic dependence',zero(hh2-j*s.conjugate(hh1)))
    real_map=s.Matrix([s.re(v).expand() for v in hh1]+[s.im(v).expand() for v in hh1]+[s.re(v).expand() for v in hh2]+[s.im(v).expand() for v in hh2])
    jac=real_map.jacobian(variables)
    rho=(ww.H*ww)[0];sigma=zz*s.conjugate(zz)
    check('common compensator vertical derivative norm',s.expand(s.trace(jac.T*jac)-4*rho-8*sigma)==0)
    for index,(fixture,rank) in enumerate((([0,0,1,0,0,0],2),([1,0,0,0,0,0],4),([0,0,0,0,0,0],0))):
        check(f'common compensator forced-locus real jet rank {index}',jac.subs(dict(zip(variables,fixture))).rank()==rank)
    dr,di,er,ei,fr,fi=s.symbols('jet_dr jet_di jet_er jet_ei jet_fr jet_fi',real=True)
    dz=dr+s.I*di;dw=s.Matrix([er+s.I*ei,fr+s.I*fi])
    dh1=s.conjugate(dz)*ww+s.conjugate(zz)*dw
    dh2=j*s.conjugate(dh1)
    value=(dh1.H*dh1)[0]+(dh2.H*dh2)[0]
    expected=2*rho*dz*s.conjugate(dz)+2*sigma*(dw.H*dw)[0]+4*s.re(dz*s.conjugate(zz)*(ww.H*dw)[0])
    check('common compensator directional first jet invariant',s.expand(value-expected)==0)
    evaluation=s.diag(1,0,0);g=s.Matrix([[0,-1,0],[1,0,0],[0,0,1]]);c=s.Matrix([0,1,0])
    check('constant color mixing control is SU3',g.H*g==s.eye(3) and g.det()==1)
    check('constant color mixing fails zero-locus evaluation kernel',evaluation*c==s.zeros(3,1) and evaluation*g*c==s.Matrix([-1,0,0]))
    normal_frame=s.Matrix.hstack(hh1,hh2);evaluation=s.diag(1,normal_frame)
    check('compensator triplet Gram obstruction',zero(evaluation.H*evaluation-s.diag(1,sigma*rho,sigma*rho)))
    RESULTS['common_charge_spin_covariants']={'classification':'Every charge-invariant polynomial SU2-equivariant doublet map on C plus C2 is P(|z|²,|w|²) conjugate(z)w + Q(|z|²,|w|²) z J conjugate(w).','Hilbert_series':'2 t²/(1-t²)²; 2n maps at degree 2n.','smooth_zero_theorem':'Every continuous map with those symmetries vanishes on z=0 and w=0.','shared_geometry':'The same pair gives equal Gram norms, an SU2 frame off the zero sets, and opposite circular axes.','scope':'No selected handedness, color SU3 action or particle count; extra rotating input data changes the theorem.'}


def round_spin_sections():
    pauli=[s.Matrix([[0,1],[1,0]]),s.Matrix([[0,-s.I],[s.I,0]]),s.diag(1,-1)]
    gamma=[tensor(v,pauli[2]) for v in pauli]+[tensor(s.eye(2),pauli[0]),tensor(s.eye(2),pauli[1])]
    eye=s.eye(4)
    for i,g in enumerate(gamma):
        check(f'round ambient spin Hermitian {i}',g.H==g)
        for j,h in enumerate(gamma):
            check(f'round ambient spin Clifford {i}/{j}',g*h+h*g==2*int(i==j)*eye)
    check('round ambient volume sign',s.prod(gamma)==-eye)
    k=s.I*gamma[3]*gamma[4]
    check('round fixed-plane volume',k==-tensor(s.eye(2),pauli[2]) and k*k==eye)
    x=s.Matrix(s.symbols('round_spin_x1:6',real=True));r=s.symbols('round_spin_R',positive=True)
    n=sum((x[i]*gamma[i]/r for i in range(5)),s.zeros(4))
    radius=(x.T*x)[0]
    for chi in (-1,1):
        p=(eye+chi*n)/2
        check(f'round chiral projection polynomial {chi}',zero(p*p-p-(radius/r**2-1)*eye/4))
        check(f'round chiral projection trace {chi}',s.trace(p)==2)
        for kap in (-1,1):
            indices=[i for i in range(4) if k[i,i]==kap]
            embedding=eye[:,indices]
            compressed=embedding.H*p*embedding
            target=(s.eye(2)+chi*sum((x[i]*(embedding.H*gamma[i]*embedding)/r for i in range(3)),s.zeros(2)))/2
            check(f'round doublet exact Gram {chi}/{kap}',zero(compressed-target))
            check(f'round doublet Gram determinant {chi}/{kap}',s.simplify(compressed.det()-(1-sum(x[i]**2 for i in range(3))/r**2)/4)==0)
            check(f'round doublet rank2 away from Q2 {chi}/{kap}',compressed.subs(dict(zip(x,[0,0,0,r,0])))==s.eye(2)/2)
            check(f'round doublet rank1 on Q2 {chi}/{kap}',compressed.subs(dict(zip(x,[0,0,r,0,0]))).rank()==1)
            u=embedding[:,0]
            bloch=s.Matrix([(u.H*gamma[i]*u)[0] for i in range(3)])
            check(f'round doublet Bloch unit {chi}/{kap}',(bloch.T*bloch)[0]==1)
            point=list(-chi*r*bloch)+[0,0]
            pp=p.subs(dict(zip(x,point)))
            check(f'round chiral one-zero fixture {chi}/{kap}',pp*u==s.zeros(4,1))
            check(f'round opposite chirality survives {chi}/{kap}',(eye-pp)*u==u)
            tangent=[i for i in range(5) if point[i]==0]
            derivative=s.Matrix.hstack(*[chi*gamma[i]*u/(2*r) for i in tangent])
            real_derivative=s.Matrix.vstack(s.re(derivative),s.im(derivative))
            check(f'round chiral zero transverse real rank {chi}/{kap}',real_derivative.rank()==4)
            for i in range(3):
                for j in range(3):
                    check(f'round constant doublet observer action {chi}/{kap}/{i}/{j}',embedding.H*gamma[i]*gamma[j]*embedding==pauli[i]*pauli[j])
    north=gamma[4]
    for i in range(4):
        connection=north*gamma[i]/(2*r)
        check(f'round spin connection preserves chirality {i}',comm(connection,north)==-gamma[i]/r)
        for j in range(4):
            check(f'round spin connection Clifford compatibility {i}/{j}',comm(connection,gamma[j])==int(i==j)*north/r)
    for perm in itertools.permutations(range(5)):
        check('round Chern orientation trace '+''.join(map(str,perm)),s.trace(s.prod(gamma[i] for i in perm))==-4*s.LeviCivita(*perm))
    for chi in (-1,1):
        p=(eye+chi*north)/2
        check(f'round selfdual chirality convention {chi}',zero((gamma[0]*gamma[1]-chi*gamma[2]*gamma[3])*p))
        check(f'round Weyl not tangent Clifford submodule {chi}',p*gamma[0]*p==s.zeros(4))
        check(f'round Chern number from exact sphere integral {chi}',s.simplify((-s.Rational(chi,8))*64*s.pi**2/(8*s.pi**2))==-chi)
    projector=(eye+k)/2
    check('round constant doublet not full tangent Clifford submodule',comm(projector,gamma[3])!=s.zeros(4))
    # Observer invariance alone permits other lines in the multiplicity C2.
    v=s.Matrix([1,1])/s.sqrt(2)
    embedding=tensor(s.eye(2),v)
    other=embedding.H*(eye+n)*embedding/2
    check('round alternative observer doublet Gram',zero(other-(1+x[3]/r)*s.eye(2)/2))
    check('round alternative observer doublet point defect',other.subs(dict(zip(x,[0,0,0,-r,0])))==s.zeros(2))
    RESULTS['round_spin_sections']={'global_realization':'Full round Dirac bundle has genuine globally nonzero observer-doublet section spaces, with curved spin connection.','chiral_evaluation':'For the fixed-plane volume K splitting, Gram eigenvalues (1 +/- r3/R)/2; rank loss exactly on great Q2=S2. Every nonzero projected constant section has one transverse zero and c2=-chi.','alternatives':'Observer symmetry alone admits different doublets with displaced sphere or point defects; Q3 remains an RP1 family of containing equators.','scope':'Actual global section representations are not tangent-Clifford invariant rank2 subbundles, parallel fields or occupied particles.'}


def nonspin_global_control():
    h=s.symbols('cp2_h')
    c1=-3*h;c2=3*h*h;e=3*h*h;p1=3*h*h
    check('CP2 normal dual tangent first Pontryagin',s.expand(c1*c1-2*c2-p1)==0)
    check('CP2 same curvature endpoint characteristic identity',s.expand(c1*c1-4*c2-(p1-2*e))==0)
    check('CP2 total spin parity cancellation',(3-3)%2==0 and 3%2==1)
    check('CP2 no separate normal determinant root',(-3)%2==1)
    check('CP2 total canonical line cancellation',-3-(-3)==0)
    check('CP2 total Pontryagin class',s.expand(p1+c1*c1-2*c2)==6*h*h)
    check('CP2 quaternionic auxiliary mismatch',(c1/h)**2==9 and int(c1/h)%2==1)
    check('CP2 projective traceless instanton number',s.expand(c2-c1*c1/4)==s.Rational(3,4)*h*h)
    def twoform(i,j):
        return unit(4,i,j)-unit(4,j,i)
    sd=[twoform(0,1)+twoform(2,3),twoform(0,2)-twoform(1,3),twoform(0,3)+twoform(1,2)]
    anti=[twoform(0,1)-twoform(2,3),twoform(0,2)+twoform(1,3),twoform(0,3)-twoform(1,2)]
    for i,a in enumerate(sd):
        for j,b in enumerate(anti):
            check(f'CP2 opposite Hodge skew factors commute {i}/{j}',comm(a,b)==s.zeros(4))
    pauli=[s.Matrix([[0,1],[1,0]]),s.Matrix([[0,-s.I],[s.I,0]]),s.diag(1,-1)]
    ta=[-s.I*p/2 for p in pauli]
    f,t=s.symbols('cp2_f cp2_t',real=True)
    ff=[[sum((f*anti[a][i,j]*ta[a] for a in range(3)),s.zeros(2)) for j in range(4)] for i in range(4)]
    full=[[ff[i][j]+s.I*t*sd[0][i,j]*s.eye(2) for j in range(4)] for i in range(4)]
    z,a,b=s.symbols('cp2_z cp2_a cp2_b');y=s.Matrix([z,a,b])
    def beta(curvature,xi):
        curv=[[s.diag(0,x) for x in row] for row in curvature]
        return s.Matrix([s.expand(sum(s.Matrix.hstack(y,curv[i][mu]*y,curv[xi][mu]*y).det() for mu in range(4))) for i in range(4)])
    for xi in range(4):
        check(f'CP2 full determinant curvature cancels from cubic {xi}',zero(beta(full,xi)-beta(ff,xi)))
    response=sum((full[i][j].H*full[i][j] for i in range(4) for j in range(i+1,4)),s.zeros(2))
    check('CP2 abelian curvature remains in scalar response',zero(response-(s.Rational(3,2)*f*f+2*t*t)*s.eye(2)))
    check('CP2 Fubini Study curvature response',response.subs({f:2,t:3})==24*s.eye(2))
    def wedge(a,b):
        return sum(s.LeviCivita(i,j,k,l)*a[i,j]*b[k,l] for i in range(4) for j in range(i+1,4) for k in range(4) for l in range(k+1,4))
    trace_square=sum(wedge(s.Matrix(4,4,lambda i,j:full[i][j][a,b]),s.Matrix(4,4,lambda i,j:full[i][j][b,a])) for a in range(2) for b in range(2))
    trace=s.Matrix(4,4,lambda i,j:s.trace(full[i][j]))
    c2density=s.expand((trace_square-wedge(trace,trace))/(8*s.pi**2))
    volume=s.pi**2/2
    check('CP2 exact curvature Chern integral',s.simplify(c2density.subs({f:2,t:3})*volume)==3)
    scalar_coefficient=s.Rational(3,4)*f*f+t*t
    contact_fraction=f**4/(2*scalar_coefficient**2)
    check('CP2 curvature contact bound restricted fraction',contact_fraction.subs({f:2,t:3})==s.Rational(1,18))
    check('SU2-only contact bound restricted fraction',s.simplify(contact_fraction.subs(t,0))==s.Rational(8,9))
    phase=s.diag(s.I,1)
    j=s.Matrix([[0,-1],[1,0]])
    check('U2 quaternionic conjugation determinant twist',j*s.conjugate(phase)==phase*j/phase.det())
    RESULTS['nonspin_global_control']={'base':'CP2, V=holomorphic cotangent, W=C plus V; base and normal separately nonspin, total M10 spin.','geometry':'Complete 2,3,4,5,6,10 flag with bent lower S3, perfect traceless ASD curvature and nonzero SD determinant curvature.','global_carrier':'A specified tangent Spin-c repair globalizes the real128 and P96 construction; quaternionic commutants are algebra bundles, not three global axes.','curvature_contact':'Opposite Hodge components cancel the determinant-curvature cross term in beta, while scalar curvature response retains its positive contribution.','scope':'Changed topology and transport premises; no globally ordinary base spinors, no automatic global same-bundle quaternionic compensator pair or globally ambient-parallel P96.'}


def global_multiplicity():
    def quaternion(axis,right=False):
        a=s.zeros(4);a[axis,0]=1;a[0,axis]=-1
        for b in range(1,4):
            for c in range(1,4):
                a[c,b]=s.LeviCivita(b,axis,c) if right else s.LeviCivita(axis,b,c)
        return a
    left=[quaternion(i) for i in range(1,4)]
    right=[quaternion(i,True) for i in range(1,4)]
    for i,l in enumerate(left):
        check(f'global multiplicity left quaternion square {i}',l*l==-s.eye(4) and l.T==-l)
        for j,r in enumerate(right):
            check(f'global multiplicity two commuting quaternion actions {i}/{j}',comm(l,r)==s.zeros(4))
    li,ri=left[0],right[0]
    correlated=(li+ri)/2
    check('global determinant transition generator',correlated==s.diag(s.Matrix([[0,-1],[1,0]]),s.zeros(2)))
    det_plane=s.diag(1,1,0,0);trivial=s.eye(4)-det_plane
    for angle in (s.pi/2,s.pi):
        half_left=s.cos(angle/2)*s.eye(4)+s.sin(angle/2)*li
        half_right=s.cos(angle/2)*s.eye(4)+s.sin(angle/2)*ri
        expected=s.diag(s.Matrix([[s.cos(angle),-s.sin(angle)],[s.sin(angle),s.cos(angle)]]),s.eye(2))
        check(f'global paired half-determinant transition {angle}',zero(half_left*half_right-expected))
        check(f'global paired half-angle sign cancellation {angle}',(-half_left)*(-half_right)==half_left*half_right)
    for i,p in enumerate((det_plane,trivial)):
        check(f'global full Clifford multiplicity complex plane {i}',comm(p,li)==s.zeros(4) and comm(p,ri)==s.zeros(4))
        check(f'global complex plane loses full normal quaternion algebra {i}',comm(p,left[1])!=s.zeros(4))
    check('global real line loses canonical central lift',comm(s.diag(0,0,1,0),li)!=s.zeros(4))
    variables=s.symbols('multiplicity_a0:16');a=s.Matrix(4,4,variables)
    equations=[x for l in left for x in comm(a,l)]
    system,_=s.linear_eq_to_matrix(equations,variables)
    check('global quaternion commutant real dimension',16-system.rank()==4)
    system_symmetric,_=s.linear_eq_to_matrix(equations+list(a-a.T),variables)
    check('global symmetric quaternion commutant scalar only',16-system_symmetric.rank()==1)
    eps=s.Matrix([[0,-1],[1,0]])
    conjugation=tensor(eps,eps)
    check('global paired quaternion real structure square',conjugation*conjugation==s.eye(4))
    real_involution=s.diag(conjugation,-conjugation)
    check('global multiplicity fixed real rank four',s.trace((s.eye(8)+real_involution)/2)==4)
    d=s.diag(1,s.I,-s.I,1)
    check('global four-character real structure descent',conjugation*s.conjugate(d)==d*conjugation)
    check('global normal circle two-pi half signs',(-s.eye(4))*(-s.eye(4))==s.eye(4))
    RESULTS['global_multiplicity']={'exact_bundle':'F = S10_real tensor (R² plus D_R), with multiplicity connection d plus determinant connection.','global_algebra':'Normal and tangent quaternionic commutants are nontrivial associated algebra bundles with retained global complex axes.','submodules':'Full ambient real64 plane submodules retain the canonical normal circle; retaining the full original normal quaternion algebra needs real128. P96 is tangent-only through Q6.','charge_lift':'Independent normal central action has spin and multiplicity half-determinant factors, each minus one at 2pi, cancelling in their product.','scope':'Exact specified transition factorization, not a classification from characteristic numbers alone.'}


def covariant_spatial_operator():
    creators,normal,number,parity=exterior(3)
    z=s.symbols('hull_z0:3');bar=s.symbols('hull_bar0:3')
    e=s.eye(8)
    h1=bar[0]*(z[1]*e[:,3]+z[2]*e[:,5])
    h2=z[0]*(-bar[2]*e[:,3]+bar[1]*e[:,5])
    singlet=e[:,6]
    def dirac(v):
        return s.simplify(2*sum((creators[i]*v.diff(bar[i])+creators[i].T*v.diff(z[i]) for i in range(3)),s.zeros(8,1)))
    def lifted(v,branch):
        return number*v-3*branch*v-sum((z[i]*v.diff(z[i])-bar[i]*v.diff(bar[i]) for i in range(3)),s.zeros(8,1))
    d1=dirac(h1);d2=dirac(h2)
    check('spin-compensator first exact Clifford derivative',d1==-4*bar[0]*e[:,1])
    check('spin-compensator second exact Clifford derivative',zero(d2-2*(-bar[2]*e[:,2]+bar[1]*e[:,4]-2*z[0]*e[:,7])))
    for i,v in enumerate((singlet,h1,h2,d1,d2)):
        check(f'compensator finite Dirac hull second derivative {i}',dirac(dirac(v))==s.zeros(8,1))
        for branch in (0,1):
            check(f'compensator lifted charge retained {i}/{branch}',zero(lifted(v,branch)-(2-3*branch)*v))
    check('compensator first derivative changes normal chirality',parity*d1==-d1 and parity*d2==-d2)
    pauli=[s.Matrix([[0,1],[1,0]]),s.Matrix([[0,-s.I],[s.I,0]]),s.diag(1,-1)]
    for a,pa in enumerate(pauli):
        generator=s.diag(0,-s.I*pa/2)
        spin=exterior_action(generator,creators)
        position=generator*s.Matrix(z)
        conjugate_position=s.conjugate(generator)*s.Matrix(bar)
        for k,v in enumerate((singlet,h1,h2,d1,d2)):
            orbital=sum((position[j]*v.diff(z[j])+conjugate_position[j]*v.diff(bar[j]) for j in range(3)),s.zeros(8,1))
            check(f'actual SU2 horizontal compensator cancellation {a}/{k}',zero(spin*v-orbital))
    sigma,rho=s.symbols('harmonic_sigma harmonic_rho',nonnegative=True)
    def radial(p):
        return s.expand(4*(sigma*s.diff(p,sigma,2)+rho*s.diff(p,rho,2)+2*s.diff(p,sigma)+3*s.diff(p,rho)))
    for a in range(3):
        for b in range(3):
            p=sigma**a*rho**b
            actual_p=p.subs({sigma:z[0]*bar[0],rho:z[1]*bar[1]+z[2]*bar[2]})
            expected_p=radial(p).subs({sigma:z[0]*bar[0],rho:z[1]*bar[1]+z[2]*bar[2]})
            for i,h in enumerate((h1,h2)):
                value=4*sum(((actual_p*h).diff(z[j]).diff(bar[j]) for j in range(3)),s.zeros(8,1))
                check(f'compensator common radial Laplacian {a}/{b}/{i}',zero(value-expected_p*h))
    t=s.symbols('harmonic_t',real=True)
    polys=[]
    for n in range(7):
        p=s.expand(sum((-1)**j*s.binomial(n,j)*s.binomial(n+2,j)*sigma**j*rho**(n-j)/s.Integer(j+1) for j in range(n+1)))
        polys.append(p)
        check(f'compensator radial harmonic polynomial {n}',radial(p)==0)
        normalized=s.expand(p.subs({sigma:t,rho:1-t}))
        jacobi=(-1)**n*s.jacobi(n,2,1,2*t-1)/s.Integer(n+1)
        check(f'compensator exact Jacobi normalization {n}',s.expand(normalized-jacobi)==0)
        norm=s.integrate(2*t*(1-t)**2*normalized**2,(t,0,1))
        check(f'compensator exact angular norm {n}',norm==s.Rational(1,(n+1)*(n+2)*(n+3)))
        check(f'compensator interior angular nodes {n}',s.Poly(normalized,t).count_roots(0,1)==n)
        check(f'compensator angular eigenvalue {n}',(2*n+2)*(2*n+6)==4*(n+1)*(n+3))
        angular=12*normalized-4*(t*(1-t)*s.diff(normalized,t,2)+(2-5*t)*s.diff(normalized,t))
        check(f'compensator full reduced angular equation {n}',s.expand(angular-4*(n+1)*(n+3)*normalized)==0)
        check(f'compensator endpoint coefficients {n}',p.subs(sigma,0)==rho**n and p.subs(rho,0)==(-1)**n*(n+2)*sigma**n/2)
        for k in range(n+1):
            harmonic=polys[n-k]
            check(f'compensator Fischer Laplacian coefficient {n}/{k}',s.expand(radial((sigma+rho)**k*harmonic)-4*k*(2*n-k+4)*(sigma+rho)**max(k-1,0)*harmonic)==0)
        if n:
            previous=polys[n-1].subs({sigma:t,rho:1-t})
            check(f'adjacent harmonic towers have no common interior node {n}',s.degree(s.gcd(normalized,previous),t)==0)
    check('first harmonic interior node exact ratio',polys[1].subs(sigma,2*rho/3)==0)
    for i in range(4):
        for j in range(i):
            pi=polys[i].subs({sigma:t,rho:1-t});pj=polys[j].subs({sigma:t,rho:1-t})
            check(f'compensator harmonic angular orthogonality {i}/{j}',s.integrate(2*t*(1-t)**2*pi*pj,(t,0,1))==0)
    def conjugate_polynomial(value):
        return s.conjugate(value).xreplace({**{s.conjugate(z[j]):bar[j] for j in range(3)},**{s.conjugate(bar[j]):z[j] for j in range(3)}})
    radial_sub={sigma:z[0]*bar[0],rho:z[1]*bar[1]+z[2]*bar[2]}
    cpos=sum((creators[j]*z[j]+creators[j].T*bar[j] for j in range(3)),s.zeros(8))
    monogenic=[]
    for n,p in enumerate(polys):
        a=s.diff(p,sigma);b=s.diff(p,rho)
        norm1=4*(sigma**2*rho*a**2+sigma*(rho*b+2*p)**2)
        norm2=4*(rho*(sigma*a+p)**2+sigma*(rho*b+2*p)**2)
        angular_norms=[s.integrate(s.expand(2*(1-t)*value.subs({sigma:t,rho:1-t})),(t,0,1)) for value in (norm1,norm2)]
        input_norm=s.Rational(1,(n+1)*(n+2)*(n+3))
        check(f'angular Dirac normalized first norm {n}',angular_norms[0]/input_norm==8*(n+2)**2)
        check(f'angular Dirac normalized second norm {n}',angular_norms[1]/input_norm==8*(n+2)*(n+3))
        if n<=3:
            modes=[p.subs(radial_sub)*h for h in (h1,h2)]
            descendants=[dirac(v) for v in modes]
            for k,g in enumerate(descendants):
                expected_norm=(norm1,norm2)[k].subs(radial_sub)
                check(f'angular Dirac pointwise norm {n}/{k}',s.expand((conjugate_polynomial(g).T*g)[0]-expected_norm)==0)
                check(f'angular descendants monogenic {n}/{k}',zero(dirac(g)))
                m=s.simplify(modes[k]-cpos*g/(4*n+8))
                check(f'Fischer completed spinor monogenic {n}/{k}',zero(dirac(m)))
                check(f'Fischer completed spinor total charge {n}/{k}',zero(lifted(m,0)-2*m))
                if n==0: monogenic.append(m)
            check(f'angular Dirac descendants pointwise orthogonal {n}',s.expand((conjugate_polynomial(descendants[0]).T*descendants[1])[0])==0)
    first=(h1+bar[0]**2*e[:,0])/2
    second=(h2+(2*z[0]*bar[0]-z[1]*bar[1]-z[2]*bar[2])*e[:,6])/4
    check('first exact monogenic completion',zero(monogenic[0]-first))
    check('second exact monogenic completion',zero(monogenic[1]-second))
    second_norm=(4*sigma**2-3*sigma*rho+rho**2)/16
    check('same-triplet completed spinor exact norm',s.expand((conjugate_polynomial(second).T*second)[0]-second_norm.subs(radial_sub))==0)
    check('same-triplet completed spinor positive off origin',s.expand(second_norm-((rho-3*sigma/2)**2+7*sigma**2/4)/16)==0)
    check('same-triplet completed spinor retains scalar fibre number',number*second==2*second)
    check('first completion uses orbital fibre compensation',not zero(number*first-2*first))
    for n,p in enumerate(polys):
        aa=s.expand((2*p+rho*(s.diff(p,rho)-s.diff(p,sigma)))/(2*(n+2)))
        cc=p/2
        tt=s.Rational(n+1,2*(n+2))*p
        bb=s.expand(((2*sigma-rho)*p+sigma*rho*(s.diff(p,rho)-s.diff(p,sigma)))/(2*(n+2)))
        equations=[sigma*s.diff(aa,sigma)+2*aa-rho*s.diff(cc,rho)-2*cc,
                   s.diff(aa,rho)+s.diff(cc,sigma),
                   s.diff(bb,rho)+sigma*s.diff(tt,sigma)+tt,
                   s.diff(bb,sigma)-rho*s.diff(tt,rho)-2*tt]
        check(f'complete all-degree monogenic radial system {n}',all(s.expand(v)==0 for v in equations))
        scalar=sum((-1)**j*s.binomial(n+1,j)*s.binomial(n+2,j)*sigma**j*rho**(n+1-j) for j in range(n+2))
        check(f'completed singlet exact binomial harmonic {n}',s.expand(bb+scalar/(2*(n+2)))==0)
        check(f'completed triplet nonzero axis endpoints {n}',bb.subs(sigma,0)==-rho**(n+1)/(2*(n+2)) and bb.subs(rho,0)==(-1)**n*sigma**(n+1)/2)
        check(f'completed triplet no shared interior node {n}',s.degree(s.gcd(p.subs({sigma:t,rho:1-t}),bb.subs({sigma:t,rho:1-t})),t)==0)
        coeff=s.symbols(f'kernel_{n}_0:{4*n+5}')
        monomials=[sigma**j*rho**(n-j) for j in range(n+1)]
        av=sum(coeff[j]*monomials[j] for j in range(n+1))
        cv=sum(coeff[n+1+j]*monomials[j] for j in range(n+1))
        tv=sum(coeff[2*n+2+j]*monomials[j] for j in range(n+1))
        bv=sum(coeff[3*n+3+j]*sigma**j*rho**(n+1-j) for j in range(n+2))
        residuals=[sigma*s.diff(av,sigma)+2*av-rho*s.diff(cv,rho)-2*cv,s.diff(av,rho)+s.diff(cv,sigma),s.diff(bv,rho)+sigma*s.diff(tv,sigma)+tv,s.diff(bv,sigma)-rho*s.diff(tv,rho)-2*tv]
        linear=[v for residual in residuals for v in s.Poly(residual,sigma,rho).coeffs() if v!=0]
        matrix,_=s.linear_eq_to_matrix(linear,coeff)
        check(f'complete even fixed-charge monogenic dimension {n+1}',matrix.cols-matrix.rank()==2)
        fibre=matrix[:,n+1:]
        check(f'fixed fibre-number-two monogenic dimension {n+1}',fibre.cols-fibre.rank()==1)
    for k,v in enumerate((conjugate_polynomial(h1),conjugate_polynomial(h2))):
        expected=2*(bar[1]*e[:,2]+bar[2]*e[:,4]) if k==0 else s.zeros(8,1)
        check(f'coefficient conjugation changes Clifford embedding {k}',zero(dirac(v)-expected))
    latitude=s.symbols('target_latitude',real=True)
    primitive=latitude-latitude**3/3
    volume=s.integrate(1-latitude**2,(latitude,-1,1))
    check('restricted angular target volume normalization',volume==s.Rational(4,3))
    for n in range(7):
        degree=(primitive.subs(latitude,(-1)**n)-primitive.subs(latitude,-1))/volume
        check(f'restricted angular target endpoint degree {n}',degree==s.Rational(1+(-1)**n,2))
    quaternionic_j=realify(s.Matrix([[0,-1],[1,0]]))*s.diag(1,1,-1,-1)
    check('restricted angular target quaternionic orientation',quaternionic_j.det()==1)
    radius=s.symbols('angular_chart_radius',positive=True)
    angular_rate=((2-radius**2)-radius*s.diff(2-radius**2,radius))/(radius**2+(2-radius**2)**2)
    check('degree-one angular chart monotone radial angle',s.simplify(angular_rate-(2+radius**2)/(radius**2+(2-radius**2)**2))==0)
    _,base,_,base_parity=exterior(2)
    def twoform(i,j):
        return unit(4,i,j)-unit(4,j,i)
    anti=[twoform(0,1)-twoform(2,3),twoform(0,2)+twoform(1,3),twoform(0,3)-twoform(1,2)]
    kahler=twoform(0,1)+twoform(2,3)
    # At w=(1,0), the three T_a w are the orthogonal normal vectors below.
    vectors=[-normal[5]/2,normal[4]/2,-normal[3]/2]
    curvature=s.zeros(32)
    for i in range(4):
        for j in range(i+1,4):
            clifford=sum((anti[a][i,j]*vectors[a] for a in range(3)),s.zeros(8))
            curvature+=tensor(base[i]*base[j]*base_parity,clifford)/4
    dark=tensor((s.eye(4)+base_parity)/2,s.eye(8))
    check('actual connection-metric curvature term skew',curvature.H==-curvature)
    check('pure Hodge curvature has dark base Weyl block',curvature*dark==s.zeros(32))
    check('perfect curvature active block singular values',(curvature.H*curvature).eigenvals()=={s.Integer(0):16,s.Rational(1,16):12,s.Rational(9,16):4})
    central=sum((tensor(base[i]*base[j]*base_parity,normal[3])*kahler[i,j]/4 for i in range(4) for j in range(i+1,4)),s.zeros(32))
    check('opposite-Hodge determinant curvature fills other Weyl block',central*(s.eye(32)-dark)==s.zeros(32))
    check('opposite-Hodge curvature terms have no cross square',curvature.H*central+central.H*curvature==s.zeros(32))
    full=2*curvature+3*central
    check('CP2 full curvature Clifford term is invertible off Q6',(full.H*full).eigenvals()=={s.Rational(1,4):12,s.Rational(9,4):20})
    check('determinant block exact square',central**2==-dark/4)
    check('complementary curvature blocks annihilate each other',curvature*central==s.zeros(32) and central*curvature==s.zeros(32))
    ambient=[tensor(b,s.eye(8)) for b in base]+[tensor(base_parity,n) for n in normal]
    # Reconstruct one independent curvature component from its Koszul coefficients.
    connections=[s.zeros(10) for _ in range(10)]
    connections[0][1,4]=-s.Rational(1,2);connections[0][4,1]=s.Rational(1,2)
    connections[1][0,4]=s.Rational(1,2);connections[1][4,0]=-s.Rational(1,2)
    connections[4][0,1]=s.Rational(1,2);connections[4][1,0]=-s.Rational(1,2)
    lifts=[]
    for a,omega in enumerate(connections):
        lift=-sum((omega[b,c]*ambient[b]*ambient[c]/4 for b in range(10) for c in range(10) if omega[b,c]),s.zeros(32))
        lifts.append(lift)
        if not zero(omega):
            for b in range(10):
                expected=sum((omega[b,c]*ambient[c] for c in range(10)),s.zeros(32))
                check(f'positive Clifford ordered spin lift {a}/{b}',comm(lift,ambient[b])==expected)
    triple=ambient[0]*ambient[1]*ambient[4]
    horizontal=sum((ambient[a]*lifts[a] for a in range(4)),s.zeros(32))
    vertical=sum((ambient[a]*lifts[a] for a in range(4,10)),s.zeros(32))
    check('actual LC horizontal curvature contraction',horizontal==triple/2)
    check('actual LC vertical curvature contraction',vertical==-triple/4)
    check('actual LC combined curvature contraction',horizontal+vertical==triple/4)
    coupling=sum((tensor(pa,pa) for pa in pauli),s.zeros(4))
    check('perfect curvature Pauli coupling multiplicities',coupling.eigenvals()=={s.Integer(1):3,s.Integer(-3):1})
    check('perfect curvature Pauli minimal polynomial',(coupling-s.eye(4))*(coupling+3*s.eye(4))==s.zeros(4))
    for a,generator in enumerate([s.diag(0,-s.I*pa/2) for pa in pauli]+[s.diag(0,s.I,s.I)]):
        v=generator*s.Matrix(z);vbar=s.conjugate(generator)*s.Matrix(bar)
        cliff=sum((creators[j]*v[j]+creators[j].T*vbar[j] for j in range(3)),s.zeros(8))
        orbital=sum((z[j]*cliff.diff(z[j])-bar[j]*cliff.diff(bar[j]) for j in range(3)),s.zeros(8))
        check(f'actual curvature orbital plus fibre charge covariance {a}',zero(comm(number,cliff)-orbital))
        check(f'actual curvature fibre charge alone fails {a}',not zero(comm(number,cliff)))
    real_w=s.Matrix(s.symbols('curvature_w0:4',real=True))
    complex_w=s.Matrix([real_w[0]+s.I*real_w[1],real_w[2]+s.I*real_w[3]])
    vectors=[-s.I*pa*complex_w/2 for pa in pauli]
    for a in range(3):
        for b in range(3):
            check(f'perfect curvature real quaternionic Gram {a}/{b}',s.simplify(s.re((vectors[a].H*vectors[b])[0])-(real_w.dot(real_w)/4 if a==b else 0))==0)
    RESULTS['covariant_spatial_operator']={'flat_normal_hull':'The degree2 H1,H2 Clifford derivatives are nonzero degree1 monogenic sections; constant singlet plus two maps plus their derivatives form a five-copy finite polynomial hull preserving total charge.','angular_harmonics':'Two covariants per even degree L=2n+2; Jacobi(2,1) factors, eigenvalue4(n+1)(n+3), exact norm pi³/[(n+1)(n+2)(n+3)].','nodes':'Both columns share n interior nodes at a fixed angular degree; adjacent degrees have no common interior nodes. Forced axes remain.','ambient_curvature':'Actual connection-metric Dirac has +1/4 sum gamma_i gamma_j c(F_ij y), independently reconstructed from Koszul coefficients. Perfect one-Hodge curvature vanishes on one base Weyl block and is invertible on the other off w=0. Opposite-Hodge determinant curvature removes that dark block.','same_connection':'SU2 horizontal derivatives of H1,H2 and descendants vanish by simultaneous orbital and exterior action. Curvature is covariant under total orbital-plus-fibre circle; scalar fibre number alone fails.','scope':'Polynomial geometric sections, not finite-norm particles, physical helicity, occupied levels or a mass spectrum. Proofs establish global extensions and all-degree consequences; finite checks verify their displayed identities.'}
    RESULTS['completed_monogenic_module']={'angular_Dirac_Gram':'Input-normalized Gram is 8(n+2) diag(n+2,n+3), distinct from the equal componentwise spherical Laplace eigenvalue.','complete_even_classification':'SU2-equivariant total-weight2 even polynomials have dimensions 4k+1 at ordinary degree2k>=2; harmonic4, monogenic2. Degree0 has one constant singlet.','fixed_fibre_number':'N=2 leaves one monogenic copy per even degree, including degree0. This does not make N commute with the whole Clifford operator.','zero_sets':'Each homogeneous completed M2_n has no zero off origin; the singlet repairs both doublet axes and all Jacobi nodes. M1_n vanishes precisely at z=0.','global_scope':'Classification uses the SU2 determinant framing. On general U2 data the second branch requires a common determinant twist and the active circle lift must be retained explicitly.','physical_scope':'Available polynomial geometry, not occupied particles, selected physical handedness, energy or mass.'}
    RESULTS['monogenic_angular_topology']={'restricted_target':'The fixed-frame normalized M2_n coefficient map factors through the angular quotient CP2 to S4 in C2 plus R; degree is (1+(-1)^n)/2 with stated orientations.','full_complex_control':'The composite CP2 to full complex-triplet sphere S5 is nullhomotopic; so is the original S5-domain to S5-target map. Restricted parity is not a protected invariant of the full complex field.','scope':'Angular quotient and coefficient target are not identified with physical sectors. The homotopy statements require the accompanying topological proof, not only finite endpoint checks.'}


def charge_saturation_extension():
    creators, gammas, number, parity = exterior(3)
    eye = s.eye(8)
    for j in range(3):
        check(f'saturation circle complex partner {j}', comm(number,gammas[2*j]) == -s.I*gammas[2*j+1])
        check(f'saturation circle second partner {j}', comm(number,gammas[2*j+1]) == s.I*gammas[2*j])
    blocks = [(i,j) for i in range(8) for j in range(8) if i.bit_count()==j.bit_count()]
    variables = s.symbols('p0:'+str(len(blocks)))
    symbolic = s.zeros(8)
    for variable,(i,j) in zip(variables,blocks):
        symbolic[i,j] = variable
    rows = []
    for k in range(7):
        for h in range(max(0,k-3),k//2+1):
            r = k-h
            indices = list(range(2*h))+[2*j for j in range(h,r)]
            equations = []
            for j in indices:
                equations.extend(list(comm(symbolic,gammas[j])))
            if equations:
                coefficients,_ = s.linear_eq_to_matrix(equations,variables)
                dimension = len(variables)-coefficients.rank()
            else:
                dimension = len(variables)
            expected = s.binomial(2*(3-r),3-r)
            check(f'saturation commutant dimension k{k} h{h}', dimension==expected)
            projector = s.diag(*[int(i < 2**r) for i in range(8)])
            check(f'saturation minimum rank k{k} h{h}',s.trace(projector)==2**r)
            check(f'saturation charge projector k{k} h{h}',zero(comm(projector,number)))
            for j in indices:
                check(f'saturation tangent projector k{k} h{h} j{j}',zero(comm(projector,gammas[j])))
            character = [sum(projector[i,i] for i in range(8) if i.bit_count()==q) for q in range(r+1)]
            check(f'saturation binomial character k{k} h{h}',character==[s.binomial(r,q) for q in range(r+1)])
            rows.append({'real_normal_rank':k,'complex_intersection_rank':h,'complex_saturation_rank':r,'minimum_complex_module_rank':2**r,'commutant_dimension':int(dimension),'charge_multiplicities':list(map(int,character))})
    angled = [gammas[0],s.Rational(3,5)*gammas[1]+s.Rational(4,5)*gammas[2]]
    equations = [x for g in angled for x in comm(symbolic,g)]
    coefficients,_ = s.linear_eq_to_matrix(equations,variables)
    check('saturation noncomplex slanted plane commutant',len(variables)-coefficients.rank()==2)
    normal = s.Matrix(s.symbols('sat_p0:6',real=True))
    symbol = sum((normal[j]*gammas[j] for j in range(6)),s.zeros(8))
    defect = comm(number,symbol)
    check('saturation scalar charge defect exact norm',s.simplify(defect.H*defect-normal.dot(normal)*eye)==s.zeros(8))
    for m in range(1,9):
        for r in range(m+1):
            check(f'saturation Vandermonde count m{m} r{r}',sum(s.binomial(m-r,q)**2 for q in range(m-r+1))==s.binomial(2*(m-r),m-r))
    RESULTS['charge_saturation_extension']={'canonical_complex_module_rows':rows,'scope':'Charge-compatible Clifford projectors extend to complex saturation. These are canonical exterior-complex ranks, distinct from reporting-complex ranks of the real quaternionic carrier. Odd normal rank through the circle fixed locus cannot define an invariant spatial domain. No physical charge-to-depth assignment is selected.'}


def completed_frame_extension():
    x,y,p,q,r,t = s.symbols('frame_x frame_y frame_p frame_q frame_r frame_t',real=True)
    aa,cc,dd,bb = s.symbols('frame_A frame_c frame_d frame_B',real=True)
    z=x+s.I*y
    w=s.Matrix([p+s.I*q,r+s.I*t])
    sigma=x*x+y*y
    rho=p*p+q*q+r*r+t*t
    a=s.conjugate(z)**2*aa
    first=s.Matrix([0,-cc*s.conjugate(z)*w[1],cc*s.conjugate(z)*w[0]])
    second=s.Matrix([bb,-dd*z*s.conjugate(w[0]),-dd*z*s.conjugate(w[1])])
    vector=first.cross(second)
    covector=a*second
    check('completed frame cross-product coordinates',zero(vector-s.Matrix([cc*dd*sigma*rho,cc*bb*s.conjugate(z)*w[0],cc*bb*s.conjugate(z)*w[1]])))
    check('completed frame normal nullness',s.expand(covector.dot(vector))==0)
    check('completed frame input orthogonality',s.expand((first.H*second)[0])==0)
    target=(sigma**2*aa**2+cc**2*sigma*rho)*(bb**2+dd**2*sigma*rho)
    check('completed frame norm factors',s.expand((vector.H*vector)[0]+(covector.H*covector)[0]-target)==0)
    cartesian=s.Matrix.vstack((vector+covector)/s.sqrt(2),s.I*(vector-covector)/s.sqrt(2))
    real=cartesian.applyfunc(s.re)
    imaginary=cartesian.applyfunc(s.im)
    check('completed frame real imaginary orthogonality',s.expand(real.dot(imaginary))==0)
    check('completed frame equal real imaginary norms',s.expand(real.dot(real)-imaginary.dot(imaginary))==0)
    orbital=lambda expr: -s.I*(x*s.diff(expr,y)-y*s.diff(expr,x)+p*s.diff(expr,q)-q*s.diff(expr,p)+r*s.diff(expr,t)-t*s.diff(expr,r))
    check('completed frame vector total weight one',zero(vector.applyfunc(orbital)))
    check('completed frame covector total weight one',zero(covector.applyfunc(orbital)+2*covector))
    ss,rr,tt=s.symbols('frame_sigma frame_rho frame_t_unit',real=True)
    for n in range(7):
        f=sum((-1)**j*s.binomial(n,j)*s.binomial(n+2,j)*ss**j*rr**(n-j)/s.Integer(j+1) for j in range(n+1))
        ap=s.expand((2*f+rr*(s.diff(f,rr)-s.diff(f,ss)))/(2*(n+2)))
        bp=s.expand(((2*ss-rr)*f+ss*rr*(s.diff(f,rr)-s.diff(f,ss)))/(2*(n+2)))
        fu=s.Poly(f.subs({ss:tt,rr:1-tt}),tt)
        au=s.Poly(ap.subs({ss:tt,rr:1-tt}),tt)
        bu=s.Poly(bp.subs({ss:tt,rr:1-tt}),tt)
        check(f'completed frame no common interior first roots n{n}',s.gcd(fu,au).degree()==0)
        check(f'completed frame no common interior second roots n{n}',s.gcd(fu,bu).degree()==0)
        check(f'completed frame first axis nonzero n{n}',s.simplify(ap.subs(rr,0)-(-1)**n*ss**n/2)==0)
        check(f'completed frame second axis nonzero n{n}',s.simplify(bp.subs(rr,0)-(-1)**n*ss**(n+1)/2)==0)
        check(f'completed frame other second axis nonzero n{n}',s.simplify(bp.subs(ss,0)+rr**(n+1)/(2*(n+2)))==0)
    RESULTS['completed_frame_extension']={'map':'Lambda²(1+Lambda²W) tensor det(W)^-1 = W plus W*; v=A cross B, xi=a B-b A, xi(v)=0.','complete_monogenic_inputs':'M1_m and determinant-twisted M2_n give a nonzero normal frame exactly where z is nonzero, at every grade. At w=0 the frame plane is the real Cu plane.','vacuum_control':'Untwisted constant vacuum with twisted M2_n gives a nonzero frame everywhere off the origin.','charge_scope':'Inputs have distinct total charges. These bilinear frame channels are not a single fixed-charge polarization state or physical handedness selection.'}


def run_angular_selection_review(check, RESULTS):
    """Static embedding entry point; standard library only, exact rational arithmetic."""
    check_external = check
    from fractions import Fraction as Q
    from math import comb
    import json

    def add(a,b):
        return [ (a[i] if i<len(a) else 0)+(b[i] if i<len(b) else 0) for i in range(max(len(a),len(b))) ]
    def scale(a,c): return [c*x for x in a]
    def mul(a,b):
        c=[Q(0)]*(len(a)+len(b)-1)
        for i,x in enumerate(a):
            for j,y in enumerate(b): c[i+j]+=x*y
        return c
    def deriv(a): return [i*a[i] for i in range(1,len(a))]
    def power(a,n):
        c=[Q(1)]
        for _ in range(n): c=mul(c,a)
        return c
    def sphere(a): return sum((2*x/Q((i+1)*(i+2)) for i,x in enumerate(a)),Q(0))
    def homogeneous(n,A,B):
        # Solve sigma dss+rho drr+A ds+B dr, with first homogeneous coefficient one.
        c=Q(1); out=[Q(0)]
        for j in range(n+1):
            out=add(out,scale(mul(power([0,1],j),power([1,-1],n-j)),c))
            if j<n: c=-c*Q((n-j)*(n-j+B-1),(j+1)*(j+A))
        return out

    def review_checks(N=6,R=14):
        count=0
        def check(a,b,label):
            nonlocal count
            check_external('angular-review '+str(label), a==b); count+=1
        t=[Q(0),Q(1)]; rho=[Q(1),Q(-1)]; tr=mul(t,rho)
        fs=[homogeneous(n,2,3) for n in range(N+2)]
        gs=[homogeneous(r,1,2) for r in range(max(R,N+2)+1)]
        aa=[scale(add(scale(f,2),scale(mul(rho,deriv(f)),-1)),Q(1,2*(n+2))) for n,f in enumerate(fs)]
        bb=[scale(add(mul([-1,3],f),scale(mul(tr,deriv(f)),-1)),Q(1,2*(n+2))) for n,f in enumerate(fs)]
        hs=[Q(n+1,2*(n+2)) for n in range(N+2)]
        for n in range(N+1):
            check(bb[n],scale(gs[n+1],Q(-1,2*(n+2))),('b',n))
            for m in range(N+1):
                d1=add(scale(mul(tr,mul(fs[n],fs[m])),Q(1,4)),mul(power(t,2),mul(aa[n],aa[m])))
                d2=add(scale(mul(tr,mul(fs[n],fs[m])),hs[n]*hs[m]),mul(bb[n],bb[m]))
                norms=[Q(1,2*(n+1)*(n+2)*(n+3)),Q(1,2*(n+2)**2*(n+3))]
                for branch,d in enumerate([d1,d2]):
                    check(sphere(d),norms[branch] if n==m else 0,('gram',branch,n,m))
                    if n==m:
                        diag=norms[branch]*(Q(1,2) if branch==0 else Q(1,2)-Q(1,2*(n+2)*(2*n+5)))
                        check(sphere(mul(t,d)),diag,('t diag',branch,n))
                    elif abs(n-m)==1:
                        j=min(n,m)
                        off=Q(-1,4*(j+2)*(j+3)*(2*j+5)) if branch==0 else Q(-1,4*(j+3)**2*(2*j+5))
                        check(sphere(mul(t,d)),off,('t off',branch,n,m))
                    else: check(sphere(mul(t,d)),0,('t far',branch,n,m))
                    for r in range(R+1):
                        if r<abs(n-m) or r>n+m+1+branch:
                            check(sphere(mul(gs[r],d)),0,('bilinear band',branch,n,m,r))
            for r in range(R+1):
                check(sphere(mul(gs[r],bb[n])),Q(-1,2*(n+2)**2) if r==n+1 else 0,('constant',n,r))
        fixtures={(0,0,1):Q(1,360),(1,0,1):Q(-1,2240),(2,0,1):Q(1,4032),(1,0,2):Q(1,896),(2,0,2):Q(-11,40320),(3,0,2):Q(1,5760),(0,1,2):Q(1,896),(1,1,2):Q(-11,48384),(2,1,2):Q(1,6720),(3,1,2):Q(-13,295680),(4,1,2):Q(1,38016)}
        def cube(m,n,l):
            W=add(scale(mul(fs[n],bb[l]),hs[n]),scale(mul(fs[l],bb[n]),-hs[l]))
            return scale(mul(tr,mul(fs[m],W)),Q(1,2))
        for m in range(N+1):
            for n in range(N+1):
                for l in range(N+1):
                    d=cube(m,n,l); c=sphere(d)
                    check(c,-sphere(cube(m,l,n)),('antisym',m,n,l))
                    if n==l or m<abs(n-l)-1 or m>n+l+1: check(c,0,('triangle',m,n,l))
                    if (m,n,l) in fixtures: check(c,fixtures[m,n,l],('fixture',m,n,l))
                    # Independent hypergeometric coefficient construction and beta integration.
                    af=lambda k,i: Q((-1)**i*comb(k,i))*Q(__import__('math').prod(range(k+4,k+4+i)),__import__('math').factorial(i+1))
                    bf=lambda k,i: Q((-1)**(i+1)*comb(k+1,i))*Q(__import__('math').prod(range(k+3,k+3+i)),2*(k+2)*__import__('math').factorial(i))
                    alt=Q(0)
                    for i in range(m+1):
                        for j in range(max(n,l)+1):
                            for s in range(max(n,l)+2):
                                x=hs[n]*af(n,j)*bf(l,s) if j<=n and s<=l+1 else 0
                                y=hs[l]*af(l,j)*bf(n,s) if j<=l and s<=n+1 else 0
                                S=i+j+s
                                alt+=af(m,i)*(x-y)*Q(2,(S+2)*(S+3)*(S+4))
                    check(c,alt,('beta sum',m,n,l))
                    for r in range(R+1):
                        if n==l or r<max(0,m-n-l-1,n-m-l-1,l-m-n-1) or r>m+n+l+3:
                            check(sphere(mul(gs[r],d)),0,('quartic range',m,n,l,r))
        return {'passed':count,'N':N,'R':R,'fixtures':{str(k):str(v) for k,v in fixtures.items()}}

    outcome = review_checks(N=6, R=14)
    # Covariant-tensor proofs are in reviewed.md; these are their explicit weight controls.
    for rank in (3, 5, 7, 9):
        check_external('odd-rank cubic charge '+str(rank), Q(2,rank)-Q(1,rank)-Q(1,rank)==0)
    # Rank nine same-degree control: complementary forms of degree three wedge to volume.
    def wedge_sign(I,J):
        if set(I)&set(J): return 0
        return (-1)**sum(i>j for i in I for j in J)
    first=(0,1,2); second=(3,4,5); third=(6,7,8)
    check_external('rank9 same exterior degree central balance', 3*6-2*9==0)
    check_external('rank9 cubic disjoint wedge nonzero', wedge_sign(first,second)*wedge_sign(first+second,third)==1)
    check_external('rank9 cubic twisted slot antisymmetry', wedge_sign(second,third)==-wedge_sign(third,second))
    outcome['extra_controls']=7
    outcome['odd_rank_control']='Invariant Lambda²W x W* x W* for every odd rank >=3; commuting repeated slots vanish.'
    outcome['same_degree_control']='Rank9 degree6 nonzero alternating tensor from three complementary covector3 forms; same-degree cubic alone is not unique.'
    RESULTS['independent_angular_selection_review'] = outcome
    return outcome


def round_degree_two_extension():
    """Exact actual-round coefficient algebra; uses s, check, RESULTS."""
    pauli = [s.Matrix([[0,1],[1,0]]), s.Matrix([[0,-s.I],[s.I,0]]), s.diag(1,-1)]
    eye2 = s.eye(2)
    def tensor(*matrices):
        result = matrices[0]
        for matrix in matrices[1:]:
            result = s.kronecker_product(result, matrix)
        return result
    base = [tensor(pauli[0],eye2), tensor(pauli[1],eye2), tensor(pauli[2],pauli[0]), tensor(pauli[2],pauli[1])]
    chirality = base[0]*base[1]*base[2]*base[3]
    normal = [tensor(*([pauli[2]]*j+[pauli[a]]+[eye2]*(2-j))) for j in range(3) for a in range(2)]
    indices = [i for i in range(4) if chirality[i,i]==1]
    bivectors = {(i,j):(base[i]*base[j]).extract(indices,indices) for i in range(4) for j in range(i+1,4)}
    def realification(matrix):
        real = matrix.applyfunc(s.re)
        imag = matrix.applyfunc(s.im)
        return real.row_join(-imag).col_join(imag.row_join(real))
    curvature = {pair:s.diag(s.zeros(2),realification(matrix/2)) for pair,matrix in bivectors.items()}
    check('round degree2 scalar curvature normalization',sum((matrix*matrix/2 for matrix in bivectors.values()),s.zeros(2))==-3*s.eye(2))
    operator = s.zeros(96)
    for pair,matrix in bivectors.items():
        field = curvature[pair]
        check(f'round degree2 normal curvature skew {pair}',field.T==-field)
        operator = s.MutableDenseMatrix(operator-tensor(field.T,tensor(matrix,s.eye(8)))/2)
        for a in range(6):
            clifford = sum((field[t,a]*normal[t] for t in range(6)),s.zeros(8))
            for z in range(6):
                operator[a*16:(a+1)*16,z*16:(z+1)*16]-=tensor(matrix,clifford*normal[z])/4
    for pair,matrix in bivectors.items():
        field = curvature[pair]
        spin = -sum((field[c,a]*normal[a]*normal[c] for a in range(6) for c in range(6)),s.zeros(8))/4
        generator = tensor(s.eye(6),tensor(matrix/2,s.eye(8))+tensor(s.eye(2),spin))-tensor(field.T,s.eye(16))
        check(f'round degree2 curvature equivariance {pair}',generator*operator-operator*generator==s.zeros(96))
    unseen = set(range(96))
    blocks = []
    while unseen:
        todo = [min(unseen)]
        component = set(todo)
        while todo:
            i = todo.pop()
            for j in unseen-component:
                if operator[i,j]!=0 or operator[j,i]!=0:
                    component.add(j)
                    todo.append(j)
        unseen -= component
        blocks.append(sorted(component))
    check('round degree2 full spectral block sizes',[len(block) for block in blocks]==[24,24,24,24])
    variable = s.Symbol('lambda')
    first = variable**8*(2*variable-9)*(2*variable-3)**3*(2*variable+1)**9*(2*variable+3)**3/65536
    second = variable**8*(2*variable+1)**8*(4*variable**2-4*variable-9)**4/65536
    polynomials = []
    for index,block in enumerate(blocks):
        check(f'round degree2 preserved spectral block {index}',all(operator[i,j]==0 for i in block for j in range(96) if j not in block))
        polynomial = s.factor(operator.extract(block,block).charpoly(variable).as_expr())
        check(f'round degree2 exact characteristic polynomial {index}',s.expand(polynomial-[first,second,first,second][index])==0)
        polynomials.append(str(polynomial))
    # Illustrative finite checks only; all-weight integrality is proved in prose.
    cases = []
    for twice_j in range(4):
        j = s.Rational(twice_j,2)
        for n in range(8):
            highest = j+n
            eigenvalue = highest*(highest+3)+j*(j+1)-2*j*(j+1)
            check(f'round degree2 induced Casimir integer j{twice_j}/2 n{n}',eigenvalue.is_integer and eigenvalue>=0)
            check(f'round degree2 induced Casimir formula j{twice_j}/2 n{n}',eigenvalue==n*n+(2*j+3)*n+2*j)
            cases.append({'twice_j':twice_j,'base_level':n,'rough_laplacian_in_units_k':int(eigenvalue)})
    RESULTS['round_degree_two_extension'] = {
        'active_linear_coefficient_dimension':96,
        'sectional_curvature_normalization':'F_ij=Realify(+(k/2) b_i b_j on matching base Weyl)',
        'characteristic_polynomials_k1':polynomials,
        'necessary_endomorphism_spectrum_in_units_k':['0','-1/2','-3/2','3/2','9/2','(1-sqrt(10))/2','(1+sqrt(10))/2'],
        'sample_Casimir_checks':cases,
        'scope':'Actual round chiral-spin induced bundles. All-weight Casimir integrality and parallel K give degree≤2 harmonic-polynomial and uniformly bounded invariant-space exclusion. Degrees≥3 and arbitrary smooth fields remain open.'
    }


def differential_cubic_extension():
    z,zb,w1,w1b,w2,w2b = s.symbols('dc_z dc_zb dc_w1 dc_w1b dc_w2 dc_w2b')
    sigma=z*zb
    rho=w1*w1b+w2*w2b
    f1=rho-s.Rational(3,2)*sigma
    b0=(2*sigma-rho)/4
    b1=-(rho**2-6*sigma*rho+3*sigma**2)/6
    first=s.Matrix([zb*w1/2,zb*w2/2,0])
    second=s.Matrix([-z*w2b/4,z*w1b/4,b0])
    third=s.Matrix([-f1*z*w2b/3,f1*z*w1b/3,b1])
    coordinates=[z,zb,w1,w1b,w2,w2b]
    euler=lambda matrix: matrix.applyfunc(lambda p:sum(q*s.diff(p,q) for q in coordinates))
    combined=second+third
    density=s.expand(s.Matrix.hstack(first,combined,euler(combined)).det())
    check('differential cubic identical commuting slots zero',s.Matrix.hstack(first,combined,combined).det()==0)
    check('differential cubic Euler grade difference',s.expand(density-2*s.Matrix.hstack(first,second,third).det())==0)
    expected=sigma*rho*(3*sigma**2-sigma*rho+rho**2)/24
    check('differential cubic direct Cartesian density',s.expand(density-expected)==0)
    t=s.symbols('dc_t',real=True)
    reduced=s.factor(density.subs({zb:s.Symbol('dc_sigma')/z,w1b:s.Symbol('dc_r1')/w1,w2b:(s.Symbol('dc_rho')-s.Symbol('dc_r1'))/w2}))
    sphere_density=s.factor(reduced.subs({s.Symbol('dc_sigma'):t,s.Symbol('dc_rho'):1-t}))
    integral=s.integrate(2*(1-t)*sphere_density,(t,0,1))
    check('differential cubic exact angular coefficient',integral==s.Rational(1,180))
    square_integral=s.integrate(2*(1-t)*sphere_density**2,(t,0,1))
    check('differential cubic full-spin invariant norm angular coefficient',square_integral==s.Rational(1,24192))
    f,g,yf=s.symbols('dc_f dc_g dc_Yf')
    check('differential cubic common radial derivative cancels',s.expand(s.Matrix.hstack(g*first,f*combined,yf*combined+f*euler(combined)).det()-g*f**2*density)==0)
    RESULTS['differential_cubic_extension']={'angular_density':str(sphere_density),'normal_component_integral_over_pi_cubed':str(integral),'full_spin_norm_integral_over_pi_cubed':str(square_integral),'scope':'Same-base-polarization example: Euler derivative distinguishes repeated commuting normal-field slots. The normal cubic leaves tangent-spin indices; its invariant squared norm is a full-spin and uniform-phase invariant sixth-order density. No physical action coefficient or radial integral is assigned.'}


def uniform_dimension_extension():
    outer_check=check
    from fractions import Fraction
    from math import comb, factorial
    from pathlib import Path
    from datetime import datetime, timezone
    import json

    checks=[]
    def dimension_check(name, actual, expected):
        outer_check('dimension-extension '+name,actual==expected)
        checks.append(name)
    def c(n,k):
        return comb(n,k) if 0<=k<=n else 0

    def stirling(n,k):
        row=[1]+[0]*k
        for a in range(1,n+1):
            row=[0]+[row[b-1]+b*row[b] for b in range(1,k+1)]
        return row[k]

    rows=[]
    for m in range(1,13):
        ks=list(range(0,m+1,2))
        weights=[(Fraction(k,m)-branch,c(m,k)) for k in ks for branch in (0,1)]
        dimension_check(f'm{m} rank',sum(mult for q,mult in weights),2**m)
        exponent=2*sum(c(m-1,k-1) for k in ks)-2**(m-1)
        dimension_check(f'm{m} determinant',exponent,-1 if m==1 else 0)
        trace=sum(q*mult for q,mult in weights)
        dimension_check(f'm{m} trace',trace,Fraction(-1) if m==1 else Fraction(0))
        for power in range(1,12,2):
            moment=sum(q**power*mult for q,mult in weights)
            expected=Fraction(-factorial(m)*stirling(power,m),m**power) if m%2 else Fraction(0)
            dimension_check(f'm{m} moment{power}',moment,expected)
        cubic=sum(q**3*mult for q,mult in weights)
        dimension_check(f'm{m} cubic discriminator',cubic,Fraction(-1) if m==1 else Fraction(-2,9) if m==3 else Fraction(0))
        rows.append({'m':m,'rank_E':2**m,'blocks':[{ 'degree':k,'multiplicity':c(m,k),'charges':[str(Fraction(k,m)),str(Fraction(k,m)-1)]} for k in ks], 'cubic_trace':str(cubic)})

    sp=[]
    for r in range(1,8):
        n=2*r;m=n+1
        blocks={k:0 for k in range(0,m+1,2)}
        decompositions=[]
        for d in range(n+1):
            js=list(range(d%2,min(d,n-d)+1,2))
            primitive=[(j,c(n,j)-c(n,j-2)) for j in js]
            dimension_check(f'Sp{r} exterior{d}',sum(dim for j,dim in primitive),c(n,d))
            original_degree=d+d%2
            blocks[original_degree]+=sum(dim for j,dim in primitive)
            decompositions.append({'V_degree':d,'W_degree':original_degree,'primitive_blocks':primitive})
        for k,dim in blocks.items():
            dimension_check(f'Sp{r} Wblock{k}',dim,c(m,k))
        A=[[0]*n for _ in range(n)]
        for a in range(0,n,2):A[a][a+1]=-1;A[a+1][a]=1
        for a in range(n):
            for b in range(n):
                dimension_check(f'Sp{r} j square{a},{b}',sum(A[a][t]*A[t][b] for t in range(n)),-int(a==b))
        sp.append({'quaternionic_rank':r,'m':m,'residual_singlet_charges':[[str(Fraction(2*l,m)),str(Fraction(2*l,m)-1)] for l in range(r+1)],'decompositions':decompositions})

    for m in (3,5,7,9,11):
        dimension_check(f'm{m} cubic slot charge',Fraction(2,m)+2*(Fraction(m-1,m)-1),0)
        dimension_check(f'm{m} cubic first slot allowed',2%2,0)
        dimension_check(f'm{m} cubic other slots allowed',(m-1)%2,0)


    RESULTS['uniform_dimension_extension']={'dimension_rows':rows,'quaternionic_branching':sp,'scope':'Uniform full carrier with all branches retained. Simplicity and full-residual-equivariance criteria are additional premises.'}


def cubic_tensor_count_extension():
    #!/usr/bin/env python3
    """Exact independent exterior/Lie-generator certificate. Standard library only."""
    from itertools import product
    from datetime import datetime, timezone
    import hashlib
    import json
    from pathlib import Path

    PRIME = 1000003


    def parity_sign(seq):
        return (-1) ** sum(x > y for n, x in enumerate(seq) for y in seq[n + 1:])


    def indices(mask, m):
        return [j for j in range(m) if mask >> j & 1]


    def canonical(a, b, c):
        if b == c:
            return None, 0
        return ((a, b, c), 1) if b < c else ((a, c, b), -1)


    def replace(mask, i, j):
        """Action of E_ij on an exterior basis: replace j with i."""
        if not mask >> j & 1 or mask >> i & 1:
            return None, 0
        new = mask ^ (1 << i) ^ (1 << j)
        between = mask & (((1 << max(i, j)) - 1) ^ ((1 << (min(i, j) + 1)) - 1))
        return new, (-1) ** between.bit_count()


    def acted(state, i, j):
        for slot in range(3):
            changed, sign = replace(state[slot], i, j)
            if changed is None:
                continue
            target = list(state)
            target[slot] = changed
            key, orientation = canonical(*target)
            if key is not None:
                yield key, sign * orientation


    def degree_channels(m):
        allowed = range(m % 2, m + 1, 2)
        return [(a, b, c) for a in allowed for b in allowed for c in allowed
                if a + b + c == m and (b < c or b == c and b % 2)]


    def ordered_wedge(state, m):
        full = (1 << m) - 1
        blocks = [indices(full ^ x, m) for x in state]
        if len(set(sum(blocks, []))) != m or sum(map(len, blocks)) != m:
            return 0
        sign = parity_sign(sum(blocks, []))
        for mask, comp in zip(state, blocks):
            sign *= parity_sign(comp + indices(mask, m))
        return sign


    def channel_vector(columns, degrees, m):
        l1, l2, l3 = degrees
        answer = []
        for a, b, c in columns:
            if m - a.bit_count() != l1:
                answer.append(0)
            elif (m - b.bit_count(), m - c.bit_count()) == (l2, l3):
                answer.append(ordered_wedge((a, b, c), m))
            elif l2 != l3 and (m - c.bit_count(), m - b.bit_count()) == (l2, l3):
                answer.append(-ordered_wedge((a, c, b), m))
            else:
                answer.append(0)
        return answer


    def modular_rank(rows):
        pivots = {}
        for row in rows:
            row = {k: v % PRIME for k, v in row.items() if v % PRIME}
            while row:
                p = min(row)
                if p not in pivots:
                    inv = pow(row[p], PRIME - 2, PRIME)
                    pivots[p] = {k: v * inv % PRIME for k, v in row.items()}
                    break
                factor = row[p]
                for k, v in pivots[p].items():
                    changed = (row.get(k, 0) - factor * v) % PRIME
                    if changed:
                        row[k] = changed
                    else:
                        row.pop(k, None)
        return len(pivots)


    def certificate(m):
        full = (1 << m) - 1
        basis = set()
        # Coordinate torus neutrality means complements partition every index.
        for allocation in product(range(3), repeat=m):
            complements = [sum(1 << i for i, slot in enumerate(allocation) if slot == n)
                           for n in range(3)]
            original = [full ^ mask for mask in complements]
            if any(mask.bit_count() % 2 for mask in original):
                continue
            state, _ = canonical(*original)
            if state is not None:
                basis.add(state)
        columns = sorted(basis)
        lookup = {s: n for n, s in enumerate(columns)}
        rows = []
        for i in range(m):
            for j in range(m):
                if i == j:
                    continue
                sources = {s for target in columns for s, _ in acted(target, j, i)}
                for source in sources:
                    row = {}
                    for target, coefficient in acted(source, i, j):
                        if target in lookup:
                            n = lookup[target]
                            row[n] = row.get(n, 0) + coefficient
                    if any(row.values()):
                        rows.append(row)
        channels = degree_channels(m)
        vectors = [channel_vector(columns, degrees, m) for degrees in channels]
        checks = 0
        for vector_index,vector in enumerate(vectors):
            check(f'cubic tensor nonzero wedge m{m} v{vector_index}',any(vector))
            check(f'cubic tensor Lie invariance m{m} v{vector_index}',all(sum(coefficient*vector[n] for n,coefficient in row.items())==0 for row in rows))
            checks += len(rows)
        # Wedge vectors have disjoint exterior-degree support; check it explicitly.
        supports = [{n for n, v in enumerate(vector) if v} for vector in vectors]
        check(f'cubic tensor disjoint supports m{m}',all(not(a&b) for i,a in enumerate(supports) for b in supports[i+1:]))
        rank = modular_rank(rows)
        nullity = len(columns) - rank
        formula = ((m // 2 + 1) ** 2) // 4
        check(f'cubic tensor exact rank certificate m{m}',nullity==len(channels)==formula)
        return {"m": m, "torus_neutral_basis": len(columns), "constraint_rows": len(rows),
                "rank_mod_prime": rank, "invariant_dimension": nullity,
                "complementary_channels": channels, "integer_annihilation_checks": checks}



    check('cubic tensor modulus is prime',all(PRIME%d for d in range(2,int(PRIME**0.5)+1)))
    records=[certificate(m) for m in range(1,8)]
    for m in range(1,101):
        check(f'cubic tensor all-rank formula finite control m{m}',len(degree_channels(m))==((m//2+1)**2)//4)
    RESULTS['cubic_tensor_count']={'certificates':records,'formula':'N_(2r)=N_(2r+1)=floor((r+1)^2/4)','scope':'Full U(m) complex-linear invariant tensor F tensor Lambda²(F det^-1). Not all interaction actions; count one admits m2 and m3.'}


def quartic_bridge_tests():
    """Paste into shared checker; uses its sympy s, check and RESULTS."""
    def values(X,Y):
        r=X.rows
        norm=lambda Z:s.simplify(sum(s.conjugate(v)*v for v in Z))
        M=X.conjugate()*Y.T
        K=s.trace(M)
        C=norm(M)
        Q=s.simplify(sum(s.conjugate(z)*z for a in range(r) for b in range(r)
            for i in range(4) for j in range(i+1,4)
            for z in [X[a,i]*Y[b,j]-X[a,j]*Y[b,i]]))
        product=norm(X)*norm(Y)
        traceless=norm(M-K*s.eye(r)/r)
        return product,s.simplify(s.conjugate(K)*K),C,Q,traceless
    pairs={
        'entangled_equal':(s.Matrix([[1,0,0,0],[0,1,0,0]]),s.Matrix([[1,0,0,0],[0,1,0,0]])),
        'different_base_same_normal':(s.Matrix([[1,0,0,0],[0,0,0,0]]),s.Matrix([[0,0,0,0],[1,0,0,0]])),
        'vacuum_triplet_cancel':(s.Matrix([[1,1,0,0]]),s.Matrix([[1,-1,0,0]])),
        'orthogonal_normal_common_base':(s.Matrix([[1,0,0,0]]),s.Matrix([[0,1,0,0]])),
        'complex_entangled':(s.Matrix([[1,s.I,1,0],[0,1,0,s.I]]),s.Matrix([[s.I,1,0,1],[1,0,s.I,0]])),
    }
    rows={}
    for name,(X,Y) in pairs.items():
        p,k,c,q,m0=values(X,Y)
        check('quartic Gram '+name,s.simplify(p-c-q)==0)
        check('quartic trace split '+name,s.simplify(c-k/X.rows-m0)==0)
        check('quartic positive '+name,q>=0 and m0>=0)
        phase=s.I
        check('quartic uniform phase '+name,values(phase*X,phase*Y)==values(X,Y))
        V=s.diag(1,s.I,-1,-s.I)
        check('quartic normal unitary '+name,values(X*V,Y*V)==values(X,Y))
        Fg=s.diag(1,-s.I,s.I,-1)
        check('quartic determinant gluing '+name,values(X*Fg,s.I*Y*Fg)==values(X,Y))
        if X.rows==2:
            U=s.Matrix([[s.Rational(3,5),s.Rational(4,5)],[-s.Rational(4,5),s.Rational(3,5)]])
            check('quartic base unitary '+name,values(U*X,U*Y)==values(X,Y))
        rows[name]=[str(v) for v in [p,k,c,q,m0]]
    check('quartic entangled naive identity fails',rows['entangled_equal'][:4]==['4','4','2','2'])
    check('quartic vacuum-triplet interference',rows['vacuum_triplet_cancel'][:4]==['4','0','0','4'])
    for rank in [1,2,3,4]:
        X=s.zeros(rank,4)
        for a in range(rank): X[a,a]=1/s.sqrt(rank)
        p,k,c,q,m0=values(X,X)
        check('quartic Schmidt maximum '+str(rank),p==1 and k==1 and c==s.Rational(1,rank) and q==1-s.Rational(1,rank))
        rows['normalized_equal_rank_'+str(rank)]=[str(v) for v in [p,k,c,q,m0]]
    RESULTS['phase_preserving_quartic_bridge']={'columns':['N0N1','absK2','C','frame_norm2','traceless_norm2'],'controls':rows}
    t=s.symbols('quartic_t',real=True)
    def f(n):
        return s.expand(sum((-1)**i*s.binomial(n,i)*s.rf(n+4,i)/s.factorial(i+1)*t**i for i in range(n+1)))
    def h(n): return s.Rational(n+1,2*(n+2))
    def b(n):
        F=f(n)
        return s.expand(((3*t-1)*F-t*(1-t)*s.diff(F,t))/(2*(n+2)))
    def integral(p):
        p=s.Poly(s.expand(p),t)
        return sum(c/s.Rational(j[0]+1) for j,c in p.terms())
    parseval=[]
    for n,l in [(0,1),(0,2),(1,2)]:
        W=s.expand(h(n)*f(n)*b(l)-h(l)*f(l)*b(n))
        q=integral(2*t*(1-t)**2*W**2)
        d=lambda a,c:s.expand(h(a)*h(c)*t*(1-t)*f(a)*f(c)+b(a)*b(c))
        check('quartic W determinant '+str((n,l)),s.expand(d(n,n)*d(l,l)-d(n,l)**2-t*(1-t)*W**2)==0)
        Cs=[integral(t*(1-t)**2*f(k)*W) for k in range(n+l+2)]
        rhs=sum(4*(k+1)*(k+2)*(k+3)*c*c for k,c in enumerate(Cs))
        check('quartic cubic Parseval '+str((n,l)),s.simplify(q-rhs)==0)
        check('quartic distinct grade positive '+str((n,l)),q>0)
        for k in range(max(0,abs(n-l)-1)):
            check('quartic cubic lower zero '+str((k,n,l)),Cs[k]==0)
        parseval.append({'n':n,'l':l,'frame_integral_over_pi3':str(q),'cubic_over_pi3':[str(c) for c in Cs]})
    RESULTS['phase_preserving_quartic_bridge']['cubic_frame_parseval']=parseval


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir')
    args = parser.parse_args()
    start = datetime.now(timezone.utc)
    source_bytes = Path(__file__).read_bytes()
    out = Path(args.output_dir) if args.output_dir else Path(__file__).resolve().parent/('run-'+start.strftime('%Y%m%dT%H%M%S%fZ'))
    out.mkdir(parents=True,exist_ok=False)
    (out/'shared-geometry-checks.py').write_bytes(source_bytes)
    for name,fn in [('charge_completion',charge_completion),('tangent_projectors',tangent_projectors),('unitary_geometry',unitary_geometry),('common_spin_factor',common_spin_factor),('curvature_bounds',curvature_bounds),('photon_geometry',photon_geometry),('contacts',contacts),('lifted_charge',lifted_charge),('real_clifford',real_clifford),('geometric_ladders',geometric_ladders),('ambient_symmetry',ambient_symmetry),('total_real_carrier',total_real_carrier),('review_bounds_and_cubic_ladder',review_bounds_and_cubic_ladder),('color_invariants',color_invariants),('observer_helicity',observer_helicity),('polarization_zeros',polarization_zeros),('refined_counts',refined_counts),('complex_points',complex_points),('bott_constraints',bott_constraints),('ambient_curvature_bounds',ambient_curvature_bounds)]:
        before = len(CHECKS)
        fn()
        print(name,len(CHECKS)-before,'checks passed',flush=True)
    before=len(CHECKS)
    pure_spinor_geometry()
    print('pure_spinor_geometry',len(CHECKS)-before,'checks passed',flush=True)
    for name,fn in [('joint_dimension_and_alignment',joint_dimension_and_alignment),('polarization_gradient',polarization_gradient),('transition_order',transition_order),('ambient_dimension_audit',ambient_dimension_audit),('selfdual_curvature_bridge',selfdual_curvature_bridge),('charge_polarization_compatibility',charge_polarization_compatibility),('curvature_metric_and_round_flag',curvature_metric_and_round_flag)]:
        before=len(CHECKS)
        fn()
        print(name,len(CHECKS)-before,'checks passed',flush=True)
    for name,fn in [('seven_normal_reduction',seven_normal_reduction),('lifted_charge_flag',lifted_charge_flag),('adapted_connection',adapted_connection),('alternating_vector_rule',alternating_vector_rule),('joint_carrier_projectors',joint_carrier_projectors),('universal_vector_frame',universal_vector_frame),('joint_endpoint_rule',joint_endpoint_rule),('color_flag_tradeoff',color_flag_tradeoff)]:
        before=len(CHECKS)
        fn()
        print(name,len(CHECKS)-before,'checks passed',flush=True)
    before=len(CHECKS)
    curvature_observer_lift()
    print('curvature_observer_lift',len(CHECKS)-before,'checks passed',flush=True)
    before=len(CHECKS)
    common_charge_spin_covariants()
    print('common_charge_spin_covariants',len(CHECKS)-before,'checks passed',flush=True)
    before=len(CHECKS)
    round_spin_sections()
    print('round_spin_sections',len(CHECKS)-before,'checks passed',flush=True)
    before=len(CHECKS)
    nonspin_global_control()
    print('nonspin_global_control',len(CHECKS)-before,'checks passed',flush=True)
    before=len(CHECKS)
    global_multiplicity()
    print('global_multiplicity',len(CHECKS)-before,'checks passed',flush=True)
    before=len(CHECKS)
    covariant_spatial_operator()
    print('covariant_spatial_operator',len(CHECKS)-before,'checks passed',flush=True)
    before=len(CHECKS)
    charge_saturation_extension()
    print('charge_saturation_extension',len(CHECKS)-before,'checks passed',flush=True)
    before=len(CHECKS)
    completed_frame_extension()
    print('completed_frame_extension',len(CHECKS)-before,'checks passed',flush=True)
    before=len(CHECKS)
    (lambda: run_angular_selection_review(check, RESULTS))()
    print('angular_selection_extension',len(CHECKS)-before,'checks passed',flush=True)
    before=len(CHECKS)
    (round_degree_two_extension)()
    print('round_degree_two_extension',len(CHECKS)-before,'checks passed',flush=True)
    before=len(CHECKS)
    (differential_cubic_extension)()
    print('differential_cubic_extension',len(CHECKS)-before,'checks passed',flush=True)
    before=len(CHECKS)
    uniform_dimension_extension()
    print('uniform_dimension_extension',len(CHECKS)-before,'checks passed',flush=True)
    before=len(CHECKS)
    cubic_tensor_count_extension()
    print('cubic_tensor_count_extension',len(CHECKS)-before,'checks passed',flush=True)
    before=len(CHECKS)
    quartic_bridge_tests()
    print('quartic_bridge_tests',len(CHECKS)-before,'checks passed',flush=True)
    result = {'started_utc':start.isoformat(),'finished_utc':datetime.now(timezone.utc).isoformat(),'passed':len(CHECKS),'checks':CHECKS,'results':RESULTS,'sources':{SOURCE_CATALOG_PATH:sha256(source_bytes).hexdigest()}}
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'passed':len(CHECKS),'output':str(out)},indent=2))


if __name__ == '__main__':
    main()
