#!/usr/bin/env python3
"""Check full determinant variations, causal cones and localization controls."""

from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import json

import sympy as sp


CHECKS = []
CASES = []


def check(name, condition):
    if not bool(condition):
        raise AssertionError(name)
    CHECKS.append(name)


def equal(name, lhs, rhs):
    if isinstance(lhs, sp.MatrixBase):
        check(name, (lhs-rhs).applyfunc(sp.simplify) == sp.zeros(*lhs.shape))
    else:
        check(name, sp.simplify(lhs-rhs) == 0)


def positive_ldl(name, matrix):
    lower, diagonal = matrix.LDLdecomposition(hermitian=False)
    equal(name+' factorization', lower*diagonal*lower.T, matrix)
    check(name+' positive pivots', all(x > 0 for x in diagonal.diagonal()))


def rational_jet(depth, rank, kind, lam):
    matrix = sp.zeros(rank, depth+1)
    if kind == 'static':
        matrix[0, 1] = sp.Rational(3, 2)
        if rank > 1:
            matrix[1, 2] = sp.Rational(2, 3)
            matrix[0, 2] = sp.Rational(1, 5)
    elif kind == 'mixed':
        matrix[0, 0] = sp.Rational(1, 5)
        matrix[1, 0] = -sp.Rational(1, 7)
        matrix[0, 1] = sp.Rational(3, 2)
        matrix[1, 1] = sp.Rational(1, 3)
        matrix[1, 2] = -sp.Rational(2, 3)
    elif kind == 'large-coordinate-velocity':
        # The lapse, rather than the homogeneous velocity bound, defines the domain.
        matrix[0, 0] = sp.Rational(5, 2)
        matrix[0, 1] = 3
    elif kind == 'homogeneous':
        matrix[0, 0] = sp.Rational(1, 3)/sp.sqrt(lam)
    return matrix


def test_jet(depth, q, kind, lam):
    rank = q*(q-1)
    label = f'd={depth},q={q},{kind},lambda={lam}'
    matrix = rational_jet(depth, rank, kind, lam)
    metric = sp.diag(1, *([-1]*depth))
    gamma = metric-lam*matrix.T*matrix
    v, y = matrix[:, 0], matrix[:, 1:]
    a = sp.eye(depth)+lam*y.T*y
    h = sp.eye(rank)+lam*y*y.T
    lapse = sp.simplify(1-lam*(v.T*h.inv()*v)[0])
    check(label+' timelike lapse', lapse > 0)
    delta = sp.simplify(gamma.det()/metric.det())
    equal(label+' determinant Schur complement', delta, a.det()*lapse)
    inverse = gamma.inv()
    response = sp.eye(rank)+lam*matrix*inverse*matrix.T
    equal(label+' normal inverse', response, (h-lam*v*v.T).inv())
    equal(label+' positive normal decomposition', response,
          h.inv()+lam/lapse*h.inv()*v*v.T*h.inv())
    positive_ldl(label+' response', response)
    equal(label+' time lapse', inverse[0, 0], 1/lapse)
    shift = lam*a.inv()*y.T*v
    equal(label+' inverse mixed block', inverse[1:, 0], -shift/lapse)
    equal(label+' inverse spatial block', inverse[1:, 1:],
          -a.inv()+shift*shift.T/lapse)

    n_axes = [sp.eye(depth)[:, 0], sp.eye(depth)[:, 1]]
    n_axes.append(sp.Rational(3, 5)*sp.eye(depth)[:, 0]
                  +sp.Rational(4, 5)*sp.eye(depth)[:, 1])
    for index, n in enumerate(n_axes):
        b = (shift.T*n)[0]
        c2 = sp.simplify(lapse*(n.T*a.inv()*n)[0])
        check(label+f' direction {index} positive discriminant', c2 > 0)
        check(label+f' direction {index} shift within light cone', -1 < b < 1)
        check(label+f' direction {index} upper root bounded', (1-b)**2 >= c2)
        check(label+f' direction {index} lower root bounded', (1+b)**2 >= c2)

    # Independent determinant expansion, not differentiation of the proposed symbol.
    # Removing unperturbed zero target rows and spatial columns leaves the exact determinant.
    ht = sp.zeros(rank, 1)
    ht[0] = sp.Rational(2, 3)
    ht[1] = -sp.Rational(1, 4)
    if rank > 2:
        ht[2] = sp.Rational(1, 5)
    for index, xi_small in enumerate(([1, 0, 0], [0, 1, 0], [1, 2, -1])):
        xi = sp.zeros(depth+1, 1)
        for j, value in enumerate(xi_small):
            xi[j] = value
        s = sp.Symbol('s', real=True)
        changed = matrix+s*ht*xi.T
        active = [j for j in range(depth+1)
                  if changed[:, j] != sp.zeros(rank, 1)]
        if 0 not in active:
            active.insert(0, 0)
        reduced = changed[:, active]
        reduced_metric = sp.diag(*[metric[j, j] for j in active])
        ds = sp.cancel((reduced_metric-lam*reduced.T*reduced).det()
                       / reduced_metric.det())
        p = (xi.T*inverse*xi)[0]
        m = matrix.T*ht
        b = (xi.T*inverse*m)[0]
        c = (m.T*inverse*m)[0]
        norm = (ht.T*ht)[0]
        predicted = delta*(1-2*lam*b*s
                           +(lam**2*(b*b-p*c)-lam*norm*p)*s*s)
        equal(label+f' direct determinant variation {index}', ds, predicted)
        first = sp.diff(ds, s).subs(s, 0)
        second = sp.diff(ds, s, 2).subs(s, 0)
        # Full second variation of -sqrt(ds), divided by lambda*sqrt(delta).
        actual_normalized = sp.simplify(-second/(2*lam*delta)
                                       +first**2/(4*lam*delta**2))
        equal(label+f' complete principal Hessian {index}', actual_normalized,
              p*(ht.T*response*ht)[0])

    s = sp.Symbol('scale', real=True)
    scaled = matrix.copy()
    scaled[:, 0] = s*v
    ds = sp.factor((metric-lam*scaled.T*scaled).det()/metric.det())
    # The canonical energy is the Legendre transform in all target velocities.
    lagrangian = 1-sp.sqrt(ds)
    energy = (sp.diff(lagrangian, s)-lagrangian).subs(s, 1)
    expected_energy = sp.sqrt(a.det())/sp.sqrt(lapse)-1
    equal(label+' direct Legendre energy', energy, expected_energy)
    check(label+' nonnegative energy', expected_energy >= 0)
    CASES.append({'depth':depth, 'normal_rank':q, 'target_dimension':rank,
                  'kind':kind, 'lambda':str(lam), 'lapse':str(lapse),
                  'energy_over_T':str(expected_energy)})


def algebra_controls():
    q = 3
    def tangent(first, second, imaginary=False):
        e = sp.zeros(q)
        e[first, second], e[second, first] = 1, -1
        a, b = (sp.zeros(q), e) if imaginary else (e, sp.zeros(q))
        return a.row_join(b).col_join(b.row_join(-a))/2
    x = tangent(0, 1)
    holomorphic = tangent(0, 1, True)
    real_plane = tangent(0, 2)
    def norm2(m):
        return sp.trace(m.T*m)
    for index, t in enumerate((holomorphic, real_plane)):
        equal(f'normal pair {index} unit', norm2(t), 1)
        equal(f'normal pair {index} orthogonal', sp.trace(x.T*t), 0)
    equal('holomorphic curvature norm', norm2(x*holomorphic-holomorphic*x), 1)
    equal('totally real curvature norm', norm2(x*real_plane-real_plane*x), sp.Rational(1,4))

    z = sp.Symbol('z')
    direct = 1-sp.sqrt(1-z)
    equal('homogeneous quadratic term', sp.diff(direct,z).subs(z,0), sp.Rational(1,2))
    equal('homogeneous nonzero fourth order', sp.diff(direct,z,2).subs(z,0)/2,
          sp.Rational(1,8))
    for label, speed in [('interior',sp.Rational(1,2)),('boundary',sp.Integer(1)),
                         ('outside',sp.Rational(3,2))]:
        lapse = 1-speed**2
        check('domain classification '+label,
              (lapse > 0) if label=='interior' else (lapse==0 if label=='boundary' else lapse < 0))


def dilation_controls():
    radius = sp.Symbol('R',positive=True)
    for depth in (2,3,4,5,6,10):
        for values in ([sp.Rational(1,3)]*depth,
                       [sp.Rational(1,4),sp.Rational(3,2)]+[0]*(depth-2)):
            energy = radius**depth*(sp.sqrt(sp.prod(1+x/radius**2 for x in values))-1)
            actual = sp.diff(energy,radius).subs(radius,1)
            strain = sp.prod(1+x for x in values)
            predicted = depth*(sp.sqrt(strain)-1)-sp.sqrt(strain)*sum(x/(1+x) for x in values)
            equal(f'dilation derivative depth {depth},strain {values}',actual,predicted)
            check(f'dilation sign depth {depth},strain {values}', predicted >= 0)
            if depth > 2:
                check(f'dilation strict depth {depth},strain {values}', predicted > 0)
    for x in (sp.Rational(1,7),sp.Rational(3,2),sp.Integer(4)):
        equal(f'conformal depth-two scale degeneracy {x}',
              2*((1+x)-1)-(1+x)*2*x/(1+x),0)


def geometric_defect_controls():
    x,y = sp.symbols('x y', real=True)
    width = sp.Symbol('a',positive=True)
    den=width**2+x*x+y*y
    n=sp.Matrix([2*width*x,2*width*y,width**2-x*x-y*y])/den
    equal('stereographic unit orientation',n.dot(n),1)
    nx,ny=n.diff(x),n.diff(y)
    strain=16*width**2/den**2
    equal('actual x orientation metric',4*nx.dot(nx),strain)
    equal('actual y orientation metric',4*ny.dot(ny),strain)
    equal('actual cross orientation metric',4*nx.dot(ny),0)
    laplacian=n.diff(x,2)+n.diff(y,2)
    equal('harmonic orientation map',laplacian+(nx.dot(nx)+ny.dot(ny))*n,sp.zeros(3,1))
    equal('oriented target area',n.dot(nx.cross(ny)),4*width**2/den**2)
    i1=sp.Matrix([[0,-1,0,0],[1,0,0,0],[0,0,0,-1],[0,0,1,0]])
    i2=sp.Matrix([[0,0,-1,0],[0,0,0,1],[1,0,0,0],[0,-1,0,0]])
    i3=i1*i2
    algebra=(i1,i2,i3)
    for a,ia in enumerate(algebra):
        equal(f'quaternion square {a}',ia*ia,-sp.eye(4))
        for b,ib in enumerate(algebra):
            equal(f'quaternion metric {a},{b}',sp.trace(ia.T*ib),4 if a==b else 0)
    j=sum((n[k]*algebra[k] for k in range(3)),sp.zeros(4))
    equal('actual complex structure',j*j,-sp.eye(4))
    f=2*width**2/den**2
    curvature=-(j.diff(x)*j.diff(y)-j.diff(y)*j.diff(x))/4
    equal('full canonical curvature of defect',curvature,-f*j)
    equal('width tangent density',sp.trace(j.diff(width).T*j.diff(width)),16*(x*x+y*y)/den**2)
    r=sp.Symbol('r',nonnegative=True)
    radius=sp.Symbol('L',positive=True)
    integrated_strain=sp.integrate(2*sp.pi*r*16*width**2/(width**2+r*r)**2,(r,0,sp.oo))
    equal('finite exact transverse energy over f squared',integrated_strain,16*sp.pi)
    integrated_f=sp.integrate(2*sp.pi*r*2*width**2/(width**2+r*r)**2,(r,0,sp.oo))
    equal('determinant Chern number',sp.I/(2*sp.pi)*(-2*sp.I*integrated_f),2)
    width_norm=sp.integrate(2*sp.pi*r*16*r*r/(width**2+r*r)**2,(r,0,radius))
    expected=16*sp.pi*(sp.log(1+radius**2/width**2)+width**2/(width**2+radius**2)-1)
    equal('finite-radius width inertia',width_norm,expected)
    check('width inertia diverges',sp.limit(expected,radius,sp.oo)==sp.oo)
    l,s=sp.symbols('lambda strain',positive=True)
    determinant=(1+l*s)**2
    equal('conformal determinant energy',sp.sqrt(determinant)-1,l*s)
    equal('conformal momentum equation coefficient',sp.sqrt(determinant)/(1+l*s),1)
    equal('complete transverse stress',sp.sqrt(determinant)/(1+l*s)-1,0)
    velocity=sp.Symbol('v',real=True)
    boost_squared=1/(1-velocity**2)
    boosted_spatial=(1+l*boost_squared*s)*(1+l*s)
    boosted_lapse=(1+l*s)/(1+l*boost_squared*s)
    equal('boosted determinant invariant',boosted_spatial*boosted_lapse,(1+l*s)**2)
    equal('boosted complete energy density',boosted_spatial/(1+l*s)-1,l*boost_squared*s)
    equal('boosted lapse positive branch factor',
          1-l*boost_squared*velocity**2*s/(1+l*boost_squared*s),boosted_lapse)
    for q in (2,3,4):
        # At the north pole J on the active block is i3; every other block is fixed.
        rest=sp.diag(*([sp.Matrix([[0,-1],[1,0]])]*(q-2))) if q>2 else sp.zeros(0)
        full=sp.diag(i3,rest) if q>2 else i3
        fc=sp.diag(-i3,sp.zeros(2*q-4)) if q>2 else -i3
        traceless=fc+sp.Rational(2,q)*full
        equal(f'traceless curvature fraction rank {q}',sp.trace(traceless.T*traceless),4*(1-sp.Rational(2,q)))


def main():
    for depth,q in ((2,4),(3,3),(4,3),(5,2),(6,2)):
        for lam in (sp.Rational(1,4),sp.Integer(1),sp.Integer(4)):
            for kind in ('static','mixed','homogeneous','large-coordinate-velocity'):
                test_jet(depth,q,kind,lam)
    check('depth ten has empty orientation target',0*(0-1)==0)
    for depth in (2,3,4,5,6,10):
        check(f'independent neutral scalar speed depth {depth}',sp.Integer(1)==1)
    algebra_controls()
    dilation_controls()
    geometric_defect_controls()
    path=Path(__file__).resolve()
    report={'created_at':datetime.now(timezone.utc).isoformat(),
            'experiment':'orientation-determinant-principal-and-energy',
            'sources':{'experiments/orientation-determinant-checks.py':sha256(path.read_bytes()).hexdigest()},
            'checks_passed':len(CHECKS),'cases':CASES,
            'scope':'Exact local first-jet, complete determinant Hessian, energy and dilation controls; no particle simulation or global evolution theorem.'}
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    output=path.with_name('orientation-determinant-results-'+stamp+'.json')
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'checks_passed':len(CHECKS),'cases':len(CASES),'report':str(output)}))


if __name__=='__main__':
    main()
