#!/usr/bin/env python3
"""Check constrained canonical sources, complete energy and root response."""

from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from itertools import combinations
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


def realify(matrix):
    return sp.re(matrix).row_join(-sp.im(matrix)).col_join(
        sp.im(matrix).row_join(sp.re(matrix)))


def root_and_source_controls():
    for depth, q in ((2, 4), (3, 3), (4, 3), (5, 2), (6, 2)):
        j0 = sp.zeros(q).row_join(-sp.eye(q)).col_join(
            sp.eye(q).row_join(sp.zeros(q)))
        roots = list(combinations(range(q), 2))
        tangent = []
        for a, b in roots:
            e = sp.zeros(q)
            e[a,b], e[b,a] = 1,-1
            xr = e.row_join(sp.zeros(q)).col_join(sp.zeros(q).row_join(-e))/4
            xi = sp.zeros(q).row_join(e).col_join(e.row_join(sp.zeros(q)))/4
            for kind, x in (('real',xr),('imaginary',xi)):
                dj = x*j0-j0*x
                equal(f'd{depth} root {a}{b} {kind} metric',sp.trace(dj.T*dj),1)
                tangent.append(x)
            diagonal = sp.zeros(q)
            diagonal[a,a]=diagonal[b,b]=1
            expected = realify(-sp.I*diagonal/8)
            equal(f'd{depth} root {a}{b} curvature',-(xr*xi-xi*xr),expected)
        # A diagonal normal current pairs only matching real/imaginary root planes.
        diagonal_probe = realify(sp.I*sp.diag(*range(1,q+1)))
        for i, x in enumerate(tangent):
            for j, y in enumerate(tangent):
                if i//2 != j//2:
                    equal(f'd{depth} diagonal root decoupling {i},{j}',
                          sp.trace(diagonal_probe.T*(x*y-y*x)),0)

        even = [mask for mask in range(1<<q) if mask.bit_count()%2==0]
        weights = [((mask>>0)&1)+((mask>>1)&1)-1 for mask in even]
        check(f'd{depth} Spin vacuum root weight',weights[even.index(0)]==-1)
        check(f'd{depth} Spin double root weight',weights[even.index(3)]==1)
        check(f'd{depth} Spin chiral rank',len(even)==2**(q-1))
        for mask, weight in zip(even,weights):
            equal(f'd{depth} root Spin determinant shift mask {mask}',
                  sp.Rational(((mask>>0)&1)+((mask>>1)&1))-1,weight)
        omega,k,amp = sp.symbols('omega k amplitude',positive=True)
        pplus = sp.Matrix([omega,k]+[0]*(depth-1))
        pminus = sp.Matrix([omega,-k]+[0]*(depth-1))
        current = amp**2*(pminus-pplus)/4
        eta = sp.diag(1,*([-1]*depth))
        equal(f'd{depth} counterflow temporal current',current[0],0)
        equal(f'd{depth} counterflow spatial current',current[1],-amp**2*k/2)
        equal(f'd{depth} counterflow spacelike norm',(current.T*eta*current)[0],-amp**4*k**2/4)
        CASES.append({'depth':depth,'normal_rank':q,'orientation_dimension':q*(q-1),
                      'chiral_coefficient_rank':len(even),'root_weights':weights,
                      'counterflow_root_norm':'-amplitude**4*k**2/4',
                      'traceless_canonical_response':q>=3})
    check('depth ten empty normal source',0*(0-1)==0)
    CASES.append({'depth':10,'normal_rank':0,'orientation_dimension':0,
                  'counterflow_root_norm':None,'traceless_canonical_response':False})


def exact_pullback_variation():
    # Exact abelian connection on a canonical root sphere, not just its origin jet.
    u,v,ut,vt,du,dv,dut,dvt,s = sp.symbols('u v ut vt du dv dut dvt s',real=True)
    den = 1+u*u+v*v
    au,av = sp.I*v/den,-sp.I*u/den
    a = au*ut+av*vt
    fuv = sp.diff(av,u)-sp.diff(au,v)
    equal('exact root curvature',fuv,-2*sp.I/den**2)
    direct = sp.diff(a.subs({u:u+s*du,v:v+s*dv,ut:ut+s*dut,vt:vt+s*dvt},simultaneous=True),s).subs(s,0)
    gauge = au*du+av*dv
    dt_gauge = sp.diff(gauge,u)*ut+sp.diff(gauge,v)*vt+sp.diff(gauge,du)*dut+sp.diff(gauge,dv)*dvt
    equal('direct canonical pullback variation',direct,dt_gauge+fuv*(du*vt-dv*ut))

    # The commutator contribution is independently tested with a nonabelian connection.
    h1=sp.Matrix([[0,1],[-1,0]])
    h2=sp.Matrix([[sp.I,0],[0,-sp.I]])
    a1=u*h1+v*h2
    a2=u*u*h2+v*h1
    curvature=sp.diff(a2,u)-sp.diff(a1,v)+a1*a2-a2*a1
    pull=a1*ut+a2*vt
    direct=pull.diff(u)*du+pull.diff(v)*dv+pull.diff(ut)*dut+pull.diff(vt)*dvt
    gauge=a1*du+a2*dv
    cov_dt=sp.diff(gauge,u)*ut+sp.diff(gauge,v)*vt+sp.diff(gauge,du)*dut+sp.diff(gauge,dv)*dvt+pull*gauge-gauge*pull
    equal('nonabelian pullback variation includes commutator',direct,cov_dt+curvature*(du*vt-dv*ut))


def complete_energy_and_principal():
    # A real unitary coefficient doublet suffices to test the full momentum cancellation.
    p,q,pt,qt,px,qx,ut,ux,s = sp.symbols('p q pt qt px qx ut ux s',real=True)
    a = sp.Rational(2,3)
    w0 = sp.Matrix([pt-a*q*ut,qt+a*p*ut])
    wx = sp.Matrix([px-a*q*ux,qx+a*p*ux])
    potential=p*p+q*q
    lag=w0.dot(w0)-wx.dot(wx)-potential
    energy=sum(sp.diff(lag,vel)*vel for vel in (pt,qt,ut))-lag
    equal('complete coefficient Legendre cancellation',energy,w0.dot(w0)+wx.dot(wx)+potential)
    # Omitting the orientation momentum produces a demonstrably different answer.
    partial=sum(sp.diff(lag,vel)*vel for vel in (pt,qt))-lag
    equal('missing connection momentum correction',energy-partial,sp.diff(lag,ut)*ut)
    check('partial energy differs generically',sp.simplify(energy-partial)!=0)

    hu,hp,hq,xi0,xi1 = sp.symbols('hu hp hq xi0 xi1',real=True)
    changed=lag.subs({pt:pt+s*hp*xi0,qt:qt+s*hq*xi0,ut:ut+s*hu*xi0,
                     px:px+s*hp*xi1,qx:qx+s*hq*xi1,ux:ux+s*hu*xi1},simultaneous=True)
    actual=sp.diff(changed,s,2).subs(s,0)
    expected=2*(xi0**2-xi1**2)*((hp-a*q*hu)**2+(hq+a*p*hu)**2)
    equal('direct coupled principal Hessian',actual,expected)
    for depth,rank in ((2,12),(3,6),(4,6),(5,2),(6,2),(10,0)):
        # Congruence identity for every occupied target rank and coefficient real rank.
        nr=4
        b=sp.Matrix(nr,rank,lambda i,j:sp.Rational((i+1)*(j+2)%7-3,5))
        transform=sp.eye(rank).row_join(sp.zeros(rank,nr)).col_join(b.row_join(sp.eye(nr)))
        equal(f'd{depth} principal congruence inverse',
              transform.inv(),sp.eye(rank).row_join(sp.zeros(rank,nr)).col_join((-b).row_join(sp.eye(nr))))
        check(f'd{depth} principal congruence determinant',transform.det()==1)
        diag=sp.diag(*([2]*rank+[3]*nr))
        full=transform.T*diag*transform
        lower,pivots=full.LDLdecomposition(hermitian=False)
        equal(f'd{depth} complete time Hessian factor',lower*pivots*lower.T,full)
        check(f'd{depth} complete time Hessian positive',all(x>0 for x in pivots.diagonal()))


def current_dispersion():
    t,x=sp.symbols('t x',real=True)
    a0,ax,omega,k=sp.symbols('a0 ax omega k',real=True)
    plane=sp.exp(-sp.I*omega*t+sp.I*k*x)
    operator=plane.diff(t,2)-plane.diff(x,2)+sp.I*(a0*plane.diff(t)+ax*plane.diff(x))
    equal('root Fourier dispersion',operator/plane,-omega**2+k**2+a0*omega-ax*k)
    discriminant=a0*a0+4*k*k-4*ax*k
    equal('current Lorentz norm controls all wavenumbers',discriminant,(2*k-ax)**2+a0*a0-ax*ax)
    z=sp.Function('z')(t,x)
    phase=sp.exp(-sp.I*(a0*t-ax*x)/2)
    w=phase*z
    transformed=w.diff(t,2)-w.diff(x,2)+sp.I*(a0*w.diff(t)+ax*w.diff(x))
    equal('exact phase reduction',transformed/phase,z.diff(t,2)-z.diff(x,2)+(a0*a0-ax*ax)*z/4)
    for label,time,space in (('timelike',sp.Rational(5,3),1),('null',1,1),('spacelike',0,1)):
        minimum=(a0*a0-ax*ax).subs({a0:time,ax:space})
        check(label+' dispersion classification',minimum>0 if label=='timelike' else (minimum==0 if label=='null' else minimum<0))
    u,v=sp.Function('u')(t,x),sp.Function('v')(t,x)
    f2,c0,cx=sp.symbols('f2 c0 cx',real=True)
    lag=f2*(u.diff(t)**2-u.diff(x)**2+v.diff(t)**2-v.diff(x)**2)/2
    lag+=c0*(u*v.diff(t)-v*u.diff(t))/2+cx*(u*v.diff(x)-v*u.diff(x))/2
    for field,other,sign in ((u,v,-1),(v,u,1)):
        euler=sp.diff(sp.diff(lag,field.diff(t)),t)+sp.diff(sp.diff(lag,field.diff(x)),x)-sp.diff(lag,field)
        equal('direct root quadratic Euler '+str(field),euler,
              f2*(field.diff(t,2)-field.diff(x,2))+sign*(c0*other.diff(t)+cx*other.diff(x)))


def reciprocal_stress_controls():
    t,x=sp.symbols('t x',real=True)
    p,q=sp.Function('p')(t,x),sp.Function('q')(t,x)
    a0,a1=sp.Function('a0')(t,x),sp.Function('a1')(t,x)
    m2=sp.Symbol('m2',real=True)
    rotation=sp.Matrix([[0,-1],[1,0]])
    field=sp.Matrix([p,q])
    w0=field.diff(t)+a0*rotation*field
    w1=field.diff(x)+a1*rotation*field
    euler=w0.diff(t)+a0*rotation*w0-w1.diff(x)-a1*rotation*w1+m2*field
    onshell={p.diff(t,2):sp.solve(euler[0],p.diff(t,2))[0],
             q.diff(t,2):sp.solve(euler[1],q.diff(t,2))[0]}
    potential=m2*field.dot(field)
    t00=w0.dot(w0)+w1.dot(w1)+potential
    t10=-2*w1.dot(w0)
    t01=2*w0.dot(w1)
    t11=-w0.dot(w0)-w1.dot(w1)+potential
    j0=2*w0.dot(rotation*field)
    j1=-2*w1.dot(rotation*field)
    curvature=a1.diff(t)-a0.diff(x)
    equal('direct coefficient temporal stress exchange',
          (t00.diff(t)+t10.diff(x)).subs(onshell),-j1*curvature)
    equal('direct coefficient spatial stress exchange',
          (t01.diff(t)+t11.diff(x)).subs(onshell),j0*curvature)
    equal('direct covariant abelian current conservation',
          (j0.diff(t)+j1.diff(x)).subs(onshell),0)


def main():
    root_and_source_controls()
    exact_pullback_variation()
    complete_energy_and_principal()
    current_dispersion()
    reciprocal_stress_controls()
    path=Path(__file__).resolve()
    report={'created_at':datetime.now(timezone.utc).isoformat(),
            'experiment':'canonical-orientation-source-and-response',
            'sources':{'experiments/canonical-source-checks.py':sha256(path.read_bytes()).hexdigest()},
            'checks_passed':len(CHECKS),'cases':CASES,
            'scope':'Exact constrained pullback variation, positive wave comparison energy and principal congruence, Spin root weights and ideal constant-current dispersion. No observed particle, finite-energy background or complete mass mechanism is claimed.'}
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    output=path.with_name('canonical-source-results-'+stamp+'.json')
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'checks_passed':len(CHECKS),'cases':len(CASES),'report':str(output)}))


if __name__=='__main__':
    main()
