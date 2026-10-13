#!/usr/bin/env python3
"""Exact phase-complete coefficient geometry and polarization controls."""

from datetime import datetime, timezone
from hashlib import sha256
from itertools import combinations
from pathlib import Path
import json
import sympy as sp


CHECKS=[]


def equal(name,lhs,rhs):
    residual=sp.trigsimp(sp.simplify(lhs-rhs))
    if residual!=0:raise AssertionError((name,residual))
    CHECKS.append(name)


def derive():
    f,p,pp,r,d,beta,w,m2,b,T=sp.symbols('F p pp r d beta omega m2 b T',real=True)
    y=sp.sin(f)**2
    A=1+beta*p*p
    B=1-beta*w*w*y
    C0=sp.sqrt(A)/sp.sqrt(B)
    Cr=sp.sqrt(B)/sp.sqrt(A)
    K=m2-2*b*y-C0*w*w
    force=A*(sp.sin(f)*sp.cos(f)*((m2-2*b*y)*C0-w*w/B)-(d-1)*p/r)
    radial_derivative=lambda expr:sp.diff(expr,f)*p+sp.diff(expr,p)*force
    div=lambda expr:radial_derivative(expr)+(d-1)*expr/r
    equal('complete radial momentum equation',div(Cr*p),sp.sin(f)*sp.cos(f)*K)
    a=sp.symbols('a',real=True)
    W=lambda weight:(weight**2-y)*K-Cr*p*p
    h0,h1=sp.cos(f),sp.sin(f)
    equal('weight zero positive solution',-div(Cr*sp.diff(h0,f)*p)+W(0)*h0,0)
    equal('weight one positive solution',-div(Cr*sp.diff(h1,f)*p)+W(1)*h1,0)
    equal('all weights convex form',W(a),(1-a*a)*W(0)+a*a*W(1))

    eps,u,v,ut,vt,ur,vr=sp.symbols('eps u v ut vt ur vr',real=True)
    rho=u*u+v*v
    base=sp.Matrix([sp.cos(f),sp.sin(f),0])
    base_t=sp.Matrix([0,sp.I*w*sp.sin(f),0])
    base_r=sp.Matrix([-p*sp.sin(f),p*sp.cos(f),0])
    ray=(1-eps**2*rho/2)*base+sp.Matrix([0,0,eps*(u+sp.I*v)])
    dt=(1-eps**2*rho/2)*base_t-eps**2*(u*ut+v*vt)*base+sp.Matrix([0,0,eps*(ut+sp.I*vt+sp.I*a*w*(u+sp.I*v))])
    dr=(1-eps**2*rho/2)*base_r-eps**2*(u*ur+v*vr)*base+sp.Matrix([0,0,eps*(ur+sp.I*vr)])
    inner=lambda x,z:(sp.conjugate(x).T*z)[0]
    coefficient=lambda expr:sp.expand(expr).coeff(eps,2)
    time_norm=(ut-a*w*v)**2+(vt+a*w*u)**2
    equal('full coefficient time metric',coefficient(inner(dt,dt)),time_norm-w*w*y*rho)
    equal('full coefficient radial metric',coefficient(inner(dr,dr)),ur*ur+vr*vr-p*p*rho)
    generator=sp.diag(0,1,a)
    equal('full coefficient Killing norm',coefficient(inner(ray,generator**2*ray)),(a*a-y)*rho)

    aa,bb,bt=sp.symbols('A B beta',positive=True)
    gtt,grr,gtr=sp.symbols('gtt grr gtr',real=True)
    delta=(bb-bt*eps**2*gtt)*(aa+bt*eps**2*grr)+(bt*eps**2*gtr)**2
    quadratic=sp.diff(T*(1-sp.sqrt(delta)),eps,2).subs(eps,0)/2
    equal('complete determinant quadratic metric variation',quadratic,T*bt*(sp.sqrt(aa/bb)*gtt-sp.sqrt(bb/aa)*grr)/2)
    c0,cr,ct=sp.symbols('C0 Cr Ct',positive=True)
    kinetic=c0*time_norm-cr*(ur*ur+vr*vr)
    rest=-c0*w*w*y*rho+cr*p*p*rho-(m2-2*b*y)*(a*a-y)*rho
    lag=kinetic+rest
    ham=sp.diff(lag,ut)*ut+sp.diff(lag,vt)*vt-lag
    expected=c0*(ut*ut+vt*vt)+cr*(ur*ur+vr*vr)+((a*a-y)*(m2-2*b*y-c0*w*w)-cr*p*p)*rho
    equal('complete rotating-frame transverse energy',ham,expected)

    # The two active coefficient phases have positive spatial forms.
    phase_gradient=sp.symbols('phase_gradient',real=True)
    delta_phase1=B*(A+beta*y*phase_gradient**2)+beta**2*w*w*y*y*phase_gradient**2
    equal('active rotating phase radial determinant',delta_phase1,A*B+beta*y*phase_gradient**2)
    delta_phase0=B*(A+beta*(1-y)*phase_gradient**2)
    equal('active stationary phase radial determinant',delta_phase0,A*B+beta*B*(1-y)*phase_gradient**2)


def carrier_weights():
    rows=[]
    for depth,q in ((2,4),(3,3),(4,3),(5,2),(6,2)):
        weights=[]
        for degree in range(0,q+1,2):
            for occupied in combinations(range(q),degree):
                shifted=sp.Rational(sum(j in occupied for j in (0,1)),2)
                spin=sp.Rational(sum(j in occupied for j in (0,1))-1,2)
                equal(f'd{depth} actual Spin shift {occupied}',shifted,spin+sp.Rational(1,2))
                weights.append(str(shifted))
        equal(f'd{depth} full chiral coefficient count',len(weights),2**(q-1))
        rows.append({'depth':depth,'normal_rank':q,'weights':weights,
                     'zero_weight_complex_count':weights.count('0'),
                     'half_weight_complex_count':weights.count('1/2'),
                     'unit_weight_complex_count':weights.count('1')})
    rows.append({'depth':10,'normal_rank':0,'weights':['0'],'status':'no-active-normal-generator'})
    return rows


def main():
    derive()
    rows=carrier_weights()
    path=Path(__file__).resolve()
    report={'created_at':datetime.now(timezone.utc).isoformat(),
            'experiment':'phase-complete-normal-coefficient-polarization',
            'sources':{'experiments/full-coefficient-checks.py':sha256(path.read_bytes()).hexdigest()},
            'exact_checks':len(CHECKS),'checks':CHECKS,'cases':rows,
            'scope':'A fixed-norm normal-spinor-valued positive wave field with a selected combined phase/normal generator. The exact polarization positivity is conditional on a smooth monotone localized radial branch. No Lorentz Dirac kinetic law, observed spin, electric charge or discrete frequency selection is derived.'}
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    output=path.with_name('full-coefficient-results-'+stamp+'.json')
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'exact_checks':len(CHECKS),'cases':len(rows),'report':str(output)}))


if __name__=='__main__':main()
