#!/usr/bin/env python3
"""Exact and finite-gap checks of a reciprocal geometric velocity interaction."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import importlib.util
import json
from math import comb, pi
from pathlib import Path

import mpmath as mp
import numpy as np
import sympy as sp


CHECKS = 0
CASES = []


def check(ok, message):
    global CHECKS
    if not ok:
        raise AssertionError(message)
    CHECKS += 1


def response(omega, stiffness, gap, a, b, inertia=1.):
    squared=omega*omega
    denominator=(stiffness-inertia*squared)*(gap*gap-squared)-gap*a*a*squared
    return gap*b*b*squared*(stiffness-inertia*squared+gap*a*a)/denominator


def full_hessian(kq, kl, gap, a, b, mq=1., ml=1.):
    # Canonical order q,pq,lambda,plambda,x,y, c=(x+iy)/sqrt(2).
    h=np.diag([kq+gap*a*a,1/mq,kl+gap*b*b,1/ml,gap,gap])
    h[0,4]=h[4,0]=-gap*a
    h[2,5]=h[5,2]=-gap*b
    return h


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output',required=True)
    ap.add_argument('--sawtooth-source',required=True)
    args=ap.parse_args()
    # Independent exact Schur reduction of the full quadratic Fourier action.
    w,d,a,b,k,m,kl,ml=sp.symbols('Omega Delta A B K M Kl Ml',positive=True,real=True)
    action=sp.Matrix([[k+d*a*a-m*w*w,0,-d*a,0],
                      [0,kl+d*b*b-ml*w*w,0,-d*b],
                      [-d*a,0,d,-sp.I*w],
                      [0,-d*b,sp.I*w,d]])
    keep=[1]
    eliminate=[0,2,3]
    reduced=action.extract(keep,keep)-action.extract(keep,eliminate)*action.extract(eliminate,eliminate).inv()*action.extract(eliminate,keep)
    r=d*b*b*w*w*(k-m*w*w+d*a*a)/((k-m*w*w)*(d*d-w*w)-d*a*a*w*w)
    check(sp.simplify(reduced[0]-(kl-ml*w*w-r))==0,'complete source Schur response')
    check(sp.simplify(sp.limit(r/w**2,w,0)-b*b*(a*a*d+k)/(d*k))==0,'low-frequency coefficient')
    check(sp.simplify(sp.limit(r,w,sp.oo)+d*b*b)==0,'high-frequency constant')
    check(sp.simplify(sp.limit(r,w,d)-b*b*(d*d*m-k-d*a*a)/a**2)==0,'bare gap cancellation')
    determinant=sp.factor(action.det())
    check(sp.Poly(determinant,w).degree()==6,'three complete oscillators')

    # Frame curvature is invariant under a coordinate-dependent phase.
    q,l,theta,alpha,beta,chi=sp.symbols('q lambda theta alpha beta chi',real=True)
    u=sp.Matrix([sp.cos((theta+alpha*q)/2),sp.exp(sp.I*beta*l)*sp.sin((theta+alpha*q)/2)])
    aq=sp.simplify(sp.I*(u.H*sp.diff(u,q))[0])
    al=sp.simplify(sp.I*(u.H*sp.diff(u,l))[0])
    curvature=sp.simplify(sp.diff(al,q)-sp.diff(aq,l))
    check(sp.simplify(curvature+alpha*beta*sp.sin(theta+alpha*q)/2)==0,'CP1 mixed curvature')
    phase=chi*q*l
    transformed=sp.exp(sp.I*phase)*u
    tq=sp.simplify(sp.I*(transformed.H*sp.diff(transformed,q))[0])
    tl=sp.simplify(sp.I*(transformed.H*sp.diff(transformed,l))[0])
    check(sp.simplify(tq-aq+sp.diff(phase,q))==0,'phase gauge q')
    check(sp.simplify(tl-al+sp.diff(phase,l))==0,'phase gauge lambda')
    check(sp.simplify(sp.diff(tl,q)-sp.diff(tq,l)-curvature)==0,'curvature gauge invariant')
    induced_mass_q=a*a/d
    induced_mass_l=b*b/d
    check(sp.simplify((a*b)**2/(induced_mass_q*induced_mass_l)-d*d)==0,'curvature-induced inertia relation')

    j=np.kron(np.eye(3),np.array([[0.,1.],[-1.,0.]]))
    for gap in (.3,1.3,3.7,10.,30.):
        for coupling in (.03,.1,.3):
            h=full_hessian(2.,3.,gap,coupling,coupling)
            check(np.linalg.eigvalsh(h)[0]>0,'positive complete Hamiltonian')
            dynamic=j@h
            eigen=np.linalg.eigvals(dynamic)
            check(np.max(np.abs(eigen.real))<1e-10,'oscillatory full dynamics')
            frequencies=sorted(v.imag for v in eigen if v.imag>1e-9)
            check(len(frequencies)==3,'retained fast mode')
            test=np.arange(1.,7.)/7
            check(abs(test@(h@dynamic+dynamic.T@h)@test)<1e-12,'energy conservation')
            for frequency in (.13,.91,2.11,7.3,100.):
                numeric=np.array(action.subs({k:2.,kl:3.,m:1.,ml:1.,d:gap,a:coupling,b:coupling,w:frequency}),dtype=complex)
                indexes=[0,2,3]
                schur=numeric[1,1]-numeric[1,indexes]@np.linalg.solve(numeric[np.ix_(indexes,indexes)],numeric[indexes,1])
                expected=3.-frequency**2-response(frequency,2.,gap,coupling,coupling)
                check(np.isclose(schur,expected,rtol=1e-10,atol=1e-10),'numeric full versus reduced response')
            CASES.append({'name':'complete-Hamiltonian','parameters':{'gap':gap,'A':coupling,'B':coupling,
                'bare_masses':[1,1],'bare_stiffnesses':[2,3]},
                'measurements':{'complete_frequencies':frequencies,'minimum_energy_eigenvalue':float(np.linalg.eigvalsh(h)[0])}})

    source=Path(args.sawtooth_source)
    spec=importlib.util.spec_from_file_location('frozen_sawtooth',source)
    saw=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(saw)
    mp.mp.dps=50
    electron=mp.mpf('0.51099895000')
    mu6=electron/18564
    mu5=mp.sqrt(mp.mpf('2.534e-3')/(65780**2-2002**2))/mp.mpf(10**6)
    scales={2:electron*mp.sqrt(2890),3:electron*mp.sqrt(32*mp.sqrt(7)),
            4:electron*mp.sqrt(48/mp.sqrt(7))/15,5:mu5,6:mu6,10:mu6}
    assigned=[('W',2,76),('Z',2,81),('Higgs',2,95),('down',3,1),('strange',3,4),
              ('up',4,3),('charm',4,20),('top',4,72),('nu1',5,10),('nu2',5,15),
              ('nu3',5,22),('electron',6,13),('muon',6,35),('tau',10,23)]
    for gap in (1.3,3.7,10.,30.):
        a=b=.1
        for label,depth,n in assigned:
            cutoff=comb(n+depth-1,depth)
            ratio=float(scales[depth]/mu6)
            limit=2*cutoff if cutoff<=150000 else 65536
            _,coeff=saw.coefficients(cutoff,limit)
            omega=ratio*np.arange(1,len(coeff)+1)
            rq=response(omega,2.,gap,a,b)
            rt=response(omega,3.,gap,a,b)
            average=(3*rq+2*rt)/5
            quadratic=float(np.sum(coeff*coeff*average)/2)
            check(np.isfinite(quadratic),'finite all-sector source sum')
            # Exact complete response has no DC term and a bounded high-frequency tail.
            check(response(0.,2.,gap,a,b)==0.,'zero static source response')
            omitted=2*cutoff-len(coeff)
            tail=0.
            if omitted:
                low_tail_omega=ratio*(len(coeff)+1)
                check(low_tail_omega**2>=2*(3+gap*gap+gap*a*a),'tail beyond both shifted poles')
                coefficient_tail=4*pi*pi/cutoff+1/(3*len(coeff)**3)
                tail=3*gap*b*b*coefficient_tail
            low_coefficient=b*b/gap+a*a*b*b*(13/30)
            derivative_norm=float(2*cutoff-mp.harmonic(cutoff)-(mp.zeta(2)-mp.polygamma(1,cutoff+1))/2)
            low_estimate=low_coefficient*ratio*ratio*derivative_norm
            if depth==5:
                check(np.isclose(quadratic,low_estimate,rtol=.01),'small neutrino source follows derivative law')
            CASES.append({'name':'all-sector-finite-gap-response','real_depth':depth,'mode_index':n,
                'parameters':{'particle':label,'gap':gap,'A':a,'B':b,'fundamental_ratio':ratio,
                    'cutoff':cutoff,'evaluated_harmonics':len(coeff),
                    'source_map':'conditional lambda=f_K squared; fixed isotropic direction'},
                'measurements':{'signed_complete_response':quadratic,'absolute_tail_bound':tail,
                                'low_frequency_estimate':low_estimate}})

    report={'experiment_key':'mixed-curvature-velocity-response-20261009',
        'created_utc':datetime.now(timezone.utc).isoformat(),
        'sources':{'experiments/mixed-curvature-velocity-checks.py':sha256(Path(__file__).read_bytes()).hexdigest(),
                   'experiments/sawtooth-sector-coupling-checks.py':sha256(source.read_bytes()).hexdigest()},
        'scope':'Conditional positive same-field mode action, exact complement and common all-sector Fourier comparison. Not a derived physical rate or electric charge.',
        'summary':{'checks_passed':CHECKS,'case_count':len(CASES)},'cases':CASES}
    Path(args.output).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['summary']))
    for c in CASES:
        if c['name']=='all-sector-finite-gap-response' and c['parameters']['gap']==10.:
            print(json.dumps(c))


if __name__=='__main__':
    main()
