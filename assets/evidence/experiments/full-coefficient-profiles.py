#!/usr/bin/env python3
"""Reuse the audited radial solver for the changed complete-coefficient action."""

from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import importlib.util
import json
import sympy as sp


ROOT=Path(__file__).resolve().parent
SOLVER=ROOT/'rotating-determinant-checks.py'
spec=importlib.util.spec_from_file_location('radial_solver',SOLVER)
radial=importlib.util.module_from_spec(spec)
spec.loader.exec_module(radial)
radial.BETA=1.0
radial.TENSION=4.0


def differential_functions():
    f,p,w,slope=sp.symbols('F P omega slope',real=True)
    beta=sp.Rational(1)
    tension=sp.Rational(4)
    kappa=tension*beta
    y=sp.sin(f)**2
    bb=1-beta*w*w*y
    margin=bb-beta*p*p
    g=sp.Matrix([p/sp.sqrt(margin),sp.sin(f)*sp.cos(f)*(1-y/2)-w*w*sp.sin(f)*sp.cos(f)/sp.sqrt(margin)])
    jac=[[sp.lambdify((f,p,w),sp.diff(g[i],v),'numpy') for v in (f,p)] for i in range(2)]
    src=[sp.lambdify((f,p,w),sp.diff(g[i],w),'numpy') for i in range(2)]
    charge=kappa*w*y/sp.sqrt(margin)
    energy=tension*(1/sp.sqrt(margin)-1)+kappa*(y-y*y/4)/2
    qdiff=[sp.lambdify((f,p,w),sp.diff(charge,v),'numpy') for v in (f,p,w)]
    ediff=[sp.lambdify((f,p,w),sp.diff(energy,v),'numpy') for v in (f,p,w)]
    aa=1+beta*slope*slope
    # The shared spectrum routine uses coefficients divided by kappa.
    routh=(sp.sqrt(aa)*sp.sqrt(bb)-1)/beta+(y-y*y/4)/2
    rfp=sp.diff(routh,f,slope)
    hessian=[sp.lambdify((f,slope,w),expr,'numpy') for expr in
             (sp.diff(routh,f,2),rfp,sp.diff(rfp,f),sp.diff(rfp,slope))]
    return jac,src,qdiff,ediff,hessian


def main():
    functions=differential_functions()
    cases=[]
    for omega in (0.95,0.97):
        for depth in (2,3,4,5,6):
            base=radial.solve_depth(depth,omega)
            refined=radial.solve_depth(depth,omega,radius=110.0,tolerance=2e-10)
            if refined['status']=='collocation-converged' and base['status']=='collocation-converged':
                refined['convergence']={key:abs(refined[key]-base[key])/abs(refined[key]) for key in ('energy','charge','center')}
                if max(refined['convergence'].values())>1e-6:raise AssertionError(('boundary convergence',refined))
                solution=radial.PROFILES[(depth,omega,110.0,2e-10)]
                response=radial.sensitivity(solution,depth,omega,110.0,functions)
                # The reused integral has unit kappa; restore the common kappa=4.
                response['fixed_shape_charge_susceptibility']*=4
                response['fixed_charge_rank_update_ratio']/=4
                refined.update(response)
                spectra={}
                for intervals in (1100,2200):
                    spectra[str(intervals)]={
                        'shape_l'+str(ell):radial.radial_spectrum(solution,depth,omega,110.0,intervals,functions,'shape',ell)
                        for ell in (0,1)}
                refined['radial_quadratic_spectra']=spectra
                base['refined']=refined
            cases.append(base)
            print(json.dumps({key:base[key] for key in ('depth','omega','status')}),flush=True)
    cases.append({'depth':10,'normal_rank':0,'status':'no-active-normal-generator'})
    path=Path(__file__).resolve()
    report={'created_at':datetime.now(timezone.utc).isoformat(),
            'experiment':'phase-complete-normal-coefficient-localization',
            'sources':{'experiments/full-coefficient-profiles.py':sha256(path.read_bytes()).hexdigest(),
                       'experiments/rotating-determinant-checks.py':sha256(SOLVER.read_bytes()).hexdigest()},
            'parameters':{'beta':1,'T':4,'kappa':4,'m_squared':1,'b':'1/4'},
            'cases':cases,
            'scope':'Changed fixed-norm complete-coefficient action with its combined phase/normal generator. Complete radial energy, stress and fixed-charge response are tested; unused coefficient polarizations have a separate analytic convex-form positivity proof. No observed particle identification, Lorentz Dirac kinetics or simplex frequency selection is claimed.'}
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    output=path.with_name('full-coefficient-profiles-'+stamp+'.json')
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'cases':len(cases),'report':str(output)}),flush=True)


if __name__=='__main__':main()
