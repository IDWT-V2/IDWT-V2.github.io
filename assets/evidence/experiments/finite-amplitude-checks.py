#!/usr/bin/env python3
"""Finite amplitude continuation of the complete-coefficient determinant law."""

from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import importlib.util
import json
import math
import numpy as np
import sympy as sp
from scipy.integrate import solve_bvp,simpson
from scipy.special import kve,gamma


ROOT=Path(__file__).resolve().parent
PARENT=ROOT/'rotating-determinant-checks.py'
spec=importlib.util.spec_from_file_location('radial',PARENT)
radial=importlib.util.module_from_spec(spec)
spec.loader.exec_module(radial)
radial.BETA=1.0
radial.TENSION=4.0
CHECKS=[]


def equal(name,lhs,rhs):
    result=sp.trigsimp(sp.simplify(lhs-rhs))
    if result!=0:raise AssertionError((name,result))
    CHECKS.append(name)


def derive():
    a,c,pa,pc,beta,w,m2,b,T,M2=sp.symbols('a c pa pc beta omega m2 b T M2',real=True)
    kappa=T*beta
    s=a*a+c*c
    A=1+beta*(pa*pa+pc*pc)
    B=1-beta*w*w*c*c
    V=kappa*(M2*(s-1)**2/4+(m2*c*c-b*c**4)/2)
    routh=T*(sp.sqrt(A)*sp.sqrt(B)-1)+V
    equal('first radial amplitude force',sp.diff(routh,a)/kappa,M2*(s-1)*a)
    equal('second radial amplitude force',sp.diff(routh,c)/kappa,
          (M2*(s-1)+m2-2*b*c*c-w*w*sp.sqrt(A)/sp.sqrt(B))*c)
    equal('first radial momentum',sp.diff(routh,pa)/kappa,pa*sp.sqrt(B)/sp.sqrt(A))
    equal('second radial momentum',sp.diff(routh,pc)/kappa,pc*sp.sqrt(B)/sp.sqrt(A))
    lag=-routh
    q=sp.diff(lag,w)
    equal('complete charge',q,kappa*w*c*c*sp.sqrt(A)/sp.sqrt(B))
    equal('complete energy',w*q-lag,T*(sp.sqrt(A)/sp.sqrt(B)-1)+V)
    weight=sp.symbols('weight',real=True)
    W=lambda x:M2*(s-1)+x*x*(m2-2*b*c*c-w*w*sp.sqrt(A)/sp.sqrt(B))
    equal('finite amplitude convex polarization identity',W(weight),(1-weight**2)*W(0)+weight**2*W(1))
    z=sp.symbols('z',real=True)
    p=M2*(z-1)**2/4+(m2*z-b*z*z)/2
    equal('large norm polynomial',p,(M2-2*b)*z*z/4+(m2-M2)*z/2+M2/4)
    equal('large norm derivative at one',sp.diff(p,z).subs(z,1),(m2-2*b)/2)
    zmin=(M2-m2)/(M2-2*b)
    equal('large norm minimum',p.subs(z,zmin),(2*M2*(m2-b)-m2*m2)/(4*(M2-2*b)))
    x,bb=sp.symbols('x Bstar',positive=True)
    g=(M2-2*b)*x+m2-M2-w*w/sp.sqrt(1-beta*w*w*x)
    equal('core maximum derivative',sp.diff(g,x).subs(x,(1-bb)/(beta*w*w)),
          M2-2*b-beta*w**4/(2*bb**sp.Rational(3,2)))
    equal('global sufficient bound exceeds quartic threshold',
          m2*m2/(2*(m2-b))-2*b,(m2-2*b)**2/(2*(m2-b)))


def solve_amplitudes(depth,omega,stiffness,radius,tolerance,seed_solution):
    beta=1.0
    tension=4.0
    kappa=4.0
    mu=math.sqrt(1-omega*omega)
    nu=(depth-2)/2
    robin=mu*kve(nu+1,mu*radius)/kve(nu,mu*radius)
    mesh=np.linspace(0,radius,int(radius*16)+1)
    if seed_solution.y.shape[0]==2:
        f,p=seed_solution.sol(mesh)
        # Stable cos(F)-1 avoids cancellation in the vacuum norm equation.
        initial=np.array([-2*np.sin(f/2)**2,np.sin(f),-np.sin(f)*p,np.cos(f)*p])
    else:initial=seed_solution.sol(mesh)
    def fun(rr,y):
        delta,c,pa,pc=y
        a=1+delta
        norm_delta=delta*(delta+2)+c*c
        B=1-beta*omega*omega*c*c
        margin=B-beta*(pa*pa+pc*pc)
        root=np.sqrt(np.maximum(margin,1e-14))
        return np.array([pa/root,pc/root,stiffness*norm_delta*a,
                         (stiffness*norm_delta+1-c*c/2-omega*omega/root)*c])
    def boundary(left,right):
        B=1-omega*omega*right[1]**2
        margin=B-right[2]**2-right[3]**2
        return np.array([left[2],left[3],right[0],right[3]/math.sqrt(max(margin,1e-14))+robin*right[1]])
    singular=np.diag([0,0,-(depth-1),-(depth-1)])
    solution=solve_bvp(fun,boundary,mesh,initial,S=singular,tol=tolerance,max_nodes=24000)
    if not solution.success:
        return {'depth':depth,'omega':omega,'norm_stiffness':stiffness,'radius':radius,
                'status':'collocation-failure','message':solution.message},solution
    rr=np.linspace(0,radius,int(radius*300)+1)
    delta,c,pa,pc=solution.sol(rr)
    a=1+delta
    norm_delta=delta*(delta+2)+c*c
    B=1-omega*omega*c*c
    margin=B-pa*pa-pc*pc
    if np.min(B)<=0 or np.min(margin)<=0:raise AssertionError('inadmissible converged branch')
    A=B/margin
    V=kappa*(stiffness*norm_delta**2/4+(c*c-c**4/4)/2)
    ratio=np.sqrt(A/B)
    energy=tension*(A-B)/(B*(ratio+1))+V
    charge=kappa*omega*c*c*ratio
    pr=tension*(A-B)/(A*(1+np.sqrt(B/A)))-V
    pt=-tension*(A*B-1)/(np.sqrt(A*B)+1)-V
    measure=2*math.pi**(depth/2)/gamma(depth/2)*rr**(depth-1)
    integrate=lambda values:float(simpson(values*measure,x=rr))
    E,Q=integrate(energy),integrate(charge)
    virial=integrate(pr+(depth-1)*pt)
    if np.max(c)<0.1 or np.min(c)<-1e-7 or np.min(a)<=0:raise AssertionError('trivial or sign-changing branch')
    if abs(virial/E)>1e-5:raise AssertionError(('complete virial',depth,omega,stiffness,virial/E))
    result={'depth':depth,'normal_rank':{2:4,3:3,4:3,5:2,6:2}[depth],'omega':omega,
            'norm_stiffness':stiffness,'radius':radius,'status':'collocation-converged',
            'energy':E,'charge':Q,'energy_over_charge':E/Q,
            'center_first_coefficient':float(a[0]),'center_rotating_coefficient':float(c[0]),
            'maximum_norm_squared_deviation':float(np.max(np.abs(norm_delta))),
            'minimum_first_coefficient':float(np.min(a)),'minimum_rotating_coefficient':float(np.min(c)),
            'minimum_lapse':float(np.min(B)),'minimum_momentum_margin':float(np.min(margin)),
            'relative_virial':virial/E,'max_bvp_residual':float(np.max(solution.rms_residuals)),
            'collocation_nodes':len(solution.x),'bvp_tolerance':tolerance}
    return result,solution


def main():
    derive()
    cases=[]
    diagnostics=[]
    for omega in (0.95,0.97):
        for stiffness in (25,100,400,1600):
            Bstar=(omega**4/(2*(stiffness-0.5)))**(2/3)
            maximum=(stiffness-0.5)/omega**2*(1-3*Bstar)+1-stiffness
            diagnostics.append({'omega':omega,'norm_stiffness':stiffness,'core_maximum':maximum})
        for depth in (2,3,4,5,6):
            constrained=radial.solve_depth(depth,omega,radius=110.0,tolerance=2e-9)
            if constrained['status']!='collocation-converged':raise AssertionError(constrained)
            seed=radial.PROFILES[(depth,omega,110.0,2e-9)]
            for stiffness in (1600,400):
                base,_=solve_amplitudes(depth,omega,stiffness,80.0,1e-7,seed)
                refined,solution=solve_amplitudes(depth,omega,stiffness,110.0,1e-9,seed)
                if refined['status']=='collocation-converged' and base['status']=='collocation-converged':
                    refined['convergence']={key:abs(refined[key]-base[key])/max(abs(refined[key]),1) for key in ('energy','charge','center_rotating_coefficient')}
                    if max(refined['convergence'].values())>1e-5:raise AssertionError(('boundary convergence',refined))
                    refined['relative_to_constrained']={key:(refined[key]-constrained[key])/constrained[key] for key in ('energy','charge')}
                    seed=solution
                refined['coarse_status']=base['status']
                cases.append(refined)
                print(json.dumps({key:refined[key] for key in ('depth','omega','norm_stiffness','status')}),flush=True)
    cases.append({'depth':10,'normal_rank':0,'status':'no-active-normal-generator'})
    path=Path(__file__).resolve()
    report={'created_at':datetime.now(timezone.utc).isoformat(),
            'experiment':'finite-amplitude-complete-coefficient-localization',
            'sources':{'experiments/finite-amplitude-checks.py':sha256(path.read_bytes()).hexdigest(),
                       'experiments/rotating-determinant-checks.py':sha256(PARENT.read_bytes()).hexdigest()},
            'exact_checks':len(CHECKS),'checks':CHECKS,
            'parameters':{'beta':1,'T':4,'kappa':4,'m_squared':1,'quartic_b':'1/4'},
            'core_diagnostics':diagnostics,'cases':cases,
            'scope':'Unconstrained complex positive wave field with a common polynomial norm stiffness. All unused polarizations have conditional positive comparison solutions when both active radial coefficients are positive. Coupled two-amplitude radial stability, Dirac kinetics and simplex frequency selection remain unresolved.'}
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    output=path.with_name('finite-amplitude-results-'+stamp+'.json')
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'exact_checks':len(CHECKS),'cases':len(cases),'report':str(output)}),flush=True)


if __name__=='__main__':main()
