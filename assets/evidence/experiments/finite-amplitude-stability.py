#!/usr/bin/env python3
"""Complete two-amplitude Hessian and charge response of the finite-stiffness law."""

from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import importlib.util
import json
import math
import numpy as np
import sympy as sp
from scipy.integrate import solve_bvp,simpson
from scipy.linalg import eig_banded
from scipy.special import kve,gamma


ROOT=Path(__file__).resolve().parent
PARENT=ROOT/'finite-amplitude-checks.py'
spec=importlib.util.spec_from_file_location('finite_amplitude',PARENT)
finite=importlib.util.module_from_spec(spec)
spec.loader.exec_module(finite)
EXACT=[]


def equal(name,lhs,rhs):
    value=sp.simplify(lhs-rhs)
    if value!=0:raise AssertionError((name,value))
    EXACT.append(name)


def differential_functions():
    delta,c,pa,pc,w,M2=sp.symbols('delta c Pa Pc omega M2',real=True)
    nd=delta*(delta+2)+c*c
    S=1-w*w*c*c-pa*pa-pc*pc
    g=[pa/sp.sqrt(S),pc/sp.sqrt(S),M2*nd*(1+delta),
       (M2*nd+1-c*c/2-w*w/sp.sqrt(S))*c]
    variables=(delta,c,pa,pc)
    args=(*variables,w,M2)
    jac=[[sp.lambdify(args,sp.diff(g[i],v),'numpy') for v in variables] for i in range(4)]
    src=[sp.lambdify(args,sp.diff(expr,w),'numpy') for expr in g]
    charge=4*w*c*c/sp.sqrt(S)
    energy=4*(1/sp.sqrt(S)-1)+4*(M2*nd**2/4+(c*c-c**4/4)/2)
    qdiff=[sp.lambdify(args,sp.diff(charge,v),'numpy') for v in (*variables,w)]
    ediff=[sp.lambdify(args,sp.diff(energy,v),'numpy') for v in (*variables,w)]
    a,p0,p1=sp.symbols('a p0 p1',real=True)
    A=1+p0*p0+p1*p1
    B=1-w*w*c*c
    R=sp.sqrt(A)*sp.sqrt(B)-1+M2*(a*a+c*c-1)**2/4+(c*c-c**4/4)/2
    H=sp.eye(2)-sp.Matrix([p0,p1])*sp.Matrix([p0,p1]).T/A
    pp=sp.hessian(R,(p0,p1))
    for i in range(2):
        for j in range(2):equal(f'complete radial kinetic Hessian {i},{j}',pp[i,j],sp.sqrt(B/A)*H[i,j])
    eps,rho,weight,m2,b=sp.symbols('eps rho weight m2 b',real=True)
    s=a*a+c*c+eps*eps*rho
    z=c*c+eps*eps*weight*weight*rho
    V=M2*(s-1)**2/4+(m2*z-b*z*z)/2
    equal('unconstrained coefficient potential variation',sp.expand(V).coeff(eps,2),
          (M2*(a*a+c*c-1)+weight*weight*(m2-2*b*c*c))*rho/2)
    return jac,src,qdiff,ediff


def sensitivity(solution,depth,omega,stiffness,functions):
    jac,src,qdiff,ediff=functions
    radius=110.0
    def fun(rr,z):
        vals=solution.sol(rr)
        args=(*vals,omega,stiffness)
        return np.array([sum(jac[i][j](*args)*z[j] for j in range(4))+src[i](*args) for i in range(4)])
    mu=math.sqrt(1-omega*omega)
    nu=(depth-2)/2
    zz=mu*radius
    ratio=kve(nu+1,zz)/kve(nu,zz)
    robin=mu*ratio
    dr=-omega/mu*(ratio+zz*(ratio*ratio-1-(2*nu+1)*ratio/zz))
    end=solution.sol(radius)
    args=(*end,omega,stiffness)
    def boundary(left,right):
        last=sum(jac[1][j](*args)*right[j] for j in range(4))+src[1](*args)+robin*right[1]+dr*end[1]
        return np.array([left[2],left[3],right[0],last])
    response=solve_bvp(fun,boundary,solution.x,np.zeros((4,len(solution.x))),
                       S=np.diag([0,0,-(depth-1),-(depth-1)]),tol=1e-8,max_nodes=24000)
    if not response.success:raise AssertionError(response.message)
    rr=np.linspace(0,radius,33001)
    vals=solution.sol(rr)
    z=response.sol(rr)
    args=(*vals,omega,stiffness)
    measure=2*math.pi**(depth/2)/gamma(depth/2)*rr**(depth-1)
    integrate=lambda f:float(simpson(f*measure,x=rr))
    Qprime=integrate(sum(qdiff[j](*args)*z[j] for j in range(4))+qdiff[4](*args))
    Eprime=integrate(sum(ediff[j](*args)*z[j] for j in range(4))+ediff[4](*args))
    c,pa,pc=vals[1:]
    B=1-omega*omega*c*c
    A=B/(B-pa*pa-pc*pc)
    fixed=integrate(4*c*c*np.sqrt(A)/B**1.5)
    error=(Eprime-omega*Qprime)/max(abs(Eprime),1)
    if abs(error)>1e-6:raise AssertionError(('variational identity',error))
    return {'dcharge_domega':Qprime,'denergy_domega':Eprime,
            'fixed_shape_charge_susceptibility':fixed,'fixed_charge_rank_update_ratio':Qprime/fixed,
            'family_identity_relative_error':error,'sensitivity_residual':float(np.max(response.rms_residuals))}


def spectrum(solution,depth,omega,stiffness,intervals,ell):
    radius=110.0
    step=radius/intervals
    xi=np.array([-math.sqrt(3/5),0,math.sqrt(3/5)])
    gauss=np.array([5/9,8/9,5/9])
    rr=np.arange(intervals)[:,None]*step+step*(xi+1)/2
    delta,c,pa,pc=solution.sol(rr.reshape(-1)).reshape(4,intervals,3)
    a=1+delta
    B=1-omega*omega*c*c
    S=B-pa*pa-pc*pc
    A=B/S
    slopes=np.stack([pa,pc],axis=-1)/np.sqrt(S)[...,None]
    H=np.eye(2)-slopes[..., :,None]*slopes[...,None,:]/A[...,None,None]
    Cr=np.sqrt(B/A)
    P=Cr[...,None,None]*H
    U=np.empty((*rr.shape,2,2))
    nd=delta*(delta+2)+c*c
    U[...,0,0]=stiffness*(nd+2*a*a)
    U[...,0,1]=U[...,1,0]=2*stiffness*a*c
    U[...,1,1]=stiffness*(nd+2*c*c)+1-1.5*c*c-omega*omega*np.sqrt(A)/B**1.5
    U+=np.sqrt(A*B)[...,None,None]*H*(ell*(ell+depth-2)/rr**2)[...,None,None]
    J=np.zeros_like(U)
    J[...,1,:]=-omega*omega*c[...,None]*slopes/np.sqrt(A*B)[...,None]
    weights=step/2*gauss[None,:]*rr**(depth-1)
    basis=[(1-xi)/2,(1+xi)/2]
    deriv=[-1/step,1/step]
    count=2*(intervals+1)
    bands=np.zeros((4,count))
    lump=np.zeros(count)
    for node in (0,1):
        values=np.sum(weights*basis[node],axis=1)
        lump[2*node:2*(intervals+node):2]+=values
        lump[2*node+1:2*(intervals+node)+1:2]+=values
    for n in (0,1):
        for m in (0,1):
            block=np.sum(weights[...,None,None]*(
                P*deriv[n]*deriv[m]+U*(basis[n]*basis[m])[None,:,None,None]
                +J*(basis[n]*deriv[m])[None,:,None,None]
                +np.swapaxes(J,-1,-2)*(deriv[n]*basis[m])[None,:,None,None]),axis=1)
            for i in (0,1):
                for j in (0,1):
                    offset=2*(n-m)+i-j
                    if offset<0:continue
                    columns=2*np.arange(intervals)+2*m+j
                    np.add.at(bands[offset],columns,block[:,i,j])
    first=2 if ell else 0
    last=count-2
    bands=bands[:,first:last]
    lump=lump[first:last]
    for offset in range(4):
        length=len(lump)-offset
        bands[offset,:length]/=np.sqrt(lump[:length]*lump[offset:])
    values=eig_banded(bands,lower=True,eigvals_only=True,select='i',select_range=(0,3))
    return [float(v) for v in values]


def main():
    finite.derive()
    functions=differential_functions()
    cases=[]
    for omega in (0.95,0.97):
        for depth in (2,3,4,5,6):
            constrained=finite.radial.solve_depth(depth,omega,radius=110.0,tolerance=2e-9)
            if constrained['status']!='collocation-converged':raise AssertionError(constrained)
            seed=finite.radial.PROFILES[(depth,omega,110.0,2e-9)]
            for stiffness in (1600,400):
                coarse,_=finite.solve_amplitudes(depth,omega,stiffness,80.0,1e-7,seed)
                result,solution=finite.solve_amplitudes(depth,omega,stiffness,110.0,1e-9,seed)
                if result['status']!='collocation-converged' or coarse['status']!='collocation-converged':raise AssertionError(result)
                result['convergence']={key:abs(result[key]-coarse[key])/max(abs(result[key]),1) for key in ('energy','charge','center_rotating_coefficient')}
                if max(result['convergence'].values())>1e-5:raise AssertionError(('boundary convergence',result))
                result.update(sensitivity(solution,depth,omega,stiffness,functions))
                result['coupled_amplitude_energy_form_spectra']={str(n):{
                    'ell'+str(ell):spectrum(solution,depth,omega,stiffness,n,ell) for ell in (0,1)} for n in (1100,2200)}
                cases.append(result)
                seed=solution
                print(json.dumps({key:result[key] for key in ('depth','omega','norm_stiffness','dcharge_domega')}),flush=True)
    cases.append({'depth':10,'normal_rank':0,'status':'no-active-normal-generator'})
    path=Path(__file__).resolve()
    sources={str('experiments/'+p.name):sha256(p.read_bytes()).hexdigest() for p in (path,PARENT,finite.PARENT)}
    report={'created_at':datetime.now(timezone.utc).isoformat(),
            'experiment':'finite-amplitude-complete-coefficient-stability','sources':sources,
            'exact_checks':len(finite.CHECKS)+len(EXACT),'checks':finite.CHECKS+EXACT,'cases':cases,
            'scope':'Complete coupled radial amplitude energy-form index, charge sensitivity and boundary/mesh controls for one finite-stiffness positive wave action. Energy-form eigenvalues use a positive ordinary radial mass for index counting and are not physical oscillation or growth frequencies. Unused-polarization positivity is a conditional analytic result. No nonlinear stability theorem, Lorentz Dirac dynamics or selected simplex spectrum is claimed.'}
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    output=path.with_name('finite-amplitude-stability-'+stamp+'.json')
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'exact_checks':report['exact_checks'],'cases':len(cases),'report':str(output)}),flush=True)


if __name__=='__main__':main()
