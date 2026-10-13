#!/usr/bin/env python3
"""Derive and solve a bounded common rotating-orientation comparison."""

from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import json
import math

import numpy as np
from scipy.integrate import solve_ivp, solve_bvp, simpson
from scipy.special import kve, gamma
from scipy.linalg import eigh_tridiagonal
import sympy as sp


BETA=0.25
TENSION=4.0
MASS2=1.0
ATTRACTIVE=0.25
CHECKS=[]
PROFILES={}


def equal(name,lhs,rhs):
    value=sp.simplify(lhs-rhs)
    if value!=0:
        raise AssertionError((name,value))
    CHECKS.append(name)


def derive():
    f,p,pp,beta,omega,m2,b,T,d,r=sp.symbols('f p pp beta omega m2 b T d r',real=True)
    a=1+beta*p*p
    bb=1-beta*omega*omega*sp.sin(f)**2
    potential=T*beta*(m2*sp.sin(f)**2-b*sp.sin(f)**4)/2
    lag=T*(1-sp.sqrt(a)*sp.sqrt(bb))-potential
    momentum=sp.diff(lag,p)
    euler=sp.diff(momentum,f)*p+sp.diff(momentum,p)*pp+(d-1)*momentum/r-sp.diff(lag,f)
    predicted=a*(sp.sin(f)*sp.cos(f)*((m2-2*b*sp.sin(f)**2)*sp.sqrt(a)/sp.sqrt(bb)-omega**2/bb)-(d-1)*p/r)
    equal('complete radial Euler equation',euler,sp.diff(momentum,p)*(pp-predicted))
    charge=sp.diff(lag,omega)
    equal('complete target charge',charge,T*beta*omega*sp.sin(f)**2*sp.sqrt(a)/sp.sqrt(bb))
    energy=omega*charge-lag
    equal('complete rest energy',energy,T*(sp.sqrt(a)/sp.sqrt(bb)-1)+potential)
    equal('stationary Routhian',energy-omega*charge,T*(sp.sqrt(a)*sp.sqrt(bb)-1)+potential)
    pr=T*(1-sp.sqrt(bb)/sp.sqrt(a))-potential
    equal('radial stress balance',sp.diff(pr,f)*p+sp.diff(pr,p)*predicted,
          -(d-1)*T*beta*p*p*sp.sqrt(bb)/(r*sp.sqrt(a)))
    threshold=sp.Rational(3,4)-sp.Rational(1,4)*sp.Rational(3,4)**2/4
    equal('equator negative-region threshold',threshold,sp.Rational(183,256))


def factors(f,p):
    # Here p is the bounded canonical radial momentum, not F-prime.
    b=1-BETA*OMEGA**2*np.sin(f)**2
    margin=b-BETA*p*p
    return b,margin


def transverse_geometry_checks():
    # Expand actual normalized ray jets, independently of the radial Hessian.
    eps,f,p,w,u,v,ut,vt,ur,vr=sp.symbols('eps f p w u v ut vt ur vr',real=True)
    rho=u*u+v*v
    c,s=sp.cos(f/2),sp.sin(f/2)
    base=sp.Matrix([c,s,0])
    base_t=sp.Matrix([-sp.I*w*c/2,sp.I*w*s/2,0])
    base_r=sp.Matrix([-p*s/2,p*c/2,0])
    extra=sp.Matrix([0,0,u+sp.I*v])
    ray=(1-eps**2*rho/2)*base+eps*extra
    dt=(1-eps**2*rho/2)*base_t-eps**2*(u*ut+v*vt)*base+sp.Matrix([0,0,eps*(ut+sp.I*vt)])
    dr=(1-eps**2*rho/2)*base_r-eps**2*(u*ur+v*vr)*base+sp.Matrix([0,0,eps*(ur+sp.I*vr)])
    inner=lambda a,b:(sp.conjugate(a).T*b)[0]
    metric=lambda a,b:sp.re(inner(a,b)-inner(a,ray)*inner(ray,b))
    coefficient=lambda expr:sp.trigsimp(sp.expand(expr).coeff(eps,2))
    norm_t=(ut-w*sp.cos(f)*v/2)**2+(vt+w*sp.cos(f)*u/2)**2
    expected_t=norm_t-w*w*sp.sin(f)**2*rho/4
    expected_r=ur*ur+vr*vr-p*p*rho/4
    expected_tr=(ut-w*sp.cos(f)*v/2)*ur+(vt+w*sp.cos(f)*u/2)*vr
    equal('normalized ray transverse time metric',sp.trigsimp(coefficient(metric(dt,dt))-expected_t),0)
    equal('normalized ray transverse radial metric',sp.trigsimp(coefficient(metric(dr,dr))-expected_r),0)
    equal('normalized ray mixed metric',sp.trigsimp(coefficient(metric(dt,dr))-expected_tr),0)
    generator=sp.diag(-sp.Rational(1,2),sp.Rational(1,2),0)
    killing=4*(inner(ray,generator*generator*ray)-inner(ray,generator*ray)**2)
    equal('full Killing potential transverse variation',
          sp.trigsimp(coefficient(killing)-rho*(1-2*sp.sin(f)**2)),0)
    aa,bb,beta,T,gtt,grr,gtr=sp.symbols('A B beta T gtt grr gtr',positive=True)
    # FS has metric factor sixteen; beta=4 lambda, hence 4 beta times FS.
    delta=(bb-4*beta*eps**2*gtt)*(aa+4*beta*eps**2*grr)+(4*beta*eps**2*gtr)**2
    quadratic=sp.diff(T*(1-sp.sqrt(delta)),eps,2).subs(eps,0)/2
    equal('full determinant transverse quadratic form',quadratic,
          2*T*beta*(sp.sqrt(aa/bb)*gtt-sp.sqrt(bb/aa)*grr))
    alpha=sp.symbols('alpha',real=True)
    n=sp.Matrix([sp.sin(f)*sp.cos(alpha),sp.sin(f)*sp.sin(alpha),sp.cos(f)])
    equal('quaternion radial electric curvature coefficient',
          (w*n.diff(alpha)).cross(p*n.diff(f))[0],-w*p*sp.sin(f)*n[0])
    for index in (1,2):
        equal(f'quaternion electric curvature component {index}',
              (w*n.diff(alpha)).cross(p*n.diff(f))[index],-w*p*sp.sin(f)*n[index])


def rhs(r,y,depth):
    f,p=y
    b,margin=factors(f,p)
    root=np.sqrt(np.maximum(margin,1e-14))
    sc=np.sin(f)*np.cos(f)
    return [p/root,sc*(MASS2-2*ATTRACTIVE*np.sin(f)**2)-OMEGA**2*sc/root-(depth-1)*p/r]


def shoot(center,depth,tolerance=1e-9):
    eps=1e-6
    b0=1-BETA*OMEGA**2*math.sin(center)**2
    sc=math.sin(center)*math.cos(center)
    second=sc*((MASS2-2*ATTRACTIVE*math.sin(center)**2)/math.sqrt(b0)-OMEGA**2/b0)/depth
    p0=eps*second*math.sqrt(b0)
    if p0>=0:
        return 'invalid-center',None
    def crossing(r,y):return y[0]
    crossing.terminal=True
    crossing.direction=-1
    def turning(r,y):return y[1]
    turning.terminal=True
    turning.direction=1
    def boundary(r,y):return factors(y[0],y[1])[1]-1e-8
    boundary.terminal=True
    boundary.direction=-1
    sol=solve_ivp(lambda r,y:rhs(r,y,depth),(eps,120.0),[center+second*eps**2/2,p0],
                  events=(crossing,turning,boundary),rtol=tolerance,atol=tolerance/100,max_step=0.25,dense_output=True)
    if not sol.success:return 'integration-failure',sol
    if len(sol.t_events[0]):return 'overshoot',sol
    if len(sol.t_events[1]):return 'undershoot',sol
    if len(sol.t_events[2]):return 'momentum-boundary',sol
    return 'radius-limit',sol


def solve_depth(depth,omega,radius=80.0,tolerance=1e-8):
    global OMEGA
    OMEGA=omega
    trials=[]
    lower=upper=None
    centers=list(np.linspace(0.4,1.5,18))+[math.pi/2-10.0**(-j) for j in range(1,12)]
    previous=None
    for center in sorted(set(centers)):
        outcome,sol=shoot(center,depth)
        trials.append({'center':center,'outcome':outcome})
        if outcome=='undershoot':previous=center
        if outcome=='overshoot' and previous is not None:
            lower,upper=previous,center
            break
    if lower is None:
        return {'depth':depth,'omega':omega,'status':'no-shooting-bracket','trials':trials}
    for iteration in range(44):
        center=(lower+upper)/2
        outcome,sol=shoot(center,depth)
        if outcome=='undershoot':lower=center
        elif outcome=='overshoot':upper=center
        else:return {'depth':depth,'omega':omega,'status':outcome,'center':center,'trials':trials}
    center=(lower+upper)/2
    outcome,sol=shoot(center,depth,tolerance=1e-10)
    decay=math.sqrt(MASS2-omega**2)
    nu=(depth-2)/2
    robin=decay*kve(nu+1,decay*radius)/kve(nu,decay*radius)
    # Preserve only the decreasing, positive shooting segment, then append its linear tail.
    sample=np.linspace(sol.t[0],sol.t[-1],2000)
    ff,pp=sol.sol(sample)
    good=np.where(ff>1e-4)[0]
    if not len(good):raise AssertionError('no positive seed')
    last=good[-1]
    match=sample[last]
    matchf=ff[last]
    mesh=np.linspace(0,radius,int(radius*12)+1)
    seed=np.empty((2,len(mesh)))
    inner=mesh<=match
    seed[:,inner]=sol.sol(np.maximum(mesh[inner],sol.t[0]))
    rr=mesh[~inner]
    tail=matchf*(rr/match)**(-nu)*np.exp(-decay*(rr-match))*kve(nu,decay*rr)/kve(nu,decay*match)
    slope=-decay*kve(nu+1,decay*rr)/kve(nu,decay*rr)*tail
    seed[0,~inner]=tail
    seed[1,~inner]=slope*np.sqrt(1-BETA*omega**2*np.sin(tail)**2)/np.sqrt(1+BETA*slope**2)
    seed[1,0]=0
    def fun(r,y):
        f,p=y
        bb,margin=factors(f,p)
        root=np.sqrt(np.maximum(margin,1e-14))
        sc=np.sin(f)*np.cos(f)
        return np.array([p/root,sc*(MASS2-2*ATTRACTIVE*np.sin(f)**2)-omega**2*sc/root])
    def boundary(left,right):
        bb,margin=factors(right[0],right[1])
        return np.array([left[1],right[1]/math.sqrt(max(margin,1e-14))+robin*right[0]])
    singular=np.diag([0,-(depth-1)])
    solution=solve_bvp(fun,boundary,mesh,seed,S=singular,tol=tolerance,max_nodes=18000)
    if not solution.success:
        return {'depth':depth,'omega':omega,'status':'collocation-failure','message':solution.message,'center':center}
    rr=np.linspace(0,radius,int(radius*250)+1)
    f,p=solution.sol(rr)
    bb,margin=factors(f,p)
    slope=p/np.sqrt(margin)
    aa=1+BETA*slope*slope
    kappa=TENSION*BETA
    pot=kappa*(MASS2*np.sin(f)**2-ATTRACTIVE*np.sin(f)**4)/2
    # Rationalized expressions avoid subtracting one from a near-vacuum square root.
    ratio=np.sqrt(aa/bb)
    energy=TENSION*(aa-bb)/(bb*(ratio+1))+pot
    charge=kappa*omega*np.sin(f)**2*ratio
    prod=np.sqrt(aa*bb)
    pressure_r=TENSION*(aa-bb)/(aa*(1+np.sqrt(bb/aa)))-pot
    pressure_t=-TENSION*(aa*bb-1)/(prod+1)-pot
    measure=2*math.pi**(depth/2)/gamma(depth/2)*rr**(depth-1)
    integrate=lambda arr:float(simpson(arr*measure,x=rr))
    total=integrate(energy)
    q=integrate(charge)
    virial=integrate(pressure_r+(depth-1)*pressure_t)
    result={'depth':depth,'normal_rank':{2:4,3:3,4:3,5:2,6:2}[depth],
            'omega':omega,'status':'collocation-converged','center':float(f[0]),
            'energy':total,'charge':q,'energy_over_charge':total/q,
            'energy_minus_rotation_charge':total-omega*q,
            'virial':virial,'relative_virial':virial/total,
            'minimum_lapse':float(np.min(bb)),'minimum_momentum_margin':float(np.min(margin)),
            'maximum_principal_radial_speed':float(np.sqrt(np.max(bb/aa))),
            'maximum_principal_tangential_speed':float(np.sqrt(np.max(bb))),
            'max_bvp_residual':float(np.max(solution.rms_residuals)),
            'bvp_tolerance':tolerance,
            'boundary_amplitude':float(f[-1]),'radius':radius,'collocation_nodes':len(solution.x),
            'half_center_radius':float(np.interp(f[0]/2,f[::-1],rr[::-1])),
            'shooting_bracket':[lower,upper]}
    if np.max(f)<0.1 or np.min(f)<-1e-6:raise AssertionError('trivial or nodeful branch')
    if np.min(bb)<=0 or np.min(margin)<=0:raise AssertionError('inadmissible branch')
    if abs(virial/total)>1e-5:raise AssertionError(('virial',result))
    PROFILES[(depth,omega,radius,tolerance)]=solution
    return result


def normal_generator_checks():
    ranks=[]
    for depth,q in ((2,4),(3,3),(4,3),(5,2),(6,2)):
        j0=sp.zeros(q).row_join(-sp.eye(q)).col_join(sp.eye(q).row_join(sp.zeros(q)))
        hcomplex=sp.diag(sp.I/2,sp.I/2,*([0]*(q-2)))
        h=sp.re(hcomplex).row_join(-sp.im(hcomplex)).col_join(sp.im(hcomplex).row_join(sp.re(hcomplex)))
        equal(f'd{depth} common root generator norm',sp.trace(h.T*h),1)
        weights=[]
        for a in range(q):
            for b in range(a+1,q):
                e=sp.zeros(q);e[a,b],e[b,a]=1,-1
                for imaginary in (False,True):
                    x=(sp.zeros(q).row_join(e).col_join(e.row_join(sp.zeros(q))) if imaginary else e.row_join(sp.zeros(q)).col_join(sp.zeros(q).row_join(-e)))/4
                    dj=x*j0-j0*x
                    dk=h*dj-dj*h
                    weight=sp.Rational((1 if a<2 else 0)+(1 if b<2 else 0),2)
                    equal(f'd{depth} full target Killing Hessian {a},{b},{imaginary}',sp.trace(dk.T*dk),weight**2)
                    weights.append(str(weight))
        ranks.append({'depth':depth,'normal_rank':q,'target_weights':weights,
                      'zero_weight_dimension':weights.count('0')})
    ranks.append({'depth':10,'normal_rank':0,'target_weights':[],'zero_weight_dimension':0})
    return ranks


def differential_functions():
    f,p,w=sp.symbols('F P omega',real=True)
    beta=sp.Rational(1,4)
    y=sp.sin(f)**2
    bb=1-beta*w*w*y
    margin=bb-beta*p*p
    g=sp.Matrix([p/sp.sqrt(margin),sp.sin(f)*sp.cos(f)*(1-y/2)-w*w*sp.sin(f)*sp.cos(f)/sp.sqrt(margin)])
    jac=[[sp.lambdify((f,p,w),sp.diff(g[i],v),'numpy') for v in (f,p)] for i in range(2)]
    src=[sp.lambdify((f,p,w),sp.diff(g[i],w),'numpy') for i in range(2)]
    charge=w*y/sp.sqrt(margin)
    energy=4*(1/sp.sqrt(margin)-1)+(y-y*y/4)/2
    qdiff=[sp.lambdify((f,p,w),sp.diff(charge,v),'numpy') for v in (f,p,w)]
    ediff=[sp.lambdify((f,p,w),sp.diff(energy,v),'numpy') for v in (f,p,w)]
    # Physical slope, rather than the momentum variable, is used in this Hessian.
    slope=sp.Symbol('slope',real=True)
    aa=1+beta*slope*slope
    routh=4*(sp.sqrt(aa)*sp.sqrt(bb)-1)+(y-y*y/4)/2
    rfp=sp.diff(routh,f,slope)
    hessian=[sp.lambdify((f,slope,w),expr,'numpy') for expr in
             (sp.diff(routh,f,2),rfp,sp.diff(rfp,f),sp.diff(rfp,slope))]
    return jac,src,qdiff,ediff,hessian


def sensitivity(solution,depth,omega,radius,functions):
    jac,src,qdiff,ediff,_=functions
    def fun(rr,z):
        f,p=solution.sol(rr)
        rows=[]
        for i in range(2):
            rows.append(jac[i][0](f,p,omega)*z[0]+jac[i][1](f,p,omega)*z[1]+src[i](f,p,omega))
        return np.array(rows)
    decay=math.sqrt(MASS2-omega**2)
    nu=(depth-2)/2
    zz=decay*radius
    ratio=kve(nu+1,zz)/kve(nu,zz)
    robin=decay*ratio
    robin_derivative=-omega/decay*(ratio+zz*(ratio*ratio-1-(2*nu+1)*ratio/zz))
    fr,pr=solution.sol(radius)
    def boundary(left,right):
        value=(jac[0][0](fr,pr,omega)+robin)*right[0]+jac[0][1](fr,pr,omega)*right[1]
        value+=src[0](fr,pr,omega)+robin_derivative*fr
        return np.array([left[1],value])
    derivative=solve_bvp(fun,boundary,solution.x,np.zeros((2,len(solution.x))),
                         S=np.diag([0,-(depth-1)]),tol=2e-9,max_nodes=18000)
    if not derivative.success:raise AssertionError(derivative.message)
    rr=np.linspace(0,radius,int(radius*250)+1)
    f,p=solution.sol(rr)
    zf,zp=derivative.sol(rr)
    bb,margin=factors(f,p)
    aa=bb/margin
    measure=2*math.pi**(depth/2)/gamma(depth/2)*rr**(depth-1)
    integrate=lambda values:float(simpson(values*measure,x=rr))
    qprime=integrate(qdiff[0](f,p,omega)*zf+qdiff[1](f,p,omega)*zp+qdiff[2](f,p,omega))
    eprime=integrate(ediff[0](f,p,omega)*zf+ediff[1](f,p,omega)*zp+ediff[2](f,p,omega))
    fixed=integrate(np.sin(f)**2*np.sqrt(aa)/bb**1.5)
    identity=(eprime-omega*qprime)/max(abs(eprime),1)
    if abs(identity)>1e-6:raise AssertionError(('family variational identity',depth,omega,identity))
    return {'dcharge_domega':qprime,'denergy_domega':eprime,
            'fixed_shape_charge_susceptibility':fixed,'fixed_charge_rank_update_ratio':qprime/fixed,
            'family_identity_relative_error':identity,
            'sensitivity_bvp_residual':float(np.max(derivative.rms_residuals))}


def radial_spectrum(solution,depth,omega,radius,intervals,functions,kind,ell):
    # Piecewise-linear radial finite elements, three-point Gaussian quadrature,
    # lumped positive mass and a symmetric tridiagonal generalized operator.
    grid=np.linspace(0,radius,intervals+1)
    step=radius/intervals
    xi=np.array([-math.sqrt(3/5),0,math.sqrt(3/5)])
    weights=np.array([5/9,8/9,5/9])
    rr=grid[:-1,None]+step*(xi+1)/2
    f,p=solution.sol(rr.reshape(-1))
    f,p=f.reshape(rr.shape),p.reshape(rr.shape)
    bb,margin=factors(f,p)
    slope=p/np.sqrt(margin)
    aa=1+BETA*slope*slope
    c0=np.sqrt(aa/bb)
    cr=np.sqrt(bb/aa)
    ct=np.sqrt(aa*bb)
    yy=np.sin(f)**2
    if kind=='overlap':
        potential=((1-2*yy)*(MASS2-2*ATTRACTIVE*yy-c0*omega**2)-cr*slope*slope)/4
        mass=c0
        gradient=cr
        angular=ct
    else:
        rff,rfp,rfpf,rfpp=functions[-1]
        sc=np.sin(f)*np.cos(f)
        second=aa*(sc*((MASS2-2*ATTRACTIVE*yy)*np.sqrt(aa/bb)-omega**2/bb)-(depth-1)*slope/rr)
        potential=rff(f,slope,omega)-rfpf(f,slope,omega)*slope-rfpp(f,slope,omega)*second-(depth-1)*rfp(f,slope,omega)/rr
        mass=1/np.sqrt(aa*bb)
        gradient=np.sqrt(bb)/aa**1.5
        angular=cr
    potential=potential+angular*ell*(ell+depth-2)/rr**2
    weight=step/2*weights[None,:]*rr**(depth-1)
    n0,n1=(1-xi)/2,(1+xi)/2
    stiff=np.sum(weight*gradient,axis=1)/step**2
    v00=np.sum(weight*potential*n0*n0,axis=1)
    v11=np.sum(weight*potential*n1*n1,axis=1)
    v01=np.sum(weight*potential*n0*n1,axis=1)
    diag=np.zeros(intervals+1)
    massdiag=np.zeros(intervals+1)
    diag[:-1]+=stiff+v00
    diag[1:]+=stiff+v11
    massdiag[:-1]+=np.sum(weight*mass*n0,axis=1)
    massdiag[1:]+=np.sum(weight*mass*n1,axis=1)
    off=-stiff+v01
    first=1 if ell else 0
    diag=diag[first:-1]
    off=off[first:-1]
    massdiag=massdiag[first:-1]
    eigenvalues=eigh_tridiagonal(diag/massdiag,off/np.sqrt(massdiag[:-1]*massdiag[1:]),
                                  select='i',select_range=(0,3),eigvals_only=True)
    return [float(v) for v in eigenvalues]


def main():
    derive()
    transverse_geometry_checks()
    normal_ranks=normal_generator_checks()
    functions=differential_functions()
    cases=[]
    for omega in (0.95,0.97):
        for depth in (2,3,4,5,6):
            result=solve_depth(depth,omega)
            refined=solve_depth(depth,omega,radius=110.0,tolerance=2e-10)
            if result['status']=='collocation-converged' and refined['status']=='collocation-converged':
                refined['convergence']={key:abs(refined[key]-result[key])/abs(refined[key]) for key in ('energy','charge','center')}
                if max(refined['convergence'].values())>1e-6:raise AssertionError(('boundary convergence',refined))
                solution=PROFILES[(depth,omega,110.0,2e-10)]
                refined.update(sensitivity(solution,depth,omega,110.0,functions))
                spectra={}
                for intervals in (1100,2200):
                    values={}
                    for ell in (0,1):
                        values['shape_l'+str(ell)]=radial_spectrum(solution,depth,omega,110.0,intervals,functions,'shape',ell)
                    if depth in (2,3,4):
                        values['overlap_l0']=radial_spectrum(solution,depth,omega,110.0,intervals,functions,'overlap',0)
                    spectra[str(intervals)]=values
                refined['radial_quadratic_spectra']=spectra
                result['refined']=refined
            cases.append(result)
            print(json.dumps({key:result[key] for key in ('depth','omega','status')}),flush=True)
    cases.append({'depth':10,'normal_rank':0,'status':'empty-normal-target'})
    path=Path(__file__).resolve()
    report={'created_at':datetime.now(timezone.utc).isoformat(),
            'experiment':'rotating-determinant-orientation',
            'sources':{'experiments/rotating-determinant-checks.py':sha256(path.read_bytes()).hexdigest()},
            'parameters':{'beta':BETA,'T':TENSION,'m_squared':MASS2,'b':ATTRACTIVE},
            'exact_checks':len(CHECKS),'normal_generator_comparison':normal_ranks,'cases':cases,
            'scope':'A fixed normal generator and bounded Killing-norm potential are added constitutive premises. A finite radial boundary calculation tests a common action; no discrete mass-frequency selection, complete stability theorem, source force or color identification is claimed.'}
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    output=path.with_name('rotating-determinant-results-'+stamp+'.json')
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'exact_checks':len(CHECKS),'cases':len(cases),'report':str(output)}),flush=True)


if __name__=='__main__':
    main()
