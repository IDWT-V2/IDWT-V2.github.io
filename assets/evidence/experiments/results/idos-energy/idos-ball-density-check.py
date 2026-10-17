"""Independent separated Jacobi kernel and derivative checks, no DB writes."""
import math, json, hashlib
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
from scipy.special import eval_jacobi, eval_gegenbauer, gammaln, roots_jacobi
import sympy as s
DIMS=(2,3,4,5,6,10)
def h(d,l):
    return math.comb(l+d-1,d-1)-(math.comb(l+d-3,d-1) if l>=2 else 0)
def W(d): return math.pi**(d/2)*math.gamma(1.5)/math.gamma((d+3)/2)
def kernel(d,N,r):
    r=np.asarray(r,dtype=float); K=np.zeros_like(r); deriv=np.zeros_like(r)
    area=2*math.pi**(d/2)/math.gamma(d/2)
    for l in range(N+1):
        beta=l+(d-2)/2
        for j in range((N-l)//2+1):
            norm=.5*math.exp(gammaln(j+1.5)+gammaln(j+beta+1)-gammaln(j+1)-gammaln(j+beta+1.5))/(2*j+beta+1.5)
            p=eval_jacobi(j,.5,beta,2*r*r-1)
            f=r**l*p
            df=(l*r**(l-1)*p if l else np.zeros_like(r))
            if j: df+=2*r**(l+1)*(j+.5+beta+1)*eval_jacobi(j-1,1.5,beta+1,2*r*r-1)
            coeff=h(d,l)/area/norm
            K+=coeff*f*f; deriv+=2*coeff*f*df
    return K,deriv
checks=[]
def check(name,ok):
    checks.append({'name':name,'passed':bool(ok)})
    assert ok,name
# Exact d=2,N=3 normalized polynomial via moments and Gram inverse.
x,y,t=s.symbols('x y t')
mon=[x**a*y**b for a in range(4) for b in range(4-a)]
def moment(a,b):
    if a%2 or b%2:return s.S(0)
    return s.rf(s.Rational(1,2),a//2)*s.rf(s.Rational(1,2),b//2)/s.rf(s.Rational(5,2),(a+b)//2)
G=s.Matrix([[moment(s.degree(a*b,x),s.degree(a*b,y)) for b in mon] for a in mon])
v=s.Matrix(mon); exact=s.expand((v.T*G.inv()*v)[0].subs(y,0))
check('d2N3 exact centre',exact.subs(x,0)==s.Rational(10,3))
for d in DIMS:
    r=np.array([.0,.3,.6,.9])
    k,_=kernel(d,1,r)
    check(f'd{d} degree-one exact',np.max(abs(k*W(d)-(1+(d+3)*r*r)))<1e-10)
    for N in (0,1,2,3,8):
        nodes,weights=roots_jacobi(N+3,.5,(d-2)/2)
        rr=np.sqrt((nodes+1)/2)
        kk,_=kernel(d,N,rr)
        integ=np.dot(weights,kk)*2*math.pi**(d/2)/math.gamma(d/2)/2**((d+3)/2)
        check(f'd{d}N{N} trace',abs(integ/math.comb(N+d,d)-1)<1e-11)
        rr=np.array([.2,.5,.8]); step=1e-6
        _,der=kernel(d,N,rr)
        fd=(kernel(d,N,rr+step)[0]-kernel(d,N,rr-step)[0])/(2*step)
        check(f'd{d}N{N} derivative',np.max(abs(der-fd)/(1+abs(der)))<1e-7)
k,_=kernel(2,3,np.array([0,.6]))
check('separated equals exact Gram',max(abs(k*W(2)-np.array([float(exact.subs(x,z)) for z in (0,s.Rational(3,5))])))<1e-10)
# Closed diagonal from SO(3) fibre averaging of the auxiliary sphere kernel.
for d in DIMS:
    for N in (0,1,2,3,8,16):
        r=np.array([0,.1,.3,.6,.8,.9])
        k,kp=kernel(d,N,r); lam=(d+1)/2
        num=math.comb(N+d+1,d)+math.comb(N+d,d)-eval_gegenbauer(N+1,lam,2*r*r-1)-eval_gegenbauer(N,lam,2*r*r-1)
        closed=num/(2*(d+1)*W(d)*(1-r*r))
        check(f'd{d}N{N} closed diagonal',max(abs(k/closed-1))<1e-11)
        at1=kernel(d,N,np.array([1.]))[0][0]*W(d)
        expected=math.comb(N+d+2,d+2)+(math.comb(N+d+1,d+2) if N else 0)
        check(f'd{d}N{N} boundary diagonal',abs(at1/expected-1)<1e-11)
profiles=[]
for d in DIMS:
    c=math.gamma((d+1)/2)/math.pi**((d+1)/2)
    r=np.linspace(.1,.9,81)
    for N in (16,64,128):
        k,kp=kernel(d,N,r); D=math.comb(N+d,d)
        ratio=k*(1-r*r)/(D*c)
        derivative=((1-r*r)*kp-2*r*k)/(D*c)
        profiles.append(dict(d=d,N=N,max_relative_error=float(max(abs(ratio-1))),min_ratio=float(min(ratio)),max_ratio=float(max(ratio)),max_radial_derivative=float(max(abs(derivative)))))
labels=[('W',2,75),('Z',2,80),('Higgs',2,94),('down',3,0),('strange',3,3),('up',4,2),('charm',4,19),('top',4,71),('nu1',5,9),('nu2',5,14),('nu3',5,21),('electron',6,12),('muon',6,34),('tau',10,22)]
assigned=[]
for name,d,N in labels:
    r=np.array([0,.3,.6,.8,.9,1.]); k,_=kernel(d,N,r)
    c=math.gamma((d+1)/2)/math.pi**((d+1)/2)
    ratios=k*(1-r*r)/(math.comb(N+d,d)*c)
    assigned.append(dict(name=name,d=d,N=N,D=math.comb(N+d,d),r=r.tolist(),uw_over_asymptotic=ratios.tolist()))
report=dict(date=datetime.now(timezone.utc).isoformat(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),checks=checks,exact_d2N3_normalized_kernel=str(exact),exact_at_three_fifths=str(exact.subs(x,s.Rational(3,5))),convergence=profiles,assigned=assigned)
Path(__file__).with_name('idos-ball-density-checks.json').write_text(json.dumps(report,indent=2)+'\n')
print('Checks',len(checks)); print('Exact',exact,exact.subs(x,s.Rational(3,5))); print(json.dumps(profiles,indent=2))
