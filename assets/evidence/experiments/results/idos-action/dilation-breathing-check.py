#!/usr/bin/env python3
"""Bounded exact checks; no large tensors or PDE discretization."""
import sympy as s, json, math, hashlib
from datetime import datetime, timezone
from pathlib import Path
out=Path(__file__).resolve().parent
checks=[]
def ck(name, value):
    assert value, name
    checks.append(name)
L,al,be,ss,J,dim=s.symbols('L alpha beta s2 J d',positive=True)
V=J*dim/4*(al/L**2-be*s.log(1+2*ss/L**2))
M=J*dim/(8*al)
force=4*al**2/L**3-8*al*be*ss/(L*(L**2+2*ss))
ck('Gaussian radial full energy variation',s.simplify(-s.diff(V,L)/M-force)==0)
beta_eq=al*(L**2+2*ss)/(2*ss*L**2)
ck('Gaussian equilibrium',s.simplify(force.subs(be,beta_eq))==0)
wb=16*al**2*ss/(L**4*(L**2+2*ss))
ck('Gaussian breathing curvature',s.simplify(-s.diff(force,L).subs(be,beta_eq)-wb)==0)
r=L**2/(L**2+2*ss)
ck('degree two spectrum agreement',s.simplify(8*al**2/L**4*(1-r)-wb)==0)
# Polynomial Hermite derivative, including normalization, tested symbolically.
x=s.symbols('x',real=True)
for n in range(9):
    f=s.hermite(n,x)*s.exp(-x*x/2)/s.sqrt(2**n*s.factorial(n))
    up=s.hermite(n+2,x)*s.exp(-x*x/2)/s.sqrt(2**(n+2)*s.factorial(n+2))
    low=0 if n<2 else s.hermite(n-2,x)*s.exp(-x*x/2)/s.sqrt(2**(n-2)*s.factorial(n-2))
    rhs=(s.sqrt((n+1)*(n+2))*up-s.sqrt(n*(n-1))*low)/2
    ck(f'Hermite dilation n={n}',s.simplify(-f/2-x*s.diff(f,x)-rhs)==0)
# Compositions only small cutoffs. Large cutoffs use generating-function identities.
def comps(k,d):
    if d==1: yield (k,); return
    for j in range(k+1):
        for tail in comps(k-j,d-1): yield (j,)+tail
rows=[]
for d in (2,3,4,5,6,10):
    for N in range(5):
        D=math.comb(N+d,d)
        active=sum(sum((n+1)*(n+2) for n in ns) for k in (N,N-1) if k>=0 for ns in comps(k,d))/s.Integer(4)
        exact=s.Rational(d,2)*(math.comb(N+d+1,d+1)+(math.comb(N+d,d+1) if N else 0))
        ck(f'active leakage d={d} N={N}',active==exact)
        ambient=active+s.Rational(10-d,2)*D
        if N==0:
            ck(f'ground active leakage d={d}',active==s.Rational(d,2))
            ck(f'ground ambient leakage d={d}',ambient==5)
        rows.append({'d':d,'N':N,'D':D,'L2_active_leakage':str(active),'L2_full10_leakage':str(ambient)})
# Exact connection-induced counterexample, d arbitrary: modes 0 and 2e1,
# Q0=1,P0=0,Q2=0,P2=1; equal I and equal v is sufficient.
g=s.sqrt(2)/2
Q=s.Matrix([1,0]); P=s.Matrix([0,1]); G=s.Matrix([[0,-g],[g,0]])
Qdot=-G*Q; Pdot=-G*P
Idot=[s.simplify(Q[j]*Qdot[j]+P[j]*Pdot[j]) for j in range(2)]
# Above quadrature choice has zero instantaneous transfer. Use in-phase Q values.
Q=s.Matrix([1,1]); P=s.Matrix([0,0]); Qdot=-G*Q; Pdot=-G*P
Idot=[s.simplify(Q[j]*Qdot[j]+P[j]*Pdot[j]) for j in range(2)]
ck('connection equal-action tangent fails',Idot==[g,-g])
# Nondegenerate equal energy: Q0=sqrt(2/v0), Q2=sqrt(2/v2), P=0.
v0,v2=s.symbols('v0 v2',positive=True)
Q=s.Matrix([s.sqrt(2/v0),s.sqrt(2/v2)]); Qdot=-G*Q
bdot=[s.simplify(v*Q[i]*Qdot[i]) for i,v in enumerate((v0,v2))]
ck('nondegenerate equal energy immediately separates',s.simplify(bdot[0]-bdot[1]-s.sqrt(2)*(v0+v2)/s.sqrt(v0*v2))==0)
# Chirped Gaussian continuity and radial phase coefficients.
Ld,Ldd=s.symbols('Ld Ldd',real=True)
c=Ld/(2*al*L)
cd=Ldd/(2*al*L)-Ld**2/(2*al*L**2)
phase_rhs=2*al/L**4-2*al*c**2-2*be*(1-r)/L**2
ck('quadratic phase radial equation',s.simplify((cd-phase_rhs).subs(Ldd,force))==0)
source_paths={
'content/notes/topics/contact-operators/idos-energy-and-coherence.md':'0a36f94efff55521408e75faa7f8fd29cbe63475b489f1c4ecc73f1bd5482e3f',
'content/notes/topics/contact-operators/self-supported-real-harmonic-core.md':'628dfb7c82e9c90f58981484644ac869190e5791136bbbf25399975a08a66145',
'support_model':'CEEM-S1 as stated in idos-energy-and-coherence.md'}
report={'utc':datetime.now(timezone.utc).isoformat(),'checks':len(checks),'passed':checks,'sources':source_paths,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'leakage':rows}
p=out/('checks-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'.json')
p.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'passed':len(checks),'report':str(p)}))
