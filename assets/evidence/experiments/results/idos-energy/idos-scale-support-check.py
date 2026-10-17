#!/usr/bin/env python3
"""Exact finite checks of CEEM-S1 full canonical scale extension."""
import sympy as s
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
I=s.symbols('I0:3',positive=True); t=s.symbols('t0:3',real=True)
L,M,a,p,q,k,eta,ref=s.symbols('L M a p q k eta ref',positive=True); P=s.symbols('P',real=True)
v=(1,2,3); r=(1,-2,1); b=[v[j]*I[j] for j in range(3)]; B=sum(b)
Cb=sum((z-B/3)**2 for z in b)/(2*ref); F=1+eta*s.cos(sum(r[j]*t[j] for j in range(3)))
G=k*Cb*F; K=B+G; H=P**2/(2*M)+a*L**p+L**(-q)*K
checks=[]
def ck(n,x):
    assert s.simplify(x)==0,n
    checks.append(n)
def rate(f):
    return sum(s.diff(f,t[j])*s.diff(H,I[j])-s.diff(f,I[j])*s.diff(H,t[j]) for j in range(3))+s.diff(f,L)*s.diff(H,P)-s.diff(f,P)*s.diff(H,L)
for n,f in [('B',B),('A',sum(I)),('G',G),('H',H)]:ck(n+' conserved',rate(f))
ck('radial velocity',s.diff(H,P)-P/M)
ck('radial force',-s.diff(H,L)+p*a*L**(p-1)-q*K*L**(-q-1))
for j in range(3):
    ck('phase variation '+str(j),s.diff(H,I[j])-L**(-q)*v[j]*(1+k*(b[j]-B/3)*F/ref))
b0=s.symbols('b0',positive=True); uniform={I[j]:b0/v[j] for j in range(3)}
for j in range(3):
    ck('uniform b invariant '+str(j),rate(b[j]).subs(uniform))
    ck('breathing energy law '+str(j),rate(b[j]*L**(-q)).subs(uniform)+q*b0*L**(-q-1)*P/M)
D,e=s.symbols('D epsilon',positive=True)
U=a*L**p+D*b0*L**(-q)
# Eliminate b0 using the stationary condition, preserving arbitrary positive p,q.
equilibrium={b0:p*a*L**(p+q)/(q*D)}
ck('radial equilibrium',s.diff(U,L).subs(equilibrium))
ck('positive radial curvature',s.diff(U,L,2).subs(equilibrium)-p*(p+q)*a*L**(p-2))
ck('complete energy support factor',(U-(1+q/p)*D*b0*L**(-q)).subs(equilibrium))
# Check general scaling exponents algebraically; evaluate rational p,q examples exactly.
for pp,qq in [(1,1),(2,2),(2,1),(1,2),(s.Rational(1,2),s.Rational(3,2))]:
    pp,qq=s.Rational(pp),s.Rational(qq)
    for dd in (2,3,4,5,6,10):
        count=s.binomial(dd+2,dd)
        aa=s.Rational(3,2); ee=s.Rational(2,3)
        ll=(qq*count*ee/(pp*aa))**(1/pp)
        bb=ee*(qq*count*ee/(pp*aa))**(qq/pp)
        ck(f'common epsilon equilibrium {pp} {qq} {dd}',pp*aa*ll**(pp+qq)-qq*count*bb)
        ck(f'common epsilon full E {pp} {qq} {dd}',aa*ll**pp+count*bb*ll**(-qq)-(1+qq/pp)*count*ee)
        ck(f'common epsilon modal e {pp} {qq} {dd}',bb*ll**(-qq)-ee)
# Shared action: energy and scale variables multiply all modal frequencies equally.
ck('uniform carrier remains scale-rate',s.diff(H,I[0]).subs(uniform)-L**(-q)*v[0])
now=datetime.now(timezone.utc); path=Path(__file__)
out=path.parent/('scale-support-checks-'+now.strftime('%Y%m%dT%H%M%SZ')+'.json')
out.write_text(json.dumps({'created_utc':now.isoformat(),'status':'passed','check_count':len(checks),'checks':checks,'script_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'action':'CEEM-S1','scope':'Exact canonical identities and bounded rational exponent controls; no field PDE claim.'},indent=2)+'\n')
print(out, len(checks))
