#!/usr/bin/env python3
"""Bounded analytic checks for the normal-kernel comparison; no DB writes."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib,json,math
import sympy as s
root=Path(__file__).resolve().parent
x=s.symbols('x', real=True)
a,k,g=s.symbols('a k g', positive=True)
q=s.sech(k*x); th=s.tanh(k*x)
checks=[]
def check(name, expression):
    residual=s.simplify(s.expand_trig(expression).rewrite(s.exp))
    passed=residual==0
    checks.append({'name':name,'passed':bool(passed),'residual':str(residual)})
    if not passed: raise AssertionError((name,residual))
# Exact Helmholtz-Hartree solution. Parameters obey ell=1/(2k).
ell=1/(2*k); A=6*a*k*k; B=s.sqrt(9*a*k*k/g)
phi=B*q*q; W=A*q*q; eps=4*a*k*k
check('Hartree stationary profile',-a*s.diff(phi,x,2)-W*phi+eps*phi)
check('Hartree Helmholtz same-field response',W-ell**2*s.diff(W,x,2)-g*phi**2)
check('Hartree phase zero mode',-a*s.diff(phi,x,2)-W*phi+eps*phi)
check('Hartree translation response source',s.diff(W,x)-ell**2*s.diff(W,x,3)-2*g*phi*s.diff(phi,x))
check('Hartree translation zero mode',-a*s.diff(phi,x,3)-W*s.diff(phi,x)-s.diff(W,x)*phi+eps*s.diff(phi,x))
chi=q*th
check('Hartree frozen first excited bound profile',-a*s.diff(chi,x,2)-W*chi+a*k*k*chi)
# Use exact beta integral C_r=int sech(t)^r dt; C4=4/3,C6=16/15.
C4=s.Rational(4,3); C6=s.Rational(16,15)
Q=B**2*C4/k
T=a*B**2*4*k*(C4-C6)
Wint=B**2*A*C6/k
E=T-Wint/2
check('Hartree canonical branch norm',Q-12*a*k/g)
check('Hartree complete energy',E+8*a*k*k*Q/5)
check('Hartree stationary contraction',T-Wint+eps*Q)
check('Hartree curvature-depth squared',2*a*s.diff(-W,x,2).subs(x,0)/A**2-s.Rational(2,3))
check('Hartree bound-gap depth',3*a*k*k/A-s.Rational(1,2))
# Generic power family, written with lambda=1/sigma, depth a lambda(lambda+1)k².
lam=s.symbols('lam',positive=True)
f=q**lam
V=-a*lam*(lam+1)*k*k*q*q
check('Generic power/PT stationary profile',-a*s.diff(f,x,2)+V*f+a*lam**2*k*k*f)
check('Generic curvature-depth squared',2*a*s.diff(V,x,2).subs(x,0)/(a*lam*(lam+1)*k*k)**2-4/(lam*(lam+1)))
golden=(1+s.sqrt(5))/2
check('Ratio-two exponent exact',golden**2-golden-1)
check('Ratio-two curvature exact',4*golden**2/(golden+1)-4)
# Unitary weighted-kinetic conjugation on ordinary warped geometry.
p=s.symbols('p',positive=True)
J=s.cosh(k*x)**p
chi=s.Function('chi')(x)
f=chi/s.sqrt(J)
Vgeo=a*k*k*(p*p/4-p*(p-2)*q*q/4)
check('Warped metric induced kinetic potential',s.sqrt(J)*(-a*(s.diff(f,x,2)+s.diff(s.log(J),x)*s.diff(f,x)))-(-a*s.diff(chi,x,2)+Vgeo*chi))
# Positive d>=4 branch; lambda=(d-3)/2.
depths=[]
for d in (2,3,4,5,6,10):
    lv=s.Rational(d-3,2)
    bound=lv>0
    row={'physical_depth':d,'p':d-1,'normal_bound_ground':bool(bound),'full_local_power_sigma_d_minus_2':float(d*golden-2),'full_Helmholtz_high_concentration':('kinetic dominates' if d<4 else 'critical coefficient' if d==4 else 'attraction dominates')}
    if bound:
        Ag=lv*(lv+1)
        ratio=s.simplify(2/s.sqrt(Ag))
        row.update({'geometric_depth_over_a_k2':str(Ag),'curvature_depth_ratio':str(ratio),'curvature_depth_decimal':float(ratio),'bound_normal_levels':[j for j in range(10) if j<lv], 'ground_energy_over_a_k2':str(s.Integer(d-2))})
        ground=q**lv
        check(f'd{d} geometric ground profile',-a*s.diff(ground,x,2)+Vgeo.subs(p,d-1)*ground-a*k*k*(d-2)*ground)
    depths.append(row)
# Inclusions preserve a norm-density normal kernel exactly; compare two internal bands.
u=s.Matrix([s.Rational(3,5),4*s.I/5]); inc=s.Matrix([[1,0],[0,1],[0,0]])
check('Internal band norm preserved',(inc*u).conjugate().dot(inc*u)-u.conjugate().dot(u))
check('Identity normal block compression',s.trace(inc.T*s.eye(3)*inc-s.eye(2)))
# Coordinate rescaling y=b x: m_x=b²m, curvature_x=b²curvature_y.
b,m,c=s.symbols('b m c',positive=True)
check('Curvature ratio coordinate invariance',(b*b*c)/(b*b*m)-c/m)
result={'created_utc':datetime.now(timezone.utc).isoformat(),'scope':'Exact normal profiles, frozen spectra, responding zero modes, kinetic measure, depth comparisons. No particle-mass or stability claim.','code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'sources':{'experiments/exact-normal-response-kernels.py':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},'checks':checks,'all_passed':all(r['passed'] for r in checks),'depths':depths,'hartree':{'Q':'12 a k/g','ell':'1/(2 k)','A':'6 a k^2','stationary_frequency':'-4 a k^2/hbar','energy':'-8 a k^2 Q/5','curvature_over_depth':'sqrt(2/3)','frozen_gap_over_depth':'1/2'},'ratio_two_local_power':{'sigma':str(golden),'sigma_float':float(golden),'frozen_bound_level_count':1,'interpretation':'Target-selected exponent, no geometric derivation; curvature 2 is not a discrete gap.'}}
out=root/('exact-normal-response-kernels-results-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'.json')
out.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'result':str(out),'checks':len(checks),'all_passed':result['all_passed'],'depths':depths},indent=2))
