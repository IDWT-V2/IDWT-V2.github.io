"""Bounded exact checks for full Gaussian covariance/chirp dynamics."""
import sympy as s
import json, hashlib
from pathlib import Path
from datetime import datetime, timezone
checks=[]
def check(name, expr):
    ok=all(s.simplify(x)==0 for x in expr) if isinstance(expr,s.MatrixBase) else s.simplify(expr)==0
    if not ok: raise AssertionError((name,expr))
    checks.append(name)
a,b,k=s.symbols('alpha beta k',positive=True)
g,h,c=s.symbols('g h c',positive=True)
G=s.diag(g,h); C=s.Matrix([[0,c],[c,0]])
F=lambda q:2*a/q**2-2*b*k/(q*(q+k))
Gdot=2*a*(C*G+G*C)
Cdot=2*a*(G.inv()**2-C**2)-2*b*(G.inv()-(G+k*s.eye(2)).inv())
check('angular tensor conservation',Gdot*C+G*Cdot-Cdot*G-C*Gdot)
# Energy derivative for a general rational, noncommuting symmetric point in each depth.
counts=[]
for d in (2,3,4,5,6,10):
    GG=s.diag(*[s.Rational(i+2,i+1) for i in range(d)])
    CC=s.Matrix(d,d,lambda i,j:s.Rational((i+1)*(j+1),37) if i!=j else s.Rational(i+1,29))
    GD=2*(CC*GG+GG*CC)
    CD=2*(GG.inv()**2-CC**2)-2*(GG.inv()-(GG+2*s.eye(d)).inv())
    check(f'd{d} angular conservation',GD*CC+GG*CD-CD*GG-CC*GD)
    # W derivative = -G^-2+2[G^-1(G+2I)^-1].
    WP=-GG.inv()**2+2*GG.inv()*(GG+2*s.eye(d)).inv()
    check(f'd{d} energy conservation',s.trace((CC*CD+CD*CC)*GG+CC**2*GD+WP*GD))
    # Homogeneity phase identity, removing identical mu/log constants.
    lhs=s.trace(GG.inv())-s.trace((GG+2*s.eye(d)).inv())
    rhs=s.trace(GG.inv()+CC**2*GG)/2+s.trace(GG*CD)/4
    check(f'd{d} phase-energy identity',lhs-rhs)
    # Cross block zero and stationary transverse block checked with symbolic 2x block.
    if d<10:
        GT=s.diag(GG,2*s.eye(10-d)); CT=s.diag(CC,s.zeros(10-d))
        GDT=2*(CT*GT+GT*CT); CDT=2*(GT.inv()**2-CT**2)-2*(GT.inv()-(GT+2*s.eye(10)).inv())
        check(f'd{d} stationary transverse block',GDT[d:,d:]+CDT[d:,d:])
        check(f'd{d} no cross-plane generation',GDT[:d,d:]+CDT[:d,d:])
    counts.append({'depth':d,'quadratic_spatial_shell':d*(d+1)//2,'independent_rotation_planes':d//2,'cumulative_IDOS_through_degree_2':(d+1)*(d+2)//2})
# Algebraic rotating solution checked modulo its defining quadratic, avoiding numerical root tests.
y,w=s.symbols('y w',positive=True)
poly=27*y*y-13*y-54
w2=9*(y-1)/(5*y*y*(y+1))
GG=s.diag(2*y,y); CC=s.Matrix([[0,w/6],[w/6,0]]); Om=s.Matrix([[0,-w],[w,0]])
GD=2*(CC*GG+GG*CC); CD=2*(GG.inv()**2-CC**2)-2*(GG.inv()-(GG+2*s.eye(2)).inv())
check('relative equilibrium G equation',GD-(Om*GG-GG*Om))
for i in range(2):
    expr=s.factor((CD-(Om*CC-CC*Om))[i,i].subs(w*w,w2))
    num=s.together(expr).as_numer_denom()[0]
    check(f'relative equilibrium C equation {i}',s.rem(num,poly,y))
check('rotation phase deficit',s.trace(GG*(Om*CC-CC*Om))/4+w*(GG*CC-CC*GG)[0,1]/2)
# Small-amplitude frequency.
g0=a*k/(b*k-a)
fprime=s.diff(F(g),g).subs(g,g0)
nu2=16*a*a*(k/2)/(g0*g0*(g0+k))
check('quadratic disturbance frequency',-4*a*g0*fprime-nu2)
yv=(s.Integer(13)+s.sqrt(6001))/54
wv=s.sqrt(w2.subs(y,yv)); cv=wv/6
e0=s.Rational(1,2)-s.log(2)
excess=(1/(2*yv)+1/yv+3*yv*cv*cv-s.log(1+1/yv)-s.log(1+2/yv)-2*e0)/2
numeric={key:str(s.N(value,25)) for key,value in {'h':yv,'rotation_rate':wv,'chirp':cv,'excess_energy_per_A':excess,'escape_threshold_per_A':-e0/2,'angular_momentum_per_A':yv*cv/2,'carrier_deficit':wv*yv*cv/2}.items()}
assert 0<float(excess)<float(-e0/2)
checks.append('rotating sample lies strictly below escape threshold (numerical diagnostic)')
report={'date_utc':datetime.now(timezone.utc).isoformat(),'checks':checks,'check_count':len(checks),'depths':counts,'rotating_control':numeric,'sources':{'check.py':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},'scope':'Exact rational/algebraic matrix identities; one non-interval numerical diagnostic. No PDE evolution, particle simulation or full-field stability claim.'}
output=Path(__file__).with_name('checks-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'.json')
output.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'check_count':len(checks),'output':str(output),'rotating_control':numeric},indent=2))
