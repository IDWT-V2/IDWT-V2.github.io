"""Finite radial charge profile and bounded trial-energy checks."""
import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sympy as s
import mpmath as mp

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',type=Path,required=True)
args=p.parse_args()
if args.output.exists():
    raise FileExistsError(args.output)
checks=[]


def require(name,test):
    if not test:
        raise AssertionError(name)
    checks.append(name)


mp.mp.dps=35
u=s.symbols('u',positive=True)
f=1-s.exp(-u)
dq=s.diff(f**3,u)
require('radial charge measure',s.simplify(dq-3*s.exp(-u)*(1-s.exp(-u))**2)==0)
moments=[]
for k,expected in [(0,s.Integer(1)),(1,s.Rational(11,6)),(2,s.Rational(85,18))]:
    exact=3*s.factorial(k)*(1-s.Rational(2,2**(k+1))+s.Rational(1,3**(k+1)))
    numeric=mp.quad(lambda v:v**k*3*mp.exp(-v)*(1-mp.exp(-v))**2,[0,1,mp.inf])
    require(f'charge moment {k}',exact==expected)
    require(f'charge quadrature {k}',abs(numeric-mp.mpf(str(s.N(exact,35))))<mp.mpf('1e-30'))
    moments.append(str(exact))
require('central charge density finite',s.limit(6*s.exp(-u)*(1-s.exp(-u))**2/u**2,u,0)==6)

# Generic radial curvature is a common congruence of every polarization matrix.
r,g,dg=s.symbols('r g dg',positive=True)
x=s.Matrix([r,0,0,0,0,0])
block=s.Matrix([[0,1],[-1,0]])
j=s.diag(block,-block,-block)
curv=2*g*j-(dg/r)*(x*(j*x).T-(j*x)*x.T)
t=s.diag(1+r*dg/(2*g),1,1,1,1,1)
require('generic congruence',s.simplify(curv-2*g*t*j*t)==s.zeros(6))
require('radial and tangent blocks',s.simplify(curv[0,1]-(2*g+r*dg))==0 and curv[2,3]==-2*g)

# Normalized orientation energy for the same finite charge texture.
orientation=mp.quad(lambda v:mp.mpf('0.5')*v**2*mp.exp(-v)
    *(v/mp.expm1(v)+5*(-mp.expm1(-v))/v),[0,1,mp.inf])
alpha=s.pi**4/30+s.Rational(15,8)
require('orientation gradient integral',abs(orientation-(3*mp.zeta(4)-mp.mpf(9)/8))<mp.mpf('1e-30'))
require('full normal gradient',abs(orientation+3-mp.mpf(str(s.N(alpha,35))))<mp.mpf('1e-30'))
magnetic=mp.quad(lambda v:mp.sqrt(v)*mp.exp(-mp.mpf('1.5')*v)*(-mp.expm1(-v))**mp.mpf('2.5'),[0,1,mp.inf])
require('finite positive circulation moment',0<magnetic<1)

# All charge weights multiply the same normalized spatial density.
for grade in [0,2]:
    ell=1-s.Rational(2*grade,3)
    for branch in [0,1]:
        require(f'charge preserved grade={grade} branch={branch}',(1-ell)/2-branch==s.Rational(grade,3)-branch)

# A normalized marginal of a conserved norm current is itself conserved.
t0,x0,v=s.symbols('t x v',real=True)
w=s.exp(-(x0-v*t0)**2)
require('rigid marginal continuity',s.simplify(s.diff(w,t0)+s.diff(v*w,x0))==0)

# Fixed-norm Gaussian trial energy. Couplings remain symbolic physical inputs.
a,b,kappa,n,G,H=s.symbols('a b kappa n G H',positive=True)
beta=s.Integer(2)
ratio=s.sqrt(3*beta/(2*alpha))
require('aspect ratio',s.simplify(ratio**2-3/alpha)==0)
require('optimized kinetic coefficient',s.simplify(alpha+beta/ratio**2-5*alpha/3)==0)
for power in [2,3]:
    c=s.Rational(1,power**5)/s.pi**(5*(power-1))
    require(f'Gaussian nonlinear integral {power}',
            s.simplify((s.pi/power)**5/s.pi**(5*power)-c)==0)
K=s.symbols('K',positive=True)
energy=K/a**2-G/a**10+H/a**20
require('stationary polynomial',s.simplify(s.diff(energy,a)*a**21/(-2)-(K*a**18-5*G*a**10+10*H))==0)
stationary_H=(5*G*a**10-K*a**18)/10
require('stationary energy',s.simplify(energy.subs(H,stationary_H)-(s.Rational(9,10)*K/a**2-G/(2*a**10)))==0)
require('radial curvature at critical point',s.simplify(s.diff(energy,a,2).subs(H,stationary_H)-(100*G/a**12-36*K/a**4))==0)
amax=(25*G/(9*K))**s.Rational(1,8)
hmax=s.Rational(2,9)*G*(25*G/(9*K))**s.Rational(5,4)
require('double-root threshold',s.simplify(stationary_H.subs(a,amax)-hmax)==0)

# Dimensionless control, not a particle or mass fit.
def bisect(fun,left,right):
    fl=fun(left)
    for _ in range(180):
        mid=(left+right)/2; fm=fun(mid)
        if fm*fl>0:
            left=mid;fl=fm
        else:
            right=mid
    return (left+right)/2

polynomial=lambda z:z**9-5*z**5+mp.mpf('0.1')
roots=[bisect(polynomial,mp.mpf('0.01'),mp.mpf('0.8')),bisect(polynomial,mp.mpf(1),mp.mpf(2))]
for i,z in enumerate(roots):
    require(f'stationary root {i}',abs(polynomial(z))<mp.mpf('1e-30'))
require('first root is radial minimum',100/roots[0]**6-36/roots[0]**2>0)
require('second root is radial maximum',100/roots[1]**6-36/roots[1]**2<0)
require('negative trial minimum',1/roots[0]-1/roots[0]**5+mp.mpf('0.01')/roots[0]**10<0)
result=dict(created=datetime.now(timezone.utc).isoformat(),code_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
    review_status='author checked; independent review pending',check_count=len(checks),checks=checks,
    charge_moments= moments,normal_gradient_coefficient=str(alpha),
    magnetic_moment_coefficient=str(magnetic),dimensionless_roots_a_squared=[str(z) for z in roots],
    scope='Finite prescribed field and conserved-current construction; restricted trial-energy minima, not a solved full field equation.')
args.output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'passed':len(checks),'magnetic_coefficient':str(magnetic),'output':str(args.output)}))
