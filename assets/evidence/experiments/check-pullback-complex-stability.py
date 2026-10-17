"""Small exact checks for the complex-profile quadratic action; no PDE numerics."""
import sympy as s
import json, hashlib
from pathlib import Path
from datetime import datetime, timezone
p,q,a,h,at,ht,alpha,omega,nu,omk=s.symbols('p q a h at ht alpha omega nu omk', real=True)
p1,q1,a1,h1,z,vz=s.symbols('p1 q1 a1 h1 z vz', real=True)
u0=p+s.I*q
eta=a+s.I*h
d0=-s.I*omega*alpha*u0
d1=at+s.I*ht-s.I*omega*eta+alpha*z*(p1+s.I*q1)
d2=z*(a1+s.I*h1)-alpha*vz*(p1+s.I*q1)
actual=s.expand(s.re(s.conjugate(d1)*d1)/2+s.re(s.conjugate(d0)*d2))
expected=((at+alpha*z*p1)**2+(ht+alpha*z*q1)**2)/2+omega*(h*at-a*ht)+omega**2*(a*a+h*h)/2+omega*alpha*z*(h*p1-p*h1+q*a1-a*q1)+omega*alpha**2*(p*q1-q*p1)*vz
checks={'quadratic_kinetic_expansion':s.expand(actual-expected)==0}
# Carrier row for the helicity: mass and gyro terms before inserting the mode.
g,g1=s.symbols('g g1')
carrier=-nu**2*(p1*a+q1*h)+s.I*nu*omega*(h*p1-p*h1+q*a1-a*q1)
low=s.expand(carrier.subs({a:g,h:-s.I*g,a1:g1,h1:-s.I*g1})-nu*((omega-nu)*g*(p1-s.I*q1)-omega*(p-s.I*q)*g1))
up=s.expand(carrier.subs({a:g,h:s.I*g,a1:g1,h1:s.I*g1})+nu*((omega+nu)*g*(p1+s.I*q1)-omega*(p+s.I*q)*g1))
checks.update(lower_helicity_source=low==0,upper_helicity_source=up==0)
# Two-dimensional displacement bracket and weighted exact ball moments.
x,y,R=s.symbols('x y R', real=True)
f=1-x*x-y*y
z1=s.Matrix([f,0]); z2=s.Matrix([0,f]); j=s.Matrix([-y,x])
bracket=z2.jacobian([x,y])*z1-z1.jacobian([x,y])*z2
checks['gyro_bracket_sign']=s.expand(-j.dot(bracket)-2*f*(x*x+y*y))==0
rows=[]
for d in [2,3,4,5,6,10]:
    # Weighted ball is the Dirichlet moment law with d half parameters
    # and radial remainder parameter 3/2.
    ex2=s.Rational(1,d+3)
    ex2R=s.Rational(d+2,(d+3)*(d+5))
    eR=s.Rational(d,d+3)
    eR2=s.Rational(d*(d+2),(d+3)*(d+5))
    gyro=s.factor(4*(ex2-ex2R))
    mass=s.factor(1-2*eR+eR2)
    norm=2*ex2
    checks[f'd{d}_gyro']=gyro==s.Rational(12,(d+3)*(d+5))
    checks[f'd{d}_mass']=mass==s.Rational(15,(d+3)*(d+5))
    checks[f'd{d}_no_first_shell_collision']=s.binomial(d+2,d)>2*(d+1)-1
    rows.append(dict(d=d,gyro=str(gyro),mass=str(mass),norm=str(norm),count=d+1))
N,m,l=75,30,102
DN,Dm,Dl=[s.binomial(n+2,2) for n in [N,m,l]]
checks['exception_counts']=Dm+Dl==2*DN
checks['exception_lower_euler']=N*Dm-m*DN==-50580
checks['exception_upper_euler']=N*Dl-l*DN==103248
checks['exception_degrees']=(N+m-1,N+l-1)==(104,176)
# General second-order carrier jet: EL of omega J_i v_j d_j(vdot_i).
# Its variational row is -omega[d_j(J_i z_j)+J_j d_i z_j].
# Test pairing is the negative bracket form checked above.
checks={key:bool(value) for key,value in checks.items()}
base=Path(__file__).parent
out={'created_utc':datetime.now(timezone.utc).isoformat(),'all_pass':all(checks.values()),'checks':checks,'rows':rows,'canonical_source':{'path':'content/notes/topics/operator-tools/material-pullback-count-action.md','snapshot':1125,'sha256':'010780d2b4131bea0302b8067533cffb398080b4d0faadf3f03bd38fda281d23'},'sources':{'experiments/check-pullback-complex-stability.py':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}}
name=base/('pullback-complex-stability-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'.json')
name.write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({'all_pass':out['all_pass'],'checks':len(checks),'result':str(name)}))
assert out['all_pass']
