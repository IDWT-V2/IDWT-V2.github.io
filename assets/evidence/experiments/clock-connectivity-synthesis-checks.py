"""Bounded exact checks of the proposed clock/connectivity synthesis."""
from pathlib import Path
import json
import sympy as s

root = Path(__file__).resolve().parent
checks = []

def check(name, condition):
    assert bool(condition), name
    checks.append(name)
    print('PASS ' + name, flush=True)

def equal(name, left, right=0):
    check(name, s.simplify(left - right) == 0)

I1, I2, Om, hbar, c, p, U, lam = s.symbols('I1 I2 Om hbar c p U lam', positive=True)
C = s.symbols('C', integer=True, positive=True)
B = Om * (I1 + I2 + hbar * (C - 1)) + U
H = s.sqrt(c*c*p*p + B*B)
equal('complete canonical clock derivative', s.diff(H, I1), Om*B/H)
equal('translation velocity', s.diff(H, p), c*c*p/H)
equal('rest momentum Hessian', s.diff(H, p, 2).subs(p, 0), c*c/B)
bad = lam*(I1-I2)**2/(I1+I2)**2
equal('degree-zero source homogeneity', I1*s.diff(bad,I1)+I2*s.diff(bad,I2))
check('degree-zero does not preserve individual clocks', s.simplify(s.diff(bad,I1)-s.diff(bad,I2)) != 0)

def laplacian(n, edges):
    L = s.zeros(n)
    for i,j,w in edges:
        L[i,i] += w
        L[j,j] += w
        L[i,j] -= w
        L[j,i] -= w
    return L

for n, edges, components in [
    (1,[],1), (3,[(0,1,1),(1,2,1)],1),
    (4,[(0,1,1),(2,3,2)],2),
    (4,[(0,1,1),(1,2,1),(2,3,1),(3,0,1)],1),
]:
    L = laplacian(n,edges)
    check(f'graph rank n={n}, components={components}',L.rank()==n-components)
    check(f'constant kernel n={n}',L*s.ones(n,1)==s.zeros(n,1))
    equal(f'calibrated rest count n={n}',Om*(components*hbar+hbar*L.rank()),n*hbar*Om)

t = s.symbols('t', real=True)
f = 6*t**5-15*t**4+10*t**3
equal('spectral smoothing lower endpoint',f.subs(t,0))
equal('spectral smoothing upper endpoint',f.subs(t,1),1)
for order in (1,2):
    for endpoint in (0,1):
        equal(f'smoothing endpoint derivative {order}, {endpoint}',s.diff(f,t,order).subs(t,endpoint))
equal('nonnegative spectral smoothing derivative',s.diff(f,t),30*t*t*(1-t)**2)
w, eps = s.symbols('w eps',positive=True)
L2=laplacian(2,[(0,1,w)])
check('two-site actual spectrum',L2.eigenvals()=={0:1,2*w:1})
equal('two-site transition force coefficient',s.diff(f.subs(t,2*w/eps),w),2/eps*s.diff(f,t).subs(t,2*w/eps))

k, F, x = s.symbols('k F x',positive=True)
energy = k*x*x/2-F*x
equal('shape equilibrium',s.diff(energy,x).subs(x,F/k))
equal('relaxed susceptibility energy',energy.subs(x,F/k),-F*F/(2*k))
P1,P2,C1,C2=s.symbols('P1 P2 C1 C2')
joined=Om*(P1+P2+hbar*(C1+C2-1))
split=Om*(P1+hbar*(C1-1))+Om*(P2+hbar*(C2-1))
equal('rest merger reference cost',joined-split,hbar*Om)
report={'status':'passed','count':len(checks),'checks':checks,
        'scope':'Exact finite algebra and small graph fixtures; no field PDE, continuum stability or physical selection test.'}
(root/'clock-connectivity-synthesis-checks.json').write_text(json.dumps(report,indent=2)+'\n')
