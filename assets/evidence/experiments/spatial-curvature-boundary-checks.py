"""Author checks for spatial curvature, boundary invariants and projected covariance."""
import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sympy as s

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
if args.output.exists():
    raise FileExistsError(args.output)
checks = []


def require(name, condition):
    if not condition:
        raise AssertionError(name)
    checks.append(name)


def pfaffian(a):
    if a.rows == 0:
        return s.Integer(1)
    return sum((-1)**(j+1)*a[0, j]*pfaffian(a.minor_submatrix(0, j).minor_submatrix(j-1, 0))
               for j in range(1, a.rows))


def wedge(a, b):
    out = {}
    for ia, va in a.items():
        for ib, vb in b.items():
            if set(ia) & set(ib):
                continue
            indices = ia + ib
            sign = (-1)**sum(indices[i] > indices[j]
                             for i in range(len(indices)) for j in range(i+1, len(indices)))
            key = tuple(sorted(indices))
            out[key] = out.get(key, 0) + sign*va*vb
    return {k: s.simplify(v) for k, v in out.items() if v != 0}


rows = []
for d in [2, 4, 6, 10]:
    m = d//2
    j = s.diag(*[s.Matrix([[0, 1], [-1, 0]]) for _ in range(m)])
    x = s.Matrix([s.Rational(i+1, d+1) for i in range(d)])
    a = s.Rational(3, 2)
    den = a*a + (x.T*x)[0]
    w = j*x
    f = 2*j/den + 2*(x*w.T-w*x.T)/den**2
    pf = s.factor(pfaffian(f))
    require(f'curvature Pfaffian d={d}', pf == 2**m*a*a/den**(m+1))
    require(f'conjugation parity d={d}', pfaffian(-f) == (-1)**m*pf)
    sphere = 2*s.pi**m/s.gamma(m)
    integral = s.simplify(s.factorial(m)*2**m*a*a*sphere
        * s.gamma(m)*s.gamma(1)/s.gamma(m+1)/(2*a*a)/(2*s.pi)**m)
    require(f'whole-space integral d={d}', integral == 1)
    r = s.symbols('r', positive=True)
    ball = (r*r/(a*a+r*r))**m
    radial = s.factorial(m)*2**m*a*a*sphere*r**(2*m-1)/(a*a+r*r)**(m+1)/(2*s.pi)**m
    require(f'finite-ball derivative d={d}', s.simplify(s.diff(ball, r)-radial) == 0)
    rows.append({'depth':d, 'normalized_integral':str(integral),
                 'conjugation_sign':(-1)**m, 'finite_norm_profile': d > 2})

# Differentiate the proposed connection explicitly in four coordinates.
x = s.Matrix(s.symbols('x0:4', real=True))
a = s.symbols('a', positive=True)
j = s.diag(s.Matrix([[0, 1], [-1, 0]]), s.Matrix([[0, 1], [-1, 0]]))
den = a*a+(x.T*x)[0]
connection = -j*x/den
f = s.Matrix(4, 4, lambda i,k:s.diff(connection[k], x[i])-s.diff(connection[i], x[k]))
expected = 2*j/den+2*(x*(j*x).T-(j*x)*x.T)/den**2
require('connection differentiation', (f-expected).applyfunc(s.simplify) == s.zeros(4))
require('Bianchi identity representative', s.simplify(s.diff(f[1,2],x[0])+s.diff(f[2,0],x[1])+s.diff(f[0,1],x[2])) == 0)

# Exact normalized-spinor check, including a concrete Clifford representation.
sx = s.Matrix([[0,1],[1,0]])
sy = s.Matrix([[0,-s.I],[s.I,0]])
sz = s.diag(1,-1)
gammas = [s.kronecker_product(sx,s.eye(2)),s.kronecker_product(sy,s.eye(2)),
          s.kronecker_product(sz,sx),s.kronecker_product(sz,sy)]
chi = s.Matrix([1,0,0,0])
z = (a*s.eye(4)+s.I*sum((x[i]*gammas[i] for i in range(4)),s.zeros(4)))*chi/s.sqrt(den)
require('normalized spinor', s.simplify((s.conjugate(z).T*z)[0]) == 1)
for i in range(4):
    ai = s.simplify(-s.I*(s.conjugate(z).T*z.diff(x[i]))[0])
    require(f'spinor connection component {i}', s.simplify(ai-connection[i]) == 0)

# Pullback permits products of oriented planes only when enough directions exist.
for d in [2,3,4,5,6,10]:
    two = {(i,i+1):s.Integer(1) for i in range(0,d-1,2)}
    product = {():s.Integer(1)}
    for k in range(1,7):
        product = wedge(product,two)
        require(f'exterior rank d={d} k={k}', bool(product) == (2*k <= d))

# A deterministic shear changes projected covariance without adding trajectories.
t = s.symbols('t', real=True)
cov = s.eye(4)
transport = s.Matrix([[1,0,0,0],[0,1,0,-t],[t,0,1,0],[0,0,0,1]])
omega = s.diag(s.Matrix([[0,1],[-1,0]]),s.Matrix([[0,1],[-1,0]]))
require('shear symplectic', transport.T*omega*transport == omega)
out = transport*cov*transport.T
require('full covariance volume preserved', out.det() == 1)
require('observed covariance increases', s.simplify(out[:2,:2].det()-(1+t*t)) == 0)

result = {'created':datetime.now(timezone.utc).isoformat(),
          'status':'author-checked; independent review pending',
          'code_sha256':sha256(Path(__file__).read_bytes()).hexdigest(),
          'checks':checks,'check_count':len(checks),'curvature_integrals':rows,
          'scope':'Exact sampled Pfaffians plus analytic formulas and representative symbolic derivatives; no physical charge identification.'}
args.output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'passed':len(checks),'output':str(args.output)}))
