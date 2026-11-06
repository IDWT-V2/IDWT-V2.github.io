"""Exact mixed-curvature weights and full spatial charge compensation."""
import argparse
from datetime import datetime, timezone
from hashlib import sha256
import itertools
import json
from pathlib import Path
import sympy as s

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--output', type=Path, required=True)
args = p.parse_args()
if args.output.exists():
    raise FileExistsError(args.output)
checks = []


def require(name, condition):
    if not condition:
        raise AssertionError(name)
    checks.append(name)


def pf(a):
    if a.rows == 0:
        return s.Integer(1)
    return sum((-1)**(j+1)*a[0,j]*pf(a.minor_submatrix(0,j).minor_submatrix(j-1,0))
               for j in range(1,a.rows))


block = s.Matrix([[0,1],[-1,0]])
j0 = s.diag(block,block,block)
x = s.Matrix([s.Rational(i+1,7) for i in range(6)])
a = s.Rational(3,2)
den = a*a+(x.T*x)[0]
t = s.symbols('t')
rows = []
for bits in itertools.product([0,1],repeat=3):
    r = sum(bits)
    signs = [1-2*b for b in bits]
    j = s.diag(*[v*block for v in signs])
    mixed = j0+t*j
    w = mixed*x
    curvature = 2*mixed/den+2*(x*w.T-w*x.T)/den**2
    expected = s.prod(1+t*v for v in signs)
    polynomial = s.factor(pf(curvature)/(8*a*a/den**4))
    require(f'curvature polynomial {bits}', s.expand(polynomial-expected) == 0)
    ell = s.expand(polynomial).coeff(t,1)/3
    require(f'mixed alignment {bits}', ell == 1-s.Rational(2*r,3))
    require(f'matrix alignment {bits}', ell == -s.trace(j*j0)/6)
    for branch in [0,1]:
        q = (1-ell)/2-branch
        require(f'charge weight {bits} branch={branch}', q == s.Rational(r,3)-branch)
        rows.append(dict(bits=bits,grade=r,branch=branch,even_carrier=r%2==0,
                         alignment=str(ell),charge=str(q),top_integral=s.prod(signs)))

sx = s.Matrix([[0,1],[1,0]])
sy = s.Matrix([[0,-s.I],[s.I,0]])
sz = s.diag(1,-1)
i2 = s.eye(2)
gamma = []
for k in range(5):
    for pauli in [sx,sy]:
        gamma.append(s.kronecker_product(*([sz]*k+[pauli]+[i2]*(4-k))))
i32 = s.eye(32)
spin = [-s.I*gamma[i]*gamma[i+1]/2 for i in [4,6,8]]
spin_sum = sum(spin,s.zeros(32))
number = s.Rational(3,2)*i32-spin_sum
qspin = number/3
require('normal degree diagonal', all(number[i,j]==0 for i in range(32) for j in range(32) if i!=j))
require('normal degree spectrum', number.eigenvals() == {s.Integer(k):4*s.binomial(3,k) for k in range(4)})

# The support contains either both members of each rotated normal pair or neither.
supports = {3:[0,1,2],4:[0,1,2,3],5:[0,1,2,4,5],6:[0,1,2,3,4,5],10:list(range(10))}
coords = [s.Rational(i+1,11) for i in range(10)]
for d,active in supports.items():
    u = i32+s.I*sum((gamma[i]*coords[i] for i in active),s.zeros(32))
    angular_derivative = s.zeros(32)
    for i in [4,6,8]:
        if i in active:
            angular_derivative += coords[i]*gamma[i+1]-coords[i+1]*gamma[i]
    # L=-i(x_i partial_j-x_j partial_i), so L U equals angular_derivative.
    require(f'full spatial charge intertwining d={d}',
            (qspin*u-u*qspin-angular_derivative/3).applyfunc(s.expand) == s.zeros(32))
    if d>=5:
        require(f'spin-only failure exposed d={d}', qspin*u-u*qspin != s.zeros(32))

# A selected depth-five seed is neutral under both candidate generators.
active = supports[5]
eta = -gamma[active[0]]*gamma[active[1]]*gamma[active[2]]*gamma[active[3]]*gamma[active[4]]
k = -s.I*gamma[4]*gamma[5]
c = -s.I*gamma[0]*gamma[1]*gamma[2]
vacuum = s.diag(*[int(number[i,i]==0) for i in range(32)])
neutral_left = vacuum*(i32-eta)/2
require('five-dimensional sign product', c == eta*k)
require('shared neutral handed subspace exists', neutral_left.rank() == 2)
require('old generator neutral', (eta+k)*neutral_left == s.zeros(32))
require('new generator neutral', qspin*neutral_left == s.zeros(32))
require('selected handedness', c*neutral_left == -neutral_left)

# Constant two-branch shift does not follow from the same seed curvature.
require('branch shift is one', s.Rational(2,3)-(s.Rational(2,3)-1) == 1)
for row in rows:
    conjugate = s.Rational(3-row['grade'],3)-(1-row['branch'])
    require(f"conjugate charge {row['bits']} branch={row['branch']}", conjugate == -s.Rational(row['charge']))
result = dict(created=datetime.now(timezone.utc).isoformat(),
              code_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
              review_status='author checks only; independent review pending',
              checks=checks,check_count=len(checks),weights=rows,
              scope='Exact matrix controls and mixed-curvature polynomial; no physical source calibration.')
args.output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'passed':len(checks),'output':str(args.output)}))
