#!/usr/bin/env python3
"""Exact exterior charge moments and parity controls; no particle simulation."""
from collections import Counter
import argparse
from datetime import datetime, timezone
from fractions import Fraction as F
from hashlib import sha256
import json
from math import comb, factorial
from pathlib import Path

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output-dir',type=Path,help='A new directory for paired code and results.')
args=parser.parse_args()
script_bytes=Path(__file__).read_bytes()
checks = 0


def require(condition):
    global checks
    assert condition
    checks += 1


def spectrum(m, parity=None):
    values = Counter()
    for grade in range(m+1):
        if parity is None or grade % 2 == parity:
            values[F(grade,m)] += comb(m,grade)
            values[F(grade,m)-1] += comb(m,grade)
    return values


def moment(values, power):
    return sum((mult * charge**power for charge,mult in values.items()), F(0))


def exact(x):
    return str(x.numerator) + '/' + str(x.denominator)


def pfaffian(matrix):
    if not matrix:
        return F(1)
    total = F(0)
    for j in range(1,len(matrix)):
        rest = [i for i in range(len(matrix)) if i not in (0,j)]
        minor = [[matrix[a][b] for b in rest] for a in rest]
        total += (-1)**(j+1)*matrix[0][j]*pfaffian(minor)
    return total


def multiply(a,b):
    return [[sum((a[i][k]*b[k][j] for k in range(len(b))),F(0))
             for j in range(len(b[0]))] for i in range(len(a))]


def transpose(a):
    return list(map(list,zip(*a)))


def rank(matrix):
    a = [row[:] for row in matrix]
    r = 0
    for c in range(len(a[0])):
        pivot = next((i for i in range(r,len(a)) if a[i][c]),None)
        if pivot is None:
            continue
        a[r],a[pivot]=a[pivot],a[r]
        scale=a[r][c]
        a[r]=[v/scale for v in a[r]]
        for i in range(len(a)):
            if i != r:
                scale=a[i][c]
                a[i]=[v-scale*w for v,w in zip(a[i],a[r])]
        r += 1
        if r == len(a):
            break
    return r


def frame_tensor(x,y):
    a,aa=x[0],x[1:]
    b,bb=y[0],y[1:]
    v=[aa[1]*bb[2]-aa[2]*bb[1],aa[2]*bb[0]-aa[0]*bb[2],aa[0]*bb[1]-aa[1]*bb[0]]
    xi=[a*z-b*w for w,z in zip(aa,bb)]
    return v,xi


def aggregate_rotation(xs,ys):
    omega=[[F(0) for _ in range(6)] for _ in range(6)]
    q=F(0)
    for x in xs:
        for y in ys:
            v,xi=frame_tensor(x,y)
            require(sum(a*b for a,b in zip(v,xi)) == 0)
            q += sum(a*a for a in v+xi)
            u=[a+b for a,b in zip(v,xi)]
            w=[a-b for a,b in zip(v,xi)]
            for i in range(3):
                for j in range(3):
                    omega[2*i][2*j+1] -= u[i]*w[j]
                    omega[2*i+1][2*j] += w[i]*u[j]
    n0=sum(a*a for x in xs for a in x)
    n1=sum(a*a for y in ys for a in y)
    mixed=[[sum(a*b for a,b in zip(x,y)) for y in ys] for x in xs]
    require(q == n0*n1-sum(a*a for row in mixed for a in row))
    return omega,q,F(n0*n1)


def gamma_action(state,axis):
    plane=axis//2
    out={}
    for mask,coefficient in state.items():
        sign=(-1)**((mask & ((1<<plane)-1)).bit_count())
        if axis%2:
            sign *= 1j*(-1)**((mask>>plane)&1)
        target=mask^(1<<plane)
        out[target]=out.get(target,0)+sign*coefficient
    return out


def spinor_rotation(m,state):
    def gaussian_norm(c):
        c=complex(c)
        require(c.real==int(c.real) and c.imag==int(c.imag))
        return int(c.real)**2+int(c.imag)**2
    norm=sum(gaussian_norm(c) for c in state.values())
    result=[[F(0) for _ in range(2*m)] for _ in range(2*m)]
    for i in range(2*m):
        for j in range(i+1,2*m):
            acted=gamma_action(gamma_action(state,j),i)
            scalar=1j*sum(complex(c).conjugate()*acted.get(mask,0) for mask,c in state.items())
            require(scalar.imag == 0 and scalar.real == int(scalar.real))
            result[i][j]=F(int(scalar.real),norm)
            result[j][i]=-result[i][j]
    number=sum(F(mask.bit_count()*gaussian_norm(c),norm) for mask,c in state.items())
    return result,number


rows = []
for m in range(1,17):
    even,odd,full = spectrum(m,0),spectrum(m,1),spectrum(m)
    require(sum(even.values()) == 2**m)
    require(sum(full.values()) == 2**(m+1))
    require(even+odd == full)
    require(all(full[q] == full[-q] for q in full))
    if m % 2 == 0:
        require(all(even[q] == even[-q] for q in even))
    for power in range(0,2*m+4):
        e,o,f = moment(even,power),moment(odd,power),moment(full,power)
        require(e+o == f)
        if power % 2:
            require(f == 0)
            if m%2 == 0 or power < m:
                require(e == 0)
            if m%2 and power == m:
                require(e == -F(factorial(m),m**m))
    # Independent finite-character identity in unnormalized integer weights.
    for x in [F(1,2),F(2,3),F(1),F(3,2),F(2)]:
        ge = sum(mult*x**int(q*m) for q,mult in even.items())
        gi = sum(mult*x**int(-q*m) for q,mult in even.items())
        closed = (1+x**(-m))*((1+x)**m+(1-x)**m)/2
        supertrace = sum((even[q]-odd[q])*x**int(q*m) for q in full)
        require(ge == closed)
        require(ge-gi == ((1+x**(-m))*(1-x)**m if m%2 else 0))
        require(supertrace == (1+x**(-m))*(1-x)**m)
    if m >= 3:
        require(moment(even,2)/sum(even.values()) == F(m+1,4*m))
    rows.append({'complex_normal_rank':m,'real_normal_directions':2*m,
                 'orthogonal_complex_structure_orbit_real_dimension':m*(m-1),
                 'ambient_depth_if_base_four':4+2*m,'even_carrier_rank':sum(even.values()),
                 'first_nonzero_odd_moment':m if m%2 else None,
                 'first_odd_moment_trace':exact(-F(factorial(m),m**m)) if m%2 else None,
                 'normalized_cubic_trace':exact(moment(even,3)),
                 'full_cubic_trace':exact(moment(full,3)),
                 'spectrum':[{'q':exact(q),'multiplicity':even[q]} for q in sorted(even)]})
require(moment(spectrum(3,0),3) == -F(2,9))
require(moment(spectrum(3,1),3) == F(2,9))
require(moment(spectrum(3),3) == 0)
require([m for m in range(2,17) if moment(spectrum(m,0),3)] == [3])
require([m for m in range(1,17) if m*(m-1)==2*m] == [3])

# Multivariate parity character with independent plane weights x_i.
# Its lowest polynomial degree is m, with coefficient 2*(-1)^m*x_1...x_m.
for m in range(1,9):
    plane_weights = [F(i+1,i+2) for i in range(m)]
    degree = m
    super_moment = F(0)
    for mask in range(1<<m):
        grade = mask.bit_count()
        w = sum((plane_weights[i] for i in range(m) if mask & (1<<i)),F(0))
        determinant_weight = sum(plane_weights,F(0))
        super_moment += (-1)**grade * (w**degree + (w-determinant_weight)**degree)
    product = F(1)
    for w in plane_weights:
        product *= w
    require(super_moment == 2*(-1)**m*factorial(m)*product)
    if m <= 4:
        size = 2*m
        a = [[F(0) for _ in range(size)] for _ in range(size)]
        for i,w in enumerate(plane_weights):
            a[2*i][2*i+1] = -w
            a[2*i+1][2*i] = w
        require(super_moment == 2*factorial(m)*pfaffian(a))
        rotation = [[F(int(i==j)) for j in range(size)] for i in range(size)]
        if m > 1:
            for i,j in [(0,2),(1,3)]:
                elementary = [[F(int(x==y)) for y in range(size)] for x in range(size)]
                elementary[i][i]=elementary[j][j]=F(3,5)
                elementary[i][j]=-F(4,5)
                elementary[j][i]=F(4,5)
                rotation = multiply(elementary,rotation)
        require(multiply(rotation,transpose(rotation)) ==
                [[F(int(i==j)) for j in range(size)] for i in range(size)])
        rotated = multiply(multiply(rotation,a),transpose(rotation))
        require(pfaffian(rotated) == pfaffian(a))
        reflection = [[F(int(i==j)) for j in range(size)] for i in range(size)]
        reflection[0][0]=-1
        reflected=multiply(multiply(reflection,a),transpose(reflection))
        require(pfaffian(reflected) == -pfaffian(a))

e=[[F(int(i==j)) for i in range(4)] for j in range(4)]
one=aggregate_rotation([e[0]],[e[1]])
three=aggregate_rotation([e[0],e[1]],[e[0],e[2]])
cancel=aggregate_rotation([e[0],e[1]],[e[2],e[3]])
partial=aggregate_rotation([e[0],e[1]],
         [[a+b for a,b in zip(e[0],e[2])],[a+b for a,b in zip(e[1],e[3])]])
require(rank(one[0]) == 2 and pfaffian(one[0]) == 0)
require(rank(three[0]) == 6 and pfaffian(three[0]) == -1)
require(multiply(three[0],three[0]) == [[-F(int(i==j)) for j in range(6)] for i in range(6)])
require(rank(cancel[0]) == 0 and cancel[1] == 4)
require(rank(partial[0]) == 2 and pfaffian(partial[0]) == 0)
require(three[1]/three[2] == partial[1]/partial[2] == F(3,4))
rotation_cases=[]
for name,case in [('single-plane',one),('three-plane',three),('circulation-cancels',cancel),('same-normalized-frame-power',partial)]:
    omega,q,n=case
    rotation_cases.append({'name':name,'rotation_rank':rank(omega),'pfaffian':exact(pfaffian(omega)),
                           'frame_power':exact(q),'intensity_product':exact(n),
                           'normalized_frame_power':exact(q/n),
                           'volume_efficiency':exact(27*abs(pfaffian(omega))/q**3),
                           'rotation_matrix':[[exact(a) for a in row] for row in omega]})
    require(0 <= 27*abs(pfaffian(omega)) <= q**3)
require(27*abs(pfaffian(three[0]))/three[1]**3 == 1)

spinor_cases=[]
for m,state,name in [(2,{0:1},'rank-two-vacuum'),(2,{3:1},'rank-two-top-singlet'),
                     (3,{0:1},'rank-three-vacuum'),(3,{3:1},'rank-three-grade-two-basis'),
                     (3,{3:1,5:2,6:3j},'rank-three-arbitrary-grade-two'),
                     (3,{0:1,3:1},'rank-three-mixed-charge'),
                     (4,{3:1},'rank-four-pure-grade-two'),
                     (4,{3:1,12:1},'rank-four-nonpure-same-charge')]:
    j,number=spinor_rotation(m,state)
    j0,_=spinor_rotation(m,{0:1})
    product=multiply(j,j0)
    alignment=-sum(product[i][i] for i in range(2*m))/F(2*m)
    require(alignment == 1-2*number/m)
    require((1-alignment)/2 == number/m)
    is_complex=multiply(j,j)==[[-F(int(i==k)) for k in range(2*m)] for i in range(2*m)]
    if m<=3 or len(state)==1:
        require(is_complex)
    else:
        require(not is_complex and rank(j)==0)
    spinor_cases.append({'name':name,'complex_normal_rank':m,'grade_expectation':exact(number),
                         'unshifted_charge_expectation':exact(number/m),'rotation_alignment':exact(alignment),
                         'rotation_is_orthogonal_complex_structure':is_complex,'rotation_rank':rank(j)})
# The same rank-four counterexample embeds at every higher rank tested.
for m in range(5,9):
    j,number=spinor_rotation(m,{3:1,12:1})
    require(number==2 and rank(j)==2*(m-4))
    require(multiply(j,j)!=[[-F(int(i==k)) for k in range(2*m)] for i in range(2*m)])

result = {'status':'passed','checks':checks,'comparison_ranks':[1,16],
          'scope':'Exact pointwise characters, not selected particles or physical anomaly coefficients',
          'rows':rows,'aggregate_rotation_cases':rotation_cases,
          'spinor_rotation_cases':spinor_cases,
          'sources':{'experiments/normal-charge-character-checks.py':sha256(script_bytes).hexdigest()}}
output_dir=args.output_dir or Path(__file__).parent/('run-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ'))
output_dir.mkdir(parents=True,exist_ok=False)
(output_dir/'executed-source.py').write_bytes(script_bytes)
output = output_dir/'normal-charge-character-checks.json'
output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'status':result['status'],'checks':checks,'output':str(output),
                  'rank_three_cubic_trace':'-2/9','full_completion_cubic_trace':'0'},indent=2))
