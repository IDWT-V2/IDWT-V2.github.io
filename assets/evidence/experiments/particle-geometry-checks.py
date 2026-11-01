"""Reproduce finite calculations in the particle geometry case; no particle solve."""
from pathlib import Path
from fractions import Fraction as F
from itertools import combinations, product
from math import comb, log, exp
import hashlib
import json
import numpy as np
import sympy as s

ROOT = Path(__file__).resolve().parent
report = {}


def count(n, d):
    return comb(n + d - 1, d) if n else 0


def close(a, b):
    assert np.max(np.abs(a - b)) < 1e-11


def kron_all(items):
    out = np.ones((1, 1), complex)
    for item in items:
        out = np.kron(out, item)
    return out


# Mass prescription and diagnostics.
me = s.Rational('0.51099895000')
units = {2: me*s.sqrt(2890), 3: me*s.sqrt(32*s.sqrt(7)),
         4: me*s.sqrt(48/s.sqrt(7))/15, 6: me/18564, 10: me/18564}
families = {
    2: [('W', 76, '80369.2'), ('Z', 81, '91188'), ('H', 95, '125200')],
    3: [('down', 1, '4.70'), ('strange', 4, '93.5')],
    4: [('up', 3, '2.16'), ('charm', 20, '1273'), ('top', 72, '172570')],
    6: [('electron', 13, '0.51099895000'), ('muon', 35, '105.6583755')],
    10: [('tau', 23, '1776.93')]}
mass_rows = []
for d, rows in families.items():
    for name, n, observed in rows:
        predicted = units[d]*count(n, d)
        mass_rows.append(dict(name=name, d=d, n=n, C=count(n,d),
                              exact=str(predicted), MeV=float(predicted),
                              archived_MeV=observed))
report['units_MeV'] = {str(d): {'exact': str(u), 'decimal': float(u)}
                       for d, u in units.items()}
report['masses'] = mass_rows
for d in range(1, 16):
    for n in range(1, 33):
        assert count(n,d) == sum(comb(j+d-1,d-1) for j in range(n))
        if d > 1:
            assert count(n,d) == count(n,d-1)+count(n-1,d)
        assert F(count(n+1,d),count(n,d)) == F(n+d,n)
diagnostics = {}
for original in (2,3,4,6):
    scores = {}
    rows = families[original]
    for d in range(1,16):
        offsets = [log(float(M)/count(n,d)) for _,n,M in rows]
        anchor = sum(offsets)/len(offsets)
        scores[d] = sum((v-anchor)**2 for v in offsets)/len(offsets)
    assert min(scores, key=scores.get) == original
    diagnostics[str(original)] = scores
report['fixed_index_log_variance'] = diagnostics
nu = [count(n,5) for n in (10,15,22)]
ratio = F(nu[1]**2-nu[0]**2,nu[2]**2-nu[0]**2)
report['neutrinos'] = dict(counts=nu, exact_ratio=str(ratio), ratio=float(ratio),
    benchmark_A=float(F('0.0000753')/(F('0.002455')+F('0.0000753'))),
    benchmark_B=float(F('0.0000749')/F('0.002534')),
    solar_B_masses_eV=[float(s.sqrt(s.Rational('0.0000749')/
                          (nu[1]**2-nu[0]**2))*v) for v in nu])

# Exact all-positive-index brackets; no arbitrary index cutoff.
a4 = {2:F(2890**2),3:F(7168),4:F(256,39375),
      6:F(1,18564**4),10:F(1,18564**4)}


def bracket(d, target4):
    def value(n):
        return a4[d]*count(n,d)**4
    if value(1) > target4:
        return 0,1
    hi = 2
    while value(hi) < target4:
        hi *= 2
    lo = 1
    while hi-lo > 1:
        mid = (hi+lo)//2
        if value(mid) <= target4:
            lo = mid
        else:
            hi = mid
    if value(hi) == target4:
        return hi,hi
    if value(lo) == target4:
        return lo,lo
    return lo,hi


electron = {d: bracket(d,F(1)) for d in a4}
assert [d for d, (lo,hi) in electron.items() if lo==hi] == [6]
assert electron[6] == (13,13) and electron[10] == (7,8)
w4 = (F('80369.2')/F('0.51099895000'))**4
certificates = []
for d, expected in [(3,(45,46)),(4,(58,59))]:
    lo,hi = bracket(d,w4)
    assert (lo,hi) == expected
    low = a4[d]*count(lo,d)**4/w4
    high = a4[d]*count(hi,d)**4/w4
    assert low < F('0.995')**4 and high > F('1.005')**4
    certificates.append(dict(d=d, low_n=lo, high_n=hi,
                             low_ratio_fourth=str(low),high_ratio_fourth=str(high)))
assert a4[3]/(F('2.16')/F('0.51099895000'))**4 > 16
report['assignment_certificates'] = dict(electron=electron,W=certificates,
    scope='Exact electron forces 6; equal lepton unit and distinct depths force tau10; W forces2; up forces4; remaining block3.')
nearest=[]
for d, rows in families.items():
    for name,n,M in rows:
        target=(F(M)/F('0.51099895000'))**4
        lo,hi=bracket(d,target)
        if lo==0:
            chosen=1
        elif lo==hi:
            chosen=lo
        else:
            boundary=a4[d]**2*(count(lo,d)*count(hi,d))**4
            chosen=lo if target**2 < boundary else hi
        assert chosen==n
        nearest.append(dict(name=name,d=d,n=chosen))
report['nearest_logarithmic_indices']=nearest
bottom_target=(F('4183')/F('0.51099895000'))**4
lo,hi=bracket(3,bottom_target)
assert (lo,hi)==(16,17)
assert bottom_target**2 > a4[3]**2*(count(lo,3)*count(hi,3))**4
report['bottom_control']=dict(nearest_ordinary_n=17,ordinary_count=count(17,3),
    ordinary_MeV=float(units[3]*count(17,3)),
    geometric_mean_MeV=float(units[3]*s.sqrt(count(16,3)*count(17,3))),
    interpretation='Unresolved exception; no beat mechanism derived.')
report['neutrino_cubic_ansatz']=dict(
    unit_MeV=float(s.Rational(3,4)*units[6]**3/units[4]**2),
    unit_eV=float(10**6*s.Rational(3,4)*units[6]**3/units[4]**2),
    interpretation='Separate phenomenological scale expression, not a consequence of the anchor relation.')

# Sawtooth identity, including negative arguments and exact rational phases.
def saw(x):
    return x-(x.numerator//x.denominator)-F(1,2)


for C in range(1,33):
    for numerator in range(-19,20):
        x = F(numerator,37)
        assert sum(saw(x+F(j,C)) for j in range(C)) == saw(C*x)
report['sawtooth_exact_checks'] = 32*39
report['bubble_norm_coefficients'] = {str(d):str(s.simplify(
    s.pi**s.Rational(d,2)*s.gamma(s.Rational(d,2)-1)/s.gamma(d-1)))
    for d in (3,4,5,6,10)}

# Real Pauli words diagonalize conjugation by every Clifford monomial.
# Labels: 0=I, 1=X, 2=Z, 3=[[0,-1],[1,0]]. Signs do not affect commutation.
def commute(a,b):
    return sum(x != y and x != 0 and y != 0 for x,y in zip(a,b)) % 2 == 0


def xor(a,b):
    return tuple(x^y for x,y in zip(a,b))


e7 = [(3,0,0),(2,3,0),(2,2,3),(2,1,3),(1,0,3),(1,3,1),(1,3,2)]
assert all(not commute(a,b) for a,b in combinations(e7,2))
g9 = [(3,)+a for a in e7]+[(1,0,0,0),(2,0,0,0)]
g10 = [(2,)+a for a in g9]+[(1,0,0,0,0)]
q = [(3,0),(2,3),(1,3)]
generators = {1:[(0,)],2:[(3,)],3:q,4:q}
for d in (5,6,7):
    generators[d] = [xor(a,b) for a,b in combinations(e7[:d],2)]
generators[8] = e7+[xor(a,b) for a,b in combinations(e7,2)]
generators[9] = [xor(a,b) for a,b in combinations(g9,2)]
generators[10] = [xor(a,b) for a,b in combinations(g10,2)]
clock_rows = []
for d, gen in generators.items():
    words = [w for w in product(range(4),repeat=len(gen[0]))
             if all(commute(w,g) for g in gen)]
    skew = sum(w.count(3)%2 for w in words)
    if d==1:
        words=[(0,)]; skew=0
    assert skew == {1:0,2:1,3:3,4:3,5:3,6:1,7:0,8:0,9:0,10:1}[d]
    clock_rows.append(dict(d=d,commutant_dimension=len(words),skew_dimension=skew))
report['minimal_real_clocks'] = clock_rows
band=[]
for d,sd in [(7,8),(8,8),(9,16)]:
    for n in range(1,33):
        k=n-1
        dimension=0
        kernel=0
        for ell in range(k+1):
            rank=sd*comb(ell+d-2,d-2)
            mult=[k-ell+1] if d!=8 else [(k-ell)//2+1,(k-ell+1)//2]
            dimension+=sum(mult)*rank
            kernel+=sum(m%2 for m in mult)*rank
        assert dimension==sd*count(n,d)
        assert F(kernel,dimension)>=F(d*(d-1),(n+d-1)*(n+d-2))
        if n==2:
            band.append(dict(d=d,dimension=dimension,minimal_kernel=kernel))
report['complete_band_n2']=band
adjoint=[]
for d in (2,3,4,5,6,10):
    grades=[]
    for t in range(0,d+1,2):
        if d%2==0 and t>d//2:
            continue
        multiplicity=comb(d,t)//2 if d%2==0 and t==d//2 else comb(d,t)
        grades.append((t*(d-t),multiplicity))
    assert sum(m for _,m in grades)==2**(2*((d-1)//2))
    positive=[(c,m) for c,m in grades if c>0]
    value=min((c for c,_ in positive),default=0)
    rank=sum(m for c,m in positive if c==value)
    adjoint.append(dict(d=d,lowest_positive=value,rank=rank))
assert [v['rank'] for v in adjoint]==[0,3,3,5,15,45]
report['minimal_adjoint_channels']=adjoint

# Complete middle Hodge ranks and signed-permutation star matrices.
hodge = []
for m in range(1,6):
    D = 2*m
    basis = list(combinations(range(D),m))
    indices = {b:i for i,b in enumerate(basis)}
    star = np.zeros((len(basis),len(basis)),dtype=np.int64)
    for col,b in enumerate(basis):
        complement = tuple(i for i in range(D) if i not in b)
        inversions = sum(i>j for i in b for j in complement)
        star[indices[complement],col] = (-1)**inversions
    close(star@star,(-1)**m*np.eye(len(basis)))
    close(star.T,(-1)**m*star)
    H = (star+star.T)/2
    assert np.count_nonzero(H) == (len(basis) if m%2==0 else 0)
    hodge.append(dict(parent=D,shadow=D-1,rank=comb(D-1,m-1),
                      instantaneous_active=bool(m%2==0)))
report['hodge'] = hodge

# SU(3) and the spin/color clock, with separate indices.
T=[]
for i,j in [(0,1),(0,2),(1,2)]:
    a=np.zeros((3,3),complex); a[i,j]=a[j,i]=.5; T.append(a)
    b=np.zeros((3,3),complex); b[i,j]=-.5j; b[j,i]=.5j; T.append(b)
T += [np.diag([1,-1,0])/2,np.diag([1,1,-2])/(2*np.sqrt(3))]
close(sum(t@t for t in T),np.eye(3)*4/3)
swap=np.zeros((9,9))
for i,j in product(range(3),repeat=2):
    swap[3*j+i,3*i+j]=1
pair=sum(np.kron(t,t) for t in T)
close(pair,(swap-np.eye(9)/3)/2)
report['color_pair_eigenvalues'] = np.linalg.eigvalsh(pair).tolist()


def realify(a):
    return np.block([[a.real,-a.imag],[a.imag,a.real]])


epsilon=np.array([[0,-1],[1,0]])
J0=realify(1j*np.eye(6))
J1=realify(np.kron(epsilon,np.eye(3)))@np.diag([1]*6+[-1]*6)
J2=J0@J1
defect=np.zeros((3,3))
for t in T:
    r=realify(1j*np.kron(np.eye(2),t))
    comm=[j@r-r@j for j in (J0,J1,J2)]
    defect += np.array([[np.sum(a*b)/12 for b in comm] for a in comm])
close(defect,np.diag([0,10/3,10/3]))
report['joint_clock_defect_matrix'] = defect.tolist()

# Explicit full ten-space Clifford representation and lowest angular charges.
I=np.eye(2); X=np.array([[0,1],[1,0]]);Y=np.array([[0,-1j],[1j,0]]);Z=np.diag([1,-1])
gamma=[]
for j in range(5):
    for a in (X,Y):
        gamma.append(kron_all([Z]*j+[a]+[I]*(4-j)))
for i,j in product(range(10),repeat=2):
    close(gamma[i]@gamma[j]+gamma[j]@gamma[i],2*np.eye(32)*(i==j))
G4=gamma[0]@gamma[1]@gamma[2]@gamma[3]
normal_spin=sum(-.5j*gamma[i]@gamma[i+1] for i in (4,6,8))
normal_parity=np.eye(32,dtype=complex)
for i in (4,6,8):
    normal_parity=normal_parity@(-1j*gamma[i]@gamma[i+1])
# even j corresponds to normal parity -1 with this occupation convention.
P=(np.eye(32)-normal_parity)/2
values, vectors=np.linalg.eigh(P)
U=vectors[:,values>.5]
Q=(1.5*G4+normal_spin)/3
qvals=np.linalg.eigvalsh(U.conj().T@Q@U)
close(qvals,np.array([-1]*2+[-1/3]*6+[0]*2+[2/3]*6))
J=[-.5j*gamma[1]@gamma[2],-.5j*gamma[2]@gamma[0],-.5j*gamma[0]@gamma[1]]
close(sum(j@j for j in J),np.eye(32)*.75)
for j in J:
    close(j@Q-Q@j,np.zeros((32,32)))
report['lowest_charge_eigenvalues'] = qvals.tolist()

# Five-depth handed/neutral sign bridge and normal radial integral.
alpha=[np.kron(np.kron(Z,Z),a) for a in (X,Y,Z)]
alpha += [np.kron(np.kron(Z,a),I) for a in (X,Y)]
eta=np.kron(np.kron(Z,I),I)
K=-1j*alpha[3]@alpha[4]
C5=-1j*alpha[0]@alpha[1]@alpha[2]
close(C5,eta@K)
Q5=(eta+K)/3
close(Q5@Q5,2/9*(np.eye(8)+C5))
I0=s.beta(s.Rational(5,2),s.Rational(5,2))/2
I2=s.beta(s.Rational(7,2),s.Rational(3,2))/2
assert s.simplify(s.expand_func(I2/I0))==s.Rational(5,3)
report['depth5_bridge']=dict(sign_product=True,neutral_left=True,
                             reversed_norm_fraction='1/4')

# Nonlinear repair identity, independent of numerical profiles.
rho,A,g,h=s.symbols('rho A g h')
W=lambda t:-g*t*t+h*t**3
assert s.expand(W(rho+A)-W(rho)+g*g*A/(3*h)
                -3*h*A*(rho+A/2-g/(3*h))**2-h*A**3/4)==0
report['nonlinear_singlet_identity'] = True
report['scope'] = 'Finite mathematics only; physical premises and continuum existence are not validated by this script.'
report['source_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
(ROOT/'particle-geometry-checks.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'status':'passed','source_sha256':report['source_sha256'],
                  'mass_rows':len(mass_rows),'clock_rows':len(clock_rows)}),flush=True)
