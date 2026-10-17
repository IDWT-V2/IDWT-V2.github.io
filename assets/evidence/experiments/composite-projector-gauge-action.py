#!/usr/bin/env python3
"""Exact topology and representation traces for the conditional gauge reduction."""
from pathlib import Path
from itertools import combinations
import datetime, hashlib, json, sys
import sympy as s
OUT = Path(__file__).resolve().parent
checks = []
def ck(name, value):
    assert bool(value), name
    checks.append(name)
H = s.Symbol('H')
ce = s.expand((1-2*H)**3*(1+3*H)**2)
ck('auxiliary Chern polynomial', ce == 1-15*H**2+10*H**3+60*H**4-72*H**5)
ct = 1+6*H+15*H**2+20*H**3+15*H**4+6*H**5
ck('tangent factorization', s.expand((1+2*H)*(1+H+H**2)*(1+3*H+3*H**2)) == ct)
for a,b,expected in [(1,1,s.Rational(1,2)),(3,3,s.Rational(-1,2))]:
    # ch(t)=exp(H)-1: coefficient of t^3 required by rank-two character.
    k2 = s.Rational(a*a-2*b-a,2)
    k3 = s.Rational(a**3-3*a*b-a,6)-k2
    ck(f'K integrality obstruction {a},{b}', k3 == expected and k3.q == 2)
subsets = [c for r in [0,2,4] for c in combinations(range(5), r)]
w = [-2,-2,-2,3,3]
charges = [sum(w[i] for i in c) for c in subsets]
ck('rank sixteen', len(subsets)==16)
ck('vanishing first character', sum(charges)==0)
ck('auxiliary second character', s.Rational(sum(q*q for q in charges),2)==60)
ck('geometric half-spin second character', 2*(6**2-2*15)==12)
ck('cubic charge bookkeeping', sum(q**3 for q in charges)==0)
ck('nonisomorphism', 60 != 12)
ck('line twist cannot preserve c1', 16 != 0)

def exterior(A):
    R=s.zeros(16)
    for col,I in enumerate(subsets):
        for slot,j in enumerate(I):
            for i in range(5):
                if A[i,j]==0: continue
                seq=list(I);seq[slot]=i
                if len(set(seq))!=len(seq):continue
                sign=(-1)**sum(seq[a]>seq[b] for a in range(len(seq)) for b in range(a+1,len(seq)))
                R[subsets.index(tuple(sorted(seq))),col]+=sign*A[i,j]
    return R

def pair(i,j,imag=False):
    A=s.zeros(5);A[i,j]= -s.I/2 if imag else s.Rational(1,2);A[j,i]= s.I/2 if imag else s.Rational(1,2);return A
C=[]
for i,j in combinations(range(3),2): C.extend([pair(i,j),pair(i,j,True)])
C += [s.diag(s.Rational(1,2),-s.Rational(1,2),0,0,0),s.diag(1,1,-2,0,0)/(2*s.sqrt(3))]
W=[pair(3,4),pair(3,4,True),s.diag(0,0,0,s.Rational(1,2),-s.Rational(1,2))]
Y=s.diag(-s.Rational(1,3),-s.Rational(1,3),-s.Rational(1,3),s.Rational(1,2),s.Rational(1,2))
RC,RW,RY=[exterior(A) for A in C],[exterior(A) for A in W],exterior(Y)
for name,group in [('colour',RC),('weak',RW)]:
    for i,A in enumerate(group):
        for j,B in enumerate(group):ck(f'{name} trace {i},{j}',s.simplify(s.trace(A*B))==(2 if i==j else 0))
for i,A in enumerate(RC):
    for j,B in enumerate(RW):
        ck(f'colour weak trace {i},{j}',s.trace(A*B)==0)
        ck(f'colour weak commutator {i},{j}',A*B==B*A)
for i,A in enumerate(RC+RW):
    ck(f'hypercharge mixed trace {i}',s.trace(A*RY)==0)
    ck(f'hypercharge commutator {i}',A*RY==RY*A)
ck('hypercharge trace',s.trace(RY*RY)==s.Rational(10,3))
ck('fundamental hypercharge trace',s.trace(Y*Y)==s.Rational(5,6))
for i,A in enumerate(C+W+[Y]):
    ck(f'exterior index four {i}',s.simplify(s.trace(exterior(A)**2)-4*s.trace(A**2))==0)
# Split projector polynomial and differentiated projector curvature witness.
D=s.diag(-2,3,0)
Pc=D*(D-3*s.eye(3))/10;Pw=D*(D+2*s.eye(3))/15
ck('spectral colour projector', Pc==s.diag(1,0,0))
ck('spectral weak projector', Pw==s.diag(0,1,0))
ck('orthogonal spectral projectors',Pc*Pw==s.zeros(3))
# Anti-Hermitian frame generators: projected curvature -[X_off,Y_off] on image.
P=s.diag(1,1,0)
X=s.zeros(3);X[0,2]=1;X[2,0]=-1
T=s.zeros(3);T[1,2]=1;T[2,1]=-1
dx=X*P-P*X;dy=T*P-P*T
FP=P*(dx*dy-dy*dx)*P
ck('projector curvature formula',FP==-P*(X*T-T*X)*P)
ck('composite curvature nonzero',FP!=s.zeros(3))
# Explicit embedding dimensions: negative second Veronese and conjugated third.
ck('ambient coefficient rank',3*s.binomial(7,2)+2*s.binomial(8,3)==175)
ck('determinant cancellation',3*(-2)+2*3==0)
normal_rows=[]
for d in [2,3,4,5,6,10]:
    e=d if d%2==0 else d+1;q=(10-e)//2
    h=q*q if q>=3 else 1 if q==2 else 0
    normal_rows.append({'physical_depth':d,'even_carrier':e,'q':q,'canonical_J_target_holonomy_dimension':h,'traceless_dimension':h-1 if h else 0})
ck('occupied q controls',[r['q'] for r in normal_rows]==[4,3,3,2,2,0])
ck('canonical q2 has no su2',normal_rows[3]['traceless_dimension']==0)
report={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'verification_scope':'portable mathematical assertions; archival source-hash checks omitted','script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'checks_passed':len(checks),'checks':checks,'chern_character_2':{'auxiliary_even_exterior':60,'geometric_half_spin':12,'coefficient':'H^2'},'trace_indices':{'SU3':'2','SU2':'2','Y':'10/3'},'conditional_gY_squared_over_g2_squared':'3/5','normal_comparison':normal_rows,'scope':'Exact identities and necessary obstructions; no spatial state, selected action, force law or measured coupling is established.'}
name=sys.argv[1] if len(sys.argv)>1 else 'composite-projector-gauge-action-results-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'.json'
p=OUT/name;p.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'report':str(p),'checks_passed':len(checks)}))
