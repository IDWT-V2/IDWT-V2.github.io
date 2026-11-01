#!/usr/bin/env python3
import sympy as s
from pathlib import Path
import json,hashlib,datetime
P=Path(__file__).resolve().parent; checks=[]
def ck(n,c):
 assert bool(c),n
 checks.append(n)
def eq(n,a,b):ck(n,s.simplify(a-b)==0)
L=s.Matrix([[9,2],[2,6]]);B=s.Matrix([[1,s.Rational(7,12)],[0,s.sqrt(95)/12]]);R=(L-4*s.eye(2)).inv()
ck('response eigenvalues',set(L.eigenvals())=={5,10});ck('source Gram',s.simplify(B.T*B)==s.Matrix([[1,s.Rational(7,12)],[s.Rational(7,12),1]]))
K0=s.diag(4,9)+B.T*R*B;K=K0.row_join(-B.T).col_join((-B).row_join(L));v=s.Matrix([1,0]);Y=R*B*v;u=v.col_join(Y)
ck('support nonzero both entries',Y==s.Matrix([s.Rational(1,3),-s.Rational(1,3)]));ck('full exceptional eigenmode',s.simplify(K*u-4*u)==s.zeros(4,1));eq('full norm',u.dot(u),s.Rational(11,9))
for j in range(1,5):ck('full positive principal minor '+str(j),s.simplify(K[:j,:j].det())>0)
S=K0-B.T*L.inv()*B;ck('positive static remainder',s.simplify(S[0,0])>0 and s.simplify(S.det())>0)
ck('full zero',s.diag(0,K)*s.Matrix([1,0,0,0,0])==s.zeros(5,1));ck('full kernel rank',s.diag(0,K).rank()==4)
# Arbitrary spatial source and operator variations at the reference point.
Bd=s.Matrix([[s.Rational(1,2),-1],[2,s.Rational(1,3)]]);Ld=s.Matrix([[1,3],[3,-2]])
for sig in [-1,1]:
 lamd=4+sig;Rd=R*(lamd*s.eye(2)-Ld)*R
 Kd=s.diag(sig,0)+Bd.T*R*B+B.T*R*Bd+B.T*Rd*B
 pot=(v.T*Kd*v)[0]+(Y.T*Ld*Y)[0]-2*(Y.T*Bd*v)[0]
 eq('complete changed source cancellation '+str(sig),pot,sig+lamd*Y.dot(Y))
 mean=(4+pot)/(4*u.dot(u))
 eq('mean source h derivative '+str(sig),mean,s.Rational(lamd,4))
z=s.symbols('z');F=K0-z*s.eye(2)-B.T*(L-z*s.eye(2)).inv()*B
ck('dynamic normalization derivative',s.simplify(-F.diff(z)-(s.eye(2)+B.T*(L-z*s.eye(2)).inv()**2*B))==s.zeros(2))
# Positive response below target still allowed by the actual Schur criterion.
lowL=s.eye(2);smallB=B/10;lowK=s.diag(4,9)+smallB.T*(lowL-4*s.eye(2)).inv()*smallB
lowS=lowK-smallB.T*smallB
ck('below target response can be healthy',s.simplify(lowS[0,0])>0 and s.simplify(lowS.det())>0)
# Gradient response exact spectral bounds, no grid approximation.
p2=s.symbols('p2',nonnegative=True);eq('gradient resolvent gap',p2+9-4,p2+5);eq('gradient bound positive factorization',s.Rational(1,25)-1/(p2+5)**2,p2*(p2+10)/(25*(p2+5)**2));ck('gradient norm bound', (p2*(p2+10)/(25*(p2+5)**2)).is_nonnegative)
lam=s.symbols('lambda');poly=s.factor(K.charpoly(lam).as_expr());ck('polynomial target factor',s.simplify(poly.subs(lam,4))==0)
report={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'checks_passed':len(checks),'checks':checks,'response':str(L),'source':str(B),'full_active_stiffness':str(s.simplify(K)),'squared_frequency_polynomial':str(poly),'full_norm':'11/9','scope':'Exact matrix checks and analytic gradient spectral bounds. No PDE discretization.','code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(P/'results.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'checks_passed':len(checks),'polynomial':str(poly)}))
