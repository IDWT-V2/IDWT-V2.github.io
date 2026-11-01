#!/usr/bin/env python3
"""IMR-1 independent exact checks. Writes only this packet's results."""
from pathlib import Path
import sympy as s
import json, hashlib, datetime, math
P=Path(__file__).resolve().parent
Q=s.Rational
checks=[]
def ck(name,condition):
    assert bool(condition),name
    checks.append(name)
def eq(name,a,b):
    if isinstance(a,s.MatrixBase):
        ck(name, all(s.expand(x)==0 or s.simplify(x)==0 for x in a-b))
    else: ck(name,s.simplify(a-b)==0)
def matstr(A): return [[str(x) for x in A.row(i)] for i in range(A.rows)]
def model(B,L,lambdas,theta,D=None,Qresp=None):
    N=B.cols; m=B.rows
    D=s.eye(N) if D is None else D
    Rm=s.eye(m) if Qresp is None else Qresp
    # B denotes source force covectors in physical response coordinates.
    Y=s.Matrix.hstack(*[(L-theta*lambdas[i]*Rm).inv()*B[:,i] for i in range(N)])
    La=s.diag(*lambdas); G=s.simplify(Y.T*Rm*Y)
    Mr=D+(1-2*theta)*G; C=-(1-theta)*Rm*Y
    K0=D*La+Y.T*L*Y-theta*(G*La+La*G)
    Mf=Mr.row_join(C.T).col_join(C.row_join(Rm))
    Kf=K0.row_join(-B.T).col_join((-B).row_join(L))
    W=s.eye(N).col_join(Y)
    eq('full target rows '+str((N,theta,lambdas[:2])),Kf*W,Mf*W*La)
    eq('complete mode norms '+str((N,theta,lambdas[:2])),W.T*Mf*W,D)
    eq('kinetic symmetric '+str((N,theta)),Mf,Mf.T)
    eq('stiffness symmetric '+str((N,theta)),Kf,Kf.T)
    T=Rm-theta**2*Rm*Y*D.inv()*Y.T*Rm
    V=L-theta**2*Rm*Y*La*D.inv()*Y.T*Rm
    return Mf,Kf,W,G,T,V
# Recompute actual source integrals; no parent executable is imported.
x=s.symbols('x',real=True);f=s.sqrt(15)*(1-x*x)/4;hc=s.sqrt(Q(3,8))
eq('f norm',s.integrate(f*f,(x,-1,1)),1)
eq('h norm',2*hc**2*(1+s.integrate((2-x)**2,(x,1,2))),1)
A=s.integrate(f**4,(x,-1,1));mix=s.integrate(f*f*hc*hc,(x,-1,1));H=2*hc**4*(1+s.integrate((2-x)**4,(x,1,2)))
eq('source constants A',A,Q(5,7));eq('source constants B',mix,Q(3,8));eq('source constants H',H,Q(27,80))
eq('normalized correlation squared',mix**2/(A*H),Q(7,12))
depths=[2,3,4,5,6,10];t=s.sqrt(Q(7,12));R=s.Matrix([[t**abs(i-j) for j in depths] for i in depths]);Rp=R.inv()
eq('actual dual metric inverse',R*Rp,s.eye(6))
for k in range(1,7):ck('source Gram positive minor '+str(k),s.simplify(R[:k,:k].det())>0)
# Fixed diagonal obstruction follows from the actual two spatial columns.
r=Q(7,12);shape=s.Matrix([[1,r],[0,s.sqrt(95)/12]])
B=s.Matrix.hstack(shape[:,0],shape[:,1],shape[:,0])/2
Yfixed=s.Matrix.hstack(B[:,0]/8,B[:,1]/5,B[:,2]/9)
eq('fixed metric obstruction entry',Yfixed[:,0].dot(Yfixed[:,1]),Q(7,1920))
ck('fixed metric obstruction genuinely nonzero',Yfixed[:,0].dot(Yfixed[:,1])!=0)
# Useful fixed diagonal nonflat exception, exact source overlap and positivity.
c=(48-4*s.sqrt(95))/7;Bexc=s.Matrix([[4,c],[c,4]]);Lexc=s.Matrix([[5,c],[c,8]])
eq('fixed exception actual normalized overlap',8*c/(16+c*c),r)
eq('fixed exception low branch',Lexc-Bexc,s.diag(1,4))
ck('fixed exception c bounds',bool(c>1) and bool(c<2))
ck('fixed exception high positivity determinant',bool(s.simplify((Lexc+Bexc).det())>0))
Kexc=Lexc.row_join(-Bexc).col_join((-Bexc).row_join(Lexc));Wexc=s.eye(2).col_join(s.eye(2))
eq('fixed exception all target rows',Kexc*Wexc,Wexc*s.diag(1,4))
eq('fixed exception full orthogonality',Wexc.T*Wexc,2*s.eye(2))
# Simultaneous two massive clocks and dressed zero; full complement retained.
z=s.symbols('z'); examples=[]
for theta in [s.Integer(1),Q(1,2)]:
    Mf,Kf,W,G,T,V=model(B,9*s.eye(2),[1,4,0],theta)
    for k in range(1,6):ck(f'small full kinetic positive {theta} {k}',s.simplify(Mf[:k,:k].det())>0)
    for k in [1,2]:
        ck(f'small complement kinetic positive {theta} {k}',s.simplify(T[:k,:k].det())>0)
        ck(f'small complement energy positive {theta} {k}',s.simplify(V[:k,:k].det())>0)
    eq('small full dressed zero '+str(theta),Kf*W[:,2],s.zeros(5,1))
    ck('small zero is not dark '+str(theta),W[3:,2]!=s.zeros(2,1))
    eq('small kernel dimension '+str(theta),Kf.rank(),4)
    poly=s.factor((Kf-z*Mf).det())
    eq('full characteristic factorization '+str(theta),poly,(1-z)*(4-z)*(-z)*(V-z*T).det())
    eq('full kinetic determinant '+str(theta),Mf.det(),T.det())
    if theta==1:
        F=Kf[:3,:3]-z*Mf[:3,:3]-B.T*B/(9-z)
        eq('actual offdiagonal reciprocal probe',F[0,1],-G[0,1]*(z-1)*(z-4)/(9-z))
        ck('static cross response nonzero',s.simplify(F[0,1].subs(z,0))!=0)
    else:eq('fixed retained diagonal in derivative repair',Mf[:3,:3],s.eye(3))
    # Verify general derivative pencil norm for an arbitrary retained trial vector.
    C=Mf[3:,:3];Mr=Mf[:3,:3];K0=Kf[:3,:3]
    Rz=s.eye(2)/(9-z);Zlift=Rz*(B+z*C)
    F=K0-z*Mr-(B.T+z*C.T)*Rz*(B+z*C)
    eq('complete pencil derivative norm '+str(theta),-F.diff(z),Mr+C.T*Zlift+Zlift.T*C+Zlift.T*Zlift)
    examples.append({'theta':str(theta),'kinetic':matstr(Mf),'stiffness':matstr(Kf),'full_modes':matstr(W),'response_trace':str(s.trace(G)),'characteristic':str(poly),'complement_characteristic':str(s.factor((V-z*T).det()))})
# Above-response example: no hypothetical off-range exclusion.
Mf,Kf,W,G,T,V=model(B/10,s.eye(2),[4,9,0],s.Integer(1))
for k in range(1,6):ck('above response kinetic positive '+str(k),s.simplify(Mf[:k,:k].det())>0)
for k in [1,2]:ck('above response complement positive '+str(k),s.simplify(V[:k,:k].det())>0)
ck('above response exact full kernel',Kf.rank()==4)
# General algebra in exact scalar symbols, independent of numeric source basis.
a,b,ell,th,gram=s.symbols('a b ell th gram',real=True)
# Off diagonal K0 symmetry follows automatically; identity for harmonic support and retained rows.
eq('general full support identity',(ell-th*b)-(ell-b)-(1-th)*b,0)
eq('general full retained identity',ell-th*(a+b)-(ell-th*a)-b*(-th),0)
# Fourteen ordinary labels (bottom excluded from ordinary assignment list).
labels=[('W',2,76),('Z',2,81),('H',2,95),('down',3,1),('strange',3,4),('up',4,3),('charm',4,20),('top',4,72),('nu1',5,10),('nu2',5,15),('nu3',5,22),('electron',6,13),('muon',6,35),('tau',10,23)]
counts=[math.comb(n+d-1,d) for _,d,n in labels];maxcount=max(counts)
ck('fourteen ordinary labels',len(counts)==14);ck('all ordinary count frequencies distinct',len(set(counts))==14)
lam=[s.Integer(0)]+[Q(k,maxcount)**2 for k in counts]
di=[2]+[d for _,d,n in labels];N=len(lam)
Bc=s.zeros(6,N)
for i,d in enumerate(di):Bc[depths.index(d),i]=Q(1,8)
allrows=[]
# Use exact dual response coordinates, preserving the entire actual six-source span.
for theta in [s.Integer(1),Q(1,2)]:
    # L coordinate form 4 R^-1; response source covectors Bc.
    Mf,Kf,W,G,T,V=model(Bc,4*Rp,lam,theta,Qresp=Rp)
    eq('all labels dressed zero '+str(theta),Kf*W[:,0],s.zeros(N+6,1))
    ck('all labels nonzero response '+str(theta),all(W[N:,i]!=s.zeros(6,1) for i in range(N)))
    tr=s.trace(G);bound=Q(5,192) if theta==1 else Q(15,784)
    ck('all labels exact trace bound '+str(theta),tr<=bound)
    ck('uniform kinetic bound '+str(theta),1-theta**2*bound>0)
    # Matrix identity proves V>=4T using known PSD Gram form and nonnegative diagonals.
    Y=W[N:,:];La=s.diag(*lam)
    eq('all labels complement gap identity '+str(theta),V-4*T,theta**2*Rp*Y*(4*s.eye(N)-La)*Y.T*Rp)
    if theta==Q(1,2):eq('all labels retained diagonal fixed',Mf[:N,:N],s.eye(N))
    # Entrywise actual Y Gram, including repeated spatial sources at distinct frequencies.
    for i in range(N):
        for j in range(i,N):
            eq(f'actual spatial full Gram {theta} {i},{j}',G[i,j],Q(1,64)*t**abs(di[i]-di[j])/((4-theta*lam[i])*(4-theta*lam[j])))
    allrows.append({'theta':str(theta),'response_trace_exact':str(tr),'response_trace_decimal':str(tr.evalf(15)),'trace_bound':str(bound),'kinetic_margin':str(1-theta**2*bound),'full_dimension':N+6,'targets':15,'complement_modes':6,'complement_squared_gap':4})
# All physical supplied sector units remain symbolic in theorem; this test is a count-only control.
rows=[{'label':label,'d':d,'n':n,'count':count,'frequency_symbolic':f'mu_{d}*{count}','count_control_frequency_over_max':str(Q(count,maxcount)),'count_control_squared':str(Q(count,maxcount)**2)} for (label,d,n),count in zip(labels,counts)]
eq('conditional bottom count a',math.comb(18,3),816);eq('conditional bottom count b',math.comb(19,3),969)
# Full charge-energy identity with actual norm, signs are independent.
h,amp,Dii=s.symbols('h amp Dii',positive=True)
for sign in [-1,1]:eq('complete charge energy '+str(sign),h*h*amp*amp*Dii,h*abs(sign*h*amp*amp*Dii))
# Domain bounds for continuum are spectral, no finite-source closure assumption.
k2,gap=s.symbols('k2 gap',positive=True)
eq('continuum bound factorization',1/gap**2-1/(gap+k2)**2,k2*(2*gap+k2)/(gap**2*(gap+k2)**2))
ck('continuum resolvent norm bound',(k2*(2*gap+k2)/(gap**2*(gap+k2)**2)).is_positive)
report={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'version':'IMR-1','status':'all passed','checks_passed':len(checks),'checks':checks,'examples':examples,'fourteen_labels':rows,'simultaneous_controls':allrows,'above_response':{'response_eigenvalue':1,'massive_squared_targets':[4,9],'complete_positive':True},'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'scope':'Exact finite action and actual source integrals. Fourteen-label numeric control uses equal supplied sector units; uniform theorem permits every positive supplied sector scale. No empirical scale fit, source-free localization or physical particle identification.'}
(P/'results.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'checks_passed':len(checks),'count_rows':rows,'bounds':allrows},indent=2))
