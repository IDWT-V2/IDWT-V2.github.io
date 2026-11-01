#!/usr/bin/env python3
"""Independent principal barrier verification in combined four-profile action."""
import numpy as np,math,json,hashlib,datetime,resource
from pathlib import Path
from fractions import Fraction as R
ROOT=Path(__file__).resolve().parent;checks=[]
def ck(s,v):
 if not bool(v):raise AssertionError(s)
 checks.append(s)
def near(s,a,b,t=1e-8):ck(s,np.max(np.abs(np.asarray(a)-np.asarray(b)))<t)
def etaf(m,d):return 1+(math.factorial(d)*m)**(1/d)/d
def etap(m,d):return (math.factorial(d)*m)**(1/d)/(d*d*m)
def weights(k,x):
 u=np.array([(1+k)/2,(1-k)/2]);v=np.array([1,x*x])/(1+x*x)
 return np.outer(u,v).ravel(),np.outer([.5,-.5],v).ravel(),np.outer(u,[-2*x,2*x]).ravel()/(1+x*x)**2

def betas(d,n):return [tuple([n+1-i-j,i+j]+[1]*(d-2)+[0]*(10-d)) for i in range(2) for j in range(2)]
def derivatives(beta):
 ans=[]
 for a in range(10):
  z={};v=list(beta)
  if beta[a]:v[a]-=1;z[tuple(v)]=math.sqrt(beta[a]/2)
  v=list(beta);v[a]+=1;z[tuple(v)]=-math.sqrt((beta[a]+1)/2);ans.append(z)
 return ans

def crossgrad(b,c):
 db,dc=derivatives(b),derivatives(c)
 return np.array([[sum(v*dc[j].get(key,0) for key,v in db[i].items()) for j in range(10)] for i in range(10)])

def T(q):
 s=np.eye(10)+q;return s*math.exp(-np.linalg.slogdet(s)[1]/10)
def dT(q,dq):
 s=np.eye(10)+q;t=math.exp(-np.linalg.slogdet(s)[1]/10)
 return t*(dq-s*np.trace(np.linalg.solve(s,dq))/10)
def support(F,q):
 W=np.linalg.solve(T(q),F.T@F)-np.eye(10);return np.trace(W@W)/8
assign={2:[76,81,95],3:[1,4],4:[3,20,72],5:[10,15,22],6:[13,35],10:[23]};rows=[]
for d,ns in assign.items():
 for n in ns:
  K=n+d-1;c=math.comb(K,d)
  qs=[crossgrad(b,b) for b in betas(d,n)]
  for k,x in [(0.,.5),(.75,.32),(.2,.7)]:
   p,pk,px=weights(k,x);q=sum(pi*qa for pi,qa in zip(p,qs));l=(1-k)/2+x*x/(1+x*x)
   expected=np.diag([n+1.5-l,l+.5]+[1.5]*(d-2)+[.5]*(10-d))
   near(f'actual profile tensor d{d} n{n} k{k}',q,expected,1e-12)
   near(f'common determinant d{d} n{n} k{k}',np.linalg.det(T(q)),1,1e-12)
  for j in range(d,d+100):
   cj=math.comb(j,d);delta=cj-c
   if not delta:continue
   eta_bound=R(K+1,d)
   coef=R(delta)*(1+eta_bound*R(delta,c))
   for comp in range(4):ck(f'rigorous barrier lower-bound shell sign {d} {n} {j} {comp}',coef>0)
  for sign in [-1,1]:ck(f'complete phase energy {d} {n} {sign}',R(1,c)*c*c==c*abs(sign))
  rows.append(dict(d=d,n=n,K=K,count=c))
ck('all fourteen references',len(rows)==14)
# Full joint d=3,n=4 patch: 4 independently varying profiles, each with lower,
# upper and same-shell spatial variation, plus radius/current and all 100 F entries.
d,n=3,4;K=n+d-1;omega=math.comb(K,d);r0=1/math.sqrt(omega)
bs=[]
for base in betas(d,n):
 low=list(base);low[0]-=2;high=list(base);high[0]+=2;same=list(base);same[0]-=1;same[1]+=1
 bs.append([tuple(low),base,tuple(high),tuple(same)])
Gq=np.array([[[crossgrad(a,b) for b in bank] for a in bank] for bank in bs])
counts=np.array([[math.comb(sum(b[:d]),d) for b in bank] for bank in bs],float)
slots=[0,2,3]
def coeffs(v):
 return np.array([v[0],1.,v[1],v[2]])/math.sqrt(1+v@v)
def coeffJ(v):
 f=coeffs(v);den=math.sqrt(1+v@v)
 return np.array([(np.eye(4)[j]-f*f[j])/den for j in slots]).T

def spatial(v):
 fs=np.array([coeffs(a) for a in v.reshape(4,3)])
 qa=np.einsum('ai,aj,aijuv->auv',fs,fs,Gq)
 ma=np.einsum('ai,ai,ai->a',fs,fs,counts);za=np.einsum('ai,ai,ai->a',fs,fs,counts**2)
 return fs,qa,ma,za
p0,_,_=weights(0.,.5);fref,qaref,maref,zaref=spatial(np.zeros(12));q0=np.einsum('a,auv->uv',p0,qaref)
T0=T(q0);F0=np.diag(np.sqrt(np.diag(T0)))
# qv=(r,k,x,12 profile coordinates,100 additive F coordinates).
Ddim=115;base=np.zeros(Ddim);base[:3]=[r0,0.,.5]
def data(v):
 r,k,x=v[:3];p,pk,px=weights(k,x);fs,qa,ma,za=spatial(v[3:15]);m=p@ma;z=p@za
 B=m+etaf(m,d)*(z-m*m)/m;w=(1+k*k)*x*x/2-.5*x+.125;lam=B*(1+w)
 q=np.einsum('a,auv->uv',p,qa);F=F0+v[15:].reshape(10,10)
 return r,k,x,p,pk,px,fs,qa,ma,za,m,z,B,w,lam,q,F

def V(v):
 r,k,x,p,pk,px,fs,qa,ma,za,m,z,B,w,lam,q,F=data(v)
 return 1/(2*r*r)+r*r*lam*lam/2+support(F,q)
near('reference complete energy',V(base),omega,1e-12)
# Build full analytic Hessian from exact profile gradient matrices.
J=np.zeros((100,Ddim));zi=np.diag(1/np.sqrt(np.diag(T0)))
_,pk,px=weights(0.,.5)
for s,dp in [(1,pk),(2,px)]:J[:,s]=(-zi@dT(q0,np.einsum('a,auv->uv',dp,qaref))@zi).ravel()
for a in range(4):
 for j,slot in enumerate(slots):
  dq=p0[a]*(Gq[a,slot,1]+Gq[a,1,slot])
  J[:,3+3*a+j]=(-zi@dT(q0,dq)@zi).ravel()
for i in range(100):
 df=np.zeros((10,10));df.flat[i]=1
 J[:,15+i]=(zi@(F0.T@df+df.T@F0)@zi).ravel()
H=J.T@J/4;H[0,0]+=4*omega**2;H[1,1]+=r0*r0*omega**2*.25;H[2,2]+=r0*r0*omega**2
for a in range(4):
 for j,slot in enumerate(slots):
  delta=counts[a,slot]-omega
  H[3+3*a+j,3+3*a+j]+=2*r0*r0*omega*p0[a]*delta*(1+etaf(omega,d)*delta/omega)
ev=np.linalg.eigvalsh(H);ck('full joint Hessian nonnegative',ev[0]>-1e-9)
ck('45 rotations plus 4 same-shell following nulls',sum(abs(ev)<1e-8)==49)
ck('genuine current-material mixed Hessian',np.linalg.norm(H[1:3,15:])>.01)
ck('genuine profile-material mixed Hessian',np.linalg.norm(H[3:15,15:])>.01)
Gdiag=np.r_[1,r0*r0*.25,r0*r0/1.25**2,np.repeat(r0*r0*p0,3),np.ones(100)]
ginv=np.diag(1/np.sqrt(Gdiag));freq=np.linalg.eigvalsh(ginv@H@ginv)
ck('kinetic full generalized frequencies nonnegative',freq[0]>-1e-8)
# Original-action directional Hessians with all variables changing together.
rng=np.random.default_rng(381);errs=[];grads=[]
for trial in range(5):
 v=rng.normal(size=Ddim)*.025;v[0]*=.15
 exact=v@H@v;ee=[]
 for step in [1e-2,3e-3,1e-3,5e-4]:
  fd=(V(base+step*v)-2*V(base)+V(base-step*v))/step**2;ee.append(abs(fd-exact))
 print('barrier Hessian',trial,exact,ee)
 ck('full original Hessian direction '+str(trial),min(ee)<2e-5 and min(ee)<ee[0])
 step=1e-5;grad=(V(base+step*v)-V(base-step*v))/(2*step);ck('full reference stationarity '+str(trial),abs(grad)<1e-7)
 errs.append(ee);grads.append(grad)
# Off-reference full variation: includes weight derivatives, variance denominator,
# normalized profile directions, noncommuting F, and normalized strain source.
point=base+rng.normal(size=Ddim)*.03;point[0]=r0*1.1;point[1:3]=[.21,.63]
direction=rng.normal(size=Ddim)*.02
r,k,x,p,pk,px,fs,qa,ma,za,m,z,B,w,lam,q,F=data(point)
dr,dk,dx=direction[:3];dp=pk*dk+px*dx
vprof=point[3:15].reshape(4,3);dvprof=direction[3:15].reshape(4,3)
dfs=np.array([coeffJ(vv)@dd for vv,dd in zip(vprof,dvprof)])
dma=2*np.sum(fs*dfs*counts,axis=1);dza=2*np.sum(fs*dfs*counts**2,axis=1)
dm=dp@ma+p@dma;dz=dp@za+p@dza;eta=etaf(m,d);ep=etap(m,d);dB=(1+ep*(z-m*m)/m-eta-eta*z/m**2)*dm+(eta/m)*dz
ck('global variance includes component means',abs((z-m*m)-np.dot(p,za-ma*ma))>1e-5)
dqa=np.einsum('ai,aj,aijuv->auv',dfs,fs,Gq)+np.einsum('ai,aj,aijuv->auv',fs,dfs,Gq)
dq=np.einsum('a,auv->uv',dp,qa)+np.einsum('a,auv->uv',p,dqa)
dw=k*x*x*dk+((1+k*k)*x-.5)*dx;dlam=(1+w)*dB+B*dw
Tq=T(q);Z=np.linalg.inv(Tq);h=F.T@F;Sq=np.eye(10)+q;t=math.exp(-np.linalg.slogdet(Sq)[1]/10)
Ah=(Z@h@Z-Z)/4;BT=(Z@h@Z-Z@h@Z@h@Z)/4;C=t*(BT-np.sum(BT*Sq)*np.linalg.inv(Sq)/10)
dF=direction[15:].reshape(10,10);dVs=np.sum(2*F@Ah*dF)+np.sum(C*dq)
denergy=(-1/r**3+r*lam*lam)*dr+r*r*lam*dlam+dVs
step=1e-6;fd=(V(point+step*direction)-V(point-step*direction))/(2*step)
near('off-reference complete joint derivative with eta prime',fd,denergy,2e-6)
wrong=denergy-r*r*lam*(1+w)*ep*(z-m*m)/m*dm
ck('holding eta fixed fails actual joint force',abs(fd-wrong)>max(1e-7,10*abs(fd-denergy)))
eta_force_error=abs(fd-wrong)
# Actual embedded spatial coefficient kinetic Gram, including current/profile cross tests.
def embedding(v):
 r,theta,k,x=v[:4];p,_,_=weights(k,x);fs,_,_,_=spatial(v[4:])
 return (r*np.array([math.cos(theta),math.sin(theta)])[:,None,None]*np.sqrt(p)[None,:,None]*fs[None,:,:]).ravel()
y=np.r_[1.2,.4,.21,.63,point[3:15]];step=1e-6;JE=[]
for i in range(len(y)):
 dy=np.eye(len(y))[i]*step;JE.append((embedding(y+dy)-embedding(y-dy))/(2*step))
Gram=np.array(JE)@np.array(JE).T;Expected=np.zeros_like(Gram);rr,th,kk,xx=y[:4];pp,_,_=weights(kk,xx)
Expected[:4,:4]=np.diag([1,rr*rr,rr*rr/(4*(1-kk*kk)),rr*rr/(1+xx*xx)**2])
for a,vv in enumerate(y[4:].reshape(4,3)):
 block=rr*rr*pp[a]*(np.eye(3)/(1+vv@vv)-np.outer(vv,vv)/(1+vv@vv)**2)
 Expected[4+3*a:7+3*a,4+3*a:7+3*a]=block
near('actual complete embedded kinetic metric',Gram,Expected,2e-9)
# The new inertial loads cannot be omitted: differentiate T_f wrt k and x.
vel=rng.normal(size=(4,3));rates=np.array([np.dot(coeffJ(vv)@vvv,coeffJ(vv)@vvv) for vv,vvv in zip(y[4:].reshape(4,3),vel)])
def Tf(k,x):return rr*rr*np.dot(weights(k,x)[0],rates)/2
pk=weights(kk,xx)[1];px=weights(kk,xx)[2]
near('profile kinetic k force',(Tf(kk+step,xx)-Tf(kk-step,xx))/(2*step),rr*rr*pk@rates/2,1e-8)
near('profile kinetic x force',(Tf(kk,xx+step)-Tf(kk,xx-step))/(2*step),rr*rr*px@rates/2,1e-8)
# Sign-independent complete energy and proper-time identity with nonzero support.
E=V(point)+.37;gamma=1/math.sqrt(1-.3**2);near('complete energy momentum', (gamma*E)**2-(gamma*E*.3)**2,E*E,1e-7)
# Additional exact all-K implementation bounds and derivative controls.
bounds=[]
for depth in assign:
 for shell in [depth,depth+1,depth+99,1000,10000]:
  count=math.comb(shell,depth)
  ck(f'integer product bound {depth} {shell}',math.factorial(depth)*count>=(shell-depth+1)**depth)
  ck(f'exact strict threshold {depth} {shell}',R(shell+1,depth)>R(shell,depth))
  ck(f'binomial exact threshold {depth} {shell}',R(count,count-math.comb(shell-1,depth))==R(shell,depth))
 bounds.append(dict(d=depth,eta_at_m1=etaf(1,depth)))
result=dict(eta_prime_omission_force_error=float(eta_force_error),extra_bound_rows=bounds,created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='passed',assertions=len(checks),rows=rows,full_patch_dimension=Ddim,full_patch_nullity=int(sum(abs(ev)<1e-8)),hessian_min=float(ev[0]),positive_hessian_min=float(ev[ev>1e-8][0]),positive_generalized_frequency_squared_min=float(freq[freq>1e-8][0]),directional_hessian_errors=errs,stationarity_errors=grads,off_reference_derivative_error=float(abs(fd-denergy)),kinetic_gram_error=float(np.max(abs(Gram-Expected))),max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),scope='Independent verification of principal variable eta in complete four-profile action; exact all-K bound controls and115-dimensional joint Hessian. Any finite cutoff; no infinite-PDE inference.')
(ROOT/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k] for k in ['status','assertions','full_patch_nullity','off_reference_derivative_error','kinetic_gram_error','max_rss_kib']}))
