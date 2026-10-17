"""Exact full signed spectral bound. Uses integer Sturm endpoint checks only."""
from fractions import Fraction as F
from math import factorial, comb, isqrt, ceil
from pathlib import Path
import json, hashlib
P=Path(__file__).parent
import argparse
from datetime import datetime, timezone
parser=argparse.ArgumentParser()
parser.add_argument('--input',type=Path,default=P/'results/free-count-stability-20261014.json')
src=parser.parse_args().input
data=json.loads(src.read_text()); s=data['spectral']; M=s['M']; b=s['b']; r=F(1,100000)
checks=0
def check(x):
 global checks
 assert x
 checks+=1
def iv(x):return tuple(map(F,x))
def sturm(d,lam,f):
 z=(f*factorial(d)/M**(d-1))**2
 p,q=z.numerator,z.denominator
 a=[lam+j*(d+1-j) for j in range(1,d+1)]
 diag=[x+M*M for x in a[:-1]]+[a[-1]]
 old,cur=1,diag[0]*q-p; signs=[1]
 if cur:signs.append(1 if cur>0 else -1)
 for j in range(1,d):
  old,cur=cur,(diag[j]*q-p)*cur-M*M*a[j]*q*q*old
  if cur:signs.append(1 if cur>0 else -1)
 check(cur!=0)
 return sum(x!=y for x,y in zip(signs,signs[1:]))
roots=0
for dd,dep in s['depths'].items():
 d=int(dd); cutoff=dep['cutoff_lambda']; fam={0:[dict(grade=0,degree=0,h=1,signed_chain_multiplicity=1)]}
 for k in range(d):
  n=1
  while True:
   lam=(n+k)*(n+d+1-k)*(1 if k==0 else b*b)
   if lam>=cutoff:break
   h=comb(n+d-1,d-k-1)*comb(n+k-1,k)
   fam.setdefault(lam,[]).append(dict(grade=k,degree=n,h=h,signed_chain_multiplicity=2*h));n+=1
 check({x['lam']:x['families'] for x in dep['families']}==fam)
 for f in dep['families']+[dict(lam=cutoff,frequency=dep['cutoff_frequency'])]:
  lo,hi=iv(f['frequency']);check(0<lo<hi);check(sturm(d,f['lam'],lo)==0);check(sturm(d,f['lam'],hi)==1);roots+=1
 maxweak=isqrt(cutoff+max(j*(d+1-j) for j in range(1,d+1)))+1
 heavy=F((M-maxweak)*M**(d-1),factorial(d));check(heavy==F(dep['heavy_normalized_lower']))
 bg=max(F(x['target_frequency'][1]) for x in s['rows'] if x['depth']==d)
 check(heavy>2*bg);check(F(dep['cutoff_frequency'][0])>2*bg)
rows=[]
for row in s['rows']:
 d,N=row['depth'],row['degree']; target=row['target_lambda']; omlo,omhi=iv(row['target_frequency']);dep=s['depths'][str(d)]
 check(next(x['frequency'] for x in dep['families'] if x['lam']==target)==row['target_frequency'])
 candidates=[];signed=0
 for f in dep['families']:
  lam=f['lam'];lo,hi=iv(f['frequency'])
  if lam==target:continue # BOTH full signed eigenspaces, including all multiplicities
  check(hi<omlo or lo>omhi)
  for sign in (-1,1):
   nl,nh=(lo,hi) if sign==1 else (-hi,-lo);signed+=1
   if nl>omhi or nh<-omhi:continue # ratio (nu+omega)/(nu-omega) positive
   check(-omlo<nl<=nh<omlo)
   # -(nu+omega)/(nu-omega)=(omega+nu)/(omega-nu).
   # Monotone increasing in nu. For nu positive use omega lower,
   # for nu negative use omega upper (derivative -2nu/(omega-nu)^2).
   om=omlo if nh>=0 else omhi
   candidates.append((F(om+nh,om-nh),lam,sign))
 beta,near,sign=max(candidates,default=(F(0),None,None))
 B=ceil(beta);w=1-r*r*(1+B);a=w/(r*r)-1
 check(w>r*r)
 rows.append(dict(label=row['label'],depth=d,degree=N,target_frequency=row['target_frequency'],nearest_lower_lambda=near,nearest_sign=sign,beta_upper=str(beta),beta_integer_upper=B,w_lower=str(w),inertia_factor=str(a),signed_branches_checked=signed))
B=max(x['beta_integer_upper'] for x in rows);w=1-r*r*(1+B);A=w/r**2-1
for row in rows:
 d,N=row['depth'],row['degree'];lam=N*(N+d+1);g=F(lam,d+2)
 row['angular_norm_ratio']=str(g)
 if N:
  row['common_inertia_over_profile_norm_lower']=str(A*g)
  row['omega_ratio_squared_over_epsilon_upper']=str(2/(A*g))
  # pi < 22/7 gives a wholly rational squared per-cycle bound.
  q=4*F(22,7)**2*2/(A*g)
  row['cycle_angle_squared_over_epsilon_upper']=str(q)
  c=F(isqrt(q.numerator*10**16//q.denominator)+1,10**8)
  check(c*c>q)
  row['cycle_angle_over_sqrt_epsilon_rational_upper']=str(c)
 else:row['orientation']='absent; no inverse or angular speed bound'
report=dict(status='passed',checks=checks,root_intervals_rechecked=roots,M=M,b=b,r=str(r),common_beta_integer_upper=B,common_w_lower=str(w),common_inertia_factor=str(A),rows=rows,sources={'experiments/results/free-count-stability-20261014.json':hashlib.sha256(src.read_bytes()).hexdigest(),'experiments/check-orientation-inertia.py':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
(P/('orientation-inertia-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'.json')).write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ('rows','sources')},indent=2))
for x in rows:print(x['label'],x['beta_integer_upper'],x.get('common_inertia_over_profile_norm_lower'),x.get('cycle_angle_squared_over_epsilon_upper'))
