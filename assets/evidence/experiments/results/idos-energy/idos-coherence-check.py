"""Light exact and finite-dimensional numerical coherence checks; no DB mutation."""
from fractions import Fraction as F
from math import comb,sqrt
import json,hashlib,pathlib,datetime
import numpy as np
from scipy.integrate import solve_ivp
root=pathlib.Path(__file__).resolve().parent
checks=0
rows=[]
def check(x):
 global checks
 assert x
 checks+=1
for d in (2,3,4,5,6,10):
 for N in range(35):
  D=comb(N+d,d)
  check(sum(comb(k+d-1,d-1) for k in range(N+1))==D)
  check(sum((F(comb(k+d-1,d-1),comb(k+d,d)) for k in range(N+1)),F())==d*sum((F(1,k) for k in range(d,N+d+1)),F()))
 C=F(3*(d+3),d+5);M6=F(15*(d+3)**2,(d+5)*(d+7))
 J=F(10);Delta=F(1);g=F(1);nu0=F(1);nu1=F(2)
 I1=(Delta+2*g*J)/(g*(5-C));I0=J-I1
 Om=nu0+g*(I0+3*I1)
 check(0<I1<J)
 check(Om==nu1+g*(C*I1+3*I0))
 H2=nu0*I0+nu1*I1;H4=g/2*(I0**2+C*I1**2+6*I0*I1)
 check(Om*J==H2+2*H4)
 nu_rel2=4*g*g*(5-C)*I0*I1
 check(nu_rel2>0)
 leakage2=g*g*(9*I0*I1**2*(C-1)+I1**3*(M6-C*C))
 check(leakage2>0)
 cf=float(C)
 def rhs(t,z):
  a,b=z
  return -1j*np.array([a+(abs(a)**2+2*abs(b)**2)*a+b*b*a.conjugate(),2*b+(cf*abs(b)**2+2*abs(a)**2)*b+a*a*b.conjugate()])
 def energy(z):
  a,b=z
  return abs(a)**2+2*abs(b)**2+0.5*(abs(a)**4+cf*abs(b)**4+4*abs(a*b)**2+2*np.real((a.conjugate()*b)**2))
 z0=np.array([sqrt(float(I0)),sqrt(float(I1))*np.exp(-0.015j)])
 sol=solve_ivp(rhs,(0,20),z0,method='DOP853',rtol=2e-11,atol=2e-12,max_step=0.015)
 check(sol.success)
 norms=np.sum(abs(sol.y)**2,axis=0);ens=np.array([energy(z) for z in sol.y.T]);angles=np.angle(sol.y[1]*sol.y[0].conjugate())
 nd=float(np.max(abs(norms-10)));ed=float(np.max(abs(ens-ens[0])));ph=float(np.max(abs(angles)));ir=[float(np.min(abs(sol.y[1])**2)),float(np.max(abs(sol.y[1])**2))]
 check(nd<1e-7);check(ed<2e-6);check(ph<0.016)
 rows.append(dict(depth=d,C=str(C),threshold_J=str(Delta/(g*(3-C))),I0=str(I0),I1=str(I1),Omega=str(Om),H2=str(H2),H4=str(H4),H=str(H2+H4),relative_frequency_squared=str(nu_rel2),unprojected_cubic_leakage_squared=str(leakage2),numerical={'time_end':20,'initial_relative_phase':0.015,'max_abs_relative_phase':ph,'I1_range':ir,'norm_drift':nd,'energy_drift':ed,'steps':len(sol.t)}))
# Ground+first-shell beats recur exactly; at every depth D_1-D_0=d.
for d in (2,3,4,5,6,10): check(comb(1+d,d)-1==d)
# Forced carrier zero-data formula and resonant finite-time envelope checked analytically in report.
out=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),checks=checks,source_script_sha256=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),rows=rows)
(root/'idos-coherence-checks.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
