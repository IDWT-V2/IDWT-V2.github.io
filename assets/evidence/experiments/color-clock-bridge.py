#!/usr/bin/env python3
"""Bounded color/spinor bridge checks; no DB writes, simulations or fits."""
import math,json,hashlib,pathlib,datetime
from fractions import Fraction as F
import numpy as np
from scipy.integrate import quad
from scipy.special import kv,gamma
P=pathlib.Path(__file__).resolve().parent
D=[2,3,4,5,6,10]
particles=[('W',2,76),('Z',2,81),('Higgs',2,95),('down',3,1),('strange',3,4),('up',4,3),('charm',4,20),('top',4,72),('nu1',5,10),('nu2',5,15),('nu3',5,22),('electron',6,13),('muon',6,35),('tau',10,23)]
checks=[]
def ck(name,err,tol=1e-11):
 checks.append({'name':name,'error':float(err),'tolerance':tol,'pass':bool(err<tol)})
def radial(d,q,p=0):
 a=(d+p)/2;b=q*(d-1)-a
 return math.inf if b<=0 else math.pi**(d/2)*gamma(a)/gamma(d/2)*gamma(b)/gamma(q*(d-1))
def form(d,z):
 nu=(d-2)/2
 return 1. if z==0 else 2**(1-nu)*z**nu*kv(nu,z)/gamma(nu)
# Full four-dimensional Clifford tangent, with color generators kept separate.
s1=np.array([[0,1],[1,0]],complex);s2=np.array([[0,-1j],[1j,0]]);s3=np.diag([1,-1]);I=np.eye(2)
gam=[np.kron(s1,I),np.kron(s2,I),np.kron(s3,s1),np.kron(s3,s2)]
K=[1j*gam[i]@gam[j] for i in range(4) for j in range(i+1,4)]
Ts=[]
for i in range(3):
 for j in range(i+1,3):
  t=np.zeros((3,3),complex);t[i,j]=t[j,i]=.5;Ts.append(t)
  t=np.zeros((3,3),complex);t[i,j]=-.5j;t[j,i]=.5j;Ts.append(t)
Ts += [np.diag([1,-1,0])/2,np.diag([1,1,-2])/(2*math.sqrt(3))]
seed=np.array([1,0,0,0],complex);color=np.array([1,0,0],complex)
cc=sum(float(np.vdot(color,t@color).real)**2 for t in Ts)
ck('pure color adjoint expectation norm',abs(cc-1/3))
rng=np.random.default_rng(20261018)
for d in [3,4]:
 for j in range(16):
  x=rng.normal(size=d);t=float(x@x);v=(np.eye(4)+1j*sum(x[i]*gam[i] for i in range(d)))@seed
  actual=cc*sum(float(np.vdot(v,k@v).real)**2 for k in K)/(1+t)**(2*d)
  target=2/3*(1+t*t)/(1+t)**(2*d)
  ck(f'full Clifford bubble current d{d} point{j}',abs(actual-target))
# General seed polarizations: exact angular second moments and Clifford identity.
chir=gam[0]@gam[1]@gam[2]@gam[3]
polarization_controls=[]
for j in range(12):
 chi=rng.normal(size=4)+1j*rng.normal(size=4);chi/=np.linalg.norm(chi)
 delta=float(np.vdot(chi,chir@chi).real)
 v=np.array([float(np.vdot(chi,1j*chir@g@chi).real) for g in gam])
 ck(f'Spin5 bilinear unit identity seed{j}',abs(delta*delta+float(v@v)-1))
 for d in [3,4]:
  predicted=(5-v[3]**2)/12 if d==3 else (25+delta**2)/63
  alpha=d/2;beta=3*d/2-2;total=alpha+beta
  # Exact spherical averaging needs only antipodal coordinate axes here.
  def angular(r):
   vals=[]
   for axis in range(d):
    for sign in [-1,1]:
     w=(np.eye(4)+1j*sign*r*gam[axis])@chi
     vals.append(cc*sum(float(np.vdot(w,k@w).real)**2 for k in K))
   return sum(vals)/len(vals)
  area=2*math.pi**(d/2)/gamma(d/2)
  integral=quad(lambda r:r**(d-1)*angular(r)/(1+r*r)**(2*d),0,np.inf,epsabs=2e-11)[0]*area
  ratio=integral/radial(d,2)
  ck(f'general seed integrated current d{d} seed{j}',abs(ratio-predicted),2e-10)
  polarization_controls.append({'seed':j,'d':d,'delta_squared':delta**2,'normal_bilinear_squared':v[3]**2,'predicted_I2_coefficient':predicted,'actual':ratio})
# Analytic beta integrals independently checked by radial quadrature.
rows=[]
for d in D:
 alpha=F(d,2);beta=F(3*d,2)-2
 R=(alpha*(alpha+1)+beta*(beta+1))/((alpha+beta)*(alpha+beta+1))
 gate=int(math.ceil(d/2)%2==0)
 norm=radial(d,1);quartic=radial(d,2)
 area=2*math.pi**(d/2)/gamma(d/2)
 integral=quad(lambda r:r**(d-1)*(1+r*r)**(-2*(d-1)),0,np.inf,epsabs=1e-12)[0]*area
 ck(f'quartic integral d{d}',abs(integral-quartic))
 row={'d':d,'schur':{2:'C',3:'H',4:'H',5:'H',6:'C',10:'C'}[d], 'minimal_real_rank':{2:2,3:4,4:4,5:8,6:8,10:32}[d], 'skew_rank':3 if d in [3,4,5] else 1,'middle_rank':math.comb(2*math.ceil(d/2)-1,math.ceil(d/2)-1),'hodge_gate':gate,'normal_commutant_dimension':8 if d<=4 else (3 if d<=6 else 0),'finite_Q':d>2,'finite_second_moment':d>4,'second_moment_over_a2':str(F(d,d-4)) if d>4 else None,'Q_over_A2_ad':None if d==2 else norm,'I2_over_A4_ad':quartic,'chirality_radial_factor':str(R),'color_contact_over_A4_ad':(2/3*float(R)*quartic if d in [3,4] else 0)}
 if d in [3,4]:
  integ=quad(lambda r:r**(d-1)*(1+r**4)*(1+r*r)**(-2*d),0,np.inf,epsabs=1e-12)[0]*area*2/3
  ck(f'actual color current integral d{d}',abs(integ-row['color_contact_over_A4_ad']))
  row['color_contact_over_Q2_a_minus_d']=row['color_contact_over_A4_ad']/norm**2
 if d>2:
  row['form_factors']={str(z):form(d,z) for z in [.01,.1,1,3,10]}
  # Schwinger parameter formula independently integrates the Fourier kernel.
  nu=(d-2)/2
  for z in [.1,1,3]:
   integral=quad(lambda t:t**(nu-1)*math.exp(-t-z*z/(4*t)),0,np.inf,epsabs=2e-12)[0]/gamma(nu)
   ck(f'form factor integral d{d} z{z}',abs(integral-form(d,z)),2e-10)
 rows.append(row)
labels=[]
for name,d,n in particles:
 S=math.comb(n+d-1,d);H=math.comb(n+d-2,d-1)
 # This is the published width/current branch, not an inferred particle lifetime.
 inertia_reduced=F(n*n,2*(n+2))
 # ν+ν-=λ/(I/N); k=n-1. Scale-free diagnostic c²b/μ=1, λ=12.
 u=24*(n+2)/(S*n*n)
 low=u/(math.sqrt(36+u)+6)
 labels.append({'name':name,'d':d,'n':n,'S':S,'H':H,'H_over_S':str(F(H,S)), 'I_over_Nomega_times_c2b':str(inertia_reduced),'nu_minus_unit_scale_lambda12':low if d<10 else None,'has_cross_plane_in_source_family':d<10,'color_current_S2_coefficient_over_gamma_q02_a_minus_d':next(x for x in rows if x['d']==d)['color_contact_over_A4_ad']/(next(x for x in rows if x['d']==d)['Q_over_A2_ad'] or 1)**2*S*S if d in [3,4] else 0})
# Exact modular rank certificate for the minimal joint Spin(3) x SU(3) carrier.
def realify(A):
 return np.rint(np.block([[A.real,-A.imag],[A.imag,A.real]])).astype(np.int64)
spin_gens=[realify(np.kron(1j*s,np.eye(3))) for s in [s1,s2,s3]]
color_anti=[]
for i in range(3):
 for j in range(i+1,3):
  a=np.zeros((3,3),complex);a[i,j]=1;a[j,i]=-1;color_anti.append(a)
  a=np.zeros((3,3),complex);a[i,j]=a[j,i]=1j;color_anti.append(a)
color_anti += [1j*np.diag([1,-1,0]),1j*np.diag([1,1,-2])]
joint=spin_gens+[realify(np.kron(np.eye(2),a)) for a in color_anti]
basis=[]
for i in range(12):
 for j in range(12):
  e=np.zeros((12,12),dtype=np.int64);e[i,j]=1;basis.append(e)
def constraints(bs):
 return np.array([np.concatenate([(e@g-g@e).reshape(-1) for g in joint]) for e in bs],dtype=np.int64).T
def modular_rank(A,p=1000003):
 a=A.copy()%p;row=0
 for col in range(a.shape[1]):
  nz=np.flatnonzero(a[row:,col])
  if not len(nz):continue
  pivot=row+int(nz[0]);a[[row,pivot]]=a[[pivot,row]]
  a[row]=(a[row]*pow(int(a[row,col]),p-2,p))%p
  nzrows=np.flatnonzero(a[:,col]);nzrows=nzrows[nzrows!=row]
  a[nzrows]=(a[nzrows]-a[nzrows,col,None]*a[row])%p
  row+=1
  if row==a.shape[0]:break
 return row
rank=modular_rank(constraints(basis))
J12=realify(1j*np.eye(6));I12=np.eye(12,dtype=np.int64)
ck('joint complex structure commutes exactly',max(np.max(abs(J12@g-g@J12)) for g in joint))
ck('joint full commutant modular rank142',abs(rank-142))
skew=[]
for i in range(12):
 for j in range(i+1,12):
  a=np.zeros((12,12),dtype=np.int64);a[i,j]=1;a[j,i]=-1;skew.append(a)
skewrank=modular_rank(constraints(skew))
ck('joint skew commutant modular rank65',abs(skewrank-65))
# The displayed I,J kernel basis plus these rank lower bounds certify rational nullities2,1.
joint_clock={'real_carrier_rank':12,'generators':len(joint),'modulus':1000003,'commutator_rank_mod_p':rank,'commutant_dimension':144-rank,'skew_constraint_rank_mod_p':skewrank,'skew_dimension':66-skewrank,'exact_kernel_basis':'I12 and J12; J12 alone for skew part'}
# The two discarded quaternionic clock axes have a definite color symmetry defect.
def real_matrix(A):
 return np.block([[A.real,-A.imag],[A.imag,A.real]])
eps=1j*s2
A=np.kron(eps,np.eye(3))
Janti=np.block([[A.real,A.imag],[A.imag,-A.real]])
Jphase=real_matrix(1j*np.eye(6));Janti2=Jphase@Janti
clocks=[Jphase,Janti,Janti2]
colors=[real_matrix(np.kron(np.eye(2),1j*t)) for t in Ts]
defects=np.zeros((3,3))
for i,a in enumerate(clocks):
 for j,b in enumerate(clocks):
  defects[i,j]=sum(np.sum((a@g-g@a)*(b@g-g@b)) for g in colors)/12
ck('color quaternionic clock-axis defect',np.max(abs(defects-np.diag([0,10/3,10/3]))))
joint_clock['normalized_axis_defect']=defects.tolist()
# Explicit real-type carrier times complex color always has a clock.
J=np.kron(np.eye(3),np.array([[0,-1],[1,0]]))
ck('real six-dimensional color clock antisymmetric',np.max(abs(J.T+J)))
ck('real six-dimensional color clock nondegenerate',np.max(abs(J@J+np.eye(6))))
out={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'script_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'depth_rows':rows,'particle_rows':labels,'polarization_controls':polarization_controls,'joint_spin_color_clock':joint_clock,'checks':checks,'all_passed':all(c['pass'] for c in checks),'fitted_constants':[],'empirical_lifetimes_used':False}
(P/'results.json').write_text(json.dumps(out,indent=2))
print(json.dumps({'checks':len(checks),'all_passed':out['all_passed'],'max_error':max(c['error'] for c in checks),'depth_rows':rows,'particle_rows':labels},indent=2))
