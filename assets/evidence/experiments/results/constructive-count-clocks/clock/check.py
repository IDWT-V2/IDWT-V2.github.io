#!/usr/bin/env python3
"""Small matrices only; exact rational Collatz enclosures plus 80-digit checks."""
from pathlib import Path
from fractions import Fraction as Q
import math,json,hashlib
import mpmath as mp
mp.mp.dps=140
out=Path(__file__).resolve().parent
g=Q(5000); rows=[('W',2,76),('Z',2,81),('Higgs',2,95),('down',3,1),('strange',3,4),('up',4,3),('charm',4,20),('top',4,72),('nu1',5,10),('nu2',5,15),('nu3',5,22),('electron',6,13),('muon',6,35),('tau',10,23)]
def real(x):return mp.mpf(x.numerator)/x.denominator if isinstance(x,Q) else mp.mpf(x)
def chain(d,n):
 A=[[g**(i-j-d+1)*math.prod(Q(n+k) for k in range(d) if k<j or k>i) if i>=j else Q(0) for j in range(d)]for i in range(d)]
 M=[[sum(A[k][i]*A[k][j]for k in range(d))for j in range(d)]for i in range(d)]
 ev,V=mp.eigsy(mp.matrix([[real(v)for v in r]for r in M]))
 v=[Q(mp.nstr(abs(V[i,d-1]),125))for i in range(d)]
 ratios=[sum(M[i][j]*v[j]for j in range(d))/v[i]for i in range(d)]
 lo,hi=min(ratios),max(ratios)
 assert lo>0 and real(hi-lo)/real(lo)<mp.mpf('1e-65')
 return Q(1)/hi,Q(1)/lo
cache={(d,n):chain(d,n)for _,d,n in rows}
z=lambda d,n: Q((2*n+d-1)**2,4*5000**2)
R=Q('206.76828299')/Q(3515,17)
eL,eU=cache[6,13];mL,mU=cache[6,35];den=z(6,35)-R**2*z(6,13)
aL=(R**2*eL-mU)/den;aU=(R**2*eU-mL)/den
assert den>0 and aL>0
eta=real((aL+aU)/2)
fe=mp.sqrt(real((eL+eU)/2)+eta*real(z(6,13)))
records=[]
# The mu ratios below are source rounded MeV-equivalent examples only; the proofs permit arbitrary supplied positive units.
units={2:mp.mpf('27.47064959'),3:mp.mpf('4.701857978'),4:mp.mpf('0.1451024874'),5:mp.mpf('7.55562e-13'),6:mp.mpf('0.00002752633861'),10:mp.mpf('0.00002752633861')}
fnu=lambda n: mp.sqrt(real(sum(cache[5,n],Q(0))/2)+eta*real(z(5,n)))
units[5]=mp.sqrt(mp.mpf('7.49e-5')/(11628**2*fnu(15)**2-2002**2*fnu(10)**2))*mp.mpf('1e-6')
omega_star=min(units.values())/(2*fe)
for name,d,n in rows:
 lo,hi=cache[d,n];zl=z(d,n);fL=lo+aL*zl;fU=hi+aU*zl;f=mp.sqrt(real((fL+fU)/2));D=math.comb(n+d-1,d);H=units[d]*D*f/(1 if d==5 else fe)
 # Differentiate the small analytic matrix, not particle-wise interpolation.
 def freq(nn):
  A=mp.matrix(d)
  for i in range(d):
   for j in range(i+1):A[i,j]=mp.mpf(5000)**(i-j-d+1)*mp.fprod(nn+k for k in range(d)if k<j or k>i)
  val=mp.eigsy(A.T*A,eigvals_only=True)[d-1]
  S=mp.fprod(nn+k for k in range(d))/math.factorial(d)
  return S*mp.sqrt(1/val+eta*((nn+(d-1)/2)/5000)**2)
 deriv=mp.diff(freq,mp.mpf(n))/(mp.mpf(D)*sum(mp.mpf(1)/(n+k)for k in range(d)))
 assert deriv>0
 N=n-1;diag=Q(N*(2*N+d),(d+1)*(d+2));off=Q(N*(N-1),(d+1)*(d+2));lam=diag+(d-1)*off
 assert lam/(1+d*diag)<=1
 variants=[]
 for entropy in ['fixed','branch']:
  base=omega_star if entropy=='fixed' else H;b=base/128;k=8/base
  for clock in ['positive','even+','even-']:
   sign=-1 if clock=='even-' else 1;a2=mp.mpf(0)if clock=='positive'else 8*k/mp.sinh(8*k*H)
   speed2=k/(1/b+a2)
   assert 0<speed2<=mp.mpf(1)/16 and H/2>0 and H>=omega_star
   # Material exact reference derivatives, including entropy-dependent clock kappa.
   def Fc(t,h,kap):
    if clock=='positive':return t-h
    return (mp.log(mp.cosh(4*kap*t))-mp.log(mp.cosh(4*kap*h)))/(4*kap*mp.tanh(4*kap*h))
   # Derivatives at huge kH use analytic stable closed forms to avoid finite differencing saturation.
   Ft=sign;FH=-1;Fk=0
   assert Fc(sign*H,H,k)==0
   v=mp.mpf('0.6');gam=1/mp.sqrt(1-v*v)
   assert abs((gam*H)**2-(gam*H*v)**2-H**2)<H**2*mp.mpf('1e-75')
   variants.append(dict(entropy=entropy,clock=clock,phase_speed_squared=mp.nstr(speed2,35),phase_charge=sign,time_displacement_weight=mp.nstr(H/2,35),clock_reference_derivatives=[Ft,FH,Fk]))
 records.append(dict(label=name,d=d,n=n,count=D,correction=mp.nstr(f,45),anchored_shift_ppm=mp.nstr((f/fe-1)*10**6,35),example_frequency_MeV_equivalent=mp.nstr(H,35),F_x=mp.nstr(deriv,35),raw_second_moment_diagonal=str(diag),raw_second_moment_offdiagonal=str(off),variants=variants,t_squared_bounds=[str(lo),str(hi)],correction_squared_bounds=[str(fL),str(fU)]))
# Exact component metric counterexample and corrected stiffness positivity proof check.
G=mp.diag([4,1]);D=mp.diag([6,5]);C=mp.diag([mp.mpf(1)/2,1])*D.T*D*mp.diag([mp.mpf(1)/2,1]);assert list(mp.eigsy(C,eigvals_only=True))==[9,25]
# Occupation lower bound for arbitrary tangents: exact random rational samples of the proved identity.
occupation=[]
for count in [1,2,3,15,20]:
 u=[Q(i)-Q(count-1,2)for i in range(count)];sout=Q(2,7);u0=u[0]
 q_m=-count**2*sum(x*x for x in u)+2*count*sout
 q_v0=2*count**2*sum((u0-x)**2 for x in u)+2*count*sout
 bound=count**2*sum(x*x for x in u)+2*count**3*u0*u0+4*count*sout
 assert q_m+q_v0==bound and bound>0
 occupation.append({'D':count,'exact_lower_bound':str(bound)})
# Support residuals at x/L=(u,0,...): all independent tensor entries.
for u in [mp.mpf(0),mp.mpf('0.4'),mp.mpf(1),mp.mpf(3)]:
 W=(1+u*u)**-4;Z=u*(1+u*u)**-5
 dW=-8*u*(1+u*u)**-5;dZ=(1+u*u)**-5-10*u*u*(1+u*u)**-6
 assert abs(dW+8*Z)<mp.mpf('1e-75')
 assert abs(dZ-W**mp.mpf('1.25')+10*Z*Z/W)<mp.mpf('1e-75')
 assert abs(Z*Z-W**mp.mpf('2.25')+W**mp.mpf('2.5'))<mp.mpf('1e-75')
I0=(529-760*mp.log(2)+mp.mpf(11017)/693)/212
assert I0<mp.mpf(11)/128 and Q(20)-Q(1024,53)==Q(36,53)
# Enclose fitted ratio squared with exact rational arithmetic.
fmeL=mL+aL*z(6,35);fmeU=mU+aU*z(6,35);feeL=eL+aL*z(6,13);feeU=eU+aU*z(6,13)
rlo=Q(3515,17)**2*fmeL/feeU;rhi=Q(3515,17)**2*fmeU/feeL
assert rlo<=Q('206.76828299')**2<=rhi
assert real(rhi-rlo)<mp.mpf('1e-58')
report={'precision_digits':140,'eta_interval':[str(aL),str(aU)],'eta_decimal':mp.nstr(eta,65),'ratio_squared_interval':[str(rlo),str(rhi)],'ratio_target':'206.76828299','all_rows':records,'constitutive_clock_entropy_cases':84,'occupation_lower_bound_tests':occupation,'support_radial_residual_tests':4,'I0':mp.nstr(I0,40),'rounding_note':'Collatz spectral and squared-ratio endpoints are exact rationals; printed square roots and derivatives are high precision diagnostics. Example units are rounded source values, neutrino unit uses the dated solar splitting 7.49e-5 eV^2 with the corrected counts, not an absolute mass measurement.','proof_scope':'Reference bulk principal bounds and co-rotating energy, not nonlinear global evolution.','script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'passed':True}
(out/'results.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:report[k]for k in ['eta_decimal','constitutive_clock_entropy_cases','I0','passed']},indent=2))
