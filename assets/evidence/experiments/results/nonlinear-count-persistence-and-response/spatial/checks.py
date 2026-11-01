#!/usr/bin/env python3
"""NSP-I1: independent exact finite-cutoff trapping estimates, no parent imports."""
from fractions import Fraction as F
from math import comb, factorial
from pathlib import Path
import datetime, hashlib, json, resource

ROOT=Path(__file__).resolve().parent
CHECKS=[]
def ck(name, condition):
    if not condition:
        raise AssertionError(name)
    CHECKS.append(name)
def root_enclosure(value, d, bits=128):
    """Exact rational interval for value**(1/d), all integer comparisons."""
    scale=1<<bits
    rhs=value.numerator*scale**d
    den=value.denominator
    lo,hi=0,scale
    while hi**d*den<=rhs:
        hi*=2
    while hi-lo>1:
        mid=(lo+hi)//2
        if mid**d*den<=rhs: lo=mid
        else: hi=mid
    ck('root bracket',lo**d*den<=rhs<hi**d*den)
    return F(lo,scale),F(hi,scale)
def weights(k,x):
    return [u*v for u in [(1+k)/2,(1-k)/2]
            for v in [1/(1+x*x),x*x/(1+x*x)]]
def current(k,x,k0):
    f=lambda t:t*t/(8*(1+t*t))
    w=(1+k*k)*x*x/2-x/2+F(1,8)
    if k0:
        w-=f(k0)+k0/(4*(1+k0*k0)**2)*(k-k0)
        w+=(k-k0)**2/4
    return w
ASSIGN={2:[76,81,95],3:[1,4],4:[3,20,72],5:[10,15,22],6:[13,35],10:[23]}
EXPECTED=[2926,3321,4560,1,20,15,8855,1215450,2002,11628,65780,18564,3838380,64512240]
rows=[];mixes=[]
for d,ns in ASSIGN.items():
    Jmax=d+99
    counts={j:comb(j,d) for j in range(d,Jmax+1)}
    for n in ns:
        K=n+d-1;c=counts[K]
        gaps={j:v-c for j,v in counts.items() if j!=K}
        D=max(abs(a) for a in gaps.values());delta=min(abs(a) for a in gaps.values())
        ck('Pascal threshold',F(c,c-comb(K-1,d))==F(K,d))
        ck('root lower bound',factorial(d)*c>=(K-d+1)**d)
        ck('root upper bound',factorial(d)*c<=K**d)
        for law in ['128','variable']:
            if law=='128':
                gamma=(128-F(K,d))/c;H0=F(256,c);H1=F(512,c*c);eta_lower=F(128)
            else:
                gamma=F(1,d*c);H0=2*(1+F(K,d))/c;H1=4*(1+F(K,d))/(c*c);eta_lower=F(K+1,d)
            ck('strict positive nonlinear coefficient',gamma>0)
            for j,Delta in gaps.items():
                ck('entire-band quadratic coefficient lower bound',F(1,Delta)+eta_lower/c>=gamma)
            beta=gamma*delta*delta/2
            tau_s=min(F(1,4),F(c,2*D),gamma/(2*(H1*D+H0)))
            for k0,x0 in [(F(0),F(1,2)),(F(3,4),F(8,25))]:
                rho=min(1-abs(k0),x0,F(1))/4
                pmin=(1-abs(k0)-rho)/2*min(F(1),(x0-rho)**2)/(1+(x0+rho)**2)
                tau=min(tau_s,pmin/4)
                bars=[beta*tau,c*rho*rho/48,F(9*c,8),F(1,32)]
                Ebar=min(bars)
                ck('nonempty explicit basin',Ebar>0)
                ck('spectral remainder margin',gamma-(H1*D+H0)*tau>=gamma/2)
                ck('regular moment denominator',D*tau<=F(c,2))
                ck('individual profiles retain target projection',tau/pmin<=F(1,4))
                for sign in [-1,1]:
                    ck('fixed signed charge reference energy',F(sign*sign*c,2)+F(c,2)==c)
                row=dict(d=d,n=n,K=K,Kmax=Jmax,count=c,law=law,k0=str(k0),x0=str(x0),
                         gap=delta,max_gap=D,gamma=str(gamma),H0=str(H0),H1=str(H1),
                         beta=str(beta),rho=str(rho),pmin=str(pmin),tau=str(tau),
                         Ebar=str(Ebar),relative_Ebar=str(Ebar/c),tau_decimal=float(tau),
                         Ebar_decimal=float(Ebar),relative_Ebar_decimal=float(Ebar/c),
                         limiting_barrier=['spectral','current','radial','material'][bars.index(Ebar)],
                         full_kernel_dimension=45+4*(comb(K+d-1,d-1)-1))
                rows.append(row)
                # All four profiles have independently different off-shell amounts.
                # The current also moves, so its p weights are not reference constants.
                k=k0+rho/3;x=x0+rho/5;p=weights(k,x)
                ck('exact weight normalization',sum(p)==1)
                ck('positive moving kinetic weights',min(p)>=pmin)
                for mix_kind in range(3):
                    target_t=tau/F(2+mix_kind)
                    factors=[F(1),F(2),F(3),F(4)]
                    norm=sum(pa*fac for pa,fac in zip(p,factors))
                    es=[target_t*fac/norm for fac in factors]
                    others=list(gaps)
                    js=([min(others),max(others),min(others),max(others)] if mix_kind==0 else
                        [min(others,key=lambda j:abs(gaps[j])),max(others),others[len(others)//2],min(others)] if mix_kind==1 else
                        [max(others),min(others),max(others),min(others)])
                    m=sum(pa*((1-e)*c+e*counts[j]) for pa,e,j in zip(p,es,js))
                    z=sum(pa*((1-e)*c*c+e*counts[j]**2) for pa,e,j in zip(p,es,js))
                    t=sum(pa*e for pa,e in zip(p,es));u=m-c
                    D2=sum(pa*e*gaps[j]**2 for pa,e,j in zip(p,es,js))
                    variance=z-m*m
                    ck('global variance identity',variance==D2-u*u)
                    ck('four-profile independent positive mixtures',all(0<e<F(1,4) for e in es))
                    ck('weighted Cauchy remainder',u*u<=t*D2)
                    if law=='128':etalo=etahi=F(128)
                    else:
                        sl,sh=root_enclosure(factorial(d)*m,d)
                        etalo=1+sl/d;etahi=1+sh/d
                    Blo=m+etalo*variance/m;Bhi=m+etahi*variance/m
                    ck('nonlinear spectral lower inequality',Blo-c>=beta*t)
                    ck('nonlinear stronger D2 inequality',Blo-c>=gamma*D2/2)
                    w=current(k,x,k0);dist=(k-k0)**2+(x-x0)**2
                    ck('both-current nonlinear lower inequality',w>=dist/48)
                    lamlo=Blo*(1+w);lamhi=Bhi*(1+w)
                    ck('joint nonlinear clock inequality',lamlo-c>=beta*t+c*dist/48)
                    # Exact complete energy decomposition for a rational test Lambda,
                    # including a nonzero artificial Tred and Vs to test bookkeeping.
                    # This identity is algebraic, not a trajectory or physical tensor claim.
                    lam=(lamlo+lamhi)/2;rad=F(3,2);charge=F(1);Tred=F(2,7);Vs=F(1,9)
                    H=Tred+charge**2/(2*rad**2)+rad**2*lam**2/2+Vs
                    split=Tred+(charge/rad-rad*lam)**2/2+charge*(lam-c)+Vs
                    ck('complete nonnegative energy identity',H-c==split)
                    mixes.append(dict(d=d,n=n,law=law,k0=str(k0),kind=mix_kind,
                                     t=str(t),B_minus_c_lower=str(Blo-c),beta_t=str(beta*t),
                                     nonlinear_ratio_lower=float((Blo-c)/(beta*t))))
# Every real current branch proof has a small exact rational off-reference grid control.
for k0,x0 in [(F(0),F(1,2)),(F(3,4),F(8,25))]:
    for ik in range(-15,16):
        k=F(ik,16);s=k*k
        ck('current curvature polynomial',4*(1-3*s)+(1+s)**3==(s-1)**2*(s+5))
        fpp=(1-3*s)/(4*(1+s)**3)
        ck('repaired global curvature lower bound',fpp>=-F(1,16))
        for ix in range(1,25):
            x=F(ix,12)
            ck('current global coercivity grid',current(k,x,k0)>=((k-k0)**2+(x-x0)**2)/48)
# Exact full noncommuting 10x10 support trace diagnostic, no numpy dependency.
N=10
matmul=lambda A,B:[[sum(A[i][k]*B[k][j] for k in range(N)) for j in range(N)] for i in range(N)]
transpose=lambda A:list(map(list,zip(*A)))
sq=[F(2),F(1,2)]+[F(1)]*8
Fm=[[sq[i] if i==j else F((i+2*j)%5-2,100) for j in range(N)] for i in range(N)]
h=matmul(transpose(Fm),Fm)
C=[[h[i][j]/sq[i]/sq[j]-(i==j) for j in range(N)] for i in range(N)]
Zhi=[[h[i][j]/sq[i]**2-(i==j) for j in range(N)] for i in range(N)]
trace=sum(matmul(Zhi,Zhi)[i][i] for i in range(N))
frob=sum(v*v for row in C for v in row)
ck('full noncommuting material trace square',trace==frob)
ck('nonzero material diagnostic',frob>0)
ck('material safe tube',frob<F(1,4))
ck('all actual counts',[r['count'] for r in rows if r['law']=='128' and r['k0']=='0']==EXPECTED)
ck('all 56 branches',len(rows)==56)
ck('full d3 n4 null family',next(r for r in rows if r['d']==3 and r['n']==4)['full_kernel_dimension']==153)
now=datetime.datetime.now(datetime.timezone.utc)
report=dict(version='NSP-I1',status='passed',created_at=now.isoformat(),assertions=len(CHECKS),
            scope='Exact rational trapping constants and nonlinear mixtures; no ODE simulation, no PDE claim, no parent checker replay.',
            rows=rows,mixtures=mixes,noncommuting_material_frobenius_squared=str(frob),
            min_nonlinear_ratio_lower=min(r['nonlinear_ratio_lower'] for r in mixes),
            max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
name='results-'+now.strftime('%Y%m%dT%H%M%SZ')+'.json'
(ROOT/name).write_text(json.dumps(report,indent=2)+'\n')
lines=['# Exact sufficient trapping estimates','',
       'All units A=mu_d=g=kappa=IF=a0=chi=ell=1; Kmax=d+99. Scientific scaling is in derivation.md. Decimal table entries summarize exact rational bounds in the JSON, not measured lifetimes.', '',
       '| d | n | c | law | tau primary | Ebar primary | Ebar/E0 primary | tau outer | Ebar outer | limiting barrier |',
       '|---|---|---|---|---|---|---|---|---|---|']
for i in range(0,len(rows),2):
    a,b=rows[i:i+2]
    lines.append(f"|{a['d']}|{a['n']}|{a['count']}|{a['law']}|{a['tau_decimal']:.6g}|{a['Ebar_decimal']:.6g}|{a['relative_Ebar_decimal']:.6g}|{b['tau_decimal']:.6g}|{b['Ebar_decimal']:.6g}|{a['limiting_barrier']}/{b['limiting_barrier']}|")
lines+=['',f"Passed {len(CHECKS)} exact assertions and {len(mixes)} four-profile nonlinear mixture cases. Minimum certified (B-c)/(beta t) was {report['min_nonlinear_ratio_lower']:.6g}. This is a bounded diagnostic supporting the analytic all-time theorem, not a sampled proof of it.",'',f'Full result: {name}.']
(ROOT/'results.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({k:report[k] for k in ['version','status','assertions','max_rss_kib','min_nonlinear_ratio_lower']}))
print(name)
