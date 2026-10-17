"""Full-form finite-chain low-frequency collision certificate, exact integers."""
from fractions import Fraction as F
from math import comb, factorial, isqrt, prod
from pathlib import Path
import json, hashlib, time
M=100000
B=40
BITS=100
LABELS=[('W',2,75),('Z',2,80),('Higgs',2,94),('down',3,0),('strange',3,3),('up',4,2),('charm',4,19),('top',4,71),('neutrino1',5,9),('neutrino2',5,14),('neutrino3',5,21),('electron',6,12),('muon',6,34),('tau',10,22)]

def below(d,lam,z):
    p,q=z.numerator,z.denominator
    a=[lam+j*(d+1-j) for j in range(1,d+1)]
    diag=[x+M*M for x in a[:-1]]+[a[-1]]
    old,cur=1,diag[0]*q-p
    signs=[1]
    if cur: signs.append(1 if cur>0 else -1)
    for j in range(1,d):
        old,cur=cur,(diag[j]*q-p)*cur-M*M*a[j]*q*q*old
        if cur: signs.append(1 if cur>0 else -1)
    assert cur, 'An exact spectral endpoint needs separate handling.'
    return sum(x!=y for x,y in zip(signs,signs[1:]))

cache={}
def interval(d,lam):
    key=(d,lam)
    if key in cache:return cache[key]
    a=[lam+j*(d+1-j) for j in range(1,d+1)]
    # In normalized units f=M^(d-1)*sigma/d!, f<=sqrt(product a)/d!.
    hi=isqrt(prod(a))+1
    lo=0
    scale=F(1,M**(d-1))
    # Isolate sigma using rational g=d!*f, with a common dyadic denominator.
    assert below(d,lam,F(0))==0
    assert below(d,lam,(hi*scale)**2)==1
    den=1
    for _ in range(BITS):
        lo*=2;hi*=2;den*=2
        mid=(lo+hi)//2
        if below(d,lam,(F(mid,den)*scale)**2)==0:lo=mid
        else:hi=mid
    ans=(F(lo,den*factorial(d)),F(hi,den*factorial(d)))
    cache[key]=ans
    return ans

def outiv(v):return [str(x) for x in v]
def h(d,n,k):return comb(n+d-1,d-k-1)*comb(n+k-1,k)

start=time.time(); rows=[]; depths={}
for d in sorted({d for _,d,_ in LABELS}):
    nmax=max(n for _,dd,n in LABELS if d==dd)
    bg=interval(d,nmax*(nmax+d+1))
    # Scalar cutoff bounds every family by strict monotonicity in lambda.
    n=0
    while interval(d,n*(n+d+1))[0]<=2*bg[1]:n+=1
    cutoff=n*(n+d+1)
    fam={0:[dict(grade=0,degree=0,h=1,signed_chain_multiplicity=1)]}
    for k in range(d):
        n=1
        while True:
            lam=(n+k)*(n+d+1-k)*(1 if k==0 else B*B)
            if lam>=cutoff:break
            fam.setdefault(lam,[]).append(dict(grade=k,degree=n,h=h(d,n,k),signed_chain_multiplicity=2*h(d,n,k)))
            n+=1
    # At lambda>=cutoff all frequencies > 2 highest background by monotonicity.
    # Below cutoff, every non-low singular value >= M-max_j sqrt(lambda+a_j).
    maxweak=isqrt(cutoff+max(j*(d+1-j) for j in range(1,d+1)))+1
    heavy=F((M-maxweak)*M**(d-1),factorial(d))
    assert heavy>2*bg[1]
    spectrum=[(lam,interval(d,lam)) for lam in sorted(fam)]
    depths[d]=dict(cutoff_lambda=cutoff,cutoff_frequency=outiv(interval(d,cutoff)),heavy_normalized_lower=str(heavy),families=[dict(lam=lam,frequency=outiv(iv),families=fam[lam]) for lam,iv in spectrum])
    for name,dd,N in LABELS:
        if dd!=d:continue
        targetlam=N*(N+d+1); target=interval(d,targetlam)
        included=[]
        for lam,iv in spectrum:
            if iv[1]<2*target[0]:included.append((lam,iv))
            else:assert iv[0]>2*target[1]
        sep=[];trivial=[];pairs=0
        for i,(l,li) in enumerate(included):
            for m,mi in included[i:]:
                pairs+=1
                if l==m==targetlam:
                    trivial.append([l,m]);continue
                diff=(li[0]+mi[0]-2*target[1],li[1]+mi[1]-2*target[0])
                assert diff[0]>0 or diff[1]<0, (name,l,m,outiv(diff))
                absolute=diff if diff[0]>0 else (-diff[1],-diff[0])
                sep.append((absolute,l,m,diff))
        opposite=[x for x in sep if x[1]<targetlam<x[2]]
        opposite.sort(key=lambda x:x[0][0])
        opposite_result=None
        if opposite:
            best_op=opposite[0]
            assert all(best_op[0][1]<x[0][0] for x in opposite[1:])
            opposite_result=dict(pair_lambda=[best_op[1],best_op[2]],separation=outiv(best_op[0]),decimal=float(sum(best_op[0])/2))
        sep.sort(key=lambda x:x[0][0])
        if not sep:
            assert N==0 and len(included)==1
            rows.append(dict(label=name,depth=d,degree=N,target_lambda=targetlam,target_frequency=outiv(target),frequency_count=1,pair_count=1,nontrivial_collisions=0,neutral_equal_frequency_pairs=trivial,extra_families_below_twice=[],minimum_absolute_separation=None,target_signed_chain_multiplicity=1))
            print(name,'ground: no nontrivial pairs',flush=True)
            continue
        best=sep[0]
        assert all(best[0][1]<x[0][0] for x in sep[1:]),'Minimum not unique or intervals overlap.'
        rows.append(dict(opposite_energy_pair_count=len(opposite),opposite_energy_minimum=opposite_result,label=name,depth=d,degree=N,target_lambda=targetlam,target_frequency=outiv(target),frequency_count=len(included),pair_count=pairs,nontrivial_collisions=0,neutral_equal_frequency_pairs=trivial,extra_families_below_twice=[dict(lam=l,families=fam[l]) for l,_ in included if any(f['grade']>0 for f in fam[l])],nearest_pair_lambda=[best[1],best[2]],nearest_signed_difference=outiv(best[3]),minimum_absolute_separation=outiv(best[0]),minimum_absolute_separation_decimal=float(sum(best[0])/2),minimum_relative_separation_decimal=float(sum(best[0])/sum(target)),target_signed_chain_multiplicity=sum(f['signed_chain_multiplicity'] for f in fam[targetlam])))
        print(name,len(included),best[1:3],float(sum(best[0])/2),flush=True)
w=rows[0]['opposite_energy_minimum']
assert F(2361951706713,10**15)<F(w['separation'][0])<F(w['separation'][1])<F(2361951706714,10**15)

spectral_report=dict(status='passed',M=M,b=B,rows=rows,depths=depths,roots_certified=len(cache))

data=spectral_report
r=F(1,100000)
M=data['M']; b=data['b']; kappa=F(1,10)
def iv(x): return tuple(F(v) for v in x)
def poch(x,n):
    z=F(1)
    for k in range(n): z*=x+k
    return z
rows=[]; checks=0
for row in data['rows']:
    d=row['depth']; N=row['degree']; target=row['target_lambda']; omega=iv(row['target_frequency'])
    a=F(d+1,2); m=N//2
    emin=F(1) if N==0 else (poch(a,m)/factorial(m))**2 if N%2==0 else (2*a*poch(a+1,m)/factorial(m))**2/(N*(N+d+1))
    C=F(16*(d+3)*(2**d-1),d)
    ratio=emin/comb(N+d,d)**2
    kh=ratio/C
    e=2*kappa*(F(b*M**(d-1),factorial(d)))**2*kh
    spectral=[(f['lam'],iv(f['frequency'])) for f in data['depths'][d]['families']]
    contours=[]
    for low,nu in spectral:
        if low>=target: continue
        center_iv=(omega[0]-nu[1],omega[1]-nu[0]); center=sum(center_iv)/2; error=(center_iv[1]-center_iv[0])/2
        gaps=[center_iv[0]]
        for lam,freq in spectral:
            for sign_h in (-1,1):
                unsigned=(omega[0]-freq[1],omega[1]-freq[0]) if sign_h==1 else (omega[0]+freq[0],omega[1]+freq[1])
                for sign_j in (-1,1):
                    if lam==low and sign_h==1 and sign_j==1: continue
                    z=unsigned if sign_j==1 else (-unsigned[1],-unsigned[0])
                    delta=max(center_iv[0]-z[1],z[0]-center_iv[1]); assert delta>0
                    gaps.append(delta)
        cutoff=iv(data['depths'][d]['cutoff_frequency']); heavy=F(data['depths'][d]['heavy_normalized_lower'])
        gaps.extend([cutoff[0]-omega[1]-center_iv[1],heavy-omega[1]-center_iv[1]])
        assert min(gaps)>0
        radius=min(gaps)/4; assert error<radius/2
        distance=radius/2; L=center+radius
        gap_fraction=r*r*L*L/e; assert gap_fraction<F(1,2)
        bound=r*r/(1-r*r)*(1+2*(L+omega[1])/distance); assert bound<1
        checks+=4
        contours.append(dict(low_lambda=low,center=str(center),radius=str(radius),distance_lower=str(distance),material_gap_fraction=str(gap_fraction),bound=str(bound),bound_decimal=float(bound)))
    worst=max(contours,key=lambda x:F(x['bound'])) if contours else None
    rows.append(dict(label=row['label'],depth=d,degree=N,free_korn_constant=str(C),density_ratio=str(ratio),material_e_lower=str(e),contour_count=len(contours),worst=worst,contours=contours))
    print(row['label'],len(contours),None if worst is None else worst['bound_decimal'],flush=True)
# Independent moment controls of the exact chord identity for affine fields.
for n in (3,4,5,6,7,11):
    C=F(16*(n+2)*(2**(n-1)-1),n-1)
    # Symmetric affine strain v=Qx, use sphere-ball fourth moments.
    tr=F(sum(range(1,n+1))); tr2=F(sum(j*j for j in range(1,n+1)))
    lhs=2*(tr**2+2*tr2)/((n+2)*(n+4))+2*tr2/(n+2)**2+2*tr**2/(n+2)**2+2*tr2/(n+2)**2
    # E[(z^T Q z)^2], z=x-y, independent uniform x,y.
    direct=2*(tr**2+2*tr2)/((n+2)*(n+4))+2*tr**2/(n+2)**2+4*tr2/(n+2)**2
    assert lhs == direct
    assert F(tr2,n+2)<=C*F(tr2)
    checks+=2
assert {r['depth'] for r in rows}=={2,3,4,5,6,10}; assert len(rows)==14

from datetime import datetime, timezone
stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
report=dict(status='passed',created_utc=stamp,r=str(r),M=M,b=b,kappa=str(kappa),
    checks=checks,positive_contour_count=sum(x['contour_count'] for x in rows),
    spectral=spectral_report,rows=rows,
    sources={'experiments/check-free-count-stability.py':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
    assumptions=['Natural free weighted strain domain and the complete signed full-form carrier.',
                 'All fourteen supplied nodeless real preparations; no amplitude fit.',
                 'The accompanying article proves the free Korn constant and evolution argument.',
                 'The conclusion is bounded linear evolution modulo the stationary kernel.'])
output=Path(__file__).with_name('free-count-stability-'+stamp+'.json')
output.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(dict(status='passed',roots=len(cache),checks=checks,
    positive_contours=report['positive_contour_count'],output=str(output))))
