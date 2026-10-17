#!/usr/bin/env python3
"""Independent pointwise weighted-adjoint and spectral trace certificates."""
from fractions import Fraction as Q
from math import comb, prod
from itertools import combinations
from pathlib import Path
import hashlib,json
from datetime import datetime,timezone

def add(*vs):
    out={}
    for v in vs:
        for m,c in v.items(): out[m]=out.get(m,0)+c
    return {m:c for m,c in out.items() if c}
def sc(v,c):return {m:x*c for m,x in v.items() if x*c}
def ei(v,i):
    return {m|(1<<i):c*(-1)**((m&((1<<i)-1)).bit_count()) for m,c in v.items() if not m&(1<<i)}
def ii(v,i):
    return {m^(1<<i):c*(-1)**((m&((1<<i)-1)).bit_count()) for m,c in v.items() if m&(1<<i)}
def ev(v,y):return add(*(sc(ei(v,i),a) for i,a in enumerate(y)))
def iv(v,y):return add(*(sc(ii(v,i),a) for i,a in enumerate(y)))
def B(v,y):return add(v,sc(ev(iv(v,y),y),-1))
def Binv(v,y):return add(v,sc(ev(iv(v,y),y),1/(1-sum(a*a for a in y))))
def compositions(n,d):
    if d==1:yield (n,);return
    for j in range(n+1):
        for tail in compositions(n-j,d-1):yield (j,)+tail

def h(d,n,k):return comb(n+d-1,d-k-1)*comb(n+k-1,k)

rows=[]
for d in (2,3,4,5,6,10):
    checks=0
    y=tuple(Q(i+1,4*d*d) for i in range(d));s=1-sum(a*a for a in y)
    for k in range(1,d+1):
        masks=[sum(1<<i for i in I) for I in combinations(range(d),k)]
        if d==10:masks=list(dict.fromkeys([masks[0],masks[len(masks)//2],masks[-1]]))
        for m in masks:
            for n in (0,1,3):
                # Mixed monomial p, independently differentiated at a rational point.
                p=tuple((n if i==0 else 1 if i==d-1 else 0) for i in range(d))
                f=prod(a**r for a,r in zip(y,p));a={m:f}
                derivatives=[sc(a,Q(p[i])/y[i]) for i in range(d)]
                raw={}
                for i in range(d):
                    # (partial_i(w B a))/w, with actual component metric.
                    derivative_B=add(sc(ei(iv(a,y),i),-1),sc(ev(ii(a,i),y),-1))
                    jet=add(B(derivatives[i],y),derivative_B,sc(B(a,y),-y[i]/s))
                    raw=add(raw,sc(ii(jet,i),-1))
                actual=Binv(raw,y)
                expected=add(*(sc(ii(v,i),-1) for i,v in enumerate(derivatives)),sc(iv(a,y),sum(p)+d+3-k))
                assert actual==expected,(d,k,m,n)
                assert Binv(B(a,y),y)==a
                # The radial boundary contraction is exactly s times contraction.
                assert iv(B(a,y),y)==sc(iv(a,y),s)
                checks+=3
    # Full leading-symbol trace over every homogeneous monomial, all grades.
    # This is independent of the claimed multiplicity formula.
    for q in range(1,5):
        dim=trace=0
        for k in range(min(d,q)+1):
            n=q-k
            for p in compositions(n,d):
                for I in combinations(range(d),k):
                    dim+=1
                    trace+=q*(n+d+1-k)+2*sum(p[i]+1 for i in I)
        pairs=[(q-k,k) for k in range(min(d-1,q-1)+1)]
        assert dim==2*sum(h(d,n,k) for n,k in pairs)
        assert trace==2*sum(h(d,n,k)*(n+k)*(n+d+1-k) for n,k in pairs)
        checks+=2
    maxN={2:94,3:3,4:71,5:21,6:34,10:22}[d]
    scalar=maxN*(maxN+d+1);extra=1600*2*(d+1);M=100000
    aj=[j*(d+1-j) for j in range(1,d+1)]
    assert extra>scalar
    assert (M*M-extra-max(aj))**2*prod(extra+a for a in aj)>M**4*prod(scalar+a for a in aj)
    checks+=2
    rows.append({'d':d,'checks':checks})
root=Path(__file__).parent
inputs={}
report={'utc':datetime.now(timezone.utc).isoformat(),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'checks':sum(x['checks'] for x in rows),'rows':rows,'inputs':inputs,'scope':'Independent rational-point weighted adjoint on every grade; every component basis for depths <=6, three representative components per grade at d10; exact radial-flux identity; exhaustive homogeneous full-spectrum dimension and trace for total degrees 1..4. Finite-M separation checked independently with integers at b40 and M100000. Analytic all-degree proof is in the accompanying count-chain note.'}
(root/('count-chain-forms-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'.json')).write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
