"""Independent finite arithmetic audit; no spatial discretization or DB writes."""
from math import comb
from fractions import Fraction
from pathlib import Path
import hashlib, json
from datetime import datetime, timezone
LABELS=((2,75),(2,80),(2,94),(3,0),(3,3),(4,2),(4,19),(4,71),(5,9),(5,14),(5,21),(6,12),(6,34),(10,22))
rows=[]
for d,N in LABELS:
    C=lambda n: comb(n+d,d)
    mult=lambda n: comb(n+d-1,d-1)
    pairs=[]
    # Independent monotone integer bisection, unlike principal's lookup table.
    for m in range(N):
        target=2*C(N)-C(m)
        lo,hi=N+1,2*N+d+2
        assert C(hi)>target
        while lo<hi:
            mid=(lo+hi)//2
            if C(mid)<target: lo=mid+1
            else: hi=mid
        if C(lo)==target:
            pairs.append({'lower':m,'upper':lo,'lower_count':C(m),'upper_count':C(lo),
                'beat':C(N)-C(m),'lower_euler':N*C(m)-m*C(N),
                'upper_euler':N*C(lo)-lo*C(N),'lower_dim':mult(m),'upper_dim':mult(lo)})
    neg=2*sum(mult(m) for m in range(N))
    assert neg==(2*comb(N+d-1,d) if N else 0)
    rows.append({'depth':d,'degree':N,'count':C(N),'negative_index':neg,
        'kernel_dimension':2*mult(N),'collisions':pairs,
        'sufficient_kappa_upper_over_pi_squared_times_mu_squared':str(Fraction(d,4*(C(N)-1)**2)) if N else None})
assert [(r['depth'],r['degree'],p['lower'],p['upper']) for r in rows for p in r['collisions']]==[(2,75,30,102)]
p=rows[0]['collisions'][0]
assert (p['lower_euler'],p['upper_euler'],p['lower_dim'],p['upper_dim'])==(-50580,103248,31,103)
report={'status':'passed','utc':datetime.now(timezone.utc).isoformat(),'rows':rows,
    'assumptions':['The declared count law and fourteen assigned labels are inputs.'],
    'checker_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'scope':'Exact counts, complete negative index and collision enumeration; analytic domain and stability proof are separately reviewed in the accompanying spectral-stability note.'}
output=Path(__file__).with_name('pullback-excited-stability-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'.json')
output.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'status':report['status'],'labels':len(rows),'exceptional_collision':p,'output':str(output)}))
