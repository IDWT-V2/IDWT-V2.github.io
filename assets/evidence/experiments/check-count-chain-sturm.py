"""Certify count-chain low roots with exact rational Sturm arithmetic."""
from fractions import Fraction as F
from math import prod, factorial, comb
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

LABELS = [(2,75),(2,80),(2,94),(3,0),(3,3),(4,2),(4,19),(4,71),
          (5,9),(5,14),(5,21),(6,12),(6,34),(10,22)]
M = 100000

def below(diagonal, off_squared, z):
    previous, current = F(1), diagonal[0] - z
    signs = [1]
    if current:
        signs.append(1 if current > 0 else -1)
    for k in range(1, len(diagonal)):
        previous, current = current, (diagonal[k]-z)*current-off_squared[k-1]*previous
        if current:
            signs.append(1 if current > 0 else -1)
    if not current:
        raise ArithmeticError('An endpoint is exactly an eigenvalue.')
    return sum(a != b for a,b in zip(signs, signs[1:]))

rows = []
for d,n in LABELS:
    lam = n*(n+d+1)
    weak_squared = [lam+j*(d+1-j) for j in range(1,d+1)]
    assert prod(weak_squared) == (factorial(d)*comb(n+d,d))**2
    diagonal = [x+M*M for x in weak_squared[:-1]]+[weak_squared[-1]]
    off_squared = [M*M*x for x in weak_squared[1:]]
    target = F(prod(weak_squared), M**(2*d-2))
    lo, hi = F(0), F(1)
    assert below(diagonal,off_squared,lo*target) == 0
    assert below(diagonal,off_squared,hi*target) == 1
    for _ in range(100):
        mid = (lo+hi)/2
        count = below(diagonal,off_squared,mid*target)
        assert count in (0,1)
        if count == 0:
            lo = mid
        else:
            hi = mid
    assert lo > F(999999,1000000)**2
    # Rational square-root enclosure; no floating point enters the certificate.
    flo, fhi = F(0), F(1)
    for _ in range(110):
        mid = (flo+fhi)/2
        if mid*mid < lo:
            flo = mid
        else:
            fhi = mid
    lower_frequency = flo
    flo, fhi = F(0), F(1)
    for _ in range(110):
        mid = (flo+fhi)/2
        if mid*mid < hi:
            flo = mid
        else:
            fhi = mid
    upper_frequency = fhi
    rows.append(dict(depth=d,degree=n,
        squared_frequency_ratio_bounds=[str(lo),str(hi)],
        relative_frequency_error_bounds=[str(lower_frequency-1),str(upper_frequency-1)],
        approximate_error=float((lower_frequency+upper_frequency)/2-1),
        target_squared_frequency=str(target),roots_below_lower=0,roots_below_upper=1))
    print(d,n,rows[-1]['approximate_error'],flush=True)

stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
report = dict(status='passed',M=M,rows=rows,
    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    method='Exact rational Sturm counts and rational square-root bounds.',
    scope='Lowest positive finite-chain frequency; weighted PDE realization is separate.')
path = Path(__file__).with_name('count-chain-sturm-'+stamp+'.json')
path.write_text(json.dumps(report,indent=2)+'\n')
print(path)
