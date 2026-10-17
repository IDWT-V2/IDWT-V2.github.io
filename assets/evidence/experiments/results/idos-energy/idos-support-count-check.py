from fractions import Fraction
from math import comb
import json
rows=[]
for d,n in [(2,75),(2,80),(2,94),(3,0),(3,3),(4,2),(4,19),(4,71),(5,9),(5,14),(5,21),(6,12),(6,34),(10,22)]:
 D=comb(n+d,d); p=d; q=1; eps=Fraction(1); a=Fraction(1)
 Lp=Fraction(q*D,p)
 carrier=a*Lp
 wave=D*eps
 assert p*carrier==q*wave
 total=carrier+wave
 assert total==Fraction(p+q,p)*D*eps
 # L^2 H'' at fixed actions, using stationarity, no root rounding.
 hess_scaled=p*(p+q)*carrier
 assert hess_scaled>0
 rows.append(dict(depth=d,cutoff=n,count=D,carrier_fraction=str(carrier/total),
                  energy_factor=str(total/wave),radius_power=str(Lp),scaled_hessian=str(hess_scaled)))
print(json.dumps(dict(checks=42,rows=rows),indent=2))
