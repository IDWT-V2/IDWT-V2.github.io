"""Neutral integrated charge with unequal radial orientation shapes."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import sympy as s

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',type=Path,required=True)
args=p.parse_args()
if args.output.exists():
    raise FileExistsError(args.output)
u,lam=s.symbols('u lam',positive=True)
f0=1-s.exp(-u)
fj=1-s.exp(-lam*u)
enclosed=f0**2*(f0-fj)/2
checks=[]
def require(name,test):
    if not test:
        raise AssertionError(name)
    checks.append(name)
require('neutral boundary origin',s.limit(enclosed,u,0)==0)
require('neutral boundary infinity',s.limit(enclosed,u,s.oo)==0)
require('matched shape zero density',s.simplify(enclosed.subs(lam,1))==0)
moment=-s.Rational(1,2)*(1/lam-2/(lam+1)+1/(lam+2)-s.Rational(1,3))
require('signed second moment',s.simplify(moment-(s.Rational(1,6)-1/(lam*(lam+1)*(lam+2))))==0)
require('linear shape sensitivity',s.diff(moment,lam).subs(lam,1)==s.Rational(11,36))
for value in [s.Rational(1,2),s.Integer(1),s.Integer(2)]:
    integrated=-s.integrate(enclosed.subs(lam,value),(u,0,s.oo))
    require(f'explicit integral lambda={value}',s.simplify(integrated-moment.subs(lam,value))==0)
result={'code_sha256':sha256(Path(__file__).read_bytes()).hexdigest(),
        'check_count':len(checks),'checks':checks,'status':'author checked; independent review pending',
        'signed_second_moment_over_reference_radius_squared':str(s.factor(moment)),
        'scope':'Integrated neutrality and conditional spatial charge form factor; not a measured neutrino charge radius.'}
args.output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'passed':len(checks)}))
