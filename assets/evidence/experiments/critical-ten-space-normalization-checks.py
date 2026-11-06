"""Normalization and charge on the existing ten-dimensional critical profile."""
import argparse,json
from hashlib import sha256
from pathlib import Path
import sympy as s
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',type=Path,required=True)
args=p.parse_args()
if args.output.exists(): raise FileExistsError(args.output)
checks=[]
def require(name,test):
    if not test: raise AssertionError(name)
    checks.append(name)
a,A=s.symbols('a A',positive=True)
norm=A**2*a**10*s.pi**5*s.gamma(4)/s.gamma(9)
require('full norm',s.simplify(norm-A**2*a**10*s.pi**5/6720)==0)
require('normal marginal constant',s.pi**3*s.gamma(6)/s.gamma(9)==s.pi**3/336)
require('four-direction normalization',20*s.gamma(4)/s.gamma(6)==1)
require('normal curvature normalization',6*s.gamma(1)/s.gamma(4)==1)
require('critical Hamiltonian energy over norm',s.gamma(5)*s.gamma(9)/(s.gamma(10)*s.gamma(4))==s.Rational(4,9))
require('normal source RMS divergent',6+2-2*4==-0)
result={'code_sha256':sha256(Path(__file__).read_bytes()).hexdigest(),
        'checks':checks,'check_count':len(checks),'review_status':'author checks only',
        'scope':'Known static profile; no nonzero physical clock or dynamical stability claim'}
args.output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'passed':len(checks)}))
