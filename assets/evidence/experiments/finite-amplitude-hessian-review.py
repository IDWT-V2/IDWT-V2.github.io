#!/usr/bin/env python3
"""Independent symbolic differentiation of the implemented radial Hessian."""

from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import json
import sympy as sp


def main():
    a,c,p0,p1,w,M2=sp.symbols('a c p0 p1 omega M2',real=True)
    A=1+p0*p0+p1*p1
    B=1-w*w*c*c
    R=sp.sqrt(A)*sp.sqrt(B)-1+M2*(a*a+c*c-1)**2/4+(c*c-c**4/4)/2
    norm_delta=a*a+c*c-1
    U=sp.Matrix([[M2*(norm_delta+2*a*a),2*M2*a*c],
                 [2*M2*a*c,M2*(norm_delta+2*c*c)+1-3*c*c/2-w*w*sp.sqrt(A)/B**sp.Rational(3,2)]])
    J=sp.Matrix([[0,0],[-w*w*c*p0/(sp.sqrt(A)*sp.sqrt(B)),-w*w*c*p1/(sp.sqrt(A)*sp.sqrt(B))]])
    checks=[]
    for i,u in enumerate((a,c)):
        for j,v in enumerate((a,c)):
            residual=sp.simplify(sp.diff(R,u,v)-U[i,j])
            if residual!=0:raise AssertionError(('amplitude potential Hessian',i,j,residual))
            checks.append(f'complete amplitude potential Hessian {i},{j}')
        for j,p in enumerate((p0,p1)):
            residual=sp.simplify(sp.diff(R,u,p)-J[i,j])
            if residual!=0:raise AssertionError(('mixed Hessian',i,j,residual))
            checks.append(f'complete mixed amplitude-gradient Hessian {i},{j}')
    path=Path(__file__).resolve()
    report={'created_at':datetime.now(timezone.utc).isoformat(),
            'experiment':'finite-amplitude-radial-hessian-symbolic-review',
            'sources':{'experiments/finite-amplitude-hessian-review.py':sha256(path.read_bytes()).hexdigest()},
            'exact_checks':len(checks),'checks':checks,
            'scope':'Direct symbolic differentiation verifies the two-amplitude potential and mixed gradient Hessians used by the finite-element energy-form calculation. It does not certify its numerical spectrum or nonlinear stability.'}
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    output=path.with_name('finite-amplitude-hessian-review-'+stamp+'.json')
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'exact_checks':len(checks),'report':str(output)}))


if __name__=='__main__':main()
