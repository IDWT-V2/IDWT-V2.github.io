#!/usr/bin/env python3
"""Test whether geometrically conjugate Hodge channels force neutral dynamics."""
from datetime import datetime, timezone
from hashlib import sha256
from itertools import combinations
from math import comb
from pathlib import Path
import argparse
import json
import sympy as sp


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output',required=True)
    args=ap.parse_args()
    checks=0
    cases=[]
    for m in range(1,7):
        d=2*m
        basis=list(combinations(range(d),m))
        seen=set()
        eigenvalue=sp.I if m%2 else sp.Integer(1)
        for indices in basis:
            if indices in seen:
                continue
            other=tuple(i for i in range(d) if i not in indices)
            sign=(-1)**sum(i>j for i in indices for j in other)
            reverse=(-1)**sum(i>j for i in other for j in indices)
            block=sp.Matrix([[0,reverse],[sign,0]])
            assert block*block==(-1)**m*sp.eye(2)
            checks+=1
            plus=(sp.eye(2)+block/eigenvalue)/2
            minus=(sp.eye(2)-block/eigenvalue)/2
            assert plus*plus==plus and minus*minus==minus and plus*minus==sp.zeros(2)
            checks+=1
            assert plus.conjugate()==(minus if m%2 else plus)
            checks+=1
            seen.update((indices,other))
        k=comb(2*m-1,m-1)
        cases.append({'name':'middle-hodge-reality','real_depth':d,
            'parameters':{'middle_degree':m,'shadow_depth':d-1},
            'measurements':{'each_channel_rank':k,'star_squared':(-1)**m,
                'chirality_exchanged_by_conjugation':bool(m%2),
                'eigenvalues':['i','-i'] if m%2 else ['1','-1']}})

    # A concrete second-order kinetic Noether current on a realified representation.
    t=sp.symbols('t',real=True)
    omega=sp.symbols('omega',positive=True,real=True)
    x=sp.Matrix([sp.exp(-sp.I*omega*t),0,0])
    y=x.conjugate()
    generator=sp.diag(sp.Rational(1,2),-sp.Rational(1,2),0)
    dual=-generator.T
    density_x=(x.H*generator*x)[0]
    density_y=(y.H*dual*y)[0]
    current_x=sp.simplify(2*sp.im((x.H*generator*sp.diff(x,t))[0]))
    current_y=sp.simplify(2*sp.im((y.H*dual*sp.diff(y,t))[0]))
    assert sp.simplify(density_x+density_y)==0
    checks+=1
    assert sp.simplify(current_x+current_y)==-2*omega
    checks+=1
    kinetic=sp.simplify((sp.diff(x,t).H*sp.diff(x,t))[0]+(sp.diff(y,t).H*sp.diff(y,t))[0])
    assert kinetic==2*omega**2
    checks+=1
    schrodinger_x=sp.I*(x.H*sp.diff(x,t)-sp.diff(x,t).H*x)[0]/2
    schrodinger_y=sp.I*(y.H*sp.diff(y,t)-sp.diff(y,t).H*y)[0]/2
    assert sp.simplify(schrodinger_x+schrodinger_y)==0
    checks+=1
    cases.append({'name':'conjugate-reality-noether-control',
        'parameters':{'kinetics':'positive second-order norm','representations':'3 direct-sum conjugate3'},
        'measurements':{'sum_algebraic_densities':'0','sum_noether_currents':str(current_x+current_y),
                        'kinetic_norm':str(kinetic),'same-sign_first-order_kinetic_sum':'0'}})
    report={'experiment_key':'hodge-reality-cancellation-20261009',
        'created_utc':datetime.now(timezone.utc).isoformat(),
        'sources':{'experiments/hodge-reality-cancellation-checks.py':sha256(Path(__file__).read_bytes()).hexdigest()},
        'scope':'Middle-Hodge conjugation and explicit same-law kinetic current counterexample; no particle simulations.',
        'summary':{'checks_passed':checks,'case_count':len(cases),
                   'six_and_ten_have_naturally_conjugate_middle_channels':True,
                   'conjugate_reality_alone_forces_neutral_current':False},
        'cases':cases}
    Path(args.output).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
