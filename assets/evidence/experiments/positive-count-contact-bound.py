"""Interval Bernstein certificates for all maintained orthogonal-column pair energies."""
from datetime import datetime,timezone
from fractions import Fraction
from hashlib import sha256
from importlib.util import spec_from_file_location,module_from_spec
from itertools import combinations
from math import comb
from pathlib import Path
import json

import mpmath as mp

ROOT=Path(__file__).resolve().parent
spec=spec_from_file_location('envelope',ROOT/'convex-count-envelope.py')
envelope=module_from_spec(spec); spec.loader.exec_module(envelope)
spec=spec_from_file_location('contact',ROOT/'positive-count-contact.py')
contact=module_from_spec(spec); spec.loader.exec_module(contact)
CHECKS=0


def interval(value):
    if isinstance(value,Fraction):
        return mp.iv.mpf(value.numerator)/value.denominator
    return mp.iv.mpf(value)


def low(value):
    return mp.mpf(value.a)


def high(value):
    return mp.mpf(value.b)


def add(one,two):
    return [(one[i] if i<len(one) else interval(0))+(two[i] if i<len(two) else interval(0))
            for i in range(max(len(one),len(two)))]


def scale(one,factor):
    return [x*factor for x in one]


def multiply(one,two):
    result=[interval(0) for _ in range(len(one)+len(two)-1)]
    for i,a in enumerate(one):
        for j,b in enumerate(two): result[i+j]+=a*b
    return result


def row_polynomial(own,mass,eta):
    dm=add(mass,[-own['m']]); de=add(eta,[-own['eta']])
    mismatch=add(add(multiply(dm,dm),multiply(de,de)),scale(multiply(dm,de),2*own['x']))
    return add(mismatch,scale(add(scale(dm,own['x']),de),2*own['omega']))


def bernstein(power):
    nn=len(power)-1
    return [sum((power[k]*interval(Fraction(comb(j,k),comb(nn,k))) for k in range(j+1)),interval(0))
            for j in range(nn+1)]


def subdivide(coefficients):
    rows=[coefficients]
    while len(rows[-1])>1:
        rows.append([(a+b)/2 for a,b in zip(rows[-1],rows[-1][1:])])
    return [row[0] for row in rows],[row[-1] for row in rows[::-1]]


def certify(coefficients,depth=0):
    global CHECKS
    CHECKS+=1
    if min(low(v) for v in coefficients)>0:
        return True,min(low(v) for v in coefficients),depth
    if depth>=10:
        return False,min(low(v) for v in coefficients),depth
    left,right=subdivide(coefficients)
    a=certify(left,depth+1); b=certify(right,depth+1)
    return a[0] and b[0],min(a[1],b[1]),max(a[2],b[2])


def sector(depth):
    epsilon=Fraction(3,10)/max(envelope.forward.count_data(depth,n-1)[1]
        for _,dd,n in envelope.forward.radial.ASSIGNMENTS if dd==depth)
    values=[];level=0
    while True:
        count,first,second=envelope.forward.count_data(depth,level)
        cc=epsilon*first; hh=Fraction(2*level+depth,2)
        if cc>=1: break
        if hh*epsilon*second-cc*(1-cc*cc)>0:
            xx=mp.iv.sqrt(1-interval(cc)**2)
            values.append(dict(level=level,x=xx,m=interval(hh)*xx/interval(cc),
                eta=interval(epsilon*count-hh/cc),omega=interval(epsilon*count)))
        level+=1
    lines=values[::-1]
    edges=[(a['eta']-b['eta'])/(b['m']-a['m']) for a,b in zip(lines,lines[1:])]
    margins=[]
    for i,line in enumerate(lines):
        if i: margins.append(line['x']-edges[i-1])
        if i<len(edges): margins.append(edges[i]-line['x'])
    minimum=mp.iv.mpf([min(low(v) for v in margins),min(high(v) for v in margins)])
    sigma=minimum/4
    assert low(sigma)>0
    return values,lines,edges,sigma


def main():
    global CHECKS
    mp.mp.dps=100; mp.iv.dps=80
    cases=[]
    for depth in envelope.forward.radial.DEPTHS:
        values,lines,edges,sigma=sector(depth)
        _,reference_nodes,_=envelope.depth_carrier(depth)
        _,reference_edges,reference_width,reference_source=contact.smooth_data(reference_nodes)
        assert low(sigma)<=reference_width<=high(sigma)
        CHECKS+=1
        maintained=sorted(n-1 for _,dd,n in envelope.forward.radial.ASSIGNMENTS if dd==depth)
        pairs=list(combinations(maintained,2))
        if depth==10: pairs.append((22,23))
        for first,second in pairs:
            one,two=[next(c for c in values if c['level']==level) for level in [first,second]]
            direct=[]
            for own in [one,two]:
                for selected in values:
                    if selected['level']==own['level']: continue
                    change=row_polynomial(own,[selected['m']],[selected['eta']])[0]
                    direct.append(low(change))
            assert min(direct)>0
            bands=[]
            for i,edge in enumerate(edges):
                # Count nodes stay at least four widths from the band edges.
                if high(edge)<=low(two['x']) or low(edge)>=high(one['x']): continue
                assert low(edge-sigma)>high(two['x']) and high(edge+sigma)<low(one['x'])
                jump=lines[i+1]['m']-lines[i]['m']
                cdf=list(map(interval,[0,0,0,10,-15,6]))
                moment=list(map(interval,[0,0,0,-10,30,-30,10]))
                mass=add([lines[i]['m']],scale(cdf,jump))
                eta=add([lines[i]['eta']],scale(add(scale(cdf,edge),scale(moment,sigma)),-jump))
                w1=[(edge-sigma-two['x'])/(one['x']-two['x']),2*sigma/(one['x']-two['x'])]
                w2=add([interval(1)],scale(w1,-1))
                polynomial=add(multiply(w1,row_polynomial(one,mass,eta)),
                               multiply(w2,row_polynomial(two,mass,eta)))
                certificate=bernstein(polynomial)
                result=certify(certificate)
                own1,own2=[next(c for c in reference_nodes if c['level']==level) for level in [first,second]]
                for fraction in [Fraction(0),Fraction(1,4),Fraction(1,2),Fraction(3,4),Fraction(1)]:
                    tt=mp.mpf(fraction.numerator)/fraction.denominator
                    xx=reference_edges[i]+reference_width*(2*tt-1)
                    ff,mm,_=reference_source(xx); ee=ff-xx*mm
                    weight=(xx-own2['x'])/(own1['x']-own2['x'])
                    direct_value=weight*contact.row_delta(own1,mm,ee)+(1-weight)*contact.row_delta(own2,mm,ee)
                    evaluated=interval(0)
                    for coefficient in polynomial[::-1]: evaluated=evaluated*interval(fraction)+coefficient
                    assert low(evaluated)<=direct_value<=high(evaluated)
                    CHECKS+=1
                bands.append(dict(lower_level=lines[i+1]['level'],upper_level=lines[i]['level'],
                    strict_positive_certificate=result[0],normalized_contact_lower_bound=str(result[1]),
                    subdivision_depth=result[2],bernstein_coefficient_intervals=[str(v) for v in certificate]))
            success=all(b['strict_positive_certificate'] for b in bands)
            case=dict(depth=depth,shell_levels=[first,second],
                other_unsmoothed_chart_minimum_source_cost_lower_bound=str(min(direct)),
                common_smoothing_width_interval=str(sigma),smoothing_bands=bands,
                pointwise_nonnegative_pair_energy_certified=success,
                second_is_unassigned_mathematical_control=depth==10)
            cases.append(case)
            print(json.dumps(dict(depth=depth,levels=[first,second],certified=success,
                                  bands=len(bands),max_subdivision=max([b['subdivision_depth'] for b in bands],default=0))),flush=True)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    names=['positive-count-contact-bound.py','positive-count-contact.py','positive-count-wave.py','convex-count-envelope.py','forward-count-action.py',
           'inverse-current-radial.py','excited-shell-orientation-current.py']
    report=dict(experiment_key='interval-positive-count-pair-contact-bound',
        created_utc=datetime.now(timezone.utc).isoformat(),interval_decimal_precision=80,
        sources={'experiments/'+n:sha256((ROOT/n).read_bytes()).hexdigest() for n in names},
        evaluations=CHECKS,cases=cases,
        limits=['certificates concern orthogonal column blocks with their isolated initial clocks',
                'all maintained distinct pairs are checked; depth ten adds one unassigned adjacent control',
                'a pointwise energy inequality is not a physical radial force calculation',
                'coherent fields and time-evolved encounters are not covered',
                'the diagnostic geometric scales and stated compact smoothing remain premises'])
    path=ROOT/('positive-count-contact-bound-results-'+stamp+'.json')
    path.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(report=str(path),evaluations=CHECKS,
                         certified_pairs=sum(c['pointwise_nonnegative_pair_energy_certified'] for c in cases))),flush=True)


if __name__=='__main__':
    main()
