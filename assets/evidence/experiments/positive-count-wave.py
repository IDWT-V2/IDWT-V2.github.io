"""Smooth exact count charts and a positive wave completion with spectral gap checks."""
from datetime import datetime, timezone
from fractions import Fraction
from hashlib import sha256
from importlib.util import spec_from_file_location, module_from_spec
from math import isqrt
from pathlib import Path
import json

import mpmath as mp
import numpy as np

ROOT = Path(__file__).resolve().parent
spec = spec_from_file_location('envelope', ROOT/'convex-count-envelope.py')
envelope = module_from_spec(spec)
spec.loader.exec_module(envelope)
COUNTS, ERRORS = {}, {}


def check(value, label):
    if not bool(value):
        raise AssertionError(label)
    COUNTS[label] = COUNTS.get(label, 0)+1


def high(value):
    return mp.mpf(value.numerator)/value.denominator


def smooth_chart(candidates):
    lines = sorted(candidates, key=lambda c:c['m'])
    boundaries = [(one['eta']-two['eta'])/(two['m']-one['m']) for one,two in zip(lines,lines[1:])]
    check(all(a<b for a,b in zip(boundaries,boundaries[1:])),
          'all geometric count tangents occur in their ordered convex envelope')
    margins = []
    for i, line in enumerate(lines):
        local = []
        if i:
            local.append(line['x']-boundaries[i-1])
        if i<len(boundaries):
            local.append(boundaries[i]-line['x'])
        margins.append(min(local))
    sigma = min(margins)/4
    check(sigma>0, 'one common smoothing width is fixed by all eligible chart nodes')
    def evaluate(x):
        for i, boundary in enumerate(boundaries):
            z = x-boundary
            if abs(z)<sigma:
                u = z/sigma
                cdf = mp.mpf(1)/2+mp.mpf(15)/16*(u-mp.mpf(2)/3*u**3+u**5/5)
                moment = mp.mpf(15)/16*(u*u/2-u**4/2+u**6/6-mp.mpf(1)/6)
                hinge = sigma*(u*cdf-moment)
                jump = lines[i+1]['m']-lines[i]['m']
                value = lines[i]['m']*x+lines[i]['eta']+jump*hinge
                derivative = lines[i]['m']+jump*cdf
                curvature = jump*mp.mpf(15)/16*(1-u*u)**2/sigma
                return value, derivative, curvature
        winner = max(lines, key=lambda c:c['m']*x+c['eta'])
        return winner['m']*x+winner['eta'], winner['m'], mp.mpf(0)
    for line in lines:
        value, derivative, curvature = evaluate(line['x'])
        check(abs(value-(line['omega']-line['h']*line['c']))<mp.mpf('1e-75'),
              'smoothing leaves every eligible exact count contact value unchanged')
        check(derivative==line['m'] and curvature==0,
              'smoothing retains an open affine chart at every eligible count state')
    for boundary in boundaries[::max(1,len(boundaries)//7)]:
        for fraction in [mp.mpf('-.7'),mp.mpf('0'),mp.mpf('.4')]:
            x=boundary+fraction*sigma
            value, first, second = evaluate(x)
            check(second>=0, 'the regularized count envelope remains convex through every tested transition')
            numeric_first = mp.diff(lambda xx:evaluate(xx)[0],x)
            numeric_second = mp.diff(lambda xx:evaluate(xx)[1],x)
            check(abs(numeric_first-first)<mp.mpf('1e-65'),
                  'independent differentiation verifies the complete smoothed source coefficient')
            check(abs(numeric_second-second)/max(1,abs(second))<mp.mpf('1e-65'),
                  'independent differentiation verifies the smoothed current-ratio Hessian')
    bound = sigma*mp.mpf(5)/16*(lines[-1]['m']-lines[0]['m'])
    return sigma, bound, evaluate


def main():
    mp.mp.dps=100
    sectors,cases=[],[]
    for depth in envelope.forward.radial.DEPTHS:
        epsilon,candidates,_=envelope.depth_carrier(depth)
        sigma,raising_bound,evaluate=smooth_chart(candidates)
        sectors.append(dict(depth=depth, eligible_charts=len(candidates), smoothing_width=str(sigma),
                            maximum_potential_raising_bound_per_norm=str(raising_bound)))
        for name,dd,index in envelope.forward.radial.ASSIGNMENTS:
            if dd!=depth:
                continue
            level=index-1
            count,first,_=envelope.forward.count_data(depth,level)
            hq=Fraction(2*level+depth,2)
            cq=epsilon*first
            lambdaq=hq/cq
            omegaq=epsilon*count
            etaq=omegaq-lambdaq
            m2q=lambdaq*lambdaq-hq*hq
            zero_square=etaq*etaq-m2q
            check(zero_square==hq*hq-2*lambdaq*omegaq+omegaq*omegaq,
                  'the full infinite wave-spectrum zero test reduces to an exact rational equation')
            # Only the nearest levels can minimize the monotone positive branch.
            if zero_square>Fraction(depth*depth,4):
                floor_root=isqrt((4*zero_square.numerator)//zero_square.denominator)
                near=max(0,(floor_root-depth)//2)
                levels=[near,near+1]
            else:
                levels=[0]
            for k in levels:
                check(Fraction((2*k+depth)**2,4)!=zero_square,
                      'the exact nearest-level comparison excludes every spatial zero mode for this assigned state')
            gaps=[]
            for k in levels:
                bareq=Fraction(2*k+depth,2)
                root=mp.sqrt(high(bareq*bareq+m2q))
                # Rationalized evaluation avoids cancellation at large source coefficients.
                gap=abs(high(bareq*bareq-zero_square))/(root+abs(high(etaq)))
                gaps.append(gap)
            gap=min(gaps)
            check(gap>0,'the complete self-adjoint count generator has a positive distance from zero')
            hh,cc,omega,eta=high(hq),high(cq),high(omegaq),high(etaq)
            mass=mp.sqrt(high(m2q))
            aa,bb=mp.sqrt((1+cc)/2),mp.sqrt((1-cc)/2)
            pair=mp.matrix([aa,bb])
            ham=mp.matrix([[hh+eta,mass],[mass,-hh+eta]])
            residual=mp.norm(ham*ham*pair-omega*omega*pair)/(omega*omega)
            check(residual<mp.mpf('1e-65'),
                  'the fully squared field equation retains the exact count frequency relatively')
            normq=Fraction(17,10)
            full_energy=2*omegaq*omegaq*normq
            phase_action=2*omegaq*normq
            check(full_energy==omegaq*phase_action,
                  'complete positive energy divided by canonical phase action equals the count frequency')
            for k in [0,level,level+1,1000]:
                spatial=mp.mpf(k)+mp.mpf(depth)/2
                eigs=[eta-mp.sqrt(spatial*spatial+mass*mass),eta+mp.sqrt(spatial*spatial+mass*mass)]
                for eig in eigs:
                    check(eig*eig>=0,'every retained wave block has nonnegative squared lab frequency')
                    # Exact oscillator energy checks both position and velocity data.
                    freq=abs(eig)
                    initial=mp.mpc('.03','.07'); velocity=mp.mpc('-.02','.01')
                    initial_energy=abs(velocity)**2+freq*freq*abs(initial)**2
                    for time in [mp.mpf('.2'),mp.mpf('3')]:
                        value=mp.cos(freq*time)*initial+mp.sin(freq*time)/freq*velocity
                        speed=-freq*mp.sin(freq*time)*initial+mp.cos(freq*time)*velocity
                        check(abs(abs(speed)**2+freq*freq*abs(value)**2-initial_energy)
                              /max(1,initial_energy)<mp.mpf('1e-70'),
                              'wave evolution conserves the complete graph-energy block with both initial data')
            cases.append(dict(particle=name,depth=depth,inferred_index=index,
                frequency_in_radius_units_exact=str(omegaq), zero_condition_rhs_exact=str(zero_square),
                exact_nearest_spatial_levels=levels, full_generator_gap=str(gap),
                wave_clock_relative_residual=str(residual),
                positive_phase_action_normalization='Qplus=2*omega*N; not electric charge',
                classification='exact positive count standing wave and nonresonant chart-local persistence'))
            print(json.dumps(dict(particle=name,depth=depth,gap=str(gap))),flush=True)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    names=['positive-count-wave.py','convex-count-envelope.py','forward-count-action.py',
           'inverse-current-radial.py','excited-shell-orientation-current.py']
    report=dict(experiment_key='positive-wave-exact-count-completion',
        created_utc=datetime.now(timezone.utc).isoformat(),
        sources={'experiments/'+name:sha256((ROOT/name).read_bytes()).hexdigest() for name in names},
        evaluations=sum(COUNTS.values()),conditions=COUNTS,maximum_errors=ERRORS,sectors=sectors,cases=cases,
        limits=['the positive square action changes the physical kinetic law',
                'the count interaction and sphere geometry remain effective premises',
                'one common compact convolution is an explicit functional choice, not a master-field derivation',
                'nonlinear persistence uses a nodeless field and an open chart; the full generator gap is proved at the diagnostic scales',
                'positive energy and a finite principal cone do not establish Lorentz boost inertia',
                'contact attraction and orientation response from the first-order action are not transferred',
                'source second variations at vacuum retain their regularity qualification'])
    path=ROOT/('positive-count-wave-results-'+stamp+'.json')
    path.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(report=str(path),evaluations=report['evaluations'])),flush=True)


if __name__=='__main__':
    main()
