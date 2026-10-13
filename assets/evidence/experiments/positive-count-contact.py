"""Fully varied smooth count source and pair energy of its positive wave action."""
from bisect import bisect_right
from datetime import datetime, timezone
from hashlib import sha256
from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path
import json

import mpmath as mp

ROOT = Path(__file__).resolve().parent
spec = spec_from_file_location('wave', ROOT/'positive-count-wave.py')
wave = module_from_spec(spec)
spec.loader.exec_module(wave)
COUNTS, ERRORS = {}, {}


def check(condition, label):
    if not bool(condition):
        raise AssertionError(label)
    COUNTS[label] = COUNTS.get(label, 0)+1


def close(actual, expected, label, tolerance=mp.mpf('1e-55')):
    error = abs(actual-expected)/max(1, abs(expected))
    ERRORS[label] = max(ERRORS.get(label, mp.mpf(0)), error)
    check(error<tolerance, label)


def real_inner(one, two):
    return mp.re(sum(mp.conj(a)*b for a,b in zip(one,two)))


def norm2(value):
    return real_inner(value,value)


def smooth_data(candidates):
    lines = sorted(candidates, key=lambda c:c['m'])
    boundaries = [(a['eta']-b['eta'])/(b['m']-a['m']) for a,b in zip(lines,lines[1:])]
    margins = []
    for i,line in enumerate(lines):
        margins.extend([line['x']-boundaries[i-1]] if i else [])
        margins.extend([boundaries[i]-line['x']] if i<len(boundaries) else [])
    sigma = min(margins)/4
    check(sigma>0, 'all eligible count states determine one common smoothing width')
    check(all(b-a>8*sigma for a,b in zip(boundaries,boundaries[1:])),
          'the complete smoothing bands are disjoint')
    def evaluate(x):
        index = bisect_right(boundaries,x)
        nearby = index if index<len(boundaries) and boundaries[index]-x<sigma else index-1
        if 0<=nearby<len(boundaries) and abs(x-boundaries[nearby])<sigma:
            u = (x-boundaries[nearby])/sigma
            cdf = mp.mpf(1)/2+mp.mpf(15)/16*(u-2*u**3/3+u**5/5)
            moment = mp.mpf(15)/16*(u*u/2-u**4/2+u**6/6-mp.mpf(1)/6)
            hinge = sigma*(u*cdf-moment)
            jump = lines[nearby+1]['m']-lines[nearby]['m']
            f = lines[nearby]['m']*x+lines[nearby]['eta']+jump*hinge
            first = lines[nearby]['m']+jump*cdf
            second = jump*mp.mpf(15)/16*(1-u*u)**2/sigma
            return f,first,second
        line = lines[index]
        return line['m']*x+line['eta'],line['m'],mp.mpf(0)
    return lines,boundaries,sigma,evaluate


def full_variation(depth, boundaries, sigma, evaluate):
    # A finite Hermitian carrier checks the full real source algebra, not
    # the actual spatial Dirac domain, which is treated analytically.
    beta = mp.matrix([[1,0],[0,-1]])
    h0 = mp.matrix([[mp.mpf('.7'),mp.mpc('.2','.3')],
                    [mp.mpc('.2','-.3'),mp.mpf('-.4')]])
    def unpack(values):
        return mp.matrix([mp.mpc(values[0],values[1]),mp.mpc(values[2],values[3])])
    def source(values):
        phi = unpack(values)
        rho = norm2(phi)
        scalar = real_inner(phi,beta*phi)
        x = scalar/rho
        f,m,curvature = evaluate(x)
        eta = f-x*m
        return (h0+m*beta+eta*mp.eye(2))*phi
    def generating(values):
        phi = unpack(values)
        rho = norm2(phi)
        x = real_inner(phi,beta*phi)/rho
        return real_inner(phi,h0*phi)+rho*evaluate(x)[0]
    def squared(values):
        return norm2(source(values))
    boundary = boundaries[len(boundaries)//2]
    for fraction in [mp.mpf('-.6'),mp.mpf('0'),mp.mpf('.5')]:
        x = boundary+fraction*sigma
        phi = mp.matrix([mp.sqrt((1+x)/2)*mp.exp(mp.j*mp.mpf('.3')),
                         mp.sqrt((1-x)/2)*mp.exp(-mp.j*mp.mpf('.7'))])
        values = [mp.re(phi[0]),mp.im(phi[0]),mp.re(phi[1]),mp.im(phi[1])]
        f,m,curvature = evaluate(x)
        eta = f-x*m
        vec = (beta-x*mp.eye(2))*phi
        base = h0+m*beta+eta*mp.eye(2)
        def derivative(u):
            return base*u+2*curvature/norm2(phi)*vec*real_inner(vec,u)
        ff = source(values)
        for i in range(4):
            direction = mp.matrix([0,0])
            direction[i//2] = 1 if i%2==0 else mp.j
            def changed(t):
                updated = values.copy(); updated[i]+=t
                return updated
            first = mp.diff(lambda t:generating(changed(t)),0)
            close(first,2*real_inner(ff,direction),
                  'independent real differentiation retains the complete complex count gradient')
            numeric = mp.matrix([mp.diff(lambda t:source(changed(t))[j],0) for j in range(2)])
            close(mp.norm(numeric-derivative(direction)),0,
                  'independent source differentiation verifies the full real Hessian',mp.mpf('1e-45'))
            gradient = mp.diff(lambda t:squared(changed(t)),0)
            close(gradient,2*real_inner(derivative(ff),direction),
                  'variation of the actual positive square equals twice the full adjoint source',mp.mpf('1e-45'))
        uu=mp.matrix([mp.mpc('.13','.21'),mp.mpc('-.17','.04')])
        vv=mp.matrix([mp.mpc('.09','-.11'),mp.mpc('.07','.19')])
        close(real_inner(uu,derivative(vv)),real_inner(derivative(uu),vv),
              'the complete current-ratio Hessian is real self-adjoint')
        close(mp.norm(derivative(phi)-ff),0,
              'the full smoothed source retains the Euler homogeneity identity')


def row_delta(own,mass,eta):
    dm,de = mass-own['m'],eta-own['eta']
    gap = -dm*own['x']-de
    mismatch = dm*dm+de*de+2*dm*de*own['x']
    return mismatch-2*own['omega']*gap


def aligned_pair(depth, levels, candidates, data, normalization):
    lines,boundaries,sigma,evaluate = data
    one,two = [next(c for c in candidates if c['level']==level) for level in levels]
    power = levels[1]-levels[0]
    bb = mp.mpf(depth-1)/2
    norms = [mp.mpf(1),mp.mpf(1)] if normalization=='unit-norm' else [1/(2*c['omega']) for c in [one,two]]
    coefficients = [norm*mp.rf(mp.mpf(depth+1)/2,level)/mp.factorial(level)
                    for norm,level in zip(norms,levels)]
    ratio_coefficient = coefficients[1]/coefficients[0]
    def x_at(t):
        ratio=ratio_coefficient*t**power
        return (one['x']+ratio*two['x'])/(1+ratio)
    def t_at(x):
        ratio=(one['x']-x)/(x-two['x'])
        return (ratio/ratio_coefficient)**(mp.mpf(1)/power)
    points = [mp.mpf(0),mp.mpf(1)]
    for boundary in boundaries:
        if x_at(1)<boundary<one['x']:
            points.append(t_at(boundary))
    points.sort()
    unsmoothed = mp.mpf(0)
    for low,high in zip(points,points[1:]):
        x = x_at((low+high)/2)
        winner = max(candidates,key=lambda c:c['m']*x+c['eta'])
        for own,level,norm in zip([one,two],levels,norms):
            fraction = mp.betainc(level+1,bb,low,high,regularized=True)
            unsmoothed += norm*row_delta(own,winner['m'],winner['eta'])*fraction
    correction = mp.mpf(0)
    for boundary in boundaries:
        low=max(boundary-sigma,x_at(1)); high=min(boundary+sigma,one['x'])
        if low>=high:
            continue
        def integrand(u):
            x=boundary+sigma*u
            t=t_at(x)
            f,mass,curvature=evaluate(x)
            eta=f-x*mass
            winner=lines[bisect_right(boundaries,x)]
            rho=[coefficient*t**level for coefficient,level in zip(coefficients,levels)]
            change=sum(density*(row_delta(own,mass,eta)-row_delta(own,winner['m'],winner['eta']))
                       for own,density in zip([one,two],rho))
            jacobian=sigma*t*(one['x']-two['x'])/(power*(one['x']-x)*(x-two['x']))
            return change*bb*(1-t)**(bb-1)*jacobian
        u0,u1=(low-boundary)/sigma,(high-boundary)/sigma
        splits=[u0]+([mp.mpf(0)] if u0<0<u1 else [])+[u1]
        correction += mp.quad(integrand,splits)
    delta = unsmoothed+correction
    isolated = sum(2*c['omega']**2*norm for c,norm in zip([one,two],norms))
    complete = isolated+delta
    check(complete>0,'the recalculated smooth pair has positive complete wave energy')
    # Independently check the source-mismatch formula on all traversed charts.
    for selected in candidates:
        if levels[0]<=selected['level']<=levels[1]:
            for own in [one,two]:
                a,b=mp.sqrt((1+own['c'])/2),mp.sqrt((1-own['c'])/2)
                uu=own['h']*a+selected['m']*b+selected['eta']*a
                vv=-own['h']*b+selected['m']*a+selected['eta']*b
                direct=uu*uu+vv*vv-own['omega']**2
                close(direct,row_delta(own,selected['m'],selected['eta']),
                      'independent full two-component source norm verifies the changed pair energy')
    return dict(depth=depth,shell_levels=levels,normalization=normalization,
        norms=[str(n) for n in norms], isolated_complete_energy=str(isolated),
        unsmoothed_contact_energy_difference=str(unsmoothed), smoothing_correction=str(correction),
        smooth_contact_energy_difference=str(delta), complete_energy=str(complete),
        contact_over_isolated_energy=str(delta/isolated),
        aligned_contact_sign='positive' if delta>0 else 'negative' if delta<0 else 'zero',
        second_is_unassigned_adjacent_shell_control=depth==10,
        classification='actual smooth positive-action aligned pair preparation; no radial force inferred')


def main():
    mp.mp.dps=100
    levels={2:(75,80),3:(0,3),4:(2,19),5:(9,14),6:(12,34),10:(22,23)}
    cases=[]
    for depth in wave.envelope.forward.radial.DEPTHS:
        epsilon,candidates,_=wave.envelope.depth_carrier(depth)
        data=smooth_data(candidates)
        full_variation(depth,data[1],data[2],data[3])
        for normalization in ['unit-norm','equal-phase-action']:
            case=aligned_pair(depth,levels[depth],candidates,data,normalization)
            cases.append(case)
            print(json.dumps(dict(depth=depth,normalization=normalization,
                relative_contact=case['contact_over_isolated_energy'])),flush=True)
        for own in candidates:
            if own['level'] in [n-1 for _,d,n in wave.envelope.forward.radial.ASSIGNMENTS if d==depth]:
                f,m,curvature=data[3](own['x'])
                close(row_delta(own,m,f-own['x']*m),0,
                      'all maintained same-branch orthogonal-column pairs retain exactly zero contact')
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    names=['positive-count-contact.py','positive-count-wave.py','convex-count-envelope.py',
           'forward-count-action.py','inverse-current-radial.py','excited-shell-orientation-current.py']
    report=dict(experiment_key='positive-count-full-variation-and-beam-contact',
        created_utc=datetime.now(timezone.utc).isoformat(),
        sources={'experiments/'+name:sha256((ROOT/name).read_bytes()).hexdigest() for name in names},
        evaluations=sum(COUNTS.values()),conditions=COUNTS,maximum_errors={k:str(v) for k,v in ERRORS.items()},cases=cases,
        limits=['the action and its normalization differ from the first-order contact law',
                'aligned prepared energy is not a physical radial force or encounter trajectory',
                'orthogonal field columns eliminate relative-phase dependence in this preparation',
                'the depth-ten second branch is an unassigned mathematical control',
                'the source-variation check is finite algebra; the spatial domain proof remains analytical',
                'no confinement, color law or short-range repulsion is inferred from an energy difference alone'])
    path=ROOT/('positive-count-contact-results-'+stamp+'.json')
    path.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(report=str(path),evaluations=report['evaluations'])),flush=True)


if __name__=='__main__':
    main()
