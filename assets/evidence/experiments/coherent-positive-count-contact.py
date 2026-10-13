"""Phase-resolved coherent count beams under the same positive wave action."""
from datetime import datetime, timezone
from hashlib import sha256
from importlib.util import spec_from_file_location,module_from_spec
from pathlib import Path
import json

import mpmath as mp
import numpy as np
from scipy.special import roots_jacobi

ROOT=Path(__file__).resolve().parent
spec=spec_from_file_location('contact',ROOT/'positive-count-contact.py')
contact=module_from_spec(spec)
spec.loader.exec_module(contact)
COUNTS,ERRORS={},{}


def check(condition,label):
    if not bool(condition):
        raise AssertionError(label)
    COUNTS[label]=COUNTS.get(label,0)+1


def close(actual,expected,label,tolerance=2e-9):
    error=float(np.linalg.norm(np.asarray(actual)-np.asarray(expected))/max(1,np.linalg.norm(expected)))
    ERRORS[label]=max(ERRORS.get(label,0),error)
    check(error<tolerance,label)


def coefficients(candidates,data):
    lines,boundaries,sigma,evaluate=data
    masses=np.array([float(c['m']) for c in lines])
    offsets=np.array([float(c['eta']) for c in lines])
    edges=np.array([float(b) for b in boundaries])
    width=float(sigma)
    def source(x):
        indices=np.searchsorted(edges,x,side='right')
        mm,ee=masses[indices].copy(),offsets[indices].copy()
        for shift in [0,-1]:
            nearby=indices+shift
            valid=(nearby>=0)&(nearby<len(edges))
            safe=np.clip(nearby,0,len(edges)-1)
            u=(x-edges[safe])/width
            mask=valid&(abs(u)<1)
            if not np.any(mask):
                continue
            kk=safe[mask]; uu=u[mask]
            cdf=.5+15/16*(uu-2/3*uu**3+uu**5/5)
            moment=15/16*(uu**2/2-uu**4/2+uu**6/6-1/6)
            jump=masses[kk+1]-masses[kk]
            mm[mask]=masses[kk]+jump*cdf
            ee[mask]=offsets[kk]-jump*(edges[kk]*cdf+width*moment)
        return mm,ee
    return source


def matrix_check(depth,one,two):
    gg=contact.wave.envelope.forward.radial.gammas(depth+1)
    size=gg[0].shape[0]
    eye=np.eye(size)
    rotor=gg[1]@gg[2]
    for theta,tt,phase in [(.37,.41,.0),(1.13,.73,.57),(.91,.29,np.pi/2)]:
        position=np.zeros(depth+1)
        position[0]=np.sqrt(1-tt)
        position[1]=np.sqrt(tt)*np.cos(theta)
        position[2]=np.sqrt(tt)*np.sin(theta)
        beta=sum(value*g for value,g in zip(position,gg))
        fields=[];bare=[]
        for own in [one,two]:
            level=own['level']
            cc=float(own['c'])
            aa,bb=np.sqrt((1+cc)/2),np.sqrt((1-cc)/2)
            uu=np.cos(level*theta)*eye+np.sin(level*theta)*rotor
            bare.append(uu/np.sqrt(size))
            fields.append((aa*eye+bb*beta)@uu/np.sqrt(size))
        delta=(two['level']-one['level'])*theta
        predicted_beta=1j*position[0]*np.sin(delta) if depth==2 else 0
        close(np.trace(bare[0].conj().T@bare[1]),np.cos(delta),
              'actual full-matrix spinor traces verify the coherent bare overlap')
        close(np.trace(bare[0].conj().T@beta@bare[1]),predicted_beta,
              'actual Clifford traces retain the exceptional depth-two cross grading')
        ai,bi=np.sqrt((1+float(one['c']))/2),np.sqrt((1-float(one['c']))/2)
        aj,bj=np.sqrt((1+float(two['c']))/2),np.sqrt((1-float(two['c']))/2)
        av,bv=ai*aj+bi*bj,ai*bj+bi*aj
        actual=fields[0]+np.exp(1j*phase)*fields[1]
        predicted_rho=2+2*np.real(np.exp(1j*phase)*(av*np.cos(delta)+bv*predicted_beta))
        predicted_scalar=float(one['x'])+float(two['x'])+2*np.real(np.exp(1j*phase)*(bv*np.cos(delta)+av*predicted_beta))
        close(np.vdot(actual,actual),predicted_rho,
              'direct physical matrix fields verify coherent intensity with relative phase')
        close(np.vdot(actual,beta@actual),predicted_scalar,
              'direct physical matrix fields verify coherent scalar density with relative phase')


def integrate(depth,one,two,source,normalization,phase,order):
    bb=(depth-1)/2
    points,weights=roots_jacobi(order,bb-1,0)
    tt=((points+1)/2)[:,None]
    weights=weights/weights.sum()
    # Integrate one complete period of the difference harmonic. Uniform
    # physical azimuth gives this same measure for every nonzero difference.
    angular_order=order
    theta=2*np.pi*(np.arange(angular_order)+.5)/angular_order
    delta=theta[None,:]
    cc=np.cos(phase)*np.cos(delta)
    qq=-np.sin(phase)*np.sqrt(1-tt)*np.sin(delta) if depth==2 else np.zeros_like(tt+delta)
    norms=[1,1] if normalization=='unit-norm' else [1/(2*float(c['omega'])) for c in [one,two]]
    log_rho=[np.log(norm*float(mp.rf(mp.mpf(depth+1)/2,c['level'])/mp.factorial(c['level'])))+c['level']*np.log(tt)
             for norm,c in zip(norms,[one,two])]
    common_log=np.maximum(*log_rho)
    common_factor=np.exp(common_log)
    rho=[np.exp(value-common_log) for value in log_rho]
    root=np.sqrt(rho[0]*rho[1])
    ai,bi=np.sqrt((1+float(one['c']))/2),np.sqrt((1-float(one['c']))/2)
    aj,bj=np.sqrt((1+float(two['c']))/2),np.sqrt((1-float(two['c']))/2)
    av,bv=ai*aj+bi*bj,ai*bj+bi*aj
    intensity=rho[0]+rho[1]+2*root*(av*cc+bv*qq)
    scalar=float(one['x'])*rho[0]+float(two['x'])*rho[1]+2*root*(bv*cc+av*qq)
    check(np.all(intensity>0),'different-shell coherent matrix preparations have positive sampled density')
    x=scalar/intensity
    check(np.max(abs(x))<1+2e-12,'coherent scalar-to-intensity ratios obey the physical grading bound')
    mass,eta=source(np.clip(x,-1,1))
    uv=[]
    for own,a,b in [(one,ai,bi),(two,aj,bj)]:
        dm=mass-float(own['m']); de=eta-float(own['eta'])
        uv.append((float(own['omega'])*a+dm*b+de*a,
                   float(own['omega'])*b+dm*a+de*b))
    w1,w2=float(one['omega']),float(two['omega'])
    kinetic=w1*w1*rho[0]+w2*w2*rho[1]+2*w1*w2*root*(av*cc+bv*qq)
    squared=sum((u*u+v*v)*density for (u,v),density in zip(uv,rho))
    squared+=2*root*((uv[0][0]*uv[1][0]+uv[0][1]*uv[1][1])*cc+
                    (uv[0][0]*uv[1][1]+uv[0][1]*uv[1][0])*qq)
    check(np.min(squared)>-1e-10*max(1,np.max(squared)),
          'the reconstructed complete coherent source norm is nonnegative')
    integral=lambda value:float(weights@np.mean(common_factor*value,axis=1))
    isolated=sum(2*float(c['omega'])**2*norm for c,norm in zip([one,two],norms))
    close(integral(kinetic),isolated/2,
          'azimuthal integration removes the full distinct-shell kinetic cross term',tolerance=2e-8)
    energy=integral(kinetic+squared)
    return dict(order=order,spatial_nodes=order*angular_order,complete_energy=energy,
                contact_energy=energy-isolated,relative_contact=(energy-isolated)/isolated,
                minimum_sampled_log_density=float(np.min(common_log+np.log(intensity))),
                zero_underflow_energy_weight_nodes=int(np.count_nonzero(common_factor==0)))


def main():
    mp.mp.dps=100
    levels={2:(75,80),3:(0,3),4:(2,19),5:(9,14),6:(12,34),10:(22,23)}
    cases=[]; local_diagnostics=[]
    for depth in contact.wave.envelope.forward.radial.DEPTHS:
        epsilon,candidates,_=contact.wave.envelope.depth_carrier(depth)
        one,two=[next(c for c in candidates if c['level']==level) for level in levels[depth]]
        data=contact.smooth_data(candidates)
        source=coefficients(candidates,data)
        matrix_check(depth,one,two)
        for name,dd,index in contact.wave.envelope.forward.radial.ASSIGNMENTS:
            if dd!=depth:
                continue
            own=next(c for c in candidates if c['level']==index-1)
            values=[contact.row_delta(own,c['m'],c['eta']) for c in candidates if c['level']!=own['level']]
            local_diagnostics.append(dict(particle=name,depth=depth,
                minimum_other_unsmoothed_chart_source_energy_change=str(min(values)),
                negative_other_chart_count=sum(value<0 for value in values)))
        for normalization in ['unit-norm','equal-phase-action']:
            for phase in [0,np.pi/4,np.pi/2]:
                estimates=[integrate(depth,one,two,source,normalization,phase,order) for order in [96,192,384]]
                case=dict(depth=depth,shell_levels=levels[depth],normalization=normalization,
                    relative_complex_phase_radians=phase,quadrature=estimates,
                    finest_relative_change=abs(estimates[-1]['contact_energy']-estimates[-2]['contact_energy'])/max(1,abs(estimates[-1]['contact_energy'])),
                    second_is_unassigned_mathematical_control=depth==10)
                cases.append(case)
                print(json.dumps(dict(depth=depth,normalization=normalization,phase=phase,
                    relative_contacts=[e['relative_contact'] for e in estimates])),flush=True)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    names=['coherent-positive-count-contact.py','positive-count-contact.py','positive-count-wave.py',
           'convex-count-envelope.py','forward-count-action.py','inverse-current-radial.py','excited-shell-orientation-current.py']
    report=dict(experiment_key='phase-resolved-coherent-positive-count-beams',created_utc=datetime.now(timezone.utc).isoformat(),
        sources={'experiments/'+name:sha256((ROOT/name).read_bytes()).hexdigest() for name in names},
        evaluations=sum(COUNTS.values()),conditions=COUNTS,maximum_errors=ERRORS,cases=cases,local_chart_diagnostics=local_diagnostics,
        limits=['full matrix columns are a declared coherent carrier, not the minimum one-column branch',
                'finite quadrature resolution is reported explicitly; positive node density is not a global theorem',
                'the very narrow smoothing bands may require adaptive integration beyond these illustrations',
                'each prepared component retains its own isolated clock; no locking is imposed',
                'phase-dependent prepared energy does not establish a physical radial force or trajectory',
                'depth two retains its exceptional cross-grading trace',
                'depth ten includes an unassigned adjacent-shell control'])
    path=ROOT/('coherent-positive-count-contact-results-'+stamp+'.json')
    path.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(report=str(path),evaluations=report['evaluations'])),flush=True)


if __name__=='__main__':
    main()
