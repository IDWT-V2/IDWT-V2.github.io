"""Ground harmonic reduction of the original continuous count wave action."""
from datetime import datetime,timezone
from fractions import Fraction
from hashlib import sha256
from importlib.util import spec_from_file_location,module_from_spec
from pathlib import Path
import json

import mpmath as mp
import numpy as np
from scipy.linalg import cholesky,eigvals

ROOT=Path(__file__).resolve().parent
spec=spec_from_file_location('wave',ROOT/'positive-count-wave.py')
wave=module_from_spec(spec)
spec.loader.exec_module(wave)
COUNTS={}


def check(value,label):
    if not bool(value):
        raise AssertionError(label)
    COUNTS[label]=COUNTS.get(label,0)+1


def harmonic_block(depth,cc,degree):
    h=depth/2
    hd=sum(Fraction(1,j) for j in range(1,depth+1))
    second=hd*hd-sum(Fraction(1,j*j) for j in range(1,depth+1))
    epsilon=cc/float(hd)
    xx=np.sqrt(1-cc*cc)
    omega=epsilon
    mass=h*xx/cc
    eta=omega-h/cc
    curvature_q=2*h/cc-2*xx*xx/(epsilon*float(second))
    aa,bb=np.sqrt((1+cc)/2),np.sqrt((1-cc)/2)
    ll=degree
    dd=np.array([[h+ll,0,0,0],[0,-h-ll,0,0],
                 [0,1,h+ll-1,0],[-1,0,0,-h-ll+1]],float)
    beta=np.array([[0,1,0,0],[1,0,0,0],[0,0,0,1],[0,0,1,0]],float)
    gram=np.array([[1,0,0,ll],[0,1,ll,0],
                   [0,ll,ll*(2*ll+depth-1),0],
                   [ll,0,0,ll*(2*ll+depth-1)]],float)
    ham=dd+mass*beta+eta*np.eye(4)
    pp=np.array([-bb,aa,0,0]); rr=gram@pp
    br=ham+curvature_q*np.outer(pp,rr)
    check(np.linalg.norm(gram@ham-ham.T@gram)<1e-10*max(1,np.linalg.norm(gram@ham)),
          'the derived scalar harmonic Dirac block is self-adjoint in its physical Gram metric')
    check(np.linalg.norm(gram@br-br.T@gram)<1e-10*max(1,np.linalg.norm(gram@br)),
          'the complete continuous scalar source derivative is real self-adjoint')
    change=cholesky(gram,lower=False)
    inverse=np.linalg.inv(change)
    ham=change@ham@inverse
    br=change@br@inverse
    eye=np.eye(4); zero=np.zeros((4,4))
    real_stiffness=br@br-omega*omega*eye
    imaginary_stiffness=ham@ham-omega*omega*eye
    generator=np.block([[zero,zero,eye,zero],[zero,zero,zero,eye],
                        [-real_stiffness,zero,zero,-2*omega*eye],
                        [zero,-imaginary_stiffness,2*omega*eye,zero]])
    spectrum=eigvals(generator)
    growth=float(max(np.real(spectrum)))
    # A near-zero root may be a symmetry mode; report its numerical scale.
    values=np.linalg.eigvalsh(real_stiffness)
    secular=1+curvature_q*float(rr@np.linalg.solve(dd+mass*beta+(eta+omega)*np.eye(4),pp))
    return dict(depth=depth,ground_polarization_c=cc,scalar_harmonic_degree=degree,
        curvature_coefficient_q=curvature_q,maximum_numerical_growth=growth,
        minimum_real_rotating_stiffness=float(values[0]),
        secular_value_at_negative_count_frequency=secular,
        numerical_frequency_generator_eigenvalues=[[float(v.real),float(v.imag)] for v in spectrum])


def main():
    mp.mp.dps=100
    pairs=[];cases=[]
    for name,depth,index in wave.envelope.forward.radial.ASSIGNMENTS:
        epsilon,candidates,_=wave.envelope.depth_carrier(depth)
        count,first,second=wave.envelope.forward.count_data(depth,index-1)
        condition=epsilon*epsilon*(first*first+count*second)
        check(condition<1,'every maintained continuous count pair passes the exact positive-wave pair criterion')
        pairs.append(dict(particle=name,depth=depth,index=index,
                          exact_pair_criterion_left_side=str(condition),positive_pair_margin_exact=str(1-condition)))
    for depth in wave.envelope.forward.radial.DEPTHS:
        for cc in [.3,.6]:
            for degree in [1,2,3,5,10,30,100]:
                case=harmonic_block(depth,cc,degree)
                cases.append(case)
            print(json.dumps(dict(depth=depth,ground_c=cc,
                first_harmonic=cases[-7]['maximum_numerical_growth'],
                minimum_secular=min(c['secular_value_at_negative_count_frequency'] for c in cases[-7:]))),flush=True)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    names=['continuous-count-wave-ground.py','positive-count-wave.py','convex-count-envelope.py',
           'forward-count-action.py','inverse-current-radial.py','excited-shell-orientation-current.py']
    report=dict(experiment_key='continuous-count-positive-wave-ground-harmonics',
        created_utc=datetime.now(timezone.utc).isoformat(),
        sources={'experiments/'+n:sha256((ROOT/n).read_bytes()).hexdigest() for n in names},
        evaluations=sum(COUNTS.values()),conditions=COUNTS,assigned_pair_criteria=pairs,cases=cases,
        limits=['ground spatial controls are distinct from assigned excited-field persistence',
                'the four-field closure and full harmonic coverage require analytical verification',
                'numerical roots near zero do not prove growth or exclude secular symmetry modes',
                'the raw continuous count law has no envelope convolution in this calculation',
                'finite harmonic degrees are illustrations until an all-degree criterion is established'])
    path=ROOT/('continuous-count-wave-ground-results-'+stamp+'.json')
    path.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(report=str(path),evaluations=report['evaluations'])),flush=True)


if __name__=='__main__':
    main()
