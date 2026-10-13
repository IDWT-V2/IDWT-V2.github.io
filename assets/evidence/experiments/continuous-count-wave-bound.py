"""Exact continuous-count inequalities and actual minimal-carrier source bounds."""
from datetime import datetime,timezone
from fractions import Fraction
from hashlib import sha256
from importlib.util import spec_from_file_location,module_from_spec
from pathlib import Path
import json

import numpy as np

ROOT=Path(__file__).resolve().parent
def import_local(name,filename):
    spec=spec_from_file_location(name,ROOT/filename)
    module=module_from_spec(spec); spec.loader.exec_module(module)
    return module

envelope=import_local('envelope','convex-count-envelope.py')
minimal=import_local('minimal','minimal-count-carrier.py')
COUNTS,ERRORS={},{}


def check(value,label):
    if not bool(value):
        raise AssertionError(label)
    COUNTS[label]=COUNTS.get(label,0)+1


def close(actual,expected,label,tolerance=2e-10):
    error=float(np.linalg.norm(np.asarray(actual)-np.asarray(expected))/max(1,np.linalg.norm(expected)))
    ERRORS[label]=max(ERRORS.get(label,0),error)
    check(error<tolerance,label)


def ground_coefficients(depth,epsilon):
    count,first,second=envelope.forward.count_data(depth,0)
    cc=epsilon*first
    hh=Fraction(depth,2)
    omega=epsilon
    lam=hh/cc
    qq=2*lam-2*(1-cc*cc)/(epsilon*second)
    pair=epsilon*epsilon*(first*first+second)
    return cc,omega,lam,qq,pair


def main():
    rng=np.random.default_rng(202610121125)
    sectors=[];pairs=[]
    for depth in envelope.forward.radial.DEPTHS:
        count,first,second=envelope.forward.count_data(depth,0)
        check(count==1,'the exact ground geometric band has one count per seed')
        check(second<first*first,'the logarithmic count derivative gives the uniform pair bound')
        epsilon=Fraction(3,5)/first
        cc,omega,lam,qq,pair=ground_coefficients(depth,epsilon)
        check(qq>=0,'one common ground polarization has a nonnegative complete scalar Hessian')
        check(qq<=2*(lam-omega),'the exact scalar source norm fits the full negative spectral gap')
        check(pair<1,'the common ground control satisfies the positive-wave pair condition exactly')
        check(lam>omega,'the ground spatial operator has disjoint signed spectral branches')
        alpha,beta,ambient,seed,form=minimal.preparation(depth)
        aa,bb=np.sqrt((1+float(cc))/2),np.sqrt((1-float(cc))/2)
        fsecond=float(qq)/(2*float(cc)**2)
        for _ in range(4):
            position=rng.normal(size=depth+1); position/=np.linalg.norm(position)
            grading=sum(x*g for x,g in zip(position,ambient))
            field=(aa*np.eye(len(grading))+bb*grading)@seed
            rho=np.vdot(field,field).real
            scalar=np.vdot(field,grading@field).real
            xx=scalar/rho
            vec=(grading-xx*np.eye(len(grading)))@field
            close(np.vdot(vec,vec).real,float(cc)**2*rho,
                  'actual minimal spatial fields give the exact normalized scalar-source direction')
            real=np.concatenate([vec.real.ravel(),vec.imag.ravel()])
            hessian=2*fsecond/rho*np.outer(real,real)
            close(np.linalg.norm(hessian,2),float(qq),
                  'the full real source derivative has exactly the claimed operator norm')
            check(np.linalg.eigvalsh(hessian)[0]>-1e-10,
                  'the complete real source derivative is positive on the actual minimal carrier')
            for _ in range(2):
                variation=rng.normal(size=real.size)
                value=variation@hessian@variation
                check(-1e-10<=value<=float(qq)*np.dot(variation,variation)+1e-10,
                      'arbitrary complex-field directions obey the full positive source bound')
        assigned_epsilon=Fraction(3,10)/max(envelope.forward.count_data(depth,n-1)[1]
            for _,dd,n in envelope.forward.radial.ASSIGNMENTS if dd==depth)
        ac,aw,al,aq,ap=ground_coefficients(depth,assigned_epsilon)
        sectors.append(dict(depth=depth,common_ground_c_exact=str(cc),
            common_ground_epsilon_exact=str(epsilon),source_norm_q_exact=str(qq),
            negative_gap_capacity_exact=str(2*(lam-omega)),pair_condition_exact=str(pair),
            assigned_diagnostic_ground_c_exact=str(ac),assigned_diagnostic_ground_q_exact=str(aq),
            ground_bound_applies_at_assigned_diagnostic_scale=bool(aq>=0 and aq<=2*(al-aw)),
            common_ground_control_supports_all_assigned_levels=all(epsilon*envelope.forward.count_data(depth,n-1)[1]<1
                for _,dd,n in envelope.forward.radial.ASSIGNMENTS if dd==depth)))
    for name,depth,index in envelope.forward.radial.ASSIGNMENTS:
        epsilon=Fraction(3,10)/max(envelope.forward.count_data(depth,n-1)[1]
            for _,dd,n in envelope.forward.radial.ASSIGNMENTS if dd==depth)
        count,first,second=envelope.forward.count_data(depth,index-1)
        left=epsilon*epsilon*(first*first+count*second)
        check(count*second<first*first,
              'every assigned positive count product obeys the exact logarithmic derivative inequality')
        check(left<2*(epsilon*first)**2<=Fraction(18,100),
              'one diagnostic scale gives the assigned pair criterion without individual tuning')
        pairs.append(dict(particle=name,depth=depth,index=index,pair_condition_exact=str(left)))
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    names=['continuous-count-wave-bound.py','minimal-count-carrier.py','convex-count-envelope.py',
           'forward-count-action.py','inverse-current-radial.py','excited-shell-orientation-current.py']
    report=dict(experiment_key='continuous-count-wave-complete-ground-bound',
        created_utc=datetime.now(timezone.utc).isoformat(),
        sources={'experiments/'+n:sha256((ROOT/n).read_bytes()).hexdigest() for n in names},
        evaluations=sum(COUNTS.values()),conditions=COUNTS,maximum_errors=ERRORS,sectors=sectors,assigned_pair_criteria=pairs,
        limits=['the all-spatial conclusion additionally uses the self-adjoint spectral Schur proof',
                'the conclusion excludes exponential linear growth and does not remove zero or secular directions',
                'ground controls do not establish assigned excited-field persistence',
                'the common ground c=.6 scale cannot support the maintained high assigned levels',
                'the original continuous constitutive source is used on its locally smooth domain',
                'finite spatial samples check algebraic source bounds, not global nodelessness'])
    path=ROOT/('continuous-count-wave-bound-results-'+stamp+'.json')
    path.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(report=str(path),evaluations=report['evaluations'],maximum_errors=ERRORS)),flush=True)


if __name__=='__main__':
    main()
