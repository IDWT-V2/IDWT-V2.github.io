"""Actual zonal count-wave Galerkin pencils with exact projected squared source."""
from datetime import datetime,timezone
from fractions import Fraction
from hashlib import sha256
from math import comb,sqrt
from pathlib import Path
import json
import argparse

import numpy as np
from scipy.linalg import solve_triangular,eigvalsh
from scipy.optimize import brentq
from scipy.special import roots_jacobi,eval_gegenbauer

ROOT=Path(__file__).resolve().parent
COUNTS,ERRORS={},{}
ASSIGNMENTS={2:[('W',76),('Z',81),('Higgs',95)],3:[('down',1),('strange',4)],
             4:[('up',3),('charm',20),('top',72)],
             5:[('nu1-comparison',10),('nu2-comparison',15),('nu3-comparison',22)],
             6:[('electron',13),('muon',35)],10:[('tau',23)]}


def check(value,label):
    if not bool(value):
        raise AssertionError(label)
    COUNTS[label]=COUNTS.get(label,0)+1


def close(actual,expected,label,tolerance=2e-9):
    error=float(np.linalg.norm(np.asarray(actual)-np.asarray(expected))/max(1,np.linalg.norm(expected)))
    ERRORS[label]=max(ERRORS.get(label,0),error)
    check(error<tolerance,label)


def count_data(depth,level):
    count=Fraction(comb(level+depth,depth))
    rec=[Fraction(1,level+j) for j in range(1,depth+1)]
    first=count*sum(rec)
    second=count*(sum(rec)**2-sum(v*v for v in rec))
    return count,first,second


def beta(field,t):
    out=np.zeros_like(field)
    out[:,0]=field[:,1]+2*t*field[:,2]
    out[:,1]=field[:,0]+2*t*field[:,3]
    out[:,2]=-field[:,3]
    out[:,3]=-field[:,2]
    return out


def right_gamma(field,t):
    out=np.zeros_like(field)
    out[:,0]=2*t*field[:,1]+field[:,2]
    out[:,1]=-field[:,3]
    out[:,2]=field[:,0]+2*t*field[:,3]
    out[:,3]=-field[:,1]
    return out


def gram_apply(field,t):
    out=field.copy()
    out[:,0]+=t*field[:,3]
    out[:,3]+=t*field[:,0]
    out[:,1]+=t*field[:,2]
    out[:,2]+=t*field[:,1]
    return out


def weighted_field(field,t,weights):
    # Explicit positive trace-Gram square root, without a near-endpoint Cholesky.
    plus=np.sqrt((1+t)/2)*np.sqrt(weights)
    minus=np.sqrt((1-t)/2)*np.sqrt(weights)
    out=np.empty_like(field)
    out[:,0,:]=plus[:,None]*(field[:,0,:]+field[:,3,:])
    out[:,3,:]=minus[:,None]*(field[:,0,:]-field[:,3,:])
    out[:,1,:]=plus[:,None]*(field[:,1,:]+field[:,2,:])
    out[:,2,:]=minus[:,None]*(field[:,1,:]-field[:,2,:])
    return out.reshape(-1,field.shape[-1])


def zonal(depth,level,t):
    aa=(level+depth-1)/(depth-1)*eval_gegenbauer(level,(depth-1)/2,t)
    bb=np.zeros_like(t) if level==0 else eval_gegenbauer(level-1,(depth+1)/2,t)
    norm=sqrt(comb(level+depth-1,depth-1))
    out=np.zeros((len(t),4))
    out[:,0]=(aa-t*bb)/norm;out[:,3]=bb/norm
    return out


def zonal_derivative(depth,level,t):
    bb=np.zeros_like(t) if level==0 else eval_gegenbauer(level-1,(depth+1)/2,t)
    bp=np.zeros_like(t) if level<2 else (depth+1)*eval_gegenbauer(level-2,(depth+3)/2,t)
    ap=(level+depth-1)*bb
    norm=sqrt(comb(level+depth-1,depth-1))
    out=np.zeros((len(t),4));out[:,0]=(ap-bb-t*bp)/norm;out[:,3]=bp/norm
    return out


def beta_derivative(field,derivative,t):
    out=beta(derivative,t);out[:,0]+=2*field[:,2];out[:,1]+=2*field[:,3]
    return out


def right_derivative(field,derivative,t):
    out=right_gamma(derivative,t);out[:,0]+=2*field[:,1];out[:,2]+=2*field[:,3]
    return out


def dirac(field,derivative,t,depth):
    h=depth/2
    out=np.empty_like(field)
    out[:,0]=h*field[:,0]-t*derivative[:,0]+2*(depth-1)*t*field[:,3]-derivative[:,3]
    out[:,1]=-h*field[:,1]-t*derivative[:,1]-derivative[:,2]
    out[:,2]=derivative[:,1]+h*field[:,2]+t*derivative[:,2]
    out[:,3]=derivative[:,0]+(2-h)*field[:,3]+t*derivative[:,3]
    return out


def gram_derivative(field):
    return field[:,[3,2,1,0]]


def preparation(depth,level,epsilon,indices,nodes):
    t,weights=roots_jacobi(nodes,(depth-2)/2,(depth-2)/2)
    weights/=weights.sum()
    count,first,second=count_data(depth,level)
    h=Fraction(2*level+depth,2);c=epsilon*first
    omega=epsilon*count;lam=h/c
    mass2=h*h*(1-c*c)/(c*c)
    q=2*lam-2*(1-c*c)/(epsilon*second)
    check(q>0,'every displayed background has the positive continuous source curvature')
    cc,ww,ll=float(c),float(omega),float(lam)
    aa,bb=sqrt((1+cc)/2),sqrt((1-cc)/2)
    zl=zonal(depth,level,t);bzl=beta(zl,t)
    zlp=zonal_derivative(depth,level,t);bzlp=beta_derivative(zl,zlp,t)
    background=aa*zl+bb*bzl
    direction=-bb*zl+aa*bzl
    directionp=-bb*zlp+aa*bzlp
    rho=np.sum(zl*gram_apply(zl,t),axis=1)
    rhop=2*np.sum(zlp*gram_apply(zl,t),axis=1)+np.sum(zl*gram_derivative(zl),axis=1)
    check(rho.min()>0,'actual zonal background has positive density on all integration nodes')
    close(np.sum(direction*gram_apply(direction,t),axis=1),rho,
          'actual normalized source direction has the same density as its zonal field')
    fields=[];derivatives=[];scores=[];ham=[];ai=[];labels=[];low=[];high=[]
    for kk in indices:
        hh=Fraction(2*kk+depth,2)
        lk=sqrt(float(hh*hh+mass2))
        ck=cc if kk==level else float(hh)/lk
        ak,bk=sqrt((1+ck)/2),sqrt((1-ck)/2)
        zk=zonal(depth,kk,t);bzk=beta(zk,t)
        zkp=zonal_derivative(depth,kk,t);bzkp=beta_derivative(zk,zkp,t)
        close(dirac(zk,zkp,t,depth),float(hh)*zk,
              'actual four-function spatial Dirac formula gives each zonal shell eigenvalue',2e-8)
        positive=ak*zk+bk*bzk;negative=-bk*zk+ak*bzk
        positivep=ak*zkp+bk*bzkp;negativep=-bk*zkp+ak*bzkp
        overlap=np.sum(zl*gram_apply(zk,t),axis=1)
        crossed=np.sum(zl*gram_apply(beta(right_gamma(zk,t),t),t),axis=1)
        cosine=aa*ak+bb*bk
        sine=(cc-ck)/(2*(aa*bk+bb*ak))
        cosine_sum=(cc+ck)/(2*cosine)
        sine_sum=aa*bk+bb*ak
        delta2=hh*hh-h*h
        delta=float(delta2)/(lk+ll)
        plus_value=ww+delta
        plus_ai=delta*float(delta2+4*omega*(lam-omega))/(lk+ll-2*ww)
        minus_value=ww-ll-lk
        minus_ai=(ll+lk)*(ll+lk-2*ww)
        for seed in (0,1):
            for sign in (1,-1):
                field=positive if sign==1 else negative
                derivative=positivep if sign==1 else negativep
                if seed:
                    derivative=right_derivative(field,derivative,t)
                    field=right_gamma(field,t)
                if not seed:
                    score=sine*overlap if sign==1 else cosine*overlap
                else:
                    score=cosine_sum*crossed if sign==1 else -sine_sum*crossed
                measured=np.sum(direction*gram_apply(field,t),axis=1)
                source_metric=np.sqrt(weights/rho)
                close(source_metric*measured,source_metric*score,
                      'stable polarization factors reproduce the independent physical source projection')
                (low if sign==1 else high).append(len(fields))
                fields.append(field);derivatives.append(derivative);scores.append(score)
                ham.append(plus_value if sign==1 else minus_value)
                ai.append(plus_ai if sign==1 else minus_ai)
                labels.append(dict(level=kk,dirac_polarity=sign,right_seed=seed))
    fields=np.stack(fields,axis=-1);derivatives=np.stack(derivatives,axis=-1);scores=np.stack(scores,axis=-1)
    ham=np.array(ham);ai=np.array(ai)
    physical=weighted_field(fields,t,weights)
    close(physical.T@physical,np.eye(len(ham)),
          'actual spatial shell fields are orthonormal in the normalized sphere measure',2e-8)
    nmat=float(q)*(scores.T@(weights[:,None]*scores/rho[:,None]))
    bfields=fields*ham[None,None,:]+float(q)*direction[:,:,None]*scores[:,None,:]/rho[:,None,None]
    factor=weighted_field(bfields,t,weights)
    expected=np.diag(ham*ham)+(ham[:,None]+ham[None,:]+float(q))*nmat
    close(factor.T@factor,expected,
          'direct squared full source agrees with H squared plus HN plus NH plus qN',2e-8)
    low=np.array(low);high=np.array(high)
    qh,rh=np.linalg.qr(factor[:,high],mode='reduced')
    projected=qh.T@factor[:,low]
    response=solve_triangular(rh,projected)
    remainder=factor[:,low]-qh@projected
    gram=rh.T@rh
    check(eigvalsh(gram,subset_by_index=[0,0])[0]>ww*ww,
          'eliminated negative-Dirac block has strictly positive rotating stiffness')
    residual_gram=remainder.T@remainder
    def reduced(ratio):
        sigma2=(ww*ratio)**2
        diagonal=sigma2-ww*ww+4*ww*ww*sigma2/(ai+sigma2)
        dd=diagonal[high]
        correction=dd[:,None]*response
        correction-=dd[:,None]*np.linalg.solve(gram+np.diag(dd),correction)
        value=residual_gram+np.diag(diagonal[low])+response.T@correction
        return (value+value.T)/(2*ww*ww)
    def field_residual(ratio):
        values,vectors=np.linalg.eigh(reduced(ratio))
        ul=vectors[:,np.argmin(abs(values))]
        sigma=ww*ratio;sigma2=sigma*sigma
        diagonal=sigma2-ww*ww+4*ww*ww*sigma2/(ai+sigma2)
        leading=response@ul
        uh=-leading+np.linalg.solve(gram+np.diag(diagonal[high]),diagonal[high]*leading)
        real=np.zeros(len(ham));real[low]=ul;real[high]=uh
        imag=2*ww*sigma*real/(ai+sigma2)
        norm=sqrt(real@real+imag@imag);real/=norm;imag/=norm
        variation=fields@real;variationp=derivatives@real
        score=scores@real
        scorep=(np.sum(directionp*gram_apply(variation,t),axis=1)+
                np.sum(direction*gram_apply(variationp,t),axis=1)+
                np.sum(direction*gram_derivative(variation),axis=1))
        source=float(q)*direction*score[:,None]/rho[:,None]
        sourcep=float(q)*(directionp*score[:,None]/rho[:,None]+
                         direction*(scorep/rho-score*rhop/(rho*rho))[:,None])
        hsource=dirac(source,sourcep,t,depth)+sqrt(float(mass2))*beta(source,t)+(ww-ll)*source
        second_score=scores@(ham*real)
        nh=float(q)*direction*second_score[:,None]/rho[:,None]
        mixed=hsource+nh+float(q)*source
        weighted_mixed=weighted_field(mixed[:,:,None],t,weights)[:,0]
        outside=weighted_mixed-physical@(physical.T@weighted_mixed)
        full=weighted_mixed+physical@((ai+sigma2)*real+2*ww*sigma*imag)
        return dict(full_spatial_residual_over_clock_squared=float(np.linalg.norm(full)/(ww*ww)),
                    omitted_spatial_residual_over_clock_squared=float(np.linalg.norm(outside)/(ww*ww)),
                    normalized_real_variation_norm=float(np.linalg.norm(real)),
                    normalized_imaginary_variation_norm=float(np.linalg.norm(imag)))
    return dict(reduced=reduced,ai=ai,omega=ww,epsilon=epsilon,c=c,q=q,
                field_residual=field_residual,
                nodes=nodes,indices=indices,labels=labels,
                minimum_sampled_density=float(rho.min()),
                factor_residual_relative=float(np.linalg.norm(qh.T@remainder)/max(1,np.linalg.norm(remainder))))


def pencil_roots(prepared,maximum_only=False):
    ww=prepared['omega'];ai=prepared['ai'];fun=prepared['reduced']
    poles=sorted(set(sqrt(-value)/ww for value in ai if value<0))
    start=1e-4
    poles=[p for p in poles if start<p<1.000001]
    boundaries=[start,*poles,1.001]
    initial=eigvalsh(fun(start))
    remaining=sum(1 for value in ai if value<-(ww*start)**2)
    initial_negative=int(np.sum(initial<0))
    roots=[]
    intervals=range(len(boundaries)-1)
    if maximum_only:
        intervals=reversed(list(intervals))
    for i in intervals:
        left,right=boundaries[i:i+2]
        margin=(right-left)*1e-5
        left+=margin if i>0 else 0
        right-=margin if i<len(boundaries)-2 else 0
        locations=np.linspace(left,right,7)
        for first,second in zip(locations[:-1],locations[1:]):
            va,vb=eigvalsh(fun(first)),eigvalsh(fun(second))
            na,nb=int(np.sum(va<0)),int(np.sum(vb<0))
            if nb>na:
                for index in range(na,nb):
                    root=brentq(lambda rr:float(eigvalsh(fun(rr),subset_by_index=[index,index])[0]),
                                first,second,xtol=1e-12)
                    residual=float(min(abs(eigvalsh(fun(root)))))
                    roots.append(dict(growth_over_count_clock=root,
                                      dimensional_growth=ww*root,reduced_pencil_residual=residual,
                                      **prepared['field_residual'](root)))
        if maximum_only and roots:
            break
    return dict(real_growing_roots=roots,
                initial_reduced_negative_index=initial_negative,
                remaining_negative_imaginary_stiffness_poles=remaining,
                initial_inertia_margin=float(min(abs(initial))),
                inertia_count_forces_real_growth=remaining>initial_negative,
                positive_growth_detected=bool(roots),
                maximum_growth_over_count_clock=max([v['growth_over_count_clock'] for v in roots],default=0),
                pole_ratios=poles)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--full-offset',type=int)
    parser.add_argument('--full-factor',type=int,default=1)
    parser.add_argument('--quadrature-multiplier',type=int,default=4)
    parser.add_argument('--only',nargs='*')
    args=parser.parse_args()
    cases=[]
    for depth,assignments in ASSIGNMENTS.items():
        epsilon=Fraction(3,10)/max(count_data(depth,index-1)[1] for _,index in assignments)
        jobs=[(name,index-1,epsilon) for name,index in assignments]
        jobs.append(('common-ground-control',0,Fraction(3,5)/count_data(depth,0)[1]))
        for name,level,eps in jobs:
            if args.only and name not in args.only:
                continue
            indices=(list(range(args.full_factor*level+args.full_offset+1)) if args.full_offset is not None else
                     sorted(set(range(min(level+5,13)))|set(range(max(0,level-3),level+5))))
            nodes=max(128,args.quadrature_multiplier*(max(indices)+level+depth+5))
            prepared=preparation(depth,level,eps,indices,nodes)
            outcome=pencil_roots(prepared,maximum_only=args.full_offset is not None)
            case=dict(particle=name,depth=depth,background_level=level,
                      epsilon_exact=str(eps),polarization_exact=str(prepared['c']),
                      source_norm_q_exact=str(prepared['q']),count_clock=prepared['omega'],
                      shell_indices=indices,quadrature_nodes=nodes,
                      minimum_sampled_density=prepared['minimum_sampled_density'],
                      qr_orthogonality_relative=prepared['factor_residual_relative'],**outcome)
            cases.append(case)
            print(json.dumps(dict(particle=name,depth=depth,growth_ratio=outcome['maximum_growth_over_count_clock'],
                                  finite_inertia_count=outcome['inertia_count_forces_real_growth'])),flush=True)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    report=dict(experiment_key='continuous-count-assigned-zonal-spatial-pencils',
                created_utc=datetime.now(timezone.utc).isoformat(),
                sources={'experiments/'+Path(__file__).name:sha256(Path(__file__).read_bytes()).hexdigest()},
                evaluations=sum(COUNTS.values()),conditions=COUNTS,maximum_errors=ERRORS,cases=cases,
                comparison_options=vars(args),
                limits=['finite true Galerkin pencils are not full infinite-spectrum proofs',
                        'the full matrix zonal carrier differs from minimum-column particle candidates',
                        'pointwise squared source is projected before forming the finite wave stiffness',
                        'rationalized signed eigenvalues and QR elimination avoid heavy/light cancellation',
                        'cutoff, quadrature and precision refinement remain required for detected growth',
                        'no particle lifetime or observed decay channel follows from a linear growth rate',
                        'the smoothed envelope law is a different action with its own persistence theorem'])
    path=ROOT/('continuous-count-excited-spatial-results-'+stamp+'.json')
    path.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(report=str(path),evaluations=report['evaluations'],maximum_errors=ERRORS)),flush=True)


if __name__=='__main__':
    main()
