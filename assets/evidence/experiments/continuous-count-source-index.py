"""Exact finite source projections certify a full count-wave instability index."""
from datetime import datetime,timezone
from fractions import Fraction
from functools import lru_cache
from hashlib import sha256
from math import comb
from pathlib import Path
import json

import mpmath as mp
import numpy as np

ROOT=Path(__file__).resolve().parent
COUNTS={}
ASSIGNMENTS={2:[('W',76),('Z',81),('Higgs',95)],3:[('down',1),('strange',4)],
             4:[('up',3),('charm',20),('top',72)],
             5:[('nu1-comparison',10),('nu2-comparison',15),('nu3-comparison',22)],
             6:[('electron',13),('muon',35)],10:[('tau',23)]}


def check(value,label):
    if not bool(value):raise AssertionError(label)
    COUNTS[label]=COUNTS.get(label,0)+1


def iv(value):
    if isinstance(value,Fraction):return mp.iv.mpf(value.numerator)/value.denominator
    return mp.iv.mpf(value)


def lo(value):return mp.mpf(value.a)
def hi(value):return mp.mpf(value.b)


def add(a,b):
    return tuple((a[i] if i<len(a) else Fraction(0))+(b[i] if i<len(b) else Fraction(0))
                 for i in range(max(len(a),len(b))))


def scale(a,c):return tuple(v*c for v in a)


def mul(a,b):
    out=[Fraction(0)]*(len(a)+len(b)-1)
    for i,aa in enumerate(a):
        for j,bb in enumerate(b):out[i+j]+=aa*bb
    return tuple(out)


@lru_cache(None)
def gegenbauer(degree,parameter):
    if degree<0:return (Fraction(0),)
    if degree==0:return (Fraction(1),)
    if degree==1:return (Fraction(0),2*parameter)
    a=scale(mul((Fraction(0),Fraction(1)),gegenbauer(degree-1,parameter)),
            2*(degree-1+parameter)/degree)
    b=scale(gegenbauer(degree-2,parameter),-Fraction(degree-2+2*parameter,degree))
    return add(a,b)


@lru_cache(None)
def moment(depth,power):
    if power%2:return Fraction(0)
    out=Fraction(1)
    for k in range(1,power//2+1):out*=Fraction(2*k-1,depth+2*k-1)
    return out


def integral(poly,depth):
    return sum((value*moment(depth,i) for i,value in enumerate(poly)),Fraction(0))


@lru_cache(None)
def zonal_polynomials(depth,level):
    aa=scale(gegenbauer(level,Fraction(depth-1,2)),Fraction(level+depth-1,depth-1))
    bb=gegenbauer(level-1,Fraction(depth+1,2))
    return add(aa,scale(mul((Fraction(0),Fraction(1)),bb),-1)),bb


@lru_cache(None)
def overlap_polynomials(depth,level,other):
    p,b=zonal_polynomials(depth,level);q,c=zonal_polynomials(depth,other)
    pq=mul(p,q);bc=mul(b,c);cross=add(mul(p,c),mul(b,q))
    uu=add(add(pq,bc),mul((Fraction(0),Fraction(1)),cross))
    vv=add(add(mul((Fraction(0),Fraction(1)),pq),
               mul((Fraction(-1),Fraction(0),Fraction(2)),cross)),
           mul((Fraction(0),Fraction(-3),Fraction(0),Fraction(4)),bc))
    return uu,vv


def count_data(depth,level):
    count=Fraction(comb(level+depth,depth));rec=[Fraction(1,level+j) for j in range(1,depth+1)]
    return count,count*sum(rec),count*(sum(rec)**2-sum(v*v for v in rec))


def parameters(depth,level,curvature=Fraction(1)):
    epsilon=Fraction(3,10)/max(count_data(depth,n-1)[1] for _,n in ASSIGNMENTS[depth])
    count,first,second=count_data(depth,level)
    h=Fraction(2*level+depth,2);c=epsilon*first;lam=h/c;omega=epsilon*count
    mass2=h*h*(1-c*c)/(c*c);q=curvature*(2*lam-2*(1-c*c)/(epsilon*second))
    h0=Fraction(depth,2)
    return dict(epsilon=epsilon,c=c,lam=lam,omega=omega,mass2=mass2,q=q,h=h,
                positive_branch_above_negative_count_gap=h0*h0-h*h+4*omega*(lam-omega),
                negative_branch_gap_lower=2*lam-q-2*omega-(h*h-h0*h0)/lam)


def transpose(a):return [list(row) for row in zip(*a)]


def matmul(a,b):
    bt=transpose(b)
    return [[sum((x*y for x,y in zip(row,col)),iv(0)) for col in bt] for row in a]


def inverse(a):
    n=len(a);aug=[row.copy()+[iv(int(i==j)) for j in range(n)] for i,row in enumerate(a)]
    for i in range(n):
        pivot=aug[i][i]
        check(lo(pivot)>0 or hi(pivot)<0,'interval Gaussian pivot excludes zero')
        aug[i]=[v/pivot for v in aug[i]];aug[i][i]=iv(1)
        for j in range(n):
            if i==j:continue
            factor=aug[j][i]
            aug[j]=[x-factor*y for x,y in zip(aug[j],aug[i])];aug[j][i]=iv(0)
    return [row[n:] for row in aug]


def submatrix(a,rows,cols):return [[a[i][j] for j in cols] for i in rows]


def congruence_certificate(matrix):
    n=len(matrix)
    midpoint=np.array([[float((lo(v)+hi(v))/2) for v in row] for row in matrix])
    values,vectors=np.linalg.eigh((midpoint+midpoint.T)/2)
    transform=[[iv(Fraction.from_float(float(vectors[i,j]))) for j in range(n)] for i in range(n)]
    metric=matmul(transpose(transform),transform)
    error=max(sum(hi(abs(metric[i][j]-int(i==j))) for j in range(n)) for i in range(n))
    check(error<mp.mpf('.5'),'recorded rational congruence matrix is rigorously invertible')
    diagonalized=matmul(matmul(transpose(transform),matrix),transform)
    rows=[];negative=0
    for i in range(n):
        radius=sum(hi(abs(diagonalized[i][j])) for j in range(n) if i!=j)
        lower,upper=lo(diagonalized[i][i]),hi(diagonalized[i][i])
        if lower>radius:sign=1;margin=lower-radius
        elif upper<-radius:sign=-1;margin=-upper-radius;negative+=1
        else:raise AssertionError('congruence does not yet give strict signed row dominance')
        check(margin>0,'interval signed row dominance certifies the exact inertia')
        rows.append(dict(diagonal_lower=str(lower),diagonal_upper=str(upper),
                         off_diagonal_radius_upper=str(radius),sign=sign,margin_lower=str(margin)))
    return dict(negative=negative,positive=n-negative,
                transform_gram_error_upper=str(error),
                minimum_signed_margin_lower=min((r['margin_lower'] for r in rows),key=mp.mpf),
                rows=rows,
                transform_binary_rationals=[[str(Fraction.from_float(float(vectors[i,j]))) for j in range(n)] for i in range(n)],
                floating_eigenvalues_for_orientation_only=values.tolist())


def source_certificate(depth,level,degree,curvature=Fraction(1),require_growth=True):
    pars=parameters(depth,level,curvature)
    check(pars['positive_branch_above_negative_count_gap']>0,
          'all positive-Dirac branches lie strictly above minus the count clock')
    check(pars['negative_branch_gap_lower']>0,
          'the complete positive source cannot move a negative branch into the count gap')
    check(pars['q']>0,'source projection is a genuine positive lower operator')
    omega=iv(pars['omega']);cl=iv(pars['c']);lam=iv(pars['lam']);q=iv(pars['q'])
    al=mp.iv.sqrt((1+cl)/2);bl=mp.iv.sqrt((1-cl)/2)
    gl=comb(level+depth-1,depth-1)
    rho=scale(overlap_polynomials(depth,level,level)[0],Fraction(1,gl))
    polys=[gegenbauer(j,Fraction(depth-1,2)) for j in range(degree+1)]
    gram_exact=[[integral(mul(mul(rho,a),b),depth) for b in polys] for a in polys]
    check(gram_exact[0][0]==1,'the original source direction has exact unit spatial norm')
    gram=[[iv(v) for v in row] for row in gram_exact]
    gram_inverse=inverse(gram)
    cutoff=level+degree+2
    coefficients=[];diagonal=[];positive=[];negative=[];labels=[];selected=None
    for kk in range(cutoff+1):
        hk=Fraction(2*kk+depth,2);lk=mp.iv.sqrt(iv(hk*hk+pars['mass2']))
        ck=cl if kk==level else iv(hk)/lk
        ak=mp.iv.sqrt((1+ck)/2);bk=mp.iv.sqrt((1-ck)/2)
        cosine=al*ak+bl*bk;sine=(cl-ck)/(2*(al*bk+bl*ak))
        cosine_sum=(cl+ck)/(2*cosine);sine_sum=al*bk+bl*ak
        uu,vv=overlap_polynomials(depth,level,kk)
        norm=mp.iv.sqrt(iv(gl*comb(kk+depth-1,depth-1)))
        iu=[iv(integral(mul(uu,p),depth))/norm for p in polys]
        ivv=[iv(integral(mul(vv,p),depth))/norm for p in polys]
        for seed in (0,1):
            for sign in (1,-1):
                if not seed:row=[(sine if sign==1 else cosine)*v for v in iu]
                else:row=[(cosine_sum if sign==1 else -sine_sum)*v for v in ivv]
                if kk==level and seed==0 and sign==1:
                    selected=len(labels);row=[iv(0) for _ in row]
                (positive if sign==1 else negative).append(len(labels))
                coefficients.append(row)
                diagonal.append(iv(hk*hk-pars['h']**2)/(lk+lam) if sign==1 else -lam-lk)
                labels.append(dict(level=kk,polarity=sign,right_seed=seed))
    lower=matmul(matmul(coefficients,gram_inverse),transpose(coefficients))
    lower=[[q*value+(diagonal[i] if i==j else 0) for j,value in enumerate(row)]
           for i,row in enumerate(lower)]
    negative_block=submatrix(lower,negative,negative)
    correction=matmul(matmul(submatrix(lower,positive,negative),inverse(negative_block)),
                      submatrix(lower,negative,positive))
    schur=submatrix(lower,positive,positive)
    schur=[[(schur[i][j]-correction[i][j])/omega for j in range(len(positive))]
           for i in range(len(positive))]
    selected_positive=positive.index(selected)
    keep=[i for i in range(len(positive)) if i!=selected_positive]
    certificate=congruence_certificate(submatrix(schur,keep,keep))
    imaginary_negative=2*level
    growth=certificate['negative']<imaginary_negative
    if require_growth:
        check(growth,'full-operator gyro pole count exceeds the certified possible nonpositive real directions')
    for kk in (cutoff+1,cutoff+2):
        uu,vv=overlap_polynomials(depth,level,kk)
        check(all(integral(mul(v,p),depth)==0 for v in (uu,vv) for p in polys),
              'exact polynomial degrees make the source projection vanish beyond the stated finite support')
    return dict(depth=depth,background_level=level,source_polynomial_degree=degree,
                finite_dirac_cutoff=cutoff,exact_parameters={k:str(v) for k,v in pars.items()},
                exact_source_gram=[[str(v) for v in row] for row in gram_exact],
                schur_inertia_certificate=certificate,
                imaginary_negative_index=imaginary_negative,
                real_nonpositive_index_upper_excluding_selected_phase=certificate['negative'],
                gyroscopic_index_deficit=imaginary_negative-certificate['negative'],
                full_spatial_real_growth_established=growth,
                growth_rate_interval=('0 < sigma <= omega; no sharper rate certified' if growth else
                                      'this sufficient index test is inconclusive'))


def main():
    mp.mp.dps=100;mp.iv.dps=85
    controls=[]
    for depth,assignments in ASSIGNMENTS.items():
        for name,index in assignments:
            pars=parameters(depth,index-1)
            check(pars['negative_branch_gap_lower']>0,
                  'same exact source norm bound retains the remote negative gap at every assignment')
            controls.append(dict(particle=name,depth=depth,index=index,
                                 positive_branch_above_minus_clock=pars['positive_branch_above_negative_count_gap']>0,
                                 exact_positive_branch_margin=str(pars['positive_branch_above_negative_count_gap']),
                                 exact_negative_branch_margin=str(pars['negative_branch_gap_lower']),
                                 certificate_status='evaluated separately; applicability is not itself an instability proof'))
    certificates=[]
    for depth,level,degree in ((3,3,12),(4,2,6)):
        result=source_certificate(depth,level,degree);certificates.append(result)
        print(json.dumps(dict(depth=depth,level=level,negative_upper=result['real_nonpositive_index_upper_excluding_selected_phase'],
                              imaginary_negative=result['imaginary_negative_index'],full_spatial_growth=True)),flush=True)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    report=dict(experiment_key='continuous-count-source-projection-full-spatial-index',
                created_utc=datetime.now(timezone.utc).isoformat(),
                sources={'experiments/'+Path(__file__).name:sha256(Path(__file__).read_bytes()).hexdigest()},
                evaluations=sum(COUNTS.values()),conditions=COUNTS,
                occupied_assignment_controls=controls,certificates=certificates,
                interval_decimal_precision=85,
                limits=['proof uses the source-projection and gyroscopic spectral-flow arguments in the companion note',
                        'the actual full-matrix zonal carrier differs from minimum-column particle candidates',
                        'only the two displayed excited backgrounds receive a full spatial certificate here',
                        'the index deficit certifies existence; no precise growth multiplicity is asserted',
                        'finite source support is exact polynomial algebra, not a truncation assumption',
                        'no empirical particle decay or lifetime is identified',
                        'the envelope source remains a separate action with a separate persistence theorem'])
    path=ROOT/('continuous-count-source-index-results-'+stamp+'.json')
    path.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(report=str(path),evaluations=report['evaluations'])),flush=True)


if __name__=='__main__':main()
