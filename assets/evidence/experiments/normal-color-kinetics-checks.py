#!/usr/bin/env python3
"""Test independent normal color against the same nested spatial Clifford flag."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
from itertools import combinations, permutations
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import sympy as sp


HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('spinor_source', HERE/'spinor-hodge-source-checks.py')
source = importlib.util.module_from_spec(spec)
spec.loader.exec_module(source)
CHECKS = 0
CASES = []


def check(value, message):
    global CHECKS
    if not value:
        raise AssertionError(message)
    CHECKS += 1


def close(a, b, message, atol=1e-10):
    check(np.allclose(a,b,atol=atol,rtol=1e-10),message)


def su_generators(k):
    result = []
    for a,b in combinations(range(k),2):
        g = np.zeros((k,k),complex); g[a,b]=1; g[b,a]=-1; result.append(g)
        g = np.zeros((k,k),complex); g[a,b]=1j; g[b,a]=1j; result.append(g)
    for a in range(k-1):
        g=np.zeros((k,k),complex); g[a,a]=1j; g[a+1,a+1]=-1j; result.append(g)
    return result


def constraints(columns):
    matrix=np.column_stack(columns)
    matrix=np.rint(np.vstack((matrix.real,matrix.imag))*2).astype(int)
    matrix=matrix[np.any(matrix!=0,axis=1)]
    return sp.Matrix(matrix)


def realify_interleaved(matrix):
    k=len(matrix)
    result=np.zeros((2*k,2*k))
    for a in range(k):
        for b in range(k):
            z=matrix[a,b]
            result[2*a:2*a+2,2*b:2*b+2]=[[z.real,-z.imag],[z.imag,z.real]]
    return result


def data_for_parent(k):
    m=k+2
    gamma=[source.dense(g) for g in source.gamma_monomials(m)]
    size=len(gamma[0])
    create=[(gamma[2*j]-1j*gamma[2*j+1])/2 for j in range(m)]
    destroy=[g.conj().T for g in create]
    algebra=su_generators(k)
    lifted=[sum(a[i,j]*create[i+2]@destroy[j+2] for i in range(k) for j in range(k))
            for a in algebra]
    for a,x in zip(algebra,lifted):
        close(x.conj().T,-x,'normal SU generators are skew Hermitian on the full spatial field')
        ordinary=np.zeros((2*m,2*m)); ordinary[4:,4:]=realify_interleaved(a)
        for i in range(2*m):
            close(x@gamma[i]-gamma[i]@x,sum(ordinary[j,i]*gamma[j] for j in range(2*m)),
                  'normal color generator is an actual ordinary spatial rotation')
    # One full normal exterior grade is fundamental or conjugate fundamental.
    grade=k-1
    keep=np.array([j for j in range(size) if (j&((1<<k)-1)).bit_count()==grade])
    projector=np.zeros((size,size)); projector[keep,keep]=1
    check(len(keep)==4*k,'triplet/fundamental projection retains a full four-dimensional Dirac factor')
    for x in lifted:
        close(projector@x,x@projector,'normal exterior grade is preserved by its SU group')
    return gamma,algebra,lifted,keep,projector


def kinetic_checks(k):
    gamma,algebra,lifted,keep,p=data_for_parent(k)
    size=len(gamma[0]); d_parent=len(gamma)
    for d in range(2,d_parent+1):
        columns=[np.concatenate([(x@g-g@x).reshape(-1) for g in gamma[:d]]) for x in lifted]
        null=constraints(columns).nullspace() if columns else []
        expected=(k*k-1 if d<=4 else max(k-math.ceil((d-4)/2),0)**2-1)
        expected=max(expected,0)
        check(len(null)==expected,'normal SU commutant is fixed by the number of active normal complex planes')
        for vector in null:
            x=sum(float(c)*g for c,g in zip(vector,lifted))
            for g in gamma[:d]:
                close(x@g,g@x,'surviving independent color commutes with every tangent Clifford vector')
        invariance=[]
        compressed=[]
        for i,g in enumerate(gamma[:d]):
            residual=np.linalg.norm((np.eye(size)-p)@g@p)**2
            invariance.append(residual)
            cg=g[np.ix_(keep,keep)]
            compressed.append(cg)
            if i<4:
                close(residual,0,'normal triplet retains each base kinetic direction')
                close(cg@cg,np.eye(len(keep)),'compressed base kinetic symbol has the original spatial metric')
            else:
                close(cg,0,'fixed normal chirality compression kills an added normal kinetic direction')
                close(residual,len(keep),'all normal-triplet amplitude leaks under an added normal Clifford vector')
        for a,b in combinations(range(min(d,4)),2):
            close(compressed[a]@compressed[b]+compressed[b]@compressed[a],0,
                  'full independent-color base symbol retains Clifford anticommutators')
        if k==3:
            CASES.append({'real_depth': d,'geometry':'one flat nested flag in the selected ten-dimensional parent',
                          'parameters':{'parent_depth':10,'normal_complex_rank':3,'projected_full_spinor_rank':len(keep)},
                          'measurements':{'independent_normal_su3_subalgebra_dimension':len(null),
                                          'full_triplet_kinetic_compatibility':d<=4,
                                          'retained_spatial_symbol_directions':min(d,4),
                                          'squared_leak_by_tangent_direction':invariance,
                                          'spatial_three_form_can_be_nonzero':d>=3,
                                          'combined_kinetic_and_three_form_candidate':3<=d<=4,
                                          'all_color_interactions_proved_absent':False}})
    return gamma,algebra,lifted,keep,p


def actual_connection_and_pair(gamma,algebra,lifted,keep):
    # Select one fixed four-dimensional Dirac spin state and all normal grade-two components.
    normal_keep=np.array([j for j in keep if j>>3==0])
    color=[x[np.ix_(normal_keep,normal_keep)] for x in lifted]
    check(len(normal_keep)==3,'one base spinor component contains an independent color triplet')
    base_gamma=[source.dense(g) for g in source.gamma_monomials(2)]
    for g,cg in zip(gamma[:4],base_gamma):
        close(g[np.ix_(keep,keep)],np.kron(cg,np.eye(3)),
              'actual projected ambient spinor factors into full base Dirac spin and independent color')
    for x,c in zip(lifted,color):
        close(x[np.ix_(keep,keep)],np.kron(np.eye(4),c),
              'normal color generators act as identity on every observer spin component')
    close(sum(x@x for x in color[:6]),np.diag([-4.,-4.,-4.]),
          'off-diagonal color generators act in the usual three-dimensional exterior representation')
    # Complete orthonormal Hermitian generators for the actual projected normal representation.
    hermitian=[1j*x for x in color]
    normalized=[]
    for candidate in hermitian:
        value=candidate.copy()
        for previous in normalized:
            value-=2*np.trace(previous@value).real*previous
        norm=np.trace(value@value).real
        check(norm>0,'independent normal generators survive in the triplet')
        normalized.append(value/math.sqrt(2*norm))
    for a,t in enumerate(normalized):
        close(np.trace(t),0,'normal triplet color generators are traceless')
        for b,u in enumerate(normalized):
            close(np.trace(t@u),.5 if a==b else 0,'normal color uses the empirical Casimir convention')
    pair=sum(np.kron(t,t) for t in normalized)
    swap=np.zeros((9,9))
    for a in range(3):
        for b in range(3):swap[3*a+b,3*b+a]=1
    close(pair,(swap-np.eye(9)/3)/2,'actual normal triplet has the standard color pair-Casimir identity')
    eigenvalues=np.linalg.eigvalsh(pair)
    close(eigenvalues[:3],-2/3,'normal triplet antisymmetric pair has minus two-thirds contraction')
    close(eigenvalues[3:],1/3,'normal triplet symmetric pair has one-third contraction')
    determinant=np.zeros(27,complex)
    for perm in permutations(range(3)):
        sign=(-1)**sum(perm[i]>perm[j] for i in range(3) for j in range(i+1,3))
        determinant[9*perm[0]+3*perm[1]+perm[2]]=sign/math.sqrt(6)
    for t in normalized:
        total=(np.kron(np.kron(t,np.eye(3)),np.eye(3))
               +np.kron(np.kron(np.eye(3),t),np.eye(3))
               +np.kron(np.kron(np.eye(3),np.eye(3)),t))
        close(total@determinant,0,'three normal triplets have the invariant determinant singlet')
    # A real connection metric realizes the normal action on the zero section.
    a,b=algebra[0],algebra[1]
    curvature=a@b-b@a
    curv_spin=lifted[0]@lifted[1]-lifted[1]@lifted[0]
    curv_triplet=curv_spin[np.ix_(normal_keep,normal_keep)]
    check(np.linalg.norm(curvature)>0,'constant noncommuting normal connections have nonzero curvature')
    check(np.linalg.norm(curv_triplet)>0,'the triplet sees actual nonzero normal curvature')
    vacuum_keep=np.array([j for j in range(32) if j&7==0])
    close(curv_spin[np.ix_(vacuum_keep,vacuum_keep)],0,'normal vacuum is a color singlet for the same connection')
    # The same curvature can survive on Q2: a third-form zero does not remove minimal kinetic coupling.
    for d in (2,3,4):
        for g in gamma[:d]:close(curv_spin@g,g@curv_spin,'normal curvature acts independently of every lower-sector tangent direction')
    connection_metric_checks(gamma,algebra,lifted)
    spin_retained_volume_checks()
    charge_spin_checks(gamma,lifted)
    CASES.append({'geometry':'normal connection and standard color contractions',
                  'parameters':{'connection_metric':'g=g_B+|dy+A y|^2','normal_group':'SU3'},
                  'measurements':{'qq_antisymmetric_casimir':'-2/3','qq_symmetric_casimir':'1/3',
                                  'determinant_singlet_present':True,'observer_spin_and_color_commute':True,
                                  'normal_curvature_can_couple_a_depth_two_triplet':True,
                                  'three_form_zero_alone_removes_all_depth_two_color_coupling':False}})


def charge_spin_checks(gamma,normal_su3):
    create=[(gamma[2*j]-1j*gamma[2*j+1])/2 for j in range(5)]
    destroy=[g.conj().T for g in create]
    numbers=[a@b for a,b in zip(create,destroy)]
    observer=[gamma[a]@gamma[b]/2 for a,b in ((0,1),(0,2),(1,2))]
    y=(numbers[0]+numbers[1])/2-sum(numbers[2:])/3
    t3=(numbers[0]-numbers[1])/2
    q=y+t3
    check(sum(np.linalg.norm(q@s-s@q)>1e-9 for s in observer)==2,
          'ambient exterior Q is not independent of fixed observer spatial spin')
    weak=[]
    for a in su_generators(2):
        weak.append(sum(a[i,j]*create[i]@destroy[j] for i in range(2) for j in range(2)))
    fixed_projective=normal_su3+weak+[1j*y]
    columns=[np.concatenate([(x@s-s@x).reshape(-1) for s in observer]) for x in fixed_projective]
    check(len(constraints(columns).nullspace())==8,
          'fixed/projective pure-pair stabilizer retains only normal SU3 as an observer-spin commutant')
    both_rays=fixed_projective+[1j*(sum(numbers)-2.5*np.eye(32))]
    columns=[np.concatenate([(x@s-s@x).reshape(-1) for s in observer]) for x in both_rays]
    check(len(constraints(columns).nullspace())==9,
          'both-ray geometry retains normal U3 as an observer-spin commutant')
    volume=1j*gamma[0]@gamma[1]@gamma[2]
    close(volume.conj().T,volume,'observer three-volume gives a Hermitian Clifford branch')
    close(volume@volume,np.eye(32),'observer three-volume has two unit eigenvalues')
    for g in gamma[:3]:close(volume@g,g@volume,'observer volume commutes with every three-sector Clifford direction')
    close(volume@gamma[3],-gamma[3]@volume,'observer volume anticommutes with the fourth direction')
    branch=(np.eye(32)-volume)/2
    q_normal=sum(numbers[2:])/3-branch
    for s in observer:close(q_normal@s,s@q_normal,'normal charge with an auxiliary Clifford branch is independent of observer spin')
    for g in gamma[:3]:close(q_normal@g,g@q_normal,'auxiliary branch charge retains three-sector first-order kinetics')
    check(np.linalg.norm(q_normal@gamma[3]-gamma[3]@q_normal)>0,
          'the same branch charge is not a pointwise four-sector kinetic commutant')
    normal_even=np.array([j for j in range(32) if (j&7).bit_count()%2==0])
    values=np.linalg.eigvalsh(q_normal[np.ix_(normal_even,normal_even)])
    expected=np.array([-1.]*2+[-1/3]*6+[0.]*2+[2/3]*6)
    close(values,expected,'the root-free normal charge table is present with a separate observer spin doublet')
    CASES.append({'geometry':'charge character versus independent observer spin',
                  'parameters':{'observer_sector_depth':3,'containing_base_depth':4},
                  'measurements':{'ambient_Q_commutes_with_observer_spin':False,
                                  'fixed_projective_stabilizer_observer_commutant_dimension':8,
                                  'both_rays_stabilizer_observer_commutant_dimension':9,
                                  'auxiliary_branch_operator':'(I-i gamma1 gamma2 gamma3)/2',
                                  'normal_branch_charge':'N_normal/3-branch',
                                  'normal_charge_weights_with_spin_multiplicities':{'0':2,'2/3':6,'-1':2,'-1/3':6},
                                  'normal_branch_charge_commutes_with_Cl3':True,
                                  'normal_branch_charge_commutes_with_Cl4':False,
                                  'physical_total_charge_generator_derived':False}})


def connection_metric_checks(gamma,algebra,lifted):
    # The exact first metric jet at y=0 determines the full Levi-Civita connection there.
    selected=[algebra[0],algebra[1],np.zeros((3,3),complex),np.zeros((3,3),complex)]
    ordinary=[realify_interleaved(a) for a in selected]
    dg=np.zeros((10,10,10))
    for mu,a in enumerate(ordinary):
        for i in range(6):
            for j in range(6):
                dg[4+j,mu,4+i]=a[i,j]
                dg[4+j,4+i,mu]=a[i,j]
    christoffel=np.zeros((10,10,10))
    for a in range(10):
        for b in range(10):
            for c in range(10):
                christoffel[a,b,c]=(dg[b,c,a]+dg[c,b,a]-dg[a,b,c])/2
    for mu,a in enumerate(ordinary):
        close(christoffel[4:,mu,4:],a,'connection metric has exactly the declared ordinary normal connection')
        close(christoffel[4:,:4,:4],0,'the four-dimensional zero section is totally geodesic')
        omega=christoffel[:,mu,:]
        close(omega+omega.T,0,'metric connection in the coordinate frame is skew at the zero section')
        spin=sum(omega[i,j]*gamma[i]@gamma[j]/4 for i in range(10) for j in range(10))
        expected=lifted[mu] if mu<2 else np.zeros((32,32),complex)
        close(spin,expected,'Levi-Civita spin connection induces precisely the normal SU3 color action')
    # A connection on two base directions can already generate full SU3 holonomy.
    first=algebra[6]+3*algebra[7]
    second=algebra[0]+algebra[4]
    generated=[first,second]
    rank=2
    while True:
        additions=[]
        for a,b in combinations(generated,2):
            comm=a@b-b@a
            matrix=constraints([x.reshape(-1) for x in generated+additions+[comm]])
            new_rank=matrix.rank()
            if new_rank>rank:
                additions.append(comm);rank=new_rank
        if not additions:break
        generated.extend(additions)
    check(rank==8,'a depth-two pullback can retain full normal SU3 holonomy')


def spin_retained_volume_checks():
    from scipy.linalg import expm
    rng=np.random.default_rng(20261009)
    epsilon=np.zeros((3,3,3))
    for perm in permutations(range(3)):
        epsilon[perm]=(-1)**sum(perm[i]>perm[j] for i in range(3) for j in range(i+1,3))
    def volume(a,b,c):
        tensor=np.einsum('abc,Aa,Bb,Cc->ABC',epsilon,a,b,c)
        return np.vdot(tensor,tensor).real
    for spin_rank in (2,4):
        a=rng.normal(size=(spin_rank,3))+1j*rng.normal(size=(spin_rank,3))
        b=rng.normal(size=(spin_rank,3))+1j*rng.normal(size=(spin_rank,3))
        c=rng.normal(size=(spin_rank,3))+1j*rng.normal(size=(spin_rank,3))
        energy=volume(a,b,c)
        raw=rng.normal(size=(3,3))+1j*rng.normal(size=(3,3))
        generator=raw-raw.conj().T;generator-=np.trace(generator)*np.eye(3)/3
        color_rotation=expm(generator)
        close(volume(a@color_rotation.T,b@color_rotation.T,c@color_rotation.T),energy,
              'spin-retained normal color volume is SU3 invariant')
        raw=rng.normal(size=(spin_rank,spin_rank))+1j*rng.normal(size=(spin_rank,spin_rank))
        spin_rotation=expm(raw-raw.conj().T)
        close(volume(spin_rotation@a,spin_rotation@b,spin_rotation@c),energy,
              'independent observer spin rotation preserves the normal color volume')
        summed=0
        for index in range(3):
            third=np.zeros((spin_rank,3));third[0,index]=1
            summed+=volume(a,b,third)
        rho_a=a.T@a.conj();rho_b=b.T@b.conj()
        close(summed,np.trace(rho_a).real*np.trace(rho_b).real-np.trace(rho_a@rho_b).real,
              'third normal-color contraction gives I-minus-color-swap with every spin index retained')
        repeated=volume(a,a,a)
        if spin_rank==2:
            close(repeated,0,'one pointwise chiral spinor has rank at most two and zero repeated color volume')
        else:
            check(repeated>0,'full four-component Dirac source can have a nonzero rank-three color volume')
        CASES.append({'geometry':'independent spin and normal color volume',
                      'parameters':{'spin_rank':spin_rank,'color_rank':3},
                      'measurements':{'third_contraction':'tr(rho_a) tr(rho_b)-tr(rho_a rho_b)',
                                      'pair_operator':'I_spin tensor (I_color-swap_color)',
                                      'same_chiral_pointwise_field_volume_vanishes':spin_rank==2,
                                      'distinct_modes_or_spatial_jets_needed_for_chiral_source':spin_rank==2}})


def determinant_form_checks():
    rng=np.random.default_rng(20261009)
    for k in range(2,7):
        for d in (2,3,4,5,6,10):
            derivative=rng.normal(size=(k,d))+1j*rng.normal(size=(k,d))
            norm=sum(abs(np.linalg.det(derivative[:,index]))**2 for index in combinations(range(d),k))
            check((norm>0)==(d>=k),'one determinant differential form has the same spatial degree rule at every depth')
            if d>=k:
                close(norm,np.linalg.det(derivative@derivative.conj().T).real,'Cauchy-Binet computes the complete determinant-form energy')
            # Exact all-depth comparison: a color factor cannot retain more than the four base directions.
            allowed=d>=k and d<=4
            if k==3:check(allowed==(d in (3,4)),'conditional determinant-contact plus kinetic rule leaves only maintained depths three and four')
            CASES.append({'real_depth':d,'geometry':'determinant spatial differential-form candidate',
                          'parameters':{'normal_complex_rank':k,'contact_form_degree':k},
                          'measurements':{'spatial_form_nonzero':d>=k,'base_color_kinetics_preserved':d<=4,
                                          'combined_candidate_allowed':allowed,
                                          'measured_force_derived':False}})


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    data=None
    for k in (2,3,4):
        current=kinetic_checks(k)
        if k==3:data=current
    actual_connection_and_pair(*data[:-1])
    determinant_form_checks()
    report={'title':'Independent normal color and kinetic compatibility across nested spatial sectors',
            'created_utc':datetime.now(timezone.utc).isoformat(),'checks_passed':CHECKS,'cases':CASES,
            'scope':'Finite ordinary spatial parent, full tangent Clifford symbols, exact real commutants and derivative-form controls.',
            'limitations':['The selected ten-dimensional parent is within the infinite containing manifold; higher ambient normals are not excluded.',
                           'The normal SU3 triplet is distinct from the base middle-Hodge spin-one triplet.',
                           'Depth two also admits the triplet kinematically, and its normal curvature need not vanish.',
                           'A determinant three-form vanishes at depth two but does not remove an independently present covariant kinetic connection.',
                           'Residual SU2 and higher equivalent-mode interactions are not excluded at five/six.',
                           'No occupied particle representation, strong dynamics or all-sector force absence is derived.'],
            'sources':{'experiments/normal-color-kinetics-checks.py':sha256(Path(__file__).read_bytes()).hexdigest(),
                       'experiments/spinor-hodge-source-checks.py':sha256((HERE/'spinor-hodge-source-checks.py').read_bytes()).hexdigest()}}
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'checks_passed':CHECKS,'cases':len(CASES),'output':str(args.output)}))


if __name__=='__main__':main()
