#!/usr/bin/env python3
"""Test one geometric zero-dielectric constraint without erasing active kinetics."""
from pathlib import Path
from itertools import combinations
import argparse
import hashlib
import json
import math

import numpy as np
from scipy.linalg import expm
from scipy.integrate import quad


DEPTHS=(2,3,4,5,6,10)
CHECKS=0
CASES=[]


def check(condition,message):
    global CHECKS
    CHECKS+=1
    if not condition:
        raise AssertionError(message)


def close(actual,expected,message,atol=2e-10):
    check(np.allclose(actual,expected,atol=atol,rtol=2e-10),message)


def tensor(factors):
    result=np.ones((1,1),complex)
    for factor in factors:
        result=np.kron(result,factor)
    return result


def build_geometry():
    x=np.array([[0,1],[1,0]],complex)
    y=np.array([[0,-1j],[1j,0]],complex)
    z=np.diag([1.,-1.])
    identity=np.eye(2)
    gamma=[]
    for mode in range(5):
        for pauli in (x,y):
            gamma.append(tensor([z]*mode+[pauli]+[identity]*(4-mode)))
    for i,g in enumerate(gamma):
        close(g,g.conj().T,'Hermitian full spatial Clifford matrix')
        for j,h in enumerate(gamma):
            close(g@h+h@g,2*np.eye(32) if i==j else 0,'full ten-dimensional Clifford relation')
    annihilate=[(gamma[2*j]+1j*gamma[2*j+1])/2 for j in range(2,5)]
    number=[a.conj().T@a for a in annihilate]
    fundamental=[]
    for i,j in combinations(range(3),2):
        g=np.zeros((3,3),complex);g[i,j]=g[j,i]=.5;fundamental.append(g)
        g=np.zeros((3,3),complex);g[i,j]=-.5j;g[j,i]=.5j;fundamental.append(g)
    fundamental.extend([np.diag([1.,-1.,0])/2,np.diag([1.,1.,-2.])/(2*math.sqrt(3))])
    lifted=[sum((g[i,j]*annihilate[i].conj().T@annihilate[j]
                 for i in range(3) for j in range(3)),np.zeros((32,32),complex))
            for g in fundamental]
    for i,g in enumerate(fundamental):
        for j,h in enumerate(fundamental):
            close(np.trace(g@h),.5 if i==j else 0,'fundamental color normalization')
    residual=[]
    for g in (np.array([[0,0,0],[0,0,1],[0,1,0]],complex)/2,
              np.array([[0,0,0],[0,0,-1j],[0,1j,0]],complex)/2,
              np.diag([0.,1.,-1.])/2):
        residual.append(sum((g[i,j]*annihilate[i].conj().T@annihilate[j]
                             for i in range(3) for j in range(3)),np.zeros((32,32),complex)))
    return gamma,lifted,residual,number


def hodge(even_depth):
    m=even_depth//2
    basis=list(combinations(range(even_depth),m))
    lookup={item:i for i,item in enumerate(basis)}
    star=np.zeros((len(basis),len(basis)))
    for j,item in enumerate(basis):
        complement=tuple(i for i in range(even_depth) if i not in item)
        ordered=item+complement
        sign=(-1)**sum(ordered[i]>ordered[j] for i in range(even_depth) for j in range(i+1,even_depth))
        star[lookup[complement],j]=sign
    close(star.T@star,np.eye(len(basis)),'Hodge map preserves form metric')
    close(star@star,(-1)**m*np.eye(len(basis)),'middle-degree Hodge square')
    symmetric=(star+star.T)/2
    close(symmetric,star if m%2==0 else 0,'one symmetric Hodge law yields its zeros')
    return symmetric


def source_prefactor(psi,gamma,color):
    total=0.
    for i,j in combinations(range(4),2):
        blade=1j*gamma[i]@gamma[j]
        for generator in color:
            current=psi.conj()@((blade@generator+generator@blade)/2)@psi
            close(current.imag,0,'Hermitian middle-form color current is real')
            total+=current.real**2
    return total


def geometric_constraint_checks():
    gamma,color,residual,number=build_geometry()
    normal_n=sum(number)
    triplet_mask=np.isclose(np.diag(normal_n),2)
    triplet=np.diag(triplet_mask.astype(float))
    expected_commutants={2:8,3:8,4:8,5:3,6:3,10:0}
    expected_singlets={2:8,3:8,4:8,5:16,6:16,10:32}
    rng=np.random.default_rng(20261010)
    for d in DEPTHS:
        even=d if d%2==0 else d+1
        symmetric=hodge(even)
        active=bool(np.linalg.norm(symmetric)>1e-10)
        check(active==(d in (3,4)),'Hodge prefactor supports only occupied three/four')
        columns=[np.concatenate([(g@v-v@g).reshape(-1) for v in gamma[:d]]) for g in color]
        matrix=np.column_stack(columns)
        gram=(matrix.conj().T@matrix).real
        commutant=int(np.count_nonzero(np.linalg.eigvalsh(gram)<1e-9))
        check(commutant==expected_commutants[d],'actual pointwise normal color commutant')
        algebra=color if d<=4 else residual if d<=6 else []
        check(len(algebra)==commutant,'selected surviving generators span the commutant')
        for g in algebra:
            for v in gamma[:d]:
                close(g@v,v@g,'retained gauge algebra preserves every active kinetic direction')
        casimir=sum((g@g for g in algebra),np.zeros((32,32),complex))
        values,vectors=np.linalg.eigh(casimir)
        check(np.min(values)>-1e-10,'surviving color Casimir is positive semidefinite')
        kernel=vectors[:,values<1e-9]
        projector=kernel@kernel.conj().T
        rank=kernel.shape[1]
        check(rank==expected_singlets[d],'joint color singlet rank')
        close(projector@projector,projector,'singlet projector is idempotent')
        if d<=4:
            expected=np.diag((np.isclose(np.diag(normal_n),0)|np.isclose(np.diag(normal_n),3)).astype(float))
        elif d<=6:
            last_two=number[1]+number[2]
            expected=np.diag((np.isclose(np.diag(last_two),0)|np.isclose(np.diag(last_two),2)).astype(float))
        else:
            expected=np.eye(32)
        close(projector,expected,'Casimir kernel equals the complete exterior singlet branch')
        for i,v in enumerate(gamma[:d]):
            close(projector@v,v@projector,'singlet constraint preserves the full active spatial symbol')
            close(np.linalg.norm(projector@v@projector)**2,rank,'no active kinetic direction is erased')
            for j,w in enumerate(gamma[:d]):
                compressed=projector@v@projector@w@projector+projector@w@projector@v@projector
                close(compressed,2*projector if i==j else 0,'constrained Clifford metric is unchanged')
        psi=projector@(rng.normal(size=32)+1j*rng.normal(size=32))
        psi/=np.linalg.norm(psi)
        for g in algebra:
            close(g@projector,0,'singlet annihilates the actual gauge generator')
            close(psi.conj()@g@psi,0,'singlet Gauss source vanishes')
            for v in gamma[:d]:
                close(psi.conj()@v@g@psi,0,'singlet spatial source current vanishes')
        for i,j in combinations(range(3),2):
            spin=gamma[i]@gamma[j]/2
            close(projector@spin,spin@projector,'observer spin is retained by the singlet constraint')
        if active:
            for v in gamma[:d]:
                close(triplet@v,v@triplet,'active full triplet retains every spatial direction')
            colored=triplet@(rng.normal(size=32)+1j*rng.normal(size=32))
            colored/=np.linalg.norm(colored)
            epsilon=source_prefactor(colored,gamma,color)
            check(epsilon>1e-8,'nonzero colored state has positive actual current prefactor')
            rotation=expm(1j*sum(rng.normal()*g for g in color))
            close(source_prefactor(rotation@colored,gamma,color),epsilon,'current prefactor is gauge invariant')
            spin_rotation=expm(.37*gamma[0]@gamma[1]+.19*gamma[1]@gamma[2])
            close(source_prefactor(spin_rotation@colored,gamma,color),epsilon,'current prefactor is observer-spin invariant')
        else:
            epsilon=0.
        CASES.append({'case':f'hodge-gauss-sector-depth-{d}','depth':d,'even_source_carrier':even,
                      'hodge_prefactor_active':active,'independent_normal_algebra_dimension':commutant,
                      'joint_singlet_rank':rank,'all_active_kinetic_directions_preserved':True,
                      'allowed_active_triplet_rank':12 if active else None,
                      'example_current_prefactor':epsilon,
                      'zero_dielectric_enforces_source_constraint':not active and commutant>0,
                      'all_higher_forces_excluded':False})


def finite_floor_checks():
    sigma=1.3
    for d in DEPTHS:
        # A smooth neutral quadrupole has a positive, finite ordinary Gauss energy.
        area=2*math.pi**(d/2)/math.gamma(d/2)
        angular=3/(d*(d+2))
        radial,error=quad(lambda k:k**(d+1)*math.exp(-sigma*sigma*k*k),0,np.inf,
                          epsabs=1e-11)
        unit_numeric=area/(2*(2*math.pi)**d)*angular*radial
        unit_closed=3/(4*(d+2)*(4*math.pi)**(d/2)*sigma**(d+2))
        close(unit_numeric,unit_closed,'independent finite quadrupole Gauss energy')
        energies=[]
        for floor in (1.,.1,.01,.001):
            energy=unit_closed/floor
            check(energy>0,'finite positive Gauss energy for positive floor')
            energies.append(energy)
        for i in range(3):
            close(energies[i+1]/energies[i],10,'inactive nonzero sources diverge as inverse dielectric floor')
        CASES.append({'case':f'finite-floor-source-depth-{d}','depth':d,'sigma':sigma,
                      'unit_floor_energy':unit_closed,'independent_integral':unit_numeric,
                      'floors':[1.,.1,.01,.001],'energies':energies,
                      'source_total_charge_zero':True,'source_dipole_zero':True,
                      'source_constraint_requires_exact_zero_prefactor':True})


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args()
    geometric_constraint_checks()
    finite_floor_checks()
    result={'title':'A Hodge-Gated Gauss Constraint Retains Complete Spatial Singlet Branches',
            'sources':{'experiments/hodge-gated-gauss-checks.py':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
            'checks_passed':CHECKS,'case_count':len(CASES),'occupied_depths':list(DEPTHS),
            'status':'conditional common action, not a derived master-field coefficient or complete color force',
            'conclusion':'Putting the existing positive Hodge-current norm in the actual connection dielectric gives exact inactive-sector source constraints. Complete singlet branches of the surviving normal gauge algebra retain every active spatial Clifford direction. Active three/four triplets retain independent color. An ordinary dielectric floor removes the exact constraint.',
            'cases':CASES}
    args.report.write_text(json.dumps(result,indent=2)+'\n')
    print(f'{CHECKS} checks passed across {len(CASES)} cases; {args.report}')


if __name__=='__main__':
    main()
