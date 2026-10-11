#!/usr/bin/env python3
"""Join independent normal color and symmetric Hodge contact on actual field currents."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
from itertools import combinations
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
from scipy.linalg import expm


HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('normal_color',HERE/'normal-color-kinetics-checks.py')
normal=importlib.util.module_from_spec(spec)
spec.loader.exec_module(normal)
source=normal.source
CHECKS=0
CASES=[]


def check(value,message):
    global CHECKS
    if not value:raise AssertionError(message)
    CHECKS+=1


def close(a,b,message,atol=2e-10):
    check(np.allclose(a,b,atol=atol,rtol=2e-10),message)


def color_data():
    gamma,algebra,lifted,keep,projector=normal.data_for_parent(3)
    color_keep=np.array([j for j in keep if j>>3==0])
    selected=[]
    hermitian=[]
    for full in lifted:
        full=1j*full
        candidate=full[np.ix_(color_keep,color_keep)]
        for previous,small in zip(hermitian,selected):
            coefficient=2*np.trace(small@candidate).real
            full-=coefficient*previous
            candidate-=coefficient*small
        norm=np.trace(candidate@candidate).real
        full/=math.sqrt(2*norm);candidate/=math.sqrt(2*norm)
        selected.append(candidate);hermitian.append(full)
    return gamma,hermitian,keep,projector


def hodge_data(d,gamma,color):
    # Odd depth uses the declared neighboring even-carrier shadow.
    dimension=d+d%2
    degree=dimension//2
    blades=list(combinations(range(dimension),degree))
    lookup={blade:i for i,blade in enumerate(blades)}
    size=len(blades)
    star=np.zeros((size,size))
    matrices=[]
    for index in blades:
        complement=tuple(i for i in range(dimension) if i not in index)
        star[lookup[complement],lookup[index]]=source.parity(index+complement)
        blade=np.eye(32,dtype=complex)
        for i in index:blade=blade@gamma[i]
        blade*=1j**(degree*(degree-1)//2)
        close(blade.conj().T,blade,'middle Clifford blade has the Hermitian phase')
        currents=[]
        for t in color:
            operator=(blade@t+t@blade)/2
            close(operator.conj().T,operator,'color middle-form source is an actual real Hermitian current')
            currents.append(operator)
        matrices.append(currents)
    close(star@star,(-1)**degree*np.eye(size),'actual oriented middle-form star has the common parity')
    h=(star+star.T)/2
    close(h@h,(1+(-1)**degree)*np.eye(size)/2,'same two-sided contact selects its source parity')
    return {'dimension':dimension,'degree':degree,'blades':blades,'star':star,'h':h,
            'operators':np.asarray(matrices)}


def source_current(data,psi):
    return np.einsum('i,abij,j->ab',psi.conj(),data['operators'],psi).real


def energy_and_gradient(data,psi):
    current=source_current(data,psi)
    filtered=data['h']@current
    energy=np.vdot(filtered,filtered).real
    coefficients=data['h'].T@filtered
    gradient=2*np.einsum('ab,abij,j->i',coefficients,data['operators'],psi)
    return energy,gradient


def sector_checks(gamma,color,rng):
    for d in range(2,11):
        data=hodge_data(d,gamma,color)
        active=data['degree']%2==0
        energies=[]
        for _ in range(3):
            psi=rng.normal(size=32)+1j*rng.normal(size=32)
            psi/=np.linalg.norm(psi)
            value,gradient=energy_and_gradient(data,psi)
            energies.append(value)
            check((value>1e-12)==active,'actual spin-color source follows the same all-sector Hodge rule')
            close(np.vdot(psi,gradient).imag,0,'autonomous current Hamiltonian conserves the full field norm')
            # A positive canonical retained-mode action supplies reciprocal variation.
            omega=.7
            total_gradient=omega*psi+gradient
            velocity=-1j*total_gradient
            close(2*np.vdot(total_gradient,velocity).real,0,'complete canonical mode action conserves total positive energy')
            check(omega*np.vdot(psi,psi).real+value>0,'complete canonical retained-mode Hamiltonian is positive')
            direction=rng.normal(size=32)+1j*rng.normal(size=32)
            direction/=np.linalg.norm(direction)
            step=2e-5
            numerical=(energy_and_gradient(data,psi+step*direction)[0]
                       -energy_and_gradient(data,psi-step*direction)[0])/(2*step)
            check(abs(numerical-2*np.vdot(gradient,direction).real)<2e-7,
                  'full source variation includes both sides of the bilinear current')
        psi=rng.normal(size=32)+1j*rng.normal(size=32);psi/=np.linalg.norm(psi)
        generator=sum((i+1)*t for i,t in enumerate(color))/20
        rotation=expm(1j*.713*generator)
        original,gradient=energy_and_gradient(data,psi)
        rotated,_=energy_and_gradient(data,rotation@psi)
        if d<=4:
            close(rotated,original,'active lower-sector source preserves independent normal SU3')
            for t in color:
                close(2*np.vdot(t@psi,-1j*gradient).real,0,'normal color charge is conserved by the active current interaction')
        if d in (7,8):
            check(abs(rotated-original)>1e-7,'active higher control does not inherit an independent fixed normal SU3')
        # Rotate the field, every ordinary tangent direction and normal geometric datum together.
        u=math.cos(.271/2)*np.eye(32)+math.sin(.271/2)*gamma[0]@gamma[8]
        rotated_data=dict(data)
        rotated_data['operators']=np.einsum('ij,abjk,lk->abil',u,data['operators'],u.conj())
        covariant,_=energy_and_gradient(rotated_data,u@psi)
        close(covariant,original,'full simultaneous spatial field and geometry rotation preserves the source energy')
        CASES.append({'real_depth':d,'geometry':'independent normal color current with a symmetric middle-Hodge contact',
                      'parameters':{'even_carrier_depth':data['dimension'],'middle_degree':data['degree'],
                                    'full_kinetic_spinor_rank':32,'source_chiral_projection':False},
                      'measurements':{'instantaneous_current_channel_active':active,
                                      'sampled_unit_norm_energies':energies,
                                      'independent_normal_su3_kinetic_symmetry':d<=4,
                                      'active_independent_color_channel':active and d<=4,
                                      'normal_color_rotation_energy_difference':float(rotated-original),
                                      'particle_masses_or_force_rates_derived':False}})


def chiral_fierz_checks(gamma,color,keep,rng):
    gamma4=[source.dense(g) for g in source.gamma_monomials(2)]
    base_chirality=np.diag([1.,-1.,-1.,1.])
    base_keep=np.where(np.diag(base_chirality)>0)[0]
    # Same complete six blades, one chiral source; the kinetic field remains full.
    spin_matrices=[]
    for a,b in combinations(range(4),2):
        spin_matrices.append((1j*gamma4[a]@gamma4[b])[np.ix_(base_keep,base_keep)])
    projected_keep=np.array([index for index in keep if (index>>3).bit_count()%2==0])
    check(len(projected_keep)==6,'chiral source has two spin by three color components')
    small_color=[t[np.ix_(keep[:3],keep[:3])] for t in color]
    matrices=np.asarray([np.kron(s,t) for s in spin_matrices for t in small_color])
    data=hodge_data(4,gamma,color)
    actual=data['operators'][:,:,projected_keep,:][:,:,:,projected_keep].reshape((-1,6,6))
    close(actual,matrices,'actual projected ambient current equals independent spin-blade times normal color')
    swap=np.zeros((4,4))
    for a in range(2):
        for b in range(2):swap[2*a+b,2*b+a]=1
    close(sum(np.kron(s,s) for s in spin_matrices),2*(2*swap-np.eye(4)),
          'complete middle Clifford blades supply two copies of the spin exchange tensor')
    for _ in range(8):
        psi=rng.normal(size=(2,3))+1j*rng.normal(size=(2,3))
        n=np.vdot(psi,psi).real
        rho=psi@psi.conj().T
        purity=np.trace(rho@rho).real
        flat=psi.reshape(-1)
        j=np.einsum('i,aij,j->a',flat.conj(),matrices,flat).real
        energy=j@j
        close(energy,(7*n*n-5*purity)/3,'exact spin-color Fierz identity gives the complete quartic source')
        check(2*n*n/3-1e-8<=energy<=3*n*n/2+1e-8,'chiral color source has uniform positive norm bounds')
        # The quartic's phase average is integrated exactly by four phases.
        s=rng.normal(size=2)+1j*rng.normal(size=2);s/=np.linalg.norm(s)
        t=rng.normal(size=2)+1j*rng.normal(size=2);t/=np.linalg.norm(t)
        c=rng.normal(size=3)+1j*rng.normal(size=3);c/=np.linalg.norm(c)
        e=rng.normal(size=3)+1j*rng.normal(size=3);e/=np.linalg.norm(e)
        first,second=np.kron(s,c),np.kron(t,e)
        def quartic(z):
            current=np.einsum('i,aij,j->a',z.conj(),matrices,z).real
            return current@current
        mixed=sum(quartic(first+phase*second) for phase in (1,1j,-1,-1j))/4-quartic(first)-quartic(second)
        spin_overlap=abs(np.vdot(s,t))**2;color_overlap=abs(np.vdot(c,e))**2
        close(mixed,(14-10*spin_overlap-10*color_overlap+14*spin_overlap*color_overlap)/3,
              'phase-averaged two-pattern source retains its exact spin and color dependence')
    # Purely product spin/color and maximally spin-color entangled states attain the two bounds.
    product=np.zeros(6,complex);product[0]=1
    entangled=np.zeros(6,complex);entangled[0]=entangled[4]=1/math.sqrt(2)
    energies=[]
    for z,target in ((product,2/3),(entangled,3/2)):
        current=np.einsum('i,aij,j->a',z.conj(),matrices,z).real
        close(current@current,target,'sharp positive source bounds have actual spinor preparations')
        energies.append(float(current@current))
    # A Haar-random spin ray has mean overlap 1/2; use an exact two-spin basis average.
    c=np.array([1,0,0],complex);e=np.array([1,2j,3],complex);e/=np.linalg.norm(e)
    first=np.kron(np.array([1,0]),c)
    mixed=0
    for t in np.eye(2):
        second=np.kron(t,e)
        def quartic(z):
            current=np.einsum('i,aij,j->a',z.conj(),matrices,z).real
            return current@current
        mixed+=(sum(quartic(first+phase*second) for phase in (1,1j,-1,-1j))/4-quartic(first)-quartic(second))/2
    close(mixed,3-abs(np.vdot(c,e))**2,'unpolarized spin gives scalar plus standard color pair exchange')
    CASES.append({'real_depth':4,'geometry':'actual chiral adjoint-color middle-form current',
                  'parameters':{'source_spin_rank':2,'source_color_rank':3,'kinetic_carrier_remains_full':True},
                  'measurements':{'source_energy':'(7 n^2-5 tr(rho_spin^2))/3',
                                  'sharp_energy_bounds':['2 n^2/3','3 n^2/2'],
                                  'mixed_phase_averaged_energy':'(14-10s-10c+14sc)/3',
                                  'unpolarized_spin_pair':'3I-swap_color=(8/3)I-2 sum_A T_A tensor T_A',
                                  'attractive_force_derived_from_positive_contact':False}})


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();rng=np.random.default_rng(20261009)
    gamma,color,keep,projector=color_data()
    sector_checks(gamma,color,rng)
    chiral_fierz_checks(gamma,color,keep,rng)
    report={'title':'Independent normal color and two-sided Hodge contact share one explicit current action',
            'created_utc':datetime.now(timezone.utc).isoformat(),'checks_passed':CHECKS,'cases':CASES,
            'scope':'Full spatial field source operators, exact parity, bounded variation and spin-color Fierz tests; no particle simulation.',
            'limitations':['This is a proposed common current interaction, not a derived master law or QCD force.',
                           'Odd-sector sources use the existing neighboring even-carrier shadow premise.',
                           'The full kinetic field is retained; source projections do not replace its Clifford symbol.',
                           'The positive canonical oscillator action is a mode completion, not the derivation of its spatial kinetic reduction.',
                           'Other normal-connection and complementary Hodge interactions are not excluded.',
                           'Spin dependence remains explicit, and force attraction requires a spatial and dynamical calculation.',
                           'The active seven/eight control lacks independent fixed normal SU3 kinetics.',
                           'Nonlinear self-interaction can shift frequencies; the observed mass targets have not been rederived.'],
            'sources':{'experiments/hodge-color-current-checks.py':sha256(Path(__file__).read_bytes()).hexdigest(),
                       'experiments/normal-color-kinetics-checks.py':sha256((HERE/'normal-color-kinetics-checks.py').read_bytes()).hexdigest(),
                       'experiments/spinor-hodge-source-checks.py':sha256((HERE/'spinor-hodge-source-checks.py').read_bytes()).hexdigest()}}
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'checks_passed':CHECKS,'cases':len(CASES),'output':str(args.output)}))


if __name__=='__main__':main()
