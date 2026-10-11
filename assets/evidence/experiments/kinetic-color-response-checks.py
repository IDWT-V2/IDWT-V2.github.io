#!/usr/bin/env python3
"""Use actual positive spatial Clifford branches in the occupied-sector current."""
import argparse
from datetime import datetime, timezone
from hashlib import sha256
import importlib.util
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('current', HERE/'hodge-color-current-checks.py')
current = importlib.util.module_from_spec(spec)
spec.loader.exec_module(current)
CHECKS = 0
CASES = []


def check(condition, message):
    global CHECKS
    if not condition:
        raise AssertionError(message)
    CHECKS += 1


def close(actual, expected, message, tolerance=2e-10):
    check(np.allclose(actual, expected, atol=tolerance, rtol=tolerance), message)


def volume(gamma, dimension):
    value = (1j)**(dimension//2)*np.eye(len(gamma[0]), dtype=complex)
    for g in gamma[:dimension]:
        value = value@g
    return value


def quartic(operators, psi):
    j = np.einsum('i,abij,j->ab',psi.conj(),operators,psi).real
    return np.vdot(j,j).real


def phase_mixed(operators, first, second):
    return sum(quartic(operators,first+z*second) for z in (1,1j,-1,-1j))/4-quartic(operators,first)-quartic(operators,second)


def positive_frame(gamma, direction):
    symbol = sum(p*g for p,g in zip(direction,gamma))
    values, vectors = np.linalg.eigh(symbol)
    result = vectors[:,values>.5]
    close(result.conj().T@result,np.eye(2),'positive tangent branch retains its complete two-spin multiplicity')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    rng = np.random.default_rng(202610092350)
    gamma, colors, keep, triplet = current.color_data()
    for depth in (2,3,4,5,6,10):
        dimension = depth+depth%2
        chirality = volume(gamma,dimension)
        data = current.hodge_data(depth,gamma,colors)
        leakages = []
        for _ in range(4):
            direction = rng.normal(size=depth)
            direction /= np.linalg.norm(direction)
            symbol = sum(p*g for p,g in zip(direction,gamma))
            positive = (np.eye(32)+symbol)/2
            close(positive@positive,positive,'full spatial principal symbol gives a positive-frequency projector')
            close(positive@chirality@positive,np.zeros((32,32)),
                  'positive spatial Clifford states balance the two containing source chiralities')
            psi = positive@(rng.normal(size=32)+1j*rng.normal(size=32))
            psi /= np.linalg.norm(psi)
            close(np.vdot(psi,(np.eye(32)+chirality)@psi).real/2,.5,
                  'source chirality balance follows from the kinetic branch rather than fitted amplitudes')
            leakage = float(np.linalg.norm(triplet@symbol-symbol@triplet))
            leakages.append(leakage)
            if depth <= 4:
                close(leakage,0,'fixed normal triplet preserves every occupied lower tangent kinetic branch')
            else:
                check(leakage>1e-5,'fixed triplet is not a complete higher-depth spatial kinetic carrier')
            energy = current.energy_and_gradient(data,psi)[0]
            if depth not in (3,4):
                close(energy,0,'inactive occupied-sector current remains zero on the actual positive kinetic branch')
        CASES.append({'real_depth':depth,'geometry':'positive full spatial principal-symbol branch',
                      'parameters':{'sampled_unit_momenta':4,'source_chiralities_preserved':True},
                      'measurements':{'positive_branch_source_chiral_weights':['1/2','1/2'],
                                      'triplet_symbol_commutator_norms':leakages,
                                      'full_particle_solution_derived':False}})
    base_gamma = [current.source.dense(g) for g in current.source.gamma_monomials(2)]
    data = current.hodge_data(4,gamma,colors)
    operators = data['operators'][:,:,keep,:][:,:,:,keep]
    small_color = [t[np.ix_(keep[:3],keep[:3])] for t in colors]
    for depth in (3,4):
        for repetition in range(12):
            direction = rng.normal(size=depth)
            direction /= np.linalg.norm(direction)
            frame = positive_frame(base_gamma[:depth],direction)
            lift = np.kron(frame,np.eye(3))
            reduced = np.einsum('ij,abjk,kl->abil',lift.conj().T,operators,lift,optimize=True)
            psi = rng.normal(size=(2,3))+1j*rng.normal(size=(2,3))
            n = np.vdot(psi,psi).real
            rho = psi@psi.conj().T
            purity = np.trace(rho@rho).real
            energy = quartic(reduced,psi.reshape(-1))
            close(energy,(7*n*n-5*purity)/6,
                  'actual positive-symbol spin-color source has the exact half-chiral Fierz norm')
            check(n*n/3-1e-8 <= energy <= 3*n*n/4+1e-8,
                  'kinetically constrained source obeys its stronger uniform norm bounds')
            second_direction = rng.normal(size=depth)
            second_direction /= np.linalg.norm(second_direction)
            second_frame = positive_frame(base_gamma[:depth],second_direction)
            c = rng.normal(size=3)+1j*rng.normal(size=3)
            c /= np.linalg.norm(c)
            d = rng.normal(size=3)+1j*rng.normal(size=3)
            d /= np.linalg.norm(d)
            overlap = abs(np.vdot(c,d))**2
            spin_average = 0.
            for first_spin in frame.T:
                for second_spin in second_frame.T:
                    spin_average += phase_mixed(operators,np.kron(first_spin,c),np.kron(second_spin,d))/4
            close(spin_average,(3-overlap)/2,
                  'unpolarized positive kinetic branches have a direction-independent normal-color pair contraction')
            CASES.append({'real_depth':depth,'geometry':'independently directed positive kinetic spin-color patterns',
                          'parameters':{'momentum_direction_a':direction.tolist(),
                                        'momentum_direction_b':second_direction.tolist(),
                                        'relative_phase_uniform_average':True,
                                        'spin_preparation':'independent identity over each positive two-spin branch'},
                          'measurements':{'color_overlap_squared':float(overlap),
                                          'averaged_pair_contact':float(spin_average),
                                          'exact_pair_operator':'(3 I-swap_color)/2=(4/3)I-sum T_A tensor T_A',
                                          'color_casimir_coefficient':-1}})
    # The full color operator and its scalar offset remain, including their ordering.
    swap = np.zeros((9,9))
    for a in range(3):
        for b in range(3):
            swap[3*a+b,3*b+a] = 1
    casimir = sum(np.kron(t,t) for t in small_color)
    operator = (3*np.eye(9)-swap)/2
    close(operator,4*np.eye(9)/3-casimir,
          'kinetically constrained contraction retains its scalar offset and signed color Casimir')
    values = np.linalg.eigvalsh(operator)
    close(values,np.array([1.]*6+[2.]*3),
          'symmetric color has lower positive contact than antisymmetric color in this candidate')
    report = {'experiment':'kinetic-color-response','created_utc':datetime.now(timezone.utc).isoformat(),
              'status':'passed','checks':CHECKS,'cases':CASES,
              'scope':'Positive principal-symbol plane-wave controls, not localized mass solutions or a strong-force sign derivation.',
              'sources':{'experiments/kinetic-color-response-checks.py':sha256(Path(__file__).read_bytes()).hexdigest()}}
    for name in ('hodge-color-current-checks.py','normal-color-kinetics-checks.py','spinor-hodge-source-checks.py'):
        report['sources']['experiments/'+name] = sha256((HERE/name).read_bytes()).hexdigest()
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'checks':CHECKS,'cases':len(CASES),'report':str(args.output)}))


if __name__ == '__main__':
    main()
