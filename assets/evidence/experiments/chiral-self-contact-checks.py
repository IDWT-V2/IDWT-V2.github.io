#!/usr/bin/env python3
"""Test a parity-even chiral self-wedge current on occupied sector depths."""
import argparse
from datetime import datetime, timezone
from hashlib import sha256
import importlib.util
import json
from pathlib import Path

import numpy as np
from scipy.linalg import expm
import sympy as sp

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


def chirality(gamma, dimension):
    value = (1j)**(dimension//2)*np.eye(len(gamma[0]), dtype=complex)
    for g in gamma[:dimension]:
        value = value@g
    close(value.conj().T, value, 'oriented full-volume Clifford operator is Hermitian')
    close(value@value, np.eye(len(value)), 'full-volume chirality squares to one')
    return value


def projected_data(data, gamma):
    volume = chirality(gamma, data['dimension'])
    result = {}
    for sign in (1, -1):
        projector = (np.eye(len(volume))+sign*volume)/2
        operators = np.einsum('ij,abjk,kl->abil', projector, data['operators'], projector, optimize=True)
        result[sign] = {'projector': projector, 'operators': operators}
    return volume, result


def wedge_energy_gradient(star, blocks, psi):
    energy = 0.
    gradient = np.zeros_like(psi)
    currents = {}
    for sign, block in blocks.items():
        value = np.einsum('i,abij,j->ab', psi.conj(), block['operators'], psi).real
        currents[sign] = value
        energy += sign*np.vdot(value, star@value).real
        coefficient = sign*(star+star.T)@value
        gradient += np.einsum('ab,abij,j->i', coefficient, block['operators'], psi)
    return energy, gradient, currents


def wedge(left, right):
    result = {}
    for a, va in left.items():
        for b, vb in right.items():
            if set(a).intersection(b):
                continue
            index = tuple(sorted(a+b))
            result[index] = result.get(index, 0)+current.source.parity(a+b)*va*vb
    return {index:sp.simplify(value) for index,value in result.items() if sp.simplify(value) != 0}


def exterior_derivative(form, coordinates):
    result = {}
    for index, coefficient in form.items():
        for axis, coordinate in enumerate(coordinates):
            if axis in index:
                continue
            new_index = tuple(sorted((axis,)+index))
            result[new_index] = result.get(new_index,0)+current.source.parity((axis,)+index)*sp.diff(coefficient,coordinate)
    return {index:sp.simplify(value) for index,value in result.items() if sp.simplify(value) != 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rng = np.random.default_rng(202610092336)
    gamma, colors, keep, projector = current.color_data()
    by_dimension = {}
    for depth in (2, 3, 4, 5, 6, 10):
        dimension = depth+depth%2
        if dimension not in by_dimension:
            data = current.hodge_data(depth, gamma, colors)
            volume, blocks = projected_data(data, gamma)
            by_dimension[dimension] = data, volume, blocks
        data, volume, blocks = by_dimension[dimension]
        middle = data['degree']
        active = middle%2 == 0
        sample_energies = []
        for repetition in range(4):
            psi = rng.normal(size=32)+1j*rng.normal(size=32)
            psi /= np.linalg.norm(psi)
            energy, gradient, currents = wedge_energy_gradient(data['star'], blocks, psi)
            previous, previous_gradient = current.energy_and_gradient(data, psi)
            close(energy, previous, 'self-wedge and symmetric-current actions agree at every occupied sector')
            close(gradient, previous_gradient, 'occupied-sector agreement includes the complete field variation')
            if active:
                for sign, value in currents.items():
                    close(data['star']@value, sign*value, 'actual projected even-middle current has its Clifford chirality')
                close(energy, sum(np.vdot(v,v).real for v in currents.values()),
                      'chirality-weighted self-contact is the positive sum of both source norms')
                check(energy > 0, 'the occupied quark channel can have nonzero self-contact')
            else:
                close(data['star'].T, -data['star'], 'odd-middle order reversal is skew on real currents')
                close(energy, 0, 'the identical odd-middle self-contact cancels without a sector coefficient')
                close(gradient, 0, 'the cancelled local polynomial has no hidden first variation')
            # Reversing orientation swaps the two source chiralities and star's sign.
            reversed_blocks = {sign: blocks[-sign] for sign in (1,-1)}
            reversed_energy, reversed_gradient, _ = wedge_energy_gradient(-data['star'], reversed_blocks, psi)
            close(reversed_energy, energy, 'the contact is independent of orientation convention')
            close(reversed_gradient, gradient, 'orientation reversal preserves its field equation')
            direction = rng.normal(size=32)+1j*rng.normal(size=32)
            direction /= np.linalg.norm(direction)
            step = 2e-5
            difference = (wedge_energy_gradient(data['star'], blocks, psi+step*direction)[0]
                          -wedge_energy_gradient(data['star'], blocks, psi-step*direction)[0])/(2*step)
            close(difference, 2*np.vdot(gradient,direction).real,
                  'direct variation retains both current factors and both chiralities', 2e-7)
            close(np.vdot(psi,gradient).imag, 0, 'the mode completion conserves total canonical norm')
            omega = .7
            velocity = -1j*(omega*psi+gradient)
            close(2*np.vdot(omega*psi+gradient,velocity).real, 0,
                  'the full retained-mode energy is conserved under its reciprocal variation')
            sample_energies.append(float(energy))
        if depth in (3,4):
            unitary = expm(.31j*sum((i+1)*t for i,t in enumerate(colors))/20)
            rotated = wedge_energy_gradient(data['star'], blocks, unitary@psi)[0]
            close(rotated, energy, 'independent normal SU3 is retained in the occupied active channel')
            # A reflection in one base direction swaps source chirality and preserves the energy.
            reflected = wedge_energy_gradient(data['star'], blocks, gamma[0]@psi)[0]
            close(reflected, energy, 'including both source chiralities retains physical base parity')
            close(blocks[1]['projector']@gamma[0]@blocks[1]['projector'], np.zeros((32,32)),
                  'source projection must not be substituted for the full kinetic field')
        CASES.append({'real_depth':depth, 'geometry':'chirality-weighted identical middle-current self-wedge',
                      'parameters':{'even_carrier_depth':dimension, 'middle_degree':middle,
                                    'both_source_chiralities_retained':True,
                                    'full_kinetic_field_retained':True},
                      'measurements':{'active':active, 'sample_energies':sample_energies,
                                      'equals_previous_hodge_current_action':True,
                                      'complete_strong_force_derived':False}})
    data, volume, blocks = by_dimension[4]
    lookup = {pair:i for i,pair in enumerate(data['blades'])}
    indices = {sign:np.array([i for i in keep if blocks[sign]['projector'][i,i].real > .5])
               for sign in (1,-1)}
    for repetition in range(12):
        psi = np.zeros(32,complex)
        psi[keep] = rng.normal(size=12)+1j*rng.normal(size=12)
        norm_total = np.vdot(psi,psi).real
        energy = wedge_energy_gradient(data['star'], blocks, psi)[0]
        predicted = 0.
        for sign in (1,-1):
            matrix = psi[indices[sign]].reshape((2,3))
            norm_chiral = np.vdot(matrix,matrix).real
            purity = np.trace((matrix@matrix.conj().T)@(matrix@matrix.conj().T)).real
            predicted += (7*norm_chiral**2-5*purity)/3
        close(energy,predicted,'full Dirac triplet has the sum of two exact chiral Fierz energies')
        check(norm_total**2/3-1e-8 <= energy <= 3*norm_total**2/2+1e-8,
              'full Dirac triplet self-contact has uniform sharp positive bounds')
    lower = np.zeros(32,complex)
    lower[indices[1][0]] = 1/np.sqrt(2)
    lower[indices[-1][0]] = 1/np.sqrt(2)
    upper = np.zeros(32,complex)
    upper[indices[1][0]] = upper[indices[1][4]] = 1/np.sqrt(2)
    close(wedge_energy_gradient(data['star'],blocks,lower)[0],1/3,
          'balanced factorized source chiralities attain the full-triplet lower bound')
    close(wedge_energy_gradient(data['star'],blocks,upper)[0],3/2,
          'a maximally entangled one-chirality source attains the full-triplet upper bound')
    for repetition in range(8):
        psi = rng.normal(size=32)+1j*rng.normal(size=32)
        _, _, currents = wedge_energy_gradient(data['star'], blocks, psi)
        for sign, value in currents.items():
            spatial = np.array([value[lookup[(1,2)]], -value[lookup[(0,2)]], value[lookup[(0,1)]]])
            normal = np.array([value[lookup[(0,3)]], value[lookup[(1,3)]], value[lookup[(2,3)]]])
            close(normal, sign*spatial, 'the fourth-direction current is tied to the three-direction magnetic form by chirality')
            close(np.vdot(value,data['star']@value).real, 2*np.vdot(normal,spatial).real,
                  'normal contraction of the parent top form has the explicit three-shadow density')
            close(sign*2*np.vdot(normal,spatial).real, np.vdot(value,value).real,
                  'the three-shadow contact has exactly the inherited positive norm')
    # A local algebraic current is not automatically a closed curvature form.
    x = sp.symbols('x0:4', real=True)
    amplitude = sp.exp(-sum(t*t for t in x)/2)
    # Selfdual e12+e34 is a simple current-form control, not a particle solution.
    form = {(0,1):amplitude**2, (2,3):amplitude**2}
    derivative = {}
    for pair, coefficient in form.items():
        for axis in range(4):
            if axis in pair:
                continue
            index = tuple(sorted((axis,)+pair))
            derivative[index] = derivative.get(index,0)+current.source.parity((axis,)+pair)*sp.diff(coefficient,x[axis])
    check(any(sp.simplify(value) != 0 for value in derivative.values()),
          'a varying algebraic current need not obey a curvature Bianchi identity')
    norm = sp.integrate(sp.exp(-2*x[0]**2),(x[0],-sp.oo,sp.oo))**4
    check(sp.simplify(norm-sp.pi**2/4) == 0, 'a decaying trial density has a finite positive bulk self-contact integral')
    alpha = sp.symbols('alpha', real=True)
    check(sp.diff(alpha**4*norm,alpha).subs(alpha,1) == sp.pi**2,
          'the decaying algebraic current has a nonzero bulk amplitude variation')
    product = np.zeros(32,complex)
    selected = [i for i in keep if blocks[1]['projector'][i,i].real > .5]
    product[selected[0]] = 1
    product_energy, _, actual_currents = wedge_energy_gradient(data['star'], blocks, product)
    close(product_energy, 2/3, 'an actual factorized spin-color current attains the established source norm')
    excluded = [i for i,pair in enumerate(data['blades']) if 0 not in pair]
    nonclosed_norm = sum(np.vdot(value[excluded],value[excluded]).real for value in actual_currents.values())
    close(nonclosed_norm, product_energy/2, 'an actual selfdual spinor current has a nonzero exterior derivative under a varying profile')
    for dimension in (2,4,6,10):
        coordinates = sp.symbols('q0:'+str(dimension), real=True)
        degree = dimension//2
        potential = {tuple(range(1,degree)):coordinates[0],
                     tuple(range(degree+1,dimension)):coordinates[degree]}
        curvature = exterior_derivative(potential,coordinates)
        check(exterior_derivative(curvature,coordinates) == {}, 'exact differential form obeys its Bianchi identity')
        check(exterior_derivative(wedge(potential,curvature),coordinates) == wedge(curvature,curvature),
              'exact closed curvature self-contact is a boundary term at every occupied even carrier')
    CASES.append({'real_depth':4, 'geometry':'algebraic current versus a closed exact differential form',
                  'parameters':{'trial_profile_is_particle_solution':False},
                  'measurements':{'algebraic_current_closed':False,
                                  'gaussian_fourth_power_integral':'pi^2/4',
                                  'bulk_amplitude_variation_at_one':'pi^2',
                                  'actual_factorized_spinor_current_integrated_contact':'pi^2/6',
                                  'actual_current_bulk_amplitude_variation':'2*pi^2/3',
                                  'exact_closed_form_self_wedge_is_boundary_term':True}})
    sources = {'experiments/chiral-self-contact-checks.py':sha256(Path(__file__).read_bytes()).hexdigest()}
    for filename in ('hodge-color-current-checks.py','normal-color-kinetics-checks.py','spinor-hodge-source-checks.py'):
        sources['experiments/'+filename] = sha256((HERE/filename).read_bytes()).hexdigest()
    report = {'experiment':'chiral-self-contact', 'created_utc':datetime.now(timezone.utc).isoformat(),
              'status':'passed', 'checks':CHECKS, 'cases':CASES, 'sources':sources,
              'scope':'A proposed orientation-even local self-wedge interaction equals the positive Hodge-current action at occupied sectors. Its selection, spatial dynamics and force law remain open.'}
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'checks':CHECKS,'cases':len(CASES),'report':str(args.output)}))


if __name__ == '__main__':
    main()
