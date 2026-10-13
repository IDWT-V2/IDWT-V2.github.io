"""Complete second-order spectrum of the actual rotating spatial count projector."""
from datetime import datetime, timezone
from hashlib import sha256
from importlib.util import module_from_spec, spec_from_file_location
from math import comb, pi, sqrt
from pathlib import Path
import json

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import eig, eigvals, eigvalsh
from scipy.optimize import brentq, root

ROOT = Path(__file__).resolve().parent
HELPER = ROOT/'moving-contact-checks.py'
spec = spec_from_file_location('spatial_rotation_weights', HELPER)
spatial = module_from_spec(spec)
spec.loader.exec_module(spatial)
DEPTHS, ASSIGNMENTS = spatial.DEPTHS, spatial.ASSIGNMENTS
CONDITIONS, ERRORS = {}, {}
GRID = (0., .25, .5, .75, .99, 1.1, 1.3, 1.5, 1.6, 1.75, 1.9, 2.1, 2.4, 3., 4., 6.)


def check(value, label):
    if not bool(value):
        raise AssertionError(label)
    CONDITIONS[label] = CONDITIONS.get(label, 0)+1


def close(actual, expected, label, tolerance=3e-9):
    actual, expected = np.asarray(actual), np.asarray(expected)
    error = float(np.max(np.abs(actual-expected))/max(1.0, float(np.max(np.abs(expected)))))
    ERRORS[label] = max(ERRORS.get(label, 0.0), error)
    check(error < tolerance, label)


def data(depth, cutoff, sigma=1):
    rotations, weights = spatial.compressed(*spatial.joint_weights(depth, cutoff, sigma)[1:])
    b = np.sqrt(weights)
    potential = 4*np.eye(len(b))-3*np.outer(b, b)
    return rotations.astype(float), weights, b, potential


def frequency_matrix(rotations, potential, speed):
    transport = np.diag(speed*rotations)
    return np.block([[transport, np.eye(len(rotations))], [potential, transport]])


def energy_threshold(rotations, weights, cutoff):
    if not cutoff:
        return None
    def deficit(g):
        return float(1-3*np.sum(weights/(4-(g*rotations/cutoff)**2)))
    threshold = brentq(deficit, 1, 2-1e-11, xtol=3e-13, rtol=3e-14)
    check(1 < threshold < 2, 'positive rotating energy threshold lies between universal bounds')
    _, _, _, potential = data_for_energy(rotations, weights)
    minimum = eigvalsh(potential-np.diag((threshold*rotations/cutoff)**2))[0]
    close(minimum, 0, 'secular energy threshold matches complete stiffness matrix')
    return threshold


def data_for_energy(rotations, weights):
    b = np.sqrt(weights)
    return rotations, weights, b, 4*np.eye(len(b))-3*np.outer(b, b)


def bright_state(rotations, weights, potential, speed):
    def secular(z):
        return float(1-3*np.sum(weights/(4-(z-speed*rotations)**2)))
    z = brentq(secular, .5, 1.5, xtol=2e-14, rtol=1e-14)
    vector = np.sqrt(weights)/(4-(z-speed*rotations)**2)
    vector /= np.linalg.norm(vector)
    covariant_frequency = z-speed*rotations
    residual = potential@vector-covariant_frequency**2*vector
    close(residual, np.zeros_like(vector), 'bright frequency solves complete quadratic pencil')
    charge = 2*float(np.dot(covariant_frequency, vector**2))
    angular_charge = -2*float(np.dot(rotations*covariant_frequency, vector**2))
    energy = 2*float(np.dot(covariant_frequency**2, vector**2))
    close(energy-speed*angular_charge, z*charge,
          'second-order rotating energy retains signed phase and rotation charges')
    check(charge > 0, 'continued slow bright branch has positive signed phase charge')
    return z, vector, charge, angular_charge, energy


def slow_checks(depth, cutoff, rotations, weights, potential):
    if not cutoff:
        z, _, charge, _, energy = bright_state(rotations, weights, potential, 1)
        close(z, 1, 'ground clock is unchanged by any source angle speed')
        close(energy/charge, 1, 'ground field energy per positive charge equals its clock')
        return
    h = .001/cutoff
    def coefficient(step):
        plus = bright_state(rotations, weights, potential, step)
        minus = bright_state(rotations, weights, potential, -step)
        zp, zm = plus[0], minus[0]
        field = (plus[4]/plus[2]+minus[4]/minus[2]-2)/(2*step*step)
        return (zp+zm-2)/(2*step*step), (zp-zm)/(2*step), field
    full, first, full_field = coefficient(h)
    half, _, half_field = coefficient(h/2)
    estimate = (4*half-full)/3
    metric = float(spatial.geometry(depth, cutoff))
    close(estimate, -7*metric/6, 'finite-inertia second-order shift reuses the spatial metric', 7e-5)
    close(first, cutoff/(depth+1), 'slow bright linear shift is the geometric rotation mean', 5e-6)
    close((4*half_field-full_field)/3, 7*metric/6,
          'field energy per signed charge retains the opposite quadratic shift', 7e-5)


def augmented_energy_checks(rotations, weights, potential, speed, count):
    z, u, phase, _, _ = bright_state(rotations, weights, potential, speed)
    shifted = z-speed*rotations
    stiffness = potential-np.diag(shifted**2)
    spectrum = eigvalsh(stiffness)
    close(spectrum[0], 0, 'relative-equilibrium augmented field form has its exact bright zero mode')
    gap = float(spectrum[1]) if len(spectrum) > 1 else None
    if gap is not None:
        check(gap > 0, 'continued relative equilibrium minimizes complete augmented energy transverse to symmetry')
    rng = np.random.default_rng(671)
    q = rng.normal(size=len(u))+1j*rng.normal(size=len(u))
    momentum = rng.normal(size=len(u))+1j*rng.normal(size=len(u))
    source_p = .83
    charge = -2*np.vdot(q, momentum).imag
    angular = source_p+2*np.vdot(q, rotations*momentum).imag
    energy = source_p**2/(2*count)+np.vdot(momentum, momentum).real+np.vdot(q, potential@q).real
    augmented = energy-speed*angular-z*charge+count*speed**2/2
    squares = (source_p-count*speed)**2/(2*count)
    squares += np.vdot(momentum+1j*shifted*q, momentum+1j*shifted*q).real
    squares += np.vdot(q, stiffness@q).real
    close(augmented, squares, 'complete autonomous augmented energy is the sum of rotor momentum and field forms')
    q = u/sqrt(phase)
    momentum = -1j*shifted*q
    source_p = count*speed
    close(-2*np.vdot(q, momentum).imag, 1, 'relative-equilibrium action normalizer is its signed phase charge')
    check(z+abs(speed)*max(abs(rotations)) < 2,
          'continued branch has a positive diagonal secular factor at every retained rotation')
    return {'transverse_augmented_stiffness_gap': gap,
            'positive_diagonal_margin': float(2-max(abs(shifted))),
            'conditional_autonomous_orbital_stability': True}


def onset_checks():
    cases = []
    for depth in DEPTHS:
        cutoff = 2
        rotations, weights, b, potential = data(depth, cutoff)
        low, high = 1.2, 1.6
        def growth(g):
            return float(np.max(eigvals(frequency_matrix(rotations, potential, g/cutoff)).imag))
        check(growth(low) < 1e-8 and growth(high) > .1,
              'all depths have a resolved stable/unstable control bracket')
        for _ in range(34):
            middle = (low+high)/2
            if growth(middle) > 1e-6:
                high = middle
            else:
                low = middle
        zlist = eigvals(frequency_matrix(rotations, potential, high/cutoff))
        starting_z = float(zlist[np.argmax(zlist.imag)].real)
        def equations(x):
            z, g = x
            shifted = z-g*rotations/cutoff
            den = 4-shifted**2
            return [1-3*np.sum(weights/den), -6*np.sum(weights*shifted/den**2)]
        solution = root(equations, [starting_z, high], tol=1e-11)
        close(equations(solution.x), [0, 0], 'instability onset solves secular equation and its frequency derivative')
        z, g = map(float, solution.x)
        check(abs(g-high) < 2e-7, 'independent double-root onset agrees with spectral bisection')
        vector = b/(4-(z-g*rotations/cutoff)**2)
        vector /= np.linalg.norm(vector)
        signed_charge = 2*float(np.dot(z-g*rotations/cutoff, vector**2))
        close(signed_charge, 0, 'colliding frequency branches have zero limiting signed phase charge')
        threshold = energy_threshold(rotations, weights, cutoff)
        check(g > threshold+1e-4, 'loss of positive rotating energy precedes this instability onset')
        cases.append({'depth': depth, 'cutoff': cutoff, 'energy_threshold_g': threshold,
                      'first_control_instability_g': g, 'double_root_frequency_over_lambda': z,
                      'bisection_bracket': [low, high], 'double_root_signed_phase_charge': signed_charge})
    return cases


def growth_checks():
    cases = []
    for depth in DEPTHS:
        rotations, weights, b, potential = data(depth, 2)
        speed = .8
        values, vectors = eig(frequency_matrix(rotations, potential, speed))
        selected = int(np.argmax(values.imag))
        z = values[selected]
        q = vectors[:len(b), selected]
        q /= np.linalg.norm(q)
        momentum = -1j*(z-speed*rotations)*q
        check(z.imag > .1, 'direct growth test uses a resolved unstable frequency')
        def rhs(time, y):
            position, velocity = y[:len(b)], y[len(b):2*len(b)]
            moving = b*np.exp(1j*rotations*speed*time)
            overlap = np.vdot(moving, position)
            mixed = np.vdot(moving, rotations*position)
            force = -4*position+3*moving*overlap
            work = -6*speed*np.imag(np.conj(overlap)*mixed)
            return np.concatenate((velocity, force, [work]))
        endpoint = 2/float(z.imag)
        times = np.linspace(0, endpoint, 25)
        initial = np.concatenate((q, momentum, [0j]))
        solution = solve_ivp(rhs, [0, endpoint], initial, t_eval=times,
                             method='DOP853', rtol=2e-11, atol=2e-12)
        check(solution.success, 'direct second-order moving-source integration completed')
        expected = np.array([np.exp(1j*rotations*speed*t)*q*np.exp(-1j*z*t) for t in times]).T
        close(solution.y[:len(b)], expected, 'direct laboratory growth agrees with complete quadratic pencil', 5e-9)
        energies = []
        for time, y in zip(times, solution.y.T):
            position, velocity = y[:len(b)], y[len(b):2*len(b)]
            moving = b*np.exp(1j*rotations*speed*time)
            energies.append(float(np.vdot(velocity, velocity).real+4*np.vdot(position, position).real-
                                  3*abs(np.vdot(moving, position))**2))
        close(np.array(energies)/energies[0], np.exp(2*z.imag*times),
              'unstable mode energy grows at twice its amplitude growth rate', 5e-9)
        close(np.array(energies)-energies[0], solution.y[-1].real,
              'prescribed source work accounts for the growing field energy', 5e-9)
        cases.append({'depth': depth, 'cutoff': 2, 'rotation_g': 1.6,
                      'amplitude_growth_over_lambda': float(z.imag),
                      'amplitude_efolding_in_target_clock_cycles': 1/(2*pi*float(z.imag)),
                      'time_in_lambda_units': endpoint, 'integration_evaluations': solution.nfev})
    return cases


def reciprocal_checks():
    cases = []
    for name, depth, cutoff in [('control', d, 2) for d in DEPTHS]+[('electron', 6, 12), ('muon', 6, 34)]:
        rotations, weights, b, potential = data(depth, cutoff)
        count = comb(cutoff+depth, depth)
        q = b+(.07+.03j)*(rotations-np.dot(rotations, weights))*b/(cutoff+1)
        q /= sqrt(2)*np.linalg.norm(q)
        momentum = -1j*q
        n = len(b)
        initial = np.concatenate(([.17, .83], q.real, q.imag, momentum.real, momentum.imag))
        def unpack(y):
            return y[0], y[1], y[2:2+n]+1j*y[2+n:2+2*n], y[2+2*n:2+3*n]+1j*y[2+3*n:]
        def rhs(time, y):
            theta, source_p, position, velocity = unpack(y)
            moving = b*np.exp(1j*rotations*theta)
            overlap = np.vdot(moving, position)
            mixed = np.vdot(moving, rotations*position)
            force = -4*position+3*moving*overlap
            torque = 6*np.imag(np.conj(overlap)*mixed)
            return np.concatenate(([source_p/count, torque], velocity.real, velocity.imag, force.real, force.imag))
        solution = solve_ivp(rhs, [0, 6], initial, t_eval=np.linspace(0, 6, 31),
                             method='DOP853', rtol=2e-11, atol=2e-12)
        check(solution.success, 'autonomous second-order source integration completed')
        energies, phases, angular, norms, source_momenta = [], [], [], [], []
        for y in solution.y.T:
            theta, source_p, position, velocity = unpack(y)
            moving = b*np.exp(1j*rotations*theta)
            energy = source_p**2/(2*count)+np.vdot(velocity, velocity).real
            energy += 4*np.vdot(position, position).real-3*abs(np.vdot(moving, position))**2
            energies.append(float(energy))
            phases.append(-2*np.vdot(position, velocity).imag)
            angular.append(source_p+2*np.vdot(position, rotations*velocity).imag)
            norms.append(float(np.vdot(position, position).real))
            source_momenta.append(source_p)
        close(energies, np.full(len(energies), energies[0]), 'complete second-order field/source energy conserved', 5e-9)
        close(phases, np.ones(len(phases)), 'signed phase charge is conserved with both oscillator branches', 5e-9)
        close(angular, np.full(len(angular), angular[0]), 'complete second-order source/field rotation charge conserved', 5e-9)
        check(np.ptp(norms) > 1e-7, 'configuration norm is not substituted for the signed phase charge')
        check(np.ptp(source_momenta) > 1e-7, 'second-order source backreaction is retained')
        cases.append({'name': name, 'depth': depth, 'cutoff': cutoff, 'simplex_count': count,
                      'interval_in_lambda_units': 6, 'source_momentum_range': float(np.ptp(source_momenta)),
                      'configuration_norm_range': float(np.ptp(norms)),
                      'relative_energy_drift': float(np.max(np.abs(np.array(energies)/energies[0]-1))),
                      'signed_phase_charge_drift': float(np.max(np.abs(np.array(phases)-1))),
                      'combined_rotation_charge_drift': float(np.max(np.abs(np.array(angular)-angular[0])))})
    return cases


def convective_checks():
    for depth in DEPTHS:
        rotations, weights, b, potential = data(depth, 2)
        speed, theta, inertia = .73, .37, 1.3
        phase = np.exp(1j*rotations*theta)
        q = b+(.07+.03j)*(rotations-np.dot(rotations, weights))*b
        qdot = (.11-.31j)*q+.07*rotations*q
        velocity = qdot-1j*speed*rotations*q
        chi = np.conj(phase)*q
        chidot = np.conj(phase)*velocity
        moving = b*phase
        energy = 4*np.vdot(q, q).real-3*abs(np.vdot(moving, q))**2
        original_l = np.vdot(velocity, velocity).real-energy+inertia*speed**2/2
        reduced_l = np.vdot(chidot, chidot).real-np.vdot(chi, potential@chi).real+inertia*speed**2/2
        close(original_l, reduced_l, 'full convective second-order kinetic action rotates without omitted terms')
        j = 2*np.vdot(q, rotations*velocity).imag
        canonical_p = inertia*speed-j
        complete_h = (canonical_p+j)**2/(2*inertia)+np.vdot(velocity, velocity).real+energy
        close(complete_h, inertia*speed**2/2+np.vdot(chidot, chidot).real+
              np.vdot(chi, potential@chi).real,
              'convective source momentum shift retains positive complete Hamiltonian')
        close(potential@b, b, 'inverse bright frequency square remains one in the full convective action')


def main():
    assignments, scans = [], []
    for name, depth, index in ASSIGNMENTS:
        cutoff, count = index-1, comb(index-1+depth, depth)
        rotations, weights, b, potential = data(depth, cutoff)
        threshold = energy_threshold(rotations, weights, cutoff)
        slow_checks(depth, cutoff, rotations, weights, potential)
        for ratio in (.1, 1., 10.):
            speed = ratio/count
            matrix = frequency_matrix(rotations, potential, speed)
            values = eigvals(matrix)
            growth = float(np.max(values.imag))
            if cutoff == 0 or ratio*cutoff/count < threshold:
                check(growth < 2e-8, 'maintained state is spectrally stable whenever rotating energy is positive')
            record = {'name': name, 'depth': depth, 'mode_index': index, 'cutoff': cutoff,
                      'simplex_count': count, 'source_omega_over_sector_mu': ratio,
                      'source_omega_over_target_lambda': speed, 'rotation_g': speed*cutoff,
                      'energy_threshold_g': threshold, 'amplitude_growth_over_lambda': growth,
                      'amplitude_efolding_in_target_clock_cycles': 1/(2*pi*growth) if growth > 2e-8 else None,
                      'resolved_growing_mode': growth > 2e-8, 'oscillator_frequency_count': len(values)}
            if speed*cutoff < .25:
                z, vector, charge, j, energy = bright_state(rotations, weights, potential, speed)
                record.update(rotating_bright_frequency_over_lambda=z, field_energy_per_signed_charge_over_lambda=energy/charge,
                              field_rotation_charge_per_signed_charge=j/charge,
                              complete_energy_per_unit_signed_charge_over_lambda=energy/charge+ratio*ratio/(2*count))
                record.update(augmented_energy_checks(rotations, weights, potential, speed, count))
            assignments.append(record)
        for g in GRID:
            speed = g/max(1, cutoff)
            values = eigvals(frequency_matrix(rotations, potential, speed))
            growth = float(np.max(values.imag))
            if not cutoff or g < threshold:
                check(growth < 2e-8, 'all sampled positive-energy spectra retain both real frequency branches')
            scans.append({'name': name, 'depth': depth, 'cutoff': cutoff, 'rotation_g': g,
                          'source_omega_over_target_lambda': speed, 'amplitude_growth_over_lambda': growth,
                          'amplitude_efolding_in_target_clock_cycles': 1/(2*pi*growth) if growth > 2e-8 else None})
        other = data(depth, cutoff, -1)[3]
        reversal = np.eye(len(b))[::-1]
        transform = np.block([[reversal, np.zeros_like(reversal)], [np.zeros_like(reversal), -reversal]])
        plus = frequency_matrix(rotations, potential, .37/max(1, cutoff))
        minus = frequency_matrix(rotations, other, .37/max(1, cutoff))
        close(transform@plus@transform, -minus, 'complete opposite-polarization frequency spectra are exact mirrors')
    for depth in DEPTHS:
        rotations, weights, _, _ = data(depth, 1)
        threshold = energy_threshold(rotations, weights, 1)
        close(threshold, 2*sqrt((depth+1)/(4*depth+1)), 'first-cutoff positive energy threshold has the exact common formula')
    onset = onset_checks()
    growth = growth_checks()
    reciprocal = reciprocal_checks()
    convective_checks()
    report = {'experiment_key': 'actual-spatial-second-order-moving-contact',
              'created_utc': datetime.now(timezone.utc).isoformat(),
              'sources': {'experiments/second-order-contact-checks.py': sha256(Path(__file__).read_bytes()).hexdigest(),
                          'experiments/moving-contact-checks.py': sha256(HELPER.read_bytes()).hexdigest()},
              'classification': 'conditional full finite-inertia count-projector and collective-source comparison',
              'assumptions': ['same positive sphere-band spatial evaluation kernel and actual rotation weights',
                              'inverse restoring square lambda^2(4I-3P), lambda=mu_d S_N is stipulated',
                              'positive complex second-order canonical kinetic action with both oscillator branches',
                              'bare spatial Dirac dispersion is not included in this isolated inverse-restoring control',
                              'source path is a prescribed great circle or a new positive reciprocal collective rotor',
                              'comparison source inertia I mu_d=1, never inferred as observed particle inertia'],
              'limits': ['finite scans are not an exhaustive stability chart',
                         'a growing internal mode is not an established outgoing decay channel or observed lifetime',
                         'electron persistence is assessed separately; short-lived particles need not be infinitely stable',
                         'no local master source, selected indices, full spatial support or relativistic particle is derived',
                         'no special depth-five selection or new favorable sector-assignment evidence'],
              'condition_count': len(CONDITIONS), 'evaluation_count': sum(CONDITIONS.values()),
              'conditions': CONDITIONS, 'maximum_errors': ERRORS,
              'assignments': assignments, 'rotation_scans': scans, 'low_cutoff_instability_onsets': onset,
              'direct_growth': growth, 'reciprocal_source': reciprocal}
    destination = ROOT/('second-order-contact-results-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')+'.json')
    destination.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'report': str(destination), 'conditions': report['condition_count'],
                      'evaluations': report['evaluation_count'], 'assignment_cases': len(assignments),
                      'scan_cases': len(scans), 'maximum_errors': ERRORS}, indent=2), flush=True)


if __name__ == '__main__':
    main()
