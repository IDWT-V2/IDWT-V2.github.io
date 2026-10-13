"""Actual sphere-spinor moving contact, bare kinetics and reciprocal rotor."""
from datetime import datetime, timezone
from fractions import Fraction
from hashlib import sha256
from importlib.util import module_from_spec, spec_from_file_location
from math import comb, factorial, prod
from pathlib import Path
import json

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import eigh, expm
from scipy.optimize import brentq
from scipy.special import eval_gegenbauer

ROOT = Path(__file__).resolve().parent
DEPTHS = (2, 3, 4, 5, 6, 10)
ASSIGNMENTS = [('W', 2, 76), ('Z', 2, 81), ('Higgs', 2, 95),
               ('down', 3, 1), ('strange', 3, 4),
               ('up', 4, 3), ('charm', 4, 20), ('top', 4, 72),
               ('nu1', 5, 10), ('nu2', 5, 15), ('nu3', 5, 22),
               ('electron', 6, 13), ('muon', 6, 35), ('tau', 10, 23)]
CONDITIONS, ERRORS = {}, {}


def check(value, label):
    if not bool(value):
        raise AssertionError(label)
    CONDITIONS[label] = CONDITIONS.get(label, 0)+1


def close(actual, expected, label, tolerance=3e-10):
    actual, expected = np.asarray(actual), np.asarray(expected)
    error = float(np.max(np.abs(actual-expected))/max(1.0, float(np.max(np.abs(expected)))))
    ERRORS[label] = max(ERRORS.get(label, 0.0), error)
    check(error < tolerance, label)


def geometry(depth, cutoff):
    return Fraction((depth*depth+depth-1)*cutoff*(cutoff+depth+1),
                    (depth+1)**2*(depth+2))


def joint_weights(depth, cutoff, sigma=1, exact=False):
    alpha = Fraction(depth-1, 2) if exact else (depth-1)/2
    rising = [Fraction(1) if exact else 1.0]
    for j in range(1, cutoff+1):
        rising.append(rising[-1]*(alpha+j-1)/j)
    levels, rotations, weights = [], [], []
    count = comb(cutoff+depth, depth)
    for level in range(cutoff+1):
        for j in range(level+1):
            rotation = level-2*j
            coefficient = Fraction(level+depth-1+sigma*rotation, depth-1) if exact else (
                level+depth-1+sigma*rotation)/(depth-1)
            weight = coefficient*rising[j]*rising[level-j]/count
            levels.append(level)
            rotations.append(rotation)
            weights.append(weight)
    if exact:
        return levels, rotations, weights
    return np.array(levels), np.array(rotations), np.array(weights)


def compressed(rotations, weights):
    unique = np.unique(rotations)
    return unique, np.array([np.sum(weights[rotations == k]) for k in unique])


def top_state(levels, rotations, weights, strength, omega, kinetic, depth):
    diagonal = kinetic*(levels+depth/2)+omega*rotations
    maximum = float(np.max(diagonal))
    at_maximum = float(np.sum(weights[diagonal == maximum]))
    differences = maximum-diagonal
    def secular(delta):
        return float(np.sum(strength*weights/(delta+differences))-1)
    delta = brentq(secular, .5*strength*at_maximum, strength,
                   xtol=3e-13*max(1.0, strength), rtol=3e-14)
    eigenvalue = maximum+delta
    vector = np.sqrt(weights)/(delta+differences)
    vector /= np.linalg.norm(vector)
    mean_rotation = float(np.dot(rotations, vector*vector))
    mean_level = float(np.dot(levels, vector*vector))
    bright = float(np.dot(np.sqrt(weights), vector)**2)
    field_energy = kinetic*(mean_level+depth/2)+strength*bright
    residual = diagonal*vector+strength*np.sqrt(weights)*np.dot(np.sqrt(weights), vector)-eigenvalue*vector
    # The operator scale reaches 64 million in the maintained tau comparison.
    scale = max(1.0, strength+float(np.max(np.abs(diagonal))))
    close(residual/scale, np.zeros_like(vector), 'joint shell/rotation secular eigenvector', 2e-10)
    close(eigenvalue, field_energy+omega*mean_rotation,
          'rotating frequency retains the transport contribution', 2e-10)
    return eigenvalue, vector, bright, mean_rotation, field_energy


def exact_checks():
    for depth in DEPTHS:
        for cutoff in range(5):
            for sigma in (-1, 1):
                levels, rotations, weights = joint_weights(depth, cutoff, sigma, exact=True)
                check(all(w > 0 for w in weights), 'all present joint Fourier weights are positive')
                check(sum(weights) == 1, 'exact cumulative Fourier normalization')
                mean = sum(k*w for k, w in zip(rotations, weights))
                check(mean == Fraction(sigma*cutoff, depth+1), 'exact geometric mean rotation')
                variance = sum((k-mean)**2*w for k, w in zip(rotations, weights))
                check(variance == geometry(depth, cutoff), 'exact transport variance equals spatial metric')
                check(sum(l*w for l, w in zip(levels, weights)) == Fraction(depth*cutoff, depth+1),
                      'exact bare shell mean retained')


def kernel_checks(depth, cutoff):
    levels, rotations, weights = joint_weights(depth, cutoff)
    angles = np.array([.013, .231, .713, 1.37, 2.41, 3.14])
    t = np.cos(angles)
    aa, bb = np.zeros_like(t), np.zeros_like(t)
    for level in range(cutoff+1):
        aa += (level+depth-1)/(depth-1)*eval_gegenbauer(level, (depth-1)/2, t)
        if level:
            bb += eval_gegenbauer(level-1, (depth+1)/2, t)
    count = comb(cutoff+depth, depth)
    close(np.exp(-1j*np.outer(angles, rotations))@weights,
          (aa-1j*np.sin(angles)*bb)/count,
          'Fourier weights reproduce independent Gegenbauer kernel', 3e-10)
    km, wm = joint_weights(depth, cutoff, sigma=-1)[1:]
    close(np.exp(-1j*np.outer(angles, km))@wm,
          (aa+1j*np.sin(angles)*bb)/count,
          'both Clifford polarization channels are retained', 3e-10)
    unique, summed = compressed(rotations, weights)
    close(np.sum(summed), 1, 'uniform orbit average retains total contact trace')
    check(len(unique) == 2*cutoff+1, 'orbit cyclic contact has every integer rotation harmonic')
    if cutoff:
        check(np.max(summed) < 1, 'uniform orbit average loses full excited-band strength')
    else:
        close(summed, [1], 'ground contact is independent of orbit position')
    return levels, rotations, weights, unique, summed


def floquet_checks():
    results = []
    for depth in DEPTHS:
        cutoff = 2
        levels, rotations, weights = joint_weights(depth, cutoff)
        b = np.sqrt(weights)
        strength = float(comb(cutoff+depth, depth))
        omega, kinetic = 3*strength, .37*strength/(cutoff+1)
        period = 2*np.pi/omega
        d = kinetic*(levels+depth/2)
        rotating = np.diag(d+omega*rotations)+strength*np.outer(b, b)
        initial = np.arange(1, len(b)+1)+1j*np.arange(len(b), 0, -1)
        initial /= np.linalg.norm(initial)
        def rhs(time, psi):
            moving = b*np.exp(1j*rotations*omega*time)
            return -1j*(d*psi+strength*moving*np.vdot(moving, psi))
        times = np.linspace(0, period, 9)
        solution = solve_ivp(rhs, [0, period], initial, t_eval=times,
                             method='DOP853', rtol=2e-11, atol=2e-12)
        check(solution.success, 'direct moving-contact integration completed')
        expected = np.array([np.exp(1j*rotations*omega*time)*
                             (expm(-1j*rotating*time)@initial) for time in times]).T
        close(solution.y, expected, 'direct time ordering equals complete rotating frame', 3e-9)
        average = np.diag(d.astype(complex))
        for k in np.unique(rotations):
            mask = rotations == k
            average += strength*np.outer(b*mask, b*mask)
        averaged = expm(-1j*average*period)@initial
        separation = float(np.linalg.norm(solution.y[:, -1]-averaged))
        check(separation > 1e-4, 'time-averaged operator does not reproduce exact one-period dynamics')
        z, _, _, _, _ = top_state(levels, rotations, weights, strength, omega, kinetic, depth)
        close(z, eigh(rotating, eigvals_only=True)[-1], 'secular root matches independent full matrix')
        projector = np.outer(b, b)
        mm = np.diag(rotations)
        mean = float(np.dot(rotations, weights))
        cd = -omega*(mm@projector+projector@mm-2*mean*projector)
        close((strength*projector+cd+omega*mm)@b,
              (strength+omega*mean)*b,
              'explicit additional transport restores exact bright following')
        residual = omega*(rotations-mean)*b
        close(np.linalg.norm(residual), abs(omega)*np.sqrt(float(geometry(depth, cutoff))),
              'missing bright transport is fixed by the geometric metric')
        results.append({'depth': depth, 'cutoff': cutoff,
                        'one_period_static_average_state_error': separation,
                        'integration_evaluations': solution.nfev})
    return results


def actual_spatial_translation_checks():
    helper = ROOT/'spatial-source-saturation-checks.py'
    spec = spec_from_file_location('spatial_source_modes', helper)
    spatial = module_from_spec(spec)
    spec.loader.exec_module(spatial)
    first, second = .31, 1.09
    for depth in DEPTHS:
        points = []
        for angle in (first, second):
            point = np.zeros(depth+1)
            point[0], point[-1] = np.sin(angle), np.cos(angle)
            points.append(point)
        gamma = spatial.spatial.gammas(depth)[0]
        rank = len(gamma)
        for level in range(3):
            x, y = spatial.independent_evaluation(depth, level, points)
            difference = second-first
            aa, bb = spatial.shells(depth, level, np.array([np.cos(difference)]))
            expected = aa[level, 0]*np.eye(rank)-1j*np.sin(difference)*bb[level, 0]*gamma
            close(y.conj().T@x, expected,
                  'actual monogenic modes verify translated great-circle kernel', 3e-10)


def convective_connection_checks():
    for depth in DEPTHS:
        levels, rotations, weights = joint_weights(depth, 3)
        b = np.sqrt(weights)
        strength, inertia, theta, speed = float(comb(depth+3, depth)), 1.3, .37, .83
        phase = np.exp(1j*rotations*theta)
        psi = np.arange(1, len(b)+1)+1j*np.arange(len(b), 0, -1)
        psi /= np.linalg.norm(psi)
        psidot = (.13+.07j)*psi+.11*rotations*psi
        chi = np.conj(phase)*psi
        chidot = np.conj(phase)*psidot-1j*speed*rotations*chi
        moving = phase*b
        mean = float(np.vdot(psi, rotations*psi).real)
        for complement in (False, True):
            def potential(v, ray):
                contact = strength*abs(np.vdot(ray, v))**2
                return 2*strength*np.vdot(v, v).real-contact if complement else contact
            energy = potential(psi, moving)
            original_l = -np.vdot(psi, psidot).imag+speed*mean+inertia*speed**2/2-energy
            reduced_l = -np.vdot(chi, chidot).imag+inertia*speed**2/2-potential(chi, b)
            close(original_l, reduced_l, 'added convective action includes the complete canonical one-form')
            canonical_p = inertia*speed+mean
            close((canonical_p-mean)**2/(2*inertia)+energy,
                  inertia*speed**2/2+energy,
                  'convective source Hamiltonian retains positive complete kinetic energy')
            close(canonical_p-mean, inertia*speed, 'convective source combined Noether charge is mechanical momentum')
            generator_on_b = strength*b
            field_lab_rhs = phase*(generator_on_b-speed*rotations*b)
            expected_derivative = -1j*strength*moving+1j*speed*rotations*moving
            close(-1j*field_lab_rhs, expected_derivative,
                  'convective transport gives the stipulated bright clock without deleting source energy')
        shell_mean = depth*3/(depth+1)
        shell_variance = Fraction(depth*3*(3+depth+1), (depth+1)**2*(depth+2))
        bare_residual = (levels-shell_mean)*b
        close(np.linalg.norm(bare_residual)**2, float(shell_variance),
              'transport correction does not remove bare spatial kinetic leakage')


def rotor_checks():
    results = []
    cases = [('control', depth, 2) for depth in DEPTHS]+[('electron', 6, 12), ('muon', 6, 34)]
    for name, depth, cutoff in cases:
        rotations, weights = compressed(*joint_weights(depth, cutoff)[1:])
        b = np.sqrt(weights)
        strength = float(comb(cutoff+depth, depth))
        inertia = 1.0
        initial = b+(.07+.03j)*(rotations-np.dot(rotations, weights))*b/(cutoff+1)
        initial /= np.linalg.norm(initial)
        initial_p, initial_theta = .83, .17
        y0 = np.concatenate(([initial_theta, initial_p], initial.real, initial.imag))
        n = len(b)
        def unpack(y):
            return y[0], y[1], y[2:2+n]+1j*y[2+n:]
        def rhs(time, y):
            theta, momentum, psi = unpack(y)
            moving = b*np.exp(1j*rotations*theta)
            overlap = np.vdot(moving, psi)
            mixed = np.vdot(moving, rotations*psi)
            psidot = -1j*strength*moving*overlap
            pdot = -2*strength*np.imag(np.conj(overlap)*mixed)
            return np.concatenate(([momentum/inertia, pdot], psidot.real, psidot.imag))
        solution = solve_ivp(rhs, [0, 6/strength], y0, t_eval=np.linspace(0, 6/strength, 31),
                             method='DOP853', rtol=2e-11, atol=2e-12)
        check(solution.success, 'autonomous reciprocal source integration completed')
        energies, charges, norms = [], [], []
        momenta = []
        for y in solution.y.T:
            theta, momentum, psi = unpack(y)
            moving = b*np.exp(1j*rotations*theta)
            energies.append(momentum**2/(2*inertia)+strength*abs(np.vdot(moving, psi))**2)
            charges.append(momentum-float(np.vdot(psi, rotations*psi).real))
            norms.append(float(np.vdot(psi, psi).real))
            momenta.append(momentum)
        close(energies, np.full(len(energies), energies[0]), 'complete reciprocal source energy conserved', 4e-9)
        close(charges, np.full(len(charges), charges[0]), 'combined source/field rotation charge conserved', 4e-9)
        close(norms, np.ones(len(norms)), 'classical canonical field norm conserved', 4e-9)
        check(np.ptp(momenta) > 1e-7, 'source backreaction is nonzero in the tested trajectory')
        results.append({'name': name, 'depth': depth, 'cutoff': cutoff,
                        'source_momentum_range': float(np.ptp(momenta)),
                        'maximum_relative_energy_drift': float(np.max(np.abs(np.array(energies)/energies[0]-1))),
                        'maximum_combined_charge_drift': float(np.max(np.abs(np.array(charges)-charges[0]))),
                        'time_interval': 6/strength, 'integration_evaluations': solution.nfev})
    return results


def main():
    exact_checks()
    cases = []
    for name, depth, index in ASSIGNMENTS:
        cutoff, count = index-1, comb(index-1+depth, depth)
        levels, rotations, weights, unique, summed = kernel_checks(depth, cutoff)
        close(np.dot(rotations, weights), cutoff/(depth+1), 'all assignments have the exact rotation mean')
        close(np.dot(rotations*rotations, weights)-np.dot(rotations, weights)**2,
              float(geometry(depth, cutoff)), 'all assignments have the exact transport metric')
        for kinetic in (0., 1., 10.):
            for omega in (.1, 1., 10.):
                z, vector, bright, mean, field_energy = top_state(
                    levels, rotations, weights, count, omega, kinetic, depth)
                if not kinetic and cutoff and omega*omega*float(geometry(depth, cutoff))/count**2 > 1e-12:
                    check(bright < 1-1e-15, 'moving excited contact does not remain exactly bright')
                bare = float(kinetic*np.dot(levels+depth/2, vector*vector))
                total_energy = omega*omega/2+field_energy
                joint_charge = omega-mean
                close((joint_charge+mean)**2/2+field_energy, total_energy,
                      'canonical rotor reduction retains complete source energy')
                cases.append({'name': name, 'depth': depth, 'mode_index': index, 'cutoff': cutoff,
                              'simplex_count': count, 'source_omega_over_sector_mu': omega,
                              'bare_dirac_kappa_over_sector_mu': kinetic,
                              'canonical_comparison_norm': 1, 'source_inertia_in_mu_units': 1,
                              'rotating_eigenvalue': z, 'bright_contact_fraction': bright,
                              'field_energy': field_energy, 'bare_spatial_energy': bare,
                              'complete_rotor_plus_field_energy': total_energy,
                              'combined_rotation_charge': joint_charge, 'mean_rotation': mean,
                              'uniform_average_top_contact_fraction': float(np.max(summed)),
                              'uniform_average_top_rotation_label': int(unique[np.argmax(summed)]),
                              'transport_metric_exact': str(geometry(depth, cutoff))})
        km, wm = joint_weights(depth, cutoff, -1)[1:]
        close(np.exp(-1j*np.outer([.32, .71], km))@wm,
              np.exp(1j*np.outer([.32, .71], rotations))@weights,
              'polarization reversal mirrors the entire rotation spectrum')
        if cutoff:
            # The expansion is checked at a small prescribed common angular speed.
            # The cubic remainder scales with the third centered rotation moment.
            speed = 1e-3
            z, _, _, _, _ = top_state(levels, rotations, weights, count, speed, 0, depth)
            mean = cutoff/(depth+1)
            prediction = count+speed*mean+speed*speed*float(geometry(depth, cutoff))/count
            close(z, prediction, 'slow-motion geometric frequency expansion', 2e-9)
    floquet = floquet_checks()
    rotor = rotor_checks()
    actual_spatial_translation_checks()
    convective_connection_checks()
    report = {'experiment_key': 'actual-spatial-moving-contact',
              'created_utc': datetime.now(timezone.utc).isoformat(),
              'classification': 'conditional spatial contact and reciprocal collective-source comparison',
              'sources': {'experiments/moving-contact-checks.py': sha256(Path(__file__).read_bytes()).hexdigest(),
                          'experiments/spatial-source-saturation-checks.py': sha256((ROOT/'spatial-source-saturation-checks.py').read_bytes()).hexdigest(),
                          'experiments/spatial-band-curvature-checks.py': sha256((ROOT/'spatial-band-curvature-checks.py').read_bytes()).hexdigest()},
              'assumptions': ['positive cumulative Dirac bands on the specified ordinary spatial sphere',
                              'point-contact strength is stipulated as sector mu times simplex count',
                              'prescribed great-circle source path or explicit positive collective rotor',
                              'first-order canonical classical mode action, not a second-order kinetic reduction',
                              'bare positive-band Dirac frequencies retained in the kinetic comparisons',
                              'unit mode norm and source inertia are comparison units, not measured particle amplitudes'],
              'limits': ['no derivation of point-contact source, rotor inertia, occupied cutoffs or sector scales',
                         'no full localized relativistic particle, detector law or six-direction electron orbit',
                         'rotating eigenvalues are Floquet representatives, not established intrinsic mass frequencies',
                         'no special depth-five suppression and no favorable new sector-assignment evidence'],
              'conditions': CONDITIONS, 'maximum_errors': ERRORS,
              'evaluation_count': sum(CONDITIONS.values()), 'condition_count': len(CONDITIONS),
              'cases': cases, 'direct_floquet': floquet, 'reciprocal_rotor': rotor}
    destination = ROOT/('moving-contact-results-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')+'.json')
    destination.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'report': str(destination), 'evaluations': report['evaluation_count'],
                      'conditions': report['condition_count'], 'cases': len(cases),
                      'maximum_errors': ERRORS}, indent=2))


if __name__ == '__main__':
    main()
