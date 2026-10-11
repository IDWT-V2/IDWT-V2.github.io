#!/usr/bin/env python3
"""Bounded passive Hodge interference, retarded return and all-sector combs."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import importlib.util
from itertools import combinations
import json
import math
from pathlib import Path

import numpy as np
import sympy as sp


CHECKS = 0
CASES = []


def check(condition, message):
    global CHECKS
    if not condition:
        raise AssertionError(message)
    CHECKS += 1


def close(actual, expected, message, atol=2e-12):
    check(np.allclose(actual, expected, atol=atol, rtol=2e-11), message)


def orientation(index):
    return (-1)**sum(index[i] > index[j] for i in range(len(index)) for j in range(i+1, len(index)))


def transfers(star, omega, delay, common_delay=0.):
    phase = np.exp(1j*omega*delay)
    common = np.exp(1j*omega*common_delay)
    return common*(star+phase*star.T)/2, common*(star-phase*star.T)/2


def algebra_checks():
    rng = np.random.default_rng(2026100922)
    for m in range(1, 7):
        seen = set()
        block_count = 0
        for index in combinations(range(2*m), m):
            if index in seen:
                continue
            other = tuple(i for i in range(2*m) if i not in index)
            star = np.array([[0., orientation(other+index)], [orientation(index+other), 0.]])
            seen.update((index, other))
            block_count += 1
            close(star.T, (-1)**m*star, 'actual oriented Hodge block adjoint')
            for omega in (0., .17, 1., 2.3, math.pi):
                t, r = transfers(star, omega, .7, .2)
                matrix = np.block([[t, r], [r, t]])
                close(matrix.conj().T@matrix, np.eye(4), 'complete converter conserves power')
                close(t.conj().T@t+r.conj().T@r, np.eye(2), 'two output gains sum to identity')
                gain = (np.sin(omega*.7/2) if m % 2 else np.cos(omega*.7/2))**2
                close(t.conj().T@t, gain*np.eye(2), 'exact parity-selective forward gain')
                if block_count == 1:
                    scattering = np.block([[np.zeros((4, 4)), matrix.T],
                                           [matrix, np.zeros((4, 4))]])
                    close(scattering.T, scattering, 'complete bidirectional scattering is reciprocal')
                    close(scattering.conj().T@scattering, np.eye(8), 'reciprocal completion is unitary')
                    tm, rm = transfers(star, -omega, .7, .2)
                    close(tm, t.conj(), 'transfer is real in time domain')
                    f = np.exp(1j*omega*.2)*(1+(-1)**m*np.exp(1j*omega*.7))/2
                    close(t.T@t, f*f*np.eye(2), 'causal roundtrip uses transpose at same frequency')
                    if omega > 0:
                        check(abs(f*f-abs(f)**2) > 1e-3, 'forward gain and retarded return differ')
                    k, mass, gamma = 2., 1., .3
                    load = (k-mass*omega**2+1j*gamma*omega)/(k-mass*omega**2-1j*gamma*omega)
                    close(abs(load), 1., 'positive oscillator wave-line reflection is allpass')
                    u = rng.normal(size=2)+1j*rng.normal(size=2)
                    out1 = t.T@(load*t@u)
                    out2 = r.T@(load*t@u)
                    out3 = r@u
                    close(sum(np.vdot(v, v).real for v in (out1, out2, out3)), np.vdot(u, u).real,
                          'loaded three-output network conserves total field energy')
            close(transfers(star, 0., .7)[0], (star+star.T)/2, 'zero frequency is symmetric Hodge filter')
        check(block_count == math.comb(2*m-1, m-1), 'all middle-form orientation blocks covered')
        CASES.append({'name': 'complete-Hodge-passive-converter', 'real_depth': 2*m,
                      'parameters': {'middle_degree': m, 'oriented_blocks': block_count},
                      'measurements': {'static_transmission': 'zero' if m % 2 else 'unit gain',
                                       'gain': 'sin^2(Omega tau/2)' if m % 2 else 'cos^2(Omega tau/2)',
                                       'full_scattering_unitary_and_symmetric': True}})
    for k, mass, gamma in ((2., 1., .3), (3., 2., 7.)):
        poles = np.roots([mass, 1j*gamma, -k])
        check(np.max(poles.imag) < 0, 'positive oscillator reflection poles strictly below real axis')
    # Exact algebra keeps the time arguments of the retarded and advanced terms explicit.
    z, s = sp.symbols('z s', real=True)
    check(sp.expand(((1+s*z)/2)**2) == (1+2*s*z+s**2*z**2)/4,
          'causal return only has nonnegative delays zero tau two-tau')
    for m in range(1, 7):
        for zc in (.3+.2j, -.7+1.4j, 1.+3j):
            star = np.array([[0., (-1)**m], [1., 0.]])
            t, r = transfers(star, zc, .7, .2)
            check(np.linalg.norm(np.block([[t, r], [r, t]]), 2) <= 1+1e-12,
                  'delayed full converter is contractive in causal upper half plane')


def pulse_checks():
    rng = np.random.default_rng(997)
    length, lag, common = 64, 7, 3
    inputs = rng.normal(size=(2, 2, length))
    total_length = length+lag+common
    for m in range(1, 7):
        star = np.array([[0., (-1)**m], [1., 0.]])
        route1 = star@(inputs[0]+inputs[1])/np.sqrt(2)
        route2 = star.T@(inputs[0]-inputs[1])/np.sqrt(2)
        a = np.zeros((2, total_length)); b = a.copy()
        a[:, common:common+length] = route1
        b[:, common+lag:common+lag+length] = route2
        outputs = np.stack([(a+b)/np.sqrt(2), (a-b)/np.sqrt(2)])
        close(np.sum(outputs**2), np.sum(inputs**2), 'finite pulse complete two-port energy conservation')
        check(np.max(abs(outputs[:, :, :common])) == 0., 'no pulse output before first physical delay')
        # Two actual input vectors, arbitrary overlap, no quasistatic energy approximation.
        CASES.append({'name': 'causal-finite-pulse', 'real_depth': 2*m,
                      'parameters': {'input_samples': length, 'path_delays_samples': [common, common+lag]},
                      'measurements': {'input_norm2': float(np.sum(inputs**2)),
                                       'output_norm2': float(np.sum(outputs**2)),
                                       'first_allowed_output_sample': common}})


def local_wave_action_checks():
    """Real boundary projectors realize each converter with positive wave energy."""
    rng = np.random.default_rng(231)
    hadamard = np.array([[1., 1.], [1., -1.]])/np.sqrt(2)
    for m in range(1, 7):
        star = np.array([[0., (-1)**m], [1., 0.]])
        for converter in (np.kron(hadamard, np.eye(2)), star, star.T):
            size = len(converter)
            scattering = np.block([[np.zeros((size, size)), converter.T],
                                   [converter, np.zeros((size, size))]])
            plus = (np.eye(2*size)+scattering)/2
            minus = np.eye(2*size)-plus
            close(plus@minus, 0., 'Dirichlet and Neumann boundary projectors are orthogonal')
            for wave_number in (.01, .7, 3., 100.):
                derived = -np.linalg.solve(minus+1j*wave_number*plus, minus-1j*wave_number*plus)
                close(derived, scattering, 'local selfadjoint wave boundary produces desired scattering')
                incident = rng.normal(size=2*size)+1j*rng.normal(size=2*size)
                outgoing = derived@incident
                value = incident+outgoing
                derivative = 1j*wave_number*(outgoing-incident)
                close(minus@value, 0., 'incident plus outgoing satisfies value boundary condition')
                close(plus@derivative, 0., 'incident plus outgoing satisfies derivative boundary condition')
            boundary_velocity = plus@rng.normal(size=2*size)
            boundary_derivative = minus@rng.normal(size=2*size)
            close(boundary_velocity@boundary_derivative, 0., 'local wave energy flux vanishes at junction')
            # Independent positive spatial-energy form with a finite ordinary path discretization.
            eigen, vectors = np.linalg.eigh(plus)
            basis0 = vectors[:, eigen > .5]
            segments = 5
            dofs = basis0.shape[1]+(segments-1)*2*size
            nodes = []
            node0 = np.zeros((2*size, dofs)); node0[:, :basis0.shape[1]] = basis0
            nodes.append(node0)
            for j in range(segments-1):
                node = np.zeros((2*size, dofs))
                start = basis0.shape[1]+j*2*size
                node[:, start:start+2*size] = np.eye(2*size)
                nodes.append(node)
            nodes.append(np.zeros((2*size, dofs)))
            stiffness = sum(((nodes[j+1]-nodes[j]).T@(nodes[j+1]-nodes[j])
                             for j in range(segments)), np.zeros((dofs, dofs)))
            check(np.min(np.linalg.eigvalsh(stiffness)) > 0.,
                  'finite path action with outer Dirichlet control has strictly positive wave energy')
        CASES.append({'name': 'positive-local-wave-action', 'real_depth': 2*m,
                      'parameters': {'boundary_conditions': 'P_minus u=0, P_plus outward_derivative u=0',
                                     'wave_action': '1/2 sum integral[(u_t)^2-c^2(u_s)^2] ds dt',
                                     'converter_maps': 'Hadamard, star, star transpose'},
                      'measurements': {'boundary_energy_flux': '0', 'wave_energy_nonnegative': True,
                                       'local_boundary_scattering_frequency_independent': True}})


def limiting_volume_moment(delta, odd):
    delta = delta % (2*math.pi)
    remainder = 2*math.pi-delta
    if odd:
        return (delta*remainder**6+remainder*delta**6)/(2*math.pi*4**6)
    return (delta**7+remainder**7)/(14*math.pi*4**6)


def finite_volume_moment(cutoff, delta, odd):
    # More than 6K samples integrate the sixth-power trigonometric polynomial exactly.
    samples = 1 << (6*cutoff).bit_length()
    spectrum = np.zeros(samples, dtype=complex)
    k = np.arange(1, cutoff+1, dtype=float)
    phase = (delta % (2*math.pi))
    s = -1 if odd else 1
    values = -1j/(2*k)*(1+s*np.exp(-1j*k*phase))/2
    spectrum[1:cutoff+1] = values
    spectrum[-cutoff:] = values[::-1].conj()
    waveform = np.fft.ifft(spectrum).real*samples
    return float(np.mean(waveform**6))


def volume_delay_checks(ratios):
    delta = sp.symbols('delta', positive=True)
    theta = sp.symbols('theta', real=True)
    odd_exact = (delta*(2*sp.pi-delta)**6+(2*sp.pi-delta)*delta**6)/(2*sp.pi*4**6)
    even_exact = (delta**7+(2*sp.pi-delta)**7)/(14*sp.pi*4**6)
    integrated = (sp.integrate(((delta-2*theta)/4)**6, (theta, 0, delta))
                  +sp.integrate(((2*sp.pi+delta-2*theta)/4)**6, (theta, delta, 2*sp.pi)))/(2*sp.pi)
    check(sp.simplify(integrated-even_exact) == 0, 'piecewise limiting even-filter sixth moment')
    check(sp.limit(odd_exact/delta, delta, 0) == sp.pi**5/128,
          'infinite-cutoff odd-color leakage starts linearly in delay')
    reference = math.pi**6/448
    mb, me = math.pi/2, 2*math.pi
    remainder_coefficient = sum(math.comb(6, j)*mb**(6-j)*me**(j-2) for j in range(2, 7))
    for cutoff in (1, 2, 8, 128, 2002, 18564):
        for phase in (.01, .1, 1., math.pi):
            for odd in (False, True):
                actual = finite_volume_moment(cutoff, phase, odd)
                limiting = limiting_volume_moment(phase, odd)
                bound = 6*mb**5/math.sqrt(2*cutoff)+remainder_coefficient/(2*cutoff)
                check(abs(actual-limiting) <= bound, 'finite-sixth-moment convergence obeys rigorous source bound')
                if cutoff == 1:
                    expected = 5/16*(math.sin(phase/2) if odd else math.cos(phase/2))**6
                    close(actual, expected, 'independent exact single-sine sixth moment')
    assigned = [('W', 2, 76), ('Z', 2, 81), ('Higgs', 2, 95), ('down', 3, 1),
                ('strange', 3, 4), ('up', 4, 3), ('charm', 4, 20), ('top', 4, 72),
                ('nu1', 5, 10), ('nu2', 5, 15), ('nu3', 5, 22),
                ('electron', 6, 13), ('muon', 6, 35), ('tau', 10, 23)]
    for name, depth, n in assigned:
        cutoff = math.comb(n+depth-1, depth)
        m = (depth+1)//2
        odd = bool(m % 2)
        rank = math.comb(2*m-1, m-1)
        for law in ('common-receiver-delay', 'source-fundamental-delay'):
            for eta in (.1, 1.):
                phase = ratios[depth]*eta if law == 'common-receiver-delay' else eta
                limiting = limiting_volume_moment(phase, odd)
                bound = 6*mb**5/math.sqrt(2*cutoff)+remainder_coefficient/(2*cutoff)
                value = finite_volume_moment(cutoff, phase, odd) if cutoff <= 100000 else limiting
                error = 0. if cutoff <= 100000 else bound
                if cutoff <= 100000:
                    check(value >= 0., 'prepared finite color-volume energy is nonnegative')
                if depth in (6, 10) and eta == 1.:
                    check(value-error > .5, 'bounded delayed Hodge replacement reactivates lepton higher-volume channel')
                CASES.append({'name': 'finite-delay-color-volume-control', 'real_depth': depth,
                              'mode_index': n, 'basis_cutoff': cutoff,
                              'parameters': {'particle': name, 'delay_law': law, 'shared_eta': eta,
                                             'preparation': 'three equal-frequency equal-phase patterns with nonzero mode volume',
                                             'this_is_a_control_not_synchronization_of_unlike_particles': True},
                              'measurements': {'sixth_waveform_moment': value,
                                               'rigorous_large_cutoff_approximation_bound': error,
                                               'limiting_unfiltered_reference': reference,
                                               'limiting_filtered_reference_ratio': limiting/reference,
                                               'mode_rank': rank,
                                               'three_mode_volume_exists': rank >= 3,
                                               'instantaneous_H_strict_volume_zero': odd}})


def comb_checks(saw):
    assigned = [('W', 2, 76), ('Z', 2, 81), ('Higgs', 2, 95), ('down', 3, 1),
                ('strange', 3, 4), ('up', 4, 3), ('charm', 4, 20), ('top', 4, 72),
                ('nu1', 5, 10), ('nu2', 5, 15), ('nu3', 5, 22),
                ('electron', 6, 13), ('muon', 6, 35), ('tau', 10, 23)]
    ratios = {2: 18564*math.sqrt(2890), 3: 18564*math.sqrt(32*math.sqrt(7)),
              4: 18564/15*math.sqrt(48/math.sqrt(7)),
              5: math.sqrt(2.534e-3/(65780**2-2002**2))*18564/(.51099895000e6),
              6: 1., 10: 1.}
    eta_values = [.01, .1, 1., math.pi, 2*math.pi]
    records = {}
    for name, depth, n in assigned:
        cutoff = math.comb(n+depth-1, depth)
        limit = 2*cutoff if cutoff <= 150000 else 65536
        c0, coefficients = saw.coefficients(cutoff, limit)
        p = np.arange(1, len(coefficients)+1, dtype=float)
        ratio = ratios[depth]
        omega = p*ratio
        chi = (3/(2-omega**2)+2/(3-omega**2))/5
        m = (depth+1)//2
        even = m % 2 == 0
        static = c0*c0*13/30 if even else 0.
        tail = saw.response_sum(cutoff, ratio)['density_tail_bound']
        base = float(np.sum(.5*coefficients**2*chi))
        for law in ('common-receiver-delay', 'source-fundamental-delay'):
            for eta in eta_values:
                phase = omega*eta if law == 'common-receiver-delay' else p*eta
                gain = (np.cos(phase/2) if even else np.sin(phase/2))**2
                value = float(np.sum(.5*coefficients**2*gain*chi))
                complement = float(np.sum(.5*coefficients**2*(1-gain)*chi))
                close(value+complement, base, 'selected plus complementary diagnostic equals unfiltered sum')
                record = {'name': 'all-sector-passive-sawtooth', 'real_depth': depth,
                          'mode_index': n, 'basis_cutoff': cutoff,
                          'parameters': {'particle': name, 'delay_law': law, 'shared_eta': eta,
                                         'fundamental_ratio': ratio, 'evaluated_harmonics': len(coefficients),
                                         'receiver': '(3/(2-Omega^2)+2/(3-Omega^2))/5',
                                         'quantity_is_forward_receiver_diagnostic': True},
                          'measurements': {'dynamic_response': value, 'static_response': static,
                                           'total_response': value+static,
                                           'absolute_omitted_tail_bound': tail,
                                           'complementary_dynamic_response': complement}}
                CASES.append(record)
                records[(name, law, eta)] = record['measurements']
        if depth == 5:
            derivative = saw.response_sum(cutoff, ratio)['derivative_source_response']
            small = records[(name, 'common-receiver-delay', .01)]['dynamic_response']
            close(small/(.01**2/4*derivative), 1., 'common-delay odd channel has checked low-frequency derivative limit', 2e-9)
            intrinsic = records[(name, 'source-fundamental-delay', 1.)]['dynamic_response']
            check(intrinsic > .01, 'source-relative delay removes slow-clock neutrino suppression')
        if depth in (6, 10):
            dark = records[(name, 'common-receiver-delay', 2*math.pi)]['dynamic_response']
            check(abs(dark) < 1e-20, 'unfavorable common two-pi delay makes all charged-lepton comb lines dark')
        dark_or_bright = records[(name, 'source-fundamental-delay', 2*math.pi)]['dynamic_response']
        close(dark_or_bright, base if even else 0., 'intrinsic integer-period delay control')
    # Symmetry-breaking floors are common coherent path perturbations, not fitted particle coefficients.
    for epsilon in (0., 1e-8, 1e-5, .01):
        alpha, beta = (1+epsilon)/2, (1-epsilon)/2
        for phase in (0., 1e-8, 1e-5, .01):
            floor = abs(alpha-beta*np.exp(1j*phase))**2
            expected = epsilon**2+2*alpha*beta*(1-np.cos(phase))
            close(floor, expected, 'odd-channel static mismatch floor')
            CASES.append({'name': 'coherent-path-robustness',
                          'parameters': {'common_amplitude_imbalance': epsilon, 'common_DC_phase_offset': phase},
                          'measurements': {'odd_channel_DC_gain': floor,
                                           'neutrino_limiting_DC_response': float((math.pi**2/12)**2*13/30*floor)}})
    return ratios


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sawtooth-source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location('sawtooth', args.sawtooth_source)
    saw = importlib.util.module_from_spec(spec); spec.loader.exec_module(saw)
    algebra_checks()
    pulse_checks()
    local_wave_action_checks()
    ratios = comb_checks(saw)
    volume_delay_checks(ratios)
    report = {'title': 'Passive Hodge interference and causal return across all sectors',
              'created_utc': datetime.now(timezone.utc).isoformat(),
              'checks_passed': CHECKS, 'cases': CASES,
              'fundamental_ratios': ratios,
              'scope': 'Conditional same-field modal network, not particle simulations or derived measured forces.',
              'physical_premises': ['coherent equal-weight orientation paths',
                                    'raw unphased spatial Hodge contact',
                                    'selected transmission port before lower-dimensional readout',
                                    'prepared finite sawtooth density source',
                                    'fixed source orientation and common receiver'],
              'limitations': ['Forward gain-weighted receiver sum is not a retarded roundtrip force kernel.',
                              'Seven/eight and eleven/twelve are active static higher-rank controls.',
                              'Source-relative delays can remove neutrino slow-clock suppression.',
                              'Replacing the instantaneous color filter by its delayed transfer generally restores higher-sector volume interactions.',
                              'Complementary modes retain energy and may support other interactions.'],
              'sources': {'experiments/passive-hodge-transfer-checks.py': sha256(Path(__file__).read_bytes()).hexdigest(),
                          'experiments/sawtooth-sector-coupling-checks.py': sha256(args.sawtooth_source.read_bytes()).hexdigest()}}
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'checks_passed': CHECKS, 'cases': len(CASES), 'output': str(args.output)}))


if __name__ == '__main__':
    main()
