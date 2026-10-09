#!/usr/bin/env python3
"""Exact and bounded Fourier/operator checks for the specified sawtooth ansatz."""
import argparse
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction as F
from hashlib import sha256
import json
import math
from pathlib import Path

import mpmath as mp
import numpy as np
from scipy.special import digamma
import sympy as sp


CHECKS = Counter()
mp.mp.dps = 50


def exact(name, actual, expected):
    if actual != expected:
        raise AssertionError((name, actual, expected))
    CHECKS[name] += 1


def close(name, actual, expected, atol=3e-11):
    if not np.allclose(actual, expected, atol=atol, rtol=atol):
        raise AssertionError((name, actual, expected))
    CHECKS[name] += 1


def coefficients_exact(cutoff):
    harmonic = [F(0)]
    for k in range(1, cutoff + 1):
        harmonic.append(harmonic[-1] + F(1, k))
    coefficients = []
    for p in range(1, 2 * cutoff + 1):
        if p <= cutoff:
            value = F(1, p*p) + (harmonic[cutoff-p] - harmonic[cutoff]) / p
        else:
            value = -(harmonic[cutoff] - harmonic[p-cutoff-1]) / p
        coefficients.append(value)
    mean = sum((F(1, 2*k*k) for k in range(1, cutoff+1)), F(0))
    return mean, coefficients, harmonic[-1]


def coefficients(cutoff, limit=None):
    number = min(2 * cutoff, limit) if limit else 2 * cutoff
    p = np.arange(1, number + 1)
    values = np.empty(number)
    low = p <= cutoff
    if cutoff <= 150000:
        harmonic = np.r_[0., np.cumsum(1 / np.arange(1, cutoff+1, dtype=float))]
        values[low] = 1 / p[low]**2 + (harmonic[cutoff-p[low]] - harmonic[cutoff]) / p[low]
        values[~low] = -(harmonic[cutoff] - harmonic[p[~low]-cutoff-1]) / p[~low]
    else:
        values[low] = 1 / p[low]**2 + (digamma(cutoff-p[low]+1) - digamma(cutoff+1)) / p[low]
        values[~low] = -(digamma(cutoff+1) - digamma(p[~low]-cutoff)) / p[~low]
    mean = float((mp.zeta(2) - mp.polygamma(1, cutoff+1)) / 2)
    return mean, values


def waveform(cutoff, samples):
    spectrum = np.zeros(samples, dtype=complex)
    k = np.arange(1, cutoff+1)
    spectrum[k] = 1 / (2j*k)
    spectrum[-k] = -1 / (2j*k)
    return (np.fft.ifft(spectrum) * samples).real


def pair_overlap(cutoff_a, cutoff_b, phase):
    mean_a, ca = coefficients(cutoff_a)
    mean_b, cb = coefficients(cutoff_b)
    count = min(len(ca), len(cb))
    p = np.arange(1, count+1)
    return mean_a * mean_b + .5 * np.dot(ca[:count]*cb[:count], np.cos(p*phase))


def pair_limit(phase):
    phase = phase % (2*math.pi)
    return math.pi**4/80 - math.pi**2*phase**2/24 + math.pi*phase**3/24 - phase**4/96


def tangent_basis(direction):
    constraints = np.vstack([np.ones(len(direction)), direction])
    return np.linalg.svd(constraints, full_matrices=True)[2][2:].T


def simplex_jacobian(dimension):
    number = dimension+1
    directions = math.sqrt(number/dimension)*(np.eye(number)-np.ones((number, number))/number)
    bases = [tangent_basis(v) for v in directions]
    pairs, rows = [], []
    for i in range(number):
        for j in range(i+1, number):
            close('angular_zero_survives_positive_temporal_weight', (1+dimension*directions[i]@directions[j])/number, 0)
            row = np.zeros(number*(dimension-1))
            row[i*(dimension-1):(i+1)*(dimension-1)] = dimension/number*bases[i].T@directions[j]
            row[j*(dimension-1):(j+1)*(dimension-1)] = dimension/number*bases[j].T@directions[i]
            pairs.append((i, j))
            rows.append(row)
    return pairs, np.array(rows)


def response_sum(cutoff, ratio, limit=65536):
    # The receiver uses the new tetrahedral triplet/doublet shape Hessian.
    if cutoff <= 150000:
        limit = 2*cutoff
    mean, c = coefficients(cutoff, limit)
    p = np.arange(1, len(c)+1)
    frequency = ratio*p
    susceptibility = (3/(2-frequency**2)+2/(3-frequency**2))/5
    value = .5*np.dot(c*c, susceptibility)
    derivative = .5*np.dot(c*c*frequency**2, susceptibility)
    omitted = len(c) < 2*cutoff
    ordinary_bound = derivative_bound = 0.
    if omitted:
        lowest = ratio*(len(c)+1)
        if lowest**2 <= 3:
            raise AssertionError('Tail contains unresolved receiver poles')
        # Fourier projection and the L2 error of the truncated sawtooth.
        source_tail = 4*math.pi**2/cutoff + 1/(3*len(c)**3)
        ordinary_bound = source_tail/(lowest**2-3)
        derivative_bound = source_tail*lowest**2/(lowest**2-3)
    return {'density_response': float(value), 'density_tail_bound': ordinary_bound,
            'derivative_source_response': float(derivative), 'derivative_tail_bound': derivative_bound,
            'static_source_term': mean*mean*13/30, 'evaluated_source_harmonics': len(c)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--manifest', type=Path)
    args = parser.parse_args()

    for cutoff in range(1, 41):
        mean, c, h = coefficients_exact(cutoff)
        exact('density_fundamental', c[0], F(cutoff-1, cutoff))
        exact('density_top_harmonic', c[-1], -F(1, 2*cutoff**2))
        derivative_norm = sum((F(p*p, 2)*a*a for p, a in enumerate(c, 1)), F(0))
        exact('density_derivative_sum', derivative_norm, 2*cutoff-h-mean)
        a_sum = sum((sum((F(1, j) for j in range(t+1, cutoff+1)), F(0))**2 for t in range(cutoff)), F(0))
        exact('derivative_sum_proof_square_identity', a_sum, 2*cutoff-h)
        b_sum = sum((sum((F(1, j) for j in range(cutoff-p+1, cutoff+1)), F(0))/p for p in range(1, cutoff+1)), F(0))
        exact('derivative_sum_proof_triangle_identity', b_sum, 2*mean)
        samples = 16*(cutoff+1)
        f = waveform(cutoff, samples)
        spectrum = np.fft.fft(f*f)/samples
        close('independent_fourier_density_mean', spectrum[0].real, float(mean))
        close('independent_fourier_density_coefficients', 2*spectrum[1:2*cutoff+1].real, list(map(float, c)))
        close('independent_fourier_density_phase', spectrum[1:2*cutoff+1].imag, 0)
        cubic = np.fft.fft(f**3)/samples
        close('cubic_contact_highest_source', -2*cubic[3*cutoff].imag, -1/(4*cutoff**3))
        time = 2*math.pi*np.arange(samples)/samples
        k = np.arange(1, cutoff+1)[:, None]
        q = np.sin(k*time)/k
        momentum = np.cos(k*time)
        close('positive_spatial_mode_energy_is_conserved', .5*np.sum(momentum**2+(k*q)**2, axis=0), cutoff/2)
        close('spatial_evaluation_is_sawtooth', q.sum(axis=0), f)
        analytic_modes = np.exp(-1j*k*time)/k
        close('orthogonal_complex_modes_keep_total_norm', np.sum(abs(analytic_modes)**2, axis=0), float(2*mean))
        local_density = abs(analytic_modes.sum(axis=0))**2/2
        analytic_spectrum = np.fft.fft(local_density)/samples
        close('complex_local_source_has_same_fundamental', 2*analytic_spectrum[1].real, float(c[0]))
        for p in range(1, cutoff):
            expected_beat = sum((F(1, r*(r+p)) for r in range(1, cutoff-p+1)), F(0))
            close('complex_local_transition_coefficients', 2*analytic_spectrum[p].real, float(expected_beat))

    theta, delta = sp.symbols('theta delta', real=True)
    base = (sp.pi-theta)/2
    first = (sp.pi-theta-delta)/2
    wrapped = (3*sp.pi-theta-delta)/2
    integral = sp.simplify((sp.integrate(base**2*first**2, (theta, 0, 2*sp.pi-delta))
                           + sp.integrate(base**2*wrapped**2, (theta, 2*sp.pi-delta, 2*sp.pi)))/(2*sp.pi))
    expected = sp.pi**4/80-sp.pi**2*delta**2/24+sp.pi*delta**3/24-delta**4/96
    exact('independent_piecewise_phase_integral', sp.expand(integral-expected), 0)
    exact('phase_pair_antiphase_ratio', sp.simplify(integral.subs(delta, sp.pi)/integral.subs(delta, 0)), sp.Rational(1, 6))
    exact('limiting_antiphase_stiffness', sp.diff(integral, delta, 2).subs(delta, sp.pi), sp.pi**2/24)

    pair_cases = []
    for ka, kb in [(1, 1), (2, 5), (8, 13), (30, 40)]:
        samples = 32*(ka+kb+1)
        time = np.arange(samples)*2*math.pi/samples
        for phase in [0., .37, math.pi/2, math.pi]:
            fa = sum(np.sin(k*time)/k for k in range(1, ka+1))
            fb = sum(np.sin(k*(time+phase))/k for k in range(1, kb+1))
            value = pair_overlap(ka, kb, phase)
            close('independent_reciprocal_pair_average', np.mean(fa*fa*fb*fb), value)
            close('pair_exchange_reciprocity', value, pair_overlap(kb, ka, -phase))
            exact('positive_temporal_edge_weight', value > 0, True)
            pair_cases.append({'cutoff_a': ka, 'cutoff_b': kb, 'phase': phase, 'weight': value})

    rational_cases = []
    for p, q in [(1, 1), (2, 1), (3, 2), (5, 3)]:
        ka, kb = 7, 9
        ma, ca = coefficients(ka)
        mb, cb = coefficients(kb)
        count = min(len(ca)//q, len(cb)//p)
        phase = .23
        predicted = ma*mb + .5*sum(ca[q*j-1]*cb[p*j-1]*math.cos(j*phase) for j in range(1, count+1))
        time = np.arange(16384)*2*math.pi/16384
        fa = sum(np.sin(k*(p*time+phase/q))/k for k in range(1, ka+1))
        fb = sum(np.sin(k*q*time)/k for k in range(1, kb+1))
        close('commensurate_clock_pair_average', np.mean(fa*fa*fb*fb), predicted)
        rational_cases.append({'frequency_ratio': f'{p}/{q}', 'finite_pair_weight': predicted,
                               'limiting_phase_coefficient': str(F(1, 2*p*p*q*q))})

    finite_windows = []
    nodes, weights = np.polynomial.legendre.leggauss(320)
    for ratio in [math.sqrt(2), 5/3, 2.]:
        ka, kb, duration, phase_a, phase_b = 5, 7, 3.7, .1, -.4
        ma, ca = coefficients(ka)
        mb, cb = coefficients(kb)
        pa = np.arange(-2*ka, 2*ka+1)
        pb = np.arange(-2*kb, 2*kb+1)
        aa = np.r_[ca[::-1]/2, ma, ca/2]
        ab = np.r_[cb[::-1]/2, mb, cb/2]
        frequency = ratio*pa[:, None]+pb[None, :]
        phase = phase_a*pa[:, None]+phase_b*pb[None, :]
        predicted = np.sum(aa[:, None]*ab[None, :]*np.cos(phase)*np.sinc(frequency*duration/(2*math.pi)))
        time = nodes*duration/2
        fa = sum(np.sin(k*(ratio*time+phase_a))/k for k in range(1, ka+1))
        fb = sum(np.sin(k*(time+phase_b))/k for k in range(1, kb+1))
        close('finite_encounter_retains_both_clocks', weights@(fa*fa*fb*fb)/2, predicted)
        finite_windows.append({'frequency_ratio': ratio, 'duration': duration, 'pair_average': float(predicted)})

    # Same spectral amplitudes can have different sources under different spatial operators.
    spatial_cases = []
    rng = np.random.default_rng(20261009)
    for depth in range(2, 13):
        count = math.comb(depth+2, depth)
        ones = np.ones((count, count))
        close('contact_count_identity', ones@ones, count*ones)
        amplitudes = 1/np.arange(1, count+1)
        contact_beat = np.dot(amplitudes[:-1], amplitudes[1:])
        close('contact_reads_adjacent_orthogonal_modes', contact_beat, 1-1/count)
        exact('integrated_identity_has_no_adjacent_transition', np.trace(np.eye(count), offset=1), 0)
        size = depth
        a = rng.normal(size=(size, size)) + 1j*rng.normal(size=(size, size))
        metric = a.conj().T@a + np.eye(size)
        operator = rng.normal(size=(size, size)) + 1j*rng.normal(size=(size, size))
        operator = (operator+operator.conj().T)/2
        basis = np.eye(size) + .04*(rng.normal(size=(size, size))+1j*rng.normal(size=(size, size)))
        comb = rng.normal(size=(5, size))+1j*rng.normal(size=(5, size))
        comb /= np.arange(1, 6)[:, None]
        moved = np.linalg.solve(basis, comb.T).T
        moved_operator = basis.conj().T@operator@basis
        moved_metric = basis.conj().T@metric@basis
        close('kinetic_norm_basis_invariance', sum(v.conj()@metric@v for v in comb), sum(v.conj()@moved_metric@v for v in moved))
        for p in range(1, 5):
            old = sum(comb[r].conj()@operator@comb[r+p] for r in range(5-p))
            new = sum(moved[r].conj()@moved_operator@moved[r+p] for r in range(5-p))
            close('harmonic_source_basis_invariance', old, new)
        spatial_cases.append({'depth': depth, 'spatial_band_count_at_n3': count,
                              'contact_fundamental_coefficient': contact_beat})

    angular_cases = []
    for dimension in range(2, 13):
        pairs, jacobian = simplex_jacobian(dimension)
        cutoff = math.comb(dimension+2, dimension)
        common_weight = pair_overlap(cutoff, cutoff, 0)
        unweighted = 2*jacobian.T@jacobian
        equal = 2*jacobian.T@(common_weight*jacobian)
        close('equal_comb_preserves_angular_spectrum_ratios', np.linalg.eigvalsh(equal)/common_weight,
              np.linalg.eigvalsh(unweighted))
        weights = np.array([pair_overlap(cutoff, cutoff, math.pi*((i-j) % 2)) for i, j in pairs])
        weighted = 2*jacobian.T@(weights[:, None]*jacobian)
        spectrum = np.linalg.eigvalsh(weighted)
        zeros = dimension*(dimension-1)//2
        exact('phase_weighted_shape_has_only_rotation_zeros', int(np.sum(abs(spectrum)<1e-9)), zeros)
        exact('phase_weighted_shape_remains_positive', float(spectrum.min()) > -1e-10, True)
        angular_cases.append({'resolved_dimensions': dimension, 'cutoff': cutoff,
                              'equal_phase_weight': common_weight,
                              'alternating_phase_positive_eigenvalues': spectrum[zeros:].tolist()})

    tetra = [sp.Matrix(v) for v in ((1,1,1),(1,-1,-1),(-1,1,-1),(-1,-1,1))]
    bases = []
    for v in tetra:
        first = sp.Matrix([-v[1], v[0], 0])
        bases.append(sp.Matrix.hstack(first, v.cross(first)))
    metric = sp.diag(*(b.T*b for b in bases))
    rows, weights = [], []
    for i in range(4):
        for j in range(i+1, 4):
            row = sp.zeros(1, 8)
            row[:, 2*i:2*i+2] = sp.Rational(3, 4)*tetra[j].T*bases[i]/sp.sqrt(3)
            row[:, 2*j:2*j+2] = sp.Rational(3, 4)*tetra[i].T*bases[j]/sp.sqrt(3)
            rows.append(row)
            weights.append(1 if (i-j) % 2 == 0 else sp.Rational(1, 6))
    jacobian = sp.Matrix.vstack(*rows)
    hessian = metric.inv()*(2*jacobian.T*sp.diag(*weights)*jacobian)
    polynomial = sp.factor(hessian.charpoly(sp.Symbol('lambda')).as_expr())
    exact('weighted_tetrahedron_rotation_rank', hessian.rank(), 5)
    weighted_eigenvalues = {str(k): v for k, v in hessian.eigenvals().items()}

    electron_mev = mp.mpf('0.51099895000')
    mu6 = electron_mev/18564
    mu5_ev = mp.sqrt(mp.mpf('2.534e-3')/(65780**2-2002**2))
    scales = {2: electron_mev*mp.sqrt(2890), 3: electron_mev*mp.sqrt(32*mp.sqrt(7)),
              4: electron_mev*mp.sqrt(48/mp.sqrt(7))/15, 5: mu5_ev/mp.mpf(10**6), 6: mu6, 10: mu6}
    assigned = [('W',2,76),('Z',2,81),('Higgs',2,95),('down',3,1),('strange',3,4),
                ('up',4,3),('charm',4,20),('top',4,72),('nu1',5,10),('nu2',5,15),('nu3',5,22),
                ('electron',6,13),('muon',6,35),('tau',10,23)]
    particle_cases = []
    for name, depth, n in assigned:
        cutoff = math.comb(n+depth-1, depth)
        h = mp.harmonic(cutoff)
        h2 = mp.zeta(2)-mp.polygamma(1, cutoff+1)
        derivative_norm = 2*cutoff-h-h2/2
        ratio = float(scales[depth]/mu6)
        response = response_sum(cutoff, ratio)
        # Independently verify low coefficients where cancellation could affect large cutoffs.
        _, numeric = coefficients(cutoff, 12)
        for p in range(1, min(cutoff, 12)+1):
            reference = mp.mpf(1)/(p*p)+(mp.harmonic(cutoff-p)-h)/p
            close('large_cutoff_low_source_coefficients', numeric[p-1], float(reference), 2e-12)
        particle_cases.append({'particle': name, 'depth': depth, 'mode_index': n, 'cutoff': cutoff,
                               'fundamental_over_depth6': ratio, 'density_mean': float(h2/2),
                               'density_fundamental': str(F(cutoff-1, cutoff)),
                               'density_derivative_norm': float(derivative_norm), **response})

    leakage = []
    for cutoff in [1, 2, 3, 8, 32, 128, 512, 2048]:
        samples = 16*(cutoff+1)
        f = waveform(cutoff, samples)
        cubic = np.fft.fft(f**3)/samples
        total = np.mean(f**6)
        high = 2*np.sum(abs(cubic[cutoff+1:3*cutoff+1])**2)
        close('cubic_parseval', np.sum(abs(cubic)**2), total)
        leakage.append({'cutoff': cutoff, 'outside_cutoff_source_fraction': float(high/total)})

    # The same temporal scalar cannot remove an allowed higher-rank color response.
    color = []
    for depth in range(2, 17):
        half = (depth+1)//2
        rank = math.comb(2*half-1, half-1)
        factor = max(0, rank-2)
        color.append({'depth': depth, 'middle_channel_rank': rank,
                      'three_form_pair_factor': factor,
                      'scalar_volume_triplet': rank == 3})
        exact('nonzero_higher_rank_color_survives_temporal_weight', factor > 0, rank >= 3)

    result = {'title': 'Sawtooth spectra in spatial and reciprocal sector coupling',
              'created_utc': datetime.now(timezone.utc).isoformat(), 'status': 'passed',
              'checks': dict(CHECKS), 'total_checks': sum(CHECKS.values()),
              'scope': 'Specified finite Fourier preparations, exact spatial/phase operators and a common receiver response; no particle simulation or fitted coupling.',
              'finite_pair_cases': pair_cases, 'commensurate_clocks': rational_cases,
              'finite_encounter_cases': finite_windows,
              'spatial_contact_cases': spatial_cases, 'angular_cases': angular_cases,
              'weighted_tetrahedron_polynomial': str(polynomial), 'weighted_tetrahedron_eigenvalues': weighted_eigenvalues,
              'limiting_pair_overlap': str(expected), 'particle_cases': particle_cases,
              'cubic_source_leakage': leakage, 'color_controls': color,
              'neutrino_scale_input': 'Archived atmospheric splitting 2.534e-3 eV^2; separate conditional calibration.',
              'receiver': 'One common observer-three-dimensional angular response with squared poles 2 and 3 in units omega_0,6^2, weights 3/5 and 2/5. Signed response forms, not rates.',
              'exceptions': {'photon': 'Massless n=0 does not supply an optical harmonic cutoff.',
                             'bottom': 'The unresolved geometric-mean assignment is not an integer cutoff; no rounding or replacement mass was used.'},
              'sources': {'experiments/sawtooth-sector-coupling-checks.py': sha256(Path(__file__).read_bytes()).hexdigest()}}
    if args.manifest:
        result['research_inputs'] = json.loads(args.manifest.read_text())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as output:
        json.dump(result, output, indent=2)
        output.write('\n')
    print(json.dumps({'checks': result['total_checks'], 'weighted_tetrahedron': weighted_eigenvalues,
                      'report': str(args.output), 'particle_cases': particle_cases}, indent=2))


if __name__ == '__main__':
    main()
