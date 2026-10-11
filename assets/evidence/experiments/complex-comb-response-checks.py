#!/usr/bin/env python3
"""Compare real and positive-frequency density sources at occupied V2 depths."""
import argparse
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction as F
from hashlib import sha256
import importlib.util
import json
import math
from pathlib import Path

import mpmath as mp
import numpy as np
from scipy.special import digamma

mp.mp.dps = 60
CHECKS = Counter()
VARIANCE_LIMIT = float(17 * mp.zeta(4) / 4)


def exact(label, actual, expected):
    if actual != expected:
        raise AssertionError((label, actual, expected))
    CHECKS[label] += 1


def close(label, actual, expected, tolerance=3e-11):
    if not np.allclose(actual, expected, atol=tolerance, rtol=tolerance):
        raise AssertionError((label, actual, expected))
    CHECKS[label] += 1


def require(label, condition):
    if not condition:
        raise AssertionError(label)
    CHECKS[label] += 1


def harmonics(cutoff):
    h = [F(0)]
    h2 = [F(0)]
    for k in range(1, cutoff + 1):
        h.append(h[-1] + F(1, k))
        h2.append(h2[-1] + F(1, k*k))
    return h, h2


def derivative_identity(cutoff, h, h2):
    return ((cutoff+1)*(h*h-2*h2)-(4*cutoff+2)*h+8*cutoff)/2


def tail_envelope(last):
    logarithm = math.log(last)
    return (logarithm**2+4*logarithm+5)/last


def variance_deficit_bound(cutoff):
    if cutoff == 1:
        return VARIANCE_LIMIT
    h = mp.harmonic(cutoff-1)
    h2 = mp.zeta(2)-mp.polygamma(1, cutoff)
    hk = h+mp.mpf(1)/cutoff
    return float((h*h+h2+2*h*(hk-1))/(cutoff+1))+tail_envelope(cutoff-1)


def coefficients(cutoff, limit=None):
    number = cutoff-1 if limit is None else min(cutoff-1, limit)
    p = np.arange(1, number+1, dtype=float)
    if cutoff <= 150000:
        h = np.r_[0., np.cumsum(1/np.arange(1, cutoff+1, dtype=float))]
        values = (h[np.arange(1, number+1)]+h[cutoff-np.arange(1, number+1)]-h[-1])/p
    else:
        values = (digamma(p+1)+np.euler_gamma+digamma(cutoff-p+1)-digamma(cutoff+1))/p
    mean = float((mp.zeta(2)-mp.polygamma(1, cutoff+1))/2)
    return mean, values


def susceptibility(x):
    return (3/(2-x*x)+2/(3-x*x))/5


def response(cutoff, ratio, limit=65536):
    count = cutoff-1 if cutoff <= 150000 else min(cutoff-1, limit)
    mean, values = coefficients(cutoff, count)
    p = np.arange(1, count+1, dtype=float)
    x = ratio*p
    q0 = float(np.dot(values*values, susceptibility(x))/2)
    q1 = float(np.dot(values*values*x*x, susceptibility(x))/2)
    b0 = b1 = 0.
    if count < cutoff-1:
        lowest = ratio*(count+1)
        require('omitted_frequencies_above_poles', lowest*lowest > 3)
        tail = tail_envelope(count)
        b0 = tail/(2*(lowest*lowest-3))
        b1 = tail/(2*(1-3/(lowest*lowest)))
    result = {'density_response': q0, 'density_tail_bound': b0,
              'derivative_response': q1, 'derivative_tail_bound': b1,
              'static_source_term': mean*mean*13/30,
              'evaluated_harmonics': count}
    if count < cutoff-1:
        # Separate the exactly known limiting variance from a fast-decaying correction.
        correction = float(.6*np.dot(values*values, 1/(2-x*x)+1/(3-x*x)))
        correction_bound = 1.2*tail_envelope(count)/(lowest*lowest-3)
        variance_bound = variance_deficit_bound(cutoff)
        center = -VARIANCE_LIMIT/2+correction
        result.update(derivative_interval_lower=center-correction_bound,
                      derivative_interval_upper=center+variance_bound/2,
                      variance_deficit_bound=variance_bound,
                      derivative_interval_method='limiting variance plus exact finite coefficients')
        require('response_tail_and_variance_intervals_overlap',
                max(q1-b1, result['derivative_interval_lower']) <=
                min(q1, result['derivative_interval_upper'])+1e-12)
    else:
        result.update(derivative_interval_lower=q1, derivative_interval_upper=q1,
                      variance_deficit_bound=variance_deficit_bound(cutoff))
    h = mp.harmonic(cutoff)
    h2 = mp.zeta(2)-mp.polygamma(1, cutoff+1)
    derivative_norm = float(derivative_identity(cutoff, h, h2))
    result['density_derivative_norm'] = derivative_norm
    if cutoff > 1 and ratio*(cutoff-1) < math.sqrt(2):
        leading = (13/30)*ratio*ratio*derivative_norm
        maximum = float(susceptibility(ratio*(cutoff-1)))/float(susceptibility(0))
        require('slow_comb_derivative_sandwich', leading*(1-2e-11) <= q1 <= leading*maximum*(1+2e-11))
        result.update(slow_derivative_leading=leading,
                      slow_derivative_relative_upper=maximum-1)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location('real_comb_baseline', args.baseline)
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)
    finite_cases = []
    for cutoff in range(1, 41):
        h, h2 = harmonics(cutoff)
        values = [(h[p]+h[cutoff-p]-h[cutoff])/p for p in range(1, cutoff)]
        for p, value in enumerate(values, 1):
            direct = sum((F(1, k*(k+p)) for k in range(1, cutoff-p+1)), F(0))
            exact('telescoping_density_coefficient', value, direct)
            exact('limiting_coefficient_deficit', h[p]/p-value, (h[cutoff]-h[cutoff-p])/p)
            require('coefficient_deficit_bound', h[p]/p-value <= F(1, cutoff-p+1))
        norm = sum(((p+1)**2*v*v/2 for p, v in enumerate(values)), F(0))
        exact('finite_derivative_identity', norm, derivative_identity(cutoff, h[-1], h2[-1]))
        exact('partial_harmonic_square_identity', sum((v*v for v in h[1:-1]), F(0)),
              cutoff*h[-1]**2-(2*cutoff+1)*h[-1]+2*cutoff)
        exact('harmonic_convolution_identity', sum((h[p]*h[cutoff-p] for p in range(1, cutoff)), F(0)),
              (cutoff+1)*(h[-1]**2-h2[-1])-2*cutoff*h[-1]+2*cutoff)
        samples = 1 << (4*cutoff).bit_length()
        spectrum = np.zeros(samples, dtype=complex)
        spectrum[np.arange(1, cutoff+1)] = 1/np.arange(1, cutoff+1)
        z = np.fft.ifft(spectrum)*samples
        density = np.abs(z)**2/2
        density_spectrum = np.fft.fft(density)/samples
        close('density_mean_independent_fft', density_spectrum[0], float(h2[-1]/2))
        close('density_coefficients_independent_fft', 2*density_spectrum[1:cutoff], [float(v) for v in values])
        derivative_spectrum = np.zeros(samples, dtype=complex)
        for p, value in enumerate(values, 1):
            derivative_spectrum[p] = 1j*p*float(value)/2
            derivative_spectrum[-p] = -1j*p*float(value)/2
        derivative = np.fft.ifft(derivative_spectrum)*samples
        close('derivative_norm_independent_fft', np.mean(np.abs(derivative)**2), float(norm))
        variance = sum((v*v for v in values), F(0))
        deficit = VARIANCE_LIMIT-float(variance)
        require('finite_variance_deficit_bound', 0 <= deficit <= variance_deficit_bound(cutoff))
        finite_cases.append({'cutoff': cutoff, 'density_derivative_norm_exact': str(norm),
                             'variance_exact': str(variance)})
    for cutoff in (64, 1024, 18564, 100000):
        _, values = coefficients(cutoff)
        close('numeric_derivative_identity', np.dot(values*values, np.arange(1, cutoff, dtype=float)**2)/2,
              float(derivative_identity(cutoff, mp.harmonic(cutoff), mp.zeta(2)-mp.polygamma(1, cutoff+1))), 1e-9)
        require('large_finite_variance_bound',
                0 <= VARIANCE_LIMIT-float(np.dot(values, values)) <= variance_deficit_bound(cutoff))
        full = response(cutoff, 1.)
        for limit in (16, 64, 256):
            # Exercise the same analytic tail bounds against an independently full sum.
            p = np.arange(1, limit+1, dtype=float)
            head = values[:limit]
            if len(head) < limit:
                continue
            center = -VARIANCE_LIMIT/2+float(.6*np.dot(head*head, 1/(2-p*p)+1/(3-p*p)))
            bound = 1.2*tail_envelope(limit)/((limit+1)**2-3)
            require('variance_response_bound_vs_full_sum',
                    center-bound <= full['derivative_response'] <= center+variance_deficit_bound(cutoff)/2)
    last = 65536
    p = np.arange(1, last+1, dtype=float)
    limiting = (digamma(p+1)+np.euler_gamma)/p
    variance_head = float(np.dot(limiting, limiting))
    require('external_euler_sum_independent_partial_bound',
            variance_head < VARIANCE_LIMIT < variance_head+tail_envelope(last))
    limit_q0 = float(np.dot(limiting*limiting, susceptibility(p))/2)
    limit_q1 = -VARIANCE_LIMIT/2+float(.6*np.dot(limiting*limiting, 1/(2-p*p)+1/(3-p*p)))
    limit_bound = 1.2*tail_envelope(last)/((last+1)**2-3)
    electron = mp.mpf('0.51099895000')
    mu6 = electron/18564
    mu5 = mp.sqrt(mp.mpf('2.534e-3')/(65780**2-2002**2))/10**6
    scales = {2: electron*mp.sqrt(2890), 3: electron*mp.sqrt(32*mp.sqrt(7)),
              4: electron*mp.sqrt(48/mp.sqrt(7))/15, 5: mu5, 6: mu6, 10: mu6}
    assigned = [('W',2,76),('Z',2,81),('Higgs',2,95),('down',3,1),('strange',3,4),
                ('up',4,3),('charm',4,20),('top',4,72),('nu1',5,10),('nu2',5,15),('nu3',5,22),
                ('electron',6,13),('muon',6,35),('tau',10,23)]
    particles = []
    for name, depth, index in assigned:
        cutoff = math.comb(index+depth-1, depth)
        ratio = float(scales[depth]/mu6)
        actual = response(cutoff, ratio)
        real = baseline.response_sum(cutoff, ratio)
        mean, low = coefficients(cutoff, 12)
        for p, value in enumerate(low, 1):
            reference = (mp.harmonic(p)+mp.harmonic(cutoff-p)-mp.harmonic(cutoff))/p
            close('large_cutoff_coefficients_high_precision', value, float(reference), 3e-13)
        if cutoff > 1:
            close('fundamental_matches_real_preparation', low[0], 1-1/cutoff)
        close('equal_mean_source', mean*mean*13/30, real['static_source_term'])
        particles.append({'particle': name, 'real_depth': depth, 'mode_index': index,
                          'cutoff': cutoff, 'fundamental_over_depth6': ratio,
                          'complex': actual, 'real': real})
    report = {'experiment': 'complex-comb-sector-response',
              'created_utc': datetime.now(timezone.utc).isoformat(),
              'status': 'passed', 'checks': sum(CHECKS.values()), 'check_counts': dict(CHECKS),
              'scope': 'Conditional real versus positive-frequency combs at occupied sectors; signed conservative receiver diagnostics, not charges or rates.',
              'finite_cases': finite_cases, 'particle_cases': particles,
              'limiting_variance': {'exact': '17*zeta(4)/4', 'value': VARIANCE_LIMIT,
                                    'partial_sum': variance_head, 'tail_bound': tail_envelope(last)},
              'limiting_response_r1': {'density_response': limit_q0,
                                       'derivative_response': limit_q1,
                                       'derivative_tail_bound': limit_bound},
              'sources': {'experiments/complex-comb-response-checks.py': sha256(Path(__file__).read_bytes()).hexdigest(),
                          'experiments/sawtooth-sector-coupling-checks.py': sha256(args.baseline.read_bytes()).hexdigest()}}
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'checks': report['checks'], 'particle_cases': particles,
                      'limiting_response': report['limiting_response_r1']}, indent=2))


if __name__ == '__main__':
    main()
