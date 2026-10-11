#!/usr/bin/env python3
"""Exact source-support rules and bounded square-comb sector comparisons."""
import argparse
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction as F
from functools import reduce
from hashlib import sha256
from itertools import product
import json
import math
from pathlib import Path

import mpmath as mp
import numpy as np
from scipy.special import digamma

CHECKS = Counter()
mp.mp.dps = 60


def require(label, condition):
    if not condition:
        raise AssertionError(label)
    CHECKS[label] += 1


def close(label, actual, expected, tolerance=2e-11):
    require(label, np.allclose(actual,expected,atol=tolerance,rtol=tolerance))


def odd_sum(count, degree=1):
    return sum((F(1,(2*j-1)**degree) for j in range(1,count+1)),F(0))


def direct_coefficients(support):
    largest = max(support)
    return [sum((F(1,k*(k+p)) for k in support if k+p in support),F(0))
            for p in range(1,largest)]


def coefficient_formula(cutoff, repaired=False):
    count = (cutoff+1)//2
    values = []
    for p in range(1,cutoff):
        if p%2 == 0:
            q = p//2
            value = (odd_sum(q)+odd_sum(count-q)-odd_sum(count))/p if q<count else F(0)
        else:
            value = F(1,cutoff*(cutoff-p)) if repaired and cutoff%2 == 0 else F(0)
        values.append(value)
    return values


def numeric_coefficients(cutoff, number, repaired):
    count = (cutoff+1)//2
    p = np.arange(1,number+1)
    values = np.zeros(number)
    even = p%2 == 0
    q = p[even]/2
    # O_n = (psi(n+1/2)-psi(1/2))/2; keep the large differences adjacent.
    values[even] = ((digamma(q+.5)-digamma(.5)+
                    digamma(count-q+.5)-digamma(count+.5))/2)/p[even]
    if repaired and cutoff%2 == 0:
        values[~even] = 1/(float(cutoff)*(cutoff-p[~even]))
    return p,values


def susceptibility(x):
    return (3/(2-x*x)+2/(3-x*x))/5


def envelope(last):
    value = math.log(last)
    return (value*value+4*value+5)/last


def response(cutoff, ratio, repaired):
    number = cutoff-1 if cutoff <= 150000 else min(cutoff-1,65536)
    p,values = numeric_coefficients(cutoff,number,repaired)
    x = ratio*p
    chi = susceptibility(x)
    q0 = float(np.dot(values*values,chi)/2)
    q1 = float(np.dot(values*values*x*x,chi)/2)
    tail0 = tail1 = 0.
    if number < cutoff-1:
        lowest = ratio*(number+1)
        require('uncomputed_tail_above_receiver_poles',lowest*lowest>3)
        tail = envelope(number)
        tail0 = tail/(2*(lowest*lowest-3))
        tail1 = tail/(2*(1-3/(lowest*lowest)))
    fundamental = 1/(float(cutoff)*(cutoff-1)) if repaired and cutoff%2 == 0 else 0.
    return {'density_response':q0,'density_interval':[q0-tail0,q0],
            'derivative_response':q1,'derivative_interval':[q1-tail1,q1],
            'fundamental_density_coefficient':fundamental,
            'mass_endpoint_present':cutoff%2 == 1 or repaired,
            'evaluated_density_harmonics':number}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    finite = []
    for cutoff in range(1,51):
        support = set(range(1,cutoff+1,2))
        for repaired in (False,True):
            actual_support = support|({cutoff} if repaired else set())
            direct = direct_coefficients(actual_support)
            formula = coefficient_formula(cutoff,repaired)
            if len(direct)<len(formula):
                direct += [F(0)]*(len(formula)-len(direct))
            require('odd_comb_exact_telescoping',formula == direct)
            p,values = numeric_coefficients(cutoff,cutoff-1,repaired)
            close('odd_comb_high_precision_formula',values,[float(v) for v in formula])
            expected_fundamental = F(1,cutoff*(cutoff-1)) if repaired and cutoff%2 == 0 else F(0)
            require('fundamental_parity_and_endpoint_rule',not formula or formula[0] == expected_fundamental)
            if len(actual_support)>1:
                gaps = [b-a for a,b in zip(sorted(actual_support),sorted(actual_support)[1:])]
                spacing = reduce(math.gcd,gaps)
                require('support_difference_frequency_lattice',spacing == (1 if repaired and cutoff%2 == 0 else 2))
            samples = 1<<(4*cutoff).bit_length()
            spectrum = np.zeros(samples,dtype=complex)
            for k in actual_support:
                spectrum[k] = 1/k
            z = np.fft.ifft(spectrum)*samples
            fft = np.fft.fft(np.abs(z)**2/2)/samples
            close('independent_fft_density_support',2*fft[1:cutoff],values)
            norm = sum((F(p*p,2)*v*v for p,v in enumerate(formula,1)),F(0))
            close('density_derivative_parseval',norm,float(np.dot((p*values)**2,np.ones(len(values)))/2))
        if cutoff%2 == 0:
            count = cutoff//2
            pure = coefficient_formula(cutoff,False)
            repair = coefficient_formula(cutoff,True)
            increment = sum((F(p*p,2)*(b*b-a*a) for p,(a,b) in enumerate(zip(pure,repair),1)),F(0))
            expected = (odd_sum(count,2)-2*odd_sum(count)/cutoff+F(count,cutoff*cutoff))/2
            require('endpoint_derivative_increment_exact',increment == expected)
        finite.append({'cutoff':cutoff,'square_mass_endpoint_present':bool(cutoff%2),
                       'odd_only_fundamental_density_exact':'0'})
    phase_cases = []
    for cutoff in range(2,7):
        roots = np.asarray([1,1j,-1,-1j])
        phases = roots[np.asarray(list(product(range(4),repeat=cutoff)))]
        amplitude = phases/np.arange(1,cutoff+1)
        fundamental = np.sum(np.conj(amplitude[:,:-1])*amplitude[:,1:],axis=1)
        expected_variance = 2*sum((F(1,k*k) for k in range(1,cutoff+1)),F(0))-3+F(2,cutoff)-F(1,cutoff*cutoff)
        close('independent_phase_mean_fundamental',np.mean(fundamental),0.)
        close('independent_phase_fundamental_variance',np.mean(np.abs(fundamental)**2),float(expected_variance))
        coherent = sum((F(1,k*(k+1)) for k in range(1,cutoff)),F(0))
        require('coherent_sawtooth_fundamental_exact',coherent == 1-F(1,cutoff))
        # A local rank-one evaluation retains interference; the integrated identity does not.
        theta = np.arange(64)*2*np.pi/64
        mode = np.exp(-1j*theta[:,None]*np.arange(1,cutoff+1))/np.arange(1,cutoff+1)
        integrated = np.sum(np.abs(mode)**2,axis=1)/2
        close('orthogonal_integrated_norm_has_no_beats',integrated,integrated[0])
        phase_cases.append({'cutoff':cutoff,'independent_phase_samples':len(phases),
                            'coherent_coefficient_exact':str(coherent),
                            'random_phase_mean_exact':'0','random_phase_variance_exact':str(expected_variance)})
    baseline = json.loads(args.baseline.read_text())
    particles = []
    for item in baseline['particle_cases']:
        cutoff,ratio = item['cutoff'],item['fundamental_over_depth6']
        pure = response(cutoff,ratio,False)
        repaired = response(cutoff,ratio,True)
        if cutoff>1:
            saw_fundamental = 1-1/cutoff
            saw_contribution = .5*saw_fundamental*saw_fundamental*float(susceptibility(ratio))
        else:
            saw_fundamental = saw_contribution = 0.
        if item['real_depth'] in (6,10):
            require('square_lepton_dynamic_density_above_poles',pure['density_response']<0)
            require('square_lepton_dynamic_derivative_above_poles',pure['derivative_response']<0)
        if cutoff == 1:
            require('single_harmonic_has_constant_complex_density',pure['density_response'] == pure['derivative_response'] == 0)
        particles.append({'particle':item['particle'],'real_depth':item['real_depth'],
                          'mode_index':item['mode_index'],'cutoff':cutoff,
                          'fundamental_over_depth6':ratio,'square':pure,'square_with_mass_endpoint':repaired,
                          'sawtooth_fundamental_density_coefficient':saw_fundamental,
                          'sawtooth_density_fundamental_response':saw_contribution,
                          'sawtooth_complex_density_response':item['complex']['density_response'],
                          'sawtooth_complex_derivative_interval':[item['complex']['derivative_interval_lower'],
                                                                 item['complex']['derivative_interval_upper']]})
    report = {'experiment':'harmonic-support-sector-response','created_utc':datetime.now(timezone.utc).isoformat(),
              'status':'passed','checks':sum(CHECKS.values()),'check_counts':dict(CHECKS),
              'finite_cases':finite,'phase_cases':phase_cases,'particle_cases':particles,
              'scope':'Frequency support and local quadratic readout are conditional physical preparations. The fundamental is not a complete coupling law.',
              'sources':{'experiments/harmonic-support-checks.py':sha256(Path(__file__).read_bytes()).hexdigest(),
                         'results/complex-comb-response-results.json':sha256(args.baseline.read_bytes()).hexdigest()}}
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'checks':report['checks'],'particle_cases':len(particles),
                      'phase_cases':len(phase_cases),'report':str(args.output)}))


if __name__ == '__main__':
    main()
