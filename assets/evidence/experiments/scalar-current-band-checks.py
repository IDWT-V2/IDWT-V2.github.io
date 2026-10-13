"""Actual signed scalar-current source on complete round-sphere spinor bands."""
from datetime import datetime, timezone
from hashlib import sha256
from importlib.util import module_from_spec, spec_from_file_location
from math import comb, sqrt
from pathlib import Path
import json

import numpy as np
from scipy.linalg import eigh
import sympy as sp

ROOT = Path(__file__).resolve().parent
HELPER = ROOT/'spatial-source-saturation-checks.py'
spec = spec_from_file_location('sphere_sources', HELPER)
sphere = module_from_spec(spec)
spec.loader.exec_module(sphere)
CONDITIONS, ERRORS = {}, {}


def check(value, label):
    if not bool(value):
        raise AssertionError(label)
    CONDITIONS[label] = CONDITIONS.get(label, 0)+1


def close(actual, expected, label, tolerance=4e-8):
    a, e = np.asarray(actual), np.asarray(expected)
    error = float(np.max(abs(a-e))/max(1., float(np.max(abs(e)))))
    ERRORS[label] = max(ERRORS.get(label, 0.), error)
    check(error < tolerance, label)


def clifford(depth):
    matrices = sphere.spatial.gammas(depth+1)
    gamma, beta = matrices[:depth], matrices[depth]
    scalar_vector = [beta]+[1j*g@beta for g in gamma]
    identity = np.eye(len(beta))
    for i, g in enumerate(gamma):
        close(g@beta+beta@g, 0, 'physical scalar grading anticommutes with every spatial Clifford direction')
        close(g@g, identity, 'physical spatial Clifford direction squares to identity')
    close(beta@beta, identity, 'physical scalar grading is a Hermitian involution')
    close(beta, beta.conj().T, 'physical scalar grading is Hermitian')
    return gamma, beta, scalar_vector


def algebra_checks():
    rng = np.random.default_rng(1123)
    cases = []
    for depth in sphere.DEPTHS:
        gamma, beta, operators = clifford(depth)
        rank = len(beta)
        u = rng.normal(size=rank)+1j*rng.normal(size=rank)
        u /= np.linalg.norm(u)
        polar = np.array([np.vdot(u, g@u).real for g in operators])
        for theta in (.17, .73, 1.91, 2.73):
            direction = rng.normal(size=depth)
            direction /= np.linalg.norm(direction)
            g = sum(x*matrix for x, matrix in zip(direction, gamma))
            t, s = np.cos(theta), np.sin(theta)
            frame = np.cos(theta/2)*np.eye(rank)-1j*np.sin(theta/2)*g
            at_point = (t*np.eye(rank)+1j*s*g)@beta
            close(frame.conj().T@beta@frame, at_point,
                  'parallel scalar grading acquires its exact spatial Killing-frame dependence')
            close(at_point@at_point, np.eye(rank), 'spatial scalar grading stays an involution')
            a, b = sphere.shells(depth, 5, np.array([t]))
            aa, bb = a[:4].sum(), b[:4].sum()
            count = comb(3+depth, depth)
            psi = (aa*u-1j*s*bb*g@u)/sqrt(count)
            f = (t*(aa*aa-s*s*bb*bb)-2*s*s*aa*bb)/count
            h = (aa*aa+2*t*aa*bb-s*s*bb*bb)/count
            predicted = f*polar[0]+s*h*np.dot(direction, polar[1:])
            close(np.vdot(psi, at_point@psi), predicted,
                  'general signed scalar current retains scalar and directional polarization moments')
            minus_a = t*a[:, 0]-s*s*b[:, 0]
            minus_b = -(a[:, 0]+t*b[:, 0])
            for level in range(6):
                plus = (a[level, 0]*np.eye(rank)-1j*s*b[level, 0]*g)
                minus = (minus_a[level]*np.eye(rank)-1j*s*minus_b[level]*g)
                close(at_point@plus, minus@beta,
                      'physical scalar grading maps every tested positive spatial shell into its opposite sign')
        if depth % 2:
            chirality = np.eye(rank, dtype=complex)
            for matrix in operators:
                chirality = chirality@matrix
            chirality *= (1j)**((depth+1)*depth//2)
            close(chirality, chirality.conj().T, 'even Clifford family has an exact Hermitian chirality')
            values, vectors = eigh(chirality)
            zero = vectors[:, np.argmax(values)]
        elif depth >= 6:
            zero = np.zeros(rank, complex)
            zero[0] = zero[7] = 1/sqrt(2)
        else:
            zero = None
        if zero is not None:
            moments = [np.vdot(zero, matrix@zero) for matrix in operators]
            close(moments, np.zeros(depth+1), 'explicit pure polarization has identically zero signed scalar current')
            for theta in (.13, .67, 1.7, 2.89):
                direction = rng.normal(size=depth)
                direction /= np.linalg.norm(direction)
                g = sum(x*matrix for x, matrix in zip(direction, gamma))
                t, s = np.cos(theta), np.sin(theta)
                at_point = (t*np.eye(rank)+1j*s*g)@beta
                a, b = sphere.shells(depth, 7, np.array([t]))
                for time in (.11, .43, 1.17):
                    phases = np.exp(-1j*(np.arange(8)+depth/2)*time)
                    aa, bb = phases@a[:, 0], phases@b[:, 0]
                    field = aa*zero-1j*s*bb*g@zero
                    close(np.vdot(field, at_point@field)/max(1., np.vdot(field, field).real), 0,
                          'zero signed scalar current survives distinct bare spatial mode phases')
        if depth in (2, 4):
            variables = sp.symbols('x:'+str(2*rank), real=True)
            vector = sp.Matrix([variables[2*j]+sp.I*variables[2*j+1] for j in range(rank)])
            norm = (vector.conjugate().T@vector)[0]
            bilinears = []
            for matrix in operators:
                exact = sp.Matrix([[sp.nsimplify(z) for z in row] for row in matrix])
                bilinears.append((vector.conjugate().T@exact@vector)[0])
            check(sp.expand(sum(value**2 for value in bilinears)-norm**2) == 0,
                  'complete pure polarization has unit scalar-vector length at depths two and four')
        cases.append({'depth': depth, 'physical_graded_spinor_rank': rank,
                      'explicit_zero_scalar_current_polarization': zero is not None,
                      'universal_unit_scalar_vector_length_proved': depth in (2, 4),
                      'odd_depth_graded_doubling': bool(depth % 2)})
    return cases


def band_data(depth, cutoff, extra=0):
    maximum = 3*cutoff+2
    t, w = sphere.quadrature(depth, max(24, 3*cutoff+5+extra))
    a, b = sphere.shells(depth, maximum, t)
    aa, bb = a[:cutoff+1].sum(axis=0), b[:cutoff+1].sum(axis=0)
    s2 = 1-t*t
    count = comb(cutoff+depth, depth)
    density = (aa*aa+s2*bb*bb)/count
    f = (t*(aa*aa-s2*bb*bb)-2*s2*aa*bb)/count
    h = (aa*aa+2*t*aa*bb-s2*bb*bb)/count
    minus_a, minus_b = t*a-s2*b, -(a+t*b)
    source_a = f*(t*aa-s2*bb)
    source_b = -f*(aa+t*bb)
    degeneracies = np.array([comb(l+depth-1, depth-1) for l in range(maximum+1)], float)
    positive = (a@(w*source_a)+b@(w*s2*source_b))/np.sqrt(degeneracies*count)
    negative = (minus_a@(w*source_a)+minus_b@(w*s2*source_b))/np.sqrt(degeneracies*count)
    i4 = float(w@(f*f))
    source_norm = float(w@(f*f*density))
    close(w@density, 1, 'scalar-source trial field retains canonical unit spatial norm')
    close(w@f, 0, 'a pure positive spatial band has zero integrated signed scalar current')
    close(np.sum(positive**2)+np.sum(negative**2), source_norm,
          'complete signed cubic source obeys both-sign Parseval')
    along = np.sqrt(degeneracies[:cutoff+1]/count)@positive[:cutoff+1]
    close(along, i4, 'scalar quartic action derivative equals the complete source component along its trial field')
    close(positive[-1]/sqrt(source_norm), 0,
          'highest positive polynomial allowance vanishes relative to complete source norm', 2e-9)
    if not cutoff:
        close(i4, 1/(depth+1), 'ground signed scalar quartic is the exact inverse ambient dimension')
        close(positive[0], 1/(depth+1), 'ground source retains its exact selected-shell coefficient')
        close(negative[1], sqrt(depth)/(depth+1), 'ground signed source generates the first opposite Dirac shell')
        close(1-positive[0]**2/source_norm, depth/(depth+1),
              'ground signed source has an exact nonzero omitted fraction')
    angular_coefficient = float(w@(s2*h*h))/depth
    check(i4 > 0 and angular_coefficient > 0, 'both signed-current polarization coefficients are positive')
    check(i4 <= float(w@(density*density))*(1+2e-8),
          'signed scalar quartic obeys the pointwise positive-intensity bound')
    return {'simplex_count': count, 'scalar_quartic_coefficient': i4,
            'scalar_quartic_per_count': i4/count, 'directional_quartic_coefficient': angular_coefficient,
            'directional_quartic_per_count': angular_coefficient/count,
            'minimum_scalar_quartic_over_pure_polarizations': min(i4, angular_coefficient) if depth in (2, 4) else 0.,
            'maximum_scalar_quartic_over_pure_polarizations': max(i4, angular_coefficient),
            'source_squared_norm': source_norm,
            'selected_positive_fraction': float(np.sum(positive[:cutoff+1]**2)/source_norm),
            'omitted_positive_fraction': float(np.sum(positive[cutoff+1:]**2)/source_norm),
            'negative_Dirac_fraction': float(np.sum(negative**2)/source_norm),
            'total_omitted_fraction': float(1-np.sum(positive[:cutoff+1]**2)/source_norm),
            'along_trial_source_fraction': float(i4*i4/source_norm)}


def main():
    polarization = algebra_checks()
    for depth in sphere.DEPTHS:
        band_data(depth, 0)
    cases = []
    for name, depth, index in sphere.ASSIGNMENTS:
        original, refined = band_data(depth, index-1), band_data(depth, index-1, 23)
        for key in original:
            close(original[key], refined[key], 'independent signed-source quadrature refinement', 8e-8)
        cases.append(dict(name=name, depth=depth, mode_index=index, cutoff=index-1, **refined))
        print(name, depth, index, 'scalar/count', refined['scalar_quartic_per_count'],
              'omitted', refined['total_omitted_fraction'], flush=True)
    ratios = []
    for depth in (2, 3, 4, 5, 6):
        family = [c for c in cases if c['depth'] == depth]
        low, high = family[0], family[-1]
        relative = high['scalar_quartic_per_count']/low['scalar_quartic_per_count']-1
        ratios.append({'depth': depth, 'low': low['name'], 'high': high['name'],
                       'relative_scalar_quartic_ratio_drift_from_count': relative})
    report = {'experiment_key': 'signed-scalar-current-spatial-band-source',
              'created_utc': datetime.now(timezone.utc).isoformat(),
              'sources': {'experiments/scalar-current-band-checks.py': sha256(Path(__file__).read_bytes()).hexdigest(),
                          'experiments/spatial-source-saturation-checks.py': sha256(HELPER.read_bytes()).hexdigest(),
                          'experiments/spatial-band-curvature-checks.py': sha256((ROOT/'spatial-band-curvature-checks.py').read_bytes()).hexdigest()},
              'classification': 'actual spatial signed-current and complete source comparison on a declared graded Dirac carrier',
              'assumptions': ['ordinary round sphere as a spatial control with undetermined physical radius',
                              'parallel Hermitian scalar grading beta anticommutes with the full spatial Dirac operator',
                              'odd spatial depths use the full graded double of the intrinsic spinor carrier',
                              'common local scalar-current quartic with positive unit-norm bright trial fields',
                              'radial complete-source calculations use beta-at-pole eigenpolarization, not every polarization'],
              'limits': ['trial fields are not nonlinear stationary solutions or localized particles',
                         'zero scalar current is not zero electric charge, absence of all coupling or a particle identification',
                         'other polarizations retain the explicitly calculated angular-current dependence',
                         'no stability, physical lifetime, selected mass clock or favorable sector preference is established'],
              'conditions': CONDITIONS, 'maximum_errors': ERRORS,
              'condition_count': len(CONDITIONS), 'evaluation_count': sum(CONDITIONS.values()),
              'polarization': polarization, 'cases': cases, 'family_ratios': ratios}
    path = ROOT/('scalar-current-band-results-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')+'.json')
    path.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'report': str(path), 'conditions': len(CONDITIONS),
                      'evaluations': sum(CONDITIONS.values()), 'maximum_errors': ERRORS}, indent=2), flush=True)


if __name__ == '__main__':
    main()
