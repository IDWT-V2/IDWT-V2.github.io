"""Spatial Dirac source saturation, two-point smearing and full cubic leakage."""
from datetime import datetime, timezone
from hashlib import sha256
from importlib.util import module_from_spec, spec_from_file_location
from math import comb, factorial, prod, sqrt
from pathlib import Path
import json

import numpy as np
from scipy.linalg import cho_factor, cho_solve, eigvalsh
from scipy.special import eval_gegenbauer, roots_jacobi
import sympy as sp

ROOT = Path(__file__).resolve().parent
HELPER = ROOT/'spatial-band-curvature-checks.py'
spec = spec_from_file_location('spatial_band', HELPER)
spatial = module_from_spec(spec)
spec.loader.exec_module(spatial)
DEPTHS = (2, 3, 4, 5, 6, 10)
ASSIGNMENTS = [('W', 2, 76), ('Z', 2, 81), ('Higgs', 2, 95),
               ('down', 3, 1), ('strange', 3, 4),
               ('up', 4, 3), ('charm', 4, 20), ('top', 4, 72),
               ('nu1', 5, 10), ('nu2', 5, 15), ('nu3', 5, 22),
               ('electron', 6, 13), ('muon', 6, 35), ('tau', 10, 23)]
conditions = {}
errors = {}


def check(value, label):
    if not bool(value):
        raise AssertionError(label)
    conditions[label] = conditions.get(label, 0)+1


def close(actual, expected, label, tolerance=3e-9):
    actual, expected = np.asarray(actual), np.asarray(expected)
    relative = float(np.max(np.abs(actual-expected))/max(1.0, float(np.max(np.abs(expected)))))
    errors[label] = max(errors.get(label, 0.0), relative)
    check(relative < tolerance, label)


def shells(depth, maximum, t):
    a, b = np.zeros((maximum+1, len(t))), np.zeros((maximum+1, len(t)))
    lam = (depth-1)/2
    for level in range(maximum+1):
        a[level] = (level+depth-1)/(depth-1)*eval_gegenbauer(level, lam, t)
        if level:
            b[level] = eval_gegenbauer(level-1, lam+1, t)
    return a, b


def quadrature(depth, nodes):
    t, weights = roots_jacobi(nodes, (depth-2)/2, (depth-2)/2)
    return t, weights/np.sum(weights)


def independent_evaluation(depth, level, points):
    gamma = spatial.gammas(depth)
    rank, ambient = len(gamma[0]), depth+1
    symbol = gamma+[1j*np.eye(rank)]
    indices = list(spatial.compositions(level, ambient))
    lookup = {alpha: i for i, alpha in enumerate(indices)}
    dimension = len(indices)*rank
    if level:
        lower = list(spatial.compositions(level-1, ambient))
        c = np.zeros((len(lower)*rank, dimension), complex)
        for i, beta in enumerate(lower):
            for axis in range(ambient):
                alpha = list(beta)
                alpha[axis] += 1
                j = lookup[tuple(alpha)]
                c[i*rank:(i+1)*rank, j*rank:(j+1)*rank] = symbol[axis]*sqrt(alpha[axis])
        factor = cho_factor(c@c.conj().T)
        def project(value):
            return value-c.conj().T@cho_solve(factor, c@value)
    else:
        def project(value):
            return value
    normalization = sqrt(prod(ambient+2*j for j in range(level)))
    result = []
    for point in points:
        evaluation = np.hstack([np.eye(rank)*normalization*np.prod(np.asarray(point)**alpha)/
                                sqrt(prod(factorial(i) for i in alpha)) for alpha in indices])
        result.append(project(evaluation.conj().T))
    return result


def exact_checks():
    t, d = sp.symbols('t d', real=True)
    numerator = (1+d*t)**2+1-t*t
    deficit = sp.expand((d+1)**2-numerator)
    check(sp.expand(deficit-(1-t)*((d**2-1)*t+d**2+2*d-1)) == 0,
          'ground-plus-first-shell exact overlap deficit')
    check(sp.expand(2*d-2*d*t-2*d*(1-t)) == 0,
          'equal ground polarization leaves a strict first-shell separation')
    for depth in DEPTHS:
        for level in (1, 2, 3):
            r, z = sp.symbols('r z', positive=True)
            radius = sp.sqrt(r*r+z*z)
            lam = sp.Rational(depth-1, 2)
            aa = sp.cancel(sp.Rational(level+depth-1, depth-1)*radius**level*
                           sp.gegenbauer(level, lam, z/radius))
            bb = sp.cancel(radius**(level-1)*sp.gegenbauer(level-1, lam+1, z/radius))
            check(sp.simplify(sp.diff(aa, z)-depth*bb-r*sp.diff(bb, r)) == 0,
                  'zonal kernel is ambient monogenic scalar component')
            check(sp.simplify(sp.diff(aa, r)/r+sp.diff(bb, z)) == 0,
                  'zonal kernel is ambient monogenic vector component')
        x, w = quadrature(depth, 20)
        a, b = shells(depth, 4, x)
        g = np.array([comb(l+depth-1, depth-1) for l in range(5)], float)
        minus_a, minus_b = x*a-(1-x*x)*b, -(a+x*b)
        plus_gram = (a*w)@a.T+(b*w*(1-x*x))@b.T
        minus_gram = (minus_a*w)@minus_a.T+(minus_b*w*(1-x*x))@minus_b.T
        cross = (a*w)@minus_a.T+(b*w*(1-x*x))@minus_b.T
        close(plus_gram, np.diag(g), 'positive zonal shells have physical orthogonal norms')
        close(minus_gram, np.diag(g), 'negative zonal shells have physical orthogonal norms')
        close(cross, np.zeros_like(cross), 'both Dirac signs are explicitly retained')


def matrix_kernel_checks():
    for depth in DEPTHS:
        rank = 2**(depth//2)
        north = np.eye(depth+1)[-1]
        angle = .713
        point = north*np.cos(angle)
        point[0] = np.sin(angle)
        frames_n, frames_y = [], []
        for level in (0, 1, 2):
            n, y = independent_evaluation(depth, level, [north, point])
            frames_n.append(n)
            frames_y.append(y)
            aa, bb = shells(depth, level, np.array([np.cos(angle)]))
            expected = aa[level, 0]*np.eye(rank)-1j*np.sin(angle)*bb[level, 0]*spatial.gammas(depth)[0]
            close(y.conj().T@n, expected, 'kernel agrees with independently projected polynomial modes')
            count = comb(level+depth, depth)
            un, uy = np.vstack(frames_n)/sqrt(count), np.vstack(frames_y)/sqrt(count)
            overlap = uy.conj().T@un
            cumulative = np.sum(aa[:level+1, 0]), np.sum(bb[:level+1, 0])
            singular = sqrt(cumulative[0]**2+np.sin(angle)**2*cumulative[1]**2)/count
            close(overlap.conj().T@overlap, singular**2*np.eye(rank),
                  'all spatial polarization overlap singular values agree')
            source_gram = count/2*np.block([[np.eye(rank), un.conj().T@uy],
                                           [uy.conj().T@un, np.eye(rank)]])
            target = sorted([count*(1-singular)/2]*rank+[count*(1+singular)/2]*rank)
            close(eigvalsh(source_gram), target, 'two-point contact nonzero eigenvalues from independent frames')
            if level:
                check(singular < 1, 'distinct points cannot saturate an excited cumulative band')
                first = sqrt((1+depth*np.cos(angle))**2+np.sin(angle)**2)/(depth+1)
                check(singular <= 1-(depth+1)/count*(1-first)+3e-12,
                      'ground-plus-first-shell overlap bounds every larger band')


def band_data(depth, cutoff, extra_nodes=0):
    maximum = max(1, 3*cutoff)
    t, weights = quadrature(depth, max(20, 3*cutoff+3+extra_nodes))
    a, b = shells(depth, maximum, t)
    aa, bb = np.sum(a[:cutoff+1], axis=0), np.sum(b[:cutoff+1], axis=0)
    count = comb(cutoff+depth, depth)
    density = (aa*aa+(1-t*t)*bb*bb)/count
    close(weights@density, 1, 'bright spatial field has canonical unit norm', 2e-8)
    i4, i6 = float(weights@(density*density)), float(weights@(density**3))
    g = np.array([comb(l+depth-1, depth-1) for l in range(maximum+1)], float)
    minus_a, minus_b = t*a-(1-t*t)*b, -(a+t*b)
    positive = ((a*(weights*density))@aa+(b*(weights*density*(1-t*t)))@bb)/np.sqrt(g*count)
    negative = ((minus_a*(weights*density))@aa+
                (minus_b*(weights*density*(1-t*t)))@bb)/np.sqrt(g*count)
    close(np.sum(positive**2)+np.sum(negative**2), i6,
          'complete cubic source Parseval identity across both Dirac signs', 4e-8)
    selected = np.sum(positive[:cutoff+1]**2)
    positive_out = np.sum(positive[cutoff+1:]**2)
    negative_out = np.sum(negative**2)
    check(-1e-10 <= 1-selected/i6 <= 1+1e-10, 'full source leakage fraction is admissible')
    if cutoff:
        check(i4 < count, 'smooth bright self-density strictly misses full contact saturation')
    return dict(count=count, quartic_overlap=i4, self_density_fraction_of_point_contact=i4/count,
                cubic_source_squared_norm=i6, selected_positive_fraction=float(selected/i6),
                omitted_positive_fraction=float(positive_out/i6),
                negative_Dirac_fraction=float(negative_out/i6),
                total_omitted_fraction=float(1-selected/i6))


def pair_source_fraction(depth, cutoff, angle):
    aa, bb = shells(depth, cutoff, np.array([np.cos(angle)]))
    count = comb(cutoff+depth, depth)
    singular = sqrt(float(np.sum(aa)**2+np.sin(angle)**2*np.sum(bb)**2))/count
    return (1+min(1.0, singular))/2


def main():
    exact_checks()
    matrix_kernel_checks()
    cases = []
    for label, depth, mode in ASSIGNMENTS:
        cutoff = mode-1
        data = band_data(depth, cutoff)
        refined = band_data(depth, cutoff, 19)
        for key in ('quartic_overlap', 'cubic_source_squared_norm', 'total_omitted_fraction',
                    'negative_Dirac_fraction'):
            close(refined[key], data[key], 'independent quadrature refinement', 3e-8)
        metric = (depth-1+1/(depth+2))*cutoff*(cutoff+depth+1)/(depth+1)**2
        angle = 1e-4/max(1, cutoff)
        fraction = pair_source_fraction(depth, cutoff, angle)
        close((1-fraction)/angle**2, metric/4,
              'two-point width correction agrees with existing spatial quantum metric', 8e-4)
        sources = {str(a): pair_source_fraction(depth, cutoff, a) for a in (.001, .01, .05)}
        cases.append(dict(particle=label, depth=depth, mode_index=mode, cutoff=cutoff,
                          sphere_radius='R; all angles dimensionless, physical separation R times angle',
                          **data, two_equal_point_contact_fractions=sources))
        print(label, depth, mode, 'self/point', data['self_density_fraction_of_point_contact'],
              'leakage', data['total_omitted_fraction'], flush=True)
    comparisons = []
    for depth in DEPTHS:
        available = [case for case in cases if case['depth'] == depth]
        low, high = available[0], available[-1]
        comparisons.append(dict(depth=depth, low=low['particle'], high=high['particle'],
                                within_sector_ratio_available=len(available) > 1,
                                quartic_ratio_relative_to_simplex=(high['quartic_overlap']/low['quartic_overlap'])/
                                (high['count']/low['count'])-1))
    report = dict(created_at=datetime.now(timezone.utc).isoformat(),
                  sources={'experiments/spatial-source-saturation-checks.py': sha256(Path(__file__).read_bytes()).hexdigest(),
                           'experiments/spatial-band-curvature-checks.py': sha256(HELPER.read_bytes()).hexdigest()},
                  conditions=conditions, max_relative_errors=errors, cases=cases, family_comparisons=comparisons,
                  scope='Actual positive spherical Dirac bands; density self-contact is a classical comparison, not electron position probability or a Lorentz-scalar master derivation.')
    destination = ROOT/('spatial-source-saturation-results-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')+'.json')
    destination.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(report=str(destination), checks=sum(conditions.values()), conditions=len(conditions)), indent=2))


if __name__ == '__main__':
    main()
