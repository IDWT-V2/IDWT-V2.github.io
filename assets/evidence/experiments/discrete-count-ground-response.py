"""Exact first spatial-harmonic response of the integer-count current action."""
from datetime import datetime, timezone
from hashlib import sha256
from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path
import json

import numpy as np
import sympy as sp

ROOT = Path(__file__).resolve().parent
spec = spec_from_file_location('radial', ROOT/'inverse-current-radial.py')
radial = module_from_spec(spec)
spec.loader.exec_module(radial)
COUNTS, ERRORS = {}, {}


def check(value, label):
    if not bool(value):
        raise AssertionError(label)
    COUNTS[label] = COUNTS.get(label, 0)+1


def close(actual, expected, label):
    error = float(np.linalg.norm(np.asarray(actual)-np.asarray(expected))/max(1., np.linalg.norm(expected)))
    ERRORS[label] = max(ERRORS.get(label, 0.), error)
    check(error < 2e-10, label)


def exact_blocks(depth, harmonic):
    h, c, x = sp.Rational(depth, 2), sp.Rational(3, 5), sp.Rational(4, 5)
    hd = sum(sp.Rational(1, j) for j in range(1, depth+1))
    epsilon = c/hd
    lam, vector = h/c, 1-epsilon*c/h
    upper, lower, divisor = harmonic+depth-1, harmonic, 2*harmonic+depth-1
    grading = sp.zeros(4)
    grading[0, 1] = grading[1, 0] = grading[2, 3] = grading[3, 2] = 1
    free = sp.diag(harmonic+h, -harmonic-h, -harmonic+1-h, harmonic-1+h)/lam+x*grading-sp.eye(4)
    scalar = sp.Matrix([-1, 2, -1, 2])*sp.Matrix([[-upper, 2*upper, -lower, 2*lower]])/(5*divisor)
    radial_form = sp.Matrix([2*lower, -lower, -2*upper, upper])*sp.Matrix([[2, -1, -2, 1]])/(5*divisor)
    real, imaginary = free+2*scalar, free+2*vector*radial_form
    if depth == 2:
        transverse = sp.Matrix([lower, -2*lower, -upper, 2*upper])*sp.Matrix([[1, -2, -1, 2]])/(5*divisor)
        real += 2*vector*transverse
    return real, imaginary, epsilon


def physical_blocks(depth, harmonic):
    c, x, h = .6, .8, depth/2
    a, b = 2/np.sqrt(5), 1/np.sqrt(5)
    u, v = np.sqrt((harmonic+depth-1)/(2*harmonic+depth-1)), np.sqrt(harmonic/(2*harmonic+depth-1))
    epsilon = c/sum(1/j for j in range(1, depth+1))
    grading = np.zeros((4, 4)); grading[0, 1] = grading[1, 0] = grading[2, 3] = grading[3, 2] = 1
    free = np.diag([harmonic+h, -harmonic-h, -harmonic+1-h, harmonic-1+h])/(h/c)+x*grading-np.eye(4)
    scalar = np.array([-b*u, a*u, -b*v, a*v])
    vector = np.array([a*v, -b*v, -a*u, b*u])
    real, imaginary = free+2*np.outer(scalar, scalar), free+2*(1-epsilon*c/h)*np.outer(vector, vector)
    if depth == 2:
        transverse = np.array([b*v, -a*v, -b*u, a*u])
        real += 2*(1-epsilon*c/h)*np.outer(transverse, transverse)
    return real, imaginary


def main():
    z = sp.Symbol('z')
    cases = []
    for depth in radial.DEPTHS:
        for harmonic in [1, 2, 3, 5, 10]:
            real, imaginary, epsilon = exact_blocks(depth, harmonic)
            numeric_r, numeric_i = physical_blocks(depth, harmonic)
            scaling = np.diag([np.sqrt(harmonic+depth-1)]*2+[np.sqrt(harmonic)]*2)
            close(np.linalg.solve(scaling, numeric_r@scaling), np.array(real, float),
                  'independent orthonormal physical blocks agree with exact rational similarity')
            close(np.linalg.solve(scaling, numeric_i@scaling), np.array(imaginary, float),
                  'independent orthonormal physical blocks agree with exact rational similarity')
            char = sp.Poly((imaginary*real).charpoly(z).as_expr(), z)
            numeric = np.poly(numeric_i@numeric_r)
            close(numeric, np.array(char.all_coeffs(), float),
                  'independent full physical eigenproblem agrees with the exact squared-frequency polynomial')
            nonzero, zero_multiplicity = char, 0
            while nonzero.eval(0) == 0:
                nonzero = sp.div(nonzero, sp.Poly(z, z))[0]
                zero_multiplicity += 1
            negative_real_roots = int(nonzero.count_roots(-sp.oo, 0))
            real_roots = int(nonzero.count_roots(-sp.oo, sp.oo))
            nonreal_roots = nonzero.degree()-real_roots
            eigenvalues = np.linalg.eigvals(numeric_i@numeric_r)
            cases.append(dict(depth=depth, angular_harmonic=harmonic,
                              ground_c_exact='3/5', diagnostic_epsilon_exact=str(epsilon),
                              scalar_hessian_factor_exact='1', polynomial_exact=str(char.as_expr()),
                              negative_real_squared_frequencies_exact=negative_real_roots,
                              zero_squared_frequency_multiplicity_exact=zero_multiplicity,
                              nonreal_squared_frequency_roots_exact=nonreal_roots,
                              eigenvalues=[dict(real=float(v.real), imaginary=float(v.imag)) for v in eigenvalues],
                              classification='exact reducing ground rotational spatial block, not an assigned particle'))
        print(json.dumps(dict(completed_depth=depth, first_negative_count=cases[-5]['negative_real_squared_frequencies_exact'],
                              first_nonreal_count=cases[-5]['nonreal_squared_frequency_roots_exact'])), flush=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    report = dict(experiment_key='discrete-count-ground-spatial-response', created_utc=datetime.now(timezone.utc).isoformat(),
                  sources={'experiments/'+p.name: sha256(p.read_bytes()).hexdigest() for p in [Path(__file__), ROOT/'inverse-current-radial.py']},
                  evaluations=sum(COUNTS.values()), conditions=COUNTS, maximum_errors=ERRORS, cases=cases,
                  limits=['these are ground comparison preparations with c3/5 and epsilon=c/Hd',
                          'the scalar curvature belongs to the discrete count law, not the original reconstructed law',
                          'one exact negative squared frequency proves growth in that reducing spatial block',
                          'zero negative-real roots do not exclude complex frequencies or establish persistence',
                          'ground results do not identify the assigned excited electron or other particles'])
    output = ROOT/('discrete-count-ground-response-results-'+stamp+'.json')
    output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(report=str(output), evaluations=report['evaluations'], maximum_errors=ERRORS)), flush=True)


if __name__ == '__main__':
    main()
