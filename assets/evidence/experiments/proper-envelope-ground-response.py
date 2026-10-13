"""Full current response of a unique quadratic count chart, without a clock block."""
from datetime import datetime, timezone
from hashlib import sha256
from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path
import json

import numpy as np
import sympy as sp

ROOT = Path(__file__).resolve().parent
spec = spec_from_file_location('discrete', ROOT/'discrete-count-ground-response.py')
discrete = module_from_spec(spec)
spec.loader.exec_module(discrete)
CHECKS, ERRORS = {}, {}


def close(actual, expected, label):
    error = float(np.linalg.norm(np.asarray(actual)-np.asarray(expected))/max(1., np.linalg.norm(expected)))
    assert error < 3e-10, (label, error)
    CHECKS[label] = CHECKS.get(label, 0)+1
    ERRORS[label] = max(ERRORS.get(label, 0.), error)


def main():
    z = sp.Symbol('z')
    cases = []
    for depth in [2, 3, 4, 5, 6, 10]:
        for harmonic in [1, 2, 3, 5, 10]:
            real, imaginary, epsilon = discrete.exact_blocks(depth, harmonic)
            upper, lower, divisor = harmonic+depth-1, harmonic, 2*harmonic+depth-1
            scalar = sp.Matrix([-1, 2, -1, 2])*sp.Matrix([[-upper, 2*upper, -lower, 2*lower]])/(5*divisor)
            real -= 2*scalar
            physical_r, physical_i = discrete.physical_blocks(depth, harmonic)
            u, v = np.sqrt(upper/divisor), np.sqrt(lower/divisor)
            scalar_vector = np.array([-u, 2*u, -v, 2*v])/np.sqrt(5)
            physical_r -= 2*np.outer(scalar_vector, scalar_vector)
            scaling = np.diag([np.sqrt(upper)]*2+[np.sqrt(lower)]*2)
            close(np.linalg.solve(scaling, physical_r@scaling), np.array(real, float),
                  'independent physical source blocks have exactly zero scalar-ratio curvature in the affine chart')
            close(np.linalg.solve(scaling, physical_i@scaling), np.array(imaginary, float),
                  'the complete proper-current Hessian survives the affine count chart')
            polynomial = sp.Poly((imaginary*real).charpoly(z).as_expr(), z)
            close(np.poly(physical_i@physical_r), np.array(polynomial.all_coeffs(), float),
                  'the exact squared-frequency polynomial matches the physical normalized linearization')
            nonzero, zeros = polynomial, 0
            while nonzero.eval(0) == 0:
                nonzero = sp.div(nonzero, sp.Poly(z, z))[0]
                zeros += 1
            negatives = int(nonzero.count_roots(-sp.oo, 0))
            nonreal = nonzero.degree()-int(nonzero.count_roots(-sp.oo, sp.oo))
            first_order = sp.BlockMatrix([[sp.zeros(4), imaginary], [-real, sp.zeros(4)]]).as_explicit()
            nullities = [8-(first_order**k).rank() for k in [1, 2, 3]] if zeros else []
            cases.append(dict(depth=depth, angular_harmonic=harmonic, ground_c_exact='3/5',
                              diagnostic_epsilon_exact=str(epsilon), scalar_hessian_factor_exact='0',
                              polynomial_exact=str(sp.factor(polynomial.as_expr())),
                              zero_squared_frequency_multiplicity=zeros, negative_real_squared_frequencies=negatives,
                              nonreal_squared_frequency_roots=nonreal, zero_generator_power_nullities=nullities,
                              classification='ground full-matrix reducing spatial family; not an assigned excited or reduced-column state'))
        print(json.dumps(dict(depth=depth, first=cases[-5], last=cases[-1])), flush=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    sources = [Path(__file__), ROOT/'discrete-count-ground-response.py', ROOT/'inverse-current-radial.py']
    report = dict(experiment_key='proper-current-count-envelope-ground-response', created_utc=datetime.now(timezone.utc).isoformat(),
                  sources={'experiments/'+p.name: sha256(p.read_bytes()).hexdigest() for p in sources},
                  evaluations=sum(CHECKS.values()), conditions=CHECKS, maximum_errors=ERRORS, cases=cases,
                  limits=['the Lorentz-scalar proper-current chart is V=m*s+eta*sqrt(j_mu*j^mu)',
                          'the preferred-time envelope nonlinear unitary theorem does not transfer',
                          'these full-matrix ground controls are not the assigned excited particles or minimum-column Hessians',
                          'nonnegative real squared frequencies in this family would not establish full angular or nonlinear persistence'])
    path = ROOT/('proper-envelope-ground-response-results-'+stamp+'.json')
    path.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(report=str(path), evaluations=report['evaluations'], maximum_errors=ERRORS)), flush=True)


if __name__ == '__main__':
    main()
