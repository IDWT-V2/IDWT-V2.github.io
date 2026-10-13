"""Scalar and coexact-vector ground current harmonics with a responding clock."""
from datetime import datetime, timezone
from hashlib import sha256
from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path
import json

import numpy as np
import sympy as sp

ROOT = Path(__file__).resolve().parent
spec = spec_from_file_location('pilot', ROOT/'covariant-clock-ground.py')
pilot = module_from_spec(spec)
spec.loader.exec_module(pilot)
L, Z = pilot.L, pilot.Z


def form_blocks(depth, degree, family, load=sp.Rational(1, 5)):
    h, c = sp.Rational(depth, 2), sp.Rational(3, 5)
    lam, mass = h/c, 4*h/3
    source = c/sum(sp.Rational(1, j) for j in range(1, depth+1))-lam
    if family == 'scalar':
        eigenvalue = degree*(degree+depth-1)
        p, q, vector_index = h, 1-h, 1
    else:
        eigenvalue = (degree+1)*(degree+depth-2)
        p, q, vector_index = 1-h, h-2, 0
    two = sp.Matrix([[p, 1], [eigenvalue, q]])
    kinetic = sp.diag(two, -two)
    beta = sp.zeros(4)
    beta[:2, 2:] = sp.eye(2)
    beta[2:, :2] = sp.eye(2)
    particle = kinetic+mass*beta-lam*sp.eye(4)
    background = kinetic-h*sp.eye(4)
    real = sp.diag(particle, background)
    imaginary = real.copy()
    rb = sp.eye(4)[:, vector_index]
    # a sqrt(load) and b sqrt(load) have no irrational factor at load=1/5.
    rp = sp.sqrt(load/5)*(2*rb-beta*rb)
    imaginary[:4, 4:] += -2*source*rp*rb.T
    imaginary[4:, :4] += -2*source*rb*rp.T
    imaginary[4:, 4:] += 2*source*load*rb*rb.T
    if depth == 2:
        assert family == 'scalar'
        sb = beta*rb
        sparticle = sp.sqrt(load/5)*(2*sb-beta*sb)
        real[:4, 4:] += -2*source*sparticle*sb.T
        real[4:, :4] += -2*source*sb*sparticle.T
        real[4:, 4:] += 2*source*load*sb*sb.T
    return real, imaginary


def main():
    cases = []
    for depth in [2, 3, 4, 5, 6, 10]:
        for family in (['scalar'] if depth == 2 else ['scalar', 'coexact-vector']):
            real, imaginary = form_blocks(depth, L, family)
            polynomial = sp.factor((imaginary*real).charpoly(Z).as_expr())
            if family == 'scalar':
                px, py = pilot.blocks(depth)
                difference = sp.cancel(polynomial-(py*px).charpoly(Z).as_expr())
                if difference != 0:
                    raise AssertionError(('independent scalar harmonic map', depth))
            root_data = []
            for harmonic in [1, 2, 3, 5, 10, 20]:
                xp, yp = form_blocks(depth, sp.Integer(harmonic), family)
                poly = sp.Poly(polynomial.subs(L, harmonic), Z)
                factors = sp.factor_list(poly.as_expr(), Z)[1]
                zero_multiplicity = int(sum(m for f, m in factors if sp.Poly(f, Z).as_expr() == Z))
                nonzero = sp.Poly(sp.cancel(poly.as_expr()/Z**zero_multiplicity), Z)
                # Sturm counts count distinct real roots; multiplicity is retained separately.
                roots = nonzero.count_roots(0, sp.oo)
                generator = sp.BlockMatrix([[sp.zeros(8), yp], [-xp, sp.zeros(8)]]).as_explicit()
                nullities = []
                if harmonic == 1:
                    power = sp.eye(16)
                    for order in range(1, 4):
                        power = power*generator
                        nullities.append(int(16-power.rank()))
                eigenvalues = np.linalg.eigvals(np.asarray((yp*xp).evalf(), float))
                growth = float(max(np.sqrt(-eigenvalues.astype(complex)).real))
                root_data.append(dict(harmonic=harmonic, zero_squared_frequency_multiplicity=zero_multiplicity,
                                      positive_distinct_sturm_roots=int(roots), nonzero_degree=int(nonzero.degree()),
                                      zero_generator_nullities=nullities, floating_growth=growth))
            record = dict(depth=depth, family=family, c_exact='3/5', load_exact='1/5',
                          polynomial=str(polynomial), checks=root_data)
            cases.append(record)
            print(json.dumps({k: v for k, v in record.items() if k != 'polynomial'}), flush=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    output = ROOT/('covariant-clock-angular-results-'+stamp+'.json')
    report = dict(experiment_key='covariant-clock-angular-harmonics', created_utc=datetime.now(timezone.utc).isoformat(),
                  sources={'experiments/'+p.name: sha256(p.read_bytes()).hexdigest()
                           for p in [Path(__file__), ROOT/'covariant-clock-ground.py']}, cases=cases,
                  limits=['ground-shell control with explicit particle/background density ratio',
                          'scalar family has independent basis-polynomial agreement',
                          'coexact family remains conditional on the stated Killing-frame differential-form reduction',
                          'finite harmonic tests do not prove uniform stability'])
    output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(report=str(output))), flush=True)


if __name__ == '__main__':
    main()
