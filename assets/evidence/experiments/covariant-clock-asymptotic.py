"""Check complete factors and physical high-harmonic spectral separation."""
from datetime import datetime, timezone
from hashlib import sha256
from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path
import json

import sympy as sp

ROOT = Path(__file__).resolve().parent
spec = spec_from_file_location('angular', ROOT/'covariant-clock-angular.py')
angular = module_from_spec(spec)
spec.loader.exec_module(angular)
L, Z = angular.L, angular.Z
T = sp.Symbol('t')


def main():
    path = sorted(ROOT.glob('covariant-clock-uniform-results-*.json'))[-1]
    report = json.loads(path.read_text())
    cases = []
    for case in report['cases']:
        assert case['certified_all_harmonics']
        depth, family = case['depth'], case['family']
        full = sp.sympify(case['polynomial'], locals={'L': L, 'z': Z})
        n = int(case['coupled_degree'])
        coupled = max((sp.Poly(f, Z).monic() for f, m in sp.factor_list(full, Z)[1]), key=lambda p: p.degree())
        assert coupled.degree() == n
        if depth == 2:
            dark, dark_limit = sp.Integer(1), sp.Integer(1)
        elif family == 'scalar':
            dark = (Z-(L-1)**2)*(Z-(L+depth)**2)
            dark_limit = (T+2)*(T-2*depth)
        else:
            dark = (Z-L**2)*(Z-(L+depth-1)**2)
            dark_limit = T*(T-2*(depth-1))
        assert sp.cancel(full-coupled.as_expr()*dark) == 0
        shifted = sp.Poly(sp.expand(coupled.as_expr().subs(Z, L*L+T*L)), L)
        assert shifted.degree() == n
        limiting = sp.Poly(shifted.LC(), T)
        assert limiting.LC() == 1 and limiting.degree() == n
        h, c, g = sp.Rational(depth, 2), sp.Rational(3, 5), sp.Rational(1, 100)
        lam, mass = h/c, 4*h/3
        if family == 'scalar':
            p, q, eigenvalue = h, 1-h, depth
        else:
            p, q, eigenvalue = 1-h, h-2, 2*(depth-1)
        sigma = sp.Matrix([[0, 1], [1, 0]])
        c0 = sp.diag(p, q)+sp.Rational(depth-1, 2)*sigma
        d0, d1 = sp.diag(c0, -c0), sp.diag(sigma, -sigma)
        beta = sp.zeros(4)
        beta[:2, 2:] = beta[2:, :2] = sp.eye(2)
        base0 = sp.diag(d0+mass*beta-lam*sp.eye(4), d0-h*sp.eye(4))
        lead = sp.diag(d1, d1)
        one = sp.Matrix([[p, 1], [eigenvalue, q]])
        dd = sp.diag(one, -one)
        base_one = sp.diag(dd+mass*beta-lam*sp.eye(4), dd-h*sp.eye(4))
        xx, yy = angular.form_blocks(depth, sp.Integer(1), family, g)
        dx, dy = xx-base_one, yy-base_one
        limit_matrix = lead*(base0+dx)+(base0+dy)*lead
        scale = sp.sqrt(g/5)
        limit_matrix[:4, 4:] = (limit_matrix[:4, 4:]/scale).applyfunc(sp.simplify)
        limit_matrix[4:, :4] = (scale*limit_matrix[4:, :4]).applyfunc(sp.simplify)
        assert sp.cancel(limit_matrix.charpoly(T).as_expr()-limiting.as_expr()*dark_limit) == 0
        discriminant = sp.factor(sp.discriminant(limiting.as_expr(), T))
        real_roots = int(limiting.count_roots(-sp.oo, sp.oo))
        simple_real_limit = discriminant != 0 and real_roots == n
        record = dict(depth=depth, family=family, coupled_degree=n,
                      high_harmonic_offset_polynomial=str(limiting.as_expr()),
                      offset_discriminant=str(discriminant), offset_real_distinct_roots=real_roots,
                      asymptotic_simple_real_limit=bool(simple_real_limit),
                      actual_physical_limit_matrix_agrees=True,
                      all_other_factors_independently_verified=True)
        cases.append(record)
        print(json.dumps({k: v for k, v in record.items() if k not in ['offset_discriminant', 'high_harmonic_offset_polynomial']}), flush=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    output = ROOT/('covariant-clock-asymptotic-results-'+stamp+'.json')
    result = dict(experiment_key='covariant-clock-uniform-physical-asymptotics',
                  created_utc=datetime.now(timezone.utc).isoformat(),
                  sources={'experiments/'+p.name: sha256(p.read_bytes()).hexdigest()
                           for p in [Path(__file__), ROOT/'covariant-clock-angular.py', ROOT/'covariant-clock-ground.py']} | {str(path.relative_to(Path.cwd())): sha256(path.read_bytes()).hexdigest()},
                  cases=cases,
                  limits=['simple distinct real limiting offsets support uniformly bounded diagonalization at large harmonic index',
                          'zero-frequency symmetry directions require the separate quotient argument',
                          'no nonlinear or assigned excited-state persistence theorem'])
    output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(report=str(output))), flush=True)


if __name__ == '__main__':
    main()
