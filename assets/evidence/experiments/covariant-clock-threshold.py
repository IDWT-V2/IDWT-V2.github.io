"""Exact density-ratio threshold for the responding coexact current block."""
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


def main():
    cases = []
    load = sp.symbols('g', positive=True)
    c = sp.Rational(3, 5)
    for depth in [3, 4, 5, 6, 10]:
        h = sp.Rational(depth, 2)
        omega = c/sum(sp.Rational(1, j) for j in range(1, depth+1))
        source = omega-h/c
        k = -2*source
        delta = (L-1)*(L+depth)
        critical = sp.factor(delta**2/(2*k*delta+4*k*k*c))
        real, imaginary = angular.form_blocks(depth, L, 'coexact-vector', load)
        determinant_ratio = sp.factor(imaginary.det()/real.det())
        expected = 1-load/critical
        assert sp.cancel(determinant_ratio-expected) == 0
        at_two = sp.factor(critical.subs(L, 2))
        ratio = sp.factor(expected.subs({L: 2, load: sp.Rational(1, 5)}))
        record = dict(depth=depth, c_exact=str(c),
                      det_Y_over_det_X=str(determinant_ratio),
                      critical_load_all_harmonics=str(critical),
                      harmonic_two_critical_load_exact=str(at_two),
                      harmonic_two_critical_load_decimal=float(at_two),
                      harmonic_two_det_ratio_at_load_one_fifth=str(ratio),
                      forced_negative_squared_root_at_load_one_fifth=bool(ratio < 0))
        cases.append(record)
        print(json.dumps(record), flush=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    output = ROOT/('covariant-clock-threshold-results-'+stamp+'.json')
    report = dict(experiment_key='covariant-clock-coexact-density-threshold',
                  created_utc=datetime.now(timezone.utc).isoformat(),
                  sources={'experiments/'+p.name: sha256(p.read_bytes()).hexdigest()
                           for p in [Path(__file__), ROOT/'covariant-clock-angular.py', ROOT/'covariant-clock-ground.py']},
                  cases=cases,
                  limits=['determinant sign gives a sufficient instability test, not a sufficient stability test',
                          'ground coexact modes at d>=3 only; depth two has an additional current channel',
                          'field density ratio is explicit and remains physically uncalibrated'])
    output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(report=str(output))), flush=True)


if __name__ == '__main__':
    main()
