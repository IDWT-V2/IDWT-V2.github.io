"""Exact identification of the secular ground current directions."""
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


def main():
    cases = []
    for depth in [2, 3, 4, 5, 6, 10]:
        family = 'scalar' if depth == 2 else 'coexact-vector'
        for load in [sp.Rational(1, 100), sp.Rational(1, 5)]:
            real, imaginary = angular.form_blocks(depth, sp.Integer(1), family, load)
            if depth == 2:
                vector = sp.Matrix([1, -2])
                particle = vector.col_join(2*vector).col_join(sp.zeros(4, 1))
                background = sp.zeros(4, 1).col_join(sp.zeros(2, 1)).col_join(vector)
                kernel_block, complementary_block = imaginary, real
                vectors = [sp.zeros(8, 1).col_join(particle), sp.zeros(8, 1).col_join(background)]
            else:
                vector = sp.Matrix([1, depth-1])
                particle = (2*vector).col_join(vector).col_join(sp.zeros(4, 1))
                background = sp.zeros(4, 1).col_join(vector).col_join(sp.zeros(2, 1))
                kernel_block, complementary_block = real, imaginary
                vectors = [particle.col_join(sp.zeros(8, 1)), background.col_join(sp.zeros(8, 1))]
            assert kernel_block*particle == sp.zeros(8, 1)
            assert kernel_block*background == sp.zeros(8, 1)
            assert kernel_block.rank() == 6 and complementary_block.rank() == 8
            generator = sp.BlockMatrix([[sp.zeros(8), imaginary], [-real, sp.zeros(8)]]).as_explicit()
            nullities = [int(16-(generator**j).rank()) for j in [1, 2, 3]]
            assert nullities == [2, 4, 4]
            assert sp.Matrix.hstack(*vectors).rank() == 2
            for vector in vectors:
                assert generator*vector == sp.zeros(16, 1)
            record = dict(depth=depth, family=family, load_exact=str(load),
                          zero_generator_nullities=nullities,
                          kernel_vectors=list(map(str, vectors)),
                          classification='complete secular eigendirections are the two independently verified right-unitary tangents')
            cases.append(record)
            print(json.dumps({k: v for k, v in record.items() if k != 'kernel_vectors'}), flush=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    output = ROOT/('covariant-clock-zero-results-'+stamp+'.json')
    report = dict(experiment_key='covariant-clock-right-unitary-zero-modes',
                  created_utc=datetime.now(timezone.utc).isoformat(),
                  sources={'experiments/'+p.name: sha256(p.read_bytes()).hexdigest()
                           for p in [Path(__file__), ROOT/'covariant-clock-angular.py', ROOT/'covariant-clock-ground.py']},
                  cases=cases,
                  limits=['secular eigenvectors change neither left density matrix to first order',
                          'nonlinear orbital persistence and uniform spatial norm bounds are not proved',
                          'ground controls do not identify observed excited particles'])
    output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(report=str(output))), flush=True)


if __name__ == '__main__':
    main()
