"""Exact-shell geometry and current response for all inferred assignments."""
from datetime import datetime, timezone
from hashlib import sha256
from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path
from math import comb
import json

import numpy as np
from scipy.special import eval_gegenbauer
import sympy as sp

ROOT = Path(__file__).resolve().parent
spec = spec_from_file_location('radial', ROOT/'inverse-current-radial.py')
radial = module_from_spec(spec)
spec.loader.exec_module(radial)
COUNTS, ERRORS = {}, {}


def close(actual, expected, label, tolerance=3e-10):
    error = float(np.linalg.norm(np.asarray(actual)-np.asarray(expected))/max(1., np.linalg.norm(expected)))
    ERRORS[label] = max(ERRORS.get(label, 0.), error)
    if error >= tolerance:
        raise AssertionError((label, error))
    COUNTS[label] = COUNTS.get(label, 0)+1


def shell_jets(depth, level, position, anchor, tangent, alpha, column):
    rank = len(alpha[0])
    t = np.dot(position, anchor)
    vector = tangent.T@anchor
    signed = sum(v*a for v, a in zip(vector, alpha))
    aa = (level+depth-1)/(depth-1)*eval_gegenbauer(level, (depth-1)/2, t)
    bb = eval_gegenbauer(level-1, (depth+1)/2, t) if level else 0.
    ap = (level+depth-1)*bb
    bp = (depth+1)*eval_gegenbauer(level-2, (depth+3)/2, t) if level >= 2 else 0.
    factor = np.sqrt(rank*comb(level+depth-1, depth-1))
    matrix = aa*np.eye(rank)-1j*bb*signed
    derivatives = [ap*v*np.eye(rank)-1j*(bp*v*signed-t*bb*a) for v, a in zip(vector, alpha)]
    field = matrix@column/factor
    covariant = [(dm+.5j*matrix@a)@column/factor for dm, a in zip(derivatives, alpha)]
    kinetic = -1j*sum(a@dm for a, dm in zip(alpha, covariant))
    close(kinetic, (level+depth/2)*field, 'actual arbitrary-center spatial kernel satisfies its positive Dirac shell')
    return field, covariant, kinetic


def main():
    rng = np.random.default_rng(202610120600)
    cases = []
    for name, depth, index in radial.ASSIGNMENTS:
        matrices = radial.gammas(depth+1)
        alpha, beta = matrices[:depth], matrices[depth]
        rank, level, h = len(beta), index-1, index-1+depth/2
        largest = max(n for _, d, n in radial.ASSIGNMENTS if d == depth)
        p, derivative, _ = radial.polynomial(depth, level)
        epsilon = .3/float(radial.polynomial(depth, largest-1)[1])
        c = epsilon*float(derivative)
        x = np.sqrt(1-c*c)
        a, b = np.sqrt((1+c)/2), np.sqrt((1-c)/2)
        mass, lam = h*x/c, h/c
        mixing = a*np.eye(rank)+b*beta
        for sample in range(3):
            position = rng.normal(size=depth+1)
            position /= np.linalg.norm(position)
            frame, _ = np.linalg.qr(np.column_stack([position, rng.normal(size=(depth+1, depth))]))
            tangent = frame[:, 1:]
            anchors = []
            for _ in range(2):
                anchor = rng.normal(size=depth+1)
                anchors.append(anchor/np.linalg.norm(anchor))
            column = rng.normal(size=(rank, rank))+1j*rng.normal(size=(rank, rank))
            column /= np.linalg.norm(column)
            first, first_derivative, first_kinetic = shell_jets(depth, level, position, anchors[0], tangent, alpha, np.eye(rank))
            second, second_derivative, second_kinetic = shell_jets(depth, level, position, anchors[1], tangent, alpha, column)
            base, disturbance = mixing@first, mixing@second
            base_derivative = [mixing@v for v in first_derivative]
            disturbance_derivative = [mixing@v for v in second_derivative]
            # beta is parallel and anticommutes with the spatial Dirac operator.
            kinetic = a*second_kinetic-b*beta@second_kinetic
            close((kinetic+mass*beta@disturbance)/lam, disturbance,
                  'arbitrary positive-shell variations lie in the actual massive co-rotating zero shell')
            actual = np.array([2*np.vdot(base, al@disturbance).real for al in alpha])
            bare = np.array([2*np.vdot(first, al@second).real for al in alpha])
            close(actual, c*bare, 'selected-shell vector vertex is the bare spatial vertex times c')
            divergence = 2*sum(np.vdot(db, al@disturbance)+np.vdot(base, al@dv)
                               for db, dv, al in zip(base_derivative, disturbance_derivative, alpha)).real
            scale = max(1., sum(np.linalg.norm(v) for v in base_derivative)*np.linalg.norm(disturbance),
                        sum(np.linalg.norm(v) for v in disturbance_derivative)*np.linalg.norm(base))
            close(divergence/scale, 0., 'every selected-shell linear current is divergence free in the actual spatial geometry')
        cases.append(dict(particle=name, depth=depth, inferred_index=index, spatial_level=level,
                          diagnostic_c=c, epsilon=epsilon,
                          classification='actual two-center spatial shell currents; no persistence or particle-property identification'))
    degree = sp.Symbol('L', integer=True, positive=True)
    static = []
    for depth in radial.DEPTHS:
        eigenvalue = (degree+1)*(degree+depth-2)
        block = sp.Matrix([[1-depth, 1], [eigenvalue, -2]])
        response = sp.factor(block.inv()[0, 0])
        assert sp.cancel(response-2/((degree-1)*(degree+depth))) == 0
        static.append(dict(depth=depth, background_coexact_static_response=str(response),
                           first_harmonic_determinant=str(block.det().subs(degree, 1)),
                           classification='positive static response for every integer L>=2; L=1 remains resonant'))
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    output = ROOT/('covariant-clock-excited-shell-results-'+stamp+'.json')
    report = dict(experiment_key='covariant-clock-assigned-shell-current-geometry',
                  created_utc=datetime.now(timezone.utc).isoformat(),
                  sources={'experiments/'+p.name: sha256(p.read_bytes()).hexdigest()
                           for p in [Path(__file__), ROOT/'inverse-current-radial.py']},
                  evaluations=sum(COUNTS.values()), conditions=COUNTS, maximum_errors=ERRORS,
                  cases=cases, static_background_cases=static,
                  limits=['measured masses precede the inferred assignments being tested',
                          'epsilon is one diagnostic sector value, not an independently fitted particle amplitude',
                          'zero-shell current geometry does not establish full spatial persistence',
                          'the first background coexact harmonic has a pole and must remain dynamically retained',
                          'static elimination on a compatibility subspace does not prove that subspace is dynamically invariant'])
    output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(report=str(output), evaluations=report['evaluations'], maximum_errors=ERRORS)), flush=True)


if __name__ == '__main__':
    main()
