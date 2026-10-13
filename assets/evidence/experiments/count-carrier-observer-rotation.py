"""Full spinorial Lie derivatives of the actual zonal count fields."""
from datetime import datetime, timezone
from hashlib import sha256
from importlib.util import spec_from_file_location, module_from_spec
from itertools import combinations
from math import comb
from pathlib import Path
import json

import numpy as np

ROOT = Path(__file__).resolve().parent
spec = spec_from_file_location('carrier', ROOT/'minimal-count-carrier.py')
carrier = module_from_spec(spec)
spec.loader.exec_module(carrier)
COUNTS, ERRORS = {}, {}


def close(actual, expected, label, tolerance=2e-10):
    error = float(np.linalg.norm(np.asarray(actual)-np.asarray(expected))/max(1., np.linalg.norm(expected)))
    if error >= tolerance:
        raise AssertionError((label, error))
    COUNTS[label] = COUNTS.get(label, 0)+1
    ERRORS[label] = max(ERRORS.get(label, 0.), error)


def main():
    rng = np.random.default_rng(202610120905)
    prepared = {d: carrier.preparation(d) for d in [2, 3, 4, 5, 6, 10]}
    cases = []
    for name, depth, index in carrier.forward.radial.ASSIGNMENTS:
        alpha, beta, ambient, seed, phi = prepared[depth]
        eye = np.eye(len(beta))
        level = index-1
        h = level+depth/2
        first = carrier.forward.radial.polynomial(depth, level)[1]
        largest = max(carrier.forward.radial.polynomial(depth, n-1)[1]
                      for _, d, n in carrier.forward.radial.ASSIGNMENTS if d == depth)
        c = float(.3*first/largest)
        mixing = np.sqrt((1+c)/2)*eye+np.sqrt((1-c)/2)*beta
        normal = np.sqrt(comb(level+depth-1, depth-1))
        observer = list(combinations(range(3 if depth >= 3 else 2), 2))
        generators = [-.5j*alpha[a]@alpha[b] for a, b in observer]
        casimir = sum((g@g for g in generators), np.zeros_like(eye, dtype=complex))
        close(casimir, (.75 if depth >= 3 else .25)*eye,
              'the center-fixing observer rotation Casimir is three quarters for matter depths and one quarter for the depth-two circle')
        for generator in generators:
            close(np.vdot(seed, generator@seed), 0.,
                  'balanced seed has zero mean observer generator despite a nonzero representation Casimir')
        for sample in range(5):
            position = rng.normal(size=depth+1)
            position /= np.linalg.norm(position)
            t = position[0]
            sine = np.linalg.norm(position[1:])
            direction = position[1:]/sine
            frame_rotation = np.eye(depth+1)
            frame_rotation[0, 0] = t
            frame_rotation[0, 1:] = -sine*direction
            frame_rotation[1:, 0] = sine*direction
            frame_rotation[1:, 1:] += (t-1)*np.outer(direction, direction)
            frame = frame_rotation[:, 1:]
            close(frame_rotation[:, 0], position, 'the physical radial frame matches its sphere position')
            killing = np.sqrt((1+t)/2)*eye+1j*np.sqrt((1-t)/2)*sum(ui*al for ui, al in zip(direction, alpha))
            close(killing.conj().T@beta@killing,
                  sum(xi*ga for xi, ga in zip(position, ambient)),
                  'the actual Killing frame maps its grading direction to the sphere normal')
            v = frame[0, :]
            av = sum(vi*al for vi, al in zip(v, alpha))
            aa, bb, ap, bp = carrier.forward.orientation.polynomials(depth, level, np.array(t))
            matrix = aa*eye-1j*bb*av
            bare = matrix@killing@seed/normal
            jets = [(ap*v[i]*eye-1j*(bp*v[i]*av-t*bb*al)+.5j*matrix@al)@killing@seed/normal
                    for i, al in enumerate(alpha)]
            close(-1j*sum(al@jet for al, jet in zip(alpha, jets)), h*bare,
                  'the actual globally related Killing value and covariant jets remain in the intended shell')
            for (a, b), generator in zip(observer, generators):
                rotation = np.zeros((depth+1, depth+1))
                rotation[b+1, a+1] = 1
                rotation[a+1, b+1] = -1
                velocity = frame.T@rotation@position
                gradient = frame.T@rotation@frame
                covariant_transport = sum(vi*jet for vi, jet in zip(velocity, jets))
                spin_term = sum(gradient[j, i]*alpha[i]@alpha[j] for i in range(depth) for j in range(depth))/4
                full_rotation = 1j*(-covariant_transport-spin_term@bare)
                right_rotation = matrix@killing@generator@seed/normal
                close(full_rotation, right_rotation,
                      'the full spinorial Lie derivative including orbital transport intertwines exactly with the seed rotation')
                close(mixing@full_rotation, mixing@right_rotation,
                      'the same full observer rotation intertwining survives the massive count map')
                close(np.linalg.norm(covariant_transport+spin_term@bare), np.linalg.norm(right_rotation),
                      'orbital and spin connection terms are both retained in the total rotation')
        cases.append(dict(particle=name, depth=depth, inferred_index=index,
                          observer_group='center-fixing Spin(3)' if depth >= 3 else 'center-fixing Spin(2)',
                          observer_casimir='3/4' if depth >= 3 else '1/4',
                          mean_noether_rotation='0 under the balanced seed',
                          classification='exact full geometric transformation law, not a derived observed classical spin charge'))
        print(json.dumps(dict(completed=name, depth=depth)), flush=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    sources = [Path(__file__), ROOT/'minimal-count-carrier.py', ROOT/'forward-count-action.py',
               ROOT/'inverse-current-radial.py', ROOT/'excited-shell-orientation-current.py']
    report = dict(experiment_key='count-carrier-observer-rotation', created_utc=datetime.now(timezone.utc).isoformat(),
                  sources={'experiments/'+p.name: sha256(p.read_bytes()).hexdigest() for p in sources},
                  evaluations=sum(COUNTS.values()), conditions=COUNTS, maximum_errors=ERRORS, cases=cases,
                  limits=['the observer rotates three tangent directions about the zonal kernel center at matter depths',
                          'spinor columns transform as scalar-labelled spinors; another column lift changes the representation premise',
                          'Casimir three quarters is an exact transformation property, not a nonzero classical spin Noether charge',
                          'depth two has only a center-fixing rotation circle, so this does not derive photon or massive boson spin',
                          'literal deterministic electron orbit motion is not replaced by this internal field transformation'])
    path = ROOT/('count-carrier-observer-rotation-results-'+stamp+'.json')
    path.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(report=str(path), evaluations=report['evaluations'], maximum_errors=ERRORS)), flush=True)


if __name__ == '__main__':
    main()
