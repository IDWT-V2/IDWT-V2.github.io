"""The exact dynamical background current response, including its zero shell."""
from datetime import datetime, timezone
from hashlib import sha256
from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path
import json

import numpy as np
from scipy.integrate import solve_ivp
import sympy as sp

ROOT = Path(__file__).resolve().parent
spec = spec_from_file_location('radial', ROOT/'inverse-current-radial.py')
radial = module_from_spec(spec)
spec.loader.exec_module(radial)
COUNTS, ERRORS = {}, {}


def close(actual, expected, label, tolerance=2e-9):
    error = float(np.linalg.norm(np.asarray(actual)-np.asarray(expected))/max(1., np.linalg.norm(expected)))
    ERRORS[label] = max(ERRORS.get(label, 0.), error)
    if error >= tolerance:
        raise AssertionError((label, error))
    COUNTS[label] = COUNTS.get(label, 0)+1


def main():
    degree, z = sp.symbols('L z')
    cases = []
    for depth in radial.DEPTHS:
        eigenvalue = (degree+1)*(degree+depth-2)
        block = sp.Matrix([[1-depth, 1], [eigenvalue, -2]])
        exact = sp.factor((block*(block*block-z*sp.eye(2)).inv())[0, 0])
        delta = (degree-1)*(degree+depth)
        expected = (2*delta+(depth-1)*z)/(((degree-1)**2-z)*((degree+depth)**2-z))
        assert sp.cancel(exact-expected) == 0
        first = sp.factor(exact.subs(degree, 1))
        assert sp.cancel(first-(depth-1)/(z-(depth+1)**2)) == 0
        static = sp.factor(first.subs(z, 0))
        physical = np.array([[1-depth, np.sqrt(2*(depth-1))],
                             [np.sqrt(2*(depth-1)), -2.]])
        source = np.array([1., 0.])
        eigenvalues, vectors = np.linalg.eigh(physical)
        zero_vector = vectors[:, np.argmin(abs(eigenvalues))]
        negative = vectors[:, np.argmin(eigenvalues)]
        projector = np.outer(zero_vector, zero_vector)
        pseudoinverse = -np.outer(negative, negative)/(depth+1)
        close(physical@zero_vector, np.zeros(2), 'actual first coexact block retains its background zero mode')
        close(source@pseudoinverse@source, float(static), 'charge-fixed zero-frequency current response equals the negative-branch pseudoinverse')
        # A real current source drives u. The measured vector current is 2v_1.
        # The zero mode of u is an actual right-unitary tangent, while its
        # conjugate v coordinate is the conserved matrix-charge variation.
        times = np.linspace(0., 4., 241)
        for forcing in ['constant', 'cosine']:
            nu = .7
            def force(t):
                return 1. if forcing == 'constant' else np.cos(nu*t)
            def rhs(t, state):
                u, v = state[:2], state[2:]
                return np.concatenate([physical@v-source*force(t), -physical@u])
            solution = solve_ivp(rhs, [0., times[-1]], np.zeros(4), t_eval=times,
                                 rtol=2e-12, atol=2e-13)
            assert solution.success
            u, v = solution.y[:2], solution.y[2:]
            close(zero_vector@v, np.zeros(len(times)), 'the conserved zero-shell charge coordinate has no driven current response')
            integral = times if forcing == 'constant' else np.sin(nu*times)/nu
            close(zero_vector@u, -np.dot(zero_vector, source)*integral,
                  'the complete zero-shell field response is retained as a right-unitary drift')
            amp = -(depth-1)/(depth+1)**2
            if forcing == 'constant':
                predicted = 2*amp*(1-np.cos((depth+1)*times))
            else:
                response = (depth-1)/(nu*nu-(depth+1)**2)
                predicted = 2*response*(np.cos(nu*times)-np.cos((depth+1)*times))
            close(2*v[0], predicted, 'independent complete time evolution verifies the causal first-harmonic current response', 2e-8)
        eta = sp.Rational(3, 5)/sum(sp.Rational(1, j) for j in range(1, depth+1))-sp.Rational(5*depth, 6)
        dressed = []
        for load in [sp.Rational(1, 100), sp.Rational(1, 5)]:
            susceptibility = sp.factor(first/(1+2*eta*load*first))
            frequency_squared = (depth+1)**2-2*eta*load*(depth-1)
            assert sp.cancel(susceptibility-(depth-1)/(z-frequency_squared)) == 0
            def rhs_dressed(t, state):
                u, v = state[:2], state[2:]
                return np.concatenate([physical@v-source+2*float(eta*load)*source*v[0], -physical@u])
            solution = solve_ivp(rhs_dressed, [0., times[-1]], np.zeros(4), t_eval=times,
                                 rtol=2e-12, atol=2e-13)
            assert solution.success
            u, v = solution.y[:2], solution.y[2:]
            close(zero_vector@v, np.zeros(len(times)), 'retained background self-current feedback still conserves the zero-shell charge coordinate')
            frequency = np.sqrt(float(frequency_squared))
            predicted = -2*(depth-1)/float(frequency_squared)*(1-np.cos(frequency*times))
            close(2*v[0], predicted, 'full background self-current dynamics verifies the resummed first-harmonic susceptibility', 2e-8)
            close(physical@projector, np.zeros((2, 2)), 'background self-current feedback does not remove the retained free zero-shell field direction')
            dressed.append(dict(load_exact=str(load), ground_eta_exact=str(eta),
                                squared_response_frequency_exact=str(frequency_squared),
                                susceptibility=str(susceptibility)))
        cases.append(dict(depth=depth, all_harmonic_current_susceptibility=str(exact),
                          first_harmonic_current_susceptibility=str(first),
                          first_harmonic_static_response_exact=str(static),
                          absolute_static_response=float(-static),
                          ground_self_current_responses=dressed,
                          classification='retained zero field mode and finite charge-fixed current response'))
    d = sp.Symbol('d', positive=True)
    difference = sp.factor(sp.Rational(1, 8)-(d-1)/(d+1)**2)
    assert sp.cancel(difference-(d-3)**2/(8*(d+1)**2)) == 0
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    report = dict(experiment_key='covariant-clock-dynamical-current-kernel',
                  created_utc=datetime.now(timezone.utc).isoformat(),
                  sources={'experiments/'+p.name: sha256(p.read_bytes()).hexdigest()
                           for p in [Path(__file__), ROOT/'inverse-current-radial.py']},
                  evaluations=sum(COUNTS.values()), conditions=COUNTS, maximum_errors=ERRORS, cases=cases,
                  response_bound_difference=str(difference),
                  limits=['free positive Killing background and first-order current response',
                          'a fixed full matrix-charge variation removes the conjugate zero-current preparation',
                          'the secular field response is retained rather than discarded',
                          'a dimension-dependent response maximum is not a color, charge or force derivation',
                          'coupled excited-state current dynamics and persistence remain open'])
    output = ROOT/('covariant-clock-current-kernel-results-'+stamp+'.json')
    output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(report=str(output), evaluations=report['evaluations'], maximum_errors=ERRORS)), flush=True)


if __name__ == '__main__':
    main()
