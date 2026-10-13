"""Closure-first finite spatial response of the minimal count carriers."""
from datetime import datetime, timezone
from hashlib import sha256
from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path
import json

import numpy as np

ROOT = Path(__file__).resolve().parent
spec = spec_from_file_location('carrier', ROOT/'minimal-count-carrier.py')
carrier = module_from_spec(spec)
spec.loader.exec_module(carrier)


def at_point(position, gammas, seed, h, mass, lam, kappa, a, b):
    gx = sum(xi*gamma for xi, gamma in zip(position, gammas))
    raw = [seed, gx@seed]
    dirac = [h*seed, -h*gx@seed]
    for xi, gamma in zip(position, gammas):
        raw.extend([xi*seed, xi*gx@seed, gamma@seed, gx@gamma@seed])
        dirac.extend([(1+h)*xi*seed-gx@gamma@seed,
                      -(1+h)*xi*gx@seed+gamma@seed, h*gamma@seed, -h*gx@gamma@seed])
    volume = np.eye(len(gx), dtype=complex)
    for gamma in gammas:
        volume = volume@gamma
    # The same volume completion works at every occupied depth; odd generator
    # counts make it redundant, which the numerical rank calculation detects.
    raw += [volume@field for field in raw.copy()]
    dirac += [volume@field for field in dirac.copy()]
    fields = raw+[1j*field for field in raw]
    derivatives = dirac+[1j*field for field in dirac]
    background = (a*np.eye(len(gx))+b*gx)@seed
    current_vectors = [1j*(gamma@gx-xi*np.eye(len(gx)))@background for xi, gamma in zip(position, gammas)]
    output = []
    for field, derivative in zip(fields, derivatives):
        current = [2*np.vdot(vector, field).real for vector in current_vectors]
        source = kappa*sum(ji*vector for ji, vector in zip(current, current_vectors))
        output.append(-1j*(derivative+mass*gx@field-lam*field+source))
    def real_columns(values):
        matrix = np.column_stack([value.ravel() for value in values])
        return np.vstack([matrix.real, matrix.imag])
    return real_columns(fields), real_columns(output)


def main():
    rng = np.random.default_rng(202610120925)
    cases = []
    for depth in [2, 3, 4, 5, 6, 10]:
        alpha, beta, gammas, seed, phi = carrier.preparation(depth)
        h, c, x = depth/2, .6, .8
        eps = c/sum(1/j for j in range(1, depth+1))
        mass, lam, kappa = h*x/c, h/c, h/c-eps
        a, b = 2/np.sqrt(5), 1/np.sqrt(5)
        parameters = (gammas, seed, h, mass, lam, kappa, a, b)
        points = rng.normal(size=(28, depth+1)); points /= np.linalg.norm(points, axis=1)[:, None]
        data = [at_point(point, *parameters) for point in points]
        fields = np.vstack([item[0] for item in data])
        output = np.vstack([item[1] for item in data])
        u, singular, vt = np.linalg.svd(fields, full_matrices=False)
        rank = int(np.sum(singular > singular[0]*1e-10))
        transform = vt[:rank].T/singular[:rank]
        basis, action = u[:, :rank], output@transform
        matrix = basis.T@action
        training = np.linalg.norm(action-basis@matrix)/max(1., np.linalg.norm(action))
        independent = rng.normal(size=(12, depth+1)); independent /= np.linalg.norm(independent, axis=1)[:, None]
        data = [at_point(point, *parameters) for point in independent]
        extra_fields = np.vstack([item[0] for item in data])@transform
        extra_output = np.vstack([item[1] for item in data])@transform
        verification = np.linalg.norm(extra_output-extra_fields@matrix)/max(1., np.linalg.norm(extra_output))
        # Eigenvalues have physical meaning here only after independent closure.
        closed = training < 1e-9 and verification < 1e-9
        eigenvalues = np.linalg.eigvals(matrix) if closed else []
        cases.append(dict(depth=depth, minimal_columns=seed.shape[1], candidate_real_rank=rank,
                          training_closure_residual=float(training), independent_closure_residual=float(verification),
                          numerical_invariant_space_verified=bool(closed),
                          time_generator_eigenvalues=[dict(real=float(z.real), imaginary=float(z.imag)) for z in eigenvalues],
                          maximum_growth_rate=float(max(z.real for z in eigenvalues)) if closed else None,
                          classification='numerically verified finite ground spatial family' if closed else
                          'candidate family leaks; projected eigenvalues are not reported'))
        print(json.dumps(cases[-1]), flush=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    paths = [Path(__file__), ROOT/'minimal-count-carrier.py', ROOT/'forward-count-action.py',
             ROOT/'inverse-current-radial.py', ROOT/'excited-shell-orientation-current.py']
    report = dict(experiment_key='minimal-carrier-ground-proper-current-response', created_utc=datetime.now(timezone.utc).isoformat(),
                  sources={'experiments/'+p.name: sha256(p.read_bytes()).hexdigest() for p in paths}, cases=cases,
                  limits=['the operator uses the affine proper-current chart with c3/5 and epsilon=c/Hd',
                          'only independent numerical closure is claimed at this stage; exact coefficient proof remains separate',
                          'these are ground controls, not assigned excited particle states',
                          'a closed-family growing eigenvalue would disprove linear persistence of that preparation',
                          'absence of growth in this family would not establish complete angular or nonlinear persistence'])
    path = ROOT/('minimal-carrier-ground-response-results-'+stamp+'.json')
    path.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(report=str(path))), flush=True)


if __name__ == '__main__':
    main()
