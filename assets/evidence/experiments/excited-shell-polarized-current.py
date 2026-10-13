"""A bounded polarized zero-shell current map beyond orientation alone."""
from datetime import datetime, timezone
from hashlib import sha256
from importlib.util import spec_from_file_location, module_from_spec
from itertools import combinations
from pathlib import Path
import json

import numpy as np
from scipy.special import roots_jacobi
import sympy as sp

ROOT = Path(__file__).resolve().parent
spec = spec_from_file_location('orientation', ROOT/'excited-shell-orientation-current.py')
orientation = module_from_spec(spec)
spec.loader.exec_module(orientation)
radial = orientation.radial
GAMMA = np.asarray(radial.gammas(4)[:3])
LABELS = [axis+' '+matrix+' '+phase for axis in ['a', 'b']
          for matrix in ['I', 'alpha_a', 'alpha_b', 'i alpha_a alpha_b']
          for phase in ['real', 'imaginary']]
COUNTS, ERRORS = {}, {}


def close(actual, expected, label, tolerance=3e-8):
    error = float(np.linalg.norm(np.asarray(actual)-np.asarray(expected))/max(1., np.linalg.norm(expected)))
    ERRORS[label] = max(ERRORS.get(label, 0.), error)
    if error >= tolerance:
        raise AssertionError((label, error))
    COUNTS[label] = COUNTS.get(label, 0)+1


def rule(order, exponent):
    values, weights = roots_jacobi(order, exponent, exponent)
    return values, weights/sum(weights)


def calculate(depth, level, angular_order, extra):
    from math import comb
    tv, tw = rule(2*level+12+extra, (depth-2)/2)
    uv, uw = rule(angular_order, (depth-3)/2)
    if depth == 2:
        xv, xw = np.array([-1., 1.]), np.array([.5, .5])
    else:
        xv, xw = rule(angular_order, (depth-4)/2)
    aa, bb, ap, bp = orientation.polynomials(depth, level, tv)
    normal = comb(level+depth-1, depth-1)
    gram = np.zeros((16, 16), complex)
    currents = np.zeros((16, 16))
    first = np.zeros((3, 16))
    maximum_algebra_error = 0.
    maximum_profile_error = 0.
    for t, wt, a, b, a_prime, b_prime in zip(tv, tw, aa, bb, ap, bp):
        u, xi = np.meshgrid(uv, xv, indexing='ij')
        angle_weights = np.outer(uw, xw).ravel()
        coords = np.column_stack([np.full(u.size, t), np.sqrt(1-t*t)*u.ravel(),
                                 np.sqrt(1-t*t)*np.sqrt(1-u*u).ravel()*xi.ravel()])
        metric = np.eye(3)[None, :, :]-np.einsum('na,nb->nab', coords, coords)
        values, vectors = np.linalg.eigh(metric)
        root = np.sqrt(np.maximum(values, 0.))[:, :, None]*vectors.transpose(0, 2, 1)
        clifford = np.einsum('nka,kij->naij', root, GAMMA)
        av, az, ay = [clifford[:, i] for i in range(3)]
        w, y = coords[:, 1], coords[:, 2]
        qa = t*az-w[:, None, None]*av-.5j*(av@az-az@av)
        qb = t*ay-y[:, None, None]*av-.5j*(av@ay-ay@av)
        eye = np.broadcast_to(np.eye(4), av.shape)
        maximum_algebra_error = max(maximum_algebra_error, float(np.max(abs(qa@qa-eye))),
                                    float(np.max(abs(qb@qb-eye))), float(np.max(abs(qa@qb+qb@qa))))
        matrix = a*eye-1j*b*av
        da = a_prime*w[:, None, None]*eye-1j*(b_prime*w[:, None, None]*av+b*az)
        db = a_prime*y[:, None, None]*eye-1j*(b_prime*y[:, None, None]*av+b*ay)
        wa = da+1j*level/depth*matrix@qa
        wb = db+1j*level/depth*matrix@qb
        fields = []
        for derivative in [wa, wb]:
            for right in [eye, qa, qb, 1j*qa@qb]:
                field = derivative@right
                fields.extend([field, 1j*field])
        fields = np.stack(fields, axis=1)
        local_gram = np.einsum('naij,nbij->nab', fields.conj(), fields)/(4*normal)
        current = []
        for gamma in GAMMA:
            vertex = np.einsum('ij,najk->naik', gamma, fields)
            current.append(2*np.einsum('nij,naij->na', matrix.conj(), vertex).real/(4*normal))
        current = np.stack(current, axis=2)
        phi = 2*b*(a+t*b)/normal
        prime = 2*(b_prime*(a+t*b)+b*(a_prime+b+t*b_prime))/normal
        expected = ((phi-(t*prime+depth*phi)*w*w)[:, None]*root[:, :, 0]
                    +((- (1-t*t)*prime+depth*t*phi)*w)[:, None]*root[:, :, 1])
        maximum_profile_error = max(maximum_profile_error, float(np.linalg.norm(current[:, 2]-expected)/max(1., np.linalg.norm(expected))))
        gram += wt*np.einsum('n,nab->ab', angle_weights, local_gram)
        currents += wt*np.einsum('n,nak,nbk->ab', angle_weights, current, current)
        for row, (i, j) in enumerate(combinations(range(3), 2)):
            killing = coords[:, i, None]*root[:, :, j]-coords[:, j, None]*root[:, :, i]
            first[row] += wt*np.einsum('n,nak,nk->a', angle_weights, current, killing)
    close(maximum_algebra_error, 0., 'transported right Clifford directions satisfy their exact pointwise algebra')
    close(maximum_profile_error, 0., 'the actual right-polarized shape current equals the exact closed two-form profile')
    close(currents[[0, 8]], np.zeros((2, 16)), 'both real orientation tangents remain in the complete polarized current kernel')
    close(np.diag(gram.real), np.full(16, level*(depth-1)*(level+depth)/depth**2),
          'every unitary right-polarized derivative has the exact horizontal canonical norm')
    first_power = (depth+1)/2*first.T@first
    metric = gram.real
    eigenvalues, eigenvectors = np.linalg.eigh(metric)
    threshold = 1e-9*max(1., max(eigenvalues))
    active = eigenvalues > threshold
    transform = eigenvectors[:, active]/np.sqrt(eigenvalues[active])
    if not np.any(active):
        return dict(field_labels=LABELS, gram=metric.tolist(), symplectic=(2*gram.imag).tolist(),
                    current_gram=currents.tolist(), first_harmonic_moments=first.tolist(),
                    physical_real_rank=0, higher_current_shape_maximum=0., shape_witness=None)
    omega = transform.T@(2*gram.imag)@transform
    close(omega@omega, -4*np.eye(sum(active)), 'complete complex polarized span has a nondegenerate canonical symplectic form')
    # Remove first-harmonic current and both orientation/conjugate pairs.
    restrictions = np.vstack([first@transform, metric[[0, 1, 8, 9]]@transform])
    _, singular, vh = np.linalg.svd(restrictions, full_matrices=True)
    rank = sum(singular > 1e-9*max(1., max(singular)))
    null = vh[rank:].T
    shape_transform = transform@null
    reduced = shape_transform.T@currents@shape_transform
    values, vectors = np.linalg.eigh(reduced)
    maximum = float(max(values)) if len(values) else 0.
    witness = shape_transform@vectors[:, -1] if len(values) else np.zeros(16)
    if maximum > 1e-8:
        close(witness@metric@witness, 1., 'the higher-current shape witness has unit canonical norm')
        close(first@witness, np.zeros(3), 'the shape witness has zero current in every first coexact harmonic')
        close(metric[[0, 1, 8, 9]]@witness, np.zeros(4), 'the shape witness is orthogonal to both orientation and conjugate directions')
    return dict(field_labels=LABELS, gram=metric.tolist(), symplectic=(2*gram.imag).tolist(),
                current_gram=currents.tolist(), first_harmonic_moments=first.tolist(),
                first_harmonic_power=first_power.tolist(), physical_real_rank=int(sum(active)),
                higher_current_shape_maximum=maximum, shape_witness=witness.tolist(),
                current_rank=int(sum(np.linalg.eigvalsh(transform.T@currents@transform)>1e-8)))


def certify_shape_profile():
    t, w, d, n = sp.symbols('t w d n')
    phi = sp.Function('phi')(t)
    vector = [phi-(t*sp.diff(phi, t)+d*phi)*w*w,
              (-(1-t*t)*sp.diff(phi, t)+d*t*phi)*w]
    coordinates = [t, w]
    def laplace(f):
        return sum(((1 if i == j else 0)-x*y)*sp.diff(f, x, y)
                   for i, x in enumerate(coordinates) for j, y in enumerate(coordinates))-d*sum(x*sp.diff(f, x) for x in coordinates)
    divergence = sum(((1 if i == j else 0)-x*y)*sp.diff(vector[i], y)
                     for i, x in enumerate(coordinates) for j, y in enumerate(coordinates))-d*sum(x*f for x, f in zip(coordinates, vector))
    assert sp.simplify(divergence) == 0
    assert sp.simplify(d*phi+(1-t*t)*(-t*sp.diff(phi, t)-d*phi)-t*(-(1-t*t)*sp.diff(phi, t)+d*t*phi)) == 0
    ode = ((d+4)*t*sp.diff(phi, t)-n*(n+d+3)*phi)/(1-t*t)
    third = sp.diff(ode, t).subs(sp.diff(phi, t, 2), ode)
    eigenvalue = (n+3)*(n+d)
    for k, x in enumerate(coordinates):
        actual = -laplace(vector[k])+d*vector[k]+2*sum(y*sp.diff(f, x) for y, f in zip(coordinates, vector))
        residual = (actual-eigenvalue*vector[k]).subs(sp.diff(phi, t, 3), third).subs(sp.diff(phi, t, 2), ode)
        assert sp.simplify(residual) == 0


def exact_shape(depth, level):
    from math import comb
    t = sp.Symbol('t')
    aa = sp.Poly(sp.Rational(level+depth-1, depth-1)*sp.gegenbauer(level, sp.Rational(depth-1, 2), t), t, domain=sp.QQ)
    bb = sp.Poly(sp.gegenbauer(level-1, sp.Rational(depth+1, 2), t) if level else 0, t, domain=sp.QQ)
    remaining = sp.Rational(2, comb(level+depth-1, depth-1))*bb*(aa+sp.Poly(t, t, domain=sp.QQ)*bb)
    modes, norm, response = [], sp.S.Zero, sp.S.Zero
    for harmonic in range(2*level+1, 2, -2):
        basis = sp.Poly(sp.gegenbauer(harmonic-2, sp.Rational(depth+3, 2), t), t, domain=sp.QQ)
        coefficient = remaining.nth(harmonic-2)/basis.LC()
        remaining -= coefficient*basis
        vn = sp.Rational((harmonic+1)*(harmonic+depth-2)*(depth-1)*comb(harmonic+depth, depth+2), (depth+1)*(2*harmonic+depth-1))
        power = coefficient**2*vn
        norm += power
        response -= sp.Rational(2, (harmonic-1)*(harmonic+depth))*power
        modes.append(dict(harmonic=harmonic, coefficient_exact=str(coefficient), current_norm_squared_exact=str(power)))
    assert remaining.is_zero
    assert (response < 0) if level else (response == 0)
    return dict(current_norm_squared_exact=str(norm), static_response_form_exact=str(response),
                harmonic_powers_exact=list(reversed(modes))), norm


def main():
    certify_shape_profile()
    cases = []
    for name, depth, index in radial.ASSIGNMENTS:
        first = calculate(depth, index-1, 6, 0)
        second = calculate(depth, index-1, 8, 8)
        for key in ['gram', 'symplectic', 'current_gram', 'first_harmonic_moments']:
            close(first[key], second[key], 'independent exact-degree angular and radial rules agree on '+key, 3e-7)
        close(first['higher_current_shape_maximum'], second['higher_current_shape_maximum'],
              'independent cubature verifies the higher-current shape direction', 3e-7)
        exact, exact_norm = exact_shape(depth, index-1)
        close(second['current_gram'][2][2], float(exact_norm), 'exact rational shape harmonics verify the actual polarized current norm', 3e-7)
        cases.append(dict(particle=name, depth=depth, inferred_index=index, response=second, exact_shape=exact,
                          classification='finite polarized selected-shell current map, with no dynamics or stability conclusion'))
        print(json.dumps(dict(completed=name, depth=depth, physical_real_rank=second['physical_real_rank'],
                              higher_current_shape_maximum=second['higher_current_shape_maximum'])), flush=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    report = dict(experiment_key='inferred-shell-polarized-current-map',
                  created_utc=datetime.now(timezone.utc).isoformat(),
                  sources={'experiments/'+p.name: sha256(p.read_bytes()).hexdigest()
                           for p in [Path(__file__), ROOT/'inverse-current-radial.py', ROOT/'excited-shell-orientation-current.py']},
                  evaluations=sum(COUNTS.values()), conditions=COUNTS, maximum_errors=ERRORS, cases=cases,
                  limits=['a finite complex current span is not the full degenerate zero shell',
                          'the finite-span ranks and optimal witnesses are numerical; the explicit right-polarized shape profile and its signed response are exact',
                          'the full reciprocal frequency-dependent background and off-shell modes remain retained',
                          'a nonzero higher current with zero first harmonic has negative leading static response, not automatically growing dynamics',
                          'the massive current factor c_l and physical amplitudes remain explicit'])
    output = ROOT/('excited-shell-polarized-current-results-'+stamp+'.json')
    output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(report=str(output), evaluations=report['evaluations'], maximum_errors=ERRORS)), flush=True)


if __name__ == '__main__':
    main()
