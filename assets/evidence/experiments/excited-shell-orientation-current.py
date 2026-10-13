"""Charge-horizontal conjugate orientation currents on every inferred shell."""
from datetime import datetime, timezone
from hashlib import sha256
from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path
from math import comb
import json

import numpy as np
from scipy.special import eval_gegenbauer, roots_jacobi
import sympy as sp

ROOT = Path(__file__).resolve().parent
spec = spec_from_file_location('radial', ROOT/'inverse-current-radial.py')
radial = module_from_spec(spec)
spec.loader.exec_module(radial)
COUNTS, ERRORS = {}, {}


def close(actual, expected, label, tolerance=2e-8):
    error = float(np.linalg.norm(np.asarray(actual)-np.asarray(expected))/max(1., np.linalg.norm(expected)))
    ERRORS[label] = max(ERRORS.get(label, 0.), error)
    if error >= tolerance:
        raise AssertionError((label, error))
    COUNTS[label] = COUNTS.get(label, 0)+1


def polynomials(depth, level, t):
    aa = (level+depth-1)/(depth-1)*eval_gegenbauer(level, (depth-1)/2, t)
    bb = eval_gegenbauer(level-1, (depth+1)/2, t) if level else np.zeros_like(t)
    ap = (level+depth-1)*bb
    bp = (depth+1)*eval_gegenbauer(level-2, (depth+3)/2, t) if level >= 2 else np.zeros_like(t)
    return aa, bb, ap, bp


def current_profile(depth, level, t):
    aa, bb, ap, bp = polynomials(depth, level, t)
    ss = 1-t*t
    normal = comb(level+depth-1, depth-1)
    fq = t*(aa*aa-bb*bb*ss)-2*aa*bb*ss
    gq = -aa*aa-bb*bb*(1+t*t)-2*aa*bb*t
    ff = 2*(aa*bb-level/depth*fq)/normal
    gg = 2*(-bb*ap+aa*bp-level/depth*gq)/normal
    return ff, gg


def integrate_case(depth, level, extra):
    t, weights = roots_jacobi(2*level+12+extra, (depth-2)/2, (depth-2)/2)
    weights /= sum(weights)
    ss = 1-t*t
    aa, bb, ap, bp = polynomials(depth, level, t)
    normal = comb(level+depth-1, depth-1)
    close(np.dot(weights, (aa*aa+bb*bb*ss)/normal), 1., 'actual positive kernel has the stated canonical integrated norm')
    berry = np.dot(weights, (-(bb*ap-aa*bp)*ss/depth-aa*bb*t+bb*bb*ss*(depth-1)/depth)/normal)
    close(berry, -level/depth, 'integrated actual Clifford contraction independently verifies the matrix-charge Berry coefficient')
    raw = np.dot(weights, (ss/depth*(ap*ap+bp*bp*ss-2*t*bb*bp)+bb*bb*(1-ss/depth))/normal)
    metric = level*(depth-1)*(level+depth)/depth**2
    close(raw-level**2/depth**2, metric, 'horizontal orientation metric agrees with the exact reproducing-kernel derivative')
    ff, gg = current_profile(depth, level, t)
    def inner(f, g, u, v):
        return np.dot(weights, f*u*(1-ss/depth)-(f*v+g*u)*t*ss/depth+g*v*ss*ss/depth)
    norm = inner(ff, gg, ff, gg)
    rebuilt_f, rebuilt_g = np.zeros_like(t), np.zeros_like(t)
    modes = []
    for harmonic in range(1, 2*level+2, 2):
        phi = eval_gegenbauer(harmonic-1, (depth+1)/2, t)
        prime = (depth+1)*eval_gegenbauer(harmonic-2, (depth+3)/2, t) if harmonic >= 2 else np.zeros_like(t)
        vf = -ss*prime+(depth-1)*t*phi
        vg = -t*prime-(depth-1)*phi
        vn = inner(vf, vg, vf, vg)
        exact_vn = sp.Rational((harmonic+1)*(harmonic+depth-2)*(depth-1)*comb(harmonic+depth-1, depth), 2*harmonic+depth-1)
        close(vn, float(exact_vn), 'coexact basis norm matches the exact closed two-form normalization', 3e-8)
        coefficient = inner(ff, gg, vf, vg)/vn
        power = coefficient*coefficient*vn
        rebuilt_f += coefficient*vf
        rebuilt_g += coefficient*vg
        modes.append(dict(harmonic=harmonic, current_norm_squared=float(power), coefficient=float(coefficient), basis_norm_squared=float(vn)))
    reconstruction_error = inner(ff-rebuilt_f, gg-rebuilt_g, ff-rebuilt_f, gg-rebuilt_g)
    close(reconstruction_error/max(norm, 1.), 0., 'odd coexact polar harmonics reconstruct the full conjugate orientation current')
    close(sum(m['current_norm_squared'] for m in modes), norm, 'orthogonal coexact current powers exhaust the actual current norm')
    signed = sum((-(depth-1)/(depth+1)**2 if m['harmonic'] == 1 else 2/((m['harmonic']-1)*(m['harmonic']+depth)))*m['current_norm_squared'] for m in modes)
    return dict(metric=float(metric), current_norm_squared=float(norm), static_response_form=float(-signed),
                response_per_horizontal_norm=float(-signed/metric) if metric else None,
                reconstruction_error=float(reconstruction_error), harmonic_powers=modes)


def exact_response(depth, level):
    t = sp.Symbol('t')
    aa = sp.Poly(sp.Rational(level+depth-1, depth-1)*sp.gegenbauer(level, sp.Rational(depth-1, 2), t), t, domain=sp.QQ)
    bb = sp.Poly(sp.gegenbauer(level-1, sp.Rational(depth+1, 2), t) if level else 0, t, domain=sp.QQ)
    tt = sp.Poly(t, t, domain=sp.QQ)
    normal = comb(level+depth-1, depth-1)
    gg = (-bb*aa.diff()+aa*bb.diff()+sp.Rational(level, depth)*(aa*aa+bb*bb*(1+tt*tt)+2*aa*bb*tt))*sp.Rational(2, normal)
    # A divergence-free polar current is delta(phi dt wedge dw).
    # Its w grad(t) coefficient is -t phi'-(d-1)phi.
    remaining = sp.Poly.from_dict({(power,): -coefficient/(power+depth-1)
                                  for (power,), coefficient in gg.terms()}, t, domain=sp.QQ)
    modes, signed = [], sp.S.Zero
    for harmonic in range(2*level+1, 0, -2):
        basis = sp.Poly(sp.gegenbauer(harmonic-1, sp.Rational(depth+1, 2), t), t, domain=sp.QQ)
        coefficient = remaining.nth(harmonic-1)/basis.LC()
        remaining -= coefficient*basis
        norm = sp.Rational((harmonic+1)*(harmonic+depth-2)*(depth-1)*comb(harmonic+depth-1, depth), 2*harmonic+depth-1)
        power = coefficient**2*norm
        response = -sp.Rational(depth-1, (depth+1)**2) if harmonic == 1 else sp.Rational(2, (harmonic-1)*(harmonic+depth))
        signed -= response*power
        modes.append(dict(harmonic=harmonic, coefficient_exact=str(coefficient), current_norm_squared_exact=str(power)))
    assert remaining.is_zero
    return dict(static_response_form_exact=str(signed), harmonic_powers_exact=list(reversed(modes))), signed


def main():
    rng = np.random.default_rng(202610120639)
    cases = []
    t, d, value = sp.symbols('t d value')
    phi = sp.Function('phi')(t)
    ff = -(1-t*t)*sp.diff(phi, t)+(d-1)*t*phi
    gg = -t*sp.diff(phi, t)-(d-1)*phi
    divergence = -t*sp.diff(ff, t)-d*ff+(1-t*t)*sp.diff(gg, t)-(d+1)*t*gg
    assert sp.simplify(divergence) == 0
    # Reduce the Hodge equation by the Gegenbauer equation and its derivative.
    ode = ((d+2)*t*sp.diff(phi, t)-value*phi)/(1-t*t)
    ode3 = sp.diff(ode, t).subs(sp.diff(phi, t, 2), ode)
    hf = -(1-t*t)*sp.diff(ff, t, 2)+d*t*sp.diff(ff, t)+d*ff+2*t*gg
    hg = -(1-t*t)*sp.diff(gg, t, 2)+(d+4)*t*sp.diff(gg, t)+2*d*gg+2*sp.diff(ff, t)
    for residual in [hf-(value+2*(d-1))*ff, hg-(value+2*(d-1))*gg]:
        assert sp.simplify(residual.subs(sp.diff(phi, t, 3), ode3).subs(sp.diff(phi, t, 2), ode)) == 0
    for name, depth, index in radial.ASSIGNMENTS:
        level = index-1
        alpha = radial.gammas(depth+1)[:depth]
        rank = len(alpha[0])
        normal = comb(level+depth-1, depth-1)
        for sample in range(3):
            position = rng.normal(size=depth+1); position /= np.linalg.norm(position)
            frame, _ = np.linalg.qr(np.column_stack([position, rng.normal(size=(depth+1, depth))]))
            tangent = frame[:, 1:]
            anchor, direction = np.eye(depth+1)[0], np.eye(depth+1)[1]
            tv, wv = position@anchor, position@direction
            vv, zz = tangent.T@anchor, tangent.T@direction
            rotation = np.outer(direction, anchor)-np.outer(anchor, direction)
            rv = tangent.T@rotation@position
            derivative = (tangent.T@rotation@tangent).T
            av = sum(v*a for v, a in zip(vv, alpha))
            az = sum(v*a for v, a in zip(zz, alpha))
            ar = sum(v*a for v, a in zip(rv, alpha))
            two = sum(alpha[i]@alpha[j]*derivative[i, j] for i in range(depth) for j in range(depth))
            right = ar-.5j*two
            aa, bb, ap, bp = polynomials(depth, level, np.array(tv))
            matrix = aa*np.eye(rank)-1j*bb*av
            center_derivative = ap*wv*np.eye(rank)-1j*(bp*wv*av+bb*az)
            horizontal = center_derivative+1j*level/depth*matrix@right
            actual = np.array([2*np.vdot(matrix, al@(1j*horizontal)).real/(rank*normal) for al in alpha])
            f, g = current_profile(depth, level, np.array(tv))
            expected = f*zz+g*wv*vv
            close(actual, expected, 'actual full Clifford matrices verify the charge-horizontal conjugate orientation current')
            current_orientation = np.array([2*np.vdot(matrix, al@horizontal).real/(rank*normal) for al in alpha])
            close(current_orientation, np.zeros(depth), 'the actual orientation tangent carries no linear vector current')
        first = integrate_case(depth, level, 0)
        second = integrate_case(depth, level, 20)
        close(first['static_response_form'], second['static_response_form'], 'independent higher quadrature verifies the signed full coexact response', 3e-7)
        exact, exact_form = exact_response(depth, level)
        close(second['static_response_form'], float(exact_form), 'exact rational harmonic decomposition independently verifies the full signed response', 3e-7)
        cases.append(dict(particle=name, depth=depth, inferred_index=index, spatial_level=level,
                          bare_current_geometry=second, exact_response=exact,
                          classification='one charge-horizontal conjugate orientation direction; not the full zero shell'))
        print(json.dumps(dict(completed=name, depth=depth, signed_response=float(exact_form))), flush=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    report = dict(experiment_key='inferred-shell-horizontal-orientation-current',
                  created_utc=datetime.now(timezone.utc).isoformat(),
                  sources={'experiments/'+p.name: sha256(p.read_bytes()).hexdigest()
                           for p in [Path(__file__), ROOT/'inverse-current-radial.py']},
                  evaluations=sum(COUNTS.values()), conditions=COUNTS, maximum_errors=ERRORS, cases=cases,
                  limits=['a selected-shell massive current has an additional factor c_l',
                          'the displayed response omits the positive eta_l squared over rho_B normalization',
                          'this is a conjugate orientation direction, not a complete excited-shell stability theorem',
                          'matrix-charge horizontality uses the stated exact reproducing-kernel identity',
                          'no new independent empirical sector preference follows from a current profile'])
    output = ROOT/('excited-shell-orientation-current-results-'+stamp+'.json')
    output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(report=str(output), evaluations=report['evaluations'], maximum_errors=ERRORS,
                         responses=[dict(particle=c['particle'], depth=c['depth'], form=c['bare_current_geometry']['static_response_form']) for c in cases])), flush=True)


if __name__ == '__main__':
    main()
