"""Exact stationary shape freedom in the responding-clock shell action."""
from datetime import datetime, timezone
from hashlib import sha256
from importlib.util import spec_from_file_location, module_from_spec
from math import comb
from pathlib import Path
import json

import numpy as np
from scipy.special import eval_gegenbauer, roots_jacobi
import sympy as sp

ROOT = Path(__file__).resolve().parent
spec = spec_from_file_location('orientation', ROOT/'excited-shell-orientation-current.py')
orientation = module_from_spec(spec)
spec.loader.exec_module(orientation)
radial = orientation.radial
COUNTS, ERRORS = {}, {}


def close(actual, expected, label, tolerance=3e-9):
    error = float(np.linalg.norm(np.asarray(actual)-np.asarray(expected))/max(1., np.linalg.norm(expected)))
    ERRORS[label] = max(ERRORS.get(label, 0.), error)
    if error >= tolerance:
        raise AssertionError((label, error))
    COUNTS[label] = COUNTS.get(label, 0)+1


def exact_density_identity():
    t, w, d, l, a, b, bp = sp.symbols('t w d l A B Bprime')
    ss, k = 1-t*t, l/d
    ap = (l+d-1)*b
    tt = ap+t*bp+b
    zz = t*(ap+b)-ss*bp
    scalar = a*(b*t-bp*w*w-k*a)-b*((1+k)*b-tt*w*w)*ss+b*t*w*w*zz
    target = (b*ap-a*bp+b*b)*(w*w-ss/d)
    # The positive-shell scalar Dirac equation supplies this relation.
    residual = sp.expand((scalar-target).subs(a, (d*t*b-ss*bp)/l))
    assert sp.factor(residual) == 0
    COUNTS['generic dimension density tangent follows from the exact shell equations'] = 1


def integrated_profiles(depth, level, extra):
    t, weights = roots_jacobi(2*level+12+extra, (depth-2)/2, (depth-2)/2)
    weights /= sum(weights)
    a, b, ap, bp = orientation.polynomials(depth, level, t)
    ss = 1-t*t
    k = level/depth
    normal = comb(level+depth-1, depth-1)
    # Conditional moments of w on the sphere at fixed t.
    w2, w4 = ss/depth, 3*ss*ss/(depth*(depth+2))
    tt = ap+t*bp+b
    zz = t*(ap+b)-ss*bp
    f0, f2 = b*t-k*a, -bp
    v0, v2 = (1+k)*b, -tt
    norm = f0*f0+2*f0*f2*w2+f2*f2*w4
    norm += ss*(v0*v0+2*v0*v2*w2+v2*v2*w4)
    norm += zz*zz*(w2-w4)-2*t*zz*(v0*w2+v2*w4)
    norm += tt*tt*(ss*w2-w4)
    gram = np.dot(weights, norm/normal)
    cross = np.dot(weights, (a*(f0+f2*w2)-b*(ss*(v0+v2*w2)-t*zz*w2))/normal)
    expected = level*(depth-1)*(level+depth)/depth**2
    close(gram, expected, 'independent scalar moment integration verifies the full matrix norm of the shape generator')
    close(cross, 0., 'the shape generator has zero integrated real overlap')
    return dict(generator_norm_squared=float(gram), real_overlap=float(cross))


def main():
    exact_density_identity()
    rng = np.random.default_rng(202610120710)
    epsilon = {}
    for depth in radial.DEPTHS:
        epsilon[depth] = .3/max(float(radial.polynomial(depth, index-1)[1])
                               for _, d, index in radial.ASSIGNMENTS if d == depth)
    cases = []
    for name, depth, index in radial.ASSIGNMENTS:
        level = index-1
        h = level+depth/2
        normal = comb(level+depth-1, depth-1)
        metric = level*(depth-1)*(level+depth)/depth**2
        generators = radial.gammas(depth+1)
        alpha, beta = generators[:depth], generators[depth]
        rank = len(beta)
        eye = np.eye(rank)
        pcount, derivative, _ = radial.polynomial(depth, level)
        c = epsilon[depth]*float(derivative)
        x = np.sqrt(1-c*c)
        mixing_a, mixing_b = np.sqrt((1+c)/2), np.sqrt((1-c)/2)
        source_m, lam = h*x/c, h/c
        omega = epsilon[depth]*float(pcount)
        eta = omega-lam
        for sample in range(3):
            position = rng.normal(size=depth+1); position /= np.linalg.norm(position)
            frame, _ = np.linalg.qr(np.column_stack([position, rng.normal(size=(depth+1, depth))]))
            tangent = frame[:, 1:]
            t, w = position[0], position[1]
            v, z = tangent[0], tangent[1]
            vv = sum(vi*al for vi, al in zip(v, alpha))
            zz = sum(zi*al for zi, al in zip(z, alpha))
            qa = t*zz-w*vv-.5j*(vv@zz-zz@vv)
            close(qa@qa, eye, 'the transported right generator is pointwise Hermitian unitary')
            a, b, ap, bp = orientation.polynomials(depth, level, np.array(t))
            app = (level+depth-1)*bp
            bpp = (depth+1)*(depth+3)*eval_gegenbauer(level-3, (depth+5)/2, t) if level >= 3 else 0.
            matrix = a*eye-1j*b*vv
            da = ap*w*eye-1j*(bp*w*vv+b*zz)
            field = 1j*da@qa-level/depth*matrix
            tt, zc = ap+t*bp+b, t*(ap+b)-(1-t*t)*bp
            expanded = ((b*t-bp*w*w-level/depth*a)*eye
                        +1j*((1+level/depth)*b-tt*w*w)*vv
                        +1j*w*zc*zz+w*tt*.5*(vv@zz-zz@vv))
            close(field, expanded, 'direct Clifford multiplication agrees with the scalar and quaternion expansion')
            cross = matrix@field.conj().T+field@matrix.conj().T
            shape_tangent = 2*(b*ap-a*bp+b*b)*(w*w-(1-t*t)/depth)/normal
            close(cross/normal, shape_tangent*eye, 'actual full matrices verify the quadrupolar density tangent')
            close(field@field.conj().T, np.trace(field@field.conj().T).real/rank*eye,
                  'the complete shape generator has scalar pointwise left density')
            dm, df = [], []
            for axis, al in enumerate(alpha):
                mk = ap*v[axis]*eye-1j*(bp*v[axis]*vv-t*b*al)
                dak = (app*v[axis]*w+ap*z[axis])*eye-1j*((bpp*v[axis]*w+bp*z[axis])*vv
                        -bp*w*t*al+bp*v[axis]*zz-b*w*al)
                qak = .5j*(al@qa-qa@al)
                fk = 1j*(dak@qa+da@qak)-level/depth*mk
                dm.append(mk+.5j*matrix@al)
                df.append(fk+.5j*field@al)
            dirac_m = -1j*sum(al@jet for al, jet in zip(alpha, dm))
            dirac_f = -1j*sum(al@jet for al, jet in zip(alpha, df))
            close(dirac_m, h*matrix, 'the actual covariant spatial jet verifies the base positive Dirac shell')
            close(dirac_f, h*field, 'the actual covariant spatial jet verifies the polarized shape in the same positive shell')
            for angle in [-.19, 0., .27]:
                scale = np.sin(angle)/np.sqrt(metric) if metric else 0.
                base = np.cos(angle) if metric else 1.
                total = (base*matrix+scale*field)/np.sqrt(rank*normal)
                dirac_total = (base*dirac_m+scale*dirac_f)/np.sqrt(rank*normal)
                density = np.vdot(total, total).real
                close(total@total.conj().T, density/rank*eye,
                      'finite real shape combinations remain exactly pointwise balanced')
                massive = (mixing_a*eye+mixing_b*beta)@total
                dirac_massive = (mixing_a*eye-mixing_b*beta)@dirac_total
                current = np.array([np.vdot(massive, al@massive).real for al in alpha])
                scalar = np.vdot(massive, beta@massive).real
                close(current, np.zeros(depth), 'the finite massive shape family has exactly zero spatial particle current')
                close(scalar, x*density, 'the finite massive shape family remains in the same constitutive ratio')
                close((dirac_massive+source_m*beta@massive+eta*massive)/max(1., lam),
                      omega*massive/max(1., lam), 'the complete nonlinear particle equation retains the same inferred clock')
                background = eye/np.sqrt(rank)
                close(eta*sum(al@background*ji for al, ji in zip(alpha, current))/max(1., abs(eta)),
                      np.zeros_like(background), 'the retained background equation has no induced source on the exact family')
                energy = np.vdot(massive, dirac_massive).real+source_m*scalar+eta*density
                close(energy/max(1., lam), omega*density/max(1., lam),
                      'the local complete stationary particle energy equals its clock times canonical density')
        first, second = integrated_profiles(depth, level, 0), integrated_profiles(depth, level, 20)
        close(first['generator_norm_squared'], second['generator_norm_squared'],
              'independent radial quadratures agree on the exact shape-family charge normalization')
        cases.append(dict(particle=name, depth=depth, inferred_index=index,
                          generator_norm_squared_exact=str(sp.Rational(level*(depth-1)*(level+depth), depth**2)),
                          normalization=second, nontrivial_family=bool(level), source_m=source_m,
                          clock=omega, eta=eta, c=c,
                          classification='exact conditional stationary shape family, not a persistence theorem'))
        print(json.dumps(dict(completed=name, depth=depth, nontrivial_family=bool(level))), flush=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    report = dict(experiment_key='inferred-shell-exact-stationary-shape-family',
                  created_utc=datetime.now(timezone.utc).isoformat(),
                  sources={'experiments/'+p.name: sha256(p.read_bytes()).hexdigest()
                           for p in [Path(__file__), ROOT/'inverse-current-radial.py', ROOT/'excited-shell-orientation-current.py']},
                  evaluations=sum(COUNTS.values()), conditions=COUNTS, maximum_errors=ERRORS, cases=cases,
                  limits=['full matrix charge uses exact horizontality and pointwise scalar Gram matrices',
                          'small shape angles remain in the open nodeless constitutive chart by continuity',
                          'finite angles are pointwise equation checks, not a global nodeless theorem',
                          'no Euclidean localization, translational inertia, nonlinear persistence or particle identification follows',
                          'the ordinary sphere, inverse plateau action and equivariant field-block split remain conditional premises'])
    output = ROOT/('excited-shell-shape-family-results-'+stamp+'.json')
    output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(report=str(output), evaluations=report['evaluations'], maximum_errors=ERRORS)), flush=True)


if __name__ == '__main__':
    main()
