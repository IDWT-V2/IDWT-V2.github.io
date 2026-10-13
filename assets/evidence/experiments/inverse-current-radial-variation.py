"""High-precision nonlinear variations independent of the spatial matrices."""
from datetime import datetime, timezone
from hashlib import sha256
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import json

import mpmath as mp
import numpy as np

ROOT = Path(__file__).resolve().parent
spec = spec_from_file_location('radial', ROOT/'inverse-current-radial.py')
radial = module_from_spec(spec)
spec.loader.exec_module(radial)
CONDITIONS, ERRORS = {}, {}


def check(value, label):
    if not bool(value):
        raise AssertionError(label)
    CONDITIONS[label] = CONDITIONS.get(label, 0)+1


def inner(a, b):
    return (a.transpose_conj()*b)[0]


def norm(a):
    return mp.sqrt(mp.fsum(abs(z)**2 for z in a))


def operators(theta):
    t, s = mp.cos(theta), mp.sin(theta)
    beta = mp.matrix([[0, 0, t, -s], [0, 0, -s, -t],
                      [t, -s, 0, 0], [-s, -t, 0, 0]])
    alpha = mp.matrix([[0, -1j, 0, 0], [1j, 0, 0, 0],
                       [0, 0, 0, -1j], [0, 0, 1j, 0]])
    beta0 = mp.matrix([[0, 0, 1, 0], [0, 0, 0, -1],
                       [1, 0, 0, 0], [0, -1, 0, 0]])
    perpendicular0 = mp.matrix([[0, 0, 0, 1], [0, 0, 1, 0],
                                [0, 1, 0, 0], [1, 0, 0, 0]])
    return beta, alpha, s*beta0+t*perpendicular0


def nonlinear(z, depth, level, epsilon, beta, vectors, vector_weight):
    rho = inner(z, z).real
    scalar = inner(z, beta*z).real
    currents = [inner(z, operator*z).real for operator in vectors]
    proper = mp.sqrt(rho*rho-vector_weight*mp.fsum(q*q for q in currents))
    x = scalar/proper
    c = mp.sqrt(1-x*x)
    h = mp.findroot(lambda hh: epsilon*radial.polynomial(depth, hh-mp.mpf(depth)/2)[1]-c,
                    (mp.mpf(level)+mp.mpf(depth)/2, mp.mpf(level)+mp.mpf(depth)/2+mp.mpf('.001')),
                    tol=mp.mpf('1e-100'))
    p, _, _ = radial.polynomial(depth, h-mp.mpf(depth)/2)
    f = epsilon*p-h*c
    fp = h*x/c
    b = f-x*fp
    operator = fp*beta+b/proper*(rho*mp.eye(4)-vector_weight*sum((q*g for q, g in zip(currents, vectors)), mp.zeros(4)))
    return proper*f, operator*z


def main():
    mp.mp.dps = 120
    rng = np.random.default_rng(20261012)
    records = []
    for name, depth, index in radial.ASSIGNMENTS:
        last = max(n for _, d, n in radial.ASSIGNMENTS if d == depth)
        level = index-1
        p, first, second = radial.polynomial(depth, level)
        epsilon = mp.mpf('.3')/radial.polynomial(depth, last-1)[1]
        c = epsilon*first
        x = mp.sqrt(1-c*c)
        h = mp.mpf(level)+mp.mpf(depth)/2
        lam = h/c
        omega = epsilon*p
        a, b = mp.sqrt((1+c)/2), mp.sqrt((1-c)/2)
        eta = 1-x*x*first/(h*second)
        vf = 1-omega/lam
        for theta in map(mp.mpf, ['.37', '1.23']):
            beta, alpha, transverse = operators(theta)
            t, s = mp.cos(theta), mp.sin(theta)
            aa = mp.mpf(level+depth-1)/(depth-1)*mp.gegenbauer(level, mp.mpf(depth-1)/2, t)
            bb = mp.gegenbauer(level-1, mp.mpf(depth+1)/2, t) if level else mp.mpf(0)
            positive = mp.matrix([aa, s*bb, 0, 0])
            negative = beta*positive
            z = a*positive+b*negative
            tangent = a*negative-b*positive
            rho = inner(z, z).real
            vectors = [alpha]+([transverse] if depth == 2 else [])
            variation = mp.matrix([mp.mpc(float(rng.normal()), float(rng.normal())) for _ in range(4)])
            variation /= norm(variation)
            for weight in (0, 1):
                local = lambda zz: nonlinear(zz, depth, level, epsilon, beta, vectors, weight)
                step = c**4*mp.mpf('1e-31')*norm(z)
                up, gp = local(z+step*variation)
                um, gm = local(z-step*variation)
                u, g = local(z)
                action_error = abs((up-um)/(2*step)-2*inner(g, variation).real)/max(1, abs(2*inner(g, variation).real))
                check(action_error < mp.mpf('1e-55'), 'independent complete nonlinear action variation gives the current-ratio source: '+mp.nstr(action_error, 8))
                analytic = (x*beta-mp.eye(4))*variation+2*eta/rho*tangent*inner(tangent, variation).real
                for operator in vectors:
                    v = operator*z
                    analytic += 2*weight*vf/rho*v*inner(v, variation).real
                numerical = ((gp-gm)/(2*step)-omega*variation)/lam
                error = norm(numerical-analytic)/max(1, norm(analytic))
                check(error < mp.mpf('1e-50'), 'independent nonlinear source differences verify the complete complex spatial Hessian')
                ERRORS['complete nonlinear action variation'] = max(ERRORS.get('complete nonlinear action variation', 0.), float(action_error))
                ERRORS['complete complex local Hessian'] = max(ERRORS.get('complete complex local Hessian', 0.), float(error))
                records.append({'name': name, 'depth': depth, 'mode_index': index,
                                'theta': str(theta), 'vector_weight': weight,
                                'signed_shell_occupation': str(c), 'step': str(step),
                                'action_variation_relative_error': float(action_error),
                                'hessian_relative_error': float(error)})
        print(json.dumps({'name': name, 'depth': depth, 'verified': True}), flush=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    report = {'experiment_key': 'inverse-current-independent-radial-nonlinear-variation',
              'created_utc': datetime.now(timezone.utc).isoformat(),
              'sources': {'experiments/'+p.name: sha256(p.read_bytes()).hexdigest()
                          for p in [Path(__file__), ROOT/'inverse-current-radial.py']},
              'conditions': CONDITIONS, 'evaluations': sum(CONDITIONS.values()),
              'maximum_errors': ERRORS, 'decimal_precision': mp.mp.dps, 'cases': records,
              'limits': ['arbitrary complex variations inside the rotational matrix algebra',
                         'does not verify a continuum eigenvalue or nonlinear persistence theorem',
                         'vector_weight zero is the distinct preferred-time density action']}
    output = ROOT/('inverse-current-radial-variation-results-'+stamp+'.json')
    output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'report': str(output), 'evaluations': report['evaluations']}), flush=True)


if __name__ == '__main__':
    main()
