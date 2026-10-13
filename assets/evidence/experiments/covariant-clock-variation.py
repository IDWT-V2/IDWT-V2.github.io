"""Independent full matrix variations of a responding same-field clock."""
from datetime import datetime, timezone
from hashlib import sha256
from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path
import json

import numpy as np

ROOT = Path(__file__).resolve().parent
spec = spec_from_file_location('radial', ROOT/'inverse-current-radial.py')
radial = module_from_spec(spec)
spec.loader.exec_module(radial)
COUNTS, ERRORS = {}, {}


def close(actual, expected, label, tolerance=3e-8):
    error = float(np.linalg.norm(np.asarray(actual)-np.asarray(expected))/max(1., np.linalg.norm(expected)))
    ERRORS[label] = max(ERRORS.get(label, 0.), error)
    if error >= tolerance:
        raise AssertionError((label, error))
    COUNTS[label] = COUNTS.get(label, 0)+1


def current(field, alpha):
    return np.array([np.vdot(field, field).real]+[np.vdot(field, a@field).real for a in alpha])


def action_source(particle, background, alpha, beta, mass, source):
    jp, jb = current(particle, alpha), current(background, alpha)
    proper = np.sqrt(jb[0]**2-np.dot(jb[1:], jb[1:]))
    flow = jb/proper
    q = flow[0]*jp[0]-np.dot(flow[1:], jp[1:])
    signed = np.vdot(particle, beta@particle).real
    potential = mass*signed+source*q
    slash_time = flow[0]*np.eye(len(beta))-sum(u*a for u, a in zip(flow[1:], alpha))
    first = (mass*beta+source*slash_time)@particle
    transverse = source*(jp-q*flow)/proper
    second = (transverse[0]*np.eye(len(beta))-sum(t*a for t, a in zip(transverse[1:], alpha)))@background
    return potential, first, second


def main():
    rng = np.random.default_rng(202610120523)
    cases = []
    for depth in radial.DEPTHS:
        matrices = radial.gammas(depth+1)
        alpha, beta = matrices[:depth], matrices[depth]
        rank = len(beta)
        eye = np.eye(rank)
        c, x, load = .6, .8, .2
        a, b = np.sqrt((1+c)/2), np.sqrt((1-c)/2)
        h = depth/2
        first_harmonic = sum(1/j for j in range(1, depth+1))
        omega, lam, mass = c/first_harmonic, h/c, h*x/c
        source = omega-lam
        p0 = np.sqrt(load/rank)*(a*eye+b*beta)
        b0 = eye/np.sqrt(rank)
        for sample in range(4):
            p = p0+.03*(rng.normal(size=p0.shape)+1j*rng.normal(size=p0.shape))/rank
            bg = b0+.03*(rng.normal(size=b0.shape)+1j*rng.normal(size=b0.shape))/rank
            dp = rng.normal(size=p0.shape)+1j*rng.normal(size=p0.shape)
            db = rng.normal(size=p0.shape)+1j*rng.normal(size=p0.shape)
            dp /= np.linalg.norm(dp)
            db /= np.linalg.norm(db)
            value, gp, gb = action_source(p, bg, alpha, beta, mass, source)
            step = 2e-6
            finite = (action_source(p+step*dp, bg+step*db, alpha, beta, mass, source)[0]
                      -action_source(p-step*dp, bg-step*db, alpha, beta, mass, source)[0])/(2*step)
            close(finite, 2*np.real(np.vdot(gp, dp)+np.vdot(gb, db)), 'joint action variation retains both responding sources')
            close(np.vdot(p, gp).real, value, 'particle Euler contraction includes the complete interaction')
            close(np.vdot(bg, gb), 0., 'background Euler contraction vanishes without freezing its source')
            rapidity = .57
            boost = np.cosh(rapidity/2)*eye+np.sinh(rapidity/2)*alpha[0]
            boosted = action_source(boost@p, boost@bg, alpha, beta, mass, source)
            close(boosted[0], value, 'full action is invariant under a simultaneous actual Lorentz boost')
            close(boost.conj().T@boosted[1], gp, 'particle source has the complete contragredient boost law')
            close(boost.conj().T@boosted[2], gb, 'responding background source has the complete contragredient boost law')
            base = action_source(p0, b0, alpha, beta, mass, source)
            close(base[1], (mass*beta+source*eye)@p0, 'balanced clock preserves the stationary particle source')
            close(base[2], np.zeros_like(b0), 'balanced free background is an actual stationary solution')
            djp = np.array([2*np.vdot(p0, al@dp).real for al in alpha])
            djb = np.array([2*np.vdot(b0, al@db).real for al in alpha])
            linear_p = (mass*beta+source*eye)@dp-source*sum(j*al@p0 for j, al in zip(djb, alpha))
            linear_b = source*sum((-p+load*b)*al@b0 for p, b, al in zip(djp, djb, alpha))
            plus = action_source(p0+step*dp, b0+step*db, alpha, beta, mass, source)
            minus = action_source(p0-step*dp, b0-step*db, alpha, beta, mass, source)
            close((plus[1]-minus[1])/(2*step), linear_p, 'independent complete particle Hessian matches all vector current derivatives')
            close((plus[2]-minus[2])/(2*step), linear_b, 'independent complete background Hessian includes reciprocal response and its self term')
            # The coefficient of t squared is half the ordinary real Hessian.
            # Use a larger step to separate this second difference from roundoff.
            curvature_step = 3e-4
            vp = action_source(p0+curvature_step*dp, b0+curvature_step*db, alpha, beta, mass, source)[0]
            vm = action_source(p0-curvature_step*dp, b0-curvature_step*db, alpha, beta, mass, source)[0]
            finite_curvature = (vp+vm-2*base[0])/(2*curvature_step**2)
            predicted = np.vdot(dp, (mass*beta+source*eye)@dp).real-source*np.dot(djb, djp)+source*load*np.dot(djb, djb)/2
            close(finite_curvature, predicted, 'joint second action variation verifies the entire quadratic current energy', 5e-7)
        cases.append(dict(depth=depth, spinor_rank=rank, c_exact='3/5', load_exact='1/5',
                          classification='independent full complex matrix action and Hessian variations'))
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    report = dict(experiment_key='covariant-clock-full-matrix-variation',
                  created_utc=datetime.now(timezone.utc).isoformat(),
                  sources={'experiments/'+p.name: sha256(p.read_bytes()).hexdigest()
                           for p in [Path(__file__), ROOT/'inverse-current-radial.py']},
                  conditions=COUNTS, evaluations=sum(COUNTS.values()), maximum_errors=ERRORS, cases=cases,
                  limits=['affine shell plateau and retained two-block same-field ansatz',
                          'local checks do not establish spatial spectral completeness or nonlinear persistence',
                          'uncalibrated particle/background density ratio is an explicit control'])
    output = ROOT/('covariant-clock-variation-results-'+stamp+'.json')
    output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(report=str(output), evaluations=report['evaluations'], maximum_errors=ERRORS)), flush=True)


if __name__ == '__main__':
    main()
