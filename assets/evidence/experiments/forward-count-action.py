"""Vary the algebraic count action forward, retaining actual spatial jets."""
from datetime import datetime, timezone
from fractions import Fraction
from hashlib import sha256
from importlib.util import spec_from_file_location, module_from_spec
from math import comb, factorial
from pathlib import Path
import json

import mpmath as mp
import numpy as np
import sympy as sp

ROOT = Path(__file__).resolve().parent
spec = spec_from_file_location('orientation', ROOT/'excited-shell-orientation-current.py')
orientation = module_from_spec(spec)
spec.loader.exec_module(orientation)
radial = orientation.radial
COUNTS, ERRORS = {}, {}


def check(value, label):
    if not bool(value):
        raise AssertionError(label)
    COUNTS[label] = COUNTS.get(label, 0)+1


def close(actual, expected, label, tolerance=3e-8):
    error = float(np.linalg.norm(np.asarray(actual)-np.asarray(expected))/max(1., np.linalg.norm(expected)))
    ERRORS[label] = max(ERRORS.get(label, 0.), error)
    check(error < tolerance, label)


def count_data(depth, level):
    count = Fraction(comb(level+depth, depth))
    reciprocals = [Fraction(1, level+j) for j in range(1, depth+1)]
    first = count*sum(reciprocals)
    second = count*(sum(reciprocals)**2-sum(v*v for v in reciprocals))
    return count, first, second


def polynomial(depth, chi):
    return np.prod([chi+j-depth/2 for j in range(1, depth+1)])/factorial(depth)


def bilinears(field, alpha, beta):
    rho = np.vdot(field, field).real
    scalar = np.vdot(field, beta@field).real
    current = np.array([np.vdot(field, a@field).real for a in alpha])
    norm = np.sqrt(rho*rho-current@current)
    support = np.sqrt((rho-scalar)*(rho+scalar)-current@current)
    return rho, scalar, current, norm, support


def potential(field, chi, epsilon, alpha, beta, depth):
    _, _, _, norm, support = bilinears(field, alpha, beta)
    return epsilon*norm*polynomial(depth, chi)-chi*support


def source(field, chi, epsilon, alpha, beta, depth):
    rho, scalar, current, norm, support = bilinears(field, alpha, beta)
    m = chi*scalar/support
    eta = epsilon*polynomial(depth, chi)-chi*norm/support
    return m*beta@field+eta/norm*(rho*field-sum(j*a@field for j, a in zip(current, alpha)))


def symbolic_forward_variation():
    h, chi, c, y, epsilon = sp.symbols('h chi c y epsilon', positive=True)
    p = sp.Function('P')
    energy = h*c+epsilon*p(chi)-chi*sp.sqrt(c*c+y*y)
    check(sp.simplify(sp.diff(energy, c).subs(y, 0)-(h-chi)) == 0,
          'forward polarization variation locks the auxiliary label to the actual spatial eigenvalue')
    check(sp.simplify(sp.diff(energy, chi).subs(y, 0)-(epsilon*sp.diff(p(chi), chi)-c)) == 0,
          'forward auxiliary variation fixes the polarization by the derivative of the spatial count')
    check(sp.simplify(energy.subs({chi: h, y: 0})-epsilon*p(h)) == 0,
          'the complete stationary energy reduces to the count without discarding spatial kinetics')
    check(sp.diff(energy, y).subs(y, 0) == 0,
          'the retained relative-phase equation is stationary on the real pair')
    check(sp.simplify(sp.diff(energy, y, 2).subs({chi: h, y: 0})+h/c) == 0,
          'the forward action retains the original relative-phase curvature')
    check(sp.simplify(-sp.diff(energy, c, chi)**2/sp.diff(energy, chi, 2)).subs(y, 0)
          == -1/(epsilon*sp.diff(p(chi), chi, 2)),
          'eliminating the algebraic variable retains the original polarization curvature')


def main():
    symbolic_forward_variation()
    mp.mp.dps = 85
    rng = np.random.default_rng(202610120744)
    epsilon_exact = {d: Fraction(3, 10)/max(count_data(d, n-1)[1]
                     for _, dd, n in radial.ASSIGNMENTS if dd == d) for d in radial.DEPTHS}
    cases = []
    for name, depth, index in radial.ASSIGNMENTS:
        level, h = index-1, Fraction(2*(index-1)+depth, 2)
        count, first, second = count_data(depth, level)
        epsq, cq = epsilon_exact[depth], epsilon_exact[depth]*first
        check(0 < cq < 1 and second > 0, 'one sector comparison scale keeps every inferred shell in the convex open chart')
        def high(value):
            return mp.mpf(value.numerator)/value.denominator
        eps, ch, hh = high(epsq), high(cq), high(h)
        def count_function(label):
            return mp.fprod(label+j-mp.mpf(depth)/2 for j in range(1, depth+1))/factorial(depth)
        def reduced_energy(polarization, label, phase):
            return hh*polarization+eps*count_function(label)-label*mp.sqrt(polarization**2+phase**2)
        def stationary_equations(polarization, label, phase):
            return (mp.diff(lambda cc: reduced_energy(cc, label, phase), polarization),
                    mp.diff(lambda zz: reduced_energy(polarization, zz, phase), label),
                    mp.diff(lambda yy: reduced_energy(polarization, label, yy), phase))
        solved = mp.findroot(stationary_equations, (ch*mp.mpf('.83'), hh*mp.mpf('.97'), ch*mp.mpf('.05')),
                             tol=mp.mpf('1e-72'), maxsteps=30)
        check(mp.norm(solved-mp.matrix([ch, hh, 0])) < mp.mpf('1e-65'),
              'independent simultaneous stationary solve recovers the geometric label polarization and relative phase')
        check(abs(reduced_energy(*solved)-eps*high(count))/(eps*high(count)) < mp.mpf('1e-65'),
              'independent stationary energy evaluation recovers the count clock relatively')
        enumerated = sum(comb(k+depth-1, depth-1) for k in range(level+1))
        check(enumerated == count,
              'independent cumulative spatial shell enumeration gives the constitutive count at the solved label')
        xx = mp.sqrt(1-ch*ch)
        aa, bb = mp.sqrt((1+ch)/2), mp.sqrt((1-ch)/2)
        mass_source = hh*xx/ch
        eta = eps*high(count)-hh/ch
        omega = eps*high(count)
        pair = mp.matrix([aa, bb])
        pair_h = mp.matrix([[hh+eta, mass_source], [mass_source, -hh+eta]])
        relative_residual = mp.norm(pair_h*pair-omega*pair)/omega
        check(relative_residual < mp.mpf('1e-65'), 'high precision complete pair equation verifies the selected count clock relatively')
        check(abs(hh*ch+eps*high(count)-hh*ch-omega) < mp.mpf('1e-75'),
              'high precision complete pair energy retains the spatial kinetic contribution')
        generators = radial.gammas(depth+1)
        alpha, beta = generators[:depth], generators[depth]
        rank, eye = len(beta), np.eye(len(beta))
        ep = float(eps)
        for sample in range(3):
            field = rng.normal(size=(rank, rank))+1j*rng.normal(size=(rank, rank))
            field /= np.linalg.norm(field)
            variation = rng.normal(size=field.shape)+1j*rng.normal(size=field.shape)
            variation /= np.linalg.norm(variation)
            chi, dc = float(h)+.17, .31
            step = 2e-6
            rho, scalar, current, norm, support = bilinears(field, alpha, beta)
            factors = [chi+j-depth/2 for j in range(1, depth+1)]
            derivative = polynomial(depth, chi)*sum(1/v for v in factors)
            expected = 2*np.vdot(variation, source(field, chi, ep, alpha, beta, depth)).real
            expected += dc*(ep*norm*derivative-support)
            fd = (potential(field+step*variation, chi+step*dc, ep, alpha, beta, depth)
                  -potential(field-step*variation, chi-step*dc, ep, alpha, beta, depth))/(2*step)
            close(fd/max(1., chi), expected/max(1., chi),
                  'independent arbitrary field and auxiliary variations verify the complete local source')
            direction = rng.normal(size=depth); direction /= np.linalg.norm(direction)
            al = sum(v*a for v, a in zip(direction, alpha))
            rapidity = .47
            spin = np.cosh(rapidity/2)*eye+np.sinh(rapidity/2)*al
            close(potential(spin@field, chi, ep, alpha, beta, depth),
                  potential(field, chi, ep, alpha, beta, depth),
                  'actual spinor boosts preserve the one-field count potential')
            position = rng.normal(size=depth+1); position /= np.linalg.norm(position)
            frame, _ = np.linalg.qr(np.column_stack([position, rng.normal(size=(depth+1, depth))]))
            t, v = position[0], frame[0, 1:]
            av = sum(vi*a for vi, a in zip(v, alpha))
            a, b, ap, bp = orientation.polynomials(depth, level, np.array(t))
            matrix = a*eye-1j*b*av
            jets = [(ap*v[i]*eye-1j*(bp*v[i]*av-t*b*al)+.5j*matrix@al)
                    for i, al in enumerate(alpha)]
            dmatrix = -1j*sum(al@jet for al, jet in zip(alpha, jets))
            close(dmatrix, float(h)*matrix, 'actual covariant Clifford jets verify the spatial eigenvalue entering the forward derivation')
            field = (float(aa)*eye+float(bb)*beta)@matrix
            field /= np.linalg.norm(field)
            dfield = (float(aa)*eye-float(bb)*beta)@dmatrix/np.linalg.norm((float(aa)*eye+float(bb)*beta)@matrix)
            full = dfield+source(field, float(h), ep, alpha, beta, depth)
            close(full/float(hh/ch), float(omega)*field/float(hh/ch),
                  'actual full local field source and spatial jet satisfy the selected stationary branch', 3e-6)
        cases.append(dict(particle=name, depth=depth, inferred_index=index,
                          auxiliary_label_exact=str(h), count_exact=str(count),
                          epsilon_exact=str(epsq), polarization_exact=str(cq),
                          frequency_in_radius_units_exact=str(epsq*count),
                          pair_clock_relative_residual=str(relative_residual),
                          source_coefficient=float(mass_source),
                          eliminated_polarization_curvature_exact=str(-1/(epsq*second)),
                          classification='forward conditional derivation from the explicit count constitutive action'))
        print(json.dumps(dict(completed=name, depth=depth, count=int(count))), flush=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    report = dict(experiment_key='forward-algebraic-count-action', created_utc=datetime.now(timezone.utc).isoformat(),
                  sources={'experiments/'+p.name: sha256(p.read_bytes()).hexdigest() for p in
                           [Path(__file__), ROOT/'inverse-current-radial.py', ROOT/'excited-shell-orientation-current.py']},
                  evaluations=sum(COUNTS.values()), conditions=COUNTS, maximum_errors=ERRORS, cases=cases,
                  limits=['the spatial counting constitutive potential is a stated premise recovered by inverse research',
                          'the algebraic label adds no propagating field or mediator',
                          'this action is equivalent to the raw inverse-current law, not its flattened persistence modification',
                          'spatial jet residuals are scaled by the large source coefficient; pair clocks are checked relatively at high precision',
                          'the conditional sphere, sector scales, sparse occupation and localized support are not derived here',
                          'no new assigned-depth preference, observed particle identification or additional persistence theorem follows'])
    output = ROOT/('forward-count-action-results-'+stamp+'.json')
    output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(report=str(output), evaluations=report['evaluations'], maximum_errors=ERRORS)), flush=True)


if __name__ == '__main__':
    main()
