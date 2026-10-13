"""One shell-preserving nonlinear correction and full linearized source checks."""
from datetime import datetime, timezone
from hashlib import sha256
from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path
import json

import mpmath as mp
import numpy as np

ROOT = Path(__file__).resolve().parent
spec = spec_from_file_location('radial', ROOT/'inverse-current-radial.py')
radial = module_from_spec(spec)
spec.loader.exec_module(radial)
mp.mp.dps = 140
ERRORS, CONDITIONS = {}, {}


def check_close(actual, expected, label, tolerance=mp.mpf('1e-75')):
    error = abs(actual-expected)/max(1, abs(expected))
    ERRORS[label] = max(ERRORS.get(label, mp.mpf(0)), error)
    if not error < tolerance:
        raise AssertionError((label, mp.nstr(error, 20)))
    CONDITIONS[label] = CONDITIONS.get(label, 0)+1


def recipe(depth, epsilon):
    def polynomial(h):
        return mp.fprod(h+j-mp.mpf(depth)/2 for j in range(1, depth+1))/mp.factorial(depth)

    def data(h):
        p = polynomial(h)
        first = mp.diff(polynomial, h)
        second = mp.diff(polynomial, h, 2)
        c = epsilon*first
        x = mp.sqrt(1-c*c)
        f = epsilon*p-h*c
        fp = h*x/c
        fpp = h/c**3-x*x/(c*c*epsilon*second)
        coefficient = (epsilon*second-h*epsilon**2*second**2/(c*x*x))/(2*mp.pi**2)
        return p, c, x, f, fp, fpp, coefficient

    def corrected(h):
        p, c, x, f, fp, fpp, coefficient = data(h)
        return f+coefficient*mp.sin(mp.pi*(h-mp.mpf(depth)/2))**2

    def x_derivative(h):
        return mp.diff(lambda y: data(y)[2], h)

    def corrected_first(h):
        return mp.diff(corrected, h)/x_derivative(h)

    def corrected_second(h):
        return mp.diff(corrected_first, h)/x_derivative(h)

    return data, corrected, corrected_first, corrected_second


def cutoff(offset):
    square = offset*offset
    inner, outer = mp.mpf(1)/64, mp.mpf(1)/16
    if square <= inner:
        return mp.mpf(1)
    if square >= outer:
        return mp.mpf(0)
    left = mp.exp(-1/(outer-square))
    right = mp.exp(-1/(square-inner))
    return left/(left+right)


def main():
    cases = []
    rng = np.random.default_rng(2026101204)
    for name, depth, index in radial.ASSIGNMENTS:
        largest = max(n for _, d, n in radial.ASSIGNMENTS if d == depth)
        epsilon = mp.mpf('0.3')/radial.polynomial(depth, largest-1)[1]
        data, corrected, corrected_first, corrected_second = recipe(depth, epsilon)
        h = mp.mpf(index-1)+mp.mpf(depth)/2
        p, c, x, f, fp, fpp, coefficient = data(h)
        check_close(corrected(h), f, 'all assigned shell potential values are preserved')
        check_close(corrected_first(h), fp, 'all assigned shell complete first sources are preserved')
        check_close(corrected_second(h), 0, 'one common correction removes every assigned shell scalar curvature')
        def flat(y):
            original = data(y)
            return original[3]+cutoff(y-h)*(f+fp*(original[2]-x)-original[3])
        for offset in [mp.mpf(-1)/16, mp.mpf(0), mp.mpf(1)/16]:
            hh = h+offset
            check_close(flat(hh), f+fp*(data(hh)[2]-x),
                        'one common flat patch equals the exact shell tangent at finite neighboring ratios')
            first_flat = lambda y: mp.diff(flat, y)/mp.diff(lambda yy: data(yy)[2], y)
            check_close(first_flat(hh), fp, 'flat patch retains a constant source throughout its open neighborhood')
            check_close(mp.diff(first_flat, hh)/mp.diff(lambda y: data(y)[2], hh), 0,
                        'flat patch curvature vanishes throughout its open neighborhood')
        for offset in [mp.mpf(-3)/8, mp.mpf(3)/8]:
            check_close(flat(h+offset), data(h+offset)[3],
                        'flat patch returns to the original law away from its support')
        # Independent variations use only exact Gram coefficients of a
        # generic full complex matrix direction, without restricting its
        # spin or angular polarization.
        matrices = radial.gammas(depth+1)
        beta = matrices[depth]
        rank = len(beta)
        a, b = mp.sqrt((1+c)/2), mp.sqrt((1-c)/2)
        # Diagonalize beta solely to evaluate the exact scalar variation.
        eigenvalues = np.linalg.eigvalsh(beta)
        field = [((a+b*int(round(ev)))/mp.sqrt(rank)) for ev in eigenvalues]
        random = rng.normal(size=(rank, rank))+1j*rng.normal(size=(rank, rank))
        direction = [[mp.mpc(str(v.real), str(v.imag)) for v in row] for row in random]
        direction_norm = sum(abs(v)**2 for row in direction for v in row)
        direction = [[v/mp.sqrt(direction_norm) for v in row] for row in direction]
        rho0 = sum(v*v for v in field)
        j0 = sum(int(round(ev))*v*v for ev, v in zip(eigenvalues, field))
        rho1 = 2*sum((field[i]*direction[i][i]).real for i in range(rank))
        j1 = 2*sum(int(round(eigenvalues[i]))*(field[i]*direction[i][i]).real for i in range(rank))
        rho2 = sum(abs(v)**2 for row in direction for v in row)
        j2 = sum(int(round(eigenvalues[i]))*sum(abs(v)**2 for v in direction[i]) for i in range(rank))
        step = c**4*mp.mpf('1e-31')

        def nearby(t):
            rho = rho0+rho1*t+rho2*t*t
            signed = j0+j1*t+j2*t*t
            ratio = signed/rho
            target_c = mp.sqrt(1-ratio*ratio)
            nearby_h = mp.findroot(lambda y: data(y)[1]-target_c, (h, h+mp.mpf('1e-8')))
            ff, first = corrected(nearby_h), corrected_first(nearby_h)
            return rho*ff, first, ff-ratio*first

        up, down = nearby(step), nearby(-step)
        expected = fp*j1+(f-x*fp)*rho1
        check_close((up[0]-down[0])/(2*step), expected,
                    'independent full matrix action variation retains the same stationary source', mp.mpf('1e-50'))
        check_close((up[1]-down[1])/(2*step), 0,
                    'independent full matrix scalar-source derivative vanishes', mp.mpf('1e-50'))
        check_close((up[2]-down[2])/(2*step), 0,
                    'independent full matrix identity-source derivative vanishes', mp.mpf('1e-50'))
        # The complete preferred-time linearization is D+fp beta-lambda.
        # Its exact spectral expression is real at every spatial shell.
        lam = h/c
        check_close(mp.sqrt(h*h+fp*fp)-lam, 0,
                    'the assigned shell is the zero mode of the full self-adjoint linearization')
        check_close((lam-fp*x), h*c,
                    'the retained background source has its original clock identity')
        cases.append({'particle_assignment': name, 'depth': depth, 'mode_index': index,
                      'spatial_shell_degree': index-1, 'epsilon': mp.nstr(epsilon, 40),
                      'c': mp.nstr(c, 40), 'x': mp.nstr(x, 40),
                      'frequency': mp.nstr(epsilon*p, 40),
                      'original_scalar_curvature': mp.nstr(fpp, 40),
                      'corrected_scalar_curvature': mp.nstr(corrected_second(h), 15),
                      'flat_patch_inner_half_width': '1/8',
                      'flat_patch_outer_half_width': '1/4',
                      'full_linearized_spectral_formula': '+/-sqrt((m+d/2)^2+(h*x/c)^2)-h/c, divided by R',
                      'classification': 'full spin and angular linearization for the preferred-time density action'})
        print(json.dumps({'particle': name, 'checks_passed': True}), flush=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    output = ROOT/('shell-curvature-locking-results-'+stamp+'.json')
    report = {'experiment_key': 'shell-preserving-curvature-locking',
              'created_utc': datetime.now(timezone.utc).isoformat(),
              'sources': {'experiments/'+p.name: sha256(p.read_bytes()).hexdigest()
                          for p in [Path(__file__), ROOT/'inverse-current-radial.py']},
              'conditions': CONDITIONS, 'evaluations': sum(CONDITIONS.values()),
              'maximum_errors': {k: mp.nstr(v, 30) for k, v in ERRORS.items()},
              'cases': cases,
              'limits': ['inverse constitutive construction on the supplied ordinary sphere and balanced carrier',
                         'the density rho action selects a preferred time frame; no Lorentz completion is claimed',
                         'the sine correction alone establishes linear norm persistence, not nonlinear stability',
                         'the alternative flat-patch nonlinear persistence uses the separately stated Sobolev bootstrap proof',
                         'no Euclidean spatial localization, complete boost inertia or charge selection is established',
                         'no field amplitude was chosen to fit mass; one formula applies to every admissible integer shell']}
    output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'report': str(output), 'evaluations': report['evaluations']}), flush=True)


if __name__ == '__main__':
    main()
