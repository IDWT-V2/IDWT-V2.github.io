"""Inverse constitutive count clocks on actual spatial spinor shell pairs."""
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

mp.mp.dps = 80
ROOT = Path(__file__).resolve().parent
spec = spec_from_file_location('ratio', ROOT/'local-current-ratio-checks.py')
ratio = module_from_spec(spec)
spec.loader.exec_module(ratio)
EPSILON = mp.mpf(1)/2**32
CONDITIONS, ERRORS = {}, {}


def check(value, label):
    if not bool(value):
        raise AssertionError(label)
    CONDITIONS[label] = CONDITIONS.get(label, 0)+1


def close(actual, expected, label, tolerance=mp.mpf('1e-35')):
    error = abs(actual-expected)/max(1, abs(expected))
    ERRORS[label] = max(ERRORS.get(label, 0.), float(error))
    check(error < tolerance, label)


def polynomial(depth, h):
    level = h-mp.mpf(depth)/2
    factors = [level+j for j in range(1, depth+1)]
    value = mp.fprod(factors)/factorial(depth)
    logarithmic = mp.fsum(1/a for a in factors)
    return value, value*logarithmic, value*(logarithmic**2-mp.fsum(1/a**2 for a in factors))


def constitutive(depth, x, seed):
    c = mp.sqrt(1-x*x)
    # A half-shell extension keeps the physical ground level inside the open chart.
    h = mp.findroot(lambda z: EPSILON*polynomial(depth, z)[1]-c,
                    (seed, seed+mp.mpf('.01')), solver='secant', tol=mp.mpf('1e-70'))
    check(h > mp.mpf(depth-1)/2 and 0 < c < 1,
          'constitutive inversion remains inside its explicit open half-shell extension')
    value, first, second = polynomial(depth, h)
    dc = EPSILON*second
    f = EPSILON*value-h*c
    fp = h*x/c
    fpp = h/c**3-x*x/(c*c*dc)
    return f, fp, fpp, h


def dagger(a):
    return a.transpose_conj()


def inner(a, b):
    return mp.fsum(mp.conj(a[i, j])*b[i, j] for i in range(a.rows) for j in range(a.cols))


def norm(a):
    return mp.sqrt(inner(a, a).real)


def convert(a):
    return mp.matrix([[mp.mpc(float(z.real), float(z.imag)) for z in row] for row in a])


def local(field, alpha, beta, depth, seed):
    rho = inner(field, field).real
    vector = [inner(field, a*field).real for a in alpha]
    signed = inner(field, beta*field).real
    proper = mp.sqrt(rho*rho-mp.fsum(j*j for j in vector))
    x = signed/proper
    check(proper > 0 and 0 < x < 1, 'tested inverse-current field stays in the specified open ratio chart')
    f, fp, _, _ = constitutive(depth, x, seed)
    operator = fp*beta+(f-x*fp)/proper*(rho*mp.eye(beta.rows)-sum((j*a for j, a in zip(vector, alpha)),
                                                                mp.zeros(beta.rows)))
    return proper*f, operator*field, rho, vector, signed, x


def pair_linearization(depth, h, a, b, x, f, fp, fpp, omega, frequency):
    sx, sz = mp.matrix([[0, 1], [1, 0]]), mp.diag([1, -1])
    field = mp.matrix([a, b])
    operator = h*sz+fp*sx+(f-x*fp)*mp.eye(2)
    basis = [mp.matrix([1, 0]), mp.matrix([0, 1]), mp.matrix([1j, 0]), mp.matrix([0, 1j])]
    analytic = mp.zeros(4)
    numerical = mp.zeros(4)
    def rhs(z):
        scalar = (dagger(z)*sx*z)[0].real/(dagger(z)*z)[0].real
        value, first, _, _ = constitutive(depth, scalar, h)
        return -1j*(h*sz+first*sx+(value-scalar*first-omega)*mp.eye(2))*z
    step = mp.mpf('1e-25')
    for j, variation in enumerate(basis):
        dx = 2*(dagger(variation)*(sx-x*mp.eye(2))*field)[0].real
        response = -1j*((operator-omega*mp.eye(2))*variation+
                         fpp*dx*(sx-x*mp.eye(2))*field)
        finite = (rhs(field+step*variation)-rhs(field-step*variation))/(2*step)
        for i in range(2):
            analytic[i, j], analytic[i+2, j] = response[i].real, response[i].imag
            numerical[i, j], numerical[i+2, j] = finite[i].real, finite[i].imag
    close(norm(analytic-numerical)/norm(analytic), 0,
          'complete four-real pair Jacobian agrees with independent nonlinear source differences', mp.mpf('1e-22'))
    eigenvalues = mp.eig(analytic, left=False, right=False)
    for actual, expected in zip(sorted(eigenvalues, key=lambda z: z.imag), [-1j*frequency, 0, 0, 1j*frequency]):
        close(abs(actual-expected)/frequency, 0,
              'complete pair spectrum retains its elliptic shape and both zero directions', mp.mpf('1e-25'))


def actual_spatial_fields():
    cases = []
    rng = np.random.default_rng(1734)
    for name, depth, index in ratio.scalar.sphere.ASSIGNMENTS:
        level = index-1
        h = mp.mpf(level)+mp.mpf(depth)/2
        value, first, second = polynomial(depth, h)
        count = comb(level+depth, depth)
        close(value, count, 'the continuous polynomial is the exact maintained binomial count')
        c = EPSILON*first
        dc = EPSILON*second
        check(0 < c < 1 and dc > 0, 'one common epsilon admits the convex count branch at every maintained assignment')
        x = mp.sqrt(1-c*c)
        f, fp, fpp, recovered = constitutive(depth, x, h)
        close(recovered, h, 'the same constitutive inversion recovers each actual bare shell frequency')
        close(f, EPSILON*value-h*c, 'inverse local energy retains the count Legendre relation')
        close(h-fp*c/x, 0, 'complete constrained shape derivative vanishes')
        close(fpp*c*c/(x*x)-fp/(x**3), -1/dc,
              'full pair curvature equals minus the reciprocal count curvature', mp.mpf('1e-25'))
        check(f < 0, 'an attractive local contribution is required on every tested count branch')
        a, b = mp.sqrt((1+c)/2), mp.sqrt((1-c)/2)
        omega = EPSILON*value
        close(h*c+f, omega, 'complete pair energy per norm is the exact target count clock')
        frequency = 2*mp.sqrt((1-c*c)*h/(c*dc))
        check(frequency > 0, 'both fixed-norm pair curvatures give an elliptic internal shape mode')
        pair_linearization(depth, h, a, b, x, f, fp, fpp, omega, frequency)
        aa, beta0, _ = ratio.scalar.clifford(depth)
        alpha0 = [convert(z) for z in aa]
        beta0 = convert(beta0)
        rank = beta0.rows
        for theta in (mp.mpf('.41'), mp.mpf('1.39')):
            t, s = mp.cos(theta), mp.sin(theta)
            gamma = alpha0[0]
            frame = mp.cos(theta/2)*mp.eye(rank)-1j*mp.sin(theta/2)*gamma
            beta = dagger(frame)*beta0*frame
            alpha = [dagger(frame)*z*frame for z in alpha0]
            al = (mp.mpf(level+depth-1)/(depth-1))*mp.gegenbauer(level, mp.mpf(depth-1)/2, t)
            bl = mp.gegenbauer(level-1, mp.mpf(depth+1)/2, t) if level else mp.mpf(0)
            positive = (al*mp.eye(rank)-1j*s*bl*gamma)/mp.sqrt(rank*comb(level+depth-1, depth-1))
            negative = beta*positive
            field = a*positive+b*negative
            kinetic = h*(a*positive-b*negative)
            potential, gradient, rho, vector, signed, current_ratio = local(field, alpha, beta, depth, h)
            close(max(abs(j)/rho for j in vector), 0, 'actual balanced shell multiplet retains zero spatial current')
            close(current_ratio, x, 'actual scalar/proper current selects the inverse constitutive ratio')
            close(norm(kinetic+gradient-omega*field)/(omega*norm(field)), 0,
                  'full pointwise nonlinear spatial residual vanishes relative to its selected count clock', mp.mpf('1e-45'))
            if theta == mp.mpf('.41'):
                variation = convert(rng.normal(size=(rank, rank))+1j*rng.normal(size=(rank, rank)))
                variation /= norm(variation)
                step = mp.mpf('1e-35')*norm(field)
                pp = local(field+step*variation, alpha, beta, depth, h)[0]
                pm = local(field-step*variation, alpha, beta, depth, h)[0]
                close((pp-pm)/(2*step), 2*inner(gradient, variation).real,
                      'arbitrary complex spatial-field variation retains all inverse current derivatives', mp.mpf('1e-25'))
        cases.append({'name': name, 'depth': depth, 'mode_index': index, 'spatial_shell_level': level,
                      'count_exact': str(count), 'epsilon_exact': '1/4294967296',
                      'selected_frequency_exact': str(Fraction(count, 2**32)),
                      'bare_shell_frequency': float(h), 'signed_shell_occupation': float(c),
                      'one_minus_scalar_ratio': float(1-x), 'potential_per_norm': float(f),
                      'retained_shape_frequency': float(frequency),
                      'retained_shape_over_particle_frequency': float(frequency/omega),
                      'full_spatial_stability_tested': False})
    return cases


def exact_identities():
    x, c, dc, h = sp.symbols('x c dc h', positive=True)
    fp = h*x/c
    fpp = h/c**3-x*x/(c*c*dc)
    cc = sp.simplify((fpp*c*c/x**2-fp/x**3+1/dc).subs(x*x, 1-c*c))
    check(cc == 0, 'exact inverse-pair Hessian equals minus inverse clock convexity')
    n, e = sp.symbols('n e', positive=True)
    for depth in ratio.scalar.sphere.DEPTHS:
        p = sp.prod(n+j for j in range(1, depth+1))/factorial(depth)
        first, second = sp.diff(p, n), sp.diff(p, n, 2)
        offset = sp.Rational(depth, 2)
        attractive = p-(n+offset)*first
        check(sp.simplify(sp.diff(attractive, n)+(n+offset)*second) == 0,
              'inverse potential decreases from its strictly negative ground value')
        check(attractive.subs(n, 0) < 0,
              'every occupied depth requires negative inverse potential already at its ground shell')
        check(all(z > 0 for z in sp.Poly(second, n).all_coeffs()),
              'count curvature is strictly positive on every nonnegative shell level')
    check(Fraction(22, 3838380-18564) == Fraction(11, 1909908),
          'general connected-pair slope bound gives the exact electron-muon epsilon ceiling')


def main():
    exact_identities()
    cases = actual_spatial_fields()
    domain = []
    for depth in ratio.scalar.sphere.DEPTHS:
        upper = mp.findroot(lambda n: EPSILON*polynomial(depth, n+mp.mpf(depth)/2)[1]-1,
                            (mp.mpf((factorial(depth-1)*2**32)**(1/(depth-1)))-depth,
                             mp.mpf((factorial(depth-1)*2**32)**(1/(depth-1)))+depth))
        check(upper > 0, 'the finite inverse ratio chart has a positive upper shell limit')
        domain.append({'depth': depth, 'largest_continuous_shell_level_at_slope_one': float(upper)})
    files = ['inverse-current-ratio-checks.py','local-current-ratio-checks.py','scalar-current-band-checks.py',
             'spatial-source-saturation-checks.py','spatial-band-curvature-checks.py']
    report = {'experiment_key': 'inverse-local-current-count-spatial-branch',
              'created_utc': datetime.now(timezone.utc).isoformat(),
              'sources': {'experiments/'+f: sha256((ROOT/f).read_bytes()).hexdigest() for f in files},
              'classification': 'inverse constitutive benchmark with target count polynomial supplied',
              'assumptions': ['actual round-sphere spatial shell pair with complete balanced spinor columns',
                              'one common dimensionless diagnostic epsilon=mu_d R_d=1/4294967296',
                              'dimension-dependent constitutive function is reconstructed by one Legendre formula from the supplied count polynomial',
                              'the constitutive action is only specified on its open inverse-current ratio chart with h>(d-1)/2',
                              'the test map equates actual shell degree l with cumulative cutoff N'],
              'limits': ['counts are encoded in the inverse action and have not been independently derived',
                         'negative local potential and unrestricted Dirac energy do not establish a globally positive action',
                         'elliptic pair persistence is not full spatial stability, localization, decay or observed spin',
                         'the constitutive chart cannot support an unlimited convex count ladder at one radius',
                         'a single parent law for the dimension-dependent constitutive functions and physical scales is unestablished'],
              'precision_digits': mp.mp.dps, 'condition_count': len(CONDITIONS),
              'evaluation_count': sum(CONDITIONS.values()), 'conditions': CONDITIONS,
              'maximum_errors': ERRORS, 'cases': cases, 'finite_chart_limits': domain}
    path = ROOT/('inverse-current-ratio-results-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')+'.json')
    path.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'report': str(path), 'condition_count': len(CONDITIONS),
                      'evaluation_count': sum(CONDITIONS.values()), 'maximum_errors': ERRORS,
                      'finite_chart_limits': domain}, indent=2), flush=True)


if __name__ == '__main__':
    main()
