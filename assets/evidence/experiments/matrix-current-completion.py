"""Lorentz-invariant matrix-current completion and exact ground controls."""
from datetime import datetime, timezone
from hashlib import sha256
from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path
from math import comb
import json

import numpy as np
from scipy.linalg import eigvals
import sympy as sp

ROOT = Path(__file__).resolve().parent
spec = spec_from_file_location('ground', ROOT/'inverse-current-ground-proof.py')
ground = module_from_spec(spec)
spec.loader.exec_module(ground)
radial = ground.radial
CONDITIONS, ERRORS = {}, {}


def check(value, label):
    if not bool(value):
        raise AssertionError(label)
    CONDITIONS[label] = CONDITIONS.get(label, 0)+1


def close(a, b, label, tolerance=3e-8):
    error = np.linalg.norm(np.asarray(a)-np.asarray(b))/max(1., np.linalg.norm(b))
    ERRORS[label] = max(ERRORS.get(label, 0.), float(error))
    check(error < tolerance, label)


def invariants(field, beta):
    rank = len(beta)
    matrix = field@field.conj().T
    right_scalar = field.conj().T@beta@field
    rho = np.trace(matrix).real
    signed = np.trace(right_scalar).real
    square = np.trace(right_scalar@right_scalar).real
    proper = np.sqrt(rank*square-signed*signed)
    return matrix, right_scalar, rho, signed, square, proper


def potential_gradient(field, beta):
    # A separate smooth test function isolates the matrix chain rule.
    # Exact count backgrounds are checked with their own f and f' below.
    matrix, _, rho, signed, square, proper = invariants(field, beta)
    x = signed/proper
    f, fp = .37+1.21*x*x/2, 1.21*x
    density_gradient = (len(beta)*beta@matrix@beta-signed*beta)/proper
    gradient = (fp*beta+(f-x*fp)*density_gradient)@field
    return proper*f, gradient, density_gradient


def nonlinear_checks():
    rng = np.random.default_rng(2026101204)
    cases = []
    for depth in radial.DEPTHS:
        matrices = radial.gammas(depth+1)
        alpha, beta = matrices[:depth], matrices[depth]
        rank = len(beta)
        for sample in range(3):
            field = rng.normal(size=(rank, rank))+1j*rng.normal(size=(rank, rank))
            field /= np.linalg.norm(field)
            matrix, right_scalar, rho, signed, square, proper = invariants(field, beta)
            check(proper+1e-14 >= abs(signed), 'matrix inertia bound keeps the completed ratio inside its physical unit interval')
            close(square, np.trace(beta@matrix@beta@matrix).real,
                  'right scalar Gram matrix and left density give the same Lorentz scalar square')
            value, gradient, _ = potential_gradient(field, beta)
            variation = rng.normal(size=field.shape)+1j*rng.normal(size=field.shape)
            variation /= np.linalg.norm(variation)
            step = 2e-6
            finite = (potential_gradient(field+step*variation, beta)[0]-potential_gradient(field-step*variation, beta)[0])/(2*step)
            close(finite, 2*np.vdot(gradient, variation).real,
                  'independent complete matrix action variation verifies the invariant source')
            close(potential_gradient(1.7*field, beta)[0], 1.7**2*value,
                  'completed invariant potential retains amplitude degree two')
            close(potential_gradient(1.7*field, beta)[1], 1.7*gradient,
                  'completed invariant source retains amplitude degree one')
            angle = .43
            boost = np.cosh(angle/2)*np.eye(rank)+np.sinh(angle/2)*alpha[0]
            close(boost.conj().T@beta@boost, beta,
                  'actual spinor boost preserves its indefinite scalar metric')
            transformed = invariants(boost@field, beta)
            close(transformed[1], right_scalar, 'entire right scalar Gram matrix is invariant under the actual Lorentz boost')
            close(transformed[-1], proper, 'completed proper density is Lorentz invariant')
            close(potential_gradient(boost@field, beta)[0], value,
                  'completed local potential is Lorentz invariant')
            close(boost.conj().T@potential_gradient(boost@field, beta)[1], gradient,
                  'complete nonlinear source has the Lorentz contragredient transformation required by its action')
            if depth == 2:
                currents = [np.vdot(field, a@field).real for a in alpha]
                ordinary = np.sqrt(rho*rho-sum(j*j for j in currents))
                close(proper, ordinary, 'depth two completed and vector proper densities coincide identically')
        for c in (.1, .3, .6, .8):
            x = np.sqrt(1-c*c)
            a, b = np.sqrt((1+c)/2), np.sqrt((1-c)/2)
            field = (a*np.eye(rank)+b*beta)/np.sqrt(rank)
            matrix, _, rho, signed, _, proper = invariants(field, beta)
            density_gradient = (rank*beta@matrix@beta-signed*beta)/proper
            close(proper, rho, 'balanced count background retains exactly the same density')
            close(density_gradient, np.eye(rank), 'completed density has the same first field gradient on the exact count background')
            close(signed/proper, x, 'completed action preserves the balanced scalar ratio')
            h = depth/2
            p, first, _ = radial.polynomial(depth, 0)
            epsilon = c/float(first)
            lam = h/c
            f, fp = epsilon*float(p)-h*c, h*x/c
            original = (fp*beta+(f-x*fp)*np.eye(rank))@field
            completed = (fp*beta+(f-x*fp)*density_gradient)@field
            close(original, completed, 'actual inverse count source and complete energy remain unchanged on the background')
        # Generic complex fields in the exact rotational matrix algebra
        # independently identify its additional tensor-current norm.
        gamma = alpha[0]
        tensor = -1j*gamma@beta
        algebra = np.stack([np.eye(rank), -1j*gamma, beta, tensor])/np.sqrt(rank)
        for sample in range(3):
            z = rng.normal(size=4)+1j*rng.normal(size=4)
            field = np.einsum('a,aij->ij', z, algebra)
            _, _, rho, signed, _, proper = invariants(field, beta)
            jr, jt = np.vdot(field, gamma@field).real, np.vdot(field, tensor@field).real
            close(proper*proper, rho*rho-jr*jr-jt*jt,
                  'rotational matrix algebra derives the additional tensor-current square')
        cases.append({'depth': depth, 'spinor_rank': rank,
                      'classification': 'full spinor matrix variations and Lorentz transformations'})
    for name, depth, index in radial.ASSIGNMENTS:
        matrices = radial.gammas(depth+1)
        gamma, beta0 = matrices[0], matrices[depth]
        rank, level = len(beta0), index-1
        last = max(n for _, d, n in radial.ASSIGNMENTS if d == depth)
        p, first, _ = radial.polynomial(depth, level)
        epsilon = .3/float(radial.polynomial(depth, last-1)[1])
        c = epsilon*float(first)
        x = np.sqrt(1-c*c)
        a, b = np.sqrt((1+c)/2), np.sqrt((1-c)/2)
        h = level+depth/2
        f, fp = epsilon*float(p)-h*c, h*x/c
        for theta in (.37, 1.23):
            t, s = np.cos(theta), np.sin(theta)
            beta = (t*np.eye(rank)+1j*s*gamma)@beta0
            aa = (level+depth-1)/(depth-1)*radial.eval_gegenbauer(level, (depth-1)/2, t)
            bb = radial.eval_gegenbauer(level-1, (depth+1)/2, t) if level else 0.
            positive = (aa*np.eye(rank)-1j*s*bb*gamma)/np.sqrt(rank*comb(level+depth-1, depth-1))
            field = (a*np.eye(rank)+b*beta)@positive
            matrix, _, rho, signed, _, proper = invariants(field, beta)
            density_gradient = (rank*beta@matrix@beta-signed*beta)/proper
            close(proper/rho, 1., 'every maintained actual spatial shell retains its original proper density')
            close(density_gradient, np.eye(rank), 'every maintained actual spatial shell retains its complete density gradient')
            original = (fp*beta+(f-x*fp)*np.eye(rank))@field
            completed = (fp*beta+(f-x*fp)*density_gradient)@field
            close(completed/max(1., np.linalg.norm(original)), original/max(1., np.linalg.norm(original)),
                  'completed local source preserves all fourteen exact spatial count branches')
    return cases


def symbolic_blend(depth, c_value):
    z, kappa = sp.symbols('z kappa')
    x = sp.Symbol('x', real=True)
    c = sp.Rational(c_value)
    lx, ly = ground.exact_block(depth, 1, c, x)
    if depth != 2:
        h = sp.Rational(depth, 2)
        harmonic = 1
        u, v = sp.Rational(depth, depth+1), sp.Rational(1, depth+1)
        aa, bb, ab = (1+c)/2, (1-c)/2, x/2
        tensor = sp.Matrix([[v*bb, -v*ab, -v*bb, v*ab],
                            [-v*ab, v*aa, v*ab, -v*aa],
                            [-u*bb, u*ab, u*bb, -u*ab],
                            [u*ab, -u*aa, -u*ab, u*aa]])
        first = sum(sp.Rational(1, j) for j in range(1, depth+1))
        vector = 1-c*c/(h*first)
        lx += 2*kappa*vector*tensor
    raw = sp.Poly((ly*lx).charpoly(z).as_expr(), x)
    reduced = sp.factor(sp.rem(raw, sp.Poly(x*x-1+c*c, x)).as_expr())
    check(not reduced.has(x), 'completed exact harmonic polynomial retains only the physical mixing ratio')
    check(reduced.subs(z, 0) == 0, 'completed matrix action retains the same zero symmetry direction')
    cubic = sp.Poly(sp.cancel(reduced/z), z)
    disc = sp.factor(sp.discriminant(cubic.as_expr(), z))
    coefficients = cubic.all_coeffs()
    # With three real roots, these elementary symmetric signs are
    # equivalent to nonnegative roots, including their zero limits.
    inequalities = [disc >= 0, -coefficients[1] >= 0,
                    coefficients[2] >= 0, -coefficients[3] >= 0]
    allowed = sp.reduce_inequalities(inequalities, kappa)
    convex_allowed = sp.reduce_inequalities(inequalities+[kappa >= 0, kappa <= 1], kappa)
    check(convex_allowed == sp.false,
          'every positive blend retains first-harmonic growth at both common tested preparations')
    completed = sp.Poly(cubic.as_expr().subs(kappa, 1), z)
    completed_disc = sp.discriminant(completed.as_expr(), z)
    if depth != 2:
        baseline = sp.Poly(cubic.as_expr().subs(kappa, 0), z)
        check(sp.discriminant(baseline.as_expr(), z) < 0,
              'exact blend baseline reproduces the already certified spatial instability')
    record = {'depth': depth, 'c_exact': str(c), 'x_squared_exact': str(1-c*c),
              'characteristic_polynomial': str(reduced),
              'cubic_discriminant': str(disc),
              'completed_cubic_discriminant': str(completed_disc),
              'real_blend_coefficients_with_nonnegative_first_harmonic_squared_frequencies': str(allowed),
              'convex_blend_coefficients_with_nonnegative_first_harmonic_squared_frequencies': str(convex_allowed),
              'classification': 'exact first spatial harmonic only; absence of its growth is not full stability'}
    print(json.dumps(record), flush=True)
    return record


def main():
    nonlinear = nonlinear_checks()
    symbolic = []
    for c in ['3/10', '3/5']:
        for depth in radial.DEPTHS:
            symbolic.append(symbolic_blend(depth, c))
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    report = {'experiment_key': 'lorentz-invariant-matrix-current-completion',
              'created_utc': datetime.now(timezone.utc).isoformat(),
              'sources': {'experiments/'+p.name: sha256(p.read_bytes()).hexdigest()
                          for p in [Path(__file__), ROOT/'inverse-current-ground-proof.py', ROOT/'inverse-current-radial.py']},
              'conditions': CONDITIONS, 'evaluations': sum(CONDITIONS.values()),
              'maximum_errors': ERRORS, 'nonlinear_cases': nonlinear, 'cases': symbolic,
              'limits': ['count constitutive function and balanced matrix carrier remain supplied premises',
                         'same stationary frequencies do not prove localization, observed spin or particle identity',
                         'exact first-harmonic stability intervals do not cover higher harmonics or assigned excited shells',
                         'negative blend weights are explicitly permitted in the comparison and require their own global analysis',
                         'no assigned-depth preference, observed color mechanism or full persistence is claimed']}
    output = ROOT/('matrix-current-completion-results-'+stamp+'.json')
    output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'report': str(output), 'evaluations': report['evaluations']}), flush=True)


if __name__ == '__main__':
    main()
