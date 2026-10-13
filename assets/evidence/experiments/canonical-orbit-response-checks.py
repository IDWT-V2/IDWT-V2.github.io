#!/usr/bin/env python3
"""Check a deterministic orbital coordinate and a canonical skew response."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
import sympy as sp


CHECKS = []


def zero(name, expression):
    if isinstance(expression, sp.MatrixBase):
        passed = all(sp.simplify(x) == 0 for x in expression)
    else:
        passed = sp.simplify(expression) == 0
    if not passed:
        raise AssertionError((name, expression))
    CHECKS.append(name)


def pairs(depth):
    return [(i, j) for i in range(depth) for j in range(i+1, depth)]


def skew(values, depth):
    result = np.zeros((depth, depth), dtype=np.asarray(values).dtype)
    for value, (i, j) in zip(values, pairs(depth)):
        result[i, j], result[j, i] = value, -value
    return result


def upper(matrix):
    return np.array([matrix[i, j] for i, j in pairs(len(matrix))])


def unpack(state, depth):
    size = depth*(depth-1)//2
    return (state[:depth], state[depth:2*depth],
            skew(state[2*depth:2*depth+size], depth),
            skew(state[2*depth+size:], depth))


def rhs(time, state, depth, coupling, inertia, rotating=False):
    q, p, B, P = unpack(state, depth)
    velocity = p-coupling*B@q
    qdot = velocity
    pdot = -coupling*B@velocity-q/(q@q)**1.5
    Bdot = P/inertia
    Pdot = -B-coupling*(np.outer(q, velocity)-np.outer(velocity, q))
    if rotating:
        omega = 1/np.sqrt(1+2*coupling**2)
        A = np.zeros((depth, depth))
        A[1, 0], A[0, 1] = omega, -omega
        qdot, pdot = qdot-A@q, pdot-A@p
        Bdot = Bdot-A@B+B@A
        Pdot = Pdot-A@P+P@A
    return np.concatenate([qdot, pdot, upper(Bdot), upper(Pdot)])


def energy_and_charge(state, depth, coupling, inertia):
    q, p, B, P = unpack(state, depth)
    v = p-coupling*B@q
    energy = v@v/2+np.sum(upper(P)**2)/(2*inertia)+np.sum(upper(B)**2)/2-1/np.sqrt(q@q)
    charge = np.outer(q, p)-np.outer(p, q)-(B@P-P@B)
    return energy, charge


def symbolic():
    g, M, omega = sp.symbols('g M omega', positive=True)
    z = sp.Symbol('z')
    r, ell, k, b = sp.symbols('r ell k b', positive=True)
    U = ell**2/(2*r**2)+g*b*ell+g**2*b**2*r**2/2+b**2/2-k/r
    bst = -g*ell/(1+g**2*r**2)
    zero('static response equation', sp.diff(U, b).subs(b, bst))
    reduced = sp.simplify(U.subs(b, bst))
    zero('reduced angular inertia', reduced-(ell**2/(2*r**2*(1+g**2*r**2))-k/r))
    circular = {r: 1, ell: (1+g**2)*omega, b: -g*omega, k: (1+2*g**2)*omega**2}
    zero('circular radial equation', sp.diff(U, r).subs(circular))
    hessian = sp.hessian(U, (r, b)).subs(circular)
    zero('radial Hessian', hessian[0, 0]-(1+2*g**2+4*g**4)*omega**2)
    zero('mixed Hessian', hessian[0, 1]+2*g**3*omega)
    zero('source Hessian', hessian[1, 1]-(1+g**2))
    ratio = (1+3*g**2+6*g**4)/(1+g**2)
    zero('positive Hessian determinant', hessian.det()-omega**2*(1+3*g**2+6*g**4))
    zero('adiabatic radial clock', hessian[0, 0]-hessian[0, 1]**2/hessian[1, 1]-omega**2*ratio)
    zero('four-to-one small-oscillation ratio', ratio.subs(g, sp.sqrt(3))-16)

    q = sp.Matrix(sp.symbols('q0:3'))
    p = sp.Matrix(sp.symbols('p0:3'))
    bs = sp.Matrix(sp.symbols('b0:3'))
    ps = sp.Matrix(sp.symbols('s0:3'))
    B = sp.zeros(3)
    P = sp.zeros(3)
    for a, (i, j) in enumerate(pairs(3)):
        B[i, j], B[j, i] = bs[a], -bs[a]
        P[i, j], P[j, i] = ps[a], -ps[a]
    v = p-g*B*q
    A = sp.Matrix([[0, -omega, 0], [omega, 0, 0], [0, 0, 0]])
    norm = (q.dot(q))**sp.Rational(3, 2)
    qd, pd = v-A*q, -g*B*v-k*q/norm-A*p
    Bd = P/M-A*B+B*A
    Pd = -B-g*(q*v.T-v*q.T)-A*P+P*A
    state = sp.Matrix.vstack(q, p, bs, ps)
    field = sp.Matrix.vstack(qd, pd, sp.Matrix([Bd[i, j] for i, j in pairs(3)]),
                             sp.Matrix([Pd[i, j] for i, j in pairs(3)]))
    equilibrium = dict(zip(state, [1, 0, 0, 0, (1+g**2)*omega, 0, -g*omega, 0, 0, 0, 0, 0]))
    equilibrium[k] = (1+2*g**2)*omega**2
    zero('full rotating equilibrium', field.subs(equilibrium))
    J = field.jacobian(state).subs(equilibrium)
    active = J.extract([0, 1, 3, 4, 6, 9], [0, 1, 3, 4, 6, 9])
    transverse = J.extract([2, 5, 7, 8, 10, 11], [2, 5, 7, 8, 10, 11])
    active_poly = sp.factor(active.charpoly(z).as_expr())
    transverse_poly = sp.factor(transverse.charpoly(z).as_expr())
    expected_active = z**2*(z**4+(hessian[0, 0]+hessian[1, 1]/M)*z**2+hessian.det()/M)
    zero('complete active characteristic polynomial', active_poly-expected_active)
    x, eta, s = sp.symbols('x eta s', positive=True)
    a2 = 2*(1+x)*eta**2+(x+2)*eta
    a0 = (eta-1)*((1+2*x)*eta-(1+x))
    normalized_transverse = eta**2*s**2+a2*s+a0
    expected_transverse = (z**2+omega**2)*normalized_transverse.subs(
        {x: g**2, eta: M*omega**2, s: z**2/omega**2})/M**2
    zero('complete transverse characteristic polynomial', transverse_poly-expected_transverse)
    zero('transverse discriminant', a2**2-4*eta**2*a0-
         eta**2*(4*x**2*eta**2+4*(x**2+6*x+4)*eta+x**2))
    zero('lower transverse threshold', a0.subs(eta, (1+x)/(1+2*x)))
    zero('upper transverse threshold', a0.subs(eta, 1))

    H = v.dot(v)/2+sum(y*y for y in ps)/(2*M)+sum(y*y for y in bs)/2-k/sp.sqrt(q.dot(q))
    canonical = sp.Matrix.vstack(sp.Matrix([sp.diff(H, y) for y in p]),
                                sp.Matrix([-sp.diff(H, y) for y in q]),
                                sp.Matrix([sp.diff(H, y) for y in ps]),
                                sp.Matrix([-sp.diff(H, y) for y in bs]))
    lab_Bd = P/M
    lab_Pd = -B-g*(q*v.T-v*q.T)
    lab = sp.Matrix.vstack(v, -g*B*v-k*q/norm,
                           sp.Matrix([lab_Bd[i, j] for i, j in pairs(3)]),
                           sp.Matrix([lab_Pd[i, j] for i, j in pairs(3)]))
    zero('all canonical Hamilton equations', canonical-lab)
    charge = q*p.T-p*q.T-(B*P-P*B)
    zero('exact total rotational charge', sp.Matrix([sum(sp.diff(charge[i, j], y)*dy for y, dy in zip(state, lab))
                                                     for i, j in pairs(3)]))
    zero('exact energy conservation', sum(sp.diff(H, y)*dy for y, dy in zip(state, lab)))
    return {'active_polynomial': str(active_poly), 'transverse_polynomial': str(transverse_poly),
            'radial_hessian': str(hessian), 'adiabatic_clock_ratio_squared': str(ratio),
            'normalized_transverse_quadratic': str(normalized_transverse),
            'instability_window': '(1+x)/(1+2*x) < M*omega**2 < 1, x=coupling**2 at unit radius'}


def numerical():
    equilibria = []
    differentials = []
    integrations = []
    growth_integrations = []
    rng = np.random.default_rng(20261011)
    for depth in (2, 3, 4, 5, 6, 10):
        size = depth*(depth-1)//2
        for coupling in (0.2, 1.0, np.sqrt(3)):
            for inertia in (0.05, 1.0, 20.0, 1+1.5*coupling**2):
                omega = 1/np.sqrt(1+2*coupling**2)
                state = np.zeros(2*depth+2*size)
                state[0], state[depth+1] = 1, (1+coupling**2)*omega
                state[2*depth] = -coupling*omega
                defect = np.max(np.abs(rhs(0, state, depth, coupling, inertia, True)))
                if defect > 3e-14:
                    raise AssertionError(('equilibrium', depth, defect))
                # Complex-step differentiation retains the full skew-coordinate complement.
                jac = np.column_stack([np.imag(rhs(0, state.astype(complex)+1e-25j*np.eye(len(state))[i],
                                                  depth, coupling, inertia, True))/1e-25 for i in range(len(state))])
                eig = np.linalg.eigvals(jac)
                # Each block retains its full multiplicity; no transverse mode is discarded.
                pair_indices = {pair: i for i, pair in enumerate(pairs(depth))}
                blocks = [[0, 1, depth, depth+1, 2*depth, 2*depth+size]]
                for axis in range(2, depth):
                    i, j = pair_indices[(0, axis)], pair_indices[(1, axis)]
                    blocks.append([axis, depth+axis, 2*depth+i, 2*depth+j, 2*depth+size+i, 2*depth+size+j])
                for i, j in pairs(depth):
                    if i >= 2:
                        a = pair_indices[(i, j)]
                        blocks.append([2*depth+a, 2*depth+size+a])
                retained = np.zeros_like(jac)
                for block in blocks:
                    retained[np.ix_(block, block)] = jac[np.ix_(block, block)]
                block_defect = np.max(np.abs(jac-retained))
                if block_defect > 3e-14 or sum(map(len, blocks)) != len(state):
                    raise AssertionError(('complete block decomposition', depth, block_defect))
                x = coupling**2
                eta = inertia*omega**2
                c2 = 2*(1+x)*eta**2+(x+2)*eta
                c0 = (eta-1)*((1+2*x)*eta-(1+x))
                discriminant = eta**2*(4*x*x*eta**2+4*(x*x+6*x+4)*eta+x*x)
                positive_s = -2*c0/(c2+np.sqrt(discriminant))
                growth = omega*np.sqrt(max(0, positive_s)) if depth > 2 else 0
                if abs(max(eig.real)-growth) > 2e-7:
                    raise AssertionError(('transverse growth prediction', depth, coupling, inertia, max(eig.real), growth))
                equilibria.append({'depth': depth, 'coupling': coupling, 'source_inertia': inertia,
                                   'clock': omega, 'dimension': len(state), 'equilibrium_defect': defect,
                                   'max_real_eigenvalue': float(np.max(eig.real)),
                                   'predicted_growth_rate': float(growth), 'block_defect': block_defect,
                                   'eigenvalues': [[float(e.real), float(e.imag)] for e in eig]})
                if coupling == 1.0 and inertia == 2.5 and depth > 2:
                    values, vectors = np.linalg.eig(jac)
                    index = int(np.argmax(values.real))
                    direction = vectors[:, index].real
                    direction /= np.linalg.norm(direction)
                    amplitude = 1e-7
                    duration = 6/growth
                    solution = solve_ivp(lambda t, y: rhs(t, y, depth, coupling, inertia, True),
                                         (0, duration), state+amplitude*direction, method='DOP853',
                                         rtol=3e-12, atol=3e-13)
                    if not solution.success:
                        raise AssertionError(solution.message)
                    amplification = np.linalg.norm(solution.y[:, -1]-state)/amplitude
                    relative_error = abs(amplification/np.exp(6)-1)
                    if relative_error > 2e-3:
                        raise AssertionError(('nonlinear growth', depth, amplification, relative_error))
                    initial_energy, initial_charge = energy_and_charge(solution.y[:, 0], depth, coupling, inertia)
                    final_energy, final_charge = energy_and_charge(solution.y[:, -1], depth, coupling, inertia)
                    growth_integrations.append({'depth': depth, 'growth_rate': growth, 'duration': duration,
                                                'amplification': amplification, 'relative_growth_error': relative_error,
                                                'energy_error': abs(final_energy-initial_energy),
                                                'charge_error': float(np.max(np.abs(final_charge-initial_charge))),
                                                'rhs_evaluations': solution.nfev})
        state = rng.normal(size=2*depth+2*size)*0.1
        state[:depth] += np.eye(depth)[0]
        tangent = rhs(0, state, depth, 0.7, 1.3)
        rotation, _ = np.linalg.qr(rng.normal(size=(depth, depth)))
        if np.linalg.det(rotation) < 0:
            rotation[:, 0] *= -1
        def rotate(y):
            q, p, B, P = unpack(y, depth)
            return np.concatenate([rotation@q, rotation@p, upper(rotation@B@rotation.T), upper(rotation@P@rotation.T)])
        covariance_defect = np.max(np.abs(rhs(0, rotate(state), depth, 0.7, 1.3)-rotate(tangent)))
        if covariance_defect > 3e-14:
            raise AssertionError(('rotation covariance', depth, covariance_defect))
        eps = 2e-6
        ep, jp = energy_and_charge(state+eps*tangent, depth, 0.7, 1.3)
        em, jm = energy_and_charge(state-eps*tangent, depth, 0.7, 1.3)
        ed, jd = abs((ep-em)/(2*eps)), np.max(np.abs((jp-jm)/(2*eps)))
        if max(ed, jd) > 1e-8:
            raise AssertionError(('conservation', depth, ed, jd))
        differentials.append({'depth': depth, 'energy_derivative_defect': ed, 'charge_derivative_defect': jd,
                              'rotation_covariance_defect': covariance_defect})

        # A short zero-source evolution tests induction and exact planarity, not atom fitting.
        state = np.zeros(2*depth+2*size)
        state[0], state[depth+1] = 1, 1
        initial_e, initial_j = energy_and_charge(state, depth, 0.4, 1.0)
        solution = solve_ivp(lambda t, y: rhs(t, y, depth, 0.4, 1.0), (0, 12), state,
                             method='DOP853', rtol=2e-11, atol=2e-12, t_eval=np.linspace(0, 12, 121))
        if not solution.success:
            raise AssertionError(solution.message)
        errors, charges, outside, strengths, radii = [], [], [], [], []
        for y in solution.y.T:
            energy, charge = energy_and_charge(y, depth, 0.4, 1.0)
            q, p, B, P = unpack(y, depth)
            errors.append(abs(energy-initial_e));charges.append(np.max(np.abs(charge-initial_j)))
            outside.append(max(np.linalg.norm(q[2:]), np.linalg.norm(B[2:, :]), np.linalg.norm(P[2:, :])))
            strengths.append(np.linalg.norm(upper(B)));radii.append(np.linalg.norm(q))
        if max(errors) > 2e-8 or max(charges) > 2e-8 or max(outside) > 1e-12 or max(strengths) < 0.1:
            raise AssertionError(('induction', depth, max(errors), max(charges), max(outside), max(strengths)))
        integrations.append({'depth': depth, 'max_energy_error': max(errors), 'max_charge_error': max(charges),
                             'max_outside_initial_plane': max(outside), 'max_induced_strength': max(strengths),
                             'radius_range': [min(radii), max(radii)], 'rhs_evaluations': solution.nfev})
    return equilibria, differentials, integrations, growth_integrations


if __name__ == '__main__':
    symbols = symbolic()
    print(json.dumps(symbols, indent=2), flush=True)
    equilibria, differentials, integrations, growth_integrations = numerical()
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    path = Path(__file__)
    report = {'created_utc': stamp, 'sources': {'experiments/canonical-orbit-response-checks.py': hashlib.sha256(path.read_bytes()).hexdigest()},
              'exact_checks': CHECKS, 'symbolic': symbols, 'equilibria': equilibria,
              'differential_conservation': differentials, 'zero_source_integrations': integrations,
              'nonlinear_growth_integrations': growth_integrations}
    destination = path.with_name('canonical-orbit-response-results-'+stamp+'.json')
    destination.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'report': str(destination), 'exact_checks': len(CHECKS), 'equilibria': len(equilibria),
                      'largest_real_part': max(x['max_real_eigenvalue'] for x in equilibria),
                      'predicted_unstable_cases': sum(x['predicted_growth_rate'] > 0 for x in equilibria),
                      'zero_source_integrations': integrations,
                      'nonlinear_growth_integrations': growth_integrations}, indent=2))
