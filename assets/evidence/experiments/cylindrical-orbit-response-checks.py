#!/usr/bin/env python3
"""Test actual relative equilibria with a frozen lower-depth nuclear support."""

import hashlib
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
import sympy as sp


ROOT = Path(__file__).parent
SPEC = importlib.util.spec_from_file_location('canonical_orbit', ROOT/'canonical-orbit-response-checks.py')
BASE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BASE)
CHECKS = []


def zero(name, expression):
    values = list(expression) if isinstance(expression, sp.MatrixBase) else [expression]
    if any(sp.simplify(value) != 0 for value in values):
        raise AssertionError((name, expression))
    CHECKS.append(name)


def symbolic():
    w, n, M, g = sp.symbols('w n M g', positive=True)
    A = sp.diag(sp.Matrix([[0, -w], [w, 0]]), sp.Matrix([[0, -n], [n, 0]]))
    q = sp.Matrix([1, 0, 1, 0])
    v = A*q
    D = 1-M*(w*w+n*n)
    denominator = (1-M*(w+n)**2)*(1-M*(w-n)**2)
    c = g*n*(-1+M*(3*w*w+n*n))/denominator
    d = g*w*(1-M*(w*w+3*n*n))/denominator
    B = sp.Matrix([[0, -g*w, 0, c], [g*w, 0, d, 0],
                   [0, -d, 0, -g*n], [-c, 0, g*n, 0]])
    zero('complete rotating source response', B+M*(A*(A*B-B*A)-(A*B-B*A)*A)+g*(q*v.T-v*q.T))
    F = (-(w*w+n*n)+M*(w**4+n**4+6*w*w*n*n))/denominator
    zero('mixed response contraction', n*c-w*d-g*F)
    force = (A*A+g*(A*B+B*A))*q
    zero('visible force balance', force[0]-(-w*w*(1+2*g*g)+g*g*F))
    zero('unsupported force balance', force[2]-(-n*n*(1+2*g*g)+g*g*F))
    zero('equal rotation cannot bind unsupported circle', F.subs(n, w)+2*w*w)
    L, y = sp.symbols('L y', positive=True)
    f = (-(L*L+1)+y*(L**4+6*L*L+1))/((1-y*(L+1)**2)*(1-y*(L-1)**2))
    zero('response as sum and difference clocks', F.subs({w: L*n, M: y/n**2})-n**2*f)
    C = 2+1/g**2
    polynomial = C*(L*L-1)**2*y*y-(2*C*(L*L+1)+L**4+6*L*L+1)*y+C+L*L+1
    zero('equal-radius source equation', (C-f)*(1-y*(L+1)**2)*(1-y*(L-1)**2)-polynomial)
    q5 = sp.Matrix(sp.symbols('Q0:5'))
    support = sp.diag(1, 1, 1, 0, 0)
    radius = sp.sqrt(q5.dot(support*q5))
    force = -support*q5/radius**3
    potential = -1/radius
    zero('supported-space potential gradient', force+sp.Matrix([sp.diff(potential, z) for z in q5]))
    torque = q5*force.T-force*q5.T
    zero('supported rotational charge', torque[:3, :3])
    zero('unsupported rotational charge', torque[3:, 3:])
    return {'ratio_polynomial': str(polynomial), 'response': str(F),
            'source_resonances': 'M*(w+n)**2=1 or M*(w-n)**2=1'}


def rhs(time, state, depth, coupling, inertia, A):
    q, p, B, P = BASE.unpack(state, depth)
    support = np.zeros_like(q)
    support[:min(3, depth)] = q[:min(3, depth)]
    v = p-coupling*B@q
    qd = v-A@q
    pd = -coupling*B@v-support/(support@support)**1.5-A@p
    Bd = P/inertia-A@B+B@A
    Pd = -B-coupling*(np.outer(q, v)-np.outer(v, q))-A@P+P@A
    return np.concatenate([qd, pd, BASE.upper(Bd), BASE.upper(Pd)])


def equilibrium(depth, ratio, high, coupling=1.0):
    C = 2+1/coupling**2
    qa = C*(ratio**2-1)**2
    qb = -(2*C*(ratio**2+1)+ratio**4+6*ratio**2+1)
    qc = C+ratio**2+1
    discriminant = qb*qb-4*qa*qc
    y = (-qb+np.sqrt(discriminant))/(2*qa) if high else 2*qc/(-qb+np.sqrt(discriminant))
    nu = np.sqrt(1/((1+2*coupling**2)*(ratio**2-1)))
    omega = ratio*nu
    inertia = y/(nu*nu)
    A = np.zeros((depth, depth))
    A[0, 1], A[1, 0], A[3, 4], A[4, 3] = -omega, omega, -nu, nu
    q = np.zeros(depth)
    q[0], q[3] = 1, 1
    v = A@q
    basis = []
    for i, j in BASE.pairs(depth):
        E = np.zeros((depth, depth));E[i, j], E[j, i] = 1, -1
        basis.append(E)
    operator = np.column_stack([BASE.upper(E+inertia*(A@(A@E-E@A)-(A@E-E@A)@A)) for E in basis])
    b = np.linalg.solve(operator, -coupling*BASE.upper(np.outer(q, v)-np.outer(v, q)))
    B = BASE.skew(b, depth)
    P = inertia*(A@B-B@A)
    p = (A+coupling*B)@q
    state = np.concatenate([q, p, BASE.upper(B), BASE.upper(P)])
    return state, A, inertia, operator, omega, nu


def energy(state, depth, coupling, inertia):
    q, p, B, P = BASE.unpack(state, depth)
    v = p-coupling*B@q
    radius = np.linalg.norm(q[:min(3, depth)])
    return v@v/2+np.sum(BASE.upper(P)**2)/(2*inertia)+np.sum(BASE.upper(B)**2)/2-1/radius


if __name__ == '__main__':
    symbols = symbolic()
    print(json.dumps(symbols, indent=2), flush=True)
    cases = []
    structural = [{'depth': d, 'nuclear_support_rank': min(3, d),
                   'unsupported_rotation_planes': (d-min(3, d))//2,
                   'two-plane_branch_available': d >= 5} for d in (2, 3, 4, 5, 6, 10)]
    growth_runs = []
    conservation = []
    for depth in (2, 3, 4, 5, 6, 10):
        rng = np.random.default_rng(20261011+depth)
        size = depth*(depth-1)//2
        state = rng.normal(size=2*depth+2*size)*0.1
        state[0] += 1
        derivative = rhs(0, state, depth, 0.7, 1.3, np.zeros((depth, depth)))
        step = 2e-6
        high, low = state+step*derivative, state-step*derivative
        energy_defect = abs((energy(high, depth, 0.7, 1.3)-energy(low, depth, 0.7, 1.3))/(2*step))
        def charge(y):
            q, p, B, P = BASE.unpack(y, depth)
            return np.outer(q, p)-np.outer(p, q)-(B@P-P@B)
        torque = (charge(high)-charge(low))/(2*step)
        rank = min(3, depth)
        allowed_defect = max(np.linalg.norm(torque[:rank, :rank]), np.linalg.norm(torque[rank:, rank:]))
        if max(energy_defect, allowed_defect) > 2e-8:
            raise AssertionError(('energy or applicable rotational charge', depth, energy_defect, allowed_defect))
        conservation.append({'depth': depth, 'energy_derivative_defect': energy_defect,
                             'applicable_charge_derivative_defect': allowed_defect,
                             'full_charge_derivative_norm': float(np.linalg.norm(torque))})
    for ratio in (1.5, 2.0, 2.5, 3.0, 4.0):
        for high in (False, True):
            for depth in (5, 6, 10):
                state, A, inertia, operator, omega, nu = equilibrium(depth, ratio, high)
                residual = np.max(np.abs(rhs(0, state, depth, 1.0, inertia, A)))
                if residual > 2e-10:
                    raise AssertionError(('equilibrium', ratio, high, depth, residual))
                jac = np.column_stack([np.imag(rhs(0, state.astype(complex)+1e-25j*np.eye(len(state))[i],
                                                  depth, 1.0, inertia, A))/1e-25 for i in range(len(state))])
                direction = np.random.default_rng(20261011+depth).normal(size=len(state))
                direction /= np.linalg.norm(direction)
                step = 1e-6
                finite = (rhs(0, state+step*direction, depth, 1.0, inertia, A)-
                          rhs(0, state-step*direction, depth, 1.0, inertia, A))/(2*step)
                jacobian_defect = np.max(np.abs(finite-jac@direction))
                if jacobian_defect > 2e-8:
                    raise AssertionError(('independent Jacobian', ratio, high, depth, jacobian_defect))
                values, vectors = np.linalg.eig(jac)
                growth = float(max(values.real))
                cases.append({'depth': depth, 'ratio': ratio, 'branch': 'high-inertia' if high else 'low-inertia',
                              'source_inertia': inertia, 'orbit_clock': omega, 'unsupported_clock': nu,
                              'source_operator_condition': float(np.linalg.cond(operator)),
                              'equilibrium_residual': residual, 'max_real_eigenvalue': growth,
                              'independent_jacobian_defect': jacobian_defect,
                              'phase_space_dimension': len(state),
                              'eigenvalues': [[float(z.real), float(z.imag)] for z in values]})
                if ratio == 2 and depth == 6 and growth > 1e-5:
                    idx = int(np.argmax(values.real))
                    mode = vectors[:, idx]
                    initial = mode.real/np.linalg.norm(mode.real)
                    duration = 4/growth
                    amplitude = 1e-8
                    solution = solve_ivp(lambda t, y: rhs(t, y, depth, 1.0, inertia, A),
                                         (0, duration), state+amplitude*initial, method='DOP853',
                                         rtol=1e-12, atol=1e-13)
                    if not solution.success:
                        raise AssertionError(solution.message)
                    # Compare with the complete real linear flow, including any imaginary frequency.
                    from scipy.linalg import expm
                    expected = expm(jac*duration)@initial
                    actual = (solution.y[:, -1]-state)/amplitude
                    relative = np.linalg.norm(actual-expected)/np.linalg.norm(expected)
                    if relative > 2e-3:
                        raise AssertionError(('nonlinear unstable mode', high, relative))
                    growth_runs.append({'branch': 'high-inertia' if high else 'low-inertia',
                                        'depth': depth, 'ratio': ratio, 'duration': duration,
                                        'linear_amplification': float(np.linalg.norm(expected)),
                                        'relative_growth_error': float(relative),
                                        'energy_error': abs(energy(solution.y[:, -1], depth, 1, inertia)-
                                                            energy(solution.y[:, 0], depth, 1, inertia)),
                                        'rhs_evaluations': solution.nfev})
    path = Path(__file__)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    report = {'created_utc': stamp, 'sources': {
        'experiments/cylindrical-orbit-response-checks.py': hashlib.sha256(path.read_bytes()).hexdigest(),
        'experiments/canonical-orbit-response-checks.py': hashlib.sha256((ROOT/'canonical-orbit-response-checks.py').read_bytes()).hexdigest()},
        'exact_checks': CHECKS, 'symbolic': symbols, 'structural_controls': structural,
        'equilibria': cases, 'nonlinear_growth': growth_runs, 'conservation': conservation}
    destination = path.with_name('cylindrical-orbit-response-results-'+stamp+'.json')
    destination.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'report': str(destination), 'exact_checks': len(CHECKS), 'structural_controls': structural,
                      'cases': len(cases), 'resolved_unstable_cases': sum(c['max_real_eigenvalue'] > 1e-5 for c in cases),
                      'max_equilibrium_residual': max(c['equilibrium_residual'] for c in cases),
                      'max_independent_jacobian_defect': max(c['independent_jacobian_defect'] for c in cases),
                      'nonlinear_growth': growth_runs}, indent=2), flush=True)
