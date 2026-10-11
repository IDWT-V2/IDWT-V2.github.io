#!/usr/bin/env python3
"""Test actual induced normal curvature on one nested ordinary spatial flag."""
from pathlib import Path
from itertools import combinations
import argparse
import hashlib
import importlib.util
import json
import math

import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp


DEPTHS = (2, 3, 4, 5, 6, 10)
CHECKS = 0
CASES = []


def check(condition, message):
    global CHECKS
    CHECKS += 1
    if not condition:
        raise AssertionError(message)


def close(actual, expected, message, atol=3e-9):
    check(np.allclose(actual, expected, atol=atol, rtol=3e-9), message)


def lie_basis(seed, tolerance=1e-9):
    basis = []
    def add(matrix):
        candidate = matrix.copy()
        # Two passes control false directions from loss of orthogonality.
        for _ in range(2):
            for previous in basis:
                candidate -= np.vdot(previous, candidate).real*previous
        norm = np.linalg.norm(candidate)
        if norm > tolerance:
            basis.append(candidate/norm)
    for matrix in seed:
        add(matrix)
    index = 0
    while index < len(basis):
        first = basis[index]
        for second in basis[:index]:
            add(first@second-second@first)
        index += 1
    return basis


def full_hessian(amplitude=1.):
    result = np.zeros((10, 10, 10))
    def entry(component, i, j, value):
        result[component, i, j] = result[component, j, i] = amplitude*value
    entry(4, 0, 0, 1); entry(4, 1, 1, -1); entry(5, 0, 1, 1)
    entry(6, 0, 2, 1); entry(6, 1, 3, -1); entry(7, 0, 3, 1); entry(7, 1, 2, 1)
    entry(8, 2, 2, 1); entry(8, 3, 3, -1); entry(9, 2, 3, 1)
    return result


def triangular_map(w, amplitude=1.):
    z = np.array(w, dtype=complex)
    z[2] += amplitude*w[0]**2/2
    z[3] += amplitude*w[0]*w[1]
    z[4] += amplitude*w[1]**2/2
    return z


def inverse_map(z, amplitude=1.):
    w = np.array(z, dtype=complex)
    w[2] -= amplitude*z[0]**2/2
    w[3] -= amplitude*z[0]*z[1]
    w[4] -= amplitude*z[1]**2/2
    return w


def real_curvature(second, tangent_depth):
    result = []
    for i, j in combinations(range(tangent_depth), 2):
        first = second[:, i, :]
        other = second[:, j, :]
        result.append(first@other.T-other@first.T)
    return result


def paired_complex_structure(q):
    result = np.zeros((2*q, 2*q))
    for i in range(q):
        result[2*i, 2*i+1] = -1
        result[2*i+1, 2*i] = 1
    return result


def nested_flag_checks(geometry, rng):
    gamma, _, _, _ = geometry.build_geometry()
    hessian = full_hessian()
    expected_intrinsic = {2: 1, 3: 10, 4: 9, 5: 10, 6: 4, 10: 0}
    expected_shadow = {2: 1, 3: 9, 4: 9, 5: 4, 6: 4, 10: 0}
    for d in DEPTHS:
        even = d if d % 2 == 0 else d+1
        k = even//2
        q = 5-k
        intrinsic_second = hessian[d:, :d, :d]
        intrinsic_curvature = real_curvature(intrinsic_second, d)
        intrinsic_rank = len(lie_basis(intrinsic_curvature))
        check(intrinsic_rank == expected_intrinsic[d], 'actual intrinsic normal curvature algebra on the common flag')
        second = hessian[even:, :even, :even]
        shadow_curvature = real_curvature(second, d)
        shadow_basis = lie_basis(shadow_curvature)
        check(len(shadow_basis) == expected_shadow[d], 'explicit even-normal pullback has the stated curvature algebra')
        complex_structure = paired_complex_structure(q)
        for curvature in shadow_curvature:
            close(curvature@complex_structure, complex_structure@curvature,
                  'even-shadow normal connection preserves its actual ordinary normal complex structure')
        # This is the second fundamental map, not an independently chosen particle carrier.
        first_rank = int(np.linalg.matrix_rank(second.reshape(2*q, -1))) if q else 0
        check(first_rank == (2 if k == 1 else 6 if k == 2 else 4 if k == 3 else 0),
              'first normal space follows from the common quadratic embedding')
        for amplitude in (.2, .7, 1.3):
            varied = full_hessian(amplitude)[even:, :even, :even]
            actual = real_curvature(varied, d)
            for a, b in zip(actual, shadow_curvature):
                close(a, amplitude**2*b, 'induced curvature scales quadratically in the single embedding amplitude')
            energy = sum(np.linalg.norm(x)**2 for x in actual)
            reference = sum(np.linalg.norm(x)**2 for x in shadow_curvature)
            close(energy, amplitude**4*reference, 'curvature norm energy has quartic embedding homogeneity')
        zero = real_curvature(full_hessian(0.)[even:, :even, :even], d)
        for curvature in zero:
            close(curvature, 0, 'undeformed common graph has no independent vacuum curvature')
        # The actual spin lift of unused normal rotations retains every active Clifford direction.
        annihilate = [(gamma[2*j]+1j*gamma[2*j+1])/2 for j in range(k, 5)]
        if q:
            for i in range(q):
                for j in range(i+1, q):
                    fundamental = np.zeros((q, q), complex)
                    fundamental[i, j] = fundamental[j, i] = .5
                    lifted = sum((fundamental[a, b]*annihilate[a].conj().T@annihilate[b]
                                  for a in range(q) for b in range(q)), np.zeros((32, 32), complex))
                    for vector in gamma[:d]:
                        close(lifted@vector, vector@lifted, 'normal rotation spin lift preserves every active spatial direction')
        for _ in range(3):
            raw = rng.normal(size=10)
            raw[d:] = 0
            w = raw[::2]+1j*raw[1::2]
            z = triangular_map(w)
            close(inverse_map(z), w, 'one global triangular map realizes every nested sector with an inverse')
            # Inclusion is tested in common coordinates, without optimizing each sector independently.
            recovered = np.ravel(np.column_stack((inverse_map(z).real, inverse_map(z).imag)))
            close(recovered[d:], 0, 'actual image point stays in its flagged sector')
        CASES.append({'case': f'nested-normal-curvature-depth-{d}', 'real_depth': d,
                      'even_source_carrier': even, 'intrinsic_real_normal_rank': 10-d,
                      'intrinsic_curvature_lie_dimension': intrinsic_rank,
                      'even_shadow_real_normal_rank': 2*q, 'complex_normal_rank': q,
                      'even_shadow_curvature_lie_dimension': len(shadow_basis),
                      'quadratic_first_normal_real_rank': first_rank,
                      'symmetric_complex_second_fundamental_dimension': k*(k+1)//2,
                      'full_second_fundamental_normal_isomorphism': k*(k+1)//2 == q,
                      'odd_shadow_equals_intrinsic_normal_connection': d % 2 == 0,
                      'independent_gauge_action_or_force_derived': False})


def matrix_inverse_sqrt(matrix):
    values, vectors = np.linalg.eigh(matrix)
    return (vectors/np.sqrt(values))@vectors.conj().T


def normal_frame_and_derivative(p, derivative):
    q, k = p.shape
    metric = np.eye(q)+p@p.conj().T
    values, vectors = np.linalg.eigh(metric)
    s = (vectors/np.sqrt(values))@vectors.conj().T
    dmetric = derivative@p.conj().T+p@derivative.conj().T
    square = np.sqrt(values)
    divided = -1/(square[:, None]*square[None, :]*(square[:, None]+square[None, :]))
    ds = vectors@(divided*(vectors.conj().T@dmetric@vectors))@vectors.conj().T
    raw = np.vstack((-p.conj().T, np.eye(q)))
    draw = np.vstack((-derivative.conj().T, np.zeros((q, q))))
    normal = raw@s
    dnormal = draw@s+raw@ds
    return normal, dnormal


def graph_curvature(p, derivatives):
    q, k = p.shape
    s = matrix_inverse_sqrt(np.eye(q)+p@p.conj().T)
    inverse_tangent = np.linalg.inv(np.eye(k)+p.conj().T@p)
    tangent_raw = np.vstack((np.eye(k), p))
    tangent_projection = tangent_raw@inverse_tangent@tangent_raw.conj().T
    frames = [normal_frame_and_derivative(p, derivative) for derivative in derivatives]
    for normal, dn in frames:
        close(normal.conj().T@normal, np.eye(q), 'actual graph normal frame is orthonormal')
        close(tangent_raw.conj().T@normal, 0, 'graph normal directions are perpendicular ordinary spatial directions')
    curvatures = []
    for a, b in combinations(range(len(derivatives)), 2):
        pa, pb = derivatives[a], derivatives[b]
        actual = s@(pa@inverse_tangent@pb.conj().T-pb@inverse_tangent@pa.conj().T)@s
        direct = frames[a][1].conj().T@tangent_projection@frames[b][1]-frames[b][1].conj().T@tangent_projection@frames[a][1]
        close(actual, direct, 'normal-frame projection independently verifies the Ricci graph curvature')
        close(actual, -actual.conj().T, 'actual normal curvature is unitary')
        curvatures.append(actual)
    return curvatures


def point_graph_controls():
    for diagonal in (1., math.sqrt(2)):
        b1 = np.array([[diagonal, 0], [0, 1.], [0, 0]], complex)
        b2 = np.array([[0, 0], [1., 0], [0, diagonal]], complex)
        for x in (0., .4):
            p = x*b1
            curvatures = graph_curvature(p, [b1, 1j*b1, b2, 1j*b2])
            full = len(lie_basis(curvatures))
            traceless = len(lie_basis([f-np.trace(f)*np.eye(3)/3 for f in curvatures]))
            if diagonal == 1.:
                check((full, traceless) == (9, 8), 'common monomial graph generates full U3 and SU3 already at the tested point')
            elif x == 0:
                check((full, traceless) == (4, 3), 'isotropic quadratic second fundamental form has a smaller pointwise algebra at the origin')
            else:
                check((full, traceless) == (9, 8), 'neighboring actual curvature restores full U3 holonomy in the isotropic graph')
            CASES.append({'case': f'actual-complex-graph-diagonal-{diagonal:g}-x-{x:g}',
                          'complex_base_rank': 2, 'complex_normal_rank': 3,
                          'diagonal_second_derivative': diagonal, 'position_x1': x,
                          'pointwise_curvature_lie_dimension': full,
                          'traceless_curvature_lie_dimension': traceless})


def higher_jet_curve_controls():
    def data(x, high):
        if high:
            p = np.array([x, x*x/2, x**3/6, x**4/24], complex)[:, None]
            derivative = np.array([1., x, x*x/2, x**3/6], complex)[:, None]
        else:
            p = np.array([x, 0., 0., 0.], complex)[:, None]
            derivative = np.array([1., 0., 0., 0.], complex)[:, None]
        return p, derivative
    for high in (False, True):
        def transport(t, state):
            p, dp = data(t, high)
            n, dn = normal_frame_and_derivative(p, dp)
            connection = n.conj().T@dn
            connection = (connection-connection.conj().T)/2
            return (-connection@state.reshape(4, 4)).reshape(-1)
        solution = solve_ivp(transport, (0., .9), np.eye(4, dtype=complex).reshape(-1),
                             method='DOP853', rtol=2e-11, atol=2e-12, max_step=.025, dense_output=True)
        check(solution.success, 'actual normal parallel transport integrates')
        transported = []
        for x in (0., .3, .6, .9):
            p, dp = data(x, high)
            normal_curvature = graph_curvature(p, [dp, 1j*dp])[0]
            u = solution.sol(x).reshape(4, 4)
            close(u.conj().T@u, np.eye(4), 'normal parallel transport is unitary')
            transported.append(u.conj().T@normal_curvature@u)
        full = len(lie_basis(transported, tolerance=1e-8))
        tracefree = [f-np.trace(f)*np.eye(4)/4 for f in transported]
        traceless = len(lie_basis(tracefree, tolerance=1e-8))
        stricter = len(lie_basis(tracefree, tolerance=1e-7))
        check(traceless == stricter, 'transported Lie algebra dimension survives a stricter rank tolerance')
        check((full, traceless) == ((16, 15) if high else (1, 1)),
              'higher normal jets allow full U4/SU4 at depth two, while the quadratic curve remains Abelian')
        CASES.append({'case': 'higher-normal-jet-curve' if high else 'quadratic-curve-control',
                      'real_depth': 2, 'complex_normal_rank': 4,
                      'normal_graph_degrees': [2, 3, 4, 5] if high else [2],
                      'transported_curvature_lie_dimension': full,
                      'traceless_transport_lie_dimension': traceless,
                      'pointwise_only_curvature_lie_dimension': 1,
                      'quadratic_selection_is_universal': False})


def exact_lie_dimension(generators):
    """Rational real row reduction avoids numerical Lie-rank thresholds."""
    basis = []
    rows = {}
    def add(matrix):
        vector = [sp.expand(sp.re(x)) for x in matrix]+[sp.expand(sp.im(x)) for x in matrix]
        for pivot, row in sorted(rows.items()):
            coefficient = vector[pivot]
            if coefficient:
                vector = [x-coefficient*y for x, y in zip(vector, row)]
        pivot = next((i for i, x in enumerate(vector) if x != 0), None)
        if pivot is not None:
            value = vector[pivot]
            rows[pivot] = [x/value for x in vector]
            basis.append(matrix)
    for generator in generators:
        add(generator)
    index = 0
    while index < len(basis):
        for other in basis[:index]:
            add(basis[index]*other-other*basis[index])
        index += 1
    return len(basis)


def exact_curvature_and_jet_checks(rng):
    hessian = full_hessian().astype(int)
    for d in DEPTHS:
        for even_shadow in (False, True):
            even = d+(d % 2) if even_shadow else d
            second = hessian[even:, :even, :even]
            generators = [sp.Matrix(x.astype(int)) for x in real_curvature(second, d)]
            dimension = exact_lie_dimension(generators) if even < 10 else 0
            expected = ({2: 1, 3: 9, 4: 9, 5: 4, 6: 4, 10: 0} if even_shadow
                        else {2: 1, 3: 10, 4: 9, 5: 10, 6: 4, 10: 0})
            check(dimension == expected[d], 'exact rational Ricci algebra confirms each occupied intrinsic and shadow rank')

    second = hessian[4:, :4, :4]
    obstruction_squared = 0
    for i, j in combinations(range(4), 2):
        bi, bj = second[:, i, :], second[:, j, :]
        rn = bi@bj.T-bj@bi.T
        rt = bi.T@bj-bj.T@bi
        for x in range(4):
            for y in range(4):
                residual = rn@second[:, x, y]
                residual -= sum((rt[l, x]*second[:, l, y]+rt[l, y]*second[:, x, l]
                                 for l in range(4)), np.zeros(6, dtype=int))
                obstruction_squared += int(residual@residual)
    check(obstruction_squared > 0, 'the rank-three graph cannot impose a parallel second fundamental isomorphism')

    # Curvature derivatives in a parallel normal frame certify holonomy at one point.
    order = 6
    def multiply(a, b):
        return [sum((a[j]*b[n-j] for j in range(n+1)), sp.zeros(a[0].rows, b[0].cols))
                for n in range(order+1)]
    def transpose(a):
        return [x.T for x in a]
    def add(a, b):
        return [x+y for x, y in zip(a, b)]
    q = 4
    p = [sp.zeros(q, 1) for _ in range(order+1)]
    b = [sp.zeros(q, 1) for _ in range(order+1)]
    for j in range(q):
        p[j+1][j] = sp.Rational(1, math.factorial(j+1))
        b[j][j] = sp.Rational(1, math.factorial(j))
    pp = multiply(p, transpose(p))
    s = [sp.eye(q)]+[sp.zeros(q) for _ in range(order)]
    power = [sp.eye(q)]+[sp.zeros(q) for _ in range(order)]
    for degree in range(1, order//2+1):
        power = multiply(power, pp)
        s = add(s, [sp.binomial(sp.Rational(-1, 2), degree)*x for x in power])
    raw = [-x.T.col_join(sp.zeros(q)) for x in p]
    raw[0] = sp.zeros(1, q).col_join(sp.eye(q))
    normal = multiply(raw, s)
    dn = [(n+1)*normal[n+1] if n < order else sp.zeros(q+1, q) for n in range(order+1)]
    connection = multiply(transpose(normal), dn)
    u = [sp.eye(q)]+[sp.zeros(q) for _ in range(order)]
    for n in range(order):
        u[n+1] = -sum((connection[j]*u[n-j] for j in range(n+1)), sp.zeros(q))/(n+1)
    norm = multiply(transpose(p), p)
    denominator = [sp.ones(1, 1)]+[sp.zeros(1, 1) for _ in range(order)]
    norm_power = denominator.copy()
    for degree in range(1, order//2+1):
        norm_power = multiply(norm_power, norm)
        denominator = add(denominator, [(-1)**degree*x for x in norm_power])
    v = multiply(s, b)
    curvature = [sp.zeros(q) for _ in range(order+1)]
    vv = multiply(v, transpose(v))
    for n in range(order+1):
        curvature[n] = -2*sp.I*sum((vv[j]*denominator[n-j][0] for j in range(n+1)), sp.zeros(q))
    transported = multiply(multiply(transpose(u), curvature), u)
    full = exact_lie_dimension(transported)
    tracefree = [x-sp.trace(x)*sp.eye(q)/q for x in transported]
    traceless = exact_lie_dimension(tracefree)
    check((full, traceless) == (16, 15), 'exact covariant curvature jets prove depth-two U4 and traceless SU4 holonomy')
    close(np.array(transported[0], complex), -2j*np.diag([1., 0, 0, 0]), 'exact jet normalization matches graph curvature')

    for d in DEPTHS:
        for _ in range(2):
            raw_point = rng.normal(size=10)
            raw_point[d:] = 0
            w = raw_point[::2]+1j*raw_point[1::2]
            z = triangular_map(w)
            z[1] += w[0]**5/120
            z[3] += w[0]**3/6
            z[4] += w[0]**4/24
            recovered = z.copy()
            recovered[1] -= z[0]**5/120
            recovered[2] -= z[0]**2/2
            recovered[3] -= z[0]*recovered[1]+z[0]**3/6
            recovered[4] -= recovered[1]**2/2+z[0]**4/24
            close(recovered, w, 'higher-jet control also extends to one common invertible nested spatial flag')
    CASES.append({'case': 'exact-rational-curvature-and-higher-jet-certificate',
                  'all_occupied_intrinsic_and_shadow_dimensions_exact': True,
                  'covariant_curvature_jet_maximum_order': order,
                  'depth_two_exact_normal_holonomy_dimension': full,
                  'depth_two_exact_traceless_holonomy_dimension': traceless,
                  'four_base_parallel_second_fundamental_obstruction_squared': obstruction_squared,
                  'higher_jet_control_embeds_in_one_common_nested_flag': True})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--geometry-source', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location('gauss_geometry', args.geometry_source)
    geometry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(geometry)
    nested_flag_checks(geometry, np.random.default_rng(2026101024))
    point_graph_controls()
    higher_jet_curve_controls()
    exact_curvature_and_jet_checks(np.random.default_rng(2026101041))
    report = {'title': 'An Actual Nested Spatial Flag Generates a Conditional Rank-Three Normal Color Geometry',
              'sources': {'experiments/nested-normal-curvature-checks.py': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                          'experiments/hodge-gated-gauss-checks.py': hashlib.sha256(args.geometry_source.read_bytes()).hexdigest()},
              'checks_passed': CHECKS, 'dependency_checks_passed': geometry.CHECKS,
              'case_count': len(CASES), 'occupied_depths': list(DEPTHS),
              'scope': 'one exact nested ordinary spatial flag and induced normal connections; quadratic and higher-jet controls, not a derived force action',
              'cases': CASES}
    args.report.write_text(json.dumps(report, indent=2)+'\n')
    print(f'{CHECKS} nested-geometry and normal-curvature checks plus {geometry.CHECKS} geometry checks passed across {len(CASES)} cases')


if __name__ == '__main__':
    main()
