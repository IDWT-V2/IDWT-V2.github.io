#!/usr/bin/env python3
"""Exact retained-normal-structure checks; read-only inputs, no database access."""
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
import hashlib
import argparse
import json
import sympy as s


TASK = Path(__file__).resolve().parent
REPO = next(p for p in TASK.parents if (p/"database"/"research.py").is_file())
CHECKS = 0


def check(condition, label):
    global CHECKS
    CHECKS += 1
    if not condition:
        raise AssertionError(label)


def comm(x, y):
    return x*y-y*x


def inner(x, y):
    return s.trace(x.T*y)


def norm2(x):
    return inner(x, x)


def jmat(q):
    j = s.zeros(2*q)
    for r in range(q):
        j[2*r, 2*r+1] = -1
        j[2*r+1, 2*r] = 1
    return j


def skew(n, i, j):
    x = s.zeros(n)
    x[i, j], x[j, i] = 1, -1
    return x


def ph(x, j):
    return (x-j*x*j)/2


def pm(x, j):
    return (x+j*x*j)/2


def vec(x):
    return x.reshape(x.rows*x.cols, 1)


def independent_matrices(matrices, n):
    if not matrices:
        return []
    cols = s.Matrix.hstack(*(vec(x) for x in matrices)).columnspace()
    return [x.reshape(n, n) for x in cols]


sources = {'experiments/normal-structure-response-checks.py': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}

rank_cases = []
for q in (4, 3, 2, 0):
    n = 2*q
    j = jmat(q)
    ident = s.eye(n)
    check(j.T == -j, 'J is skew')
    check(j*j == -ident, 'J squares to minus identity')
    basis = [skew(n, i, k) for i, k in combinations(range(n), 2)]
    hs = [ph(x, j) for x in basis]
    ms = [pm(x, j) for x in basis]
    hb = independent_matrices(hs, n)
    mb = independent_matrices(ms, n)
    check(len(hb) == q*q, 'commuting rank')
    check(len(mb) == q*(q-1), 'complement rank')
    check(len(basis) == q*(2*q-1), 'full skew rank')
    check(len(hb)+len(mb) == len(basis), 'direct sum dimension')
    for x, h, m in zip(basis, hs, ms):
        check(h+m == x, 'splitting reconstructs X')
        check(h.T == -h and m.T == -m, 'splitting is metric')
        check(comm(h, j) == s.zeros(n), 'h commutes with J')
        check(m*j+j*m == s.zeros(n), 'm anticommutes with J')
        check(ph(h, j) == h and pm(m, j) == m, 'idempotent projections')
        check(pm(h, j) == s.zeros(n) and ph(m, j) == s.zeros(n),
              'orthogonal projection composition')
        check(comm(comm(x, j), j) == -4*m, 'commutator quadratic eigenvalue')
        check(norm2(comm(x, j)) == 4*norm2(m), 'commutator norm identity')
        check(norm2(x) == norm2(h)+norm2(m), 'Pythagorean splitting')
    for h in hb:
        for m in mb:
            check(inner(h, m) == 0, 'h and m are orthogonal')
            check(ph(comm(h, m), j) == s.zeros(n), '[h,m] lies in m')
    for h1, h2 in combinations(hb, 2):
        check(pm(comm(h1, h2), j) == s.zeros(n), '[h,h] lies in h')
        check(s.trace(j*comm(h1, h2)) == 0, 'commutator has zero complex trace')
    for m1, m2 in combinations(mb, 2):
        check(pm(comm(m1, m2), j) == s.zeros(n), '[m,m] lies in h')

    if q:
        su = [h-inner(j, h)*j/(2*q) for h in hb]
        sub = independent_matrices(su, n)
        check(len(sub) == q*q-1, 'traceless unitary rank')
        for h in hb:
            central = inner(j, h)*j/(2*q)
            traceless = h-central
            check(s.trace(j*traceless) == 0, 'zero complex trace')
            check(inner(central, traceless) == 0, 'su and central are orthogonal')
            check(comm(j, h) == s.zeros(n), 'central J is central')
        # Exact spectrum on the skew basis, whose upper entries are coordinates.
        coords = list(combinations(range(n), 2))
        quad = s.Matrix.hstack(*(s.Matrix([
            (-comm(comm(x, j), j))[i, k] for i, k in coords
        ]) for x in basis))
        eig = {str(value): int(mult) for value, mult in quad.eigenvals().items()}
        expected_eig = {'0': q*q, '4': q*(q-1)}
        check(eig == expected_eig, 'full exact positive-commutator spectrum')
        # The orientation tangent map has no kernel on m.
        tangent = independent_matrices([comm(x, j) for x in mb], n)
        check(len(tangent) == q*(q-1), 'orientation coordinate count')
    else:
        eig = {}
        check(not basis and not hb and not mb, 'empty normal boundary')
    rank_cases.append({'q': q, 'real_normal_rank': n,
                       'so_dimension': len(basis), 'u_dimension': len(hb),
                       'su_dimension': q*q-1 if q else 0,
                       'central_dimension': 1 if q else 0,
                       'orientation_dimension': len(mb),
                       'commutator_quadratic_eigenvalues': eig})

# Derive the reviewed depth-four second fundamental matrices from the graph.
x = s.symbols('x0:4', real=True)
z1, z2 = x[0]+s.I*x[1], x[2]+s.I*x[3]
f_complex = (z1*z1/2, z1*z2, z2*z2/2)
f_real = [v.expand(complex=True) for z in f_complex for v in (s.re(z), s.im(z))]
B = [s.Matrix([[s.diff(y, x[i], x[k]) for k in range(4)]
               for y in f_real]) for i in range(4)]
j = jmat(3)
for i, k in combinations(range(4), 2):
    fij = B[i]*B[k].T-B[k]*B[i].T
    check(comm(fij, j) == s.zeros(6), 'static graph curvature is unitary')
eta, n = s.eye(6)[:, 0], s.eye(4)[:, 0]
check(B[2]*n == s.eye(6)[:, 2], 'reviewed axial bending vector')
w2 = eta*(B[2]*n).T-(B[2]*n)*eta.T
check(w2 == skew(6, 0, 2), 'reviewed W2 exact matrix')
check(norm2(w2) == 2, 'W2 norm')
check(norm2(comm(w2, j)) == 4, 'reviewed nonunitary electric curvature')
check(norm2(pm(w2, j)) == 1, 'W2 nonunitary projection norm')
check(norm2(ph(w2, j)) == 1, 'W2 unitary projection norm')
delta_j = comm(skew(6, 0, 4)+skew(6, 1, 3), j)
f0i = s.zeros(6)
check(comm(w2, j)+comm(f0i, delta_j) == comm(w2, j),
      'zero background electric curvature cannot cancel first variation')
check(comm(w2, j) != s.zeros(6), 'linear parallel continuation obstructed')

# Verify the actual compatible connection, its derivative correction and curvature.
# Use J(s,t)=exp(sX) exp(tY) J0 exp(-tY) exp(-sX) at s=t=0.
# First and mixed jets are exact; no matrix exponential approximation is used.
X = skew(6, 0, 2)+skew(6, 1, 4)
Y = skew(6, 0, 5)-skew(6, 2, 3)
A = skew(6, 0, 1)+2*skew(6, 0, 3)-skew(6, 2, 5)
C = skew(6, 1, 4)+skew(6, 2, 3)+skew(6, 3, 5)
js, jt = comm(X, j), comm(Y, j)
jst = comm(X, comm(Y, j))
ks, kt = js+comm(A, j), jt+comm(C, j)
for k in (ks, kt):
    check(k.T == -k and k*j+j*k == s.zeros(6), 'DJ tangent constraints')
    corr = -j*k/2
    check(corr.T == -corr, 'projected connection is metric')
    check(ph(corr, j) == s.zeros(6), 'projection correction is orthogonal')
    check(k+comm(corr, j) == s.zeros(6), 'projection makes J parallel')
    check(comm(j, k) == 2*j*k, 'vacuum algebraic orientation equation')
    check(norm2(comm(j, k)) == 4*norm2(k), 'vacuum current is injective on DJ')
corr_s, corr_t = -j*ks/2, -j*kt/2
check(A+corr_s == ph(A, j)-j*js/2, 'varying-J connection formula')
check(C+corr_t == ph(C, j)-j*jt/2, 'second varying-J connection formula')
ds_kt = jst+comm(C, js)
dt_ks = jst+comm(A, jt)
ds_corr_t = -(js*kt+j*ds_kt)/2
dt_corr_s = -(jt*ks+j*dt_ks)/2
F = comm(A, C)
check(ds_kt+comm(A, kt)-dt_ks-comm(C, ks) == comm(F, j),
      'curvature integrability identity')
Fhat_direct = (F+ds_corr_t-dt_corr_s+comm(A, corr_t)
               +comm(corr_s, C)+comm(corr_s, corr_t))
Fhat_formula = ph(F, j)-comm(ks, kt)/4
check(Fhat_direct == Fhat_formula, 'projected curvature exact formula')
check(comm(Fhat_direct, j) == s.zeros(6), 'projected curvature preserves J')

# Explicit first-order orientation kinetic normalization on every m basis
# is already verified by norm identity; verify it on one mixed orientation jet.
theta_jet = pm(X+2*Y, j)
a_complement = pm(A, j)
linear_K = comm(theta_jet+a_complement, j)
check(norm2(linear_K)/2 == 2*norm2(theta_jet+a_complement),
      'orientation quadratic kinetic coefficient')

# Common Cartan history is traceless unitary and leaves the entire J term zero.
j1, j2 = s.zeros(6), s.zeros(6)
j1[:2, :2] = jmat(1)
j2[2:4, 2:4] = jmat(1)
T = (j1-j2)/2
check(norm2(T) == 1, 'common Cartan Frobenius normalization')
check(comm(T, j) == s.zeros(6), 'common Cartan preserves J')
check(s.trace(j*T) == 0, 'common Cartan is already in su3')
t, x1, x2 = s.symbols('t x1 x2', real=True)
b = s.Function('b')(t)
A2 = b*x1*T
F12, F02 = A2.diff(x1), A2.diff(t)
check(F12 == b*T, 'Cartan magnetic history')
check(F02 == b.diff(t)*x1*T, 'Cartan electric history')
check(F12.diff(t)-F02.diff(x1) == s.zeros(6), 'Cartan Bianchi identity')
check(comm(A2, j) == s.zeros(6), 'Cartan DJ stays exactly zero')
check(comm(F12, j) == s.zeros(6) and comm(F02, j) == s.zeros(6),
      'Cartan curvature obstruction vanishes')
check(F02.diff(x2) == s.zeros(6), 'Cartan vacuum divergence')
u, kappa, h, settled_B, omega = s.symbols('u kappa h B Omega', real=True)
epsilon = kappa*h*u**4
check(epsilon.subs(u, 0) == 0 and s.diff(epsilon, u).subs(u, 0) == 0,
      'exact vacuum dielectric and amplitude derivative vanish')
V = omega**2*u**2/2+epsilon*settled_B**2/2
check(s.diff(V, u) == omega**2*u+2*kappa*h*settled_B**2*u**3,
      'identical later neutral probe equation unchanged')
check(s.diff(A2, x1, x1) == s.zeros(6), 'settled constant magnetic curvature')

depths = [2, 3, 4, 5, 6, 10]
even_carrier = {2: 2, 3: 4, 4: 4, 5: 6, 6: 6, 10: 10}
occupied = []
for depth in depths:
    even = even_carrier[depth]
    q = (10-even)//2
    m = even//2
    hd = 1 if m % 2 == 0 else 0
    check(q in (4, 3, 2, 0), 'only assigned normal ranks')
    check(hd == (1 if depth in (3, 4) else 0), 'separate Hodge premise')
    occupied.append({'depth': depth, 'even_carrier': even, 'q': q,
                     'u_dimension': q*q, 'su_dimension': q*q-1 if q else 0,
                     'orientation_dimension': q*(q-1), 'hodge_coefficient': hd})

report = {
    'title': 'Retained Ordinary Normal Complex Structure: Exact Geometric Response',
    'created_utc': datetime.now(timezone.utc).isoformat(),
    'sources': sources,
    'checks_passed': CHECKS,
    'exact_arithmetic': 'SymPy integers and rational numbers; no numerical rank tolerance',
    'rank_cases': rank_cases,
    'occupied_depths': occupied,
    'depth_four_real_wave': {
        'W2_norm_squared': 2, 'commutator_norm_squared': 4,
        'unitary_projection_norm_squared': 1, 'complement_projection_norm_squared': 1,
        'first_order_parallel_J_continuation': 'obstructed: background F0i is zero'},
    'projected_metric_connection': 'Ahat=A-J(DJ)/2',
    'projected_curvature': 'Fhat=P_h(F)-[K_mu,K_nu]/4',
    'vacuum_cartan_history': {
        'generator': '(j_1-j_2)/2', 'generator_norm_squared': 1,
        'generator_algebra': 'su3, hence also u3',
        'A2': 'b(t)*x1*T', 'DJ': '0', 'vacuum_energy': '0',
        'independent_connection_vacuum_ambiguity': 'survives'},
    'scope': 'local geometric identities and explicit comparison action; no graph evolution or particle simulation'
}
parser = argparse.ArgumentParser()
parser.add_argument('--report', type=Path, required=True)
output = parser.parse_args().report
output.write_text(json.dumps(report, indent=2, sort_keys=True)+'\n')
print(json.dumps({'checks_passed': CHECKS, 'result': str(output)}, indent=2))
