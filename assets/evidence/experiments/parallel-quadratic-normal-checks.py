#!/usr/bin/env python3
"""Exact finite representation checks for the quadratic Veronese comparison."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import argparse
import json
import sympy as s

HERE = Path(__file__).resolve().parent
REPOSITORY = next(p for p in HERE.parents if (p/"database"/"research.py").is_file())
checks = []


def check(name, condition):
    assert bool(condition), name
    checks.append(name)


def sym2(A):
    a, b, c, d = A[0, 0], A[0, 1], A[1, 0], A[1, 1]
    return s.Matrix([[2*a, b, 0], [2*c, a+d, 2*b], [0, c, 2*d]])


def rank(matrices):
    return s.Matrix.hstack(*(M.reshape(M.rows*M.cols, 1) for M in matrices)).rank()


I = s.I
X = s.Matrix([[0, 1], [-1, 0]])
Y = s.Matrix([[0, I], [I, 0]])
Z = s.diag(I, -I)
C = I*s.eye(2)
u2 = [X, Y, Z, C]
normal = [sym2(A) for A in u2]
H = s.diag(1, s.Rational(1, 2), 1)
K = s.Matrix([[0, 0, -2], [0, 1, 0], [-2, 0, 0]])
check('u2 normal image rank four', rank(normal) == 4)
check('su2 normal image rank three', rank(normal[:3]) == 3)
check('central phase has normal weight two', normal[3] == 2*I*s.eye(3))
for i, A in enumerate(u2):
    R = sym2(A)
    check(f'unitarity {i}', R.conjugate().T*H + H*R == s.zeros(3))
    check(f'discriminant character {i}', R.T*K + K*R == 2*s.trace(A)*K)
    check(f'determinant character {i}', s.trace(R) == 3*s.trace(A))
    for j, B in enumerate(u2):
        check(f'Lie homomorphism {i},{j}', sym2(A*B-B*A) == R*sym2(B)-sym2(B)*R)
        if i < 3 and j < 3:
            comm = R*sym2(B)-sym2(B)*R
            check(f'su2 closure {i},{j}', rank(normal[:3]+[comm]) == 3)


def invariant_tensors(generators, dimension):
    eye = s.eye(dimension)
    stacked = s.Matrix.vstack(*(
        s.kronecker_product(A, eye) + s.kronecker_product(eye, A)
        for A in generators
    ))
    return stacked.nullspace()


su2_invariants = invariant_tensors(normal[:3], 3)
u2_invariants = invariant_tensors(normal, 3)
check('spin-one tensor square has one singlet', len(su2_invariants) == 1)
check('the spin-one singlet is symmetric', su2_invariants[0].reshape(3, 3).T == su2_invariants[0].reshape(3, 3))
check('full u2 tensor square has no untwisted singlet', len(u2_invariants) == 0)

sl3 = []
for i in range(3):
    for j in range(3):
        if i != j:
            E = s.zeros(3)
            E[i, j] = 1
            sl3.append(E)
sl3.extend([s.diag(1, -1, 0), s.diag(0, 1, -1)])
check('su3 complexified algebra has rank eight', rank(sl3) == 8)
check('su3 fundamental tensor square has no singlet', len(invariant_tensors(sl3, 3)) == 0)

t = s.symbols('t', nonzero=True)
chi1 = t**2 + 1 + t**-2
chi2 = t**4 + t**2 + 1 + t**-2 + t**-4
check('spin-one tensor-square character is one plus three plus five', s.expand(chi1**2-1-chi1-chi2) == 0)

x, y, h = s.symbols('x y h')
roots = [2*x, x+y, 2*y]
c1 = sum(roots)
c2 = sum(roots[i]*roots[j] for i in range(3) for j in range(i+1, 3))
check('symmetric-square c1 identity', s.expand(c1-3*(x+y)) == 0)
check('symmetric-square c2 identity', s.expand(c2-2*(x+y)**2-4*x*y) == 0)
check('CP2 c1 normal', 3*3 == 9)
check('CP2 c2 normal', 2*3**2+4*3 == 30)
ambient_normal_chern = s.series((1+2*h)**6/(1+3*h+3*h**2), h, 0, 3).removeO().expand()
check('ambient quotient Chern identity', ambient_normal_chern == 1+9*h+30*h**2)
check('determinant-free c2', 30 + 2*(-3)*9 + 3*(-3)**2 == 3)
check('normal real p1', 9**2-2*30 == 21)
check('adjoint real p1', 3**2-4*3 == -3)
check('normal bundle Euler degree exceeds base dimension', 2*3 > 4)

report = {
    'generated_utc': datetime.now(timezone.utc).isoformat(),
    'scope': 'Exact finite algebra and characteristic identities; not a numerical simulation or a proof of metric completeness.',
    'checks_passed': len(checks),
    'checks': checks,
    'results': {
        'normal_holonomy_algebra_dimension': 4,
        'normal_traceless_algebra_dimension': 3,
        'spin_one_tensor_square_singlets': len(su2_invariants),
        'full_u2_tensor_square_untwisted_singlets': len(u2_invariants),
        'su3_fundamental_tensor_square_singlets': 0,
        'normal_c1_H': 9,
        'normal_c2_H2': 30,
        'determinant_free_c2_H2': 3,
        'normal_real_p1_H2': 21,
        'adjoint_real_p1_H2': -3,
    },
    'sources': {'experiments/parallel-quadratic-normal-checks.py': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
}
parser = argparse.ArgumentParser()
parser.add_argument('--report', type=Path, required=True)
destination = parser.parse_args().report
destination.write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps({'checks_passed': len(checks), 'report': str(destination), **report['results']}, indent=2))
