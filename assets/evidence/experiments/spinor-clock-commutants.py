"""Exact integer Clifford commutants and real invariant clock constructions."""
import hashlib
import itertools
import json
from pathlib import Path
import sys
import subprocess

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

HERE = Path(__file__).resolve().parent
checks = 0


def require(condition):
    global checks
    assert condition
    checks += 1


def tensor(factors):
    result = np.array([[1]])
    for factor in factors:
        result = np.kron(result, factor)
    return result


I = np.eye(2, dtype=np.int64)
X = np.array([[0, 1], [1, 0]], dtype=np.int64)
Z = np.diag([1, -1])
J = np.array([[0, -1], [1, 0]], dtype=np.int64)
Y = 1j * J
gamma = []
for k in range(5):
    for middle in (X, Y):
        gamma.append(tensor([Z] * k + [middle] + [I] * (4 - k)))
eye32 = np.eye(32, dtype=np.int64)
for i in range(10):
    for j in range(10):
        require(np.array_equal(gamma[i] @ gamma[j] + gamma[j] @ gamma[i],
                               (2 if i == j else 0) * eye32))
positive = [i for i in range(32) if i.bit_count() % 2 == 0]


def realify(a):
    require(np.all(a.real == np.rint(a.real)))
    require(np.all(a.imag == np.rint(a.imag)))
    return np.block([[a.real, -a.imag], [a.imag, a.real]]).astype(np.int64)


generators = [realify((gamma[i] @ gamma[i + 1])[np.ix_(positive, positive)])
              for i in range(9)]
for g in generators:
    require(np.array_equal(g.T, -g))
    require(np.array_equal(g @ g, -eye32))

# The 4^5 real Pauli tensor words are an orthogonal basis of all real
# endomorphisms. Clifford adjoint conjugations diagonalize on this basis.
words = list(itertools.product(range(4), repeat=5))
basis = np.array([tensor([(I, X, Z, J)[i] for i in w]) for w in words])
skew = np.array([sum(i == 3 for i in w) % 2 == 1 for w in words])
require(basis.shape == (1024, 32, 32))
for i, b in enumerate(basis):
    require(np.array_equal(b.T, (-1 if skew[i] else 1) * b))
    require(np.array_equal(b @ b.T, eye32))

surviving = np.ones(1024, dtype=bool)
ambient_j = np.kron(J, np.eye(16, dtype=np.int64))
ambient = []
expected = {2: (512, 256), 3: (256, 136), 4: (128, 72),
            5: (64, 36), 6: (32, 16), 7: (16, 6),
            8: (8, 2), 9: (4, 1), 10: (2, 1)}
for d, g in enumerate(generators, 2):
    commute = np.all(basis @ g == g @ basis, axis=(1, 2))
    anticommute = np.all(basis @ g == -(g @ basis), axis=(1, 2))
    require(np.all(commute | anticommute))
    surviving &= commute
    dimensions = (int(surviving.sum()), int((surviving & skew).sum()))
    require(dimensions == expected[d])
    require(np.array_equal(ambient_j @ g, g @ ambient_j))
    ambient.append(dict(depth=d, real_commutant_dimension=dimensions[0],
                        skew_commutant_dimension=dimensions[1],
                        ambient_complex_structure_survives=True))

# Quaternion units on real R^4: an explicit family of invariant clocks.
units = [np.kron(J, I), np.kron(Z, J), np.kron(X, J)]
eye4 = np.eye(4, dtype=np.int64)
for i, a in enumerate(units):
    require(np.array_equal(a.T, -a))
    for j, b in enumerate(units):
        require(np.array_equal(a @ b + b @ a, (-2 if i == j else 0) * eye4))
for a in itertools.product(range(-2, 3), repeat=3):
    b = sum((x * u for x, u in zip(a, units)), np.zeros((4, 4), dtype=np.int64))
    require(np.array_equal(b @ b, -sum(x*x for x in a) * eye4))
    for c in ((1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 2, -1)):
        k = sum((x * u for x, u in zip(c, units)), np.zeros((4, 4), dtype=np.int64))
        require(-int(np.trace(b @ k)) == 4 * sum(x*y for x, y in zip(a, c)))

# Quaternionic J reversal is an internal symmetry, not a distinguished
# antiparticle operation. Two copies of a real carrier also admit a clock.
require(np.array_equal(units[1] @ units[0] @ units[1].T, -units[0]))
for n in (1, 8, 16):
    b = np.kron(J, np.eye(n, dtype=np.int64))
    c = np.kron(Z, np.eye(n, dtype=np.int64))
    require(np.array_equal(c @ c, np.eye(2*n, dtype=np.int64)))
    require(np.array_equal(c @ b @ c, -b))
    require(np.array_equal(b @ b, -np.eye(2*n, dtype=np.int64)))

# Real irreducible spin modules: type from the real even Clifford algebra.
types = {0: 'R', 1: 'R', 2: 'C', 3: 'H', 4: 'H', 5: 'H', 6: 'C', 7: 'R'}
minimal = [dict(depth=d, type=types[d % 8],
                skew_dimension={'R': 0, 'C': 1, 'H': 3}[types[d % 8]])
           for d in range(1, 13)]
selected = [r['depth'] for r in minimal if r['depth'] <= 10 and r['skew_dimension']]
require(selected == [2, 3, 4, 5, 6, 10])

sources = {'experiments/spinor-clock-commutants.py': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
report = dict(checks_passed=checks, verification_scope="portable exact algebra", sources=sources,
              arithmetic='integer int64; all matrix entries and products in bounded small integers',
              minimal_modules=minimal, ambient_half_spin_restrictions=ambient,
              selected_minimal_depths=selected,
              assumptions=['Spin(d) is a candidate rotation representation, not derived particle spin.',
                           'Ambient carrier is one realified complex half-spin representation of Spin(10).',
                           'Real matrix components describe fields; physical coordinates remain R10.'])
(HERE / 'spinor-clock-commutants-results.json').write_text(json.dumps(report, indent=2) + '\n')
# A small, separately evaluated exact rank certificate for the quaternionic
# commutant. The left and right actions are distinct real matrix algebras.
left = [np.kron(I, J), np.kron(J, Z), np.kron(J, X)]
skew_basis = []
for i in range(4):
    for j in range(i + 1, 4):
        b = np.zeros((4, 4), dtype=np.int64)
        b[i, j], b[j, i] = 1, -1
        skew_basis.append(b)
constraints = np.array([np.concatenate([(b @ l - l @ b).ravel() for l in left])
                        for b in skew_basis]).T
job = dict(assumptions=report['assumptions'] + [
    'The 48 by 6 matrix constrains a real skew operator to commute with left quaternion multiplication.',
    'Nullity three supplies the quaternionic clock axes; it does not identify colour or weak interaction.'],
    sources=sources,
    jobs=[dict(name='quaternionic-invariant-skew-forms', operation='matrix',
               matrix=constraints.tolist()),
          dict(name='normalized-clock-positive-metric', operation='matrix',
               matrix=np.eye(4, dtype=np.int64).tolist())])
(HERE / 'spinor-clock-commutants-matrix.json').write_text(json.dumps(job, indent=2) + '\n')
print(json.dumps(report, indent=2))
