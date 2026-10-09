#!/usr/bin/env python3
"""Verify the combined spectral, color and directed-state constructions."""

import hashlib
import itertools
import json
import math
import os
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction as F
from pathlib import Path

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import mpmath as mp
import numpy as np
import sympy as sp


BASE = Path(__file__).resolve().parents[2]
COUNTS = Counter()
mp.mp.dps = 65


def exact(group, a, b):
    difference = sp.simplify(a - b)
    valid = difference.is_zero_matrix if isinstance(difference, sp.MatrixBase) else difference == 0
    if not valid:
        raise AssertionError((group, a, b))
    COUNTS[group] += 1


def close(group, a, b, tol=1e-10):
    error = float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
    if error > tol:
        raise AssertionError((group, error))
    COUNTS[group] += 1


def S(n, d):
    return math.comb(n + d - 1, d)


def shell(k, d):
    return math.comb(k + d - 1, d - 1)


def hodge_rank(d):
    m = (d + 1) // 2
    return math.comb(2 * m - 1, m - 1)


def mp_fraction(value):
    return mp.mpf(value.numerator) / value.denominator


spectral = []
for depth in range(2, 13):
    for n in range(1, 15):
        N = n - 1
        weights = [shell(l, depth) for l in range(n)]
        D = S(n, depth)
        mean = F(depth * N, depth + 1)
        variance = F(depth * N * (N + depth + 1), (depth + 1) ** 2 * (depth + 2))
        exact("spatial_band_count", sum(weights), D)
        exact("band_mean", F(sum(l * weights[l] for l in range(n)), D), mean)
        exact("band_variance", sum(F(weights[l], D) * (l - mean) ** 2 for l in range(n)), variance)
        exact("shell_weighted_tail", F(shell(n - 1, depth), S(n, depth)), F(depth, n + depth - 1))
        if depth % 2 == 0:
            m = depth // 2
            exact("projective_reference_count", F(math.comb(N + m, m) * math.comb(N + 2 * m, m),
                                                 math.comb(2 * m, m)), D)
    for cutoff in (2, 5, 12):
        n = cutoff + 1
        weights = [shell(l, depth) for l in range(n)]
        D = sum(weights)
        vector = np.sqrt(np.array(weights, dtype=float))
        contact = np.outer(vector, vector)
        close("normalized_contact_projector", contact @ contact / (D * D), contact / D, 2e-12)
        kinetic = np.arange(n, dtype=float) + depth / 2
        matrix = np.diag(kinetic) + contact
        values, vectors = np.linalg.eigh(matrix)
        mean = F(depth, 2) + F(depth * cutoff, depth + 1)
        variance = F(depth * cutoff * (cutoff + depth + 1),
                     (depth + 1) ** 2 * (depth + 2))
        bound = F(variance, D - cutoff)
        lo, hi = mp.mpf(0), mp_fraction(bound)
        def secular(shift):
            root = D + mp_fraction(mean) + shift
            return sum(mp.mpf(w) / (root - (mp.mpf(l) + mp.mpf(depth) / 2))
                       for l, w in enumerate(weights)) - 1
        if secular(lo) < -mp.mpf("1e-55") or secular(hi) > mp.mpf("1e-55"):
            raise AssertionError(("kinetic_bracket", depth, cutoff))
        for _ in range(160):
            midpoint = (lo + hi) / 2
            if secular(midpoint) > 0:
                lo = midpoint
            else:
                hi = midpoint
        correction = (lo + hi) / 2
        root = D + mp_fraction(mean) + correction
        close("independent_kinetic_eigenvalue", values[-1] / D, float(root / D), 3e-12)
        bright = vector / math.sqrt(D)
        leakage = math.sqrt(max(0, 1 - abs(bright @ vectors[:, -1]) ** 2))
        if leakage > math.sqrt(float(variance)) / (D - cutoff) + 1e-7:
            raise AssertionError(("kinetic_leakage_bound", depth, cutoff, leakage))
        COUNTS["kinetic_leakage_bound"] += 1
        spectral.append({"depth": depth, "cutoff": cutoff, "count": D,
                         "mean_kinetic": str(mean), "variance": str(variance),
                         "correction": mp.nstr(correction, 16), "upper_bound": str(bound)})


def permutation_sign(p):
    return -1 if sum(p[i] > p[j] for i in range(len(p)) for j in range(i + 1, len(p))) % 2 else 1


def swap_matrix(rank):
    out = sp.zeros(rank * rank)
    for i in range(rank):
        for j in range(rank):
            out[j * rank + i, i * rank + j] = 1
    return out


for rank in range(2, 7):
    groups = {}
    for permutation in itertools.permutations(range(rank)):
        groups.setdefault(permutation[2:], []).append((permutation[:2], permutation_sign(permutation)))
    reduced = sp.zeros(rank * rank)
    for terms in groups.values():
        for (i, j), sign_a in terms:
            for (k, l), sign_b in terms:
                reduced[i * rank + j, k * rank + l] += sign_a * sign_b
    target = math.factorial(rank - 2) * (sp.eye(rank * rank) - swap_matrix(rank))
    exact("determinant_partial_trace", (reduced - target).norm(), 0)
    exact("normalized_determinant_trace", sp.trace(reduced) / math.factorial(rank), 1)

for rank in range(2, 9):
    # Contract the three-form projector against a unit covariance in slot three.
    reduced = sp.zeros(rank * rank)
    for i, j, a, b in itertools.product(range(rank), repeat=4):
        value = 0
        for c in range(rank):
            left, right = (i, j, c), (a, b, c)
            value += sp.det(sp.Matrix(3, 3, lambda row, col: int(left[row] == right[col])))
        reduced[i * rank + j, a * rank + b] = value
    exact("three_form_partial_trace", (reduced - (rank - 2) * (sp.eye(rank * rank) - swap_matrix(rank))).norm(), 0)

rank = 3
generators = []
for i in range(rank):
    for j in range(i + 1, rank):
        symmetric, antisymmetric = sp.zeros(rank), sp.zeros(rank)
        symmetric[i, j] = symmetric[j, i] = sp.Rational(1, 2)
        antisymmetric[i, j], antisymmetric[j, i] = -sp.I / 2, sp.I / 2
        generators += [symmetric, antisymmetric]
generators += [sp.diag(1, -1, 0) / 2, sp.diag(1, 1, -2) / (2 * sp.sqrt(3))]
pair_casimir = sum((sp.kronecker_product(t, t) for t in generators), sp.zeros(9))
exact("color_pair_fierz", (2 * pair_casimir + sp.eye(9) / 3 - swap_matrix(3)).norm(), 0)
exact("color_fundamental_casimir", (sum((t * t for t in generators), sp.zeros(3)) - sp.Rational(4, 3) * sp.eye(3)).norm(), 0)
exact("color_pair_weights", (pair_casimir + sp.Rational(2, 3) * sp.eye(9)) *
      (pair_casimir - sp.Rational(1, 3) * sp.eye(9)), sp.zeros(9))

color_table = []
for depth in range(2, 17):
    rank = hodge_rank(depth)
    yq = F(1, 2 * rank)
    yl, yu, yd, ye, yn = F(-1, 2), yq + F(1, 2), yq - F(1, 2), F(-1), F(0)
    exact("charge_linear_anomalies", 2 * yq - yu - yd, 0)
    exact("charge_linear_anomalies", rank * yq + yl, 0)
    exact("charge_linear_anomalies", rank * (2 * yq - yu - yd) + 2 * yl - ye - yn, 0)
    exact("charge_cubic_anomaly", rank * (2 * yq ** 3 - yu ** 3 - yd ** 3) + 2 * yl ** 3 - ye ** 3 - yn ** 3, 0)
    m = (depth + 1) // 2
    exact("middle_channel_parity", rank % 2, int(m & (m - 1) == 0))
    color_table.append({"depth": depth, "channel_rank": rank, "adjoint_rank": rank * rank - 1,
                        "fundamental_casimir": str(F(rank * rank - 1, 2 * rank)),
                        "up_charge": str(yu), "down_charge": str(yd),
                        "weak_doublets_with_one_lepton": rank + 1,
                        "even_weak_doublet_count": (rank + 1) % 2 == 0,
                        "scalar_trilinear_invariant": rank in (1, 3),
                        "isotropic_three_form_pair_factor": max(0, rank - 2)})


def tangent_basis(n):
    constraints = np.vstack([np.ones(len(n)), n])
    return np.linalg.svd(constraints, full_matrices=True)[2][2:].T


angular = []
for k in range(2, 13):
    number = k + 1
    directions = math.sqrt(number / k) * (np.eye(number) - np.ones((number, number)) / number)
    bases = [tangent_basis(n) for n in directions]
    rows = []
    for i in range(number):
        for j in range(i + 1, number):
            exact_overlap = (1 + k * (directions[i] @ directions[j])) / number
            close("simplex_zero_energy", exact_overlap, 0, 2e-14)
            row = np.zeros(number * (k - 1))
            row[i * (k - 1):(i + 1) * (k - 1)] = k / number * bases[i].T @ directions[j]
            row[j * (k - 1):(j + 1) * (k - 1)] = k / number * bases[j].T @ directions[i]
            rows.append(row)
    jacobian = np.array(rows)
    hessian = 2 * jacobian.T @ jacobian
    spectrum = np.linalg.eigvalsh(hessian)
    zeros, first, second = k * (k - 1) // 2, k, (k + 1) * (k - 2) // 2
    expected = sorted([0.] * zeros + [2.] * first + [4 * k / (k + 1)] * second)
    close("simplex_shape_spectrum", spectrum, expected, 2e-12)
    angular.append({"resolved_directions": k, "states": number, "rotation_zero_modes": zeros,
                    "first_shape_eigenvalue": "2", "first_multiplicity": first,
                    "second_shape_eigenvalue": str(F(4 * k, k + 1)), "second_multiplicity": second})

cross_polytope = []
for k in range(2, 13):
    directions = np.vstack([np.eye(k), -np.eye(k)])
    bases = [np.eye(k)[:, np.flatnonzero(n == 0)] for n in directions]
    rows = []
    for i in range(2 * k):
        for j in range(i + 1, 2 * k):
            x = directions[i] @ directions[j]
            close("cross_polytope_zero_energy", x * (x + 1) / 2, 0)
            row = np.zeros(2 * k * (k - 1))
            row[i * (k - 1):(i + 1) * (k - 1)] = (x + 0.5) * bases[i].T @ directions[j]
            row[j * (k - 1):(j + 1) * (k - 1)] = (x + 0.5) * bases[j].T @ directions[i]
            rows.append(row)
    jacobian = np.array(rows)
    spectrum = np.linalg.eigvalsh(2 * jacobian.T @ jacobian)
    pairs = k * (k - 1) // 2
    close("cross_polytope_shape_spectrum", spectrum, sorted([0.] * pairs + [1.] * (2 * pairs) + [2.] * pairs), 2e-12)
    cross_polytope.append({"resolved_directions": k, "states": 2 * k, "rotation_zero_modes": pairs,
                          "first_shape_eigenvalue": "1", "first_multiplicity": 2 * pairs,
                          "second_shape_eigenvalue": "2", "second_multiplicity": pairs})

# Exact generalized Hessian in an integer tangent basis of the tetrahedron.
tetra = [sp.Matrix(v) for v in ((1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1))]
bases = []
for v in tetra:
    first = sp.Matrix([-v[1], v[0], 0])
    bases.append(sp.Matrix.hstack(first, v.cross(first)))
metric = sp.diag(*(b.T * b for b in bases))
rows = []
for i in range(4):
    for j in range(i + 1, 4):
        row = sp.zeros(1, 8)
        row[:, 2*i:2*i+2] = sp.Rational(3, 4) * tetra[j].T * bases[i] / sp.sqrt(3)
        row[:, 2*j:2*j+2] = sp.Rational(3, 4) * tetra[i].T * bases[j] / sp.sqrt(3)
        rows.append(row)
jacobian = sp.Matrix.vstack(*rows)
symbol = sp.Symbol("lambda")
tetra_polynomial = sp.factor((metric.inv() * (2 * jacobian.T * jacobian)).charpoly(symbol).as_expr())
exact("exact_tetrahedral_shape_spectrum", tetra_polynomial, symbol ** 3 * (symbol - 2) ** 3 * (symbol - 3) ** 2)

axes = [sp.Matrix(v) for v in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))]
bases = [sp.Matrix.hstack(*(sp.eye(3)[:, j] for j in range(3) if n[j] == 0)) for n in axes]
rows = []
for i in range(6):
    for j in range(i + 1, 6):
        x = axes[i].dot(axes[j])
        exact("octahedral_zero_energy", x * (x + 1) / 2, 0)
        row = sp.zeros(1, 12)
        row[:, 2*i:2*i+2] = (x + sp.Rational(1, 2)) * axes[j].T * bases[i]
        row[:, 2*j:2*j+2] = (x + sp.Rational(1, 2)) * axes[i].T * bases[j]
        rows.append(row)
jacobian = sp.Matrix.vstack(*rows)
octa_polynomial = sp.factor((2 * jacobian.T * jacobian).charpoly(symbol).as_expr())
exact("exact_octahedral_shape_spectrum", octa_polynomial, symbol ** 3 * (symbol - 1) ** 6 * (symbol - 2) ** 3)

# The same band maps give a normalized mixing defect.
rng = np.random.default_rng(936)
for rank in (1, 2, 3, 5, 6, 10):
    first = rng.normal(size=(2 * rank, rank)) + 1j * rng.normal(size=(2 * rank, rank))
    second = rng.normal(size=(2 * rank, rank)) + 1j * rng.normal(size=(2 * rank, rank))
    u = np.linalg.qr(first)[0]
    v = np.linalg.qr(second)[0]
    kernel = u.conj().T @ v
    defect = np.eye(rank) - kernel.conj().T @ kernel
    close("mixing_projector_defect", defect, v.conj().T @ (np.eye(2 * rank) - u @ u.conj().T) @ v, 2e-12)
    if np.linalg.eigvalsh(defect).min() < -1e-12:
        raise AssertionError("negative overlap defect")
    COUNTS["positive_mixing_defect"] += 1

report = {
    "title": "Integrated spatial-band, color and angular-geometry calculations",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "status": "passed",
    "checks": dict(COUNTS),
    "total_checks": sum(COUNTS.values()),
    "scope": "Finite exact algebra and bounded eigenvalue comparisons; no particle evolution, parameter fitting or empirical claim.",
    "spectral_dressing": spectral,
    "color_comparison": color_table,
    "angular_comparison": angular,
    "cross_polytope_comparison": cross_polytope,
    "exact_tetrahedron_characteristic_polynomial": str(tetra_polynomial),
    "exact_octahedron_characteristic_polynomial": str(octa_polynomial),
    "sources": {"experiments/spatial-band-integration-checks.py": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
    "input_snapshot": 935,
}
output = BASE / "spatial-band-integration-results.json"
output.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({"status": report["status"], "checks": report["total_checks"],
                  "tetrahedron": str(tetra_polynomial), "octahedron": str(octa_polynomial), "report": str(output)}))
