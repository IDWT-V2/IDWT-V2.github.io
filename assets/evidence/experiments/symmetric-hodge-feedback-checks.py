#!/usr/bin/env python3
"""Symmetric Hodge feedback, color pair operators and nested-source controls."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import importlib.util
from itertools import combinations
import json
from math import comb
from pathlib import Path

import numpy as np
import mpmath as mp
import sympy as sp


CHECKS = 0
CASES = []


def check(condition, message):
    global CHECKS
    if not condition:
        raise AssertionError(message)
    CHECKS += 1


def sign_sort(index):
    return (-1)**sum(index[i] > index[j] for i in range(len(index)) for j in range(i+1, len(index)))


def hodge_blocks():
    for m in range(1, 7):
        depth = 2*m
        seen = set()
        lam = sp.I if m % 2 else sp.Integer(1)
        rank = comb(depth-1, m-1)
        for index in combinations(range(depth), m):
            if index in seen:
                continue
            other = tuple(i for i in range(depth) if i not in index)
            star = sp.Matrix([[0, sign_sort(other+index)], [sign_sort(index+other), 0]])
            hermitian = (star+star.H)/2
            complementary = (star-star.H)/(2*sp.I)
            check(star.H == (-1)**m*star, 'real Hodge adjoint parity')
            check(hermitian == (sp.zeros(2) if m % 2 else star), 'canonical symmetric feedback filter')
            check(hermitian**2 == (sp.zeros(2) if m % 2 else sp.eye(2)), 'symmetric activation projector')
            check(hermitian**2+complementary**2 == sp.eye(2), 'complementary Hermitian source remains available')
            check(complementary.conjugate() == -complementary, 'complementary current is conjugation odd')
            plus = (sp.eye(2)+star/lam)/2
            check(hermitian*plus == sp.re(lam)*plus, 'complete chiral-channel filter eigenvalue')
            seen.update((index, other))
        for actual_depth in (depth-1, depth):
            CASES.append({'name': 'all-sector-symmetric-Hodge-rule', 'real_depth': actual_depth,
                          'parameters': {'even_carrier_depth': depth, 'middle_degree': m,
                                         'odd_depth_uses_declared_isometric_shadow': actual_depth != depth,
                                         'common_kernel': 'Hermitian part of raw real Hodge star'},
                          'measurements': {'channel_rank': rank, 'symmetric_filter_active': not bool(m % 2),
                                           'nonabelian_volume_channel_active': not bool(m % 2) and rank >= 3,
                                           'isotropic_pair_coefficient_sigma2_units': rank-2 if not m % 2 and rank >= 3 else 0,
                                           'complementary_Hermitian_channel_available': bool(m % 2)}})


def filtered_volume(patterns, gate):
    mapped = gate*patterns
    gram = mapped.conj().T@mapped
    return float(np.linalg.det(gram).real)


def norm_volume_and_pair():
    rng = np.random.default_rng(20261009)
    for m in range(1, 7):
        rank = comb(2*m-1, m-1)
        lam = 1j if m % 2 else 1.
        gate = float(np.real(lam))
        for _ in range(4):
            patterns = rng.normal(size=(rank, 3))+1j*rng.normal(size=(rank, 3))
            value = filtered_volume(patterns, gate)
            check(value >= -1e-8, 'positive common filtered volume action')
            check(abs(value) < 1e-10 if m % 2 else True, 'odd-middle-degree source vanishes for every pattern')
            phases = np.exp(1j*rng.normal(size=3))
            check(np.isclose(filtered_volume(patterns*phases, gate), value, rtol=1e-9, atol=1e-8),
                  'independent electric phase characters preserve volume energy')
            # A nontrivial unitary mode change preserves the same action at every rank.
            unitary = np.eye(rank, dtype=complex)
            if rank >= 2:
                angle = .731
                unitary[:2, :2] = [[np.cos(angle), 1j*np.sin(angle)],
                                   [1j*np.sin(angle), np.cos(angle)]]
            check(np.isclose(filtered_volume(unitary@patterns, gate), value, rtol=1e-9, atol=1e-8),
                  'full complex unitary mode covariance')
        # Independently contract the third slot using a complete isotropic basis.
        if rank <= 35:
            a = rng.normal(size=rank)+1j*rng.normal(size=rank)
            b = rng.normal(size=rank)+1j*rng.normal(size=rank)
            contracted = sum(filtered_volume(np.column_stack([a, b, np.eye(rank)[:, j]]), gate)
                             for j in range(rank))
            expected = gate**6*(rank-2)*(np.vdot(a, a).real*np.vdot(b, b).real-abs(np.vdot(a, b))**2)
            check(np.isclose(contracted, expected, rtol=1e-8, atol=1e-8), 'same-law isotropic third-slot contraction')
    # Exact triplet pair operator and its familiar Casimir decomposition.
    rank = 3
    swap = sp.zeros(rank**2)
    for i in range(rank):
        for j in range(rank):
            swap[i*rank+j, j*rank+i] = 1
    pair = sp.eye(rank**2)-swap
    check(pair.eigenvals() == {sp.Integer(2): 3, sp.Integer(0): 6}, 'triplet volume pair retains antisymmetric channel')
    generators = []
    for i, j in combinations(range(rank), 2):
        t = sp.zeros(rank); t[i, j] = t[j, i] = sp.Rational(1, 2); generators.append(t)
        t = sp.zeros(rank); t[i, j] = -sp.I/2; t[j, i] = sp.I/2; generators.append(t)
    generators += [sp.diag(1, -1, 0)/2, sp.diag(1, 1, -2)/(2*sp.sqrt(3))]
    contraction = sum((sp.kronecker_product(t, t) for t in generators), sp.zeros(9))
    check(pair == sp.Rational(2, 3)*sp.eye(9)-2*contraction, 'existing color Casimir pair operator preserved')
    check(contraction.eigenvals() == {sp.Rational(-2, 3): 3, sp.Rational(1, 3): 6}, 'standard qq pair Casimirs retained')
    CASES.append({'name': 'filtered-color-pair', 'parameters': {'active_rank': 3, 'third_slot_covariance': 'sigma^2 I'},
                  'measurements': {'pair_operator': 'sigma^2 (I-swap)',
                                   'qq_pair_weights': ['-2/3', '1/3'],
                                   'odd_middle_degree_pair_operator': '0',
                                   'higher_control_rank35_pair_coefficient': '33 sigma^2'}})


def alternative_sources_and_restriction():
    # The naive complex self-wedge pairing does not have full SU(3) symmetry.
    a = np.array([1., 1., 0.], dtype=complex)/np.sqrt(2)
    u = np.diag([np.exp(1j*np.pi/4), np.exp(-1j*np.pi/4), 1])
    check(np.isclose(np.linalg.det(u), 1), 'symmetry control lies in SU3')
    check(not np.isclose(abs(a@a)**2, abs((u@a)@(u@a))**2), 'complex bilinear self-wedge is not SU3 invariant')
    check(np.isclose(np.vdot(a, a), np.vdot(u@a, u@a)), 'Hermitian filtered source preserves SU3')
    star = np.array([[0., -1.], [1., 0.]])
    complementary = star/1j
    field = np.array([1., 1j])/np.sqrt(2)
    current = float(np.vdot(field, complementary@field).real)
    reversed_current = float(np.vdot(field.conjugate(), complementary@field.conjugate()).real)
    check(np.isclose(current, -reversed_current) and abs(current) > .9, 'complementary source changes under time reversal')
    check(np.isclose(current**2, reversed_current**2), 'squared complementary energy is time-reversal even')
    CASES.append({'name': 'allowed-alternative-channel', 'parameters': {'odd_middle_degree': True},
                  'measurements': {'symmetric_Hodge_source': '0', 'complementary_source': current,
                                   'time_reversal_forbids_complementary_squared_energy': False,
                                   'generic_norm_overlap_also_remains_allowed': True}})
    # Contract selected ordinary normal directions, then restrict to a four-plane.
    for parent_depth in (6, 10):
        degree = parent_depth//2
        parent_basis = list(combinations(range(parent_depth), degree))
        child_basis = list(combinations(range(4), 2))
        selected_normals = tuple(range(4, 4+degree-2))
        r = np.zeros((6, len(parent_basis)))
        for j, child in enumerate(child_basis):
            index = tuple(sorted(child+selected_normals))
            r[j, parent_basis.index(index)] = sign_sort(selected_normals+child)
        child_star = np.zeros((6, 6))
        for j, child in enumerate(child_basis):
            other = tuple(i for i in range(4) if i not in child)
            child_star[child_basis.index(other), j] = sign_sort(child+other)
        check(np.allclose(r@r.T, np.eye(6)), 'ordinary normal contraction reaches full child two-form space')
        mismatch = child_star@r  # Parent symmetric Hodge filter is zero.
        check(np.isclose(np.linalg.norm(mismatch)**2, 6), 'symmetric filter does not commute with lower-depth contraction')
        CASES.append({'name': 'nested-source-ordering', 'real_depth': parent_depth,
                      'parameters': {'child_depth': 4, 'normal_contractions': degree-2},
                      'measurements': {'filter_parent_then_restrict': '0',
                                       'restrict_then_filter_squared_Hilbert_Schmidt_norm': 6,
                                       'intrinsic_source_first_is_an_additional_physical_premise': True}})


def contact_order_and_reciprocity():
    q1, q2, v1, v2 = sp.symbols('q1 q2 v1 v2', real=True)
    q, v = sp.Matrix([q1, q2]), sp.Matrix([v1, v2])
    for middle_degree in range(1, 7):
        star = sp.Matrix([[0, (-1)**middle_degree], [1, 0]])
        symmetric, skew = (star+star.T)/2, (star-star.T)/2
        energy = (q.T*star*q)[0]/2
        velocity_action = (q.T*star*v)[0]/2
        euler_velocity = sp.Matrix([sum(sp.diff(sp.diff(velocity_action, vv), qq)*dv
                                        for qq, dv in zip(q, v))-sp.diff(velocity_action, qq)
                                   for vv, qq in zip(v, q)])
        check(sp.expand(energy-(q.T*symmetric*q)[0]/2) == 0, 'real instantaneous self-energy uses symmetric Hodge part')
        check(euler_velocity == -skew*v, 'real velocity term uses antisymmetric Hodge part')
        check((v.T*skew*v)[0].expand() == 0, 'Hodge gyroscopic response performs no instantaneous work')
        # Complete positive second-order control; no simulated particle trajectory.
        kk = 2*np.eye(2)+.3*np.array(symmetric, dtype=float)
        aa = .4*np.array(skew, dtype=float)
        dynamics = np.block([[np.zeros((2, 2)), np.eye(2)], [-kk, aa]])
        eigen = np.linalg.eigvals(dynamics)
        frequencies = sorted(x.imag for x in eigen if x.imag > 1e-10)
        check(np.linalg.eigvalsh(kk)[0] > 0 and np.max(np.abs(eigen.real)) < 1e-10,
              'same-law static and gyro oscillator controls remain positive and oscillatory')
        expected = sorted([np.sqrt(2.04)-.2, np.sqrt(2.04)+.2] if middle_degree % 2 else
                          [np.sqrt(1.7), np.sqrt(2.3)])
        check(np.allclose(frequencies, expected), 'complete Hodge oscillator frequencies')
        # Both participants are canonical coordinates of the same field reduction.
        # Positive potential and gyroscopic reciprocity are checked before prescribing a source.
        hs = np.array(symmetric, dtype=float)
        sa = np.array(skew, dtype=float)
        full_k = np.block([[2*np.eye(2), -hs], [-hs, 3*np.eye(2)]])
        full_g = np.block([[np.zeros((2, 2)), sa], [sa, np.zeros((2, 2))]])
        full_dynamics = np.block([[np.zeros((4, 4)), np.eye(4)], [-full_k, full_g]])
        full_eigen = np.linalg.eigvals(full_dynamics)
        energy_matrix = sp.diag(*([1]*8))
        check(np.linalg.eigvalsh(full_k)[0] > 0, 'common reciprocal q/source potential positive')
        check(np.allclose(full_g+full_g.T, 0), 'complete gyro coupling antisymmetric across both participants')
        check(np.max(np.abs(full_eigen.real)) < 1e-9, 'complete reciprocal q/source modes oscillatory')
        metric_energy = la_block_energy = np.block([[full_k, np.zeros((4, 4))],
                                                   [np.zeros((4, 4)), np.eye(4)]])
        check(np.linalg.norm(metric_energy@full_dynamics+full_dynamics.T@metric_energy) < 1e-10,
              'complete reciprocal mode energy conserved')
    # Reciprocity applies to the full kernel, not the fibre factor alone.
    skew = sp.Matrix([[0, -1], [1, 0]])
    cross_kernel = sp.BlockMatrix([[sp.zeros(2), skew], [-skew, sp.zeros(2)]]).as_explicit()
    check(cross_kernel == cross_kernel.T and cross_kernel != sp.zeros(4),
          'antisymmetric fibre map can survive in a reciprocal two-slot kernel')
    check((2*sp.eye(4)+cross_kernel).is_positive_definite is True,
          'reciprocal cross-kernel counterexample admits positive complete energy')
    CASES.append({'name': 'position-versus-velocity-Hodge-contact',
                  'parameters': {'common_bare_stiffness': 2, 'static_coefficient': .3, 'gyro_coefficient': .4},
                  'measurements': {'even_middle_degree_static_active': True,
                                   'odd_middle_degree_velocity_active': True,
                                   'reciprocity_alone_forces_symmetric_fibre_filter': False,
                                   'velocity_term_requires_time_reversal_parities_or_circulation': True}})


def all_sector_spectral_response(source):
    spec = importlib.util.spec_from_file_location('frozen_sawtooth', source)
    saw = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(saw)
    mp.mp.dps = 50
    electron = mp.mpf('0.51099895000')
    mu6 = electron/18564
    mu5 = mp.sqrt(mp.mpf('2.534e-3')/(65780**2-2002**2))/mp.mpf(10**6)
    scales = {2: electron*mp.sqrt(2890), 3: electron*mp.sqrt(32*mp.sqrt(7)),
              4: electron*mp.sqrt(48/mp.sqrt(7))/15, 5: mu5, 6: mu6, 10: mu6}
    assigned = [('W', 2, 76), ('Z', 2, 81), ('Higgs', 2, 95), ('down', 3, 1),
                ('strange', 3, 4), ('up', 4, 3), ('charm', 4, 20), ('top', 4, 72),
                ('nu1', 5, 10), ('nu2', 5, 15), ('nu3', 5, 22),
                ('electron', 6, 13), ('muon', 6, 35), ('tau', 10, 23)]
    for label, depth, n in assigned:
        middle_degree = (depth+1)//2
        static_gate = int(middle_degree % 2 == 0)
        derivative_gate = 1-static_gate
        cutoff = comb(n+depth-1, depth)
        ratio = float(scales[depth]/mu6)
        response = saw.response_sum(cutoff, ratio)
        dynamic = (static_gate*response['density_response']+
                   derivative_gate*response['derivative_source_response'])
        static = static_gate*response['static_source_term']
        bound = (static_gate*response['density_tail_bound']+
                 derivative_gate*response['derivative_tail_bound'])
        check(np.isfinite(dynamic+static), 'all-sector common-order response finite')
        check(not static_gate if depth in (2, 5, 6, 10) else bool(static_gate),
              'same real-Hodge contact law chooses all maintained interaction orders')
        if depth == 5:
            check(abs(dynamic) < 1e-9 and static == 0, 'slow neutrino source suppression survives common Hodge order rule')
        CASES.append({'name': 'all-sector-Hodge-order-response', 'real_depth': depth, 'mode_index': n,
                      'parameters': {'particle': label, 'cutoff': cutoff, 'fundamental_ratio': ratio,
                                     'source_map': 'conditional real sawtooth density in fixed unit form direction',
                                     'common_g0_g1_in_reference_time_units': [1, 1],
                                     'receiver': 'scalar susceptibility times form identity; shape weights 3/5 at2,2/5 at3',
                                     'interaction_order': 'density' if static_gate else 'velocity'},
                      'measurements': {'signed_dynamic_response': dynamic, 'static_source_response': static,
                                       'signed_total_response': dynamic+static, 'absolute_omitted_tail_bound': bound}})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--sawtooth-source', required=True)
    args = parser.parse_args()
    hodge_blocks()
    norm_volume_and_pair()
    alternative_sources_and_restriction()
    contact_order_and_reciprocity()
    all_sector_spectral_response(Path(args.sawtooth_source))
    report = {'experiment_key': 'symmetric-Hodge-feedback-20261009',
              'created_utc': datetime.now(timezone.utc).isoformat(),
              'sources': {'experiments/symmetric-hodge-feedback-checks.py': sha256(Path(__file__).read_bytes()).hexdigest(),
                          'experiments/sawtooth-sector-coupling-checks.py': sha256(Path(args.sawtooth_source).read_bytes()).hexdigest()},
              'scope': 'A conditional common raw-Hodge reciprocal contact kernel and filtered color-volume action. It selects 3/4 among maintained sectors, allows 7/8 controls, and retains alternative interactions and nested-source ordering as explicit physical premises.',
              'summary': {'checks_passed': CHECKS, 'case_count': len(CASES)}, 'cases': CASES}
    Path(args.output).write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report['summary']))


if __name__ == '__main__':
    main()
