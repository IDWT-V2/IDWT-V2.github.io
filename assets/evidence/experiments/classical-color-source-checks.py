#!/usr/bin/env python3
"""Test classical color-current cancellation with the actual spatial symbol."""
from pathlib import Path
from itertools import combinations
import argparse
import hashlib
import importlib.util
import json
import math

import numpy as np


DEPTHS = (2, 3, 4, 5, 6, 10)
CHECKS = 0
CASES = []


def check(condition, message):
    global CHECKS
    CHECKS += 1
    if not condition:
        raise AssertionError(message)


def close(actual, expected, message, atol=3e-10):
    check(np.allclose(actual, expected, atol=atol, rtol=3e-10), message)


def unit(vector):
    return vector / np.linalg.norm(vector)


def currents(psi, gamma, algebra):
    density = np.array([np.vdot(psi, t @ psi).real for t in algebra])
    spatial = np.array([[np.vdot(psi, g @ t @ psi).real for t in algebra]
                        for g in gamma])
    return density, spatial


def symbol_projector(momentum, gamma):
    direction = momentum / np.linalg.norm(momentum)
    symbol = sum(p * g for p, g in zip(direction, gamma))
    return (np.eye(len(symbol)) + symbol) / 2


def color_paired_line(gamma, algebra, number, rng):
    # Complementary occupation states have opposite SU(3) weights.
    eta = np.zeros(8, complex)
    eta[4] = eta[3] = 1 / math.sqrt(2)
    embed = np.kron(np.eye(4), eta[:, None])
    close(embed.conj().T @ embed, np.eye(4), 'complex-linear embedding has positive norm')
    casimir = sum(t @ t for t in algebra)
    close(embed.conj().T @ casimir @ embed, 4 * np.eye(4) / 3,
          'classically canceled normal line retains nonzero SU(3) Casimir')
    central = sum(number)
    close(embed.conj().T @ central @ embed, 1.5 * np.eye(4), 'normal number mean')
    close(embed.conj().T @ central @ central @ embed - 2.25 * np.eye(4),
          np.eye(4) / 4, 'canceled normal line is not one central-weight eigenline')
    for t in algebra:
        close(embed.conj().T @ t @ embed, 0, 'all color densities vanish on paired line')
    for d in (2, 3, 4):
        base = [g[np.ix_([0, 8, 16, 24], [0, 8, 16, 24])] for g in gamma[:d]]
        for g, b in zip(gamma[:d], base):
            close(g @ embed, embed @ b, 'complete free spatial dynamics preserves paired line')
            for t in algebra:
                close(embed.conj().T @ g @ t @ embed, 0, 'all spatial current compressions vanish')
        # Arbitrary positive-energy directions, amplitudes and interference are retained.
        momenta = rng.normal(size=(5, d))
        modes = []
        for p in momenta:
            spin = unit(symbol_projector(p, base) @ (rng.normal(size=4) + 1j*rng.normal(size=4)))
            modes.append(embed @ spin)
        maximum = 0.
        for time in (0., .31, 1.2):
            for position in (np.zeros(d), rng.normal(size=d), rng.normal(size=d)):
                phases = np.exp(1j * (momenta @ position - np.linalg.norm(momenta, axis=1)*time))
                wave = sum(phase * mode for phase, mode in zip(phases, modes))
                density, spatial = currents(wave, gamma[:d], algebra)
                maximum = max(maximum, float(np.max(np.abs(density))), float(np.max(np.abs(spatial))))
                close(density, 0, 'interfering positive-frequency packet density remains canceled')
                close(spatial, 0, 'interfering packet spatial currents remain canceled')
        CASES.append({'case': f'complex-paired-normal-line-depth-{d}', 'real_depth': d,
                      'linear_branch_rank': 4, 'normal_casimir': '4/3',
                      'normal_number_variance': '1/4', 'complex_symplectic_norm': 'I_4',
                      'maximum_packet_source': maximum,
                      'full_free_spatial_dynamics_preserved': True,
                      'fibre_singlet': False, 'physical_particle_identified': False})


def pure_triplet_floor(gamma, algebra, number, rng):
    normal_number = np.diag(sum(number)).real[:8]
    indices = np.flatnonzero(np.isclose(normal_number, 2))
    for d in (2, 3, 4):
        p = rng.normal(size=d)
        base = [g[np.ix_([0, 8, 16, 24], [0, 8, 16, 24])] for g in gamma[:d]]
        values, vectors = np.linalg.eigh(symbol_projector(p, base))
        spin_basis = vectors[:, values > .5]
        check(spin_basis.shape[1] == 2, 'positive base symbol has exactly two spin multiplicities')
        eigenvalues = []
        for sample in range(5):
            if sample == 0:
                coefficients = np.array([[1., 0, 0], [0, 1., 0]], complex) / math.sqrt(2)
            else:
                coefficients = rng.normal(size=(2, 3)) + 1j*rng.normal(size=(2, 3))
                coefficients /= np.linalg.norm(coefficients)
            field = np.zeros((4, 8), complex)
            field[:, indices] = spin_basis @ coefficients
            psi = field.reshape(32)
            reduced = coefficients.conj().T @ coefficients
            q, j = currents(psi, gamma[:d], algebra)
            norm = float(q @ q)
            purity = float(np.trace(reduced @ reduced).real)
            close(norm, (purity - 1/3) / 2, 'SU(3) completeness fixes pure triplet source norm')
            check(norm >= 1/12 - 1e-10, 'positive pure triplet cannot satisfy zero classical Gauss density')
            if sample == 0:
                close(norm, 1/12, 'rank-two color reduction saturates exact source lower bound')
            eigenvalues.append(norm)
        CASES.append({'case': f'positive-pure-triplet-floor-depth-{d}', 'real_depth': d,
                      'positive_spin_multiplicity': 2, 'exact_density_norm_floor': '1/12',
                      'sample_density_norms': eigenvalues, 'zero_source_allowed': False})


def residual_entangled_state(momentum, gamma, algebra, rng):
    positive = symbol_projector(momentum, gamma)
    up = (np.eye(32) + 2*algebra[2])/2
    # Remove the residual singlets: they have Tz=0 rather than +1/2.
    up = up @ (2*algebra[2] @ algebra[2] + algebra[2])
    projector = positive @ up
    values, vectors = np.linalg.eigh(projector)
    basis = vectors[:, values > .9]
    check(basis.shape[1] == 4, 'positive residual up sector has four spin multiplicities')
    random = rng.normal(size=(4, 2)) + 1j*rng.normal(size=(4, 2))
    orthogonal, _ = np.linalg.qr(random)
    first, second = (basis @ orthogonal[:, i] for i in range(2))
    lowered = (algebra[0] - 1j*algebra[1]) @ second
    close(np.linalg.norm(lowered), 1, 'SU(2) lowering preserves fundamental norm')
    psi = (first + lowered)/math.sqrt(2)
    return psi


def positive_symbol_controls(gamma, color, residual, geometry, rng):
    for d in DEPTHS:
        hodge = geometry.hodge(d if d % 2 == 0 else d + 1)
        hodge_active = bool(np.linalg.norm(hodge) > 1e-10)
        check(hodge_active == (d in (3, 4)), 'actual Hodge coefficient is tested at every occupied depth')
        algebra = color if d <= 4 else residual if d <= 6 else []
        casimir = sum((t @ t for t in algebra), np.zeros((32, 32), complex))
        for sample in range(4):
            p = rng.normal(size=d)
            projector = symbol_projector(p, gamma[:d])
            psi = unit(projector @ (rng.normal(size=32) + 1j*rng.normal(size=32)))
            q, j = currents(psi, gamma[:d], algebra)
            close(j, np.outer(p/np.linalg.norm(p), q) if algebra else np.empty((d, 0)),
                  'positive symbol relates every current to density without dropping a spatial direction')
            close(np.vdot(psi, sum(a*g for a, g in zip(p, gamma[:d])) @ psi),
                  np.linalg.norm(p), 'positive mode has positive bare spatial energy')
            if d in (5, 6):
                canceled = residual_entangled_state(p, gamma[:d], residual, rng)
                q0, j0 = currents(canceled, gamma[:d], residual)
                close(q0, 0, 'entangled nonsinglet residual SU(2) density cancels')
                close(j0, 0, 'all residual SU(2) spatial currents cancel in a positive mode')
                close(np.vdot(canceled, casimir @ canceled), .75,
                      'current-canceled residual mode retains fundamental Casimir')
                close(np.vdot(canceled, canceled), 1, 'common complex mode amplitude has positive kinetic norm')
            CASES.append({'case': f'positive-symbol-depth-{d}-sample-{sample}', 'real_depth': d,
                          'momentum': p.tolist(), 'algebra_dimension': len(algebra),
                          'hodge_coefficient_active': hodge_active,
                          'maximum_symbol_identity_error': float(np.max(np.abs(j-np.outer(p/np.linalg.norm(p), q)))) if algebra else 0.,
                          'residual_canceled_nonsinglet_exists': d in (5, 6),
                          'residual_casimir': '3/4' if d in (5, 6) else None})


def packet_and_reality_controls(gamma, residual, rng):
    spin_gamma = [g[::4, ::4] for g in gamma[:6]]
    conjugation = spin_gamma[0] @ spin_gamma[2] @ spin_gamma[4]
    close(conjugation @ conjugation.conj(), -np.eye(8), 'spatial charge conjugation is quaternionic')
    for g in spin_gamma:
        close(conjugation @ g.conj(), g @ conjugation, 'antiunitary map commutes with full spatial Clifford action')
    def embed_pair(f):
        result = np.zeros((8, 4), complex)
        result[:, 2] = f
        result[:, 1] = conjugation @ f.conj()
        return result.reshape(32)
    for d in (5, 6):
        # Reality pairing cancels all currents but pulls the first-order symplectic form to zero.
        f = rng.normal(size=8) + 1j*rng.normal(size=8)
        psi = embed_pair(f)
        q, j = currents(psi, gamma[:d], residual)
        close(q, 0, 'quaternionic reality pairing cancels algebraic density')
        close(j, 0, 'quaternionic reality pairing cancels every spatial current')
        close(np.vdot(psi, psi).real, 2*np.vdot(f, f).real, 'reality-paired norm is positive')
        for _ in range(4):
            h = rng.normal(size=8) + 1j*rng.normal(size=8)
            k = rng.normal(size=8) + 1j*rng.normal(size=8)
            close(np.vdot(embed_pair(h), embed_pair(k)).imag, 0,
                  'imposed antiunitary relation has degenerate first-order symplectic form')
        p = rng.normal(size=d)
        positive = symbol_projector(p, spin_gamma[:d])
        f_positive = unit(positive @ f)
        conjugate_partner = conjugation @ f_positive.conj()
        symbol = sum(p_i*g for p_i, g in zip(p, spin_gamma[:d]))
        first_energy = np.vdot(f_positive, symbol @ f_positive).real
        # Conjugating the full spatial wave reverses its momentum as well as its phase.
        second_energy = np.vdot(conjugate_partner, -symbol @ conjugate_partner).real
        close(first_energy + second_energy, 0, 'imposed conjugate full waves cancel bare first-order Hamiltonian')
        # Two individually positive source-free modes need not remain source-free when interfered.
        p = rng.normal(size=d)
        r = rng.normal(size=d)*1.7
        first = residual_entangled_state(p, gamma[:d], residual, rng)
        second = residual_entangled_state(r, gamma[:d], residual, rng)
        sources = []
        for phase in (0., .4, 1.3, 2.2):
            wave = first + np.exp(1j*phase)*second
            density, spatial = currents(wave, gamma[:d], residual)
            sources.append(float(np.sum(density*density) + np.sum(spatial*spatial)))
        check(max(sources) > 1e-4, 'positive canceled modes generically source color when coherently superposed')
        # Matrix-algebra dimensions prove the absence of a fixed colored complex-linear Clifford-invariant dark subspace.
        blades = []
        for order in range(d+1):
            for indices in combinations(range(d), order):
                blade = np.eye(8, dtype=complex)
                for i in indices:
                    blade = blade @ spin_gamma[i]
                blades.append(blade.reshape(-1))
        algebra_rank = int(np.linalg.matrix_rank(np.column_stack(blades), tol=1e-9))
        check(algebra_rank == (32 if d == 5 else 64), 'full complex Clifford algebra has the stated irreducible spatial blocks')
        for _ in range(4):
            eta = unit(rng.normal(size=2) + 1j*rng.normal(size=2))
            f = unit(rng.normal(size=8) + 1j*rng.normal(size=8))
            separable = np.zeros((8, 4), complex)
            separable[:, [2, 1]] = np.outer(f, eta)
            density, _ = currents(separable.reshape(32), gamma[:d], residual)
            close(density @ density, .25, 'a fundamental color line has exact SU(2) source norm one quarter')
        CASES.append({'case': f'residual-packet-and-reality-depth-{d}', 'real_depth': d,
                      'full_spatial_clifford_algebra_dimension': algebra_rank,
                      'coherent_two_mode_source_norms': sources,
                      'antiunitary_current_cancellation': True,
                      'antiunitary_first_order_symplectic_form': '0',
                      'antiunitary_full_wave_energy': '0',
                      'pure_residual_color_line_density_norm': '1/4',
                      'fixed_colored_complex_linear_full_clifford_dark_subspace': False,
                      'all_nonlinear_source_free_packets_excluded': False})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--geometry-source', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location('gauss_geometry', args.geometry_source)
    geometry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(geometry)
    gamma, color, residual, number = geometry.build_geometry()
    rng = np.random.default_rng(2026101021)
    color_paired_line(gamma, color, number, rng)
    pure_triplet_floor(gamma, color, number, rng)
    positive_symbol_controls(gamma, color, residual, geometry, rng)
    packet_and_reality_controls(gamma, residual, rng)
    report = {'title': 'Classical Zero Color Currents and Complete Spatial Dynamics',
              'sources': {'experiments/classical-color-source-checks.py': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                          'experiments/hodge-gated-gauss-checks.py': hashlib.sha256(args.geometry_source.read_bytes()).hexdigest()},
              'checks_passed': CHECKS, 'dependency_checks_passed': geometry.CHECKS,
              'case_count': len(CASES), 'occupied_depths': list(DEPTHS),
              'scope': 'actual finite-parent Clifford and source algebra; free-mode controls, not localized particles or derived occupation',
              'cases': CASES}
    args.report.write_text(json.dumps(report, indent=2) + '\n')
    print(f'{CHECKS} current and kinetic checks plus {geometry.CHECKS} dependency checks passed across {len(CASES)} cases')


if __name__ == '__main__':
    main()
