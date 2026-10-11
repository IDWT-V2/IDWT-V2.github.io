#!/usr/bin/env python3
"""Retain a dynamical pair phase in zero-moment color reduction and audit its lift."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import math

import numpy as np
from scipy.linalg import expm


CHECKS = 0
CASES = []
DEPTHS = (2, 3, 4, 5, 6, 10)


def check(condition, message):
    global CHECKS
    CHECKS += 1
    if not condition:
        raise AssertionError(message)


def close(actual, expected, message, atol=3e-10):
    check(np.allclose(actual, expected, atol=atol, rtol=3e-10), message)


def unit(v):
    return v/np.linalg.norm(v)


def su_generators(k):
    result = []
    for i in range(k):
        for j in range(i+1, k):
            t = np.zeros((k, k), complex); t[i, j] = t[j, i] = .5; result.append(t)
            t = np.zeros((k, k), complex); t[i, j] = -.5j; t[j, i] = .5j; result.append(t)
    for j in range(1, k):
        diagonal = np.zeros(k); diagonal[:j] = 1; diagonal[j] = -j
        result.append(np.diag(diagonal)/math.sqrt(2*j*(j+1)))
    return result


def quotient_controls(rng):
    generators = su_generators(3)
    for sample in range(8):
        r = rng.uniform(.2, 2.)
        phi = rng.uniform(-3., 3.)
        u = unit(rng.normal(size=3)+1j*rng.normal(size=3))
        z = r*np.exp(.5j*phi)*u
        w = r*np.exp(.5j*phi)*u.conj()
        pair = np.r_[z, w]
        density = np.array([np.vdot(z, t@z)-np.vdot(w, t.T@w) for t in generators])
        close(density, 0, 'actual direct-sum color moment vanishes')
        close(np.outer(z, z.conj())-np.outer(w.conj(), w), 0,
              'zero-moment rank-one Hermitian density equality')
        invariant = w @ z
        close(invariant, r*r*np.exp(1j*phi), 'bilinear invariant retains the independent pair phase')
        dr = pair/r
        dphi = .5j*pair
        close(np.vdot(dr, dr).real, 2, 'reduced radial metric')
        close(np.vdot(dphi, dphi).real, r*r/2, 'reduced angular metric')
        close(np.vdot(dr, dphi).real, 0, 'radial and angular coordinates are metric orthogonal')
        close(2*np.vdot(dr, dphi).imag, 2*r, 'nonzero canonical symplectic pairing survives the free pair phase')
        close(-np.vdot(pair, dphi).imag, -r*r, 'first-order canonical one-form is minus R dphi')
        for t in generators:
            gauge = np.r_[1j*t@z, -1j*t.T@w]
            close(2*np.vdot(dr, gauge).imag, 0, 'gauge direction is symplectically null on radial tangent')
            close(2*np.vdot(dphi, gauge).imag, 0, 'gauge direction is symplectically null on phase tangent')
            close(np.vdot(dr, gauge).real, 0, 'radial representative is metric horizontal')
            close(np.vdot(dphi, gauge).real, 0, 'symmetric pair-phase representative is metric horizontal')
        rotation = expm(1j*sum(rng.normal()*t for t in generators))
        close((rotation.conj()@w) @ (rotation@z), invariant, 'non-Abelian gauge orbit preserves the invariant bilinear')
        charge_cases = []
        for label, weights in [('exterior', (1., 2.)), ('spin-lift', (-.5, .5)),
                                ('common-inverse-determinant', (-2., -1.)),
                                ('mixed-determinant-branches', (1., -1.))]:
            theta = .37
            rotated_invariant = (np.exp(1j*weights[1]*theta)*w) @ (np.exp(1j*weights[0]*theta)*z)
            total = sum(weights)
            close(rotated_invariant, np.exp(1j*total*theta)*invariant, 'pair central character uses the actual specified lift')
            close((weights[0]*np.vdot(z, z)+weights[1]*np.vdot(w, w))/3,
                  total*r*r/3, 'electric moment agrees with the reduced phase generator')
            central_velocity = np.r_[1j*weights[0]*z, 1j*weights[1]*w]
            difference = (weights[0]-weights[1])/2
            gauge_generator = difference/(1-1/3)*(np.outer(u, u.conj())-np.eye(3)/3)
            gauge_velocity = np.r_[1j*gauge_generator@z, -1j*gauge_generator.T@w]
            close(central_velocity, total*dphi+gauge_velocity,
                  'central action splits into the measured quotient phase plus a true SU3 gauge motion')
            charge_cases.append({'lift': label, 'component_weights': list(weights),
                                 'bilinear_weight': total, 'determinant_normalized_weight': total/3})
        omega = .83
        time = .51
        evolved_z = np.exp(-1j*omega*time)*z
        evolved_w = np.exp(-1j*omega*time)*w
        close(evolved_w@evolved_z, np.exp(-2j*omega*time)*invariant,
              'bilinear frequency is twice the constituent field frequency under the common linear clock')
        CASES.append({'case': f'normal-SU3-phase-quotient-{sample}', 'radius_r': r, 'pair_phase': phi,
                      'quotient_coordinate': 'B = w^T z = R exp(i phi), R = r^2',
                      'canonical_one_form': '-R dphi', 'canonical_two_form': '-dR wedge dphi',
                      'reduced_metric': '(dR^2 + R^2 dphi^2)/(2R)',
                      'central_characters': charge_cases,
                      'common_clock_bilinear_frequency_factor': 2,
                      'fixed_absolute_electric_charge_derived': False})


def sector_current_controls(geometry, rng):
    gamma, color, residual, number = geometry.build_geometry()
    parity4 = np.diag([1., -1., -1., 1.])
    epsilon2 = np.array([[0., 1.], [-1., 0.]])
    for d in DEPTHS:
        hodge = geometry.hodge(d if d % 2 == 0 else d+1)
        active = bool(np.linalg.norm(hodge)>1e-10)
        check(active == (d in (3, 4)), 'phase quotient audit retains the same all-sector Hodge gate')
        if d <= 4:
            r = .9
            phi = .4
            normal = np.zeros(8, complex)
            normal[4] = normal[3] = r*np.exp(.5j*phi)
            spin = unit(rng.normal(size=4)+1j*rng.normal(size=4))
            psi = np.kron(spin, normal)
            q = np.array([np.vdot(psi, t@psi) for t in color])
            j = np.array([[np.vdot(psi, g@t@psi) for t in color] for g in gamma[:d]])
            close(q, 0, 'normal pair phase permits zero SU3 density on every base spinor')
            close(j, 0, 'normal pair phase permits every full base spatial current to vanish')
            available = True
            residual_norm = None
        elif d <= 6:
            r = .9
            z = r*unit(rng.normal(size=2)+1j*rng.normal(size=2))
            w = np.exp(.4j)*epsilon2@z.conj()
            normal = np.zeros(8, complex)
            normal[[2, 1]] = z
            normal[[6, 5]] = w
            spin = unit(rng.normal(size=4)+1j*rng.normal(size=4))
            psi = np.kron(spin, normal)
            q = np.array([np.vdot(psi, t@psi) for t in residual])
            j = np.array([[np.vdot(psi, g@t@psi).real for t in residual] for g in gamma[:d]])
            close(q, 0, 'a residual SU2 density quotient exists in the actual exterior carrier')
            close(j[:4], 0, 'base directions retain zero color current')
            chirality = np.vdot(spin, parity4@spin).real
            close(j[4]@j[4], chirality**2*r**4,
                  'added fifth spatial direction has an exact residual current obstruction')
            if d == 6:
                close(j[5]@j[5], chirality**2*r**4,
                      'sixth direction independently retains the residual current obstruction')
            check(np.linalg.norm(j[4:])>1e-4, 'zero density alone does not satisfy the full inactive source constraints')
            available = False
            residual_norm = float(np.sum(j[4:]**2))
        else:
            available = False
            residual_norm = None
        CASES.append({'case': f'actual-sector-phase-quotient-depth-{d}', 'real_depth': d,
                      'hodge_coefficient_active': active, 'normal_algebra_dimension': 8 if d<=4 else 3 if d<=6 else 0,
                      'normal_SU3_pair_reduction_retains_arbitrary_base_spinor': available,
                      'residual_added_direction_current_norm': residual_norm,
                      'all_particle_charges_or_occupations_selected': False})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--geometry-source', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location('gauss_geometry', args.geometry_source)
    geometry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(geometry)
    rng = np.random.default_rng(2026101023)
    quotient_controls(rng)
    sector_current_controls(geometry, rng)
    report = {'title': 'A Dynamical Zero-Color Pair Phase and Its Actual Central Character',
              'sources': {'experiments/color-phase-quotient-checks.py': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                          'experiments/hodge-gated-gauss-checks.py': hashlib.sha256(args.geometry_source.read_bytes()).hexdigest()},
              'checks_passed': CHECKS, 'dependency_checks_passed': geometry.CHECKS,
              'case_count': len(CASES), 'occupied_depths': list(DEPTHS),
              'scope': 'explicit local commuting-amplitude zero-moment reduction and full spatial-current audit; not physical charge quantization or a particle identification',
              'cases': CASES}
    args.report.write_text(json.dumps(report, indent=2)+'\n')
    print(f'{CHECKS} quotient and current checks plus {geometry.CHECKS} geometry checks passed across {len(CASES)} cases')


if __name__ == '__main__':
    main()
