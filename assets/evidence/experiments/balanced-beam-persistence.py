"""Real balanced shell sections and nodeless additions to count beams."""
from datetime import datetime, timezone
from hashlib import sha256
from importlib.util import spec_from_file_location, module_from_spec
from itertools import combinations
from math import comb
from pathlib import Path
import json

import numpy as np
import sympy as sp

ROOT = Path(__file__).resolve().parent
spec = spec_from_file_location('beams', ROOT/'balanced-count-beams.py')
beams = module_from_spec(spec)
spec.loader.exec_module(beams)
COUNTS, ERRORS = {}, {}


def check(value, label):
    if not bool(value):
        raise AssertionError(label)
    COUNTS[label] = COUNTS.get(label, 0)+1


def close(actual, expected, label, scale=1., tolerance=2e-10):
    error = float(np.linalg.norm(np.asarray(actual)-np.asarray(expected))/scale)
    ERRORS[label] = max(ERRORS.get(label, 0.), error)
    check(error<tolerance, label)


def zonal_jets(position, ambient, seed, level):
    depth, t = len(ambient)-1, position[0]
    aa, bb, ap, bp = beams.forward.orientation.polynomials(depth, level, np.array(t))
    gx = sum(x*g for x, g in zip(position, ambient))
    eye = np.eye(len(gx))
    matrix = (aa-t*bb)*eye+bb*ambient[0]@gx
    g = comb(level+depth-1, depth-1)
    field = matrix@seed/np.sqrt(g)
    grad = np.array([((int(i==0)*((ap-bb-t*bp)*eye+bp*ambient[0]@gx)
                       +bb*ambient[0]@gamma)@seed)/np.sqrt(g)
                     for i, gamma in enumerate(ambient)])
    return field, grad, float((aa*aa+bb*bb*(1-t*t))/g), matrix/np.sqrt(g)


def exact_real_structure(depth, ambient):
    for g in ambient:
        check(np.array_equal(g.real, np.rint(g.real)) and np.array_equal(g.imag, np.rint(g.imag)),
              'the Clifford entries are exactly Gaussian integers before symbolic conversion')
    matrices = [sp.Matrix(g.shape[0], g.shape[1], lambda i, j:
                 sp.Integer(int(g[i, j].real))+sp.I*sp.Integer(int(g[i, j].imag))) for g in ambient]
    bb = sp.eye(8)
    for ix in ([0, 2, 4] if depth==5 else [0, 2, 4, 6]):
        bb *= matrices[ix]
    check(bb*bb.conjugate()==sp.eye(8), 'the balanced real structure is an exact involution')
    for gamma in matrices:
        check(bb*gamma.conjugate()==-gamma*bb,
              'every Hermitian vector has odd antiunitary parity and zero expectation on real spinors')
    for a, b in combinations(matrices, 2):
        hh = sp.I*a*b
        check(bb*hh.conjugate()==-hh*bb,
              'every Hermitian bivector also has odd antiunitary parity')
    # Realification of B conjugation; its fixed-space dimension is exactly eight.
    real_map = bb.applyfunc(sp.re); imag_map = bb.applyfunc(sp.im)
    involution = real_map.row_join(imag_map).col_join(imag_map.row_join(-real_map))
    check((involution-sp.eye(16)).rank()==8,
          'the invariant real balanced seed space has exactly eight real dimensions')
    return np.array(bb, complex)


def main():
    rng = np.random.default_rng(202610121012)
    cases = []
    for depth in beams.forward.radial.DEPTHS:
        _, _, ambient, seed, _ = beams.minimal.preparation(depth)
        real_structure = exact_real_structure(depth, ambient) if depth in [5, 6] else None
        real_seed = seed*np.exp(-.25j*np.pi) if depth==5 else seed
        if real_structure is not None:
            close(real_structure@real_seed.conj(), real_seed,
                  'the beam seed has a real-structure phase choice')
        for name, dd, index in beams.forward.radial.ASSIGNMENTS:
            if dd!=depth:
                continue
            level, h = index-1, index-1+depth/2
            plane = (4, 5) if depth>=5 else ((0, 4) if depth==4 else (1, 2))
            normal = np.sqrt(float(beams.normalization(depth, level)))
            for sample in range(8):
                position = rng.normal(size=depth+1); position /= np.linalg.norm(position)
                # First verify a concrete direct sum that has no zeros at any depth.
                beam, bg, _ = beams.beam_jets(position, ambient, seed, level, plane, normal)
                zonal, zg, density, matrix = zonal_jets(position, ambient, seed, level)
                close(matrix.conj().T@matrix, density*np.eye(len(matrix)),
                      'the actual global zonal map is invertible with a scalar positive norm', scale=max(1., density))
                check(density>0, 'the zonal background is nonzero at every evaluated position')
                delta = .1
                field = np.concatenate([np.sqrt(1-delta*delta)*beam, delta*zonal], axis=1)
                grad = np.concatenate([np.sqrt(1-delta*delta)*bg, delta*zg], axis=2)
                scale = np.linalg.norm(field)
                close(beams.sphere_dirac(position, ambient, field, grad), h*field,
                      'the concrete nodeless direct sum retains the exact positive shell', scale=h*scale)
                rho = np.vdot(field, field).real
                close(rho, (1-delta*delta)*np.vdot(beam, beam).real+delta*delta*density,
                      'the direct-sum density has its rigorous zonal floor', scale=rho)
                gx = sum(x*g for x, g in zip(position, ambient))
                close(np.vdot(field, gx@field), 0., 'nodeless direct sums preserve scalar balance', scale=rho)
                for i, gamma in enumerate(ambient):
                    alpha = 1j*(gamma@gx-position[i]*np.eye(len(gx)))
                    close(np.vdot(field, alpha@field), 0., 'nodeless direct sums preserve current balance', scale=rho)
                if real_structure is not None:
                    raw = rng.normal(size=(8, 1))+1j*rng.normal(size=(8, 1))
                    v = raw+real_structure@raw.conj(); v /= np.linalg.norm(v)
                    close(real_structure@v.conj(), v, 'independent real seeds remain in the balanced space')
                    bz, bj, _ = beams.beam_jets(position, ambient, real_seed, level, plane, normal)
                    zz, zj, _, zm = zonal_jets(position, ambient, v, level)
                    coherent = np.sqrt(1-delta*delta)*bz+delta*zz
                    cj = np.sqrt(1-delta*delta)*bj+delta*zj
                    rr = np.vdot(coherent, coherent).real
                    close(real_structure@coherent.conj(), coherent,
                          'a coherent beam plus zonal mode remains a real balanced section', scale=np.sqrt(rr))
                    close(beams.sphere_dirac(position, ambient, coherent, cj), h*coherent,
                          'coherent real beam additions retain the full spatial eigenvalue', scale=h*np.sqrt(rr))
                    low = ambient+[1j*a@b for a, b in combinations(ambient, 2)]
                    for moment in low:
                        close(np.vdot(coherent, moment@coherent), 0.,
                              'arbitrary coherent real sections annihilate all local low Clifford moments', scale=rr)
                    inverse = zm.conj().T/density
                    bad_seed = -np.sqrt(1-delta*delta)/delta*inverse@bz
                    close(coherent, delta*zm@(v-bad_seed),
                          'the exact bad-seed map identifies every possible coherent zero', scale=np.sqrt(rr))
                    close(rr, delta*delta*density*np.linalg.norm(v-bad_seed)**2,
                          'distance from the compact bad-seed image gives an exact density bound', scale=rr)
                    close(real_structure@bad_seed.conj(), bad_seed,
                          'the entire bad-seed map belongs to the same real eight-dimensional space',
                          scale=max(1., np.linalg.norm(bad_seed)))
            cases.append(dict(particle=name, depth=depth, inferred_index=index,
                direct_sum_columns=2*seed.shape[1], zonal_energy_fraction_exact='1/100',
                coherent_pure_nodeless_existence_by_dimension=depth in [5, 6],
                bad_seed_image_dimension_bound=depth if depth in [5, 6] else None,
                unit_real_seed_sphere_dimension=7 if depth in [5, 6] else None,
                classification='constructive direct-sum persistence; generic pure existence at five and six'))
            print(json.dumps(dict(completed=name, depth=depth, direct_sum_columns=2*seed.shape[1])), flush=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    source_names = ['balanced-beam-persistence.py', 'balanced-count-beams.py', 'minimal-count-carrier.py',
                    'forward-count-action.py', 'inverse-current-radial.py', 'excited-shell-orientation-current.py']
    report = dict(experiment_key='balanced-beam-nodeless-persistence',
        created_utc=datetime.now(timezone.utc).isoformat(),
        sources={'experiments/'+name: sha256((ROOT/name).read_bytes()).hexdigest() for name in source_names},
        evaluations=sum(COUNTS.values()), conditions=COUNTS, maximum_errors=ERRORS, cases=cases,
        proof_dependencies=['nonzero Gegenbauer polynomial and derivative cannot share a root',
            'a smooth image of a compact d-manifold has Hausdorff dimension at most d',
            'for d five or six this image cannot cover the unit seven-sphere of real seeds',
            'the preferred-time convex-envelope graph-norm proof applies to a positive density floor'],
        limits=['generic pure existence is analytic, not certified by finite spatial samples',
            'no specific random seed is asserted globally nodeless by the checker',
            'the direct sum is globally nodeless by its exact positive density sum',
            'no covariant proper-current or responding-clock stability theorem is transferred',
            'no full containing-space localization or translational inertia theorem follows'])
    path = ROOT/('balanced-beam-persistence-results-'+stamp+'.json')
    path.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(report=str(path), evaluations=report['evaluations'], maximum_errors=ERRORS)), flush=True)


if __name__ == '__main__':
    main()
