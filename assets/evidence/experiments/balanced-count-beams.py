"""Exact balanced monogenic beams, count clocks and observer rotation controls."""
from datetime import datetime, timezone
from fractions import Fraction
from hashlib import sha256
from importlib.util import spec_from_file_location, module_from_spec
from math import comb, factorial
from pathlib import Path
import json

import numpy as np
from scipy.integrate import quad
from scipy.special import betainc, betaincinv, beta as beta_function

ROOT = Path(__file__).resolve().parent
spec = spec_from_file_location('minimal', ROOT/'minimal-count-carrier.py')
minimal = module_from_spec(spec)
spec.loader.exec_module(minimal)
forward = minimal.forward
COUNTS, ERRORS = {}, {}


def check(value, label):
    if not bool(value):
        raise AssertionError(label)
    COUNTS[label] = COUNTS.get(label, 0)+1


def close(actual, expected, label, scale=None, tolerance=2e-10):
    actual, expected = np.asarray(actual), np.asarray(expected)
    if scale is None:
        scale = max(1., np.linalg.norm(expected))
    error = float(np.linalg.norm(actual-expected)/scale)
    ERRORS[label] = max(ERRORS.get(label, 0.), error)
    check(error < tolerance, label)


def normalization(depth, level):
    answer = Fraction(1)
    for k in range(1, level+1):
        answer *= Fraction(2*k+depth-1, 2*k)
    return answer


def beam_jets(position, ambient, seed, level, plane, normal):
    left, right = plane
    rank, dim = len(ambient[0]), len(ambient)
    eye = np.eye(rank)
    involution = 1j*ambient[left]@ambient[right]
    pminus, pplus = (eye-involution)/2, (eye+involution)/2
    w = position[left]+1j*position[right]
    dw = np.zeros(dim, complex); dw[left] = 1; dw[right] = 1j
    field = normal*(w**level*pminus+np.conj(w)**level*pplus)@seed
    gradient = np.zeros((dim,)+field.shape, complex)
    hessian = np.zeros((dim, dim)+field.shape, complex)
    if level:
        for i in range(dim):
            gradient[i] = normal*level*(w**(level-1)*dw[i]*pminus
                +np.conj(w)**(level-1)*np.conj(dw[i])*pplus)@seed
    if level >= 2:
        for i in range(dim):
            for j in range(dim):
                hessian[i, j] = normal*level*(level-1)*(w**(level-2)*dw[i]*dw[j]*pminus
                    +np.conj(w)**(level-2)*np.conj(dw[i]*dw[j])*pplus)@seed
    return field, gradient, hessian


def mass_jets(position, ambient, jets, aa, bb):
    bare, grad, hess = jets
    gx = sum(x*g for x, g in zip(position, ambient))
    mix = aa*np.eye(len(gx))+bb*gx
    field = mix@bare
    gradient = np.array([mix@g+bb*gamma@bare for gamma, g in zip(ambient, grad)])
    hessian = np.array([[mix@hess[i, j]+bb*(ambient[i]@grad[j]+ambient[j]@grad[i])
                         for j in range(len(ambient))] for i in range(len(ambient))])
    return field, gradient, hessian


def sphere_dirac(position, ambient, field, gradient):
    gx = sum(x*g for x, g in zip(position, ambient))
    euler = sum(x*g for x, g in zip(position, gradient))
    return -gx@sum(g@jet for g, jet in zip(ambient, gradient))+euler+(len(ambient)-1)/2*field


def rotation(position, ambient, jets, pair, squared=False):
    field, grad, hess = jets
    left, right = pair
    tangent = np.zeros(len(position)); tangent[left] = -position[right]; tangent[right] = position[left]
    first = sum(v*g for v, g in zip(tangent, grad))
    spin = -.5j*ambient[left]@ambient[right]
    if not squared:
        return -1j*first+spin@field
    acceleration = np.zeros(len(position)); acceleration[left] = -position[left]; acceleration[right] = -position[right]
    second = sum(acc*g for acc, g in zip(acceleration, grad))
    second += sum(tangent[i]*tangent[j]*hess[i, j] for i in range(len(position)) for j in range(len(position)))
    return -second-2j*spin@first+spin@spin@field


def main():
    rng = np.random.default_rng(202610121006)
    cases, sectors = [], []
    prepared = {d: minimal.preparation(d) for d in forward.radial.DEPTHS}
    for depth, (_, _, ambient, seed, _) in prepared.items():
        eye = np.eye(len(ambient[0]))
        for left in range(depth+1):
            for right in range(left+1, depth+1):
                jj = 1j*ambient[left]@ambient[right]
                check(np.array_equal((ambient[left]+1j*ambient[right])@(eye-jj), np.zeros_like(eye)),
                      'exact Clifford arithmetic makes each holomorphic beam monogenic')
                check(np.array_equal((ambient[left]-1j*ambient[right])@(eye+jj), np.zeros_like(eye)),
                      'exact Clifford arithmetic makes each conjugate beam monogenic')
        sectors.append(dict(depth=depth, minimal_columns=seed.shape[1],
            observer_fixed_embedding_dimension=max(0, depth-2),
            fully_normal_spatial_dimension=max(0, depth-3),
            observer_invariant_circle_available=depth>=4,
            observer_invariant_fully_normal_circle_available=depth>=5,
            restriction_zero_not_unique_to_five=depth>=5))
    for name, depth, index in forward.radial.ASSIGNMENTS:
        _, _, ambient, seed, _ = prepared[depth]
        level, h = index-1, index-1+depth/2
        count, first, _ = forward.count_data(depth, level)
        epsilon = Fraction(3, 10)/max(forward.count_data(depth, n-1)[1]
                    for _, d, n in forward.radial.ASSIGNMENTS if d == depth)
        cc = float(epsilon*first); xx = np.sqrt(1-cc*cc)
        aa, bb = np.sqrt((1+cc)/2), np.sqrt((1-cc)/2)
        mass, lam, omega = h*xx/cc, h/cc, float(epsilon*count)
        eta = omega-lam
        nq = normalization(depth, level); normal = np.sqrt(float(nq))
        transverse_mean = Fraction(depth-1, 2*level+depth+1)
        beta_a, beta_b = level+1, (depth-1)/2
        density_t = lambda t: t**level*(1-t)**(beta_b-1)/beta_function(beta_a, beta_b)
        integrated = quad(density_t, 0, 1, epsabs=2e-10, epsrel=2e-10)[0]
        close(integrated, 1., 'independent scalar integration normalizes the classical beam energy')
        close(quad(lambda t: (1-t)*density_t(t), 0, 1, epsabs=2e-10)[0], float(transverse_mean),
              'independent energy integration gives the exact transverse width ratio')
        tube_angles = {}
        for fraction in [.5, .9, .99]:
            boundary = betaincinv(beta_a, beta_b, 1-fraction)
            angle = np.arccos(np.sqrt(boundary))
            close(quad(density_t, boundary, 1, epsabs=2e-10, epsrel=2e-10)[0], fraction,
                  'independent tube integration checks inverse-beta concentration radii')
            tube_angles[str(fraction)] = float(angle)
        planes = [(1, 2)]
        if depth>=4:
            planes.append((0, 4))
        if depth>=5:
            planes.append((4, 5))
        for plane in planes:
            for sample in range(4):
                position = rng.normal(size=depth+1)
                position[list(plane)] *= 4
                position /= np.linalg.norm(position)
                bare_jets = beam_jets(position, ambient, seed, level, plane, normal)
                bare, gradient, _ = bare_jets
                scale = np.linalg.norm(bare)
                gx = sum(v*g for v, g in zip(position, ambient))
                rho = np.vdot(bare, bare).real
                close(sum(g@jet for g, jet in zip(ambient, gradient)), np.zeros_like(bare),
                      'actual ambient jets verify complete beam monogenicity', scale=scale*max(1, level))
                close(sphere_dirac(position, ambient, bare, gradient), h*bare,
                      'the full sphere Dirac operator gives the intended positive shell', scale=h*scale)
                close(np.vdot(bare, gx@bare), 0., 'the bare balanced beam has zero scalar source', scale=rho)
                for i in range(depth+1):
                    alpha = 1j*(ambient[i]@gx-position[i]*np.eye(len(gx)))
                    close(np.vdot(bare, alpha@bare), 0., 'the bare balanced beam has zero tangential current', scale=rho)
                jets = mass_jets(position, ambient, bare_jets, aa, bb)
                field, grad, _ = jets
                density, scalar = np.vdot(field, field).real, np.vdot(field, gx@field).real
                close(density, rho, 'the massive beam retains its complete classical density', scale=rho)
                close(scalar, xx*rho, 'the massive beam has the intended constant constitutive ratio', scale=rho)
                df = sphere_dirac(position, ambient, field, grad)
                source = mass*gx@field+eta*field
                close((df+source)/lam, omega*field/lam,
                      'the complete spatial field equation retains the count clock', scale=scale, tolerance=5e-10)
                energy = np.vdot(field, df).real+mass*scalar+eta*density
                close(energy/lam, omega*density/lam,
                      'the complete pointwise energy retains spatial kinetic and contact terms', scale=rho, tolerance=5e-10)
                if plane == (1, 2):
                    close(rotation(position, ambient, jets, plane, True), (level+.5)**2*field,
                          'the visible-plane control has extra full orbital angular momentum', scale=(level+.5)**2*scale)
                else:
                    observer_pairs = [(1, 2), (2, 3), (3, 1)]
                    casimir = sum(rotation(position, ambient, jets, pair, True) for pair in observer_pairs)
                    close(casimir, .75*field,
                          'observer-invariant beams retain the full half-spin Casimir', scale=scale)
                    for pair in observer_pairs:
                        spin = -.5j*ambient[pair[0]]@ambient[pair[1]]
                        seed_jets = beam_jets(position, ambient, spin@seed, level, plane, normal)
                        expected = mass_jets(position, ambient, seed_jets, aa, bb)[0]
                        close(rotation(position, ambient, jets, pair), expected,
                              'full observer rotation intertwines the actual massive beam and seed', scale=scale)
            if plane == (4, 5):
                position = np.zeros(depth+1); position[:4] = rng.normal(size=4); position /= np.linalg.norm(position)
                restricted = mass_jets(position, ambient,
                    beam_jets(position, ambient, seed, level, plane, normal), aa, bb)
                check(np.count_nonzero(restricted[0]) == 0,
                      'fully normal positive-level beams vanish on the observer equatorial sector')
                if level>=2:
                    check(np.count_nonzero(restricted[1]) == 0,
                          'the normal beam has a higher-order zero rather than a prescribed readout coefficient')
        cases.append(dict(particle=name, depth=depth, inferred_index=index,
            shell_level=level, count_exact=str(count), normalization_exact=str(nq),
            mean_transverse_squared_distance_over_radius_squared_exact=str(transverse_mean),
            tube_half_angles_radians=tube_angles, planes_tested=[list(p) for p in planes],
            omega_in_radius_units_exact=str(epsilon*count),
            classification='exact classical count-compatible circle-concentrated field, not identified observed particle'))
        print(json.dumps(dict(completed=name, depth=depth, transverse_width_exact=str(transverse_mean))), flush=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    sources = [Path(__file__), ROOT/'minimal-count-carrier.py', ROOT/'forward-count-action.py',
               ROOT/'inverse-current-radial.py', ROOT/'excited-shell-orientation-current.py']
    report = dict(experiment_key='balanced-monogenic-count-beams', created_utc=datetime.now(timezone.utc).isoformat(),
        sources={'experiments/'+p.name: sha256(p.read_bytes()).hexdigest() for p in sources},
        evaluations=sum(COUNTS.values()), conditions=COUNTS, maximum_errors=ERRORS, sectors=sectors, cases=cases,
        limits=['round sector spheres and effective count action remain conditional',
                'tube concentration around a great circle is not a mobile localized object in unbounded space',
                'energy integration is classical and does not introduce electron position probabilities',
                'fully normal circles occur at five six and ten; no unique neutrino coupling suppression follows',
                'an observer-invariant circle already exists at depth four using the center coordinate',
                'pure beam nodes prevent transfer of the nodeless nonlinear persistence proof',
                'half-spin field transformation does not identify a nonzero classical mean spin'])
    output = ROOT/('balanced-count-beams-results-'+stamp+'.json')
    output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(report=str(output), evaluations=report['evaluations'], maximum_errors=ERRORS)), flush=True)


if __name__ == '__main__':
    main()
