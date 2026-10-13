"""Mass-chart contact energy and exact alignment theorem for balanced beam columns."""
from datetime import datetime, timezone
from hashlib import sha256
from importlib.util import spec_from_file_location, module_from_spec
from math import pi
from pathlib import Path
import json

import mpmath as mp
import numpy as np
from scipy.special import betaincinv, ndtri
from scipy.stats import qmc

ROOT = Path(__file__).resolve().parent
spec = spec_from_file_location('envelope', ROOT/'convex-count-envelope.py')
envelope = module_from_spec(spec)
spec.loader.exec_module(envelope)
spec = spec_from_file_location('beams', ROOT/'balanced-count-beams.py')
beams = module_from_spec(spec)
spec.loader.exec_module(beams)
COUNTS, ERRORS = {}, {}


def check(value, label):
    if not bool(value):
        raise AssertionError(label)
    COUNTS[label] = COUNTS.get(label, 0)+1


def close(actual, expected, label, tolerance=2e-10):
    error = float(np.linalg.norm(np.asarray(actual)-np.asarray(expected))/max(1., np.linalg.norm(expected)))
    ERRORS[label] = max(ERRORS.get(label, 0.), error)
    check(error<tolerance, label)


def norm_mp(depth, level):
    return mp.rf(mp.mpf(depth+1)/2, level)/mp.factorial(level)


def exact_contact(depth, level1, level2, candidates):
    one = next(c for c in candidates if c['level']==level1)
    two = next(c for c in candidates if c['level']==level2)
    active = [c for c in candidates if level1<=c['level']<=level2]
    def f(c):
        return c['m']*c['x']+c['eta']
    a = [f(one)-(c['m']*one['x']+c['eta']) for c in active]
    b = [f(two)-(c['m']*two['x']+c['eta']) for c in active]
    check(a[0]==0 and b[-1]==0, 'the two isolated branches have their exact zero contact deficits')
    for aa, bb in zip(a, b):
        check(aa>=0 and bb>=0, 'every geometric chart has nonnegative endpoint deficits')
    radii, weights = [], []
    for i in range(len(active)-1):
        weight = b[i]-b[i+1]
        radius = (a[i+1]-a[i])/weight
        check(weight>0 and radius>0, 'the contact has an exact positive layer-overlap decomposition')
        if radii:
            check(radius>radii[-1], 'the complete intermediate-chart crossings occur in their geometric order')
        radii.append(radius); weights.append(weight)
    check(abs(sum(weights)-b[0])<mp.mpf('1e-75'), 'layer weights telescope to the first endpoint deficit')
    check(abs(sum(w*r for w, r in zip(weights, radii))-a[-1])<mp.mpf('1e-75'),
          'weighted crossing radii telescope to the other endpoint deficit')
    for exponent in [-20, -8, -2, 0, 2, 8, 20]:
        ratio = mp.mpf(10)**exponent
        direct = min(aa+ratio*bb for aa, bb in zip(a, b))
        layered = sum(w*min(ratio, r) for w, r in zip(weights, radii))
        check(abs(direct-layered)/max(1, abs(direct))<mp.mpf('1e-75'),
              'independent minimization verifies the exact layer-cake contact identity')
    for ratio in [mp.mpf('.03'), mp.mpf('.7'), mp.mpf('5')]:
        scalar = one['x']+ratio*two['x']
        rho = 1+ratio
        all_deficit = f(one)+ratio*f(two)-max(c['m']*scalar+c['eta']*rho for c in candidates)
        direct = min(aa+ratio*bb for aa, bb in zip(a, b))
        check(abs(all_deficit-direct)<mp.mpf('1e-73'),
              'the actual maximum over all admissible charts equals the active contact minimum')
    n1, n2 = norm_mp(depth, level1), norm_mp(depth, level2)
    beta_b = mp.mpf(depth-1)/2
    aligned = mp.mpf(0)
    for weight, radius in zip(weights, radii):
        boundary = (radius*n1/n2)**(mp.mpf(1)/(level2-level1))
        if boundary>=1:
            contribution = mp.mpf(1)
        else:
            contribution = mp.betainc(level2+1, beta_b, 0, boundary, regularized=True)
            contribution += radius*mp.betainc(level1+1, beta_b, boundary, 1, regularized=True)
        aligned += weight*contribution
    orthogonal_overlap = None
    orthogonal_bound = None
    if depth>=3:
        orthogonal_overlap = mp.sqrt(n1*n2)*mp.gamma(1+mp.mpf(level1)/2)*mp.gamma(1+mp.mpf(level2)/2)
        orthogonal_overlap /= mp.rf(mp.mpf(depth+1)/2, mp.mpf(level1+level2)/2)
        orthogonal_bound = mp.sqrt(a[-1]*b[0])*orthogonal_overlap
    # Independently integrate the complete kinetic and winning contact terms,
    # without using the layer-overlap expression for the energy difference.
    boundaries = [mp.mpf(0), mp.mpf(1)]
    for radius in radii:
        boundary = (radius*n1/n2)**(mp.mpf(1)/(level2-level1))
        if 0<boundary<1:
            boundaries.append(boundary)
    boundaries.sort()
    interaction_integral = mp.mpf(0)
    for low, high in zip(boundaries, boundaries[1:]):
        middle = (low+high)/2
        rho1, rho2 = n1*middle**level1, n2*middle**level2
        winning = max(candidates, key=lambda c:
                      c['m']*(one['x']*rho1+two['x']*rho2)+c['eta']*(rho1+rho2))
        for own, level in [(one, level1), (two, level2)]:
            fraction = mp.betainc(level+1, beta_b, low, high, regularized=True)
            interaction_integral += (winning['m']*own['x']+winning['eta'])*fraction
    kinetic_integral = one['h']*one['c']+two['h']*two['c']
    total_energy = kinetic_integral+interaction_integral
    check(abs(total_energy-(one['omega']+two['omega']-aligned))<mp.mpf('1e-70'),
          'independent complete kinetic and winning-contact integration verifies the pair energy')
    check((total_energy<0)==(aligned>one['omega']+two['omega']),
          'the sign of complete pair energy is retained rather than hidden in an overlap deficit')
    return one, two, a, b, weights, radii, aligned, orthogonal_overlap, orthogonal_bound, total_energy


def sample_beam(depth, level, plane, size_power, seed):
    sobol = qmc.Sobol(depth+1, scramble=True, seed=seed)
    uu = sobol.random_base2(size_power)
    tt = betaincinv(level+1, (depth-1)/2, uu[:, 0])
    theta = 2*pi*uu[:, 1]
    normals = ndtri(uu[:, 2:]); normals /= np.linalg.norm(normals, axis=1)[:, None]
    a, b = plane
    complement, _ = np.linalg.qr(np.column_stack([a, b, np.eye(depth+1)]))
    complement = complement[:, 2:]
    positions = np.sqrt(tt)[:, None]*(np.cos(theta)[:, None]*a+np.sin(theta)[:, None]*b)
    positions += np.sqrt(1-tt)[:, None]*(normals@complement.T)
    return positions


def density(positions, depth, level, plane):
    a, b = plane
    return float(norm_mp(depth, level))*((positions@a)**2+(positions@b)**2)**level


def plane_at(depth, angle):
    eye = np.eye(depth+1)
    return eye[0], np.cos(angle)*eye[1]+np.sin(angle)*eye[2]


def contact_values(rho1, rho2, a, b):
    return np.min(np.array([float(aa)*rho1+float(bb)*rho2 for aa, bb in zip(a, b)]), axis=0)


def quadrature(depth, level1, level2, a, b, angle, power, seed, orthogonal=False):
    eye = np.eye(depth+1)
    plane1 = (eye[0], eye[1])
    plane2 = (eye[2], eye[3]) if orthogonal else plane_at(depth, angle)
    first = sample_beam(depth, level1, plane1, power, seed)
    second = sample_beam(depth, level2, plane2, power, seed+1)
    positions = np.concatenate([first, second])
    rho1, rho2 = density(positions, depth, level1, plane1), density(positions, depth, level2, plane2)
    proposal = (rho1+rho2)/2
    values = contact_values(rho1, rho2, a, b)
    answer = float(np.mean(values/proposal))
    check(answer>=0, 'the sampled complete contact energy has the analytically required attractive sign')
    if not orthogonal and angle>0:
        step = 1e-6
        plus = density(positions, depth, level2, plane_at(depth, angle+step))
        minus = density(positions, depth, level2, plane_at(depth, angle-step))
        fd = (contact_values(rho1, plus, a, b)-contact_values(rho1, minus, a, b))/(2*step)
        a2, b2 = plane2
        derivative_direction = -np.sin(angle)*eye[1]+np.cos(angle)*eye[2]
        tt = (positions@a2)**2+(positions@b2)**2
        drho = 2*float(norm_mp(depth, level2))*level2*tt**(level2-1)*(positions@b2)*(positions@derivative_direction)
        winners = np.argmin(np.array([float(aa)*rho1+float(bb)*rho2 for aa, bb in zip(a, b)]), axis=0)
        analytic = np.array([float(bb) for bb in b])[winners]*drho
        # Integrate both variations against the SAME fixed importance proposal.
        close(np.mean(fd/proposal), np.mean(analytic/proposal),
              'independent fixed-proposal field rotation variation verifies the contact torque', tolerance=3e-5)
    # No orientation samples are used to prove the global alignment theorem.
    return answer


def main():
    mp.mp.dps = 100
    pair_levels = {2:(75,80), 3:(0,3), 4:(2,19), 5:(9,14), 6:(12,34), 10:(22,23)}
    cases = []
    for depth in beams.forward.radial.DEPTHS:
        epsilon, candidates, _ = envelope.depth_carrier(depth)
        level1, level2 = pair_levels[depth]
        one, two, a, b, weights, radii, aligned, overlap, bound, total_energy = exact_contact(depth, level1, level2, candidates)
        estimates = []
        for angle in [pi/6, pi/3, pi/2]:
            values = []
            for power in [10, 12]:
                for replicate in range(2):
                    result = quadrature(depth, level1, level2, a, b, angle, power,
                                        202610120000+depth*100+replicate*10)
                    values.append(dict(samples=2*(2**power), replicate=replicate, contact_magnitude=result))
            estimates.append(dict(relative_plane_angle_radians=angle, quadrature_estimates=values))
        if depth>=3:
            values = [quadrature(depth, level1, level2, a, b, 0, power,
                                 202610125000+depth*100, True) for power in [10,12]]
        else:
            values = None
        # Same-branch isolated columns interact neither in energy nor source, for every orientation.
        for candidate in candidates:
            if candidate['level'] in [n-1 for _, dd, n in beams.forward.radial.ASSIGNMENTS if dd==depth]:
                xx = candidate['x']; isolated = candidate['m']*xx+candidate['eta']
                for ratio in [mp.mpf('.1'), mp.mpf('1'), mp.mpf('7')]:
                    potential = max(c['m']*xx*(1+ratio)+c['eta']*(1+ratio) for c in candidates)
                    check(abs(potential-isolated*(1+ratio))<mp.mpf('1e-74'),
                          'all maintained same-branch orthogonal-column controls have exactly zero contact energy')
        cases.append(dict(depth=depth, first_shell_level=level1, second_shell_level=level2,
            second_is_unassigned_mathematical_control=depth==10,
            aligned_contact_magnitude=str(aligned), aligned_contact_over_sum_isolated_energy=str(aligned/(one['omega']+two['omega'])),
            aligned_complete_pair_energy=str(total_energy), aligned_complete_energy_is_negative=bool(total_energy<0),
            layer_count=len(weights), layer_weights=[str(w) for w in weights],
            layer_density_ratio_thresholds=[str(r) for r in radii],
            orthogonal_circle_classical_overlap=str(overlap) if overlap is not None else None,
            orthogonal_contact_upper_bound=str(bound) if bound is not None else None,
            rigorous_aligned_exceeds_orthogonal_bound=bool(aligned>bound) if bound is not None else None,
            finite_orientation_estimates=estimates, orthogonal_circle_quadrature_estimates=values,
            classification='exact attractive contact and global alignment bound; numerical angular illustrations'))
        print(json.dumps(dict(depth=depth, aligned_contact_magnitude=str(aligned),
                              orthogonal_upper_bound=str(bound), nonconstant_pair=level1>0)), flush=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    source_names = ['count-envelope-beam-contact.py', 'convex-count-envelope.py', 'balanced-count-beams.py',
        'minimal-count-carrier.py', 'forward-count-action.py', 'inverse-current-radial.py', 'excited-shell-orientation-current.py']
    report = dict(experiment_key='count-envelope-balanced-beam-contact',
        created_utc=datetime.now(timezone.utc).isoformat(),
        sources={'experiments/'+name: sha256((ROOT/name).read_bytes()).hexdigest() for name in source_names},
        evaluations=sum(COUNTS.values()), conditions=COUNTS, maximum_errors=ERRORS, cases=cases,
        limits=['orthogonal column blocks are a stated two-pattern preparation of one field',
                'this preparation has no relative-phase dependence in the count envelope',
                'instantaneous energy and torque do not constitute encounter dynamics or a measured force',
                'cross-depth field extension and physical interaction rates remain unspecified',
                'the depth-three ground density makes its displayed pair orientation independent',
                'depth ten uses an adjacent unassigned-shell comparison as well as the maintained same-branch zero',
                'the aligned electron-muon preparation has negative complete energy in this diagnostic action',
                'rare orthogonal-circle quadrature is unresolved at the displayed sizes; use the exact upper bound',
                'global alignment follows from the positive layer-overlap proof, not sampled angle ordering',
                'all coefficients are from the same existing count action; no independent pair coefficient is fitted'])
    path = ROOT/('count-envelope-beam-contact-results-'+stamp+'.json')
    path.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(report=str(path), evaluations=report['evaluations'], maximum_errors=ERRORS)), flush=True)


if __name__=='__main__':
    main()
