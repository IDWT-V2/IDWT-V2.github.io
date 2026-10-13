"""Check breathing projections of the existing autonomous Coulomb petals."""
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import json
import sys

import numpy as np
from scipy.linalg import expm
import sympy as sp

import completed_coulomb_petals_checks as completed

ROOT = Path(__file__).resolve().parent
DEPTHS = (2, 3, 4, 5, 6, 10)
ACTIVE = (0, 1, 3, 4)
conditions = {}
errors = {}


def check(value, name):
    if not bool(value):
        raise AssertionError(name)
    conditions[name] = conditions.get(name, 0) + 1


def close(actual, expected, name, tolerance=2e-9):
    error = float(np.max(np.abs(np.asarray(actual) - np.asarray(expected))))
    errors[name] = max(errors.get(name, 0.0), error)
    check(error < tolerance, name)


def symbolic_checks():
    theta, level, chi = sp.symbols('theta level chi', real=True)
    q = sp.Matrix([sp.cos(level*theta)*sp.cos(theta),
                   sp.cos(level*theta)*sp.sin(theta),
                   sp.sin(level*theta)*sp.cos(theta),
                   sp.sin(level*theta)*sp.sin(theta)])
    transform = sp.Matrix([[1, 0, 0, 1], [0, 1, -1, 0],
                           [1, 0, 0, -1], [0, 1, 1, 0]]) / sp.sqrt(2)
    close(np.array(transform.T*transform, float), np.eye(4),
          'eigenplane change is orthogonal')
    expected = sp.Matrix([sp.cos((1-level)*theta), sp.sin((1-level)*theta),
                          sp.cos((1+level)*theta), sp.sin((1+level)*theta)]) / sp.sqrt(2)
    for entry in transform*q - expected:
        check(sp.trigsimp(sp.expand_trig(entry)) == 0,
              'two physical circular eigenplanes')
    projection = sp.Matrix([[sp.cos(chi), 0, sp.sin(chi), 0],
                            [0, sp.cos(chi), 0, sp.sin(chi)],
                            [-sp.sin(chi), 0, sp.cos(chi), 0]])
    for entry in projection*projection.T-sp.eye(3):
        check(sp.trigsimp(entry) == 0, 'three orthonormal observed axes')
    observed = projection*expected
    radius_squared = (1 + sp.sin(2*chi)*sp.cos(2*level*theta))/2
    check(sp.trigsimp(sp.expand_trig(observed[0]**2+observed[1]**2-radius_squared)) == 0,
          'exact projected breathing law')
    for i in (0, 1):
        check(sp.trigsimp(sp.expand_trig(observed[i].subs(chi, sp.pi/4)-q[i])) == 0,
              'aligned observer recovers signed petals')
    return transform


def wedge(q, p):
    return np.outer(q, p)-np.outer(p, q)


def field(q, p, source):
    coupling = completed.LAM
    velocity = p-coupling*source@q
    force = -q/(q@q)**1.5-coupling*source@p+coupling**2*source@source@q
    gradient = coupling*wedge(q, p)-coupling**2*wedge(q, source@q)
    return velocity, force, source@gradient-gradient@source


def energy(q, p, source):
    v = p-completed.LAM*source@q
    return float(v@v/2-1/np.linalg.norm(q)-np.trace(source@source)/4)


def projection_matrix(depth, chi, transform):
    p = np.zeros((3, depth))
    p[:, :4] = np.array([[np.cos(chi), 0, np.sin(chi), 0],
                         [0, np.cos(chi), 0, np.sin(chi)],
                         [-np.sin(chi), 0, np.cos(chi), 0]]) @ transform
    return p


def main():
    transform = np.array(symbolic_checks(), float)
    rng = np.random.default_rng(610111539)
    cases = []
    for level in (np.sqrt(2), 1.5, 2.0, 4.0):
        state, generator6, rate, solved = completed.solve_equilibrium(1, level)
        q6, p6, source6 = completed.base.unpack(state, 1)
        for depth in DEPTHS:
            if depth < 4:
                cases.append(dict(depth=depth, level=float(level),
                                  branch_available=False,
                                  reason='Two independent rotation planes need four spatial dimensions.'))
                continue
            q0, p0 = np.zeros(depth), np.zeros(depth)
            q0[:4], p0[:4] = q6[0, ACTIVE], p6[0, ACTIVE]
            source0, generator = np.zeros((depth, depth)), np.zeros((depth, depth))
            source0[:4, :4] = source6[np.ix_(ACTIVE, ACTIVE)]
            generator[:4, :4] = generator6[np.ix_(ACTIVE, ACTIVE)]
            v, force, source_dot = field(q0, p0, source0)
            residual = max(np.linalg.norm(v-generator@q0),
                           np.linalg.norm(force-generator@p0),
                           np.linalg.norm(source_dot-(generator@source0-source0@generator)))
            close(residual, 0, 'autonomous relative equilibrium after cross-depth embedding', 2e-9)
            initial_energy = energy(q0, p0, source0)
            total_angular = wedge(q0, p0)+source0
            for theta in (0.0, .137, .428, .991, 2.771, 5.103):
                rotation = expm(generator*theta/rate)
                q, p = rotation@q0, rotation@p0
                source = rotation@source0@rotation.T
                q_formula = np.array([np.cos(level*theta)*np.cos(theta),
                                      np.cos(level*theta)*np.sin(theta),
                                      np.sin(level*theta)*np.cos(theta),
                                      np.sin(level*theta)*np.sin(theta)])
                close(q[:4], q_formula, 'exact existing autonomous orbit', 2e-9)
                close(q@q, 1, 'full physical radius stays fixed')
                close(energy(q, p, source), initial_energy, 'full energy conserved')
                close(wedge(q, p)+source, total_angular, 'complete angular charge conserved')
                for chi in (0.0, np.pi/12, np.pi/8, np.pi/4):
                    observer = projection_matrix(depth, chi, transform)
                    displayed = observer@q
                    predicted = (1+np.sin(2*chi)*np.cos(2*level*theta))/2
                    close(displayed[:2]@displayed[:2], predicted,
                          'projected breathing at every supported depth')
                    close(observer@observer.T, np.eye(3), 'observer is an ordinary orthogonal projection')
                    matrix, _ = np.linalg.qr(rng.normal(size=(depth, depth)))
                    close((observer@matrix.T)@(matrix@q), displayed,
                          'common physical rotation leaves projection unchanged')
            period = 4*np.pi if level == 1.5 else 2*np.pi
            rotation = expm(generator*period/rate)
            recurrence = float(np.linalg.norm(rotation@q0-q0))
            if level != np.sqrt(2):
                close(recurrence, 0, 'commensurate full trajectory recurrence', 3e-9)
            else:
                check(recurrence > .1, 'irrational ratio fails a single clock-turn recurrence')
            cases.append(dict(depth=depth, level=float(level), branch_available=True,
                              clock_rate=float(rate), equilibrium_residual=float(residual),
                              linear_trajectory_span=4, recurrence_error_at_tested_turn=recurrence,
                              stability='Not newly tested; observer changes cannot alter an unchanged full-state flow.'))

    # A tilted ordinary three-dimensional circle is a lower-depth breathing control.
    control = []
    theta = np.linspace(0, 2*np.pi, 1001)
    for depth in DEPTHS:
        if depth == 2:
            radius_squared = np.cos(theta)**2+np.sin(theta)**2
            close(radius_squared, np.ones_like(theta), 'depth-two full plane circle has no projected breathing')
            control.append(dict(depth=depth, tilted_circle_available=False))
            continue
        inclination = np.pi/3
        q = np.zeros((depth, len(theta)))
        q[0], q[1], q[2] = np.cos(theta), np.cos(inclination)*np.sin(theta), np.sin(inclination)*np.sin(theta)
        close(np.sum(q*q, axis=0), 1, 'tilted circle full radius fixed')
        radius_squared = q[0]**2+q[1]**2
        close(radius_squared, np.cos(theta)**2+np.cos(inclination)**2*np.sin(theta)**2,
              'lower-depth tilted circle breathes in projection')
        control.append(dict(depth=depth, tilted_circle_available=True,
                            projected_radius_min=float(np.min(np.sqrt(radius_squared))),
                            projected_radius_max=float(np.max(np.sqrt(radius_squared)))))

    # A radial dwell-time histogram does not establish azimuthally complete rings.
    chi = np.pi/12
    rmin2, rmax2 = (1-np.sin(2*chi))/2, (1+np.sin(2*chi))/2
    sample_theta = (np.arange(20000)+.5)*2*np.pi/20000
    samples = (1+np.sin(2*chi)*np.cos(4*sample_theta))/2
    for fraction in (.1, .3, .5, .7, .9):
        boundary = rmin2+fraction*(rmax2-rmin2)
        predicted = 1-np.arccos(2*fraction-1)/np.pi
        close(np.mean(samples <= boundary), predicted, 'deterministic radial dwell-time CDF', 3e-4)
    report = dict(created_at=datetime.now(timezone.utc).isoformat(),
                  sources={f'experiments/{name}': sha256((ROOT/name).read_bytes()).hexdigest()
                           for name in ('breathing-orbit-checks.py',
                                        'completed_coulomb_petals_checks.py',
                                        'coulomb_angular_response_checks.py')},
                  conditions=conditions, max_errors=errors, cases=cases, lower_depth_controls=control,
                  scope='Existing conditional classical central-source model and ordinary spatial projections.',
                  conclusions=['Annular breathing and petal projections can share one unchanged full trajectory.',
                               'This branch occupies a four-dimensional linear span even when embedded in six.',
                               'Radial breathing alone is not specific to depth six.',
                               'Noninteger frequency-ratio relative equilibria also exist; no integer selection follows.',
                               'Finite commensurate traces are not azimuthally complete concentric rings.',
                               'No new source induction, physical detector law, or particle mass derivation.'])
    destination = ROOT/('breathing-orbit-results-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')+'.json')
    destination.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(report=str(destination), checks=sum(conditions.values()),
                          conditions=len(conditions), available_cases=sum(c['branch_available'] for c in cases),
                          max_equilibrium_residual=max(c.get('equilibrium_residual', 0) for c in cases)), indent=2))


if __name__ == '__main__':
    main()
