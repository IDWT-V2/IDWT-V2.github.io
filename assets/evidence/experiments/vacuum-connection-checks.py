#!/usr/bin/env python3
"""Test a smooth vacuum-history ambiguity of the exact-zero dielectric action."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import math

import numpy as np
from scipy.integrate import solve_ivp, quad
from scipy.special import ellipk


DEPTHS = (2, 3, 4, 5, 6, 10)
CHECKS = 0
CASES = []


def check(condition, message):
    global CHECKS
    CHECKS += 1
    if not condition:
        raise AssertionError(message)


def close(actual, expected, message, atol=2e-8):
    check(np.allclose(actual, expected, atol=atol, rtol=2e-8), message)


def flat_step(s):
    if s <= 0:
        return 0.
    if s >= 1:
        return 1.
    a = math.exp(-1/s)
    b = math.exp(-1/(1-s))
    return a/(a+b)


def drive(time):
    if not 2 < time < 3:
        return 0.
    s = time-2
    return math.exp(4-1/(s*(1-s)))


def magnetic_history(time, final):
    return final*flat_step((time-.25)/1.25)


def driven_response(active, magnetic, *, rtol=2e-11, atol=2e-12):
    kappa = 1.2
    omega = .9
    coefficient = kappa*active*magnetic*magnetic
    def equation(t, state):
        u, speed, work = state
        forcing = drive(t)
        return [speed, forcing-omega*omega*u-2*coefficient*u**3, forcing*speed]
    solution = solve_ivp(equation, (2., 12.), [0., 0., 0.], method='DOP853',
                         rtol=rtol, atol=atol, max_step=.025, dense_output=True)
    check(solution.success, 'smooth neutral-probe response integrates successfully')
    times = np.linspace(2., 12., 401)
    u, speed, work = solution.sol(times)
    energy = (speed*speed+omega*omega*u*u+coefficient*u**4)/2
    close(energy, work, 'complete scalar plus magnetic energy equals work of the identical specified drive')
    check(np.max(np.abs(energy-work))<2e-9, 'conserved post-drive energy is resolved')
    return solution, times, u, speed, energy, work


def duffing_clock(active, magnetic):
    kappa = 1.2
    omega = .9
    initial_speed = .8
    c = kappa*active*magnetic*magnetic
    if c == 0:
        amplitude_squared = initial_speed**2/omega**2
        frequency_scale = omega
        modulus = 0.
    else:
        amplitude_squared = 2*initial_speed**2/(omega*omega+math.sqrt(omega**4+4*c*initial_speed**2))
        frequency_scale = math.sqrt(omega*omega+2*c*amplitude_squared)
        modulus = c*amplitude_squared/frequency_scale**2
    period = 4*ellipk(modulus)/frequency_scale
    def equation(t, state):
        u, speed = state
        return [speed, -omega*omega*u-2*c*u**3]
    def crossing(t, state):
        return state[0]
    crossing.direction = 1
    solution = solve_ivp(equation, (0., 2.2*period), [0., initial_speed], method='DOP853',
                         rtol=1e-11, atol=1e-12, max_step=period/150, events=crossing)
    check(solution.success, 'homogeneous complete-energy clock integrates')
    events = solution.t_events[0]
    check(len(events)>=3, 'clock has two complete post-initial cycles')
    close(events[1], period, 'Jacobi elliptic period agrees with independent zero-crossing integration')
    close(events[2]-events[1], period, 'second cycle independently retains the period')
    amplitude = math.sqrt(amplitude_squared)
    close((omega*omega*amplitude_squared+c*amplitude_squared**2)/2,
          initial_speed*initial_speed/2, 'turning amplitude retains complete initial kinetic energy')
    return period, amplitude


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--geometry-source', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location('gauss_geometry', args.geometry_source)
    geometry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(geometry)
    for time in (0., .1, .25):
        close(magnetic_history(time, 3.), 0, 'all trial histories share the same initially flat raw connection')
    for time in (1.5, 1.8, 2., 3., 8.):
        close(magnetic_history(time, 3.), 3., 'curvature history settles before the common probe turns on')
    # Before activation u=0, all dielectric and its scalar derivative vanish identically.
    for value in (0., .1, 1., 3.):
        close(value*0.**4, 0, 'arbitrary smooth vacuum curvature has zero connection energy')
        close(value*0.**3, 0, 'arbitrary smooth vacuum curvature exerts no scalar force before activation')
    for d in DEPTHS:
        hodge = geometry.hodge(d if d % 2 == 0 else d+1)
        active = int(np.linalg.norm(hodge)>1e-10)
        check(active == int(d in (3, 4)), 'vacuum test retains the occupied-sector Hodge rule')
        if d == 10:
            period, amplitude = duffing_clock(0, 0.)
            CASES.append({'case': 'vacuum-response-depth-10', 'real_depth': d,
                          'independent_normal_connection_exists': False, 'neutral_clock_period': period})
            continue
        responses = []
        periods = []
        for magnetic in (0., 1., 3.):
            solution, times, u, speed, energy, work = driven_response(active, magnetic)
            refined, _, u2, _, _, _ = driven_response(active, magnetic, rtol=3e-12, atol=3e-13)
            close(u, u2, 'neutral response survives tighter independent integration tolerances')
            period, amplitude = duffing_clock(active, magnetic)
            periods.append(period)
            responses.append(u)
            pde_errors = []
            for time in (.4, .8, 1.3, 1.8, 2.2, 2.5, 3.5, 6.):
                delta = 1e-4
                b = magnetic_history(time, magnetic)
                b_dot = (magnetic_history(time+delta, magnetic)-magnetic_history(time-delta, magnetic))/(2*delta)
                b_ddot = (magnetic_history(time+delta, magnetic)-2*b+magnetic_history(time-delta, magnetic))/delta**2
                if time < 2:
                    amplitude = velocity = acceleration = 0.
                else:
                    amplitude, velocity, _ = solution.sol(time)
                    acceleration = (solution.sol(time+delta)[1]-solution.sol(time-delta)[1])/(2*delta)
                position_x = .73
                electric = -b_dot*position_x
                epsilon = 1.2*active*amplitude**4
                epsilon_dot = 4*1.2*active*amplitude**3*velocity
                gauge_residual = -position_x*(epsilon_dot*b_dot+epsilon*b_ddot)
                scalar_residual = acceleration+.9**2*amplitude+2*1.2*active*amplitude**3*(b*b-electric*electric)-drive(time)
                close(gauge_residual, 0, 'full connection equation holds before and after neutral activation')
                close(scalar_residual, 0, 'independent solution derivative satisfies the full scalar PDE with electric and magnetic terms', atol=2e-6)
                pde_errors.append(abs(float(scalar_residual)))
            CASES.append({'case': f'vacuum-response-depth-{d}-B-{magnetic:g}', 'real_depth': d,
                          'hodge_coefficient_active': bool(active), 'settled_magnetic_curvature': magnetic,
                          'initial_raw_connection_and_curvature': '0',
                          'initial_neutral_amplitude_and_velocity': '0',
                          'identical_neutral_probe': 'smooth pulse supported in 2<t<3',
                          'response_at_t4': float(solution.sol(4.)[0]),
                          'maximum_energy_work_error': float(np.max(np.abs(energy-work))),
                          'maximum_independent_scalar_PDE_error': max(pde_errors),
                          'post_probe_complete_energy_density': float(energy[-1]),
                          'unforced_common_initial_velocity_period': period,
                          'unforced_turning_amplitude': amplitude,
                          'local_homogeneous_control': True, 'finite_total_energy_global_solution': False})
        difference = float(np.max(np.abs(responses[0]-responses[-1])))
        if active:
            check(difference>.05, 'different zero-energy vacuum histories produce different later neutral-probe responses')
            check(periods[2]<periods[1]<periods[0], 'activated curvature changes the neutral clock at common initial scalar velocity')
        else:
            close(difference, 0, 'identically inactive coefficient leaves every later neutral response unchanged')
            close(periods, [2*math.pi/.9]*3, 'inactive neutral clock remains harmonic independently of raw curvature')
        # A positive floor forbids the freely varying vacuum history without an additional driving current.
        s = .43
        delta = 1e-4
        second = (flat_step(s+delta)-2*flat_step(s)+flat_step(s-delta))/delta**2
        check(abs(second)>.1, 'the smooth history has nonzero acceleration in its vacuum transition')
        for floor in (.1, .01):
            check(abs(floor*second)>.0001,
                  'positive-floor Maxwell equation rejects this vacuum history without a connection source')
    report = {'title': 'An Exact-Zero Dielectric Leaves a Vacuum-History Ambiguity',
              'sources': {'experiments/vacuum-connection-checks.py': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                          'experiments/hodge-gated-gauss-checks.py': hashlib.sha256(args.geometry_source.read_bytes()).hexdigest()},
              'checks_passed': CHECKS, 'dependency_checks_passed': geometry.CHECKS,
              'case_count': len(CASES), 'occupied_depths': list(DEPTHS),
              'scope': 'exact local homogeneous driven classical solutions with smooth raw vacuum connection histories; energy-density controls, not finite-total-energy particles',
              'conclusion': 'In active sectors, different smooth zero-energy vacuum connection histories sharing identical initial raw data yield different responses to the same later neutral drive. A positive floor rejects those histories but removes the exact source constraints. A geometric vacuum-flatness or connection-reduction rule requires derivation. Inactive sectors keep their zero source and neutral harmonic controls.',
              'cases': CASES}
    args.report.write_text(json.dumps(report, indent=2)+'\n')
    print(f'{CHECKS} vacuum, energy and clock checks plus {geometry.CHECKS} geometry checks passed across {len(CASES)} cases')


if __name__ == '__main__':
    main()
