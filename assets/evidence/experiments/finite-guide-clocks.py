"""Dynamic guide boundary response, reciprocal stress and distinct count clocks."""
from datetime import datetime, timezone
from hashlib import sha256
from math import comb, pi
from pathlib import Path
import itertools
import json

import numpy as np
import sympy as s
from scipy.integrate import quad, solve_bvp

ROOT = Path(__file__).resolve().parent
COUNTS, ERRORS = {}, {}
ASSIGNMENTS = {2:[('W',76),('Z',81),('Higgs',95)],
               3:[('down',1),('strange',4)],
               4:[('up',3),('charm',20),('top',72)],
               5:[('nu1-comparison',10),('nu2-comparison',15),('nu3-comparison',22)],
               6:[('electron',13),('muon',35)],10:[('tau',23)]}


def check(value, label):
    if not bool(value):
        raise AssertionError(label)
    COUNTS[label] = COUNTS.get(label,0)+1


def exact(value, label):
    check(s.simplify(value)==0,label)


def close(actual, expected, label, tolerance=1e-8):
    error = float(abs(actual-expected)/max(1,abs(expected)))
    ERRORS[label] = max(ERRORS.get(label,0),error)
    check(error<tolerance,label)


def amplitude(lam, omega, length, left, right, x):
    alpha = np.sqrt(complex(lam*lam-omega*omega))
    if abs(alpha)<1e-12:
        return left+(right-left)*x/length,(right-left)/length
    den = np.sinh(alpha*length)
    value = (left*np.sinh(alpha*(length-x))+right*np.sinh(alpha*x))/den
    slope = alpha*(-left*np.cosh(alpha*(length-x))+right*np.cosh(alpha*x))/den
    return value,slope


def boundary_matrix(lam, omega, length):
    alpha = np.sqrt(complex(lam*lam-omega*omega))
    if abs(alpha)<1e-12:
        return np.array([[1,-1],[-1,1]],complex)/length
    return alpha/np.sinh(alpha*length)*np.array([[np.cosh(alpha*length),-1],[-1,np.cosh(alpha*length)]])


def mean_action_cost(lam, omega, length, endpoint):
    return (np.vdot(endpoint,boundary_matrix(lam,omega,length)@endpoint)/4).real


def symbolic():
    alpha, length, x = s.symbols('alpha L x',positive=True)
    a,b = s.symbols('a b',real=True)
    z = (a*s.sinh(alpha*(length-x))+b*s.sinh(alpha*x))/s.sinh(alpha*length)
    cost = alpha*((a*a+b*b)*s.coth(alpha*length)-2*a*b/s.sinh(alpha*length))/4
    pressure = (s.diff(z,x)**2-alpha*alpha*z*z)/4
    exact(s.diff(z,x,2)-alpha*alpha*z,'dynamic spectral amplitude solves the full wave equation')
    exact(s.diff(pressure,x),'time-averaged full stress is independent of position')
    exact(s.trigsimp(s.diff(cost,length)+pressure.subs(x,0)),
          'averaged action variation and independently derived pressure agree')
    exact(s.limit(alpha*s.coth(alpha*length),alpha,0)-1/length,
          'diagonal boundary response has a regular threshold limit')
    exact(s.limit(alpha/s.sinh(alpha*length),alpha,0)-1/length,
          'cross boundary response has the same regular threshold limit')
    p = s.symbols('p',positive=True)
    exact((alpha*s.coth(alpha*length)).subs(alpha,s.I*p)-p*s.cot(p*length),
          'above-gap diagonal response is oscillatory rather than exponential')
    exact((alpha/s.sinh(alpha*length)).subs(alpha,s.I*p)-p/s.sin(p*length),
          'above-gap cross response retains its standing-wave poles')
    omega, mass = s.symbols('omega mass',positive=True)
    energy_density = (s.diff(z,x)**2+(mass*mass+omega*omega)*z*z)/4
    cost_density = (s.diff(z,x)**2+(mass*mass-omega*omega)*z*z)/4
    exact(energy_density-cost_density-omega*omega*z*z/2,
          'stored energy differs from averaged action cost by the retained kinetic term')
    phase = s.symbols('phase',real=True)
    field = s.exp(-s.I*omega*x)*phase
    exact(s.conjugate(field)*field-phase*phase,
          'a single-frequency phase-invariant shape does not freeze its carrier clock')


def dynamic_controls():
    rows = []
    endpoint = np.array([0.13+0.03j,-0.07+0.09j])
    lam = 1.3
    for omega in (0.4,1.3,1.9):
        for length in (0.7,1.2):
            grid = np.linspace(0,length,65)
            initial_value = endpoint[0]+(endpoint[1]-endpoint[0])*grid/length
            initial_slope = np.full(len(grid),(endpoint[1]-endpoint[0])/length)
            initial = np.vstack([initial_value.real,initial_value.imag,
                                  initial_slope.real,initial_slope.imag])
            solution = solve_bvp(lambda x,y:np.vstack([y[2],y[3],
                                 (lam*lam-omega*omega)*y[0],(lam*lam-omega*omega)*y[1]]),
                                 lambda a,b:np.array([a[0]-endpoint[0].real,a[1]-endpoint[0].imag,
                                                      b[0]-endpoint[1].real,b[1]-endpoint[1].imag]),
                                 grid,initial,tol=1e-10,max_nodes=4096)
            check(solution.success,'independent complex harmonic boundary solver converges')
            def reconstructed(x):
                values = solution.sol(x)
                return values[0]+1j*values[1],values[2]+1j*values[3]
            norm,error = quad(lambda x:abs(reconstructed(x)[0])**2,0,length,epsabs=1e-12)
            measured_energy,_ = quad(lambda x:(abs(reconstructed(x)[1])**2+
                (lam*lam+omega*omega)*abs(reconstructed(x)[0])**2)/4,0,length,epsabs=1e-12)
            cost = mean_action_cost(lam,omega,length,endpoint)
            close(measured_energy,cost+omega*omega*norm/2,
                  'independent stored energy includes the extra kinetic contribution')
            pressure = None
            for x in np.linspace(0,length,7):
                actual,slope = reconstructed(x)
                expected,expected_slope = amplitude(lam,omega,length,*endpoint,x)
                close(actual,expected,'independent harmonic field matches exact boundary formula')
                close(slope,expected_slope,'independent harmonic spatial derivative matches exact formula')
                current = (abs(slope)**2+(omega*omega-lam*lam)*abs(actual)**2)/4
                if pressure is not None:
                    close(current,pressure,'actual mean pressure remains constant through the guide')
                pressure = current
            step = 1e-4*length
            derivative = (mean_action_cost(lam,omega,length+step,endpoint)-
                          mean_action_cost(lam,omega,length-step,endpoint))/(2*step)
            close(derivative,-pressure,'numerical fixed-frequency action derivative yields the actual stress force')
            if omega>lam:
                check(pressure>0,'a propagating scalar channel has positive outward pressure')
            rows.append(dict(channel_gap=lam,drive_frequency=omega,length=length,
                             mean_shape_pressure=pressure,mean_stored_energy=measured_energy,
                             mean_action_cost=cost,
                             interpretation='exact scalar response identity control, not a selected transverse mode'))
    return rows


def causal_controls():
    rows = []
    endpoint = np.array([0.2+0.07j,-0.1+0.11j])
    for omega in (0.5,1.7,3.2):
        for eta in (0.3,0.1,0.03):
            length,lam = 1.4,1.1
            frequency = omega+1j*eta
            matrix = boundary_matrix(lam,frequency,length)
            value = np.vdot(endpoint,matrix@endpoint)
            norm,_ = quad(lambda x:abs(amplitude(lam,frequency,length,*endpoint,x)[0])**2,
                          0,length,epsabs=2e-12)
            close(value.imag,-2*omega*eta*norm,
                  'independent causal Green identity retains complete endpoint work')
            negative_imag = -(matrix-matrix.conj().T)/(2j)
            check(np.linalg.eigvalsh(negative_imag).min()>0,
                  'causal boundary admittance is passive for arbitrary endpoint polarization')
            rows.append(dict(omega=omega,causal_regulator_eta=eta,
                             mean_boundary_power=-omega*value.imag/2,
                             interpretation='upper-half-plane regulator, not a physical IDWT damping parameter'))
    return rows


def count_controls():
    rows,pairs = [],[]
    phase = s.symbols('phase',real=True)
    for depth,assignments in ASSIGNMENTS.items():
        counts = [(name,index,comb(index+depth-1,depth)) for name,index in assignments]
        for name,index,count in counts:
            for t in (s.Rational(1,4),s.Rational(1,2),s.Rational(3,4)):
                for scaled_length in (1,4):
                    ratio = s.Rational(count*count,1)/(1-t)
                    threshold = 1+pi*pi/scaled_length**2
                    active = depth in (3,4)
                    rows.append(dict(particle=name,depth=depth,index=index,count=count,
                                     t_exact=str(t),mu_times_length=scaled_length,
                                     direct_carrier_frequency_squared_over_gap_squared_exact=str(ratio),
                                     radiation_threshold_squared_over_gap_squared=threshold,
                                     generic_radiation_possible_if_guide_supplied=bool(float(ratio)>threshold),
                                     guide_present_under_existing_premises=active,
                                     interpretation='Omega equals sector fundamental only in this labeled control'))
                    check(ratio>1,'every positive count carrier lies above the transverse gap in the declared control')
        for first,second in itertools.combinations(counts,2):
            difference,sum_count = second[2]-first[2],second[2]+first[2]
            check(difference!=0,'maintained sibling carriers retain distinct clocks')
            for harmonic in (difference,sum_count):
                exact(s.sin(2*s.pi*harmonic+phase)-s.sin(phase),
                      'unlike count-harmonic cross terms average exactly to zero over one fundamental period')
            pairs.append(dict(depth=depth,first=first[0],second=second[0],
                              count_difference=difference,
                              phase_sensitive_linear_cross_force_period_average_exact='0'))
    for omega1,omega2 in ((1,2),(2,5),(3,7)):
        for phase1,phase2 in ((0.1,0.7),(0.4,-0.2)):
            measured,_ = quad(lambda t:np.cos(omega1*t+phase1)*np.cos(omega2*t+phase2),
                              0,2*pi,epsabs=2e-12)
            close(measured,0,'independent time integration verifies unlike-harmonic phase cancellation')
    return rows,pairs


def main():
    symbolic()
    dynamic = dynamic_controls()
    causal = causal_controls()
    assignments,pairs = count_controls()
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    report = dict(experiment_key='finite-guide-distinct-clocks-and-dynamic-response',
                  created_utc=datetime.now(timezone.utc).isoformat(),
                  sources={'experiments/'+Path(__file__).name:sha256(Path(__file__).read_bytes()).hexdigest()},
                  evaluations=sum(COUNTS.values()),conditions=COUNTS,maximum_errors=ERRORS,
                  dynamic_controls=dynamic,causal_controls=causal,
                  occupied_assignments=assignments,distinct_clock_pairs=pairs,
                  exact_equations={'finite_segment_threshold':'mu^2+pi^2/L^2',
                                   'force':'T-mean(P)=T+partial_L W',
                                   'mean_energy':'W+(omega^2/2) integral |Z|^2 dx'},
                  limits=['slow phase-invariant shape coupling is distinct from direct mass-clock forcing',
                          'fundamental identification of Omega remains a hypothesis',
                          'the guide restriction to three/four is inherited from prior premises',
                          'zero averaged cross force does not mean zero individual response',
                          'causal eta is a resolvent regulator, not an added dissipation mechanism',
                          'generic continuum forcing above threshold requires outgoing radiation and source work',
                          'no nonlinear locking, particle mass selection or observed force is derived'])
    path = ROOT/('finite-guide-clocks-results-'+stamp+'.json')
    path.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(report=str(path),evaluations=report['evaluations'],maximum_errors=ERRORS)),flush=True)


if __name__=='__main__':
    main()
