"""Derive neutral-frame shape response and independently solve finite bridges."""
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import json

import numpy as np
import sympy as s
from scipy.integrate import quad, solve_bvp
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parent
COUNTS, ERRORS = {}, {}


def check(value, label):
    if not bool(value):
        raise AssertionError(label)
    COUNTS[label] = COUNTS.get(label, 0) + 1


def exact(value, label):
    check(s.simplify(value) == 0, label)


def close(actual, expected, label, tolerance=1e-8):
    error = float(np.linalg.norm(np.asarray(actual) - np.asarray(expected)) /
                  max(1, np.linalg.norm(expected)))
    ERRORS[label] = max(ERRORS.get(label, 0), error)
    check(error < tolerance, label)


def symbolic():
    u, v, w = s.symbols('u v w', real=True)
    frame = s.Matrix([[u+v, w], [w, u-v]]) / s.sqrt(2)
    norm = s.trace(frame.T*frame)
    area = (frame.T*frame).det()
    exact(norm-u*u-v*v-w*w, 'actual frame norm retains both shape channels')
    exact(area-(u*u-v*v-w*w)**2/4, 'actual frame Gram area has the exact shape deficit')
    derivatives = [frame.diff(a) for a in (u, v, w)]
    for i in range(3):
        for j in range(3):
            exact(s.trace(derivatives[i].T*derivatives[j])-int(i == j),
                  'frame kinetic metric is the common canonical identity')
    angle = s.symbols('angle', real=True)
    rot = s.Matrix([[s.cos(angle), -s.sin(angle)], [s.sin(angle), s.cos(angle)]])
    shape = s.Matrix([[v,w], [w,-v]])
    transformed = rot*shape*rot.T
    exact(s.trigsimp(transformed[0,0]-v*s.cos(2*angle)+w*s.sin(2*angle)),
          'shape polarization rotates twice the ordinary spatial angle')
    exact(s.trigsimp(transformed[0,1]-v*s.sin(2*angle)-w*s.cos(2*angle)),
          'both physical shape directions transform together')

    h, k2, k4, om, e0, i0, j2, j4 = s.symbols(
        'h k2 k4 om e0 i0 j2 j4', positive=True)
    dielectric = k2*norm+k4*area
    first = s.diff(dielectric.subs({v:h*v,w:h*w}),h).subs(h,0)
    second = s.diff(dielectric.subs({v:h*v,w:h*w}),h,2).subs(h,0)/2
    exact(first, 'shape has zero first dielectric source variation')
    exact(second-(k2-k4*u*u/2)*(v*v+w*w),
          'complete dielectric quadratic coefficient is derived')
    energy = (e0*i0)**2/(2*(i0+h*h*j2+h**4*j4))
    exact(s.diff(energy,h,2).subs(h,0)/2+e0*e0*j2/2,
          'fixed-flux variation retains the complete electric quadratic cost')
    potential = om*om*(v*v+w*w)/2-e0*e0*second/2
    exact(potential-(om*om-k2*e0*e0+k4*e0*e0*u*u/2)*(v*v+w*w)/2,
          'both shape operators have the derived positive-gap potential')

    x, length, k = s.symbols('x L k', positive=True)
    left, right = s.symbols('left right', real=True)
    z = (left*s.sinh(k*(length-x))+right*s.sinh(k*x))/s.sinh(k*length)
    exact(s.diff(z,x,2)-k*k*z, 'exact finite bridge solves its Euler equation')
    exact(z.subs(x,0)-left, 'exact bridge retains the first endpoint')
    exact(z.subs(x,length)-right, 'exact bridge retains the second endpoint')
    energy = k*((left*left+right*right)*s.coth(k*length)-
                2*left*right/s.sinh(k*length))/2
    boundary = (right*s.diff(z,x).subs(x,length)-left*s.diff(z,x).subs(x,0))/2
    exact(s.trigsimp(boundary-energy), 'integration by parts gives the complete bridge energy')
    plus, minus = (left+right)/2, (left-right)/2
    diagonal = k*(plus*plus*s.tanh(k*length/2)+minus*minus/s.tanh(k*length/2))
    exact(s.trigsimp(s.expand_trig((energy-diagonal).rewrite(s.exp))),
          'sum and mismatch shapes diagonalize the boundary energy')
    pressure = (s.diff(z,x)**2-k*k*z*z)/2
    exact(s.diff(pressure,x), 'longitudinal shape stress is conserved')
    exact(s.trigsimp(s.diff(energy,length)+pressure.subs(x,0)),
          'moving-end energy and independent stress give identical force')

    theta, amp = s.symbols('theta amp', real=True)
    angular = k*amp*amp*(s.coth(k*length)-s.cos(theta)/s.sinh(k*length))
    exact(s.diff(angular,theta)-k*amp*amp*s.sin(theta)/s.sinh(k*length),
          'reciprocal shape torque follows from the same energy')
    exact(s.diff(angular,theta,2).subs(theta,s.pi)+k*amp*amp/s.sinh(k*length),
          'opposite endpoint shapes have negative angular curvature')
    t, mass, omega = s.symbols('t mass omega', positive=True)
    for n in (2,3):
        grad = n*omega**2*(1-t)*mass/(4-n)
        electric = (omega**2*t*mass+(grad+omega**2*(1-t)*mass)/2)/2
        tension = electric+(grad+omega**2*mass)/2
        exact(tension/mass-omega**2*(3+(1-n)*t)/(4-n),
              'actual mixed-frame guide moments fix tension per deformation norm')
        width_bound = 2*(1-t)*(4-n)/(3+(1-n)*t)
        exact(s.diff(width_bound,t)+2*(4-n)**2/(3+(1-n)*t)**2,
              'exact derivative is negative throughout the stated physical interval zero to one')
        exact(width_bound.subs(t,0)-s.Rational(2*(4-n),3),
              'one common upper bound controls the formal equilibrium in guide-width units')
    return {'frame_norm':'u^2+v^2+w^2', 'frame_area':'(u^2-v^2-w^2)^2/4',
            'shape_operator':'-Delta_y+mu^2+(kappa4 E0^2/2)u0^2',
            'on_shell_energy':'T L+<g+,K tanh(K L/2)g+>+<g-,K coth(K L/2)g->'}


def bridge_spectral(matrix, left, right, length):
    values, vectors = np.linalg.eigh(matrix)
    rates = np.sqrt(values)
    a, b = vectors.T@left, vectors.T@right
    plus, minus = (a+b)/2, (a-b)/2
    z = rates*length/2
    energy = np.sum(rates*(plus*plus*np.tanh(z)+minus*minus/np.tanh(z)))
    inward = np.sum(values*(plus*plus/np.cosh(z)**2-minus*minus/np.sinh(z)**2))/2
    return energy, inward


def numerical():
    rng = np.random.default_rng(202610121215)
    cases = []
    for dim in (1,2,4):
        for length in (0.4,1.7,4.0):
            raw = rng.normal(size=(dim,dim))/2
            matrix = raw.T@raw+0.7**2*np.eye(dim)
            left, right = rng.normal(size=dim)/10, rng.normal(size=dim)/10
            grid = np.linspace(0,length,65)
            initial = np.vstack([left[:,None]+(right-left)[:,None]*grid/length,
                                  np.broadcast_to(((right-left)/length)[:,None],(dim,len(grid)))])
            solution = solve_bvp(lambda x,y:np.vstack([y[dim:],matrix@y[:dim]]),
                                 lambda a,b:np.r_[a[:dim]-left,b[:dim]-right],
                                 grid, initial, tol=2e-10, max_nodes=4096)
            check(solution.success, 'independent vector boundary solver converges')
            def density(x):
                data = solution.sol(x)
                return (data[dim:]@data[dim:]+data[:dim]@matrix@data[:dim])/2
            measured, error = quad(density,0,length,epsabs=2e-12,epsrel=2e-12)
            energy, derivative = bridge_spectral(matrix,left,right,length)
            close(measured,energy,'independent integrated field energy matches the operator formula')
            for x in np.linspace(0,length,7):
                data = solution.sol(x)
                pressure = (data[dim:]@data[dim:]-data[:dim]@matrix@data[:dim])/2
                close(pressure,-derivative,'independent field stress is constant and agrees with force')
            step = 1e-4*length
            ep = bridge_spectral(matrix,left,right,length+step)[0]
            em = bridge_spectral(matrix,left,right,length-step)[0]
            close((ep-em)/(2*step),derivative,'direct moving-end energy difference agrees with force')
            rotation = np.linalg.qr(rng.normal(size=(dim,dim)))[0]
            transformed = bridge_spectral(rotation@matrix@rotation.T,rotation@left,rotation@right,length)
            close(transformed,(energy,derivative),'common rotation preserves the entire pair energy and force')
            cases.append(dict(matrix_dimension=dim,length=length,shape_energy=energy,
                              shape_inward_force=derivative,quadrature_error=error,
                              interpretation='positive-operator identity control, not a physical transverse spectrum'))
    return cases


def force_controls():
    rows = []
    for mu in (0.3,1.0,2.0):
        rates = mu*np.array([1.0,1.3,2.1,3.7])
        weights = np.array([0.11,0.17,0.29,0.43])
        for cosine in (-1.0,-0.5,0.0,0.5,1.0):
            plus2, minus2 = (1+cosine)/2,(1-cosine)/2
            previous = None
            for scaled_length in (0.1,0.2,0.5,1,2,4,8):
                length = scaled_length/mu
                z = rates*length/2
                inward = 1+np.sum(weights*rates*rates*(plus2/np.cosh(z)**2-minus2/np.sinh(z)**2))/2
                slope = np.sum(weights*rates**3*(minus2/np.sinh(z)**2/np.tanh(z)-
                                                plus2/np.cosh(z)**2*np.tanh(z)))/2
                if cosine <= 0:
                    check(slope>0, 'nonpositive endpoint overlap makes inward force increase')
                    if previous is not None:
                        check(inward>previous, 'independent lengths verify increasing inward response')
                    check(inward<1, 'mismatched inward force approaches tension from below')
                if cosine == -1:
                    repulsion = 1-inward
                    bound = mu*mu/(2*np.sinh(mu*length/2)**2)
                    check(repulsion<=bound+2e-13*max(1,bound), 'gap controls the complete spectral outward response')
                    check(bound<=2/length**2, 'small-length estimate bounds the gap envelope')
                previous = inward
                rows.append(dict(mu=mu,endpoint_cosine=cosine,length=length,
                                 inward_force=inward,inward_force_slope=slope))
        def force(length):
            return 1-np.sum(weights*rates*rates/np.sinh(rates*length/2)**2)/2
        equilibrium = brentq(force,0.01/mu,20/mu,xtol=1e-13)
        negative_curvature = -np.sum(weights*rates/np.sinh(rates*equilibrium))
        check(negative_curvature<0,'formal radial balance remains an angular saddle')
        rows.append(dict(mu=mu,radial_equilibrium=equilibrium,
                         angular_curvature=negative_curvature,
                         interpretation='formal quadratic endpoint control, not a bound hadron'))
    return rows


def main():
    equations = symbolic()
    boundary_cases = numerical()
    forces = force_controls()
    sectors = []
    for depth,carrier,rank in ((2,2,4),(3,4,3),(4,4,3),(5,6,2),(6,6,2),(10,10,0)):
        middle = carrier//2
        hodge_square = (-1)**(middle*middle)
        gate = int(hodge_square == 1)
        active = gate == 1 and rank > 0
        check(active == (depth in (3,4)), 'same middle-Hodge rule retains precisely the existing active guides')
        gradient_coefficient = 4*middle-(depth-1)*(2*middle-2)
        mass_coefficient = (depth-1)*(2*middle-2)
        if depth in (3,4):
            check(gradient_coefficient>0 and mass_coefficient>0,'geometric profile identity allows the active guide')
        else:
            check(gradient_coefficient<=0 or mass_coefficient==0,'ungated geometric-degree profile control remains obstructed')
        sectors.append(dict(depth=depth,even_carrier=carrier,hodge_gate=gate,
                            normal_complex_rank=rank,guide_available_under_existing_premises=active,
                            new_independent_color_selection=False))
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    report = dict(experiment_key='finite-guide-imbalance-competing-ranges',
                  created_utc=datetime.now(timezone.utc).isoformat(),
                  sources={'experiments/'+Path(__file__).name:sha256(Path(__file__).read_bytes()).hexdigest()},
                  evaluations=sum(COUNTS.values()),conditions=COUNTS,maximum_errors=ERRORS,
                  exact_equations=equations,boundary_controls=boundary_cases,
                  force_controls=forces,sectors=sectors,
                  limits=['guide is conditional on the independently varied commuting connection action',
                          'endpoint deformations are prescribed slow shapes, not frozen unequal carrier clocks',
                          'positive matrix controls are not a discretization claim for the actual transverse spectrum',
                          'the full transverse continuum is retained by the analytic spectral proof',
                          'free endpoint orientation makes opposite shapes an angular saddle',
                          'small length can invalidate the quadratic end-core approximation',
                          'three/four restriction is inherited, not independently derived by this response',
                          'full non-Abelian confinement, physical units and the master-field reduction remain open'])
    path = ROOT/('finite-guide-imbalance-results-'+stamp+'.json')
    path.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(report=str(path),evaluations=report['evaluations'],maximum_errors=ERRORS)),flush=True)


if __name__ == '__main__':
    main()
