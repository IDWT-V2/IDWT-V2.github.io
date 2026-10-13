"""Exact frame bulk equations and relaxed guide force/stiffness comparisons."""
from datetime import datetime, timezone
from hashlib import sha256
from math import sqrt
from pathlib import Path
import json

import numpy as np
import sympy as s
from scipy.optimize import brentq

ROOT=Path(__file__).resolve().parent
COUNTS,ERRORS={},{}


def check(value,label):
    if not bool(value):
        raise AssertionError(label)
    COUNTS[label]=COUNTS.get(label,0)+1


def exact(value,label):
    check(s.simplify(value)==0,label)


def close(actual,expected,label,tolerance=2e-7):
    error=float(abs(actual-expected)/max(1,abs(expected)))
    ERRORS[label]=max(ERRORS.get(label,0),error)
    check(error<tolerance,label)


def symbolic():
    time,x,u,omega,k=s.symbols('time x u omega k',real=True)
    phase=omega*time+k*x
    rot=s.Matrix([[s.cos(phase),-s.sin(phase)],[s.sin(phase),s.cos(phase)]])
    frame=u*rot/s.sqrt(2)
    for entry in frame.diff(time,2)-frame.diff(x,2)-(k*k-omega*omega)*frame:
        exact(entry,'full frame bulk wave equation has one common shifted restoring scale')
    exact(s.trace(frame.diff(time).T*frame.diff(time))-u*u*omega*omega,
          'both rotating vectors retain the complete time kinetic energy')
    exact(s.trace(frame.diff(x).T*frame.diff(x))-u*u*k*k,
          'both twisted vectors retain the complete longitudinal gradient energy')
    exact(s.det(frame.T*frame)-u**4/4,'exact uniform twists retain the area dielectric')
    e0,k2,k4=s.symbols('E kappa2 kappa4',positive=True)
    aa=s.Matrix(s.symbols('a0:2',real=True));bb=s.Matrix(s.symbols('b0:2',real=True))
    eps=k2*(aa.dot(aa)+bb.dot(bb))+k4*(aa.dot(aa)*bb.dot(bb)-aa.dot(bb)**2)
    source=s.Matrix([s.diff(eps,a)*e0*e0/2 for a in aa])
    expected=e0*e0*(k2*aa+k4*(bb.dot(bb)*aa-aa.dot(bb)*bb))
    for entry in source-expected:
        exact(entry,'full electric variation retains the actual spatial vector source')
    replacement={aa[i]:frame[i,0] for i in range(2)}
    replacement.update({bb[i]:frame[i,1] for i in range(2)})
    for entry in (source.subs(replacement,simultaneous=True)-
                  e0*e0*(k2+k4*u*u/2)*frame[:,0]):
        exact(s.trigsimp(entry),'balanced rotating frame satisfies the same scalar profile equation')

    length,angle,base=s.symbols('L Theta Omega',positive=True)
    ss=s.symbols('sigma',positive=True)
    tau=s.Function('tau')
    scale=base*base+angle*angle/length**2
    energy=length*tau(scale)
    mass=2*s.diff(tau(ss),ss)
    force=tau(scale)-(angle/length)**2*mass.subs(ss,scale)
    exact(s.diff(energy,length)-force,'complete relaxed bulk energy gives tension minus shape-gradient stress')
    exact(s.diff(energy,angle)-(angle/length)*mass.subs(ss,scale),
          'endpoint torque follows from the same relaxed energy')
    stiffness=(mass+2*(angle/length)**2*s.diff(mass,ss)).subs(ss,scale)
    exact(s.diff(force,length)-(angle/length)**2*stiffness/length,
          'force separation slope equals longitudinal phase stiffness exactly')
    t=s.symbols('t',real=True)
    for n in (2,3):
        coeff=(3+(1-n)*t)/(4-n)
        exact(coeff-1-(n-1)*(1-t)/(4-n),
              'actual guide moment coefficient is strictly above one on zero to one')
        a=s.Rational(4-n,6)
        power=ss**a
        mass_power=2*s.diff(power,ss)
        phase_speed=(mass_power+2*k*k*s.diff(mass_power,ss))/mass_power
        exact(phase_speed.subs(ss,base*base+k*k)-
              (base*base+(2*a-1)*k*k)/(base*base+k*k),
              'pure quartic phase stiffness has the exact twist threshold')
        rotating_speed=mass_power/(mass_power-2*omega*omega*s.diff(mass_power,ss))
        exact(rotating_speed.subs(ss,base*base-omega*omega)-
              (base*base-omega*omega)/(base*base+(1-2*a)*omega*omega),
              'homogeneous rotation has positive subluminal reduced phase speed')


def full_frame_controls():
    z,zt,zx,om,k=s.symbols('z zt zx omega k',real=True)
    # A normalized actual spatial tilt retains Gram data exactly.
    theta=s.symbols('theta',real=True)
    unit=s.Matrix([s.sqrt(1-z*z)*s.cos(theta),s.sqrt(1-z*z)*s.sin(theta),z,0])
    other=s.Matrix([-s.sin(theta),s.cos(theta),0,0])
    exact(unit.dot(unit)-1,'finite full-frame tilt preserves its actual vector norm')
    exact(unit.dot(other),'finite full-frame tilt preserves relative orthogonality')
    for derivative,rate in ((zt,om),(zx,k)):
        velocity=unit.diff(z)*derivative+unit.diff(theta)*rate
        expected=derivative**2/(1-z*z)+(1-z*z)*rate*rate
        exact(s.trigsimp(velocity.dot(velocity)-expected),
              'exact tilted-vector kinetic metric retains orientation and tilt together')
    quadratic=(zt*zt-zx*zx+(k*k-om*om)*z*z)/4
    exact(s.diff(quadratic,z)-(k*k-om*om)*z/2,
          'full-frame quadratic tilt has the exact rotation-versus-twist restoring term')
    uu,electric,kappa2,kappa4,base,zztt,zzxx=s.symbols(
        'u E kappa2 kappa4 Omega ztt zxx',real=True)
    profile_laplacian=(base*base+k*k-om*om-kappa2*electric*electric)*uu-kappa4*electric*electric*uu**3/2
    direct_tilt=uu*(zztt-zzxx)-profile_laplacian*z+(
        base*base-kappa2*electric*electric-kappa4*electric*electric*uu*uu/2)*uu*z
    exact(direct_tilt-uu*(zztt-zzxx+(om*om-k*k)*z),
          'original vector equation independently gives the exact spatial tilt frequency')
    aa=s.Matrix(s.symbols('a0:4',real=True))
    pp=s.Matrix(s.symbols('p0:4',real=True))
    generator=s.Matrix([[0,-1,0,0],[1,0,0,0],[0,0,0,0],[0,0,0,0]])
    base=s.symbols('Omega',positive=True)
    angular=pp.dot(generator*aa)
    hamiltonian=(pp.dot(pp)+base*base*aa.dot(aa))/2
    shifted=pp-om*generator*aa
    decomposition=(shifted.dot(shifted)+(base*base-om*om)*aa.dot(aa)+
                   om*om*(aa.dot(aa)-(generator*aa).dot(generator*aa)))/2
    exact(hamiltonian-om*angular-decomposition,
          'complete rotation Routhian is the claimed positive-square decomposition')
    exact(aa.dot(aa)-(generator*aa).dot(generator*aa)-aa[2]**2-aa[3]**2,
          'all unused spatial frame directions have a nonnegative rotation norm deficit')
    tau,mass=s.symbols('tau M',positive=True)
    speed,clock=s.symbols('speed clock',real=True)
    gamma=1/s.sqrt(1-speed*speed)
    boosted_omega,boosted_k=gamma*clock,-gamma*speed*clock
    exact(boosted_omega**2-boosted_k**2-clock**2,
          'boosted uniform frame retains the same supported transverse restoring coefficient')
    momentum=-mass*boosted_omega*boosted_k
    exact(s.diff(momentum,speed).subs(speed,0)-mass*clock*clock,
          'complete bulk longitudinal momentum has positive motion-supplied linear response')
    controls=[]
    for angle in (0.2,1.0,2.8,3.4):
        for rotation in (0.0,0.5,1.5):
            length=2.0;kvalue=angle/length
            frequency2=(np.pi/length)**2+rotation**2-kvalue**2
            if rotation==0 and angle<np.pi:
                check(frequency2>0,'finite shortest-angle static segment retains a positive lowest tilt frequency')
            if rotation>=kvalue:
                check(frequency2>0,'timelike frame rotation removes this exact tilt growth channel')
            controls.append(dict(endpoint_angle=angle,length=length,frame_rotation=rotation,
                                 lowest_dirichlet_tilt_frequency_squared=frequency2))
    return controls


def mixed_profile(n,scale,q_at_one=10.0):
    # The inherited approximate ground moments set units, not a fitted force.
    moment={2:11.70089652,3:18.89725130}[n]
    restoring=sqrt(scale)
    flux=q_at_one*moment/(2*(4-n))
    q=q_at_one*restoring**(n-1)
    if n==2:
        t=(2/q)**(2/3)
    else:
        fold=(9-sqrt(33))/4
        t=brentq(lambda tt:(2-tt)/(sqrt(1-tt)*tt**1.5)-q,1e-12,fold,xtol=1e-14)
    gap2=scale*(1-t)
    amplitude2=(1-t)/(2*t)
    mass=amplitude2*gap2**(-n/2)*moment
    grad=n*gap2*mass/(4-n)
    quartic=4*amplitude2*mass/(4-n)
    integral=mass+quartic
    electric=sqrt(scale*t)
    tension=flux*electric/2+(grad+scale*mass)/2
    log_q_t=-(1-n/2)/(1-t)-3/(2*t)+(2-n)/(2+(2-n)*t)
    t_scale=(n-1)/(2*scale*log_q_t)
    mass_scale=mass*(-n/(2*scale)+(-(1-n/2)/(1-t)-1/t)*t_scale)
    return dict(n=n,scale=scale,flux=flux,t=t,gap_squared=gap2,
                mass=mass,mass_scale=mass_scale,tension=tension,
                recovered_flux=electric*integral,
                direct_stress=flux*electric/2+(grad+(2-scale)*mass)/2)


def mixed_controls():
    rows=[]
    for n in (2,3):
        for q in (6.0,10.0,25.0):
            for k in (0.0,0.2,0.8,1.5,3.0,10.0):
                scale=1+k*k
                profile=mixed_profile(n,scale,q)
                mass,ms,tension=profile['mass'],profile['mass_scale'],profile['tension']
                close(profile['recovered_flux'],profile['flux'],'independent complete moments recover the prescribed fixed flux')
                close(tension/mass,scale*(3+(1-n)*profile['t'])/(4-n),
                      'complete guide energy gives the exact tension-to-norm relation')
                check(ms<0,'globally bound mixed guide has decreasing profile norm with restoring scale')
                force=tension-k*k*mass
                check(force>0,'relaxed mixed uniform twist has positive net inward force')
                close(force,profile['direct_stress'],'direct full neutral and electric stress agrees with relaxed force')
                step=1e-4*scale
                lower=mixed_profile(n,scale-step,q)
                upper=mixed_profile(n,scale+step,q)
                close((upper['tension']-lower['tension'])/(2*step),mass/2,
                      'independent branch differences verify the fixed-flux envelope derivative')
                close((upper['mass']-lower['mass'])/(2*step),ms,
                      'independent branch differences verify the complete profile norm response')
                stiffness=mass+2*k*k*ms
                if k>0:
                    length=1/k
                    step_l=length*1e-4
                    def energy_force(ell):
                        kk=1/ell; pp=mixed_profile(n,1+kk*kk,q)
                        return ell*pp['tension'],pp['tension']-kk*kk*pp['mass']
                    ep,fp=energy_force(length+step_l)
                    em,fm=energy_force(length-step_l)
                    close((ep-em)/(2*step_l),force,'direct finite-separation energy derivative gives the force')
                    close((fp-fm)/(2*step_l),k*k*stiffness/length,
                          'independent mixed-profile force differences give the phase stiffness identity')
                if stiffness>0:
                    check(0<stiffness/mass<=1,'positive static phase stiffness gives subluminal reduced response')
                profile.update(twist_rate=k,inward_force=force,phase_stiffness=stiffness,
                               reduced_phase_speed_squared=stiffness/mass)
                rows.append(profile)
            root=brentq(lambda kk:((lambda pp:pp['mass']+2*kk*kk*pp['mass_scale'])
                                  (mixed_profile(n,1+kk*kk,q))),0.01,20.0,xtol=1e-12)
            pp=mixed_profile(n,1+root*root,q)
            check(pp['tension']-root*root*pp['mass']>0,
                  'mixed force minimum remains strictly inward when phase stiffness changes sign')
            rows.append(dict(n=n,reference_scaled_flux=q,phase_stiffness_zero_twist=root,
                             inward_force_at_threshold=pp['tension']-root*root*pp['mass']))
    return rows


def pure_controls():
    rows=[]
    for depth in (3,4):
        n=depth-1;a=(4-n)/6
        for rate in (0,0.1,0.5,1,2,5,20):
            y=rate*rate
            force=(1+y)**(a-1)*(1+(1-2*a)*y)
            speed=(1-(1-2*a)*y)/(1+y)
            check(force>0,'exact pure-quartic relaxed twist remains inward at all control gradients')
            rows.append(dict(depth=depth,power_exact=str(s.Rational(4-n,6)),
                             twist_over_restoring=rate,inward_force_over_tension=force,
                             reduced_phase_speed_squared=speed))
    return rows


def main():
    symbolic()
    tilts=full_frame_controls()
    mixed=mixed_controls()
    pure=pure_controls()
    sectors=[]
    for depth,carrier in ((2,2),(3,4),(4,4),(5,6),(6,6),(10,10)):
        gate=int((-1)**((carrier//2)**2)==1)
        check(bool(gate)==(depth in (3,4)), 'occupied-sector gate remains the same without a twist-specific coefficient')
        sectors.append(dict(depth=depth,hodge_gate=gate,twisted_guide_available=bool(gate),
                            independently_new_dimensional_selection=False))
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    report=dict(experiment_key='relaxed-frame-twist-force-and-longitudinal-stiffness',
                created_utc=datetime.now(timezone.utc).isoformat(),
                sources={'experiments/'+Path(__file__).name:sha256(Path(__file__).read_bytes()).hexdigest()},
                evaluations=sum(COUNTS.values()),conditions=COUNTS,maximum_errors=ERRORS,
                mixed_controls=mixed,pure_quartic_controls=pure,full_frame_tilt_controls=tilts,sectors=sectors,
                exact_equations={'shifted_restoring':'s=Omega^2+k^2-varpi^2',
                                 'force':'tau-k^2 M','phase_stiffness':'G=M+2k^2 partial_s M',
                                 'force_slope':'partial_L F_in=(k^2/L)G'},
                limits=['uniform bulk frame and supported transverse profile are exact within the retained action',
                        'finite end caps and general dynamic sources are not solved',
                        'effective modulated phase action is a leading long-wavelength reduction',
                        'phase-channel stiffness does not prove full orientation and connection stability',
                        'rotating global energetic bound does not settle the degenerate vacuum Cauchy problem',
                        'bulk longitudinal inertia is a response per length with tension, not particle rest mass',
                        'frame rotation frequency is not identified with the particle mass clock',
                        'ground moment inputs are inherited decimal approximations, not exact rational constants',
                        'three/four restriction is inherited from the guide action, not an independent new selection',
                        'master-field reduction, non-Abelian confinement and physical normalization remain open'])
    path=ROOT/('relaxed-frame-twist-results-'+stamp+'.json')
    path.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(report=str(path),evaluations=report['evaluations'],maximum_errors=ERRORS)),flush=True)


if __name__=='__main__':
    main()
