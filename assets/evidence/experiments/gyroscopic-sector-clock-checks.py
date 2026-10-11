"""Ordinary rotation-plane clocks, canonical currents and exact fast-band controls."""
from collections import Counter
from datetime import datetime,timezone
from hashlib import sha256
import json
from pathlib import Path

import mpmath as mp
import numpy as np
import sympy as sp

ROOT=Path(__file__).resolve().parent
OUTPUT_ROOT=ROOT.parent/'results' if ROOT.name=='experiments' else ROOT
counts=Counter()


def check(value,label):
    if not bool(value):
        raise AssertionError(label)
    counts[label]+=1


def close(a,b,label,tol=2e-10):
    check(np.allclose(a,b,atol=tol,rtol=tol),label)


k,omega,c,w=sp.symbols('k Omega c w',positive=True)
energy=sp.sqrt(c*c*k*k+omega*omega/4)
upper=energy+omega/2
lower=c*c*k*k/upper
for frequency,sign in ((upper,1),(lower,-1)):
    check(sp.simplify(frequency**2-sign*omega*frequency-c*c*k*k)==0,
          'exact circular wave dispersion')
    check(sp.simplify((frequency**2+c*c*k*k)/(2*energy)-frequency)==0,
          'canonical gyro wave energy normalization')
check(sp.simplify((upper+lower)/(4*energy)-sp.Rational(1,2))==0,
      'canonical gyro partial velocity vertex exactly one half')
check(sp.simplify(2*upper-omega-2*energy)==0,'upper symplectic residue')
check(sp.simplify(2*lower+omega-2*energy)==0,'lower symplectic residue')
response_frequency=sp.symbols('response_frequency',real=True)
denominator=(lower**2-response_frequency**2)*(upper**2-response_frequency**2)
gyro_response=(c*c*k*k-response_frequency**2)/denominator
bending_response=(omega**2+c*c*k*k-response_frequency**2)/denominator
check(sp.simplify(gyro_response-(lower/(upper+lower)/(lower**2-response_frequency**2)
                               +upper/(upper+lower)/(upper**2-response_frequency**2)))==0,
      'gyro response has lower branch residue suppressed at small wave number')
check(sp.simplify(bending_response-(upper/(upper+lower)/(lower**2-response_frequency**2)
                                  +lower/(upper+lower)/(upper**2-response_frequency**2)))==0,
      'bending response has the opposite diagonal residue weights')
check(sp.simplify(gyro_response.subs(response_frequency,0)-1/(c*c*k*k))==0,
      'gyro static response is second order despite quadratic lower dispersion')
check(sp.simplify(bending_response.subs(response_frequency,0)
                  -1/(c*c*k*k)-omega**2/(c**4*k**4))==0,
      'bending static response retains fourth order with identical branch frequencies')

rng=np.random.default_rng(20261012)
sector=[]
frame_controls=[]
for depth in (2,3,4,5,6,10):
    skew=np.zeros((depth,depth))
    for j in range(depth//2):
        skew[2*j,2*j+1]=-1.
        skew[2*j+1,2*j]=1.
    # An arbitrary common ordinary rotation removes any preferred axes.
    rotation=np.linalg.qr(rng.normal(size=(depth,depth)))[0]
    rotated=rotation@skew@rotation.T
    close(rotated.T,-rotated,'ordinary gyroscopic tensor antisymmetric')
    check(np.linalg.matrix_rank(rotated,tol=1e-10)==2*(depth//2),
          'ordinary even rank and odd axis')
    eigen=np.linalg.eigvalsh(1j*rotated)
    expected=sorted([-1.,1.]*(depth//2)+([0.] if depth%2 else []))
    close(eigen,expected,'rotation plane frequencies are coordinate invariant')
    for wave_number in (.03,.2,1.,5.,100.):
        operator=np.block([[np.zeros((depth,depth)),np.eye(depth)],
                            [-wave_number**2*np.eye(depth),-rotated]])
        freqs=sorted(x.imag for x in np.linalg.eigvals(operator) if x.imag>1e-8)
        up=(np.sqrt(1+4*wave_number**2)+1)/2
        low=wave_number**2/up
        expect=sorted([low,up]*(depth//2)+([wave_number] if depth%2 else []))
        close(freqs,expect,'full spatial gyro spectrum')
        vlow=np.array([1.,1j])/np.sqrt(2)
        vup=np.array([1.,-1j])/np.sqrt(2)
        e0=np.sqrt(wave_number**2+.25)
        t=2*np.pi*np.arange(128)/128
        field=(vlow[:,None]*np.exp(1j*t)+vup[:,None])/np.sqrt(2*e0)
        velocity=(-1j*low*vlow[:,None]*np.exp(1j*t)-1j*up*vup[:,None])/np.sqrt(2*e0)
        current=(1j*(field[0].conj()*velocity[0]-velocity[0].conj()*field[0])).real
        close(abs(2*np.fft.fft(current)[1]/len(t)),1.,'full canonical gyro partial current coefficient')
        total_energy=np.sum(abs(velocity)**2+wave_number**2*abs(field)**2,axis=0)
        close(total_energy,up+low,'full canonical gyro energy has no beat')
        total_mechanical=(1j*np.sum(field.conj()*velocity-velocity.conj()*field,axis=0)).real
        # Circular modes remain orthogonal even though their action
        # normalization differs from the unmodified wave normalization.
        close(np.ptp(total_mechanical),0,'total gyro identity readout has no beat')
        # Counterpropagating equal-action circular waves have zero
        # canonical translation momentum but a residual energy flux.
        # The fixed gyro background is consequently a physical frame.
        shape=(vlow+vup)/np.sqrt(2*e0)
        time_derivative=(-1j*low*vlow-1j*up*vup)/np.sqrt(2*e0)
        spatial_derivative=(1j*wave_number*vlow-1j*wave_number*vup)/np.sqrt(2*e0)
        plane_gyro=np.array([[0.,-1.],[1.,0.]])
        canonical_momentum=time_derivative+plane_gyro@shape/2
        translation=-2*np.vdot(canonical_momentum,spatial_derivative).real
        flux=-2*np.vdot(time_derivative,spatial_derivative).real
        close(translation,0,'counterpropagating gyro waves have zero translation momentum')
        close(flux,-wave_number/e0,'zero gyro translation momentum does not imply zero energy flux')
        check(abs(flux)>0,'fixed gyro background cannot be omitted from rest frame analysis')
        frame_controls.append({'real_depth':depth,'wave_number':wave_number,
                               'equal_canonical_actions':1,'complete_fixed_background_wave_momentum':translation,
                               'fixed_background_wave_energy_flux':flux,
                               'complete_particle_boost_law_derived':False})
        if depth%2:
            check(low<wave_number<up,'odd null-axis branch is not lowest energy')
        sector.append({'real_depth':depth,'wave_number':wave_number,'rotation_planes':depth//2,
                       'unpaired_axes':depth%2,'lower_frequency':low,'upper_frequency':up,
                       'fundamental_gap':1,'canonical_partial_velocity_vertex':'1/2',
                       'odd_axis_selected_by_energy':False,
                       'neutrino_neutrality_derived':False})

orientation_cases=[]
orientation_moments=[]
for depth in (2,3,4,5,6,10):
    vlow=np.zeros(depth,dtype=complex)
    vup=np.zeros(depth,dtype=complex)
    vlow[:2]=np.array([1.,1j])/np.sqrt(2)
    vup[:2]=np.array([1.,-1j])/np.sqrt(2)
    for eta in ((1.,) if depth==2 else (0.,.25,1.)):
        for angle in (0.,.3,.9,1.4):
            axis=np.zeros(depth)
            axis[:2]=np.sqrt(eta)*np.array([np.cos(angle),np.sin(angle)])
            if depth>2:
                axis[2]=np.sqrt(1-eta)
            probe=np.outer(axis,axis)
            vertex=np.vdot(vlow,probe@vup)
            close(vertex,eta*np.exp(-2j*angle)/2,'polarized gyro readout amplitude and phase')
            rotation=np.linalg.qr(rng.normal(size=(depth,depth)))[0]
            close(np.vdot(rotation@vlow,(rotation@probe@rotation.T)@(rotation@vup)),vertex,
                  'gyro source and polarized readout common rotation covariance')
            orientation_cases.append({'real_depth':depth,'readout_plane_projection_squared':eta,
                                      'readout_plane_angle':angle,'partial_vertex_absolute':abs(vertex),
                                      'reciprocal_encounter_dynamics_derived':False})
    orientation_moments.append({'real_depth':depth,
                                'isotropic_mean_absolute_vertex':str(sp.Rational(1,depth)),
                                'isotropic_mean_squared_vertex':str(sp.Rational(2,depth*(depth+2))),
                                'depth_five_uniquely_suppressed':False})

# Parallel gyroscopic tensors must commute with the spatial holonomy.
# These pointwise holonomy controls do not construct nested manifolds.
holonomy_cases=[]
for depth in (2,3,4,5,6,10):
    basis=[]
    def skew_generator(i,j):
        value=np.zeros((depth,depth),dtype=np.int64)
        value[i,j]=-1
        value[j,i]=1
        return value
    for i in range(depth):
        for j in range(i+1,depth):
            basis.append(skew_generator(i,j))
    j_tensor=sum((skew_generator(2*j,2*j+1) for j in range(depth//2)),
                 np.zeros((depth,depth),dtype=np.int64))
    unitary_generators=[skew_generator(2*j,2*j+1) for j in range(depth//2)]
    for i in range(depth//2):
        for j in range(i+1,depth//2):
            unitary_generators.extend([
                skew_generator(2*i,2*j)+skew_generator(2*i+1,2*j+1),
                skew_generator(2*i,2*j+1)-skew_generator(2*i+1,2*j)])
    for name,generators,expected in (
        ('full round-space SO(d) holonomy',basis,1 if depth==2 else 0),
        ('Kahler U(m) holonomy with one flat axis when odd',unitary_generators,1)):
        constraints=np.concatenate([
            np.column_stack([(g@b-b@g).reshape(-1) for b in basis]) for g in generators],axis=0)
        gram=constraints.T@constraints
        dimension=len(basis)-sp.Matrix(gram.tolist()).rank()
        check(dimension==expected,'exact ordinary skew holonomy commutant')
        if name.startswith('Kahler'):
            check(all(np.array_equal(g@j_tensor,j_tensor@g) for g in generators),
                  'parallel complex rotation candidate commutes with unitary holonomy')
        holonomy_cases.append({'real_depth':depth,'holonomy':name,
                               'parallel_antisymmetric_tensor_dimension':int(dimension),
                               'nested_geometry_constructed':False})

# A positive isotropic density self-interaction preserves the gap on
# exact constant-density complex circular preparations. This is a changed
# nonlinear action, not a result for the real wave with a quartic added.
nonlinear=[]
kap,lam,norm,eg,ap,am,om=sp.symbols('kappa lambda N E aplus aminus Omega',positive=True)
nonlinear_energy=(kap+lam*norm+om**2/4)**sp.Rational(1,2)
wp=nonlinear_energy+om/2
wm=nonlinear_energy-om/2
check(sp.simplify(wp-wm-om)==0,'nonlinear isotropic branch gap exact')
check(sp.simplify((wp+wm)/(4*nonlinear_energy)-sp.Rational(1,2))==0,
      'nonlinear phase action partial vertex exact')
# H=E Itotal+Omega(Iplus-Iminus)/2-lambda N²/2,
# Itotal=2EN. Its complete derivative must give the phase frequencies.
dn=sp.diff(2*nonlinear_energy*norm,norm)
de=sp.diff(nonlinear_energy,norm)
check(sp.simplify(nonlinear_energy+(2*nonlinear_energy*norm*de-lam*norm)/dn-nonlinear_energy)==0,
      'nonlinear complete energy derivative gives both phase frequencies')
for depth in (2,3,4,5,6,10):
    for stiffness in (.04,1.,100.):
        for coefficient in (.1,.8,2.):
            amplitudes=np.zeros(depth,dtype=complex)
            amplitudes[:2]=(.6,.8)
            density=np.vdot(amplitudes,amplitudes).real
            e=np.sqrt(stiffness+coefficient*density+.25)
            high=e+.5
            low=e-.5
            t=np.linspace(0,2*np.pi,129)
            z=amplitudes[:,None]*np.exp(-1j*np.array([low,high]+[low]*(depth-2))[:,None]*t)
            close(np.sum(abs(z)**2,axis=0),density,'nonlinear circular preparation constant density')
            close(low**2+low,stiffness+coefficient*density,'nonlinear lower exact equation')
            close(high**2-high,stiffness+coefficient*density,'nonlinear upper exact equation')
            # Linearize the homogeneous two-circular-component subsystem
            # in its exact rotating frame. The remaining empty components
            # are stable decoupled linear waves with positive stiffness.
            b=amplitudes[:2].real
            identity=np.eye(2)
            zero=np.zeros((2,2))
            curvature=2*coefficient*np.outer(b,b)
            operator=np.block([[zero,zero,identity,zero],
                               [zero,zero,zero,identity],
                               [-curvature,zero,zero,-2*e*identity],
                               [zero,zero,2*e*identity,zero]])
            values=np.linalg.eigvals(operator)
            check(max(abs(values.real))<1e-6,'nonlinear circular subsystem has no exponential normal modes')
            frequencies=sorted(v.imag for v in values if v.imag>1e-6)
            close(frequencies,[2*e,np.sqrt(4*e*e+2*coefficient*density)],
                  'nonlinear radial and transverse disturbance frequencies')
            energy_check=(low**2*abs(b[0])**2+high**2*abs(b[1])**2
                          +stiffness*density+coefficient*density**2/2)
            actions=2*e*abs(b)**2
            close(energy_check,low*actions[0]+high*actions[1]-coefficient*density**2/2,
                  'nonlinear complete energy retains interaction term')
            phases=2*np.pi*np.arange(128)/128
            circular_low=np.array([1.,1j])/np.sqrt(2)
            circular_high=np.array([1.,-1j])/np.sqrt(2)
            phi=b[0]*circular_low[:,None]*np.exp(1j*phases)+b[1]*circular_high[:,None]
            phidot=(-1j*low*b[0]*circular_low[:,None]*np.exp(1j*phases)
                    -1j*high*b[1]*circular_high[:,None])
            current=(1j*(phi[0].conj()*phidot[0]-phidot[0].conj()*phi[0])).real
            close(abs(2*np.fft.fft(current)[1]/len(phases)),np.sqrt(actions[0]*actions[1]),
                  'nonlinear physical partial current agrees with phase action vertex')
            nonlinear.append({'real_depth':depth,'bare_spatial_stiffness':stiffness,
                              'density_interaction_coefficient':coefficient,'prepared_density':density,
                              'lower_frequency':low,'upper_frequency':high,'gap':high-low,
                              'partial_phase_action_current_vertex':'1/2',
                              'full_spatial_stability_tested':False,
                              'particle_mass_frequency_amplitude_independent':False})

# The previously derived positive same-field completion must be retained
# when mass-frequency carriers approach its eliminated gap.
mp.mp.dps=75


def bisect(fun,left,right):
    fl,fr=fun(left),fun(right)
    if fl==0:
        return left
    if fr==0:
        return right
    check(fl*fr<0,'full fast-band root bracket')
    for iteration in range(300):
        middle=(left+right)/2
        fm=fun(middle)
        if fl*fm<=0:
            right=middle
        else:
            left,fl=middle,fm
    return (left+right)/2


completion=[]
baseline_path=ROOT/'grounding/complex-comb-response-results.json'
if not baseline_path.is_file():
    baseline_path=ROOT.parent/'results/complex-comb-response-results.json'
baseline=json.loads(baseline_path.read_text())
for case in baseline['particle_cases']:
    cutoff=case['cutoff']
    if cutoff==1:
        continue
    for gap in (100,10**10):
        delta=mp.mpf(gap)
        amplitude_squared=delta/(delta-1)
        effective_mass=amplitude_squared
        stiffness=effective_mass*cutoff*(cutoff-1)
        def poly(freq):
            return (stiffness-freq*freq)*(delta-freq)-delta*amplitude_squared*freq
        bound=mp.sqrt(stiffness)+delta+amplitude_squared+1
        negative=bisect(poly,-bound,mp.mpf(0))
        pos0=bisect(poly,mp.mpf(0),delta)
        pos1=bisect(poly,delta,bound)
        def residue(freq):
            return 2*freq+delta*delta*amplitude_squared/(delta-freq)**2
        strengths=[2*p/residue(p) for p in (pos0,pos1)]
        selected=pos0 if strengths[0]>=strengths[1] else pos1
        signed=abs(negative)-selected
        canonical_vertex=(abs(negative)+selected)/(2*mp.sqrt(abs(residue(negative))*residue(selected)))
        effective_vertex=effective_mass*canonical_vertex
        check(residue(negative)<0 and residue(pos0)>0 and residue(pos1)>0,
              'full positive energy branch symplectic signs')
        for root in (negative,pos0,pos1):
            check(abs(poly(root))/max(mp.mpf(1),bound**3)<mp.mpf('1e-65'),
                  'full fast-band cubic residual')
            fast_ratio=delta*mp.sqrt(amplitude_squared)/(delta-root)
            full_energy=root**2+stiffness+delta*(fast_ratio-mp.sqrt(amplitude_squared))**2
            check(abs(full_energy-root*residue(root))/max(mp.mpf(1),abs(full_energy))<mp.mpf('1e-65'),
                  'full completion canonical mode energy identity')
        if gap==10**10:
            check(abs(effective_vertex-mp.mpf('.5'))<mp.mpf('1e-4'),
                  'far fast-band canonical partial vertex approaches one half')
        if gap==10**10:
            check(abs(abs(signed)-1)<mp.mpf('1e-4'),
                  'far fast-band completion approaches common clock')
        if gap==100 and cutoff>10000:
            check(abs(signed)<mp.mpf('.001'),
                  'high carriers do not inherit constant adiabatic gyro clock')
        completion.append({'particle':case['particle'],'real_depth':case['real_depth'],'cutoff':cutoff,
                           'fast_gap_over_own_sector_clock':gap,'bare_slow_mass':1,
                           'common_coupling_amplitude_squared':str(amplitude_squared),
                           'exact_positive_frequencies':[str(abs(negative)),str(pos0),str(pos1)],
                           'positive_branch_coordinate_strengths':list(map(str,strengths)),
                           'full_symplectic_residues':list(map(lambda p:str(residue(p)),(negative,pos0,pos1))),
                           'bare_displacement_partial_velocity_vertex':str(canonical_vertex),
                           'effective_mass_normalized_partial_velocity_vertex':str(effective_vertex),
                           'dominant_displacement_gap_signed':str(signed),
                           'dominant_displacement_gap_absolute':str(abs(signed)),
                           'mass_and_spatial_targets_retuned':False})

report={'experiment':'gyroscopic-sector-clock','created_utc':datetime.now(timezone.utc).isoformat(),
        'status':'passed','checks':sum(counts.values()),'check_counts':dict(counts),
        'scope':'Ordinary spatial rotation-plane wave prototype with correct symplectic normalization; full finite-gap completion tests its adiabatic domain. No assigned depth or particle occupation is selected.',
        'sector_cases':sector,'fixed_background_frame_cases':frame_controls,
        'polarized_readout_cases':orientation_cases,'isotropic_orientation_moments':orientation_moments,
        'holonomy_cases':holonomy_cases,'nonlinear_cases':nonlinear,
        'full_fast_band_cases':completion,
        'sources':{'experiments/gyroscopic-sector-clock-checks.py':sha256(Path(__file__).read_bytes()).hexdigest(),
                   'results/complex-comb-response-results.json':sha256(baseline_path.read_bytes()).hexdigest()}}
out=OUTPUT_ROOT/'gyroscopic-sector-clock-results.json'
out.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'checks':report['checks'],'sector_cases':len(sector),
                  'full_completion_cases':len(completion),'report':str(out)}))
