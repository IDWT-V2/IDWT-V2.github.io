"""Positive displacement/rotation response: exact branch gap and static pole."""
from collections import Counter
from datetime import datetime,timezone
from fractions import Fraction as F
from hashlib import sha256
import json
import math
from pathlib import Path

import numpy as np
import sympy as sp

ROOT=Path(__file__).resolve().parent
counts=Counter()


def check(value,label):
    if not bool(value):
        raise AssertionError(label)
    counts[label]+=1


def close(a,b,label,tol=2e-10):
    check(np.allclose(a,b,atol=tol,rtol=tol),label)


k,c,clock,z=sp.symbols('k c Omega z',positive=True)
c2=sp.symbols('c_2',positive=True)
kap,bend,mass,inertia=sp.symbols('kappa b M I',positive=True)
stiff=sp.Matrix([[kap*k*k,-kap*k],[-kap*k,kap+bend*k*k]])
check(sp.factor(stiff.det())==kap*bend*k**4,'exact positive stiffness determinant')
static=sp.factor(kap*k*k-kap**2*k*k/(kap+bend*k*k))
check(sp.simplify(static-kap*bend*k**4/(kap+bend*k*k))==0,'static Schur bending stiffness')
check(sp.simplify(1/static-1/(bend*k**4)-1/(kap*k*k))==0,'static inverse fourth plus second order')
effective=kap*k*k-mass*z-kap**2*k*k/(kap+bend*k*k-inertia*z)
check(sp.factor(effective*(kap+bend*k*k-inertia*z))==
      sp.factor((stiff-sp.diag(mass,inertia)*z).det()),'full frequency Schur complement')
matched=sp.Matrix([[c*c*k*k,-c*clock*k],[-c*clock*k,clock*clock+c*c*k*k]])
root=sp.sqrt(clock*clock+4*c*c*k*k)
plus=(root+clock)/2
minus=2*c*c*k*k/(root+clock)
for w in (plus,minus):
    check(sp.simplify((matched-w*w*sp.eye(2)).det())==0,'exact matched branch frequency')
check(sp.simplify(plus-minus-clock)==0,'exact carrier independent branch gap')
check(sp.simplify(plus*minus-c*c*k*k)==0,'exact branch frequency product')
check(sp.simplify(sp.diff(plus,k)-2*c*c*k/root)==0,'common finite group velocity')
check(sp.simplify(sp.diff(minus,k)-sp.diff(plus,k))==0,'equal branch group velocities')
raw_vertex=c*k/root
canonical_density_vertex=raw_vertex/(2*sp.sqrt(sp.simplify(plus*minus)))
check(sp.simplify(canonical_density_vertex-1/(2*root))==0,
      'canonical density vertex retains carrier suppression')
check(sp.simplify((plus+minus)*canonical_density_vertex-sp.Rational(1,2))==0,
      'canonical velocity current vertex exactly one half')
unmatched_matrix=sp.Matrix([[c*c*k*k,-c*clock*k],[-c*clock*k,clock*clock+c2*c2*k*k]])
check(sp.simplify(unmatched_matrix.trace()-2*c*c2*k*k-clock**2-(c-c2)**2*k*k)==0,
      'unmatched exact gap squared')
check(sp.factor(unmatched_matrix.det())==c*c*c2*c2*k**4,
      'unmatched frequency product determinant')

# The one-dimensional conservative energy system has an exact Dirac
# factorization plus a commuting clock polarization.
principal=sp.Matrix([[0,1,0,0],[1,0,0,0],[0,0,0,1],[0,0,1,0]])
conversion=sp.Matrix([[0,0,0,-1],[0,0,0,0],[0,0,0,0],[1,0,0,0]])
clock_polarization=sp.Matrix([[0,0,0,-sp.I],[0,0,-sp.I,0],
                              [0,sp.I,0,0],[sp.I,0,0,0]])
grading=sp.diag(1,-1,-1,1)
dirac_mass=clock_polarization*grading
full_h=-c*k*principal+sp.I*clock*conversion
dirac_h=-c*k*principal+clock*dirac_mass/2
check(principal**2==sp.eye(4),'first order spatial Clifford square')
check(dirac_mass**2==sp.eye(4),'first order mass Clifford square')
check(principal*dirac_mass+dirac_mass*principal==sp.zeros(4),
      'first order Clifford anticommutation')
check(clock_polarization**2==sp.eye(4),'clock polarization involution')
check(full_h==dirac_h+clock*clock_polarization/2,'exact Dirac plus clock decomposition')
check(full_h*clock_polarization-clock_polarization*full_h==sp.zeros(4),
      'clock polarization commutes with full first order operator')
check(sp.simplify(dirac_h**2-(c*c*k*k+clock*clock/4)*sp.eye(4))==sp.zeros(4),
      'exact Dirac dispersion factorization')

clifford_controls=[]
for depth in (2,3,4,5,6,10):
    allowed=[]
    for mask in range(1<<10):
        degree=mask.bit_count()
        commutes_beta=(degree%2==0)
        commutes_tangent=all(((degree-(1 if mask&(1<<j) else 0))%2)==0 for j in range(depth))
        if commutes_beta and commutes_tangent:
            allowed.append(mask)
    expected=2**(9-depth) if depth<10 else 1
    check(len(allowed)==expected,'full spatial Clifford and mass commutant dimension')
    fixed_plane=(1<<8)|(1<<9)
    check((fixed_plane in allowed)==(depth<10),'ordinary fixed clock plane compatibility')
    clifford_controls.append({'real_depth':depth,
                              'complex_pointwise_commutant_dimension':len(allowed),
                              'fixed_9_10_rotation_clock_commutes':fixed_plane in allowed,
                              'nontrivial_pointwise_independent_clock_allowed':len(allowed)>1,
                              'spatial_band_clock_excluded':False})

numerical=[]
for depth in (2,3,4,5,6,10):
    for wave_number in (.001,.03,.2,1.,3.,20.,1000.):
        a=np.array([[wave_number**2,-wave_number],[-wave_number,1+wave_number**2]])
        values,vectors=np.linalg.eigh(a)
        big=(math.sqrt(1+4*wave_number**2)+1)/2
        small=wave_number*wave_number/big
        close(values,[small**2,big**2],'independent full modal eigenvalues',2e-9)
        check(values[0]>0,'full positive kinetic branch')
        vminus,vplus=vectors[:,0],vectors[:,1]
        # Identity sees orthogonal branches; component-selective and strain
        # readouts see the gap. A unit-mass kinetic basis is used here.
        close(np.dot(vminus,vplus),0,'integrated identity readout has no gap beat')
        q_vertex=abs(vminus[0]*vplus[0])
        shear=np.array([-wave_number,1.])
        shear_vertex=abs(np.dot(shear,vminus)*np.dot(shear,vplus))
        expected=wave_number/math.sqrt(1+4*wave_number**2)
        close(q_vertex,expected,'displacement source gap coefficient')
        close(shear_vertex,wave_number**2*expected,'shear source adds squared wave number')
        canonical_density=q_vertex/(2*np.sqrt(big*small))
        canonical_velocity=(big+small)*canonical_density
        close(canonical_density,1/(2*np.sqrt(1+4*wave_number**2)),
              'independent canonical density source normalization')
        close(canonical_velocity,.5,'independent canonical velocity vertex universal')
        check(0<q_vertex<.5,'gap vertex bounded and nonzero')
        group=2*wave_number/math.sqrt(1+4*wave_number**2)
        check(0<group<1,'both group velocities below characteristic speed')
        t=2*np.pi*np.arange(128)/128
        # Factor the high carrier when evaluating the same canonical field.
        field=(vminus[:,None]*np.exp(1j*t)/np.sqrt(2*small)+
               vplus[:,None]/np.sqrt(2*big))
        time_derivative=(-1j*small*vminus[:,None]*np.exp(1j*t)/np.sqrt(2*small)-
                         1j*big*vplus[:,None]/np.sqrt(2*big))
        jq=(1j*(field[0].conj()*time_derivative[0]-
                time_derivative[0].conj()*field[0])).real
        jtotal=(1j*np.sum(field.conj()*time_derivative-
                         time_derivative.conj()*field,axis=0)).real
        close(jtotal,2,'total conserved current has no branch beat')
        close(abs(2*np.fft.fft(jq)[1]/len(t)),1.,
              'full canonical velocity current fundamental coefficient')
        energy=np.sum(abs(time_derivative)**2,axis=0)+np.einsum('it,ij,jt->t',field.conj(),a,field).real
        close(energy,big+small,'full canonical branch energy has no beat',2e-9)
        numerical.append({'real_depth':depth,'wave_number_in_clock_units':wave_number,
                          'upper_frequency_over_clock':big,'lower_frequency_over_clock':small,
                          'branch_gap_over_clock':1,'q_source_vertex':q_vertex,
                          'shear_source_vertex':shear_vertex,'group_velocity_over_c':group})
        numerical[-1].update({'canonical_density_vertex':canonical_density,
                              'canonical_velocity_current_vertex':canonical_velocity})

# Unmatched speeds are a physical control, not a per-particle fit.
unmatched=[]
for speed_squared in (.5,.9,1.,1.1,2.):
    for wave_number in (.2,1.,5.,20.):
        mat=np.array([[wave_number**2,-wave_number],[-wave_number,1+speed_squared*wave_number**2]])
        vals=np.linalg.eigvalsh(mat)
        check(vals[0]>0,'unmatched speeds remain positive')
        gap=float(np.sqrt(vals[1])-np.sqrt(vals[0]))
        close(gap,np.sqrt(1+(1-np.sqrt(speed_squared))**2*wave_number**2),
              'independent unequal speed gap identity')
        if speed_squared==1:
            close(gap,1,'matched control constant gap')
        unmatched.append({'rotation_speed_squared_over_displacement_speed_squared':speed_squared,
                          'wave_number':wave_number,'gap':gap})

# Full tensor stiffness, evaluated at a generic spatial wave vector, checks
# that the longitudinal response is identical at every occupied depth.
tensor=[]
for depth in (2,3,4,5,6,10):
    vector=np.arange(1,depth+1,dtype=float)
    vector/=np.linalg.norm(vector)
    vector*=.7
    norm=float(np.linalg.norm(vector))
    matrix=np.zeros((depth+1,depth+1))
    matrix[0,0]=norm**2
    matrix[0,1:]=-vector
    matrix[1:,0]=-vector
    matrix[1:,1:]=(1+norm**2)*np.eye(depth)
    vals=np.linalg.eigvalsh(matrix)
    big=(np.sqrt(1+4*norm**2)+1)/2
    small=norm**2/big
    expected=sorted([small**2,big**2]+[1+norm**2]*(depth-1))
    close(vals,expected,'full spatial longitudinal and transverse spectrum')
    inverse=np.linalg.inv(matrix)
    close(inverse[0,0],1/norm**4+1/norm**2,'full spatial q static fourth order response')
    shear_map=np.column_stack((-vector,np.eye(depth)))
    shear_response=shear_map@inverse@shear_map.T
    long=vector/norm
    close(long@shear_response@long,1.,'longitudinal shear vertex cancels static fourth order pole')
    rot_response=long@inverse[1:,1:]@long
    close(rot_response,1/norm**2,'rotation vertex has only second order pole')
    tensor.append({'real_depth':depth,'transverse_rotation_modes':depth-1,
                   'q_response':'1/k^4+1/k^2','longitudinal_rotation_response':'1/k^2',
                   'longitudinal_shear_response':'1',
                   'direct_q_neutral_confinement_window':depth in (3,4),
                   'rotation_neutral_confinement_window':depth in (2,)})

baseline_path=ROOT/'grounding/complex-comb-response-results.json'
baseline=json.loads(baseline_path.read_text())
assigned=[]
speed_sensitivity=[]
for source in baseline['particle_cases']:
    cutoff=source['cutoff']
    # Inverting the upper branch preserves the supplied target; it does
    # not select a spatial eigenvalue. Exact arithmetic avoids huge matrices.
    k_squared=cutoff*(cutoff-1)
    radicand=1+4*k_squared
    check(math.isqrt(radicand)==2*cutoff-1,'assigned adjacent branch exact square')
    vertex_squared=F(k_squared,(2*cutoff-1)**2)
    check(vertex_squared==F(1,4)-F(1,4*(2*cutoff-1)**2),'assigned gap vertex saturation identity')
    if cutoff==1:
        check(vertex_squared==0,'down endpoint has no paired dynamic gap source')
    # A common unit-radius round sphere does not generally supply the
    # required spatial eigenvalue. Test the occupied depth itself exactly.
    depth=source['real_depth']
    discr=(depth-1)**2+4*k_squared
    integer_root=math.isqrt(discr)
    spherical=(integer_root**2==discr and (integer_root-(depth-1))%2==0)
    degree=(integer_root-(depth-1))//2 if spherical else None
    assigned.append({'particle':source['particle'],'real_depth':depth,'cutoff':cutoff,
                     'required_spatial_wave_number_squared':k_squared,
                     'upper_frequency_over_sector_clock':cutoff,
                     'lower_frequency_over_sector_clock':cutoff-1,
                     'q_vertex_squared_exact':str(vertex_squared),
                     'q_vertex_squared':float(vertex_squared),
                     'canonical_density_vertex_in_clock_units':str(F(1,2*(2*cutoff-1))) if cutoff>1 else None,
                     'canonical_velocity_current_vertex':'1/2' if cutoff>1 else None,
                     'round_sphere_eigenvalue_matches':spherical,'sphere_degree':degree,
                     'spatial_target_is_derived':False})
    if cutoff>1:
        for epsilon in (-1e-6,-1e-8,1e-10,1e-8,1e-6):
            gap=math.sqrt(1+epsilon*epsilon*k_squared)
            velocity_vertex=1/(2*math.sqrt(1+epsilon)*gap)
            speed_sensitivity.append({'particle':source['particle'],'real_depth':depth,
                                      'rotation_speed_relative_error':epsilon,
                                      'gap_over_sector_clock':gap,
                                      'canonical_velocity_vertex':velocity_vertex,
                                      'spatial_wave_number_held_fixed':True})

# On a curved base, a covariant gradient needs a Weitzenbock correction.
# This formal eigenvalue control is valid only if the chosen rotation
# stiffness preserves gradients with the stated eigenvalue.
ell=sp.symbols('ell',nonnegative=True,integer=True)
check(sp.expand(1+4*ell*(ell+1)-(2*ell+1)**2)==0,'two sphere exact adjacent ladder identity')

report={'experiment':'bending-clock','created_utc':datetime.now(timezone.utc).isoformat(),
        'status':'passed','checks':sum(counts.values()),'check_counts':dict(counts),
        'scope':'Positive local coupled displacement/rotation inverse action. Equal characteristic speeds give a constant branch gap; actual IDWT carrier, vertex, spatial selection and confinement are not derived.',
        'numerical_cases':numerical,'unmatched_speed_controls':unmatched,
        'spatial_tensor_cases':tensor,'particle_cases':assigned,
        'speed_sensitivity_cases':speed_sensitivity,
        'first_order_clifford_controls':clifford_controls,
        'sources':{'experiments/bending-clock-checks.py':sha256(Path(__file__).read_bytes()).hexdigest(),
                   'results/complex-comb-response-results.json':sha256(baseline_path.read_bytes()).hexdigest()}}
out=ROOT/'bending-clock-results.json'
out.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'checks':report['checks'],'assigned_cases':len(assigned),
                  'full_spatial_cases':len(tensor),'report':str(out)}))
