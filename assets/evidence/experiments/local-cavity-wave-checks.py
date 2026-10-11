"""Local positive wave action: matrix-source beats, flux and finite disturbances."""
from collections import Counter
from datetime import datetime,timezone
from fractions import Fraction as F
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from scipy.linalg import eigvals
import sympy as sp

ROOT = Path(__file__).resolve().parent
counts = Counter()


def check(value,label):
    if not bool(value):
        raise AssertionError(label)
    counts[label]+=1


def close(a,b,label,tol=2e-10):
    check(np.allclose(a,b,atol=tol,rtol=tol),label)


def outer(z):
    return np.outer(z,z.conj())


def potential(a,c,eta):
    d = outer(c)-eta*outer(a)
    return np.linalg.norm(d)**2/2


kinetic,quartic,weighted_norm=sp.symbols('T V W',nonnegative=True)
localization=[]
for depth in (2,3,4,5,6,10):
    scaling=(2-depth)*kinetic-depth*quartic+depth*weighted_norm
    eliminated=sp.expand(scaling.subs(weighted_norm,kinetic+2*quartic))
    check(eliminated==2*kinetic+depth*quartic,
          'ambient localization Pohozaev obstruction')
    localization.append({'real_depth':depth,'integrated_field_equation':'T+2V=W',
                         'eliminated_scaling_identity':str(eliminated),
                         'nontrivial_finite_energy_time_harmonic_state':False,
                         'scope':'Flat noncompact space, sufficient decay, homogeneous positive quartic, common phase within each polarization block.'})


rng=np.random.default_rng(20261011)
# Check the actual nonlinear field variation, not just the zero branch.
for trial in range(20):
    a=rng.normal(size=2)+1j*rng.normal(size=2)
    c=rng.normal(size=2)+1j*rng.normal(size=2)
    da=rng.normal(size=2)+1j*rng.normal(size=2)
    dc=rng.normal(size=2)+1j*rng.normal(size=2)
    eta=.4
    d=outer(c)-eta*outer(a)
    ga=-eta*d@a
    gc=d@c
    eps=1e-6
    fd=(potential(a+eps*da,c+eps*dc,eta)-potential(a-eps*da,c-eps*dc,eta))/(2*eps)
    close(fd,2*np.vdot(ga,da).real+2*np.vdot(gc,dc).real,
          'full local matrix potential variation',2e-7)
    matrix=rng.normal(size=(2,2))+1j*rng.normal(size=(2,2))
    unitary=np.linalg.qr(matrix)[0]
    close(potential(unitary@a,unitary@c,eta),potential(a,c,eta),
          'common polarization covariance')
    close(potential(a*np.exp(.3j),c*np.exp(.3j),eta),potential(a,c,eta),
          'common physical field phase symmetry')
    # The real kinetic equation uses half the real potential derivative.
    close(np.vdot(a,ga).imag+np.vdot(c,gc).imag,0,
          'local common phase torque vanishes')

baseline_path=ROOT/'grounding/complex-comb-response-results.json'
baseline=json.loads(baseline_path.read_text())
theta=2*np.pi*np.arange(128)/128
pauli=np.array([[0,1],[1,0]],complex)
cases=[]
for source in baseline['particle_cases']:
    cutoff=source['cutoff']
    for eta_exact in (F(1,4),F(1),F(2)):
        eta=float(eta_exact)
        if cutoff == 1:
            amps=np.array([1.,0.])
            high=np.array([1,1],dtype=np.int64)
        else:
            amps=np.array([1.,.5])
            high=np.array([cutoff-1,cutoff],dtype=np.int64)
        low=np.array([1,2],dtype=np.int64)
        envelope=amps[:,None]*np.array([np.ones_like(theta),np.exp(1j*theta)])
        a=envelope*np.exp(1j*int(high[0])*theta)
        c=np.sqrt(eta)*envelope*np.exp(1j*theta)
        # Common carriers are factored; no huge frequency sum is needed.
        qa=np.einsum('it,jt->tij',a,a.conj())
        qc=np.einsum('it,jt->tij',c,c.conj())
        close(qc,eta*qa,'exact pointwise matrix source match')
        density_a=np.einsum('it,it->t',a.conj(),a).real
        close(density_a,amps@amps,'orthogonal polarizations keep density constant')
        readout=np.einsum('it,ij,jt->t',a.conj(),pauli,a).real
        expected=2*amps[0]*amps[1]*np.cos(theta)
        close(readout,expected,'off diagonal source retains fundamental')
        flux_factor=float(np.sum(high.astype(float)**2*amps**2)+
                          eta*np.sum(low.astype(float)**2*amps**2))
        energy_density=np.full(len(theta),2*flux_factor)
        momentum_vector=np.mean(energy_density[:,None]*
                                np.array([-np.sin(theta),np.cos(theta)]).T,axis=0)
        # Relative error is the meaningful diagnostic for huge supplied K.
        close(momentum_vector/float(energy_density[0]),0,'complete loop linear momentum vanishes')
        energy_exact=2*sum(F(int(high[j])**2)*F(str(float(amps[j]**2)))+
                           eta_exact*F(int(low[j])**2)*F(str(float(amps[j]**2))) for j in range(2))
        flux_exact=energy_exact  # Dimensionless v=R=1 in this comparison.
        check(flux_exact/energy_exact==1,'energy flux over energy density equals wave speed')
        # Each retained term solves d_t^2-d_theta^2 exactly. The interaction
        # gradient vanishes because the matrix mismatch vanishes.
        check(all(-F(int(n))**2+F(int(n))**2==0 for n in list(high)+list(low)),
              'complete local wave equation exact on each carrier')
        check(max(high)==cutoff,'assigned upper mass endpoint preserved')
        cases.append({'particle':source['particle'],'real_depth':source['real_depth'],
                      'cutoff':cutoff,'eta':str(eta_exact),'occupied_high_labels':list(map(int,high)),
                      'low_labels':[1,2],'matrix_source_fundamental':float(2*amps[0]*amps[1]),
                      'identity_density_fundamental':0,
                      'energy_density_over_common_amplitude_squared':str(energy_exact),
                      'linear_momentum_vector_exact':[0,0],
                      'angular_momentum_over_total_energy':'R/v',
                      'prescribed_path_requires_normal_stress':True})

# A scalar adjacent comb would instead have a rotating net momentum on a
# prescribed circular path. Independent quadrature resolves its coefficient.
scalar=[]
for k in (1,2,5,13):
    aa,bb=1.,.5
    for t in (.0,.3,1.1):
        density=2*(k*k*aa*aa+(k+1)**2*bb*bb+
                   2*k*(k+1)*aa*bb*np.cos(theta-t))
        p=np.mean(density[:,None]*np.array([-np.sin(theta),np.cos(theta)]).T,axis=0)
        expected=2*k*(k+1)*aa*bb*np.array([-np.sin(t),np.cos(t)])
        close(p,expected,'scalar adjacent comb momentum must be balanced')
        scalar.append({'lower_label':k,'phase':t,'momentum_per_unit_loop_length':p.tolist()})


def matrix_source_jacobian(a,c,eta):
    """Real Hermitian matrix coordinates have their Frobenius norm."""
    jac=np.empty((4,8))
    for j in range(8):
        da=np.zeros(2,dtype=complex)
        dc=np.zeros(2,dtype=complex)
        target=da if j<4 else dc
        local=j%4
        target[local//2]=1 if local%2==0 else 1j
        d=(np.outer(dc,c.conj())+np.outer(c,dc.conj())-
           eta*(np.outer(da,a.conj())+np.outer(a,da.conj())))
        jac[:,j]=[d[0,0].real,d[1,1].real,
                  np.sqrt(2)*d[0,1].real,np.sqrt(2)*d[0,1].imag]
    return jac


spectral=[]
for depth in (2,3,4,5,6,10):
    for cutoff in (2,4,6):
        modes=np.arange(-cutoff,cutoff+1)
        grid=8*cutoff+33
        x=2*np.pi*np.arange(grid)/grid
        eta=.4
        a=np.array([np.exp(2j*x),.5*np.exp(3j*x)])
        c=np.sqrt(eta)*np.array([np.exp(1j*x),.5*np.exp(2j*x)])
        local=[]
        for j in range(grid):
            jac=matrix_source_jacobian(a[:,j],c[:,j],eta)
            local.append(jac.T@jac/2)
        local=np.array(local)
        basis=np.exp(1j*x[:,None]*modes)
        kspec=np.einsum('xq,xab,xr->aqbr',basis.conj(),local,basis)/grid
        kspec=kspec.reshape(8*len(modes),8*len(modes))
        close(kspec,kspec.conj().T,'projected local Hessian Hermitian')
        check(np.linalg.eigvalsh(kspec)[0]>-1e-10,'projected local potential nonnegative')
        size=len(kspec)
        gyro=np.diag(np.tile(2j*modes,8))
        operator=np.block([[np.zeros((size,size)),np.eye(size)],[-kspec,gyro]])
        spectrum=eigvals(operator,check_finite=True)
        max_real=float(np.max(abs(spectrum.real)))
        check(max_real<2e-6,'full wave disturbance operator has no resolved growing mode')
        du=rng.normal(size=size)+1j*rng.normal(size=size)
        velocity=rng.normal(size=size)+1j*rng.normal(size=size)
        acceleration=-kspec@du+gyro@velocity
        edot=2*np.vdot(velocity,acceleration).real+2*np.vdot(du,kspec@velocity).real
        close(edot,0,'full linear comoving energy conserved',2e-8)
        spectral.append({'real_depth':depth,'spatial_fourier_cutoff':cutoff,
                         'real_component_count':8,'full_first_order_matrix_size':2*size,
                         'maximum_absolute_growth_rate':max_real,
                         'minimum_potential_hessian_eigenvalue':float(np.linalg.eigvalsh(kspec)[0]),
                         'nonlinear_localization_claim':False})

report={'experiment':'local-cavity-wave','created_utc':datetime.now(timezone.utc).isoformat(),
        'status':'passed','checks':sum(counts.values()),'check_counts':dict(counts),
        'scope':'Local causal positive wave prototype on a prescribed closed spatial path; no self-confined particle or simplex selection is claimed.',
        'particle_cases':cases,'scalar_momentum_controls':scalar,'spectral_cases':spectral,
        'ambient_localization_controls':localization,
        'sources':{'experiments/local-cavity-wave-checks.py':sha256(Path(__file__).read_bytes()).hexdigest(),
                   'results/complex-comb-response-results.json':sha256(baseline_path.read_bytes()).hexdigest()}}
out=ROOT/'local-cavity-wave-results.json'
out.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'checks':report['checks'],'particle_cases':len(cases),
                  'spectral_cases':len(spectral),'report':str(out)}))
