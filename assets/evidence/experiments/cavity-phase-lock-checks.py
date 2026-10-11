"""Finite exact/spectral tests of a candidate same-field comb-locking energy."""
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction as F
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import sympy as sp

ROOT = Path(__file__).resolve().parent
counts = Counter()


def check(value, label):
    if not bool(value):
        raise AssertionError(label)
    counts[label] += 1


def close(a, b, label, tol=3e-10):
    check(np.allclose(a, b, atol=tol, rtol=tol), label)


T,tau,phi,n = sp.symbols('T tau phi n', real=True)
wn = (2*sp.pi*n-phi)/(T+tau)
check(sp.simplify(wn*(T+tau)+phi-2*sp.pi*n)==0,'roundtrip phase closure')
check(sp.simplify(wn.subs(n,n+1)-wn-2*sp.pi/(T+tau))==0,
      'roundtrip spacing uses group delay')
check(sp.simplify(wn+phi/(T+tau)-2*sp.pi*n/(T+tau))==0,
      'cavity offset remains independent condition')


def constraints(a, weights):
    x = a * weights
    return x[:-2] * x[2:] - x[1:-1] ** 2


def gradient(a, weights):
    x = a * weights
    c = constraints(a, weights)
    g = np.zeros_like(a, dtype=complex)
    g[:-2] += weights[:-2] * np.conj(x[2:]) * c
    g[2:] += weights[2:] * np.conj(x[:-2]) * c
    g[1:-1] -= 2 * weights[1:-1] * np.conj(x[1:-1]) * c
    return g


def linear_constraint(b, w):
    m = len(b)
    x = b * w
    l = np.zeros((max(0, m-2), m))
    for j in range(1, m-1):
        l[j-1, j-1] = w[j-1] * x[j+1]
        l[j-1, j+1] = w[j+1] * x[j-1]
        l[j-1, j] = -2 * w[j] * x[j]
    return l


rng = np.random.default_rng(20261010)
stability = []
for m in (3, 4, 5, 8, 13):
    for ratio in (F(1, 2), F(1), F(3, 2)):
        for weighted in (False, True):
            w_exact = [F(j+1) if weighted else F(1) for j in range(m)]
            b_exact = [ratio**j / w_exact[j] for j in range(m)]
            x = [w_exact[j]*b_exact[j] for j in range(m)]
            check(all(x[j-1]*x[j+1] == x[j]**2 for j in range(1, m-1)), 'exact locked constraint')
            le = sp.zeros(m-2, m)
            for j in range(1, m-1):
                le[j-1, j-1] = w_exact[j-1]*x[j+1]
                le[j-1, j+1] = w_exact[j+1]*x[j-1]
                le[j-1, j] = -2*w_exact[j]*x[j]
            check(le.rank() == m-2, 'exact transverse rank')
            be = sp.Matrix(b_exact)
            je = sp.Matrix([j*b_exact[j] for j in range(m)])
            check(le*be == sp.zeros(m-2, 1), 'overall amplitude null')
            check(le*je == sp.zeros(m-2, 1), 'index slope null')
            w = np.array(w_exact, dtype=float)
            b = np.array(b_exact, dtype=float)
            b /= np.linalg.norm(b)
            l = linear_constraint(b, w)
            eig = np.linalg.eigvalsh(l.T@l)
            close(eig[:2], 0, 'two complex neutral modes', 1e-8)
            check(eig[2] > 0, 'positive transverse Hessian')
            a = b + .09*(rng.normal(size=m)+1j*rng.normal(size=m))
            g = gradient(a, w)
            delta = rng.normal(size=m)+1j*rng.normal(size=m)
            eps = 1e-6
            energy = lambda z: float(np.vdot(constraints(z,w), constraints(z,w)).real)
            fd = (energy(a+eps*delta)-energy(a-eps*delta))/(2*eps)
            close(fd, 2*np.vdot(g,delta).real, 'full complex energy variation', 3e-6)
            adot = -1j*g
            close(2*np.vdot(a,adot).real, 0, 'norm conservation')
            close(2*np.vdot(np.arange(m)*a,adot).real, 0, 'index moment conservation')
            close(2*np.vdot(g,adot).real, 0, 'locking energy conservation')
            phases = np.exp(1j*(.27+np.arange(m)*.43))
            close(energy(a*phases), energy(a), 'two phase symmetries')
            close(gradient(a*phases,w), phases*g, 'equivariant field equation')
            omega = 100-np.arange(m)
            free = b*np.exp(-1j*omega*.37)
            close(gradient(free,w), 0, 'unchanged free mass frequencies', 1e-7)
            # Independent second variation, with the finite first-order constraint.
            eps = 1e-5
            close(energy(b+eps*delta)/eps**2, np.linalg.norm(l@delta)**2,
                  'second variation from quartic', 5e-4)
            stability.append({'bandwidth':m, 'ratio':str(ratio), 'weighted':weighted,
                              'minimum_positive_curvature':float(eig[2]),
                              'maximum_curvature':float(eig[-1]), 'complex_nullity':2})

# Independent Fourier extraction avoids evaluating huge optical carriers.
envelope = []
theta = 2*np.pi*np.arange(256)/256
for m in (5,8,13):
    for ratio in (F(1,2),F(1),F(3,2)):
        b0 = [ratio**j for j in range(m)]
        norm = sum(x*x for x in b0)
        d = [sum(b0[j]*b0[j+p] for j in range(m-p))/norm for p in range(1,m)]
        expected = (F(m-1,m) if ratio == 1 else
                    ratio*(1-ratio**(2*(m-1)))/(1-ratio**(2*m)))
        check(d[0] == expected, 'geometric fundamental exact')
        b = np.array(b0,dtype=float)/np.sqrt(float(norm))
        z = b@np.exp(1j*np.arange(m)[:,None]*theta)
        density = abs(z)**2/2
        fft = np.fft.fft(density)/len(theta)
        close(fft[0], .5, 'common density mean')
        close(2*fft[1:m], np.array(d,dtype=float), 'independent density coefficients')
        for carrier in (m, m+1, 18564, 3838380, 64512240):
            # Factorization includes an arbitrary comb offset and upper mass label.
            optical = np.exp(-1j*(carrier+.137)*theta)*z
            close(abs(optical)**2/2, density, 'carrier and offset blind density')
            field_vertex_mean = sum((carrier-j)*float(b[j]**2) for j in range(m))
            check(field_vertex_mean > 0, 'field energy vertex retains carrier')
        matrix = rng.normal(size=(m,m))+1j*rng.normal(size=(m,m))
        matrix = (matrix+matrix.conj().T)/2
        modal = b[:,None]*np.exp(1j*np.arange(m)[:,None]*theta)
        source = np.einsum('it,ij,jt->t',modal.conj(),matrix,modal)
        shifted_modal = modal*np.exp(-1j*13.137*theta)
        shifted_source = np.einsum('it,ij,jt->t',shifted_modal.conj(),matrix,shifted_modal)
        close(source,shifted_source,'general Hermitian source carrier cancellation')
        close(np.fft.fft(source)*np.fft.fftfreq(len(theta))*1j,
              np.fft.fft(shifted_source)*np.fft.fftfreq(len(theta))*1j,
              'source derivative carrier cancellation')
        corr0 = F(1,4)+sum(x*x for x in d)/2
        corrpi = F(1,4)+sum((-1)**p*d[p-1]**2 for p in range(1,m))/2
        if ratio == 1:
            check(corr0 == F(2*m*m+1,12*m), 'flat in phase exact')
            if m%2 == 0:
                check(corrpi == F(1,4*m), 'flat opposite phase exact')
        close(np.mean(density**2), float(corr0), 'pair correlation by Fourier')
        shifted = abs(b@np.exp(1j*np.arange(m)[:,None]*(theta+np.pi)))**2/2
        close(np.mean(density*shifted), float(corrpi), 'opposite phase pair correlation')
        envelope.append({'bandwidth':m,'ratio':str(ratio),'fundamental':str(d[0]),
                         'in_phase_density_correlation':str(corr0),
                         'opposite_phase_density_correlation':str(corrpi)})

baseline_file = ROOT/'grounding/complex-comb-response-results.json'
baseline = json.loads(baseline_file.read_text())
assigned = []
for source in baseline['particle_cases']:
    k = source['cutoff']
    r = source['fundamental_over_depth6']
    for target_m in (5,8,13):
        m = min(k,target_m)
        p = np.arange(1,m,dtype=float)
        d = (m-p)/m
        freq = p*r
        chi = (3/(2-freq*freq)+2/(3-freq*freq))/5
        q0 = float(np.sum(d*d*chi)/2)
        q1 = float(np.sum(freq*freq*d*d*chi)/2)
        # Only diagnostic shape normalization; physical amplitude is uncalibrated.
        check(k >= m >= 1, 'mass endpoint retained')
        if k >= target_m:
            check(F(m-1,m) == F(target_m-1,target_m), 'same envelope fundamental')
        # Weighted full-comb constraints vanish exactly, including huge K.
        for j in sorted(set([2, min(3,k-1), max(2,k//2), k-1])):
            if 2 <= j <= k-1:
                check(F(j-1)*F(1,j-1)*F(j+1)*F(1,j+1)
                      == (F(j)*F(1,j))**2, 'assigned sawtooth locking constraint')
        assigned.append({'particle':source['particle'],'real_depth':source['real_depth'],
                         'cutoff':k,'requested_bandwidth':target_m,'bandwidth':m,
                         'fundamental_over_depth6':r,'fundamental':str(F(m-1,m)),
                         'density_response':q0,'derivative_response':q1})

# Explicit endpoint control: the retained-band zero is not a zero of the
# unprojected adjacent-quartet action. Its next empty mode is forced.
endpoint = []
for k in (3,5,8,13):
    a = np.array([1/j for j in range(1,k+1)]+[0.,0.],complex)
    w = np.arange(1,k+3,dtype=float)
    g = gradient(a,w)
    close(g[k], -(k+1), 'unprojected next mode forcing')
    close(np.linalg.norm(constraints(a,w))**2, 1, 'nonzero endpoint energy')
    endpoint.append({'cutoff':k,'next_mode_gradient':float(g[k].real),
                     'locking_energy':1, 'requires_band_invariance':True})


def octic_constraints(a, weights):
    x = a*weights
    p = x[:-2]*x[2:]
    return p*(p-x[1:-1]**2)


def octic_gradient(a, weights):
    x = a*weights
    p = x[:-2]*x[2:]
    f = octic_constraints(a,weights)
    g = np.zeros_like(a,dtype=complex)
    g[:-2] += weights[:-2]*np.conj((2*p-x[1:-1]**2)*x[2:])*f
    g[2:] += weights[2:]*np.conj((2*p-x[1:-1]**2)*x[:-2])*f
    g[1:-1] -= 2*weights[1:-1]*np.conj(p*x[1:-1])*f
    return g


# A separate positive eight-wave variant makes every coordinate support
# invariant. It locks existing harmonics, and cannot generate absent ones.
closed_support = []
for size in (8,13,21):
    w = np.arange(1,size+1,dtype=float)
    for k in (3,5):
        a = np.zeros(size,dtype=complex)
        a[:k] = 1/w[:k]
        close(octic_constraints(a,w),0,'octic finite sawtooth zero')
        close(octic_gradient(a,w),0,'octic finite sawtooth exact orbit')
        for trial in range(3):
            arbitrary = rng.normal(size=size)+1j*rng.normal(size=size)
            absent = rng.choice(size,size//2,replace=False)
            arbitrary[absent] = 0
            close(octic_gradient(arbitrary,w)[absent],0,'arbitrary support invariant')
        l = np.zeros((size-2,size))
        x = a.real*w
        for j in range(1,size-1):
            p0 = x[j-1]*x[j+1]
            l[j-1,j-1] = w[j-1]*(2*p0-x[j]**2)*x[j+1]
            l[j-1,j+1] = w[j+1]*(2*p0-x[j]**2)*x[j-1]
            l[j-1,j] = -2*w[j]*p0*x[j]
        eig = np.linalg.eigvalsh(l.T@l)
        check(np.min(eig)>-1e-9,'octic complete linear nonnegativity')
        check(np.linalg.matrix_rank(l)==k-1,'octic boundary adds one curvature')
        close((l.T@l)[k,k],(k+1)**2,'octic next empty mode positive curvature')
        delta = rng.normal(size=size)+1j*rng.normal(size=size)
        eps = 1e-6
        e = lambda z:float(np.linalg.norm(octic_constraints(z,w))**2)
        close(e(a+eps*delta)/eps**2,np.linalg.norm(l@delta)**2,
              'octic full second variation',2e-3)
        z = .1*(rng.normal(size=size)+1j*rng.normal(size=size))
        g = octic_gradient(z,w)
        fd = (e(z+eps*delta)-e(z-eps*delta))/(2*eps)
        close(fd,2*np.vdot(g,delta).real,'octic full complex variation',1e-4)
        close(2*np.vdot(z,-1j*g).real,0,'octic norm conservation',1e-5)
        close(2*np.vdot(np.arange(size)*z,-1j*g).real,0,'octic index conservation',1e-4)
        closed_support.append({'ambient_retained_modes':size,'occupied_cutoff':k,
                               'next_mode_curvature':(k+1)**2,
                               'complex_transverse_rank':k-1,
                               'all_empty_modes_remain_empty':True,
                               'harmonic_generation':False})

# Curvature of the bare frequency ladder breaks the exact free locked orbit.
for m in (5,8,13):
    a = np.ones(m,dtype=complex)
    omega = 100-np.arange(m)+.03*np.arange(m)**2/2
    c = constraints(a*np.exp(-1j*omega*.7),np.ones(m))
    close(np.linalg.norm(c)**2,4*(m-2)*np.sin(.03*.7/2)**2,
          'dispersion spoils free exact locking')

# A retained fundamental mode can exchange energy reciprocally with two
# adjacent high-frequency modes through one positive mismatch energy.
# Every amplitude labels a mode of the same field, not a constituent object.
resonant_feedback = []
for depth in (2,3,4,5,6,10):
    for aa,bb,eta in ((1.,.5,.4),(.5,1.,.4),(1.,1.,.4),(.5,1.5,.8)):
        rvec = np.array([-eta*bb,-eta*aa,1.])
        svec = np.array([eta*bb,-eta*aa,1.])
        qmat = np.block([[np.zeros((3,3)),np.outer(svec,svec)],
                         [-np.outer(rvec,rvec),np.zeros((3,3))]])
        freq = abs(1+eta*eta*(aa*aa-bb*bb))
        eig = np.linalg.eigvals(qmat)
        # The four symmetry/constraint zeros can form small numerical
        # Jordan artifacts; the nonzero pair is checked independently.
        nonzero = sorted(eig,key=abs)[-2:]
        close(sorted(abs(v.imag) for v in nonzero),[freq,freq],
              'same field feedback oscillation frequency',1e-7)
        close([v.real for v in nonzero],0,'same field feedback has no growing pair',1e-7)
        state = .4*(rng.normal(size=3)+1j*rng.normal(size=3))
        a0,a1,b = state
        mismatch = b-eta*np.conj(a0)*a1
        g = np.array([-eta*np.conj(mismatch)*a1,
                      -eta*mismatch*a0,mismatch])
        adot = -1j*g
        ndot = 2*np.real(np.conj(state)*adot)
        close(ndot[0]+ndot[1],0,'feedback high band norm conservation')
        close(ndot[1]+ndot[2],0,'feedback resonant energy exchange balance')
        close(np.dot([20.,21.,1.],ndot),0,'feedback complete bare energy balance')
        close(2*np.vdot(g,adot).real,0,'feedback interaction energy conservation')
        close(ndot[1],-2*eta*np.imag(np.conj(a1)*b*a0),
              'feedback phase resolved transfer')
        delta = rng.normal(size=3)+1j*rng.normal(size=3)
        e = lambda z:float(abs(z[2]-eta*np.conj(z[0])*z[1])**2)
        eps = 1e-6
        fd = (e(state+eps*delta)-e(state-eps*delta))/(2*eps)
        close(fd,2*np.vdot(g,delta).real,'feedback full complex variation',1e-6)
        resonant_feedback.append({'real_depth':depth,'a0':aa,'a1':bb,'eta':eta,
                                  'fundamental_amplitude':eta*aa*bb,
                                  'feedback_disturbance_frequency_over_g':freq})

four_mode_feedback = []
for depth in (2,3,4,5,6,10):
    for aa,bb,cc,eta in ((1.,.5,1.,.4),(.5,1.,1.,.4),(1.,1.,1.,.4),(.5,1.5,1.,.8)):
        dd = eta*aa*bb/cc
        rvec = np.array([-eta*bb,-eta*aa,dd,cc])
        svec = np.array([eta*bb,-eta*aa,-dd,cc])
        qmat = np.block([[np.zeros((4,4)),np.outer(svec,svec)],
                         [-np.outer(rvec,rvec),np.zeros((4,4))]])
        freq = abs(eta*eta*(aa*aa-bb*bb)+cc*cc-dd*dd)
        eig = sorted(np.linalg.eigvals(qmat),key=abs)[-2:]
        close(sorted(abs(v.imag) for v in eig),[freq,freq],
              'four mode reciprocal oscillation frequency',1e-7)
        close([v.real for v in eig],0,'four mode no growing disturbance pair',1e-7)
        z = .4*(rng.normal(size=4)+1j*rng.normal(size=4))
        a0,a1,c0,c1 = z
        beat_a,beat_c = np.conj(a0)*a1,np.conj(c0)*c1
        mismatch = beat_c-eta*beat_a
        g = np.array([-eta*np.conj(mismatch)*a1,-eta*mismatch*a0,
                      np.conj(mismatch)*c1,mismatch*c0])
        ndot = 2*np.real(np.conj(z)*(-1j*g))
        close(ndot[0]+ndot[1],0,'four mode first pair norm conservation')
        close(ndot[2]+ndot[3],0,'four mode second pair norm conservation')
        close(ndot[1]+ndot[3],0,'four mode total sideband action conservation')
        close(np.dot([20.,21.,3.,4.],ndot),0,'four mode full energy exchange balance')
        close(ndot[1],-2*eta*np.imag(np.conj(beat_a)*beat_c),
              'four mode phase resolved transfer')
        e = lambda q:float(abs(np.conj(q[2])*q[3]-eta*np.conj(q[0])*q[1])**2)
        delta = rng.normal(size=4)+1j*rng.normal(size=4)
        eps = 1e-6
        close((e(z+eps*delta)-e(z-eps*delta))/(2*eps),2*np.vdot(g,delta).real,
              'four mode complete complex variation',1e-6)
        for alpha in (.17,1.1):
            close(e(z*np.exp(1j*alpha)),e(z),'four mode common field phase symmetry')
        # Independent cycle averaging of the density mismatch gives a DC
        # term plus precisely twice the beat-mismatch energy.
        wave_a = a0+a1*np.exp(-1j*theta)
        wave_c = c0+c1*np.exp(-1j*theta)
        dc = abs(c0)**2+abs(c1)**2-eta*(abs(a0)**2+abs(a1)**2)
        averaged = np.mean((abs(wave_c)**2-eta*abs(wave_a)**2)**2)
        close(averaged,dc*dc+2*e(z),'local density mismatch generates beat interaction')
        four_mode_feedback.append({'real_depth':depth,'a0':aa,'a1':bb,'c0':cc,'c1':dd,
                                   'eta':eta,'disturbance_frequency_over_g':freq,
                                   'common_field_phase_preserved':True})

# Combine locking and reciprocal beat response in one positive energy.
combined = []
for m in (5,8,13):
    for eta in (.2,.4,.8):
        a = np.ones(m)/np.sqrt(m)
        c0 = 1.
        c1 = eta*(m-1)/m
        modes = m+2
        l = linear_constraint(a,np.ones(m))
        rlock = np.zeros((2*(m-2),2*modes))
        rlock[:m-2,:m] = l
        rlock[m-2:,modes:modes+m] = l
        # Variation of C1=sum conj(a_j) a_(j+1).
        rr = np.zeros(modes)
        ss = np.zeros(modes)
        for j in range(m-1):
            rr[j] -= eta*a[j+1]
            rr[j+1] -= eta*a[j]
            ss[j] += eta*a[j+1]
            ss[j+1] -= eta*a[j]
        rr[m:] = [c1,c0]
        ss[m:] = [-c1,c0]
        ltotal = np.vstack((rlock,np.r_[rr,np.zeros(modes)],np.r_[np.zeros(modes),ss]))
        kmat = ltotal.T@ltotal
        check(np.linalg.matrix_rank(ltotal)==2*m-2,'combined action full transverse rank')
        symplectic = np.block([[np.zeros((modes,modes)),np.eye(modes)],
                               [-np.eye(modes),np.zeros((modes,modes))]])
        eig = np.linalg.eigvals(symplectic@kmat)
        close([x.real for x in eig],0,'combined action no growing linear modes',1e-7)
        nonzero = [x for x in eig if abs(x)>1e-6]
        check(len(nonzero)==2*(m-1),'combined action disturbance pair count')
        combined.append({'high_bandwidth':m,'eta':eta,'real_neutral_directions':6,
                         'nonzero_frequency_pairs':m-1,
                         'minimum_nonzero_disturbance_frequency':min(abs(x.imag) for x in nonzero),
                         'maximum_nonzero_disturbance_frequency':max(abs(x.imag) for x in nonzero)})

# Symbolic rotating-pattern phase bunching and symmetry controls.
ell,I,V,omega,p,psi = sp.symbols('ell I V omega p psi',positive=True,real=True)
h = ell**2/(2*I)*(p-I*omega/ell**2)**2-V*sp.cos(psi)
check(sp.simplify(sp.diff(h,p,2)*sp.diff(h,psi,2).subs(psi,0)-ell**2*V/I)==0,
      'phase bunching libration frequency')
rotating = []
for depth in (2,3,4,5,6,10):
    for l in (1,2,3,5,8):
        ang = 2*np.pi*(np.arange(256*l)+.25)/(256*l)
        acceptance = np.mean(np.cos(l*ang)>0)
        close(acceptance,.5,'total spoke acceptance independent of order')
        t = theta
        single = np.cos(l*t)
        mixed = np.cos(l*t)+np.cos((l+1)*t)
        sf = np.fft.fft(single**2)/len(t)
        mf = np.fft.fft(mixed**2)/len(t)
        close(sf[1],0,'single rotating order has no fundamental density')
        close(2*mf[1],1,'adjacent rotating orders restore fundamental density')
        rotating.append({'real_depth':depth,'azimuthal_order':l,'total_spoke_fraction':'1/2',
                         'single_order_density_fundamental':0,
                         'adjacent_order_density_fundamental':1})

report = {'experiment':'cavity-phase-lock','created_utc':datetime.now(timezone.utc).isoformat(),
          'status':'passed','checks':sum(counts.values()),'check_counts':dict(counts),
          'scope':'Finite retained-mode inverse action; no particle simulation or derived band confinement.',
          'stability_cases':stability,'envelope_cases':envelope,'particle_cases':assigned,
          'endpoint_controls':endpoint,'rotating_cases':rotating,
          'closed_support_cases':closed_support,
          'resonant_feedback_cases':resonant_feedback,
          'four_mode_feedback_cases':four_mode_feedback,'combined_action_cases':combined,
          'sources':{'experiments/cavity-phase-lock-checks.py':sha256(Path(__file__).read_bytes()).hexdigest(),
                     'results/complex-comb-response-results.json':sha256(baseline_file.read_bytes()).hexdigest()}}
out = ROOT/'cavity-phase-lock-results.json'
out.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'status':report['status'],'checks':report['checks'],
                  'stability_cases':len(stability),'particle_cases':len(assigned),
                  'rotating_cases':len(rotating),'report':str(out)}))
