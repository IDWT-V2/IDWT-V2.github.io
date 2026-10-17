"""One-off exact and numerical branch calculation for real same-field clocks.

Positions remain ordinary R10. All small matrices below act on real field
components or explicitly conditional collective coordinates. Numerical runs
are reduced ODE controls, not simulations of identified particles.
"""
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import itertools
import json
import math
from pathlib import Path
import subprocess
import time

import numpy as np
import scipy
from scipy.integrate import solve_ivp
import sympy as sp

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
START = time.monotonic()
CHECKS = {}
SECTIONS = []
DATA = {}
SNAPSHOT = 1187


def check(group, condition):
    if not bool(condition):
        raise AssertionError(group)
    CHECKS[group] = CHECKS.get(group, 0) + 1


def zero(group, expr):
    if isinstance(expr, sp.MatrixBase):
        check(group, all(sp.simplify(x) == 0 for x in expr))
    else:
        check(group, sp.simplify(expr) == 0)


def section(title, equations, result, assumptions):
    SECTIONS.append(dict(title=title, equations=equations, result=result,
                         assumptions=assumptions))
    print(title, flush=True)


def tensor(factors):
    out = np.array([[1]])
    for factor in factors:
        out = np.kron(out, factor)
    return out


I2 = np.eye(2, dtype=np.int64)
X = np.array([[0, 1], [1, 0]], dtype=np.int64)
Z = np.diag([1, -1])
J2 = np.array([[0, -1], [1, 0]], dtype=np.int64)
UNITS = [np.kron(J2, I2), np.kron(Z, J2), np.kron(X, J2)]
J1, J2H, J3 = [sp.Matrix(m) for m in UNITS]
E4 = sp.eye(4)

# 1. Minimal modules and all small multiplicity branches.
types = {0: 'R', 1: 'R', 2: 'C', 3: 'H', 4: 'H', 5: 'H', 6: 'C', 7: 'R'}
dimensions = {1: 1, 2: 2, 3: 4, 4: 4, 5: 8, 6: 8, 7: 8,
              8: 8, 9: 16, 10: 32, 11: 64, 12: 64}
minimal = []
copies = []
for d in range(1, 13):
    f = types[d % 8]
    minimal.append(dict(depth=d, real_module_dimension=dimensions[d], type=f,
                        skew_dimension={'R': 0, 'C': 1, 'H': 3}[f]))
    for m in range(1, 9):
        skew_dim = {'R': m*(m-1)//2, 'C': m*m, 'H': m*(2*m+1)}[f]
        if f == 'R':
            b = sp.zeros(m)
            for k in range(0, m-1, 2):
                b[k, k+1], b[k+1, k] = -1, 1
            expected_rank = m-m % 2
        elif f == 'C':
            b = sp.kronecker_product(sp.eye(m), sp.Matrix(J2))
            expected_rank = 2*m
        else:
            b = sp.kronecker_product(sp.eye(m), J1)
            expected_rank = 4*m
        check('multiplicity', b.rank() == expected_rank)
        zero('multiplicity', b+b.T)
        copies.append(dict(depth=d, type=f, multiplicity=m, skew_dimension=skew_dim,
                           nondegenerate=(f != 'R' or m % 2 == 0)))
check('minimal', [r['depth'] for r in minimal if r['depth'] <= 10 and r['skew_dimension']]
      == [2, 3, 4, 5, 6, 10])
DATA['minimal'] = minimal
DATA['multiplicities'] = copies
section('1. Minimal versus repeated real carriers',
        ['End_G(V)=F in {R,C,H}; dim(skew F)=(0,1,3).',
         'End_G(V^m)=M_m(F); dim(skew)=(m(m-1)/2,m^2,m(2m+1)).',
         'For F=R, a nondegenerate invariant alternating form exists iff m is even.'],
        'The minimal rule selects the occupied depths; every real-type depth regains a clock with two copies. Odd real multiplicity has a null direction even when skew generators exist.',
        'One irreducible real spin module, or m identical copies. Spatial mode count is not automatically representation multiplicity.')

# 2. Recompute full ambient restrictions exactly, including negative controls.
Y = 1j*J2
gamma = [tensor([Z]*k+[middle]+[I2]*(4-k)) for k in range(5) for middle in (X, Y)]
eye32 = np.eye(32, dtype=np.int64)
for i, a in enumerate(gamma):
    for j, b in enumerate(gamma):
        check('clifford', np.array_equal(a@b+b@a, (2 if i == j else 0)*eye32))
positive = [i for i in range(32) if i.bit_count() % 2 == 0]
generators = []
for i in range(9):
    a = (gamma[i]@gamma[i+1])[np.ix_(positive, positive)]
    g = np.block([[a.real, -a.imag], [a.imag, a.real]]).astype(np.int64)
    check('clifford', np.array_equal(g.T, -g))
    generators.append(g)
words = list(itertools.product(range(4), repeat=5))
basis = np.array([tensor([(I2, X, Z, J2)[i] for i in w]) for w in words])
skew = np.array([sum(i == 3 for i in w) % 2 == 1 for w in words])
alive = np.ones(1024, dtype=bool)
j0 = np.kron(J2, np.eye(16, dtype=np.int64))
ambient = []
expected = [(512,256),(256,136),(128,72),(64,36),(32,16),(16,6),(8,2),(4,1),(2,1)]
for d, g in enumerate(generators, 2):
    bg, gb = basis@g, g@basis
    commute = np.all(bg == gb, axis=(1,2))
    check('ambient', np.all(commute | np.all(bg == -gb, axis=(1,2))))
    alive &= commute
    pair = (int(alive.sum()), int((alive & skew).sum()))
    check('ambient', pair == expected[d-2])
    check('ambient', np.array_equal(j0@g, g@j0))
    ambient.append(dict(depth=d, commutant_dimension=pair[0], skew_dimension=pair[1]))
DATA['ambient'] = ambient
section('2. Shared Spin(10) carrier versus minimal carriers',
        ['J0=I_16 tensor J_2 up to component ordering; [J0,Spin(10)]=0.',
         '[J0,Spin(d)]=0 for every embedded subgroup.',
         'At d=7,8,9 the full skew-commutant dimensions are 6,2,1.'],
        'All depths retain a nondegenerate clock in the shared 32-real-component carrier. Ambient restriction alone does not implement the minimal selector.',
        'A realified complex half-spin carrier; no spatial compactification.')
del basis, gamma

# 3. Energy symmetry need not preserve the symplectic form.
symmetry = []
for i, t in enumerate((J1, J2H, J3)):
    defect = t.T*J3+J3*t
    zero('kinetic-symmetry', t.T+t)
    symmetry.append(dict(generator=i+1, preserves_isotropic_energy=True,
                         preserves_fixed_J3_kinetics=(defect == sp.zeros(4)),
                         kinetic_defect_rank=defect.rank()))
check('kinetic-symmetry', [r['preserves_fixed_J3_kinetics'] for r in symmetry] == [False,False,True])
DATA['fixed_J_symmetry'] = symmetry
section('3. Quaternionic energy symmetry: first-order and second-order branches',
        ['H=omega psi^T psi/2 is invariant under all three quaternion units.',
         'L1=psi^T J3 dot(psi)/2-H requires T^T J3+J3 T=0.',
         'Only the J3 generator preserves this fixed first-order kinetic term.',
         'L2=(|dot(psi)|^2-omega^2|psi|^2)/2 preserves all three skew generators.'],
        'An H-type commutant supplies three possible phase axes, but a fixed first-order clock retains only one continuous canonical phase symmetry. The isotropic second-order action retains all three.',
        'Classical commuting real fields. A Noether charge requires symmetry of the complete action, not only the energy.')

# 4. Conjugation branches and conserved quantities.
q = sp.Matrix(sp.symbols('q0:4', real=True))
p = sp.Matrix(sp.symbols('p0:4', real=True))
omega = sp.symbols('omega', positive=True)
charge_rows = []
for i, t in enumerate((J1,J2H,J3)):
    charge = (q.T*t*p)[0]
    derivative = sum(sp.diff(charge,q[k])*p[k]-sp.diff(charge,p[k])*omega**2*q[k] for k in range(4))
    zero('second-order-charges', derivative)
    charge_rows.append(str(charge))
c = sp.diag(1,1,-1,-1)
j = sp.kronecker_product(sp.Matrix(J2), sp.eye(2))
zero('conjugation', c*c-sp.eye(4))
zero('conjugation', c*j*c+j)
zero('conjugation', J2H*J1*J2H.T+J1)
zero('conjugation', J2H**2+E4)
u,v = sp.symbols('u v', real=True)
reverse_h = u*J2H+v*J3
zero('conjugation', reverse_h*J1+J1*reverse_h)
zero('conjugation', reverse_h**2+(u*u+v*v)*E4)
common_b = sp.diag(sp.Matrix(J2), sp.Matrix(J2))
opposite_phase = sp.diag(sp.Matrix(J2), -sp.Matrix(J2))
swap = sp.BlockMatrix([[sp.zeros(2),sp.eye(2)],[sp.eye(2),sp.zeros(2)]]).as_explicit()
charge_matrix = common_b*opposite_phase
zero('conjugation', swap.T*common_b*swap-common_b)
zero('conjugation', swap.T*charge_matrix*swap+charge_matrix)
zero('conjugation', swap.T*swap-E4)
zero('conjugation', common_b*common_b+E4)
DATA['second_order_charges'] = charge_rows
section('4. Conjugation, circulation and both orders of time evolution',
        ['C=diag(I,-I), C^2=I, CJC=-J on two real copies.',
         'For L2, Q_T=q^T T p is conserved for each T=J1,J2,J3.',
         'For fixed first-order J, C reverses the symplectic form: it is not a same-time canonical symmetry.',
         'Inside one H commutant, an element reversing J1 is u J2+v J3; its square is -(u^2+v^2)I, never +I for nonzero real u,v.',
         'For one fixed first-order phase generator T=J, its moment map is Q=psi^T JT psi/2=-|psi|^2/2: sign-definite.',
         'For two copies, B=diag(J,J), T=diag(J,-J), Q=(|psi_2|^2-|psi_1|^2)/2.',
         'The copy swap obeys C^T B C=B, C^T(BT)C=-BT and preserves the common positive isotropic energy.'],
        'Both signs of circulation exist in the second-order control. A paired first-order carrier also admits an explicit same-time canonical sign reversal, using a specified opposite-weight phase generator. A single uniform first-order phase has sign-definite action. Quaternionic reversal can square to minus identity; an involutive antipattern map is an additional requirement. No electric-charge identification follows.',
        'Internal classical circulation is distinct from observed electric charge. Reversing J together with the kinetic sector is a different branch from holding the action fixed.')

# 5. Time-dependent clock axis: derive the full six-coordinate symplectic form.
# In a north-pole chart a=(x,y,sqrt(1-x^2-y^2)), derivatives of J are J1,J2.
norm2 = (q.T*q)[0]
cross = sp.Matrix.hstack(J1*q/2, J2H*q/2)
form = J3.row_join(cross).col_join((-cross.T).row_join(sp.zeros(2)))
schur = sp.simplify(-cross.T*J3*cross)
target_schur = sp.Matrix([[0,norm2/4],[-norm2/4,0]])
zero('moving-axis-form', schur-target_schur)
zero('moving-axis-form', form+form.T)
zero('moving-axis-form', sp.det(form)-norm2**2/16)
axis_inverse = sp.Matrix([[0,-4/norm2],[4/norm2,0]])
zero('moving-axis-form', schur*axis_inverse-sp.eye(2))
zero('moving-axis-form', cross.T*(-J3)*q)
ax = sp.symbols('a1:4',real=True)
ja = sum((ax[i]*t for i,t in enumerate((J1,J2H,J3))),sp.zeros(4))
for i,t in enumerate((J1,J2H,J3)):
    zero('moving-axis-noether', (q.T*ja*t*q)[0]/4+norm2*ax[i]/4)
DATA['moving_axis'] = dict(symplectic_matrix=str(form), schur_complement=str(schur),
                          determinant=str(norm2**2/16), axis_inverse=str(axis_inverse),
                          spin_coefficient='s=|psi|^2/4=I/2', singular_branch='psi=0')
section('5. Derive a dynamical axis instead of adding an assumed rotor inertia',
        ['theta=psi^T J(a) dpsi/2, |a|=1; F=dtheta.',
         'At a=e3: F=[[J3,C],[-C^T,0]], C=(J1 psi,J2 psi)/2.',
         'F_axis,eff=-C^T J3 C=(|psi|^2/4)*[[0,1],[-1,0]].',
         'det(F)=|psi|^4/16; dot(a)=(1/s) a cross grad_a(H), s=|psi|^2/4.',
         'The norm I=|psi|^2/2 Poisson-commutes with a. At psi=0 this form is degenerate.',
         'Simultaneous psi -> exp(t J_i/2)psi and a -> rotation_i(t)a preserves the kinetic term.',
         'Its moment map is Q_i=psi^T J(a)J_i psi/4=-s a_i, with s=|psi|^2/4.'],
        'Allowing an independent clock axis in the original first-order term generates first-order precession, with coefficient fixed by the field norm. It does not generate the earlier second-order rotor law. Both kinetic branches must be kept.',
        'The axis is an independent internal pattern variable in this extended action. If it is a function of psi, its pullback must be recomputed. No claim that the fundamental field already contains this variable.')

# 6. A derived-axis pullback: an explicit different branch.
K = sp.kronecker_product(sp.eye(2), sp.Matrix(J2))
for t in (J1,J2H,J3):
    zero('derived-axis', K*t-t*K)
hopf = sp.Matrix([(q.T*K*t*q)[0]/norm2 for t in (J1,J2H,J3)])
zero('derived-axis', (hopf.T*hopf)[0]-1)
j_hopf = sum((hopf[i]*t for i,t in enumerate((J1,J2H,J3))), sp.zeros(4))
zero('derived-axis', j_hopf*q+K*q)
zero('derived-axis', q.T*j_hopf+q.T*K)
DATA['derived_axis'] = [str(sp.factor(x)) for x in hopf]
section('6. Independent versus field-derived axis',
        ['Choose a fixed commuting K with K^2=-I; a_i=(psi^T K J_i psi)/|psi|^2.',
         '|a|^2=1; J(a)psi=-K psi.',
         'Pullback[psi^T J(a)dpsi/2]=-psi^T K dpsi/2.'],
        'This explicit derived-axis choice collapses to a fixed kinetic structure and supplies no new independent axis degree of freedom. Its fixed K is additional structure; it is not a universal rotation-invariant selection law.',
        'Nonzero psi, fixed K. This branch is a test of coordinate dependence, not a new spatial dimension.')

# 7. Exact paired precession for both g signs, aligned and antialigned.
sa,sb,g,lam = sp.symbols('s_a s_b g lambda', nonzero=True, real=True)
R = sp.Matrix([[0,-1],[1,0]])
precession = []
for epsilon in (1,-1):
    matrix = sp.BlockMatrix([[epsilon*g*R/sa, -g*R/sa],
                             [-epsilon*g*R/sb, g*R/sb]]).as_explicit()
    expected_poly = lam**2*(lam**2+g**2*(epsilon/sa+1/sb)**2)
    zero('pair-precession', (lam*sp.eye(4)-matrix).det()-expected_poly)
    equal = matrix.subs({sa:1,sb:1,g:1})
    precession.append(dict(relative_orientation=epsilon,
                           characteristic_polynomial=str(sp.factor(expected_poly)),
                           equal_coefficient_rank=equal.rank(),
                           equal_coefficient_square_rank=(equal*equal).rank()))
check('pair-precession', precession[1]['equal_coefficient_square_rank']==0)
DATA['precession_spectra'] = precession
section('7. First-order paired axes: both signs and both equilibria',
        ['V=-g a dot b; dot(a)=-(g/s_a) a cross b; dot(b)=(g/s_b) a cross b.',
         'L=s_a a+s_b b and a dot b are exactly conserved.',
         'dot(a)=g/(s_a s_b) L cross a; Omega_prec=|g| |L|/|s_a s_b|.',
         'At b=epsilon a: chi(lambda)=lambda^2[lambda^2+g^2(epsilon/s_a+1/s_b)^2].'],
        'Both contact signs give conservative precession, not alignment. Generic nonzero branches have one relative oscillatory pair and two zero eigenvalues. Equal positive coefficients at antiparallel alignment give a nonzero nilpotent linear block: the zero frequency is a degenerate limit, not a positive restoring resonance.',
        'First-order axis action derived in section 5, with fixed nonzero norms. No assumed damping. Signed coefficients are also covered by the symbolic equation.')

# 8. Newtonian rotors, both signs and orientations, including anisotropy.
ia,ib = sp.symbols('iota_a iota_b', positive=True)
rotor = []
for epsilon in (1,-1):
    stiffness = g*sp.Matrix([[epsilon,-1],[-1,epsilon]])
    mass = sp.diag(ia,ib)
    a = mass.inv()*stiffness
    expected = lam*(lam-epsilon*g*(1/ia+1/ib))
    zero('pair-rotor', (lam*sp.eye(2)-a).det()-expected)
    for sign in (-1,0,1):
        rotor.append(dict(orientation=epsilon, coupling=sign,
                          restoring=epsilon*sign>0, unstable=epsilon*sign<0,
                          neutral=sign==0))
DATA['rotor_signs'] = rotor
section('8. Second-order paired axes: restoring and unstable branches',
        ['L_axes=(iota_a|dot(a)|^2+iota_b|dot(b)|^2)/2+g a dot b.',
         'iota_a(ddot(a)+|dot(a)|^2 a)=g[b-(a dot b)a].',
         'At b=epsilon a: nu_relative^2=epsilon*g*(1/iota_a+1/iota_b), twice.',
         'g*epsilon>0: two restoring modes; g*epsilon<0: two growing/decaying pairs; g=0: free axes.'],
        'The second-order law distinguishes energy minima from maxima through exponential stability, unlike pure precession. It has twice the orientation phase-space dimension and needs separately derived positive inertias.',
        'An alternative effective kinetic action. The rate is not inherited from the first-order field term.')

# 9. Interpolate the kinetic laws and classify all gyroscopic sign branches.
inertia,s,kx,ky = sp.symbols('iota s k_x k_y', real=True)
dispersion = (inertia*lam**2+kx)*(inertia*lam**2+ky)+s*s*lam**2
zero('gyro-polynomial', (inertia*lam**2*sp.eye(2)-s*lam*R+sp.diag(kx,ky)).det()-dispersion)
gyro_cases=[]
for iv in (Fraction(1,2),Fraction(1),Fraction(2)):
    for sv in (Fraction(0),Fraction(1),Fraction(3)):
        for xv,yv in itertools.product((-3,-1,0,1,3),repeat=2):
            A=iv*iv; B=iv*(xv+yv)+sv*sv; C=Fraction(xv*yv); disc=B*B-4*A*C
            if C<0:
                label='real-growth'
            elif C==0:
                label='zero-mode-with-growth' if B<0 else 'zero-mode-marginal'
            elif disc<0:
                label='complex-growth'
            elif B<0:
                label='real-growth'
            elif disc==0:
                label='imaginary-collision'
            else:
                label='pure-imaginary'
            M=np.array([[0,0,1,0],[0,0,0,1],[-xv/float(iv),0,0,-float(sv/iv)],
                        [0,-yv/float(iv),float(sv/iv),0]],float)
            eig=np.linalg.eigvals(M)
            if label=='pure-imaginary':
                check('gyro-numeric-control', max(abs(eig.real))<1e-7)
            elif label in ('real-growth','complex-growth','zero-mode-with-growth'):
                check('gyro-numeric-control', max(eig.real)>1e-6)
            gyro_cases.append(dict(inertia=str(iv),gyro=str(sv),kx=xv,ky=yv,
                                   classification=label,discriminant=str(disc),
                                   maximum_real_eigenvalue=float(max(eig.real))))
DATA['gyroscopic_cases']=gyro_cases
boundary=[]
for label,iv,sv,xv,yv in [('isotropic-repeated-stable',1,0,1,1),
                        ('inverted-gyro-threshold',1,2,-1,-1),
                        ('free-inertial',1,0,0,0),('free-gyroscopic',1,1,0,0),
                        ('one-flat-stiffness',1,1,1,0)]:
    m=sp.Matrix([[0,0,1,0],[0,0,0,1],[-sp.Rational(xv,iv),0,0,-sp.Rational(sv,iv)],
                 [0,-sp.Rational(yv,iv),sp.Rational(sv,iv),0]])
    ev=m.eigenvals()
    eigen_rows=[]
    for value,multiplicity in ev.items():
        geometric=4-(m-value*sp.eye(4)).rank()
        eigen_rows.append(dict(eigenvalue=str(value),algebraic=int(multiplicity),geometric=geometric))
    jordan=any(r['algebraic']>r['geometric'] for r in eigen_rows)
    check('gyro-boundaries',jordan==(label in ('inverted-gyro-threshold','free-inertial','one-flat-stiffness')))
    boundary.append(dict(case=label,eigenvalues=eigen_rows,has_jordan_drift=jordan))
DATA['gyroscopic_boundaries']=boundary
section('9. Mixed kinetic action and anisotropic response',
        ['iota ddot(delta)-s J_2 dot(delta)+diag(k_x,k_y)delta=0.',
         'chi(lambda)=iota^2 lambda^4+[iota(k_x+k_y)+s^2]lambda^2+k_x k_y.',
         'Strict imaginary spectrum: C=k_x k_y>0, B=iota(k_x+k_y)+s^2>0, B^2-4 iota^2 C>0.',
         'A mixed-sign stiffness (C<0) always has growth. Equality cases require a Jordan-block check.',
         'For k_x=k_y=2g: signed circular rates nu_+/-=[sqrt(s^2+8 iota g)+/-s]/(2 iota); magnitudes give frequencies.',
         'iota -> 0: one slow rate 2g/s and one fast rate s/iota; s -> 0 recovers the degenerate rotor rates.'],
        'Gyroscopic terms can spectrally stabilize an inverted isotropic contact when s^2+8 iota g>0, but cannot stabilize a mixed-sign stiffness. Exact eigenvector counts distinguish a bounded isotropic repeated frequency from Jordan drift at the inverted threshold. Free inertial motion drifts, whereas the fully gyroscopic free control has semisimple zero modes. The first-order limit discards a fast branch rather than reproducing two independent slow modes.',
        'Positive inertia for numerical controls; real signed g and anisotropy. Spectral classification is not nonlinear stability. No depth-specific coefficients.')

# 10. Stiff partner, zero stiffness, sign reversals, and finite selection window.
delta,kappa,vel,qp,pp=sp.symbols('Delta kappa velocity q p',real=True)
lp=pp*vel-delta*pp**2/2-kappa*qp**2/2
zero('stiff-partner', lp.subs(pp,vel/delta)-vel**2/(2*delta)+kappa*qp**2/2)
stiff=[]
for dv,kv in itertools.product((-4,-1,0,1,4),repeat=2):
    product=dv*kv
    stiff.append(dict(Delta=dv,kappa=kv,frequency_squared=product,
                      classification='oscillatory' if product>0 else 'exponential' if product<0 else 'degenerate',
                      positive_energy=dv>0 and kv>0,
                      elimination_regular=dv!=0))
DATA['stiff_branches']=stiff
section('10. Exact isolation versus finite stiffness',
        ['L=p dot(q)-Delta p^2/2-kappa q^2/2.',
         'Delta !=0: L_eff=dot(q)^2/(2 Delta)-kappa q^2/2; Omega^2=kappa Delta.',
         'Delta*kappa>0 oscillates; Delta*kappa<0 grows; a zero coefficient is degenerate.',
         'Both negative gives oscillation but negative energy. Delta>0,kappa>0 gives positive energy.',
         'A finite positive pair is below a frequency window Lambda iff kappa Delta <= Lambda^2.'],
        'A gap does not exactly remove a canonical partner. A cutoff can hide it only with a specified dynamical scale. Negative-energy oscillation is separately labelled rather than counted as a physical success.',
        'Quadratic internal control. Zero Delta cannot use the eliminated formula.')

# 11. First-order invariance of a proposed selected subspace.
projector_cases=[]
for keep in ([0],[0,1],[0,2],[0,3],[1,2],[0,1,2],[0,1,2,3]):
    projector=sp.diag(*[int(i in keep) for i in range(4)])
    leakage=(E4-projector)*J3*projector
    block=(projector*J3*projector).extract(keep,keep)
    projector_cases.append(dict(kept_components=keep,leakage_rank=leakage.rank(),
                                restricted_skew_rank=block.rank(),
                                invariant=leakage==sp.zeros(4),
                                kinetic_nondegenerate=block.det()!=0))
check('projector-branches',sum(r['invariant'] for r in projector_cases)==3)
check('projector-branches',all(r['kinetic_nondegenerate']==r['invariant'] for r in projector_cases))
DATA['projector_controls']=projector_cases
section('11. A subspace must close under its clock generator',
        ['A constant projector P is invariant for dot(psi)=-omega J psi iff (I-P)JP=0.',
         'The pulled-back kinetic form is PJP on im(P), and must be nondegenerate there.',
         'For a general quadratic energy K: closure requires (I-P)J^{-1}KP=0.'],
        'A small support projector is not automatically a closed first-order particle sector. Rank-one real projections lose the alternating form and leak under the unprojected clock. The script checks both invariant and leaking component choices.',
        'Linear positive-frequency control. A nonlinear selector needs the analogous full vector-field tangency condition.')

# 12. Unequal-frequency conservative interaction, with both signs and positivity.
wa,wb,e=sp.symbols('omega_a omega_b epsilon',real=True)
clock_h=sp.Matrix([[wa,e],[e,wb]])
zero('detuned-exchange',(lam*sp.eye(2)-clock_h).det()-((lam-wa)*(lam-wb)-e**2))
families={'electron-muon':(18564,3838380), 'electron-tau-shared-unit':(18564,64512240),
          'down-strange':(1,20), 'up-charm':(15,8855), 'charm-top':(8855,1215450),
          'neutrino-1-2':(2002,11628), 'neutrino-2-3':(11628,65780),
          'W-Z':(2926,3321), 'Z-Higgs':(3321,4560), 'equal-frequency-control':(1,1)}
exchange=[]
for family,(na,nb) in families.items():
    for fraction in (Fraction(0),Fraction(1,100),Fraction(1,10),Fraction(1,2),Fraction(1),Fraction(2)):
        for sign in (-1,1):
            eps_squared=fraction*fraction*na*nb
            denominator=Fraction((na-nb)**2)+4*eps_squared
            transfer=4*eps_squared/denominator if denominator else Fraction(0)
            exchange.append(dict(family=family,omega_ratio=str(Fraction(nb,na)),
                                  coupling_sign=sign,coupling_over_geometric_frequency=str(fraction),
                                  maximum_action_transfer=str(transfer),
                                  maximum_action_transfer_decimal=float(transfer),
                                  energy_positive=fraction<1,
                                  zero_energy_direction=fraction==1))
            check('detuned-exchange', 0<=transfer<=1)
DATA['exchange']=exchange
section('12. Unequal natural clocks: mixing without prescribed synchronization',
        ['H=omega_a|z_a|^2+omega_b|z_b|^2+2 epsilon Re(z_a^* z_b).',
         'Omega_+/-=(omega_a+omega_b)/2 +/- sqrt((omega_a-omega_b)^2/4+epsilon^2).',
         'Maximum transferred action fraction=4 epsilon^2/[(omega_a-omega_b)^2+4 epsilon^2].',
         'For positive omega_a,omega_b, H>0 iff epsilon^2<omega_a omega_b.',
         'Changing sign(epsilon) changes eigenvector phase, not the frequencies or maximum fraction.'],
        'The two eigenstates have common clocks but a generic unlike-particle preparation beats; conservation does not force phase locking. Large maintained frequency ratios strongly suppress weak-coupling exchange. Strong coupling can cross the positive-energy boundary and is explicitly marked.',
        'Two real quadratures written in paired notation, with an admitted common component map. Counts set same-sector frequency ratios; the electron/tau comparison additionally assumes their maintained common unit. No physical decay or interaction-rate identification.')

# 13. Frequency, action and total energy: both linear and nonlinear branches.
action,mu,nonlinear=sp.symbols('I mu lambda_nl',real=True)
ham=mu*action+nonlinear*action**2/2
frequency=sp.diff(ham,action)
zero('energy-clock',frequency-ham/action-nonlinear*action/2)
assignments=[('W',2,76),('Z',2,81),('Higgs',2,95),('down',3,1),('strange',3,4),
             ('up',4,3),('charm',4,20),('top',4,72),('nu1',5,10),('nu2',5,15),
             ('nu3',5,22),('electron',6,13),('muon',6,35),('tau',10,23)]
counts=[]
for name,d,n in assignments:
    N=n-1; count=math.comb(N+d,d)
    check('counts',sum(math.comb(k+d-1,d-1) for k in range(N+1))==count)
    neutral=1+(d if N>=1 else 0)
    rate_squared=Fraction(N)*(Fraction(N)-2+Fraction(2)**(1-N))
    check('counts',rate_squared>=0)
    counts.append(dict(particle=name,depth=d,n=n,N=N,IDOS=count,
                       neutral_disturbance_modes=neutral,positive_disturbance_modes=count-neutral,
                       top_core_disturbance_frequency_squared=str(rate_squared)))
DATA['counts']=counts
section('13. Count energy versus actual phase equation',
        ['H(I)=mu I+lambda_nl I^2/2; Omega=dH/dI=mu+lambda_nl I.',
         'H/I=mu+lambda_nl I/2; Omega-H/I=lambda_nl I/2.',
         'For independent positive-rate modes H=sum nu_j I_j, equal energies imply I_j=epsilon/nu_j and H=D epsilon, but dot(theta_j)=nu_j.',
         'D=C(N+d,d)=sum_{k=0}^N C(k+d-1,d-1).',
         'Original entropy-core control: nu_k^2=k[k-2+2^(1-k)], with k=0,1 neutral.',
         'A neutral disturbance is not a proof of zero complete bare field energy.'],
        'Linear energy-to-phase identification holds for this homogeneous oscillator; the nonlinear branch differs. Equal modal energies reproduce a total count but do not create a count-proportional carrier. All fourteen count assignments and the N=0 boundary are evaluated.',
        'Known mode assignments are inputs. No amplitude is fitted to an observed mass. Disturbance frequencies and particle carrier frequencies remain distinct.')

# 14. Finite-time reduced ODE controls, retaining both signs and initial angles.
ode=[]
max_errors={'norm':0.,'angle':0.,'angular_momentum':0.,'trajectory':0.,'rotor_energy':0.}
for gav in (-2.,0.,2.):
    for sav,sbv in ((1.,1.),(1.,2.)):
        for angle in (.2,1.2,math.pi-.2):
            a0=np.array([0.,0.,1.]); b0=np.array([math.sin(angle),0.,math.cos(angle)])
            total=sav*a0+sbv*b0
            def rhs(t,y):
                cross=np.cross(y[:3],y[3:])
                return np.r_[-gav*cross/sav,gav*cross/sbv]
            times=np.linspace(0.,8.,161)
            sol=solve_ivp(rhs,(0.,8.),np.r_[a0,b0],method='DOP853',rtol=2e-10,atol=2e-12,t_eval=times)
            check('precession-ode',sol.success)
            axis=total/np.linalg.norm(total)
            rate=gav*np.linalg.norm(total)/(sav*sbv)
            exact=np.array([a0*math.cos(rate*t)+np.cross(axis,a0)*math.sin(rate*t)
                            +axis*np.dot(axis,a0)*(1-math.cos(rate*t)) for t in times]).T
            av,bv=sol.y[:3],sol.y[3:]
            errors=dict(norm=float(max(np.max(abs(np.linalg.norm(av,axis=0)-1)),np.max(abs(np.linalg.norm(bv,axis=0)-1)))),
                        angle=float(np.max(abs(np.sum(av*bv,axis=0)-np.dot(a0,b0)))),
                        angular_momentum=float(np.max(abs(sav*av+sbv*bv-total[:,None]))),
                        trajectory=float(np.max(abs(av-exact))))
            for key,value in errors.items():
                max_errors[key]=max(max_errors[key],value)
                check('precession-ode',value<2e-7)
            ode.append(dict(kind='first-order',g=gav,s_a=sav,s_b=sbv,angle=angle,errors=errors))
for gav in (-1.,1.):
    for orientation in (-1.,1.):
        iav,ibv=1.,2.
        a0=np.array([0.,0.,1.]);b0=np.array([.05,0.,orientation*math.sqrt(1-.05**2)])
        def rhs_rotor(t,y):
            av,bv,va,vb=y[:3],y[3:6],y[6:9],y[9:]
            c=np.dot(av,bv)
            return np.r_[va,vb,gav*(bv-c*av)/iav-np.dot(va,va)*av,
                         gav*(av-c*bv)/ibv-np.dot(vb,vb)*bv]
        sol=solve_ivp(rhs_rotor,(0.,4.),np.r_[a0,b0,np.zeros(6)],method='DOP853',rtol=2e-10,atol=2e-12,
                      t_eval=np.linspace(0.,4.,121))
        check('rotor-ode',sol.success)
        av,bv,va,vb=sol.y[:3],sol.y[3:6],sol.y[6:9],sol.y[9:]
        energy=(iav*np.sum(va*va,axis=0)+ibv*np.sum(vb*vb,axis=0))/2-gav*np.sum(av*bv,axis=0)
        drift=float(np.max(abs(energy-energy[0])))
        normerror=float(max(np.max(abs(np.linalg.norm(av,axis=0)-1)),np.max(abs(np.linalg.norm(bv,axis=0)-1))))
        check('rotor-ode',drift<2e-7)
        check('rotor-ode',normerror<2e-7)
        max_errors['rotor_energy']=max(max_errors['rotor_energy'],drift)
        ode.append(dict(kind='second-order',g=gav,orientation=orientation,energy_drift=drift,norm_error=normerror,
                        initial_dot=float(np.dot(a0,b0)),final_dot=float(np.dot(av[:,-1],bv[:,-1]))))
DATA['ode_controls']=ode
DATA['ode_max_errors']=max_errors
section('14. Exact histories versus numerical reduced evolution',
        ['First-order pair: both axes rotate about the conserved L with rate g|L|/(s_a s_b).',
         'Second-order pair: E=(iota_a|dot(a)|^2+iota_b|dot(b)|^2)/2-g a dot b is conserved.'],
        'Eighteen first-order runs check the exact rotation formula and conservation for both signs and zero coupling. Four second-order runs check both stationary orientations and both contact signs. These are short consistency controls, not lifetime or nonlinear-stability theorems.',
        'DOP853 with rtol=2e-10, atol=2e-12. Numerical errors and trajectories are summarized in the JSON report.')

# 15. Explicit branch ledger: conclusions must retain their assumptions.
ledger=[
    ['minimal irreducible first-order carrier','C/H depths admit clocks','R depths lack a nondegenerate invariant first-order form'],
    ['repeated real-type carrier','even multiplicity permits a clock','odd multiplicity leaves null directions'],
    ['ambient restriction','clocks survive at every tested depth','does not implement minimal selection'],
    ['fixed quaternionic first-order J','one continuous canonical phase symmetry','other two energy symmetries fail kinetic invariance'],
    ['isotropic second-order kinetics','three conserved quaternionic circulations','no identification with charge or colour'],
    ['first-order phase charge','one uniform phase has sign-definite action','paired opposite weights allow canonical sign-reversing copy swap'],
    ['independent dynamical J','first-order sphere precession with s=I/2','zero field norm is singular'],
    ['derived Hopf J(psi)','pullback reduces to fixed -K','no independent axis degree of freedom'],
    ['pure first-order pair','both signs precess; angle conserved','no conservative attraction to alignment'],
    ['second-order pair','energy-minimum branch restores','energy-maximum branch grows'],
    ['mixed gyroscopic pair','sufficient gyro strength can stabilize two negative stiffnesses spectrally','mixed-sign stiffness always grows'],
    ['repeated or zero eigenvalues','semisimple branches remain bounded linearly','defective branches have Jordan drift; both explicitly checked'],
    ['finite stiff partner','positive finite gap retains a second-order oscillator','exact projection is a distinct assumption'],
    ['equal frequency mixing','nonzero coupling permits complete action transfer','zero coupling transfers nothing'],
    ['unequal frequency mixing','detuning suppresses weak transfer','large coupling can violate positive energy'],
    ['linear homogeneous oscillator','Omega=H/I','nonlinear H generally gives Omega different from H/I'],
    ['equal energy per counted mode','total energy is D epsilon','actual mode clocks remain their Hamiltonian derivatives'],
]
DATA['branch_ledger']=ledger
section('15. Conditional branch ledger',[],
        'Every calculation branch and its alternative are listed explicitly in branch_ledger. A passing algebraic check does not choose the physical branch.',
        'No observed particle, electric charge, mediator or probability interpretation is inferred from the matrix labels.')

sources={'experiments/spinor-clock-branches.py':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
report=dict(created_utc=datetime.now(timezone.utc).isoformat(),verification_scope="portable branch equations and ODE controls",
            runtime_seconds=time.monotonic()-START,checks=CHECKS,total_checks=sum(CHECKS.values()),
            sources=sources,libraries=dict(numpy=np.__version__,sympy=sp.__version__,scipy=scipy.__version__),
            sections=SECTIONS,data=DATA)
result_path=HERE/('spinor-clock-branches-results-'+stamp+'.json')
result_path.write_text(json.dumps(report,indent=2)+'\n')
lines=['# Both Branches of the Real-Clock Calculation','',
       'This is the executed equation ledger. Each section states its assumptions and alternative outcomes.',
       f'Exact and control assertions passed: {sum(CHECKS.values())}.','']
for item in SECTIONS:
    lines+=['## '+item['title'],'',item['assumptions'],'']
    lines += ['```text']+item['equations']+['```',''] if item['equations'] else []
    lines += [item['result'],'']
lines+=['## Branch Ledger','','| Condition | First outcome | Other outcome or limit |','| --- | --- | --- |']
lines+=['| '+' | '.join(row)+' |' for row in ledger]
lines+=['','## Exact Minimal-Carrier Table','','| d | Type | Real dimension | Skew generators |','| ---: | --- | ---: | ---: |']
lines += [f"| {r['depth']} | {r['type']} | {r['real_module_dimension']} | {r['skew_dimension']} |" for r in minimal]
lines+=['','## Common Ambient Carrier','','| d | Real commutant dimension | Skew dimension |','| ---: | ---: | ---: |']
lines += [f"| {r['depth']} | {r['commutant_dimension']} | {r['skew_dimension']} |" for r in ambient]
lines+=['','## Weak Unequal-Frequency Exchange Control','',
        'Coupling magnitude is one hundredth of the geometric mean frequency. Entries are transferred action fractions, not decay rates or probabilities.','',
        '| Pair | Frequency ratio | Maximum fraction |','| --- | ---: | ---: |']
lines += [f"| {r['family']} | {r['omega_ratio']} | {r['maximum_action_transfer']} ({r['maximum_action_transfer_decimal']:.8g}) |"
          for r in exchange if r['coupling_over_geometric_frequency']=='1/100' and r['coupling_sign']==1]
lines+=['','## Numerical Errors','',json.dumps(max_errors,indent=2),'',
        'The JSON report contains every multiplicity, stiffness, gyroscopic, exchange and ODE case, including unfavorable and singular branches.',
        'No empirical particle assignment was changed. This script tests explicit candidate laws; it does not establish which action nature realizes.','']
(HERE/'equations-and-results.md').write_text('\n'.join(lines))
print(json.dumps(dict(result_path=str(result_path.relative_to(ROOT)),total_checks=sum(CHECKS.values()),
                      runtime_seconds=report['runtime_seconds'],ode_max_errors=max_errors),indent=2),flush=True)
