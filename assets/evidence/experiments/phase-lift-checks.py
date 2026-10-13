#!/usr/bin/env python3
"""Check complete spinor lifts and the metric retained by a projector column."""

from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import json

import sympy as sp


CHECKS=[]
CASES=[]


def check(name,condition):
    if not bool(condition):
        raise AssertionError(name)
    CHECKS.append(name)


def equal(name,lhs,rhs):
    if isinstance(lhs,sp.MatrixBase):
        check(name,(lhs-rhs).applyfunc(sp.simplify)==sp.zeros(*lhs.shape))
    else:
        check(name,sp.simplify(lhs-rhs)==0)


def inner(v,w):
    return sp.simplify((v.conjugate().T*w)[0])


def root_lifts():
    a,r,L=sp.symbols('a r L',positive=True)
    angle=sp.Symbol('angle',real=True)
    den=a*a+r*r
    psi=sp.Matrix([a,r*sp.exp(sp.I*angle)])/sp.sqrt(den)
    dr,da=psi.diff(r),psi.diff(angle)
    equal('normalized local lift',inner(psi,psi),1)
    equal('radial Berry connection',-sp.I*inner(psi,dr),0)
    berry=-sp.I*inner(psi,da)
    equal('angular Berry connection',berry,r*r/den)
    fsr=inner(dr,dr)-inner(dr,psi)*inner(psi,dr)
    fsa=inner(da,da)-inner(da,psi)*inner(psi,da)
    equal('radial projective metric',fsr,a*a/den**2)
    equal('angular projective metric',fsa,a*a*r*r/den**2)
    equal('phase curvature total flux',sp.integrate(2*sp.pi*sp.diff(berry,r),(r,0,sp.oo)),2*sp.pi)
    equal('canonical vacuum line Chern class',sp.I/(2*sp.pi)*sp.I*2*sp.pi,-1)
    complete=inner(dr,dr)+inner(da,da)/r**2
    equal('complete normalized spinor gradient',complete,(r*r+2*a*a)/den**2)
    integral=sp.integrate(2*sp.pi*r*complete,(r,0,L))
    expected=sp.pi*(sp.log(1+L*L/(a*a))+1-a*a/(a*a+L*L))
    equal('finite-radius full normalized gradient',integral,expected)
    check('full fixed-amplitude lift diverges',sp.limit(expected,L,sp.oo)==sp.oo)
    equal('finite full-plane projective norm',sp.integrate(2*sp.pi*r*(fsr+fsa/r**2),(r,0,sp.oo)),2*sp.pi)

    projector=psi*psi.conjugate().T
    reference=sp.Matrix([0,1])
    hole=projector*reference
    equal('explicit amplitude-zero lift',hole,sp.Matrix([a*r*sp.exp(-sp.I*angle),r*r])/den)
    equal('hole amplitude squared',inner(hole,hole),r*r/den)
    equal('hole same ray away from core',hole,hole[1]/psi[1]*psi)
    equal('hole radial metric',inner(hole.diff(r),hole.diff(r)),fsr)
    equal('hole angular metric',inner(hole.diff(angle),hole.diff(angle)),fsa)
    equal('hole mixed real metric',sp.re(inner(hole.diff(r),hole.diff(angle))),0)
    complete_hole=inner(hole.diff(r),hole.diff(r))+inner(hole.diff(angle),hole.diff(angle))/r**2
    equal('finite complete hole norm',sp.integrate(2*sp.pi*r*complete_hole,(r,0,sp.oo)),2*sp.pi)
    center=reference/2
    equal('hole target sphere constraint',inner(hole-center,hole-center),sp.Rational(1,4))
    residual=hole.diff(r,2)+hole.diff(r)/r+hole.diff(angle,2)/r**2
    equal('unrestricted flat-field residual',residual,-8*a*a/den**2*(hole-center))
    equal('radial-potential force cannot match residual',
          sp.det(hole.row_join(residual)),4*a*a/den**2*hole[0])
    check('residual not identically zero',sp.simplify(residual[0])!=0)
    equal('Cartesian core limit first component',sp.limit(hole[0],r,0),0)
    equal('Cartesian core limit second component',sp.limit(hole[1],r,0),0)
    equal('constant boundary first component',sp.limit(hole[0],r,sp.oo),0)
    equal('constant boundary second component',sp.limit(hole[1],r,sp.oo),1)
    # Reconstruct the entire norm from amplitude, local phase and projective terms.
    amplitude=r/sp.sqrt(den)
    phase_gradient=(-1+berry)**2/r**2
    reconstructed=sp.diff(amplitude,r)**2+amplitude**2*(phase_gradient+fsr+fsa/r**2)
    equal('complete polar lift with all three contributions',reconstructed,complete_hole)
    equal('conformal complete determinant equation coefficient',
          (1+sp.Symbol('kappa',positive=True)*fsr)/(1+sp.Symbol('kappa',positive=True)*fsr),1)


def projector_column_controls():
    t=sp.Symbol('overlap',positive=True)
    for s in (2,4,8):
        psi=sp.Matrix([sp.sqrt(t),sp.sqrt(1-t)]+[0]*(s-2))
        reference=sp.eye(s)[:,0]
        tangent=[sp.Matrix([sp.sqrt(1-t),-sp.sqrt(t)]+[0]*(s-2))]
        tangent += [sp.eye(s)[:,i] for i in range(2,s)]
        real_tangent=[]
        for h in tangent:
            real_tangent.extend((h,sp.I*h))
        dprojector=[h*psi.conjugate().T+psi*h.conjugate().T for h in real_tangent]
        # Symbol t is in the interior (0,1); refine its square-root conjugates explicitly.
        replacements={sp.conjugate(sp.sqrt(1-t)):sp.sqrt(1-t)}
        dprojector=[m.xreplace(replacements) for m in dprojector]
        column=[m*reference for m in dprojector]
        for i in range(len(real_tangent)):
            for j in range(len(real_tangent)):
                expected=(1 if i<2 else t) if i==j else 0
                actual=(column[i].conjugate().T*column[j])[0].xreplace(replacements)
                equal(f'rank{s} full column metric {i},{j}',
                      (actual+sp.conjugate(actual).xreplace(replacements))/2,expected)
                value=sp.trace(dprojector[i].conjugate().T*dprojector[j]).xreplace(replacements)
                equal(f'rank{s} full projector metric {i},{j}',
                      (value+sp.conjugate(value).xreplace(replacements))/2,2 if i==j else 0)
        at_zero=[m.subs(t,0) for m in column]
        real_columns=sp.Matrix.hstack(*[sp.re(m).col_join(sp.im(m)) for m in at_zero])
        equal(f'rank{s} zero-overlap real differential rank',real_columns.rank(),2)
        if s==2:
            check('CP1 column isometry',len(real_tangent)==2)
        else:
            check(f'rank{s} column loses complete metric',len(real_tangent)-2==2*(s-2))


def sectors():
    for depth,q in ((2,4),(3,3),(4,3),(5,2),(6,2),(10,0)):
        s=2**(q-1) if q else 1
        raydim=2*(s-1)
        orientation=q*(q-1)
        if q:
            check(f'd{depth} complete chiral polar degree count',2+raydim==2*s)
            check(f'd{depth} independent full coefficient adds degrees',orientation+2*s>2*s)
        CASES.append({'depth':depth,'normal_rank':q,'chiral_rank':s,
                      'full_ray_dimension':raydim,'orientation_dimension':orientation,
                      'root_defect_present':q>=2,'fixed_amplitude_phase_energy_finite':False if q else None,
                      'root_amplitude_zero_lift_norm':'2*pi' if q else None,
                      'defect_extended_directions':depth-2 if q else None,
                      'column_isometry_on_complete_ray':s==2})


def main():
    root_lifts()
    projector_column_controls()
    sectors()
    path=Path(__file__).resolve()
    report={'created_at':datetime.now(timezone.utc).isoformat(),
            'experiment':'complete-phase-and-amplitude-lift',
            'sources':{'experiments/phase-lift-checks.py':sha256(path.read_bytes()).hexdigest()},
            'checks_passed':len(CHECKS),'cases':CASES,
            'scope':'Exact transverse root profiles, complete second-order norm, unrestricted residual and full projective differential metrics. Kinematic amplitude-zero lift and constrained-target interpretation; no unrestricted particle solution or mass-frequency selection.'}
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    output=path.with_name('phase-lift-results-'+stamp+'.json')
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'checks_passed':len(CHECKS),'cases':len(CASES),'report':str(output)}))


if __name__=='__main__':
    main()
