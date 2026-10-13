"""An explicit count-energy curvature law preserving every stationary shell."""
from datetime import datetime, timezone
from fractions import Fraction
from hashlib import sha256
from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path
import json

import mpmath as mp
import numpy as np
import sympy as sp

ROOT = Path(__file__).resolve().parent
spec = spec_from_file_location('forward', ROOT/'forward-count-action.py')
forward = module_from_spec(spec)
spec.loader.exec_module(forward)
COUNTS, ERRORS = {}, {}


def check(value, label):
    if not bool(value):
        raise AssertionError(label)
    COUNTS[label] = COUNTS.get(label, 0)+1


def tiny(error, scale, label):
    relative = abs(error)/max(mp.mpf(1), abs(scale))
    ERRORS[label] = max(ERRORS.get(label, mp.mpf(0)), relative)
    check(relative < mp.mpf('1e-65'), label)


def high(value):
    return mp.mpf(value.numerator)/value.denominator


def symbolic_schur():
    x, c, h, density = sp.symbols('x c h density', positive=True)
    curvature = x*x*c/h
    hessian = h/(density*c**3)*sp.Matrix([[x*x, -x], [-x, 1]])
    coupling = sp.Matrix([-x*x/c, x/c])
    reduced = hessian-coupling*coupling.T/(density*curvature)
    check(reduced.applyfunc(sp.simplify) == sp.zeros(2),
          'generic density and scalar Schur Hessian vanishes at the explicit target count curvature')


def main():
    mp.mp.dps = 110
    symbolic_schur()
    cases = []
    for name, depth, index in forward.radial.ASSIGNMENTS:
        level = index-1
        hh = mp.mpf(level)+mp.mpf(depth)/2
        count, first, second = forward.count_data(depth, level)
        epsq = Fraction(3, 10)/max(forward.count_data(depth, n-1)[1]
                                 for _, d, n in forward.radial.ASSIGNMENTS if d == depth)
        epsilon, c = high(epsq), high(epsq*first)
        x = mp.sqrt(1-c*c)
        target = c*x*x/hh
        def original(label):
            return epsilon*mp.fprod(label+j-mp.mpf(depth)/2 for j in range(1, depth+1))/mp.factorial(depth)
        def correction(label):
            slope = mp.diff(original, label)
            return slope*(1-slope*slope)/label-mp.diff(original, label, 2)
        def changed(label):
            return original(label)+correction(label)*mp.sin(mp.pi*(label-mp.mpf(depth)/2))**2/(2*mp.pi**2)
        ff, fp, fpp = changed(hh), mp.diff(changed, hh), mp.diff(changed, hh, 2)
        tiny(ff-original(hh), original(hh), 'the explicit forward response law preserves every count frequency')
        tiny(fp-c, c, 'the explicit forward response law preserves every stationary polarization and source')
        tiny(fpp-target, target, 'the explicit forward response law gives the required algebraic curvature')
        check(fpp > 0, 'positive auxiliary curvature gives a valid local elimination chart at every assignment')
        def potential(q, scalar, label):
            return q*changed(label)-label*mp.sqrt(q*q-scalar*scalar)
        # Independent derivatives of the actual three-variable potential.
        hessian = mp.matrix(3)
        args = (mp.mpf(1), x, hh)
        for i in range(3):
            for j in range(3):
                order = tuple(int(k == i)+int(k == j) for k in range(3))
                hessian[i, j] = mp.diff(potential, args, order)
        schur = hessian[:2, :2]-hessian[:2, 2:3]*hessian[2:3, :2]/hessian[2, 2]
        tiny(mp.norm(schur), mp.norm(hessian),
             'independent derivatives of the complete potential verify its vanishing eliminated invariant Hessian')
        curvature_of_f = hh/c**3-x*x/(c*c*fpp)
        tiny(curvature_of_f, hh/c**3,
             'the eliminated constitutive scalar curvature vanishes without an implicit inverse patch')
        mass, lam, eta = hh*x/c, hh/c, original(hh)-hh/c
        # A direct massive shell diagonalization tests the full spectral formula.
        for harmonic in [0, 1, level, level+1, level+3, 1000]:
            bare = mp.mpf(harmonic)+mp.mpf(depth)/2
            operator = mp.matrix([[bare-lam, mass], [mass, -bare-lam]])
            values = mp.eigsy(operator, eigvals_only=True)
            root = mp.sqrt(bare*bare+mass*mass)
            tiny(values[0]-(-root-lam), root+lam,
                 'independent massive Dirac pair diagonalization matches the complete self-adjoint spectral formula')
            tiny(values[1]-(root-lam), root+lam,
                 'independent massive Dirac pair diagonalization matches the complete self-adjoint spectral formula')
        pair_shape = 2*mp.sqrt(x*x*hh/(c*fpp))
        tiny(pair_shape-2*lam, 2*lam,
             'the reduced shape disturbance clock agrees with the retained massive pair gap')
        # Show that the analytic replacement is not an exactly affine patch.
        chi_s = -x/(c*fpp)
        third = 3*hh*x/c**5+3*chi_s/c**3+mp.diff(changed, hh, 3)*chi_s**3
        check(abs(third) > mp.mpf('1e-30'),
              'a retained nonzero cubic derivative prevents transferring the nonlinear flat-patch theorem')
        cases.append(dict(particle=name, depth=depth, inferred_index=index,
                          epsilon_exact=str(epsq), polarization_exact=str(epsq*first),
                          target_count_curvature_exact=str((epsq*first)*(1-(epsq*first)**2)/Fraction(2*level+depth, 2)),
                          clock_in_radius_units_exact=str(epsq*count),
                          eliminated_hessian_relative_residual=str(mp.norm(schur)/max(1, mp.norm(hessian))),
                          scalar_cubic_derivative=str(third),
                          shape_disturbance_frequency_in_radius_units=str(pair_shape),
                          classification='explicit analytic forward law with unchanged count clocks and scalar source Hessian'))
        print(json.dumps(dict(completed=name, depth=depth)), flush=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    report = dict(experiment_key='forward-count-energy-response', created_utc=datetime.now(timezone.utc).isoformat(),
                  sources={'experiments/'+p.name: sha256(p.read_bytes()).hexdigest() for p in
                           [Path(__file__), ROOT/'forward-count-action.py', ROOT/'inverse-current-radial.py',
                            ROOT/'excited-shell-orientation-current.py']},
                  evaluations=sum(COUNTS.values()), conditions=COUNTS,
                  maximum_errors={key: str(value) for key, value in ERRORS.items()}, cases=cases,
                  limits=['the periodic count-energy correction is an explicit constitutive choice',
                          'local convex elimination is established at each shell, not on the whole continuous label interval',
                          'complete linear unitary persistence applies to the preferred-time density realization',
                          'proper-current and responding-background realizations retain their additional current Hessians',
                          'a nonzero cubic response excludes transfer of the nonlinear affine-patch proof',
                          'no new sector preference, localization, complete inertia or observed-particle identification follows'])
    output = ROOT/('forward-count-response-results-'+stamp+'.json')
    output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(report=str(output), evaluations=report['evaluations'], maximum_errors=report['maximum_errors'])), flush=True)


if __name__ == '__main__':
    main()
