"""Complete frozen local spinor-current symbol, separate from spatial stability."""
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from importlib.util import spec_from_file_location, module_from_spec
import json

import numpy as np
from scipy.linalg import eigvals, svd

ROOT = Path(__file__).resolve().parent
spec = spec_from_file_location('inverse', ROOT/'inverse-current-ratio-checks.py')
inverse = module_from_spec(spec)
spec.loader.exec_module(inverse)
CONDITIONS, ERRORS = {}, {}


def check(value, label):
    if not bool(value):
        raise AssertionError(label)
    CONDITIONS[label] = CONDITIONS.get(label, 0)+1


def close(a, b, label, tolerance=2e-10):
    error = np.linalg.norm(a-b)/max(1., np.linalg.norm(b))
    ERRORS[label] = max(ERRORS.get(label, 0.), float(error))
    check(error < tolerance, label)


def vector(a):
    return a.reshape(-1, order='F')


def reduced_symbol(depth, c, dc, h, omega, axis):
    alpha, beta, _ = inverse.ratio.scalar.clifford(depth)
    rank = len(beta)
    x = np.sqrt(1-c*c)
    a, b = np.sqrt((1+c)/2), np.sqrt((1-c)/2)
    plus, minus = (np.eye(rank)+beta)/2, (np.eye(rank)-beta)/2
    background = ((a+b)*plus+c/(a+b)*minus)/np.sqrt(rank)
    tangent = (c*(a+b)/(1+x)*plus-(1+x)/(a+b)*minus)/np.sqrt(rank)
    scalar_factor = 1-x*x*c/(h*dc)
    vector_factor = 1-omega*c/h
    # Endpoint column projectors avoid a singular-value basis polluted by tiny c.
    seeds = [plus, minus]+[g@p for g in alpha for p in (plus, minus)]
    seeds += [alpha[axis]@z for z in list(seeds)]
    real_columns = np.column_stack([z for s in seeds for z in (vector(s).real, vector(s).imag)])
    u, values, _ = svd(real_columns, full_matrices=False)
    basis = u[:, values > 1e-11*values[0]]
    def apply(matrix, columns):
        return np.column_stack([vector(matrix@column.reshape((rank, rank), order='F')) for column in columns.T])
    kinetic = basis.T@apply(alpha[axis], basis)
    grading = basis.T@apply(beta, basis)
    close(apply(alpha[axis], basis), basis@kinetic, 'retained real matrix range is exactly invariant under the frozen Dirac symbol')
    close(apply(beta, basis), basis@grading, 'retained matrix range is exactly invariant under physical grading')
    sources = [vector(tangent)]+[vector(g@background) for g in alpha]
    for source in sources:
        close(source, basis@(basis.T@source), 'every full local-current feedback source lies in the retained range')
    sources = [basis.T@z for z in sources]
    coefficients = [scalar_factor]+[vector_factor]*depth
    hermitian = sum((q*np.outer(z, z.conj()) for q, z in zip(coefficients, sources)),
                    np.zeros_like(grading, dtype=complex))
    anomalous = sum((q*np.outer(z, z) for q, z in zip(coefficients, sources)),
                    np.zeros_like(grading, dtype=complex))
    close(hermitian, hermitian.conj().T, 'full current Hessian retains its Hermitian complex-linear part')
    close(anomalous, anomalous.T, 'full current Hessian retains its symmetric conjugate-linear part')
    base = x*grading-np.eye(len(grading))+hermitian
    cases = []
    for wave_number in (0., .01, .03, .1, .3, 1., 3., 10.):
        ap = base+wave_number*kinetic
        am = base-wave_number*kinetic
        generator = np.block([[ap, anomalous], [-anomalous.conj(), -am.conj()]])
        if wave_number == 0:
            columns = np.column_stack([basis.T@vector(alpha[axis]@p)/np.sqrt(rank/2)
                                       for p in (plus, minus)])
            lift = np.block([[columns, np.zeros_like(columns)],
                             [np.zeros_like(columns), columns.conj()]])
            if depth == 2:
                ca = -(1-vector_factor)*np.diag([1+x, 1-x])
                cb = vector_factor*c*np.array([[0, 1], [1, 0]])
            else:
                v = np.array([(a+b)/np.sqrt(2), c/(np.sqrt(2)*(a+b))])
                ca = np.diag([-1-x, -1+x])+vector_factor*np.outer(v, v)
                cb = vector_factor*np.outer(v, v)
            exact_block = np.block([[ca, cb], [-cb, -ca]])
            close(generator@lift, lift@exact_block,
                  'complete frozen symbol retains the independently derived zero-wavevector current block')
        signature = np.diag([1.]*len(grading)+[-1.]*len(grading))
        energy = signature@generator
        close(energy, energy.conj().T, 'both Fourier wavevectors give the complete Hamiltonian frozen Hessian')
        roots = eigvals(generator)
        cases.append({'dimensionless_wave_number': wave_number,
                      'maximum_growth_over_lambda': float(np.max(roots.imag)),
                      'maximum_frequency_over_lambda': float(np.max(abs(roots.real))),
                      'resolved_complex_root_count': int(np.sum(abs(roots.imag)>3e-7))})
    return {'depth': depth, 'axis': axis, 'full_spinor_matrix_complex_dimension': rank*rank,
            'retained_complex_dimension': len(grading), 'signed_shell_occupation': c,
            'scalar_feedback_factor': scalar_factor, 'vector_feedback_factor': vector_factor,
            'lambda_over_inverse_radius': h/c, 'probes': cases}


def main():
    sp = inverse.sp
    rr, xx = sp.symbols('r x', real=True)
    cc = sp.sqrt(1-xx*xx)
    ll = sp.diag(-1-xx, -1+xx)
    hh = sp.Matrix([[-rr*(1+xx), (1-rr)*cc], [(1-rr)*cc, -rr*(1-xx)]])
    product = ll*hh
    check(sp.simplify(sp.trace(product)-2*rr*(1+xx*xx)) == 0,
          'exact zero-wavevector squared-frequency trace retains the full current variation')
    check(sp.simplify(product.det()-(1-xx*xx)**2*(2*rr-1)) == 0,
          'exact zero-wavevector squared-frequency determinant has its stated negative sign')
    z = sp.symbols('z')
    two = sp.Matrix([[-rr*(1+xx),(1-rr)*cc],[-(1-rr)*cc,rr*(1-xx)]])
    check(sp.simplify(two.charpoly(z).as_expr()-(z*z+2*rr*xx*z+(1-xx*xx)*(1-2*rr))) == 0,
          'depth-two overlapping vector currents have their separate exact channel polynomial')
    cases = []
    for name, depth, index in inverse.ratio.scalar.sphere.ASSIGNMENTS:
        h = inverse.mp.mpf(index-1)+inverse.mp.mpf(depth)/2
        p, first, second = inverse.polynomial(depth, h)
        c, dc = inverse.EPSILON*first, inverse.EPSILON*second
        result = reduced_symbol(depth, float(c), float(dc), float(h), float(inverse.EPSILON*p), 0)
        rr = inverse.EPSILON*p*c/h
        xx = inverse.mp.sqrt(1-c*c)
        if depth == 2:
            growth_squared = (1-rr)**2*c*c-rr*rr
        else:
            trace = 2*rr*(1+xx*xx)
            determinant = c**4*(2*rr-1)
            growth_squared = -2*determinant/(inverse.mp.sqrt(trace*trace-4*determinant)+trace)
        check(growth_squared > 0, 'exact frozen zero-wavevector current channel has a positive growth square')
        growth = inverse.mp.sqrt(growth_squared)
        result['exact_zero_wavevector_growth_over_lambda'] = float(growth)
        result['exact_zero_wavevector_growth_in_inverse_radius_units'] = float(growth*h/c)
        close(result['probes'][0]['maximum_growth_over_lambda'], float(growth),
              'floating full frozen symbol agrees with the exact low-wavevector channel within its resolved precision', 3e-7)
        result.update(name=name, mode_index=index, spatial_shell_level=index-1)
        cases.append(result)
    covariance = []
    for depth in inverse.ratio.scalar.sphere.DEPTHS:
        record = next(a for a in cases if a['depth']==depth)
        name, _, index = next(a for a in inverse.ratio.scalar.sphere.ASSIGNMENTS if a[1]==depth)
        h = inverse.mp.mpf(index-1)+inverse.mp.mpf(depth)/2
        p, first, second = inverse.polynomial(depth, h)
        control = reduced_symbol(depth, float(inverse.EPSILON*first), float(inverse.EPSILON*second),
                                 float(h), float(inverse.EPSILON*p), depth-1)
        for a, b in zip(record['probes'], control['probes']):
            close(a['maximum_growth_over_lambda'], b['maximum_growth_over_lambda'],
                  'common physical rotation preserves the frozen growth spectrum', 3e-7)
        covariance.append(control)
    files = ['inverse-current-frozen-symbol.py','inverse-current-ratio-checks.py','local-current-ratio-checks.py',
             'scalar-current-band-checks.py','spatial-source-saturation-checks.py','spatial-band-curvature-checks.py']
    report = {'experiment_key': 'inverse-current-complete-frozen-local-symbol',
              'created_utc': datetime.now(timezone.utc).isoformat(),
              'sources': {'experiments/'+f:sha256((ROOT/f).read_bytes()).hexdigest() for f in files},
              'classification': 'complete local coefficient-frozen dispersion diagnostic, not a full spatial eigenvalue calculation',
              'limits': ['actual spatial background and spin-connection derivatives are not part of the frozen symbol',
                         'complex frozen frequencies do not by themselves prove a global spatial exponential eigenmode or decay',
                         'the exact zero-wavevector growth is of the same scale as omitted spatial geometry and cannot be transferred to the actual sphere solution',
                         'floating roots below 3e-7 in lambda units are unresolved; exact zero-wavevector formulas are reported separately',
                         'the coupled matrix range is complete for this frozen symbol; its complement is self-adjoint and uncoupled'],
              'condition_count':len(CONDITIONS),'evaluation_count':sum(CONDITIONS.values()),
              'floating_complex_root_resolution_over_lambda': 3e-7,
              'conditions':CONDITIONS,'maximum_errors':ERRORS,'cases':cases,'axis_covariance':covariance}
    path=ROOT/('inverse-current-frozen-results-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')+'.json')
    path.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'report':str(path),'conditions':len(CONDITIONS),'evaluations':sum(CONDITIONS.values()),
                      'growth_at_wave_number_one':[{k:c[k] for k in ('name','depth','retained_complex_dimension')}|
                         {'growth_over_lambda':next(p['maximum_growth_over_lambda'] for p in c['probes'] if p['dimensionless_wave_number']==1.)}
                         for c in cases]},indent=2),flush=True)


if __name__=='__main__':
    main()
