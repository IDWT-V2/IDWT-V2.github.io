"""Exact polynomial closure and scalar-curvature freedom in the low spatial family."""
from datetime import datetime, timezone
from hashlib import sha256
from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path
import json

import numpy as np
import sympy as sp

ROOT = Path(__file__).resolve().parent
spec = spec_from_file_location('carrier', ROOT/'minimal-count-carrier.py')
carrier = module_from_spec(spec)
spec.loader.exec_module(carrier)
spec = spec_from_file_location('numerical', ROOT/'minimal-carrier-ground-response.py')
numerical = module_from_spec(spec)
spec.loader.exec_module(numerical)
GAMMA, TIME, Z = sp.symbols('scalar_curvature time_eigenvalue z', real=True)


def exact(array):
    return sp.Matrix(np.asarray(array).tolist()).applyfunc(sp.nsimplify)


def coefficient_columns(fields, coordinates):
    vectors, monomials = [], set()
    for field in fields:
        real = [sp.re(sp.expand(value)) for value in field]
        imag = [sp.im(sp.expand(value)) for value in field]
        row = [sp.Poly(value, *coordinates).as_dict() for value in real+imag]
        vectors.append(row)
        for entry in row:
            monomials.update(entry)
    monomials = sorted(monomials)
    return vectors, monomials


def run(depth):
    _, beta, arrays, normalized, _ = carrier.preparation(depth)
    raw = np.rint(normalized.real/np.max(np.abs(normalized))).astype(int)+1j*np.rint(
        normalized.imag/np.max(np.abs(normalized))).astype(int)
    seed, gammas = exact(raw), [exact(g) for g in arrays]
    norm = sp.trace(seed.conjugate().T*seed)
    coordinates = sp.symbols('x0:'+str(depth+1), real=True)
    square_relation = coordinates[0]**2+sum(x*x for x in coordinates[1:])-1
    modulus = sp.Poly(square_relation, coordinates[0])
    def reduce(value):
        value = sp.expand(value)
        return sp.expand(sp.rem(sp.Poly(value, coordinates[0]), modulus).as_expr())
    def reduced(matrix):
        return matrix.applyfunc(reduce)
    def dot(a, b):
        return reduce(sum(sp.conjugate(u)*v for u, v in zip(a, b) if u != 0 and v != 0))
    h, c, x = sp.Rational(depth, 2), sp.Rational(3, 5), sp.Rational(4, 5)
    hd = sum(sp.Rational(1, j) for j in range(1, depth+1))
    lam, mass, kappa = h/c, h*x/c, h/c-c/hd
    fpp = GAMMA*lam/c**2
    gseed = [g*seed for g in gammas]
    gxseed = sum((xi*v for xi, v in zip(coordinates, gseed)), sp.zeros(*seed.shape))
    gxg0seed = sum((xi*g*gseed[0] for xi, g in zip(coordinates, gammas)), sp.zeros(*seed.shape))
    fields = [coordinates[0]*seed, coordinates[0]*gxseed, gseed[0], gxg0seed]
    derivatives = [(h+1)*fields[0]-fields[3], -(h+1)*fields[1]+fields[2], h*fields[2], -h*fields[3]]
    beta_fields = [fields[1], fields[0], fields[3], fields[2]]
    fields += [sp.I*v for v in fields.copy()]
    derivatives += [sp.I*v for v in derivatives.copy()]
    beta_fields += [sp.I*v for v in beta_fields.copy()]
    currents = []
    for xi, g, gs in zip(coordinates, gammas, gseed):
        ggx = sum((xj*g*v for xj, v in zip(coordinates, gseed)), sp.zeros(*seed.shape))
        currents.append(sp.I*(gs+2*ggx-2*xi*seed-xi*gxseed))
    scalar = (1-2*x)*seed+(2-x)*gxseed
    output = []
    for field, derivative, bf in zip(fields, derivatives, beta_fields):
        source = sp.zeros(*seed.shape)
        for vector in currents:
            overlap = sp.re(dot(vector, field)).expand()
            if overlap != 0:
                source += vector*overlap
        source *= 2*kappa/(5*norm)
        source += 2*fpp/(5*norm)*scalar*sp.re(dot(scalar, field)).expand()
        output.append(reduced(-sp.I*(derivative+mass*bf-lam*field+source)))
    fields = [reduced(v) for v in fields]
    field_columns, field_monomials = coefficient_columns(fields, coordinates)
    output_columns, output_monomials = coefficient_columns(output, coordinates)
    monomials = sorted(set(field_monomials+output_monomials))
    rows = [(component, monomial) for component in range(2*seed.rows*seed.cols) for monomial in monomials]
    basis = sp.Matrix([[column[component].get(monomial, 0) for column in field_columns] for component, monomial in rows])
    action = sp.Matrix([[column[component].get(monomial, 0) for column in output_columns] for component, monomial in rows])
    independent_rows = basis.T.rref()[1]
    if len(independent_rows) != 8:
        return dict(depth=depth, exact_rank=len(independent_rows), exact_closure=False,
                    classification='the fixed-label eight-field basis is dependent; no eigenvalue claim')
    selected = list(independent_rows)
    matrix = basis[selected, :].inv()*action[selected, :]
    residual = (basis*matrix-action).applyfunc(sp.expand)
    leakage = sum(value != 0 for value in residual)
    if leakage:
        return dict(depth=depth, exact_rank=8, exact_closure=False, nonzero_residual_coefficients=leakage,
                    first_residual=str(next(value for value in residual if value != 0)),
                    classification='the fixed-label family is not exactly closed; no projected eigenvalue claim')
    # Verify the exact operator against independent pointwise NumPy sources,
    # including nonzero curvature rather than only the original fitted block.
    rng = np.random.default_rng(202610120940+depth)
    maximum_error = 0.
    for _ in range(5):
        point = rng.normal(size=depth+1); point /= np.linalg.norm(point)
        data, action_at_point = numerical.at_point(point, arrays, normalized, float(h), float(mass), float(lam),
                                                   float(kappa), 2/np.sqrt(5), 1/np.sqrt(5))
        count = 2*(2+4*(depth+1))
        indices = [2, 3, 4, 5]+[count+2, count+3, count+4, count+5]
        small_data, small_action = data[:, indices], action_at_point[:, indices]
        gx = sum(xi*g for xi, g in zip(point, arrays))
        back = (2*np.eye(len(gx))+gx)@normalized/np.sqrt(5)
        scalar_vector = (gx-.8*np.eye(len(gx)))@back
        real_fields = [point[0]*normalized, point[0]*gx@normalized, arrays[0]@normalized, gx@arrays[0]@normalized]
        real_fields += [1j*v for v in real_fields.copy()]
        for trial in [sp.Rational(-3, 2), 0, sp.Rational(5, 4)]:
            fpp_numeric = float(trial*lam/c**2)
            changes = [-2j*fpp_numeric*scalar_vector*np.vdot(scalar_vector, field).real for field in real_fields]
            changes = np.column_stack([v.ravel() for v in changes])
            changes = np.vstack([changes.real, changes.imag])
            expected = small_action+changes
            actual = small_data@np.array(matrix.subs(GAMMA, trial), float)
            error = float(np.linalg.norm(actual-expected)/max(1., np.linalg.norm(expected)))
            assert error < 2e-12, (depth, trial, error)
            maximum_error = max(maximum_error, error)
    characteristic = matrix.charpoly(TIME)
    polynomial = sp.Poly(characteristic.as_expr().subs(characteristic.gen, TIME), TIME)
    print(json.dumps(dict(depth=depth, exact_closure=True, degree=polynomial.degree(),
                          characteristic=str(sp.factor(polynomial.as_expr())))), flush=True)
    assert polynomial.degree() == 8 and all(polynomial.nth(k) == 0 for k in [0, 1, 3, 5, 7])
    cubic = sp.Poly(Z**3-polynomial.nth(6)*Z**2+polynomial.nth(4)*Z-polynomial.nth(2), Z)
    elementary = [-cubic.nth(2), cubic.nth(1), -cubic.nth(0)]
    discriminant = sp.factor(sp.discriminant(cubic.as_expr(), Z))
    conditions = [value >= 0 for value in elementary]+[discriminant >= 0]
    allowed = sp.reduce_inequalities(conditions, GAMMA)
    disc_poly = sp.Poly(discriminant, GAMMA)
    sum_bound = sp.solve(elementary[0], GAMMA)[0]
    certificates = dict(discriminant_real_root_count=int(disc_poly.count_roots(-sp.oo, sp.oo)),
                        necessary_sum_bound=str(sum_bound),
                        discriminant_roots_at_or_below_sum_bound=int(disc_poly.count_roots(-sp.oo, sum_bound)),
                        discriminant_at_zero_sign=int(sp.sign(disc_poly.eval(0))))
    if depth in [6, 10]:
        assert allowed == sp.false
        assert sum_bound > 0 and certificates['discriminant_roots_at_or_below_sum_bound'] == 0
        assert disc_poly.eval(0) < 0
    flat = sp.Poly(cubic.as_expr().subs(GAMMA, 0), Z)
    real_count = int(flat.count_roots(-sp.oo, sp.oo))
    negative_count = int(flat.count_roots(-sp.oo, 0)) if flat.eval(0) != 0 else None
    return dict(depth=depth, exact_rank=8, exact_closure=True, real_generator_matrix=[[str(v) for v in matrix.row(i)] for i in range(8)],
                time_characteristic_polynomial=str(sp.factor(polynomial.as_expr())),
                nonzero_squared_frequency_cubic=str(sp.factor(cubic.as_expr())),
                elementary_symmetric_coefficients=list(map(str, map(sp.factor, elementary))),
                cubic_discriminant=str(discriminant), all_real_scalar_curvature_allowed_set=str(allowed),
                exact_root_certificates=certificates, independent_pointwise_source_maximum_error=maximum_error,
                flat_nonreal_squared_frequency_roots=3-real_count, flat_negative_squared_frequency_roots=negative_count,
                classification='exact closed spatial family for every real scalar curvature; not a full persistence theorem')


def main():
    cases = []
    for depth in [2, 3, 4, 5, 6, 10]:
        case = run(depth)
        cases.append(case)
        print(json.dumps(case), flush=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    paths = [Path(__file__), ROOT/'minimal-count-carrier.py', ROOT/'forward-count-action.py',
             ROOT/'inverse-current-radial.py', ROOT/'excited-shell-orientation-current.py',
             ROOT/'minimal-carrier-ground-response.py']
    report = dict(experiment_key='minimal-carrier-exact-low-harmonic-curvature', created_utc=datetime.now(timezone.utc).isoformat(),
                  sources={'experiments/'+p.name: sha256(p.read_bytes()).hexdigest() for p in paths}, cases=cases,
                  limits=['one common c3/5 ground preparation and the fixed ambient-label low spatial family',
                          'scalar_curvature=f_double_prime*c_squared/lambda varies the local action Hessian while retaining the background',
                          'all polynomial coefficients are reduced exactly modulo sphere norm squared equal to one',
                          'failed closure prevents any projected spectral claim',
                          'allowed coefficients in one low family do not prove all-angular or assigned excited persistence'])
    path = ROOT/('minimal-carrier-exact-curvature-results-'+stamp+'.json')
    path.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(report=str(path))), flush=True)


if __name__ == '__main__':
    main()
