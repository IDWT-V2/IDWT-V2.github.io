"""Exact tensor identities of the reduced count carriers; no trajectory simulation."""
from datetime import datetime, timezone
from hashlib import sha256
from importlib.util import spec_from_file_location, module_from_spec
from itertools import combinations
from pathlib import Path
import json

import numpy as np
import sympy as sp

ROOT = Path(__file__).resolve().parent
spec = spec_from_file_location('carrier', ROOT/'minimal-count-carrier.py')
carrier = module_from_spec(spec)
spec.loader.exec_module(carrier)
CHECKS = {}


def check(value, label):
    if not bool(value):
        raise AssertionError(label)
    CHECKS[label] = CHECKS.get(label, 0)+1


def exact(matrix):
    return sp.Matrix(np.asarray(matrix).tolist()).applyfunc(lambda z: sp.nsimplify(z))


def expectation(seed, matrix):
    return sp.simplify(sp.trace(seed.conjugate().T*matrix*seed)/sp.trace(seed.conjugate().T*seed))


def main():
    sectors = []
    for depth in [2, 3, 4, 5, 6, 10]:
        alpha, beta, gammas, seed_float, tensor = carrier.preparation(depth)
        gammas = [exact(g) for g in gammas]
        # Recover Gaussian-integer columns so every subsequent calculation is exact.
        raw = np.rint(seed_float.real/np.max(np.abs(seed_float))).astype(int)+1j*np.rint(
            seed_float.imag/np.max(np.abs(seed_float))).astype(int)
        seed = exact(raw)
        dimension = depth+1
        eye = sp.eye(dimension)
        vectors = sp.eye(dimension).columnspace()
        forms = [sp.Matrix(dimension, dimension, lambda i, j: int(round(tensor[k, i, j])))
                 for k in range(dimension)]
        for i, j, k in combinations(range(dimension), 3):
            check(forms[i][j, k] == expectation(seed, sp.I*gammas[i]*gammas[j]*gammas[k]),
                  'every three-form coefficient is verified directly in exact rational Clifford arithmetic')
        for gamma in gammas+[sp.I*gammas[i]*gammas[j] for i, j in combinations(range(dimension), 2)]:
            check(expectation(seed, gamma) == 0, 'every seed balance moment is independently verified in exact rational arithmetic')
        if depth in [5, 6]:
            conjugation = sp.eye(len(beta))
            for i in ([0, 2, 4] if depth == 5 else [0, 2, 4, 6]):
                conjugation *= gammas[i]
            check(conjugation*conjugation.conjugate() == sp.eye(len(beta)),
                  'the spinor real structure is an exact antiunitary involution')
            for gamma in gammas:
                check(conjugation*gamma.conjugate() == -gamma*conjugation,
                      'the real structure anticommutes with every Clifford vector and commutes with every real rotation')
            real_seed = (1-sp.I)*seed if depth == 5 else seed
            check((conjugation*real_seed.conjugate()-real_seed).applyfunc(sp.simplify) == sp.zeros(*real_seed.shape),
                  'the balanced pure seed belongs to the invariant eight-dimensional real spinor space after one phase choice')
        if depth == 5:
            chirality = sp.I*sp.prod(gammas)
            check(chirality == chirality.conjugate().T and chirality**2 == sp.eye(len(beta)),
                  'six-generator chirality is an exact Hermitian involution')
            complex_form = sp.Matrix(dimension, dimension, lambda i, j: 0 if i == j else
                                    expectation(seed, -sp.I*chirality*gammas[i]*gammas[j]))
            check(complex_form.T == -complex_form and complex_form**2 == -eye,
                  'depth-five seed defines an exact orthogonal complex structure on its Clifford vector space')
        elif depth == 10:
            axis = vectors[0]
            complex_form = forms[0]
            check(complex_form**2 == -eye+axis*axis.T,
                  'depth-ten seed defines five equal planes perpendicular to its distinguished three-form axis')
            for k in range(dimension):
                for i in range(dimension):
                    for j in range(dimension):
                        check(forms[k][i, j] == axis[k]*complex_form[i, j]+
                              axis[i]*complex_form[j, k]+axis[j]*complex_form[k, i],
                              'the depth-ten three-form is exactly the axis one-form wedged with its perpendicular two-form')
        if depth in [5, 6]:
            def target(v):
                result = v*v.T-(v.T*v)[0]*eye
                if depth == 5:
                    result += (complex_form*v)*(complex_form*v).T
                return result
            for i in range(dimension):
                check(forms[i]**2 == target(vectors[i]), 'exact diagonal coefficients of the normal-contraction identity')
                for j in range(i+1, dimension):
                    check(forms[i]*forms[j]+forms[j]*forms[i] == target(vectors[i]+vectors[j])-
                          target(vectors[i])-target(vectors[j]),
                          'exact mixed coefficients prove the contraction identity for every normal, not only sampled normals')
            if depth == 5:
                for i in range(dimension):
                    for j in range(dimension):
                        check(forms[i]*complex_form*vectors[j]+forms[j]*complex_form*vectors[i] == sp.zeros(dimension, 1),
                              'the depth-five contraction annihilates the pattern-defined complex normal direction')
        if depth in [5, 6, 10]:
            dims = {}
            for group, indices in [('ambient', range(dimension)), ('physical', range(1, dimension))]:
                products = [gammas[i]*gammas[j]*seed for i, j in combinations(indices, 2)]
                columns = [sp.Matrix(list(p)).applyfunc(sp.re).col_join(sp.Matrix(list(p)).applyfunc(sp.im)) for p in products]
                real_map = sp.Matrix.hstack(*columns)
                kernel = real_map.nullspace()
                dims[group] = len(kernel)
                check(len(kernel) == {(5, 'ambient'): 8, (5, 'physical'): 3,
                                      (6, 'ambient'): 14, (6, 'physical'): 8,
                                      (10, 'ambient'): 24, (10, 'physical'): 24}[(depth, group)],
                      'exact real infinitesimal seed stabilizer dimension')
                for coefficients in kernel:
                    check(real_map*coefficients == sp.zeros(real_map.rows, 1),
                          'every exact stabilizer basis element annihilates the actual seed')
            if depth == 10:
                gram = sp.Matrix(dimension, dimension, lambda i, j: sum(
                    forms[a][i, b]*forms[a][j, b] for a in range(dimension) for b in range(dimension))/2)
                check(gram == eye+4*axis*axis.T, 'the depth-ten three-form determines its axis by a simple isolated metric eigenvalue')
                # Symbolic arbitrary normal: direct trace and minimal polynomial, not random testing.
                x = sp.Matrix(sp.symbols('x0:'+str(dimension), real=True))
                contraction = sum((x[i]*forms[i] for i in range(dimension)), sp.zeros(dimension))
                norm_squared, t = (x.T*x)[0], x[0]
                check(contraction*x == sp.zeros(dimension, 1), 'the normal is always in the contraction kernel')
                check(sp.expand(sp.trace(contraction**2)+2*(norm_squared+4*t*t)) == 0,
                      'exact depth-ten rotation trace fixes one full-strength pair and four axial pairs')
                polynomial = contraction*(contraction**2+norm_squared*eye)*(contraction**2+t*t*eye)
                check(polynomial.applyfunc(sp.expand) == sp.zeros(dimension),
                      'exact depth-ten contraction minimal polynomial holds for every normal')
            sectors.append(dict(depth=depth, stabilizer_dimensions=dims))
        print(json.dumps(dict(completed_depth=depth, evaluations=sum(CHECKS.values()))), flush=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    paths = [Path(__file__), ROOT/'minimal-count-carrier.py', ROOT/'forward-count-action.py',
             ROOT/'inverse-current-radial.py', ROOT/'excited-shell-orientation-current.py']
    report = dict(experiment_key='count-carrier-exact-rotation-identities', created_utc=datetime.now(timezone.utc).isoformat(),
                  sources={'experiments/'+p.name: sha256(p.read_bytes()).hexdigest() for p in paths},
                  evaluations=sum(CHECKS.values()), conditions=CHECKS, sectors=sectors,
                  limits=['the normal-contraction identities describe local spin density, not total particle spin',
                          'the Clifford vector space includes the grading direction, not an extra physical spatial depth',
                          'the depth-five kernel axis is local; fixed encounter-axis dynamics is not derived',
                          'the seed preparation and sphere geometry remain conditional'])
    path = ROOT/('count-carrier-rotation-identities-results-'+stamp+'.json')
    path.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(report=str(path), evaluations=report['evaluations'])), flush=True)


if __name__ == '__main__':
    main()
