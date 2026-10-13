"""Actual Clifford and spatial-jet checks for the coexact current family."""
from datetime import datetime, timezone
from hashlib import sha256
from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path
from itertools import combinations
import json

import numpy as np

ROOT = Path(__file__).resolve().parent
spec = spec_from_file_location('radial', ROOT/'inverse-current-radial.py')
radial = module_from_spec(spec)
spec.loader.exec_module(radial)
COUNTS, ERRORS = {}, {}


def close(actual, expected, label, tolerance=2e-11):
    error = float(np.linalg.norm(np.asarray(actual)-np.asarray(expected))/max(1., np.linalg.norm(expected)))
    ERRORS[label] = max(ERRORS.get(label, 0.), error)
    if error >= tolerance:
        raise AssertionError((label, error))
    COUNTS[label] = COUNTS.get(label, 0)+1


def main():
    rng = np.random.default_rng(202610120530)
    cases = []
    for depth in radial.DEPTHS:
        matrices = radial.gammas(depth+1)
        alpha, beta = matrices[:depth], matrices[depth]
        rank = len(beta)
        eye = np.eye(rank)
        clifford = []
        for grade in range(depth+1):
            for indices in combinations(range(depth), grade):
                matrix = eye.copy()
                for index in indices:
                    matrix = matrix@alpha[index]
                clifford.append(matrix)
                tvalue = (-1)**grade*(depth-2*grade)/2
                actual = sum(a@matrix@a for a in alpha)/2
                close(actual, tvalue*matrix, 'all exterior grades have the exact Killing-frame connection term')
                close(sum(a@beta@matrix@a for a in alpha)/2, -tvalue*beta@matrix,
                      'parallel beta reverses the complete Killing-frame connection term')
                close(np.trace(matrix), rank if grade == 0 else 0,
                      'all nontrivial Clifford characters have zero trace in the retained carrier')
        if depth % 2:
            clifford += [beta@m for m in clifford]
        close(len(clifford), rank*rank, 'Clifford forms with the stated odd-depth doubling span the complete matrix carrier')
        flat = np.array(clifford).reshape(len(clifford), -1)/np.sqrt(rank)
        # For this monomial Clifford basis the product of distinct monomials
        # is a phase times a nonidentity monomial; the trace check above
        # proves its Gram orthogonality without a dense cubic Gram product.
        close(np.sum(abs(flat)**2, axis=1), np.ones(rank*rank), 'complete Clifford matrix basis has unit diagonal Gram entries')
        if depth % 2:
            close([np.trace(m) for m in clifford[2**depth:]], np.zeros(2**depth),
                  'odd-depth beta monomials remain trace orthogonal to the original Clifford algebra')
        for level in [1, 2]:
            eigenvalue = (level+1)*(level+depth-2)
            for sample in range(3):
                ambient = rng.normal(size=depth+1)
                ambient /= np.linalg.norm(ambient)
                orthogonal, _ = np.linalg.qr(np.column_stack([ambient, rng.normal(size=(depth+1, depth))]))
                tangent = orthogonal[:, 1:]
                rotation = np.zeros((depth+1, depth+1))
                rotation[0, 1], rotation[1, 0] = 1., -1.
                linear = np.zeros(depth+1)
                if level == 2:
                    linear[2] = 1.
                f = np.dot(linear, ambient) if level == 2 else 1.
                vector = f*rotation@ambient
                jacobian = f*rotation+np.outer(rotation@ambient, linear)
                values = tangent.T@vector
                derivative = tangent.T@jacobian@tangent
                derivative = derivative.T  # derivative index first, component index second
                second = np.zeros((depth, depth, depth))
                for k in range(depth):
                    for i in range(depth):
                        ambient_second = (np.dot(linear, tangent[:, k])*rotation@tangent[:, i]
                                          +np.dot(linear, tangent[:, i])*rotation@tangent[:, k])
                        second[k, i, :] = tangent.T@ambient_second
                        if k == i:
                            second[k, i, :] -= level*values
                        second[k, i, k] += values[i]
                close(np.dot(ambient, vector), 0., 'ambient polynomial vector is exactly tangent to the ordinary sphere')
                close(np.trace(derivative), 0., 'the spatial vector harmonic is divergence free')
                rough = np.einsum('iij->j', second)
                close(-rough+(depth-1)*values, eigenvalue*values,
                      'independent ambient second jets verify the coexact Hodge eigenvalue')
                m1 = -1j*sum(v*a for v, a in zip(values, alpha))
                dm1 = [-1j*sum(derivative[k, j]*alpha[j] for j in range(depth)) for k in range(depth)]
                m2 = -sum(alpha[i]@alpha[j]*derivative[i, j] for i in range(depth) for j in range(depth))/np.sqrt(eigenvalue)
                dm2 = [-sum(alpha[i]@alpha[j]*second[k, i, j] for i in range(depth) for j in range(depth))/np.sqrt(eigenvalue)
                       for k in range(depth)]
                d1 = -1j*sum(a@dm for a, dm in zip(alpha, dm1))+sum(a@m1@a for a in alpha)/2
                d2 = -1j*sum(a@dm for a, dm in zip(alpha, dm2))+sum(a@m2@a for a in alpha)/2
                close(d1, (1-depth/2)*m1+np.sqrt(eigenvalue)*m2,
                      'actual full spatial Dirac derivative closes the coexact vector into its two-form')
                close(d2, np.sqrt(eigenvalue)*m1+(depth/2-2)*m2,
                      'actual full spatial Dirac derivative closes the two-form back into its vector')
                if level == 1:
                    positive = m1+np.sqrt((depth-1)/2)*m2
                    close(positive.conj().T, -positive,
                          'first coexact positive Killing endomorphism is anti-Hermitian')
                    for k in range(depth):
                        derivative_positive = dm1[k]+np.sqrt((depth-1)/2)*dm2[k]
                        close(derivative_positive, .5j*(alpha[k]@positive-positive@alpha[k]),
                              'zero coexact endomorphism is a constant right generator in the Killing frame')
                    base = (2*eye+beta)/np.sqrt(5*rank)
                    variation = (2*eye+beta)@positive/np.sqrt(5*rank)
                    close(variation@base.conj().T+base@variation.conj().T, np.zeros_like(base),
                          'particle right-unitary zero mode changes no left density matrix to first order')
                    close(positive+positive.conj().T, np.zeros_like(positive),
                          'background right-unitary zero mode changes no left density matrix to first order')
                if depth >= 3:
                    # The two-form and its beta partner carry no physical
                    # vector trace in this full retained Clifford module.
                    close([np.trace(a@m2) for a in alpha], np.zeros(depth),
                          'coexact two-form carries no direct vector current')
                    close([np.trace(a@beta@m2) for a in alpha], np.zeros(depth),
                          'beta coexact two-form carries no direct vector current')
                    close([np.trace(a@beta@m1) for a in alpha], np.zeros(depth),
                          'unmixed beta vector has no independent vector trace above depth two')
                close([np.trace(a@m1)/rank for a in alpha], -1j*values,
                      'actual Clifford vector trace derives the current-source normalization')
                cases.append(dict(depth=depth, spinor_rank=rank, coexact_degree=level,
                                  sample=sample, hodge_eigenvalue=eigenvalue,
                                  classification='actual tangent vector, covariant second jets and full matrix Dirac closure'))
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    report = dict(experiment_key='covariant-clock-spatial-geometry', created_utc=datetime.now(timezone.utc).isoformat(),
                  sources={'experiments/'+p.name: sha256(p.read_bytes()).hexdigest()
                           for p in [Path(__file__), ROOT/'inverse-current-radial.py']},
                  evaluations=sum(COUNTS.values()), conditions=COUNTS, maximum_errors=ERRORS, cases=cases,
                  limits=['ordinary unit sphere and full balanced matrix carrier are supplied premises',
                          'first two coexact harmonics receive explicit ambient polynomial jet checks',
                          'global Hodge completeness requires the accompanying analytical argument'])
    output = ROOT/('covariant-clock-geometry-results-'+stamp+'.json')
    output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(report=str(output), evaluations=report['evaluations'], maximum_errors=ERRORS)), flush=True)


if __name__ == '__main__':
    main()
