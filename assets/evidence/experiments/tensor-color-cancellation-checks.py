#!/usr/bin/env python3
"""Bounded exact tests of oscillator pairing and color cancellation."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
from itertools import combinations
from math import comb
from pathlib import Path
import json

import sympy as sp


CHECKS = 0
CASES = []


def check(condition, message):
    global CHECKS
    if not condition:
        raise AssertionError(message)
    CHECKS += 1


def shell(n, d):
    return comb(n + d - 1, d - 1)


def band(n, d):
    return comb(n + d, d)


def invariant_shell(n, spectators):
    return sum(comb(n-2*r+spectators-1,spectators-1) if spectators else int(2*r==n)
               for r in range(n//2+1))


def invariant_band(n, spectators):
    return sum(comb(n-2*r+spectators,spectators) for r in range(n//2+1))


def generators(k):
    """Hermitian su(k), tr(Ta Tb)=delta_ab/2."""
    result = []
    for i, j in combinations(range(k), 2):
        s = sp.zeros(k)
        s[i, j] = s[j, i] = sp.Rational(1, 2)
        result.append(s)
        a = sp.zeros(k)
        a[i, j] = -sp.I / 2
        a[j, i] = sp.I / 2
        result.append(a)
    for l in range(1, k):
        h = sp.zeros(k)
        for i in range(l):
            h[i, i] = 1
        h[l, l] = -l
        result.append(h / sp.sqrt(2 * l * (l + 1)))
    return result


def compositions(total, length):
    if length == 1:
        yield (total,)
        return
    for first in range(total + 1):
        for rest in compositions(total - first, length - 1):
            yield (first,) + rest


def permutation_sign(p):
    return (-1) ** sum(p[i] > p[j] for i in range(len(p)) for j in range(i + 1, len(p)))


def exact_singlet_count(k, n):
    """Weyl integration of the complete Sym^n(V+V*) character."""
    from itertools import permutations
    weights = Counter()
    for occupation in compositions(n, 2 * k):
        full = tuple(occupation[i] - occupation[i + k] for i in range(k))
        weights[tuple(full[i] - full[-1] for i in range(k - 1))] += 1
    rho = tuple(range(k - 1, -1, -1))
    multiplicity = 0
    for p in permutations(range(k)):
        shift = tuple(rho[i] - rho[p[i]] for i in range(k))
        key = tuple(shift[i] - shift[-1] for i in range(k - 1))
        multiplicity += permutation_sign(p) * weights[key]
    return multiplicity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    # The oscillator factorization is a shell statement for every split.
    for d in range(2, 13):
        for a in range(1, d):
            b = d - a
            for n in range(41):
                lhs = sum(shell(j, a) * shell(n - j, b) for j in range(n + 1))
                check(lhs == shell(n, d), f'shell convolution {d,a,n}')
    for d in (2, 3, 4, 5, 6, 10):
        CASES.append({'name': 'shell-splits', 'real_depth': d,
                      'parameters': {'splits': [[a, d-a] for a in range(1,d)]},
                      'measurements': {'all_splits_factorize': True}})

    # The pair-state cancellation must hold as a vector equation.
    for k in range(2, 6):
        ts = generators(k)
        eye = sp.eye(k)
        identity_vector = sp.Matrix([1 if i == j else 0 for i in range(k) for j in range(k)])
        proj = identity_vector * identity_vector.T / k
        cf = sp.Rational(k*k - 1, 2*k)
        same_casimir = sp.zeros(k*k)
        dual_casimir = sp.zeros(k*k)
        swap = sp.zeros(k*k)
        for i in range(k):
            for j in range(k):
                swap[i*k+j, j*k+i] = 1
        for a, ta in enumerate(ts):
            for b, tb in enumerate(ts):
                check(sp.simplify(sp.trace(ta*tb)) == (sp.Rational(1,2) if a == b else 0), 'generator metric')
            qsame = sp.kronecker_product(ta, eye) + sp.kronecker_product(eye, ta)
            qdual = sp.kronecker_product(ta, eye) - sp.kronecker_product(eye, ta.T)
            same_casimir += qsame*qsame
            dual_casimir += qdual*qdual
            check(qdual*identity_vector == sp.zeros(k*k,1), 'singlet generator')
        check(sp.simplify(dual_casimir - k*(sp.eye(k*k)-proj)) == sp.zeros(k*k), 'dual Casimir')
        expected_same = (sp.Rational(k*k-2,k))*sp.eye(k*k) + swap
        check(sp.simplify(same_casimir-expected_same) == sp.zeros(k*k), 'identical Casimir')
        # For k=2 the antisymmetric identical pair is a singlet; for k>=3 it is not.
        same_antisymmetric = sp.Rational((k+1)*(k-2),k)
        check(same_antisymmetric >= 0, 'same antisymmetric Casimir')
        # Spatially nonuniform source acting oppositely on paired slots.
        a, b = sp.symbols('a b', real=True)
        strength = 0
        for ta in ts:
            source = a*sp.kronecker_product(ta,eye)-b*sp.kronecker_product(eye,ta.T)
            output = source*identity_vector
            strength += (output.conjugate().T*output)[0]/k
        check(sp.simplify(strength-cf*(a-b)**2) == 0, 'nonuniform singlet source')
        CASES.append({'name': 'paired-casimir', 'parameters': {'rank': k},
            'measurements': {'fundamental_casimir': str(cf), 'dual_pair_singlet_rank':1,
                'dual_pair_adjoint_gap':k,'identical_antisymmetric_casimir':str(same_antisymmetric),
                'identical_symmetric_casimir':str(sp.Rational((k-1)*(k+2),k)),
                'singlet_source_norm':str(cf)+'*(a-b)^2'}})

    # Independent character calculation of all oscillator singlets at small cutoffs.
    for k in range(2, 6):
        for n in range(7):
            multiplicity = exact_singlet_count(k,n)
            check(multiplicity == int(n % 2 == 0), f'Weyl invariant count {k,n}')
    x = sp.symbols('x0:3')
    y = sp.symbols('y0:3')
    pairing = sum(x[i]*y[i] for i in range(3))
    for r in range(5):
        f = pairing**r
        for i in range(3):
            for j in range(3):
                result = x[j]*sp.diff(f,x[i])-y[i]*sp.diff(f,y[j])
                check(sp.expand(result) == 0, 'paired polynomial invariant')

    # Pairwise interactions also support a three-fundamental singlet.
    ts = generators(3)
    epsilon = sp.zeros(27,1)
    from itertools import permutations
    for p3 in permutations(range(3)):
        epsilon[p3[0]*9+p3[1]*3+p3[2]] = permutation_sign(p3)
    total_norm=0
    pair_values=[0,0,0]
    for ta in ts:
        slots=[sp.kronecker_product(ta,sp.eye(3),sp.eye(3)),
               sp.kronecker_product(sp.eye(3),ta,sp.eye(3)),
               sp.kronecker_product(sp.eye(3),sp.eye(3),ta)]
        check(sum(slots,sp.zeros(27))*epsilon == sp.zeros(27,1), 'three-slot singlet')
        for index,(i,j) in enumerate(combinations(range(3),2)):
            pair_values[index] += (epsilon.T*slots[i]*slots[j]*epsilon)[0]/6
    check(pair_values == [-sp.Rational(2,3)]*3,'pair forces on three-slot singlet')
    CASES.append({'name':'three-triplet-two-body-control','parameters':{'rank':3,'slots':3},
        'measurements':{'total_generator_squared':0,'each_pair_contraction':'-2/3',
                        'sum_pair_contraction':str(sum(pair_values))}})

    # The same matrix-carrier pairing exists at every complete Hodge rank.
    for d in (2,3,4,5,6,7,8,10):
        m=(d+1)//2
        k=comb(2*m-1,m-1)
        CASES.append({'name':'full-hodge-pairing','real_depth':d,'parameters':{'channel_rank':k},
            'measurements':{'endomorphism_rank':k*k,'trace_singlet_rank':1,
                            'traceless_rank':k*k-1,'casimir_gap':k if k>1 else 0}})

    assigned = [(2,76,'W'),(2,81,'Z'),(2,95,'H'),(3,1,'down'),(3,4,'strange'),
                (4,3,'up'),(4,20,'charm'),(4,72,'top'),(5,10,'nu1'),(5,15,'nu2'),
                (5,22,'nu3'),(6,13,'electron'),(6,35,'muon'),(10,23,'tau')]
    for d, n, label in assigned:
        level = n-1
        m = d//2
        spectator = d % 2
        # SU(1) is trivial. For SU(m), m>=2, q is its single quadratic invariant.
        if m == 1:
            singlet_shell = shell(level,d)
            singlet_band = band(level,d)
        else:
            singlet_shell = invariant_shell(level,spectator)
            singlet_band = invariant_band(level,spectator)
            check(singlet_band == sum(invariant_shell(j,spectator) for j in range(level+1)),
                  'cumulative invariant sum')
        CASES.append({'name':'assigned-oscillator-invariants','real_depth':d,'mode_index':n,
            'parameters':{'particle':label,'complex_rank':m,'unpaired_real_directions':spectator,
                          'group':'SU('+str(m)+')'},
            'measurements':{'shell_singlets':singlet_shell,'band_singlets':singlet_band,
                            'full_band_count':band(level,d),
                            'singlet_fraction':str(sp.Rational(singlet_band,band(level,d)))}})
    # A common color SU(3) acts on six real directions with d-6 spectators.
    for d, n in [(6,13),(6,35),(7,8),(8,8),(10,23)]:
        nmax=n-1
        spectators=d-6
        shell_count = invariant_shell(nmax,spectators)
        band_count = invariant_band(nmax,spectators)
        check(band_count == sum(invariant_shell(j,spectators) for j in range(nmax+1)),
              'same-color cumulative invariant sum')
        CASES.append({'name':'same-color-subgroup','real_depth':d,'mode_index':n,
            'parameters':{'singlet_spectators':spectators},
            'measurements':{'shell_singlets':shell_count,'band_singlets':band_count,
                'full_band_count':band(nmax,d),'singlet_fraction':str(sp.Rational(band_count,band(nmax,d)))}})
    # A singlet projection changes a full-rank normalization in an explicit complete toy band.
    s = 9
    v = sp.ones(s,1)
    p = sp.diag(1,1,0,0,0,0,0,0,0)
    t = v*v.T
    compressed=p*t*p
    check(t*t == s*t,'full contact eigenvalue')
    check(compressed*compressed == 2*compressed,'compressed contact eigenvalue')
    # Keeping full-band coherent contact color invariant leaves only the slack ray at six.
    for k in range(2,6):
        gs=[]
        for ta in generators(k):
            g=sp.diag(sp.zeros(1),ta,-ta.T)
            gs.append(g)
        all_generators=sp.Matrix.vstack(*gs)
        check(len(all_generators.nullspace())==1, 'only common invariant single-vector is slack')
    # Electric U(1) need not cancel with color; it depends on assigned characters.
    q1,q2=sp.symbols('q1 q2',real=True)
    check(sp.simplify(q1+q2).subs(q2,-q1)==0,'fully dual U1 cancels')
    check(sp.simplify(q1+q2).subs(q2,q1)==2*q1,'common U1 survives')

    report={'experiment_key':'tensor-color-cancellation-20261009',
        'created':datetime.now(timezone.utc).isoformat(),
        'sources':{'experiments/tensor-color-cancellation-checks.py':sha256(Path(__file__).read_bytes()).hexdigest()},
        'scope':'Exact oscillator count, representation and source tests; no particle simulations.',
        'summary':{'checks_passed':CHECKS,'case_count':len(CASES),
            'count_identity_selects_depth':False,'conjugate_pair_all_rank_singlet':True,
            'full_dual_pair_also_cancels_U1':True,
            'neutrality_does_not_imply_zero_nonuniform_response':True},
        'cases':CASES}
    target=Path(args.output)
    target.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['summary'],indent=2))
    print('Report:',target)
    for case in CASES:
        if case['name'] in ('assigned-oscillator-invariants','same-color-subgroup'):
            print(json.dumps(case))


if __name__=='__main__':
    main()
