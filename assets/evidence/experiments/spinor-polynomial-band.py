from fractions import Fraction as F
from math import comb
from pathlib import Path
import hashlib
import json

checks = []


def check(name, condition):
    if not condition:
        raise AssertionError(name)
    checks.append(name)


def weyl_dimension(d, degree, chirality=1):
    rank = d // 2
    weight = [F(1, 2)] * rank
    weight[0] += degree
    if d % 2 == 0:
        weight[-1] *= chirality
    rho = [F(d, 2) - i - 1 for i in range(rank)]
    value = F(1)
    for i in range(rank):
        if d % 2:
            value *= (weight[i] + rho[i]) / rho[i]
        for j in range(i + 1, rank):
            value *= (weight[i] + rho[i] - weight[j] - rho[j]) / (rho[i] - rho[j])
            value *= (weight[i] + rho[i] + weight[j] + rho[j]) / (rho[i] + rho[j])
    check(f'integer Weyl dimension d{d} l{degree} c{chirality}', value.denominator == 1)
    return int(value)


def decomposition(d, n):
    degree = n - 1
    spinor_rank = {7: 8, 8: 8, 9: 16}[d]
    blocks = []
    for l in range(degree + 1):
        dimension = spinor_rank * comb(l + d - 2, d - 2)
        if d % 2:
            blocks.append((l, 'single', degree - l + 1, dimension))
        else:
            plus = (degree - l) // 2 + 1
            minus = (degree - l + 1) // 2
            blocks.append((l, '+', plus, dimension))
            if minus:
                blocks.append((l, '-', minus, dimension))
    total = sum(m * dimension for _, _, m, dimension in blocks)
    kernel = sum((m % 2) * dimension for _, _, m, dimension in blocks)
    skew_dimension = sum(m * (m - 1) // 2 for _, _, m, _ in blocks)
    return dict(d=d, n=n, N=degree, S=comb(n+d-1,d), total_rank=total,
                minimum_kernel=kernel, maximum_skew_rank=total-kernel,
                skew_commutant_dimension=skew_dimension,
                kernel_fraction=str(F(kernel,total)),
                blocks=[dict(l=l,chirality=c,multiplicity=m,rank=r) for l,c,m,r in blocks])


for d in (7, 8, 9):
    spinor_rank = {7: 8, 8: 8, 9: 16}[d]
    for degree in range(96):
        expected = spinor_rank * comb(degree+d-2,d-2)
        check(f'Weyl versus monogenic d{d} l{degree}', weyl_dimension(d,degree)==expected)
        if d == 8:
            check(f'chiral equal dimension l{degree}', weyl_dimension(d,degree,-1)==expected)
        case = decomposition(d,degree+1)
        check(f'full actual count d{d} N{degree}', case['total_rank']==spinor_rank*comb(degree+d,d))
        check(f'nonzero unavoidable kernel d{d} N{degree}', case['minimum_kernel']>=expected)
        check(f'achievable even maximum skew rank d{d} N{degree}', case['maximum_skew_rank']%2==0)
        check(f'odd real color multiplicity preserves kernel d{d} N{degree}',
              sum((3*b['multiplicity']%2)*b['rank'] for b in case['blocks'])==case['minimum_kernel'])
        check(f'complex color six real copies remove kernel d{d} N{degree}',
              sum((6*b['multiplicity']%2)*b['rank'] for b in case['blocks'])==0)

labels = [('W',2,76),('Z',2,81),('Higgs',2,95),('down',3,1),('strange',3,4),
          ('up',4,3),('charm',4,20),('top',4,72),('nu1',5,10),('nu2',5,15),
          ('nu3',5,22),('electron',6,13),('muon',6,35),('tau',10,23)]
controls=[]
for name, assigned_d, n in labels:
    for d in (7,8,9):
        case=decomposition(d,n)
        case.pop('blocks')
        case.update(label=name,assigned_d=assigned_d,control_only=True)
        controls.append(case)

report=dict(status='pass',check_count=len(checks),checks=checks,
            examples=[decomposition(d,n) for d in (7,8,9) for n in (1,2,3,4,13,23,35)],
            actual_index_controls=controls,
            script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            scope='Exact dimension and multiplicity checks. Irreducibility, real type and invariant-form conclusions require the analytic proof; no physical lifetime or selection of observed indices is inferred.')
Path(__file__).with_name('polynomial-spinor-results.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(dict(status='pass',checks=len(checks),controls=len(controls))),flush=True)
