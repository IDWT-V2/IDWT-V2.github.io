"""Small exact checks of the canonically normalized closure obstruction.

No spatial simulation, physical amplitude fit, or database mutation is made.
Run with python3 -B PATH/check.py (installed SymPy). A dated JSON report is written.
"""
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from math import comb
import json
import sympy as s

ROOT = Path(__file__).resolve().parent
checks = []


def check(condition, label):
    if not bool(condition):
        raise AssertionError(label)
    checks.append(label)


def zero(expr, label):
    check(s.simplify(expr) == 0, label)


def main():
    a, b, c, z, mu, inertia, hb = s.symbols('a b c z mu inertia hbar', positive=True)
    G = s.Matrix([[a, b], [b, c]])
    zero((G-z*s.eye(2)).det().subs(b**2, a*c)-z*(z-a-c), 'rank-one full spectral polynomial')
    zero((a-b**2/c).subs(b**2, a*c), 'static relaxed stiffness vanishes')
    zero((b**2/c).subs(b**2, a*c)-a, 'removed relaxation contribution is a')
    zero((mu*a/inertia).subs(inertia, hb**2/(mu*a))-(mu*a/hb)**2,
         'target clamped frequency requires the specified inverse stiffness inertia')

    h, cp, yy, chi, fp2, qnorm, radius = s.symbols('h c y chi Fpp Q R', positive=True)
    F = s.Function('F')
    e = h*cp+F(chi)-chi*s.sqrt(cp**2+yy**2)
    zero(s.diff(e, cp).subs(yy, 0)-(h-chi), 'stationarity retains the spatial kinetic term')
    zero(s.diff(e, chi).subs(yy, 0)-(s.diff(F(chi), chi)-cp), 'auxiliary equation')
    ecc = (s.diff(e, cp, 2)-s.diff(e, cp, chi)**2/s.diff(e, chi, 2)).subs(yy, 0)
    zero(ecc+1/s.diff(F(chi), chi, 2), 'complete eliminated polarization Hessian')
    eyy = s.diff(e, yy, 2).subs({yy: 0, chi: h})
    zero(eyy+h/cp, 'relative phase Hessian')
    # Canonical coefficient dynamics gives {c,y}=-2x/Q, x²=1-c².
    # H=Q e/R, hence the norm cancels in the linear matrix.
    x = s.symbols('x', positive=True)
    J = s.Matrix([[0, -2*x/qnorm], [2*x/qnorm, 0]])
    A = J*(qnorm/radius)*s.diag(-1/fp2, -h/cp)
    expected_squared = 4*x*x*h/(radius**2*cp*fp2)
    check(s.simplify(A*A+expected_squared*s.eye(2)) == s.zeros(2), 'canonical pair frequency and amplitude cancellation')

    # An independent exact local Taylor calculation: sin² has second jet 2π².
    xi, t = s.symbols('xi t', real=True)
    F0, F1, F2, F3 = s.symbols('F0 F1 F2 F3')
    Taylor = F0+F1*xi+F2*xi**2/2+F3*xi**3/6
    deformed = Taylor+t*s.diff(Taylor, xi, 2)*s.sin(s.pi*xi)**2/(2*s.pi**2)
    zero(deformed.subs(xi, 0)-F0, 'analytic deformation preserves value')
    zero(s.diff(deformed, xi).subs(xi, 0)-F1, 'analytic deformation preserves source')
    zero(s.diff(deformed, xi, 2).subs(xi, 0)-(1+t)*F2, 'analytic deformation rescales curvature')

    n, total, hop, kap, v3, v4, e3, e4, phi = s.symbols('n N J kappa v3 v4 e3 e4 phi', real=True)
    weight = v4*total+(v3-v4)*n
    hn = e3*n+e4*(total-n)+kap*weight**2/2-2*hop*s.sqrt(n*(total-n))*s.cos(phi)
    phase_hessian = s.diff(hn, phi, 2).subs(phi, 0)
    population_hessian = s.diff(hn, n, 2).subs(phi, 0)
    expected_pair = 2*hop*kap*(v3-v4)**2*s.sqrt(n*(total-n))+hop**2*total**2/(n*(total-n))
    zero(s.expand(phase_hessian*population_hessian)-expected_pair, 'dimer canonical relative frequency')
    zero(s.diff(hn, phi).subs(hop, 0), 'density-only rank-one action has no phase restoring term')
    n3, n4, r12 = s.symbols('n3 n4 r12', positive=True)
    hh = e3*n3+e4*n4+kap*(v3*n3+v4*n4)**2/2-2*hop*s.sqrt(n3*n4)
    zero(n3*s.diff(hh,n3)+n4*s.diff(hh,n4)-hh-kap*(v3*n3+v4*n4)**2/2,
         'dimer complete energy differs from canonical frequency contraction by quartic energy')

    source = json.loads((ROOT/'results/canonical-count-response-inputs.json').read_text())
    cases = []
    for old in source['cases']:
        depth, index = old['depth'], old['inferred_index']
        level = index-1
        hv = s.Rational(2*level+depth, 2)
        eps = s.Rational(old['epsilon_exact'])
        P = s.prod(chi+j-s.Rational(depth, 2) for j in range(1, depth+1))/s.factorial(depth)
        Fbase = eps*P
        count = s.binomial(level+depth, depth)
        cv = s.diff(Fbase, chi).subs(chi, hv)
        fpp = s.diff(Fbase, chi, 2).subs(chi, hv)
        check(0 < cv < 1 and fpp > 0, f'{old["particle"]}: open convex baseline')
        zero(P.subs(chi, hv)-count, f'{old["particle"]}: count polynomial')
        check(sum(comb(k+depth-1, depth-1) for k in range(level+1)) == count,
              f'{old["particle"]}: independent cumulative shell count')
        om0sq = 4*hv*(1-cv**2)/(cv*fpp)
        row = {'particle': old['particle'], 'depth': depth, 'index': index,
               'h_exact': str(hv), 'count_exact': str(count),
               'epsilon_exact': str(eps), 'polarization_exact': str(cv),
               'particle_frequency_radius_units_exact': str(eps*count),
               'shape_frequency_squared_radius_units_exact': str(om0sq), 'deformations': []}
        for tv in [0, 1, 3]:
            Ft = Fbase+tv*s.diff(Fbase, chi, 2)*s.sin(s.pi*(chi-s.Rational(depth, 2)))**2/(2*s.pi**2)
            fv = Ft.subs(chi, hv)
            f1v = s.diff(Ft, chi).subs(chi, hv)
            f2v = s.diff(Ft, chi, 2).subs(chi, hv)
            zero(fv-eps*count, f'{old["particle"]} t={tv}: unchanged clock')
            zero(f1v-cv, f'{old["particle"]} t={tv}: unchanged field polarization')
            zero(f2v-(1+tv)*fpp, f'{old["particle"]} t={tv}: exact changed curvature')
            omsq = 4*hv*(1-cv**2)/(cv*f2v)
            zero(omsq/om0sq-s.Rational(1, 1+tv), f'{old["particle"]} t={tv}: canonical shape frequency ratio')
            row['deformations'].append({'t': tv, 'shape_frequency_squared_ratio_exact': str(s.Rational(1, 1+tv))})
        cases.append(row)
    check(sorted({v['depth'] for v in cases}) == [2, 3, 4, 5, 6, 10], 'all occupied depths covered')
    check(len(cases) == 14, 'all fourteen maintained massive assignments covered')
    sources = {'experiments/results/canonical-count-response-inputs.json': sha256((ROOT/'results/canonical-count-response-inputs.json').read_bytes()).hexdigest(), 'experiments/canonical-count-response-freedom.py': sha256(Path(__file__).read_bytes()).hexdigest()}
    report = {'created_utc': datetime.now(timezone.utc).isoformat(), 'input_kind':'public diagnostic assignment table',
              'experiment_key':'closure-a-canonical-response-freedom', 'checks_passed':len(checks),
              'checks': checks, 'sources':sources, 'cases':cases,
              'limits':['Exact reduced balanced-pair calculation, not a full spatial perturbation computation.',
                        'No collective baryon clock, particle identification, localization or boost inertia derived.',
                        'Canonical phase norm is arbitrary and unchanged; no measured field amplitude is used.',
                        'The photon n=0 and unresolved bottom branch are not the fourteen single-shell massive assignments.']}
    output=ROOT/('canonical-count-response-freedom-results-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'.json')
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'checks_passed':len(checks),'report':str(output),'depths':[2,3,4,5,6,10]}))


if __name__ == '__main__':
    main()
