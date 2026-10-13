"""Common source-curvature interpolation with unchanged assigned count clocks."""
from datetime import datetime, timezone
from fractions import Fraction
from hashlib import sha256
from pathlib import Path
import importlib.util
import json

ROOT = Path(__file__).resolve().parent
DEPENDENCY = ROOT / 'continuous-count-source-index.py'
spec = importlib.util.spec_from_file_location('source_index', DEPENDENCY)
index = importlib.util.module_from_spec(spec)
spec.loader.exec_module(index)


def main():
    index.mp.mp.dps = 100
    index.mp.iv.dps = 85
    gammas = [Fraction(1,16), Fraction(1,8), Fraction(1,4),
              Fraction(1,2), Fraction(3,4), Fraction(1)]
    controls = []
    for depth, assignments in index.ASSIGNMENTS.items():
        for name, mode_index in assignments:
            pars = index.parameters(depth, mode_index-1)
            q = pars['q']
            pair_limit = 2*(pars['lam']-pars['omega'])/q
            index.check(pair_limit > 1,
                        'the restricted polarization pair remains non-growing throughout the common interpolation')
            second = pars['epsilon']*index.count_data(depth, mode_index-1)[2]
            x2 = 1-pars['c']**2
            for gamma in [Fraction(0), *gammas]:
                adjusted = 1/(gamma/second+(1-gamma)*pars['lam']/x2)
                curvature = pars['lam']/pars['c']**2-x2/(pars['c']**2*adjusted)
                index.check(adjusted > 0,
                            'the explicit forward correction has positive eliminated-label curvature at every node')
                index.check(2*pars['c']**2*curvature == gamma*q,
                            'independent rational elimination gives the same scaled scalar source derivative')
            controls.append(dict(particle=name, depth=depth, index=mode_index,
                                 epsilon_exact=str(pars['epsilon']),
                                 count_clock_exact=str(pars['omega']),
                                 source_norm_at_gamma_one_exact=str(q),
                                 restricted_pair_curvature_limit_exact=str(pair_limit),
                                 all_clocks_unchanged=True,
                                 pair_no_growth_for_zero_through_one=True,
                                 complete_spatial_persistence_established=False))
    certificates = []
    for depth, level, degree in ((3,3,12), (4,2,6)):
        for gamma in gammas:
            result = index.source_certificate(depth, level, degree, gamma, require_growth=False)
            result['curvature_fraction_exact'] = str(gamma)
            certificates.append(result)
            print(json.dumps(dict(depth=depth, level=level, gamma=str(gamma),
                                  real_index_upper=result['real_nonpositive_index_upper_excluding_selected_phase'],
                                  imaginary_index=result['imaginary_negative_index'],
                                  full_growth=result['full_spatial_real_growth_established'])), flush=True)
    bounds = []
    for depth, level in ((3,3), (4,2)):
        passing = [Fraction(x['curvature_fraction_exact']) for x in certificates
                   if x['depth']==depth and x['full_spatial_real_growth_established']]
        bounds.append(dict(depth=depth, level=level,
                           sufficient_continuous_curvature_fraction_for_growth=str(min(passing)),
                           growth_proved_through_gamma_one=True,
                           smaller_gamma_status='not decided by failure of this sufficient lower-source test'))
    sources = {'experiments/'+path.name: sha256(path.read_bytes()).hexdigest()
               for path in (Path(__file__), DEPENDENCY)}
    report = dict(experiment_key='count-preserving-common-source-curvature',
                  created_utc=datetime.now(timezone.utc).isoformat(),
                  sources=sources, evaluations=sum(index.COUNTS.values()),
                  conditions=index.COUNTS, occupied_assignment_controls=controls,
                  certificates=certificates, sufficient_growth_intervals=bounds,
                  interval_decimal_precision=85,
                  limits=['f_gamma=(1-gamma)f_envelope+gamma f_continuous is one local common source interpolation',
                          'count values and first derivatives agree at every eligible node; only curvature changes',
                          'the zero endpoint inherits the separate envelope persistence theorem',
                          'failure of the sufficient index test does not establish stability',
                          'growth intervals use Schur monotonicity and the negative-gap bound at gamma one',
                          'the full-matrix carrier differs from minimum-column particle candidates',
                          'no observed lifetime or new depth preference follows'])
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    target = ROOT / ('count-curvature-response-results-'+stamp+'.json')
    target.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(report=str(target),evaluations=report['evaluations'])),flush=True)


if __name__ == '__main__':
    main()
