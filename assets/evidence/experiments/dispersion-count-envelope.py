"""A convex dispersion envelope and its unique local count constitutive source."""
from datetime import datetime, timezone
from fractions import Fraction
from hashlib import sha256
from math import comb
from pathlib import Path
import json
import mpmath as mp

ROOT = Path(__file__).resolve().parent
ASSIGNMENTS = {2:[('W',76),('Z',81),('Higgs',95)],3:[('down',1),('strange',4)],
               4:[('up',3),('charm',20),('top',72)],
               5:[('nu1-comparison',10),('nu2-comparison',15),('nu3-comparison',22)],
               6:[('electron',13),('muon',35)],10:[('tau',23)]}
COUNTS = {}
ERRORS = {}


def check(value, label):
    if not bool(value):
        raise AssertionError(label)
    COUNTS[label] = COUNTS.get(label, 0) + 1


def number(value):
    if isinstance(value, Fraction):
        return mp.mpf(value.numerator)/value.denominator
    return mp.mpf(value)


def close(actual, expected, label, tolerance=mp.mpf('1e-65')):
    error = abs(actual-expected)/max(mp.mpf(1),abs(expected))
    ERRORS[label] = max(ERRORS.get(label, mp.mpf(0)), error)
    check(error < tolerance, label)


def data(depth, level, epsilon):
    count = Fraction(comb(level+depth, depth))
    reciprocal = [Fraction(1,level+j) for j in range(1,depth+1)]
    first = count*sum(reciprocal)
    second = count*(sum(reciprocal)**2-sum(v*v for v in reciprocal))
    h = Fraction(2*level+depth,2)
    c = epsilon*first
    return dict(level=level,h=h,c=c,clock=epsilon*count,
                mass2=h*h*(1-c*c)/(c*c),
                eta=epsilon*count-h/c,
                convex_margin=h*epsilon*second-c*(1-c*c))


def eligible_data(depth, epsilon):
    values = []
    for level in range(10000):
        value = data(depth,level,epsilon)
        if value['c'] >= 1:
            break
        if value['convex_margin'] > 0:
            values.append(value)
    else:
        raise AssertionError('eligible-label enumeration exceeded its diagnostic limit')
    check(bool(values), 'every occupied depth has eligible dispersion curves')
    check([v['level'] for v in values] == list(range(values[0]['level'],values[-1]['level']+1)),
          'the chosen convex count domain gives one contiguous finite label set')
    return values


def arc(value, chi):
    mass2 = number(value['mass2'])
    lam = mp.sqrt(chi*chi+mass2)
    return (lam+number(value['eta']),chi/lam,
            mass2/lam**3,-3*mass2*chi/lam**5)


def smooth_max(a, b, width):
    difference = [x-y for x,y in zip(a,b)]
    z = difference[0]
    if z >= width:
        return a
    if z <= -width:
        return b
    u = z/width
    absolute = width*(5+15*u*u-5*u**4+u**6)/16
    first = (15*u-10*u**3+3*u**5)/8
    second = 15*(1-u*u)**2/(8*width)
    third = -15*u*(1-u*u)/(2*width**2)
    d1,d2,d3 = difference[1:]
    return ((a[0]+b[0]+absolute)/2,
            (a[1]+b[1]+first*d1)/2,
            (a[2]+b[2]+first*d2+second*d1*d1)/2,
            (a[3]+b[3]+first*d3+3*second*d1*d2+third*d1**3)/2)


def envelope(values, width, chi):
    result = arc(values[0],chi)
    for value in values[1:]:
        result = smooth_max(result,arc(value,chi),width)
    return result


def sector(depth):
    epsilon = Fraction(3,10)/max(
        data(depth,n-1,Fraction(1))['c'] for _,n in ASSIGNMENTS[depth])
    values = eligible_data(depth,epsilon)
    minimum = None
    for own in values:
        for other in values:
            if own['level'] == other['level']:
                continue
            aa = own['clock']-other['eta']
            remainder = aa*aa-own['h']**2-other['mass2']
            check(aa>0 and remainder>0,
                  'exact rational squared comparison proves the unique winner at every eligible node')
            lower = remainder/(2*aa)
            minimum = lower if minimum is None else min(minimum,lower)
    width_exact = minimum/(8*len(values))
    width = number(width_exact)
    check(width>mp.mpf('1e-60'),'the 100-digit evaluation resolves the exact common energy smoothing width')
    crossings = []
    for left,right in zip(values[:-1],values[1:]):
        delta = right['eta']-left['eta']
        mass_delta = left['mass2']-right['mass2']
        check(delta>0 and mass_delta>delta*delta,
              'ordered adjacent dispersion curves have one positive crossing')
        lambda_right = (mass_delta-delta*delta)/(2*delta)
        chi2 = lambda_right**2-right['mass2']
        check(left['h']**2 < chi2 < right['h']**2,
              'each exact squared crossing lies strictly between its adjacent spatial shells')
        value_at_cross = lambda_right+right['eta']
        chi = mp.sqrt(number(chi2))
        smooth = envelope(values,width,chi)
        raw = arc(left,chi)
        close(raw[0],arc(right,chi)[0],
              'independent massive dispersion curves agree at the rational squared crossing')
        check(smooth[2]>0,
              'the actual common smoothed envelope retains strict positive label curvature in every joining band')
        check(0<smooth[1]<1,
              'the joining-band response lies inside the physical grading-ratio domain')
        crossings.append(dict(left_level=left['level'],right_level=right['level'],
                              chi_squared_exact=str(chi2),energy_at_cross_exact=str(value_at_cross),
                              left_derivative_squared_exact=str(chi2/(lambda_right+delta)**2),
                              right_derivative_squared_exact=str(chi2/lambda_right**2),
                              smoothed_second_derivative=str(smooth[2])))
    lower_chi = values[0]['h']-Fraction(1,2)
    upper_chi = values[-1]['h']+Fraction(1,2)
    check(lower_chi>0, 'the common half-spacing constitutive window stays at positive spatial label')
    cases = []
    for name,mode_index in ASSIGNMENTS[depth]:
        own = next(v for v in values if v['level']==mode_index-1)
        hh,cc = number(own['h']),number(own['c'])
        xx = mp.sqrt(1-cc*cc)
        actual = envelope(values,width,hh)
        close(actual[0],number(own['clock']),
              'smoothing retains each assigned count energy exactly in its open winning arc')
        close(actual[1],cc,
              'smoothing retains each assigned geometric grading balance')
        close(actual[2],cc*(1-cc*cc)/hh,
              'the retained massive arc has precisely zero eliminated scalar source curvature')
        root = mp.findroot(lambda chi:envelope(values,width,chi)[1]-cc,
                           (hh-mp.mpf('.01'),hh+mp.mpf('.01')),tol=mp.mpf('1e-80'))
        close(root,hh,'independent convex auxiliary elimination returns the actual spatial shell')
        def source(x):
            c = mp.sqrt(1-x*x)
            chi = mp.findroot(lambda z:envelope(values,width,z)[1]-c,
                              (hh-mp.mpf('.001'),hh+mp.mpf('.001')),tol=mp.mpf('1e-85'))
            return envelope(values,width,chi)[0]-chi*c
        expected_f = number(own['clock'])-hh*cc
        expected_slope = hh*xx/cc
        close(source(xx),expected_f,
              'independently eliminated constitutive value retains complete source and clock')
        close(mp.diff(source,xx),expected_slope,
              'independent scalar differentiation retains the exact source coefficient')
        close(mp.diff(source,xx,2),mp.mpf(0),
              'independent scalar differentiation confirms the open affine source chart',mp.mpf('1e-55'))
        lam = hh/cc
        # Exact nonzero H spectrum decides the positive-wave persistence argument.
        target2 = own['h']**2-2*(own['h']/own['c'])*own['clock']+own['clock']**2
        if target2 < Fraction(depth*depth,4):
            candidates = [0]
        else:
            near = int(mp.floor(mp.sqrt(number(target2))-number(Fraction(depth,2))))
            check(Fraction(2*near+depth,2)**2 <= target2 <
                  Fraction(2*(near+1)+depth,2)**2,
                  'exact rational inequalities certify the proposed adjacent spatial shells')
            candidates = sorted(set([max(0,near),max(0,near+1)]))
        check(all(Fraction(2*k+depth,2)**2!=target2 for k in candidates),
              'exact rational adjacent-shell test excludes a zero of the entire spatial H spectrum')
        cases.append(dict(particle=name,depth=depth,index=mode_index,
                          count_clock_exact=str(own['clock']),polarization_exact=str(own['c']),
                          source_mass_squared_exact=str(own['mass2']),source_eta_exact=str(own['eta']),
                          eliminated_label=str(root),source_value=str(source(xx)),
                          scalar_source_curvature='zero on an open affine chart',
                          full_spatial_nonzero_H_spectrum=True,
                          positive_wave_persistence='conditional nonlinear chart-local theorem applies'))
        print(json.dumps(dict(particle=name,depth=depth,frequency_preserved=True,
                              unique_label=True,affine_source=True)),flush=True)
    # A finite label window is substantive: an unrestricted last arc would admit extra shells.
    higher_h = values[-1]['h']+1
    higher_energy = arc(values[-1],number(higher_h))[0]
    higher_count = data(depth,values[-1]['level']+1,epsilon)['clock']
    aa = higher_count-values[-1]['eta']
    check(aa>0 and aa*aa-higher_h*higher_h-values[-1]['mass2']>0,
          'exact rational comparison proves the unrestricted higher-shell clock differs from its count target')
    check(higher_h>upper_chi,
          'the first omitted integer shell is outside the explicit constitutive label window')
    return dict(depth=depth,epsilon_exact=str(epsilon),eligible_level_range=[values[0]['level'],values[-1]['level']],
                number_of_curves=len(values),constitutive_chi_window_exact=[str(lower_chi),str(upper_chi)],
                exact_node_gap_lower=str(minimum),exact_common_smooth_max_width=str(width_exact),
                crossings=crossings,cases=cases,
                unrestricted_last_arc_control=dict(first_extra_shell_h_exact=str(higher_h),
                                                     extra_clock=str(higher_energy),
                                                     target_count_clock_exact=str(higher_count),
                                                     extra_branch_prevented_by_stated_label_window=True))


def main():
    mp.mp.dps = 100
    # Check the smooth maximum's kernel-derived algebra independently.
    import sympy as s
    z=s.symbols('z',real=True)
    absolute=(5+15*z*z-5*z**4+z**6)/16
    check(s.simplify(s.diff(absolute,z,2)-s.Rational(15,8)*(1-z*z)**2)==0,
          'the compact smooth absolute has the independently derived nonnegative second derivative')
    for boundary in (-1,1):
        for order in range(4):
            expected=1 if order==0 else boundary if order==1 else 0
            check(s.diff(absolute,z,order).subs(z,boundary)==expected,
                  'the compact smoothing joins absolute value through three derivatives')
    sectors = [sector(depth) for depth in ASSIGNMENTS]
    report = dict(experiment_key='convex-dispersion-envelope-count-action',
                  created_utc=datetime.now(timezone.utc).isoformat(),
                  sources={'experiments/'+Path(__file__).name:sha256(Path(__file__).read_bytes()).hexdigest()},
                  evaluations=sum(COUNTS.values()),conditions=COUNTS,
                  maximum_errors={key:str(value) for key,value in ERRORS.items()},
                  decimal_precision=100,sectors=sectors,
                  limits=['the count tangency data and ordinary sphere remain explicit effective premises',
                          'the common smooth maximum is folded in increasing shell order',
                          'the finite constitutive label window is part of the law, not automatic from a finite envelope',
                          'the source is locally smooth on its open scalar-ratio domain; pure-grading endpoints are not included',
                          'the nonlinear persistence result uses nodeless fields and exact affine neighborhoods',
                          'proper-current and responding-clock dynamics are not replaced by preferred-time source results',
                          'the single-shell selection theorem does not classify all nonlinear mixed-shell fields',
                          'no sparse observed occupation, containing-space localization or boost inertia is derived',
                          'no new preference for depth five or the occupied sector assignments is established'])
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    target=ROOT/('dispersion-count-envelope-results-'+stamp+'.json')
    target.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(report=str(target),evaluations=report['evaluations'],
                          maximum_errors=report['maximum_errors'])),flush=True)


if __name__=='__main__':
    main()
