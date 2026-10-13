#!/usr/bin/env python3
"""Bound the available rotation clocks without identifying energy with amplitude."""

from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import json
import sympy as sp


def main():
    z=sp.symbols('z',positive=True)
    rows=[]
    for beta,name in ((sp.Rational(1,4),'orientation-only'),(sp.Rational(1),'phase-complete-fixed-norm')):
        y=sp.symbols('y',positive=True)
        h=1-y/4-beta*y*(1-y/4)**2/4
        derivative=sp.factor(sp.diff(h,y))
        at_one=sp.factor(h.subs(y,1))
        rows.append({'action':name,'minimum_omega_squared':str(at_one),
                     'threshold_derivative':str(derivative),'maximum_clock_ratio':str(1/sp.sqrt(at_one)),
                     'maximum_clock_ratio_numeric':float(1/sp.sqrt(at_one))})
    for stiffness in (400,1600):
        v=sp.Rational(stiffness,4)*(z-1)**2+z/2-z*z/8
        h=(2*v-v*v)/z
        numerator=sp.factor(sp.together(sp.diff(h,z))).as_numer_denom()[0]
        candidates=[]
        for root in sp.nroots(numerator,n=50,maxsteps=200):
            if abs(sp.im(root))>1e-10:continue
            value=float(sp.re(root))
            if value>=1 and float(v.subs(z,value))<1:
                candidates.append((float(h.subs(z,value)),value))
        minimum,argmin=min(candidates)
        rows.append({'action':'finite-amplitude','norm_stiffness':stiffness,
                     'threshold_stationary_polynomial':str(numerator),
                     'minimum_omega_squared':minimum,'minimizing_squared_coefficient':argmin,
                     'maximum_clock_ratio_numeric':minimum**-0.5})
    stiffness=sp.symbols('M2',positive=True)
    v=stiffness*(z-1)**2/4+z/2-z*z/8
    certificate=(stiffness-sp.Rational(2,3))*(z-1)**2/4+(z-2)**2/24
    assert sp.expand(v-z/3-certificate)==0
    ratio=sp.Rational(3838380,18564)
    path=Path(__file__).resolve()
    report={'created_at':datetime.now(timezone.utc).isoformat(),
            'experiment':'localized-rotation-clock-window',
            'sources':{'experiments/rotation-clock-window.py':sha256(path.read_bytes()).hexdigest()},
            'cases':rows,'finite_amplitude_exact_bound':{
                'condition':'M_squared >= 2/3, m_squared=1, b=1/4, beta=1',
                'large_norm_potential_certificate':str(certificate),
                'minimum_omega_squared_strict_lower_bound':'1/3',
                'maximum_clock_ratio_strict_upper_bound':'sqrt(3)'},
            'maintained_muon_electron_frequency_ratio':str(ratio),
            'maintained_muon_electron_frequency_ratio_numeric':float(ratio),
            'scope':'Necessary negative-Routhian and decaying-tail conditions bound the common radial rotation frequency window. If this rotation is identified with the assigned particle mass-frequency, these simple branches cannot reproduce electron and muon together. This does not exclude a different same-field spectral mechanism or identify their complete energies with their rotation clock.'}
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    output=path.with_name('rotation-clock-window-'+stamp+'.json')
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    print(str(output))


if __name__=='__main__':main()
