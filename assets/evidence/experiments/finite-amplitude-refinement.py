#!/usr/bin/env python3
"""Resolve the stiffest low-spectrum and translation-mode mesh controls."""

from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import importlib.util
import json


ROOT=Path(__file__).resolve().parent
PARENT=ROOT/'finite-amplitude-stability.py'
spec=importlib.util.spec_from_file_location('stability',PARENT)
stability=importlib.util.module_from_spec(spec)
spec.loader.exec_module(stability)


def main():
    cases=[]
    for depth,stiffness in ((2,1600),(3,400),(6,400),(6,1600)):
        omega=0.97
        radial=stability.finite.radial
        seed_case=radial.solve_depth(depth,omega,radius=110.0,tolerance=2e-9)
        if seed_case['status']!='collocation-converged':raise AssertionError(seed_case)
        seed=radial.PROFILES[(depth,omega,110.0,2e-9)]
        result,solution=stability.finite.solve_amplitudes(depth,omega,stiffness,110.0,1e-9,seed)
        if result['status']!='collocation-converged':raise AssertionError(result)
        result['coupled_amplitude_energy_form_spectra']={str(n):{
            'ell'+str(ell):stability.spectrum(solution,depth,omega,stiffness,n,ell)
            for ell in (0,1)} for n in (2200,4400,8800)}
        cases.append(result)
        print(json.dumps({'depth':depth,'stiffness':stiffness,'spectra':result['coupled_amplitude_energy_form_spectra']}),flush=True)
    path=Path(__file__).resolve()
    sources={'experiments/'+p.name:sha256(p.read_bytes()).hexdigest() for p in
             (path,PARENT,stability.PARENT,stability.finite.PARENT)}
    report={'created_at':datetime.now(timezone.utc).isoformat(),
            'experiment':'finite-amplitude-stiff-spectrum-refinement','sources':sources,'cases':cases,
            'scope':'Four selected radial energy-form controls refine the translation zero mode and negative radial index. Finite-domain mesh convergence is numerical evidence, not a certified continuum spectrum or physical growth frequency.'}
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    output=path.with_name('finite-amplitude-refinement-'+stamp+'.json')
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'cases':len(cases),'report':str(output)}),flush=True)


if __name__=='__main__':main()
