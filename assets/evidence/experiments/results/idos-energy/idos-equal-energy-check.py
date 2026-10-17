#!/usr/bin/env python3
"""Bounded exact checks of the conservative mismatch action; no DB writes."""
from pathlib import Path
from math import comb
from fractions import Fraction as Q
import sympy as s
import json,hashlib,subprocess
from datetime import datetime, timezone
OUT=Path(__file__).resolve().parent
checks=[]
def check(name,value):
    assert value,name
    checks.append(name)
I=s.symbols('I0:3',positive=True); th=s.symbols('t0:3',real=True)
k,eta,eps=s.symbols('k eta eps',positive=True)
nu=(1,2,3); r=(1,-2,1)
e=[nu[j]*I[j] for j in range(3)]; H0=sum(e); mean=H0/3
C=sum((v-mean)**2 for v in e)/(2*eps)
psi=sum(r[j]*th[j] for j in range(3)); F=1+eta*s.cos(psi)
H=H0+k*C*F
Idot=[-s.diff(H,t) for t in th]; tdot=[s.diff(H,a) for a in I]
check('changed amplitude Euler identity',s.simplify(2*sum(I[j]*tdot[j] for j in range(3))-2*H0-4*k*C*F)==0)
check('H0 conservation',s.simplify(sum(nu[j]*Idot[j] for j in range(3)))==0)
check('total action conservation',s.simplify(sum(Idot))==0)
check('complete H conservation',s.simplify(sum(s.diff(H,I[j])*Idot[j]+s.diff(H,th[j])*tdot[j] for j in range(3)))==0)
uniform={I[j]:eps/nu[j] for j in range(3)}
for j in range(3):
    check(f'uniform invariant {j}',s.simplify(Idot[j].subs(uniform))==0)
    check(f'uniform rate {j}',s.simplify(tdot[j].subs(uniform)-nu[j])==0)
    check(f'complete canonical phase formula {j}',s.simplify(tdot[j]-nu[j]*(1+k*(e[j]-mean)*F/eps))==0)
point={I[0]:s.Rational(11,10),I[1]:s.Rational(3,10),I[2]:s.Rational(13,30),th[0]:s.pi/2,th[1]:0,th[2]:0,eps:1,k:1,eta:s.Rational(1,2)}
flow=[s.simplify(v.subs(point)) for v in Idot]
check('nonzero reciprocal exchange',flow==[s.Rational(13,200),-s.Rational(13,100),s.Rational(13,200)])
check('same H0 as uniform',H0.subs(point)==3)
check('same action as uniform',sum(I).subs(point)==s.Rational(11,6))
# Exact generic local U(1)-invariant quartic counterexample.
g,B,A,Bb=s.symbols('g B A Bb',positive=True)
Hloc=g*(A*I[0]**2+Bb*I[1]**2+4*B*I[0]*I[1]+2*B*I[0]*I[1]*s.cos(2*(th[0]-th[1])))
difference=-nu[0]*s.diff(Hloc,th[0])+nu[1]*s.diff(Hloc,th[1])
check('local quartic breaks equal energy',s.simplify(difference-4*g*B*(nu[0]+nu[1])*I[0]*I[1]*s.sin(2*(th[0]-th[1])))==0)
labels=[('W',2,75),('Z',2,80),('Higgs',2,94),('down',3,0),('strange',3,3),('up',4,2),('charm',4,19),('top',4,71),('nu1',5,9),('nu2',5,14),('nu3',5,21),('electron',6,12),('muon',6,34),('tau',10,22)]
rows=[]
for name,d,N in labels:
    mult=[comb(a+d-1,d-1) for a in range(N+1)]
    D=comb(N+d,d); inv=sum((Q(v,a+1) for a,v in enumerate(mult)),Q(0))
    check(f'count {name}',sum(mult)==D)
    check(f'equal action extra factor {name}',sum(Q((a+1)*v,D) for a,v in enumerate(mult))==1+Q(d*N,d+1))
    check(f'positive mode capacity {name}',D>0 and inv>0)
    rows.append({'name':name,'d':d,'N':N,'D':D,'E_over_epsilon':D,'J_times_omega_over_epsilon':str(inv),'equal_action_energy_over_D_I_omega':str(1+Q(d*N,d+1)),'same_depth_gap_law':'nu_k=omega*(k+1)'})
for d in (2,3,4,5,6,10):
    for N in range(7):
        es=[Q(j+1,j+2) for j in range(comb(N+d,d))] if N<3 else [Q(2),Q(3),Q(5)]
        # The variance identity is dimension independent; use small finite witnesses.
        av=sum(es)/len(es)
        check(f'variance identity {d} {N}',sum((v-av)**2 for v in es)==sum((es[i]-es[j])**2 for i in range(len(es)) for j in range(i))/len(es))
        check(f'count boundary {d} {N}',sum(comb(a+d-1,d-1) for a in range(N+1))==comb(N+d,d))
sources={'canonical_core_sha256':'a53a7568c1b590028dbf77d1a6aec7fa29ab42f5e362da2c980dca48b000122f'}
result={'created_utc':datetime.now(timezone.utc).isoformat(),'status':'passed','check_count':len(checks),'checks':checks,'sources':sources,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'assignments':rows,'exchange_actions':[str(v) for v in flow],'exchange_modal_energies':[str(nu[j]*flow[j]) for j in range(3)],'limits':'Finite exact identities, not spatial PDE evolution or a particle-clock derivation.'}
name='checks-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'.json'
(OUT/name).write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'path':str(OUT/name),'checks':len(checks),'status':'passed'}))
