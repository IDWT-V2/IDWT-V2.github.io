#!/usr/bin/env python3
"""Scoped checks for Responding Spatial Count Patterns, Python standard library.
Run: python3 -B check.py --output results.json
New independent checks of selected displayed identities, not a replay of any
original checker, spatial quadrature suite, or nonlinear encounter simulation.
"""
import argparse
import datetime
from fractions import Fraction as F
import hashlib
import json
import math
from pathlib import Path

checks = []
def check(name, condition):
    if not condition:
        raise AssertionError(name)
    checks.append(name)
def dot(a, b):
    return sum(x*y for x, y in zip(a, b))
def add(a, b):
    return [x+y for x, y in zip(a, b)]
def scale(a, c):
    return [c*x for x in a]
def sub(a, b):
    return add(a, scale(b, -1))
def J(a):
    return [v for x,y in zip(a[::2],a[1::2]) for v in (-y,x)]
def close(a, b):
    return abs(a-b) <= 2e-12 * max(1,abs(a),abs(b))

# Exact rational substitution in the full rotating identity, including an
# independent spatial response complement. Source s has unit norm (3/5,4/5).
h, om, g, lam = F(2), F(5), F(1,5), F(9,2)
delta=om*om-h*h
z=[F(3,2),F(-2,3)]; zd=[F(2,7),F(5,9)]
s=[F(3,5),F(4,5)]
u=[s0*z0 for s0 in s for z0 in z]+[F(0),F(0)]
y=[F(1,3),F(2,5),F(-3,7),F(1,11),F(2),F(-1)]
yd=[F(1,2),F(-2,3),F(3,4),F(4,5),F(1,7),F(2,9)]
r2=dot(z,z)
energy=(dot(zd,zd)+r2*(lam+g*g/(om*om-lam))+dot(yd,yd)+om*om*dot(y,y))/2-g*dot(u,y)
charge=dot(J(z),zd)+dot(J(y),yd)
for sig in [-1,1]:
    rotating=(dot(sub(zd,scale(J(z),sig*h)),sub(zd,scale(J(z),sig*h)))
       +dot(sub(yd,scale(J(y),sig*h)),sub(yd,scale(J(y),sig*h))))/2
    residual=sub(y,scale(u,g/delta))
    rhs=rotating+delta*dot(residual,residual)/2+r2*(lam-h*h)*(1+g*g/((om*om-lam)*delta))/2
    check('rational complete rotating identity sign '+str(sig),energy-sig*h*charge==rhs)

# Exact complete potential squares for both neutral actions, with arbitrary
# response, positive unit density directions and explicit radial derivatives.
s1=[F(1),F(0)];s2=[F(3,5),F(4,5)]; response=[F(2,3),F(-1,4)]
r1,r2,rs=F(3,2),F(5,4),F(7,6)
for xi in [0,2]:
    def amp(r): return r*r if xi==2 else r*r/(rs*rs+r*r)
    def damp(r): return 2*r if xi==2 else 2*r*rs*rs/(rs*rs+r*r)**2
    j1=scale(s1,g*amp(r1));j2=scale(s2,g*amp(r2));js=add(j1,j2)
    v=om*om*dot(response,response)/2-dot(js,response)+(dot(j1,j1)+dot(j2,j2))/(2*om*om)+xi*dot(j1,j2)/(om*om)
    diff=sub(response,scale(js,1/(om*om)))
    vc=om*om*dot(diff,diff)/2+(xi-1)*dot(j1,j2)/(om*om)
    check('rational neutral square xi='+str(xi),v==vc)
    dj=scale(s1,g*damp(r1))
    check('rational isolated radial self force xi='+str(xi),dot(dj,scale(j1,1/(om*om)))-dot(dj,j1)/(om*om)==0)
    pairforce=dot(dj,scale(js,1/(om*om)))-dot(dj,add(j1,scale(j2,xi)))/(om*om)
    check('rational static radial cross sign xi='+str(xi),pairforce==(1-xi)*dot(dj,j2)/(om*om))
    if xi==0:
        check('saturated bounded cross',0<=dot(j1,j2)<=g*g)
        check('saturated shifted budget positive',g*g-dot(j1,j2)>0)

# Exact conservative lower factors used in the common cap proof.
A=F(3);Z=1+g*g/delta**2;E0=h*A
check('charge roots at maximal epsilon',F(1)+F(1,4)-F(3,4)==F(1,2) and F(1)+F(1,4)+F(3,4)==2)
check('radius lower budget factor',(A/(2*h)-4*(A*delta/(16*h))/delta)/(2*Z)==A/(8*h*Z))
cap=(om*om+h*h)/2
check('cap barrier factor',(A/(8*h*Z))*(cap-h*h)/2==E0*(cap/h**2-1)/(16*Z))
check('material half strain barrier',F(1,8)*F(1,2)**2==F(1,32))

# Actual fourteen supplied count labels; no empirical recalibration.
rows=[('W',2,76,2926),('Z',2,81,3321),('Higgs',2,95,4560),
 ('down',3,1,1),('strange',3,4,20),('up',4,3,15),('charm',4,20,8855),
 ('top',4,72,1215450),('nu1',5,10,2002),('nu2',5,15,11628),
 ('nu3',5,22,65780),('electron',6,13,18564),('muon',6,35,3838380),('tau',10,23,64512240)]
for name,d,n,c in rows:
    K=n+d-1
    check(name+' count',math.comb(K,d)==c)
    if K>d:
        check(name+' Pascal relative gap',F(c,c-math.comb(K-1,d))==F(K,d))
check('all maintained depths',set(d for _,d,_,_ in rows)=={2,3,4,5,6,10})
check('pair counts',math.comb(14,2)==91 and math.comb(15,2)==105)

# Explicit exact-invariant solution evaluated in floating arithmetic; analytic
# time derivatives are used, not a time-step integrator. h=2, Omega=5,g=1/5.
hf,of,gf,R=2.,5.,.2,1.3
D=of*of-hf*hf;k=hf*hf+gf*gf/D
ws2=(k+of*of-math.sqrt((of*of-k)**2+8*gf*gf))/2
ws,wa=math.sqrt(ws2),math.sqrt(k);bs=math.sqrt(2)*gf/(of*of-ws2);Zs=1+bs*bs
check('bright/dark positive',0<ws<wa)
max_residual=0.;max_energy_error=0.;max_charge_error=0.
for index,phase in enumerate([math.pi/3,math.pi/2,2*math.pi/3,math.pi]):
    for sig in [-1,1]:
        # A common phase may be chosen freely at each sample. This evaluates
        # the analytic solution, its first derivatives and second derivatives.
        plus=[R,0.];minus=[R*math.cos(phase),sig*R*math.sin(phase)]
        pd=scale(J(plus),sig*ws);md=scale(J(minus),sig*wa)
        Y=scale(plus,bs);Yd=scale(pd,bs);Ydd=scale(Y,-ws2)
        z1=scale(add(plus,minus),1/math.sqrt(2));z2=scale(sub(plus,minus),1/math.sqrt(2))
        d1=scale(add(pd,md),1/math.sqrt(2));d2=scale(sub(pd,md),1/math.sqrt(2))
        dd1=scale(add(scale(plus,-ws2),scale(minus,-wa*wa)),1/math.sqrt(2))
        dd2=scale(sub(scale(plus,-ws2),scale(minus,-wa*wa)),1/math.sqrt(2))
        residuals=add(dd1,sub(scale(z1,k),scale(Y,gf)))+add(dd2,sub(scale(z2,k),scale(Y,gf)))+add(Ydd,sub(scale(Y,of*of),scale(add(z1,z2),gf)))
        max_residual=max(max_residual,max(map(abs,residuals)))
        check('oscillator rows '+str((index,sig)),all(close(x,0) for x in residuals))
        E=(dot(d1,d1)+dot(d2,d2)+dot(Yd,Yd)+k*(dot(z1,z1)+dot(z2,z2))+of*of*dot(Y,Y))/2-gf*dot(add(z1,z2),Y)
        Q=dot(J(z1),d1)+dot(J(z2),d2)+dot(J(Y),Yd)
        Et=R*R*(ws2*Zs+wa*wa);Qt=sig*R*R*(ws*Zs+wa)
        max_energy_error=max(max_energy_error,abs(E-Et));max_charge_error=max(max_charge_error,abs(Q-Qt))
        check('complete transfer energy '+str((index,sig)),close(E,Et))
        check('complete transfer charge '+str((index,sig)),close(Q,Qt))
        check('transfer radius '+str((index,sig)),close(dot(z1,z1),R*R*(1+math.cos(phase))))
        check('transfer retained charge '+str((index,sig)),close(dot(J(z1),d1),sig*R*R*(ws+wa)*(1+math.cos(phase))/2))
        if index<3:
            check('finite coexistence '+str((index,sig)),R*R/2-1e-12<=dot(z1,z1)<=3*R*R/2+1e-12)
        else:
            check('boundary velocity nonzero '+str(sig),dot(d1,d1)>0 and close(dot(d1,d1),R*R*(wa-ws)**2/2))

parser=argparse.ArgumentParser();parser.add_argument('--output',default='results.json');args=parser.parse_args()
source=Path(__file__).read_bytes()
result={'title':'New scoped responding-spatial identity checker','status':'passed','assertions':len(checks),
 'executed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'source_sha256':hashlib.sha256(source).hexdigest(),'source_bytes':len(source),
 'requirements':'Python 3 standard library only','checks':checks,
 'transfer_example':{'h':hf,'Omega':of,'g':gf,'amplitude':R,'omega_s':ws,'omega_a':wa,
 'max_equation_residual':max_residual,'max_energy_error':max_energy_error,'max_charge_error':max_charge_error,
 'boundary_velocity_squared':R*R*(wa-ws)**2/2},
 'scope':'Exact rational substitutions for rotating identity with complement and both signs, neutral squares/radial signs, cap factors, fourteen counts; analytic invariant-transfer solution evaluated numerically.',
 'not_executed':['Original 1567/366/530/152 suites','105 spatial overlap quadratures','full nonlinear encounter integration','gradient-response evolution','empirical validation']}
Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'status':'passed','assertions':len(checks),'source_sha256':result['source_sha256'],'output':args.output}))
