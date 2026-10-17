import sympy as s,json,pathlib,datetime,hashlib
root=pathlib.Path(__file__).resolve().parent
x,a,t=s.symbols('x a t', positive=True)
w=a/x**2-s.log(1+1/x**2);q=x*x
amin=q*(3*q+1)/(3*(q+1)**2)
checks={}
def check(name,expr):
 assert s.factor(expr)==0,name
 checks[name]=True
check('convexity boundary',s.diff(w,x,2)-6*(a-amin)/x**4)
P=(5*s.diff(w,x,3)**2-3*s.diff(w,x,2)*s.diff(w,x,4))*x**10*(q+1)**6/8
positive=90*(q+1)**6*t**2+3*q*(q+1)**2*(35*q*q+126*q+19)*t+10*q*q*(5*q+1)**2
check('Schaaf positive shifted polynomial',P.subs(a,amin+t)-positive)
check('Schaaf inflection sign', (s.diff(w,x)*s.diff(w,x,3)).subs(a,amin)+16*(5*x*x+1)/(3*x**4*(x*x+1)**5))
x0=s.sqrt(a/(1-a)); v2,v3,v4=[s.diff(w,x,k).subs(x,x0) for k in (2,3,4)]
C=(2*a*a+32*a+11)/(24*(1-a)**2)
check('small amplitude period coefficient',v4/(8*v2**2)-5*v3**2/(24*v2**3)+C)
L,alpha,beta,S2,v,mu,J,d=s.symbols('L alpha beta S2 v mu J d',positive=True)
W=alpha/L**2-beta*s.log(1+2*S2/L**2)
c=v/(2*alpha*L)
cdot=2*alpha/L**4-2*alpha*c*c-4*beta*S2/(L**2*(L**2+2*S2))
phase=mu+d*(alpha/L**2-beta*s.log(1+2*S2/L**2)/2-beta*S2/(L**2+2*S2))
H=J*mu/2+J*d*(W+alpha*c*c*L*L)/4
check('exact Euler center phase factors',2*H-J*phase+J*d*L*L*cdot/4)
for depth in (2,3,4,5,6,10):
 check('radial mass depth '+str(depth),(-s.diff(J*depth*W/4,L)/(J*depth/(8*alpha)))-(4*alpha*alpha/L**3-8*alpha*beta*S2/(L*(L*L+2*S2))))
report={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'script_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'checks':checks}
(root/'symbolic-checks.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
