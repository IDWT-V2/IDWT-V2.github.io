#!/usr/bin/env python3
"""Exact color algebra and uniform spatial-depth controls, without simulations."""
from pathlib import Path
from datetime import datetime, timezone
from hashlib import sha256
from itertools import combinations, permutations
from math import comb
import argparse
import json
import sympy as s

parser = argparse.ArgumentParser()
parser.add_argument('--output-dir', type=Path)
args = parser.parse_args()
checks = 0


def require(condition):
    global checks
    assert condition
    checks += 1


def clean(matrix):
    return matrix.applyfunc(s.simplify)


def zero(matrix):
    return clean(matrix) == s.zeros(*matrix.shape)


def comm(a, b):
    return clean(a*b-b*a)


def norm2(matrix):
    return s.simplify(s.trace(matrix.H*matrix))


def plane(d, a, b):
    result = s.zeros(d)
    result[a,b] = 1
    result[b,a] = -1
    return result


def wedge_action(matrix):
    d = matrix.rows
    basis = list(combinations(range(d), 2))
    result = s.zeros(len(basis))
    for col,(a,b) in enumerate(basis):
        for k in range(d):
            if k != b:
                pair = tuple(sorted((k,b)))
                result[basis.index(pair),col] += matrix[k,a]*(1 if k<b else -1)
            if k != a:
                pair = tuple(sorted((a,k)))
                result[basis.index(pair),col] += matrix[k,b]*(1 if a<k else -1)
    return result


def chiral_embeddings(d):
    basis = list(combinations(range(d),2))
    plus = s.zeros(len(basis),3)
    minus = s.zeros(len(basis),3)
    # B = (e23,e31,e12); star4 B = (e14,e24,e34).
    for i,(pair,sign) in enumerate([((1,2),1),((0,2),-1),((0,1),1)]):
        plus[basis.index(pair),i] = sign
        minus[basis.index(pair),i] = sign
        plus[basis.index((i,3)),i] = 1
        minus[basis.index((i,3)),i] = -1
    return plus,minus


I = s.I
root3 = s.sqrt(3)
T = [s.Matrix(x)/2 for x in [
    [[0,1,0],[1,0,0],[0,0,0]],
    [[0,-I,0],[I,0,0],[0,0,0]],
    [[1,0,0],[0,-1,0],[0,0,0]],
    [[0,0,1],[0,0,0],[1,0,0]],
    [[0,0,-I],[0,0,0],[I,0,0]],
    [[0,0,0],[0,0,1],[0,1,0]],
    [[0,0,0],[0,0,-I],[0,I,0]],
    [[1/root3,0,0],[0,1/root3,0],[0,0,-2/root3]],
]]
for a in range(8):
    require(T[a].H == T[a] and s.trace(T[a]) == 0)
    for b in range(8):
        require(s.simplify(s.trace(T[a]*T[b])) == s.Rational(a==b,2))
CF = clean(sum((t*t for t in T),s.zeros(3)))
require(CF == s.eye(3)*s.Rational(4,3))
f = [[[s.simplify(-2*I*s.trace(comm(T[a],T[b])*T[c])) for c in range(8)]
      for b in range(8)] for a in range(8)]
for a in range(8):
    for b in range(8):
        require(zero(comm(T[a],T[b])-I*sum((f[a][b][c]*T[c] for c in range(8)),s.zeros(3))))
        require(s.simplify(sum(f[a][c][d]*f[b][c][d] for c in range(8) for d in range(8))) == 3*int(a==b))
for a,b,c,d in [(a,b,c,d) for a in range(3) for b in range(3) for c in range(3) for d in range(3)]:
    lhs = s.simplify(sum(t[a,b]*t[c,d] for t in T))
    rhs = (s.Rational(1,2)*int(a==d and b==c)-s.Rational(1,6)*int(a==b and c==d))
    require(lhs == rhs)

rho = s.eye(3)/3
require(all(s.trace(rho*t)==0 for t in T))
require(s.trace(rho*CF)==s.Rational(4,3))
v = s.ones(3,1)
means = [s.simplify((v.H*t*v)[0]/3) for t in T]
require(means[2] == means[7] == 0)
require(s.simplify(sum(a*a for a in means)) == s.Rational(1,3))
require(s.simplify((v.H*CF*v)[0]/3) == s.Rational(4,3))

meson = s.zeros(9,1)
for i in range(3):
    meson[3*i+i] = 1
meson_generators = [s.kronecker_product(t,s.eye(3))-s.kronecker_product(s.eye(3),t.T) for t in T]
require(all(zero(g*meson) for g in meson_generators))
baryon = s.zeros(27,1)
for p in permutations(range(3)):
    inversions = sum(p[i]>p[j] for i in range(3) for j in range(i+1,3))
    baryon[9*p[0]+3*p[1]+p[2]] = (-1)**inversions
baryon_generators = [s.kronecker_product(t,s.eye(3),s.eye(3))+
                    s.kronecker_product(s.eye(3),t,s.eye(3))+
                    s.kronecker_product(s.eye(3),s.eye(3),t) for t in T]
require(all(zero(g*baryon) for g in baryon_generators))
require((baryon.H*baryon)[0] == 6)
baryon_C=clean(sum((g*g for g in baryon_generators),s.zeros(27)))
require(baryon_C.eigenvals()=={s.S.Zero:1,s.Integer(3):16,s.Integer(6):10})
meson_C = clean(sum((g*g for g in meson_generators),s.zeros(9)))
require(meson_C.eigenvals() == {s.S.Zero:1,s.Integer(3):8})
qq_generators=[s.kronecker_product(t,s.eye(3))+s.kronecker_product(s.eye(3),t) for t in T]
qq_C=clean(sum((g*g for g in qq_generators),s.zeros(9)))
require(qq_C.eigenvals() == {s.Rational(4,3):3,s.Rational(10,3):6})
qq_pair=(qq_C-s.Rational(8,3)*s.eye(9))/2
require(qq_pair.eigenvals() == {s.Rational(-2,3):3,s.Rational(1,3):6})
meson_pair=(meson_C-s.Rational(8,3)*s.eye(9))/2
require(meson_pair.eigenvals() == {s.Rational(-4,3):1,s.Rational(1,6):8})
mixed_baryon_generators=[s.kronecker_product(t,s.eye(3),s.eye(3))+
                        s.kronecker_product(s.eye(3),t,s.eye(3))-
                        s.kronecker_product(s.eye(3),s.eye(3),t.T) for t in T]
mixed_baryon_C=clean(sum((g*g for g in mixed_baryon_generators),s.zeros(27)))
require(mixed_baryon_C.eigenvals()=={s.Rational(4,3):6,s.Rational(10,3):6,s.Rational(16,3):15})
require(mixed_baryon_C.det()!=0)

plus,minus = chiral_embeddings(4)
star4 = s.zeros(6)
for a,b,sign in [(0,5,1),(1,4,-1),(2,3,1)]:
    star4[a,b] = star4[b,a] = sign
require(star4*star4 == s.eye(6))
require(star4*plus == plus and star4*minus == -minus)
require(plus.T*plus == minus.T*minus == 2*s.eye(3))
require(zero(plus.T*minus))
restriction = s.zeros(3,6)
restriction[0,3] = 1
restriction[1,1] = -1
restriction[2,0] = 1
require(restriction*plus == restriction*minus == s.eye(3))
observer_actions=[]
extra_actions=[]
for a,b in combinations(range(4),2):
    action = wedge_action(plane(4,a,b))
    left = clean(plus.T*action*plus/2)
    right = clean(minus.T*action*minus/2)
    require(action*plus == plus*left and action*minus == minus*right)
    if b<3:
        require(left == right)
        observer_actions.append(left)
    else:
        require(left == -right and not zero(left))
        extra_actions.append(left)
for r in observer_actions:
    require(r.T == -r)
variables = s.symbols('m0:9')
M = s.Matrix(3,3,variables)
equations = [entry for r in observer_actions for entry in comm(M,r)]
coefficients,_ = s.linear_eq_to_matrix(equations,variables)
require(len(coefficients.nullspace()) == 1)
color_rotation_commutators = sum(not zero(comm(t,r)) for t in T for r in observer_actions)
require(color_rotation_commutators > 0)

# A relative frame is a constructive repair of the fixed-vector Schur
# obstruction. It is extra geometric data, not a derived particle field.
R=s.Matrix([[0,-1,0],[1,0,0],[0,0,1]])
require(R.T*R==s.eye(3) and R.det()==1)
B=s.Matrix([1,I,2])
relative_frame_checks=0
for k in observer_actions:
    frame_variation=k*R
    pattern_variation=k*B
    require(zero(frame_variation.T*B+R.T*pattern_variation))
    for t in T:
        C=I*R*t*R.T
        varied_C=I*(frame_variation*t*R.T+R*t*frame_variation.T)
        require(zero(varied_C*B+C*pattern_variation-k*C*B))
        relative_frame_checks+=1
require(clean(sum(((R*t*R.T)**2 for t in T),s.zeros(3)))==CF)
half_spin=[s.Matrix([[0,1],[1,0]])/2,s.Matrix([[0,-I],[I,0]])/2,
           s.diag(s.Rational(1,2),s.Rational(-1,2))]
require(all(zero(comm(s.kronecker_product(a,s.eye(3)),
                      s.kronecker_product(s.eye(2),t))) for a in half_spin for t in T))

# A spatial triple's eight endomorphisms are a spin-one plus spin-two space.
adjoint_rotation_casimir = s.zeros(8)
for b,t in enumerate(T):
    image = -sum((comm(r,comm(r,t)) for r in observer_actions),s.zeros(3))
    for a in range(8):
        adjoint_rotation_casimir[a,b] = s.simplify(2*s.trace(T[a]*image))
require(adjoint_rotation_casimir.eigenvals() == {s.Integer(2):3,s.Integer(6):5})

# Canonical rank-three projected-mode connection: all nine u(3) curvature
# directions are possible if the modes can have complex relative variation.
def projected_curvature(v,w):
    return v.H*w-w.H*v

mode_tangents = []
for i in range(3):
    row=s.zeros(1,3)
    row[i]=1
    mode_tangents.extend([row,I*row])
berry_curvatures = [projected_curvature(v,w) for v,w in combinations(mode_tangents,2)]
berry_columns = s.Matrix.hstack(*(c.reshape(9,1) for c in berry_curvatures))
require(berry_columns.rank()==9)
traceless_curvatures = [c-s.trace(c)*s.eye(3)/3 for c in berry_curvatures]
require(s.Matrix.hstack(*(c.reshape(9,1) for c in traceless_curvatures)).rank()==8)
real_curvatures = [projected_curvature(v,w) for v,w in combinations(mode_tangents[::2],2)]
require(s.Matrix.hstack(*(c.reshape(9,1) for c in real_curvatures)).rank()==3)

def traceless(matrix):
    return clean(matrix-s.trace(matrix)*s.eye(matrix.rows)/matrix.rows)


def lie_span(generators,target_rank):
    basis=[]
    for candidate in generators:
        if s.Matrix.hstack(*(x.reshape(9,1) for x in basis+[candidate])).rank()>len(basis):
            basis.append(candidate)
    while len(basis)<target_rank:
        previous=len(basis)
        for a in basis[:]:
            for b in basis[:]:
                candidate=comm(a,b)
                if s.Matrix.hstack(*(x.reshape(9,1) for x in basis+[candidate])).rank()>len(basis):
                    basis.append(candidate)
                if len(basis)==target_rank:
                    break
            if len(basis)==target_rank:
                break
        if len(basis)==previous:
            break
    return len(basis)


# Actual two-dimensional pullback Z(x,y)=x*v1+y*v2+x*x*v3/2.
# At the origin A=0, Fxy=M(v1,v2), and covariant d_x Fxy=M(v3,v2).
v1=s.Matrix([[1,0,0]])
v2=s.Matrix([[I,1,0]])
v3=s.Matrix([[0,I,1]])
curvature0=traceless(projected_curvature(v1,v2))
curvature_derivative=traceless(projected_curvature(v3,v2))
require(lie_span([curvature0,curvature_derivative],8)==8)
u=s.diag(I,1,-I)
c=s.Matrix([1,1,0])
require(u.H*u==s.eye(3) and u.det()==1)
require((c.H*c)[0]==((u*c).H*(u*c))[0])
require(s.Abs((c.T*c)[0])**2==4 and s.Abs(((u*c).T*(u*c))[0])**2==0)

spin10_blades=[set(b) for k in [0,2,4] for b in combinations(range(10),k)]
require(len(spin10_blades)==256)

depth_rows=[]
leakage_rows=[]
for d in range(2,17):
    bivector_rank = comb(d,2)
    factors = [1] if d==2 else [3,3] if d==4 else [bivector_rank]
    require((3 in factors) == (d in (3,4)))
    # Standard fixed Spin(10) positive chiral carrier, restricted to Spin(d).
    if d<=10:
        if d==10:
            copies=[1]
            spin_dimensions=[16]
        elif d%2==0:
            copies=[2**((10-d)//2-1)]*2
            spin_dimensions=[2**(d//2-1)]*2
        else:
            copies=[2**((9-d)//2)]
            spin_dimensions=[2**((d-1)//2)]
        require(sum(a*b for a,b in zip(copies,spin_dimensions)) == 16)
        # Independent Clifford-monomial centralizer count, with the chiral
        # complement relation reducing every even blade to degree zero/two/four.
        allowed=[b for b in spin10_blades if all((a in b)==(c in b)
                 for a,c in combinations(range(d),2))]
        require(len(allowed)==sum(k*k for k in copies))
        commuting_su3_available = any(k>=3 for k in copies)
    else:
        copies=spin_dimensions=None
        commuting_su3_available=None
    real_spin_commutant={0:'R',1:'R',2:'C',3:'H',4:'H',5:'H',6:'C',7:'R'}[d%8]
    # Full tangent-Clifford compatibility uses the parent Dirac carrier,
    # not its chiral half, because individual generators exchange chirality.
    if d<=10:
        clifford_copies=([2**((10-d)//2)] if d%2==0
                        else [2**((9-d)//2)]*2)
        require(sum(clifford_copies)*2**(d//2)==32)
    else:
        clifford_copies=None
    mixing_factors=[{'rank':k,'traceless_unitary_dimension':k*k-1,
                     'fundamental_Casimir':str(s.Rational(k*k-1,2*k)),
                     'adjoint_Casimir':k if k>1 else 0} for k in factors]
    depth_rows.append({'spatial_depth':d,'maintained_sector':d in [2,3,4,5,6,10],
                       'bivector_rank':bivector_rank,'full_rotation_factor_ranks':factors,
                       'rank_three_rotation_factor':3 in factors,
                       'Spin10_chiral_rotation_multiplicities':copies,
                       'Spin10_restricted_irrep_dimensions':spin_dimensions,
                       'commuting_su3_available_in_this_carrier':commuting_su3_available,
                       'full_Spin10_Dirac_tangent_Clifford_multiplicities':clifford_copies,
                       'full_tangent_Clifford_commuting_su3_available':any(k>=3 for k in clifford_copies) if clifford_copies else None,
                       'minimal_real_spinor_commutant':real_spin_commutant,
                       'quaternionic_three_units_available':real_spin_commutant=='H',
                       'full_rotation_factor_unitary_mixing':mixing_factors})
    if d>=4:
        p,_ = chiral_embeddings(d)
        projector=p*p.T/2
        require(projector*projector==projector and projector.rank()==3)
        total=s.S.Zero
        mixed_count=0
        for a,b in combinations(range(d),2):
            action=wedge_action(plane(d,a,b))
            leaked=(s.eye(bivector_rank)-projector)*action*p
            leakage=norm2(leaked)/2
            if a<4<=b:
                require(leakage==s.Rational(3,2))
                mixed_count+=1
            else:
                require(leakage==0)
            total+=leakage
        require(total == 6*(d-4))
        leakage_rows.append({'spatial_depth':d,'chiral_triple_rank':3,
                             'mixed_rotation_generators':mixed_count,
                             'total_squared_leakage':str(total),
                             'invariant_under_full_sector_rotations':total==0})

# Uniform round-sphere spectrum controls. These are actual spectral
# multiplicities, unlike the ranks of a two-form bundle at one point.
sphere_rows=[]
for d in range(2,17):
    candidates=[{'type':'exact-two-form','eigenvalue':2*(d-1),'multiplicity':comb(d+1,2)}]
    if d>=3:
        candidates.append({'type':'coexact-two-form','eigenvalue':3*(d-2),'multiplicity':comb(d+1,3)})
    first=min(c['eigenvalue'] for c in candidates)
    first_multiplicity=sum(c['multiplicity'] for c in candidates if c['eigenvalue']==first)
    require((first_multiplicity==3)==(d==2))
    sphere_rows.append({'spatial_depth':d,'first_nonzero_Hodge_two_form_eigenvalue':first,
                        'first_nonzero_Hodge_two_form_multiplicity':first_multiplicity,
                        'first_scalar_harmonic_multiplicity':d+1,
                        'candidates':candidates})
require(sphere_rows[1]['first_nonzero_Hodge_two_form_multiplicity']==4)
require(sphere_rows[2]['first_nonzero_Hodge_two_form_multiplicity']==20)

# Boundary/self-duality degree tests do not require a gauge group.
curvature_degree_rows=[]
for d in range(2,17):
    curvature_degree_rows.append({'spatial_depth':d,'star_two_form_degree':d-2,
        'ordinary_self_dual_two_form_possible':d==4,
        'scalar_gradient_Bogomolny_degree_match':d==3,
        'four_dimensional_self_dual_embedding_in_container_possible':d>=4,
        'quaternionic_geometry_pair_dimensions':d%4 in [0,3]})

def hodge_basis(indices,dimension):
    complement=tuple(i for i in range(dimension) if i not in indices)
    sign=(-1)**(sum(indices)-len(indices)*(len(indices)-1)//2)
    return complement,sign


def sparse_hodge(form,dimension):
    out={}
    for indices,value in form.items():
        dual,sign=hodge_basis(indices,dimension)
        out[dual]=out.get(dual,0)+sign*value
    return {k:s.simplify(v) for k,v in out.items() if v!=0}


middle_hodge_rows=[]
for m in range(1,9):
    even=2*m
    odd=even-1
    k=comb(odd,m-1)
    require(2*k==comb(even,m))
    require((k==3)==(m==2))
    if m>1:
        previous=comb(2*m-3,m-2)
        require(s.Rational(k,previous)==4-s.Rational(2,m))
    eigenvalue=s.S.One if m%2==0 else I
    count=0
    for indices in combinations(range(odd),m-1):
        a,sign=hodge_basis(indices,odd)
        form={a:s.Integer(sign),indices+(odd,):1/eigenvalue}
        opposite={a:s.Integer(sign),indices+(odd,):-1/eigenvalue}
        require(sparse_hodge(form,even)=={key:s.simplify(eigenvalue*value) for key,value in form.items()})
        require(sparse_hodge(opposite,even)=={key:s.simplify(-eigenvalue*value) for key,value in opposite.items()})
        require(sum(s.conjugate(form[key])*opposite[key] for key in form)==0)
        require(sum(s.conjugate(value)*value for value in form.values())==2)
        count+=1
    require(count==k)
    for d in [odd,even]:
        middle_hodge_rows.append({'spatial_depth':d,'paired_depth':even if d==odd else odd,
            'complete_middle_Hodge_channel_rank':k,
            'three_channel_candidate':k==3,
            'even_partner_Hodge_eigenvalue':str(eigenvalue),
            'full_channel_mixing_group':'SU('+str(k)+')' if k>1 else 'trivial',
            'full_channel_fundamental_Casimir':str(s.Rational(k*k-1,2*k)),
            'scope':'Full-channel mixing is a physical hypothesis; Hodge rank and restriction are exact'})

def blade_lie_closure(initial):
    basis=set(initial)
    while True:
        previous=len(basis)
        for a,b in combinations(tuple(basis),2):
            if (len(a)*len(b)-len(a&b))%2:
                basis.add(a^b)
        if len(basis)==previous:
            return basis


endomorphism_rows=[]
for d in range(2,17):
    spinor_dimension=2**((d-1)//2)
    components=[]
    if d%2:
        for k in range(0,d,2):
            components.append({'blade_degree':k,'multiplicity':comb(d,k),
                               'rotation_Casimir':k*(d-k)})
    else:
        for k in range(0,d//2+1,2):
            multiplicity=comb(d,k)
            if 2*k==d:
                multiplicity//=2
            components.append({'blade_degree':k,'multiplicity':multiplicity,
                               'rotation_Casimir':k*(d-k)})
    require(sum(x['multiplicity'] for x in components)==spinor_dimension**2)
    positive=[x for x in components if x['rotation_Casimir']>0]
    first=min((x['rotation_Casimir'] for x in positive),default=None)
    first_rank=sum(x['multiplicity'] for x in positive if x['rotation_Casimir']==first)
    require((first_rank==3)==(d in [3,4]))
    require((spinor_dimension**2-1==3)==(d in [3,4]))
    # A rotation generator commutes with a Clifford blade unless exactly one
    # of its two indices lies in that blade. There are k*(d-k) such pairs.
    for component in components:
        k=component['blade_degree']
        representative=set(range(k))
        counted=sum((a in representative)!=(b in representative)
                    for a,b in combinations(range(d),2))
        require(counted==component['rotation_Casimir'])
    lie_rank=0
    closed=True
    if positive:
        if d==4:
            # The half-middle two-forms are one simple so(3) ideal.
            lie_rank=3
        else:
            lowest=set(frozenset(indices) for component in positive
                       if component['rotation_Casimir']==first
                       for indices in combinations(range(d),component['blade_degree']))
            closure=blade_lie_closure(lowest)
            lie_rank=len(closure)
            closed=closure==lowest
            require(lie_rank==(comb(d,2) if d%2==0 else (3 if d==3 else comb(d,2)+d)))
            require(closed==(d%2==0 or d==3))
    endomorphism_rows.append({'spatial_depth':d,'minimal_rotation_spinor_dimension':spinor_dimension,
        'traceless_endomorphism_rank':spinor_dimension**2-1,
        'lowest_positive_adjoint_rotation_Casimir':first,
        'lowest_positive_channel_rank':first_rank,
        'lowest_channel_closed_under_commutators':closed,
        'Lie_algebra_generated_by_lowest_channel_rank':lie_rank,
        'components':components})

paired_carrier_rows=[]
for d in range(2,17):
    rank=2**((d+1)//2)
    grades=range(d+1) if d%2==0 else range(0,d,2)
    factor=1 if d%2==0 else 4
    components=[{'blade_degree':k,'rotation_Casimir':k*(d-k),
                 'multiplicity':factor*comb(d,k)} for k in grades]
    require(sum(x['multiplicity'] for x in components)==rank**2)
    first=min(x['rotation_Casimir'] for x in components if x['rotation_Casimir']>0)
    first_rank=sum(x['multiplicity'] for x in components if x['rotation_Casimir']==first)
    require(first_rank==(2 if d==2 else (2*d if d%2==0 else 4*d)))
    paired_carrier_rows.append({'spatial_depth':d,'maintained_paired_carrier_rank':rank,
        'lowest_positive_rotation_Casimir':first,
        'lowest_positive_channel_rank':first_rank,
        'components':components,
        'scope':'Full paired Dirac carrier without choosing a minimal rotation irrep'})

low_degree_rows=[]
for k in [1,2,3,5,10,15,35,45,126]:
    cubic_allowed=3%k==0
    # The SU(k) center acts on three fundamental slots with phase z^3.
    # At k=3 the previously checked epsilon tensor supplies the invariant.
    require(cubic_allowed==(k in [1,3]))
    a=s.zeros(k,1)
    b=s.zeros(k,1)
    a[0]=1
    b[0]=1 if k==1 else 1/s.sqrt(2)
    if k>1:
        b[1]=1/s.sqrt(2)
    require(norm2(a)==1 and norm2(b)==1)
    overlap=(b.H*a)[0]
    tangent_response=clean(b*overlap-a*s.conjugate(overlap)*overlap)
    require(norm2(tangent_response)==(0 if k==1 else s.Rational(1,4)))
    low_degree_rows.append({'complete_complex_channel_rank':k,
        'three_fundamental_slots_singlet_dimension':1 if cubic_allowed else 0,
        'quartic_overlap_has_nonzero_relative_orientation_response':k>1,
        'quartic_overlap_projected_gradient_norm_squared':str(norm2(tangent_response)),
        'scope':'Full SU(k) equivalent-mode hypothesis; not the native rotation algebra'})

result={'status':'passed','checks':checks,
        'scope':'Representation and rotation-form geometry, not a derived physical color interaction',
        'depth_rows':depth_rows,'chiral_embedding_controls':leakage_rows,
        'color_factors':{'CF':'4/3','CA':'3','TR':'1/2'},
        'pair_color_eigenvalues':{'qq_antitriplet':'-2/3','qq_sextet':'1/3',
                                  'q_antiquark_singlet':'-4/3','q_antiquark_octet':'1/6',
                                  'scope':'Algebraic weights, not forces without a derived interaction kernel'},
        'mixed_quark_family_control':{'three_same_color_types_allow_baryon_singlet':True,
                                      'two_triplets_one_antitriplet_Casimir_has_zero':False},
        'round_sphere_spectral_controls':sphere_rows,
        'curvature_degree_controls':curvature_degree_rows,
        'middle_hodge_pair_controls':middle_hodge_rows,
        'spinor_endomorphism_Casimir_controls':endomorphism_rows,
        'low_degree_interaction_controls':low_degree_rows,
        'maintained_paired_spinor_Casimir_controls':paired_carrier_rows,
        'neutrality_controls':{'unpolarized_triplet_mean_generators':'0',
                              'unpolarized_triplet_Casimir':'4/3',
                              'equal_component_pure_triplet_mean_square':'1/3',
                              'meson_total_generators':'0','baryon_total_generators':'0',
                              'three_triplets_Casimir_spectrum':{'0':1,'3':16,'6':10},
                              'zero_triality_decuplet_Casimir':'6'},
        'three_four_restriction':{'both_chiral_restrictions_rank':3,
                                 'observer_actions_equal':True,
                                 'extra_rotation_actions_opposite':True,
                                 'rotation_commutant_complex_dimension':1,
                                 'nonzero_color_rotation_commutators':color_rotation_commutators,
                                 'eight_endomorphisms_rotation_Casimir':{'2':3,'6':5}},
        'relative_frame_completion':{'mixed_generator_checks':relative_frame_checks,
                                     'relative_amplitudes_are_rotation_invariant':True,
                                     'color_Casimir_preserved':'4/3',
                                     'separate_spin_half_and_color_actions_commute':True,
                                     'scope':'Requires a geometric moving frame and a justified equivalent-mode sector'},
        'projected_mode_curvature':{'complex_variations_u3_span':9,
                                   'traceless_complex_variations_su3_span':8,
                                   'real_variations_so3_span':3,
                                   'two_dimensional_pullback_holonomy_Lie_rank':8,
                                   'same_pullback_extends_to_depths':[2,16],
                                   'SO3_invariant_quartic_SU3_counterexample':[4,0],
                                   'scope':'Availability if an actual geometric rank-three mode projector is derived'},
        'sources':{'experiments/color-depth-algebra-checks.py':sha256(Path(__file__).read_bytes()).hexdigest()}}
out=args.output_dir or Path(__file__).parent/('run-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ'))
out.mkdir(parents=True,exist_ok=False)
(out/'executed-source.py').write_bytes(Path(__file__).read_bytes())
(out/'color-depth-algebra-checks.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'status':'passed','checks':checks,'output':str(out)},indent=2),flush=True)
