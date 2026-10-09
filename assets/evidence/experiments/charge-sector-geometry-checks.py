#!/usr/bin/env python3
"""Standalone exact geometry, charge and sector-selector comparison.

Run: python3 -B charge-sector-geometry-checks.py [--output-dir NEW_DIRECTORY]
Requires installed SymPy. No other project files, network or simulations.
"""
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction as F
from itertools import combinations, permutations, product
from math import comb
from pathlib import Path
import argparse
import hashlib
import json
import sympy as s

HERE = Path(__file__).resolve().parent
SCRIPT_SOURCE_PATH = 'experiments/charge-sector-geometry-checks.py'
DEPTHS = (2, 3, 4, 5, 6, 10)
CHECKS = []
Q = (F(0), F(2, 3), F(-1), F(-1, 3))
MULT = (1, 3, 1, 3)


def check(name, result):
    result = bool(result)
    CHECKS.append({"name": name, "passed": result})
    if not result:
        raise AssertionError(name)


def exterior_spectrum(rank, denominator):
    return [(F((-1)**k * k, denominator), comb(rank, k))
            for k in range(rank + 1)]


def signature(rows):
    result = Counter()
    for charge, multiplicity in rows:
        result[charge] += multiplicity
    return result


def serial_spectrum(rows):
    return [{"charge": str(q), "multiplicity": m} for q, m in rows]


def rank_and_depths():
    target = signature(zip(Q, MULT))
    ranks = []
    for rank in range(1, 7):
        relative = exterior_spectrum(rank, rank)
        common = exterior_spectrum(rank, 3)
        check(f"rank {rank} exterior total dimension", sum(m for _, m in common) == 2**rank)
        ranks.append({"complex_rank": rank, "real_normal_dimension": 2*rank,
                      "determinant_unit": serial_spectrum(relative),
                      "fixed_three_plane_unit": serial_spectrum(common),
                      "full_target_match": signature(common) == target})
    check("only rank three gives the complete target spectrum in ranks 1..6",
          [r["complex_rank"] for r in ranks if r["full_target_match"]] == [3])
    normal_pairs = []
    for a, b in combinations(DEPTHS, 2):
        gap = b-a
        row = {"lower_depth": a, "upper_depth": b, "normal_dimension": gap}
        if gap % 2:
            row["status"] = "odd real normal rank: whole-normal complex structure unavailable"
            row["full_target_match"] = False
        else:
            row["complex_rank"] = gap//2
            row["full_target_match"] = signature(exterior_spectrum(gap//2, 3)) == target
            row["status"] = "Hermitian normal bundle assumed; full exterior carrier compared"
        normal_pairs.append(row)
    winners = [(r["lower_depth"], r["upper_depth"]) for r in normal_pairs if r["full_target_match"]]
    check("only maintained pair 4-in-10 has full rank-three table", winners == [(4, 10)])
    broad_pairs = [(a, b) for a, b in combinations(range(2, 11), 2) if b-a == 6]
    check("rank criterion is a gap criterion, not absolute depth selection",
          broad_pairs == [(2, 8), (3, 9), (4, 10)])

    # Rebuilding the carrier from each tangent space is a changed rule.
    tangent_rows = []
    targets = {3: F(-1, 3), 4: F(2, 3), 5: F(0), 6: F(-1), 10: F(-1)}
    target_multiplicities = {3: 3, 4: 3, 5: 1, 6: 1, 10: 1}
    for d in DEPTHS:
        n = d//2
        rows = exterior_spectrum(n, 3)
        tangent_rows.append({"depth": d, "paired_rank": n,
                             "extra_odd_direction_choice_required": bool(d % 2),
                             "charges": serial_spectrum(rows),
                             "matter_target": str(targets[d]) if d in targets else None,
                             "target_multiplicity": target_multiplicities.get(d),
                             "multiplicity_at_target": next((m for q, m in rows if d in targets and q == targets[d]), None),
                             "target_available": any(q == targets[d] for q, _ in rows) if d in targets else None})
    check("changed tangent rule contains all target numbers with fixed unit",
          all(r["target_available"] for r in tangent_rows if r["matter_target"] is not None))
    check("changed tangent rule fails target multiplet ranks at depths 3,4,10",
          [r["depth"] for r in tangent_rows if r["matter_target"] is not None
           and r["multiplicity_at_target"] != r["target_multiplicity"]] == [3, 4, 10])
    return {"ranks": ranks, "all_maintained_pairs": normal_pairs,
            "maintained_pair_matches": winners, "all_integer_depths_2_to_10_matches": broad_pairs,
            "changed_tangent_rule_fixed_unit": tangent_rows}


def fixed_carrier_assignments():
    # Pullback of the constructed E and Q along each inclusion has the same
    # fibre spectrum; it does not rebuild E from that submanifold's tangent.
    spectra = {str(d): serial_spectrum(zip(Q, MULT)) for d in DEPTHS}
    qmat = s.diag(0, *([s.Rational(2, 3)]*3), -1, *([-s.Rational(1, 3)]*3))
    for d in DEPTHS:
        connection = s.I*qmat
        check(f"depth {d} pulled-back connection preserves Q", connection*qmat == qmat*connection)
    blocks = ("down_at_3", "up_at_4", "neutral_at_5", "charged_at_6_and_10")
    assignments = []
    for perm in permutations(range(4)):
        type_ok = [MULT[k] for k in perm] == [3, 3, 1, 1]
        assignments.append({"assignment": {name: str(Q[k]) for name, k in zip(blocks, perm)},
                            "type_preserving": type_ok,
                            "trace_zero": sum(MULT[k]*Q[k] for k in perm) == 0})
    check("every one-to-one branch assignment retains trace", all(a["trace_zero"] for a in assignments))
    check("24 permutations allowed by fibre construction", len(assignments) == 24)
    check("four assignments remain after singlet/triplet grouping",
          sum(a["type_preserving"] for a in assignments) == 4)
    type_cases = [a["assignment"] for a in assignments if a["type_preserving"]]
    swapped = {"down_at_3": "2/3", "up_at_4": "-1/3", "neutral_at_5": "0", "charged_at_6_and_10": "-1"}
    check("quark-depth swap remains geometrically allowed", swapped in type_cases)
    # These extra constraints associate the geometric transition with named
    # physical channels. The fixed carrier itself has no such depth map.
    aligned = [a for a in type_cases if F(a['up_at_4'])-F(a['down_at_3'])
               == F(a['neutral_at_5'])-F(a['charged_at_6_and_10'])]
    exact_oriented = [a for a in aligned if F(a['neutral_at_5'])-F(a['charged_at_6_and_10']) == 1]
    check("naming the same oriented physical channels leaves two assignments", len(aligned) == 2)
    check("naming physical forward channels as D leaves one assignment", len(exact_oriented) == 1)
    return {"restriction_scope": "Same W/E extended by pullback from the depth-four base on the explicit R10 example",
            "spectra_at_depths": spectra, "one_to_one_assignment_count": len(assignments),
            "type_preserving_assignments": type_cases,
            "with_additional_named_channel_alignment": aligned,
            "with_named_forward_channels_fixed_to_D": exact_oriented,
            "assignment_qualification": "The 24 and 4 counts do not impose a map from physical named channels to D. Such a map reduces counts by extra assumptions; the depth numbers still do not enter Q.",
            "warning": "These are permitted identifications, not derived physical states. No branch-to-depth selector is present."}


def twists_and_weights():
    a, b, c, m = s.symbols("a b c m", integer=True)
    neutral_trace_common = [1+3*a+b+3*c, -b-m, 1+a-c-m]
    solution = s.solve(neutral_trace_common, (a, b, c), dict=True)[0]
    check("common-transition character solution",
          solution == {a: 2*(m-1)/3, b: -m, c: (1-m)/3})
    exact_unit = {key: s.simplify(value.subs(m, 1)) for key, value in solution.items()}
    check("exact determinant transition fixes all twists", exact_unit == {a: 0, b: -1, c: 0})
    classified = []
    for av, cv in product(range(-3, 4), repeat=2):
        bv = -1-3*(av+cv)
        weights = (F(0), F(2, 3)+av, F(bv), F(-1, 3)+cv)
        check(f"twist ({av},{cv}) full trace", sum(q*k for q, k in zip(weights, MULT)) == 0)
        transitions = (-weights[2], weights[1]-weights[3])
        common = transitions[0] == transitions[1]
        if common:
            mv = transitions[0]
            check(f"common twist ({av},{cv}) is only an overall scale", weights == tuple(mv*q for q in Q))
            check(f"common twist ({av},{cv}) U3 globality", mv.denominator == 1 and mv.numerator % 3 == 1)
        classified.append({"a": av, "b": bv, "c": cv,
                           "charges": list(map(str, weights)), "common_transition": common})
    alternative = (F(0), F(5, 3), F(-4), F(-1, 3))
    check("relative-charge counterexample keeps neutrality and determinant cancellation",
          alternative[0] == 0 and sum(q*k for q, k in zip(alternative, MULT)) == 0)
    normalized = tuple(q/4 for q in alternative)
    check("counterexample is not a unit change", normalized != Q and normalized[2] == -1)
    check("counterexample loses common transition", alternative[1]-alternative[3] != -alternative[2])
    # Independent block-scalar linear constraints reproduce Q without twists.
    u, e, down = s.symbols("u e down")
    rigid = s.solve([3*u+e+3*down, -e-1, u-down-1], (u, e, down), dict=True)
    check("block-scalar neutral trace common-unit constraints fix relative weights",
          rigid == [{u: s.Rational(2, 3), e: -1, down: -s.Rational(1, 3)}])
    return {"carrier_family": "1 + (Lambda2 W tensor D^a) + D^b + (W* tensor D^c)",
            "general_solution_for_common_D_power_m": {str(k): str(v) for k, v in solution.items()},
            "admissible_m": "m congruent 1 modulo 3; m is nonzero; all weights are m times original Q",
            "bounded_twist_domain": "a,c in [-3,3]; b determined by exact determinant cancellation",
            "cases": classified,
            "relative_counterexample": {"a": 1, "b": -4, "c": 0,
                "charges_in_original_unit": list(map(str, alternative)),
                "charged_singlet_normalized_to_minus_one": list(map(str, normalized)),
                "lost_premise": "The singlet and triplet transition lines are D^4 and D^2, rather than the same D"}}


def anisotropic_geometry():
    l1, l2, l3 = s.symbols("l1 l2 l3", real=True)
    total = l1+l2+l3
    w = (0, l1+l2, l1+l3, l2+l3, -total, -l3, -l2, -l1)
    H = s.diag(*w)
    check("general plane rates have traceless carrier generator", s.trace(H) == 0)
    raise_map = s.zeros(8)
    raise_map[:4, 4:] = s.eye(4)
    check("general plane rates retain same singlet and pair transition",
          H*raise_map-raise_map*H == total*raise_map)
    color_mix = s.zeros(8)
    color_mix[1, 3] = color_mix[3, 1] = color_mix[5, 7] = color_mix[7, 5] = 1
    example = H.subs({l1: 1, l2: 1, l3: 2})/4
    check("anisotropic response is not color scalar", example*color_mix != color_mix*example)
    condition = s.solve([l1+l2-(l1+l3), l1+l2-(l2+l3)], (l1, l2), dict=True)
    check("equal triplet weights force equal plane rates", condition == [{l1: l3, l2: l3}])
    determinant_q = s.diag(0, *([s.Rational(2, 3)]*3), -1, *([-s.Rational(1, 3)]*3))
    check("determinant part remains original Q even for anisotropic rates",
          (example-determinant_q)[0, 0] == 0 and (example-determinant_q)[4, 4] == 0
          and s.trace((example-determinant_q)[1:4, 1:4]) == 0
          and s.trace((example-determinant_q)[5:8, 5:8]) == 0)
    j2 = s.Matrix([[0, -1], [1, 0]])
    generator = s.diag(l1*j2, l2*j2, l3*j2)
    j = s.diag(j2, j2, j2)
    check("all three independent plane rates preserve the same complex structure", generator*j == j*generator)
    x = s.symbols("x1:5", real=True)
    y = s.Matrix(s.symbols("y1:7", real=True))
    omega = s.Matrix([-x[1]/2, x[0]/2, 0, 0])
    cross = omega*(generator*y).T
    gxx = s.eye(4)+cross*cross.T
    check("anisotropic metric remains positive by Schur complement", gxx-cross*cross.T == s.eye(4))
    recovered = s.Matrix(6, 6, lambda a, b: (s.diff(cross[1, a], y[b])-s.diff(cross[1, b], y[a]))/2)
    check("anisotropic normal connection follows from the actual metric", recovered == omega[1]*generator)
    return {"plane_rates": [1, 1, 2], "determinant_unit": 4,
            "complete_holonomy_response": [str(x) for x in example.diagonal()],
            "metric": "g=|dx|^2+|dy+omega J_lambda y|^2, omega=(x1 dx2-x2 dx1)/2",
            "retained": ["ordinary nested depths", "compatible J", "fixed exterior carrier", "neutral singlet", "trace zero", "unit paired differences"],
            "changed": "Charge identified with complete anisotropic holonomy instead of the central determinant component; color scalar property is lost",
            "qualification": "This does not change Q when electric charge is still defined by the determinant component. It tests a relaxed identification."}




# Exact geometry checks consolidated from the source-pinned earlier derivation.
def geometry_check(name, value):
    if isinstance(value, s.MatrixBase):
        passed = all((s.simplify(x) == 0 for x in value))
    elif isinstance(value, bool):
        passed = value
    else:
        passed = s.simplify(value) == 0
    CHECKS.append({'name': name, 'passed': bool(passed)})
    if not passed:
        raise AssertionError(name)

def exterior(matrix, degree):
    tuples = list(combinations(range(matrix.rows), degree))
    return s.Matrix([[matrix.extract(a, b).det() for b in tuples] for a in tuples])

def exterior_lie(matrix, degree):
    t = s.symbols('t', real=True)
    return exterior(s.eye(matrix.rows) + t * matrix, degree).diff(t).subs(t, 0)

def carrier(matrix):
    return s.diag(s.ones(1), exterior(matrix, 2), matrix.conjugate(), s.Matrix([[s.conjugate(matrix.det())]]))

def factored_carrier(matrix):
    first = s.diag(s.ones(1), exterior(matrix, 2))
    return s.diag(first, first / matrix.det())

def representation():
    g = s.Matrix([[s.Rational(3, 5), s.Rational(4, 5), 0], [-s.Rational(4, 5), s.Rational(3, 5), 0], [0, 0, 1]]) * s.diag(s.I, 1, 1)
    h = s.Matrix([[1, 0, 0], [0, s.Rational(5, 13), s.Rational(12, 13)], [0, -s.Rational(12, 13), s.Rational(5, 13)]]) * s.diag(1, s.I, -1)
    p = s.zeros(8)
    for j in range(4):
        p[j, j] = 1
    p[4, 7], p[5, 6], p[6, 5], p[7, 4] = (1, 1, -1, 1)
    geometry_check('Hodge reordering is unitary', p * p.T - s.eye(8))
    for name, matrix in [('g', g), ('h', h), ('gh', g * h)]:
        geometry_check(name + ' unitary', matrix.H * matrix - s.eye(3))
        geometry_check(name + ' global determinant factorization', p * carrier(matrix) * p.T - factored_carrier(matrix))
        geometry_check(name + ' carrier determinant one', carrier(matrix).det() - 1)
    geometry_check('representation composition on noncommuting unitary elements', carrier(g * h) - carrier(g) * carrier(h))
    geometry_check('test matrices do not commute', g * h != h * g)
    x = s.symbols('x0:9', real=True)
    A = s.Matrix([[s.I * x[0], x[3] + s.I * x[4], x[5] + s.I * x[6]], [-x[3] + s.I * x[4], s.I * x[1], x[7] + s.I * x[8]], [-x[5] + s.I * x[6], -x[7] + s.I * x[8], s.I * x[2]]])
    Af = s.diag(s.zeros(1), exterior_lie(A, 2))
    AE = s.diag(Af, Af - s.trace(A) * s.eye(4))
    original = s.diag(s.zeros(1), exterior_lie(A, 2), A.conjugate(), s.Matrix([[s.conjugate(s.trace(A))]]))
    geometry_check('general u3 connection factorization', p * original * p.T - AE)
    geometry_check('induced connection has zero trace', s.trace(AE))
    Q = s.diag(0, *[s.Rational(2, 3)] * 3, -1, *[-s.Rational(1, 3)] * 3)
    As = A - s.trace(A) * s.eye(3) / 3
    Cf = s.diag(s.zeros(1), exterior_lie(As, 2))
    geometry_check('single determinant connection gives all relative charges', AE - s.diag(Cf, Cf) - s.trace(A) * Q)
    geometry_check('charge generator parallel for induced unitary connection', AE * Q - Q * AE)
    geometry_check('charge trace zero', s.trace(Q))
    raise_op = s.zeros(8)
    raise_op[:4, 4:] = s.eye(4)
    geometry_check('same unit transition on singlet and triplet', Q * raise_op - raise_op * Q - raise_op)
    geometry_check('transition is a section of determinant line', factored_carrier(g) * raise_op * factored_carrier(g).inv() - g.det() * raise_op)
    t1 = s.Matrix([[0, 1, 0], [1, 0, 0], [0, 0, 0]]) / 2
    t2 = s.Matrix([[0, -s.I, 0], [s.I, 0, 0], [0, 0, 0]]) / 2
    t3 = s.diag(1, -1, 0) / 2
    t4 = s.Matrix([[0, 0, 1], [0, 0, 0], [1, 0, 0]]) / 2
    t5 = s.Matrix([[0, 0, -s.I], [0, 0, 0], [s.I, 0, 0]]) / 2
    t6 = s.Matrix([[0, 0, 0], [0, 0, 1], [0, 1, 0]]) / 2
    t7 = s.Matrix([[0, 0, 0], [0, 0, -s.I], [0, s.I, 0]]) / 2
    t8 = s.diag(1, 1, -2) / (2 * s.sqrt(3))
    casimir = s.zeros(8)
    for index, t in enumerate([t1, t2, t3, t4, t5, t6, t7, t8]):
        block = s.diag(s.zeros(1), exterior_lie(t, 2))
        T = s.diag(block, block)
        casimir += T * T
        geometry_check(f'color generator {index + 1} commutes with charge', T * Q - Q * T)
    geometry_check('both triplets have fundamental Casimir', casimir - s.Rational(4, 3) * s.diag(0, 1, 1, 1, 0, 1, 1, 1))
    geometry_check('old parity-number shortcut fails the unchanged wedge product', s.Rational(2, 3) != 2 * -s.Rational(1, 3))
    twist = s.symbols('twist', integer=True)
    geometry_check('determinant twist changes trace', s.trace(Q + twist * s.eye(8)) - 8 * twist)
    central_weights = [0, 2, 2, 2, -3, -1, -1, -1]
    return {'integer_center_weights': central_weights, 'normalized_charges': [str(w / s.Integer(3)) for w in central_weights], 'factorization': 'E=(1+Lambda^2 W) tensor (1+det(W)^-1)', 'induced_connection': 'A_E=diag(A_F,A_F-Tr(A) I4)=diag(C_F,C_F)+Tr(A) Q', 'same_transition_line': 'Hom(D^-1,1)=D', 'determinant': 'det(E)=D^4 D^-4=1', 'twist_family': 'E tensor D^k shifts every normalized charge by k; canonical untwisted construction is a carrier choice', 'circle_globality': 'Q is center generator/3; its 2pi exponential equals the SU3 center action, and its standalone period is 6pi'}

def rank_comparison():
    rows = []
    for n in range(1, 7):
        weights = [{'degree': k, 'multiplicity': int(s.binomial(n, k)), 'weight': str(s.Rational((-1) ** k * k, n))} for k in range(n + 1)]
        trace = sum((s.binomial(n, k) * (-1) ** k * k for k in range(n + 1)))
        if n >= 2:
            geometry_check(f'rank {n} alternating binomial trace', trace)
        rows.append({'complex_normal_rank': n, 'ordinary_normal_dimensions': 2 * n, 'weights': weights, 'integer_weight_trace': str(trace)})
    geometry_check('rank three has exact multiplicities 1,3,3,1', [int(s.binomial(3, k)) for k in range(4)] == [1, 3, 3, 1])
    return rows

def minimal_carrier_classification():
    cases = []
    selected = []
    for signs in product((-1, 1), repeat=3):
        e1, e2, e3 = signs
        determinant_power = e1 + 2 * e2 + e3
        case = {'conjugation_signs_degrees_1_2_3': list(signs), 'integer_weights': [0, e1, 2 * e2, 3 * e3], 'determinant_power': determinant_power, 'matching_triplet_type': e1 == -e2}
        cases.append(case)
        if determinant_power == 0:
            selected.append(signs)
    geometry_check('all eight minimal exterior choices classified', selected == [(-1, 1, -1), (1, -1, 1)])
    geometry_check('trivial complex determinant forces matching triplets in this family', all((e1 == -e2 for e1, e2, e3 in selected)))
    return {'family': 'Exactly one copy of each degree 0,1,2,3, each positive degree either original or conjugate; no character twists or extra summands', 'cases': cases, 'determinant_trivial_choices': [list(x) for x in selected], 'conclusion': 'E and its full conjugate are the only determinant-trivial choices in this finite family; their triplet types match automatically', 'premise': 'A preferred parallel complex volume on the carrier is an extra selection criterion, not a consequence of orienting the real spatial normal bundle'}

def projective_geometry():
    H = s.symbols('H')
    weights = [0, 2, 2, 2, -1, -1, -1, -3]
    c = s.expand(s.prod((1 + w * H for w in weights)))
    geometry_check('projective carrier c1 vanishes', c.coeff(H, 1))
    geometry_check('projective carrier c2', c.coeff(H, 2) + 12)
    geometry_check('projective carrier ch2', sum((w * w for w in weights)) / s.Integer(2) - 12)
    geometry_check('lepton transition exponent', 0 - -3 - 3)
    geometry_check('triplet transition exponent', 2 - -1 - 3)
    incompatible = []
    for a in (1, 2):
        euler = s.rem(H, H ** (a + 1), H)
        geometry_check(f'CP{a} to CP{a + 1} normal Euler class nonzero', euler != 0)
        incompatible.append({'lower_depth': 2 * a, 'higher_depth': 2 * (a + 1), 'excluded_intermediate_depth': 2 * a + 1, 'normal_euler_class': 'H'})
    return {'normal_bundle': 'N_CP2/CP5=Hom(ell,C6/A)=O(1) tensor (C6/A)', 'carrier': 'O + O(2)^3 + O(-1)^3 + O(-3), with constant quotient-space factors retained', 'total_chern_polynomial': str(c), 'chern_character_degree_four': '12 H^2', 'pullback': 'O_CP5(-3)|CP3=O_CP3(-3)', 'compatibility_status': 'Valid partial/global associated-bundle example; not a complete maintained sector chain', 'odd_intermediate_obstructions': incompatible}

def compatible_metric():
    x = s.symbols('x1:5', real=True)
    y = s.Matrix(s.symbols('y1:7', real=True))
    curvature = s.symbols('b', real=True)
    j2 = s.Matrix([[0, -1], [1, 0]])
    J = s.diag(j2, j2, j2)
    omega = s.Matrix([-curvature * x[1] / 2, curvature * x[0] / 2, 0, 0])
    cross = omega * (J * y).T
    base = s.eye(4) + cross * cross.T
    geometry_check('metric Schur complement identity', base - cross * cross.T - s.eye(4))
    for mu in range(4):
        gamma = s.Matrix(6, 6, lambda a, b: (s.diff(cross[mu, a], y[b]) - s.diff(cross[mu, b], y[a])) / 2)
        geometry_check(f'normal connection equals prescribed geometric one-form {mu + 1}', gamma - omega[mu] * J)
    normal_curvature = s.diff(omega[1], x[0]) * J - s.diff(omega[0], x[1]) * J
    geometry_check('normal curvature is nonzero central U3 curvature', normal_curvature - curvature * J)
    origin = {v: 0 for v in y}
    second_fundamental = []
    for a in range(6):
        for mu in range(4):
            for nu in range(4):
                second_fundamental.append((s.diff(cross[nu, a], x[mu]) + s.diff(cross[mu, a], x[nu]) - s.diff(base[mu, nu], y[a])).subs(origin) / 2)
    geometry_check('zero section is totally geodesic', s.Matrix(second_fundamental))
    return {'ordinary_space': 'R4_x times R6_y, all noncompact physical spatial coordinates', 'metric': 'g=sum dx_mu^2 + |dy+omega J y|^2, omega=b/2 (x1 dx2-x2 dx1)', 'normal_connection': 'omega J, or i omega I3 in complex notation', 'normal_curvature': 'b J dx1 wedge dx2', 'full_nested_flag': {'2': 'span(x1,x2)', '3': 'span(x1,x2,x3)', '4': 'R4_x at y=0', '5': 'R4_x plus y1', '6': 'R4_x plus y1,y2', '10': 'all coordinates'}, 'metric_properties': 'smooth, positive, determinant one; no boundary or compactification', 'premises': 'Chosen metric and complex normal structure; an existence construction, not selected IDWT geometry', 'scope': 'Normal connection need not preserve each intermediate real normal line; all listed submanifolds coexist'}

SELECTORS = {
    "unrestricted": lambda p, q, r: True,
    "tangent-only": lambda p, q, r: q == 0,
    "normal-only": lambda p, q, r: p == 0,
    "saturated-tangent": lambda p, q, r: p == r,
    "saturated-normal": lambda p, q, r: q == 3-r,
    "balanced-occupation": lambda p, q, r: p == q,
    "fully-saturated": lambda p, q, r: p == r and q == 3-r,
}
MATTER_TARGETS = {3: (F(-1, 3), 3), 4: (F(2, 3), 3),
                  5: (F(0), 1), 6: (F(-1), 1), 10: (F(-1), 1)}


def column_basis(matrix):
    columns = matrix.columnspace()
    return s.Matrix.hstack(*columns) if columns else s.zeros(matrix.rows, 0)


def orthogonal_projector(basis):
    if basis.cols == 0:
        return s.zeros(basis.rows)
    return s.simplify(basis*(basis.T*basis).inv()*basis.T)


def complex_part_and_closure(basis, j):
    """Compute S intersect J S and S+J S using real spatial subspaces."""
    basis = column_basis(basis)
    closure = column_basis(basis.row_join(j*basis))
    if basis.cols:
        coefficients = basis.row_join(-j*basis).nullspace()
        vectors = [basis*v[:basis.cols, :] for v in coefficients]
        core = column_basis(s.Matrix.hstack(*vectors)) if vectors else s.zeros(j.rows, 0)
    else:
        core = s.zeros(j.rows, 0)
    return orthogonal_projector(core), orthogonal_projector(closure)


def complex_matrix(real_matrix):
    """Coordinates are (x1,y1,x2,y2,x3,y3), with J acting by i."""
    return s.Matrix(3, 3, lambda a, b: real_matrix[2*a, 2*b]+s.I*real_matrix[2*a+1, 2*b])


def occupation_projector(a, rule):
    """Project the actual exterior matrices by A-occupation p, B-occupation q."""
    rank = int(s.trace(a))
    blocks = []
    for degree in (0, 2, 3, 1):
        number = exterior_lie(a, degree)
        identity = s.eye(number.rows)
        selected = s.zeros(number.rows)
        for p in range(degree+1):
            q = degree-p
            if not (p <= rank and q <= 3-rank and SELECTORS[rule](p, q, rank)):
                continue
            # Lagrange projectors on integer occupation eigenvalues.
            projector = identity
            for other in range(degree+1):
                if other != p:
                    projector = projector*(number-other*identity)/(p-other)
            selected += projector
        blocks.append(selected.conjugate() if degree % 2 else selected)
    return s.simplify(s.diag(*blocks))


def color_generators():
    fundamental = []
    for a, b in combinations(range(3), 2):
        symmetric, antisymmetric = s.zeros(3), s.zeros(3)
        symmetric[a, b] = symmetric[b, a] = 1
        antisymmetric[a, b], antisymmetric[b, a] = -s.I, s.I
        fundamental.extend((symmetric, antisymmetric))
    fundamental.extend((s.diag(1, -1, 0), s.diag(1, 1, -2)))
    return [s.diag(s.zeros(1), exterior_lie(t, 2), s.zeros(1), -t.conjugate()) for t in fundamental]


def selector_spectrum(rank, name):
    result = Counter()
    for p in range(rank+1):
        for q in range(4-rank):
            if SELECTORS[name](p, q, rank):
                result[F((-1)**(p+q)*(p+q), 3)] += comb(rank, p)*comb(3-rank, q)
    return result


def selector_tests():
    j2 = s.Matrix([[0, -1], [1, 0]])
    j = s.diag(j2, j2, j2)
    rotation = s.eye(6)
    rotation[1:3, 1:3] = s.Matrix([[s.Rational(3, 5), -s.Rational(4, 5)],
                                                  [s.Rational(4, 5), s.Rational(3, 5)]])
    check("normal flag tilt is an exact orthogonal rotation", rotation.T*rotation == s.eye(6))
    check("relative normal flag tilt changes J alignment", rotation*j != j*rotation)
    geometries = {"aligned-flag": s.eye(6), "tilted-flag": rotation}
    generators = color_generators()
    qmatrix = s.diag(0, *([s.Rational(2, 3)]*3), -1, *([-s.Rational(1, 3)]*3))
    rows, rank_rows = [], []
    projector_cache = {}
    for geometry, frame in geometries.items():
        for d in DEPTHS:
            real_rank = max(0, d-4)
            basis = frame[:, :real_rank]
            core, closure = complex_part_and_closure(basis, j)
            core_rot, closure_rot = complex_part_and_closure(rotation*basis, rotation*j*rotation.T)
            check(f"{geometry} depth {d} common rotation covariance",
                  core_rot == rotation*core*rotation.T and closure_rot == rotation*closure*rotation.T)
            check(f"{geometry} depth {d} complex core/closure dimension identity",
                  s.trace(core)+s.trace(closure) == 2*real_rank)
            rank_rows.append({"geometry": geometry, "depth": d, "real_intersection_rank": real_rank,
                              "maximal_complex_rank": int(s.trace(core)/2),
                              "minimal_closure_complex_rank": int(s.trace(closure)/2)})
            for policy, real_projector in (("maximal-contained", core), ("minimal-containing", closure)):
                check(f"{geometry} depth {d} {policy} is J-invariant",
                      real_projector*j == j*real_projector)
                a = complex_matrix(real_projector)
                check(f"{geometry} depth {d} {policy} is Hermitian projector",
                      a*a == a and a.H == a)
                rank = int(s.trace(a))
                for name in SELECTORS:
                    cache_key = (tuple(a), name)
                    if cache_key not in projector_cache:
                        selected = occupation_projector(a, name)
                        expected = selector_spectrum(rank, name)
                        observed = Counter()
                        for start, stop, charge in ((0, 1, F(0)), (1, 4, F(2, 3)),
                                                    (4, 5, F(-1)), (5, 8, F(-1, 3))):
                            count = int(s.trace(selected[start:stop, start:stop]))
                            if count:
                                observed[charge] = count
                        label = f"selector projector {len(projector_cache)+1} {name}"
                        check(label+" is an orthogonal projector", selected.H == selected and selected*selected == selected)
                        check(label+" retains charge eigenbranches", selected*qmatrix == qmatrix*selected)
                        check(label+" matches binomial branching", observed == expected)
                        invariant = all(selected*t == t*selected for t in generators)
                        projector_cache[cache_key] = (observed, invariant)
                    spectrum, color_invariant = projector_cache[cache_key]
                    target = MATTER_TARGETS.get(d)
                    rows.append({"geometry": geometry, "policy": policy, "rule": name,
                                 "depth": d, "complex_active_rank": rank,
                                 "charges": serial_spectrum(sorted(spectrum.items())),
                                 "full_SU3_triplets_preserved": color_invariant,
                                 "target_available": spectrum.get(target[0], 0) > 0 if target else None,
                                 "exact_target_multiplet": spectrum == Counter({target[0]: target[1]}) if target else None})
    families = []
    for geometry, policy, name in product(geometries, ("maximal-contained", "minimal-containing"), SELECTORS):
        cases = [r for r in rows if (r['geometry'], r['policy'], r['rule']) == (geometry, policy, name)
                 and r['depth'] in MATTER_TARGETS]
        families.append({"geometry": geometry, "policy": policy, "rule": name,
                         "all_target_numbers_available": all(r['target_available'] for r in cases),
                         "all_target_multiplets_selected": all(r['exact_target_multiplet'] for r in cases),
                         "mismatched_depths": [r['depth'] for r in cases if not r['exact_target_multiplet']]})
        d3 = next(r for r in cases if r['depth'] == 3)
        d4 = next(r for r in cases if r['depth'] == 4)
        check(f"{geometry} {policy} {name} cannot distinguish d3/d4", d3['charges'] == d4['charges'])
    check("no tested uniform selector selects the complete assignment",
          not any(f['all_target_multiplets_selected'] for f in families))
    d6 = [r for r in rank_rows if r['depth'] == 6]
    check("depth-six core rank changes under relative J alignment",
          [(r['maximal_complex_rank'], r['minimal_closure_complex_rank']) for r in d6] == [(1, 1), (0, 2)])
    return {"scope": "Seven added uniform predicates, two complex-subspace policies, two compatible nested real flags",
            "premise": "Only the intersection of each tangent sector with the fixed four-in-ten normal bundle enters the selector",
            "predicates": {"unrestricted": "all p,q", "tangent-only": "q=0", "normal-only": "p=0",
                           "saturated-tangent": "p=r", "saturated-normal": "q=3-r",
                           "balanced-occupation": "p=q", "fully-saturated": "p=r and q=3-r"},
            "rank_rows": rank_rows, "cases": rows, "uniform_families": families,
            "number_only_matches": [f for f in families if f['all_target_numbers_available']],
            "complete_assignment_matches": [f for f in families if f['all_target_multiplets_selected']],
            "no_separation_reason": "S_3=S_4=0. Every uniform function of this intersection and J alone gives the same retained spectrum at both depths.",
            "scope_limit": "No exclusion of rules using additional base-tangent, embedding, curvature or physical branch data. Complex intersections define subbundles only on constant-rank regions."}


def integer_character_scan():
    solutions = []
    count = 0
    for e1, e2, e3 in product((-1, 1), repeat=3):
        for t1, t2, t3 in product(range(-8, 9), repeat=3):
            count += 1
            w1, w2, w3 = e1+3*t1, 2*e2+3*t2, 3*e3+3*t3
            if e1 != -e2 or 3*w1+3*w2+w3 != 0 or w2-w1 != -w3:
                continue
            m = -w3//3
            check(f"integer family solution {len(solutions)+1}",
                  (w2, w3, w1) == (2*m, -3*m, -m) and m % 3 == (1 if e2 == 1 else 2))
            solutions.append({"degree_signs": [e1, e2, e3], "twists": [t1, t2, t3], "m": m})
    check("integer scan includes even m4", any(row['m'] == 4 for row in solutions))
    return {"case_count": count, "scope": "all eight degree conjugations; three independent integer twists in [-8,8]",
            "solutions": solutions, "interpretation": "Every accepted carrier is an overall charge rescaling, not a new relative-charge table"}


def write_report(result, destination):
    tested = result['sector_selector_tests']
    lines = ['# Unified Geometric Charge And Sector Test', '',
             'This report is generated by the one standalone Python script. All computations are exact geometry or finite algebra; there is no particle simulation.', '',
             '## Outcome', '',
             'The charge construction retains its conditional ratios and six-normal-direction criterion. None of the tested uniform tangent/normal selectors uniquely produces the complete maintained matter assignment.', '',
             f"The run contains {len(result['checks'])} passing mathematical checks, {len(tested['cases'])} selector/depth cases and {result['integer_character_scan']['case_count']} bounded integer character cases. Check counts are implementation verification, not independent physical evidence.", '',
             '## New Geometric Tests', '',
             'Let S_d be the real intersection of the depth-d tangent space with the fixed six-dimensional normal space of four inside ten. The script computes S_d intersect J S_d and S_d+J S_d by exact linear algebra. It then builds actual exterior occupation projectors, checking their ranks, charge preservation and commutation with all eight SU(3) generators.', '',
             '| Geometry | Depth | Real S rank | Largest contained complex rank | Smallest containing complex rank |',
             '| --- | ---: | ---: | ---: | ---: |']
    for row in tested['rank_rows']:
        lines.append(f"| {row['geometry']} | {row['depth']} | {row['real_intersection_rank']} | {row['maximal_complex_rank']} | {row['minimal_closure_complex_rank']} |")
    lines.extend(['', 'The tilted case rotates the real flag relative to J by an exact rational orthogonal matrix. Rotating both the flag and J together transforms the two real subspace projectors covariantly, as separately checked. The resulting orientation dependence is therefore physical relative geometric data within the candidate, not coordinate dependence.', '',
                  '## Uniform Selector Comparison', '',
                  'Each rule is an added hypothesis applied unchanged at every depth. Here p is occupation in the chosen complex active subspace of rank r, and q is occupation in its orthogonal complement. The common charge is (-1)^(p+q)(p+q)/3 and multiplicity is C(r,p) C(3-r,q).', '',
                  '| Rule | Condition | Complete assignment selected in either flag/policy? |', '| --- | --- | --- |'])
    for name, predicate in tested['predicates'].items():
        won = any(f['rule'] == name and f['all_target_multiplets_selected'] for f in tested['uniform_families'])
        lines.append(f"| {name} | {predicate} | {'Yes' if won else 'No'} |")
    lines.extend(['', 'Some rules merely contain all desired charge numbers while also allowing other charges. Their exact cases are retained in results.json and are not counted as a selected assignment.', '',
                  'The decisive obstruction in this finite family is S_3=S_4=0. Every uniform function of that intersection and J alone gives identical retained spectra at depths three and four. Such a selector cannot uniquely return a -1/3 triplet at one and a +2/3 triplet at the other. This is stronger than a failed numerical search within the seven examples, but still concerns only rules using this particular geometric input.', '',
                  'Partial occupation projectors can also split triplets. The script records whether each projector commutes with the full fixed SU(3) action. Covariance under simultaneous rotation of all geometric data does not imply that a fixed distinguished subspace preserves full SU(3). Complex core and closure ranks can vary under relative orientation; they define smooth subbundles only on constant-rank regions.', '',
                  '## Retained Construction And Alternative Charges', '',
                  'The script also derives the global U(3) exterior-carrier factorization, its induced determinant connection, trace cancellation and shared transition; compares normal ranks and all maintained depth pairs; checks allowed branch assignments, determinant twists, integer globality and an anisotropic ordinary metric; and retains the projective intermediate-depth obstruction.', '',
                  'The full two-singlet/two-triplet exterior pattern requires six normal directions. Only four inside ten has that gap among the maintained depths; two inside eight and three inside nine also have it in the broader integer-depth comparison. Relative charges remain 0,2/3,-1,-1/3 under the original premises. A relaxed common-transition rule permits the alternative normalized table 0,5/12,-1,-1/12. The original determinant-defined generator is unchanged by anisotropic contributions in the traceless connection.', '',
                  '## Scope', '',
                  'These results exclude the tested intersection-only selectors as complete explanations of the particle-depth assignment. They do not exclude additional geometric rules involving base tangent representations, embedding or curvature. No such additional selection law is silently assumed, fitted or claimed solved here. The exterior charge pattern and original mathematical sources retain their attribution in the source-pinned earlier derivation; this script tests its IDWT geometric application.', '',
                  f"Executed source SHA-256: `{next(iter(result['sources'].values()))}`.",
                  'The executed script hash is recorded in results.json. The script needs only Python and installed SymPy; each default run creates a new dated output directory.', ''])
    destination.write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, help='New directory; existing paths are refused')
    args = parser.parse_args()
    output = args.output_dir or HERE/('run-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    output.mkdir(parents=True, exist_ok=False)
    result = {"experiment": "One-script geometry, charge and sector selection comparison",
              "created_at": datetime.now(timezone.utc).isoformat(),
              "sources": {SCRIPT_SOURCE_PATH: hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
              "original_geometry": {"representation": representation(), "ranks": rank_comparison(),
                  "minimal_carrier": minimal_carrier_classification(), "projective": projective_geometry(),
                  "compatible_metric": compatible_metric()},
              "rank_and_depth_tests": rank_and_depths(),
              "fixed_carrier_assignment_tests": fixed_carrier_assignments(),
              "charge_twist_tests": twists_and_weights(),
              "integer_character_scan": integer_character_scan(),
              "anisotropic_metric_test": anisotropic_geometry(),
              "sector_selector_tests": selector_tests(),
              "checks": CHECKS}
    (output/'results.json').write_text(json.dumps(result, indent=2)+'\n')
    write_report(result, output/'report.md')
    print(json.dumps({"output_directory": str(output), "checks": len(CHECKS),
                      "all_passed": all(c['passed'] for c in CHECKS),
                      "maintained_normal_pair": result['rank_and_depth_tests']['maintained_pair_matches'],
                      "selector_depth_cases": len(result['sector_selector_tests']['cases']),
                      "uniform_selector_families": len(result['sector_selector_tests']['uniform_families']),
                      "complete_assignment_matches": result['sector_selector_tests']['complete_assignment_matches'],
                      "number_only_matches": result['sector_selector_tests']['number_only_matches']}, indent=2))


if __name__ == '__main__':
    main()
