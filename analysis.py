"""
Supply Chain Optimisation for GreenGlow Cosmetics
Deterministic MILP + Stochastic extension (five scenarios)
Implementation: PuLP

Repo design choice:
This script contains a self contained parameter set so it can run without external files.
Replace the parameter dictionaries with your dataset later if you want.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, List
import random

import pulp as pl


# =========================
# Sets
# =========================
SUPPLIERS = ["S1", "S2", "S3", "S4"]
PLANTS = ["Europe", "NorthAmerica", "Asia", "SouthAmerica"]
MODES = ["Sea", "Air"]
PRODUCTS = ["A", "B", "C"]
REGIONS = ["Africa", "Asia", "Europe", "MiddleEast", "NorthAmerica", "SouthAmerica"]


# =========================
# Base parameters (editable)
# =========================
# Supplier monthly capacities
q_s: Dict[str, float] = {
    "S1": 250_000,
    "S2": 180_000,
    "S3": 400_000,
    "S4": 220_000,
}

# Supplier unit procurement costs
c_s: Dict[str, float] = {
    "S1": 15,
    "S2": 25,
    "S3": 10,
    "S4": 20,
}

# Mode availability by supplier (1 allowed, 0 not allowed)
a_sm: Dict[Tuple[str, str], int] = {}
for s in SUPPLIERS:
    for m in MODES:
        a_sm[(s, m)] = 1

# Constraints from your write up
# S2 is air only, S3 is sea only
a_sm[("S2", "Sea")] = 0
a_sm[("S2", "Air")] = 1
a_sm[("S3", "Sea")] = 1
a_sm[("S3", "Air")] = 0

# Raw material shipping cost from supplier by mode (simplified)
ship_cost_sm: Dict[Tuple[str, str], float] = {
    ("S1", "Sea"): 1.2,
    ("S1", "Air"): 4.5,
    ("S2", "Sea"): 99.0,  # blocked by availability anyway
    ("S2", "Air"): 5.5,
    ("S3", "Sea"): 1.0,
    ("S3", "Air"): 99.0,  # blocked
    ("S4", "Sea"): 1.4,
    ("S4", "Air"): 4.8,
}

# Raw material shipping emissions from supplier by mode (simplified, kg CO2 per unit)
ship_emis_sm: Dict[Tuple[str, str], float] = {
    ("S1", "Sea"): 0.06,
    ("S1", "Air"): 0.22,
    ("S2", "Sea"): 99.0,
    ("S2", "Air"): 0.25,
    ("S3", "Sea"): 0.05,
    ("S3", "Air"): 99.0,
    ("S4", "Sea"): 0.07,
    ("S4", "Air"): 0.23,
}

# Plant capacities (units)
cap_p: Dict[str, float] = {
    "Europe": 800_000,
    "NorthAmerica": 1_200_000,
    "Asia": 1_500_000,
    "SouthAmerica": 600_000,
}

# Plant operating cost per unit produced
oc_p: Dict[str, float] = {
    "Europe": 12,
    "NorthAmerica": 10,
    "Asia": 8,
    "SouthAmerica": 11,
}

# Plant production emissions per unit produced (kg CO2)
ce_p: Dict[str, float] = {
    "Europe": 0.8,
    "NorthAmerica": 0.7,
    "Asia": 0.6,
    "SouthAmerica": 0.9,
}

# Transport cost from plant to region (per unit)
# Simplified: intra region cheaper than inter region
tc_pd: Dict[Tuple[str, str], float] = {}
te_pd: Dict[Tuple[str, str], float] = {}

for p in PLANTS:
    for d in REGIONS:
        same = (
            (p == "Asia" and d == "Asia")
            or (p == "Europe" and d == "Europe")
            or (p == "NorthAmerica" and d == "NorthAmerica")
            or (p == "SouthAmerica" and d == "SouthAmerica")
        )
        tc_pd[(p, d)] = 1.5 if same else 4.0
        te_pd[(p, d)] = 0.10 if same else 0.28

# Demand by product and region (units)
# If you have your exact demand table, replace these values.
dem_kd: Dict[Tuple[str, str], float] = {}
for k in PRODUCTS:
    for d in REGIONS:
        base = {
            "Asia": 1_000_000,
            "NorthAmerica": 800_000,
            "Europe": 700_000,
            "MiddleEast": 500_000,
            "Africa": 400_000,
            "SouthAmerica": 500_000,
        }[d]
        # product split (A slightly higher)
        mult = {"A": 0.44, "B": 0.33, "C": 0.23}[k]
        dem_kd[(k, d)] = base * mult

# Shortage penalty by region (per unit)
pen_d: Dict[str, float] = {
    "Asia": 4,
    "NorthAmerica": 5,
    "Europe": 6,
    "SouthAmerica": 5,
    "MiddleEast": 8,
    "Africa": 7,
}

# Raw material requirements rho[k,s] per unit of product
# Replace with your formulation matrix if different.
rho_ks: Dict[Tuple[str, str], float] = {
    # Product A uses S1 S2 S3
    ("A", "S1"): 0.2,
    ("A", "S2"): 0.2,
    ("A", "S3"): 0.2,
    ("A", "S4"): 0.0,
    # Product B uses all four
    ("B", "S1"): 0.1,
    ("B", "S2"): 0.1,
    ("B", "S3"): 0.2,
    ("B", "S4"): 0.1,
    # Product C uses S2 S3 S4
    ("C", "S1"): 0.0,
    ("C", "S2"): 0.2,
    ("C", "S3"): 0.2,
    ("C", "S4"): 0.1,
}

# Objective weights
ALPHA_COST = 0.3
ALPHA_EMIS = 0.3
ALPHA_PEN = 0.4

# Minimum service level beta
BETA = 0.12


# =========================
# Deterministic model
# =========================
@dataclass
class SolutionSummary:
    status: str
    weighted_objective: float
    total_cost: float
    total_emissions: float
    total_penalties: float
    production: Dict[Tuple[str, str], float]
    shortages: Dict[Tuple[str, str], float]


def solve_deterministic() -> SolutionSummary:
    prob = pl.LpProblem("GreenGlow_Supply_Chain_Deterministic", pl.LpMinimize)

    # Variables
    X = pl.LpVariable.dicts("X", (SUPPLIERS, PLANTS, MODES), lowBound=0)  # raw materials
    Y = pl.LpVariable.dicts("Y", (PLANTS, PRODUCTS), lowBound=0)         # production
    Z = pl.LpVariable.dicts("Z", (PLANTS, PRODUCTS, REGIONS), lowBound=0)  # distribution
    S = pl.LpVariable.dicts("Short", (PRODUCTS, REGIONS), lowBound=0)    # shortages

    # Objective components
    # Procurement + inbound shipping
    cost_inbound = pl.lpSum(
        (c_s[s] + ship_cost_sm[(s, m)]) * X[s][p][m]
        for s in SUPPLIERS for p in PLANTS for m in MODES
        if a_sm[(s, m)] == 1
    )

    # Production cost
    cost_prod = pl.lpSum(
        oc_p[p] * Y[p][k] for p in PLANTS for k in PRODUCTS
    )

    # Outbound distribution cost
    cost_out = pl.lpSum(
        tc_pd[(p, d)] * Z[p][k][d] for p in PLANTS for k in PRODUCTS for d in REGIONS
    )

    total_cost = cost_inbound + cost_prod + cost_out

    # Emissions: inbound shipping + production + outbound shipping
    emis_inbound = pl.lpSum(
        ship_emis_sm[(s, m)] * X[s][p][m]
        for s in SUPPLIERS for p in PLANTS for m in MODES
        if a_sm[(s, m)] == 1
    )
    emis_prod = pl.lpSum(
        ce_p[p] * Y[p][k] for p in PLANTS for k in PRODUCTS
    )
    emis_out = pl.lpSum(
        te_pd[(p, d)] * Z[p][k][d] for p in PLANTS for k in PRODUCTS for d in REGIONS
    )
    total_emis = emis_inbound + emis_prod + emis_out

    # Penalties
    total_pen = pl.lpSum(
        pen_d[d] * S[k][d] for k in PRODUCTS for d in REGIONS
    )

    # Weighted objective
    prob += ALPHA_COST * total_cost + ALPHA_EMIS * total_emis + ALPHA_PEN * total_pen

    # Constraints
    # Supplier capacity
    for s in SUPPLIERS:
        prob += pl.lpSum(X[s][p][m] for p in PLANTS for m in MODES if a_sm[(s, m)] == 1) <= q_s[s]

    # Mode availability
    for s in SUPPLIERS:
        for p in PLANTS:
            for m in MODES:
                if a_sm[(s, m)] == 0:
                    prob += X[s][p][m] == 0

    # Raw material sufficient for production at each plant
    for p in PLANTS:
        for s in SUPPLIERS:
            prob += pl.lpSum(X[s][p][m] for m in MODES if a_sm[(s, m)] == 1) >= pl.lpSum(
                rho_ks[(k, s)] * Y[p][k] for k in PRODUCTS
            )

    # Plant capacity
    for p in PLANTS:
        prob += pl.lpSum(Y[p][k] for k in PRODUCTS) <= cap_p[p]

    # Shipments cannot exceed production
    for p in PLANTS:
        for k in PRODUCTS:
            prob += pl.lpSum(Z[p][k][d] for d in REGIONS) <= Y[p][k]

    # Demand satisfaction with shortages
    for k in PRODUCTS:
        for d in REGIONS:
            prob += pl.lpSum(Z[p][k][d] for p in PLANTS) + S[k][d] >= dem_kd[(k, d)]

    # Minimum service level constraint
    for k in PRODUCTS:
        for d in REGIONS:
            prob += pl.lpSum(Z[p][k][d] for p in PLANTS) >= BETA * dem_kd[(k, d)]

    # Solve
    prob.solve(pl.PULP_CBC_CMD(msg=False))

    status = pl.LpStatus[prob.status]
    weighted = float(pl.value(prob.objective))

    # Extract components
    cost_val = float(pl.value(total_cost))
    emis_val = float(pl.value(total_emis))
    pen_val = float(pl.value(total_pen))

    production = {(p, k): float(Y[p][k].value()) for p in PLANTS for k in PRODUCTS}
    shortages = {(k, d): float(S[k][d].value()) for k in PRODUCTS for d in REGIONS}

    return SolutionSummary(
        status=status,
        weighted_objective=weighted,
        total_cost=cost_val,
        total_emissions=emis_val,
        total_penalties=pen_val,
        production=production,
        shortages=shortages,
    )


# =========================
# Stochastic extension
# =========================
def solve_stochastic(num_scenarios: int = 5, seed: int = 25) -> None:
    random.seed(seed)

    scenarios = list(range(num_scenarios))
    prob_s = 1.0 / num_scenarios

    # Scenario multipliers
    # Supplier capacity varies 80 to 120 percent
    cap_mult = {(s, w): random.uniform(0.8, 1.2) for s in SUPPLIERS for w in scenarios}
    # Demand varies 90 to 110 percent
    dem_mult = {(k, d, w): random.uniform(0.9, 1.1) for k in PRODUCTS for d in REGIONS for w in scenarios}

    prob = pl.LpProblem("GreenGlow_Supply_Chain_Stochastic", pl.LpMinimize)

    # First stage expansion decisions
    Expand = pl.LpVariable.dicts("Expand", PLANTS, cat=pl.LpBinary)

    # Second stage variables per scenario
    X = pl.LpVariable.dicts("X", (scenarios, SUPPLIERS, PLANTS, MODES), lowBound=0)
    Y = pl.LpVariable.dicts("Y", (scenarios, PLANTS, PRODUCTS), lowBound=0)
    Z = pl.LpVariable.dicts("Z", (scenarios, PLANTS, PRODUCTS, REGIONS), lowBound=0)
    S = pl.LpVariable.dicts("Short", (scenarios, PRODUCTS, REGIONS), lowBound=0)

    # Expansion cost
    # 50 percent capacity addition, cost 1000 per unit of added capacity
    expand_cost = pl.lpSum(Expand[p] * (0.5 * cap_p[p] * 1000) for p in PLANTS)

    # Expected objective across scenarios
    expected_cost = []
    expected_emis = []
    expected_pen = []

    for w in scenarios:
        cost_in = pl.lpSum(
            (c_s[s] + ship_cost_sm[(s, m)]) * X[w][s][p][m]
            for s in SUPPLIERS for p in PLANTS for m in MODES
            if a_sm[(s, m)] == 1
        )
        cost_prod = pl.lpSum(oc_p[p] * Y[w][p][k] for p in PLANTS for k in PRODUCTS)
        cost_out = pl.lpSum(tc_pd[(p, d)] * Z[w][p][k][d] for p in PLANTS for k in PRODUCTS for d in REGIONS)
        expected_cost.append(cost_in + cost_prod + cost_out)

        emis_in = pl.lpSum(
            ship_emis_sm[(s, m)] * X[w][s][p][m]
            for s in SUPPLIERS for p in PLANTS for m in MODES
            if a_sm[(s, m)] == 1
        )
        emis_prod = pl.lpSum(ce_p[p] * Y[w][p][k] for p in PLANTS for k in PRODUCTS)
        emis_out = pl.lpSum(te_pd[(p, d)] * Z[w][p][k][d] for p in PLANTS for k in PRODUCTS for d in REGIONS)
        expected_emis.append(emis_in + emis_prod + emis_out)

        expected_pen.append(pl.lpSum(pen_d[d] * S[w][k][d] for k in PRODUCTS for d in REGIONS))

    exp_cost = prob_s * pl.lpSum(expected_cost)
    exp_emis = prob_s * pl.lpSum(expected_emis)
    exp_pen = prob_s * pl.lpSum(expected_pen)

    prob += expand_cost + (ALPHA_COST * exp_cost + ALPHA_EMIS * exp_emis + ALPHA_PEN * exp_pen)

    # Constraints per scenario
    for w in scenarios:
        # Supplier capacity with uncertainty
        for s in SUPPLIERS:
            prob += pl.lpSum(
                X[w][s][p][m] for p in PLANTS for m in MODES if a_sm[(s, m)] == 1
            ) <= q_s[s] * cap_mult[(s, w)]

        # Mode availability
        for s in SUPPLIERS:
            for p in PLANTS:
                for m in MODES:
                    if a_sm[(s, m)] == 0:
                        prob += X[w][s][p][m] == 0

        # Raw materials for production
        for p in PLANTS:
            for s in SUPPLIERS:
                prob += pl.lpSum(X[w][s][p][m] for m in MODES if a_sm[(s, m)] == 1) >= pl.lpSum(
                    rho_ks[(k, s)] * Y[w][p][k] for k in PRODUCTS
                )

        # Plant capacity with expansion
        for p in PLANTS:
            prob += pl.lpSum(Y[w][p][k] for k in PRODUCTS) <= cap_p[p] * (1 + 0.5 * Expand[p])

        # Shipments <= production
        for p in PLANTS:
            for k in PRODUCTS:
                prob += pl.lpSum(Z[w][p][k][d] for d in REGIONS) <= Y[w][p][k]

        # Demand with uncertainty + shortages
        for k in PRODUCTS:
            for d in REGIONS:
                dem = dem_kd[(k, d)] * dem_mult[(k, d, w)]
                prob += pl.lpSum(Z[w][p][k][d] for p in PLANTS) + S[w][k][d] >= dem
                prob += pl.lpSum(Z[w][p][k][d] for p in PLANTS) >= BETA * dem

    prob.solve(pl.PULP_CBC_CMD(msg=False))

    print("\nStochastic model status:", pl.LpStatus[prob.status])
    print("Objective value:", float(pl.value(prob.objective)))

    print("\nExpansion decisions")
    for p in PLANTS:
        print(p, int(round(Expand[p].value())))

    # Print one scenario summary
    w0 = 0
    print("\nScenario 0 production (units)")
    for p in PLANTS:
        for k in PRODUCTS:
            v = float(Y[w0][p][k].value())
            if v > 1e-6:
                print(p, k, round(v, 2))

    print("\nScenario 0 shortages (units)")
    for k in PRODUCTS:
        for d in REGIONS:
            v = float(S[w0][k][d].value())
            if v > 1e-6:
                print(k, d, round(v, 2))


def main() -> None:
    det = solve_deterministic()
    print("\nDeterministic model status:", det.status)
    print("Weighted objective:", round(det.weighted_objective, 2))
    print("Total cost:", round(det.total_cost, 2))
    print("Total emissions (kg CO2):", round(det.total_emissions, 2))
    print("Total penalties:", round(det.total_penalties, 2))

    print("\nProduction allocation (units)")
    for p in PLANTS:
        tot = sum(det.production[(p, k)] for k in PRODUCTS)
        print(p, round(tot, 2), "total")
        for k in PRODUCTS:
            print(" ", k, round(det.production[(p, k)], 2))

    print("\nShortages (units)")
    for k in PRODUCTS:
        for d in REGIONS:
            v = det.shortages[(k, d)]
            if v > 1e-6:
                print(k, d, round(v, 2))

    solve_stochastic(num_scenarios=5, seed=25)


if __name__ == "__main__":
    main()
