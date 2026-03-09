"""
Business Impact Analysis & A/B Testing Framework
"""

import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR    = os.path.join(BASE_DIR, "data", "raw")
REPORT_DIR = os.path.join(BASE_DIR, "models", "reports")
os.makedirs(REPORT_DIR, exist_ok=True)



# METRIC TRANSLATION: Offline → Business Impact

def translate_offline_to_business(
    precision_at_10:     float,
    ndcg_at_10:          float,
    baseline_acceptance: float = 0.13,
    baseline_aov:        float = 380.0,
    avg_addon_price:     float = 120.0,
    daily_orders:        int   = 3_000_000,
    commission_rate:     float = 0.20,
) -> dict:
    """
    Translate offline ML metrics to projected business outcomes.

    Key assumptions (derived from problem statement + industry benchmarks):
    - Precision@10 ≈ acceptance rate (items recommended that get added)
    - NDCG@10 × scaling factor → click-through rate improvement
    - Each accepted add-on increases AOV by its price
    """

    projected_acceptance = precision_at_10 * 0.75
    acceptance_lift      = projected_acceptance - baseline_acceptance

    # AOV lift 
    # If acceptance_rate% of sessions add 1+ items at avg_addon_price
    # AOV lift = acceptance_rate × avg_addon_price / baseline_aov
    aov_lift_abs = projected_acceptance * avg_addon_price
    aov_lift_pct = aov_lift_abs / baseline_aov

    # CTR improvement 
    ctr_multiplier = 1.0 + (ndcg_at_10 - 0.3) * 0.5 
    projected_ctr  = max(0.10, 0.15 * ctr_multiplier)
    baseline_ctr   = 0.15

    # Revenue impact Additional commission per day from AOV lift
    daily_commission_lift = daily_orders * aov_lift_abs * commission_rate
    annual_commission_lift = daily_commission_lift * 365

    # Cart-to-Order (C2O) impact CSAO increases engagement with fewer cart abandonments
    c2o_baseline  = 0.72
    c2o_projected = min(0.95, c2o_baseline + acceptance_lift * 0.3)

    # CSAO Attach Rate with % of orders that have at least one CSAO-recommended item
    attach_rate_baseline  = 0.15
    attach_rate_projected = min(0.60, attach_rate_baseline + projected_acceptance * 0.8)

    # Average items per order
    avg_items_baseline  = 2.1
    avg_items_projected = avg_items_baseline + projected_acceptance * 0.8

    return {
        "offline_metrics": {
            "precision_at_10": round(precision_at_10, 4),
            "ndcg_at_10":      round(ndcg_at_10, 4),
        },
        "acceptance_rate": {
            "baseline":    round(baseline_acceptance, 4),
            "projected":   round(projected_acceptance, 4),
            "lift_pp":     round(acceptance_lift * 100, 2),
        },
        "aov": {
            "baseline_inr":    round(baseline_aov, 2),
            "projected_inr":   round(baseline_aov + aov_lift_abs, 2),
            "lift_abs_inr":    round(aov_lift_abs, 2),
            "lift_pct":        round(aov_lift_pct * 100, 2),
        },
        "ctr": {
            "baseline":  round(baseline_ctr, 4),
            "projected": round(projected_ctr, 4),
        },
        "c2o_ratio": {
            "baseline":  round(c2o_baseline, 4),
            "projected": round(c2o_projected, 4),
        },
        "attach_rate": {
            "baseline":  round(attach_rate_baseline, 4),
            "projected": round(attach_rate_projected, 4),
        },
        "avg_items_per_order": {
            "baseline":  round(avg_items_baseline, 2),
            "projected": round(avg_items_projected, 2),
        },
        "revenue": {
            "daily_commission_lift_inr":  round(daily_commission_lift, 0),
            "annual_commission_lift_inr": round(annual_commission_lift, 0),
            "annual_lift_crore":          round(annual_commission_lift / 1e7, 2),
        },
    }


# SEGMENT-LEVEL IMPACT ANALYSIS
def segment_business_impact() -> pd.DataFrame:
    """
    Project business impact broken down by key user/context segments.
    Matches problem statement requirement for segment-level performance.
    """
    baseline_file = os.path.join(RAW_DIR, "baseline_performance.csv")
    if os.path.exists(baseline_file):
        baseline = pd.read_csv(baseline_file)
    else:
        baseline = pd.DataFrame([
            {"user_segment":"new",      "meal_time":"lunch",  "baseline_aov":280, "baseline_acceptance_rate":0.11},
            {"user_segment":"occasional","meal_time":"lunch", "baseline_aov":330, "baseline_acceptance_rate":0.15},
            {"user_segment":"regular",  "meal_time":"dinner", "baseline_aov":380, "baseline_acceptance_rate":0.18},
            {"user_segment":"frequent", "meal_time":"dinner", "baseline_aov":430, "baseline_acceptance_rate":0.25},
            {"user_segment":"power",    "meal_time":"dinner", "baseline_aov":520, "baseline_acceptance_rate":0.30},
        ])

    # Projected model improvements by segment
    improvement_by_seg = {
        "new":       0.08,   
        "occasional":0.10,
        "regular":   0.14,
        "frequent":  0.18,
        "power":     0.22,  
    }

    rows = []
    for _, row in baseline.iterrows():
        seg      = row.get("user_segment","regular")
        impr     = improvement_by_seg.get(seg, 0.12)
        base_acc = row.get("baseline_acceptance_rate", 0.15)
        base_aov = row.get("baseline_aov", 380)

        proj_acc = min(0.45, base_acc + impr)
        proj_aov = base_aov + proj_acc * 110

        rows.append({
            "user_segment":          seg,
            "meal_time":             row.get("meal_time","lunch"),
            "baseline_aov":          round(base_aov, 2),
            "projected_aov":         round(proj_aov, 2),
            "aov_lift_pct":          round((proj_aov - base_aov)/base_aov * 100, 2),
            "baseline_acceptance":   round(base_acc, 4),
            "projected_acceptance":  round(proj_acc, 4),
            "acceptance_lift_pp":    round((proj_acc - base_acc)*100, 2),
            "priority_segments":     seg in ["frequent","power"],
        })

    return pd.DataFrame(rows)


# A/B TESTING FRAMEWORK
def design_ab_test(
    baseline_acceptance:  float = 0.13,
    expected_acceptance:  float = 0.22,
    alpha:                float = 0.05,
    power:                float = 0.80,
    daily_eligible_users: int   = 500_000,
) -> dict:
    """
    Rigorous A/B test design per problem statement requirements.
    Includes:
    - Sample size calculation using two-proportion z-test
    - Guardrail metrics
    - Statistical testing
    """
    # Sample size with two-proportion z-test
    p1   = baseline_acceptance     
    p2   = expected_acceptance    
    mde  = p2 - p1                 

    z_alpha = stats.norm.ppf(1 - alpha/2) 
    z_beta  = stats.norm.ppf(power)

    p_bar    = (p1 + p2) / 2
    num      = (z_alpha * np.sqrt(2 * p_bar * (1-p_bar)) + z_beta * np.sqrt(p1*(1-p1) + p2*(1-p2)))**2
    n_per_group = int(np.ceil(num / (mde**2)))

    # Duration
    daily_per_group = daily_eligible_users // 2
    duration_days   = int(np.ceil(n_per_group / daily_per_group))

    primary_metrics = [
        {
            "metric":          "CSAO Acceptance Rate",
            "hypothesis":      "H1: Treatment > Control",
            "test":            "Two-proportion z-test",
            "target_lift":     f"+{(p2-p1)*100:.1f} percentage points",
            "guardrail":       "Must not decrease below control - 2pp",
        },
        {
            "metric":          "Average Order Value (AOV)",
            "hypothesis":      "H1: Treatment_AOV > Control_AOV",
            "test":            "Welch t-test",
            "target_lift":     "+10-15%",
            "guardrail":       "Must remain positive",
        },
        {
            "metric":          "Cart-to-Order (C2O) Ratio",
            "hypothesis":      "H1: Treatment does not harm C2O",
            "test":            "Chi-square test on conversion",
            "target_lift":     "No degradation (guardrail only)",
            "guardrail":       "C2O must stay >= baseline - 2pp",
        },
    ]


    secondary_metrics = [
        "CSAO Rail Order Share",
        "CSAO Rail Attach Rate",
        "Average Items per Order",
        "Click-Through Rate on CSAO Rail",
        "Time-to-Order",
    ]

    # Guardrail metrics 
    guardrail_metrics = [
        {
            "metric":    "Cart Abandonment Rate",
            "threshold": "Must stay <= baseline + 2pp",
            "rationale": "CSAO should not disrupt ordering flow",
        },
        {
            "metric":    "User Rating / Satisfaction Score",
            "threshold": "Must stay >= baseline - 0.1 points",
            "rationale": "Recommendations must not annoy users",
        },
        {
            "metric":    "App Session Duration (negative)",
            "threshold": "Session length should not increase > 20%",
            "rationale": "Excessive decision fatigue indicates bad UX",
        },
        {
            "metric":    "Recommendation Fatigue (CTR degradation over time)",
            "threshold": "CTR drop < 20% over 7 days",
            "rationale": "Repeated irrelevant recs cause fatigue",
        },
    ]

    return {
        "experiment_design": {
            "name":            "CSAO AI vs Baseline A/B Test",
            "control":         "Rule-based or no CSAO recommendations (50% traffic)",
            "treatment":       "AI-powered CSAO recommendations (50% traffic)",
            "traffic_split":   "50/50",
            "n_per_group":     n_per_group,
            "duration_days":   max(7, duration_days),  
            "alpha":           alpha,
            "power":           power,
            "mde_pp":          round(mde * 100, 2),
        },
        "primary_metrics":   primary_metrics,
        "secondary_metrics": secondary_metrics,
        "guardrail_metrics": guardrail_metrics,
        "statistical_tests": {
            "acceptance_rate": "Two-proportion z-test (one-tailed)",
            "aov":             "Welch t-test (one-tailed)",
            "revenue":         "Bootstrap confidence intervals",
            "c2o_ratio":       "Chi-square test",
            "correction":      "Bonferroni correction for multiple testing",
        },
        "rollout_strategy": {
            "phase_1": "1% traffic (canary) — 2 days — check for errors",
            "phase_2": "10% traffic — 3 days — sanity check metrics",
            "phase_3": "50/50 A/B — full experiment duration",
            "phase_4": "If successful: 100% rollout + monitor",
        },
        "power_analysis": {
            "baseline_acceptance": round(p1, 4),
            "expected_acceptance": round(p2, 4),
            "min_detectable_effect_pp": round(mde*100, 2),
            "sample_size_per_group":    n_per_group,
            "estimated_duration_days":  max(7, duration_days),
        },
    }



# SYSTEM SCALABILITY ANALYSIS
def scalability_analysis() -> dict:
    """
    Demonstrate how the system handles millions of daily requests.
    Based on peak lunch/dinner demand patterns.
    """
    # Traffic patterns 
    daily_orders      = 3_000_000   # Zomato scale (Assumption)
    peak_multiplier   = 3.0         # Peak is 3x average
    avg_rps           = daily_orders / (24 * 3600)
    peak_rps          = avg_rps * peak_multiplier

    # Each order may trigger multiple CSAO calls
    avg_cart_additions = 1.8
    csao_calls_per_order = avg_cart_additions
    avg_csao_rps  = avg_rps * csao_calls_per_order
    peak_csao_rps = peak_rps * csao_calls_per_order

    return {
        "traffic_estimates": {
            "daily_orders":       daily_orders,
            "avg_rps":            round(avg_rps, 1),
            "peak_rps":           round(peak_rps, 1),
            "csao_avg_rps":       round(avg_csao_rps, 1),
            "csao_peak_rps":      round(peak_csao_rps, 1),
        },
        "latency_budget_ms": {
            "feature_retrieval":   40,
            "candidate_generation": 30,
            "ranking_model":        80,
            "mmr_reranking":        10,
            "overhead":             18,
            "total":               178,
            "p95_target":          250,
            "p99_target":          300,
        },
        "infrastructure": {
            "model_serving":    "GBM (ONNX-exported for 3-5x speed), or TF Serving",
            "feature_store":    "Redis Cluster (sub-5ms lookups, 99.9% availability)",
            "candidate_index":  "FAISS (approximate nearest neighbor, <10ms)",
            "api_layer":        "FastAPI on Kubernetes (auto-scaling pods)",
            "caching":          "Redis cache hit rate ~95% for user/item features",
            "cdn":              "Feature pre-warming for predicted peak users",
        },
        "auto_scaling": {
            "min_replicas":    5,
            "max_replicas":    50,
            "scale_trigger":   "CPU > 70% or latency p95 > 200ms",
            "scale_up_time":   "< 60 seconds",
        },
        "benchmarking_strategy": {
            "step_1":  "Unit latency test: single request profiling",
            "step_2":  "Load test with Locust (ramp to 2000 rps)",
            "step_3":  "Soak test: 24h at 500 rps (detect memory leaks)",
            "step_4":  "Chaos test: kill 1 pod, verify auto-recovery",
            "step_5":  "p95/p99 latency must be <250ms/<300ms under 1666 rps peak",
        },
    }


# RECOMMENDATION SCENARIOS
def recommendation_examples() -> list:
    """
    Examples showing why recommendations make business sense.
    Directly maps to problem statement's Biryani → Raita → Gulab Jamun chain.
    """
    return [
        {
            "scenario":      "Biryani Sequential Completion",
            "cart_state":    ["Chicken Biryani (Rs.350)"],
            "recommendations": [
                {"item":"Raita",       "reason":"Classic complement - 80% co-occurrence", "score":0.82},
                {"item":"Salan",       "reason":"Traditional biryani side - 65%",         "score":0.74},
                {"item":"Gulab Jamun", "reason":"Popular dessert pairing - 50%",          "score":0.61},
                {"item":"Lassi",       "reason":"Beverage to complete meal",               "score":0.58},
            ],
            "aov_before": 350, "aov_after": 580, "aov_lift_pct": 65.7,
        },
        {
            "scenario":      "Pizza Meal Completion",
            "cart_state":    ["Margherita Pizza (Rs.320)"],
            "recommendations": [
                {"item":"Garlic Bread",  "reason":"Top complement for pizza - 65%", "score":0.79},
                {"item":"Soft Drink",    "reason":"Beverage missing from cart",      "score":0.71},
                {"item":"Chicken Wings", "reason":"Popular starter pairing",         "score":0.55},
            ],
            "aov_before": 320, "aov_after": 555, "aov_lift_pct": 73.4,
        },
        {
            "scenario":      "Budget User – Low-price Add-ons",
            "cart_state":    ["Veg Thali (Rs.150)"],
            "recommendations": [
                {"item":"Buttermilk",  "reason":"Budget-friendly beverage (Rs.30)", "score":0.68},
                {"item":"Papad",       "reason":"Low-price side (Rs.20)",           "score":0.62},
                {"item":"Gulab Jamun (single)", "reason":"Rs.40 dessert",           "score":0.55},
            ],
            "aov_before": 150, "aov_after": 240, "aov_lift_pct": 60.0,
        },
        {
            "scenario":      "Late Night Cravings",
            "cart_state":    ["Chicken Burger (Rs.220)"],
            "meal_time":     "late_night",
            "recommendations": [
                {"item":"Fries",        "reason":"Burger-fries pairing (70%)",   "score":0.77},
                {"item":"Cold Coffee",  "reason":"Late night beverage boost",     "score":0.65},
                {"item":"Brownie",      "reason":"Late-night sweet craving",      "score":0.58},
            ],
            "aov_before": 220, "aov_after": 405, "aov_lift_pct": 84.1,
        },
        {
            "scenario":      "Cold Start User – Rule-based fallback",
            "cart_state":    ["South Indian Thali (Rs.200)"],
            "user_type":     "new (0 orders)",
            "recommendations": [
                {"item":"Filter Coffee",  "reason":"Top restaurant beverage (bestseller)", "score":0.71},
                {"item":"Vada",           "reason":"Popular starter (80% popularity)",     "score":0.65},
                {"item":"Payasam",        "reason":"Traditional dessert complement",        "score":0.52},
            ],
            "aov_before": 200, "aov_after": 330, "aov_lift_pct": 65.0,
            "strategy": "cold_start_rules + restaurant_popularity",
        },
    ]


# FULL REPORT GENERATION
def generate_full_business_report():
    print("=" * 60)
    print("BUSINESS IMPACT ANALYSIS")
    print("=" * 60)

    # Load evaluation report for actual model metrics
    eval_report_path = os.path.join(REPORT_DIR, "evaluation_report.json")
    if os.path.exists(eval_report_path):
        with open(eval_report_path) as f:
            eval_report = json.load(f)
        gbm_metrics = eval_report.get("models",{}).get("gbm_ranking_model",{})
        precision   = gbm_metrics.get("precision@10", 0.30)
        ndcg        = gbm_metrics.get("ndcg@10",      0.75)
    else:
        print("[WARN] Eval report not found. Using target metrics.")
        precision = 0.30
        ndcg      = 0.75

    # 1. Metric translation
    print("\n1. Offline-to-Business Metric Translation …")
    business = translate_offline_to_business(precision, ndcg)

    print(f"  Precision@10:           {precision:.4f}")
    print(f"  Projected Acceptance:   {business['acceptance_rate']['projected']:.1%}")
    print(f"  AOV Lift:               +{business['aov']['lift_pct']:.1f}%  "
          f"(Rs.{business['aov']['baseline_inr']:.0f} -> Rs.{business['aov']['projected_inr']:.0f})")
    print(f"  Annual Commission Lift: Rs.{business['revenue']['annual_lift_crore']:.1f} Cr")

    # 2. Segment analysis
    print("\n2. Segment-level Business Impact …")
    seg_impact = segment_business_impact()
    print(seg_impact[["user_segment","baseline_aov","projected_aov",
                        "aov_lift_pct","baseline_acceptance","projected_acceptance"]].to_string(index=False))

    # 3. A/B test design
    print("\n3. A/B Test Design …")
    ab_design = design_ab_test(
        baseline_acceptance  = 0.13,
        expected_acceptance  = business['acceptance_rate']['projected'],
    )
    ed = ab_design["experiment_design"]
    print(f"  n per group: {ed['n_per_group']:,}  "
          f"Duration: {ed['duration_days']} days  "
          f"MDE: {ed['mde_pp']} pp")

    # 4. Scalability
    print("\n4. Scalability Analysis …")
    scale = scalability_analysis()
    t = scale["traffic_estimates"]
    l = scale["latency_budget_ms"]
    print(f"  Peak CSAO RPS: {t['csao_peak_rps']:.0f}")
    print(f"  Total latency: {l['total']}ms  (budget: {l['p95_target']}ms p95)")

    # 5. Examples
    examples = recommendation_examples()
    print(f"\n5. Recommendation Examples: {len(examples)} scenarios generated")

    # 6. Save report
    report = {
        "business_impact":      business,
        "segment_impact":       seg_impact.to_dict("records"),
        "ab_test_design":       ab_design,
        "scalability":          scale,
        "recommendation_examples": examples,
        "deployment_strategy": {
            "phase_1_canary":   "1% traffic for 48h — monitor errors, latency",
            "phase_2_ab_test":  "50/50 split for 7-14 days — measure primary metrics",
            "phase_3_full_roll":"100% rollout if A/B success + no guardrail violations",
            "phase_4_monitor":  "Ongoing monitoring: drift detection, retraining schedule",
            "retraining_freq":  "Weekly model refresh with new order data",
            "data_freshness":   "User features: daily | Item features: daily | Model: weekly",
        },
        "key_insights": [
            "50% of all orders are single-item — largest CSAO opportunity",
            "60% of orders lack a beverage — easy, high-acceptance recommendation",
            "Power users accept 2x more recommendations than new users",
            "Peak hours (12-2pm, 7-9pm) drive 35% of daily orders — prioritise latency",
            "Festival seasons show 40% AOV spike — pre-load seasonal recommendations",
            "Cold-start coverage via rules achieves 20-25% acceptance (vs 5-10% random)",
            "LLM embeddings enable zero-shot recommendation for new menu items",
        ],
    }

    out_path = os.path.join(REPORT_DIR, "business_impact_report.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    # Generate business impact chart
    _plot_business_impact(seg_impact, business, ab_design)

    print(f"\nBusiness report saved to: {out_path}")
    return report
