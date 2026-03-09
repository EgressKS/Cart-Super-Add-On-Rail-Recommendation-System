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
