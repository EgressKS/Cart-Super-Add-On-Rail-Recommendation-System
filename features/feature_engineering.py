"""
Feature Engineering Pipeline for Zomato CSAO Recommendation System
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from datetime import datetime

BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR      = os.path.join(BASE_DIR, "data", "raw")
PROC_DIR     = os.path.join(BASE_DIR, "data", "processed")
TRAIN_DIR    = os.path.join(PROC_DIR, "train")
VAL_DIR      = os.path.join(PROC_DIR, "validation")
TEST_DIR     = os.path.join(PROC_DIR, "test")

for d in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
    os.makedirs(d, exist_ok=True)


def load_raw_data():
    print("Loading raw data …")
    users       = pd.read_csv(os.path.join(RAW_DIR, "users.csv"))
    restaurants = pd.read_csv(os.path.join(RAW_DIR, "restaurants.csv"))
    items       = pd.read_csv(os.path.join(RAW_DIR, "items.csv"))
    orders      = pd.read_csv(os.path.join(RAW_DIR, "orders.csv"))
    order_items = pd.read_csv(os.path.join(RAW_DIR, "order_items.csv"))
    cart_snaps  = pd.read_csv(os.path.join(RAW_DIR, "cart_snapshots.csv"))
    csao        = pd.read_csv(os.path.join(RAW_DIR, "csao_interactions.csv"))
    comp        = pd.read_csv(os.path.join(RAW_DIR, "item_complementarity.csv"))
    baseline    = pd.read_csv(os.path.join(RAW_DIR, "baseline_performance.csv"))
    print(f"  Users={len(users):,}  Restaurants={len(restaurants):,}  Items={len(items):,}")
    print(f"  Orders={len(orders):,}  CSAO={len(csao):,}  Snapshots={len(cart_snaps):,}")
    return users, restaurants, items, orders, order_items, cart_snaps, csao, comp, baseline



# 1. USER FEATURES
def build_user_features(users: pd.DataFrame, orders: pd.DataFrame, order_items: pd.DataFrame) -> pd.DataFrame:
    """Combine static user profile with behavioral aggregates from order history."""
    print("Building user features …")

    seg_map  = {"new":0,"occasional":1,"regular":2,"frequent":3,"power":4}
    budg_map = {"budget":0,"midrange":1,"premium":2}
    zone_map = {"student":0,"business":1,"residential":2,"ithub":3}
    meal_map = {"all":0,"lunch_dinner":1,"dinner_only":2,"lunch_only":3,"random":4}
    tier_map = {"metro":2,"tier1":1,"tier2":0}

    u = users.copy()
    u["segment_enc"]      = u["user_segment"].map(seg_map).fillna(1)
    u["budget_enc"]       = u["budget_tier"].map(budg_map).fillna(1)
    u["zone_enc"]         = u["zone_type"].map(zone_map).fillna(0)
    u["meal_pattern_enc"] = u["meal_time_pattern"].map(meal_map).fillna(0)
    u["city_tier_enc"]    = u["city_tier"].map(tier_map).fillna(0)
    u["log_total_orders"] = np.log1p(u["total_orders"])

    if len(orders) > 0:
        order_agg = orders.groupby("user_id").agg(
            order_count=("order_id","count"),
            mean_order_value=("final_value","mean"),
            std_order_value=("final_value","std"),
            total_spend=("final_value","sum"),
            weekend_orders=("is_weekend","sum"),
            peak_hour_orders=("is_peak_hour","sum"),
            offer_used_count=("has_offer","sum"),
        ).reset_index()
        order_agg["weekend_ratio"] = order_agg["weekend_orders"] / order_agg["order_count"].clip(1)
        order_agg["peak_ratio"]    = order_agg["peak_hour_orders"]  / order_agg["order_count"].clip(1)
        order_agg["offer_ratio"]   = order_agg["offer_used_count"]  / order_agg["order_count"].clip(1)
        order_agg["std_order_value"] = order_agg["std_order_value"].fillna(0)
        u = u.merge(order_agg, on="user_id", how="left")
        u["order_count"]      = u["order_count"].fillna(0)
        u["mean_order_value"] = u["mean_order_value"].fillna(u["avg_order_value"])
        u["total_spend"]      = u["total_spend"].fillna(0)
        u["weekend_ratio"]    = u["weekend_ratio"].fillna(0.28)
        u["peak_ratio"]       = u["peak_ratio"].fillna(0.35)
        u["offer_ratio"]      = u["offer_ratio"].fillna(u["offer_redemption_rate"])

    if len(order_items) > 0:
        oi_agg = order_items.groupby(["user_id","item_category"]).size().unstack(fill_value=0)
        oi_agg.columns = [f"user_cat_{c}_count" for c in oi_agg.columns]
        oi_agg = oi_agg.reset_index()
        total_items = order_items.groupby("user_id").size().rename("user_total_items")
        oi_agg = oi_agg.merge(total_items, on="user_id", how="left")
        for col in oi_agg.columns:
            if col.startswith("user_cat_"):
                oi_agg[col.replace("_count","_rate")] = oi_agg[col] / oi_agg["user_total_items"].clip(1)
        u = u.merge(oi_agg, on="user_id", how="left")

    u["is_cold_start"]  = (u["total_orders"] <= 2).astype(int)
    u["is_sparse_user"] = (u["total_orders"].between(3, 10)).astype(int)

    num_cols = u.select_dtypes(include=[np.number]).columns
    u[num_cols] = u[num_cols].fillna(0)

    print(f"  User feature matrix: {u.shape}")
    return u
