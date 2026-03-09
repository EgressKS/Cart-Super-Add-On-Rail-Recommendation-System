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


# RESTAURANT FEATURES
def build_restaurant_features(restaurants: pd.DataFrame, orders: pd.DataFrame) -> pd.DataFrame:
    print("Building restaurant features …")
    r = restaurants.copy()

    rtype_map = {
        "chain":0,"independent_premium":1,
        "local_favorite":2,"cloud_kitchen":3,"street_food":4
    }
    r["rtype_enc"] = r["restaurant_type"].map(rtype_map).fillna(2)

    le_cuisine = LabelEncoder()
    r["cuisine_enc"] = le_cuisine.fit_transform(r["cuisine_type"].fillna("Other"))

    if len(orders) > 0:
        ord_agg = orders.groupby("restaurant_id").agg(
            total_orders=("order_id","count"),
            avg_final_value=("final_value","mean"),
            weekend_order_ratio=("is_weekend","mean"),
            peak_ratio=("is_peak_hour","mean"),
        ).reset_index()
        r = r.merge(ord_agg, on="restaurant_id", how="left")
        r["total_orders"]      = r["total_orders"].fillna(0)
        r["avg_final_value"]   = r["avg_final_value"].fillna(r["avg_order_value"])
        r["weekend_order_ratio"] = r["weekend_order_ratio"].fillna(0.28)
        r["peak_ratio"]          = r["peak_ratio"].fillna(0.35)

    r["log_total_orders"] = np.log1p(r.get("total_orders", pd.Series(0, index=r.index)))
    r["price_range_norm"] = r["price_range"] / 4.0
    r["rating_norm"]      = r["restaurant_rating"] / 5.0

    num_cols = r.select_dtypes(include=[np.number]).columns
    r[num_cols] = r[num_cols].fillna(0)
    print(f"  Restaurant feature matrix: {r.shape}")
    return r


# ITEM FEATURES
def build_item_features(items: pd.DataFrame, order_items: pd.DataFrame) -> pd.DataFrame:
    print("Building item features …")
    it = items.copy()

    cat_map = {
        "main_course":0,"starter":1,"side":2,
        "bread":3,"beverage":4,"dessert":5,"combo":6
    }
    avail_map = {"all_day":0,"morning_only":1,"evening_only":2,"lunch_dinner":3}
    it["category_enc"]  = it["category"].map(cat_map).fillna(0)
    it["avail_enc"]     = it["availability"].map(avail_map).fillna(0)
    it["log_price"]     = np.log1p(it["price"])
    it["rating_norm"]   = it["item_rating"] / 5.0
    it["pop_score"]     = it["popularity_score"]
    it["is_bestseller"] = it["is_bestseller"].astype(int)
    it["is_veg"]        = it["is_veg"].astype(int)
    it["is_new_item"]   = it["is_new_item"].astype(int)

    if len(order_items) > 0:
        oi_agg = order_items.groupby("item_id").agg(
            order_count=("order_id","count"),
            order_as_addon_count=("is_add_on","sum"),
        ).reset_index()
        oi_agg["addon_ratio"] = oi_agg["order_as_addon_count"] / oi_agg["order_count"].clip(1)
        it = it.merge(oi_agg, on="item_id", how="left")
        it["order_count"]    = it["order_count"].fillna(0)
        it["addon_ratio"]    = it["addon_ratio"].fillna(0)
        it["log_order_count"]= np.log1p(it["order_count"])

    num_cols = it.select_dtypes(include=[np.number]).columns
    it[num_cols] = it[num_cols].fillna(0)
    print(f"  Item feature matrix: {it.shape}")
    return it


# CART CONTEXT FEATURES  
def compute_cart_features(cart_snapshot_row: pd.Series) -> dict:
    """
    Given a single cart_snapshot row, compute all cart-level features.
    Called at inference time for real-time recommendations.
    """
    return {
        "cart_item_count":       cart_snapshot_row.get("cart_item_count", 0),
        "cart_total_value":      cart_snapshot_row.get("cart_total_value", 0.0),
        "has_main_course":       int(cart_snapshot_row.get("has_main_course", False)),
        "has_beverage":          int(cart_snapshot_row.get("has_beverage", False)),
        "has_dessert":           int(cart_snapshot_row.get("has_dessert", False)),
        "has_starter":           int(cart_snapshot_row.get("has_starter", False)),
        "has_bread":             int(cart_snapshot_row.get("has_bread", False)),
        "meal_completeness":     float(cart_snapshot_row.get("meal_completeness_score", 0.0)),
        "is_single_item_cart":   int(cart_snapshot_row.get("cart_item_count", 0) == 1),
        "cart_value_log":        np.log1p(cart_snapshot_row.get("cart_total_value", 0.0)),
        "missing_beverage":      int(not cart_snapshot_row.get("has_beverage", False)),
        "missing_dessert":       int(not cart_snapshot_row.get("has_dessert", False)),
        "missing_starter":       int(not cart_snapshot_row.get("has_starter", False)),
        "missing_bread":         int(not cart_snapshot_row.get("has_bread", False)),
    }


def build_cart_features_from_snapshots(cart_snaps: pd.DataFrame) -> pd.DataFrame:
    """Build cart features for all snapshots (for training)."""
    print("Building cart snapshot features …")
    cs = cart_snaps.copy()
    cs["is_single_item_cart"] = (cs["cart_item_count"] == 1).astype(int)
    cs["cart_value_log"]      = np.log1p(cs["cart_total_value"])
    cs["missing_beverage"]    = (~cs["has_beverage"]).astype(int)
    cs["missing_dessert"]     = (~cs["has_dessert"]).astype(int)
    cs["missing_starter"]     = (~cs["has_starter"]).astype(int)
    cs["missing_bread"]       = (~cs["has_bread"]).astype(int)
    cs["has_main_course"]     = cs["has_main_course"].astype(int)
    cs["has_beverage"]        = cs["has_beverage"].astype(int)
    cs["has_dessert"]         = cs["has_dessert"].astype(int)
    cs["has_starter"]         = cs["has_starter"].astype(int)
    cs["has_bread"]           = cs["has_bread"].astype(int)
    print(f"  Cart snapshot features: {cs.shape}")
    return cs


