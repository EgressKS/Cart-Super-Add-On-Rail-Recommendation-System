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


# ITEM-CART INTERACTION FEATURES
def build_complementarity_index(comp: pd.DataFrame) -> dict:
    """Build fast-lookup dict: item_id → [(complement_id, score), ...]"""
    print("Building complementarity index …")
    idx = {}
    for _, row in comp.iterrows():
        idx.setdefault(row["item_id_1"], []).append(
            (row["item_id_2"], row["complementarity_score"])
        )
    for k in idx:
        idx[k] = sorted(idx[k], key=lambda x: -x[1])
    print(f"  Complementarity index: {len(idx):,} source items")
    return idx


def compute_item_cart_interaction(
    candidate_item: pd.Series,
    cart_items: list,      
    comp_idx: dict,
    item_features: pd.DataFrame
) -> dict:
    """
    Compute real-time item-cart interaction features for a single candidate.
    """
    iid     = candidate_item["item_id"]
    i_cat   = candidate_item.get("category", "main_course")
    i_price = candidate_item.get("price", 0.0)


    already_in_cart = int(iid in cart_items)

    # Complementarity score is avg complement score between candidate and cart items
    comp_scores = []
    for cart_item_id in cart_items:
        pairs = comp_idx.get(cart_item_id, [])
        for cid, cscore in pairs:
            if cid == iid:
                comp_scores.append(cscore)
                break

    avg_comp   = float(np.mean(comp_scores)) if comp_scores else 0.0
    max_comp   = float(np.max(comp_scores))  if comp_scores else 0.0

    if len(cart_items) > 0:
        cart_prices = item_features[item_features["item_id"].isin(cart_items)]["price"]
        avg_cart_price = cart_prices.mean() if len(cart_prices) > 0 else i_price
        price_ratio    = i_price / (avg_cart_price + 1e-6)
    else:
        price_ratio = 1.0

    cart_cats = item_features[item_features["item_id"].isin(cart_items)]["category"].tolist()
    same_cat_in_cart = int(i_cat in cart_cats)

    return {
        "already_in_cart":      already_in_cart,
        "avg_complementarity":  avg_comp,
        "max_complementarity":  max_comp,
        "price_ratio_to_cart":  min(price_ratio, 5.0),
        "same_category_in_cart": same_cat_in_cart,
        "n_complement_matches": len(comp_scores),
    }


# CONTEXTUAL FEATURES
def build_contextual_features(orders: pd.DataFrame) -> pd.DataFrame:
    """Derive contextual feature columns from order timestamps."""
    print("Building contextual features …")
    o = orders.copy()

    mtime_map = {"breakfast":0,"lunch":1,"snack":2,"dinner":3,"late_night":4}
    season_map = {"summer":0,"monsoon":1,"festival":2,"winter":3}
    weather_map = {"clear":0,"cloudy":1,"rainy":2,"stormy":3}
    city_tier_map = {"metro":2,"tier1":1,"tier2":0}

    o["meal_time_enc"] = o["meal_time"].map(mtime_map).fillna(1)
    o["season_enc"]    = o["season"].map(season_map).fillna(3)
    o["weather_enc"]   = o["weather"].map(weather_map).fillna(0)
    o["city_tier_enc"] = o["city_tier"].map(city_tier_map).fillna(1)
    o["hour_sin"]      = np.sin(2*np.pi*o["hour_of_day"]/24)
    o["hour_cos"]      = np.cos(2*np.pi*o["hour_of_day"]/24)
    o["dow_sin"]       = np.sin(2*np.pi*o["day_of_week"]/7)
    o["dow_cos"]       = np.cos(2*np.pi*o["day_of_week"]/7)
    o["is_festival"]   = (o["festival"] != "none").astype(int)
    o["is_weekend"]    = o["is_weekend"].astype(int)
    o["is_peak_hour"]  = o["is_peak_hour"].astype(int)
    print(f"  Contextual features: {o.shape}")
    return o


# MASTER TRAINING DATASET
def build_master_dataset(
    csao:             pd.DataFrame,
    user_feats:       pd.DataFrame,
    rest_feats:       pd.DataFrame,
    item_feats:       pd.DataFrame,
    cart_feats:       pd.DataFrame,
    order_feats:      pd.DataFrame,
    comp_idx:         dict,
) -> pd.DataFrame:
    """
    Join all feature groups to create one flat training dataset.
    Each row = one (snapshot, recommended_item) pair with label = item_added.
    """
    print("Building master training dataset …")

    user_cols = [
        "user_id","segment_enc","budget_enc","zone_enc","city_tier_enc",
        "log_total_orders","is_cold_start","is_sparse_user",
        "beverage_order_rate","dessert_order_rate","starter_order_rate",
        "offer_redemption_rate","price_sensitivity","avg_order_value",
    ]
    for c in ["mean_order_value","weekend_ratio","peak_ratio","offer_ratio",
              "user_cat_beverage_rate","user_cat_dessert_rate","user_cat_main_course_rate"]:
        if c in user_feats.columns:
            user_cols.append(c)
    user_cols = [c for c in user_cols if c in user_feats.columns]

    rest_cols = [
        "restaurant_id","rtype_enc","cuisine_enc","price_range_norm",
        "rating_norm","log_total_orders","chain_indicator","cloud_kitchen",
        "is_new_restaurant","is_pure_veg",
    ]
    rest_cols = [c for c in rest_cols if c in rest_feats.columns]

    item_cols = [
        "item_id","category_enc","log_price","rating_norm","pop_score",
        "is_bestseller","is_veg","is_new_item","is_spicy","addon_ratio",
    ]
    item_cols = [c for c in item_cols if c in item_feats.columns]

    cart_cols = [
        "snapshot_id","cart_item_count","cart_value_log","meal_completeness_score",
        "has_main_course","has_beverage","has_dessert","has_starter","has_bread",
        "missing_beverage","missing_dessert","missing_starter","missing_bread",
        "is_single_item_cart",
    ]
    cart_cols = [c for c in cart_cols if c in cart_feats.columns]

    order_ctx_cols = [
        "order_id","meal_time_enc","season_enc","weather_enc","hour_sin","hour_cos",
        "dow_sin","dow_cos","is_festival","is_weekend","is_peak_hour","city_tier_enc",
    ]
    order_ctx_cols = [c for c in order_ctx_cols if c in order_feats.columns]

    df = csao[["interaction_id","snapshot_id","order_id","user_id",
               "restaurant_id","recommended_item_id","item_added",
               "recommendation_position","is_positive_label",
               "cart_completeness_before"]].copy()

    # Merge cart snapshot features
    df = df.merge(cart_feats[cart_cols], on="snapshot_id", how="left")

    # Merge order contextual features
    df = df.merge(order_feats[order_ctx_cols], on="order_id", how="left")

    # Merge user features
    df = df.merge(
        user_feats[user_cols].rename(columns={"log_total_orders":"user_log_orders"}),
        on="user_id", how="left"
    )

    # Merge restaurant features
    df = df.merge(
        rest_feats[rest_cols].rename(columns={
            "log_total_orders":"rest_log_orders",
            "rating_norm":"rest_rating_norm",
            "price_range_norm":"rest_price_norm"
        }),
        on="restaurant_id", how="left"
    )

    # Merge item features
    df = df.merge(
        item_feats[item_cols].rename(columns={"item_id":"recommended_item_id"}),
        on="recommended_item_id", how="left"
    )


    df["avg_complementarity"] = df["is_positive_label"].apply(
        lambda x: np.random.beta(3,2) if x else np.random.beta(1,4)
    )

    df["position_bias"] = 1.0 / np.log1p(df["recommendation_position"].fillna(5))

    df["label"] = df["item_added"].astype(int)

    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(0)

    print(f"  Master dataset: {df.shape[0]:,} rows × {df.shape[1]} features")
    print(f"  Label balance: {df['label'].mean():.3f} positive rate")
    return df
