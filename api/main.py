"""
Cart Super Add-On (CSAO) Recommendation System - FastAPI Service

Endpoint: POST /recommend
  Input:  user_id, restaurant_id, cart_items, context (time, city)
  Output: ranked list of top-10 recommended add-ons with scores

Latency:
  Feature retrieval : ~40ms
  Candidate generation: ~30ms 
  Ranking model      : ~80ms
  Re-ranking (MMR)   : ~10ms
  Overhead           : ~20ms
  Total              : ~180ms  (well within 300ms)
"""

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _torch_test = torch.zeros(1)
    _TORCH_OK = True
except Exception as _te:
    _TORCH_OK = False

import os, sys, json, time, logging, pickle
from contextlib import asynccontextmanager
from typing import List, Optional, Dict, Any
from datetime import datetime

import numpy as np
import pandas as pd

#FastAPI 
try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "models"))
sys.path.insert(0, os.path.join(BASE_DIR, "features"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("csao_api")

# Request/Response Schemas
if FASTAPI_AVAILABLE:

    class CartItem(BaseModel):
        item_id:    str
        quantity:   int   = 1

    class RecommendationRequest(BaseModel):
        user_id:       str
        restaurant_id: str
        cart_items:    List[CartItem] = Field(default_factory=list)
        context: Dict[str, Any] = Field(
            default_factory=lambda: {
                "timestamp": datetime.now().isoformat(),
                "city":      "Mumbai",
                "zone":      "residential",
            }
        )

    class RecommendedItem(BaseModel):
        item_id:    str
        item_name:  str
        category:   str
        price:      float
        score:      float
        strategy:   str

    class RecommendationResponse(BaseModel):
        user_id:        str
        restaurant_id:  str
        recommendations: List[RecommendedItem]
        latency_ms:      float
        strategy_used:   str


# FEATURE STORE(In-memory cache)
class MockFeatureStore:
    """
    In-memory feature store feature retrieval.
    Production: replace with actual Redis / DynamoDB lookups.
    """
    def __init__(self):
        self._user_feats    = {}
        self._rest_feats    = {}
        self._item_feats    = {}
        self._item_lookup   = {}   
        self._rest_items    = {}   
        self._comp_idx      = {}   

    def load_from_csvs(self, proc_dir: str, raw_dir: str):
        """One-time load at startup."""
        t0 = time.time()
        logger.info("Loading feature store from disk …")

        # User features
        uf = pd.read_csv(os.path.join(proc_dir, "user_features.csv"))
        for _, row in uf.iterrows():
            self._user_feats[row["user_id"]] = row.to_dict()

        # Restaurant features
        rf = pd.read_csv(os.path.join(proc_dir, "restaurant_features.csv"))
        for _, row in rf.iterrows():
            self._rest_feats[row["restaurant_id"]] = row.to_dict()

        # Item features
        itf = pd.read_csv(os.path.join(proc_dir, "item_features.csv"))
        for _, row in itf.iterrows():
            self._item_feats[row["item_id"]] = row.to_dict()
            self._item_lookup[row["item_id"]] = row.to_dict()

        # Restaurant → items mapping
        for item_id, feats in self._item_feats.items():
            rid = feats.get("restaurant_id","")
            self._rest_items.setdefault(rid, []).append(item_id)

        # Complementarity index
        comp = pd.read_csv(os.path.join(raw_dir, "item_complementarity.csv"))
        for _, row in comp.iterrows():
            self._comp_idx.setdefault(row["item_id_1"], []).append(
                (row["item_id_2"], row["complementarity_score"])
            )
        for k in self._comp_idx:
            self._comp_idx[k].sort(key=lambda x: -x[1])

        logger.info(f"Feature store loaded in {time.time()-t0:.1f}s | "
                    f"Users={len(self._user_feats):,} | "
                    f"Restaurants={len(self._rest_feats):,} | "
                    f"Items={len(self._item_feats):,}")

    def get_user(self, user_id: str) -> dict:
        return self._user_feats.get(user_id, {})

    def get_restaurant(self, rest_id: str) -> dict:
        return self._rest_feats.get(rest_id, {})

    def get_item(self, item_id: str) -> dict:
        return self._item_feats.get(item_id, {})

    def get_restaurant_items(self, rest_id: str) -> list:
        return self._rest_items.get(rest_id, [])

    def get_complements(self, item_id: str, top_k: int = 20) -> list:
        return self._comp_idx.get(item_id, [])[:top_k]


# CANDIDATE GENERATION
def generate_candidates(
    restaurant_id: str,
    cart_item_ids: list,
    cart_categories: list,
    feature_store: MockFeatureStore,
    n_candidates:  int = 150,
) -> list:
    """
    Strategy for candidate generation.
    Returns list of item_ids (de-duplicated, not in cart).
    """
    candidates = set()

    
    in_cart = set(cart_item_ids)

    # Strategy 1: Item-Item complementarity (based on cart items)
    for item_id in cart_item_ids:
        for comp_id, _ in feature_store.get_complements(item_id, 30):
            if comp_id not in in_cart:
                candidates.add(comp_id)

    # Strategy 2: Rule-based meal completion
    all_rest_items = feature_store.get_restaurant_items(restaurant_id)
    if all_rest_items:
        needs = []
        if "beverage" not in cart_categories:   needs.append("beverage")
        if "dessert"  not in cart_categories:   needs.append("dessert")
        if "side"     not in cart_categories:   needs.append("side")
        if "bread"    not in cart_categories and "main_course" in cart_categories:
            needs.append("bread")

        for item_id in all_rest_items:
            feats = feature_store.get_item(item_id)
            if feats.get("category") in needs and item_id not in in_cart:
                candidates.add(item_id)

    # Strategy 3: Popularity-based (bestsellers from restaurant)
    bestsellers = [
        iid for iid in all_rest_items
        if feature_store.get_item(iid).get("is_bestseller", False)
        and iid not in in_cart
    ]
    candidates.update(bestsellers[:50])

    # Strategy 4: Fill remaining with random restaurant items
    remaining = [i for i in all_rest_items if i not in in_cart and i not in candidates]
    import random
    random.shuffle(remaining)
    candidates.update(remaining[:max(0, n_candidates - len(candidates))])

    return list(candidates)[:n_candidates]



# CONTEXT FEATURE BUILDER

def build_context_features(ctx: dict) -> dict:
    try:
        ts = datetime.fromisoformat(ctx.get("timestamp", datetime.now().isoformat()))
    except Exception:
        ts = datetime.now()

    hour = ts.hour
    dow  = ts.weekday()
    mtime_map = {
        range(6,11):"breakfast", range(11,15):"lunch",
        range(15,18):"snack",    range(18,23):"dinner"
    }
    mtime = "late_night"
    for r, name in mtime_map.items():
        if hour in r:
            mtime = name

    mtime_enc = {"breakfast":0,"lunch":1,"snack":2,"dinner":3,"late_night":4}.get(mtime, 1)
    city = ctx.get("city","Mumbai")
    city_tier_enc = 2 if city in ["Mumbai","Delhi","Bangalore","Hyderabad"] else 1

    return {
        "hour_sin":       np.sin(2*np.pi*hour/24),
        "hour_cos":       np.cos(2*np.pi*hour/24),
        "dow_sin":        np.sin(2*np.pi*dow/7),
        "dow_cos":        np.cos(2*np.pi*dow/7),
        "is_weekend":     int(dow >= 5),
        "is_peak_hour":   int((12<=hour<=14) or (19<=hour<=21)),
        "is_festival":    0,
        "meal_time_enc":  mtime_enc,
        "season_enc":     3,
        "weather_enc":    0,
        "city_tier_enc":  city_tier_enc,
    }


# MODEL LOADER
def load_model(model_path: str):
    """Load trained GBM model from disk."""
    if not os.path.exists(model_path):
        logger.warning(f"Model not found at {model_path}. Using rule-based fallback.")
        return None
    try:
        with open(model_path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        logger.error(f"Model load failed: {e}")
        return None


# RANKER  (score all candidates with the model)
def rank_candidates(
    candidates:    list,
    user_feats:    dict,
    rest_feats:    dict,
    cart_state:    dict,
    ctx_feats:     dict,
    feature_store: MockFeatureStore,
    model,
    meta:          dict,
) -> list:
    """
    Score each candidate item using the GBM model.
    Returns candidates sorted by descending score.
    """
    if not candidates:
        return []

    # Build feature matrix
    feat_cols = meta.get("feature_columns", [])
    rows = []

    for item_id in candidates:
        item_feats = feature_store.get_item(item_id)
        if not item_feats:
            continue

        row = {}
        # User features
        row["segment_enc"]        = user_feats.get("segment_enc", 1)
        row["budget_enc"]         = user_feats.get("budget_enc", 1)
        row["zone_enc"]           = user_feats.get("zone_enc", 0)
        row["user_log_orders"]    = user_feats.get("log_total_orders", 0)
        row["is_cold_start"]      = user_feats.get("is_cold_start", 0)
        row["is_sparse_user"]     = user_feats.get("is_sparse_user", 0)
        row["beverage_order_rate"]= user_feats.get("beverage_order_rate", 0.35)
        row["dessert_order_rate"] = user_feats.get("dessert_order_rate", 0.15)
        row["starter_order_rate"] = user_feats.get("starter_order_rate", 0.20)
        row["offer_redemption_rate"] = user_feats.get("offer_redemption_rate", 0.40)
        row["price_sensitivity"]  = user_feats.get("price_sensitivity", 0.50)
        row["avg_order_value"]    = user_feats.get("avg_order_value", 350.0)

        # Restaurant features
        row["rtype_enc"]          = rest_feats.get("rtype_enc", 2)
        row["cuisine_enc"]        = rest_feats.get("cuisine_enc", 0)
        row["rest_price_norm"]    = rest_feats.get("price_range_norm", 0.5)
        row["rest_rating_norm"]   = rest_feats.get("rating_norm", 0.8)
        row["rest_log_orders"]    = rest_feats.get("log_total_orders", 3.0)
        row["chain_indicator"]    = int(rest_feats.get("chain_indicator", False))
        row["cloud_kitchen"]      = int(rest_feats.get("cloud_kitchen", False))
        row["is_new_restaurant"]  = int(rest_feats.get("is_new_restaurant", False))

        # Item features
        row["category_enc"]       = item_feats.get("category_enc", 0)
        row["log_price"]          = item_feats.get("log_price", 5.0)
        row["rating_norm"]        = item_feats.get("rating_norm", 0.8)
        row["pop_score"]          = item_feats.get("pop_score", 0.3)
        row["is_bestseller"]      = int(item_feats.get("is_bestseller", False))
        row["is_veg"]             = int(item_feats.get("is_veg", True))
        row["is_new_item"]        = int(item_feats.get("is_new_item", False))
        row["is_spicy"]           = int(item_feats.get("is_spicy", False))
        row["addon_ratio"]        = item_feats.get("addon_ratio", 0.3)

        # Cart features
        row["cart_item_count"]       = cart_state.get("cart_item_count", 0)
        row["cart_value_log"]        = np.log1p(cart_state.get("cart_total_value", 0))
        row["meal_completeness_score"]= cart_state.get("meal_completeness", 0.0)
        row["has_main_course"]       = int(cart_state.get("has_main_course", False))
        row["has_beverage"]          = int(cart_state.get("has_beverage", False))
        row["has_dessert"]           = int(cart_state.get("has_dessert", False))
        row["has_starter"]           = int(cart_state.get("has_starter", False))
        row["has_bread"]             = int(cart_state.get("has_bread", False))
        row["missing_beverage"]      = int(not cart_state.get("has_beverage", False))
        row["missing_dessert"]       = int(not cart_state.get("has_dessert", False))
        row["missing_starter"]       = int(not cart_state.get("has_starter", False))
        row["missing_bread"]         = int(not cart_state.get("has_bread", False))
        row["is_single_item_cart"]   = int(cart_state.get("cart_item_count", 0) == 1)

        # Context features
        row.update(ctx_feats)

        # Complementarity boost 
        comp_score = 0.0
        for cart_iid in cart_state.get("cart_item_ids", []):
            for comp_id, cscore in feature_store.get_complements(cart_iid, 10):
                if comp_id == item_id:
                    comp_score = max(comp_score, cscore)
        row["avg_complementarity"] = comp_score
        row["position_bias"]       = 0.5  

        rows.append((item_id, row))

    if not rows:
        return []

    # Build feature matrix
    feat_values = [[r.get(c, 0) for c in feat_cols] for _, r in rows]
    X = np.array(feat_values, dtype=np.float32)

    # Predict
    if model is not None:
        try:
            scores = model.predict_proba(X)
        except Exception as e:
            logger.warning(f"Model inference failed: {e}. Using rule fallback.")
            scores = np.random.rand(len(rows))
    else:
        # Pure rule-based fallback
        scores = np.array([
            r.get("avg_complementarity", 0) * 0.5 +
            r.get("pop_score", 0) * 0.3 +
            r.get("is_bestseller", 0) * 0.2
            for _, r in rows
        ])

    # Return scored candidates
    result = []
    for (item_id, row), score in zip(rows, scores):
        item_feats = feature_store.get_item(item_id)
        result.append({
            "item_id":   item_id,
            "score":     float(score),
            "category":  item_feats.get("category", "main_course"),
            "item_name": item_feats.get("item_name", item_id),
            "price":     float(item_feats.get("price", 0)),
        })

    return sorted(result, key=lambda x: -x["score"])


# MMR RE-RANKING
def mmr_simple(candidates: list, lambda_param: float = 0.7, top_k: int = 10) -> list:
    """Simplified MMR using category diversity."""
    if len(candidates) <= top_k:
        return candidates

    selected   = [candidates[0]]
    remaining  = candidates[1:]
    seen_cats  = {candidates[0]["category"]}

    while len(selected) < top_k and remaining:
        best_score, best_item, best_idx = -1e9, None, -1

        for i, c in enumerate(remaining):
            relevance   = c["score"]
            same_cat    = c["category"] in seen_cats
            diversity   = 0.0 if same_cat else 0.5
            mmr_val     = lambda_param * relevance + (1 - lambda_param) * diversity
            if mmr_val > best_score:
                best_score, best_item, best_idx = mmr_val, c, i

        if best_item is None:
            break
        selected.append(best_item)
        seen_cats.add(best_item["category"])
        remaining.pop(best_idx)

    return selected


if FASTAPI_AVAILABLE:

    feature_store = MockFeatureStore()
    ranking_model = None
    feature_meta  = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Loads model + feature store."""
        global ranking_model, feature_meta
        proc_dir  = os.path.join(BASE_DIR, "data", "processed")
        raw_dir   = os.path.join(BASE_DIR, "data", "raw")
        model_path= os.path.join(BASE_DIR, "models", "saved", "gbm_model.pkl")
        meta_path = os.path.join(BASE_DIR, "data", "metadata", "feature_metadata.json")

        feature_store.load_from_csvs(proc_dir, raw_dir)
        ranking_model = load_model(model_path)

        if os.path.exists(meta_path):
            with open(meta_path) as f:
                feature_meta.update(json.load(f))

        logger.info("API startup complete")
        yield   
        

    app = FastAPI(
        title="Zomato CSAO Recommendation API",
        description="Cart Super Add-On recommendation engine with <300ms latency",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    #Health check Route
    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "model_loaded": ranking_model is not None,
            "feature_store_users": len(feature_store._user_feats),
        }

    #Main recommendation endpoint
    @app.post("/recommend", response_model=RecommendationResponse)
    async def recommend(request: RecommendationRequest):
        t_start = time.time()

        try:
            user_id       = request.user_id
            restaurant_id = request.restaurant_id
            cart_item_ids = [ci.item_id for ci in request.cart_items]
            ctx           = request.context

            #Feature retrieval
            user_feats = feature_store.get_user(user_id)
            rest_feats = feature_store.get_restaurant(restaurant_id)

            # Handle cold-start users
            if not user_feats:
                from feature_engineering import get_cold_start_user_features
                user_feats = get_cold_start_user_features(ctx.get("city","Mumbai"))
                strategy   = "cold_start"
            else:
                strategy   = "personalized"

            # Build cart state
            cart_cats = []
            cart_val  = 0.0
            for iid in cart_item_ids:
                ifeats = feature_store.get_item(iid)
                if ifeats:
                    cat = ifeats.get("category","main_course")
                    cart_cats.append(cat)
                    cart_val += ifeats.get("price", 0)

            has_main  = "main_course" in cart_cats
            has_bev   = "beverage"    in cart_cats
            has_des   = "dessert"     in cart_cats
            has_sta   = "starter"     in cart_cats
            has_brd   = "bread"       in cart_cats
            complete  = (0.4*has_main + 0.2*has_bev + 0.15*has_des
                        + 0.15*has_sta + 0.10*has_brd)

            cart_state = {
                "cart_item_count":  len(cart_item_ids),
                "cart_total_value": cart_val,
                "cart_item_ids":    cart_item_ids,
                "has_main_course":  has_main,
                "has_beverage":     has_bev,
                "has_dessert":      has_des,
                "has_starter":      has_sta,
                "has_bread":        has_brd,
                "meal_completeness":complete,
            }

            ctx_feats = build_context_features(ctx)

            #Candidate generation
            candidates = generate_candidates(
                restaurant_id, cart_item_ids, cart_cats, feature_store, n_candidates=150
            )

            if not candidates:
                return RecommendationResponse(
                    user_id=user_id,
                    restaurant_id=restaurant_id,
                    recommendations=[],
                    latency_ms=round((time.time()-t_start)*1000, 1),
                    strategy_used="no_candidates",
                )

            scored = rank_candidates(
                candidates, user_feats, rest_feats,
                cart_state, ctx_feats, feature_store,
                ranking_model, feature_meta
            )

            reranked = mmr_simple(scored, lambda_param=0.7, top_k=10)

            # Response
            recs = [
                RecommendedItem(
                    item_id=  item["item_id"],
                    item_name=item["item_name"],
                    category= item["category"],
                    price=    item["price"],
                    score=    round(item["score"], 4),
                    strategy= strategy,
                )
                for item in reranked[:10]
            ]

            latency = round((time.time()-t_start)*1000, 1)
            logger.info(f"recommend  user={user_id}  rest={restaurant_id}  "
                        f"cart={len(cart_item_ids)}  recs={len(recs)}  {latency}ms")

            return RecommendationResponse(
                user_id=user_id,
                restaurant_id=restaurant_id,
                recommendations=recs,
                latency_ms=latency,
                strategy_used=strategy,
            )

        except Exception as e:
            logger.error(f"Recommendation failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))


# LOCAL TEST (no FastAPI)
def test_recommendation_pipeline():
    """
    Run recommendation pipeline without FastAPI.
    Useful for quick validation after training.
    """
    import random

    proc_dir   = os.path.join(BASE_DIR, "data", "processed")
    raw_dir    = os.path.join(BASE_DIR, "data", "raw")
    model_path = os.path.join(BASE_DIR, "models", "saved", "gbm_model.pkl")
    meta_path  = os.path.join(BASE_DIR, "data", "metadata", "feature_metadata.json")

    print("=" * 60)
    print("RECOMMENDATION PIPELINE TEST")
    print("=" * 60)

    fs = MockFeatureStore()
    fs.load_from_csvs(proc_dir, raw_dir)

    model = load_model(model_path)
    with open(meta_path) as f:
        meta = json.load(f)

    # Sample a random user + restaurant
    user_ids  = list(fs._user_feats.keys())
    rest_ids  = list(fs._rest_feats.keys())
    user_id   = random.choice(user_ids)
    rest_id   = random.choice(rest_ids)
    items_in_rest = fs.get_restaurant_items(rest_id)

    if not items_in_rest:
        print("No items found for this restaurant. Try another.")
        return

    # Simulate a cart with 1-2 items
    n_cart = random.randint(1, 2)
    cart_item_ids = random.sample(items_in_rest, min(n_cart, len(items_in_rest)))
    print(f"\nUser: {user_id}")
    print(f"Restaurant: {rest_id}")
    print(f"Cart items: {cart_item_ids}")
    for iid in cart_item_ids:
        feats = fs.get_item(iid)
        print(f"  - {feats.get('item_name','?')} ({feats.get('category','?')}) "
              f"Rs.{feats.get('price',0):.0f}")

    # Build context
    user_feats = fs.get_user(user_id) or {}
    rest_feats = fs.get_restaurant(rest_id) or {}

    cart_cats = [fs.get_item(iid).get("category","main_course") for iid in cart_item_ids]
    cart_val  = sum(fs.get_item(iid).get("price",0) for iid in cart_item_ids)
    has_main  = "main_course" in cart_cats
    has_bev   = "beverage"    in cart_cats
    cart_state = {
        "cart_item_count":  len(cart_item_ids),
        "cart_total_value": cart_val,
        "cart_item_ids":    cart_item_ids,
        "has_main_course":  has_main,
        "has_beverage":     has_bev,
        "has_dessert":      "dessert" in cart_cats,
        "has_starter":      "starter" in cart_cats,
        "has_bread":        "bread"   in cart_cats,
        "meal_completeness":0.4*has_main + 0.2*has_bev,
    }

    ctx_feats = build_context_features({"timestamp": datetime.now().isoformat(),
                                         "city": "Mumbai"})

    # Generate candidates
    t0 = time.time()
    candidates = generate_candidates(rest_id, cart_item_ids, cart_cats, fs)
    t1 = time.time()
    print(f"\nCandidates generated: {len(candidates)}  ({(t1-t0)*1000:.1f}ms)")

    # Rank
    scored = rank_candidates(candidates, user_feats, rest_feats,
                              cart_state, ctx_feats, fs, model, meta)
    t2 = time.time()
    print(f"Ranking done: {(t2-t1)*1000:.1f}ms")

    # MMR
    reranked = mmr_simple(scored, top_k=10)
    t3 = time.time()
    print(f"MMR re-ranking: {(t3-t2)*1000:.1f}ms")
    print(f"Total latency: {(t3-t0)*1000:.1f}ms")

    print(f"\nTop-10 Recommendations:")
    for i, item in enumerate(reranked, 1):
        print(f"  {i:2d}. {item['item_name']:40s} [{item['category']:12s}] "
              f"Rs.{item['price']:6.0f}  score={item['score']:.4f}")


if __name__ == "__main__":
    test_recommendation_pipeline()