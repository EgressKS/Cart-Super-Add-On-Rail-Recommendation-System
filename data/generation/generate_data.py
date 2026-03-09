"""
CSAO Recommendation System - Synthetic Data Generator
Covers ALL scenarios from the problem statement:
- 100K users (5 segments, 3 budget tiers, incomplete meal patterns)
- 5K restaurants (5 types, 15+ cuisines, multi-city)
- 50K menu items (7 categories, veg/non-veg, complementarity rules)
- 1M orders (temporal patterns, seasonality, festivals)
- Cart snapshots (sequential context, meal completeness)
"""

import numpy as np
import pandas as pd
import random
import json
import os
from datetime import datetime, timedelta
from tqdm import tqdm


np.random.seed(42)
random.seed(42)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
os.makedirs(DATA_DIR, exist_ok=True)


SCALE = 0.1  # Generates 10K users, 500 restaurants, 5K items, 100K orders

N_USERS       = int(100_000 * SCALE)
N_RESTAURANTS = int(5_000  * SCALE)
N_ITEMS       = int(50_000 * SCALE)
N_ORDERS      = int(1_000_000 * SCALE)
N_SESSIONS    = int(1_500_000 * SCALE)

START_DATE = datetime(2024, 1, 1)
END_DATE   = datetime(2025, 12, 31)
DATE_RANGE_DAYS = (END_DATE - START_DATE).days

CITIES = {
    "metro":  ["Mumbai","Delhi","Bangalore","Hyderabad"],
    "tier1":  ["Pune","Kolkata","Chennai","Ahmedabad"],
    "tier2":  ["Jaipur","Lucknow","Chandigarh","Indore","Bhopal","Surat"],
}
ALL_CITIES = [c for cs in CITIES.values() for c in cs]
CITY_TIER  = {c: t for t,cs in CITIES.items() for c in cs}

ZONE_TYPES = ["business","residential","student","ithub"]

CUISINES = [
    "North Indian","South Indian","Chinese","Italian","Fast Food",
    "Biryani/Mughlai","Continental","Desserts","Beverages",
    "Thai","Mexican","Mediterranean","Japanese","Street Food","Bakery"
]

CUISINE_WEIGHTS = [0.25,0.15,0.12,0.08,0.10,0.10,0.05,0.03,0.02,
                   0.02,0.02,0.01,0.01,0.03,0.01]

RESTAURANT_TYPES = ["chain","independent_premium","local_favorite","cloud_kitchen","street_food"]
RESTAURANT_TYPE_WEIGHTS = [0.20, 0.15, 0.40, 0.15, 0.10]

ITEM_CATEGORIES = ["main_course","starter","side","bread","beverage","dessert","combo"]
CATEGORY_WEIGHTS = [0.40, 0.15, 0.10, 0.05, 0.15, 0.10, 0.05]

MEAL_TIMES = ["breakfast","lunch","snack","dinner","late_night"]

# Natural complementarity rules: category → likely next additions
COMPLEMENT_RULES = {
    "main_course": {"side": 0.65, "bread": 0.55, "beverage": 0.60, "dessert": 0.50, "starter": 0.30},
    "starter":     {"main_course": 0.80, "beverage": 0.55},
    "side":        {"main_course": 0.70, "bread": 0.45, "beverage": 0.40},
    "bread":       {"main_course": 0.75, "side": 0.50},
    "beverage":    {"dessert": 0.35},
    "dessert":     {},
    "combo":       {"beverage": 0.65, "dessert": 0.40},
}

# Cuisine-specific complement pairs
CUISINE_PAIRS = {
    "Biryani/Mughlai": [("biryani","raita",0.80),("biryani","salan",0.65),("biryani","gulab jamun",0.50)],
    "Italian":         [("pizza","garlic bread",0.65),("pizza","soft drink",0.60),("pasta","garlic bread",0.55)],
    "Fast Food":       [("burger","fries",0.70),("burger","coke",0.65),("burger","shake",0.45)],
    "North Indian":    [("curry","naan",0.70),("curry","raita",0.60),("paneer","naan",0.65)],
    "South Indian":    [("dosa","sambar",0.80),("idli","chutney",0.75),("dosa","coffee",0.50)],
    "Chinese":         [("noodles","spring roll",0.55),("fried rice","manchurian",0.60)],
}

WEATHER_CONDITIONS  = ["clear","cloudy","rainy","stormy"]
WEATHER_WEIGHTS     = [0.60,  0.20,   0.15,   0.05]

INDIAN_FESTIVALS = ["diwali","holi","eid","christmas","new_year","durga_puja","raksha_bandhan"]



def weighted_choice(options, weights):
    return random.choices(options, weights=weights, k=1)[0]

def random_date(start: datetime, days: int) -> datetime:
    return start + timedelta(days=random.randint(0, days),
                             hours=random.randint(0,23),
                             minutes=random.randint(0,59))

def meal_time_from_hour(h: int) -> str:
    if  6 <= h < 11: return "breakfast"
    if 11 <= h < 15: return "lunch"
    if 15 <= h < 18: return "snack"
    if 18 <= h < 23: return "dinner"
    return "late_night"

def is_peak_hour(h: int) -> bool:
    return (12 <= h <= 14) or (19 <= h <= 21)

def get_season(month: int) -> str:
    if month in [4,5,6]: return "summer"
    if month in [7,8,9]: return "monsoon"
    if month in [10,11]: return "festival"
    return "winter"

def festival_in_date(dt: datetime) -> str:
    m, d = dt.month, dt.day
    # approximate festival dates
    if m == 10 and 20 <= d <= 25: return "diwali"
    if m == 3  and 20 <= d <= 25: return "holi"
    if m == 4  and 10 <= d <= 15: return "eid"
    if m == 12 and 24 <= d <= 31: return "christmas"
    if m == 1  and 1  <= d <=  3: return "new_year"
    return "none"

def clamp(v, lo, hi):
    return max(lo, min(hi, v))


#USERS
def generate_users(n: int) -> pd.DataFrame:
    print(f"[1/8] Generating {n:,} users …")
    segments = ["new","occasional","regular","frequent","power"]
    seg_weights = [0.10, 0.30, 0.40, 0.15, 0.05]
    budget_tiers = ["budget","midrange","premium"]
    budget_weights = [0.30, 0.50, 0.20]

    # Meal-time pattern coverage
    meal_patterns = ["all","lunch_dinner","dinner_only","lunch_only","random"]
    meal_weights   = [0.30, 0.25, 0.20, 0.15, 0.10]

    rows = []
    for i in range(n):
        seg      = weighted_choice(segments, seg_weights)
        budget   = weighted_choice(budget_tiers, budget_weights)
        city     = weighted_choice(
            ALL_CITIES,
            [0.12,0.12,0.12,0.12, 0.08,0.08,0.08,0.08, 0.04,0.04,0.04,0.04,0.04,0.04]
        )
        zone_type = weighted_choice(ZONE_TYPES, [0.25,0.40,0.15,0.20])

        # Orders by segment
        order_ranges = {
            "new":0,"occasional":3,"regular":11,"frequent":51,"power":201
        }
        order_max = {
            "new":2,"occasional":10,"regular":50,"frequent":200,"power":600
        }
        total_orders = random.randint(order_ranges[seg], order_max[seg])

        aov_by_budget = {"budget":(150,300),"midrange":(300,600),"premium":(600,1500)}
        lo,hi = aov_by_budget[budget]
        avg_ov = round(np.random.uniform(lo, hi), 2)

        # Days since last order
        recency = {"new":0,"occasional":15,"regular":5,"frequent":2,"power":1}
        days_since = max(0, int(np.random.exponential(recency[seg])))

        # Veg preference
        veg_pref = random.choices([True,False], weights=[0.30,0.70])[0]

        # Meal pattern
        meal_pat = weighted_choice(meal_patterns, meal_weights)

        # Behavioral rates depend on segment
        bev_rate  = clamp(np.random.beta(2,3) + (0.2 if seg in ["frequent","power"] else 0), 0.05, 0.95)
        des_rate  = clamp(np.random.beta(1.5,5) + (0.1 if seg=="power" else 0), 0.01, 0.80)
        sta_rate  = clamp(np.random.beta(1.5,5), 0.01, 0.80)
        offer_rate= clamp(np.random.beta(3,2) if budget=="budget" else np.random.beta(1,3), 0.05, 0.95)
        price_sens= clamp(1.0 - avg_ov/1500.0 + np.random.normal(0,0.05), 0.0, 1.0)

        # Cuisine preferences
        n_cuisines = {"new":1,"occasional":2,"regular":3,"frequent":4,"power":5}
        preferred = random.sample(CUISINES, k=min(n_cuisines[seg], len(CUISINES)))

        reg_date = START_DATE - timedelta(days=random.randint(30,730))

        rows.append({
            "user_id":            f"U{i+1:07d}",
            "user_segment":       seg,
            "budget_tier":        budget,
            "city":               city,
            "city_tier":          CITY_TIER[city],
            "zone_type":          zone_type,
            "registration_date":  reg_date.strftime("%Y-%m-%d"),
            "total_orders":       total_orders,
            "avg_order_value":    avg_ov,
            "days_since_last_order": days_since,
            "is_premium_member":  (seg == "power") or (budget == "premium" and random.random() < 0.4),
            "veg_preference":     veg_pref,
            "meal_time_pattern":  meal_pat,
            "beverage_order_rate":round(bev_rate, 3),
            "dessert_order_rate": round(des_rate, 3),
            "starter_order_rate": round(sta_rate, 3),
            "offer_redemption_rate": round(offer_rate, 3),
            "price_sensitivity":  round(price_sens, 3),
            "preferred_cuisines": "|".join(preferred),
            "cuisine_variety_score": len(preferred),
            "is_cold_start":      total_orders <= 2,
        })

    return pd.DataFrame(rows)
