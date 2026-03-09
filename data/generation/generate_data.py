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


# RESTAURANTS
def generate_restaurants(n: int) -> pd.DataFrame:
    print(f"[2/8] Generating {n:,} restaurants …")
    price_ranges = [1, 2, 3, 4]   # 1=budget … 4=luxury
    price_weights = [0.30, 0.50, 0.15, 0.05]

    rows = []
    for i in range(n):
        rtype  = weighted_choice(RESTAURANT_TYPES, RESTAURANT_TYPE_WEIGHTS)
        cuisine= weighted_choice(CUISINES, CUISINE_WEIGHTS)
        city   = weighted_choice(ALL_CITIES, [0.12,0.12,0.12,0.12,0.08,0.08,0.08,0.08,0.04,0.04,0.04,0.04,0.04,0.04])
        zone   = weighted_choice(ZONE_TYPES, [0.25,0.40,0.15,0.20])
        price_rng = weighted_choice(price_ranges, price_weights)

        # Rating distributions
        rating = clamp(np.random.beta(7,3)*5, 2.5, 5.0)

        # Age: 5% very new → cold start
        age_days = int(np.random.exponential(400))
        is_very_new = age_days < 30
        is_new_rest = age_days < 90

        # Menu size by type
        menu_sizes = {
            "chain":100, "independent_premium":50,
            "local_favorite":60, "cloud_kitchen":35, "street_food":40
        }
        menu_size = int(np.random.normal(menu_sizes[rtype], menu_sizes[rtype]*0.2))
        menu_size = max(10, menu_size)

        avg_prep = max(10, int(np.random.normal(25,8)))
        avg_del  = max(15, int(np.random.normal(35,10)))
        aov_restaurant = price_rng * 200 + np.random.normal(0,50)

        rows.append({
            "restaurant_id":       f"R{i+1:05d}",
            "restaurant_name":     f"{cuisine.split('/')[0]} {rtype.replace('_',' ').title()} {i+1}",
            "restaurant_type":     rtype,
            "cuisine_type":        cuisine,
            "city":                city,
            "city_tier":           CITY_TIER[city],
            "zone_type":           zone,
            "price_range":         price_rng,
            "restaurant_rating":   round(rating, 1),
            "delivery_rating":     round(clamp(rating + np.random.normal(0,0.3), 1.0, 5.0), 1),
            "is_pure_veg":         (cuisine in ["Desserts","Bakery","South Indian"]) or random.random()<0.15,
            "chain_indicator":     rtype == "chain",
            "cloud_kitchen":       rtype == "cloud_kitchen",
            "menu_size":           menu_size,
            "has_combos":          random.random() < 0.60,
            "avg_prep_time_min":   avg_prep,
            "avg_delivery_time_min": avg_del,
            "avg_order_value":     round(aov_restaurant, 2),
            "age_days":            age_days,
            "is_new_restaurant":   is_new_rest,
            "is_cold_start":       is_very_new,
        })

    return pd.DataFrame(rows)


# MENU ITEMS
def generate_items(n: int, restaurants: pd.DataFrame) -> pd.DataFrame:
    print(f"[3/8] Generating {n:,} menu items …")

    # Item name templates per category
    ITEM_NAMES = {
        "main_course": ["Butter Chicken","Chicken Biryani","Paneer Tikka Masala","Dal Makhani",
                        "Veg Fried Rice","Chicken Noodles","Margherita Pizza","Pasta Arrabiata",
                        "Chicken Burger","Fish & Chips","Lamb Rogan Josh","Chole Bhature",
                        "Rajma Rice","Palak Paneer","Chicken Curry","Mutton Biryani"],
        "starter":     ["Chicken Kebab","Paneer Tikka","Spring Rolls","Chilli Chicken",
                        "Samosa","Veg Manchurian","Chicken Wings","Onion Bhaji",
                        "Fish Tikka","Veg Spring Roll","Seekh Kebab","Aloo Tikki"],
        "side":        ["Raita","Salan","Gulab Jamun Sauce","Coleslaw","Fries",
                        "Garden Salad","Garlic Dip","Mint Chutney","Tamarind Chutney",
                        "Mixed Pickle","Boondi Raita","Papad"],
        "bread":       ["Butter Naan","Garlic Naan","Roti","Paratha","Garlic Bread",
                        "Pita Bread","Kulcha","Laccha Paratha","Missi Roti"],
        "beverage":    ["Coke","Pepsi","Lassi","Mango Shake","Cold Coffee",
                        "Masala Chai","Fresh Lime Soda","Buttermilk","Orange Juice",
                        "Watermelon Juice","Iced Tea","Mojito","Filter Coffee"],
        "dessert":     ["Gulab Jamun","Ice Cream","Gajar Halwa","Brownie","Kulfi",
                        "Kheer","Ras Malai","Jalebi","Cake Slice","Chocolate Mousse",
                        "Basundi","Peda"],
        "combo":       ["Combo Meal","Family Pack","Student Combo","Value Meal",
                        "Party Pack","Meal Deal","Happy Meal"],
    }

    rest_ids = restaurants["restaurant_id"].tolist()
    rows = []
    item_counter = 1

    items_per_restaurant = {}
    total_menu = restaurants["menu_size"].sum()
    for _, r in restaurants.iterrows():
        items_per_restaurant[r["restaurant_id"]] = max(5, int(n * r["menu_size"] / total_menu))

    for rest_id in tqdm(rest_ids, desc="  items"):
        rest_row = restaurants[restaurants.restaurant_id == rest_id].iloc[0]
        cuisine  = rest_row["cuisine_type"]
        price_rng= rest_row["price_range"]
        n_items_here = items_per_restaurant[rest_id]

        for _ in range(n_items_here):
            if item_counter > n:
                break
            cat = weighted_choice(ITEM_CATEGORIES, CATEGORY_WEIGHTS)
            names = ITEM_NAMES.get(cat, ["Item"])
            base_name = random.choice(names)

            # Price by category & restaurant price_range
            price_ranges_by_cat = {
                "main_course": (150, 500), "starter": (80, 300),
                "side":        (30, 150),  "bread":   (20,  80),
                "beverage":    (20, 150),  "dessert": (60, 300),
                "combo":       (250, 700),
            }
            plo, phi = price_ranges_by_cat[cat]
            price = clamp(
                np.random.lognormal(
                    np.log((plo+phi)/2 * price_rng/2.5), 0.3
                ), plo, phi * price_rng
            )

            # Veg preference based on cuisine and restaurant
            if rest_row["is_pure_veg"]:
                is_veg = True
            elif cuisine in ["Biryani/Mughlai","Fast Food"]:
                is_veg = random.random() < 0.25
            else:
                is_veg = random.random() < 0.40

            pop = np.random.beta(2, 8)  
            is_bestseller = pop > 0.70
            is_new_item   = random.random() < 0.15

            availability  = random.choices(
                ["all_day","morning_only","evening_only","lunch_dinner"],
                weights=[0.70,0.10,0.05,0.15]
            )[0]

            rows.append({
                "item_id":            f"I{item_counter:07d}",
                "restaurant_id":      rest_id,
                "item_name":          f"{base_name} ({item_counter})",
                "category":           cat,
                "price":              round(price, 2),
                "is_veg":             is_veg,
                "is_bestseller":      is_bestseller,
                "is_spicy":           random.random() < 0.35,
                "is_combo":           cat == "combo",
                "cuisine_tag":        cuisine,
                "popularity_score":   round(pop, 4),
                "item_rating":        round(clamp(np.random.beta(7,3)*5, 2.0, 5.0), 1),
                "rating_count":       int(np.random.exponential(100)),
                "calories":           int(np.random.normal(400, 200)),
                "availability":       availability,
                "is_new_item":        is_new_item,
                "days_on_menu":       0 if is_new_item else random.randint(10, 1000),
                "preparation_time":   max(5, int(np.random.normal(20,8))),
                "has_customization":  random.random() < 0.30,
            })
            item_counter += 1

        if item_counter > n:
            break

    return pd.DataFrame(rows)


# ITEM COMPLEMENTARITY MATRIX
def generate_complementarity(items: pd.DataFrame) -> pd.DataFrame:
    print(f"[4/8] Building item complementarity pairs …")
    pairs = []
    items_by_rest = items.groupby("restaurant_id")

    for rest_id, grp in tqdm(items_by_rest, desc="  complements"):
        grp = grp.reset_index(drop=True)
        cats = grp.groupby("category")

        for i, row_a in grp.iterrows():
            cat_a = row_a["category"]
            complement_cats = COMPLEMENT_RULES.get(cat_a, {})

            for cat_b, base_prob in complement_cats.items():
                if cat_b not in cats.groups:
                    continue
                cat_b_items = cats.get_group(cat_b)
                # Sample up to 3 complements per item
                for _, row_b in cat_b_items.sample(min(3, len(cat_b_items))).iterrows():
                    score = base_prob * row_b["popularity_score"] * (0.8 + 0.4*random.random())
                    pairs.append({
                        "item_id_1":           row_a["item_id"],
                        "item_id_2":           row_b["item_id"],
                        "restaurant_id":       rest_id,
                        "complementarity_score": round(clamp(score, 0.0, 1.0), 4),
                        "co_occurrence_score": round(clamp(score * 0.9, 0.0, 1.0), 4),
                    })

    df = pd.DataFrame(pairs)

    if len(df) > 500_000:
        df = df.nlargest(500_000, "complementarity_score")
    return df


# ORDERS
def generate_orders(
    n: int,
    users: pd.DataFrame,
    restaurants: pd.DataFrame
) -> tuple:
    """Returns (orders_df, order_items_df)"""
    print(f"[5/8] Generating {n:,} orders + order items …")

    
    user_ids = users["user_id"].tolist()
    user_lookup = users.set_index("user_id")

    rest_ids  = restaurants["restaurant_id"].tolist()
    rest_lookup = restaurants.set_index("restaurant_id")

    # Temporal hour distribution (peak lunch + dinner)
    hour_weights = [
        0.003,0.002,0.001,0.001,0.002,0.005,   # 0-5 AM
        0.015,0.025,0.030,0.020,0.018,0.040,   # 6-11 AM
        0.060,0.065,0.055,0.030,0.025,0.025,   # 12-5 PM 
        0.035,0.065,0.070,0.055,0.030,0.015,   # 6-11 PM 
    ]
    hour_weights = [w/sum(hour_weights) for w in hour_weights]

    orders, order_items = [], []
    order_counter = 1
    oi_counter    = 1

    
    items_file = os.path.join(DATA_DIR, "items.csv")
    if os.path.exists(items_file):
        items = pd.read_csv(items_file)
    else:
        raise FileNotFoundError("Run item generation first.")

    items_by_rest = {rid: grp.reset_index(drop=True)
                     for rid, grp in items.groupby("restaurant_id")}

    for _ in tqdm(range(n), desc="  orders"):
        user_id = random.choice(user_ids)
        rest_id = random.choice(rest_ids)

        if rest_id not in items_by_rest or len(items_by_rest[rest_id]) == 0:
            continue

        u = user_lookup.loc[user_id]
        r = rest_lookup.loc[rest_id]

        # Timestamp: weighted by hour + seasonality
        hour  = int(np.random.choice(range(24), p=hour_weights))
        day   = random.randint(0, DATE_RANGE_DAYS)
        ts    = START_DATE + timedelta(days=day, hours=hour,
                                       minutes=random.randint(0,59))
        mtime = meal_time_from_hour(hour)
        dow   = ts.weekday()
        season= get_season(ts.month)
        festival = festival_in_date(ts)
        weather  = weighted_choice(WEATHER_CONDITIONS, WEATHER_WEIGHTS)

        # Filter items available at this restaurant
        rest_items = items_by_rest[rest_id]

        # Veg filtering
        if u["veg_preference"]:
            avail = rest_items[rest_items["is_veg"] == True]
            if len(avail) == 0:
                avail = rest_items
        else:
            avail = rest_items

        # Cart size: 50% single-item orders
        cart_probs = [0.50, 0.30, 0.12, 0.06, 0.02]  # 1,2,3,4,5+ items
        n_items_in_cart = random.choices([1,2,3,4,5], weights=cart_probs)[0]
        n_items_in_cart = min(n_items_in_cart, len(avail))
        if n_items_in_cart == 0:
            continue

        # Prioritise main course first, then complements
        main_items = avail[avail["category"] == "main_course"]
        chosen_items = []

        if len(main_items) > 0:
            chosen_items.append(main_items.sample(1).iloc[0])
            remaining = avail[~avail["item_id"].isin([ci["item_id"] for ci in chosen_items])]
            for _ in range(n_items_in_cart - 1):
                if len(remaining) == 0: break
                # Weighted by complementarity and popularity
                weights = remaining["popularity_score"].values + 0.01
                weights /= weights.sum()
                pick = remaining.sample(1, weights=weights).iloc[0]
                chosen_items.append(pick)
                remaining = remaining[remaining["item_id"] != pick["item_id"]]
        else:
            chosen_items = [avail.sample(1).iloc[0] for _ in range(n_items_in_cart)]

        # Order-level
        total_value = sum(ci["price"] for ci in chosen_items)
        has_offer   = (random.random() < u["offer_redemption_rate"])
        discount    = round(total_value * np.random.uniform(0.05,0.30), 2) if has_offer else 0.0
        final_value = max(50, total_value - discount)

        oid = f"O{order_counter:08d}"
        orders.append({
            "order_id":          oid,
            "user_id":           user_id,
            "restaurant_id":     rest_id,
            "order_date":        ts.strftime("%Y-%m-%d"),
            "order_time":        ts.strftime("%H:%M:%S"),
            "hour_of_day":       hour,
            "day_of_week":       dow,
            "is_weekend":        dow >= 5,
            "meal_time":         mtime,
            "is_peak_hour":      is_peak_hour(hour),
            "city":              r["city"],
            "city_tier":         r["city_tier"],
            "zone_type":         r["zone_type"],
            "order_status":      "completed",
            "total_items":       len(chosen_items),
            "total_value":       round(total_value, 2),
            "final_value":       round(final_value, 2),
            "has_offer":         has_offer,
            "discount_amount":   discount,
            "season":            season,
            "weather":           weather,
            "festival":          festival,
            "is_first_order":    u["total_orders"] <= 1,
        })

        for seq, ci in enumerate(chosen_items, start=1):
            qty = 1 if ci["category"] not in ["beverage"] else random.choices([1,2],[0.7,0.3])[0]
            order_items.append({
                "order_item_id":    f"OI{oi_counter:09d}",
                "order_id":         oid,
                "item_id":          ci["item_id"],
                "restaurant_id":    rest_id,
                "user_id":          user_id,
                "quantity":         qty,
                "item_price":       round(ci["price"], 2),
                "item_category":    ci["category"],
                "sequence_in_cart": seq,
                "is_add_on":        seq > 1,
            })
            oi_counter += 1

        order_counter += 1

    return pd.DataFrame(orders), pd.DataFrame(order_items)


# CART SNAPSHOTS
def generate_cart_snapshots(order_items: pd.DataFrame) -> pd.DataFrame:
    """
    For each order, create a snapshot BEFORE each item addition.
    This gives us the sequential cart-building context.
    """
    print(f"[6/8] Generating cart snapshots …")
    snapshots = []
    snap_counter = 1

    for order_id, grp in tqdm(order_items.groupby("order_id"), desc="  snapshots"):
        grp = grp.sort_values("sequence_in_cart").reset_index(drop=True)
        cart_items_so_far = []

        for idx, row in grp.iterrows():
            # Snapshot BEFORE adding this item
            cats_in_cart = [ci["item_category"] for ci in cart_items_so_far]
            cart_value   = sum(ci["item_price"] for ci in cart_items_so_far)

            has_main = any(c=="main_course" for c in cats_in_cart)
            has_bev  = any(c=="beverage"    for c in cats_in_cart)
            has_des  = any(c=="dessert"     for c in cats_in_cart)
            has_sta  = any(c=="starter"     for c in cats_in_cart)
            has_brd  = any(c=="bread"       for c in cats_in_cart)

            # Meal completeness score (0=nothing, 1=complete)
            completeness = (
                0.4 * has_main +
                0.2 * has_bev  +
                0.15* has_des  +
                0.15* has_sta  +
                0.10* has_brd
            )

            snapshots.append({
                "snapshot_id":           f"S{snap_counter:09d}",
                "order_id":              order_id,
                "user_id":               row["user_id"],
                "restaurant_id":         row["restaurant_id"],
                "cart_item_count":       len(cart_items_so_far),
                "cart_total_value":      round(cart_value, 2),
                "has_main_course":       has_main,
                "has_beverage":          has_bev,
                "has_dessert":           has_des,
                "has_starter":           has_sta,
                "has_bread":             has_brd,
                "meal_completeness_score": round(completeness, 4),
                "next_item_added":       row["item_id"],
                "next_item_category":    row["item_category"],
                "sequence":              row["sequence_in_cart"],
            })
            snap_counter += 1

            # Now add this item to cart
            cart_items_so_far.append({
                "item_id":       row["item_id"],
                "item_category": row["item_category"],
                "item_price":    row["item_price"],
            })

    return pd.DataFrame(snapshots)



# CSAO INTERACTIONS
def generate_csao_interactions(
    cart_snapshots: pd.DataFrame,
    items: pd.DataFrame,
    users: pd.DataFrame
) -> pd.DataFrame:
    """
    Simulate CSAO recommendation impressions and responses.
    For each cart snapshot, simulate 10 recommendations being shown.
    Mark accepted (=1) if it is the next_item_added.
    Also generate some negative samples.
    """
    print(f"[7/8] Generating CSAO interactions …")
    user_seg = users.set_index("user_id")["user_segment"].to_dict()
    user_bev = users.set_index("user_id")["beverage_order_rate"].to_dict()
    user_des = users.set_index("user_id")["dessert_order_rate"].to_dict()

    items_by_rest = items.groupby("restaurant_id")["item_id"].apply(list).to_dict()

    interactions = []
    inter_counter = 1

    # Only simulate CSAO for snapshots where cart is NOT empty (seq >= 1)
    eligible = cart_snapshots[cart_snapshots["cart_item_count"] >= 1].reset_index(drop=True)

    n_sample = min(len(eligible), int(N_SESSIONS * 1.3))
    eligible = eligible.sample(min(n_sample, len(eligible)), random_state=42)

    for _, snap in tqdm(eligible.iterrows(), total=len(eligible), desc="  csao"):
        uid  = snap["user_id"]
        rid  = snap["restaurant_id"]
        seg  = user_seg.get(uid, "occasional")
        next_item = snap["next_item_added"]

        rest_items = items_by_rest.get(rid, [])
        if len(rest_items) < 2:
            continue

        n_shown = random.randint(8, 10)
        shown = [next_item]
        negatives = [i for i in rest_items if i != next_item]
        shown += random.sample(negatives, min(n_shown-1, len(negatives)))
        random.shuffle(shown)

        # Acceptance rate by segment
        seg_accept_rates = {
            "new": 0.11, "occasional": 0.15,
            "regular": 0.20, "frequent": 0.27, "power": 0.32
        }
        base_accept = seg_accept_rates.get(seg, 0.20)

        for pos, iid in enumerate(shown, start=1):
            is_target = (iid == next_item)

            pos_bias = max(0.3, 1.0 - (pos-1)*0.08)

            # Acceptance probability
            accept_prob = base_accept * pos_bias
            if is_target:
                accept_prob = min(0.95, accept_prob * 2.5)  

            clicked = random.random() < (accept_prob * 1.5)
            added   = clicked and (random.random() < accept_prob)

            time_to_click = max(1, int(np.random.exponential(10))) if clicked else 0

            interactions.append({
                "interaction_id":        f"C{inter_counter:09d}",
                "snapshot_id":           snap["snapshot_id"],
                "order_id":              snap["order_id"],
                "user_id":               uid,
                "restaurant_id":         rid,
                "recommended_item_id":   iid,
                "recommendation_position": pos,
                "item_clicked":          clicked,
                "item_added":            added,
                "time_to_click_sec":     time_to_click,
                "is_positive_label":     is_target,
                "cart_completeness_before": snap["meal_completeness_score"],
            })
            inter_counter += 1

    return pd.DataFrame(interactions)


# BASELINE PERFORMANCE
def generate_baseline_performance() -> pd.DataFrame:
    print("[8/8] Generating baseline performance metrics …")
    rows = []
    segments = ["new","occasional","regular","frequent","power"]
    meal_times = MEAL_TIMES

    for seg in segments:
        for mtime in meal_times:
            base_aov_by_seg = {
                "new":280,"occasional":330,"regular":380,"frequent":430,"power":520
            }
            base_aov = base_aov_by_seg[seg] + np.random.normal(0,20)
            rows.append({
                "user_segment":         seg,
                "meal_time":            mtime,
                "baseline_aov":         round(base_aov, 2),
                "baseline_acceptance_rate": round(np.random.uniform(0.08, 0.18), 4),
                "baseline_ctr":         round(np.random.uniform(0.10, 0.22), 4),
                "baseline_attach_rate": round(np.random.uniform(0.05, 0.15), 4),
                "baseline_c2o_ratio":   round(np.random.uniform(0.60, 0.80), 4),
                "baseline_avg_items":   round(np.random.uniform(1.5, 3.0), 2),
            })
    return pd.DataFrame(rows)


# MAIN PIPELINE
def main():
    print("=" * 60)
    print("ZOMATO CSAO - SYNTHETIC DATA GENERATION")
    print(f"Scale: {SCALE:.0%}  |  Users: {N_USERS:,}  |  Restaurants: {N_RESTAURANTS:,}")
    print("=" * 60)

    # 1. Users
    users = generate_users(N_USERS)
    users.to_csv(os.path.join(DATA_DIR, "users.csv"), index=False)
    print(f"   Saved users.csv  ({len(users):,} rows)")

    # 2. Restaurants
    restaurants = generate_restaurants(N_RESTAURANTS)
    restaurants.to_csv(os.path.join(DATA_DIR, "restaurants.csv"), index=False)
    print(f"   Saved restaurants.csv  ({len(restaurants):,} rows)")

    # 3. Items
    items = generate_items(N_ITEMS, restaurants)
    items.to_csv(os.path.join(DATA_DIR, "items.csv"), index=False)
    print(f"   Saved items.csv  ({len(items):,} rows)")

    # 4. Complementarity
    comp = generate_complementarity(items)
    comp.to_csv(os.path.join(DATA_DIR, "item_complementarity.csv"), index=False)
    print(f"   Saved item_complementarity.csv  ({len(comp):,} rows)")

    # 5. Orders
    orders, order_items = generate_orders(N_ORDERS, users, restaurants)
    orders.to_csv(os.path.join(DATA_DIR, "orders.csv"), index=False)
    order_items.to_csv(os.path.join(DATA_DIR, "order_items.csv"), index=False)
    print(f"   Saved orders.csv ({len(orders):,}) + order_items.csv ({len(order_items):,})")

    # 6. Cart snapshots
    snaps = generate_cart_snapshots(order_items)
    snaps.to_csv(os.path.join(DATA_DIR, "cart_snapshots.csv"), index=False)
    print(f"   Saved cart_snapshots.csv  ({len(snaps):,} rows)")

    # 7. CSAO interactions
    csao = generate_csao_interactions(snaps, items, users)
    csao.to_csv(os.path.join(DATA_DIR, "csao_interactions.csv"), index=False)
    print(f"   Saved csao_interactions.csv  ({len(csao):,} rows)")

    # 8. Baseline performance
    baseline = generate_baseline_performance()
    baseline.to_csv(os.path.join(DATA_DIR, "baseline_performance.csv"), index=False)
    print(f"   Saved baseline_performance.csv  ({len(baseline):,} rows)")

    print("\n── DATA VALIDATION ──")
    print(f"  User segments : {users['user_segment'].value_counts().to_dict()}")
    print(f"  Cold-start users: {users['is_cold_start'].sum():,} ({users['is_cold_start'].mean():.1%})")
    print(f"  Cuisines covered: {restaurants['cuisine_type'].nunique()}")
    print(f"  Item categories : {items['category'].value_counts().to_dict()}")
    print(f"  Veg ratio       : {items['is_veg'].mean():.1%}")
    print(f"  Single-item orders: {(orders['total_items']==1).mean():.1%}")
    print(f"  Peak-hour orders  : {orders['is_peak_hour'].mean():.1%}")
    print(f"  CSAO acceptance   : {csao['item_added'].mean():.1%}")

    print("\n Data generation complete!")
    return users, restaurants, items, orders, order_items, snaps, csao


if __name__ == "__main__":
    main()