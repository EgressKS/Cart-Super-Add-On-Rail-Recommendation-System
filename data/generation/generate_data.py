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
