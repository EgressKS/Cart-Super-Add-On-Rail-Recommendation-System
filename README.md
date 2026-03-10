# Cart Super Add-On (CSAO) Recommendation System


---

## Problem Statement

Build a recommendation system that suggests relevant add-on items
to customers based on their current cart composition, contextual factors, and
historical behaviour patterns, while maintaining high acceptance rates and
customer satisfaction.Recommendations must update in real-time as items are 
added (Biryani → suggest Raita → added → suggest Gulab Jamun → added → suggest Lassi).

---

## API Reference

### POST /recommend

```json
Request:
{
  "user_id": "U0000001",
  "restaurant_id": "R00042",
  "cart_items": [
    {"item_id": "I0001234", "quantity": 1}
  ],
  "context": {
    "timestamp": "2025-12-01T13:30:00",
    "city": "Mumbai",
    "zone": "business"
  }
}

Response:
{
  "user_id": "U0000001",
  "restaurant_id": "R00042",
  "recommendations": [
    {"item_id": "I0001567", "item_name": "Raita (567)", "category": "side",
     "price": 65.0, "score": 0.8234, "strategy": "personalized"},
    {"item_id": "I0001890", "item_name": "Gulab Jamun (890)", "category": "dessert",
     "price": 85.0, "score": 0.7612, "strategy": "personalized"},
    ...
  ],
  "latency_ms": 172.3,
  "strategy_used": "personalized"
}
```

### GET /health
Returns model status and feature store stats.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate synthetic dataset
python data/generation/generate_data.py

# 3. Feature engineering + temporal split
python features/feature_engineering.py

# 4. Train model + full evaluation
python models/train_and_evaluate.py

# 5. Business impact analysis + A/B test design
python models/business_analysis.py

# 6. Standalone API test
python api/main.py

# 7. Start FastAPI server
uvicorn api.main:app --reload --port 8000
```

---

## Project Structure

```
zomato_csao/
├── data/
│   ├── generation/
│   │   └── generate_data.py      # Full synthetic data generation
│   ├── raw/                      # Generated CSVs (see Dataset section)
│   ├── processed/                # Feature matrices + train/val/test splits
│   └── metadata/                 # Feature names, statistics
│
├── features/
│   └── feature_engineering.py    # Full feature pipeline (54 features)
│
├── models/
│   ├── model.py                  # Architecture: Two-Tower + Attention + LLM
│   ├── train_and_evaluate.py     # Training, baselines, evaluation, error analysis
│   ├── business_analysis.py      # Business impact + A/B testing framework
│   ├── saved/                    # Trained model artifacts (gbm_model.pkl)
│   └── reports/                  # Evaluation + business reports, charts
│
├── api/
│   └── main.py                   # FastAPI service (<300ms latency)
│
├── requirements.txt
└── README.md
```

---

## 1. Dataset Design

### Synthetic Data — What We Generate

| File | Rows | Description |
|------|------|-------------|
| `users.csv` | 100K | 5 segments × 3 budget tiers × meal patterns |
| `restaurants.csv` | 5K | 5 types × 15 cuisines × 10 cities |
| `items.csv` | 50K | 7 categories, veg/non-veg, price ₹50–₹1,500 |
| `item_complementarity.csv` | 500K | Pre-computed item co-occurrence pairs |
| `orders.csv` | 1M | 2-year temporal span, all meal times, seasonal |
| `order_items.csv` | 3M | Sequential cart building |
| `cart_snapshots.csv` | 5M | Cart state before each item addition |
| `csao_interactions.csv` | 2M | Impressions, clicks, accepts, rejects |
| `baseline_performance.csv` | – | Rule-based baseline metrics per segment |

### Realism Features

**User diversity:**
- **New users** (10%) — cold start, 0–2 orders
- **Occasional** (30%) — sparse data, 3–10 orders
- **Regular** (40%) — 11–50 orders, emerging patterns
- **Frequent** (15%) — clear preferences, 51–200 orders
- **Power** (5%) — daily users, 200+ orders

**Meal pattern gaps (70% users have incomplete patterns):**
- Lunch+Dinner only (25%), Dinner only (20%), Lunch only (15%)
- 60% of orders missing beverage | 80% missing dessert

**Temporal realism:**
- Breakfast 15% | Lunch 35% | Snack 10% | Dinner 35% | Late night 5%
- Weekday vs weekend patterns | Festival spikes (+40% AOV on Diwali/Holi)
- Seasonal: summer beverage spike +30%, monsoon comfort food +20%

**Sequential cart context:**
- 5M cart snapshots capture what was in cart BEFORE each addition
- Tracks: has_main, has_beverage, has_dessert, meal_completeness_score
- Dynamic recommendations update as cart changes (Biryani→Raita→Salan chain)

**Cold start coverage:**
- 15% new users (0–2 orders), 15% new restaurants (<1 month)
- 15% new items (<10 orders)

---

## 2. Problem Formulation 

### Mathematical Formulation

```
R_k = argmax_{i ∈ I \ C_t} P(add = 1 | u, H_u, C_t, X_t, i)
```

Where:
- `H_u` = user's historical behaviour
- `C_t` = current cart state (sequence of items)
- `X_t` = context (time, location, restaurant, session)
- `I \ C_t` = candidate items not already in cart


### Multi-Stage Pipeline

```
Stage 1: Candidate Generation (Recall focus, <30ms)
  ├── Item-Item CF (complementarity index)
  ├── Meal-completion rules (no beverage → suggest beverages)
  ├── Restaurant bestsellers (popularity-based)
  └── Random restaurant items (exploration)
  → ~150 candidates

Stage 2: Neural Ranking (Precision focus, <80ms)
  ├── Two-Tower Neural Network (user × item × context)
  ├── Attention over cart items (sequential context)
  └── LLM semantic embeddings (item text similarity)
  → Scored list of 150 candidates

Stage 3: Business Re-ranking (<10ms)
  ├── MMR for diversity (lambda=0.7)
  ├── Category-completeness constraints
  ├── Veg/non-veg filtering
  └── Price-range constraints
  → Final top-10 recommendations
```

### Cold Start Strategies

| Scenario | Strategy | Expected Acceptance |
|----------|----------|---------------------|
| New user (0 orders) | Cart rules + restaurant bestsellers + zone patterns | 20–25% |
| New restaurant | Cuisine-based analogues + category rules | 22–28% |
| New item | LLM semantic embedding (zero-shot) + category fallback | 18–25% |

---

## 3. Model Architecture 

### Two-Tower Neural Network (Base Model)

```
User Tower  [user_id_emb(128) + behavioral_feats(50)] → Dense[256,128,64]
Item Tower  [item_id_emb(128) + item_feats(30)]       → Dense[256,128,64]
Context Tower [cart_feats(40) + temporal(15)]          → Dense[128,64]

Interaction: Concatenate(64+64+32) = 160-dim
  → Cross Network (3 cross layers)  +  Deep Network [256,128,64]
  → Combine → Dense(32) → Sigmoid
  → P(add_to_cart)
```

### Attention for Sequential Cart Context

```python
# Self-attention over cart items
cart_rep = MultiHeadAttention(cart_items, n_heads=4, dim=128)
# Cross-attention: how much does each cart item influence candidate?
attention_score = CrossAttention(candidate_item, cart_items)
```

Captures: "User added Biryani then Raita → Gulab Jamun is now the most likely next add-on"

### LLM Semantic Embeddings 

```python
item_text = f"{item_name} - {category} - {cuisine} - {'veg' if is_veg else 'nonveg'}"
item_embedding = SentenceTransformer("all-MiniLM-L6-v2").encode(item_text)  # 384-dim

# Zero-shot cold start: embed any new item immediately
# Find semantically similar items to known complements
similarity = cosine_similarity(new_item_emb, known_complement_embs)
```

**Why LLM embeddings?**
- New items get recommendations **immediately** (no cold-start wait)
- "Butter Naan" automatically recognised as similar to "Garlic Naan"
- Cross-restaurant learning: South Indian restaurant learns from similar cuisines

### Ensemble Score

```python
final_score = (
    0.60 * neural_network_score   +  # Two-tower base model
    0.20 * attention_boost        +  # Sequential cart context
    0.15 * llm_semantic_score     +  # Semantic similarity
    0.05 * rule_based_boost          # Hard business rules
)
```

### Production-Ready Fallback (GBM)

For environments where PyTorch is unavailable (e.g., constrained containers):
- **Gradient Boosting (sklearn)** with 200 trees, max_depth=5
- Trained on same feature set, achieves **AUC = 0.77** on test set
- Inference: <10ms per batch of 150 candidates

---

## 4. Evaluation Results 

### Model Comparison

| Model | AUC-ROC | NDCG@10 | P@10 | vs Rule-based |
|-------|---------|---------|------|---------------|
| Popularity Baseline | 0.537 | 0.085 | 0.10 | baseline |
| Rule-Based | 0.605 | 0.293 | 0.30 | +0 pp |
| Logistic Regression | 0.752 | 0.934 | 0.90 | +16 pp AUC |
| **GBM Ranking Model** | **0.766** | **0.927** | **0.90** | **+16 pp AUC** |

GBM beats rule-based by **+16 pp AUC** and **+63 pp NDCG@10**.

### Cold Start Performance

| Scenario | AUC | NDCG@10 |
|----------|-----|---------|
| Cold-start users | 0.694 | 0.066 |
| New restaurants | 0.749 | 0.562 |
| New items | 0.767 | 0.514 |

Cold-start users handled gracefully via rule-based fallback (still 20–25% acceptance).

### Top Feature Importance

| Feature | Importance |
|---------|-----------|
| avg_complementarity | 35.5% |
| position_bias | 22.3% |
| user_log_orders | 17.6% |
| segment_enc | 8.2% |
| pop_score | 2.6% |

**Insight:** Item-cart complementarity is the strongest signal — validates our
cart-context-first approach.

### Data Split (Temporal — No Leakage)

```
Train:      Jan 2024 – Aug 2025  (80% — 462K interactions)
Validation: Sep 2025 – Oct 2025  (10% — 58K interactions)
Test:       Nov 2025 – Dec 2025  (10% — 59K interactions)
```

No future data leaked into training — simulates real deployment.

---

## 5. Business Impact 

### Projected Business Metrics

| Metric | Baseline | Projected | Lift |
|--------|----------|-----------|------|
| Acceptance Rate | 13% | ~23% | +10 pp |
| Avg Order Value | ₹380 | ₹461 | +21% |
| CSAO Attach Rate | 15% | ~35% | +20 pp |
| Cart-to-Order Ratio | 72% | ~76% | +4 pp |
| Avg Items/Order | 2.1 | 2.9 | +0.8 items |

**Annual Commission Impact:** ₹1,774 Cr (at 20% commission, 3M daily orders)

### Segment Insights

- **Power users:** 36% projected acceptance (2x vs new users)
- **Budget users:** Respond best to low-price beverage/side suggestions
- **Premium users:** Respond to curated dessert + branded beverages
- **Festival orders:** +40% AOV spike — pre-load seasonal recommendations
- **Late-night:** Comfort food bundles outperform individual suggestions

---

## Key Design Decisions & Trade-offs

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Problem framing | Ranking (not classification) | Optimises position of relevant items |
| Candidate generation | Multi-strategy (CF + rules + popularity) | High recall; no cold-start gaps |
| Base model | GBM (prod) + Two-Tower NN (target) | GBM: fast, interpretable; NN: higher ceiling |
| LLM integration | Sentence-BERT embeddings (offline) | Zero-shot cold start; no latency cost |
| Re-ranking | MMR (λ=0.7) | Balances relevance vs category diversity |
| Data split | Temporal (80/10/10) | Prevents leakage; simulates deployment |
| Cold start | 4-strategy fallback | Guarantees recommendations for all users |

---

## Limitations & Future Improvements

1. **Multi-restaurant recommendations:** Currently per-restaurant; cross-restaurant
   complementarity (order from two places) is out of scope but valuable.

2. **Full LLM integration:** Sentence-BERT embeddings are pre-computed; GPT-based
   real-time cart narration ("This meal is missing a refreshing beverage…") would
   boost engagement but adds 50–100ms latency.

3. **Reinforcement Learning:** Position/presentation feedback could train a bandit
   model to optimise long-term acceptance rather than single-click.

---
