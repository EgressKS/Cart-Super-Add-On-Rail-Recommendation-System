"""
CSAO Recommendation Model Architecture

Three-stage model:
  1. Two-Tower Neural Network 
  2. Sequential Attention      
  3. LLM Semantic Embeddings   
Final score = weighted ensemble of the three signals + rule-based boost.
"""

import numpy as np
import os, json

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _ = torch.zeros(1)
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False
    print("[WARN] PyTorch not available. Using sklearn-based fallback model.")

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")


# PYTORCH TWO-TOWER + ATTENTION MODEL
if TORCH_AVAILABLE:
    class UserTower(nn.Module):
        """Encodes user features into a dense user embedding."""
        def __init__(self, n_user_feats: int, emb_dim: int = 64):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(n_user_feats, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, emb_dim),
                nn.ReLU(),
            )

        def forward(self, x):
            return self.net(x)


    class ItemTower(nn.Module):
        """Encodes item + restaurant features into a dense item embedding."""
        def __init__(self, n_item_feats: int, emb_dim: int = 64):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(n_item_feats, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, emb_dim),
                nn.ReLU(),
            )

        def forward(self, x):
            return self.net(x)


    class ContextTower(nn.Module):
        """Encodes cart state + temporal + geographic context."""
        def __init__(self, n_ctx_feats: int, emb_dim: int = 32):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(n_ctx_feats, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, emb_dim),
                nn.ReLU(),
            )

        def forward(self, x):
            return self.net(x)


    class TwoTowerRankingModel(nn.Module):
        """
        Full Two-Tower ranking model with:
         - User Tower, Item Tower, Context Tower
         - Deep network for non-linear transformations
         - Sigmoid output → P(item will be added to cart)
        """
        def __init__(
            self,
            n_user_feats:  int,
            n_item_feats:  int,
            n_ctx_feats:   int,
            emb_dim:       int = 64,
            dropout:       float = 0.3,
        ):
            super().__init__()
            self.user_tower = UserTower(n_user_feats, emb_dim)
            self.item_tower = ItemTower(n_item_feats, emb_dim)
            self.ctx_tower  = ContextTower(n_ctx_feats, emb_dim // 2)

            interaction_dim = emb_dim + emb_dim + emb_dim // 2

            self.cross1 = nn.Linear(interaction_dim, interaction_dim, bias=True)
            self.cross2 = nn.Linear(interaction_dim, interaction_dim, bias=True)

            # Deep Network
            self.deep = nn.Sequential(
                nn.Linear(interaction_dim, 256),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, 64),
                nn.ReLU(),
            )

            # Output
            combined_dim = interaction_dim + 64  
            self.output = nn.Sequential(
                nn.Linear(combined_dim, 32),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(32, 1),
                nn.Sigmoid(),
            )

        def forward(self, user_x, item_x, ctx_x):
            u_emb  = self.user_tower(user_x)    
            i_emb  = self.item_tower(item_x)    
            c_emb  = self.ctx_tower(ctx_x)      

            # Concatenate all tower outputs
            x = torch.cat([u_emb, i_emb, c_emb], dim=-1)  

            # Cross network
            x0    = x
            x_cross = x0 * self.cross1(x) + x0
            x_cross = x0 * self.cross2(x_cross) + x0

            # Deep network
            x_deep = self.deep(x)

            # Combine
            combined = torch.cat([x_cross, x_deep], dim=-1)
            return self.output(combined).squeeze(-1)


    class CartAttentionLayer(nn.Module):
        """
        Multi-head self-attention over cart items.
        Given N items in cart → learns which items most influence next recommendation.
        """
        def __init__(self, item_emb_dim: int = 64, n_heads: int = 4):
            super().__init__()
            self.attn = nn.MultiheadAttention(
                embed_dim=item_emb_dim,
                num_heads=n_heads,
                batch_first=True,
                dropout=0.1,
            )
            self.norm = nn.LayerNorm(item_emb_dim)

        def forward(self, cart_embeddings):
            """
            cart_embeddings: (B, seq_len, emb_dim)
            Returns: (B, emb_dim) — pooled cart representation
            """
            attn_out, _ = self.attn(
                cart_embeddings,
                cart_embeddings,
                cart_embeddings,
            )
            attn_out = self.norm(attn_out + cart_embeddings) 
            return attn_out.mean(dim=1)
        

# SKLEARN GRADIENT BOOSTING
class GBMRankingModel:
    """
    Gradient Boosting model for CSAO ranking.
    Faster to train than neural networks; used as:
     - Strong baseline for comparison
     - Fallback when PyTorch not available
    """
    def __init__(
        self,
        n_estimators:  int   = 300,
        max_depth:     int   = 6,
        learning_rate: float = 0.05,
        subsample:     float = 0.8,
        random_state:  int   = 42,
    ):
        self.model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            min_samples_leaf=20,
            random_state=random_state,
            verbose=0,
        )
        self.scaler = StandardScaler()
        self.feature_cols = None
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: list = None):
        self.feature_cols = feature_names
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_fitted = True

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]

    def feature_importance(self) -> dict:
        if not self.is_fitted or self.feature_cols is None:
            return {}
        return dict(zip(self.feature_cols,
                        self.model.feature_importances_))

    def save(self, path: str):
        import pickle
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str):
        import pickle
        with open(path, "rb") as f:
            return pickle.load(f)


# ITEM SEMANTIC EMBEDDING (LLM-based)
class ItemSemanticEmbedder:
    """
    Generates semantic embeddings for menu items using Sentence-BERT.
    Falls back to TF-IDF embeddings if sentence-transformers not installed.

    Usage:
      embedder = ItemSemanticEmbedder()
      embeddings = embedder.encode(items_df)  # (N, 384)
      sim_score  = embedder.cosine_sim(emb_a, emb_b)
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self._try_load_sentence_transformer()

    def _try_load_sentence_transformer(self):
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
            print(f"  [LLM] Sentence-BERT loaded: {self.model_name}")
        except Exception:
            print("  [LLM] sentence-transformers not available; using TF-IDF fallback")
            self.model = None

    def _make_item_text(self, row) -> str:
        """Convert item attributes to rich text for LLM encoding."""
        return (
            f"{row.get('item_name','')} "
            f"category:{row.get('category','')} "
            f"cuisine:{row.get('cuisine_tag','')} "
            f"{'veg' if row.get('is_veg', False) else 'nonveg'} "
            f"price:{row.get('price',0):.0f} "
            f"{'bestseller' if row.get('is_bestseller',False) else ''} "
            f"{'spicy' if row.get('is_spicy',False) else ''}"
        )

    def encode(self, items_df, batch_size: int = 256) -> np.ndarray:
        """
        Encode all items → embeddings matrix.
        Returns np.ndarray of shape (N, embedding_dim).
        """
        texts = [self._make_item_text(row) for _, row in items_df.iterrows()]

        if self.model is not None:
            # Sentence-BERT path
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=True,
                convert_to_numpy=True,
            )
        else:
            # TF-IDF fallback will keep dimensionality manageable
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.decomposition import TruncatedSVD
            tfidf = TfidfVectorizer(max_features=500, ngram_range=(1,2))
            X     = tfidf.fit_transform(texts).toarray()
            svd   = TruncatedSVD(n_components=min(64, X.shape[1]-1), random_state=42)
            embeddings = svd.fit_transform(X)

        return embeddings.astype(np.float32)

    @staticmethod
    def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na < 1e-8 or nb < 1e-8:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    @staticmethod
    def batch_cosine_sim(query_emb: np.ndarray, corpus_embs: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between one query and N corpus embeddings."""
        q = query_emb / (np.linalg.norm(query_emb) + 1e-8)
        c = corpus_embs / (np.linalg.norm(corpus_embs, axis=1, keepdims=True) + 1e-8)
        return (c @ q).astype(np.float32)


# RULE-BASED SCORER
CART_COMPLETION_RULES = {
    # If category X is in cart, boost category Y
    "main_course": {"beverage": 0.5, "dessert": 0.4, "side": 0.6, "bread": 0.5},
    "starter":     {"main_course": 0.8, "beverage": 0.5},
    "combo":       {"beverage": 0.6, "dessert": 0.4},
}

VEG_STRICT_CATEGORIES = {"beverage", "dessert", "side"}


def rule_based_score(
    candidate_item:  dict,
    cart_categories: list,
    cart_value:      float,
    meal_completeness: float,
    user_is_veg:     bool = False,
) -> float:
    """
    Heuristic rule-based score [0..1] for candidate item given current cart.
    This forms the 5% rule-based component of the ensemble.
    """
    score = 0.0
    cat   = candidate_item.get("category", "main_course")

    # Hard constraint: never recommend non-veg to veg user
    if user_is_veg and not candidate_item.get("is_veg", True):
        return -1.0  

    # Meal completion boost
    missing_beverage = "beverage" not in cart_categories
    missing_dessert  = "dessert"  not in cart_categories
    missing_side     = "side"     not in cart_categories

    if cat == "beverage" and missing_beverage:
        score += 0.6
    if cat == "dessert"  and missing_dessert:
        score += 0.4 + (0.2 if meal_completeness > 0.6 else 0.0)
    if cat == "side"     and missing_side and "main_course" in cart_categories:
        score += 0.55

    # Cart-completion rules
    for cart_cat in cart_categories:
        for boost_cat, boost_val in CART_COMPLETION_RULES.get(cart_cat, {}).items():
            if cat == boost_cat:
                score += boost_val * 0.5   

    # Bestseller bonus
    if candidate_item.get("is_bestseller", False):
        score += 0.1

    # Don't recommend >50% more than avg cart item price
    if cart_value > 0 and len(cart_categories) > 0:
        avg_price  = cart_value / len(cart_categories)
        item_price = candidate_item.get("price", avg_price)
        if item_price > avg_price * 2.0:
            score -= 0.2

    return min(1.0, max(0.0, score))


# ENSEMBLE SCORER
def ensemble_score(
    neural_score:    float,
    attention_boost: float,
    semantic_score:  float,
    rule_score:      float,
    weights:         tuple = (0.60, 0.20, 0.15, 0.05),
) -> float:
    """
    Weighted ensemble of all scoring signals.
    Default weights from problem document architecture.
    """
    w_nn, w_attn, w_llm, w_rule = weights
    final = (
        w_nn   * neural_score   +
        w_attn * attention_boost +
        w_llm  * semantic_score  +
        w_rule * rule_score
    )
    return float(np.clip(final, 0.0, 1.0))


# MMR RE-RANKING for diversity
def mmr_reranking(
    candidates:  list,   
    item_embs:   dict,    
    lambda_param: float = 0.7,
    top_k:        int   = 10,
) -> list:
    """
    Maximal Marginal Relevance re-ranking.
    Balances relevance score vs distance from already selected.

    lambda_param: 1.0 = pure relevance, 0.0 = pure diversity
    """
    if not candidates:
        return []

    candidates = sorted(candidates, key=lambda x: -x["score"])
    selected   = [candidates[0]]
    remaining  = candidates[1:]

    while len(selected) < top_k and remaining:
        best_mmr, best_item = -1e9, None

        for c in remaining:
            relevance = c["score"]

            # Similarity to already selected items
            if c["item_id"] in item_embs:
                sims = []
                for s in selected:
                    if s["item_id"] in item_embs:
                        sim = ItemSemanticEmbedder.cosine_sim(
                            item_embs[c["item_id"]],
                            item_embs[s["item_id"]],
                        )
                        sims.append(sim)
                max_sim = max(sims) if sims else 0.0
            else:
                # Fallback: category-based similarity
                max_sim = 0.7 if c["category"] == selected[-1].get("category") else 0.2

            mmr = lambda_param * relevance - (1 - lambda_param) * max_sim
            if mmr > best_mmr:
                best_mmr, best_item = mmr, c

        if best_item is None:
            break
        selected.append(best_item)
        remaining.remove(best_item)

    return selected


# COLD-START HANDLER
def cold_start_recommendations(
    restaurant_id:   str,
    cart_categories: list,
    meal_time:       str,
    item_features:   "pd.DataFrame",
    top_k:           int = 10,
) -> list:
    """
    Pure rule + popularity-based recommendations for new users / new restaurants.
    Covers cold-start scenario completely.
    """
    rest_items = item_features[item_features["restaurant_id"] == restaurant_id].copy()
    if len(rest_items) == 0:
        return []

    # Filter by meal-time availability
    avail_map = {
        "breakfast": ["all_day","morning_only"],
        "lunch":     ["all_day","lunch_dinner"],
        "snack":     ["all_day","evening_only"],
        "dinner":    ["all_day","lunch_dinner"],
        "late_night":["all_day"],
    }
    allowed = avail_map.get(meal_time, ["all_day"])
    rest_items = rest_items[rest_items["availability"].isin(allowed + ["all_day"])]

    # Meal-completion priority
    needs_beverage = "beverage" not in cart_categories
    needs_dessert  = "dessert"  not in cart_categories
    needs_side     = "side"     not in cart_categories

    def priority(row):
        p = row["popularity_score"] + (0.5 if row.get("is_bestseller", False) else 0)
        if row["category"] == "beverage" and needs_beverage:  p += 0.6
        if row["category"] == "dessert"  and needs_dessert:   p += 0.4
        if row["category"] == "side"     and needs_side:      p += 0.5
        return p

    rest_items["priority"] = rest_items.apply(priority, axis=1)
    top = rest_items.nlargest(top_k, "priority")

    return [
        {
            "item_id":  row["item_id"],
            "score":    round(row["priority"], 4),
            "category": row["category"],
            "name":     row["item_name"],
            "price":    row["price"],
            "strategy": "cold_start_popularity",
        }
        for _, row in top.iterrows()
    ]