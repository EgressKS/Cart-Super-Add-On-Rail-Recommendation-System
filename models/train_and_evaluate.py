"""
Model Training & Comprehensive Evaluation for CSAO System

  - Our Model baseline + neural baseline + ensemble model training
  - Full metric suite of AUC, Precision@K, Recall@K, NDCG@K, MRR
  - Segment-wise evaluation for user type, meal time, restaurant type, city tier
  - Cold-start evaluation for new users, new restaurants, new items
  - Baseline comparison for rule-based, popularity-based, AI model
  - Error analysis forfalse positives / negatives
  - Model serialization for serving
"""


try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _torch_test = torch.zeros(1)          
    _TORCH_OK = True
    print(f"[INFO] PyTorch {torch.__version__} loaded successfully")
except Exception as _te:
    _TORCH_OK = False
    print(f"[WARN] PyTorch pre-load failed: {_te}")


import os, json, warnings, time
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score,
    confusion_matrix, classification_report, average_precision_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")   
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings("ignore")


BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR  = os.path.join(BASE_DIR, "data", "processed")
TRAIN_DIR = os.path.join(PROC_DIR, "train")
VAL_DIR   = os.path.join(PROC_DIR, "validation")
TEST_DIR  = os.path.join(PROC_DIR, "test")
MODEL_DIR = os.path.join(BASE_DIR, "models", "saved")
REPORT_DIR= os.path.join(BASE_DIR, "models", "reports")

for d in [MODEL_DIR, REPORT_DIR]:
    os.makedirs(d, exist_ok=True)

META_FILE = os.path.join(BASE_DIR, "data", "metadata", "feature_metadata.json")
with open(META_FILE) as f:
    META = json.load(f)

FEATURE_COLS = META["feature_columns"]
LABEL_COL    = META["label_column"]


# DATA LOADING
def load_splits():
    print("Loading train / val / test splits …")
    train = pd.read_csv(os.path.join(TRAIN_DIR, "interactions_train.csv"))
    val   = pd.read_csv(os.path.join(VAL_DIR,   "interactions_val.csv"))
    test  = pd.read_csv(os.path.join(TEST_DIR,  "interactions_test.csv"))
    print(f"  Train: {len(train):,}  Val: {len(val):,}  Test: {len(test):,}")
    print(f"  Positive rate - Train: {train[LABEL_COL].mean():.3f}  "
          f"Val: {val[LABEL_COL].mean():.3f}  Test: {test[LABEL_COL].mean():.3f}")
    return train, val, test


def get_Xy(df: pd.DataFrame):
    feat_cols = [c for c in FEATURE_COLS if c in df.columns]
    X = df[feat_cols].fillna(0).values.astype(np.float32)
    y = df[LABEL_COL].values.astype(int)
    return X, y, feat_cols


# METRICS
def precision_at_k(y_true, y_scores, k=10):
    """Precision@K for a single session (or overall across sorted predictions)."""
    if len(y_true) == 0: return 0.0
    sorted_idx = np.argsort(y_scores)[::-1][:k]
    return np.mean(y_true[sorted_idx])

def recall_at_k(y_true, y_scores, k=10):
    if y_true.sum() == 0: return 0.0
    sorted_idx = np.argsort(y_scores)[::-1][:k]
    return y_true[sorted_idx].sum() / y_true.sum()

def ndcg_at_k(y_true, y_scores, k=10):
    """NDCG@K."""
    sorted_idx = np.argsort(y_scores)[::-1][:k]
    gains = y_true[sorted_idx]
    discounts = 1.0 / np.log2(np.arange(2, len(gains)+2))
    dcg = (gains * discounts).sum()

    ideal_gains = np.sort(y_true)[::-1][:k]
    idcg = (ideal_gains * discounts[:len(ideal_gains)]).sum()
    return dcg / (idcg + 1e-9)

def mean_reciprocal_rank(y_true, y_scores):
    sorted_idx = np.argsort(y_scores)[::-1]
    for rank, idx in enumerate(sorted_idx, 1):
        if y_true[idx] == 1:
            return 1.0 / rank
    return 0.0

def evaluate_model(y_true, y_scores, model_name="Model", threshold=0.5):
    """Full evaluation suite for a model on a test set."""
    y_pred = (y_scores >= threshold).astype(int)

    auc      = roc_auc_score(y_true, y_scores)
    ap       = average_precision_score(y_true, y_scores)
    prec_k5  = precision_at_k(y_true, y_scores, k=5)
    prec_k10 = precision_at_k(y_true, y_scores, k=10)
    rec_k10  = recall_at_k(y_true, y_scores, k=10)
    ndcg_k10 = ndcg_at_k(y_true, y_scores, k=10)
    mrr      = mean_reciprocal_rank(y_true, y_scores)

    results = {
        "model":         model_name,
        "auc_roc":       round(auc,  4),
        "avg_precision": round(ap,   4),
        "precision@5":   round(prec_k5,  4),
        "precision@10":  round(prec_k10, 4),
        "recall@10":     round(rec_k10,  4),
        "ndcg@10":       round(ndcg_k10, 4),
        "mrr":           round(mrr,  4),
    }
    return results


# BASELINE MODELS
def train_popularity_baseline(train, test):
    """
    Always recommend based on item popularity_score only.
    No ML, no user context.
    """
    if "pop_score" in test.columns:
        y_scores = test["pop_score"].fillna(0).values
    else:
        y_scores = np.random.rand(len(test))
    y_true = test[LABEL_COL].values
    return evaluate_model(y_true, y_scores, "Popularity Baseline")


def train_rule_based_baseline(train, test):
    """
    Rule-based scoring using cart completion signals only.
    Missing_beverage * 0.6 + missing_dessert * 0.4 + bestseller * 0.2
    """
    scores = np.zeros(len(test))
    if "missing_beverage" in test.columns:
        scores += test["missing_beverage"].fillna(0).values * 0.40
    if "missing_dessert" in test.columns:
        scores += test["missing_dessert"].fillna(0).values * 0.25
    if "missing_starter" in test.columns:
        scores += test["missing_starter"].fillna(0).values * 0.15
    if "avg_complementarity" in test.columns:
        scores += test["avg_complementarity"].fillna(0).values * 0.40
    if "is_bestseller" in test.columns:
        scores += test["is_bestseller"].fillna(0).values * 0.10
    scores = np.clip(scores, 0, 1)
    y_true = test[LABEL_COL].values
    return evaluate_model(y_true, scores, "Rule-Based Baseline")


def train_logistic_baseline(train, val, test):
    """Logistic Regression with all features."""
    X_tr, y_tr, fc = get_Xy(train)
    X_te, y_te, _  = get_Xy(test)

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)

    lr = LogisticRegression(max_iter=300, C=0.1, random_state=42, n_jobs=-1)
    lr.fit(X_tr, y_tr)
    y_scores = lr.predict_proba(X_te)[:, 1]

    return evaluate_model(y_te, y_scores, "Logistic Regression")


# MAIN MODEL GRADIENT BOOSTING

def train_gbm_model(train, val, test):
    """
    Full GBM ranking model strong ML baseline and production-ready model.
    Uses class_weight to handle positive rate imbalance (~5.6%).
    """
    import sys, os as _os
    sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
    from model import GBMRankingModel

    X_tr, y_tr, fc = get_Xy(train)
    X_v,  y_v,  _  = get_Xy(val)
    X_te, y_te, _  = get_Xy(test)

    print(f"\n[GBM] Training on {len(X_tr):,} samples ({y_tr.mean():.3f} positive rate) …")
    t0 = time.time()

    gbm = GBMRankingModel(n_estimators=200, max_depth=5, learning_rate=0.08)
    gbm.fit(X_tr, y_tr, feature_names=fc)

    val_scores = gbm.predict_proba(X_v)
    val_auc    = roc_auc_score(y_v, val_scores)
    print(f"  Val AUC: {val_auc:.4f}  (train time: {time.time()-t0:.1f}s)")

    # Test evaluation
    test_scores = gbm.predict_proba(X_te)
    results     = evaluate_model(y_te, test_scores, "GBM Ranking Model")

    # Save model
    gbm.save(os.path.join(MODEL_DIR, "gbm_model.pkl"))
    print(f"  Saved to models/saved/gbm_model.pkl")

    return results, gbm, test_scores, y_te, fc


# NEURAL MODEL TWO-TOWER + ATTENTION
def train_neural_model(train, val, test):
    """
    Train Two-Tower Neural Network (user tower x item tower x context tower).
    Uses balanced sampling to handle  positive rate.
    Returns (results_dict, test_scores) or (None, None) if PyTorch unavailable.
    """
    import sys, os as _os
    sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
    from model import TORCH_AVAILABLE

    if not TORCH_AVAILABLE:
        print("\n[Neural] PyTorch not available — skipping neural model")
        return None, None

    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from model import TwoTowerRankingModel

    USER_FEATS = [
        "segment_enc", "budget_enc", "zone_enc", "user_log_orders", "is_cold_start",
        "is_sparse_user", "beverage_order_rate", "dessert_order_rate", "starter_order_rate",
        "offer_redemption_rate", "price_sensitivity", "avg_order_value",
    ]
    ITEM_FEATS = [
        "rtype_enc", "cuisine_enc", "rest_price_norm", "rest_rating_norm",
        "rest_log_orders", "chain_indicator", "cloud_kitchen", "is_new_restaurant",
        "category_enc", "log_price", "rating_norm", "pop_score", "is_bestseller",
        "is_veg", "is_new_item", "is_spicy", "addon_ratio",
    ]
    CTX_FEATS = [
        "cart_item_count", "cart_value_log", "meal_completeness_score",
        "has_main_course", "has_beverage", "has_dessert", "has_starter", "has_bread",
        "missing_beverage", "missing_dessert", "missing_starter", "missing_bread",
        "is_single_item_cart", "meal_time_enc", "season_enc", "weather_enc",
        "hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_festival", "is_weekend",
        "is_peak_hour", "avg_complementarity", "position_bias",
    ]

    def to_tensor(df, feats):
        cols = [c for c in feats if c in df.columns]
        return torch.tensor(df[cols].fillna(0).values.astype(np.float32)), len(cols)

    pos_df  = train[train[LABEL_COL] == 1]
    neg_df  = train[train[LABEL_COL] == 0]
    n_pos   = min(15000, len(pos_df))
    train_b = pd.concat([
        pos_df.sample(n_pos, random_state=42),
        neg_df.sample(n_pos, random_state=42),
    ]).sample(frac=1, random_state=42).reset_index(drop=True)

    Xu_tr, nu = to_tensor(train_b, USER_FEATS)
    Xi_tr, ni = to_tensor(train_b, ITEM_FEATS)
    Xc_tr, nc = to_tensor(train_b, CTX_FEATS)
    y_tr      = torch.tensor(train_b[LABEL_COL].values.astype(np.float32))

    Xu_te, _  = to_tensor(test, USER_FEATS)
    Xi_te, _  = to_tensor(test, ITEM_FEATS)
    Xc_te, _  = to_tensor(test, CTX_FEATS)
    y_te      = test[LABEL_COL].values.astype(int)

    print(f"\n[Neural] Two-Tower NN: {len(train_b):,} balanced samples  "
          f"user_feats={nu}  item_feats={ni}  ctx_feats={nc}")

    model_nn = TwoTowerRankingModel(
        n_user_feats=nu,
        n_item_feats=ni,
        n_ctx_feats=nc,
        emb_dim=64,
        dropout=0.3,
    )

    optimizer = torch.optim.Adam(model_nn.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = nn.BCELoss()
    loader    = DataLoader(
        TensorDataset(Xu_tr, Xi_tr, Xc_tr, y_tr),
        batch_size=512, shuffle=True,
    )

    # LR scheduler will reduce LR by half every 8 epochs for better convergence
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.5)

    N_EPOCHS = 25
    t0 = time.time()
    for epoch in range(N_EPOCHS):
        model_nn.train()
        epoch_loss = 0.0
        for u_b, i_b, c_b, y_b in loader:
            optimizer.zero_grad()
            scores = model_nn(u_b, i_b, c_b)
            loss   = criterion(scores, y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_nn.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()
        print(f"  Epoch {epoch+1:2d}/{N_EPOCHS}  loss={epoch_loss / len(loader):.4f}  lr={scheduler.get_last_lr()[0]:.2e}")
    print(f"  Train time: {time.time() - t0:.1f}s")

    model_nn.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(y_te), 4096):
            end = min(start + 4096, len(y_te))
            preds.append(
                model_nn(Xu_te[start:end], Xi_te[start:end], Xc_te[start:end]).numpy()
            )
    test_scores_nn = np.concatenate(preds)

    results_nn = evaluate_model(y_te, test_scores_nn, "Two-Tower Neural Net")

    # Save model weights
    save_path = os.path.join(MODEL_DIR, "neural_model.pt")
    torch.save(model_nn.state_dict(), save_path)
    print(f"  Saved to models/saved/neural_model.pt")

    return results_nn, test_scores_nn