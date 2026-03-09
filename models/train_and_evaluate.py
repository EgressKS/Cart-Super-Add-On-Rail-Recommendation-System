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