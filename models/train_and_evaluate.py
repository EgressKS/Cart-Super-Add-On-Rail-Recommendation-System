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


# SEGMENT-WISE EVALUATION
def segment_wise_evaluation(test: pd.DataFrame, y_scores: np.ndarray):
    """
    Evaluate model on important sub-groups to check fairness and coverage.
    """
    print("\nSegment-wise evaluation …")
    test = test.copy()
    test["pred_score"] = y_scores
    y_true = test[LABEL_COL].values

    segments_config = {
        "user_segment":   ["segment_enc"],
        "meal_time":      ["meal_time_enc"],
        "restaurant_type":["rtype_enc"],
        "city_tier":      ["city_tier_enc"],
        "cold_start_user":["is_cold_start"],
        "budget_tier":    ["budget_enc"],
    }

    all_results = {}
    for seg_name, cols in segments_config.items():
        col = cols[0]
        if col not in test.columns:
            continue
        results_for_seg = {}
        for val in test[col].unique():
            mask = test[col] == val
            if mask.sum() < 10:
                continue
            seg_y_true   = test.loc[mask, LABEL_COL].values
            seg_y_scores = test.loc[mask, "pred_score"].values
            if seg_y_true.sum() == 0:
                continue
            r = evaluate_model(seg_y_true, seg_y_scores, f"{seg_name}={val}")
            results_for_seg[str(val)] = r
        all_results[seg_name] = results_for_seg

    return all_results


# COLD-START EVALUATION
def cold_start_evaluation(test: pd.DataFrame, y_scores: np.ndarray):
    """Evaluate specifically on cold-start scenarios."""
    test = test.copy()
    test["pred_score"] = y_scores
    results = {}

    # New users (is_cold_start = 1)
    if "is_cold_start" in test.columns:
        mask = test["is_cold_start"] == 1
        if mask.sum() > 10:
            r = evaluate_model(
                test.loc[mask, LABEL_COL].values,
                test.loc[mask, "pred_score"].values,
                "Cold Start Users"
            )
            results["cold_start_users"] = r
            print(f"  Cold-start users: AUC={r['auc_roc']:.4f}  NDCG@10={r['ndcg@10']:.4f}")

    # New restaurants
    if "is_new_restaurant" in test.columns:
        mask = test["is_new_restaurant"] == True
        if mask.sum() > 10:
            r = evaluate_model(
                test.loc[mask, LABEL_COL].values,
                test.loc[mask, "pred_score"].values,
                "New Restaurants"
            )
            results["new_restaurants"] = r

    # New items
    if "is_new_item" in test.columns:
        mask = test["is_new_item"] == 1
        if mask.sum() > 10:
            r = evaluate_model(
                test.loc[mask, LABEL_COL].values,
                test.loc[mask, "pred_score"].values,
                "New Items"
            )
            results["new_items"] = r

    return results


# ERROR ANALYSIS
def error_analysis(test: pd.DataFrame, y_scores: np.ndarray, threshold=0.5):
    """
    Analyse false positives and false negatives to understand failure modes.
    """
    test = test.copy()
    test["pred_score"]  = y_scores
    test["pred_label"]  = (y_scores >= threshold).astype(int)
    test["true_label"]  = test[LABEL_COL]

    fp = test[(test["pred_label"] == 1) & (test["true_label"] == 0)]
    fn = test[(test["pred_label"] == 0) & (test["true_label"] == 1)]
    tp = test[(test["pred_label"] == 1) & (test["true_label"] == 1)]

    analysis = {
        "total_positives":       int(test["true_label"].sum()),
        "total_negatives":       int((test["true_label"] == 0).sum()),
        "true_positives":        len(tp),
        "false_positives":       len(fp),
        "false_negatives":       len(fn),
        "fp_avg_price":          float(fp["log_price"].mean()) if "log_price" in fp.columns else 0.0,
        "fn_avg_complementarity":float(fn["avg_complementarity"].mean()) if "avg_complementarity" in fn.columns else 0.0,
    }

    # Top features where FP occurs
    if "category_enc" in fp.columns:
        analysis["fp_category_dist"] = fp["category_enc"].value_counts().to_dict()
    if "meal_time_enc" in fp.columns:
        analysis["fp_meal_time_dist"] = fp["meal_time_enc"].value_counts().to_dict()
    if "is_cold_start" in fp.columns:
        analysis["fp_cold_start_rate"] = float(fp["is_cold_start"].mean())

    # FN analysis missed true complements
    if "avg_complementarity" in fn.columns:
        analysis["fn_high_comp_missed"] = int((fn["avg_complementarity"] > 0.5).sum())

    return analysis


# FEATURE IMPORTANCE
def compute_feature_importance(gbm_model, feature_cols: list) -> pd.DataFrame:
    """Return sorted feature importance from trained GBM."""
    fi = gbm_model.feature_importance()
    df = pd.DataFrame(list(fi.items()), columns=["feature","importance"])
    df = df.sort_values("importance", ascending=False).reset_index(drop=True)
    return df


# VISUALIZATIONS
def plot_results(
    baseline_results:  list,
    gbm_results:       dict,
    segment_results:   dict,
    feature_importance: pd.DataFrame,
    error_analysis:    dict,
):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Zomato CSAO - Model Evaluation Dashboard", fontsize=14, fontweight="bold")

    # 1. Baseline Comparison 
    ax = axes[0, 0]
    all_models = baseline_results + [gbm_results]
    model_names = [r["model"] for r in all_models]
    auc_vals    = [r["auc_roc"] for r in all_models]
    ndcg_vals   = [r["ndcg@10"] for r in all_models]

    x = np.arange(len(model_names))
    ax.bar(x - 0.2, auc_vals,  0.35, label="AUC-ROC", color="steelblue", alpha=0.8)
    ax.bar(x + 0.2, ndcg_vals, 0.35, label="NDCG@10", color="coral",     alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=20, ha="right", fontsize=8)
    ax.set_ylim(0, 1.0)
    ax.axhline(0.85, color="green", linestyle="--", label="AUC Target", alpha=0.7)
    ax.legend(fontsize=8)
    ax.set_title("Model Comparison")
    ax.set_ylabel("Score")

    # 2. Precision-Recall @ K 
    ax = axes[0, 1]
    k_vals  = [1, 3, 5, 10]
    metrics = {
        "precision@K": [gbm_results.get(f"precision@{k}", gbm_results.get("precision@10",0)) for k in k_vals],
        "recall@K":    [gbm_results.get(f"recall@{k}",    gbm_results.get("recall@10",0))    for k in k_vals],
    }
    for name, vals in metrics.items():
        ax.plot(k_vals, vals, marker="o", label=name)
    ax.set_xlabel("K")
    ax.set_ylabel("Score")
    ax.set_title("Precision & Recall @ K")
    ax.legend(fontsize=9)
    ax.set_xticks(k_vals)

    # 3. Feature Importance
    ax = axes[0, 2]
    top15 = feature_importance.head(15)
    ax.barh(top15["feature"], top15["importance"], color="mediumpurple", alpha=0.8)
    ax.invert_yaxis()
    ax.set_title("Top-15 Feature Importance (GBM)")
    ax.set_xlabel("Importance")
    ax.tick_params(axis="y", labelsize=7)

    # 4. Segment-wise AUC 
    ax = axes[1, 0]
    if "user_segment" in segment_results:
        seg_data = segment_results["user_segment"]
        seg_labels = list(seg_data.keys())
        seg_aucs   = [seg_data[s]["auc_roc"] for s in seg_labels]
        colors = ["#ff9999" if v < 0.75 else "#99dd99" for v in seg_aucs]
        ax.bar(seg_labels, seg_aucs, color=colors)
        ax.axhline(0.75, color="red", linestyle="--", label="Min AUC", alpha=0.7)
        ax.set_title("AUC by User Segment")
        ax.set_ylabel("AUC-ROC")
        ax.set_ylim(0.5, 1.0)
        ax.legend(fontsize=8)

    # 5. Meal-time NDCG
    ax = axes[1, 1]
    mtime_names = {0:"Breakfast",1:"Lunch",2:"Snack",3:"Dinner",4:"Late Night"}
    if "meal_time" in segment_results:
        mt_data = segment_results["meal_time"]
        labels  = [mtime_names.get(int(k), k) for k in mt_data.keys()]
        ndcgs   = [mt_data[k]["ndcg@10"] for k in mt_data.keys()]
        ax.bar(labels, ndcgs, color="teal", alpha=0.8)
        ax.set_title("NDCG@10 by Meal Time")
        ax.set_ylabel("NDCG@10")
        ax.set_ylim(0, 1.0)

    # 6. Error Analysis
    ax = axes[1, 2]
    labels = ["True Positives","False Positives","False Negatives"]
    values = [
        error_analysis.get("true_positives",0),
        error_analysis.get("false_positives",0),
        error_analysis.get("false_negatives",0),
    ]
    colors_err = ["#66bb6a","#ef5350","#ffa726"]
    ax.pie(values, labels=labels, colors=colors_err, autopct="%1.1f%%", startangle=90)
    ax.set_title("Prediction Breakdown")

    plt.tight_layout()
    path = os.path.join(REPORT_DIR, "evaluation_dashboard.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Dashboard saved to {path}")



def main():
    print("=" * 60)
    print("MODEL TRAINING & EVALUATION")
    print("=" * 60)

    # 1. Load data
    train, val, test = load_splits()
    X_te, y_te, fc   = get_Xy(test)

    # 2. Baselines
    print("\nTraining baselines …")
    pop_res  = train_popularity_baseline(train, test)
    rule_res = train_rule_based_baseline(train, test)
    lr_res   = train_logistic_baseline(train, val, test)

    for r in [pop_res, rule_res, lr_res]:
        print(f"  [{r['model']:25s}] AUC={r['auc_roc']:.4f}  NDCG@10={r['ndcg@10']:.4f}  P@10={r['precision@10']:.4f}")

    # 3. GBM model
    gbm_res, gbm_model, test_scores, y_te_arr, feat_cols = train_gbm_model(train, val, test)
    print(f"\n  [GBM Ranking Model]")
    for k, v in gbm_res.items():
        if k != "model":
            print(f"    {k:20s}: {v}")

    # 3b. Two-Tower Neural Network
    print("\nTraining Two-Tower Neural Network ...")
    nn_res, nn_scores = train_neural_model(train, val, test)
    if nn_res:
        print(f"\n  [Two-Tower Neural Net]")
        for k, v in nn_res.items():
            if k != "model":
                print(f"    {k:20s}: {v}")

    # 4. Segment-wise evaluation
    best_scores = nn_scores if (nn_res and nn_res["auc_roc"] > gbm_res["auc_roc"]) else test_scores
    seg_results = segment_wise_evaluation(test, best_scores)

    # 5. Cold-start evaluation
    print("\nCold-start evaluation …")
    cs_results = cold_start_evaluation(test, best_scores)
    for name, r in cs_results.items():
        print(f"  {name}: AUC={r['auc_roc']:.4f}  NDCG@10={r['ndcg@10']:.4f}")

    # 6. Error analysis
    err_analysis = error_analysis(test, best_scores)
    print(f"\nError analysis:")
    print(f"  TP={err_analysis['true_positives']}  "
          f"FP={err_analysis['false_positives']}  "
          f"FN={err_analysis['false_negatives']}")

    # 7. Feature importance
    fi_df = compute_feature_importance(gbm_model, feat_cols)
    fi_df.to_csv(os.path.join(REPORT_DIR, "feature_importance.csv"), index=False)
    print(f"\nTop-10 features:")
    print(fi_df.head(10).to_string(index=False))

    # 8. Visualizations
    print("\nGenerating evaluation dashboard …")
    extra_baselines = [nn_res] if nn_res else []
    plot_results(
        baseline_results  = [pop_res, rule_res, lr_res] + extra_baselines,
        gbm_results       = gbm_res,
        segment_results   = seg_results,
        feature_importance= fi_df,
        error_analysis    = err_analysis,
    )

    # 9. Improvement summary vs baseline
    auc_lift_vs_rule = gbm_res["auc_roc"] - rule_res["auc_roc"]
    ndcg_lift        = gbm_res["ndcg@10"] - rule_res["ndcg@10"]
    print(f"\nGBM vs Rule-Based:        AUC lift={auc_lift_vs_rule:+.4f}  NDCG@10 lift={ndcg_lift:+.4f}")
    if nn_res:
        nn_auc_lift  = nn_res["auc_roc"] - rule_res["auc_roc"]
        nn_ndcg_lift = nn_res["ndcg@10"] - rule_res["ndcg@10"]
        print(f"Neural vs Rule-Based:     AUC lift={nn_auc_lift:+.4f}  NDCG@10 lift={nn_ndcg_lift:+.4f}")
        best_model = "Two-Tower Neural Net" if nn_res["auc_roc"] > gbm_res["auc_roc"] else "GBM Ranking Model"
        print(f"Best model: {best_model}")

    # 10. Save comprehensive report
    models_dict = {
        "popularity_baseline": pop_res,
        "rule_based_baseline": rule_res,
        "logistic_regression": lr_res,
        "gbm_ranking_model":   gbm_res,
    }
    if nn_res:
        models_dict["two_tower_neural"] = nn_res

    report = {
        "models": models_dict,
        "segment_wise": seg_results,
        "cold_start":   cs_results,
        "error_analysis": err_analysis,
        "feature_importance_top10": fi_df.head(10).to_dict("records"),
        "baseline_comparison": {
            "auc_lift_vs_rule_based":  round(auc_lift_vs_rule, 4),
            "ndcg_lift_vs_rule_based": round(ndcg_lift, 4),
            "auc_lift_vs_popularity":  round(gbm_res["auc_roc"] - pop_res["auc_roc"], 4),
            **({"neural_auc_lift_vs_rule": round(nn_auc_lift, 4)} if nn_res else {}),
        }
    }

    with open(os.path.join(REPORT_DIR, "evaluation_report.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\nEvaluation report saved to models/reports/evaluation_report.json")
    print("Training & evaluation complete!")
    return report


if __name__ == "__main__":
    main()