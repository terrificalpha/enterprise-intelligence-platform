"""Train, tune, compare and select churn models. Writes artifacts/ (model, metrics.json, plots)."""
from __future__ import annotations

import json
import time

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import loguniform, randint, uniform
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline

from ..config import settings
from .features import ID_COL, RAW_FEATURES, TARGET, build_preprocessor, clean_raw, normalise_category

SEED = settings.random_state


def _candidates():
    """(name, pipeline, param distribution). Linear model gets scaling; trees do not need it."""
    return [
        ("logistic_regression",
         Pipeline([("prep", build_preprocessor(scale=True)),
                   ("clf", LogisticRegression(max_iter=2000, solver="lbfgs"))]),
         {"clf__C": loguniform(1e-2, 1e2)}),
        ("random_forest",
         Pipeline([("prep", build_preprocessor(scale=False)),
                   ("clf", RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=SEED))]),
         {"clf__max_depth": randint(4, 14), "clf__min_samples_leaf": randint(3, 25),
          "clf__max_features": uniform(0.2, 0.6)}),
        ("hist_gradient_boosting",
         Pipeline([("prep", build_preprocessor(scale=False)),
                   ("clf", HistGradientBoostingClassifier(random_state=SEED, early_stopping=True))]),
         {"clf__learning_rate": loguniform(0.02, 0.2), "clf__max_depth": randint(2, 6),
          "clf__max_leaf_nodes": randint(6, 31), "clf__l2_regularization": loguniform(1e-3, 10),
          "clf__min_samples_leaf": randint(10, 60)}),
    ]


def _metrics(y, p, thr) -> dict:
    pred = (p >= thr).astype(int)
    return {"roc_auc": roc_auc_score(y, p), "pr_auc": average_precision_score(y, p),
            "f1": f1_score(y, pred), "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred), "accuracy": accuracy_score(y, pred),
            "brier": brier_score_loss(y, p), "threshold": float(thr)}


def _best_f1_threshold(y, p) -> float:
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-9, None)
    return float(thr[int(np.argmax(f1))])


def load_training_frame() -> pd.DataFrame:
    df = clean_raw(pd.read_csv(settings.customers_csv))
    return df


def train(n_iter: int = 12, cv_folds: int = 5, verbose: bool = True) -> dict:
    settings.ensure_dirs()
    t0 = time.time()
    df = load_training_frame()
    X, y = df[RAW_FEATURES], df[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    cv = StratifiedKFold(cv_folds, shuffle=True, random_state=SEED)
    if verbose:
        print(f"Rows: {len(df)} | churn rate {y.mean():.1%} | train {len(X_tr)} / test {len(X_te)}")

    results, fitted = {}, {}
    base = DummyClassifier(strategy="prior").fit(X_tr, y_tr)
    results["baseline_prior"] = {"cv_roc_auc": 0.5, "test": _metrics(y_te, base.predict_proba(X_te)[:, 1], 0.5)}

    for name, pipe, dist in _candidates():
        search = RandomizedSearchCV(pipe, dist, n_iter=n_iter, scoring="roc_auc", cv=cv, random_state=SEED,
                                    n_jobs=-1, refit=True)
        search.fit(X_tr, y_tr)
        oof = cross_val_predict(search.best_estimator_, X_tr, y_tr, cv=cv, method="predict_proba")[:, 1]
        thr = _best_f1_threshold(y_tr, oof)            # threshold chosen on TRAIN out-of-fold preds only
        p_te = search.best_estimator_.predict_proba(X_te)[:, 1]
        results[name] = {"cv_roc_auc": float(search.best_score_),
                         "cv_oof_pr_auc": float(average_precision_score(y_tr, oof)),
                         "best_params": {k: (float(v) if isinstance(v, (np.floating, float)) else
                                             int(v) if isinstance(v, (np.integer,)) else v)
                                         for k, v in search.best_params_.items()},
                         "test": _metrics(y_te, p_te, thr)}
        fitted[name] = search.best_estimator_
        if verbose:
            r = results[name]
            print(f"{name:24s} CV AUC {r['cv_roc_auc']:.4f} | test AUC {r['test']['roc_auc']:.4f} "
                  f"PR-AUC {r['test']['pr_auc']:.4f} F1 {r['test']['f1']:.4f}")

    # ---- model selection: CV ROC-AUC on the training set only (test set stays untouched) ----
    best_name = max(fitted, key=lambda n: results[n]["cv_roc_auc"])
    best_pipe = fitted[best_name]
    # ---- calibrate probabilities (risk tiers use absolute probability thresholds) ----
    cal = CalibratedClassifierCV(best_pipe, method="sigmoid", cv=5).fit(X_tr, y_tr)
    oof_cal = cross_val_predict(CalibratedClassifierCV(best_pipe, method="sigmoid", cv=5), X_tr, y_tr, cv=cv,
                                method="predict_proba")[:, 1]
    thr_cal = _best_f1_threshold(y_tr, oof_cal)
    p_raw = best_pipe.predict_proba(X_te)[:, 1]
    p_cal = cal.predict_proba(X_te)[:, 1]
    final = _metrics(y_te, p_cal, thr_cal)
    results["selected"] = {"model": best_name, "selection_metric": "cv_roc_auc", "calibrated": True,
                           "test_calibrated": final, "test_uncalibrated_brier": float(brier_score_loss(y_te, p_raw)),
                           "threshold": thr_cal,
                           "confusion_matrix": confusion_matrix(y_te, (p_cal >= thr_cal).astype(int)).tolist()}
    if verbose:
        print(f"\nSELECTED: {best_name} | calibrated test AUC {final['roc_auc']:.4f} F1 {final['f1']:.4f} "
              f"Brier {final['brier']:.4f} (uncalibrated {results['selected']['test_uncalibrated_brier']:.4f})")

    # ---- permutation importance on RAW features (interpretable for business users) ----
    pi = permutation_importance(best_pipe, X_te, y_te, scoring="roc_auc", n_repeats=8, random_state=SEED, n_jobs=-1)
    imp = sorted(zip(RAW_FEATURES, pi.importances_mean, pi.importances_std), key=lambda t: -t[1])
    results["feature_importance"] = [{"feature": f, "importance": float(m), "std": float(s)} for f, m, s in imp]

    # ---- persist ----
    art = settings.artifacts_dir
    medians = X_tr.select_dtypes("number").median().to_dict()
    modes = {c: normalise_category(X_tr[c]).mode().iloc[0] for c in
             ["industry", "region", "company_size", "contract_type"]}
    joblib.dump({"model": cal, "model_name": best_name, "threshold": thr_cal, "medians": medians, "modes": modes,
                 "features": RAW_FEATURES, "version": time.strftime("%Y%m%d-%H%M%S")}, art / "best_model.joblib")
    for n, m in fitted.items():
        joblib.dump(m, art / f"candidate_{n}.joblib")
    results["meta"] = {"n_rows": int(len(df)), "churn_rate": float(y.mean()), "n_train": int(len(X_tr)),
                       "n_test": int(len(X_te)), "cv_folds": cv_folds, "search_iters": n_iter, "seed": SEED,
                       "train_seconds": round(time.time() - t0, 1)}
    (art / "metrics.json").write_text(json.dumps(results, indent=2, default=float))
    _plots(fitted, cal, X_te, y_te, p_cal, imp, art)
    _score_portfolio(cal, thr_cal)
    return results


def _plots(fitted, cal, X_te, y_te, p_cal, imp, art) -> None:
    fig_dir = art / "figures"
    fig_dir.mkdir(exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for n, m in fitted.items():
        p = m.predict_proba(X_te)[:, 1]
        fpr, tpr, _ = roc_curve(y_te, p)
        ax[0].plot(fpr, tpr, label=f"{n} ({roc_auc_score(y_te, p):.3f})")
        pr, rc, _ = precision_recall_curve(y_te, p)
        ax[1].plot(rc, pr, label=f"{n} ({average_precision_score(y_te, p):.3f})")
    ax[0].plot([0, 1], [0, 1], "k--", lw=0.8)
    ax[0].set(title="ROC curves (test)", xlabel="False positive rate", ylabel="True positive rate")
    ax[1].set(title="Precision-Recall (test)", xlabel="Recall", ylabel="Precision")
    for a in ax:
        a.legend(fontsize=8)
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "roc_pr.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4.2))
    top = imp[:12][::-1]
    ax.barh([t[0] for t in top], [t[1] for t in top], xerr=[t[2] for t in top], color="#2b6cb0")
    ax.set(title="Permutation importance (drop in ROC-AUC)", xlabel="Importance")
    fig.tight_layout()
    fig.savefig(fig_dir / "feature_importance.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4.2))
    frac, mean = calibration_curve(y_te, p_cal, n_bins=8, strategy="quantile")
    ax.plot(mean, frac, "o-", label="calibrated model")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="perfect")
    ax.set(title="Calibration (test)", xlabel="Predicted probability", ylabel="Observed churn rate")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "calibration.png", dpi=130)
    plt.close(fig)


def _score_portfolio(cal, thr) -> None:
    """Pre-compute churn probabilities for the whole portfolio (used by the analytics tools)."""
    df = load_training_frame()
    out = df[[ID_COL]].copy()
    out["churn_probability"] = cal.predict_proba(df[RAW_FEATURES])[:, 1]
    out.to_csv(settings.artifacts_dir / "customer_scores.csv", index=False)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-iter", type=int, default=12)
    a = ap.parse_args()
    train(n_iter=a.n_iter)
