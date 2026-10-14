"""50 Startups 利潤預測：多元線性迴歸（CRISP-DM）

依 CRISP-DM 六個階段執行完整流程：
    python main.py
執行後產出：
    output/index.html             CRISP-DM HTML 報告（發布到 GitHub Pages 的 /report/）
    output/*.png                  圖表
    startup_profit_model.pkl      最終模型（含特徵清單）
    site/results.json             互動式網站資料：5 種特徵選擇方法比較（網頁不重算模型）
"""

import html
import json
import sys
import warnings
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 無視窗環境（GitHub Actions）也能存圖

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
import statsmodels
import statsmodels.api as sm
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.feature_selection import RFE, SequentialFeatureSelector
from sklearn.linear_model import LinearRegression, enet_path, lasso_path
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import RepeatedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

warnings.filterwarnings("ignore", category=FutureWarning)
sns.set_theme(style="whitegrid")

RANDOM_STATE = 42
DATA_URL = "https://raw.githubusercontent.com/Avik-Jain/100-Days-Of-ML-Code/master/datasets/50_Startups.csv"
DATA_PATH = Path("50_Startups.csv")
MODEL_PATH = "startup_profit_model.pkl"
OUT_DIR = Path("output")

NUM_FEATURES = ["R&D Spend", "Administration", "Marketing Spend"]
CAT_FEATURE = "State"
TARGET = "Profit"
ALL_FEATURES = NUM_FEATURES + [CAT_FEATURE]

# 互動式網站（特徵選擇方法比較）設定：State one-hot drop first（California 為基準組）+ 截距
SITE_RESULTS_PATH = Path("site/results.json")
STATES = ["California", "Florida", "New York"]
SITE_FEATURES = ["R&D Spend", "Administration", "Marketing Spend", "State_Florida", "State_New York"]
ELASTICNET_L1_RATIO = 0.5
SITE_CV = RepeatedKFold(n_splits=5, n_repeats=10, random_state=RANDOM_STATE)
SITE_SCORING = {"rmse": "neg_root_mean_squared_error", "r2": "r2"}


# ---------------------------------------------------------------------------
# 報告輔助：同時輸出到終端機與 HTML
# ---------------------------------------------------------------------------
class Report:
    def __init__(self):
        self.parts = []

    def h2(self, text):
        print(f"\n{'=' * 70}\n{text}\n{'=' * 70}")
        self.parts.append(f"<h2>{html.escape(text)}</h2>")

    def h3(self, text):
        print(f"\n--- {text} ---")
        self.parts.append(f"<h3>{html.escape(text)}</h3>")

    def p(self, text):
        print(text)
        self.parts.append(f"<p>{html.escape(text)}</p>")

    def table(self, df, floatfmt="{:,.4f}"):
        print(df.to_string(float_format=lambda x: floatfmt.format(x)))
        self.parts.append(df.to_html(float_format=lambda x: floatfmt.format(x), border=0, classes="tbl"))

    def fig(self, name, caption=""):
        path = OUT_DIR / name
        plt.tight_layout()
        plt.savefig(path, dpi=110)
        plt.close()
        print(f"[圖] {path}")
        self.parts.append(f'<figure><img src="{name}" alt="{html.escape(caption)}"></figure>')

    def save(self, path):
        css = """
        body{font-family:system-ui,"Noto Sans TC","Microsoft JhengHei",sans-serif;max-width:1000px;
             margin:0 auto;padding:24px 16px;line-height:1.6;color:#222;background:#fff}
        h1{border-bottom:3px solid #2b6cb0;padding-bottom:8px}
        h2{margin-top:40px;color:#2b6cb0;border-bottom:1px solid #ddd}
        .tbl{border-collapse:collapse;font-size:14px;margin:8px 0;display:block;overflow-x:auto}
        .tbl th,.tbl td{padding:4px 10px;border-bottom:1px solid #eee;text-align:right}
        .tbl th{background:#f3f6fa}
        img{max-width:100%}
        """
        doc = (
            "<!doctype html><html lang='zh-Hant'><head><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>50 Startups CRISP-DM</title><style>{css}</style></head><body>"
            "<h1>50 Startups 利潤預測：多元線性迴歸（CRISP-DM）</h1>"
            + "\n".join(self.parts)
            + "</body></html>"
        )
        path.write_text(doc, encoding="utf-8")
        print(f"\n報告已輸出：{path}")


def build_model(features):
    """依特徵組合建立「前處理 + 線性迴歸」Pipeline；State 視為一個整體（one-hot 三欄同進同出）。"""
    features = list(features)
    num_cols = [f for f in features if f in NUM_FEATURES]
    use_state = CAT_FEATURE in features

    transformers = []
    if num_cols:
        transformers.append(("num", "passthrough", num_cols))
    if use_state:
        transformers.append((
            "cat",
            OneHotEncoder(drop=None, categories=[["California", "Florida", "New York"]],
                          handle_unknown="ignore", sparse_output=False),
            [CAT_FEATURE],
        ))

    prep = ColumnTransformer(transformers, verbose_feature_names_out=False)
    # 含 State（完整 one-hot）時不設截距，避免 dummy variable trap
    # tol=1e-12：sklearn 預設 tol=1e-6 會截斷小奇異值，未標準化的金額欄位 + one-hot 會被誤截
    reg = LinearRegression(fit_intercept=not use_state, tol=1e-12)
    return Pipeline([("prep", prep), ("reg", reg)])


def compute_vif(data, cols):
    """以 sklearn 自行計算 VIF：VIF_j = 1 / (1 - R²_j)。"""
    out = {}
    for c in cols:
        others = [o for o in cols if o != c]
        r2 = LinearRegression().fit(data[others], data[c]).score(data[others], data[c])
        out[c] = 1 / (1 - r2)
    return pd.Series(out, name="VIF").to_frame()


# ===========================================================================
# 互動式網站：5 種特徵選擇方法比較（輸出 site/results.json）
# ===========================================================================
# ---------------------------------------------------------------------------
# 資料載入與前處理
# ---------------------------------------------------------------------------
def load_data():
    # 優先讀取本機檔案；沒有時從原始來源下載
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)
    df = pd.read_csv(DATA_URL)
    df.to_csv(DATA_PATH, index=False)
    return df


def encode_drop_first(df):
    # State one-hot，drop first（California 為基準組）
    state = pd.Categorical(df["State"], categories=STATES)
    dummies = pd.get_dummies(state, prefix="State", drop_first=True, dtype=float)
    dummies.index = df.index
    X = pd.concat([df[["R&D Spend", "Administration", "Marketing Spend"]], dummies], axis=1)
    return X[SITE_FEATURES]


# ---------------------------------------------------------------------------
# CV 評估（只用訓練集）
# ---------------------------------------------------------------------------
_cv_cache = {}


def cv_score(X_train, y_train, feats):
    key = frozenset(feats)
    if key not in _cv_cache:
        res = cross_validate(LinearRegression(), X_train[list(feats)], y_train, cv=SITE_CV, scoring=SITE_SCORING)
        rmse = -res["test_rmse"]
        _cv_cache[key] = {
            "rmse_mean": float(rmse.mean()), "rmse_std": float(rmse.std()),
            "r2_mean": float(res["test_r2"].mean()), "r2_std": float(res["test_r2"].std()),
        }
    return _cv_cache[key]


# ---------------------------------------------------------------------------
# 5 種特徵選擇方法：回傳 {k: [features]}
# ---------------------------------------------------------------------------
def path_entry_order(X_std, y_centered, path_fn, **kw):
    # 依正則化路徑上「係數第一次變成非 0」的順序排序特徵（alpha 由大到小）
    alphas, coefs, _ = path_fn(X_std, y_centered, eps=1e-5, alphas=1000, **kw)
    nonzero = np.abs(coefs) > 1e-10
    entry = [int(np.argmax(row)) if row.any() else len(alphas) for row in nonzero]
    final_abs = np.abs(coefs[:, -1])
    order = sorted(range(len(SITE_FEATURES)), key=lambda j: (entry[j], -final_abs[j]))
    detail = [{"feature": SITE_FEATURES[j],
               "entry_alpha": float(alphas[entry[j]]) if entry[j] < len(alphas) else None}
              for j in order]
    return [SITE_FEATURES[j] for j in order], detail


def select_lasso(X_std, y_centered):
    order, detail = path_entry_order(X_std, y_centered, lasso_path)
    return {k: order[:k] for k in range(1, 6)}, detail


def select_elasticnet(X_std, y_centered):
    order, detail = path_entry_order(X_std, y_centered, enet_path, l1_ratio=ELASTICNET_L1_RATIO)
    return {k: order[:k] for k in range(1, 6)}, detail


def select_forward_sfs(X_train, y_train):
    out = {}
    for k in range(1, 6):
        if k == len(SITE_FEATURES):
            out[k] = list(SITE_FEATURES)  # SFS 要求 k < 特徵數；k = 5 即全部特徵
            continue
        sfs = SequentialFeatureSelector(
            LinearRegression(), n_features_to_select=k, direction="forward",
            scoring="neg_root_mean_squared_error", cv=SITE_CV,
        ).fit(X_train, y_train)
        out[k] = [f for f, keep in zip(SITE_FEATURES, sfs.get_support()) if keep]
    return out


def select_rfe(X_std, y_train):
    out = {}
    for k in range(1, 6):
        rfe = RFE(LinearRegression(), n_features_to_select=k).fit(X_std, y_train)
        out[k] = [f for f, keep in zip(SITE_FEATURES, rfe.support_) if keep]
    return out


def select_exhaustive(X_train, y_train):
    out, tables = {}, {}
    for k in range(1, 6):
        rows = []
        for combo in combinations(SITE_FEATURES, k):
            s = cv_score(X_train, y_train, combo)
            rows.append({"features": list(combo), **s})
        rows.sort(key=lambda r: r["rmse_mean"])
        out[k] = rows[0]["features"]
        tables[k] = rows
    return out, tables


# ---------------------------------------------------------------------------
# 單一「方法 × k」的完整評估（statsmodels OLS on train）
# ---------------------------------------------------------------------------
def evaluate_subset(feats, X_train, y_train, X_test, y_test, df):
    feats = [f for f in SITE_FEATURES if f in feats]  # 固定欄位順序
    Xtr = sm.add_constant(X_train[feats], has_constant="add")
    Xte = sm.add_constant(X_test[feats], has_constant="add")
    ols = sm.OLS(y_train, Xtr).fit()
    ci = ols.conf_int(alpha=0.05)

    coefs = []
    for i, name in enumerate(Xtr.columns):
        vif = None if name == "const" else float(variance_inflation_factor(Xtr.values, i))
        coefs.append({
            "name": "Intercept" if name == "const" else name,
            "estimate": float(ols.params[name]),
            "std_error": float(ols.bse[name]),
            "t_stat": float(ols.tvalues[name]),
            "p_value": float(ols.pvalues[name]),
            "ci_low": float(ci.loc[name, 0]),
            "ci_high": float(ci.loc[name, 1]),
            "vif": vif,
        })

    pred = ols.predict(Xte)
    resid = y_test - pred
    test_rows = [{
        "sample_id": int(i),
        "state": df.loc[i, "State"],
        "actual": float(y_test[i]),
        "predicted": float(pred[i]),
        "residual": float(resid[i]),
        "pct_error": float(resid[i] / y_test[i] * 100),
    } for i in y_test.index]

    return {
        "features": feats,
        "coefficients": coefs,
        "train": {
            "r2": float(ols.rsquared), "adj_r2": float(ols.rsquared_adj),
            "f_stat": float(ols.fvalue), "f_pvalue": float(ols.f_pvalue),
            "n": int(ols.nobs),
        },
        "cv": cv_score(X_train, y_train, feats),
        "test": {
            "rmse": float(root_mean_squared_error(y_test, pred)),
            "r2": float(r2_score(y_test, pred)),
            "mae": float(mean_absolute_error(y_test, pred)),
            "rows": test_rows,
        },
    }


def export_site_results():
    """5 種特徵選擇方法 × k = 1～5，輸出網站用的 site/results.json。"""
    # ---------------- Data Understanding / Preparation ----------------
    df = load_data()
    X = encode_drop_first(df)
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=df["State"]
    )

    # 標準化只用訓練集配適（Lasso / ElasticNet / RFE 使用）
    scaler = StandardScaler().fit(X_train)
    X_std = scaler.transform(X_train)
    y_centered = (y_train - y_train.mean()).to_numpy()

    # ---------------- Modeling：5 種特徵選擇 ----------------
    lasso_sel, lasso_order = select_lasso(X_std, y_centered)
    enet_sel, enet_order = select_elasticnet(X_std, y_centered)
    exh_sel, exh_tables = select_exhaustive(X_train, y_train)
    selections = {
        "lasso": lasso_sel,
        "forward": select_forward_sfs(X_train, y_train),
        "rfe": select_rfe(X_std, y_train),
        "elasticnet": enet_sel,
        "exhaustive": exh_sel,
    }
    method_info = {
        "lasso": {"name": "Lasso", "tag": "L1 Path",
                  "description": "標準化後計算 Lasso 係數路徑，依特徵進入模型（係數第一次非 0）的順序排序，取前 k 個。"},
        "forward": {"name": "Forward SFS", "tag": "Greedy",
                    "description": "SequentialFeatureSelector（direction='forward'），每一步加入讓 CV RMSE 最低的特徵。"},
        "rfe": {"name": "RFE", "tag": "Backward",
                "description": "Recursive Feature Elimination：標準化後反覆移除 |係數| 最小的特徵，直到剩 k 個。"},
        "elasticnet": {"name": "ElasticNet", "tag": "L1 + L2",
                       "description": f"標準化後計算 ElasticNet（l1_ratio={ELASTICNET_L1_RATIO}）係數路徑，依特徵進入順序排序，取前 k 個。"},
        "exhaustive": {"name": "Exhaustive Search", "tag": "Best Subset",
                       "description": "窮舉所有大小為 k 的特徵組合，以訓練集 CV RMSE 選最佳，作為 benchmark。"},
    }

    # ---------------- Evaluation ----------------
    methods = []
    for mid, sel in selections.items():
        results = {}
        for k in range(1, 6):
            r = evaluate_subset(sel[k], X_train, y_train, X_test, y_test, df)
            r["matches_benchmark"] = set(sel[k]) == set(exh_sel[k])
            results[str(k)] = r
        best_k = min(range(1, 6), key=lambda k: (results[str(k)]["cv"]["rmse_mean"], k))
        n_match = sum(results[str(k)]["matches_benchmark"] for k in range(1, 6))
        entry = {"id": mid, **method_info[mid], "best_k": best_k,
                 "matches_benchmark_count": n_match,
                 "matches_benchmark_all": n_match == 5, "results": results}
        if mid == "lasso":
            entry["path_order"] = lasso_order
        if mid == "elasticnet":
            entry["path_order"] = enet_order
        methods.append(entry)

    # 全域最佳：所有方法 × k 中 CV RMSE 最低（同分取較小 k）
    best = min(((m, k) for m in methods for k in range(1, 6)),
               key=lambda mk: (mk[0]["results"][str(mk[1])]["cv"]["rmse_mean"], mk[1]))
    best_res = best[0]["results"][str(best[1])]
    best_feats = set(best_res["features"])
    global_best = {
        "k": best[1],
        "features": best_res["features"],
        "methods": [m["id"] for m in methods if set(m["results"][str(best[1])]["features"]) == best_feats],
        "cv_rmse_mean": best_res["cv"]["rmse_mean"],
        "cv_rmse_std": best_res["cv"]["rmse_std"],
        "test_rmse": best_res["test"]["rmse"],
    }

    out = {
        "meta": {
            "title": "50 Startups Multiple Linear Regression Explorer",
            "dataset_url": DATA_URL,
            "n_rows": int(len(df)), "n_train": int(len(X_train)), "n_test": int(len(X_test)),
            "target": TARGET,
            "features": SITE_FEATURES,
            "baseline_state": "California",
            "split": "80/20, random_state=42, stratify=State",
            "cv": "RepeatedKFold(n_splits=5, n_repeats=10, random_state=42)",
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "versions": {"python": sys.version.split()[0], "scikit-learn": sklearn.__version__,
                         "statsmodels": statsmodels.__version__, "pandas": pd.__version__},
            "input_defaults": {c: float(df[c].median()) for c in ["R&D Spend", "Administration", "Marketing Spend"]},
            "input_ranges": {c: [float(df[c].min()), float(df[c].max())]
                             for c in ["R&D Spend", "Administration", "Marketing Spend"]},
        },
        "global_best": global_best,
        "methods": methods,
        "exhaustive_tables": {str(k): v for k, v in exh_tables.items()},
    }
    SITE_RESULTS_PATH.parent.mkdir(exist_ok=True)
    SITE_RESULTS_PATH.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")

    # ---------------- 終端機摘要 ----------------
    print(f"\n{'=' * 70}\n特徵選擇方法比較（網站資料）\n{'=' * 70}")
    print("各方法在 k = 1～5 選到的特徵：")
    for m in methods:
        print(f"\n[{m['name']}]  best k = {m['best_k']}，與 Exhaustive 一致 {m['matches_benchmark_count']}/5")
        for k in range(1, 6):
            r = m["results"][str(k)]
            flag = "✓" if r["matches_benchmark"] else "✗"
            print(f"  k={k} {flag} CV RMSE {r['cv']['rmse_mean']:>10,.2f} ± {r['cv']['rmse_std']:>8,.2f} | "
                  f"Test RMSE {r['test']['rmse']:>10,.2f} | {', '.join(r['features'])}")
    print(f"\n全域最佳（CV RMSE）：k = {global_best['k']}，{global_best['features']}，"
          f"CV RMSE = {global_best['cv_rmse_mean']:,.2f} ± {global_best['cv_rmse_std']:,.2f}，"
          f"Test RMSE = {global_best['test_rmse']:,.2f}，方法：{global_best['methods']}")
    print(f"\n已輸出：{SITE_RESULTS_PATH}")



def main():
    sys.stdout.reconfigure(encoding="utf-8")  # Windows 終端機也能正確輸出中文與符號
    OUT_DIR.mkdir(exist_ok=True)
    rpt = Report()

    # =======================================================================
    # Phase 1. Business Understanding
    # =======================================================================
    rpt.h2("Phase 1. Business Understanding（商業理解）")
    rpt.p("商業問題：新創公司在研發、行政管理、行銷上的支出以及所在州別，與公司利潤（Profit）有什麼關係？"
          "能否依一家新公司的支出結構預估其利潤，作為投資評估參考？")
    rpt.p("分析目標：建立多元線性迴歸模型預測 Profit；以交叉驗證做特徵選擇；檢查迴歸假設。")
    rpt.p("成功標準：CV 與測試集 RMSE 明顯優於 baseline（以平均值預測）；模型簡潔可解讀（one-standard-error rule）。")

    # =======================================================================
    # Phase 2. Data Understanding
    # =======================================================================
    rpt.h2("Phase 2. Data Understanding（資料理解）")
    df = load_data()
    rpt.p(f"shape：{df.shape}")
    rpt.h3("欄位型態與非空值筆數（info）")
    rpt.table(pd.DataFrame({"dtype": df.dtypes.astype(str), "non-null": df.notna().sum()}))
    rpt.h3("敘述統計（describe）")
    rpt.table(df.describe(), "{:,.2f}")
    rpt.h3("State 各類筆數")
    rpt.table(df[CAT_FEATURE].value_counts().to_frame())
    rpt.h3("缺失值與支出為 0 的筆數")
    rpt.table(pd.DataFrame({
        "missing": df.isna().sum(),
        "zero": (df[NUM_FEATURES] == 0).sum().reindex(df.columns),
    }), "{:,.0f}")

    # 三個數值特徵 vs Profit 散佈圖（含迴歸線）
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, col in zip(axes, NUM_FEATURES):
        sns.regplot(data=df, x=col, y=TARGET, ax=ax,
                    scatter_kws={"alpha": 0.7}, line_kws={"color": "crimson"})
        ax.set_title(f"{col} vs Profit")
        ax.set_xlabel(col)
        ax.set_ylabel("Profit")
    rpt.fig("scatter_features.png", "Features vs Profit")

    # 各州 Profit 箱型圖
    plt.figure(figsize=(7, 4.5))
    sns.boxplot(data=df, x=CAT_FEATURE, y=TARGET, hue=CAT_FEATURE, legend=False)
    sns.stripplot(data=df, x=CAT_FEATURE, y=TARGET, color="black", alpha=0.5)
    plt.title("Profit by State")
    plt.xlabel("State")
    plt.ylabel("Profit")
    rpt.fig("box_state.png", "Profit by State")

    # 相關係數熱圖
    corr = df[NUM_FEATURES + [TARGET]].corr()
    plt.figure(figsize=(6.5, 5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, square=True)
    plt.title("Correlation Matrix")
    rpt.fig("corr_heatmap.png", "Correlation Matrix")
    rpt.h3("相關係數")
    rpt.table(corr)

    # =======================================================================
    # Phase 3. Data Preparation
    # =======================================================================
    rpt.h2("Phase 3. Data Preparation（資料準備）")
    X = df[ALL_FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=X[CAT_FEATURE]
    )
    rpt.p(f"訓練集 {X_train.shape}、測試集 {X_test.shape}（80/20，random_state=42，依 State 分層）")
    rpt.table(pd.DataFrame({"train": X_train[CAT_FEATURE].value_counts(),
                            "test": X_test[CAT_FEATURE].value_counts()}), "{:,.0f}")
    rpt.p("State 使用完整 one-hot（drop=None）：California=100、Florida=010、New York=001。"
          "含 State 時 LinearRegression(fit_intercept=False)，不含 State 時 fit_intercept=True，避免 dummy variable trap。")

    # 驗證：完整 one-hot + 無截距 與 drop first + 有截距 應為同一個模型
    demo = build_model(ALL_FEATURES).fit(X_train, y_train)
    ref = Pipeline([
        ("prep", ColumnTransformer([
            ("num", "passthrough", NUM_FEATURES),
            ("cat", OneHotEncoder(drop="first", sparse_output=False), [CAT_FEATURE]),
        ])),
        ("reg", LinearRegression(fit_intercept=True)),
    ]).fit(X_train, y_train)
    max_diff = np.abs(demo.predict(X_test) - ref.predict(X_test)).max()
    rpt.p(f"驗證：完整 one-hot（無截距）與 drop first（有截距）在測試集預測值的最大差異 = {max_diff:.2e}")
    assert max_diff < 1e-4

    # =======================================================================
    # Phase 4. Modeling
    # =======================================================================
    rpt.h2("Phase 4. Modeling（建模）")
    cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=RANDOM_STATE)
    scoring = {"r2": "r2", "rmse": "neg_root_mean_squared_error"}

    def cv_evaluate(model):
        res = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring)
        rmse = -res["test_rmse"]
        return {"CV R2 mean": res["test_r2"].mean(), "CV R2 std": res["test_r2"].std(),
                "CV RMSE mean": rmse.mean(), "CV RMSE std": rmse.std()}

    rpt.h3("Baseline vs 完整模型（RepeatedKFold 5×10）")
    rpt.table(pd.DataFrame({
        "Baseline (mean)": cv_evaluate(DummyRegressor(strategy="mean")),
        "Full model (4 features)": cv_evaluate(build_model(ALL_FEATURES)),
    }).T)

    # 窮舉 15 種特徵組合
    rows = []
    for k in range(1, len(ALL_FEATURES) + 1):
        for combo in combinations(ALL_FEATURES, k):
            rows.append({"features": " + ".join(combo), "n_features": k,
                         **cv_evaluate(build_model(combo)), "_combo": combo})
    cv_table = pd.DataFrame(rows).sort_values("CV RMSE mean").reset_index(drop=True)
    cv_table.index += 1
    rpt.h3(f"{len(cv_table)} 種特徵組合（依 CV RMSE 排序）")
    rpt.table(cv_table.drop(columns="_combo"))

    # ------------------------------------------------------------------
    # Forward Stepwise Feature Selection（逐步向前特徵選擇）
    # ------------------------------------------------------------------
    rpt.h3("Forward Stepwise Feature Selection（逐步向前特徵選擇）")
    rpt.p("依作業範例圖的設定：train/test 80/20、random_state=0（不分層），State one-hot 後 drop first"
          "（California 為基準組，共 5 個欄位）；每一步加入讓測試集 RMSE 最低的特徵，圖表為測試集 RMSE 與 R²。"
          "註：此做法由測試集決定特徵，與上方只用訓練集 CV 的分析不同。")

    # One-hot encode State（drop first），使用範例圖的切分（random_state=0，不分層）
    X_fs = pd.get_dummies(df[ALL_FEATURES], columns=[CAT_FEATURE], drop_first=True, dtype=float)
    X_train_fs, X_test_fs, y_train_fs, y_test_fs = train_test_split(
        X_fs, df[TARGET], test_size=0.2, random_state=0
    )

    all_indiv_features = list(X_train_fs.columns)
    fs_selected = []
    fs_remaining = list(all_indiv_features)
    forward_results = []

    for step in range(1, len(all_indiv_features) + 1):
        best_rmse = np.inf
        best_feat = None
        for feat in fs_remaining:
            trial = fs_selected + [feat]
            m = LinearRegression().fit(X_train_fs[trial], y_train_fs)
            rmse_val = root_mean_squared_error(y_test_fs, m.predict(X_test_fs[trial]))
            if rmse_val < best_rmse:
                best_rmse = rmse_val
                best_feat = feat
        fs_selected.append(best_feat)
        fs_remaining.remove(best_feat)
        m = LinearRegression().fit(X_train_fs[fs_selected], y_train_fs)
        pred_fs = m.predict(X_test_fs[fs_selected])
        forward_results.append({
            "Number of Features": step,
            "Selected Features": [f for f in fs_selected],
            "RMSE": root_mean_squared_error(y_test_fs, pred_fs),
            "R-squared": r2_score(y_test_fs, pred_fs),
        })

    fs_df = pd.DataFrame(forward_results)

    # Dual plot: RMSE & R-squared by number of features（matplotlib 預設樣式，與範例圖一致）
    with plt.style.context("default"):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 6.5))
        ax1.plot(fs_df["Number of Features"], fs_df["RMSE"], marker="o")
        ax1.set_title("RMSE by Number of Features")
        ax1.set_xlabel("Number of Features")
        ax1.set_ylabel("RMSE")

        ax2.plot(fs_df["Number of Features"], fs_df["R-squared"], marker="o")
        ax2.set_title("R-squared by Number of Features")
        ax2.set_xlabel("Number of Features")
        ax2.set_ylabel("R-squared")
        rpt.fig("forward_selection.png", "Forward Stepwise Feature Selection")

    # Summary table：Selected Features 以 [A, B] 形式顯示
    fs_table = fs_df.copy()
    fs_table["Selected Features"] = fs_table["Selected Features"].apply(lambda x: "[" + ", ".join(x) + "]")
    rpt.table(fs_table.set_index("Number of Features"), "{:,.6f}")

    fs_best = fs_df.loc[fs_df["RMSE"].idxmin()]
    rpt.p(f"RMSE 最低：k = {fs_best['Number of Features']}，特徵 = [{', '.join(fs_best['Selected Features'])}]，"
          f"RMSE = {fs_best['RMSE']:,.2f}，R² = {fs_best['R-squared']:.4f}")

    # one-standard-error rule
    best = cv_table.iloc[0]
    threshold = best["CV RMSE mean"] + best["CV RMSE std"]
    candidates = cv_table[cv_table["CV RMSE mean"] <= threshold]
    chosen = candidates.sort_values(["n_features", "CV RMSE mean"]).iloc[0]
    selected_features = list(chosen["_combo"])
    rpt.h3("One-standard-error rule")
    rpt.p(f"最佳組合：{best['features']}，CV RMSE = {best['CV RMSE mean']:,.2f} ± {best['CV RMSE std']:,.2f}；"
          f"門檻 = {threshold:,.2f}；門檻內共 {len(candidates)} 種組合。")
    rpt.p(f"最終選用特徵：{selected_features}（{len(selected_features)} 個）")
    rpt.p("註：sklearn 的 LinearRegression 不提供 p 值，此處特徵選擇依據是 CV 表現而非 p 值。")

    selected_model = build_model(selected_features).fit(X_train, y_train)
    reg = selected_model.named_steps["reg"]
    rpt.h3("選用模型係數（訓練集）")
    rpt.table(pd.DataFrame({"coefficient": reg.coef_},
                           index=selected_model.named_steps["prep"].get_feature_names_out()))
    if reg.fit_intercept:
        rpt.p(f"截距（intercept）= {reg.intercept_:,.4f}")
    else:
        rpt.p("模型未設截距：各州基準值由 State_California / State_Florida / State_New York 係數代表。")

    # =======================================================================
    # Phase 5. Evaluation
    # =======================================================================
    rpt.h2("Phase 5. Evaluation（評估）")
    sel_name = f"Selected ({' + '.join(selected_features)})"
    models = {
        "Baseline (mean)": DummyRegressor(strategy="mean"),
        "Full (4 features)": build_model(ALL_FEATURES),
        sel_name: build_model(selected_features),
    }
    test_rows, test_preds = {}, {}
    for name, m in models.items():
        pred = m.fit(X_train, y_train).predict(X_test)
        test_preds[name] = pred
        test_rows[name] = {"Test R2": r2_score(y_test, pred),
                           "Test RMSE": root_mean_squared_error(y_test, pred),
                           "Test MAE": mean_absolute_error(y_test, pred)}
    rpt.h3("測試集比較")
    rpt.table(pd.DataFrame(test_rows).T)

    # 迴歸假設檢查（訓練集殘差）
    rpt.h3("迴歸假設檢查（訓練集殘差）")
    train_pred = selected_model.predict(X_train)
    residuals = y_train - train_pred

    plt.figure(figsize=(7, 4.5))
    sns.scatterplot(x=train_pred, y=residuals)
    plt.axhline(0, color="gray", linestyle="--")
    plt.title("Residuals vs Fitted (training set)")
    plt.xlabel("Fitted Profit")
    plt.ylabel("Residual")
    rpt.fig("residuals_vs_fitted.png", "Residuals vs Fitted")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    sns.histplot(residuals, kde=True, ax=axes[0])
    axes[0].set_title("Residual Distribution (training set)")
    axes[0].set_xlabel("Residual")
    axes[0].set_ylabel("Count")
    stats.probplot(residuals, dist="norm", plot=axes[1])
    axes[1].set_title("Normal Q-Q Plot of Residuals")
    axes[1].set_xlabel("Theoretical Quantiles")
    axes[1].set_ylabel("Sample Quantiles")
    rpt.fig("residuals_normality.png", "Residual normality")

    sw_stat, sw_p = stats.shapiro(residuals)
    rpt.p(f"Shapiro-Wilk：W = {sw_stat:.4f}, p-value = {sw_p:.4f} → "
          + ("p < 0.05，拒絕常態假設" if sw_p < 0.05 else "p ≥ 0.05，無法拒絕常態假設"))
    worst_idx = residuals.abs().idxmax()
    sw2_stat, sw2_p = stats.shapiro(residuals.drop(worst_idx))
    rpt.p(f"敏感度分析：排除殘差最大的第 {worst_idx} 筆後，W = {sw2_stat:.4f}, p-value = {sw2_p:.4f}")
    rpt.table(df.loc[[worst_idx]], "{:,.2f}")
    rpt.p("獨立性：本資料為 50 家公司同一時間點的橫斷面資料，沒有時間順序，可視為符合獨立性假設。")

    selected_num = [f for f in selected_features if f in NUM_FEATURES]
    if len(selected_num) < 2:
        rpt.p(f"多重共線性：選用模型數值特徵只有 {len(selected_num)} 個，VIF 不適用，跳過。（參考）完整模型 VIF：")
        rpt.table(compute_vif(X_train, NUM_FEATURES))
    else:
        rpt.p("多重共線性（VIF）：")
        rpt.table(compute_vif(X_train, selected_num))

    # 測試集實際值 vs 預測值
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True, sharey=True)
    lims = [min(y_test.min(), *(p.min() for p in test_preds.values())) * 0.9,
            max(y_test.max(), *(p.max() for p in test_preds.values())) * 1.05]
    for ax, name in zip(axes, ["Full (4 features)", sel_name]):
        ax.scatter(y_test, test_preds[name], s=60, alpha=0.8)
        ax.plot(lims, lims, "--", color="crimson", label="y = x")
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_title(f"Actual vs Predicted (test set)\n{name}")
        ax.set_xlabel("Actual Profit")
        ax.set_ylabel("Predicted Profit")
        ax.legend()
    rpt.fig("actual_vs_predicted.png", "Actual vs Predicted")

    # =======================================================================
    # Phase 6. Deployment
    # =======================================================================
    rpt.h2("Phase 6. Deployment（部署）")
    final_model = build_model(selected_features).fit(df[selected_features], df[TARGET])
    joblib.dump({"model": final_model, "features": selected_features}, MODEL_PATH)
    rpt.p(f"以全部 {len(df)} 筆資料重新訓練最終模型，已儲存：{MODEL_PATH}")
    final_reg = final_model.named_steps["reg"]
    rpt.table(pd.DataFrame({"coefficient": final_reg.coef_},
                           index=final_model.named_steps["prep"].get_feature_names_out()))
    if final_reg.fit_intercept:
        rpt.p(f"截距（intercept）= {final_reg.intercept_:,.4f}")
    else:
        rpt.p("模型未設截距：各州基準值由 State one-hot 欄位係數代表。")

    # 示範：載入模型並預測一筆新公司資料
    bundle = joblib.load(MODEL_PATH)
    new_company = pd.DataFrame([{"R&D Spend": 120000.0, "Administration": 130000.0,
                                 "Marketing Spend": 300000.0, "State": "Florida"}])
    pred_profit = bundle["model"].predict(new_company[bundle["features"]])[0]
    rpt.h3("載入模型並預測新公司")
    rpt.table(new_company, "{:,.2f}")
    rpt.p(f"模型使用特徵：{bundle['features']}；預測 Profit = {pred_profit:,.2f}")

    rpt.save(OUT_DIR / "index.html")

    export_site_results()


if __name__ == "__main__":
    main()
