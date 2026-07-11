# Ячейки §7 для notebooks/39_chemai.ipynb — одна модель = одна code-ячейка

CHEM_FEATURES = r'''
# @title §4.0. Domain features (QSPR)
"""
Семь доменных признаков поверх RDKit-дескрипторов.
Используется в EDA (§4) и в ML-пайплайне (§7).
"""
# --- защита от деления на ноль в ratio-признаках ---
EPS = 1e-9


def add_chem_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Строит QSPR-признаки из известных дескрипторов.
    Если исходной колонки нет — признак не добавляется (ноутбук остаётся переносимым).
    """
    out = df.copy()
    eps = 1e-6

    # --- липофильность / полярность ---
    if "MolLogP" in out.columns and "TPSA" in out.columns:
        out["LogP_TPSA"] = out["MolLogP"] / (out["TPSA"] + eps)

    # --- доля ароматических атомов ---
    arom = out["NumAromaticRings"] if "NumAromaticRings" in out.columns else out.get("fr_benzene")
    if arom is not None and "HeavyAtomCount" in out.columns:
        out["Arom_Heavy_ratio"] = arom / (out["HeavyAtomCount"] + eps)

    # --- суммарный частичный заряд ---
    if "MaxPartialCharge" in out.columns and "MinPartialCharge" in out.columns:
        out["Charge_sum"] = out["MaxPartialCharge"] + out["MinPartialCharge"]

    # --- бинарные флаги функциональных групп ---
    if "fr_imide" in out.columns:
        out["fr_imide_flag"] = (out["fr_imide"] > 0).astype(np.float64)
    if "fr_sulfone" in out.columns:
        out["fr_sulfone_flag"] = (out["fr_sulfone"] > 0).astype(np.float64)

    # --- нелинейные комбинации колец и липофильности ---
    if "RingCount" in out.columns and "MolLogP" in out.columns:
        out["Ring_LogP"] = out["RingCount"] * out["MolLogP"]
    if "FractionCSP3" in out.columns and "MolLogP" in out.columns:
        out["FractionCSP3_LogP"] = out["FractionCSP3"] * out["MolLogP"]
    return out


print("add_chem_features — готово")
'''

SECTION7_INTRO = """\
## §7. Обучение моделей

**Схема:** ClusterKFold (5 фолдов, KMeans по MolLogP/TPSA/RingCount/HeavyAtomCount) → OOF каждой модели по IC50, CC50, SI → Ridge meta-stacking → submission.

**Инфраструктура (§7.0):** импорты → классы (по одной ячейке) → функции OOF → подготовка данных.

Далее — **§7.1–§7.9:** одна модель = одна code-ячейка + краткий разбор OOF.
"""

INFRA_CELLS = [
    r'''
# @title §7.0. Импорты и константы
"""
Библиотеки §7, гиперпараметры пайплайна (CFG) и глобальные хранилища OOF.
Запускать первой среди ячеек §7.
"""
# --- ML-библиотеки (базовые модели + meta Ridge) ---
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Callable

import lightgbm as lgb
from sklearn.cluster import KMeans
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import BayesianRidge, ElasticNetCV, RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

# --- служебные константы ---
VARIANCE_EPS = 1e-12  # порог «нулевой» дисперсии признака
INDEX_COL = "index"   # имя колонки id в submission

# --- гиперпараметры пайплайна (как в submission2, LB 349.31) ---
CFG = SimpleNamespace(
    missing_threshold=0.3,       # дроп колонок с >30% NA
    n_folds=5,
    n_clusters=5,
    random_seed=RANDOM_STATE,
    log_transform_ic50_cc50=True,  # log1p для IC50/CC50 при обучении
)

# --- фиксированный порядок моделей в матрице stacking ---
CANDIDATE_ORDER = (
    "lgb", "xgb", "ridge", "elastic_net", "hist_gbrt",
    "random_forest", "extra_trees", "grad_boosting_sklearn", "bayesian_ridge",
)

# --- накопители результатов (заполняются в §7.1–§7.9) ---
CANDIDATES = {}
OOF_STORE = {}      # name → {target → oof_vector}
MODEL_METRICS = {}  # name → {score, parts}

print("§7.0: импорты и константы — готово")
''',
    r'''
# @title §7.0. Метрики rmse и competition_score
"""
Метрики соревнования Kaggle: RMSE по каждому таргету и их среднее.
"""


def rmse(y_true, y_pred) -> float:
    """Root MSE для одного таргета."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def competition_score(y_true, y_pred):
    """
    Главная метрика хакатона: mean(RMSE_IC50, RMSE_CC50, RMSE_SI).
    y_true, y_pred — матрицы (n_samples, 3).
    """
    parts = {}
    for i, name in enumerate(TARGETS):
        parts[name] = rmse(y_true[:, i], y_pred[:, i])
    return float(np.mean(list(parts.values()))), parts


print("rmse, competition_score — готово")
''',
    r'''
# @title §7.0. Preprocessor
"""
Подготовка табличных признаков: impute → отбор колонок → StandardScaler.
На каждом CV-фолде fit только на train-части (без утечки в validation).
"""


class Preprocessor:
    def __init__(self, missing_threshold: float = 0.3) -> None:
        self.missing_threshold = missing_threshold
        self._medians = None
        self._feature_columns = None
        self._scaler = StandardScaler()
        self._dropped_columns: list[str] = []

    def fit(self, df: pd.DataFrame) -> None:
        numeric = df.select_dtypes(include=[np.number]).copy()

        # --- отбор: слишком много NA или нулевая дисперсия ---
        miss_ratio = numeric.isna().mean()
        drop_missing = miss_ratio[miss_ratio > self.missing_threshold].index.tolist()
        medians = numeric.median(numeric_only=True)
        filled = numeric.fillna(medians)
        variances = filled.var()
        drop_zero_var = variances[variances <= VARIANCE_EPS].index.tolist()
        self._dropped_columns = sorted(set(drop_missing + drop_zero_var))
        self._feature_columns = [c for c in numeric.columns if c not in self._dropped_columns]

        # --- impute + fit scaler ---
        self._medians = medians.reindex(self._feature_columns)
        train_matrix = numeric[self._feature_columns].fillna(self._medians)
        self._scaler.fit(train_matrix)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Возвращает numpy-массив масштабированных признаков."""
        numeric = df.select_dtypes(include=[np.number]).copy()
        x = numeric[self._feature_columns].fillna(self._medians)
        return self._scaler.transform(x)


print("Preprocessor — готово")
''',
    r'''
# @title §7.0. ClusterKFold
"""
Leak-free CV: KMeans строит cluster_id только на подвыборке train-фолда,
затем GroupKFold режет по кластерам (похожие молекулы не смешиваются).
"""


class ClusterKFold:
    def __init__(self, n_splits=5, n_clusters=5, random_state=42, fit_fraction=0.75):
        self.n_splits = n_splits
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.fit_fraction = fit_fraction
        # химически осмысленные дескрипторы для кластеризации
        self._cluster_cols = ("MolLogP", "RingCount", "TPSA", "HeavyAtomCount")

    def split(self, X: pd.DataFrame, y=None):
        # --- матрица для KMeans (fallback на первые 10 числовых колонок) ---
        cols = [c for c in self._cluster_cols if c in X.columns]
        if len(cols) >= 2:
            sub = X[cols].select_dtypes(include=[np.number])
        else:
            sub = X.select_dtypes(include=[np.number]).iloc[:, :10]
        sub = sub.replace([np.inf, -np.inf], np.nan).fillna(sub.median(numeric_only=True))
        matrix = sub.to_numpy(dtype=np.float64)

        # --- KMeans только на случайной подвыборке train (без val) ---
        n = len(X)
        rng = np.random.default_rng(self.random_state)
        fit_size = max(self.n_clusters * 2, int(self.fit_fraction * n))
        fit_size = min(fit_size, n - 1)
        fit_idx = rng.choice(n, size=fit_size, replace=False)

        km = KMeans(
            n_clusters=min(self.n_clusters, fit_size),
            random_state=self.random_state,
            n_init="auto",
        )
        km.fit(matrix[fit_idx])
        groups = km.predict(matrix)

        # --- GroupKFold по cluster_id ---
        n_splits_eff = min(self.n_splits, len(np.unique(groups)))
        yield from GroupKFold(n_splits=n_splits_eff).split(X, groups=groups)


print("ClusterKFold — готово")
''',
    r'''
# @title §7.0. Expm1Predictor
"""
Обёртка над моделью, обученной в log1p-шкале таргета.
При predict возвращает исходные концентрации через expm1.
"""


class Expm1Predictor:
    def __init__(self, inner):
        self.inner = inner

    def predict(self, x):
        return np.expm1(np.asarray(self.inner.predict(x), dtype=np.float64))


print("Expm1Predictor — готово")
''',
    r'''
# @title §7.0. NumpySafeLGBMRegressor
"""
LightGBM после fit запоминает имена колонок; при predict на numpy
sklearn выдаёт warning — обёртка подставляет DataFrame с теми же именами.
"""


class NumpySafeLGBMRegressor:
    def __init__(self, model):
        self._model = model

    def predict(self, X):
        names = getattr(self._model, "feature_names_in_", None)
        if names is not None and not isinstance(X, pd.DataFrame):
            x = pd.DataFrame(np.asarray(X, dtype=np.float64), columns=list(names))
        else:
            x = X
        return np.asarray(self._model.predict(x), dtype=np.float64)


print("NumpySafeLGBMRegressor — готово")
''',
    r'''
# @title §7.0. ModelCandidate
"""
Единый интерфейс базовой модели для stacking:
  fit_fold  — обучение на train-фолде CV (с val для early stopping у бустингов)
  fit_final — обучение на полном train для инференса test
"""


@dataclass(frozen=True)
class ModelCandidate:
    name: str
    fit_fold: Callable
    fit_final: Callable
    short_description: str = ""


# реестр обученных кандидатов (заполняется в §7.1–§7.9)
CANDIDATES = {}

print("ModelCandidate, CANDIDATES — готово")
''',
    r'''
# @title §7.0. Функции OOF и stacking
"""
Цикл out-of-fold для одной модели, агрегация по таргетам,
meta Ridge и постобработка submission.
"""


def _hold_split(n, rng, frac=0.1):
    """90/10 split внутри train для early stopping при fit_final."""
    order = rng.permutation(n)
    n_hold = max(1, int(frac * n))
    return order[n_hold:], order[:n_hold]


def y_train_space(y_raw, use_log):
    """Преобразование таргета в пространство обучения (log1p для IC50/CC50)."""
    return np.log1p(np.clip(y_raw, 0.0, None)) if use_log else y_raw.copy()


def pred_to_original(pred, use_log):
    """Обратно в исходную шкалу концентраций / SI."""
    if use_log:
        return np.clip(np.expm1(np.asarray(pred, dtype=np.float64)), EPS, None)
    return np.asarray(pred, dtype=np.float64)


def collect_oof_single(x_raw, y_raw, candidate, cfg, use_log):
    """
    OOF-вектор одной модели на одном таргете.
    Preprocessor переобучается на каждом фолде — без утечки scaler/imputer.
    """
    n = len(x_raw)
    oof = np.full(n, np.nan)
    y_s = y_train_space(y_raw, use_log)
    cv = ClusterKFold(cfg.n_folds, cfg.n_clusters, cfg.random_seed)

    for fold_id, (tr, va) in enumerate(cv.split(x_raw, y_raw)):
        pre = Preprocessor(cfg.missing_threshold)
        pre.fit(x_raw.iloc[tr])
        x_tr = pre.transform(x_raw.iloc[tr])
        x_va = pre.transform(x_raw.iloc[va])
        m = candidate.fit_fold(
            x_tr, y_s[tr], x_va, y_s[va], cfg.random_seed + fold_id,
        )
        oof[va] = pred_to_original(m.predict(x_va), use_log)
    return oof


def run_model_oof(candidate, x_raw, y_df, cfg):
    """
    Полный OOF одной модели по IC50, CC50, SI.
    Результаты → OOF_STORE / MODEL_METRICS для stacking (§7.10).
    """
    name = candidate.name
    OOF_STORE[name] = {}
    parts = {}
    for t in TARGETS:
        use_log = cfg.log_transform_ic50_cc50 and t in ("IC50", "CC50")
        y_raw = y_df[t].to_numpy(dtype=np.float64)
        oof = collect_oof_single(x_raw, y_raw, candidate, cfg, use_log)
        OOF_STORE[name][t] = oof
        parts[t] = rmse(y_raw, oof)

    score = float(np.mean(list(parts.values())))
    MODEL_METRICS[name] = {"score": score, "parts": parts}
    print(f"{candidate.short_description} ({name})")
    print("  OOF competition_score:", round(score, 2))
    print("  OOF RMSE по таргетам:", {k: round(v, 2) for k, v in parts.items()})
    return score, parts


def fit_all_final(candidate, x_full, y_all, random_seed):
    """Финальный fit базовой модели на полном train (с holdout внутри fit_final)."""
    rng = np.random.default_rng(random_seed)
    trn, hold = _hold_split(len(x_full), rng)
    return candidate.fit_final(
        x_full, y_all, x_full, y_all,
        x_full[trn], y_all[trn], x_full[hold], y_all[hold],
        random_seed,
    )


def fit_meta_ridge(oof, y_original):
    """Meta-уровень: StandardScaler + RidgeCV на OOF-матрице базовых моделей."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", RidgeCV(alphas=np.logspace(-3, 3, 19))),
    ]).fit(oof, y_original)


def postprocess(predictions: pd.DataFrame) -> pd.DataFrame:
    """Физически осмысленные ограничения: концентрации > 0, SI ≥ 0."""
    out = predictions.copy()
    out["IC50"] = out["IC50"].clip(lower=1e-8)
    out["CC50"] = out["CC50"].clip(lower=1e-8)
    out["SI"] = out["SI"].clip(lower=0.0)
    return out


print("OOF/stacking функции — готово")
''',
    r'''
# @title §7.0. Подготовка train/test
"""
Матрицы признаков с domain features и numpy после Preprocessor.
full_pre fit на всём train — для финального predict test (§7.10).
"""
# --- таргеты и признаки train ---
y_df = train_df[list(TARGETS)].copy()
x_train_feat = add_chem_features(
    train_df.drop(columns=list(TARGETS)).drop(columns=["index"], errors="ignore")
)

# --- test: сохраняем index для submission ---
if "index" in test_df.columns:
    idx = test_df["index"].values
    x_test_feat = add_chem_features(test_df.drop(columns=["index"]))
else:
    idx = np.arange(len(test_df))
    x_test_feat = add_chem_features(test_df.copy())

# --- preprocessor на полном train → numpy для моделей ---
full_pre = Preprocessor(CFG.missing_threshold)
full_pre.fit(x_train_feat)
x_test_np = full_pre.transform(x_test_feat)
x_full_np = full_pre.transform(x_train_feat)

print("train:", x_train_feat.shape, "| test:", x_test_feat.shape)
''',
]

SECTION7_STACKING = r'''
# @title §7.10. Ridge meta-stacking → submission
"""
Сбор OOF всех 9 моделей → meta RidgeCV → финальный fit → predict test.
Базовые модели уже обучены в §7.1–§7.9; здесь только stacking и submission.
"""
# --- проверка: все 9 моделей должны быть в OOF_STORE ---
assert len(OOF_STORE) == len(CANDIDATE_ORDER), (
    f"Ожидалось {len(CANDIDATE_ORDER)} моделей в OOF_STORE, есть {len(OOF_STORE)}. "
    "Запустите ячейки §7.1–§7.9 по порядку."
)

# --- сводная таблица OOF базового слоя ---
base_rows = []
for name in CANDIDATE_ORDER:
    m = MODEL_METRICS[name]
    row = {"model": name, "oof_score": round(m["score"], 2)}
    row.update({f"rmse_{t}": round(m["parts"][t], 2) for t in TARGETS})
    base_rows.append(row)
base_oof_df = pd.DataFrame(base_rows).sort_values("oof_score")
print("OOF базовых моделей (competition_score):")
display(base_oof_df)

metas, oof_pred, finals = {}, {}, {}

# --- meta Ridge + финальные модели для test (отдельно по каждому таргету) ---
for t in TARGETS:
    y_raw = y_df[t].to_numpy(dtype=np.float64)
    use_log = CFG.log_transform_ic50_cc50 and t in ("IC50", "CC50")
    oof = np.column_stack([OOF_STORE[name][t] for name in CANDIDATE_ORDER])
    meta = fit_meta_ridge(oof, y_raw)
    metas[t] = meta
    oof_pred[t] = meta.predict(oof)
    finals[t] = {
        name: (
            Expm1Predictor(fit_all_final(CANDIDATES[name], x_full_np, y_train_space(y_raw, use_log), CFG.random_seed))
            if use_log
            else fit_all_final(CANDIDATES[name], x_full_np, y_train_space(y_raw, use_log), CFG.random_seed)
        )
        for name in CANDIDATE_ORDER
    }

# --- OOF-метрика stacking (на train) ---
y_mat = y_df[list(TARGETS)].to_numpy(dtype=np.float64)
oof_matrix = np.column_stack([oof_pred[t] for t in TARGETS])
stack_oof, stack_parts = competition_score(y_mat, oof_matrix)

# --- predict test: базовые модели → meta → submission ---
test_stack = {}
for t in TARGETS:
    cols = [finals[t][name].predict(x_test_np) for name in CANDIDATE_ORDER]
    test_stack[t] = np.column_stack(cols)

ic_t = metas["IC50"].predict(test_stack["IC50"])
cc_t = metas["CC50"].predict(test_stack["CC50"])
si_t = metas["SI"].predict(test_stack["SI"])
submission_df = postprocess(pd.DataFrame({INDEX_COL: idx, "IC50": ic_t, "CC50": cc_t, "SI": si_t}))

# --- эталон author4 submission2 (verify_notebook_submission.py) ---
assert abs(stack_oof - 557.22) < 0.5, (
    f"OOF stacking {stack_oof:.2f} ≠ эталон ~557.22. "
    "Colab: Runtime → Restart session, затем Run All с §7.0."
)

print("OOF stacking competition_score:", round(stack_oof, 2))
print("OOF RMSE по таргетам:", {k: round(v, 2) for k, v in stack_parts.items()})
submission_df.head()
'''

# (section_num, slug, title, code, conclusion_key)
MODELS = [
    (
        1,
        "lgb",
        "LightGBM",
        r'''
# @title §7.1. LightGBM
"""
Gradient boosting (LightGBM): OOF на ClusterKFold по IC50, CC50, SI.
Early stopping на validation-фолде внутри каждого split.
"""


def train_lgb(x_tr, y_tr, x_va, y_va, rs):
    """Обучение LGBM с early stopping; возвращает numpy-safe обёртку."""
    model = lgb.LGBMRegressor(
        n_estimators=2000,
        learning_rate=0.05,
        num_leaves=31,
        feature_fraction=0.9,
        bagging_fraction=0.9,
        bagging_freq=1,
        min_child_samples=20,
        lambda_l2=1.0,
        random_state=rs,
        verbose=-1,
    )
    model.fit(
        x_tr, y_tr,
        eval_set=[(x_va, y_va)],
        callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(0)],
    )
    return NumpySafeLGBMRegressor(model)


def lgb_fold(x_tr, y_tr, x_va, y_va, rs):
    return train_lgb(x_tr, y_tr, x_va, y_va, rs)


def lgb_final(x_full, y_all, _xf, _yf, x_tr, y_tr, x_va, y_va, rs):
    return train_lgb(x_tr, y_tr, x_va, y_va, rs)


# --- регистрация кандидата и OOF по трём таргетам ---
CANDIDATES["lgb"] = ModelCandidate("lgb", lgb_fold, lgb_final, "LightGBM")
run_model_oof(CANDIDATES["lgb"], x_train_feat, y_df, CFG)
''',
        "AFTER_MODEL_LGB",
    ),
    (
        2,
        "xgb",
        "XGBoost",
        r'''
# @title §7.2. XGBoost
"""XGBoost с early stopping: OOF по IC50, CC50, SI."""


def train_xgb(x_tr, y_tr, x_va, y_va, rs):
    """Обучение XGBRegressor; val-фолд — для early stopping."""
    model = XGBRegressor(
        n_estimators=3000,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=rs,
        early_stopping_rounds=80,
    )
    model.fit(x_tr, y_tr, eval_set=[(x_va, y_va)], verbose=False)
    return model


def xgb_fold(x_tr, y_tr, x_va, y_va, rs):
    return train_xgb(x_tr, y_tr, x_va, y_va, rs)


def xgb_final(x_full, y_all, _xf, _yf, x_tr, y_tr, x_va, y_va, rs):
    return train_xgb(x_tr, y_tr, x_va, y_va, rs)


CANDIDATES["xgb"] = ModelCandidate("xgb", xgb_fold, xgb_final, "XGBoost")
run_model_oof(CANDIDATES["xgb"], x_train_feat, y_df, CFG)
''',
        "AFTER_MODEL_XGB",
    ),
    (
        3,
        "ridge",
        "RidgeCV",
        r'''
# @title §7.3. RidgeCV
"""Линейный baseline: Ridge с перебором alpha по log-сетке."""


def train_ridge(x_tr, y_tr):
    return RidgeCV(alphas=np.logspace(-4, 4, 25)).fit(x_tr, y_tr)


def ridge_fold(x_tr, y_tr, _x_va, _y_va, _rs):
    return train_ridge(x_tr, y_tr)


def ridge_final(x_full, y_all, _xf, _yf, _x_tr, _y_tr, _x_va, _y_va, _rs):
    return train_ridge(x_full, y_all)


CANDIDATES["ridge"] = ModelCandidate("ridge", ridge_fold, ridge_final, "RidgeCV")
run_model_oof(CANDIDATES["ridge"], x_train_feat, y_df, CFG)
''',
        "AFTER_MODEL_RIDGE",
    ),
    (
        4,
        "elastic_net",
        "ElasticNetCV",
        r'''
# @title §7.4. ElasticNetCV
"""L1+L2 линейная модель: перебор l1_ratio, умеренная разреженность весов."""


def elastic_fold(x_tr, y_tr, _x_va, _y_va, rs):
    return ElasticNetCV(
        l1_ratio=[0.1, 0.5, 0.9, 0.99], random_state=rs, max_iter=5000,
    ).fit(x_tr, y_tr)


def elastic_final(x_full, y_all, _xf, _yf, _x_tr, _y_tr, _x_va, _y_va, rs):
    return ElasticNetCV(
        l1_ratio=[0.1, 0.5, 0.9, 0.99], random_state=rs, max_iter=5000,
    ).fit(x_full, y_all)


CANDIDATES["elastic_net"] = ModelCandidate(
    "elastic_net", elastic_fold, elastic_final, "ElasticNetCV",
)
run_model_oof(CANDIDATES["elastic_net"], x_train_feat, y_df, CFG)
''',
        "AFTER_MODEL_ELASTIC",
    ),
    (
        5,
        "hist_gbrt",
        "HistGradientBoosting",
        r'''
# @title §7.5. HistGradientBoosting
"""Sklearn HistGBR — бустинг на гистограммах, быстрее классического GBR."""


def hgb_fold(x_tr, y_tr, _x_va, _y_va, rs):
    return HistGradientBoostingRegressor(
        max_iter=200, learning_rate=0.08, max_depth=7, random_state=rs,
    ).fit(x_tr, y_tr)


def hgb_final(x_full, y_all, _xf, _yf, _x_tr, _y_tr, _x_va, _y_va, rs):
    return HistGradientBoostingRegressor(
        max_iter=400, learning_rate=0.06, max_depth=7,
        early_stopping=True, validation_fraction=0.12, random_state=rs,
    ).fit(x_full, y_all)


CANDIDATES["hist_gbrt"] = ModelCandidate(
    "hist_gbrt", hgb_fold, hgb_final, "HistGradientBoosting",
)
run_model_oof(CANDIDATES["hist_gbrt"], x_train_feat, y_df, CFG)
''',
        "AFTER_MODEL_HGB",
    ),
    (
        6,
        "random_forest",
        "RandomForest",
        r'''
# @title §7.6. RandomForest
"""Bagging деревьев: устойчив к выбросам, добавляет diversity в stacking."""


def rf_fold(x_tr, y_tr, _x_va, _y_va, rs):
    return RandomForestRegressor(
        n_estimators=400, min_samples_leaf=2, random_state=rs, n_jobs=-1,
    ).fit(x_tr, y_tr)


def rf_final(x_full, y_all, _xf, _yf, _x_tr, _y_tr, _x_va, _y_va, rs):
    return RandomForestRegressor(
        n_estimators=400, min_samples_leaf=2, random_state=rs, n_jobs=-1,
    ).fit(x_full, y_all)


CANDIDATES["random_forest"] = ModelCandidate(
    "random_forest", rf_fold, rf_final, "RandomForest",
)
run_model_oof(CANDIDATES["random_forest"], x_train_feat, y_df, CFG)
''',
        "AFTER_MODEL_RF",
    ),
    (
        7,
        "extra_trees",
        "ExtraTrees",
        r'''
# @title §7.7. ExtraTrees
"""Extremely Randomized Trees — случайные сплиты, decorrelation с RF/LGBM."""


def et_fold(x_tr, y_tr, _x_va, _y_va, rs):
    return ExtraTreesRegressor(
        n_estimators=500, min_samples_leaf=2, random_state=rs, n_jobs=-1,
    ).fit(x_tr, y_tr)


def et_final(x_full, y_all, _xf, _yf, _x_tr, _y_tr, _x_va, _y_va, rs):
    return ExtraTreesRegressor(
        n_estimators=500, min_samples_leaf=2, random_state=rs, n_jobs=-1,
    ).fit(x_full, y_all)


CANDIDATES["extra_trees"] = ModelCandidate(
    "extra_trees", et_fold, et_final, "ExtraTrees",
)
run_model_oof(CANDIDATES["extra_trees"], x_train_feat, y_df, CFG)
''',
        "AFTER_MODEL_ET",
    ),
    (
        8,
        "grad_boosting_sklearn",
        "GradientBoosting (sklearn)",
        r'''
# @title §7.8. GradientBoosting (sklearn)
"""Классический sklearn GBR — третий бустинг в ансамбле (после LGBM/XGB)."""


def gbr_fold(x_tr, y_tr, _x_va, _y_va, rs):
    return GradientBoostingRegressor(
        n_estimators=180, learning_rate=0.06, max_depth=4, random_state=rs,
    ).fit(x_tr, y_tr)


def gbr_final(x_full, y_all, _xf, _yf, _x_tr, _y_tr, _x_va, _y_va, rs):
    return GradientBoostingRegressor(
        n_estimators=250, learning_rate=0.05, max_depth=4, random_state=rs,
    ).fit(x_full, y_all)


CANDIDATES["grad_boosting_sklearn"] = ModelCandidate(
    "grad_boosting_sklearn", gbr_fold, gbr_final, "GradientBoosting (sklearn)",
)
run_model_oof(CANDIDATES["grad_boosting_sklearn"], x_train_feat, y_df, CFG)
''',
        "AFTER_MODEL_GBR",
    ),
    (
        9,
        "bayesian_ridge",
        "BayesianRidge",
        r'''
# @title §7.9. BayesianRidge
"""Вероятностная линейная регрессия — последняя базовая модель перед stacking."""


def bayes_fold(x_tr, y_tr, _x_va, _y_va, _rs):
    return BayesianRidge(max_iter=500).fit(x_tr, y_tr)


def bayes_final(x_full, y_all, _xf, _yf, _x_tr, _y_tr, _x_va, _y_va, _rs):
    return BayesianRidge(max_iter=800).fit(x_full, y_all)


CANDIDATES["bayesian_ridge"] = ModelCandidate(
    "bayesian_ridge", bayes_fold, bayes_final, "BayesianRidge",
)
run_model_oof(CANDIDATES["bayesian_ridge"], x_train_feat, y_df, CFG)
''',
        "AFTER_MODEL_BAYES",
    ),
]


def build_section7_cells(md_fn, code_fn, conclusions: dict) -> list:
    """Список ячеек §7: intro → infra (по классу) → (model + md) × 9 → stacking → summary."""
    cells = [md_fn(SECTION7_INTRO)]
    for infra_code in INFRA_CELLS:
        cells.append(code_fn(infra_code.strip() + "\n"))
    for _num, _slug, _title, model_code, conc_key in MODELS:
        cells.append(code_fn(model_code.strip() + "\n"))
        cells.append(md_fn(conclusions[conc_key]))
    cells.append(code_fn(SECTION7_STACKING.strip() + "\n"))
    cells.append(md_fn(conclusions["AFTER_MODELS"]))
    return cells
