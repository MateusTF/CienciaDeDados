# -*- coding: utf-8 -*-
"""
04_algoritmos_individuais.py

Sprint 3 - Divisão dos dados, treinamento e avaliação dos algoritmos
individuais de regressão.

Entrada : df_pre_processado.csv (gerado por 03_pre_processamento.py)
Alvo    : cresc_pop_0a4_pct (crescimento % da população de 0 a 4 anos, 2024-2026)

Protocolo
  1. Divisão hold-out 80/20 (treino/teste), estratificada por quintis do alvo,
     random_state = 42. O conjunto de teste só é usado na avaliação final.
  2. No treino: busca em grade (GridSearchCV) com validação cruzada K-Fold
     (k = 5, shuffle, random_state = 42), otimizando o RMSE. Os folds de
     validação fazem o papel do conjunto de validação.
  3. Pré-processamento dentro do Pipeline (ajustado só nos dados de treino de
     cada fold, evitando vazamento):
       - features numéricas -> scaler (StandardScaler, MinMaxScaler ou
         RobustScaler, escolhido na própria busca em grade para os algoritmos
         sensíveis à escala);
       - uf -> One-Hot Encoding.
  4. Melhor configuração de cada algoritmo é re-treinada no treino completo e
     avaliada no teste com MAE, RMSE, R² e MedAE, além do tempo de treino.

Saídas
  resultados/resultados_individuais.csv   métricas de teste e de validação
  resultados/melhores_hiperparametros.csv melhor configuração de cada algoritmo
  resultados/comparacao_scalers.csv       RMSE de validação por scaler
  resultados/cv_folds_rmse.csv            RMSE de cada fold (melhor config.)
  figuras/s3_boxplot_cv_rmse.png
  figuras/s3_real_vs_previsto.png
  figuras/s3_residuos.png
"""

import json
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                             median_absolute_error, r2_score)
from sklearn.model_selection import (GridSearchCV, KFold, cross_val_score,
                                     train_test_split)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (MinMaxScaler, OneHotEncoder, RobustScaler,
                                   StandardScaler)
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

warnings.filterwarnings("ignore", category=ConvergenceWarning)

SEED = 42
BASE_DIR = Path(__file__).resolve().parent
ARQ = BASE_DIR / "df_pre_processado.csv"
OUT_RES = BASE_DIR / "resultados"
OUT_FIG = BASE_DIR / "figuras"
OUT_RES.mkdir(exist_ok=True)
OUT_FIG.mkdir(exist_ok=True)

ALVO = "cresc_pop_0a4_pct"
FEATURES_NUM = ["prop_0a4_2022", "cresc_pop_2022_2024_pct", "pop_total_2022_log10"]
FEATURES_CAT = ["uf"]

# ---------------------------------------------------------------
# 1) Dados e divisão treino/teste
# ---------------------------------------------------------------
df = pd.read_csv(ARQ, dtype={"cod_municipio": str})
X = df[FEATURES_NUM + FEATURES_CAT]
y = df[ALVO]

# Estratificação por quintis do alvo: treino e teste com distribuições
# semelhantes do crescimento (o alvo tem caudas longas).
estratos = pd.qcut(y, q=5, labels=False)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.20, random_state=SEED, stratify=estratos
)
print(f"Treino: {len(X_tr)} | Teste: {len(X_te)}")
print(f"Alvo treino: média={y_tr.mean():.3f} dp={y_tr.std():.3f} | "
      f"teste: média={y_te.mean():.3f} dp={y_te.std():.3f}")

cv = KFold(n_splits=5, shuffle=True, random_state=SEED)


def montar_pipeline(modelo, scaler=StandardScaler()):
    pre = ColumnTransformer([
        ("num", scaler, FEATURES_NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore"), FEATURES_CAT),
    ])
    return Pipeline([("pre", pre), ("modelo", modelo)])


SCALERS = [StandardScaler(), MinMaxScaler(), RobustScaler()]

# ---------------------------------------------------------------
# 2) Algoritmos e grades de hiperparâmetros
# ---------------------------------------------------------------
ALGORITMOS = {
    "Baseline (média)": (DummyRegressor(strategy="mean"), {}),
    "Regressão Linear": (LinearRegression(), {"pre__num": SCALERS}),
    "Ridge": (Ridge(random_state=SEED), {
        "pre__num": SCALERS,
        "modelo__alpha": [0.01, 0.1, 1, 10, 100],
    }),
    "KNN": (KNeighborsRegressor(), {
        "pre__num": SCALERS,
        "modelo__n_neighbors": [3, 5, 7, 9, 11, 15, 21],
        "modelo__weights": ["uniform", "distance"],
        "modelo__p": [1, 2],
    }),
    "Árvore de Decisão": (DecisionTreeRegressor(random_state=SEED), {
        # árvores não dependem da escala: scaler fixo, só para manter o pipeline
        "modelo__max_depth": [3, 5, 7, 10, None],
        "modelo__min_samples_leaf": [1, 5, 10, 20],
    }),
    "SVR (RBF)": (SVR(kernel="rbf"), {
        "pre__num": SCALERS,
        "modelo__C": [0.1, 1, 10, 100],
        "modelo__epsilon": [0.05, 0.1, 0.5],
        "modelo__gamma": ["scale", 0.01, 0.1],
    }),
    "MLP": (MLPRegressor(max_iter=2000, early_stopping=True,
                         learning_rate_init=1e-3, random_state=SEED), {
        "pre__num": SCALERS,
        "modelo__hidden_layer_sizes": [(32,), (64,), (64, 32)],
        "modelo__alpha": [1e-4, 1e-3, 1e-2],
        "modelo__activation": ["relu", "tanh"],
    }),
}


def nome_scaler(obj):
    return type(obj).__name__


linhas, melhores, comp_scalers, folds = [], [], [], {}

for nome, (modelo, grade) in ALGORITMOS.items():
    print(f"\n>>> {nome}")
    pipe = montar_pipeline(modelo)
    t0 = time.perf_counter()
    gs = GridSearchCV(pipe, grade or {}, cv=cv, n_jobs=-1,
                      scoring="neg_root_mean_squared_error", refit=True)
    gs.fit(X_tr, y_tr)
    t_busca = time.perf_counter() - t0

    # Tempo de treino da configuração final (re-treino no treino completo)
    final = gs.best_estimator_
    t0 = time.perf_counter()
    final.fit(X_tr, y_tr)
    t_treino = time.perf_counter() - t0
    t0 = time.perf_counter()
    y_pred = final.predict(X_te)
    t_pred = time.perf_counter() - t0

    i = gs.best_index_
    cv_rmse = -gs.cv_results_["mean_test_score"][i]
    cv_std = gs.cv_results_["std_test_score"][i]
    folds[nome] = [-gs.cv_results_[f"split{k}_test_score"][i] for k in range(5)]

    linhas.append({
        "algoritmo": nome,
        "rmse_validacao_cv": cv_rmse,
        "dp_rmse_validacao_cv": cv_std,
        "mae_teste": mean_absolute_error(y_te, y_pred),
        "rmse_teste": np.sqrt(mean_squared_error(y_te, y_pred)),
        "r2_teste": r2_score(y_te, y_pred),
        "medae_teste": median_absolute_error(y_te, y_pred),
        "tempo_treino_s": t_treino,
        "tempo_predicao_ms": t_pred * 1000,
        "tempo_busca_grade_s": t_busca,
        "n_configuracoes": len(gs.cv_results_["params"]),
    })

    params = {k: (nome_scaler(v) if k == "pre__num" else v)
              for k, v in gs.best_params_.items()}
    melhores.append({"algoritmo": nome, "melhores_hiperparametros": json.dumps(params, ensure_ascii=False, default=str)})
    print(f"   melhores: {params} | RMSE CV={cv_rmse:.4f} | RMSE teste={linhas[-1]['rmse_teste']:.4f} | R²={linhas[-1]['r2_teste']:.4f}")

    # Comparação de scalers: melhor RMSE de validação obtido com cada scaler
    if "pre__num" in grade:
        res = pd.DataFrame(gs.cv_results_)
        res["scaler"] = res["param_pre__num"].map(nome_scaler)
        for sc, g in res.groupby("scaler"):
            comp_scalers.append({"algoritmo": nome, "scaler": sc,
                                 "melhor_rmse_validacao_cv": -g["mean_test_score"].max()})

    if nome != "Baseline (média)":
        df.loc[X_te.index, f"pred_{nome}"] = y_pred

res = pd.DataFrame(linhas)
res.to_csv(OUT_RES / "resultados_individuais.csv", index=False)
pd.DataFrame(melhores).to_csv(OUT_RES / "melhores_hiperparametros.csv", index=False)
cs = pd.DataFrame(comp_scalers).pivot(index="algoritmo", columns="scaler",
                                      values="melhor_rmse_validacao_cv")
cs.to_csv(OUT_RES / "comparacao_scalers.csv")
pd.DataFrame(folds).to_csv(OUT_RES / "cv_folds_rmse.csv", index=False)

print("\n=== Resultados (teste) ===")
print(res.round(4).to_string(index=False))
print("\n=== RMSE de validação por scaler ===")
print(cs.round(4).to_string())

# ---------------------------------------------------------------
# 3) Figuras
# ---------------------------------------------------------------
sns.set_theme(style="whitegrid")
ordem = res.sort_values("rmse_validacao_cv")["algoritmo"].tolist()
ordem_sem_base = [a for a in ordem if a != "Baseline (média)"]

# 3.1 Boxplot do RMSE nos 5 folds de validação
dfo = pd.DataFrame(folds).melt(var_name="Algoritmo", value_name="RMSE")
dfo = dfo[dfo["Algoritmo"] != "Baseline (média)"]
fig, ax = plt.subplots(figsize=(8, 4.5))
sns.boxplot(data=dfo, x="Algoritmo", y="RMSE", order=ordem_sem_base, color="#9ecae1", ax=ax)
sns.stripplot(data=dfo, x="Algoritmo", y="RMSE", order=ordem_sem_base, color="#08519c", size=4, ax=ax)
base = res.loc[res.algoritmo == "Baseline (média)", "rmse_validacao_cv"].iloc[0]
ax.axhline(base, ls="--", color="gray", lw=1)
ax.text(len(ordem_sem_base) - 0.5, base, "baseline (média)", ha="right", va="bottom", fontsize=8, color="gray")
ax.set_xlabel("")
ax.set_ylabel("RMSE na validação (p.p.)")
ax.set_title("RMSE nos 5 folds da validação cruzada (melhor configuração)")
plt.xticks(rotation=20, ha="right")
fig.tight_layout()
fig.savefig(OUT_FIG / "s3_boxplot_cv_rmse.png", dpi=300)
plt.close(fig)

# 3.2 Real x previsto do melhor modelo (menor RMSE de validação)
melhor = ordem_sem_base[0]
yp = df.loc[X_te.index, f"pred_{melhor}"]
fig, ax = plt.subplots(figsize=(6, 5.5))
cores = dict(zip(sorted(df["regiao"].unique()), sns.color_palette("tab10", 5)))
for reg, g in df.loc[X_te.index].groupby("regiao"):
    ax.scatter(g[ALVO], g[f"pred_{melhor}"], s=18, alpha=0.75, label=reg, color=cores[reg])
lim = [min(y_te.min(), yp.min()) - 0.5, max(y_te.max(), yp.max()) + 0.5]
ax.plot(lim, lim, "k--", lw=1)
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("Crescimento real 2024-2026 (%)")
ax.set_ylabel("Crescimento previsto (%)")
ax.set_title(f"Real x previsto no conjunto de teste - {melhor}")
ax.legend(title="Região", fontsize=8)
fig.tight_layout()
fig.savefig(OUT_FIG / "s3_real_vs_previsto.png", dpi=300)
plt.close(fig)

# 3.3 Resíduos do melhor modelo
resid = y_te - yp
fig, ax = plt.subplots(figsize=(7, 4.2))
ax.scatter(yp, resid, s=16, alpha=0.7, color="#3182bd")
ax.axhline(0, color="k", lw=1, ls="--")
ax.set_xlabel("Crescimento previsto (%)")
ax.set_ylabel("Resíduo (real - previsto, p.p.)")
ax.set_title(f"Resíduos no conjunto de teste - {melhor}")
fig.tight_layout()
fig.savefig(OUT_FIG / "s3_residuos.png", dpi=300)
plt.close(fig)

# Maiores erros absolutos do melhor modelo (para a discussão)
top = df.loc[X_te.index, ["municipio", "uf", "regiao_metropolitana", ALVO, f"pred_{melhor}"]].copy()
top["erro_abs"] = (top[ALVO] - top[f"pred_{melhor}"]).abs()
top.sort_values("erro_abs", ascending=False).head(10).to_csv(OUT_RES / "maiores_erros_melhor_modelo.csv", index=False)
print(f"\nMelhor modelo (validação): {melhor}")
print(top.sort_values("erro_abs", ascending=False).head(10).round(3).to_string(index=False))
