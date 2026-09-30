# -*- coding: utf-8 -*-
"""
02_analise_exploratoria.py

Análise exploratória e descritiva da base de municípios (Sprint 1):
  - Estatísticas descritivas gerais e por região
  - Histogramas (população total e proporção de crianças 0-4 anos)
  - Boxplot da proporção 0-4 anos por Grande Região
  - Gráfico de dispersão: população total x proporção 0-4 anos
  - Gráfico de dispersão: proporção 0-4 anos x crescimento populacional
  - Matriz de correlação

Todas as figuras são salvas em /home/claude/proj/figuras/ como PNG
(300 dpi) para uso no relatório (docx).
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_theme(style="whitegrid", font_scale=1.0)
OUT = "/home/claude/proj/figuras"
os.makedirs(OUT, exist_ok=True)

df = pd.read_csv("/home/claude/proj/df_municipios.csv")

# =================================================================
# 1) Estatísticas descritivas gerais
# =================================================================
desc_geral = df[[
    "pop_total_2022", "pop_0a4_2022", "prop_0a4_2022",
    "pop_2024", "pop_2026", "cresc_pop_total_2024_2026_pct"
]].describe().T
desc_geral.to_csv("/home/claude/proj/estatisticas_descritivas_geral.csv")
print("=== Estatísticas descritivas gerais ===")
print(desc_geral.round(3).to_string())

# Estatísticas por região
desc_regiao = df.groupby("regiao")[
    ["pop_total_2022", "prop_0a4_2022", "cresc_pop_total_2024_2026_pct"]
].agg(["mean", "median", "std", "count"])
desc_regiao.to_csv("/home/claude/proj/estatisticas_por_regiao.csv")
print("\n=== Estatísticas por região ===")
print(desc_regiao.round(4).to_string())

# Contagem de municípios com crescimento negativo de população total
n_decrescimo = (df["cresc_pop_total_2024_2026_pct"] < 0).sum()
pct_decrescimo = n_decrescimo / len(df) * 100
print(f"\nMunicípios com crescimento populacional negativo (2024-2026): "
      f"{n_decrescimo} ({pct_decrescimo:.1f}%)")

# =================================================================
# 2) Histograma - População total 2022 (escala log, dado o forte
#    desvio à direita causado por capitais e grandes cidades)
# =================================================================
fig, ax = plt.subplots(figsize=(8, 5))
sns.histplot(np.log10(df["pop_total_2022"]), bins=40, kde=True, ax=ax, color="#2b6cb0")
ax.set_xlabel("log10(população total do município, Censo 2022)")
ax.set_ylabel("Número de municípios")
ax.set_title("Distribuição da população total dos municípios (Censo 2022)")
fig.tight_layout()
fig.savefig(f"{OUT}/hist_populacao_total.png", dpi=300)
plt.close(fig)

# =================================================================
# 3) Histograma - Proporção de crianças 0-4 anos
# =================================================================
fig, ax = plt.subplots(figsize=(8, 5))
sns.histplot(df["prop_0a4_2022"] * 100, bins=40, kde=True, ax=ax, color="#2f855a")
ax.set_xlabel("Proporção de crianças de 0 a 4 anos (%)")
ax.set_ylabel("Número de municípios")
ax.set_title("Distribuição da proporção de crianças 0-4 anos por município (Censo 2022)")
fig.tight_layout()
fig.savefig(f"{OUT}/hist_proporcao_0a4.png", dpi=300)
plt.close(fig)

# =================================================================
# 4) Boxplot - Proporção 0-4 anos por Grande Região
# =================================================================
ordem_regiao = df.groupby("regiao")["prop_0a4_2022"].median().sort_values(ascending=False).index
fig, ax = plt.subplots(figsize=(8, 5))
sns.boxplot(data=df, x="regiao", y=df["prop_0a4_2022"] * 100, order=ordem_regiao,
            hue="regiao", legend=False, palette="Blues_r", ax=ax)
ax.set_xlabel("Grande Região")
ax.set_ylabel("Proporção de crianças 0-4 anos (%)")
ax.set_title("Proporção de crianças 0-4 anos por Grande Região (Censo 2022)")
fig.tight_layout()
fig.savefig(f"{OUT}/boxplot_regiao.png", dpi=300)
plt.close(fig)

# =================================================================
# 5) Dispersão - População total x Proporção 0-4 anos
# =================================================================
fig, ax = plt.subplots(figsize=(8, 5))
sns.scatterplot(
    data=df, x=np.log10(df["pop_total_2022"]), y=df["prop_0a4_2022"] * 100,
    hue="regiao", alpha=0.45, s=18, ax=ax
)
ax.set_xlabel("log10(população total do município)")
ax.set_ylabel("Proporção de crianças 0-4 anos (%)")
ax.set_title("População total x Proporção de crianças 0-4 anos")
ax.legend(title="Região", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
fig.tight_layout()
fig.savefig(f"{OUT}/scatter_pop_x_proporcao.png", dpi=300)
plt.close(fig)

# =================================================================
# 6) Dispersão - Proporção 0-4 anos x Crescimento pop. total 2024-2026
# =================================================================
df_plot = df[df["cresc_pop_total_2024_2026_pct"].between(-15, 15)]  # remove outliers extremos p/ leitura
fig, ax = plt.subplots(figsize=(8, 5))
sns.scatterplot(
    data=df_plot, x=df_plot["prop_0a4_2022"] * 100, y="cresc_pop_total_2024_2026_pct",
    hue="regiao", alpha=0.45, s=18, ax=ax
)
ax.axhline(0, color="gray", linestyle="--", linewidth=1)
ax.set_xlabel("Proporção de crianças 0-4 anos em 2022 (%)")
ax.set_ylabel("Crescimento da população total 2024-2026 (%)")
ax.set_title("Proporção de crianças 0-4 anos x Crescimento populacional recente")
ax.legend(title="Região", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
fig.tight_layout()
fig.savefig(f"{OUT}/scatter_proporcao_x_crescimento.png", dpi=300)
plt.close(fig)

# =================================================================
# 7) Matriz de correlação
# =================================================================
cols_corr = [
    "pop_total_2022", "pop_0a4_2022", "prop_0a4_2022",
    "pop_2024", "pop_2026", "cresc_pop_total_2024_2026_pct"
]
labels_corr = [
    "Pop. total 2022", "Pop. 0-4 2022", "Prop. 0-4 2022",
    "Pop. total 2024", "Pop. total 2026", "Cresc. pop. total (%)"
]
corr = df[cols_corr].corr()
corr.index = labels_corr
corr.columns = labels_corr
corr.to_csv("/home/claude/proj/matriz_correlacao.csv")

fig, ax = plt.subplots(figsize=(7.5, 6.5))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1,
            square=True, linewidths=0.5, ax=ax, cbar_kws={"label": "Correlação (Pearson)"})
ax.set_title("Matriz de correlação entre variáveis")
fig.tight_layout()
fig.savefig(f"{OUT}/matriz_correlacao.png", dpi=300)
plt.close(fig)

print("\n=== Matriz de correlação ===")
print(corr.round(3).to_string())

print("\nFiguras salvas em:", OUT)
print(os.listdir(OUT))
