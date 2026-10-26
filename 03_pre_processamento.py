# -*- coding: utf-8 -*-
"""
03_pre_processamento.py

Pré-processamento da base de municípios (Sprint 2 - Filtro e Criação da
Variável-Alvo):

  1. Carrega df_municipios.csv (base principal) e regioes_metropolitanas.csv
     (base de filtro, na raiz do projeto).
  2. Inner merge pela coluna cod_municipio: mantém só municípios de RMs.
  3. Calcula a variável-alvo cresc_pop_0a4_pct (crescimento % da população
     de 0 a 4 anos).
  4. Remove colunas que causam vazamento de dados (drop).
  5. Aplica log10 em pop_total_2022.
  6. Isola a variável-alvo e padroniza (StandardScaler) as features numéricas.
  7. Trata nulos e exporta df_pre_processado.csv.

Colunas usadas de regioes_metropolitanas.csv: cod_municipio e
regiao_metropolitana (as demais são ignoradas). Municípios que aparecem em
mais de um recorte metropolitano são mantidos uma única vez (primeira
ocorrência), para que o merge não duplique linhas.

Uso:
    python 03_pre_processamento.py [caminho_regioes_metropolitanas.csv]
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent

ARQ_MUNICIPIOS = BASE_DIR / "df_municipios.csv"
ARQ_RM = Path(sys.argv[1]) if len(sys.argv) > 1 else BASE_DIR / "regioes_metropolitanas.csv"
ARQ_SAIDA = BASE_DIR / "df_pre_processado.csv"

ALVO = "cresc_pop_0a4_pct"

# Colunas que reproduzem o alvo (ou o seu numerador/denominador estimado) e
# não podem ser preditoras. cresc_pop_total_2024_2026_pct é igual ao alvo
# porque pop_0a4_est_* = prop_0a4_2022 * pop_* (ver 01_carregar_dados.py).
COLUNAS_VAZAMENTO = [
    "cresc_pop_total_2024_2026_pct",
    "cresc_pop_0a4_est_2024_2026_pct",
    "pop_0a4_est_2024",
    "pop_0a4_est_2026",
]

# Colunas de identificação: ficam no arquivo final, mas não são padronizadas.
COLUNAS_ID = ["cod_municipio", "municipio", "uf", "regiao", "regiao_metropolitana"]


# ---------------------------------------------------------------
# 1) Carregamento das bases
# ---------------------------------------------------------------
if not ARQ_MUNICIPIOS.exists():
    sys.exit(f"ERRO: '{ARQ_MUNICIPIOS.name}' não encontrado. Execute antes 01_carregar_dados.py.")
if not ARQ_RM.exists():
    sys.exit(f"ERRO: '{ARQ_RM}' não encontrado (esperado na raiz do projeto).")

df = pd.read_csv(ARQ_MUNICIPIOS, dtype={"cod_municipio": str})
print(f"df_municipios.csv: {len(df)} municípios")


def ler_rm(caminho: Path) -> pd.DataFrame:
    """Lê o CSV de RMs (UTF-8 ou, na falha, Windows-1252/Latin-1)."""
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            return pd.read_csv(caminho, sep=None, engine="python", encoding=enc, dtype=str)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Não foi possível decodificar {caminho.name}")


rm = ler_rm(ARQ_RM)
faltando = {"cod_municipio", "regiao_metropolitana"} - set(rm.columns)
if faltando:
    sys.exit(f"ERRO: colunas ausentes em '{ARQ_RM.name}': {sorted(faltando)}")

rm = rm[["cod_municipio", "regiao_metropolitana"]].copy()
rm["cod_municipio"] = rm["cod_municipio"].str.strip()
n_linhas_rm = len(rm)
rm = rm.drop_duplicates(subset="cod_municipio", keep="first")
print(f"regioes_metropolitanas.csv: {n_linhas_rm} linhas, {len(rm)} municípios únicos")

# ---------------------------------------------------------------
# 2) Inner merge por cod_municipio (filtra apenas municípios de RMs)
# ---------------------------------------------------------------
df = df.merge(rm, on="cod_municipio", how="inner", validate="one_to_one")
print(f"Após inner merge: {len(df)} municípios em Regiões Metropolitanas")

sem_match = set(rm["cod_municipio"]) - set(df["cod_municipio"])
if sem_match:
    print(f"AVISO: {len(sem_match)} código(s) de RM sem correspondência em df_municipios.csv "
          f"(ex.: {sorted(sem_match)[:5]})")
if df.empty:
    sys.exit("ERRO: o merge não retornou nenhum município.")

# ---------------------------------------------------------------
# 3) Variável-alvo: crescimento (%) da população de 0 a 4 anos (2024-2026)
# ---------------------------------------------------------------
df[ALVO] = (df["pop_0a4_est_2026"] / df["pop_0a4_est_2024"] - 1) * 100

# ---------------------------------------------------------------
# 4) Remove colunas que vazam o alvo (drop)
# ---------------------------------------------------------------
df = df.drop(columns=COLUNAS_VAZAMENTO)

# ---------------------------------------------------------------
# 5) Transformação log10 em pop_total_2022 (distribuição muito assimétrica)
# ---------------------------------------------------------------
df.loc[df["pop_total_2022"] <= 0, "pop_total_2022"] = np.nan  # log10 só definido para > 0
df["pop_total_2022_log10"] = np.log10(df["pop_total_2022"])
df = df.drop(columns=["pop_total_2022"])

# ---------------------------------------------------------------
# 6) Tratamento de nulos
# ---------------------------------------------------------------
df = df.replace([np.inf, -np.inf], np.nan)

features_num = [c for c in df.select_dtypes(include="number").columns if c != ALVO]

nulos = df[features_num + [ALVO] + ["regiao_metropolitana"]].isna().sum()
nulos = nulos[nulos > 0]
if nulos.empty:
    print("Nulos: nenhum valor nulo encontrado.")
else:
    print("Nulos encontrados (por coluna):")
    print(nulos.to_string())

# Alvo ou RM ausentes: a linha não serve para o modelo -> removida.
antes = len(df)
df = df.dropna(subset=[ALVO, "regiao_metropolitana"])
if antes != len(df):
    print(f"{antes - len(df)} linha(s) removida(s) por alvo/RM ausente.")

# Features numéricas ausentes: imputação pela mediana (robusta a outliers).
for col in features_num:
    if df[col].isna().any():
        mediana = df[col].median()
        print(f"Imputando {df[col].isna().sum()} nulo(s) de '{col}' com a mediana ({mediana:.4f}).")
        df[col] = df[col].fillna(mediana)

# ---------------------------------------------------------------
# 7) Isola a variável-alvo e padroniza as features numéricas
# ---------------------------------------------------------------
y = df[ALVO]
X = df[features_num]

scaler = StandardScaler()
X_pad = pd.DataFrame(scaler.fit_transform(X), columns=features_num, index=df.index)

# Obs.: o scaler é ajustado em toda a base. Ao treinar o modelo, ajuste-o
# apenas no conjunto de treino para não vazar informação do teste.

# ---------------------------------------------------------------
# 8) Exporta df_pre_processado.csv (ids + features padronizadas + alvo)
# ---------------------------------------------------------------
ids = df[[c for c in COLUNAS_ID if c in df.columns]]
df_final = pd.concat([ids, X_pad, y], axis=1).reset_index(drop=True)
df_final.to_csv(ARQ_SAIDA, index=False)

print(f"\nArquivo salvo: {ARQ_SAIDA.name} ({df_final.shape[0]} linhas x {df_final.shape[1]} colunas)")
print(f"Features padronizadas ({len(features_num)}): {features_num}")
print(f"Variável-alvo: {ALVO}")
print("\nVerificação do scaler (média ~0, desvio ~1):")
print(X_pad.agg(["mean", "std"]).round(4).T.to_string())
print(f"\nResumo da variável-alvo (%):\n{y.describe().round(4).to_string()}")
