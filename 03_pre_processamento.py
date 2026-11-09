# -*- coding: utf-8 -*-
"""
03_pre_processamento.py

Pré-processamento da base de municípios (revisado na Sprint 3):

  1. Carrega df_municipios.csv (base principal) e regioes_metropolitanas.csv
     (base de filtro, na raiz do projeto).
  2. Inner merge pela coluna cod_municipio: mantém só municípios de RMs.
  3. Calcula a variável-alvo cresc_pop_0a4_pct (crescimento % da população
     de 0 a 4 anos).
  4. Cria a feature cresc_pop_2022_2024_pct (crescimento observado antes do
     período do alvo).
  5. Remove colunas que causam vazamento de dados ou são redundantes (drop).
  6. Aplica log10 em pop_total_2022.
  7. Trata nulos e exporta df_pre_processado.csv SEM padronização.

A normalização das features NÃO é feita aqui: ela passou para dentro do
Pipeline de modelagem (04_algoritmos_individuais.py), onde o scaler é
ajustado apenas nos dados de treino de cada fold e o tipo de scaler
(StandardScaler, MinMaxScaler, RobustScaler) é avaliado como hiperparâmetro.

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

BASE_DIR = Path(__file__).resolve().parent

ARQ_MUNICIPIOS = BASE_DIR / "df_municipios.csv"
ARQ_RM = Path(sys.argv[1]) if len(sys.argv) > 1 else BASE_DIR / "regioes_metropolitanas.csv"
ARQ_SAIDA = BASE_DIR / "df_pre_processado.csv"

ALVO = "cresc_pop_0a4_pct"

# Colunas que reproduzem o alvo (ou o seu numerador/denominador estimado) e
# não podem ser preditoras. cresc_pop_total_2024_2026_pct é igual ao alvo
# porque pop_0a4_est_* = prop_0a4_2022 * pop_* (ver 01_carregar_dados.py).
# pop_2025 e pop_2026 também são removidas: o alvo é calculado diretamente a
# partir de pop_2026 / pop_2024, então mantê-las permitiria ao modelo
# "recalcular" o alvo em vez de prevê-lo. Só entram como features
# informações disponíveis até 2024 (início do horizonte de previsão).
COLUNAS_VAZAMENTO = [
    "cresc_pop_total_2024_2026_pct",
    "cresc_pop_0a4_est_2024_2026_pct",
    "pop_0a4_est_2024",
    "pop_0a4_est_2026",
    "pop_2025",
    "pop_2026",
]

# Colunas redundantes: pop_0a4_2022 tem correlação 0,996 com pop_total_2022 e
# pop_2024 é usada apenas para construir cresc_pop_2022_2024_pct.
COLUNAS_REDUNDANTES = ["pop_0a4_2022", "pop_2024"]

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
# 4) Feature de tendência recente: crescimento % da população 2022 -> 2024
#    (período anterior ao do alvo, portanto sem vazamento)
# ---------------------------------------------------------------
df["cresc_pop_2022_2024_pct"] = (df["pop_2024"] / df["pop_total_2022"] - 1) * 100

# ---------------------------------------------------------------
# 5) Remove colunas que vazam o alvo ou são redundantes (drop)
# ---------------------------------------------------------------
df = df.drop(columns=COLUNAS_VAZAMENTO + COLUNAS_REDUNDANTES)

# ---------------------------------------------------------------
# 6) Transformação log10 em pop_total_2022 (distribuição muito assimétrica)
# ---------------------------------------------------------------
df.loc[df["pop_total_2022"] <= 0, "pop_total_2022"] = np.nan  # log10 só definido para > 0
df["pop_total_2022_log10"] = np.log10(df["pop_total_2022"])
df = df.drop(columns=["pop_total_2022"])

# ---------------------------------------------------------------
# 7) Tratamento de nulos
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
# 8) Exporta df_pre_processado.csv (ids + features SEM padronização + alvo)
# ---------------------------------------------------------------
ids = df[[c for c in COLUNAS_ID if c in df.columns]]
df_final = pd.concat([ids, df[features_num], df[ALVO]], axis=1).reset_index(drop=True)
df_final.to_csv(ARQ_SAIDA, index=False)

print(f"\nArquivo salvo: {ARQ_SAIDA.name} ({df_final.shape[0]} linhas x {df_final.shape[1]} colunas)")
print(f"Features numéricas ({len(features_num)}): {features_num}")
print(f"Variável-alvo: {ALVO}")
print(f"\nResumo da variável-alvo (%):\n{df[ALVO].describe().round(4).to_string()}")
