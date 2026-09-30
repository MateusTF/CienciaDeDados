# -*- coding: utf-8 -*-
"""
01_carregar_dados.py

Carrega, limpa e junta as duas bases do SIDRA/IBGE utilizadas no projeto:

  - Tabela 6579 - População residente estimada (municípios, 2024-2026)
  - Tabela 9514 - População residente por sexo, idade e forma de
    declaração da idade (Censo 2022, total e faixa 0 a 4 anos)

Saída: df_municipios.csv, um dataframe único em nível de município com
as colunas usadas na análise exploratória (Sprint 1).
"""

import pandas as pd
import numpy as np
import re

PASTA = "/mnt/user-data/uploads/"

# ---------------------------------------------------------------
# 1) Tabela 6579 - População residente estimada (2024, 2025, 2026)
# ---------------------------------------------------------------
raw = pd.read_csv(
    PASTA + "População_residente_estimada.csv",
    sep=";",
    encoding="utf-8-sig",
    skiprows=4,          # pula as 4 linhas de cabeçalho/metadados do SIDRA
    header=None,
    dtype=str,
    engine="python",
)

# Colunas: Nível; Cód.; Local; 2024; unidade; 2025; unidade; 2026; unidade
raw = raw.iloc[:, [0, 1, 2, 3, 5, 7]]
raw.columns = ["nivel", "cod_municipio", "municipio_uf", "pop_2024", "pop_2025", "pop_2026"]

# Mantém apenas linhas de município ("MU") e remove notas de rodapé do SIDRA
est = raw[raw["nivel"] == "MU"].copy()

for col in ["pop_2024", "pop_2025", "pop_2026"]:
    est[col] = pd.to_numeric(est[col], errors="coerce")

est["cod_municipio"] = est["cod_municipio"].astype(str)

# Extrai a UF a partir do sufixo "(UF)" do nome do município
est["uf"] = est["municipio_uf"].str.extract(r"\(([A-Z]{2})\)$")
est["municipio"] = est["municipio_uf"].str.replace(r"\s*\([A-Z]{2}\)$", "", regex=True)

est = est.drop(columns=["nivel", "municipio_uf"]).dropna(subset=["pop_2024", "pop_2025", "pop_2026"])

print(f"Tabela 6579 (estimativas): {len(est)} municípios válidos")

# ---------------------------------------------------------------
# 2) Tabela 9514 - Censo 2022: população total e 0-4 anos
# ---------------------------------------------------------------
raw2 = pd.read_csv(
    PASTA + "População_residente__por_sexo__idade_e_forma_de_declaração_da_idade.csv",
    sep=";",
    encoding="utf-8-sig",
    skiprows=6,          # pula as 6 linhas de cabeçalho/metadados do SIDRA
    header=None,
    dtype=str,
    engine="python",
)

# Colunas: Nível; Cód.; Local; Forma decl.; Total; unid; 0-4 anos; unid
raw2 = raw2.iloc[:, [0, 1, 2, 3, 4, 6]]
raw2.columns = ["nivel", "cod_municipio", "municipio_uf", "forma_declaracao", "pop_total_2022", "pop_0a4_2022"]

censo = raw2[(raw2["nivel"] == "MU") & (raw2["forma_declaracao"] == "Total")].copy()

for col in ["pop_total_2022", "pop_0a4_2022"]:
    censo[col] = pd.to_numeric(censo[col], errors="coerce")

censo["cod_municipio"] = censo["cod_municipio"].astype(str)
censo = censo.drop(columns=["nivel", "municipio_uf", "forma_declaracao"]).dropna(
    subset=["pop_total_2022", "pop_0a4_2022"]
)

print(f"Tabela 9514 (Censo 2022): {len(censo)} municípios válidos")

# ---------------------------------------------------------------
# 3) Junção das duas bases por código do município (IBGE)
# ---------------------------------------------------------------
df = est.merge(censo, on="cod_municipio", how="inner")
print(f"Base final após junção: {len(df)} municípios")

# Mapeamento de UF -> Grande Região (usado nas análises por região)
REGIAO = {
    **{uf: "Norte" for uf in ["AC", "AP", "AM", "PA", "RO", "RR", "TO"]},
    **{uf: "Nordeste" for uf in ["AL", "BA", "CE", "MA", "PB", "PE", "PI", "RN", "SE"]},
    **{uf: "Sudeste" for uf in ["ES", "MG", "RJ", "SP"]},
    **{uf: "Sul" for uf in ["PR", "RS", "SC"]},
    **{uf: "Centro-Oeste" for uf in ["DF", "GO", "MS", "MT"]},
}
df["regiao"] = df["uf"].map(REGIAO)

# ---------------------------------------------------------------
# 4) Variáveis derivadas
# ---------------------------------------------------------------
# Proporção de crianças de 0 a 4 anos na população total (Censo 2022)
df["prop_0a4_2022"] = df["pop_0a4_2022"] / df["pop_total_2022"]

# Taxa de crescimento da população TOTAL entre 2024 e 2026 (estimativas)
df["cresc_pop_total_2024_2026_pct"] = (df["pop_2026"] / df["pop_2024"] - 1) * 100

# Estimativa da população 0-4 anos em 2024 e 2026, aplicando a proporção
# do Censo 2022 sobre a população total estimada de cada ano.
# ATENÇÃO (limitação documentada no relatório): esta é uma aproximação -
# assume proporção etária constante no curto intervalo 2022-2026, já que
# a base de idade só está disponível no ano censitário (2022). Não há,
# nos arquivos fornecidos, um segundo ano censitário (ex.: 2010) nem a
# série completa 2001-2025 mencionadas no esboço inicial do projeto.
df["pop_0a4_est_2024"] = df["prop_0a4_2022"] * df["pop_2024"]
df["pop_0a4_est_2026"] = df["prop_0a4_2022"] * df["pop_2026"]
df["cresc_pop_0a4_est_2024_2026_pct"] = (
    df["pop_0a4_est_2026"] / df["pop_0a4_est_2024"] - 1
) * 100

# ---------------------------------------------------------------
# 5) Salva base final
# ---------------------------------------------------------------
cols_finais = [
    "cod_municipio", "municipio", "uf", "regiao",
    "pop_total_2022", "pop_0a4_2022", "prop_0a4_2022",
    "pop_2024", "pop_2025", "pop_2026",
    "cresc_pop_total_2024_2026_pct",
    "pop_0a4_est_2024", "pop_0a4_est_2026", "cresc_pop_0a4_est_2024_2026_pct",
]
df = df[cols_finais].sort_values("pop_2026", ascending=False).reset_index(drop=True)
df.to_csv("/home/claude/proj/df_municipios.csv", index=False)

print("\nAmostra da base final:")
print(df.head(10).to_string(index=False))
print("\nResumo estatístico:")
print(df.describe().to_string())
