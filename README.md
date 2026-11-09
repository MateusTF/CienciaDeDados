# CienciaDeDados

Previsão do crescimento da população infantil (0 a 4 anos) nas regiões metropolitanas brasileiras.

## Ordem de execução

| Script | Sprint | O que faz |
|---|---|---|
| `01_carregar_dados.py` | 1 | Carrega e junta as Tabelas 6579 e 9514 do SIDRA → `df_municipios.csv` |
| `02_analise_exploratoria.py` | 1 | Estatísticas descritivas e figuras da análise exploratória (`figuras/`) |
| `03_pre_processamento.py` | 2/3 | Filtro metropolitano, variável-alvo, remoção de vazamento, log10, `cresc_pop_2022_2024_pct` → `df_pre_processado.csv` (sem normalização) |
| `04_algoritmos_individuais.py` | 3 | Divisão 80/20 estratificada, validação cruzada 5-fold, busca em grade e avaliação de 6 algoritmos + baseline → `resultados/` e `figuras/s3_*.png` |

```bash
pip install -r requirements.txt
python 01_carregar_dados.py
python 02_analise_exploratoria.py
python 03_pre_processamento.py
python 04_algoritmos_individuais.py
```

A normalização é feita dentro do `Pipeline` do scikit-learn (ajustada só no treino de cada fold); o tipo de scaler
(StandardScaler, MinMaxScaler, RobustScaler) é escolhido na busca em grade. Semente aleatória: 42.
