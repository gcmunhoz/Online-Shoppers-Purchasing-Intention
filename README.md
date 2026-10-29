# Online Shoppers — Intenção de Compra

Classificador que prevê se uma sessão de e-commerce termina em compra (`Revenue`), treinado
sobre o dataset [Online Shoppers Purchasing Intention](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset)
(12.330 sessões, ~15% de compras).

- **Modelo:** Random Forest (scikit-learn) com busca aleatória de hiperparâmetros
  (otimizando PR-AUC) e limiar de decisão escolhido para maximizar o F1.
- **Dependências:** [uv](https://docs.astral.sh/uv/) (`pyproject.toml` + `uv.lock`).
- **Dados e pipeline:** [DVC](https://dvc.org/) (`dvc.yaml`: `preprocess → train`).
- **Experimentos e modelos:** [MLflow](https://mlflow.org/) Tracking + Model Registry.
- **Execução isolada:** Docker.

## Estrutura

```
.
├── configs/params.yaml      # parâmetros do pipeline (versionados pelo DVC)
├── data/
│   ├── raw/                 # dataset original (versionado pelo DVC: *.csv.dvc)
│   └── processed/           # train.csv / test.csv (saída do estágio preprocess)
├── models/                  # modelo treinado .joblib (saída do estágio train)
├── reports/metrics.json     # métricas no teste (dvc metrics show)
├── notebooks/EDA.ipynb      # análise exploratória
├── src/projetofase2/
│   ├── config.py            # caminhos, esquema de colunas, params.yaml e .env
│   ├── data.py              # carga, limpeza e divisão treino/teste
│   ├── model.py             # pipeline sklearn, tuning, limiar e avaliação
│   ├── tracking.py          # MLflow Tracking (params, métricas, artefatos, modelo)
│   ├── registry.py          # MLflow Model Registry (promoção da melhor versão)
│   ├── preprocess.py        # estágio DVC: preprocess
│   └── train.py             # estágio DVC: train
├── tests/                   # pytest (dados sintéticos; MLflow isolado por teste)
├── dvc.yaml / dvc.lock      # definição e estado do pipeline
├── Dockerfile
├── pyproject.toml / uv.lock
└── .env.example             # modelo das variáveis de ambiente
```

## Pré-requisitos

- Python 3.13 e [uv](https://docs.astral.sh/uv/getting-started/installation/)
- Git
- Docker (opcional, para rodar em container)

## Instalação

```bash
git clone https://github.com/gcmunhoz/Online-Shoppers-Purchasing-Intention.git
cd Online-Shoppers-Purchasing-Intention
uv sync                      # cria .venv com dependências de produção + dev (lock exato)
cp .env.example .env         # ajuste se necessário
```

As dependências ficam separadas no `pyproject.toml`:

| Grupo | Pacotes | Instalação |
|---|---|---|
| produção (`[project.dependencies]`) | scikit-learn, mlflow, pandas, numpy, joblib, python-dotenv, pyyaml | `uv sync --no-dev` |
| desenvolvimento (`[dependency-groups].dev`) | pytest, ruff, dvc, seaborn, ipykernel | incluído em `uv sync` |

## Configuração

- **`configs/params.yaml`** — parâmetros de modelagem: `random_state`, tamanho do teste,
  folds de CV, número de iterações e espaço de busca dos hiperparâmetros, caminhos de saída.
  Alterar um valor aqui faz o `dvc repro` reexecutar os estágios afetados.
- **`.env`** — configurações de ambiente do MLflow (veja `.env.example`). Variáveis
  definidas no shell têm prioridade sobre o `.env`.

| Variável | Padrão | Descrição |
|---|---|---|
| `MLFLOW_TRACKING_URI` | `sqlite:///<raiz>/mlflow_data/mlflow.db` | Backend do MLflow |
| `MLFLOW_EXPERIMENT_NAME` | `online-shoppers-purchase-intention` | Experimento das runs |
| `MLFLOW_MODEL_NAME` | `rf_online_shoppers` | Nome no Model Registry |
| `MLFLOW_REGISTER_MODEL` | `true` | Registrar cada run como nova versão |
| `MLFLOW_CHAMPION_ALIAS` | `champion` | Alias da melhor versão |
| `MLFLOW_SELECTION_METRIC` | `test_pr_auc` | Métrica (maior = melhor) para escolher a campeã |

## Dados com DVC

O dataset fica em `data/raw/online_shoppers_intention.csv` e é versionado pelo DVC: o git
guarda só o `data/raw/online_shoppers_intention.csv.dvc`, com o hash md5 do arquivo. O CSV
em si **não está no repositório**: baixe-o do Google Drive antes de rodar o pipeline.

### Obtendo o dataset

1. Baixe `online_shoppers_intention.csv` do Google Drive:
   <https://drive.google.com/file/d/1DXJaPOx0EuSEnptJHUtcfIy0rPfurzwQ/view?usp=sharing>
2. Salve o arquivo exatamente em `data/raw/online_shoppers_intention.csv`.
3. (Opcional) Confira que é a mesma versão registrada no `.dvc` — o md5 deve ser
   `cc6ec1db03b4f10f8de52c56ff48b085`:

   ```bash
   uv run dvc status   # sem "changed outs" para data/raw = arquivo idêntico
   ```

   No PowerShell também é possível conferir diretamente:
   `(Get-FileHash data\raw\online_shoppers_intention.csv -Algorithm MD5).Hash`

4. Gere os splits e o modelo executando o pipeline (veja [Pipeline](#pipeline-dvc)):

   ```bash
   uv run dvc repro
   ```

Como o `random_state` é fixo em `configs/params.yaml`, o pipeline reproduz os mesmos
splits e o mesmo modelo registrados no `dvc.lock`.

### Nova versão do dataset

Ao substituir o CSV, registre a nova versão no DVC (o `.dvc` passa a ter o novo md5) e
reexecute o pipeline:

```bash
uv run dvc add data/raw/online_shoppers_intention.csv
uv run dvc repro
git add data/raw/online_shoppers_intention.csv.dvc dvc.lock reports/metrics.json
```

## Pipeline (DVC)

```
data/raw/*.csv ──► preprocess ──► data/processed/{train,test}.csv ──► train ──► models/*.joblib
                                                                          ├──► reports/metrics.json
                                                                          └──► MLflow (run + Model Registry)
```

| Estágio | Comando | O que faz |
|---|---|---|
| `preprocess` | `python -m projetofase2.preprocess` | remove duplicatas, ajusta tipos e divide treino/teste estratificado |
| `train` | `python -m projetofase2.train` | tuning (RandomizedSearchCV), limiar ótimo, avaliação, log no MLflow, registro e promoção do modelo |

```bash
uv run dvc repro             # executa só os estágios desatualizados
uv run dvc repro --force     # reexecuta tudo (gera uma nova run no MLflow)
uv run dvc dag               # mostra o grafo do pipeline
uv run dvc metrics show      # métricas do último treino
```

Os estágios também podem ser executados diretamente:

```bash
uv run preprocess
uv run train --run-name meu-experimento   # --no-register para não registrar no Registry
```

## MLflow

Cada execução do `train` cria uma run com:

- **parâmetros:** `random_state`, `test_size`, `cv_splits`, `n_iter`, `scoring`, tamanhos
  dos conjuntos e os melhores hiperparâmetros (`rf__*`);
- **métricas:** ROC-AUC, PR-AUC, F1, precisão, recall, acurácia, matriz de confusão,
  limiar de decisão e score da validação cruzada;
- **tags:** md5 do dataset bruto (igual ao do `.dvc`, ligando a run à versão dos dados);
- **artefatos:** relatório de classificação, importâncias das features, resultados do
  tuning, o `.joblib` e o modelo MLflow (com signature e o limiar embutido no `predict`).

**Model Registry:** cada run registra uma nova versão de `rf_online_shoppers`. A versão com
maior `test_pr_auc` recebe o alias **`champion`** — se a nova versão não superar a atual,
o alias permanece onde está.

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow_data/mlflow.db
# abra http://127.0.0.1:5000
```

Carregando o melhor modelo:

```python
import mlflow

mlflow.set_tracking_uri("sqlite:///mlflow_data/mlflow.db")
model = mlflow.sklearn.load_model("models:/rf_online_shoppers@champion")
model.predict(X)  # já aplica o limiar escolhido no treino
```

## Docker

A imagem instala as dependências de produção via `uv sync --locked --no-dev` e executa
`preprocess && train`. O dataset é copiado para a imagem, então ele precisa estar em
`data/raw/online_shoppers_intention.csv` antes do build (veja [Dados com DVC](#dados-com-dvc)):

```bash
docker build -t projetofase2 .
docker run --rm --env-file .env -v "$(pwd)/mlflow_data:/app/mlflow_data" projetofase2
```

O volume em `/app/mlflow_data` mantém as runs e o Model Registry fora do container. Por
isso, deixe `MLFLOW_TRACKING_URI` vazio no `.env` (ou aponte para um servidor MLflow).

## Testes e qualidade

```bash
uv run pytest                # testes (dados sintéticos; não tocam em mlflow_data/)
uv run ruff check .          # lint (PEP 8, imports, naming, bugbear...)
```
