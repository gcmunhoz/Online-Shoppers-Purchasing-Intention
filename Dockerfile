FROM python:3.13-slim

# uv na mesma versão usada no projeto
COPY --from=ghcr.io/astral-sh/uv:0.12.9 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1

WORKDIR /app

# 1) Só as dependências: esta camada fica em cache enquanto o uv.lock não mudar
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --locked --no-dev --no-install-project

# 2) Código, parâmetros e dataset (rode `dvc pull` antes do build)
COPY README.md ./
COPY src ./src
COPY configs ./configs
COPY data/raw ./data/raw
RUN uv sync --locked --no-dev

ENV PATH="/app/.venv/bin:$PATH"

# Pipeline completo: preprocess -> train (MLflow em /app/mlflow_data)
CMD ["sh", "-c", "preprocess && train"]
