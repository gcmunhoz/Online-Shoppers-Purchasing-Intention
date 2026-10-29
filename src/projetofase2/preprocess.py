"""Estágio ``preprocess``: CSV bruto -> splits de treino e teste limpos.

Uso:
    python -m projetofase2.preprocess --params configs/params.yaml
"""

import argparse
from collections.abc import Sequence
from pathlib import Path

from projetofase2.config import DEFAULT_PARAMS_PATH, Params, load_params
from projetofase2.data import load_raw_data, split_train_test


def run_preprocess(params: Params) -> tuple[Path, Path]:
    """Limpa o dataset bruto, divide em treino/teste e salva os dois CSVs."""
    df = load_raw_data(params.data.raw_path)
    train_df, test_df = split_train_test(df, params.data.test_size, params.random_state)

    params.data.processed_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(params.data.train_path, index=False)
    test_df.to_csv(params.data.test_path, index=False)

    print(f"Treino: {len(train_df)} linhas -> {params.data.train_path}")
    print(f"Teste:  {len(test_df)} linhas -> {params.data.test_path}")
    return params.data.train_path, params.data.test_path


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--params", type=Path, default=DEFAULT_PARAMS_PATH,
                        help="Arquivo YAML de parâmetros")
    args = parser.parse_args(argv)
    run_preprocess(load_params(args.params))


if __name__ == "__main__":
    main()
