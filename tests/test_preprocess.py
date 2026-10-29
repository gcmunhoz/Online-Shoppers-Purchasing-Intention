"""Testes do estágio preprocess."""
import pandas as pd

from projetofase2.config import TARGET
from projetofase2.preprocess import main, run_preprocess


def test_run_preprocess_writes_train_and_test(params, shoppers_df):
    train_path, test_path = run_preprocess(params)
    train_df, test_df = pd.read_csv(train_path), pd.read_csv(test_path)

    assert len(train_df) + len(test_df) == len(shoppers_df.drop_duplicates())
    assert abs(len(test_df) / (len(train_df) + len(test_df)) - params.data.test_size) < 0.01
    assert set(train_df[TARGET].unique()) <= {0, 1}


def test_run_preprocess_is_reproducible(params):
    first = pd.read_csv(run_preprocess(params)[0])
    second = pd.read_csv(run_preprocess(params)[0])
    pd.testing.assert_frame_equal(first, second)


def test_cli_uses_params_file(params_file, params):
    main(["--params", str(params_file)])

    assert params.data.train_path.exists()
    assert params.data.test_path.exists()
