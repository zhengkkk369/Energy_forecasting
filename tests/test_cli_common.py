import argparse

from src.energy_forecasting.cli.common import apply_dataset_overrides


def test_apply_dataset_overrides_skips_missing_attributes():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="ETTh1")
    parser.add_argument("--seq-len", dest="seq_len", default=96, type=int)

    args = parser.parse_args([])

    overrides = {"ETTh1": {"seq_len": 192, "epochs": 10}}

    apply_dataset_overrides(args, parser, overrides)

    assert args.seq_len == 192
    assert not hasattr(args, "epochs")
