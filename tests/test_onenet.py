import torch

from energy_forecasting.models.config import ModelConfig


def test_onenet_forward_shape():
    cfg = ModelConfig(
        input_dim=4,
        output_dim=2,
        pred_len=12,
        seq_len=24,
        model_type="onenet",
        d_model=16,
        num_layers=2,
    )
    model = cfg.build()

    dummy = torch.randn(3, cfg.seq_len, cfg.input_dim)
    output = model(dummy)

    assert output.shape == (3, cfg.pred_len, cfg.output_dim)
    assert not torch.isnan(output).any()
