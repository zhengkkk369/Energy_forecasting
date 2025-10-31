import torch
from torch import nn

from energy_forecasting.config import ProjectConfig
from energy_forecasting.data.datastream import DataStream
from energy_forecasting.models.backbone import BackboneFactory
from energy_forecasting.models.adapters import LayerAdapter
from energy_forecasting.training.online_loop import OnlineTrainer


def build_model(config: ProjectConfig) -> torch.nn.Module:
    factory = BackboneFactory(name=config.backbone, params={"d_model": 128, "nhead": 4, "num_layers": 2})
    return factory.build()


def main() -> None:
    config = ProjectConfig()
    model = build_model(config)
    adapters = [LayerAdapter(d_model=128) for _ in range(2)]
    trainer = OnlineTrainer(config=config, model=model, adapters=adapters)

    stream = DataStream(horizon=config.horizon, lag=4)
    criterion = nn.L1Loss()
    trainer.run(stream, criterion)


if __name__ == "__main__":
    main()
