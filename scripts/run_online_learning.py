import argparse
import copy
import json
from collections import deque
from pathlib import Path
from typing import Iterable, Iterator, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR
from torch.utils.data import DataLoader

from scripts._cli_utils import (  # noqa: E402
    DATASET_DEFAULTS,
    DATASET_REGISTRY,
    apply_dataset_defaults,
    parse_kernel_sizes,
)
from src.energy_forecasting.data import DriftConfig, DriftInjector, StreamBatch  # noqa: E402
from src.energy_forecasting.models import BaselineConfig  # noqa: E402
from src.energy_forecasting.training.ema import ModelEMA  # noqa: E402
from src.energy_forecasting.training.lr_utils import WarmupScheduler  # noqa: E402
from src.energy_forecasting.utils import metric as calc_metric  # noqa: E402
from src.energy_forecasting.utils.detector import STEPD  # noqa: E402
from src.energy_forecasting.utils.buffer import Buffer  # noqa: E402
from src.energy_forecasting.utils.logging import create_logger  # noqa: E402

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OneNet-style online continual learning loop.")
    parser.add_argument("--data_root", default="data", type=str, help="Directory containing CSV datasets.")
    parser.add_argument("--dataset", default="ETTh1", choices=DATASET_REGISTRY.keys())
    parser.add_argument("--features", default="M", choices=["S", "M", "MS"])
    parser.add_argument("--seq_len", default=96, type=int)
    parser.add_argument("--label_len", default=48, type=int)
    parser.add_argument("--pred_len", default=24, type=int)
    parser.add_argument("--learning_rate", default=5e-4, type=float)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--model",
        default="fsnet",
        choices=[
            "lstm",
            "tcn",
            "transformer",
            "fsnet",
            "nomem",
            "ncca",
            "autoformer",
            "dlinear",
            "informer",
            "patchtst",
            "fedformer",
            "onenet",
        ],
    )
    parser.add_argument("--hidden_dim", default=128, type=int)
    parser.add_argument("--num_layers", default=2, type=int)
    parser.add_argument("--dropout", default=0.1, type=float)
    parser.add_argument("--num_heads", default=4, type=int)
    parser.add_argument("--d_model", default=128, type=int)
    parser.add_argument("--ff_dim", default=256, type=int)
    parser.add_argument("--pooling", default="last", choices=["last", "mean"])
    parser.add_argument("--tcn_levels", default=4, type=int)
    parser.add_argument("--kernel_size", default=3, type=int)
    parser.add_argument("--patch_len", default=16, type=int)
    parser.add_argument("--freq_top_k", default=16, type=int)
    parser.add_argument("--rep_dim", default=128, type=int)
    parser.add_argument(
        "--onenet_kernel_sizes",
        default=(3, 5, 7),
        type=parse_kernel_sizes,
        help="Comma separated convolution kernel sizes for OneNet's multiscale blocks.",
    )
    parser.add_argument(
        "--onenet_activation",
        default="gelu",
        choices=["relu", "gelu", "silu"],
        help="Activation used inside OneNet's convolutional blocks.",
    )

    parser.add_argument("--scheduler", default="cosine", choices=["none", "step", "cosine", "plateau"])
    parser.add_argument("--scheduler_step_size", default=200, type=int)
    parser.add_argument("--scheduler_gamma", default=0.7, type=float)
    parser.add_argument("--scheduler_t_max", default=500, type=int)
    parser.add_argument("--scheduler_min_lr", default=1e-6, type=float)
    parser.add_argument("--warmup_steps", default=0, type=int)
    parser.add_argument("--warmup_start_factor", default=0.1, type=float)
    parser.add_argument("--ema_decay", default=0.0, type=float)
    parser.add_argument("--grad_clip_norm", default=0.0, type=float)
    parser.add_argument("--pretrain_steps", default=1000, type=int)
    parser.add_argument("--update_interval", default=50, type=int)
    parser.add_argument("--max_steps", default=-1, type=int)
    parser.add_argument("--log_interval", default=50, type=int)
    parser.add_argument("--label_delay", default=0, type=int)
    parser.add_argument("--enable_drift", action="store_true")
    parser.add_argument("--drift_config", default=None, type=str, help="JSON file describing drift schedules.")
    parser.add_argument("--replay_size", default=512, type=int, help="Buffer capacity (0 disables).")
    parser.add_argument("--replay_sample", default=64, type=int, help="Samples drawn from buffer per update.")
    parser.add_argument("--pretrain_epochs", default=10, type=int, help="Epochs to run on first update after pretraining.")
    parser.add_argument("--update_epochs", default=3, type=int, help="Epochs to run on subsequent updates.")
    parser.add_argument("--use_d3a", dest="use_d3a", action="store_true", help="Enable detect-then-adapt controller.")
    parser.add_argument("--d3a_window", default=168, type=int, help="Sliding window size for STEPD detector.")
    parser.add_argument("--d3a_alpha_w", default=0.05, type=float, help="Warning threshold for STEPD.")
    parser.add_argument("--d3a_alpha_d", default=0.003, type=float, help="Drift threshold for STEPD.")
    parser.add_argument("--d3a_min_lr", default=1e-4, type=float, help="Minimum LR when STEPD adapts learning rate.")
    parser.add_argument("--d3a_max_lr", default=3e-3, type=float, help="Maximum LR when STEPD adapts learning rate.")
    parser.add_argument("--d3a_plot_path", default='pics/d3a', type=str, help="Optional path to save STEPD error plot.")
    args = parser.parse_args()

    return apply_dataset_defaults(parser, args)


class DataLoaderStream:
    """Iterate dataset splits sequentially using `DataLoader` for OneNet-style adaptation."""

    def __init__(
        self,
        loaders: Iterable[DataLoader],
        pred_len: int,
        label_delay: int,
        device: torch.device,
        drift_injector: Optional[DriftInjector] = None,
    ) -> None:
        self.loaders = list(loaders)
        self.pred_len = pred_len
        self.label_delay = label_delay
        self.device = device
        self.drift_injector = drift_injector
        self.datasets = [loader.dataset for loader in self.loaders if hasattr(loader, "dataset")]
        self._queue: deque[torch.Tensor] = deque()
        self.total = sum(len(dataset) for dataset in self.datasets)

    def __len__(self) -> int:
        return self.total

    def iterator(self) -> Iterator[Tuple[int, torch.Tensor, Optional[torch.Tensor], torch.Tensor]]:
        step = 0
        for loader in self.loaders:
            for batch in loader:
                seq_x, seq_y, seq_x_mark, seq_y_mark = batch

                seq_x_np = seq_x.squeeze(0).detach().cpu().numpy().astype(np.float32)
                seq_y_np = seq_y.squeeze(0).detach().cpu().numpy().astype(np.float32)
                seq_x_mark_np = seq_x_mark.squeeze(0).detach().cpu().numpy().astype(np.float32)
                seq_y_mark_np = seq_y_mark.squeeze(0).detach().cpu().numpy().astype(np.float32)

                if self.drift_injector is not None:
                    batch_np = StreamBatch(
                        features=seq_x_np[None, ...],
                        context={
                            "seq_x_mark": seq_x_mark_np[None, ...],
                            "seq_y_mark": seq_y_mark_np[None, ...],
                        },
                        target=seq_y_np[None, ...],
                        timestamp=step,
                    )
                    drifted = self.drift_injector.apply(step, batch_np)
                    seq_x_np = drifted.features[0]
                    if drifted.target is not None:
                        seq_y_np = drifted.target[0]

                seq_x_tensor = torch.as_tensor(seq_x_np, dtype=torch.float32, device=self.device).unsqueeze(0)
                target_full = torch.as_tensor(
                    seq_y_np[-self.pred_len :], dtype=torch.float32, device=self.device
                ).unsqueeze(0)

                available_target: Optional[torch.Tensor] = None
                if self.label_delay == 0:
                    available_target = target_full
                else:
                    self._queue.append(target_full)
                    if len(self._queue) > self.label_delay:
                        available_target = self._queue.popleft()

                yield step, seq_x_tensor, available_target, target_full
                step += 1


def build_stream(
    args: argparse.Namespace, device: torch.device, drift_injector: Optional[DriftInjector]
) -> DataLoaderStream:
    dataset_cls, base_kwargs = DATASET_REGISTRY[args.dataset]
    dataset_kwargs = {
        "size": (args.seq_len, args.label_len, args.pred_len),
        "features": args.features,
        "data_path": base_kwargs["data_path"],
        "target": base_kwargs["target"],
        "timeenc": 0,
        "freq": base_kwargs["freq"],
    }
    root = Path(args.data_root)
    datasets = [dataset_cls(root_path=root, flag=flag, **dataset_kwargs) for flag in ("train", "val", "test")]
    datasets = [ds for ds in datasets if len(ds) > 0]
    loaders = [
        DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False)
        for dataset in datasets
    ]
    return DataLoaderStream(loaders, args.pred_len, args.label_delay, device, drift_injector=drift_injector)


def build_scheduler(optimizer: torch.optim.Optimizer, args: argparse.Namespace):
    scheduler = None
    if args.scheduler == "step":
        scheduler = StepLR(optimizer, step_size=args.scheduler_step_size, gamma=args.scheduler_gamma)
    elif args.scheduler == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=args.scheduler_t_max, eta_min=args.scheduler_min_lr)
    elif args.scheduler == "plateau":
        scheduler = ReduceLROnPlateau(optimizer, factor=args.scheduler_gamma, patience=max(1, args.scheduler_step_size))
    elif args.scheduler != "none":
        raise ValueError(f"Unknown scheduler: {args.scheduler}")

    if args.warmup_steps > 0:
        scheduler = WarmupScheduler(
            optimizer,
            warmup_epochs=args.warmup_steps,
            start_factor=args.warmup_start_factor,
            after_scheduler=scheduler,
        )
    return scheduler


def step_scheduler(scheduler, metric: Optional[float] = None) -> None:
    if scheduler is None:
        return
    if isinstance(scheduler, WarmupScheduler):
        scheduler.step(metric)
    elif isinstance(scheduler, ReduceLROnPlateau):
        scheduler.step(metric if metric is not None else 0.0)
    else:
        scheduler.step()


def build_drift_injector(args: argparse.Namespace, logger) -> Optional[DriftInjector]:
    configs = None
    if args.drift_config:
        path = Path(args.drift_config)
        if not path.exists():
            logger.error("Drift config file not found: %s", path)
            return None
        try:
            configs_data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            logger.error("Failed to parse drift config JSON: %s", exc)
            return None
        configs = [DriftConfig(**cfg) for cfg in configs_data]
    elif args.enable_drift:
        configs = [
            DriftConfig(drift_type="abrupt", start=200, magnitude=1.2, feature_indices=[0]),
            DriftConfig(drift_type="periodic", start=0, magnitude=0.3, period=168, applies_to="features", mode="multiplicative"),
            DriftConfig(drift_type="gradual", start=800, duration=240, magnitude=-0.5, feature_indices=[0, 1]),
        ]
    if configs is None:
        return None
    return DriftInjector(configs)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    logger = create_logger("online-loop")

    drift_injector = build_drift_injector(args, logger)
    if drift_injector is not None:
        logger.info("Drift injection enabled with %d configurations.", len(drift_injector.configs))

    stream = build_stream(args, device, drift_injector)
    args.max_steps = stream.total if args.max_steps < 0 else min(args.max_steps, stream.total)

    sample_dataset = stream.datasets[0]
    input_dim = sample_dataset.data_x.shape[1]
    output_dim = sample_dataset.data_y.shape[1]

    model_config = BaselineConfig(
        input_dim=input_dim,
        output_dim=output_dim,
        pred_len=args.pred_len,
        model_type=args.model,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        num_heads=args.num_heads,
        d_model=args.d_model,
        ff_dim=args.ff_dim,
        pooling=args.pooling,
        tcn_levels=args.tcn_levels,
        kernel_size=args.kernel_size,
        seq_len=args.seq_len,
        patch_len=args.patch_len,
        freq_top_k=args.freq_top_k,
        rep_dim=args.rep_dim,
        onenet_kernel_sizes=args.onenet_kernel_sizes,
        onenet_activation=args.onenet_activation,
    )

    model = model_config.build().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = build_scheduler(optimizer, args)
    ema = ModelEMA(model, decay=args.ema_decay) if 0.0 < args.ema_decay < 1.0 else None
    criterion = nn.MSELoss()
    detector = None
    buffer: Optional[Buffer] = None
    if args.use_d3a:
        detector = STEPD(
            new_window_size=args.d3a_window,
            alpha_w=args.d3a_alpha_w,
            alpha_d=args.d3a_alpha_d,
        )
        capacity = args.replay_size if args.replay_size > 0 else args.d3a_window
        buffer = Buffer(buffer_size=capacity, device=device, mode="fifo")
        logger.info(
            "STEPD detector enabled | window=%d | alpha_w=%.4f | alpha_d=%.4f | buffer=%d",
            args.d3a_window,
            args.d3a_alpha_w,
            args.d3a_alpha_d,
            capacity,
        )
    elif args.replay_size > 0:
        buffer = Buffer(buffer_size=args.replay_size, device=device, mode="fifo")
        logger.info("Buffer enabled | capacity=%d | sample=%d", args.replay_size, args.replay_sample)

    logger.info(
        "Starting OneNet-style online loop | dataset=%s | model=%s | total_steps=%d",
        args.dataset,
        args.model,
        args.max_steps,
    )

    metrics = {"mae": [], "mse": [], "rmse": [], "mape": [], "mspe": []}
    best_mae = float("inf")
    best_state = None
    best_state_from_ema = False

    for step, seq_x, target, raw_target in stream.iterator():
        if step >= args.max_steps:
            break

        drift_flag = 0
        suggested_lr = None

        model.eval()
        with torch.no_grad():
            prediction = model(seq_x)
            if ema is not None:
                ema.apply_shadow(model)
                prediction = model(seq_x)
                ema.restore(model)

        if target is not None:
            seq_np = seq_x.detach().cpu().numpy()
            pred_np = prediction.detach().cpu().numpy()
            target_np = target.detach().cpu().numpy()
            mae, mse, rmse, mape, mspe = calc_metric(pred_np, target_np)
            metrics["mae"].append(mae)
            metrics["mse"].append(mse)
            metrics["rmse"].append(rmse)
            metrics["mape"].append(mape)
            metrics["mspe"].append(mspe)
            if args.use_d3a and detector is not None:
                mae_error = float(np.mean(np.abs(pred_np - target_np)))
                detector.add_data(mae_error, seq_x.detach().cpu())
                drift_flag, suggested_lr = detector.run_test()
                if suggested_lr is not None:
                    lr_new = float(np.clip(suggested_lr, args.d3a_min_lr, args.d3a_max_lr))
                    for group in optimizer.param_groups:
                        group["lr"] = lr_new
                if drift_flag:
                    if args.d3a_plot_path and detector.data:
                        plot_root = Path(args.d3a_plot_path)
                        if plot_root.suffix:
                            plot_path = plot_root
                        else:
                            plot_root.mkdir(parents=True, exist_ok=True)
                            plot_path = plot_root / f"stepd_mae_step_{step + 1:06d}.pdf"
                        detector.plt_distribution(detector.data, name="mae", save_path=str(plot_path))
                        logger.info("STEPD plot saved to %s", plot_path)
                    detector.reset()
                    logger.info("STEPD drift detected at step=%d | suggested_lr=%s", step, f"{suggested_lr:.5f}" if suggested_lr is not None else "n/a")
            if buffer is not None:
                buffer.add_data(
                    examples=seq_x.detach(),
                    labels=target.detach(),
                    logits=None,
                    task_labels=None,
                )
            if mae < best_mae:
                best_mae = mae
                if ema is not None:
                    best_state = copy.deepcopy(ema.state_dict())
                    best_state_from_ema = True
                else:
                    best_state = copy.deepcopy(model.state_dict())
                    best_state_from_ema = False

        should_update = (
            step >= args.pretrain_steps and (step - args.pretrain_steps) % max(args.update_interval, 1) == 0
        )
        if args.use_d3a and drift_flag == 1:
            should_update = True

        if should_update and target is not None:
            epochs = args.update_epochs
            if step == args.pretrain_steps and args.pretrain_epochs > 0:
                epochs = max(args.pretrain_epochs, epochs)
            model.train()
            loss_accumulator = []
            for _ in range(max(1, epochs)):
                seq_batch = seq_x
                target_batch = target
                if (
                    buffer is not None
                    and args.replay_sample > 0
                    and hasattr(buffer, "examples")
                    and not buffer.is_empty()
                ):
                    sample_limit = buffer.examples.shape[0]
                    sample_size = min(args.replay_sample, sample_limit)
                    extra = buffer.get_data(sample_size)
                    extra_examples = extra[0]
                    extra_labels = extra[1] if len(extra) > 1 else None
                    if extra_labels is not None:
                        seq_batch = torch.cat([seq_batch, extra_examples], dim=0)
                        target_batch = torch.cat([target_batch, extra_labels], dim=0)

                optimizer.zero_grad()
                prediction = model(seq_batch)
                loss = criterion(prediction, target_batch)
                loss.backward()
                if args.grad_clip_norm > 0:
                    clip_grad_norm_(model.parameters(), args.grad_clip_norm)
                optimizer.step()
                if ema is not None:
                    ema.update(model)
                loss_accumulator.append(loss.item())
            if loss_accumulator:
                step_scheduler(scheduler, float(np.mean(loss_accumulator)))

        if args.log_interval > 0 and (step + 1) % args.log_interval == 0 and metrics["mae"]:
            recent_mae = float(np.mean(metrics["mae"][-args.log_interval :]))
            recent_mse = float(np.mean(metrics["mse"][-args.log_interval :]))
            recent_rmse = float(np.mean(metrics["rmse"][-args.log_interval :]))
            logger.info(
                "step=%d | mae=%.4f | rmse=%.4f | mse=%.4f | lr=%.6f",
                step + 1,
                recent_mae,
                recent_rmse,
                recent_mse,
                optimizer.param_groups[0]["lr"],
            )

    if best_state is not None:
        if best_state_from_ema and ema is not None:
            ema.load_state_dict(best_state)
            ema.copy_to_model(model)
        else:
            model.load_state_dict(best_state)

    if metrics["mae"]:
        overall_mae = float(np.mean(metrics["mae"]))
        overall_rmse = float(np.mean(metrics["rmse"]))
        overall_mse = float(np.mean(metrics["mse"]))
        overall_mape = float(np.mean(metrics["mape"]))
        logger.info(
            "Finished online adaptation | mae=%.4f | rmse=%.4f | mse=%.4f | mape=%.4f",
            overall_mae,
            overall_rmse,
            overall_mse,
            overall_mape,
        )
    else:
        logger.warning("No labels were observed during the run; unable to report metrics.")


if __name__ == "__main__":
    main()
