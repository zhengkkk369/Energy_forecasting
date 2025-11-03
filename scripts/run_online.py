import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Iterable, Iterator, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from data.data_loader import Dataset_Custom, Dataset_ETT_hour, Dataset_ETT_minute  # noqa: E402
from src.energy_forecasting.config import ProjectConfig  # noqa: E402
from src.energy_forecasting.data import DriftConfig, DriftInjector, StreamBatch  # noqa: E402
from src.energy_forecasting.drift.d3a import D3AController  # noqa: E402
from src.energy_forecasting.models import BaselineConfig  # noqa: E402
from src.energy_forecasting.training.ema import ModelEMA  # noqa: E402
from src.energy_forecasting.training.lr_utils import WarmupScheduler  # noqa: E402
from src.energy_forecasting.training.replay import ReplayBuffer  # noqa: E402
from src.energy_forecasting.utils import metric as calc_metric  # noqa: E402
from src.energy_forecasting.utils.logging import create_logger  # noqa: E402

DATASET_REGISTRY = {
    "ETTh1": (Dataset_ETT_hour, {"data_path": "ETTh1.csv", "freq": "h", "target": "OT"}),
    "ETTh2": (Dataset_ETT_hour, {"data_path": "ETTh2.csv", "freq": "h", "target": "OT"}),
    "ETTm1": (Dataset_ETT_minute, {"data_path": "ETTm1.csv", "freq": "15min", "target": "OT"}),
    "ETTm2": (Dataset_ETT_minute, {"data_path": "ETTm2.csv", "freq": "15min", "target": "OT"}),
    "ECL": (Dataset_Custom, {"data_path": "ECL.csv", "freq": "h", "target": "MT_320"}),
    "WTH": (Dataset_Custom, {"data_path": "WTH.csv", "freq": "h", "target": "WetBulbCelsius"}),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OneNet-style online continual learning loop.")
    parser.add_argument("--data-root", default="data", type=str, help="Directory containing CSV datasets.")
    parser.add_argument("--dataset", default="ETTh1", choices=DATASET_REGISTRY.keys())
    parser.add_argument("--features", default="M", choices=["S", "M", "MS"])
    parser.add_argument("--seq-len", default=96, type=int)
    parser.add_argument("--label-len", default=48, type=int)
    parser.add_argument("--pred-len", default=24, type=int)
    parser.add_argument("--learning-rate", default=5e-4, type=float)
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
        ],
    )
    parser.add_argument("--hidden-dim", default=128, type=int)
    parser.add_argument("--num-layers", default=2, type=int)
    parser.add_argument("--dropout", default=0.1, type=float)
    parser.add_argument("--num-heads", default=4, type=int)
    parser.add_argument("--d-model", default=128, type=int)
    parser.add_argument("--ff-dim", default=256, type=int)
    parser.add_argument("--pooling", default="last", choices=["last", "mean"])
    parser.add_argument("--tcn-levels", default=4, type=int)
    parser.add_argument("--kernel-size", default=3, type=int)
    parser.add_argument("--patch-len", default=16, type=int)
    parser.add_argument("--freq-top-k", default=16, type=int)
    parser.add_argument("--rep-dim", default=128, type=int)

    parser.add_argument("--scheduler", default="cosine", choices=["none", "step", "cosine", "plateau"])
    parser.add_argument("--scheduler-step-size", default=200, type=int)
    parser.add_argument("--scheduler-gamma", default=0.7, type=float)
    parser.add_argument("--scheduler-t-max", default=500, type=int)
    parser.add_argument("--scheduler-min-lr", default=1e-6, type=float)
    parser.add_argument("--warmup-steps", default=0, type=int)
    parser.add_argument("--warmup-start-factor", default=0.1, type=float)
    parser.add_argument("--ema-decay", default=0.0, type=float)
    parser.add_argument("--grad-clip-norm", default=0.0, type=float)
    parser.add_argument("--pretrain-steps", default=200, type=int)
    parser.add_argument("--update-interval", default=1, type=int)
    parser.add_argument("--max-steps", default=-1, type=int)
    parser.add_argument("--log-interval", default=50, type=int)
    parser.add_argument("--label-delay", default=0, type=int)
    parser.add_argument("--enable-drift", action="store_true")
    parser.add_argument("--drift-config", default=None, type=str, help="JSON file describing drift schedules.")
    parser.add_argument("--replay-size", default=0, type=int, help="Replay buffer capacity (0 disables).")
    parser.add_argument("--replay-sample", default=4, type=int, help="Replay samples per update.")
    parser.add_argument("--use-d3a", action="store_true", help="Enable detect-then-adapt controller.")
    return parser.parse_args()


class OnlineStream:
    """Concatenate dataset partitions to simulate a continuous online stream."""

    def __init__(
        self,
        datasets: Iterable,
        pred_len: int,
        label_delay: int,
        device: torch.device,
        drift_injector: Optional[DriftInjector] = None,
    ) -> None:
        self.datasets = list(datasets)
        self.pred_len = pred_len
        self.label_delay = label_delay
        self.device = device
        self.drift_injector = drift_injector
        self._queue: list[Tuple[torch.Tensor, torch.Tensor]] = []
        self._build_lengths()

    def _build_lengths(self) -> None:
        self._lengths = [len(ds) for ds in self.datasets]
        self.total = sum(self._lengths)

    def __len__(self) -> int:
        return self.total

    def iterator(self) -> Iterator[Tuple[int, torch.Tensor, Optional[torch.Tensor], torch.Tensor]]:
        step = 0
        for dataset in self.datasets:
            for idx in range(len(dataset)):
                seq_x, seq_y, seq_x_mark, seq_y_mark = dataset[idx]

                seq_x_np = np.asarray(seq_x, dtype=np.float32)
                seq_y_np = np.asarray(seq_y, dtype=np.float32)

                if self.drift_injector is not None:
                    batch_np = StreamBatch(
                        features=seq_x_np[None, ...],
                        context={
                            "seq_x_mark": np.asarray(seq_x_mark, dtype=np.float32)[None, ...],
                            "seq_y_mark": np.asarray(seq_y_mark, dtype=np.float32)[None, ...],
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

                available_target = None
                if self.label_delay == 0:
                    available_target = target_full
                else:
                    self._queue.append((seq_x_tensor, target_full))
                    if len(self._queue) > self.label_delay:
                        _, available_target = self._queue.pop(0)
                yield step, seq_x_tensor, available_target, target_full
                step += 1


def build_stream(args: argparse.Namespace, device: torch.device, drift_injector: Optional[DriftInjector]) -> OnlineStream:
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
    return OnlineStream(datasets, args.pred_len, args.label_delay, device, drift_injector=drift_injector)


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
    )

    model = model_config.build().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = build_scheduler(optimizer, args)
    ema = ModelEMA(model, decay=args.ema_decay) if 0.0 < args.ema_decay < 1.0 else None
    criterion = nn.MSELoss()
    replay_buffer = ReplayBuffer(capacity=args.replay_size) if args.replay_size > 0 else None
    if replay_buffer is not None:
        logger.info("Replay buffer enabled with capacity=%d and sample=%d.", args.replay_size, args.replay_sample)

    if args.use_d3a:
        project_cfg = ProjectConfig()
        d3a_controller = D3AController(project_cfg.d3a)
        logger.info(
            "D3A controller enabled | window=%d | candidate=%d | cooldown=%d",
            project_cfg.d3a.confirmation_window,
            project_cfg.d3a.candidate_window,
            project_cfg.d3a.cooldown,
        )
    else:
        d3a_controller = None

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

        instruction = None
        if d3a_controller is not None:
            representation = seq_x.detach().cpu().mean(dim=0).numpy().flatten()
            signal, instruction = d3a_controller.assess(representation)

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
            if replay_buffer is not None:
                store_allowed = instruction.store_state if instruction is not None else True
                if store_allowed:
                    replay_buffer.push(
                        StreamBatch(
                            features=seq_np.astype(np.float32),
                            context={},
                            target=target_np.astype(np.float32),
                            timestamp=step,
                        )
                    )
            if instruction is not None and instruction.trigger_adaptation:
                logger.info(
                    "D3A trigger at step=%d | strength=%.4f | type=%s",
                    step,
                    instruction.drift_strength,
                    instruction.drift_type,
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
        if d3a_controller is not None and instruction is not None:
            should_update = should_update and instruction.trigger_adaptation

        if should_update and target is not None:
            model.train()
            optimizer.zero_grad()
            seq_batch = seq_x
            target_batch = target
            if replay_buffer is not None and args.replay_sample > 0:
                samples = replay_buffer.sample(args.replay_sample)
                replay_feats = []
                replay_targets = []
                for sample in samples:
                    if sample.target is None:
                        continue
                    replay_feats.append(torch.as_tensor(sample.features, dtype=torch.float32, device=device))
                    replay_targets.append(torch.as_tensor(sample.target, dtype=torch.float32, device=device))
                if replay_feats and replay_targets:
                    seq_batch = torch.cat([seq_batch] + replay_feats, dim=0)
                    target_batch = torch.cat([target] + replay_targets, dim=0)

            prediction = model(seq_batch)
            loss = criterion(prediction, target_batch)
            loss.backward()
            if args.grad_clip_norm > 0:
                clip_grad_norm_(model.parameters(), args.grad_clip_norm)
            optimizer.step()
            if ema is not None:
                ema.update(model)
            step_scheduler(scheduler, loss.item())

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
                replay_buffer.push(
                    StreamBatch(
                        features=seq_np.astype(np.float32),
                        context={},
                        target=target_np.astype(np.float32),
                        timestamp=step,
                    )
                )
            if mae < best_mae:
                best_mae = mae
                if ema is not None:
                    best_state = copy.deepcopy(ema.state_dict())
                    best_state_from_ema = True
                else:
                    best_state = copy.deepcopy(model.state_dict())
                    best_state_from_ema = False

        should_update = step >= args.pretrain_steps and (step - args.pretrain_steps) % max(args.update_interval, 1) == 0
        if should_update and target is not None:
            model.train()
            optimizer.zero_grad()
            seq_batch = seq_x
            target_batch = target
            if replay_buffer is not None and args.replay_sample > 0:
                samples = replay_buffer.sample(args.replay_sample)
                replay_feats = []
                replay_targets = []
                for sample in samples:
                    if sample.target is None:
                        continue
                    replay_feats.append(torch.as_tensor(sample.features, dtype=torch.float32, device=device))
                    replay_targets.append(torch.as_tensor(sample.target, dtype=torch.float32, device=device))
                if replay_feats and replay_targets:
                    seq_batch = torch.cat([seq_batch] + replay_feats, dim=0)
                    target_batch = torch.cat([target] + replay_targets, dim=0)

            prediction = model(seq_batch)
            loss = criterion(prediction, target_batch)
            loss.backward()
            if args.grad_clip_norm > 0:
                clip_grad_norm_(model.parameters(), args.grad_clip_norm)
            optimizer.step()
            if ema is not None:
                ema.update(model)
            step_scheduler(scheduler, loss.item())

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
