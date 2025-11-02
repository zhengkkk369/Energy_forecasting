import argparse
import copy
import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from data.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom  # noqa: E402
from energy_forecasting.models.baseline import BaselineConfig  # noqa: E402
from energy_forecasting.training.early_stopping import EarlyStopping  # noqa: E402
from energy_forecasting.training.ema import ModelEMA  # noqa: E402
from energy_forecasting.training.lr_utils import WarmupScheduler  # noqa: E402
from energy_forecasting.utils.logging import create_logger  # noqa: E402

DATASET_REGISTRY = {
    "ETTh1": (Dataset_ETT_hour, {"data_path": "ETTh1.csv", "freq": "h", "target": "OT"}),
    "ETTh2": (Dataset_ETT_hour, {"data_path": "ETTh2.csv", "freq": "h", "target": "OT"}),
    "ETTm1": (Dataset_ETT_minute, {"data_path": "ETTm1.csv", "freq": "15min", "target": "OT"}),
    "ETTm2": (Dataset_ETT_minute, {"data_path": "ETTm2.csv", "freq": "15min", "target": "OT"}),
    "ECL": (Dataset_Custom, {"data_path": "ECL.csv", "freq": "h", "target": "MT_320"}),
    "WTH": (Dataset_Custom, {"data_path": "WTH.csv", "freq": "h", "target": "WetBulbCelsius"}),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train baseline forecasters (LSTM/TCN/Transformer) on energy datasets.")
    parser.add_argument("--data-root", default="data", type=str, help="Directory containing CSV datasets.")
    parser.add_argument("--dataset", default="ETTh1", choices=DATASET_REGISTRY.keys())
    parser.add_argument("--features", default="M", choices=["S", "M", "MS"])
    parser.add_argument("--seq-len", default=96, type=int)
    parser.add_argument("--label-len", default=48, type=int)
    parser.add_argument("--pred-len", default=24, type=int)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--epochs", default=5, type=int)
    parser.add_argument("--learning-rate", default=1e-3, type=float)
    parser.add_argument("--num-workers", default=2, type=int)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--model",
        default="lstm",
        choices=[
            "lstm",
            "tcn",
            "transformer",
            "ts2vec",
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
    parser.add_argument("--num-heads", default=4, type=int, help="Only used for transformer.")
    parser.add_argument("--d-model", default=128, type=int, help="Only used for transformer.")
    parser.add_argument("--ff-dim", default=256, type=int, help="Only used for transformer.")
    parser.add_argument("--pooling", default="last", choices=["last", "mean"], help="Transformer pooling strategy.")
    parser.add_argument("--tcn-levels", default=4, type=int, help="Only used for TCN.")
    parser.add_argument("--kernel-size", default=3, type=int, help="Only used for TCN.")
    parser.add_argument("--patch-len", default=16, type=int, help="Patch length for PatchTST.")
    parser.add_argument("--freq-top-k", default=16, type=int, help="Top frequency components for FEDformer.")
    parser.add_argument("--rep-dim", default=128, type=int, help="Representation dimension for TS2Vec.")
    parser.add_argument("--scheduler", default="none", choices=["none", "step", "cosine", "plateau"])
    parser.add_argument("--scheduler-step-size", default=5, type=int, help="Step size for StepLR scheduler.")
    parser.add_argument("--scheduler-gamma", default=0.5, type=float, help="Gamma for StepLR or ReduceLROnPlateau.")
    parser.add_argument("--scheduler-t-max", default=10, type=int, help="T_max for CosineAnnealingLR.")
    parser.add_argument(
        "--scheduler-min-lr",
        default=1e-6,
        type=float,
        help="Minimum learning rate for cosine scheduler.",
    )
    parser.add_argument("--warmup-epochs", default=0, type=int, help="Number of warmup epochs before scheduler kicks in.")
    parser.add_argument(
        "--warmup-start-factor",
        default=0.1,
        type=float,
        help="Initial learning rate factor during warmup (relative to base lr).",
    )
    parser.add_argument(
        "--early-stopping-patience",
        default=0,
        type=int,
        help="Enable early stopping when >0 and sets patience in epochs.",
    )
    parser.add_argument(
        "--early-stopping-min-delta",
        default=0.0,
        type=float,
        help="Minimum decrease in validation loss to reset early stopping patience.",
    )
    parser.add_argument(
        "--grad-clip-norm",
        default=0.0,
        type=float,
        help="If >0, clip gradient norm to this value during training.",
    )
    parser.add_argument(
        "--ema-decay",
        default=0.0,
        type=float,
        help="If >0, maintain an exponential moving average of model weights for evaluation.",
    )
    return parser.parse_args()


def build_dataloaders(args: argparse.Namespace):
    dataset_cls, base_kwargs = DATASET_REGISTRY[args.dataset]
    dataset_kwargs = {
        "size": (args.seq_len, args.label_len, args.pred_len),
        "features": args.features,
        "data_path": base_kwargs["data_path"],
        "target": base_kwargs["target"],
        "timeenc": 0,
        "freq": base_kwargs["freq"],
    }
    data_root = Path(args.data_root)
    train_dataset = dataset_cls(root_path=data_root, flag="train", **dataset_kwargs)
    val_dataset = dataset_cls(root_path=data_root, flag="val", **dataset_kwargs)
    test_dataset = dataset_cls(root_path=data_root, flag="test", **dataset_kwargs)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )
    return train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader


def train_epoch(model, loader, optimizer, criterion, device, pred_len, grad_clip_norm=None, ema=None):
    model.train()
    total_loss = 0.0
    total_samples = 0
    for batch in loader:
        seq_x, seq_y, _, _ = batch
        seq_x = torch.as_tensor(seq_x, dtype=torch.float32, device=device)
        target = torch.as_tensor(seq_y[:, -pred_len:, :], dtype=torch.float32, device=device)

        optimizer.zero_grad()
        output = model(seq_x)
        loss = criterion(output, target)
        loss.backward()
        if grad_clip_norm and grad_clip_norm > 0.0:
            clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()
        if ema is not None:
            ema.update(model)

        batch_size = seq_x.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
    return total_loss / max(total_samples, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device, pred_len, ema=None):
    model.eval()
    ema_applied = False
    if ema is not None:
        ema.apply_shadow(model)
        ema_applied = True
    total_loss = 0.0
    total_mae = 0.0
    total_samples = 0
    for batch in loader:
        seq_x, seq_y, _, _ = batch
        seq_x = torch.as_tensor(seq_x, dtype=torch.float32, device=device)
        target = torch.as_tensor(seq_y[:, -pred_len:, :], dtype=torch.float32, device=device)

        output = model(seq_x)
        loss = criterion(output, target)
        mae = torch.mean(torch.abs(output - target))

        batch_size = seq_x.size(0)
        total_loss += loss.item() * batch_size
        total_mae += mae.item() * batch_size
        total_samples += batch_size

    if ema_applied:
        ema.restore(model)
    denom = max(total_samples, 1)
    return total_loss / denom, total_mae / denom


def build_scheduler(optimizer: torch.optim.Optimizer, args: argparse.Namespace):
    scheduler = None
    if args.scheduler == "step":
        scheduler = StepLR(optimizer, step_size=args.scheduler_step_size, gamma=args.scheduler_gamma)
    elif args.scheduler == "cosine":
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=args.scheduler_t_max,
            eta_min=args.scheduler_min_lr,
        )
    elif args.scheduler == "plateau":
        scheduler = ReduceLROnPlateau(
            optimizer,
            factor=args.scheduler_gamma,
            patience=max(1, args.scheduler_step_size),
        )
    elif args.scheduler != "none":
        raise ValueError(f"Unknown scheduler: {args.scheduler}")

    if args.warmup_epochs > 0:
        scheduler = WarmupScheduler(
            optimizer,
            warmup_epochs=args.warmup_epochs,
            start_factor=args.warmup_start_factor,
            after_scheduler=scheduler,
        )
    return scheduler


def step_scheduler(scheduler, val_loss: float | None) -> None:
    if scheduler is None:
        return
    if isinstance(scheduler, WarmupScheduler):
        scheduler.step(val_loss)
    elif isinstance(scheduler, ReduceLROnPlateau):
        scheduler.step(val_loss)
    else:
        scheduler.step()


def main() -> None:
    args = parse_args()
    logger = create_logger("baseline-training")

    (
        train_dataset,
        val_dataset,
        test_dataset,
        train_loader,
        val_loader,
        test_loader,
    ) = build_dataloaders(args)

    input_dim = train_dataset.data_x.shape[1]
    output_dim = train_dataset.data_y.shape[1]

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
    model = model_config.build().to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    criterion = nn.MSELoss()
    scheduler = build_scheduler(optimizer, args)
    ema = None
    if 0.0 < args.ema_decay < 1.0:
        ema = ModelEMA(model, decay=args.ema_decay)
    early_stopper = (
        EarlyStopping(patience=args.early_stopping_patience, min_delta=args.early_stopping_min_delta)
        if args.early_stopping_patience > 0
        else None
    )

    logger.info(
        "Starting training | dataset=%s | model=%s | seq_len=%d | pred_len=%d | device=%s",
        args.dataset,
        args.model,
        args.seq_len,
        args.pred_len,
        args.device,
    )

    best_val_loss = float("inf")
    best_state = None
    best_state_from_ema = False

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            args.device,
            args.pred_len,
            grad_clip_norm=args.grad_clip_norm,
            ema=ema,
        )
        val_loss, val_mae = evaluate(
            model,
            val_loader,
            criterion,
            args.device,
            args.pred_len,
            ema=ema,
        )
        logger.info(
            "Epoch %02d | train_loss=%.4f | val_loss=%.4f | val_mae=%.4f",
            epoch,
            train_loss,
            val_loss,
            val_mae,
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if ema is not None:
                best_state = ema.state_dict()
                best_state_from_ema = True
            else:
                best_state = copy.deepcopy(model.state_dict())
                best_state_from_ema = False
        previous_lr = optimizer.param_groups[0]["lr"]
        step_scheduler(scheduler, val_loss)
        if scheduler is not None and optimizer.param_groups[0]["lr"] != previous_lr:
            logger.info("Learning rate now %.6f", optimizer.param_groups[0]["lr"])
        if early_stopper is not None and early_stopper.step(val_loss):
            logger.info("Early stopping triggered after %d epochs without improvement.", early_stopper.patience)
            break

    if best_state is not None:
        if best_state_from_ema:
            model.load_state_dict(best_state, strict=False)
        else:
            model.load_state_dict(best_state)
    test_loss, test_mae = evaluate(model, test_loader, criterion, args.device, args.pred_len)
    logger.info("Test metrics | loss=%.4f | mae=%.4f", test_loss, test_mae)


if __name__ == "__main__":
    main()
