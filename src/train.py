import os
import json
import argparse
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import SynthRAD2p5DDataset
from src.model import build_unetplusplus_resnet34
from src.losses import MaskedL1Loss, CombinedLoss, CombinedRegionLoss


def compute_loss(criterion, outputs, targets, masks):
    loss_result = criterion(outputs, targets, masks)

    if isinstance(loss_result, tuple):
        loss, loss_dict = loss_result
    else:
        loss = loss_result
        loss_dict = {"total_loss": loss.detach()}

    return loss, loss_dict


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    running = {
        "loss": 0.0,
        "l1_loss": 0.0,
        "region_l1_loss": 0.0,
        "ssim_loss": 0.0,
        "gradient_loss": 0.0
    }

    counts = {
        "l1_loss": 0,
        "region_l1_loss": 0,
        "ssim_loss": 0,
        "gradient_loss": 0
    }

    for batch in tqdm(loader, desc="Training", leave=False):
        inputs = batch["input"].to(device)
        targets = batch["target"].to(device)
        masks = batch["mask"].to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss, loss_dict = compute_loss(criterion, outputs, targets, masks)

        loss.backward()
        optimizer.step()

        running["loss"] += loss.item()

        for key in ["l1_loss", "region_l1_loss", "ssim_loss", "gradient_loss"]:
            if key in loss_dict:
                running[key] += loss_dict[key].item()
                counts[key] += 1

    n = len(loader)

    return {
        "loss": running["loss"] / n,
        "l1_loss": running["l1_loss"] / counts["l1_loss"] if counts["l1_loss"] > 0 else None,
        "region_l1_loss": running["region_l1_loss"] / counts["region_l1_loss"] if counts["region_l1_loss"] > 0 else None,
        "ssim_loss": running["ssim_loss"] / counts["ssim_loss"] if counts["ssim_loss"] > 0 else None,
        "gradient_loss": running["gradient_loss"] / counts["gradient_loss"] if counts["gradient_loss"] > 0 else None
    }


def validate_one_epoch(model, loader, criterion, device):
    model.eval()

    running = {
        "loss": 0.0,
        "l1_loss": 0.0,
        "region_l1_loss": 0.0,
        "ssim_loss": 0.0,
        "gradient_loss": 0.0
    }

    counts = {
        "l1_loss": 0,
        "region_l1_loss": 0,
        "ssim_loss": 0,
        "gradient_loss": 0
    }

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation", leave=False):
            inputs = batch["input"].to(device)
            targets = batch["target"].to(device)
            masks = batch["mask"].to(device)

            outputs = model(inputs)
            loss, loss_dict = compute_loss(criterion, outputs, targets, masks)

            running["loss"] += loss.item()

            for key in ["l1_loss", "region_l1_loss", "ssim_loss", "gradient_loss"]:
                if key in loss_dict:
                    running[key] += loss_dict[key].item()
                    counts[key] += 1

    n = len(loader)

    return {
        "loss": running["loss"] / n,
        "l1_loss": running["l1_loss"] / counts["l1_loss"] if counts["l1_loss"] > 0 else None,
        "region_l1_loss": running["region_l1_loss"] / counts["region_l1_loss"] if counts["region_l1_loss"] > 0 else None,
        "ssim_loss": running["ssim_loss"] / counts["ssim_loss"] if counts["ssim_loss"] > 0 else None,
        "gradient_loss": running["gradient_loss"] / counts["gradient_loss"] if counts["gradient_loss"] > 0 else None
    }


def build_criterion(args):
    if args.loss_type == "combined":
        print("Using CombinedLoss")
        return CombinedLoss(
            l1_weight=args.l1_weight,
            ssim_weight=args.ssim_weight,
            gradient_weight=args.gradient_weight
        )

    if args.loss_type == "region":
        print("Using CombinedRegionLoss")
        print(f"Air/lung weight: {args.air_lung_weight}")
        print(f"Soft tissue weight: {args.soft_tissue_weight}")
        print(f"Bone weight: {args.bone_weight}")

        return CombinedRegionLoss(
            l1_weight=args.l1_weight,
            ssim_weight=args.ssim_weight,
            gradient_weight=args.gradient_weight,
            min_hu=args.min_hu,
            max_hu=args.max_hu,
            air_lung_weight=args.air_lung_weight,
            soft_tissue_weight=args.soft_tissue_weight,
            bone_weight=args.bone_weight
        )

    print("Using MaskedL1Loss")
    return MaskedL1Loss()


def load_existing_log(log_path):
    if os.path.exists(log_path):
        history_df = pd.read_csv(log_path)
        history = history_df.to_dict("records")
        print(f"Loaded existing log with {len(history)} epochs.")
        return history

    return []


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    df = pd.read_csv(args.patient_csv)

    with open(args.split_json, "r") as f:
        split = json.load(f)

    train_dataset = SynthRAD2p5DDataset(
        dataframe=df,
        patient_ids=split["train"],
        input_slices=args.input_slices,
        image_size=args.image_size,
        min_hu=args.min_hu,
        max_hu=args.max_hu,
        max_patients=args.max_train_patients
    )

    val_dataset = SynthRAD2p5DDataset(
        dataframe=df,
        patient_ids=split["val"],
        input_slices=args.input_slices,
        image_size=args.image_size,
        min_hu=args.min_hu,
        max_hu=args.max_hu,
        max_patients=args.max_val_patients
    )

    print("Train patients used:", len(train_dataset.dataframe))
    print("Val patients used:", len(val_dataset.dataframe))
    print("Train samples:", len(train_dataset))
    print("Val samples:", len(val_dataset))

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    model = build_unetplusplus_resnet34(
        in_channels=args.input_slices,
        out_channels=1,
        encoder_weights=args.encoder_weights,
        dropout=args.dropout
    ).to(device)

    criterion = build_criterion(args)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    log_path = os.path.join(args.log_dir, f"training_log_fold{args.fold}.csv")

    start_epoch = 0
    best_val_loss = float("inf")
    history = []

    if args.resume_checkpoint is not None:
        print(f"Resuming from checkpoint: {args.resume_checkpoint}")

        checkpoint = torch.load(args.resume_checkpoint, map_location=device)

        model.load_state_dict(checkpoint["model_state_dict"])

        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            print("Loaded optimizer state.")

        start_epoch = checkpoint.get("epoch", 0)
        best_val_loss = checkpoint.get("best_val_loss", checkpoint.get("val_loss", float("inf")))

        if args.resume_log and os.path.exists(log_path):
            history = load_existing_log(log_path)
            if len(history) > 0:
                best_val_loss = min([row["val_loss"] for row in history if "val_loss" in row])
        else:
            history = []

        print(f"Starting from epoch: {start_epoch + 1}")
        print(f"Current best validation loss: {best_val_loss}")

    else:
        print("Starting training from scratch.")

    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        train_stats = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device
        )

        val_stats = validate_one_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device
        )

        train_loss = train_stats["loss"]
        val_loss = val_stats["loss"]

        print(f"Train loss: {train_loss:.5f}")
        print(f"Val loss:   {val_loss:.5f}")

        if train_stats["l1_loss"] is not None:
            print(f"Train L1: {train_stats['l1_loss']:.5f} | Val L1: {val_stats['l1_loss']:.5f}")

        if train_stats["region_l1_loss"] is not None:
            print(f"Train region L1: {train_stats['region_l1_loss']:.5f} | Val region L1: {val_stats['region_l1_loss']:.5f}")

        if train_stats["ssim_loss"] is not None:
            print(f"Train SSIM loss: {train_stats['ssim_loss']:.5f} | Val SSIM loss: {val_stats['ssim_loss']:.5f}")

        if train_stats["gradient_loss"] is not None:
            print(f"Train Grad loss: {train_stats['gradient_loss']:.5f} | Val Grad loss: {val_stats['gradient_loss']:.5f}")

        row = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_l1_loss": train_stats["l1_loss"],
            "val_l1_loss": val_stats["l1_loss"],
            "train_region_l1_loss": train_stats["region_l1_loss"],
            "val_region_l1_loss": val_stats["region_l1_loss"],
            "train_ssim_loss": train_stats["ssim_loss"],
            "val_ssim_loss": val_stats["ssim_loss"],
            "train_gradient_loss": train_stats["gradient_loss"],
            "val_gradient_loss": val_stats["gradient_loss"]
        }

        history.append(row)
        pd.DataFrame(history).to_csv(log_path, index=False)

        last_ckpt_path = os.path.join(args.checkpoint_dir, f"last_fold{args.fold}.pth")
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "best_val_loss": best_val_loss,
                "args": vars(args)
            },
            last_ckpt_path
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            best_ckpt_path = os.path.join(args.checkpoint_dir, f"best_fold{args.fold}.pth")
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "best_val_loss": best_val_loss,
                    "args": vars(args)
                },
                best_ckpt_path
            )

            print(f"Saved best checkpoint with val loss: {best_val_loss:.5f}")

    print("\nTraining finished.")
    print("Best validation loss:", best_val_loss)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--patient_csv", type=str, default="data/patient_list.csv")
    parser.add_argument("--split_json", type=str, default=None)

    parser.add_argument("--image_size", type=int, default=320)
    parser.add_argument("--input_slices", type=int, default=7)
    parser.add_argument("--min_hu", type=int, default=-1000)
    parser.add_argument("--max_hu", type=int, default=1600)

    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--encoder_weights", type=str, default="imagenet")

    parser.add_argument("--max_train_patients", type=int, default=10)
    parser.add_argument("--max_val_patients", type=int, default=3)
    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument("--loss_type", type=str, default="l1", choices=["l1", "combined", "region"])
    parser.add_argument("--l1_weight", type=float, default=1.0)
    parser.add_argument("--ssim_weight", type=float, default=0.2)
    parser.add_argument("--gradient_weight", type=float, default=0.05)

    parser.add_argument("--air_lung_weight", type=float, default=2.0)
    parser.add_argument("--soft_tissue_weight", type=float, default=1.0)
    parser.add_argument("--bone_weight", type=float, default=1.5)

    parser.add_argument("--resume_checkpoint", type=str, default=None)
    parser.add_argument("--resume_log", action="store_true")

    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/test_training")
    parser.add_argument("--log_dir", type=str, default="results/metrics")

    args = parser.parse_args()

    if args.split_json is None:
        args.split_json = f"splits/fold_{args.fold}.json"

    main(args)
