import os
import json
import argparse
import pandas as pd
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from src.dataset import SynthRAD2p5DDataset
from src.model import build_unetplusplus_resnet34


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    df = pd.read_csv(args.patient_csv)

    with open(args.split_json, "r") as f:
        split = json.load(f)

    val_dataset = SynthRAD2p5DDataset(
        dataframe=df,
        patient_ids=split["val"],
        input_slices=args.input_slices,
        image_size=args.image_size,
        min_hu=args.min_hu,
        max_hu=args.max_hu,
        max_patients=args.max_val_patients
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )

    model = build_unetplusplus_resnet34(
        in_channels=args.input_slices,
        out_channels=1,
        encoder_weights=None,
        dropout=args.dropout
    ).to(device)

    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    batch = next(iter(val_loader))

    inputs = batch["input"].to(device)
    targets = batch["target"].to(device)

    with torch.no_grad():
        outputs = model(inputs)

    input_np = inputs[0].cpu().numpy()
    target_np = targets[0, 0].cpu().numpy()
    output_np = outputs[0, 0].cpu().numpy()

    center = input_np.shape[0] // 2

    os.makedirs(args.output_dir, exist_ok=True)

    plt.figure(figsize=(16, 4))

    plt.subplot(1, 4, 1)
    plt.imshow(input_np[center], cmap="gray", vmin=-1, vmax=1)
    plt.title("Input CBCT")
    plt.axis("off")

    plt.subplot(1, 4, 2)
    plt.imshow(target_np, cmap="gray", vmin=-1, vmax=1)
    plt.title("Target CT")
    plt.axis("off")

    plt.subplot(1, 4, 3)
    plt.imshow(output_np, cmap="gray", vmin=-1, vmax=1)
    plt.title("Predicted sCT")
    plt.axis("off")

    plt.subplot(1, 4, 4)
    plt.imshow(abs(output_np - target_np), cmap="hot")
    plt.title("Error Map")
    plt.axis("off")

    plt.tight_layout()

    save_path = os.path.join(args.output_dir, "day7_prediction_preview.png")
    plt.savefig(save_path, dpi=200)
    plt.show()

    print("Saved preview to:", save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--patient_csv", type=str, default="data/patient_list.csv")
    parser.add_argument("--split_json", type=str, default="splits/fold_0.json")
    parser.add_argument("--checkpoint_path", type=str, required=True)

    parser.add_argument("--image_size", type=int, default=320)
    parser.add_argument("--input_slices", type=int, default=7)
    parser.add_argument("--min_hu", type=int, default=-1000)
    parser.add_argument("--max_hu", type=int, default=1600)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--max_val_patients", type=int, default=2)

    parser.add_argument("--output_dir", type=str, default="results/figures/predictions")

    args = parser.parse_args()
    main(args)
