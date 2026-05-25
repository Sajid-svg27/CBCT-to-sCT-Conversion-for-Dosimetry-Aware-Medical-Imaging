import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


def build_unetplusplus_resnet34(
    in_channels=7,
    out_channels=1,
    encoder_weights="imagenet",
    dropout=0.2
):
    

    model = smp.UnetPlusPlus(
        encoder_name="resnet34",
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=out_channels,
        activation=None
    )

    
    model.segmentation_head = nn.Sequential(
        nn.Dropout2d(p=dropout),
        model.segmentation_head
    )

    return model
