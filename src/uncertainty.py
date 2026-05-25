import torch
import torch.nn as nn


def enable_dropout_only(model):
    
    model.eval()

    for module in model.modules():
        if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            module.train()

    return model


def mc_dropout_predict(model, inputs, n_samples=20):
    
    model = enable_dropout_only(model)

    predictions = []

    with torch.no_grad():
        for _ in range(n_samples):
            pred = model(inputs)
            predictions.append(pred.unsqueeze(0))

    predictions = torch.cat(predictions, dim=0)

    mean_prediction = predictions.mean(dim=0)
    std_prediction = predictions.std(dim=0)

    return mean_prediction, std_prediction, predictions
