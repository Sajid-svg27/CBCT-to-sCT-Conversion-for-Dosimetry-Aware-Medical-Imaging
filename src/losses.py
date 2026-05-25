import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedL1Loss(nn.Module):
  

    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, prediction, target, mask=None):
        error = torch.abs(prediction - target)

        if mask is None:
            return error.mean()

        masked_error = error * mask
        return masked_error.sum() / (mask.sum() + self.eps)


class RegionWeightedL1Loss(nn.Module):
    

    def __init__(
        self,
        min_hu=-1000,
        max_hu=1600,
        air_lung_weight=2.0,
        soft_tissue_weight=1.0,
        bone_weight=1.5,
        eps=1e-6
    ):
        super().__init__()

        self.min_hu = min_hu
        self.max_hu = max_hu
        self.air_lung_weight = air_lung_weight
        self.soft_tissue_weight = soft_tissue_weight
        self.bone_weight = bone_weight
        self.eps = eps

    def normalized_to_hu(self, image_norm):
        image = (image_norm + 1.0) / 2.0
        image = image * (self.max_hu - self.min_hu) + self.min_hu
        return image

    def forward(self, prediction, target, mask=None):
        target_hu = self.normalized_to_hu(target)

        weights = torch.ones_like(target)

        air_lung = target_hu < -300
        soft_tissue = (target_hu >= -300) & (target_hu <= 200)
        bone = target_hu > 200

        weights[air_lung] = self.air_lung_weight
        weights[soft_tissue] = self.soft_tissue_weight
        weights[bone] = self.bone_weight

        error = torch.abs(prediction - target)
        weighted_error = error * weights

        if mask is not None:
            weighted_error = weighted_error * mask
            weights = weights * mask

        return weighted_error.sum() / (weights.sum() + self.eps)


class SSIMLoss(nn.Module):
    

    def __init__(self, window_size=11, eps=1e-6):
        super().__init__()
        self.window_size = window_size
        self.eps = eps

    def forward(self, prediction, target):
        prediction = (prediction + 1.0) / 2.0
        target = (target + 1.0) / 2.0

        prediction = torch.clamp(prediction, 0.0, 1.0)
        target = torch.clamp(target, 0.0, 1.0)

        pad = self.window_size // 2

        mu_pred = F.avg_pool2d(prediction, self.window_size, stride=1, padding=pad)
        mu_target = F.avg_pool2d(target, self.window_size, stride=1, padding=pad)

        mu_pred_sq = mu_pred ** 2
        mu_target_sq = mu_target ** 2
        mu_pred_target = mu_pred * mu_target

        sigma_pred_sq = F.avg_pool2d(prediction ** 2, self.window_size, stride=1, padding=pad) - mu_pred_sq
        sigma_target_sq = F.avg_pool2d(target ** 2, self.window_size, stride=1, padding=pad) - mu_target_sq
        sigma_pred_target = F.avg_pool2d(prediction * target, self.window_size, stride=1, padding=pad) - mu_pred_target

        c1 = 0.01 ** 2
        c2 = 0.03 ** 2

        ssim_map = ((2 * mu_pred_target + c1) * (2 * sigma_pred_target + c2)) / (
            (mu_pred_sq + mu_target_sq + c1) *
            (sigma_pred_sq + sigma_target_sq + c2) + self.eps
        )

        return 1.0 - ssim_map.mean()


class GradientLoss(nn.Module):
    

    def __init__(self):
        super().__init__()

    def forward(self, prediction, target, mask=None):
        pred_dx = torch.abs(prediction[:, :, :, 1:] - prediction[:, :, :, :-1])
        target_dx = torch.abs(target[:, :, :, 1:] - target[:, :, :, :-1])

        pred_dy = torch.abs(prediction[:, :, 1:, :] - prediction[:, :, :-1, :])
        target_dy = torch.abs(target[:, :, 1:, :] - target[:, :, :-1, :])

        loss_x = torch.abs(pred_dx - target_dx)
        loss_y = torch.abs(pred_dy - target_dy)

        if mask is not None:
            mask_x = mask[:, :, :, 1:] * mask[:, :, :, :-1]
            mask_y = mask[:, :, 1:, :] * mask[:, :, :-1, :]

            loss_x = (loss_x * mask_x).sum() / (mask_x.sum() + 1e-6)
            loss_y = (loss_y * mask_y).sum() / (mask_y.sum() + 1e-6)

            return loss_x + loss_y

        return loss_x.mean() + loss_y.mean()


class CombinedLoss(nn.Module):
    

    def __init__(
        self,
        l1_weight=1.0,
        ssim_weight=0.2,
        gradient_weight=0.05
    ):
        super().__init__()

        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight
        self.gradient_weight = gradient_weight

        self.l1_loss = MaskedL1Loss()
        self.ssim_loss = SSIMLoss()
        self.gradient_loss = GradientLoss()

    def forward(self, prediction, target, mask=None):
        l1 = self.l1_loss(prediction, target, mask)
        ssim = self.ssim_loss(prediction, target)
        grad = self.gradient_loss(prediction, target, mask)

        total = (
            self.l1_weight * l1
            + self.ssim_weight * ssim
            + self.gradient_weight * grad
        )

        loss_dict = {
            "total_loss": total,
            "l1_loss": l1.detach(),
            "ssim_loss": ssim.detach(),
            "gradient_loss": grad.detach()
        }

        return total, loss_dict


class CombinedRegionLoss(nn.Module):
    

    def __init__(
        self,
        l1_weight=1.0,
        ssim_weight=0.2,
        gradient_weight=0.05,
        min_hu=-1000,
        max_hu=1600,
        air_lung_weight=2.0,
        soft_tissue_weight=1.0,
        bone_weight=1.5
    ):
        super().__init__()

        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight
        self.gradient_weight = gradient_weight

        self.region_l1_loss = RegionWeightedL1Loss(
            min_hu=min_hu,
            max_hu=max_hu,
            air_lung_weight=air_lung_weight,
            soft_tissue_weight=soft_tissue_weight,
            bone_weight=bone_weight
        )

        self.ssim_loss = SSIMLoss()
        self.gradient_loss = GradientLoss()

    def forward(self, prediction, target, mask=None):
        region_l1 = self.region_l1_loss(prediction, target, mask)
        ssim = self.ssim_loss(prediction, target)
        grad = self.gradient_loss(prediction, target, mask)

        total = (
            self.l1_weight * region_l1
            + self.ssim_weight * ssim
            + self.gradient_weight * grad
        )

        loss_dict = {
            "total_loss": total,
            "region_l1_loss": region_l1.detach(),
            "ssim_loss": ssim.detach(),
            "gradient_loss": grad.detach()
        }

        return total, loss_dict
