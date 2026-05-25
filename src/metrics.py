import numpy as np
from skimage.metrics import structural_similarity as ssim_metric
from skimage.metrics import peak_signal_noise_ratio as psnr_metric


def denormalize_to_hu(image_norm, min_hu=-1000, max_hu=1600):
    
    image = (image_norm + 1.0) / 2.0
    image = image * (max_hu - min_hu) + min_hu
    return image.astype(np.float32)


def mae_hu(pred_hu, target_hu, mask=None):
    
    error = np.abs(pred_hu - target_hu)

    if mask is not None:
        mask = mask > 0
        return error[mask].mean()

    return error.mean()


def rmse_hu(pred_hu, target_hu, mask=None):
    
    error_sq = (pred_hu - target_hu) ** 2

    if mask is not None:
        mask = mask > 0
        return np.sqrt(error_sq[mask].mean())

    return np.sqrt(error_sq.mean())


def psnr_hu(pred_hu, target_hu, data_range=2600, mask=None):
    
    if mask is not None:
        mask = mask > 0

        
        pred_values = pred_hu[mask]
        target_values = target_hu[mask]

        mse = np.mean((pred_values - target_values) ** 2)
        if mse == 0:
            return float("inf")

        return 20 * np.log10(data_range / np.sqrt(mse))

    return psnr_metric(target_hu, pred_hu, data_range=data_range)


def ssim_hu(pred_hu, target_hu, data_range=2600):
    
    return ssim_metric(target_hu, pred_hu, data_range=data_range)


def region_masks_from_ct(ct_hu, body_mask=None):
    
    if body_mask is None:
        body_mask = np.ones_like(ct_hu, dtype=bool)
    else:
        body_mask = body_mask > 0

    air_lung = (ct_hu < -300) & body_mask
    soft_tissue = (ct_hu >= -300) & (ct_hu <= 200) & body_mask
    bone = (ct_hu > 200) & body_mask

    return {
        "air_lung": air_lung,
        "soft_tissue": soft_tissue,
        "bone": bone,
        "body": body_mask
    }


def calculate_all_metrics(pred_norm, target_norm, input_norm=None, mask=None, min_hu=-1000, max_hu=1600):
    
    pred_hu = denormalize_to_hu(pred_norm, min_hu, max_hu)
    target_hu = denormalize_to_hu(target_norm, min_hu, max_hu)

    if input_norm is not None:
        input_hu = denormalize_to_hu(input_norm, min_hu, max_hu)
    else:
        input_hu = None

    if mask is not None:
        mask = mask > 0

    results = {}

    results["sct_mae_hu"] = mae_hu(pred_hu, target_hu, mask)
    results["sct_rmse_hu"] = rmse_hu(pred_hu, target_hu, mask)
    results["sct_psnr"] = psnr_hu(pred_hu, target_hu, data_range=max_hu-min_hu, mask=mask)
    results["sct_ssim"] = ssim_hu(pred_hu, target_hu, data_range=max_hu-min_hu)

    if input_hu is not None:
        results["raw_cbct_mae_hu"] = mae_hu(input_hu, target_hu, mask)
        results["raw_cbct_rmse_hu"] = rmse_hu(input_hu, target_hu, mask)
        results["raw_cbct_psnr"] = psnr_hu(input_hu, target_hu, data_range=max_hu-min_hu, mask=mask)
        results["raw_cbct_ssim"] = ssim_hu(input_hu, target_hu, data_range=max_hu-min_hu)

    region_masks = region_masks_from_ct(target_hu, body_mask=mask)

    for region_name, region_mask in region_masks.items():
        if region_mask.sum() > 0:
            results[f"sct_mae_{region_name}_hu"] = mae_hu(pred_hu, target_hu, region_mask)

            if input_hu is not None:
                results[f"raw_cbct_mae_{region_name}_hu"] = mae_hu(input_hu, target_hu, region_mask)
        else:
            results[f"sct_mae_{region_name}_hu"] = np.nan

            if input_hu is not None:
                results[f"raw_cbct_mae_{region_name}_hu"] = np.nan

    return results
