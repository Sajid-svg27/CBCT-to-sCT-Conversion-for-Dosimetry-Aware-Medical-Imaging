import numpy as np
import SimpleITK as sitk


def read_mha(path):
    
    image = sitk.ReadImage(path)
    array = sitk.GetArrayFromImage(image)
    spacing = image.GetSpacing()
    origin = image.GetOrigin()
    direction = image.GetDirection()
    return image, array, spacing, origin, direction


def clip_hu(image, min_hu=-1000, max_hu=1600):
    return np.clip(image, min_hu, max_hu)


def normalize_hu(image, min_hu=-1000, max_hu=1600):
    
    image = clip_hu(image, min_hu, max_hu)
    image = 2.0 * (image - min_hu) / (max_hu - min_hu) - 1.0
    return image.astype(np.float32)


def denormalize_hu(image_norm, min_hu=-1000, max_hu=1600):
    
    image = (image_norm + 1.0) / 2.0
    image = image * (max_hu - min_hu) + min_hu
    return image.astype(np.float32)


def prepare_mask(mask):
    
    return (mask > 0).astype(np.float32)


def apply_mask(image, mask, background_value=-1000):
    
    image_masked = image.copy()
    image_masked[mask == 0] = background_value
    return image_masked


def center_crop_or_pad_2d(image, target_size=320, pad_value=0):
    
    h, w = image.shape

    crop_y_start = max((h - target_size) // 2, 0)
    crop_x_start = max((w - target_size) // 2, 0)

    crop_y_end = crop_y_start + min(h, target_size)
    crop_x_end = crop_x_start + min(w, target_size)

    cropped = image[crop_y_start:crop_y_end, crop_x_start:crop_x_end]

    
    new_h, new_w = cropped.shape

    pad_y_before = max((target_size - new_h) // 2, 0)
    pad_y_after = max(target_size - new_h - pad_y_before, 0)

    pad_x_before = max((target_size - new_w) // 2, 0)
    pad_x_after = max(target_size - new_w - pad_x_before, 0)

    fixed = np.pad(
        cropped,
        ((pad_y_before, pad_y_after), (pad_x_before, pad_x_after)),
        mode="constant",
        constant_values=pad_value
    )

    return fixed.astype(image.dtype)


def center_crop_or_pad_3d(volume, target_size=320, pad_value=0):
    
    fixed_slices = []

    for z in range(volume.shape[0]):
        fixed_slice = center_crop_or_pad_2d(
            volume[z],
            target_size=target_size,
            pad_value=pad_value
        )
        fixed_slices.append(fixed_slice)

    return np.stack(fixed_slices, axis=0)


def get_middle_slice(array):
    z = array.shape[0] // 2
    return array[z], z
