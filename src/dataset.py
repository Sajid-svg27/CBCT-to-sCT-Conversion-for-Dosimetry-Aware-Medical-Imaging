import numpy as np
import torch
from torch.utils.data import Dataset

from .preprocessing import (
    read_mha,
    normalize_hu,
    prepare_mask,
    center_crop_or_pad_3d
)


class SynthRAD2p5DDataset(Dataset):
    

    def __init__(
        self,
        dataframe,
        patient_ids,
        input_slices=7,
        image_size=320,
        min_hu=-1000,
        max_hu=1600,
        use_mask=True,
        max_patients=None
    ):
        self.dataframe = dataframe[dataframe["patient_id"].isin(patient_ids)].reset_index(drop=True)

        if max_patients is not None:
            self.dataframe = self.dataframe.iloc[:max_patients].reset_index(drop=True)

        self.input_slices = input_slices
        self.half = input_slices // 2
        self.image_size = image_size
        self.min_hu = min_hu
        self.max_hu = max_hu
        self.use_mask = use_mask

        self.volumes = {}
        self.index = []

        self._load_volumes_and_build_index()

    def _load_volumes_and_build_index(self):
        for _, row in self.dataframe.iterrows():
            patient_id = row["patient_id"]

            _, cbct, _, _, _ = read_mha(row["cbct_path"])
            _, ct, _, _, _ = read_mha(row["ct_path"])
            _, mask, _, _, _ = read_mha(row["mask_path"])

            cbct = normalize_hu(cbct, self.min_hu, self.max_hu)
            ct = normalize_hu(ct, self.min_hu, self.max_hu)
            mask = prepare_mask(mask)

            cbct = center_crop_or_pad_3d(
                cbct,
                target_size=self.image_size,
                pad_value=-1
            )

            ct = center_crop_or_pad_3d(
                ct,
                target_size=self.image_size,
                pad_value=-1
            )

            mask = center_crop_or_pad_3d(
                mask,
                target_size=self.image_size,
                pad_value=0
            )

            self.volumes[patient_id] = {
                "cbct": cbct,
                "ct": ct,
                "mask": mask,
                "region": row["region"]
            }

            num_slices = cbct.shape[0]

            
            for z in range(self.half, num_slices - self.half):
                
                if self.use_mask:
                    body_fraction = mask[z].mean()
                    if body_fraction < 0.01:
                        continue

                self.index.append((patient_id, z))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        patient_id, z = self.index[idx]

        cbct = self.volumes[patient_id]["cbct"]
        ct = self.volumes[patient_id]["ct"]
        mask = self.volumes[patient_id]["mask"]

        cbct_stack = cbct[z - self.half : z + self.half + 1]
        ct_slice = ct[z]
        mask_slice = mask[z]

        cbct_stack = torch.tensor(cbct_stack, dtype=torch.float32)
        ct_slice = torch.tensor(ct_slice[None, :, :], dtype=torch.float32)
        mask_slice = torch.tensor(mask_slice[None, :, :], dtype=torch.float32)

        return {
            "input": cbct_stack,
            "target": ct_slice,
            "mask": mask_slice,
            "patient_id": patient_id,
            "slice_index": z
        }
