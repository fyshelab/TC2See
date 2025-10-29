from typing import Union, Sequence, Optional

import h5py
from tqdm import tqdm
import numpy as np
from scipy.ndimage import map_coordinates

from pylablib.core.utils.funcargparse import is_sequence


def load_data(
        path: str,
        subject: str,
        tr_offset: Union[float, Sequence[float]],
        run_normalize: str,
        interpolation: bool,
        interpolation_order: int = 1,
        run_ids: Optional[Sequence[int]] = None,
):
    if not is_sequence(tr_offset):
        tr_offset = [tr_offset]

    with h5py.File(path, 'r') as f:
        group = f[subject]
        stimulus_ids = group['stimulus_ids'][:]
        stimulus_trs = group['stimulus_trs'][:]
        # check if data is a volume or surface
        is_data_volume = 'fmri_mask' in group and 'affine' in group

        num_runs, num_trs, num_voxels = group['bold'].shape

        if is_data_volume:
            mask = group['fmri_mask'][:]
            affine = group['affine'][:]

        if run_ids is None:
            run_ids = list(range(num_runs))

        bolds = []
        for offset in tr_offset:
            ids = []
            bold = []

            for i in run_ids:
                in_range = (stimulus_trs[i] + max(tr_offset)) < num_trs

                run_trs = stimulus_trs[i] + offset
                run_ids = stimulus_ids[i]

                if not interpolation:
                    run_trs = np.rint(run_trs).astype(int)
                run_trs = run_trs[in_range]
                run_ids = run_ids[in_range]

                if not interpolation:
                    run_bold = group['bold'][i, run_trs]
                else:
                    run_bold = group['bold'][i]
                    x = []
                    for t in tqdm(run_trs):
                        coords = np.stack([np.full(num_voxels, t), np.arange(num_voxels)])
                        x.append(map_coordinates(run_bold, coords, order=interpolation_order))
                    run_bold = np.stack(x)

                if run_normalize == 'zscore':
                    run_bold = (run_bold - group['bold_mean'][i]) / group['bold_std'][i]
                elif run_normalize == 'linear_trend':
                    trend_coeffs = np.stack([run_trs, np.ones_like(run_trs)], axis=1).astype(float)
                    predicted_bold = trend_coeffs @ group['bold_trend'][i]
                    run_bold = (run_bold - predicted_bold) / group['bold_trend_std'][i]
                bold.append(run_bold.astype(np.float32))
                ids.append(run_ids)
            bold = np.concatenate(bold)
            ids = np.concatenate(ids)

            bold = (bold - np.nanmean(bold, axis=0)) / np.nanstd(bold, axis=0)
            bolds.append(bold)
    if len(bolds) == 1:
        bold = bolds[0]
    else:
        bold = np.stack(bolds)
    
    if is_data_volume:
        return bold, ids, mask, affine # blood oxygen level dependent response
    else:
        return bold, ids
