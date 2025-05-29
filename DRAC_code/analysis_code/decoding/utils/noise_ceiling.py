import numpy as np


def compute_ncsnr(
        betas: np.ndarray,
        stimulus_ids: np.ndarray,
):
    """
    Computes the noise ceiling signal to noise ratio.

    :param betas: Array of betas or other neural data with shape (num_betas, num_voxels)
    :param stimulus_ids: Array that specifies the stimulus that betas correspond to, shape (num_betas)
    :return: Array of noise ceiling snr values with shape (num_voxels)
    """

    unique_ids = np.unique(stimulus_ids)

    betas_var = []
    for i in unique_ids:
        stimulus_betas = betas[stimulus_ids == i]
        betas_var.append(stimulus_betas.var(axis=0, ddof=1))
    betas_var_mean = np.nanmean(np.stack(betas_var), axis=0)

    std_noise = np.sqrt(betas_var_mean)

    std_signal = 1. - betas_var_mean
    std_signal[std_signal < 0.] = 0.
    std_signal = np.sqrt(std_signal)
    ncsnr = std_signal / std_noise

    return ncsnr


def compute_nc(ncsnr: np.ndarray, num_averages: int = 1):
    """
    Convert the noise ceiling snr to the actual noise ceiling estimate

    :param ncsnr: Array of noise ceiling snr values with shape (num_voxels)
    :param num_averages: Set to the number of repetitions that will be averaged together
        If there are repetitions that won't be averaged, then leave this as 1
    :return: Array of noise ceiling values with shape (num_voxels)
    """
    ncsnr_squared = ncsnr ** 2
    nc = 100. * ncsnr_squared / (ncsnr_squared + (1. / num_averages))
    return nc

