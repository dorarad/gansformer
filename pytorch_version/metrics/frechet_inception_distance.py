# Compute the Frechet Inception Distance (FID) metric

import numpy as np
import scipy.linalg
from . import metric_utils

def compute_fid(opts):
    # Direct TorchScript translation of http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
    detector_url = "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt"
    detector_kwargs = dict(return_features = True) # Return raw features before the softmax layer.

    mu_real, sigma_real = metric_utils.compute_feature_stats_for_dataset(
        opts = opts, detector_url = detector_url, detector_kwargs = detector_kwargs,
        rel_lo = 0, rel_hi = 0, capture_mean_cov = True).get_mean_cov()

    mu_gen, sigma_gen = metric_utils.compute_feature_stats_for_generator(
        opts = opts, detector_url = detector_url, detector_kwargs = detector_kwargs,
        rel_lo = 0, rel_hi = 1, capture_mean_cov = True).get_mean_cov()

    if opts.rank != 0:
        return float("nan")

    m = np.square(mu_gen - mu_real).sum()
    s, _ = scipy.linalg.sqrtm(np.dot(sigma_gen, sigma_real), disp = False)
    fid = np.real(m + np.trace(sigma_gen + sigma_real - s * 2))
    return float(fid)
