# Compute the Kernel Inception Distance (KID) metric

import numpy as np
from . import metric_utils

def compute_kid(opts, num_subsets, max_subset_size):
    # Direct TorchScript translation of http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
    detector_url = "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt"
    detector_kwargs = dict(return_features = True) # Return raw features before the softmax layer.

    real_features = metric_utils.compute_feature_stats_for_dataset(
        opts = opts, detector_url = detector_url, detector_kwargs = detector_kwargs,
        rel_lo = 0, rel_hi = 0, capture_all = True).get_all()

    gen_features = metric_utils.compute_feature_stats_for_generator(
        opts = opts, detector_url = detector_url, detector_kwargs = detector_kwargs,
        rel_lo = 0, rel_hi = 1, capture_all = True).get_all()

    if opts.rank != 0:
        return float("nan")

    n = real_features.shape[1]
    m = min(min(real_features.shape[0], gen_features.shape[0]), max_subset_size)
    t = 0
    for _subset_idx in range(num_subsets):
        x = gen_features[np.random.choice(gen_features.shape[0], m, replace = False)]
        y = real_features[np.random.choice(real_features.shape[0], m, replace = False)]
        a = (x @ x.T / n + 1) ** 3 + (y @ y.T / n + 1) ** 3
        b = (x @ y.T / n + 1) ** 3
        t += (a.sum() - np.diag(a).sum()) / (m - 1) - b.sum() * 2 / m
    kid = t / num_subsets / m
    return float(kid)
