# Compute the Inception Score (IS) metric

import numpy as np
from . import metric_utils

def compute_is(opts, num_splits):
    # Direct TorchScript translation of http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
    detector_url = "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt"
    detector_kwargs = dict(no_output_bias = True) # Match the original implementation by not applying bias in the softmax layer.

    gen_probs = metric_utils.compute_feature_stats_for_generator(
        opts = opts, detector_url = detector_url, detector_kwargs = detector_kwargs,
        capture_all = True).get_all()

    if opts.rank != 0:
        return float("nan"), float("nan")

    scores = []
    for i in range(num_splits):
        part = gen_probs[i * opts.max_items // num_splits : (i + 1) * opts.max_items // num_splits]
        kl = part * (np.log(part) - np.log(np.mean(part, axis = 0, keepdims = True)))
        kl = np.mean(np.sum(kl, axis = 1))
        scores.append(np.exp(kl))
    return float(np.mean(scores)), float(np.std(scores))
