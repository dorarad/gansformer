# Inception Score (IS).
import numpy as np
import tensorflow as tf
import dnnlib.tflib as tflib
import glob
import PIL.Image

from metrics import metric_base
from training import misc

class IS(metric_base.MetricBase):
    def __init__(self, num_splits, minibatch_per_gpu, **kwargs):
        super().__init__(**kwargs)
        self.num_splits = num_splits
        self.minibatch_per_gpu = minibatch_per_gpu

    def _evaluate(self, Gs, Gs_kwargs, num_gpus, num_imgs, paths = None, ratio = 1.0, **kwargs):
        minibatch_size = num_gpus * self.minibatch_per_gpu
        inception = misc.load_pkl("http://d36zk2xti64re0.cloudfront.net/stylegan1/networks/metrics/inception_v3_softmax.pkl")

        def inception_func(imgs): 
            return inception.run(misc.crop_tensor(imgs[:end-begin], ratio), num_gpus = num_gpus, assume_frozen = True)

        if paths is not None:
            # Extract features for local sample image files (paths)
            feats = self._paths_to_feats(paths, inception_func, minibatch_size, num_gpus, num_imgs)
        else:
            # Extract features for newly generated fake images
            feats = self._gen_feats(Gs, inception, minibatch_size, num_imgs, num_gpus, Gs_kwargs)

        # Compute IS
        scores = []
        for i in range(self.num_splits):
            part = feats[i * num_imgs // self.num_splits : (i + 1) * num_imgs // self.num_splits]
            kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
            kl = np.mean(np.sum(kl, axis = 1))
            scores.append(np.exp(kl))
        self._report_result(np.mean(scores), suffix = "_mean")
        self._report_result(np.std(scores), suffix = "_std")
