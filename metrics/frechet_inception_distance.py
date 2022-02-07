# Frechet Inception Distance (FID)
import os
import numpy as np
import scipy
import tensorflow as tf
import dnnlib.tflib as tflib
from dnnlib.tflib.ops.upfirdn_2d import upsample_2d, downsample_2d
import PIL.Image
import glob

from metrics import metric_base
from training import misc

class FID(metric_base.MetricBase):
    def __init__(self, batch_per_gpu, **kwargs):
        super().__init__(**kwargs)
        self.batch_per_gpu = batch_per_gpu

    def _feats_to_stats(self, feats):
        mu = np.mean(feats, axis = 0)
        sigma = np.cov(feats, rowvar = False)
        return mu, sigma

    def compute_fid(self, mu_real, sigma_real, mu_fake, sigma_fake):
        m = np.square(mu_fake - mu_real).sum()
        s, _ = scipy.linalg.sqrtm(np.dot(sigma_fake, sigma_real), disp = False)
        fid = np.real(m + np.trace(sigma_fake + sigma_real - 2*s))
        return fid

    def _evaluate(self, Gs, Gs_kwargs, num_gpus, num_imgs, ratio = 1.0, paths = None, **kwargs):
        batch_size = num_gpus * self.batch_per_gpu
        featurizer = misc.load_pkl("http://d36zk2xti64re0.cloudfront.net/stylegan1/networks/metrics/inception_v3_features.pkl")

        # Compute statistics for reals
        cache_file = self._get_cache_file_for_reals(num_imgs, ratio)
        os.makedirs(os.path.dirname(cache_file), exist_ok = True)
        if os.path.isfile(cache_file):
            mu_real, sigma_real = misc.load_pkl(cache_file)
        else:
            imgs_iter = self._iterate_reals(batch_size = batch_size)
            feats_real = self._get_feats(imgs_iter, featurizer, batch_size, ratio, num_gpus, num_imgs)
            mu_real, sigma_real = self._feats_to_stats(feats_real)
            misc.save_pkl((mu_real, sigma_real), cache_file)

        if paths is not None:
            # Extract features for local sample image files (paths)
            feats = self._paths_to_feats(paths, featurizer, batch_size, ratio, num_gpus, num_imgs)
        else:
            # Extract features for newly generated fake images
            feats = self._gen_feats(Gs, featurizer, batch_size, ratio, num_imgs, num_gpus, Gs_kwargs)

        # Compute FID
        mu_fake, sigma_fake = self._feats_to_stats(feats)
        self._report_result(self.compute_fid(mu_real, sigma_real, mu_fake, sigma_fake))
