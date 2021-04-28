# Common definitions for GAN metrics: provides iterators over the datasets,
# file caching, progress reports and printing.
import os
import time
import pickle
import hashlib
import numpy as np
import tensorflow as tf
import dnnlib
import dnnlib.tflib as tflib

from training import misc
from training import dataset
from tqdm import tqdm

# Base class for metrics
class MetricBase:
    def __init__(self, name):
        self.name = name
        self._dataset_obj = None
        self._progress_lo = None
        self._progress_hi = None
        self._progress_max = None
        self._progress_sec = None
        self._progress_time = None
        self._reset()

    def close(self):
        self._reset()

    # Loading data from previous training runs
    def _parse_config_for_previous_run(self, run_dir):
        with open(os.path.join(run_dir, "submit_config.pkl"), "rb") as f:
            data = pickle.load(f)
        data = data.get("run_func_kwargs", {})
        return dict(train = data, dataset = data.get("dataset_args", {}))

    def _reset(self, network_pkl = None, run_dir = None, data_dir = None, dataset_args = None,
            mirror_augment = None):
        self._dataset_args = dataset_args.copy() if dataset_args is not None else None

        if self._dataset_obj is not None:
            self._dataset_obj.close()
        self._dataset_obj = None

        self._network_pkl = network_pkl
        self._data_dir = data_dir

        self._mirror_augment = mirror_augment
        self._eval_time = 0
        self._results = []

        if run_dir is not None and (dataset_args is None or mirror_augment is None):
            run_config = self._parse_config_for_previous_run(run_dir)
            self._dataset_args = dict(run_config["dataset"])
            self._mirror_augment = run_config["train"].get("mirror_augment", False)

        # When training is performed over a subset of the data, we still want to evaluate it
        # over a new unrestricted sample of the full dataset.
        if self._dataset_args is not None:
            self._dataset_args["max_imgs"] = None
            # self._dataset_args["shuffle_mb"] = 0

    def configure_progress_reports(self, plo, phi, pmax, psec = 15):
        self._progress_lo = plo
        self._progress_hi = phi
        self._progress_max = pmax
        self._progress_sec = psec

    def run(self, network_pkl, num_imgs, run_dir = None, data_dir = None, dataset_args = None,
            mirror_augment = None, num_gpus = 1, tf_config = None, log_results = True,
            Gs_kwargs = dict(is_validation = True), eval_mod = False, **kwargs):

        self._reset(network_pkl = network_pkl, run_dir = run_dir, data_dir = data_dir,
            dataset_args = dataset_args, mirror_augment = mirror_augment)
        self.eval_mod = eval_mod

        time_begin = time.time()
        with tf.Graph().as_default(), tflib.create_session(tf_config).as_default():
            self._report_progress(0, 1)
            _G = _D = Gs = None
            if self._network_pkl is not None:
                _G, _D, Gs = misc.load_pkl(self._network_pkl)[:3]
            self._evaluate(Gs, Gs_kwargs = Gs_kwargs, num_gpus = num_gpus,
                num_imgs = num_imgs, **kwargs)
            self._report_progress(1, 1)
        self._eval_time = time.time() - time_begin

        if log_results:
            if run_dir is not None:
                log_file = os.path.join(run_dir, "metric-%s.txt" % self.name)
                with dnnlib.util.Logger(log_file, "a", screen = False):
                    print(self.get_result_str().strip())
            print(self.get_result_str(screen = True).strip())

        return self._results[0].value

    def get_result_str(self, screen = False):
        if self._network_pkl is None:
            network_name = "None"
        else:
            network_name = os.path.splitext(os.path.basename(self._network_pkl))[0]
        if len(network_name) > 29:
            network_name = "..." + network_name[-26:]

        result_str = "%-30s" % network_name
        result_str += " time %-12s" % dnnlib.util.format_time(self._eval_time)
        nums = ""
        for res in self._results:
            nums += " " + self.name + res.suffix + " "
            nums += res.fmt % res.value
        if screen:
            nums = misc.bcolored(nums, "blue")
        result_str += nums

        return result_str

    def update_autosummaries(self):
        for res in self._results:
            name = self.name
            tflib.autosummary.autosummary("Metrics/" + name + res.suffix, res.value)

    def _evaluate(self, Gs, Gs_kwargs, num_gpus, num_imgs, paths = None):
        raise NotImplementedError # to be overridden by subclasses

    def _report_result(self, value, suffix = "", fmt = "%-10.4f"):
        self._results += [dnnlib.EasyDict(value = value, suffix = suffix, fmt = fmt)]

    def _report_progress(self, pcur, pmax, status_str = ""):
        if self._progress_lo is None or self._progress_hi is None or self._progress_max is None:
            return
        t = time.time()
        if self._progress_sec is not None and self._progress_time is not None and \
                t < self._progress_time + self._progress_sec:
            return
        self._progress_time = t
        val = self._progress_lo + (pcur / pmax) * (self._progress_hi - self._progress_lo)
        dnnlib.RunContext.get().update(status_str, int(val), self._progress_max)

    def _get_cache_file_for_reals(self, num_imgs, extension = "pkl", **kwargs):
        all_args = dnnlib.EasyDict(metric_name = self.name, mirror_augment = self._mirror_augment)
        all_args.update(self._dataset_args)
        all_args.update(kwargs)
        md5 = hashlib.md5(repr(sorted(all_args.items())).encode("utf-8"))
        dataset_name = self._dataset_args.get("tfrecord_dir", None) or self._dataset_args.get("h5_file", None)
        dataset_name = os.path.splitext(os.path.basename(dataset_name))[0]
        return os.path.join(".GANsformer-cache", "%s-%s-%s-%s.%s" % (md5.hexdigest(), 
            self.name, dataset_name, num_imgs, extension))

    def _get_dataset_obj(self):
        if self._dataset_obj is None:
            self._dataset_obj = dataset.load_dataset(data_dir = self._data_dir, **self._dataset_args)
        return self._dataset_obj

    def _iterate_files(self, paths, minibatch_size):
        for idx in range(0, len(paths), minibatch_size):
            load_img = lambda img: np.asarray(PIL.Image.open(img).convert("RGB")).transpose(2, 0, 1)
            imgs = [load_img(img) for img in paths[idx : idx + minibatch_size]]
            imgs = np.stack(imgs, axis = 0)
            yield imgs

    def _iterate_reals(self, minibatch_size):
        dataset_obj = self._get_dataset_obj()
        while True:
            imgs, _labels = dataset_obj.get_minibatch_np(minibatch_size)
            if self._mirror_augment:
                imgs = misc.apply_mirror_augment(imgs)
            yield imgs

    def _iterate_fakes(self, Gs, minibatch_size, num_gpus):
        while True:
            latents = np.random.randn(minibatch_size, *Gs.input_shape[1:])
            fmt = dict(func = tflib.convert_imgs_to_uint8, nchw_to_nhwc = True)
            imgs = Gs.run(latents, None, output_transform = fmt, is_validation = True, 
                num_gpus = num_gpus, assume_frozen = True)[0]
            yield imgs

    def _get_random_labels_tf(self, minibatch_size):
        return self._get_dataset_obj().get_random_labels_tf(minibatch_size)

    def _get_random_imgs_tf(self):
        return self._get_dataset_obj().get_minibatch_tf()[0]

    # Iterate over images form the dataset and extract their features.
    # Args:
    #   img_iter: an iterator over image batches
    #   featurizer: a feature extractor featurizer (e.g. inception/vgg), that receives an image batch
    #          and returns their vector feature embeddings
    #   minibatch_size: size of batches provides by the image iterator
    #   num_imgs: number of extracted images
    # Returns the features [num_imgs, featurizer.output_shape[1]]
    def _get_feats(self, img_iter, featurizer, minibatch_size, num_gpus, num_imgs):
        feats = np.empty([num_imgs, featurizer.output_shape[1]], dtype = np.float32)
        itr = enumerate(img_iter)
        if self.eval_mod:
            itr = tqdm(itr, total = num_imgs / minibatch_size, unit_scale = minibatch_size)
        for idx, imgs in itr:
            begin = idx * minibatch_size
            end = min(begin + minibatch_size, num_imgs)

            feats[begin:end] = featurizer.run(imgs[:end-begin], num_gpus = num_gpus, assume_frozen = True)

            if end == num_imgs:
                break

        return feats

    # Generate images and extract their features using a generator model.
    # Args:
    #   generator: a model for generating images
    #   featurizer: a feature extractor model (e.g. inception/vgg), that receives an image batch
    #          and returns their vector feature embeddings
    #   minibatch_size: size of batches provides by the image iterator
    #   num_imgs: number of extracted iamges
    #   num_gpus: number of GPUs to use for generating the images
    #   g_kwargs: generator arguments
    # Returns the features [num_imgs, featurizer.output_shape[1]]
    def _gen_feats(self, generator, featurizer, minibatch_size, num_imgs, num_gpus, g_kwargs):
        # Construct TensorFlow graph
        result_expr = []
        for gpu_idx in range(num_gpus):
            with tf.device("/gpu:%d" % gpu_idx):
                latents = tf.random_normal([self.minibatch_per_gpu] + generator.input_shape[1:])
                labels = self._get_random_labels_tf(self.minibatch_per_gpu)

                imgs = generator.clone().get_output_for(latents, labels, **g_kwargs)[0]
                imgs = tflib.convert_imgs_to_uint8(imgs)

                result_expr.append(featurizer.clone().get_output_for(imgs))

        # Compute features for newly generated 'num_imgs' images
        itr = range(0, num_imgs, minibatch_size)
        if self.eval_mod:
            itr = tqdm(itr, total = num_imgs / minibatch_size, unit_scale = minibatch_size)

        feats = np.empty([num_imgs, featurizer.output_shape[1]], dtype = np.float32)
        for begin in itr:
            self._report_progress(begin, num_imgs)
            end = min(begin + minibatch_size, num_imgs)
            feats[begin:end] = np.concatenate(tflib.run(result_expr), axis = 0)[:end-begin]
        return feats

    # Iterate over image files and extract their features.
    # Args:
    #   paths: a list of image patch to extract features for.
    #   featurizer: a feature extractor featurizer (e.g. inception/vgg), that receives an image batch
    #          and returns their vector feature embeddings
    #   minibatch_size: size of batches provides by the image iterator
    #   num_imgs: number of extracted images
    # Returns the features [num_imgs, featurizer.output_shape[1]]
    def _paths_to_feats(self, paths, featurizer, minibatch_size, num_gpus, num_imgs = None):
        paths = glob.glob(paths)
        if num_imgs is not None:
            paths = paths[:num_imgs]
        num_imgs = len(paths)

        print("Evaluting FID on {} imgs.".format(num_imgs))

        imgs = self._iterate_files(paths, minibatch_size)
        feats = self._get_feats(imgs, featurizer, num_gpus, num_imgs, minibatch_size)
        return feats

# Group of multiple metrics
class MetricGroup:
    def __init__(self, metric_kwarg_list):
        self.metrics = [dnnlib.util.call_func_by_name(**kwargs) for kwargs in metric_kwarg_list]

    def run(self, *args, **kwargs):
        ret = 0.0
        for metric in self.metrics:
            ret = metric.run(*args, **kwargs)
        return ret

    def get_result_str(self):
        return " ".join(metric.get_result_str() for metric in self.metrics)

    def update_autosummaries(self):
        for metric in self.metrics:
            metric.update_autosummaries()

# Dummy metric for debugging purposes
class DummyMetric(MetricBase):
    def _evaluate(self, Gs, Gs_kwargs, num_gpus, num_imgs, paths = None):
        self._report_result(0.0)
