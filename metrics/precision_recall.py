# Precision/Recall (PR)
import os
import numpy as np
import tensorflow as tf
import dnnlib
import dnnlib.tflib as tflib
import glob
import PIL.Image

from metrics import metric_base
from training import misc

def batch_pairwise_distances(U, V):
    # Compute pairwise distances between two batches of feature vectors
    with tf.variable_scope("pairwise_dist_block"):
        # Squared norms of each row in U and V
        norm_u = tf.reduce_sum(tf.square(U), 1)
        norm_v = tf.reduce_sum(tf.square(V), 1)

        # norm_u as a row and norm_v as a column vectors
        norm_u = tf.reshape(norm_u, [-1, 1])
        norm_v = tf.reshape(norm_v, [1, -1])

        # Pairwise squared Euclidean distances
        D = tf.maximum(norm_u - 2*tf.matmul(U, V, False, True) + norm_v, 0.0)

    return D

class DistanceBlock():
    # Distance block
    def __init__(self, num_features, num_gpus):
        self.num_features = num_features
        self.num_gpus = num_gpus

        # Initialize TF graph to compute pairwise distances
        with tf.device("/cpu:0"):
            self._features_batch1 = tf.placeholder(tf.float16, shape=[None, self.num_features])
            self._features_batch2 = tf.placeholder(tf.float16, shape=[None, self.num_features])
            features_split2 = tf.split(self._features_batch2, self.num_gpus, axis = 0)
            distances_split = []
            for gpu_idx in range(self.num_gpus):
                with tf.device("/gpu:%d" % gpu_idx):
                    distances_split.append(batch_pairwise_distances(self._features_batch1, features_split2[gpu_idx]))
            self._distance_block = tf.concat(distances_split, axis = 1)

    def pairwise_distances(self, U, V):
        # Evaluate pairwise distances between two batches of feature vectors
        return self._distance_block.eval(feed_dict = {self._features_batch1: U, self._features_batch2: V})

class ManifoldEstimator():
    # Finds an estimate for the manifold of given feature vectors
    def __init__(self, distance_block, features, row_batch_size, col_batch_size, nhood_sizes, clamp_to_percentile = None):
        # Find an estimate of the manifold of given feature vectors
        num_imgs = features.shape[0]
        self.nhood_sizes = nhood_sizes
        self.num_nhoods = len(nhood_sizes)
        self.row_batch_size = row_batch_size
        self.col_batch_size = col_batch_size
        self._ref_features = features
        self._distance_block = distance_block

        # Estimate manifold of features by calculating distances to kth nearest neighbor of each sample
        self.D = np.zeros([num_imgs, self.num_nhoods], dtype = np.float16)
        distance_batch = np.zeros([row_batch_size, num_imgs], dtype = np.float16)
        seq = np.arange(max(self.nhood_sizes) + 1, dtype = np.int32)

        for begin1 in range(0, num_imgs, row_batch_size):
            end1 = min(begin1 + row_batch_size, num_imgs)
            row_batch = features[begin1:end1]

            for begin2 in range(0, num_imgs, col_batch_size):
                end2 = min(begin2 + col_batch_size, num_imgs)
                col_batch = features[begin2:end2]

                # Compute distances between batches
                distance_batch[0:end1-begin1, begin2:end2] = self._distance_block.pairwise_distances(row_batch, col_batch)

            # Find the kth nearest neighbor from the current batch
            self.D[begin1:end1, :] = np.partition(distance_batch[0:end1-begin1, :], seq, axis = 1)[:, self.nhood_sizes]

        if clamp_to_percentile is not None:
            max_distances = np.percentile(self.D, clamp_to_percentile, axis = 0)
            self.D[self.D > max_distances] = 0

    def evaluate(self, eval_features, return_realism = False, return_neighbors = False):
        # Evaluate if new feature vectors are in the estimated manifold
        num_eval_imgs = eval_features.shape[0]
        num_ref_imgs = self.D.shape[0]
        distance_batch = np.zeros([self.row_batch_size, num_ref_imgs], dtype = np.float16)
        batch_predictions = np.zeros([num_eval_imgs, self.num_nhoods], dtype = np.int32)
        realism_score = np.zeros([num_eval_imgs,], dtype = np.float32)
        nearest_indices = np.zeros([num_eval_imgs,], dtype = np.int32)

        for begin1 in range(0, num_eval_imgs, self.row_batch_size):
            end1 = min(begin1 + self.row_batch_size, num_eval_imgs)
            feature_batch = eval_features[begin1:end1]

            for begin2 in range(0, num_ref_imgs, self.col_batch_size):
                end2 = min(begin2 + self.col_batch_size, num_ref_imgs)
                ref_batch = self._ref_features[begin2:end2]

                distance_batch[0:end1-begin1, begin2:end2] = self._distance_block.pairwise_distances(feature_batch, ref_batch)

            # From the minibatch of new feature vectors, determine if they are in the estimated manifold.
            # If a feature vector is inside a hypersphere of some reference sample, then the new sample lies on the estimated manifold.
            # The radii of the hyperspheres are determined from distances of neighborhood size k.
            samples_in_manifold = distance_batch[0:end1-begin1, :, None] <= self.D
            batch_predictions[begin1:end1] = np.any(samples_in_manifold, axis = 1).astype(np.int32)

            nearest_indices[begin1:end1] = np.argmin(distance_batch[0:end1-begin1, :], axis = 1)
            realism_score[begin1:end1] = self.D[nearest_indices[begin1:end1], 0] / np.min(distance_batch[0:end1-begin1, :], axis = 1)

        if return_realism and return_neighbors:
            return batch_predictions, realism_score, nearest_indices
        elif return_realism:
            return batch_predictions, realism_score
        elif return_neighbors:
            return batch_predictions, nearest_indices

        return batch_predictions

def knn_precision_recall_features(ref_features, eval_features, feature_net, nhood_sizes,
                                  row_batch_size, col_batch_size, num_imgs, num_gpus):
    # Computes k-NN precision and recall for two sets of feature vectors
    state = dnnlib.EasyDict()
    num_features = feature_net.output_shape[1]
    state.ref_features = ref_features
    state.eval_features = eval_features

    # Initialize DistanceBlock and ManifoldEstimators.
    distance_block = DistanceBlock(num_features, num_gpus)
    state.ref_manifold = ManifoldEstimator(distance_block, state.ref_features, row_batch_size, col_batch_size, nhood_sizes)
    state.eval_manifold = ManifoldEstimator(distance_block, state.eval_features, row_batch_size, col_batch_size, nhood_sizes)

    # Evaluate precision and recall using k-nearest neighbors
    print("Evaluating k-NN precision and recall with %i samples..." % num_imgs)
    start = time.time()

    # Precision: How many points from eval_features are in ref_features manifold
    state.precision, state.realism_scores, state.nearest_neighbors = state.ref_manifold.evaluate(state.eval_features, return_realism = True, return_neighbors = True)
    state.knn_precision = state.precision.mean(axis = 0)

    # Recall: How many points from ref_features are in eval_features manifold
    state.recall = state.eval_manifold.evaluate(state.ref_features)
    state.knn_recall = state.recall.mean(axis = 0)

    elapsed_time = time.time() - start
    print("Done evaluation in: %gs" % elapsed_time)
    return state

class PR(metric_base.MetricBase):
    def __init__(self, nhood_size, minibatch_per_gpu, row_batch_size, col_batch_size, **kwargs):
        super().__init__(**kwargs)
        self.nhood_size = nhood_size
        self.minibatch_per_gpu = minibatch_per_gpu
        self.row_batch_size = row_batch_size
        self.col_batch_size = col_batch_size

    def _evaluate(self, Gs, Gs_kwargs, num_gpus, num_imgs, paths = None, **kwargs):
        minibatch_size = num_gpus * self.minibatch_per_gpu
        feature_net = misc.load_pkl("http://d36zk2xti64re0.cloudfront.net/stylegan1/networks/metrics/vgg16.pkl")

        # Compute features for reals
        cache_file = self._get_cache_file_for_reals(num_imgs)
        os.makedirs(os.path.dirname(cache_file), exist_ok = True)
        if os.path.isfile(cache_file):
            ref_features = misc.load_pkl(cache_file)
        else:
            imgs_iter = self._iterate_reals(minibatch_size = minibatch_size)
            ref_features = self._get_feats(imgs_iter, feature_net, minibatch_size, num_gpus, num_imgs)

            misc.save_pkl(ref_features, cache_file)

        if paths is not None:
            # Extract features for local sample image files (paths)
            eval_features = self._paths_to_feats(paths, feature_net, minibatch_size, num_gpus, num_imgs)
        else:
            # Extract features for newly generated fake imgs
            eval_features = self._gen_feats(Gs, feature_net, minibatch_size, num_imgs, num_gpus, Gs_kwargs)

        # Compute precision and recall
        state = knn_precision_recall_features(ref_features = ref_features, eval_features = eval_features,
            feature_net = feature_net, nhood_sizes = [self.nhood_size], row_batch_size = self.row_batch_size,
            col_batch_size = self.row_batch_size, num_imgs = num_imgs, num_gpus = num_gpus)
        self._report_result(state.knn_precision[0], suffix = "_precision")
        self._report_result(state.knn_recall[0], suffix = "_recall")
