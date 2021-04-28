# Input data pipeline. Supports multi-resolution but unused in practice.
import re
import os
import glob
import numpy as np
import tensorflow as tf
import dnnlib
import dnnlib.tflib as tflib
from training import misc
import random

# Dataset class that loads data from tfrecords files
# It supports labels as optional
class TFRecordDataset:
    def __init__(self,
        tfrecord_dir,               # Directory containing a collection of tfrecords files
        resolution      = None,     # Dataset resolution, None = autodetect
        label_file      = None,     # Relative path of the labels file, None = autodetect
        max_label_size  = 0,        # 0 = no labels, "full" = full labels, <int> = N first label components
        max_imgs        = None,     # Maximum number of images to use, None = use all images
        repeat          = True,     # Repeat dataset indefinitely?
        shuffle_mb      = 2048,     # Shuffle data within specified window (megabytes), 0 = disable shuffling
        prefetch_mb     = 512,      # Amount of data to prefetch (megabytes), 0 = disable prefetching
        buffer_mb       = 256,      # Read buffer size (megabytes)
        num_threads     = 4,        # Number of concurrent threads for input processing
        **kwargs):       

        self.tfrecord_dir       = tfrecord_dir
        self.resolution         = None
        self.resolution_log2    = None
        self.shape              = [] # [channels, height, width]
        self.dtype              = "uint8"
        self.dynamic_range      = [0, 255]
        self.label_file         = label_file
        self.label_size         = None
        self.label_dtype        = None
        self._np_labels         = None
        self._tf_minibatch_in   = None
        self._tf_labels_var     = None
        self._tf_labels_dataset = None
        self._tf_datasets       = dict()
        self._tf_iterator       = None
        self._tf_init_ops       = dict()
        self._tf_minibatch_np   = None
        self._cur_minibatch     = -1
        self._cur_lod           = -1

        # List tfrecords files and inspect their shapes
        assert os.path.isdir(self.tfrecord_dir)
        tfr_files = sorted(glob.glob(os.path.join(self.tfrecord_dir, "*.tfrecords1of*")))
        # If max_imgs is not None, take a subset of images out of the 1st file. Otherwise take all files.
        if max_imgs is None:
            tfr_files = [sorted(glob.glob(re.sub("1of.*", "*", f))) for f in tfr_files]
        else:
            tfr_files = [[f] for f in tfr_files]

        assert len(tfr_files) >= 1
        tfr_shapes = []
        for tfr_file in tfr_files:
            tfr_opt = tf.io.TFRecordOptions("")
            for record in tf.python_io.tf_record_iterator(tfr_file[0], tfr_opt):
                tfr_shapes.append(self.parse_tfrecord_np(record).shape)
                break
            random.shuffle(tfr_file)

        # Autodetect label filename
        if self.label_file is None:
            guess = sorted(glob.glob(os.path.join(self.tfrecord_dir, "*.labels")))
            if len(guess):
                self.label_file = guess[0]
        elif not os.path.isfile(self.label_file):
            guess = os.path.join(self.tfrecord_dir, self.label_file)
            if os.path.isfile(guess):
                self.label_file = guess

        # Determine shape and resolution
        max_shape = max(tfr_shapes, key = np.prod)
        self.resolution = resolution if resolution is not None else max_shape[1]
        if resolution is not None:
            assert resolution <= max_shape[1]
        self.resolution_log2 = int(np.log2(self.resolution))
        file_indexes = [i for i, tfr_shape in enumerate(tfr_shapes) if tfr_shape[1] == self.resolution]
        tfr_files = [tfr_files[i] for i in file_indexes]
        tfr_shapes = [tfr_shapes[i] for i in file_indexes]
        self.shape = [max_shape[0], self.resolution, self.resolution]
        tfr_lods = [self.resolution_log2 - int(np.log2(shape[1])) for shape in tfr_shapes]
        assert all(shape[0] == max_shape[0] for shape in tfr_shapes)
        assert all(shape[1] == shape[2] for shape in tfr_shapes)
        assert all(shape[1] == self.resolution // (2**lod) for shape, lod in zip(tfr_shapes, tfr_lods))
        #assert all(lod in range(self.resolution_log2 - 1) for lod in tfr_lods)

        # Load labels
        assert max_label_size == "full" or max_label_size >= 0
        self._np_labels = np.zeros([1<<20, 0], dtype = np.float32)
        if self.label_file is not None and max_label_size != 0:
            self._np_labels = np.load(self.label_file)
            assert self._np_labels.ndim == 2
        if max_label_size != "full" and self._np_labels.shape[1] > max_label_size:
            self._np_labels = self._np_labels[:, :max_label_size]
        if max_imgs is not None and self._np_labels.shape[0] > max_imgs:
            self._np_labels = self._np_labels[:max_imgs]
        if max_imgs is not None and self._np_labels.shape[0] < max_imgs:
            print(misc.bcolored("Too many images. increase number.", "red"))
            exit()
        self.label_size = self._np_labels.shape[1]
        self.label_dtype = self._np_labels.dtype.name

        # Build TF expressions
        with tf.name_scope("Dataset"), tf.device("/cpu:0"):
            self._tf_minibatch_in = tf.placeholder(tf.int64, name = "minibatch_in", shape = [])
            self._tf_labels_var = tflib.create_var_with_large_initial_value(self._np_labels, name = "labels_var")
            self._tf_labels_dataset = tf.data.Dataset.from_tensor_slices(self._tf_labels_var)
            for tfr_file, tfr_shape, tfr_lod in zip(tfr_files, tfr_shapes, tfr_lods):
                if tfr_lod < 0:
                    continue
                # Load dataset
                dset = tf.data.TFRecordDataset(tfr_file, compression_type = "", 
                    buffer_size = buffer_mb<<20, num_parallel_reads = num_threads)

                # If max_imgs is set, take a subset of the data
                if max_imgs is not None:
                    dset = dset.take(max_imgs)

                # Parse the TF records
                dset = dset.map(self.parse_tfrecord_tf, num_parallel_calls = num_threads)

                # Zip images with their labels (0s if no labels)
                dset = tf.data.Dataset.zip((dset, self._tf_labels_dataset))

                # Shuffle and repeat
                bytes_per_item = np.prod(tfr_shape) * np.dtype(self.dtype).itemsize
                if shuffle_mb > 0:
                    dset = dset.shuffle(((shuffle_mb << 20) - 1) // bytes_per_item + 1)
                if repeat:
                    dset = dset.repeat()

                # Prefetch and batch
                if prefetch_mb > 0:
                    dset = dset.prefetch(((prefetch_mb << 20) - 1) // bytes_per_item + 1)
                dset = dset.batch(self._tf_minibatch_in)
                self._tf_datasets[tfr_lod] = dset

            # Initialize data iterator
            self._tf_iterator = tf.data.Iterator.from_structure(self._tf_datasets[0].output_types, 
                self._tf_datasets[0].output_shapes)
            self._tf_init_ops = {lod: self._tf_iterator.make_initializer(dset) \
                for lod, dset in self._tf_datasets.items()}

    def close(self):
        pass

    # Use the given minibatch size and level-of-detail for the data returned by get_minibatch_tf()
    def configure(self, minibatch_size, lod = 0):
        lod = int(np.floor(lod))
        assert minibatch_size >= 1 and lod in self._tf_datasets
        if self._cur_minibatch != minibatch_size or self._cur_lod != lod:
            self._tf_init_ops[lod].run({self._tf_minibatch_in: minibatch_size})
            self._cur_minibatch = minibatch_size
            self._cur_lod = lod

    # Get next minibatch as TensorFlow expressions
    def get_minibatch_tf(self): # => images, labels
        return self._tf_iterator.get_next()

    # Get next minibatch as NumPy arrays
    def get_minibatch_np(self, minibatch_size, lod = 0): # => images, labels
        self.configure(minibatch_size, lod)
        with tf.name_scope("Dataset"):
            if self._tf_minibatch_np is None:
                self._tf_minibatch_np = self.get_minibatch_tf()
            return tflib.run(self._tf_minibatch_np)

    # Get random labels as TensorFlow expression
    def get_random_labels_tf(self, minibatch_size): # => labels
        with tf.name_scope("Dataset"):
            if self.label_size > 0:
                with tf.device("/cpu:0"):
                    return tf.gather(self._tf_labels_var, tf.random_uniform([minibatch_size], 
                        0, self._np_labels.shape[0], dtype = tf.int32))
            return tf.zeros([minibatch_size, 0], self.label_dtype)

    # Get random labels as NumPy array
    def get_random_labels_np(self, minibatch_size): # => labels
        if self.label_size > 0:
            return self._np_labels[np.random.randint(self._np_labels.shape[0], size=[minibatch_size])]
        return np.zeros([minibatch_size, 0], self.label_dtype)

    # Parse individual image from a tfrecords file into TensorFlow expression
    @staticmethod
    def parse_tfrecord_tf(record):
        features = tf.parse_single_example(record, features={
            "shape": tf.FixedLenFeature([3], tf.int64),
            "data": tf.FixedLenFeature([], tf.string)})
        data = tf.decode_raw(features["data"], tf.uint8)
        return tf.reshape(data, features["shape"])

    @staticmethod
    def parse_tfrecord_tf_seg(record):
        features = tf.parse_single_example(record, features={
            "shape": tf.FixedLenFeature([3], tf.int64),
            "seg": tf.FixedLenFeature([], tf.string),
            "data": tf.FixedLenFeature([], tf.string)})
        data = tf.decode_raw(features["data"], tf.uint8)
        seg = tf.decode_raw(features["seg"], tf.uint8)
        return tf.reshape(data, features["shape"]), tf.reshape(seg, tf.concat([[2], features["shape"][1:]], axis = 0))

    # Parse an individual image from a tfrecords file into a NumPy array
    @staticmethod
    def parse_tfrecord_np(record):
        ex = tf.train.Example()
        ex.ParseFromString(record)
        shape = ex.features.feature["shape"].int64_list.value
        data = ex.features.feature["data"].bytes_list.value[0]
        return np.fromstring(data, np.uint8).reshape(shape)

# A helper function for constructing a dataset object using the given options
def load_dataset(class_name = None, data_dir = None, verbose = False, **kwargs):
    kwargs = dict(kwargs)
    if "tfrecord_dir" in kwargs:
        if class_name is None:
            class_name = __name__ + ".TFRecordDataset"
        if data_dir is not None:
            kwargs["tfrecord_dir"] = os.path.join(data_dir, kwargs["tfrecord_dir"])

    assert class_name is not None
    if verbose:
        print(misc.bcolored("Streaming data using %s %s..." % (class_name, data_dir), "white"))
    dataset = dnnlib.util.get_obj_by_name(class_name)(**kwargs)
    if verbose:
        print("Dataset shape: ", misc.bcolored(np.int32(dataset.shape).tolist(), "blue"))
        print("Dynamic range: ", misc.bcolored(dataset.dynamic_range, "blue"))
        if dataset.label_size > 0:
            print("Label size: ", misc.bcolored(dataset.label_size, "blue"))
    return dataset
