# Tool for creating image datasets.
import os
import sys
import glob
import argparse
import threading
import six.moves.queue as Queue
import traceback
import numpy as np
import tensorflow as tf
import PIL.Image
import cv2
import io
from tqdm import tqdm, trange
from training import misc 
import re

def error(msg):
    print("Error: " + msg)
    exit(1)

class DatasetExporter:
    def __init__(self, dataset_dir, expected_imgs, verbose = False, progress_interval = 10):
        self.dataset_dir        = dataset_dir
        self.expected_imgs      = expected_imgs
        self.curr_imgnum        = 0
        self.shape              = None
        self.resolution_log2    = None
        self.verbose            = verbose
        self.progress_interval  = progress_interval
        self.writer_index       = 0
        self.initialized        = False

        if self.verbose:
            print("Creating dataset %s" % dataset_dir)
        os.makedirs(self.dataset_dir, exist_ok = True)
        assert os.path.isdir(self.dataset_dir)

    def close(self):
        if self.verbose:
            print("Added %d images." % self.curr_imgnum)

    def choose_shuffled_order(self): # Note: Images and labels must be added in shuffled order
        order = np.arange(self.expected_imgs)
        np.random.RandomState(123).shuffle(order)
        return order

    def add_img(self, img):
        if self.verbose and self.curr_imgnum % self.progress_interval == 0:
            print("%d / %d\r" % (self.curr_imgnum, self.expected_imgs), end = "", flush = True)

        if not self.initialized:
            self.shape = img.shape
            self.resolution_log2 = int(np.log2(self.shape[1]))
            assert self.shape[0] in [1, 2, 3]
            assert self.shape[1] == self.shape[2]
            assert self.shape[1] == 2**self.resolution_log2
            self.initialized = True

        assert img.shape == self.shape

        for lod in range(self.resolution_log2 - 1):
            resolution = 2 ** (self.resolution_log2 - lod)
            os.makedirs(f"{self.dataset_dir}/{resolution}", exist_ok = True)
            if lod:
                img = img.astype(np.float32)
                img = (img[:, 0::2, 0::2] + img[:, 0::2, 1::2] + img[:, 1::2, 0::2] + img[:, 1::2, 1::2]) * 0.25
            out_img = np.rint(img).clip(0, 255).astype(np.uint8)
            out_img = PIL.Image.fromarray(out_img.transpose(1, 2, 0))
            img_name = f"{self.dataset_dir}/{resolution}/{self.curr_imgnum}.png"
            out_img.save(img_name, format = "png")

        self.curr_imgnum += 1

    def add_labels(self, labels):
        if self.verbose:
            print("%-40s\r" % "Saving labels...", end = "", flush = True)
        assert labels.shape[0] == self.curr_imgnum
        with open(f"{self.dataset_dir}/labels.npy", "wb") as f:
            np.save(f, labels.astype(np.float32))

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

class ExceptionInfo(object):
    def __init__(self):
        self.value = sys.exc_info()[1]
        self.traceback = traceback.format_exc()

# Thread support. Used only for Celeb-HQ since its images require heavy preprocessing
class WorkerThread(threading.Thread):
    def __init__(self, task_queue):
        threading.Thread.__init__(self)
        self.task_queue = task_queue

    def run(self):
        while True:
            func, args, result_queue = self.task_queue.get()
            if func is None:
                break
            try:
                result = func(*args)
            except:
                result = ExceptionInfo()
            result_queue.put((result, args))

class ThreadPool(object):
    def __init__(self, num_threads):
        assert num_threads >= 1
        self.task_queue = Queue.Queue()
        self.result_queues = dict()
        self.num_threads = num_threads
        for _idx in range(self.num_threads):
            thread = WorkerThread(self.task_queue)
            thread.daemon = True
            thread.start()

    def add_task(self, func, args = ()):
        assert hasattr(func, "__call__") # must be a function
        if func not in self.result_queues:
            self.result_queues[func] = Queue.Queue()
        self.task_queue.put((func, args, self.result_queues[func]))

    def get_result(self, func):
        result, args = self.result_queues[func].get()
        if isinstance(result, ExceptionInfo):
            print("\n\nWorker thread caught an exception:\n" + result.traceback)
            raise result.value
        return result, args

    def finish(self):
        for _idx in range(self.num_threads):
            self.task_queue.put((None, (), None))

    def __enter__(self):
        return self

    def __exit__(self, *excinfo):
        self.finish()

    def process_items_concurrently(self, item_iterator, process_func = lambda x: x, pre_func = lambda x: x, 
            post_func = lambda x: x, max_items_in_flight = None):
        if max_items_in_flight is None: max_items_in_flight = self.num_threads * 4
        assert max_items_in_flight >= 1
        results = []
        retire_idx = [0]

        def task_func(prepared, _idx):
            return process_func(prepared)

        def retire_result():
            processed, (_prepared, idx) = self.get_result(task_func)
            results[idx] = processed
            while retire_idx[0] < len(results) and results[retire_idx[0]] is not None:
                yield post_func(results[retire_idx[0]])
                results[retire_idx[0]] = None
                retire_idx[0] += 1

        for idx, item in enumerate(item_iterator):
            prepared = pre_func(item)
            results.append(None)
            self.add_task(func = task_func, args = (prepared, idx))
            while retire_idx[0] < idx - max_items_in_flight + 2:
                for res in retire_result(): yield res
        while retire_idx[0] < len(results):
            for res in retire_result(): yield res

# ----------------------------------------------------------------------------

def display(dataset_dir):
    print("Loading dataset %s" % dataset_dir)
    tflib.init_tf({"gpu_options.allow_growth": True})
    dset = dataset.TFRecordDataset(dataset_dir, max_label_size = "full", repeat = False, shuffle_mb = 0)
    tflib.init_uninitialized_vars()

    idx = 0
    while True:
        try:
            imgs, labels = dset.get_minibatch_np(1)
        except tf.errors.OutOfRangeError:
            break
        if idx == 0:
            print("Displaying images")
            cv2.namedWindow("dataset_tool")
            print("Press SPACE or ENTER to advance, ESC to exit")
        print("\nidx = %-8d\nlabel = %s" % (i, labels[0].tolist()))
        img = imgs[0].transpose(1, 2, 0)[:, :, ::-1] # CHW => HWC, RGB => BGR
        cv2.imshow("dataset_tool", )
        idx += 1
        if cv2.waitKey() == 27:
            break
    print("\nDisplayed %d images." % idx)

def extract(dataset_dir, output_dir):
    print("Loading dataset %s" % dataset_dir)
    tflib.init_tf({"gpu_options.allow_growth": True})
    dset = dataset.TFRecordDataset(dataset_dir, max_label_size = 0, repeat = False, shuffle_mb = 0)
    tflib.init_uninitialized_vars()

    print("Extracting images to %s" % output_dir)
    os.makedirs(output_dir, exist_ok = True)
    idx = 0
    while True:
        if idx % 10 == 0:
            print("%d\r" % idx, end = "", flush = True)
        try:
            imgs, _labels = dset.get_minibatch_np(1)
        except tf.errors.OutOfRangeError:
            break
        if imgs.shape[1] == 1:
            img = PIL.Image.fromarray(imgs[0][0], "L")
        else:
            img = PIL.Image.fromarray(imgs[0].transpose(1, 2, 0), "RGB")
        img.save(os.path.join(output_dir, "image%08d.png" % idx))
        idx += 1
    print("Extracted %d images." % idx)

def compare(dataset_dir_a, dataset_dir_b, ignore_labels):
    max_label_size = 0 if ignore_labels else "full"
    print("Loading dataset %s" % dataset_dir_a)
    tflib.init_tf({"gpu_options.allow_growth": True})
    dset_a = dataset.TFRecordDataset(dataset_dir_a, max_label_size = max_label_size, repeat = False, shuffle_mb = 0)
    print("Loading dataset %s" % dataset_dir_b)
    dset_b = dataset.TFRecordDataset(dataset_dir_b, max_label_size = max_label_size, repeat = False, shuffle_mb = 0)
    tflib.init_uninitialized_vars()

    print("Comparing datasets")
    idx = 0
    identical_imgs = 0
    identical_labels = 0
    while True:
        if idx % 100 == 0:
            print("%d\r" % idx, end = "", flush = True)
        try:
            imgs_a, labels_a = dset_a.get_minibatch_np(1)
        except tf.errors.OutOfRangeError:
            imgs_a, labels_a = None, None
        try:
            imgs_b, labels_b = dset_b.get_minibatch_np(1)
        except tf.errors.OutOfRangeError:
            imgs_b, labels_b = None, None
        if imgs_a is None or imgs_b is None:
            if imgs_a is not None or imgs_b is not None:
                print("Datasets contain different number of images")
            break
        if imgs_a.shape == imgs_b.shape and np.all(imgs_a == imgs_b):
            identical_imgs += 1
        else:
            print("Image %d is different" % idx)
        if labels_a.shape == labels_b.shape and np.all(labels_a == labels_b):
            identical_labels += 1
        else:
            print("Label %d is different" % idx)
        idx += 1
    print("Identical images: %d / %d" % (identical_imgs, idx))
    if not ignore_labels:
        print("Identical labels: %d / %d" % (identical_labels, idx))

# ----------------------------------------------------------------------------

def create_mnist(dataset_dir, mnist_dir):
    print("Loading MNIST from %s" % mnist_dir)
    import gzip
    with gzip.open(os.path.join(mnist_dir, "train-images-idx3-ubyte.gz"), "rb") as file:
        imgs = np.frombuffer(file.read(), np.uint8, offset = 16)
    with gzip.open(os.path.join(mnist_dir, "train-labels-idx1-ubyte.gz"), "rb") as file:
        labels = np.frombuffer(file.read(), np.uint8, offset = 8)
    imgs = imgs.reshape(-1, 1, 28, 28)
    imgs = np.pad(imgs, [(0,0), (0,0), (2,2), (2,2)], "constant", constant_values = 0)
    assert imgs.shape == (60000, 1, 32, 32) and imgs.dtype == np.uint8
    assert labels.shape == (60000,) and labels.dtype == np.uint8
    assert np.min(imgs) == 0 and np.max(imgs) == 255
    assert np.min(labels) == 0 and np.max(labels) == 9
    onehot = np.zeros((labels.size, np.max(labels) + 1), dtype = np.float32)
    onehot[np.arange(labels.size), labels] = 1.0

    with DatasetExporter(dataset_dir, imgs.shape[0]) as tfr:
        order = tfr.choose_shuffled_order()
        for idx in range(order.size):
            tfr.add_img(imgs[order[idx]])
        tfr.add_labels(onehot[order])

def create_mnistrgb(dataset_dir, mnist_dir, num_imgs = 1000000, random_seed = 123):
    print("Loading MNIST from %s" % mnist_dir)
    import gzip
    with gzip.open(os.path.join(mnist_dir, "train-images-idx3-ubyte.gz"), "rb") as file:
        imgs = np.frombuffer(file.read(), np.uint8, offset = 16)
    imgs = imgs.reshape(-1, 28, 28)
    imgs = np.pad(imgs, [(0,0), (2,2), (2,2)], "constant", constant_values = 0)
    assert imgs.shape == (60000, 32, 32) and imgs.dtype == np.uint8
    assert np.min(imgs) == 0 and np.max(imgs) == 255

    with DatasetExporter(dataset_dir, num_imgs) as tfr:
        rnd = np.random.RandomState(random_seed)
        for _idx in range(num_imgs):
            tfr.add_img(imgs[rnd.randint(imgs.shape[0], size = 3)])

# ----------------------------------------------------------------------------

def create_cifar10(dataset_dir, cifar10_dir):
    print("Loading CIFAR-10 from %s" % cifar10_dir)
    import pickle
    imgs = []
    labels = []
    for batch in range(1, 6):
        with open(os.path.join(cifar10_dir, "data_batch_%d" % batch), "rb") as file:
            data = pickle.load(file, encoding = "latin1")
        imgs.append(data["data"].reshape(-1, 3, 32, 32))
        labels.append(data["labels"])
    imgs = np.concatenate(imgs)
    labels = np.concatenate(labels)
    assert imgs.shape == (50000, 3, 32, 32) and imgs.dtype == np.uint8
    assert labels.shape == (50000,) and labels.dtype == np.int32
    assert np.min(imgs) == 0 and np.max(imgs) == 255
    assert np.min(labels) == 0 and np.max(labels) == 9
    onehot = np.zeros((labels.size, np.max(labels) + 1), dtype = np.float32)
    onehot[np.arange(labels.size), labels] = 1.0

    with DatasetExporter(dataset_dir, imgs.shape[0]) as tfr:
        order = tfr.choose_shuffled_order()
        for idx in range(order.size):
            tfr.add_img(imgs[order[idx]])
        tfr.add_labels(onehot[order])

def create_cifar100(dataset_dir, cifar100_dir):
    print("Loading CIFAR-100 from %s" % cifar100_dir)
    import pickle
    with open(os.path.join(cifar100_dir, "train"), "rb") as file:
        data = pickle.load(file, encoding = "latin1")
    imgs = data["data"].reshape(-1, 3, 32, 32)
    labels = np.array(data["fine_labels"])
    assert imgs.shape == (50000, 3, 32, 32) and imgs.dtype == np.uint8
    assert labels.shape == (50000,) and labels.dtype == np.int32
    assert np.min(imgs) == 0 and np.max(imgs) == 255
    assert np.min(labels) == 0 and np.max(labels) == 99
    onehot = np.zeros((labels.size, np.max(labels) + 1), dtype = np.float32)
    onehot[np.arange(labels.size), labels] = 1.0

    with DatasetExporter(dataset_dir, imgs.shape[0]) as tfr:
        order = tfr.choose_shuffled_order()
        for idx in range(order.size):
            tfr.add_img(imgs[order[idx]])
        tfr.add_labels(onehot[order])

# ----------------------------------------------------------------------------

def create_svhn(dataset_dir, svhn_dir):
    print("Loading SVHN from %s" % svhn_dir)
    import pickle
    imgs = []
    labels = []
    for batch in range(1, 4):
        with open(os.path.join(svhn_dir, "train_%d.pkl" % batch), "rb") as file:
            data = pickle.load(file, encoding = "latin1")
        imgs.append(data[0])
        labels.append(data[1])
    imgs = np.concatenate(imgs)
    labels = np.concatenate(labels)
    assert imgs.shape == (73257, 3, 32, 32) and imgs.dtype == np.uint8
    assert labels.shape == (73257,) and labels.dtype == np.uint8
    assert np.min(imgs) == 0 and np.max(imgs) == 255
    assert np.min(labels) == 0 and np.max(labels) == 9
    onehot = np.zeros((labels.size, np.max(labels) + 1), dtype = np.float32)
    onehot[np.arange(labels.size), labels] = 1.0

    with DatasetExporter(dataset_dir, imgs.shape[0]) as tfr:
        order = tfr.choose_shuffled_order()
        for idx in range(order.size):
            tfr.add_img(imgs[order[idx]])
        tfr.add_labels(onehot[order])

# ----------------------------------------------------------------------------

def create_lsun(dataset_dir, lmdb_dir, resolution = 256, max_imgs = None):
    print("Loading LSUN dataset from %s" % lmdb_dir)
    import lmdb # pip install lmdb
    import io
    with lmdb.open(lmdb_dir, readonly = True).begin(write = False) as txn:
        total_imgs = txn.stat()["entries"]
        if max_imgs is None:
            max_imgs = total_imgs
        with DatasetExporter(dataset_dir, max_imgs) as tfr:
            for _idx, (_key, value) in enumerate(txn.cursor()):
                try:
                    try:
                        img = cv2.imdecode(np.fromstring(value, dtype = np.uint8), 1)
                        if img is None:
                            raise IOError("cv2.imdecode failed")
                        img = img[:, :, ::-1] # BGR => RGB
                    except IOError:
                        img = np.asarray(PIL.Image.open(io.BytesIO(value)))
                    crop = np.min(img.shape[:2])
                    img = img[(img.shape[0] - crop) // 2 : (img.shape[0] + crop) // 2, 
                        (img.shape[1] - crop) // 2 : (img.shape[1] + crop) // 2]
                    img = PIL.Image.fromarray(img, "RGB")
                    img = img.resize((resolution, resolution), PIL.Image.ANTIALIAS)
                    img = np.asarray(img)
                    img = img.transpose([2, 0, 1]) # HWC => CHW
                    tfr.add_img(img)
                except:
                    print(sys.exc_info()[1])
                if tfr.curr_imgnum == max_imgs:
                    break

def create_lsun_wide(dataset_dir, lmdb_dir, width = 512, height = 384, max_imgs = None):
    assert width == 2 ** int(np.round(np.log2(width)))
    assert height <= width
    print("Loading LSUN dataset from %s" % lmdb_dir)
    import lmdb # pip install lmdb
    # import cv2 # pip install opencv-python
    import io
    with lmdb.open(lmdb_dir, readonly = True).begin(write = False) as txn:
        total_imgs = txn.stat()["entries"]
        if max_imgs is None:
            max_imgs = total_imgs
        with DatasetExporter(dataset_dir, max_imgs, verbose = False) as tfr:
            for idx, (_key, value) in enumerate(txn.cursor()):
                try:
                    try:
                        img = cv2.imdecode(np.fromstring(value, dtype = np.uint8), 1)
                        if img is None:
                            raise IOError("cv2.imdecode failed")
                        img = img[:, :, ::-1] # BGR => RGB
                    except IOError:
                        img = np.asarray(PIL.Image.open(io.BytesIO(value)))

                    ch = int(np.round(width * img.shape[0] / img.shape[1]))
                    if img.shape[1] < width or ch < height:
                        continue

                    img = img[(img.shape[0] - ch) // 2 : (img.shape[0] + ch) // 2]
                    img = PIL.Image.fromarray(img, "RGB")
                    img = img.resize((width, height), PIL.Image.ANTIALIAS)
                    img = np.asarray(img)
                    img = img.transpose([2, 0, 1]) # HWC => CHW

                    canvas = np.zeros([3, width, width], dtype = np.uint8)
                    canvas[:, (width - height) // 2 : (width + height) // 2] = img
                    tfr.add_img(canvas)
                    print("\r%d / %d => %d " % (idx + 1, total_imgs, tfr.curr_imgnum), end = "")

                except:
                    print(sys.exc_info()[1])
                if tfr.curr_imgnum == max_imgs:
                    break
    print()

# ----------------------------------------------------------------------------

def create_celeba(dataset_dir, celeba_dir, cx = 89, cy = 121):
    print("Loading CelebA from %s" % celeba_dir)
    glob_pattern = os.path.join(celeba_dir, "img_align_celeba_png", "*.png")
    img_filenames = sorted(glob.glob(glob_pattern))
    expected_imgs = 202599
    if len(img_filenames) != expected_imgs:
        error("Expected to find %d images" % expected_imgs)

    with DatasetExporter(dataset_dir, len(img_filenames)) as tfr:
        order = tfr.choose_shuffled_order()
        for idx in range(order.size):
            img = np.asarray(PIL.Image.open(img_filenames[order[idx]]))
            assert img.shape == (218, 178, 3)
            img = img[cy - 64 : cy + 64, cx - 64 : cx + 64]
            img = img.transpose(2, 0, 1) # HWC => CHW
            tfr.add_img(img)

def create_celebahq(dataset_dir, celeba_dir, delta_dir, num_threads = 4, num_tasks = 100):
    print("Loading CelebA from '%s'" % celeba_dir)
    expected_imgs = 202599
    if len(glob.glob(os.path.join(celeba_dir, "img_celeba", "*.jpg"))) != expected_imgs:
        error("Expected to find %d images" % expected_imgs)
    with open(os.path.join(celeba_dir, "Anno", "list_landmarks_celeba.txt"), "rt") as file:
        landmarks = [[float(value) for value in line.split()[1:]] for line in file.readlines()[2:]]
        landmarks = np.float32(landmarks).reshape(-1, 5, 2)

    print("Loading CelebA-HQ deltas from '%s'" % delta_dir)
    import scipy.ndimage
    import hashlib
    import bz2
    import zipfile
    import base64
    import cryptography.hazmat.primitives.hashes
    import cryptography.hazmat.backends
    import cryptography.hazmat.primitives.kdf.pbkdf2
    import cryptography.fernet
    expected_zips = 30
    if len(glob.glob(os.path.join(delta_dir, "delta*.zip"))) != expected_zips:
        error("Expected to find %d zips" % expected_zips)
    with open(os.path.join(delta_dir, "image_list.txt"), "rt") as file:
        lines = [line.split() for line in file]
        fields = dict()
        for idx, field in enumerate(lines[0]):
            type = int if field.endswith("idx") else str
            fields[field] = [type(line[idx]) for line in lines[1:]]
    indices = np.array(fields["idx"])

    # Must use pillow version 3.1.1 for everything to work correctly
    if getattr(PIL, "PILLOW_VERSION", "") != "3.1.1":
        error("create_celebahq requires pillow version 3.1.1") # conda install pillow = 3.1.1

    # Must use libjpeg version 8d for everything to work correctly
    img = np.array(PIL.Image.open(os.path.join(celeba_dir, "img_celeba", "000001.jpg")))
    md5 = hashlib.md5()
    md5.update(img.tobytes())
    if md5.hexdigest() != "9cad8178d6cb0196b36f7b34bc5eb6d3":
        error("create_celebahq requires libjpeg version 8d") # conda install jpeg = 8d

    def rot90(v):
        return np.array([-v[1], v[0]])

    def process_func(idx):
        # Load original image
        orig_idx = fields["orig_idx"][idx]
        orig_file = fields["orig_file"][idx]
        orig_path = os.path.join(celeba_dir, "img_celeba", orig_file)
        img = PIL.Image.open(orig_path)

        # Choose oriented crop rectangle
        lm = landmarks[orig_idx]
        eye_avg = (lm[0] + lm[1]) * 0.5 + 0.5
        mouth_avg = (lm[3] + lm[4]) * 0.5 + 0.5
        eye_to_eye = lm[1] - lm[0]
        eye_to_mouth = mouth_avg - eye_avg
        x = eye_to_eye - rot90(eye_to_mouth)
        x /= np.hypot(*x)
        x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
        y = rot90(x)
        c = eye_avg + eye_to_mouth * 0.1
        quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
        zoom = 1024 / (np.hypot(*x) * 2)

        # Shrink
        shrink = int(np.floor(0.5 / zoom))
        if shrink > 1:
            size = (int(np.round(float(img.size[0]) / shrink)), int(np.round(float(img.size[1]) / shrink)))
            img = img.resize(size, PIL.Image.ANTIALIAS)
            quad /= shrink
            zoom *= shrink

        # Crop
        border = max(int(np.round(1024 * 0.1 / zoom)), 3)
        crop = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), 
            int(np.ceil(max(quad[:,1]))))
        crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]), 
            min(crop[3] + border, img.size[1]))
        if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
            img = img.crop(crop)
            quad -= crop[0:2]

        # Simulate super-resolution
        superres = int(np.exp2(np.ceil(np.log2(zoom))))
        if superres > 1:
            img = img.resize((img.size[0] * superres, img.size[1] * superres), PIL.Image.ANTIALIAS)
            quad *= superres
            zoom /= superres

        # Pad
        pad = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), 
            int(np.ceil(max(quad[:,1]))))
        pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - img.size[0] + border, 0), 
            max(pad[3] - img.size[1] + border, 0))
        if max(pad) > border - 4:
            pad = np.maximum(pad, int(np.round(1024 * 0.3 / zoom)))
            img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), "reflect")
            h, w, _ = img.shape
            y, x, _ = np.mgrid[:h, :w, :1]
            mask = 1.0 - np.minimum(np.minimum(np.float32(x) / pad[0], np.float32(y) / pad[1]), 
                np.minimum(np.float32(w-1-x) / pad[2], np.float32(h-1-y) / pad[3]))
            blur = 1024 * 0.02 / zoom
            img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
            img += (np.median(img, axis = (0,1)) - img) * np.clip(mask, 0.0, 1.0)
            img = PIL.Image.fromarray(np.uint8(np.clip(np.round(img), 0, 255)), "RGB")
            quad += pad[0:2]

        # Transform
        img = img.transform((4096, 4096), PIL.Image.QUAD, (quad + 0.5).flatten(), PIL.Image.BILINEAR)
        img = img.resize((1024, 1024), PIL.Image.ANTIALIAS)
        img = np.asarray(img).transpose(2, 0, 1)

        # Verify MD5
        md5 = hashlib.md5()
        md5.update(img.tobytes())
        assert md5.hexdigest() == fields["proc_md5"][idx]

        # Load delta image and original JPG
        with zipfile.ZipFile(os.path.join(delta_dir, "deltas%05d.zip" % (idx - idx % 1000)), "r") as zip:
            delta_bytes = zip.read("delta%05d.dat" % idx)
        with open(orig_path, "rb") as file:
            orig_bytes = file.read()

        # Decrypt delta image, using original JPG data as decryption key
        algorithm = cryptography.hazmat.primitives.hashes.SHA256()
        backend = cryptography.hazmat.backends.default_backend()
        salt = bytes(orig_file, "ascii")
        kdf = cryptography.hazmat.primitives.kdf.pbkdf2.PBKDF2HMAC(algorithm = algorithm, length = 32, 
            salt = salt, iterations = 100000, backend = backend)
        key = base64.urlsafe_b64encode(kdf.derive(orig_bytes))
        delta = np.frombuffer(bz2.decompress(cryptography.fernet.Fernet(key).decrypt(delta_bytes)), 
            dtype = np.uint8).reshape(3, 1024, 1024)

        # Apply delta image
        img = img + delta

        # Verify MD5
        md5 = hashlib.md5()
        md5.update(img.tobytes())
        assert md5.hexdigest() == fields["final_md5"][idx]
        return img

    with DatasetExporter(dataset_dir, indices.size) as tfr:
        order = tfr.choose_shuffled_order()
        with ThreadPool(num_threads) as pool:
            for img in pool.process_items_concurrently(indices[order].tolist(), process_func = process_func, 
                    max_items_in_flight = num_tasks):
                tfr.add_img(img)

# ----------------------------------------------------------------------------

# Creates TF records for images in a directory. Images are:
# 1. Cropped to rectangle of size (s, ratio * s) (Optional)
# 2. Padded to a square
# 3. Resized to the closest power of 2
# 4. Stored in format C,H,W in the tfrecord file
# Args:
# - dataset_dir: the output directory
# - img_dir: the input directory
# - shuffle: whether to shuffle the dataset before saving
def create_from_imgs(dataset_dir, img_dir, format = "png", shuffle = False, ratio = None, 
        max_imgs = None):
    print("Loading images from %s" % img_dir)
    img_filenames = sorted(glob.glob(f"{img_dir}/**/*.{format}", recursive = True))
    if len(img_filenames) == 0:
        error("No input images found")
    if max_imgs is None:
        max_imgs = len(img_filenames)

    # Check image shape
    img = np.asarray(PIL.Image.open(img_filenames[0]).convert("RGB"))
    resolution = img.shape[0]
    channels = img.shape[2] if img.ndim == 3 else 1
    if channels not in [1, 3]:
        error("Input images must be stored as RGB or grayscale")
    # if img.shape[1] != resolution:
    #     error("Input images must have the same width and height")
    # if resolution != 2 ** int(np.floor(np.log2(resolution))):
    #     error("Input image resolution must be a power-of-two")

    with DatasetExporter(dataset_dir, len(img_filenames)) as tfr:
        order = tfr.choose_shuffled_order() if shuffle else np.arange(len(img_filenames))
        for idx in trange(max_imgs):
            img = PIL.Image.open(img_filenames[order[idx]]).convert("RGB")

            img = misc.crop_max_rectangle(img, ratio)
            img = misc.pad_min_square(img)

            pow2size = 2 ** int(np.round(np.log2(img.size[0])))
            img = img.resize((pow2size, pow2size), PIL.Image.ANTIALIAS)

            img = np.asarray(img)
            if channels == 1:
                img = img[np.newaxis, :, :] # HW => CHW
            else:
                img = img.transpose([2, 0, 1]) # HWC => CHW
            tfr.add_img(img)

def create_from_tfds(dataset_dir, dataset_name, ratio = None, max_imgs = None):
    import tensorflow_datasets as tfds

    print("Loading dataset %s" % dataset_name)
    ds = tfds.load(dataset_name, split = "train", data_dir = f"{dataset_dir}/tfds")
    with DatasetExporter(dataset_dir, 0) as tfr:
        for i, ex in tqdm(enumerate(tfds.as_numpy(ds)), total = max_imgs):
            img = PIL.Image.fromarray(ex["image"])

            img = misc.crop_max_rectangle(img, ratio)
            img = misc.pad_min_square(img)

            pow2size = 2 ** int(np.round(np.log2(img.size[0])))
            img = img.resize((pow2size, pow2size), PIL.Image.ANTIALIAS)

            img = np.asarray(img)
            img = img.transpose([2, 0, 1]) # HWC => CHW
            tfr.add_img(img)
            if max_imgs is not None and i > max_imgs:
                break

def create_from_tfrecords(dataset_dir, tfrecords_dir, ratio = None, max_imgs = None):
    import tensorflow_datasets as tfds

    def parse_tfrecord_tf(record):
        features = tf.io.parse_single_example(record, features={
            "shape": tf.io.FixedLenFeature([3], tf.int64),
            "data": tf.io.FixedLenFeature([], tf.string)})
        data = tf.io.decode_raw(features["data"], tf.uint8)
        data = tf.reshape(data, features["shape"])
        data = tf.transpose(data, [1, 2, 0])
        return data

    print("Loading dataset %s" % tfrecords_dir)
    maxres_file = sorted(glob.glob(os.path.join(tfrecords_dir, "*.tfrecords1of*")))[-1]
    tffiles = sorted(glob.glob(re.sub("1of.*", "*", maxres_file)))

    dataset = tf.data.TFRecordDataset(tffiles)
    dataset = dataset.map(parse_tfrecord_tf, num_parallel_calls = 4)
    with DatasetExporter(dataset_dir, 0) as tfr:
        for i, img in tqdm(enumerate(tfds.as_numpy(dataset)), total = max_imgs):
            img = PIL.Image.fromarray(img)

            img = misc.crop_max_rectangle(img, ratio)
            img = misc.pad_min_square(img)

            pow2size = 2 ** int(np.round(np.log2(img.size[0])))
            img = img.resize((pow2size, pow2size), PIL.Image.ANTIALIAS)

            img = np.asarray(img)
            img = img.transpose([2, 0, 1]) # HWC => CHW
            tfr.add_img(img)
            if max_imgs is not None and i > max_imgs:
                break

def create_from_lmdb(dataset_dir, lmdb_dir, ratio = None, max_imgs = None):
    import lmdb
    print("Loading dataset %s" % lmdb_dir)
    bad_imgs = 0
    with lmdb.open(lmdb_dir, readonly = True).begin(write = False) as txn:
        if max_imgs is None:
            max_imgs = txn.stat()["entries"]

        with DatasetExporter(dataset_dir, max_imgs, verbose = False) as tfr:
            for idx, (_key, value) in tqdm(enumerate(txn.cursor()), total = max_imgs):
                try:
                    img = PIL.Image.open(io.BytesIO(value))

                    img = misc.crop_max_rectangle(img, ratio)
                    img = misc.pad_min_square(img)

                    pow2size = 2 ** int(np.round(np.log2(img.size[0])))
                    img = img.resize((pow2size, pow2size), PIL.Image.ANTIALIAS)

                    img = np.asarray(img)
                    img = img.transpose([2, 0, 1]) # HWC => CHW
                    tfr.add_img(img)
                except:
                    bad_imgs += 1
                    pass
                if tfr.curr_imgnum == max_imgs:
                    break

    if bad_imgs > 0:
        print(f"Couldn't read {bad_imgs} out of {max_imgs} images")

def create_from_npy(dataset_dir, npy_filename, shuffle = False, max_imgs = None):
    print("Loading NPY archive from %s" % npy_filename)
    if max_imgs is None:
        max_imgs = npy_data.shape[0]

    with open(npy_filename, "rb") as npy_file:
        npy_data = np.load(npy_file)
        with DatasetExporter(dataset_dir, npy_data.shape[0]) as tfr:
            order = tfr.choose_shuffled_order() if shuffle else np.arange(npy_data.shape[0])
            for idx in trange(max_imgs):
                tfr.add_img(npy_data[order[idx]])
            npy_filename = os.path.splitext(npy_filename)[0] + "-labels.npy"
            if os.path.isfile(npy_filename):
                tfr.add_labels(np.load(npy_filename)[order])

def create_from_hdf5(dataset_dir, hdf5_filename, shuffle = False, max_imgs = None):
    import h5py # conda install h5py
    print("Loading HDF5 archive from %s" % hdf5_filename)
    if max_imgs is None:
        max_imgs = npy_data.shape[0]

    with h5py.File(hdf5_filename, "r") as hdf5_file:
        hdf5_data = max([value for key, value in hdf5_file.items() if key.startswith("data")], 
            key = lambda lod: lod.shape[3])
        with DatasetExporter(dataset_dir, hdf5_data.shape[0]) as tfr:
            order = tfr.choose_shuffled_order() if shuffle else np.arange(hdf5_data.shape[0])
            for idx in trange(max_imgs):
                tfr.add_img(hdf5_data[order[idx]])
            npy_filename = os.path.splitext(hdf5_filename)[0] + "-labels.npy"
            if os.path.isfile(npy_filename):
                tfr.add_labels(np.load(npy_filename)[order])

# ----------------------------------------------------------------------------

def execute_cmdline(argv):
    prog = argv[0]
    parser = argparse.ArgumentParser(
        prog        = prog,
        description = "Tool for creating TFRecords datasets for the GANsformer.",
        epilog      = "Type %s <command> -h for more information." % prog)

    subparsers = parser.add_subparsers(dest = "command")
    subparsers.required = True
    def add_command(cmd, desc, example = None):
        epilog = "Example: %s %s" % (prog, example) if example is not None else None
        return subparsers.add_parser(cmd, description = desc, help = desc, epilog = epilog)

    p = add_command("display",          "Display images in dataset.",
                                        "display datasets/mnist")
    p.add_argument( "dataset_dir",      help = "Directory containing dataset")

    p = add_command("extract",          "Extract images from dataset.",
                                        "extract datasets/mnist mnist-images")
    p.add_argument( "dataset_dir",      help = "Directory containing dataset")
    p.add_argument( "output_dir",       help = "Directory to extract the images into")

    p = add_command("compare",          "Compare two datasets.",
                                        "compare datasets/mydataset datasets/mnist")
    p.add_argument( "dataset_dir_a",    help = "Directory containing first dataset")
    p.add_argument( "dataset_dir_b",    help = "Directory containing second dataset")
    p.add_argument( "--ignore_labels",  help = "Ignore labels (default: 0)", type = int, default = 0)

    p = add_command("create_mnist",     "Create dataset for MNIST.",
                                        "create_mnist datasets/mnist ~/downloads/mnist")
    p.add_argument( "dataset_dir",      help = "New dataset directory to be created")
    p.add_argument( "mnist_dir",        help = "Directory containing MNIST")

    p = add_command("create_mnistrgb",  "Create dataset for MNIST-RGB.",
                                        "create_mnistrgb datasets/mnistrgb ~/downloads/mnist")
    p.add_argument( "dataset_dir",      help = "New dataset directory to be created")
    p.add_argument( "mnist_dir",        help = "Directory containing MNIST")
    p.add_argument( "--num_imgs",       help = "Number of composite images to create (default: 1000000)", type = int, default = 1000000)
    p.add_argument( "--random_seed",    help = "Random seed (default: 123)", type = int, default = 123)

    p = add_command("create_cifar10",   "Create dataset for CIFAR-10.",
                                        "create_cifar10 datasets/cifar10 ~/downloads/cifar10")
    p.add_argument( "dataset_dir",      help = "New dataset directory to be created")
    p.add_argument( "cifar10_dir",      help = "Directory containing CIFAR-10")

    p = add_command("create_cifar100",  "Create dataset for CIFAR-100.",
                                        "create_cifar100 datasets/cifar100 ~/downloads/cifar100")
    p.add_argument( "dataset_dir",      help = "New dataset directory to be created")
    p.add_argument( "cifar100_dir",     help = "Directory containing CIFAR-100")

    p = add_command("create_svhn",      "Create dataset for SVHN.",
                                        "create_svhn datasets/svhn ~/downloads/svhn")
    p.add_argument( "dataset_dir",      help = "New dataset directory to be created")
    p.add_argument( "svhn_dir",         help = "Directory containing SVHN")

    p = add_command("create_lsun",      "Create dataset for single LSUN category.",
                                        "create_lsun datasets/lsun-car-100k ~/downloads/lsun/car_lmdb --resolution 256 --max_imgs 100000")
    p.add_argument( "dataset_dir",      help = "New dataset directory to be created")
    p.add_argument( "lmdb_dir",         help = "Directory containing LMDB database")
    p.add_argument( "--resolution",     help = "Output resolution (default: 256)", type = int, default = 256)
    p.add_argument( "--max_imgs",       help = "Maximum number of images (default: none)", type = int, default = None)

    p = add_command("create_lsun_wide", "Create LSUN dataset with non-square aspect ratio.",
                                        "create_lsun_wide datasets/lsun-car-512x384 ~/downloads/lsun/car_lmdb --width 512 --height 384")
    p.add_argument( "dataset_dir",      help = "New dataset directory to be created")
    p.add_argument( "lmdb_dir",         help = "Directory containing LMDB database")
    p.add_argument( "--width",          help = "Output width (default: 512)", type = int, default = 512)
    p.add_argument( "--height",         help = "Output height (default: 384)", type = int, default = 384)
    p.add_argument( "--max_imgs",       help = "Maximum number of imgs (default: none)", type = int, default = None)

    p = add_command("create_celeba",    "Create dataset for CelebA.",
                                        "create_celeba datasets/celeba ~/downloads/celeba")
    p.add_argument( "dataset_dir",      help = "New dataset directory to be created")
    p.add_argument( "celeba_dir",       help = "Directory containing CelebA")
    p.add_argument( "--cx",             help = "Center X coordinate (default: 89)", type = int, default = 89)
    p.add_argument( "--cy",             help = "Center Y coordinate (default: 121)", type = int, default = 121)

    p = add_command("create_celebahq",  "Create dataset for CelebA-HQ.",
                                        "create_celebahq datasets/celebahq ~/downloads/celeba ~/downloads/celeba-hq-deltas")
    p.add_argument( "dataset_dir",      help = "New dataset directory to be created")
    p.add_argument( "celeba_dir",       help = "Directory containing CelebA")
    p.add_argument( "delta_dir",        help = "Directory containing CelebA-HQ deltas")
    p.add_argument( "--num_threads",    help = "Number of concurrent threads (default: 4)", type = int, default = 4)
    p.add_argument( "--num_tasks",      help = "Number of concurrent processing tasks (default: 100)", type = int, default = 100)

    p = add_command("create_from_imgs", "Create dataset from a directory full of images.",
                                        "create_from_imgs datasets/mydataset myimagedir")
    p.add_argument( "dataset_dir",      help = "New dataset directory to be created")
    p.add_argument( "img_dir",          help = "Directory containing the images")
    p.add_argument( "--shuffle",        help = "Randomize image order (default: 1)", type = int, default = 1)
    p.add_argument( "--ratio",          help = "Crop ratio (default: no crop)", type = int, default = None)

    p = add_command("create_from_tfrecords", "Create dataset from a tfrecords direcotry.",
                                        "create_from_tfrecords datasets/mydataset mytfrecorddir")
    p.add_argument( "dataset_dir",      help = "New dataset directory to be created")
    p.add_argument( "tfrecords_dir",    help = "Directory containing the tfrecord files")
    p.add_argument( "--shuffle",        help = "Randomize image order (default: 1)", type = int, default = 1)
    p.add_argument( "--ratio",          help = "Crop ratio (default: no crop)", type = int, default = None)

    p = add_command("create_from_hdf5", "Create dataset from legacy HDF5 archive.",
                                        "create_from_hdf5 datasets/celebahq ~/downloads/celeba-hq-1024x1024.h5")
    p.add_argument( "dataset_dir",      help = "New dataset directory to be created")
    p.add_argument( "hdf5_filename",    help = "HDF5 archive containing the images")
    p.add_argument( "--shuffle",        help = "Randomize image order (default: 1)", type = int, default = 1)

    p = add_command("create_from_npy",  "Create dataset from legacy HDF5 archive.",
                                        "create_from_hdf5 datasets/celebahq ~/downloads/celeba-hq-1024x1024.h5")
    p.add_argument( "dataset_dir",      help = "New dataset directory to be created")
    p.add_argument( "npy_filename",     help = "HDF5 archive containing the images")
    p.add_argument( "--shuffle",        help = "Randomize image order (default: 1)", type = int, default = 1)

    args = parser.parse_args(argv[1:] if len(argv) > 1 else ["-h"])
    func = globals()[args.command]
    del args.command
    func(**vars(args))

if __name__ == "__main__":
    execute_cmdline(sys.argv)
