import os
import numpy as np
import PIL.Image
import json
import torch
import dnnlib
from training import misc
import glob 

#----------------------------------------------------------------------------

class Dataset(torch.utils.data.Dataset):
    def __init__(self,
        name,                       # Name of the dataset
        shape,                      # Shape of the raw image data (NCHW)
        max_items      = None,      # Limit the size of the dataset. None = no limit
        use_labels     = False,     # Enable conditioning labels? False = label dimension is zero
        mirror_augment = False,     # Augment the dataset with horizontally mirrored images
        ratio          = 1.0,       # Image height/width ratio in the dataset
        **_kwargs                   # Ignore unrecognized keyword args
    ):
        self._name = name
        self.shape = list(shape)
        self.use_labels = use_labels
        self.ratio = ratio

        self._label_shape = None # To be overridden by subclass

        # Apply max_items
        self.idx = np.arange(self.shape[0], dtype = np.int64)
        if (max_items is not None) and (self.idx.size > max_items):
            np.random.shuffle(self.idx)
            self.idx = np.sort(self.idx[:max_items])

        # Double the index size ti include mirrored images
        self.mirror_augment = np.zeros(self.idx.size, dtype = np.uint8)
        if mirror_augment:
            self.idx = np.tile(self.idx, 2)
            self.mirror_augment = np.concatenate([self.mirror_augment, np.ones_like(self.mirror_augment)])

        # Load labels (or initialize to zeros if no labels are provided for the dataset)
        self.labels = self._load_labels()

    def _load_image(self, idx): # to be overridden by subclass
        raise NotImplementedError

    def _load_labels(self): # to be overridden by subclass
        raise NotImplementedError

    def __len__(self):
        return self.idx.size

    def __getitem__(self, idx):
        image = self._load_image(self.idx[idx])

        assert isinstance(image, np.ndarray)
        assert list(image.shape) == self.image_shape
        assert image.dtype == np.uint8
        
        # Apply mirror augment
        if self.mirror_augment[idx]:
            assert image.ndim == 3 # CHW
            image = image[:, :, ::-1]
        
        return image.copy(), self.get_label(idx)

    def get_label(self, idx):
        label = self.labels[self.idx[idx]]
        if label.dtype == np.int64:
            onehot = np.zeros(self.label_shape, dtype = np.float32)
            onehot[label] = 1
            label = onehot
        return label.copy()

    def get_random_labels(self, num):
        return np.stack([self.get_label(np.random.randint(len(self))) for _ in range(num)])

    @property
    def name(self):
        return self._name

    @property
    def image_shape(self):
        return list(self.shape[1:])

    @property
    def num_channels(self):
        assert len(self.image_shape) == 3 # CHW
        return self.image_shape[0]

    @property
    def resolution(self):
        assert len(self.image_shape) == 3 # CHW
        assert self.image_shape[1] == self.image_shape[2]
        return self.image_shape[1]

    @property
    def label_shape(self):
        if self._label_shape is None:
            if self.labels.dtype == np.int64:
                self._label_shape = [int(np.max(self.labels)) + 1]
            else:
                self._label_shape = self.labels.shape[1:]
        return list(self._label_shape)

    @property
    def label_dim(self):
        assert len(self.label_shape) == 1
        return self.label_shape[0]

    @property
    def has_labels(self):
        return any(x != 0 for x in self.label_shape)

    @property
    def has_onehot_labels(self):
        return self.labels.dtype == np.int64

#----------------------------------------------------------------------------

class ImageFolderDataset(Dataset):
    def __init__(self, path, resolution, **kwargs):
        self.path = path
        if not os.path.exists(f"{path}/{resolution}"):
            misc.error(f"Dataset folder {path}/{resolution} doesn't exists. Follow data preparation instructions using the prepare_data.py script.")

        self.img_files = sorted(glob.glob(f"{path}/{resolution}/*.png"))
        # misc.log(f"Found {len(self.img_files)} images in the dataset.")

        name = os.path.splitext(os.path.basename(self.path))[0]
        
        shape = [len(self.img_files)] + list(self._load_image(0).shape)
        if resolution is not None and (shape[2] != resolution or shape[3] != resolution):
            misc.error("Image files do not match the specified resolution")
        
        super().__init__(name = name, shape = shape, **kwargs)

    def _load_image(self, idx):
        with open(self.img_files[idx], "rb") as f:
            image = np.array(PIL.Image.open(f))
        if image.ndim == 2:
            image = image[:, :, np.newaxis] # HW => HWC
        image = image.transpose(2, 0, 1) # HWC => CHW
        return image

    def _load_labels(self):
        if not self.use_labels:
            return np.zeros([self.shape[0], 0], dtype = np.float32)
        
        with self.open_file(f"{self.path}/labels.npy", "rb") as f:
            labels = np.load(f)

        labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])
        return labels
