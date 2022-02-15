# Miscellaneous utility functions
import tensorflow as tf
import numpy as np
import PIL.Image
import pickle
import dnnlib
import math
import glob
import os

import seaborn as sns
from termcolor import colored
from tqdm import tqdm

# Colorful prints
# ----------------------------------------------------------------------------

# string -> bold string
def bold(txt, **kwargs):
    return colored(str(txt),attrs = ["bold"])

# string -> colorful bold string
def bcolored(txt, color):
    return colored(str(txt), color, attrs = ["bold"])

# conditional coloring if num > maxval.
# maxval = 0 turns functionality off.
def cond_bcolored(num, maxval, color):
    num = num or 0
    txt = f"{num:>6.3f}"
    if maxval > 0 and num > maxval:
        return bcolored(txt, color)
    return txt

def error(txt):
    print(bcolored(f"Error: {txt}", "red"))
    exit()

def log(txt, color = None):
    print(bcolored(txt, color) if color is not None else bold(txt))

# File processing
# ----------------------------------------------------------------------------
# Convenience wrappers for pickle that are able to load data produced by
# older versions of the StyleGAN code, and from external URLs.

def open_file_or_url(file_or_url):
    if dnnlib.util.is_url(file_or_url):
        return dnnlib.util.open_url(file_or_url, cache_dir = ".GANformer-cache")
    return open(file_or_url, "rb")

def load_pkl(file_or_url):
    with open_file_or_url(file_or_url) as file:
        return pickle.load(file, encoding = "latin1")

def save_pkl(obj, filename):
    with open(filename, "wb") as file:
        pickle.dump(obj, file, protocol = pickle.HIGHEST_PROTOCOL)

# Save a numpy file
def save_npy(mat, filename):
    with open(filename, 'wb') as f:
        np.save(f, mat)

# Saves a list of numpy arrays with ordering and according to a path template
def save_npys(npys, path, verbose = False, offset = 0):
    npys = enumerate(npys)
    if verbose:
        npys = tqdm(list(npys))
    for i, npy in npys:
        save_npy(npy, dnnlib.make_run_dir_path(path % (offset + i)))

# Delete a list of files
def rm(files):
    for f in files:
        os.remove(f)

# Make directory
def mkdir(d):
    os.makedirs(d, exist_ok = True)

# Image utilities
# ----------------------------------------------------------------------------

def float2uint(imgs):
    imgs = adjust_dynamic_range(imgs, [-1.0, 1.0], [0, 255])
    imgs = tf.cast(tf.clip_by_value(tf.rint(imgs), 0, 255), tf.uint8)
    return imgs

def float2uint_np(imgs):
    imgs = adjust_dynamic_range(imgs, [-1.0, 1.0], [0, 255])
    imgs = np.rint(imgs).clip(0, 255).astype(np.uint8)
    return imgs

# Cut image center (to avoid computing inception score on the padding margins)
def crop_tensor(imgs, ratio = 1.0): 
    if ratio == 1.0 or ratio is None:
        return imgs
    width = imgs.shape.as_list()[-1] if isinstance(imgs, tf.Tensor) else imgs.shape[-1]
    
    start = int(math.floor(((1 - ratio) * width / 2)))
    end = int(math.ceil((1 + ratio) * width / 2))
    imgs = imgs[...,start:end + 1,:]
    return imgs

def crop_tensor_shape(shape, ratio = 1.0): 
    if ratio == 1.0 or ratio is None:
        return shape
    width = shape[2]
    start = int(math.floor(((1 - ratio) * width / 2)))
    end = int(math.ceil((1 + ratio) * width / 2))
    return (end - start, width)

def pad_tensor(imgs):
    shape = tf.shape(imgs)
    h, w = shape[-2], shape[-1]
    if h > w:
        l1 = tf.floor((h - w) / 2)
        l2 = tf.ceil((h - w) / 2)
        padding = ((0, 0), (l1, l2))
    else:
        l1 = tf.floor((w - h) / 2)
        l2 = tf.ceil((w - h) / 2)        
        padding = ((l1, l2), (0, 0))
    padding = ((0, 0), (0, 0)) + padding
    imgs = tf.pad(imgs, padding)
    return imgs

# Crop center rectangle of size (cw, ch)
def crop_center(img, cw, ch):
    w, h = img.size
    return img.crop(((w - cw) // 2, (h - ch) // 2, (w + cw) // 2, (h + ch) // 2))

# Crop max rectangle of size (s, ratio * s) where s = min(w,h)
# If ratio is None, keep same image
def crop_max_rectangle(img, ratio = 1.0):
    if ratio is None:
        return img
    s = min(img.size[0], img.size[1] / ratio)
    return crop_center(img, s, ratio * s)

# Pad an image of dimension w,h to the smallest containing square (max(w,h), max(w,h))
def pad_min_square(img, pad_color = (0, 0, 0)):
    w, h = img.size
    if w == h:
        return img
    s = max(w, h)
    result = PIL.Image.new(img.mode, (s, s), pad_color)
    offset_x = max(0, (h - w) // 2)
    offset_y = max(0, (w - h) // 2)
    result.paste(img, (offset_x, offset_y))
    return result

# Conversions between rgb and hsv (not used by default)
def rgb_to_hsv(rgb):
    input_shape = rgb.shape
    rgb = rgb.reshape(-1, 3)
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]

    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    v = maxc

    deltac = maxc - minc
    s = deltac / maxc
    deltac[deltac == 0] = 1  # to not divide by zero (those results in any way would be overridden in next lines)
    rc = (maxc - r) / deltac
    gc = (maxc - g) / deltac
    bc = (maxc - b) / deltac

    h = 4.0 + gc - rc
    h[g == maxc] = 2.0 + rc[g == maxc] - bc[g == maxc]
    h[r == maxc] = bc[r == maxc] - gc[r == maxc]
    h[minc == maxc] = 0.0

    h = (h / 6.0) % 1.0
    res = np.dstack([h, s, v])
    return res.reshape(input_shape)

def hsv_to_rgb(hsv):
    input_shape = hsv.shape
    hsv = hsv.reshape(-1, 3)
    h, s, v = hsv[:, 0], hsv[:, 1], hsv[:, 2]

    i = np.int32(h * 6.0)
    f = (h * 6.0) - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    i = i % 6

    rgb = np.zeros_like(hsv)
    v, t, p, q = v.reshape(-1, 1), t.reshape(-1, 1), p.reshape(-1, 1), q.reshape(-1, 1)
    rgb[i == 0] = np.hstack([v, t, p])[i == 0]
    rgb[i == 1] = np.hstack([q, v, p])[i == 1]
    rgb[i == 2] = np.hstack([p, v, t])[i == 2]
    rgb[i == 3] = np.hstack([p, q, v])[i == 3]
    rgb[i == 4] = np.hstack([t, p, v])[i == 4]
    rgb[i == 5] = np.hstack([v, p, q])[i == 5]
    rgb[s == 0.0] = np.hstack([v, v, v])[s == 0.0]

    return rgb.reshape(input_shape)

# Scale data from drange_in (a,b) range to drange_out (c,d) range
# and supports hsv conversions (not used by default)
def adjust_dynamic_range(data, drange_in, drange_out, hsv = False):
    if not hsv:
        return adjust_dynamic_range_aux(data, drange_in, drange_out)
    else:
        data = adjust_dynamic_range_aux(data, drange_in, [0.0, 1.0])

        axis = -1
        for i, x in enumerate(list(data.shape)):
            if x == 3:
                axis = i

        if axis != -1:
            ln = len(list(data.shape))
            shape = list(range(ln))
            shape[axis], shape[-1] = ln - 1, axis
            data = tf.transpose(data, shape) if tf.is_tensor(data) else np.transpose(data, shape)

        if drange_in == [0, 255]:
            data = tf.image.rgb_to_hsv(data) if tf.is_tensor(data) else rgb_to_hsv(data)
        elif drange_out == [0, 255]:
            data = tf.image.hsv_to_rgb(data) if tf.is_tensor(data) else hsv_to_rgb(data)
        else:
            error("Image adjustment invalid range: ", drange_in, drange_out)

        if axis != -1:
            data = tf.transpose(data, shape) if tf.is_tensor(data) else np.transpose(data, shape)

        return adjust_dynamic_range_aux(data, [0.0, 1.0], drange_out)

def adjust_dynamic_range_aux(data, drange_in, drange_out):
    if drange_in != drange_out:
        scale = (np.float32(drange_out[1]) - np.float32(drange_out[0])) / \
            (np.float32(drange_in[1]) - np.float32(drange_in[0]))
        bias = (np.float32(drange_out[0]) - np.float32(drange_in[0]) * scale)
        data = data * scale + bias
    return data

# Converts a numpy CHW image to a Pillow
# 1. Transpose channel from first to last dimension
# 2. Adjust range from drange to uint [0,255]
# 3. Convert format to Pillow
def to_pil(img, drange = [-1,1]):
    assert img.ndim == 2 or img.ndim == 3
    if img.ndim == 3:
        if img.shape[0] == 1:
            img = img[0] # grayscale CHW => HW
        else:
            img = img.transpose(1, 2, 0) # CHW -> HWC

    img = adjust_dynamic_range(img, drange, [0,255])
    img = np.rint(img).clip(0, 255).astype(np.uint8)

    fmt = "L"
    if img.ndim == 3:
        fmt = {1: "L", 3: "RGB", 4: "RGBA"}.get(img.shape[-1], "L")
    img = PIL.Image.fromarray(img, fmt)

    return img

# Randomly horizontally flip the images in the batch BCHW
def apply_mirror_augment(batch):
    mask = np.random.rand(batch.shape[0]) < 0.5
    batch = np.array(batch)
    batch[mask] = batch[mask, :, :, ::-1]
    return batch

# Size and contents of the image snapshot grids that are exported
# periodically during training
def create_img_grid(imgs, grid_size = None):
    assert imgs.ndim == 3 or imgs.ndim == 4
    num, img_w, img_h = imgs.shape[0], imgs.shape[-1], imgs.shape[-2]

    if grid_size is not None:
        grid_w, grid_h = tuple(grid_size)
    else:
        grid_w = max(int(np.ceil(np.sqrt(num))), 1)
        grid_h = max((num - 1) // grid_w + 1, 1)

    grid = np.zeros(list(imgs.shape[1:-2]) + [grid_h * img_h, grid_w * img_w], dtype = imgs.dtype)
    for idx in range(num):
        x = (idx % grid_w) * img_w
        y = (idx // grid_w) * img_h
        grid[..., y : y + img_h, x : x + img_w] = imgs[idx]
    return grid

def save_img_grid(imgs, filename, drange = [0,1], grid_size = None):
    to_pil(create_img_grid(imgs, grid_size), drange).save(filename)

def setup_snapshot_img_grid(dataset, size = "1080p", layout = "random"):
    # dataset: dataset object to iterate over
    # size:
    ### "1080p" = to be viewed on 1080p display
    ### "4k/8k" = to be viewed on 4k/8k display
    ### int for a custom number of images
    # layout:
    ### "random" = grid contents are selected randomly
    ### "row_per_class" = each row corresponds to one class label

    # Select size
    gw = 1; gh = 1
    if size == "1080p":
        gw = np.clip(1920 // 256, 3, 32)
        gh = np.clip(1080 // 256, 2, 32)
    elif size == "4k":
        gw = np.clip(3840 // dataset.shape[2], 7, 32)
        gh = np.clip(2160 // dataset.shape[1], 4, 32)
    elif size == "8k":
        gw = np.clip(7680 // dataset.shape[2], 7, 32)
        gh = np.clip(4320 // dataset.shape[1], 4, 32)
    elif isinstance(size, tuple):
        gw, gh = size
    else: # if size is int, return an image list with that number of images
        gw = size

    # Initialize data arrays
    reals = np.zeros([gw * gh] + dataset.shape, dtype = dataset.dtype)
    labels = np.zeros([gw * gh, dataset.label_size], dtype = dataset.label_dtype)

    # Random layout
    if layout == "random":
        reals[:], labels[:] = dataset.get_batch_np(gw * gh)

    # Class-conditional layouts
    class_layouts = dict(row_per_class = [gw, 1], col_per_class = [1, gh], class4x4 = [4, 4])
    if layout in class_layouts:
        bw, bh = class_layouts[layout]
        nw = (gw - 1) // bw + 1
        nh = (gh - 1) // bh + 1
        blocks = [[] for _i in range(nw * nh)]
        for _iter in range(1000000):
            (real, seg), label = dataset.get_batch_np(1)
            idx = np.argmax(label[0])
            while idx < len(blocks) and len(blocks[idx]) >= bw * bh:
                idx += dataset.label_size
            if idx < len(blocks):
                blocks[idx].append((real, seg, label))
                if all(len(block) >= bw * bh for block in blocks):
                    break
        for i, block in enumerate(blocks):
            for j, (real, seg, label) in enumerate(block):
                x = (i %  nw) * bw + j %  bw
                y = (i // nw) * bh + j // bw
                if x < gw and y < gh:
                    reals[x + y * gw] = real[0]
                    labels[x + y * gw] = label[0]

    return (gw, gh), reals, labels

# Return color palette normalized to [-1,1]
def get_colors(num):
    colors = sns.color_palette("hls", num)
    colors = [[(2 * p - 1) for p in c] for c in colors]
    return colors

# Convert a list of images to a GIF file
def save_gif(imgs, filename, duration = 50):
    imgs[0].save(filename, save_all = True, append_images = imgs[1:], duration = duration, loop = 0)

def clean_filename(filename):
    return filename.replace("_000000", "").replace("000000_", "")

# Save a list of images with ordering and according to a path template
def save_images_builder(drange, ratio, grid_size, grid = False, verbose = False):
    def save_images(imgs, path, offset = 0):
        if grid:
            save_img_grid(imgs, dnnlib.make_run_dir_path(clean_filename(path % offset)), drange, grid_size)
        else:
            imgs = enumerate(imgs)
            if verbose:
                imgs = tqdm(list(imgs))
            for i, img in imgs:
                img = to_pil(img, drange = drange)
                img = crop_max_rectangle(img, ratio)
                img.save(dnnlib.make_run_dir_path(path % (offset + i)))
    return save_images

# Save a list of blended two-layer images with ordering and according to a path template
def save_blends_builder(drange, ratio, grid_size, grid = False, verbose = False, alpha = 0.3):
    def save_blends(imgs_a, imgs_b, path, offset = 0):
        if grid:
            img_a = to_pil(create_img_grid(imgs_a, grid_size), drange)
            img_b = to_pil(create_img_grid(imgs_b, grid_size), drange)
            blend = PIL.Image.blend(img_a, img_b, alpha = alpha)
            blend.save(dnnlib.make_run_dir_path(clean_filename(path % offset)))
        else:
            img_pairs = zip(imgs_a, imgs_b)
            img_pairs = enumerate(img_pairs)
            if verbose:
                img_pairs = tqdm(list(img_pairs))            
            for i, (img_a, img_b) in img_pairs:
                img_a = to_pil(img_a, drange = drange)
                img_b = to_pil(img_b, drange = drange)
                blend = PIL.Image.blend(img_a, img_b, alpha = alpha)
                blend = crop_max_rectangle(blend, ratio)
                blend.save(dnnlib.make_run_dir_path(path % (offset + i)))
    return save_blends
