# Miscellaneous utility functions
import numpy as np
import PIL.Image
import seaborn as sns
from termcolor import colored
from tqdm import tqdm
import math
import os

# Colorful prints
# ----------------------------------------------------------------------------

# string -> bold string
def bold(txt, **kwargs):
    return colored(str(txt),attrs = ["bold"])

# string -> colorful bold string
def bcolored(txt, color):
    return colored(str(txt), color, attrs = ["bold"])

# conditional coloring if num > maxval
# maxval = 0 turns functionality off
def cond_bcolored(num, maxval, color):
    num = num or 0
    txt = f"{num:>6.3f}"
    if maxval > 0 and num > maxval:
        return bcolored(txt, color)
    return txt

def error(txt):
    print(bcolored(f"Error: {txt}", "red"))
    exit()

def log(txt, color = None, log = True):
    if log:
        print(bcolored(txt, color) if color is not None else bold(txt))

# File processing
# ----------------------------------------------------------------------------

# Delete a list of files
def rm(files):
    for f in files:
        os.remove(f)

# Make directory
def mkdir(d):
    os.makedirs(d, exist_ok = True)

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
        save_npy(npy, path % (offset + i))

# Image utilities
# ----------------------------------------------------------------------------

# Cut image center (to avoid computing detector score on the padding margins)
def crop_tensor(imgs, ratio = 1.0): 
    if ratio == 1.0 or ratio is None:
        return imgs
    width = int(imgs.shape[-1])
    start = int(math.floor(((1 - ratio) * width / 2)))
    end = int(math.ceil((1 + ratio) * width / 2))
    imgs = imgs[...,start:end + 1,:]
    return imgs

def crop_tensor_shape(shape, ratio = 1.0): 
    if ratio == 1.0 or ratio is None:
        return shape
    width = int(shape[-1])
    start = int(math.floor(((1 - ratio) * width / 2)))
    end = int(math.ceil((1 + ratio) * width / 2))
    return (end - start, width)

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

# Scale data from drange_in (a,b) range to drange_out (c,d) range
# and supports hsv conversions (not used by default)
def adjust_range(data, drange_in, drange_out):
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
def to_pil(img, drange = [-1,1]): # , to_cpu = True
    assert img.ndim == 2 or img.ndim == 3
    if img.ndim == 3:
        if img.shape[0] == 1:
            img = img[0] # grayscale CHW => HW
        else:
            img = img.transpose(1, 2, 0) # CHW -> HWC

    img = adjust_range(img, drange, [0, 255])
    img = np.rint(img).clip(0, 255).astype(np.uint8)

    fmt = "L"
    if img.ndim == 3:
        fmt = {1: "L", 3: "RGB", 4: "RGBA"}.get(img.shape[-1], "L")
    img = PIL.Image.fromarray(img, fmt)

    return img

# Size and contents of the image snapshot grids that are exported
# periodically during training
def create_img_grid(imgs, grid_size = None):
    assert imgs.ndim == 3 or imgs.ndim == 4
    num, img_w, img_h = imgs.shape[0], imgs.shape[-1], imgs.shape[-2]

    if grid_size is not None:
        gw, gh = grid_size
    else:
        gw = max(int(np.ceil(np.sqrt(num))), 1)
        gh = max((num - 1) // gw + 1, 1)

    gw, gh = grid_size
    _N, C, H, W = imgs.shape
    imgs = imgs.reshape(gh, gw, C, H, W)
    imgs = imgs.transpose(2, 0, 3, 1, 4)
    imgs = imgs.reshape(C, gh * H, gw * W)
    return imgs

def save_img_grid(imgs, filename, drange = [0,1], grid_size = None):
    to_pil(create_img_grid(imgs, grid_size), drange).save(filename) # .cpu().numpy()

def setup_snapshot_img_grid(dataset, size = (3, 2)):
    # dataset: dataset object to iterate over
    # size:
    ### "1080p" = to be viewed on 1080p display
    ### "4k/8k" = to be viewed on 4k/8k display
    ### int for a custom number of images

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

    # No labels => show random subset of training samples
    if not dataset.has_labels:
        all_indices = list(range(len(dataset)))
        np.random.shuffle(all_indices)
        grid_indices = [all_indices[i % len(all_indices)] for i in range(gw * gh)]

    else:
        # Group training samples by label
        label_groups = dict() # label => [idx, ...]
        for idx in range(len(dataset)):
            label = tuple(dataset.get_label(idx).flat[::-1])
            if label not in label_groups:
                label_groups[label] = []
            label_groups[label].append(idx)

        # Reorder
        label_order = sorted(label_groups.keys())
        for label in label_order:
            rnd.shuffle(label_groups[label])

        # Organize into grid
        grid_indices = []
        for y in range(gh):
            label = label_order[y % len(label_order)]
            indices = label_groups[label]
            grid_indices += [indices[x % len(indices)] for x in range(gw)]
            label_groups[label] = [indices[(i + gw) % len(indices)] for i in range(len(indices))]

    # Load data
    images, labels = zip(*[dataset[i] for i in grid_indices])
    return (gw, gh), np.stack(images), np.stack(labels)

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
            save_img_grid(imgs, clean_filename(path % offset), drange, grid_size)
        else:
            imgs = enumerate(imgs)
            if verbose:
                imgs = tqdm(list(imgs))
            for i, img in imgs:
                img = to_pil(img, drange = drange)
                img = crop_max_rectangle(img, ratio)
                img.save(path % (offset + i))
    return save_images

# Save a list of blended two-layer images with ordering and according to a path template
def save_blends_builder(drange, ratio, grid_size, grid = False, verbose = False, alpha = 0.3):
    def save_blends(imgs_a, imgs_b, path, offset = 0):
        if grid:
            img_a = to_pil(create_img_grid(imgs_a, grid_size), drange) # .cpu().numpy()
            img_b = to_pil(create_img_grid(imgs_b, grid_size), drange) # .cpu().numpy()
            blend = PIL.Image.blend(img_a, img_b, alpha = alpha)
            blend.save(clean_filename(path % offset))
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
                blend.save(path % (offset + i))
    return save_blends
