# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action = "ignore", category = FutureWarning)

import os
import sys
import tqdm
import time
import json
import glob
import gdown
import urllib
import zipfile
import hashlib

import argparse
import numpy as np
from training import misc 
from dnnlib import EasyDict
import dataset_tool

catalog = {
   "ffhq": EasyDict({
        "name": "FFHQ", # Dataset name for logging 
        "filename": "ffhq-r08.tfrecords1of1", # Local file name
        "url": "http://downloads.cs.stanford.edu/nlp/data/dorarad/ffhq-r08.tfrecords1of1", # download URL
        "md5": "74de4f07dc7bfb07c0ad4471fdac5e67", # MD5 checksum to potentially skip download
        "ratio": 1, # height/width ratio
        "size": 13, # download size in GB
        "shards": 1, # Number of tfrecord shards
        "img_num": 70000 # Number of images
    }),
    "bedrooms": EasyDict({
        "name": "LSUN-Bedrooms", # Dataset name for logging 
        "filename": "bedroom_train_lmdb.zip",
        "url": "http://dl.yf.io/lsun/scenes/bedroom_train_lmdb.zip",
        "md5": "f2c5d904a82a6295dbdccb322b4b0a99",
        "dir": "bedroom_train_lmdb",
        "ratio": 188/256,
        "size": 43,
        "shards": 64,
        "img_num": 3033042,
        "process": dataset_tool.create_from_lmdb # Function to convert download to tfrecords
    }),
    "cityscapes": EasyDict({
        "name": "Cityscapes", # Dataset name for logging 
        "filename": "cityscapes.zip",
        "url": "https://drive.google.com/uc?id=1t9Qhxm0iHFd3k-xTYEbKosSx_DkyoLLJ",
        "md5": "953d231046275120dc1f73a5aebc9087",
        "ratio": 0.5,        
        "size": 2,
        "shards": 16,
        "img_num": 25000
    }),
    "clevr": EasyDict({
        "name": "CLEVR", # Dataset name for logging 
        "filename": "clevr.zip",
        "url": "https://drive.google.com/uc?id=1lY4JE30yk26v0MWHNpXBOMzltufUcTXj",
        "md5": "3040bb20a29cd2f0e1e9231aebddf2a1",
        "size": 6,
        "ratio": 0.75,
        "shards": 5,
        "img_num": 100000
        ##########################################################################################
        # Currently, we download preprocessed TFrecords of CLEVR images with image ratio 0.75.
        # To process instead the dataset from scratch (with the original image ratio of 320/480), add the following:
        # "filename": "CLEVR_v1.0.zip",
        # "size": 18,
        # "dir": "CLEVR_v1.0/images", # Image directory to process while turning into tfrecords
        # "url": "https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0.zip",
        # "md5": "b11922020e72d0cd9154779b2d3d07d2",
        # "process": dataset_tool.create_from_imgs # Function to convert download to tfrecords
    })    
}

formats_catalog = {
    "png": lambda tfdir, imgdir, **kwargs: dataset_tool.create_from_imgs(tfdir, imgdir, format = "png", **kwargs),
    "jpg": lambda tfdir, imgdir, **kwargs: dataset_tool.create_from_imgs(tfdir, imgdir, format = "jpg", **kwargs),
    "npy": dataset_tool.create_from_npy, 
    "hdf5": dataset_tool.create_from_hdf5, 
    "tfds": dataset_tool.create_from_tfds, 
    "lmdb": dataset_tool.create_from_lmdb
}

def mkdir(d):
    if not os.path.exists(d):
        try:
            os.makedirs(d)
        except:
            pass

def verify_md5(filename, md5):
    print("Verify MD5 for {}...".format(filename))
    with open(filename, "rb") as f:
        new_md5 = hashlib.md5(f.read()).hexdigest()
    result = md5 == new_md5
    if result:
        print(misc.bold("MD5 matches!"))
    else:
        print("MD5 doesn't match. Will redownload the file.")
    return result

def is_unzipped(zip, dir):
    with zipfile.ZipFile(zip) as zf:
        archive = zf.namelist()
    all_exist = all(os.path.exists("{}/{}".format(dir, file)) for file in archive)
    return all_exist

def unzip(zip, dir):
    with zipfile.ZipFile(zip) as zf:
        for member in tqdm.tqdm(zf.infolist(), desc = "Extracting "):
            try:
                zf.extract(member, dir)
            except zipfile.error as e:
                pass

def get_path(url, dir = None, path = None):
    if path is None:
        path = url.split("/")[-1]
    if dir is not None:
        path = "{}/{}".format(dir, path)
    return path

def download_file(url, path, block_sz = 8192):
    if "drive.google.com" in url:
        gdown.download(url, path)
    else:
        u = urllib.request.urlopen(url)
        with open(path, "wb") as f:
            fsize = int(u.info().get_all("Content-Length")[0])
            print("Downloading: %s Bytes: %s" % (path, fsize))

            curr = 0
            while True:
                buffer = u.read(block_sz)
                if not buffer:
                    break
                curr += len(buffer)
                f.write(buffer)
                
                status  = r"%10d  [%3.2f%%]" % (curr, curr * 100. / fsize)
                status += chr(8) * (len(status) + 1)
                print(status, end = "", flush = True)

def prepare(tasks, data_dir, shards_num = 1, max_images = None, 
        ratio = 1.0, images_dir = None, format = None): # Options for custom dataset
    mkdir(data_dir)
    for task in tasks:
        # If task not in catalog, create custom task configuration
        c = catalog.get(task, EasyDict({
            "local": True,
            "name": task,
            "dir": images_dir,
            "ratio": ratio,
            "process": formats_catalog.get(format)
        }))

        dirname = "{}/{}".format(data_dir, task)
        mkdir(dirname)

        # try:
        print(misc.bold("Preparing the {} dataset...".format(c.name)))
        
        if "local" not in c:
            fname = "{}/{}".format(dirname, c.filename)
            download = not ((os.path.exists(fname) and verify_md5(fname, c.md5)))
            path = get_path(c.url, dirname, path = c.filename)

            if download:
                print(misc.bold("Downloading the data ({} GB)...".format(c.size)))
                download_file(c.url, path)
                # print(misc.bold("Completed downloading {}".format(c.name)))
            
            if path.endswith(".zip"):
                if not is_unzipped(path, dirname):
                    print(misc.bold("Unzipping {}...".format(path)))
                    unzip(path, dirname)
                    # print(misc.bold("Completed unzipping {}".format(path)))

        if "process" in c:
            imgdir = images_dir if "local" in c else ("{}/{}".format(dirname, c.dir))
            shards_num = c.shards if max_images is None else shards_num
            c.process(dirname, imgdir, ratio = c.ratio, 
                shards_num = shards_num, max_imgs = max_images)

        print(misc.bcolored("Completed preparations for {}!".format(c.name), "blue"))
        # except:
        #     print(misc.bcolored("Had an error in preparing the {} dataset. Will move on.".format(c.name), "red"))
        #     print(sys.exc_info())

def run_cmdline(argv):
    parser = argparse.ArgumentParser(prog = argv[0], description = "Download and prepare data for the GANformer.")
    parser.add_argument("--data-dir",       help = "Directory of created dataset", default = "datasets", type = str)
    parser.add_argument("--shards-num",     help = "Number of shards to split each dataset to (optional)", default = 1, type = int)
    parser.add_argument("--max-images",     help = "Maximum number of images to have in the dataset (optional). Use to reduce the produced tfrecords file size", default = None, type = int)
    # Default tasks
    parser.add_argument("--clevr",          help = "Prepare the CLEVR dataset (6.41GB download, 31GB tfrecords, 100k images)", dest = "tasks", action = "append_const", const = "clevr")
    parser.add_argument("--bedrooms",       help = "Prepare the LSUN-bedrooms dataset (42.8GB download, up to 480GB tfrecords, 3M images)", dest = "tasks", action = "append_const", const = "bedrooms")
    parser.add_argument("--ffhq",           help = "Prepare the FFHQ dataset (13GB download, 13GB tfrecords, 70k images)", dest = "tasks", action = "append_const", const = "ffhq")
    parser.add_argument("--cityscapes",     help = "Prepare the cityscapes dataset (1.8GB download, 8GB tfrecords, 25k images)", dest = "tasks", action = "append_const", const = "cityscapes")
    # Create a new task with custom images
    parser.add_argument("--task",           help = "New dataset name", type = str, dest = "tasks", action = "append")
    parser.add_argument("--images-dir",     help = "Provide source image directory to convert into tfrecords (will be searched recursively)", default = None, type = str)
    parser.add_argument("--format",         help = "Images format", default = None, choices = ["png", "jpg", "npy", "hdf5", "tfds", "lmdb"], type = str)
    parser.add_argument("--ratio",          help = "Images height/width", default = 1.0, type = float)

    args = parser.parse_args()
    if not args.tasks:
        misc.error("No tasks specified. Please see '-h' for help.")
    if args.max_images is not None and args.max_images<50000:
        misc.log("Warning: max-images is set to {}. We recommend setting it at least to 50,000 to allow statistically correct computation of the FID-50k metric.".format(args.max_images), "red")

    prepare(**vars(args))

if __name__ == "__main__":
    run_cmdline(sys.argv)
