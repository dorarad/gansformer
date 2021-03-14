# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

import os
import sys
import tqdm
import time
import json
import glob
import gdown
import urllib
import zipfile

import argparse
import numpy as np
from training import misc 
import dataset_tool

urls = {
   "ffhq": {
        "url": "https://drive.google.com/uc?id=1Oo-43djEakn6AzeZidfmHWYFNfMen8ZP", 
        "path": "ffhq/ffhq-r08.tfrecords1of1", 
        "size": 13766900000,  
        "md5": "74de4f07dc7bfb07c0ad4471fdac5e67",
    },
    "clevr": "https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0.zip",
    "bedrooms": "http://dl.yf.io/lsun/scenes/bedroom_train_lmdb.zip",
    "cityscapes": None # Cityscapes requires registration to download files
}

def mkdir(d):
    if not os.path.exists(d):
        try:
            os.makedirs(d)
        except:
            pass

def unzip_fn(zip, path):
    with zipfile.ZipFile(zip) as zf:
        for member in tqdm.tqdm(zf.infolist(), desc = "Extracting "):
            try:
                zf.extract(member, path)
            except zipfile.error as e:
                pass

def download_file(url, dir = None, path = None, unzip = False, drive = False, md5 = None):
    if path is None:
        path = url.split("/")[-1]
    if dir is not None:
        path = "{}/{}".format(dir, path)

    if drive:
        gdown.cached_download(url, path, md5 = md5)
    else:
        u = urllib.request.urlopen(url)
        f = open(path, "wb")
        meta = u.info()
        file_size = int(meta.get_all("Content-Length")[0])
        print("Downloading: %s Bytes: %s" % (path, file_size))

        file_size_dl = 0
        block_sz = 8192
        while True:
            buffer = u.read(block_sz)
            if not buffer:
                break

            file_size_dl += len(buffer)
            f.write(buffer)
            status = r"%10d  [%3.2f%%]" % (file_size_dl, file_size_dl * 100. / file_size)
            status = status + chr(8)*(len(status) + 1)
            print(status, end = "", flush = True)

        f.close()

    if unzip:
        unzip_fn(path, dir)

def task(task, name, size, dir, redownload, download = None, prepare = lambda: None):
    # try:
    if task:
        print(misc.bcolored("Preparing the {} dataset...".format(name), "blue"))
        if download is not None and (redownload or not os.path.exists("{}/{}".format(dir, task))):
            print(misc.bcolored("Downloading the data ({} GB)...".format(size), "blue"))
            download()
            print(misc.bold("Completed downloading {}".format(name)))
        
        prepare()
        print(misc.bold("Completed preparations for {}!".format(name)))
    # except:
    #     print(misc.bcolored("Had an error in preparing the {} dataset. Will move on.".format(name), "red"))
    #     print(sys.exc_info()[0])
    #     pass

def prepare(tasks, data_dir, redownload, shards_num = None, max_images = None):
    mkdir(data_dir)
    for task in tasks:
        mkdir("{}/{}".format(data_dir, task))

    if "ffhq" in tasks:
        task("ffhq", "FFHQ", 13, data_dir, redownload,
            download = lambda: download_file(urls["ffhq"]["url"], data_dir, drive = True,
                path = urls["ffhq"]["path"], md5 = urls["ffhq"]["md5"]))
    if "clevr" in tasks:
        shards_num = shards_num or 5
        task("clevr", "CLEVR", 18, data_dir, redownload, 
            download = lambda: download_file(urls["clevr"], "{}/clevr".format(data_dir), unzip = True),
            prepare = lambda: dataset_tool.create_from_imgs("{}/clevr".format(data_dir), "{}/clevr/CLEVR_v1.0/images".format(data_dir), 
                ratio = 0.75, shards_num = shards_num, max_imgs = max_images))
    if "bedrooms" in tasks:
        shards_num = shards_num or 32
        task("bedrooms", "LSUN-Bedrooms", 43, data_dir, redownload, 
            download = lambda: download_file(urls["bedrooms"], "{}/bedrooms".format(data_dir), unzip = True),
            prepare = lambda: dataset_tool.create_from_lmdb("{}/bedrooms".format(data_dir), 
                 "{}/bedrooms/bedroom_train_lmdb".format(data_dir), ratio = 188/256, 
                 shards_num = shards_num, max_imgs = max_images))

def _str_to_bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1", ""):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected")

def run_cmdline(argv):
    parser = argparse.ArgumentParser(prog = argv[0], description = "Download and prepare data for the GANsformer.")
    parser.add_argument("--clevr",          help = "Prepare the CLEVR dataset (18 GB)", dest = "tasks", action = "append_const", const = "clevr")
    parser.add_argument("--ffhq",           help = "Prepare the FFHQ dataset (13 GB)", dest = "tasks", action = "append_const", const = "ffhq")
    parser.add_argument("--bedrooms",       help = "Prepare the LSUN-bedrooms dataset (43 GB)", dest = "tasks", action = "append_const", const = "bedrooms")
    parser.add_argument("--data-dir",       help = "Path to download dataset", default = "datasets", type = str)
    parser.add_argument("--redownload",     help = "Download even if exists",  default = True, metavar = "BOOL", type = _str_to_bool, nargs = "?")
    parser.add_argument("--shards-num",     help = "Number of shards to split each dataset to (optional)", default = None, type = int)
    parser.add_argument("--max-images",     help = "Maximum number of images to have in the dataset (optional)", default = None, type = int)

    args = parser.parse_args()
    if not args.tasks:
        print("No tasks specified. Please see '-h' for help.")
        exit(1)

    # If the flag is specified without arguments (--redownload) set to True
    if args.redownload is None:
        args.redownload = True

    prepare(**vars(args))

if __name__ == "__main__":
    run_cmdline(sys.argv)