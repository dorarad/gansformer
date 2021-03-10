import os
import sys
import requests
import html
import hashlib
import numpy as np
import threading
import queue
import time
import json
import uuid
import glob
import argparse
from collections import OrderedDict, defaultdict
import urllib2
import zipfile
from training import misc 
from dataset_tool import create_from_imgs, create_from_tfds

urls = {
   "ffhq": [{
        "file_url": "https://drive.google.com/uc?id = 1LxxgVBHWgyN8jzf8bQssgVOrTLE8Gv2v", 
        "file_path": "{}/ffhq/ffhq-r08.tfrecords1of1", 
        "file_size": 13766900000,  
        "file_md5": "74de4f07dc7bfb07c0ad4471fdac5e67",
    }],
    "clevr": "https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0.zip",
    "cityscapes": None # Cityscapes requires registration to download files
}

def download_file(url, dir = None, unzip = False):
    file_path = url.split("/")[-1]
    if dir is not None:
        file_path = "{}/{}".format(dir, file_path)

    u = urllib2.urlopen(url)
    f = open(file_path, "wb")
    meta = u.info()
    file_size = int(meta.getheaders("Content-Length")[0])
    print("Downloading: %s Bytes: %s" % (file_path, file_size))

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
        with zipfile.ZipFile(file_path, "r") as zip_file:
            zip_file.extractall(dir)

def download_drive_file(dir, session, file_spec, stats, chunk_size = 128, num_attempts = 10):
    file_path = file_spec["file_path"].format(dir)
    file_url = file_spec["file_url"]
    file_dir = os.path.dirname(file_path)
    tmp_path = file_path + ".tmp." + uuid.uuid4().hex
    if file_dir:
        os.makedirs(file_dir, exist_ok = True)

    for attempts_left in reversed(range(num_attempts)):
        data_size = 0
        try:
            # Download.
            data_md5 = hashlib.md5()
            with session.get(file_url, stream = True) as res:
                res.raise_for_status()
                with open(tmp_path, "wb") as f:
                    for chunk in res.iter_content(chunk_size = chunk_size<<10):
                        f.write(chunk)
                        data_size += len(chunk)
                        data_md5.update(chunk)
                        with stats["lock"]:
                            stats["bytes_done"] += len(chunk)

            # Validate.
            if "file_size" in file_spec and data_size != file_spec["file_size"]:
                raise IOError("Incorrect file size", file_path)
            if "file_md5" in file_spec and data_md5.hexdigest() != file_spec["file_md5"]:
                raise IOError("Incorrect file MD5", file_path)
            if "pixel_size" in file_spec or "pixel_md5" in file_spec:
                with PIL.Image.open(tmp_path) as image:
                    if "pixel_size" in file_spec and list(image.size) != file_spec["pixel_size"]:
                        raise IOError("Incorrect pixel size", file_path)
                    if "pixel_md5" in file_spec and hashlib.md5(np.array(image)).hexdigest() != file_spec["pixel_md5"]:
                        raise IOError("Incorrect pixel MD5", file_path)
            break

        except:
            with stats["lock"]:
                stats["bytes_done"] -= data_size

            # Handle known failure cases.
            if data_size > 0 and data_size < 8192:
                with open(tmp_path, "rb") as f:
                    data = f.read()
                data_str = data.decode("utf-8")

                # Google Drive virus checker nag.
                links = [html.unescape(link) for link in data_str.split('"') if "export = download" in link]
                if len(links) == 1:
                    if attempts_left:
                        file_url = requests.compat.urljoin(file_url, links[0])
                        continue

                # Google Drive quota exceeded.
                if "Google Drive - Quota exceeded" in data_str:
                    if not attempts_left:
                        raise IOError("Google Drive download quota exceeded -- please try again later")

            # Last attempt => raise error.
            if not attempts_left:
                raise

    # Rename temp file to the correct name.
    os.replace(tmp_path, file_path) # atomic
    with stats["lock"]:
        stats["files_done"] += 1

    # Attempt to clean up any leftover temps.
    for filename in glob.glob(file_path + ".tmp.*"):
        try:
            os.remove(filename)
        except:
            pass


def choose_bytes_unit(num_bytes):
    b = int(np.rint(num_bytes))
    if b < (100 << 0): return "B", (1 << 0)
    if b < (100 << 10): return "kB", (1 << 10)
    if b < (100 << 20): return "MB", (1 << 20)
    if b < (100 << 30): return "GB", (1 << 30)
    return "TB", (1 << 40)

def format_time(seconds):
    s = int(np.rint(seconds))
    if s < 60: return "%ds" % s
    if s < 60 * 60: return "%dm %02ds" % (s // 60, s % 60)
    if s < 24 * 60 * 60: return "%dh %02dm" % (s // (60 * 60), (s // 60) % 60)
    if s < 100 * 24 * 60 * 60: return "%dd %02dh" % (s // (24 * 60 * 60), (s // (60 * 60)) % 24)
    return ">100d"

#----------------------------------------------------------------------------

def download_drive_files(dir, file_specs, num_threads = 32, status_delay = 0.2, timing_window = 50, **download_kwargs):
    # Determine which files to download
    done_specs = {spec["file_path"]: spec for spec in file_specs if os.path.isfile(spec["file_path"])}
    missing_specs = [spec for spec in file_specs if spec["file_path"] not in done_specs]
    files_total = len(file_specs)
    bytes_total = sum(spec["file_size"] for spec in file_specs)
    stats = {"files_done": len(done_specs), "bytes_done": sum(spec["file_size"] for spec in done_specs.values()), "lock": threading.Lock()}
    if len(done_specs) == files_total:
        print("All files already downloaded -- skipping.")
        return

    # Launch worker threads.
    spec_queue = queue.Queue()
    exception_queue = queue.Queue()
    for spec in missing_specs:
        spec_queue.put(spec)
    thread_kwargs = {"dir": dir, "spec_queue": spec_queue, "exception_queue": exception_queue, "stats": stats, "download_kwargs": download_kwargs}
    for _thread_idx in range(min(num_threads, len(missing_specs))):
        threading.Thread(target = _download_thread, kwargs = thread_kwargs, daemon = True).start()

    # Monitor status until done.
    bytes_unit, bytes_div = choose_bytes_unit(bytes_total)
    spinner = "/-\\|"
    timing = []
    while True:
        with stats["lock"]:
            files_done = stats["files_done"]
            bytes_done = stats["bytes_done"]
        spinner = spinner[1:] + spinner[:1]
        timing = timing[max(len(timing) - timing_window + 1, 0):] + [(time.time(), bytes_done)]
        bandwidth = max((timing[-1][1] - timing[0][1]) / max(timing[-1][0] - timing[0][0], 1e-8), 0)
        bandwidth_unit, bandwidth_div = choose_bytes_unit(bandwidth)
        eta = format_time((bytes_total - bytes_done) / max(bandwidth, 1))

        print("\r%s %6.2f%% done  %d/%d files  %-13s  %-10s  ETA: %-7s " % (
            spinner[0],
            bytes_done / bytes_total * 100,
            files_done, files_total,
            "%.2f/%.2f %s" % (bytes_done / bytes_div, bytes_total / bytes_div, bytes_unit),
            "%.2f %s/s" % (bandwidth / bandwidth_div, bandwidth_unit),
            "done" if bytes_total == bytes_done else "..." if len(timing) < timing_window or bandwidth == 0 else eta,
        ), end = "", flush = True)

        if files_done == files_total:
            print()
            break

        try:
            exc_info = exception_queue.get(timeout = status_delay)
            raise exc_info[1].with_traceback(exc_info[2])
        except queue.Empty:
            pass

def _download_thread(dir, spec_queue, exception_queue, stats, download_kwargs):
    with requests.Session() as session:
        while not spec_queue.empty():
            spec = spec_queue.get()
            try:
                download_drive_file(dir, session, spec, stats, **download_kwargs)
            except:
                exception_queue.put(sys.exc_info())

def task(task, name, size, dir, redownload, download = None, prepare = lambda: None):
    if task:
        print(misc.bcolored("Preparing the {} dataset...".format(name), "blue"))
        if download is not None and (redownload or not path.exists("{}/{}".format(dir, task))):
            print(misc.bold("Downloading the data ({} GB)...".format(size), "blue"))
            download()
            print(misc("Completed downloading {}".format(name)))
        
        prepare()
        print(misc.bold("Completed preparations for {}!".format(name)))
    except:
        print(misc())
        pass

def prepare(tasks, max_images, data_dir, redownload):
    if "ffhq" in tasks:
        task("ffhq", "FFHQ", 13, data_dir, redownload, 
            download = lambda: download_drive_files(data_dir, urls["ffhq"]))
    if "clevr" in tasks:
        task("clevr", "CLEVR", 18, data_dir, redownload, 
            download = lambda: download_file(urls["clevr"], data_dir, unzip = True),
            prepare = lambda: create_from_imgs("{}/clevr".format(data_dir), "{}/TODO".format(data_dir), 
                ratio = 0.75))
    if "bedrooms" in tasks:
        task("bedrooms", "LSUN-Bedrooms", 43, data_dir, redownload, 
            download = lambda: download_file(urls["clevr"], data_dir, unzip = True),
            prepare = lambda: create_from_tfds("{}/bedrooms".format(data_dir), "lsun/bedroom", 
                ratio = 188/256))

def _str_to_bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1", ""):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

def run_cmdline(argv):
    parser = argparse.ArgumentParser(prog = argv[0], description = "Download and prepare data for the GANsformer.")
    parser.add_argument("--clevr",          help = "Prepare the CLEVR dataset (18 GB)", dest = "tasks", action = "append_const", const = "clevr")
    parser.add_argument("--ffhq",           help = "Prepare the FFHQ dataset (13 GB)", dest = "tasks", action = "append_const", const = "ffhq")
    parser.add_argument("--bedrooms",       help = "Prepare the LSUN-bedrooms dataset (43 GB)", dest = "tasks", action = "append_const", const = "bedrooms")
    parser.add_argument("--data-dir",       help = "Path to download dataset", default = "datasets", type = "str")
    parser.add_argument("--redownload",     help = "Download even if exists",  default = True, metavar = "BOOL", type = _str_to_bool)

    args = parser.parse_args()
    if not args.tasks:
        print("No tasks specified. Please see '-h' for help.")
        exit(1)
    prepare(**vars(args))

#----------------------------------------------------------------------------

if __name__ == "__main__":
    run_cmdline(sys.argv)