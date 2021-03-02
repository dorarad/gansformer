import argparse
import os
import sys

import dnnlib
import dnnlib.tflib as tflib

import pretrained_networks
from metrics import metric_base
from metrics.metric_defaults import metric_defaults

def run(network_pkl, metrics, dataset, data_dir, mirror_augment, paths):
    print("Evaluating metrics %s for %s..." % (",".join(metrics), network_pkl))
    tflib.init_tf()

    network_pkl = pretrained_networks.get_path_or_url(network_pkl)
    dataset_args = dnnlib.EasyDict(tfrecord_dir = dataset, shuffle_mb = 0)
    num_gpus = dnnlib.submit_config.num_gpus
    metric_group = metric_base.MetricGroup([metric_defaults[metric] for metric in metrics])
    tf_config = {
        "rnd.np_random_seed": 1000, 
        "allow_soft_placement": True, 
        "gpu_options.per_process_gpu_memory_fraction": 1.0
    }    

    metric_group.run(network_pkl, data_dir = data_dir, dataset_args = dataset_args, tf_config = tf_config,
        mirror_augment = mirror_augment, num_gpus = num_gpus, paths = paths)

def _str_to_bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

_examples = """examples:

  python %(prog)s --data-dir = ~/datasets --network = gdrive:networks/stylegan2-ffhq-config-f.pkl --metrics = fid50k,ppl_wend --dataset = ffhq --mirror-augment = true

valid metrics:

  """ + ", ".join(sorted([x for x in metric_defaults.keys()])) + """
"""

def main():
    parser = argparse.ArgumentParser(
        description = "Run StyleGAN2 metrics.",
        epilog = _examples,
        formatter_class = argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--result-dir",     help = "Root directory for run results (default: %(default)s)", default = "results", metavar = "DIR")
    parser.add_argument("--network",        help = "Network pickle filename", default = None, dest = "network_pkl")
    parser.add_argument("--metrics",        help = "Metrics to compute (default: %(default)s)", default = "fid50k", type = lambda x: x.split(","))
    parser.add_argument("--dataset",        help = "Training dataset", required = True)
    parser.add_argument("--data-dir",       help = "Dataset root directory", required = True)
    parser.add_argument("--mirror-augment", help = "Mirror augment (default: %(default)s)", default = False, type = _str_to_bool, metavar = "BOOL")
    parser.add_argument("--gpus",           help = "Number of GPUs to use", type = str, default = None)
    parser.add_argument("--paths",          help = "Image files to run evaluation on", default = None, type = str)

    args = parser.parse_args()

    if not os.path.exists(args.data_dir):
        print ("Error: dataset root directory does not exist.")
        sys.exit(1)

    kwargs = vars(args)
    sc = dnnlib.SubmitConfig()

    # set GPUs
    gpus = kwargs.pop("gpus")
    sc.num_gpus = len(gpus.split(","))
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus

    sc.submit_target = dnnlib.SubmitTarget.LOCAL
    sc.local.do_not_copy_source_files = True
    sc.run_dir_root = kwargs.pop("result_dir")
    sc.run_desc = "run-metrics"
    dnnlib.submit_run(sc, "run_metrics.run", **kwargs)

if __name__ == "__main__":
    main()
