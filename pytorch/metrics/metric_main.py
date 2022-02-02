import os
import time
import json
import torch
import dnnlib
from training import misc

from . import metric_utils
from . import frechet_inception_distance
from . import kernel_inception_distance
from . import precision_recall
from . import perceptual_path_length
from . import inception_score

#----------------------------------------------------------------------------

_metric_dict = dict() # name => fn

def register_metric(fn):
    assert callable(fn)
    _metric_dict[fn.__name__[1:]] = fn
    return fn

def is_valid_metric(metric):
    return metric in _metric_dict

def list_valid_metrics():
    return list(_metric_dict.keys())

#----------------------------------------------------------------------------

def compute_metric(metric, **kwargs): # See metric_utils.MetricOptions for the full list of arguments
    assert is_valid_metric(metric)
    opts = metric_utils.MetricOptions(**kwargs)
    opts.dataset_args.update(max_items = None)

    # Compute
    start_time = time.time()
    results = _metric_dict[metric](opts)
    total_time = time.time() - start_time

    # Broadcast results
    for key, value in list(results.items()):
        if opts.num_gpus > 1:
            value = torch.as_tensor(value, dtype = torch.float64, device = opts.device)
            torch.distributed.broadcast(tensor = value, src = 0)
            value = float(value.cpu())
        results[key] = value

    # Decorate with metadata
    return dnnlib.EasyDict(
        results         = dnnlib.EasyDict(results),
        metric          = metric,
        total_time      = total_time,
        total_time_str  = dnnlib.util.format_time(total_time),
        num_gpus        = opts.num_gpus,
    )

#----------------------------------------------------------------------------

def print_results(jsonl_line):
    print(" " * 100 + "\r")
    if jsonl_line["snapshot_pkl"] is None:
        network_name = "None"
    else:
        network_name = os.path.splitext(os.path.basename(jsonl_line["snapshot_pkl"]))[0]
    if len(network_name) > 29:
        network_name = "..." + network_name[-26:]

    result_str = "%-30s" % network_name
    result_str += " time %-12s" % dnnlib.util.format_time(jsonl_line["total_time"])
    nums = ""
    for res, value in jsonl_line["results"].items():
        nums += f" {res} {value:10.4f}"
    nums = misc.bcolored(nums, "blue")
    result_str += nums
    print(result_str)

def report_metric(result_dict, run_dir = None, snapshot_pkl = None):
    metric = result_dict['metric']
    assert is_valid_metric(metric)
    if run_dir is not None and snapshot_pkl is not None:
        snapshot_pkl = os.path.relpath(snapshot_pkl, run_dir)

    result = dict(result_dict, snapshot_pkl = snapshot_pkl, timestamp = time.time())
    jsonl_line = json.dumps(result)
    print_results(result)

    if run_dir is not None and os.path.isdir(run_dir):
        with open(os.path.join(run_dir, f'metric-{metric}.jsonl'), 'at') as f:
            f.write(jsonl_line + '\n')

#----------------------------------------------------------------------------

@register_metric
def _fid(opts):
    fid = frechet_inception_distance.compute_fid(opts)
    return {f"fid{opts.max_items // 1000}k": fid}

@register_metric
def _kid(opts):
    kid = kernel_inception_distance.compute_kid(opts, num_subsets = 100, max_subset_size = 1000)
    return {f"kid{opts.max_items // 1000}k": kid}

@register_metric
def _pr(opts):
    precision, recall = precision_recall.compute_pr(opts, nhood_size = 3, row_batch_size = 5000, col_batch_size = 5000)
    return {f"pr{opts.max_items // 1000}k_precision": precision, 
            f"pr{opts.max_items // 1000}k_recall": recall}

@register_metric
def _ppl_zfull(opts):
    ppl = perceptual_path_length.compute_ppl(opts, epsilon = 1e-4, space='z', sampling='full', crop = False, batch_size = 2)
    return {"ppl_zfull":  ppl}

@register_metric
def _ppl_wfull(opts):
    ppl = perceptual_path_length.compute_ppl(opts, epsilon = 1e-4, space='w', sampling='full', crop = False, batch_size = 2)
    return {"ppl_wfull":  ppl}

@register_metric
def _ppl_zend(opts):
    ppl = perceptual_path_length.compute_ppl(opts, epsilon = 1e-4, space='z', sampling='end', crop = False, batch_size = 2)
    return {"ppl_zend":  ppl}

@register_metric
def _ppl_wend(opts):
    ppl = perceptual_path_length.compute_ppl(opts, epsilon = 1e-4, space='w', sampling='end', crop = False, batch_size = 2)
    return {"ppl_wend":  ppl}

@register_metric
def _is(opts):
    mean, std = inception_score.compute_is(opts, num_splits = 10)
    return {f"is{opts.max_items // 1000}k_mean": mean, 
            f"is{opts.max_items // 1000}k_std": std}
