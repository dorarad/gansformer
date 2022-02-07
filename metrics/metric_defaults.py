# Default metric definitions
from dnnlib import EasyDict

metric_defaults = EasyDict([(args.name, args) for args in [
    EasyDict(name = "fid",       func_name = "metrics.frechet_inception_distance.FID", batch_per_gpu = 8),
    EasyDict(name = "is",        func_name = "metrics.inception_score.IS",             num_splits = 10, batch_per_gpu = 8),
    EasyDict(name = "pr",        func_name = "metrics.precision_recall.PR",            nhood_size = 3, batch_per_gpu = 8, row_batch_size = 5000, col_batch_size = 5000),
    # EasyDict(name = "ppl_zfull", func_name = "metrics.perceptual_path_length.PPL",     num_samples = 50000, epsilon = 1e-4, space = "z", sampling = "full", crop = True, batch_per_gpu = 4, Gs_overrides = dict(dtype = "float32", mapping_dtype = "float32")),
    # EasyDict(name = "ppl_wfull", func_name = "metrics.perceptual_path_length.PPL",     num_samples = 50000, epsilon = 1e-4, space = "w", sampling = "full", crop = True, batch_per_gpu = 4, Gs_overrides = dict(dtype = "float32", mapping_dtype = "float32")),
    # EasyDict(name = "ppl_zend",  func_name = "metrics.perceptual_path_length.PPL",     num_samples = 50000, epsilon = 1e-4, space = "z", sampling = "end", crop = True, batch_per_gpu = 4, Gs_overrides = dict(dtype = "float32", mapping_dtype = "float32")),
    # EasyDict(name = "ppl_wend",  func_name = "metrics.perceptual_path_length.PPL",     num_samples = 50000, epsilon = 1e-4, space = "w", sampling = "end", crop = True, batch_per_gpu = 4, Gs_overrides = dict(dtype = "float32", mapping_dtype = "float32")),
    # EasyDict(name = "ppl2_wend", func_name = "metrics.perceptual_path_length.PPL",     num_samples = 50000, epsilon = 1e-4, space = "w", sampling = "end", crop = False, batch_per_gpu = 4, Gs_overrides = dict(dtype = "float32", mapping_dtype = "float32")),
    # EasyDict(name = "ls",        func_name = "metrics.linear_separability.LS",         num_samples = 200000, num_keep = 100000, attrib_indices = range(40), batch_per_gpu = 4),
]])
