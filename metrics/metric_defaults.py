# Default metric definitions
from dnnlib import EasyDict

metric_defaults = EasyDict([(args.name, args) for args in [
    EasyDict(name = "fid",       func_name = "metrics.frechet_inception_distance.FID", num_imgs = 1000, minibatch_per_gpu = 8),
    EasyDict(name = "is5k",      func_name = "metrics.inception_score.IS",             num_imgs = 5000, num_splits = 10, minibatch_per_gpu = 8),
    EasyDict(name = "is50k",     func_name = "metrics.inception_score.IS",             num_imgs = 50000, num_splits = 10, minibatch_per_gpu = 8),
    EasyDict(name = "ppl_zfull", func_name = "metrics.perceptual_path_length.PPL",     num_samples = 50000, epsilon = 1e-4, space = "z", sampling = "full", crop = True, minibatch_per_gpu = 4, Gs_overrides = dict(dtype = "float32", mapping_dtype = "float32")),
    EasyDict(name = "ppl_wfull", func_name = "metrics.perceptual_path_length.PPL",     num_samples = 50000, epsilon = 1e-4, space = "w", sampling = "full", crop = True, minibatch_per_gpu = 4, Gs_overrides = dict(dtype = "float32", mapping_dtype = "float32")),
    EasyDict(name = "ppl_zend",  func_name = "metrics.perceptual_path_length.PPL",     num_samples = 50000, epsilon = 1e-4, space = "z", sampling = "end", crop = True, minibatch_per_gpu = 4, Gs_overrides = dict(dtype = "float32", mapping_dtype = "float32")),
    EasyDict(name = "ppl_wend",  func_name = "metrics.perceptual_path_length.PPL",     num_samples = 50000, epsilon = 1e-4, space = "w", sampling = "end", crop = True, minibatch_per_gpu = 4, Gs_overrides = dict(dtype = "float32", mapping_dtype = "float32")),
    EasyDict(name = "ppl2_wend", func_name = "metrics.perceptual_path_length.PPL",     num_samples = 50000, epsilon = 1e-4, space = "w", sampling = "end", crop = False, minibatch_per_gpu = 4, Gs_overrides = dict(dtype = "float32", mapping_dtype = "float32")),
    EasyDict(name = "ls",        func_name = "metrics.linear_separability.LS",         num_samples = 200000, num_keep = 100000, attrib_indices = range(40), minibatch_per_gpu = 4),
    EasyDict(name = "pr5k3",     func_name = "metrics.precision_recall.PR",            num_imgs = 5000, nhood_size = 3, minibatch_per_gpu = 8, row_batch_size = 5000, col_batch_size = 5000),
    EasyDict(name = "pr50k3",    func_name = "metrics.precision_recall.PR",            num_imgs = 50000, nhood_size = 3, minibatch_per_gpu = 8, row_batch_size = 10000, col_batch_size = 10000),
]])
