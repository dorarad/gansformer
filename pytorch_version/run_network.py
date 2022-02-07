# Train a GANformer model (pytorch version)

import os
import re
import json
import tempfile
# import torch
from training import misc

import dnnlib
from dnnlib import EasyDict

import argparse
import glob
import sys 
import loader

# Conditional set: if property is not None, then assign d[name] := prop
# for every d in a set of dictionaries
def cset(dicts, name, prop):
    if not isinstance(dicts, list):
        dicts = [dicts]
    if prop is not None:
        for d in dicts:
            d[name] = prop

# Conditional set: if dict[name] is not populated from the command line, then assign dict[name] := prop
def nset(args, name, prop):
    flag = f"--{name.replace('_', '-')}"
    if flag not in sys.argv:
        args[name] = prop

def set_net(name, subnets, lr, reg_interval):
    net_config = EasyDict(class_name = f"training.networks.{name}", reg_interval = reg_interval)
    net_config.opt_args = EasyDict(class_name = "torch.optim.Adam", lr = lr, betas = [0, 0.99], eps = 1e-8)
    for subnet in subnets:
        net_config[f"{subnet}_kwargs"] = EasyDict()
    return net_config

# Setup configuration based on command line
def setup_config(run_dir, **args):
    args  = EasyDict(args)               # command-line options
    train = EasyDict(run_dir = run_dir)  # training loop options
    vis   = EasyDict(run_dir = run_dir)  # visualization loop options

    if args.reload:
        config_fn = os.path.join(run_dir, "training_options.json")
        if os.path.exists(config_fn):
            # Load config form the experiment existing file (and so ignore command-line arguments)
            with open(config_fn, "rt") as f:
                config = json.load(f)
            return config
        misc.log(f"Warning: --reload is set for a new experiment {args.expname}," + 
                 f" but configuration file to reload from {config_fn} doesn't exist.", "red")

    # GANformer and baselines default settings
    # ----------------------------------------------------------------------------

    if args.ganformer_default:
        task = args.dataset
        nset(args, "mirror_augment", task in ["cityscapes", "ffhq"])

        nset(args, "transformer", True)
        nset(args, "components_num", {"clevr": 8}.get(task, 16))
        nset(args, "latent_size", {"clevr": 128}.get(task, 512))

        nset(args, "normalize", "layer")
        nset(args, "integration", "mul")
        nset(args, "kmeans", True)
        nset(args, "use_pos", True)
        nset(args, "mapping_ltnt2ltnt", task != "clevr")
        nset(args, "style", task != "clevr")

        nset(args, "g_arch", "resnet")
        nset(args, "mapping_resnet", True)

        gammas = {
            "ffhq": 10, 
            "cityscapes": 20, 
            "clevr": 40, 
            "bedrooms": 100
        }
        nset(args, "gamma", gammas.get(task, 10))

    if args.baseline == "GAN":
        nset(args, "style", False)
        nset(args, "latent_stem", True)

    ## k-GAN and SAGAN  are not currently supported in the pytorch version. 
    ## See the TF version for implementation of these baselines!
    # if args.baseline == "SAGAN":
    #     nset(args, "style", False)
    #     nset(args, "latent_stem", True)
    #     nset(args, "g_img2img", 5)

    # if args.baseline == "kGAN":
    #     nset(args, "kgan", True)
    #     nset(args, "merge_layer", 5)
    #     nset(args, "merge_type", "softmax")
    #     nset(args, "components_num", 8)        

    # General setup
    # ----------------------------------------------------------------------------

     # If the flag is specified without arguments (--arg), set to True
    for arg in ["cuda_bench", "allow_tf32", "keep_samples", "style", "local_noise"]:
        if args[arg] is None:
            args[arg] = True

    if not any([args.train, args.eval, args.vis]):
        misc.log("Warning: None of --train, --eval or --vis are provided. Therefore, we only print network shapes", "red")
    for arg in ["train", "eval", "vis", "last_snapshots"]:
        cset(train, arg, args[arg])

    if args.gpus != "":
        num_gpus = len(args.gpus.split(","))
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    if not (num_gpus >= 1 and num_gpus & (num_gpus - 1) == 0):
        misc.error("Number of GPUs must be a power of two")
    args.num_gpus = num_gpus

    # CUDA settings
    for arg in ["batch_size", "batch_gpu", "allow_tf32"]:
        cset(train, arg, args[arg])
    cset(train, "cudnn_benchmark", args.cuda_bench)

    # Data setup
    # ----------------------------------------------------------------------------

    # For bedrooms, we choose the most common ratio in the 
    # dataset and crop the other images into that ratio.
    ratios = {
        "clevr": 0.75,
        "bedrooms": 188/256, 
        "cityscapes": 0.5,
        "ffhq": 1.0
    }
    args.ratio = args.ratio or ratios.get(args.dataset, 1.0)
    args.crop_ratio = 0.5 if args.resolution > 256 and args.ratio < 0.5 else None

    args.printname = args.expname
    for arg in ["total_kimg", "printname"]:
        cset(train, arg, args[arg])
    
    dataset_args = EasyDict(
        class_name     = "training.dataset.ImageFolderDataset", 
        path           = f"{args.data_dir}/{args.dataset}",
        max_items      = args.train_images_num, 
        resolution     = args.resolution,
        ratio          = args.ratio,
        mirror_augment = args.mirror_augment
    )
    dataset_args.loader_args = EasyDict(
        num_workers = args.num_threads,
        pin_memory = True, 
        prefetch_factor = 2
    )

    # Optimization setup
    # ----------------------------------------------------------------------------

    cG = set_net("Generator", ["mapping", "synthesis"], args.g_lr, 4)
    cD = set_net("Discriminator", ["mapping", "block", "epilogue"], args.d_lr, 16)
    cset([cG, cD], "crop_ratio", args.crop_ratio)

    mbstd = min(args.batch_gpu, 4) # other hyperparams behave more predictably if mbstd group size remains fixed
    cset(cD.epilogue_kwargs, "mbstd_group_size", mbstd)

    # Automatic tuning
    if args.autotune:
        batch_size = max(min(args.num_gpus * min(4096 // args.resolution, 32), 64), args.num_gpus) # keep gpu memory consumption at bay
        batch_gpu = args.batch_size // args.num_gpus
        nset(args, "batch_size", batch_size)
        nset(args, "batch_gpu", batch_gpu)

        fmap_decay = 1 if args.resolution >= 512 else 0.5 # other hyperparams behave more predictably if mbstd group size remains fixed
        lr = 0.002 if args.resolution >= 1024 else 0.0025
        gamma = 0.0002 * (args.resolution ** 2) / args.batch_size # heuristic formula

        cset([cG.synthesis_kwargs, cD], "dim_base", int(fmap_decay * 32768)) 
        nset(args, "g_lr", lr); cset(cG.opt_args, "lr", args.g_lr)
        nset(args, "d_lr", lr); cset(cD.opt_args, "lr", args.d_lr)
        nset(args, "gamma", gamma)

        train.ema_rampup = 0.05
        train.ema_kimg = batch_size * 10 / 32

    if args.batch_size % (args.batch_gpu * args.num_gpus) != 0:
        misc.error("--batch-size should be divided by --batch-gpu * 'num_gpus'")

    # Loss and regularization settings
    loss_args = EasyDict(class_name = "training.loss.StyleGAN2Loss", 
        g_loss = args.g_loss,  d_loss = args.d_loss,
        r1_gamma = args.gamma, pl_weight = args.pl_weight
    )

    # if args.fp16:
    #     cset([cG.synthesis_kwargs, cD], "num_fp16_layers", 4) # enable mixed-precision training
    #     cset([cG.synthesis_kwargs, cD], "conv_clamp", 256) # clamp activations to avoid float16 overflow

    # cset([cG.synthesis_kwargs, cD.block_args], "fp16_channels_last", args.nhwc)

    # Evaluation and visualization
    # ----------------------------------------------------------------------------

    from metrics import metric_main
    for metric in args.metrics:
        if not metric_main.is_valid_metric(metric):
            misc.error(f"Unknown metric: {metric}. The valid metrics are: {metric_main.list_valid_metrics()}")

    for arg in ["num_gpus", "metrics", "eval_images_num", "truncation_psi"]:
        cset(train, arg, args[arg])
    for arg in ["keep_samples", "num_heads"]:
        cset(vis, arg, args[arg])

    args.vis_imgs = args.vis_images
    args.vis_ltnts = args.vis_latents
    vis_types = ["imgs", "ltnts", "maps", "layer_maps", "interpolations", "noise_var", "style_mix"]
    # Set of all the set visualization types option
    vis.vis_types = list({arg for arg in vis_types if args[f"vis_{arg}"]})

    vis_args = {
        "attention": "transformer",
        "grid": "vis_grid",
        "num": "vis_num",
        "rich_num": "vis_rich_num",
        "section_size": "vis_section_size",
        "intrp_density": "interpolation_density",
        # "intrp_per_component": "interpolation_per_component",
        "alpha": "blending_alpha"
    }
    for arg, cmd_arg in vis_args.items():
        cset(vis, arg, args[cmd_arg])

    # Networks setup
    # ----------------------------------------------------------------------------

    # Networks architecture
    cset(cG.synthesis_kwargs, "architecture", args.g_arch)
    cset(cD, "architecture", args.d_arch)

    # Latent sizes
    if args.components_num > 0:
        if not args.transformer: # or args.kgan):
            misc.error("--components-num > 0 but the model is not using components. " + 
                "Add --transformer for GANformer (which uses latent components).")
        if args.latent_size % args.components_num != 0:
            misc.error(f"--latent-size ({args.latent_size}) should be divisible by --components-num (k={k})")
        args.latent_size = int(args.latent_size / args.components_num)

    cG.z_dim = cG.w_dim = args.latent_size
    cset([cG, vis], "k", args.components_num + 1) # We add a component to modulate features globally

    # Mapping network
    args.mapping_layer_dim = args.mapping_dim
    for arg in ["num_layers", "layer_dim", "resnet", "shared", "ltnt2ltnt"]:
        field = f"mapping_{arg}"
        cset(cG.mapping_kwargs, arg, args[field])

    # StyleGAN settings
    for arg in ["style", "latent_stem", "local_noise"]:
        cset(cG.synthesis_kwargs, arg, args[arg])

    # GANformer
    cset([cG.synthesis_kwargs, cG.mapping_kwargs], "transformer", args.transformer)

    # Attention related settings
    for arg in ["use_pos", "num_heads", "ltnt_gate", "attention_dropout"]:
        cset([cG.mapping_kwargs, cG.synthesis_kwargs], arg, args[arg])

    # Attention types and layers
    for arg in ["start_res", "end_res"]: # , "local_attention" , "ltnt2ltnt", "img2img", "img2ltnt"
        cset(cG.synthesis_kwargs, arg, args[f"g_{arg}"])

    # Mixing and dropout
    for arg in ["style_mixing", "component_mixing"]:
        cset(loss_args, arg, args[arg])
    cset(cG, "component_dropout", args["component_dropout"])

    # Extra transformer options
    args.norm = args.normalize
    for arg in ["norm", "integration", "img_gate", "iterative", "kmeans", "kmeans_iters"]:
        cset(cG.synthesis_kwargs, arg, args[arg])

    # Positional encoding
    # args.pos_dim = args.pos_dim or args.latent_size
    for arg in ["dim", "type", "init", "directions_num"]:
        field = f"pos_{arg}"
        cset(cG.synthesis_kwargs, field, args[field])

    # k-GAN
    # for arg in ["layer", "type", "same"]:
    #     field = "merge_{}".format(arg)
    #     cset(cG.args, field, args[field])
    # cset(cG.synthesis_kwargs, "merge", args.kgan)
    # if args.kgan and args.transformer:
        # misc.error("Either have --transformer for GANformer or --kgan for k-GAN, not both")

    config = EasyDict(train)
    config.update(cG = cG, cD = cD, loss_args = loss_args, dataset_args = dataset_args, vis_args = vis)

    # Save config file
    with open(os.path.join(run_dir, "training_options.json"), "wt") as f:
        json.dump(config, f, indent = 2)

    return config

# Setup and launching
# ----------------------------------------------------------------------------

##### Experiments management:
# Whenever we start a new experiment we store its result in a directory named 'args.expname:000'.
# When we rerun a training or evaluation command it restores the model from that directory by default.
# If we wish to restart the model training, we can set --restart and then we will store data in a new
# directory: 'args.expname:001' after the first restart, then 'args.expname:002' after the second, etc.
def setup_working_space(args):
    # Find the latest directory that matches the experiment
    exp_dir = sorted(glob.glob(f"{args.result_dir}/{args.expname}-*"))
    run_id = 0
    if len(exp_dir) > 0:
        run_id = int(exp_dir[-1].split("-")[-1])
    # If restart, then work over a new directory
    if args.restart:
        run_id += 1

    run_name = f"{args.expname}-{run_id:03d}"
    run_dir = os.path.join(args.result_dir, run_name)
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)
    return run_name, run_dir

# Resume from the specified --pretrained_pkl, or from the latest snapshot in the experiment's directory otherwise
def setup_savefile(args, run_name, run_dir, config):
    snapshot, kimg, resume = None, 0, False
    pkls = sorted(glob.glob(f"{run_dir}/network*.pkl"))
    # Load a particular snapshot is specified
    if args.pretrained_pkl is not None and args.pretrained_pkl != "None":
        # Soft links support
        if args.pretrained_pkl.startswith("gdrive"):
            if args.pretrained_pkl not in loader.pretrained_networks:
                misc.error("--pretrained_pkl {} not available in the catalog (see loader.pretrained_networks dict)")

            snapshot = args.pretrained_pkl
        else: 
            snapshot = glob.glob(args.pretrained_pkl)[0]
            if os.path.islink(snapshot):
                snapshot = os.readlink(snapshot)

        # Extract training step from the snapshot if specified
        try:
            kimg = int(snapshot.split("-")[-1].split(".")[0])
        except:
            pass

    # Find latest snapshot in the directory
    elif len(pkls) > 0:
        snapshot = pkls[-1]
        kimg = int(snapshot.split("-")[-1].split(".")[0])
        resume = True

    if snapshot:
        misc.log(f"Resuming {run_name}, from {snapshot}, kimg {kimg}", "white")
        config.resume_pkl = snapshot
        config.resume_kimg = kimg
    else:
        misc.log("Start model training from scratch", "white")

# Launch distributed processes for multi-gpu training
def subprocess_fn(rank, args, temp_dir):
    import torch
    from torch_utils import training_stats
    from torch_utils import custom_ops
    from training import training_loop

    dnnlib.util.Logger(file_name = os.path.join(args.run_dir, "log.txt"), file_mode = "a", should_flush = True)

    # Init torch.distributed
    if args.num_gpus > 1:
        init_file = os.path.abspath(os.path.join(temp_dir, ".torch_distributed_init"))
        if os.name == "nt":
            init_method = "file:///" + init_file.replace("\\", "/")
            torch.distributed.init_process_group(backend = "gloo", init_method = init_method, rank = rank, world_size = args.num_gpus)
        else:
            init_method = f"file://{init_file}"
            torch.distributed.init_process_group(backend = "nccl", init_method = init_method, rank = rank, world_size = args.num_gpus)

    # Init torch_utils
    sync_device = torch.device("cuda", rank) if args.num_gpus > 1 else None
    training_stats.init_multiprocessing(rank = rank, sync_device = sync_device)
    if rank != 0:
        custom_ops.verbosity = "none"

    # Execute training loop
    training_loop.training_loop(rank = rank, **args)

# Command line
# ----------------------------------------------------------------------------

 # Launch processes
def launch_processes(config):
    import torch

    torch.multiprocessing.set_start_method("spawn")
    with tempfile.TemporaryDirectory() as temp_dir:
        if config.num_gpus == 1:
            subprocess_fn(rank = 0, args = config, temp_dir = temp_dir)
        else:
            torch.multiprocessing.spawn(fn = subprocess_fn, args = (config, temp_dir), nprocs = config.num_gpus)

def _str_to_bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1", ""):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Error: Boolean value expected")

def _parse_comma_sep(s):
    if s is None or s.lower() == "none" or s == "":
        return []
    return s.split(",")

def main():
    parser = argparse.ArgumentParser(description = "Train the GANformer")

    # Framework
    # ------------------------------------------------------------------------------------------------------
    parser.add_argument("--expname",            help = "Experiment name", default = "exp", type = str)
    parser.add_argument("--train",              help = "Train mode (default: False)", default = None, action = "store_true")
    parser.add_argument("--eval",               help = "Evaluation mode (default: False)", default = None, action = "store_true")
    parser.add_argument("--vis",                help = "Visualization mode (default: False)", default = None, action = "store_true")
    parser.add_argument("--gpus",               help = "Comma-separated list of GPUs to be used (default: %(default)s)", default = "0", type = str)

    ## Default configurations
    parser.add_argument("--ganformer-default",  help = "Select a default GANformer configuration, either pretrained (default) or from scratch (with --pretrained-pkl None)", default = None, action = "store_true")
    parser.add_argument("--baseline",           help = "Use a baseline model configuration", default = None, choices = ["GAN", "StyleGAN2"], type = str) # , "kGAN", "SAGAN"

    ## Resumption
    parser.add_argument("--pretrained-pkl",     help = "Filename for a snapshot to resume (optional)", default = None, type = str)
    parser.add_argument("--restart",            help = "Restart training from scratch", default = False, action = "store_true")
    parser.add_argument("--reload",             help = "Reload options from the original experiment configuration file. " +
                                                       "If False, uses the command line arguments when resuming training (default: %(default)s)", default = False, action = "store_true")
    parser.add_argument("--last-snapshots",     help = "Number of last snapshots to save. -1 for all (default: 10)", default = None, type = int)

    ## Dataset
    parser.add_argument("--data-dir",           help = "Datasets root directory (default: %(default)s)", default = "datasets", metavar = "DIR")
    parser.add_argument("--dataset",            help = "Training dataset name (subdirectory of data-dir)", required = True)
    parser.add_argument("--ratio",              help = "Image height/width ratio in the dataset", default = 1.0, type = float)
    parser.add_argument("--resolution",         help = "Training resolution", default = 256, type = int)
    parser.add_argument("--num-threads",        help = "Number of input processing threads (default: %(default)s)", default = 4, type = int)
    parser.add_argument("--mirror-augment",     help = "Perform horizontal flip augmentation for the data (default: %(default)s)", default = None, action = "store_true")
    parser.add_argument("--train-images-num",   help = "Maximum number of images to train on. If not specified, train on the whole dataset.", default = None, type = int)

    ## CUDA
    # parser.add_argument("--fp16",             help = "Enable mixed-precision training", default = False, action = "store_true")
    # parser.add_argument("--nhwc",             help = "Use NHWC memory format with FP16", default = False, action = "store_true")
    parser.add_argument("--cuda-bench",         help = "Enable cuDNN benchmarking", default = True, metavar = "BOOL", type = _str_to_bool, nargs = "?")
    parser.add_argument("--allow-tf32",         help = "Allow PyTorch to use TF32 internally", default = True, metavar = "BOOL", type = _str_to_bool, nargs = "?")

    ## Training
    parser.add_argument("--batch-size",         help = "Global batch size (optimization step) (default: %(default)s)", default = 32, type = int)
    parser.add_argument("--batch-gpu",          help = "Batch size per GPU, gradients will be accumulated to match batch-size (default: %(default)s)", default = 4, type = int)
    parser.add_argument("--total-kimg",         help = "Training length in thousands of images (default: %(default)s)", metavar = "KIMG", default = 25000, type = int)
    parser.add_argument("--gamma",              help = "R1 regularization weight (default: %(default)s)", default = 10, type = float)
    parser.add_argument("--g-lr",               help = "Generator learning rate (default: %(default)s)", default = 0.002, type = float)
    parser.add_argument("--d-lr",               help = "Discriminator learning rate (default: %(default)s)", default = 0.002, type = float)
    parser.add_argument("--autotune",           help = "Set training hyper-parameters automatically using a heuristic", default = False, action = "store_true")

    ## Logging and evaluation
    parser.add_argument("--result-dir",         help = "Root directory for experiments (default: %(default)s)", default = "results", metavar = "DIR")
    parser.add_argument("--metrics",            help = "Comma-separated list of metrics or none (default: %(default)s)", default = "fid", type = _parse_comma_sep)
    parser.add_argument("--truncation-psi",     help = "Truncation Psi to be used in producing sample images " +
                                                       "(used only for visualizations, _not used_ in training or for computing metrics) (default: %(default)s)", default = 0.75, type = float)
    parser.add_argument("--keep-samples",       help = "Keep all prior samples during training, or if False, just the most recent ones (default: %(default)s)", default = True, metavar = "BOOL", type = _str_to_bool, nargs = "?")
    parser.add_argument("--eval-images-num",    help = "Number of images to evaluate metrics on (default: 50,000)", default = None, type = int)

    ## Visualization
    parser.add_argument("--vis-images",         help = "Save image samples", default = None, action = "store_true")
    parser.add_argument("--vis-latents",        help = "Save latent vectors", default = None, action = "store_true")
    parser.add_argument("--vis-maps",           help = "Save attention maps (for GANformer only)", default = None, action = "store_true")
    parser.add_argument("--vis-layer-maps",     help = "Save attention maps for all layers (for GANformer only)", default = None, action = "store_true")
    parser.add_argument("--vis-interpolations", help = "Create latent interpolations", default = None, action = "store_true")
    parser.add_argument("--vis-noise-var",      help = "Create noise variation visualization", default = None, action = "store_true")
    parser.add_argument("--vis-style-mix",      help = "Create style mixing visualization", default = None, action = "store_true")

    parser.add_argument("--vis-grid",               help = "Whether to save the samples in one large grid files (default: True in training)", default = None, action = "store_true")
    parser.add_argument("--vis-num",                help = "Number of images for which visualization will be created (default: grid-size/100 in train/eval)", default = None, type = int)
    parser.add_argument("--vis-rich-num",           help = "Number of samples for which richer visualizations will be created (default: 5)", default = None, type = int)
    parser.add_argument("--vis-section-size",       help = "Visualization section size to process at one (section-size <= vis-num) for memory footprint (default: 100)", default = None, type = int)
    parser.add_argument("--blending-alpha",         help = "Proportion for generated images and attention maps blends (default: 0.3)", default = None, type = float)
    parser.add_argument("--interpolation-density",  help = "Number of samples in between two end points of an interpolation (default: 8)", default = None, type = int)

    # Model
    # ------------------------------------------------------------------------------------------------------

    ## General architecture
    parser.add_argument("--g-arch",             help = "Generator architecture type (default: skip)", default = None, choices = ["orig", "skip", "resnet"], type = str)
    parser.add_argument("--d-arch",             help = "Discriminator architecture type (default: resnet)", default = None, choices = ["orig", "skip", "resnet"], type = str)
    parser.add_argument("--tanh",               help = "tanh on generator output (default: False)", default = None, action = "store_true")

    # Mapping network
    parser.add_argument("--mapping-num-layers", help = "Number of mapping layers (default: 8)", default = None, type = int)
    parser.add_argument("--mapping-dim",        help = "Mapping layers dimension (default: latent_size)", default = None, type = int)
    parser.add_argument("--mapping-resnet",     help = "Use resent connections in mapping layers (default: False)", default = None, action = "store_true")
    parser.add_argument("--mapping-shared",     help = "Perform one shared mapping to all latent components concatenated together using the set dimension (default: disabled)", default = None, action = "store_true")

    # Loss
    parser.add_argument("--g-loss",             help = "Generator loss type (default: %(default)s)", default = "logistic_ns", choices = ["logistic", "logistic_ns", "hinge", "wgan"], type = str)
    parser.add_argument("--d-loss",             help = "Discriminator loss type (default: %(default)s)", default = "logistic", choices = ["wgan", "logistic", "hinge"], type = str)
    parser.add_argument("--pl-weight",          help = "Generator regularization weight (default: %(default)s)", default = 0.0, type = float)
    # parser.add_argument("--d-reg",            help = "Discriminator regularization type (default: %(default)s)", default = "r1", choices = ["non", "gp", "r1", "r2"], type = str)
    # --gamma effectively functions as discriminator regularization weight

    # Mixing and dropout
    parser.add_argument("--style-mixing",       help = "Style mixing (layerwise) probability (default: %(default)s)", default = 0.9, type = float)
    parser.add_argument("--component-mixing",   help = "Component mixing (objectwise) probability (default: %(default)s)", default = 0.0, type = float)
    parser.add_argument("--component-dropout",  help = "Component dropout (default: %(default)s)", default = 0.0, type = float)
    parser.add_argument("--attention-dropout",  help = "Attention dropout (default: 0.12)", default = None, type = float)


    # StyleGAN additions
    parser.add_argument("--style",              help = "Global style modulation (default: %(default)s)", default = True, metavar = "BOOL", type = _str_to_bool, nargs = "?")
    parser.add_argument("--latent-stem",        help = "Input latent through the generator stem grid (default: False)", default = None, action = "store_true")
    parser.add_argument("--local-noise",        help = "Add stochastic local noise each layer (default: %(default)s)", default = True, metavar = "BOOL", type = _str_to_bool, nargs = "?")

    ## GANformer
    parser.add_argument("--transformer",        help = "Add transformer layers to the generator: top-down latents-to-image (default: False)", default = None, action = "store_true")
    parser.add_argument("--latent-size",        help = "Latent size, summing the dimension of all components (default: %(default)s)", default = 512, type = int)
    parser.add_argument("--components-num",     help = "Local components number. Each component has latent dimension of 'latent-size / (components-num + 1)'. " +
                                                       "as we add one additional global component. 0 for StyleGAN since it has one global latent vector (default: %(default)s)", default = 0, type = int)
    parser.add_argument("--num-heads",          help = "Number of attention heads (default: %(default)s)", default = 1, type = int)
    parser.add_argument("--normalize",          help = "Feature normalization type (optional)", default = None, choices = ["batch", "instance", "layer"], type = str)
    parser.add_argument("--integration",        help = "Feature integration type: additive, multiplicative or both (default: %(default)s)", default = "add", choices = ["add", "mul", "both"], type = str)

    # Generator attention layers
    # Transformer resolution layers
    parser.add_argument("--g-start-res",        help = "Transformer minimum generator resolution (logarithmic): first layer in which transformer will be applied (default: %(default)s)", default = 0, type = int)
    parser.add_argument("--g-end-res",          help = "Transformer maximum generator resolution (logarithmic): last layer in which transformer will be applied (default: %(default)s)", default = 8, type = int)

    # Discriminator attention layers
    # parser.add_argument("--d-transformer",    help = "Add transformer layers to the discriminator (bottom-up image-to-latents) (default: False)", default = None, action = "store_true")
    # parser.add_argument("--d-start-res",      help = "Transformer minimum discriminator resolution (logarithmic): first layer in which transformer will be applied (default: %(default)s)", default = 0, type = int)
    # parser.add_argument("--d-end-res",        help = "Transformer maximum discriminator resolution (logarithmic): last layer in which transformer will be applied (default: %(default)s)", default = 8, type = int)

    # Attention
    parser.add_argument("--ltnt-gate",          help = "Gate attention from latents, such that components may not send information " +
                                                       "when gate value is low (default: False)", default = None, action = "store_true")
    parser.add_argument("--img-gate",           help = "Gate attention for images, such that some image positions may not get updated " +
                                                       "or receive information when gate value is low (default: False)", default = None, action = "store_true")
    parser.add_argument("--kmeans",             help = "Track and update image-to-latents assignment centroids, used in the duplex attention (default: False)", default = None, action = "store_true")
    parser.add_argument("--kmeans-iters",       help = "Number of K-means iterations per transformer layer. Note that centroids are carried from layer to layer (default: %(default)s)", default = 1, type = int) # -per-layer
    parser.add_argument("--iterative",          help = "Whether to carry over attention assignments across transformer layers of different resolutions (default: False)", default = None, action = "store_true")

    # Attention directions
    # format is A2B: Elements _from_ B attend _to_ elements in A, and B elements get updated accordingly.
    # Note that it means that information propagates in the following direction: A -> B
    parser.add_argument("--mapping-ltnt2ltnt",  help = "Add self-attention over latents in the mapping network (default: False)", default = None, action = "store_true")
    # parser.add_argument("--g-ltnt2ltnt",      help = "Add self-attention over latents in the synthesis network (default: False)", default = None, action = "store_true")
    # parser.add_argument("--g-img2img",        help = "Add self-attention between images positions in that layer of the generator (SAGAN) (default: disabled)", default = 0, type = int)
    # parser.add_argument("--g-img2ltnt",       help = "Add image to latents attention (bottom-up) (default: %(default)s)", default = None, action = "store_true")
    # g-ltnt2img: default information flow direction when using --transformer

    # parser.add_argument("--d-ltnt2img",       help = "Add latents to image attention (top-down) (default: %(default)s)", default = None, action = "store_true")
    # parser.add_argument("--d-ltnt2ltnt",      help = "Add self-attention over latents in the discriminator (default: False)", default = None, action = "store_true")
    # parser.add_argument("--d-img2img",        help = "Add self-attention over images positions in that layer of the discriminator (SAGAN) (default: disabled)", default = 0, type = int)
    # d-img2ltnt: default information flow direction when using --d-transformer

    # Local attention operations (replacing convolution)
    # parser.add_argument("--g-local-attention",  help = "Local attention operations in the generation up to this layer (default: disabled)", default = None, type = int)
    # parser.add_argument("--d-local-attention",  help = "Local attention operations in the discriminator up to this layer (default: disabled)", default = None, type = int)

    # Positional encoding
    parser.add_argument("--use-pos",            help = "Use positional encoding (default: False)", default = None, action = "store_true")
    parser.add_argument("--pos-dim",            help = "Positional encoding dimension (default: latent-size)", default = None, type = int)
    parser.add_argument("--pos-type",           help = "Positional encoding type (default: %(default)s)", default = "sinus", choices = ["linear", "sinus", "trainable", "trainable2d"], type = str)
    parser.add_argument("--pos-init",           help = "Positional encoding initialization distribution (default: %(default)s)", default = "uniform", choices = ["uniform", "normal"], type = str)
    parser.add_argument("--pos-directions-num", help = "Positional encoding number of spatial directions (default: %(default)s)", default = 2, type = int)

    ## k-GAN (not currently supported in the pytorch version. See TF version for implementation of this baseline)
    # parser.add_argument("--kgan",             help = "Generate components-num images and then merge them (k-GAN) (default: False)", default = None, action = "store_true")
    # parser.add_argument("--merge-layer",      help = "Merge layer, where images get combined through alpha-composition (default: %(default)s)", default = -1, type = int)
    # parser.add_argument("--merge-type",       help = "Merge type (default: sum)", default = None, choices = ["sum", "softmax", "max", "leaves"], type = str)
    # parser.add_argument("--merge-same",       help = "Merge images with same alpha weights across all spatial positions (default: %(default)s)", default = None, action = "store_true")

    args = parser.parse_args()

    dnnlib.util.Logger(should_flush = True)
    run_name, run_dir = setup_working_space(args)
    config = setup_config(run_dir, **vars(args))
    setup_savefile(args, run_name, run_dir, config)
    launch_processes(config)

if __name__ == "__main__":
    main()
