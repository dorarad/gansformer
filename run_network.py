# Ignore all future warnings
from warnings import simplefilter
simplefilter(action = "ignore", category = FutureWarning)

import argparse
import copy
import glob
import sys
import os

import dnnlib
from dnnlib import EasyDict
from metrics.metric_defaults import metric_defaults
from training import misc
import pretrained_networks
# Suppress warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

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

# Set network (generator or discriminator): model, loss and optimizer
def set_net(net, reg_interval):
    ret = EasyDict()
    net2name = {
        "D": "Discriminator", 
        "G": "Generator"
    }
    ret.args  = EasyDict(func_name = f"training.networks.{net2name[net]}")       # network options
    ret.loss_args = EasyDict(func_name = f"training.loss.{net}_loss")            # loss options
    ret.opt_args  = EasyDict(beta1 = 0.0, beta2 = 0.99, epsilon = 1e-8)          # optimizer options
    ret.reg_interval = reg_interval
    return ret

def run(**args):
    args      = EasyDict(args)
    train     = EasyDict(run_func_name = "training.training_loop.training_loop") # training loop options
    sched     = EasyDict()                                                       # TrainingSchedule options
    vis       = EasyDict()                                                       # visualize.eval() options
    grid      = EasyDict(size = (3, 2), layout = "random")                       # setup_snapshot_img_grid() options
    sc        = dnnlib.SubmitConfig()                                            # dnnlib.submit_run() options

    # GANformer and baselines default settings
    # ----------------------------------------------------------------------------

    if args.ganformer_default:
        task = args.dataset

        nset(args, "recompile", args.pretrained_pkl is not None)
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

    if args.baseline == "SAGAN":
        nset(args, "style", False)
        nset(args, "latent_stem", True)
        nset(args, "g_img2img", 5)

    if args.baseline == "kGAN":
        nset(args, "kgan", True)
        nset(args, "merge_layer", 5)
        nset(args, "merge_type", "softmax")
        nset(args, "components_num", 8)

    # General setup
    # ----------------------------------------------------------------------------

    # If the flag is specified without arguments (--arg), set to True
    for arg in ["summarize", "keep_samples", "style", "fused_modconv", "local_noise"]:
        if args[arg] is None:
            args[arg] = True

    # Environment configuration
    tf_config = {
        "rnd.np_random_seed": 1000,
        "allow_soft_placement": True,
        "gpu_options.per_process_gpu_memory_fraction": 1.0
    }
    if args.gpus != "":
        num_gpus = len(args.gpus.split(","))
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    assert num_gpus in [1, 2, 4, 8]
    sc.num_gpus = num_gpus

    # Data setup
    # ----------------------------------------------------------------------------

    # Dataset configuration
    if not os.path.exists(f"{args.data_dir}/{args.dataset}"):
        misc.error(f"The dataset {args.data_dir}/{args.dataset} directory does not exist")

    # For bedrooms, we choose the most common ratio in the 
    # dataset and crop the other images into that ratio.
    ratios = {
        "clevr": 0.75,
        "bedrooms": 188/256, 
        "cityscapes": 0.5,
        "ffhq": 1.0
    }
    args.ratio = args.ratio or ratios.get(args.dataset, 1.0)
    args.crop_ratio = 0.5 if args.resolution > 256 and args.ratio <= 0.5 else None

    dataset_args = EasyDict(
        tfrecord_dir = args.dataset, 
        max_imgs     = args.train_images_num, 
        num_threads  = args.num_threads, 
        resolution   = args.resolution,
        crop_ratio   = args.crop_ratio, 
        ratio        = args.ratio
    )
    for arg in ["data_dir", "mirror_augment", "total_kimg"]:
        cset(train, arg, args[arg])

    # Optimization setup
    # ----------------------------------------------------------------------------

    # Networks configuration
    cG = set_net("G", reg_interval = 4)
    cD = set_net("D", reg_interval = 16)
    cset([cG, cD], "crop_ratio", args.crop_ratio)

    # Training and Optimizations configuration
    if not any([args.train, args.eval, args.vis]):
        misc.log("Warning: None of --train, --eval or --vis are provided. Therefore, we only print network shapes", "red")

    for arg in ["train", "eval", "vis", "recompile", "last_snapshots"]:
        cset(train, arg, args[arg])

    if args.batch_size % (args.batch_gpu * num_gpus) != 0:
        misc.error("--batch-size should be divided by --batch-gpu * 'num_gpus'")

    if args.latent_size % args.components_num != 0:
        misc.error(f"--latent-size ({args.latent_size}) should be divisible by --components-num (k={k})")

    sched_args = {
        "G_lrate": "g_lr",
        "D_lrate": "d_lr",
        "batch_size": "batch_size",
        "batch_gpu": "batch_gpu"
    }
    for arg, cmd_arg in sched_args.items():
        cset(sched, arg, args[cmd_arg])
    cset(train, "clip", args.clip)

    # Evaluation and visualization
    # ----------------------------------------------------------------------------

    # Logging and metrics configuration
    for metric in args.metrics:
        if metric not in metric_defaults:
            misc.error(f"Unknown metric: {metric}")
    metrics = [metric_defaults[x] for x in args.metrics]

    for arg in ["summarize", "eval_images_num"]:
        cset(train, arg, args[arg])
    cset(cG.args, "truncation_psi", args.truncation_psi)
    for arg in ["keep_samples", "num_heads"]:
        cset(vis, arg, args[arg])

    # Visualization
    args.vis_imgs = args.vis_images
    args.vis_ltnts = args.vis_latents
    vis_types = ["imgs", "ltnts", "maps", "layer_maps", "interpolations", "noise_var", "style_mix"]
    # Set of all the set visualization types option
    vis.vis_types = {arg for arg in vis_types if args[f"vis_{arg}"]}

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
    cset(cG.args, "architecture", args.g_arch)
    cset(cD.args, "architecture", args.d_arch)
    cset(cG.args, "tanh", args.tanh)

    # Latent sizes
    if args.components_num > 1:
        if not (args.transformer or args.kgan):
            misc.error("--components-num > 1 but the model is not using components. " + 
                "Either add --transformer for GANformer or --kgan for k-GAN.")

        args.latent_size = int(args.latent_size / args.components_num)
    cD.args.a_dim = cG.args.z_dim = cG.args.w_dim = args.latent_size
    cset([cG.args, cD.args, vis], "components_num", args.components_num)

    # Mapping network
    for arg in ["layersnum", "lrmul", "dim", "resnet", "shared_dim"]:
        field = f"mapping_{arg}"
        cset(cG.args, field, args[field])

    # StyleGAN settings
    for arg in ["style", "latent_stem", "fused_modconv", "local_noise"]:
        cset(cG.args, arg, args[arg])
    cD.args.mbstd_group_size = min(args.batch_size, 4)

    # GANformer
    cset(cG.args, "transformer", args.transformer)
    cset(cD.args, "transformer", args.d_transformer)

    args.norm = args.normalize
    for arg in ["norm", "integration", "ltnt_gate", "img_gate", "iterative", "kmeans", 
                "kmeans_iters", "mapping_ltnt2ltnt"]:
        cset(cG.args, arg, args[arg])

    for arg in ["use_pos", "num_heads"]:
        cset([cG.args, cD.args], arg, args[arg])

    # Positional encoding
    for arg in ["dim", "type", "init", "directions_num"]:
        field = f"pos_{arg}"
        cset([cG.args, cD.args], field, args[field])

    # k-GAN
    for arg in ["layer", "type", "same"]:
        field = f"merge_{arg}"
        cset(cG.args, field, args[field])
    cset(cG.args, "merge", args.kgan)

    if args.kgan and args.transformer:
        misc.error("Either have --transformer for GANformer or --kgan for k-GAN, not both")

    # Attention
    for arg in ["start_res", "end_res", "ltnt2ltnt", "img2img"]: # , "local_attention"
        cset(cG.args, arg, args[f"g_{arg}"])
        cset(cD.args, arg, args[f"d_{arg}"])
    cset(cG.args, "img2ltnt", args.g_img2ltnt)
    # cset(cD.args, "ltnt2img", args.d_ltnt2img)

    # Mixing and dropout
    for arg in ["style_mixing", "component_mixing", "component_dropout", "attention_dropout"]:
        cset(cG.args, arg, args[arg])

    # Loss and regularization
    gloss_args = {
        "loss_type": "g_loss",
        "reg_weight": "g_reg_weight",
        # "pathreg": "pathreg"
    }
    dloss_args = {
        "loss_type": "d_loss",
        "reg_type": "d_reg",
        "gamma": "gamma"
    }
    for arg, cmd_arg in gloss_args.items():
        cset(cG.loss_args, arg, args[cmd_arg])
    for arg, cmd_arg in dloss_args.items():
        cset(cD.loss_args, arg, args[cmd_arg])

    # Setup and launching
    # ----------------------------------------------------------------------------

    ##### Experiments management:
    # Whenever we start a new experiment we store its result in a directory named 'args.expname:000'.
    # When we rerun a training or evaluation command it restores the model from that directory by default.
    # If we wish to restart the model training, we can set --restart and then we will store data in a new
    # directory: 'args.expname:001' after the first restart, then 'args.expname:002' after the second, etc.

    # Find the latest directory that matches the experiment
    exp_dir = sorted(glob.glob(f"{args.result_dir}/{args.expname}-*"))
    run_id = 0
    if len(exp_dir) > 0:
        run_id = int(exp_dir[-1].split("-")[-1])
    # If restart, then work over a new directory
    if args.restart:
        run_id += 1

    run_name = f"{args.expname}-{run_id:03d}"
    train.printname = f"{misc.bold(args.expname)} "

    snapshot, kimg, resume = None, 0, False
    pkls = sorted(glob.glob(f"{args.result_dir}/{run_name}/network*.pkl"))
    # Load a particular snapshot is specified
    if args.pretrained_pkl is not None and args.pretrained_pkl != "None":
        # Soft links support
        if args.pretrained_pkl.startswith("gdrive"):
            if args.pretrained_pkl not in pretrained_networks.gdrive_urls:
                misc.error("--pretrained_pkl {} not available in the catalog (see pretrained_networks.py)")

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
        train.resume_pkl = snapshot
        train.resume_kimg = kimg
    else:
        misc.log("Start model training from scratch", "white")

    # Run environment configuration
    sc.run_dir_root = args.result_dir
    sc.run_desc = args.expname
    sc.run_id = run_id
    sc.run_name = run_name
    sc.submit_target = dnnlib.SubmitTarget.LOCAL
    sc.local.do_not_copy_source_files = True

    kwargs = EasyDict(train)
    kwargs.update(cG = cG, cD = cD)
    kwargs.update(dataset_args = dataset_args, vis_args = vis, sched_args = sched, 
        grid_args = grid, metric_arg_list = metrics, tf_config = tf_config)
    kwargs.submit_config = copy.deepcopy(sc)
    kwargs.resume = resume
    kwargs.load_config = args.reload

    dnnlib.submit_run(**kwargs)

# ----------------------------------------------------------------------------

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
    parser.add_argument("--baseline",           help = "Use a baseline model configuration", default = None, choices = ["GAN", "StyleGAN2", "kGAN", "SAGAN"], type = str)

    ## Resumption
    parser.add_argument("--pretrained-pkl",     help = "Filename for a snapshot to resume (optional)", default = None, type = str)
    parser.add_argument("--restart",            help = "Restart training from scratch", default = False, action = "store_true")
    parser.add_argument("--reload",             help = "Reload options from the original experiment configuration file. " +
                                                       "If False, uses the command line arguments when resuming training (default: %(default)s)", default = False, action = "store_true")
    parser.add_argument("--recompile",          help = "Recompile model from source code when resuming training. " +
                                                       "If False, loading modules created when the experiment first started", default = None, action = "store_true")
    parser.add_argument("--last-snapshots",     help = "Number of last snapshots to save. -1 for all (default: 10)", default = None, type = int)

    ## Dataset
    parser.add_argument("--data-dir",           help = "Datasets root directory (default: %(default)s)", default = "datasets", metavar = "DIR")
    parser.add_argument("--dataset",            help = "Training dataset name (subdirectory of data-dir)", required = True)
    parser.add_argument("--ratio",              help = "Image height/width ratio in the dataset", default = None, type = float)
    parser.add_argument("--resolution",         help = "Training resolution", default = 256, type = int)
    parser.add_argument("--num-threads",        help = "Number of input processing threads (default: %(default)s)", default = 4, type = int)
    parser.add_argument("--mirror-augment",     help = "Perform horizontal flip augmentation for the data (default: %(default)s)", default = None, action = "store_true")
    parser.add_argument("--train-images-num",   help = "Maximum number of images to train on. If not specified, train on the whole dataset.", default = None, type = int)

    ## Training
    parser.add_argument("--batch-size",         help = "Global batch size (optimization step) (default: %(default)s)", default = 32, type = int)
    parser.add_argument("--batch-gpu",          help = "Batch size per GPU, gradients will be accumulated to match batch-size (default: %(default)s)", default = 4, type = int)
    parser.add_argument("--total-kimg",         help = "Training length in thousands of images (default: %(default)s)", metavar = "KIMG", default = 25000, type = int)
    parser.add_argument("--gamma",              help = "R1 regularization weight (default: %(default)s)", default = 10, type = float)
    parser.add_argument("--clip",               help = "Gradient clipping threshold (optional)", default = None, type = float)
    parser.add_argument("--g-lr",               help = "Generator learning rate (default: %(default)s)", default = 0.002, type = float)
    parser.add_argument("--d-lr",               help = "Discriminator learning rate (default: %(default)s)", default = 0.002, type = float)

    ## Logging and evaluation
    parser.add_argument("--result-dir",         help = "Root directory for experiments (default: %(default)s)", default = "results", metavar = "DIR")
    parser.add_argument("--metrics",            help = "Comma-separated list of metrics or none (default: %(default)s)", default = "fid", type = _parse_comma_sep)
    parser.add_argument("--summarize",          help = "Create TensorBoard summaries (default: %(default)s)", default = True, metavar = "BOOL", type = _str_to_bool, nargs = "?")
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
    # parser.add_argument("--interpolation-per-component", help = "Whether to perform interpolation along particular latent components when true, or all of them at once otherwise (default: False)", default = None, action = "store_true")

    # Model
    # ------------------------------------------------------------------------------------------------------

    ## General architecture
    parser.add_argument("--g-arch",             help = "Generator architecture type (default: skip)", default = None, choices = ["orig", "skip", "resnet"], type = str)
    parser.add_argument("--d-arch",             help = "Discriminator architecture type (default: resnet)", default = None, choices = ["orig", "skip", "resnet"], type = str)
    parser.add_argument("--tanh",               help = "tanh on generator output (default: False)", default = None, action = "store_true")

    # Mapping network
    parser.add_argument("--mapping-layersnum",  help = "Number of mapping layers (default: 8)", default = None, type = int)
    parser.add_argument("--mapping-lrmul",      help = "Mapping network learning rate multiplier (default: 0.01)", default = None, type = float)
    parser.add_argument("--mapping-dim",        help = "Mapping layers dimension (default: latent_size)", default = None, type = int)
    parser.add_argument("--mapping-resnet",     help = "Use resent connections in mapping layers (default: False)", default = None, action = "store_true")
    parser.add_argument("--mapping-shared-dim", help = "Perform one shared mapping to all latent components concatenated together using the set dimension (default: disabled)", default = None, type = int)

    # Loss
    # parser.add_argument("--pathreg",          help = "Use path regularization in generator training (default: False)", default = None, action = "store_true")
    parser.add_argument("--g-loss",             help = "Generator loss type (default: %(default)s)", default = "logistic_ns", choices = ["logistic", "logistic_ns", "hinge", "wgan"], type = str)
    parser.add_argument("--g-reg-weight",       help = "Generator regularization weight (default: %(default)s)", default = 1.0, type = float)

    parser.add_argument("--d-loss",             help = "Discriminator loss type (default: %(default)s)", default = "logistic", choices = ["wgan", "logistic", "hinge"], type = str)
    parser.add_argument("--d-reg",              help = "Discriminator regularization type (default: %(default)s)", default = "r1", choices = ["non", "gp", "r1", "r2"], type = str)
    # --gamma effectively functions as discriminator regularization weight

    # Mixing and dropout
    parser.add_argument("--style-mixing",       help = "Style mixing (layerwise) probability (default: %(default)s)", default = 0.9, type = float)
    parser.add_argument("--component-mixing",   help = "Component mixing (objectwise) probability (default: %(default)s)", default = 0.0, type = float)
    parser.add_argument("--component-dropout",  help = "Component dropout (default: %(default)s)", default = 0.0, type = float)
    parser.add_argument("--attention-dropout",  help = "Attention dropout (default: 0.12)", default = None, type = float)


    # StyleGAN additions
    parser.add_argument("--style",              help = "Global style modulation (default: %(default)s)", default = True, metavar = "BOOL", type = _str_to_bool, nargs = "?")
    parser.add_argument("--latent-stem",        help = "Input latent through the generator stem grid (default: False)", default = None, action = "store_true")
    parser.add_argument("--fused-modconv",      help = "Fuse modulation and convolution operations (default: %(default)s)", default = True, metavar = "BOOL", type = _str_to_bool, nargs = "?")
    parser.add_argument("--local-noise",        help = "Add stochastic local noise each layer (default: %(default)s)", default = True, metavar = "BOOL", type = _str_to_bool, nargs = "?")

    ## GANformer
    parser.add_argument("--transformer",        help = "Add transformer layers to the generator: top-down latents-to-image (default: False)", default = None, action = "store_true")
    parser.add_argument("--latent-size",        help = "Latent size, summing the dimension of all components (default: %(default)s)", default = 512, type = int)
    parser.add_argument("--components-num",     help = "Components number. Each component has latent dimension of 'latent-size / components-num'. " +
                                                       "1 for StyleGAN since it has one global latent vector (default: %(default)s)", default = 1, type = int)
    parser.add_argument("--num-heads",          help = "Number of attention heads (default: %(default)s)", default = 1, type = int)
    parser.add_argument("--normalize",          help = "Feature normalization type (optional)", default = None, choices = ["batch", "instance", "layer"], type = str)
    parser.add_argument("--integration",        help = "Feature integration type: additive, multiplicative or both (default: %(default)s)", default = "add", choices = ["add", "mul", "both"], type = str)

    # Generator attention layers
    # Transformer resolution layers
    parser.add_argument("--g-start-res",        help = "Transformer minimum generator resolution (logarithmic): first layer in which transformer will be applied (default: %(default)s)", default = 0, type = int)
    parser.add_argument("--g-end-res",          help = "Transformer maximum generator resolution (logarithmic): last layer in which transformer will be applied (default: %(default)s)", default = 8, type = int)

    # Discriminator attention layers
    parser.add_argument("--d-transformer",      help = "Add transformer layers to the discriminator (bottom-up image-to-latents) (default: False)", default = None, action = "store_true")
    parser.add_argument("--d-start-res",        help = "Transformer minimum discriminator resolution (logarithmic): first layer in which transformer will be applied (default: %(default)s)", default = 0, type = int)
    parser.add_argument("--d-end-res",          help = "Transformer maximum discriminator resolution (logarithmic): last layer in which transformer will be applied (default: %(default)s)", default = 8, type = int)

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
    parser.add_argument("--g-ltnt2ltnt",        help = "Add self-attention over latents in the synthesis network (default: False)", default = None, action = "store_true")
    parser.add_argument("--g-img2img",          help = "Add self-attention between images positions in that layer of the generator (SAGAN) (default: disabled)", default = 0, type = int)
    parser.add_argument("--g-img2ltnt",         help = "Add image to latents attention (bottom-up) (default: %(default)s)", default = None, action = "store_true")
    # g-ltnt2img: default information flow direction when using --transformer

    # parser.add_argument("--d-ltnt2img",       help = "Add latents to image attention (top-down) (default: %(default)s)", default = None, action = "store_true")
    parser.add_argument("--d-ltnt2ltnt",        help = "Add self-attention over latents in the discriminator (default: False)", default = None, action = "store_true")
    parser.add_argument("--d-img2img",          help = "Add self-attention over images positions in that layer of the discriminator (SAGAN) (default: disabled)", default = 0, type = int)
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

    ## k-GAN
    parser.add_argument("--kgan",               help = "Generate components-num images and then merge them (k-GAN) (default: False)", default = None, action = "store_true")
    parser.add_argument("--merge-layer",        help = "Merge layer, where images get combined through alpha-composition (default: %(default)s)", default = -1, type = int)
    parser.add_argument("--merge-type",         help = "Merge type (default: sum)", default = None, choices = ["sum", "softmax", "max", "leaves"], type = str)
    parser.add_argument("--merge-same",         help = "Merge images with same alpha weights across all spatial positions (default: %(default)s)", default = None, action = "store_true")

    args = parser.parse_args()

    run(**vars(args))

if __name__ == "__main__":
    main()
