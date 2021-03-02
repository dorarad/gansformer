import argparse
import copy
import glob
import os

import dnnlib
from dnnlib import EasyDict
from metrics.metric_defaults import metric_defaults

# Suppress warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from warnings import simplefilter
simplefilter(action="ignore", category = FutureWarning)

# Conditional set: if property is not None, then assign d[name] := prop 
# for every d in a set of dictionaries
def cset(dicts, name, prop):
    if not isinstance(dicts, list):
        dicts = [dicts]
    if prop is not None:
        for d in dicts:
            d[name] = prop

# Set network (generator or discriminator): model, loss and optimizer
def set_net(net, reg_interval):
    ret = EasyDict()
    ret.args  = EasyDict(func_name = "training.network.{}_GANsformer".format(net[0])) # network options
    ret.loss_args = EasyDict(func_name = "training.loss.{}_loss".format(net[0]))                # loss options
    ret.opt_args  = EasyDict(beta1 = 0.0, beta2 = 0.99, epsilon = 1e-8)                         # optimizer options
    ret.reg_interval = reg_interval
    return ret

def run(**args): 
    args      = EasyDict(args)
    train     = EasyDict(run_func_name = "training.training_loop.training_loop") # training loop options
    sched     = EasyDict()                                                       # TrainingSchedule options
    vis       = EasyDict()                                                       # visualize.eval() options
    grid      = EasyDict(size = "1080p", layout = "random")                      # setup_snapshot_img_grid() options
    sc        = dnnlib.SubmitConfig()                                            # dnnlib.submit_run() options

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

    # Networks configuration
    cG = set_net("G", reg_interval = 4)
    cD = set_net("D", reg_interval = 16)

    # Dataset configuration
    ratios = {
        "clevr": 0.75, 
        "lsun-bedrooms": 0.72, 
        "cityscapes": 0.5, 
        "ffhq": 1.0
    }
    args.ratio = ratios.get(args.dataset, args.ratio)
    dataset_args = EasyDict(tfrecord_dir = args.dataset, max_imgs = args.max_images, ratio = args.ratio,
        num_threads = args.num_threads)
    for arg in ["data_dir", "mirror_augment", "total_kimg"]:
        cset(train, arg, args[arg])

    # Training and Optimizations configuration
    for arg in ["eval", "train", "recompile", "last_snapshots"]:
        cset(train, arg, args[arg])

    # Round to the closest multiply of minibatch size for validity
    args.batch_size -= args.batch_size % args.minibatch_size
    args.minibatch_std_size -= args.minibatch_std_size % args.minibatch_size
    args.latent_size -= args.latent_size % args.component_num
    if args.latent_size == 0:
        print(bcolored("Error: latent-size is too small. Must best a multiply of component-num.", "red")) 
        exit()

    sched_args = {
        "G_lrate": "g_lr",
        "D_lrate": "d_lr",
        "minibatch_size": "batch_size",
        "minibatch_gpu": "minibatch_size"
    }
    for arg, cmd_arg in sched_args.items():
        cset(sched, arg, args[cmd_arg])
    cset(train, "clip", args.clip)

    # Logging and metrics configuration
    metrics = [metric_defaults[x] for x in args.metrics]
    cset(cG.args, "truncation_psi", args.truncation_psi)
    for arg in ["summarize", "keep_samples"]:
        cset(train, arg, args[arg])

    # Visualization
    args.imgs = args.images
    args.ltnts = args.latents
    vis_types ["imgs", "ltnts", "maps", "layer_maps", "interpolations", "noise_var", "style_mix"]:
    # Set of all the set visualization types option
    vis.vis_types = {arg for arg in vis_types if args[arg]}

    vis_args = {
        "grid": "vis_grid"    ,
        "num": "vis_num"   ,
        "rich_num": "vis_rich_num",
        "section_size": "vis_section_size",
        "intrp_density": "intrpolation_density",
        "intrp_per_component": "intrpolation_per_component",
        "alpha": "blending_alpha"
    }
    for arg, cmd_arg in vis_args.items():
        cset(vis, arg, args[cmd_arg])

    # Networks architecture
    cset(cG.args, "architecture", args.g_arch)
    cset(cD.args, "architecture", args.d_arch)
    cset([cG.args, cD.args], "resnet_mlp", args.resnet_mlp)
    cset(cG.args, "tanh", args.tanh)

    # Latent sizes
    if args.component_num > 1 
        if not (args.attention or args.merge):
            print(bcolored("Error: component-num > 1 but the model is not using components.", "red")) 
            print(bcolored("Either add --attention for GANsformer or --merge for k-GAN).", "red"))
            exit()    
        args.latent_size = int(args.latent_size / args.component_num)
    cD.args.latent_size = cG.args.latent_size = cG.args.dlatent_size = args.latent_size 
    cset([cG.args, cD.args, train, vis], "component_num", args.component_num)

    # Mapping network
    for arg in ["layersnum", "lrmul", "dim", "shared"]:
        cset(cG.args, arg, args["mapping_{}".formt(arg)])    

    # StyleGAN settings
    for arg in ["style", "latent_stem", "fused_modconv", "local_noise"]:
        cset(cG.args, arg, args[arg])  
    cD.args.mbstd_group_size = args.minibatch_std_size

    # GANsformer
    cset([cG.args, train], "attention", args.transformer)
    cset(cD.args, "attention", args.d_transformer)
    cset([cG.args, cD.args], "num_heads", args.num_heads)

    args.norm = args.normalize
    for arg in ["norm", "integration", "ltnt_gate", "img_gate", "kmeans", 
                "kmeans_iters", "asgn_direct", "mapping_ltnt2ltnt"]:
        cset(cG.args, arg, args[arg])  

    for arg in ["attention_inputs", "use_pos"]:
        cset([cG.args, cD.args], arg, args[arg])  

    # Positional encoding
    for arg in ["dim", "init", "directions_num"]:
        field = "pos_{}".format(arg)
        cset([cG.args, cD.args], field, args[field])  

    # k-GAN
    for arg in ["layer", "type", "channelwise"]:
        field = "merge_{}".format(arg)
        cset(cG.args, field, args[field])  
    cset([cG.args, train], "merge", args.merge)

    # Attention
    for arg in ["start_res", "end_res", "ltnt2ltnt", "img2img", "local_attention"]:
        cset(cG.args, arg, args["g_{}".format(arg)]) 
        cset(cD.args, arg, args["d_{}".format(arg)])         
    cset(cG.args, "img2ltnt", args.g_img2ltnt)
    cset(cD.args, "ltnt2img", args.d_ltnt2img)

    # Mixing and dropout
    for arg in ["style_mixing", "component_mixing", "component_dropout", "attention_dropout"]:
        cset(cG.args, arg, args[arg])  

    # Loss and regularization
    gloss_args = {
        "loss_type": "g_loss",
        "reg_weight": "g_reg_weight"
        "pathreg": "pathreg",
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

    ##### Experiments management:
    # Whenever we start a new experiment we store its result in a directory named 'args.expname:000'.
    # When we rerun a training or evaluation command it restores the model from that directory by default.
    # If we wish to restart the model training, we can set --restart and then we will store data in a new
    # directory: 'args.expname:001' after the first restart, then 'args.expname:002' after the second, etc.

    # Find the latest directory that matches the experiment
    exp_dir = sorted(glob.glob("{}/{}:*".format(args.result_dir, args.expname)))[-1]
    run_id = int(exp_dir.split(":")[-1])
    # If restart, then work over a new directory
    if args.restart:
        run_id += 1

    run_name = "{}:{0:03d}".format(args.expname, run_id)
    train.printname = "{} ".format(misc.bold(args.expname))

    snapshot, kimg, resume = None, 0, False
    pkls = sorted(glob.glob("{}/{}/network*.pkl".format(args.result_dir, run_name)))
    # Load a particular snapshot is specified 
    if args.pretrained_pkl:
        # Soft links support
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
        print(misc.bcolored("Resuming {}, kimg {}".format(snapshot, kimg), "white"))
        train.resume_pkl = snapshot
        train.resume_kimg = kimg
    else:
        print("Start model training from scratch.", "white")

    # Run environment configuration
    sc.run_dir_root = args.result_dir
    sc.run_desc = args.expname
    sc.run_id = run_id
    sc.run_name = run_name
    sc.submit_target = dnnlib.SubmitTarget.LOCAL
    sc.local.do_not_copy_source_files = True

    kwargs = EasyDict(train)
    kwargs.update(cG = cG, cD = cD)
    kwargs.update(dataset_args = dataset_args, vis_args = vis, sched_args = sched, grid_args = grid, metric_arg_list = metrics, tf_config = tf_config)
    kwargs.submit_config = copy.deepcopy(sc)
    kwargs.resume = resume
    # If reload new options from the command line, no need to load the original configuration file
    kwargs.load_config = not args.reload

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
        raise argparse.ArgumentTypeError("Boolean value expected.")

def _parse_comma_sep(s):
    if s is None or s.lower() == "none" or s == "":
        return []
    return s.split(",")

_examples = """examples:

  # Train the GANsformer model on CLEVR
  python %(prog)s --num-gpus = 4 --data-dir=~/datasets --dataset = clevr

valid metrics:

  """ + ", ".join(sorted([x for x in metric_defaults.keys()])) + """

"""

def main():
    parser = argparse.ArgumentParser(
        description = "Train the GANsformer",
        epilog = _examples,
        formatter_class = argparse.RawDescriptionHelpFormatter
    )

    # Framework
    # ------------------------------------------------------------------------------------------------------
    
    parser.add_argument("--expname",            help = "Experiment name", default = "exp", type = str)
    parser.add_argument("--eval",               help = "Evaluation mode (default: False)", default = None, action = "store_true") 
    parser.add_argument("--train",              help = "Train mode (default: False)", default = None, metavar = "BOOL", type = _str_to_bool) 
    parser.add_argument("--gpus",               help = "Comma-separated list of GPUs to be used (default: %(default)s)", default = "0", type = str)

    ## Resumption
    parser.add_argument("--pretrained_pkl",     help = "Filename for a snapshot to resume (optional)", default = None, type = str)
    parser.add_argument("--restart",            help = "Restart training from scratch", default = False, action = "store_true") 
    parser.add_argument("--reload",             help = "Reload new options from the command line when resuming training. " + 
                                                       "If False, uses instead original experiment configuration file (default: %(default)s)", default = False, action = "store_true") 
    parser.add_argument("--recompile",          help = "Recompile model from source code when resuming training. " + 
                                                       "If False, loading modules created when the experiment first started", default = None, action = "store_true")
    parser.add_argument("--last_snapshots",     help = "Number of last snapshots to save. -1 for all (default: 8)", default = None, type = int)

    ## Dataset
    parser.add_argument("--data-dir",           help = "Datasets root directory", required = True)
    parser.add_argument("--dataset",            help = "Training dataset name (subdirectory of data-dir).", required = True)
    parser.add_argument("--ratio",              help = "Image height/width ratio in the dataset", default = 1.0, type = float) 
    parser.add_argument("--max-images",         help = "Maximum number of images to train on. If not specified, train on the whole dataset.", efault = None, type = int) 
    parser.add_argument("--num-threads",        help = "Number of input processing threads (default: %(default)s)", default = 4, type = int)
    parser.add_argument("--mirror-augment",     help = "Perform horizontal flip augmentation for the data (default: %(default)s)", default = False)

    ## Training
    parser.add_argument("--batch-size",         help = "Global batch size (optimization step) (default: %(default)s)", default = 32, type = int) 
    parser.add_argument("--minibatch-size",     help = "Batch size per GPU, gradients will be accumulated to match batch-size (default: %(default)s)", default = 4, type = int)     
    parser.add_argument("--total-kimg",         help = "Training length in thousands of images (default: %(default)s)", metavar = "KIMG", default = 25000, type = int)
    parser.add_argument("--gamma",              help = "R1 regularization weight (default: %(default)s)", default = 10, type = float)
    parser.add_argument("--clip",               help = "Gradient clipping threshold (optional)", default = None, type = float) 
    parser.add_argument("--g-lr",               help = "Generator learning rate (default: 0.002)", default = None, type = float)
    parser.add_argument("--d-lr",               help = "Discriminator learning rate (default: 0.002)", default = None, type = float)

    ## Logging and evaluation
    parser.add_argument("--result-dir",         help = "Root directory for experiments (default: %(default)s)", default = "results", metavar = "DIR")
    parser.add_argument("--metrics",            help = "Comma-separated list of metrics or none (default: %(default)s)", default = "fid20k", type = _parse_comma_sep)
    parser.add_argument("--summarize",          help = "Create TensorBoard summaries (default: %(default)s)", default = True, metavar = "BOOL", type = _str_to_bool) 
    parser.add_argument("--truncation-psi",     help = "Truncation Psi to be used in producing sample images " + 
                                                       "(used only for visualizations, _not used_ in training or for computing metrics) (default: %(default)s)", default = 0.65, type = float)
    parser.add_argument("--keep-samples",       help = "Keep all prior samples during training, or if False, just the most recent ones (default: False)", default = None, action = "store_true")     

    ## Visualization 
    parser.add_argument("--vis-images",         help = "Save image samples", default = None, action = "store_true") 
    parser.add_argument("--vis-latents",        help = "Save latent vectors", default = None, action = "store_true") 
    parser.add_argument("--vis-maps",           help = "Save attention maps (for GANsformer only)", default = None, action = "store_true") 
    parser.add_argument("--vis-layer-maps",     help = "Save attention maps for all layers (for GANsformer only)", default = None, action = "store_true") 
    parser.add_argument("--vis-interpolations", help = "Create latent interpolations", default = None, action = "store_true") 
    parser.add_argument("--vis-noise-var",      help = "Create noise variation visualization", default = None, action = "store_true") 
    parser.add_argument("--vis-style-mix",      help = "Create style mixing visualization", default = None, action = "store_true") 

    parser.add_argument("--vis-grid",               help = "Whether to save the samples in one large grid files (default: True in training)", default = None, action = "store_true") 
    parser.add_argument("--vis-num",            help = "Image height/width ratio in the dataset", default = None, type = int) 
    parser.add_argument("--vis-rich-num",       help = "Number of samples for which richer visualizations will be created (default: 5)", default = None, type = int) 
    parser.add_argument("--vis-section-size",       help = "Visualization section size to process at one (section-size <= vis-num) for memory footprint (default: 100)", default = None, type = int) 
    parser.add_argument("--blending-alpha",     help = "Proportion for generated images and attention maps blends (default: 0.3)", default = None, type = float) 
    parser.add_argument("--intrpolation-density",       help = "Number of samples in between two end points of an interpolation (default: 8)", default = None, type = int) 
    parser.add_argument("--intrpolation-per-component", help = "Whether to perform interpolation along particular latent components when true, or all of them at once otherwise (default: False)", default = None, action = "store_true") 

    # Model
    # ------------------------------------------------------------------------------------------------------

    ## General architecture
    parser.add_argument("--g-arch",             help = "Generator architecture type (default: skip)", default = None, choices = ["orig", "skip", "resnet"], type = str)
    parser.add_argument("--d-arch",             help = "Discriminator architecture type (default: resnet)", default = None, choices = ["orig", "skip", "resnet"], type = str)
    parser.add_argument("--resnet-mlp",         help = "Use resent connections in MLP layers (default: False)", default = None, action = "store_true")
    parser.add_argument("--tanh",               help = "tanh on generator output (default: False)", default = None, action = "store_true")

    # Mapping network
    parser.add_argument("--mapping-layersnum",  help = "Number of mapping layers (default: 8)", default = None, type = int)
    parser.add_argument("--mapping-lrmul",      help = "Mapping network learning rate multiplier (default: 0.01)", default = None, type = float)
    parser.add_argument("--mapping-dim",        help = "Mapping layers dimension (default: latent_size)", default = None, type = int)
    parser.add_argument("--mapping-shared",     help = "Perform one shared mapping to all latent components concatenated together (default: False)", default = None, type = int)    

    # Loss
    parser.add_argument("--pathreg",            help = "Use path regularization in generator training (default: %(default)s)", default = True, metavar = "BOOL", type = _str_to_bool)    
    parser.add_argument("--g-loss",             help = "Generator loss type (default: %(default)s)", default = "logistic_ns", choices = ["logistic", "logistic_ns", "hinge", "wgan"], type = str)    
    parser.add_argument("--g-reg-weight",       help = "Generator regularization weight (default: %(default)s)", default = 1.0, type = float) 

    parser.add_argument("--d-loss",             help = "Discriminator loss type (default: %(default)s)", default = "logistic", choices = ["wgan", "logistic", "hinge"], type = str)    
    parser.add_argument("--d-reg",              help = "Discriminator regularization type (default: %(default)s)", default = "r1", choices = ["non", "gp", "r1", "r2"], type = str) 

    # Mixing and dropout
    parser.add_argument("--style-mixing",       help = "Style mixing (layerwise) probability (default: %(default)s)", default = 0.9, type = float)
    parser.add_argument("--component-mixing",   help = "Component mixing (objectwise) probability (default: %(default)s)", default = 0.0, type = float) 
    parser.add_argument("--component-dropout",  help = "Component dropout (default: %(default)s)", default = 0.0, type = float) 
    parser.add_argument("--attention-dropout",  help = "Attention dropout (default: 0.12)", default = None, type = float) 

    # StyleGAN additions
    parser.add_argument("--style",              help = "Global style modulation (default: %(default)s)", default = True, metavar = "BOOL", type = _str_to_bool)
    parser.add_argument("--latent-stem",        help = "Input latent through the generator stem grid (default: False)", default = None, action = "store_true")
    parser.add_argument("--fused-modconv",      help = "Fuse modulation and convolution operations (default: %(default)s)", default = True, metavar = "BOOL", type = _str_to_bool)
    parser.add_argument("--local-noise",        help = "Add stochastic local noise each layer (default: %(default)s)", default = True, metavar = "BOOL", type = _str_to_bool)
    parser.add_argument("--minibatch-std-size", help = "Add minibatch standard deviation layer in the discriminator (default: %(default)s)", default = 4, type = int)

    ## GANsformer
    parser.add_argument("--transformer",        help = "Add transformer layers to the generator: top-down latents-to-image (default: False)", default = None, default = None, action = "store_true") 
    parser.add_argument("--latent-size",        help = "Latent size, summing the dimension of all components (default: %(default)s)", default = 512, type = int)
    parser.add_argument("--component-num",      help = "Components number. Each component has latent dimension of 'latent-size / component-num'. " + 
                                                       "1 for StyleGAN since it has one global latent vector (default: %(default)s)", default = 1, type = int) 
    parser.add_argument("--num-heads",          help = "Number of attention heads (default: %(default)s)", default = 1, type = int) 
    parser.add_argument("--normalize",          help = "Feature normalization type (optional)", default = None, choices = ["batch", "instance", "layer"]) 
    parser.add_argument("--integration",        help = "Feature integration type: additive, multiplicative or both (default: %(default)s)", default = "add", choices = ["add", "mul", "both"], type = str) 

    # Generator attention layers
    # Add transformer
    parser.add_argument("--g-start-res",        help = "Transformer minimum generator resolution (logarithmic): first layer in which transformer will be applied (default: %(default)s)", default = 0, type = int) 
    parser.add_argument("--g-end-res",          help = "Transformer maximum generator resolution (logarithmic): last layer in which transformer will be applied (default: %(default)s)", default = 7, type = int) 

    # Discriminator attention layers
    parser.add_argument("--d-transformer",      help = "Add transformer layers to the discriminator (bottom-up image-to-latents) (default: False)", default = None, action = "store_true") 
    parser.add_argument("--d-start-res",        help = "Transformer minimum discriminator resolution (logarithmic): first layer in which transformer will be applied (default: %(default)s)", default = 0, type = int) 
    parser.add_argument("--d-end-res",          help = "Transformer maximum discriminator resolution (logarithmic): last layer in which transformer will be applied (default: %(default)s)", default = 7, type = int) 

    # Attention 
    parser.add_argument("--ltnt-gate",          help = "Gate attention from latents, such that components potentially will not send information " + 
                                                       "when gate value is low (default: False)", default = None, action = "store_true") 
    parser.add_argument("--img-gate",           help = "Gate attention for images, such that some image positions potentially will not get updated " + 
                                                       "or receive information when gate value is low (default: False)", default = None, action = "store_true") 
    parser.add_argument("--kmeans",             help = "Track and update image-to-latents assignment centroids, used in the duplex attention (default: False)", default = None, action = "store_true") 
    parser.add_argument("--kmeans-iters",       help = "Number of K-means iterations per transformer layer. Note that centroids are carried from layer to layer (default: %(default)s)", default = 1, type = int) # -per-layer
    parser.add_argument("--asgn_direct",        help = "TODO (default: False)", default = None, action = "store_true")
    parser.add_argument("--attention-inputs",   help = "Attention function inputs: content, position or both (default: %(default)s)", default = "both", choices = ["content", "position", "both"], type = str) 

    # Attention directions
    # format is A2B: Elements _from_ B attend _to_ elements in A, and B elements get updated accordingly.
    # Note that it means that information propagates in the following direction: A -> B
    parser.add_argument("--mapping-ltnt2ltnt",  help = "Add self-attention over latents in the mapping network (default: False)", default = None, action = "store_true") 
    parser.add_argument("--g-ltnt2ltnt",        help = "Add self-attention over latents in the synthesis network (default: False)", default = None, action = "store_true") 
    parser.add_argument("--g-img2img",          help = "Add self-attention between images positions in the discriminator (SAGAN) (default: False)", default = 0, type = int) 
    parser.add_argument("--g-img2ltnt",         help = "Train mode # bottom-up (default: %(default)s)", default = None, action = "store_true")
    # g-ltnt2img: default information flow direction when using --transformer

    parser.add_argument("--d-ltnt2img",         help = "Train mode # top-down (default: %(default)s)", default = None, action = "store_true") 
    parser.add_argument("--d-ltnt2ltnt",        help = "Add self-attention over latents in the discriminator (default: False)", default = None, action = "store_true") 
    parser.add_argument("--d-img2img",          help = "Add self-attention over images positions in the discriminator (SAGAN) (default: False)", default = None, action = "store_true") 
    # d-img2ltnt: default information flow direction when using --d-transformer

    # Local attention operations (replacing convolution)
    parser.add_argument("--g-local-attention",  help = "Local attention operations in the generation (default: False)", default = None, action = "store_true") 
    parser.add_argument("--d-local-attention",  help = "Local attention operations in the discriminator (default: False)", default = None, action = "store_true") 

    # Positional encoding
    parser.add_argument("--use-pos",            help = "Use positional encoding (default: False)", default = None, action = "store_true") 
    parser.add_argument("--pos-dim",            help = "Positional encoding dimension (default: latent-size)", default = None, type = int) 
    parser.add_argument("--pos-type",           help = "Positional encoding type (default: %(default)s)", default = "sinus", choices = ["linear", "sinus", "trainable", "trainable2d"], type = str) 
    parser.add_argument("--pos-init",           help = "Positional encoding initialization distribution (default: %(default)s)", default = "uniform", choices = ["uniform", "normal"], type = str) 
    parser.add_argument("--pos-directions-num", help = "Positional encoding spatial number of directions (default: %(default)s)", default = 2, type = int) 

    ## k-GAN
    parser.add_argument("--merge",              help = "Generate component-num images and then merge them (k-GAN) (default: False)", default = None, action = "store_true") 
    parser.add_argument("--merge-layer",        help = "Merge layer, where images get combined through alpha-composition (default: %(default)s)", default = -1, type = int) 
    parser.add_argument("--merge-type",         help = "Merge type (default: additive)", default = None, choices = ["additive", "softmax", "max", "leaves"], action = str) 
    parser.add_argument("--merge-channelwise",  help = "Merge images similarly across all spatial positions (default: %(default)s)", default = None, action = "store_true") 

    args = parser.parse_args()

    if not os.path.exists(args.data_dir):
        print(bcolored("Error: dataset root directory does not exist.", "red"))
        exit()

    for metric in args.metrics:
        if metric not in metric_defaults:
            print(bcolored("Error: unknown metric \"%s\"" % metric, "red"))
            exit()

    run(**vars(args))

if __name__ == "__main__":
    main()
