# Training loop:
# 1 Sets up the environment and data
# 2 Builds the generator (g) and discriminator (d) networks
# 3 Manages the training process
# 4 Runs periodic evaluations on specified metrics
# 5 Produces sample images over the course of training

# It supports training over an image directory dataset, prepared by prepare_data.py
# Labels can optionally be provided although not essential
# If provided, image will be generated conditioned on a chosen label

import os
import time
import copy
import json
import pickle
import psutil
import PIL.Image
import numpy as np
import torch
import glob 

import dnnlib
from torch_utils import misc as torch_misc
from torch_utils import training_stats
from torch_utils.ops import conv2d_gradfix
from training import visualize
from training import misc 

import loader
from metrics import metric_main
from metrics import metric_utils

# Data processing
# ----------------------------------------------------------------------------

# Load dataset
def load_dataset(dataset_args, batch_size, rank, num_gpus, log):
    misc.log("Loading training set...", "white", log)
    dataset = dnnlib.util.construct_class_by_name(**dataset_args) # subclass of training.datasetDataset
    dataset_sampler = torch_misc.InfiniteSampler(dataset = dataset, rank = rank, num_replicas = num_gpus)
    dataset_iter = iter(torch.utils.data.DataLoader(dataset = dataset, sampler = dataset_sampler, 
        batch_size = batch_size//num_gpus, **dataset_args.loader_args))
    misc.log(f"Num images: {misc.bcolored(len(dataset), 'blue')}", log = log)
    misc.log(f"Image shape: {misc.bcolored(dataset.image_shape, 'blue')}", log = log)
    misc.log(f"Label shape: {misc.bcolored(dataset.label_shape, 'blue')}", log = log)
    return dataset, dataset_iter

# Fetch real images and their corresponding labels, and sample latents/labels
def fetch_data(dataset, dataset_iter, input_shape, drange_net, device, batches_num, batch_size, batch_gpu):
    with torch.autograd.profiler.record_function("data_fetch"):
        real_img, real_c = next(dataset_iter)
        real_img = real_img.to(device).to(torch.float32)
        real_img = misc.adjust_range(real_img, [0, 255], drange_net).split(batch_gpu)
        real_c = real_c.to(device).split(batch_gpu)

        gen_zs = torch.randn([batches_num * batch_size, *input_shape[1:]], device = device)
        gen_zs = [gen_z.split(batch_gpu) for gen_z in gen_zs.split(batch_size)]

        gen_cs = [dataset.get_label(np.random.randint(len(dataset))) for _ in range(batches_num * batch_size)]
        gen_cs = torch.from_numpy(np.stack(gen_cs)).pin_memory().to(device)
        gen_cs = [gen_c.split(batch_gpu) for gen_c in gen_cs.split(batch_size)]

    return real_img, real_c, gen_zs, gen_cs

# Networks (construction/distribution, loading/saving, and printing)
# ----------------------------------------------------------------------------

# Construct networks
def construct_nets(cG, cD, dataset, device, log):
    misc.log("Constructing networks...", "white", log)
    common_kwargs = dict(c_dim = dataset.label_dim, img_resolution = dataset.resolution, img_channels = dataset.num_channels)
    G = dnnlib.util.construct_class_by_name(**cG, **common_kwargs).train().requires_grad_(False).to(device) # subclass of torch.nnnet
    D = dnnlib.util.construct_class_by_name(**cD, **common_kwargs).train().requires_grad_(False).to(device) # subclass of torch.nnnet
    Gs = copy.deepcopy(G).eval()
    return G, D, Gs

# Distribute models onto the GPUs
def distribute_nets(G, D, Gs, device, num_gpus, log):
    misc.log(f"Distributing across {num_gpus} GPUs...", "white", log)
    networks = {}
    for name, net in [("G", G), ("D", D), (None, Gs)]: # ("G_mapping", G.mapping), ("G_synthesis", G.synthesis)
        if (num_gpus > 1) and (net is not None) and len(list(net.parameters())) != 0:
            net.requires_grad_(True)
            net = torch.nn.parallel.DistributedDataParallel(net, device_ids = [device], broadcast_buffers = False, 
                find_unused_parameters = True)
            net.requires_grad_(False)
        if name is not None:
            networks[name] = net
    return networks

# Resume from existing pickle
def load_nets(load_pkl, nets, device, log):
    if (load_pkl is not None) and log:
        misc.log(f"Resuming from {load_pkl}", "white", log)
        resume_data = loader.load_network(load_pkl)

        if nets is not None:
            G, D, Gs = nets
            for name, net in [("G", G), ("D", D), ("Gs", Gs)]:
                torch_misc.copy_params_and_buffers(resume_data[name], net, require_all = False)
        else:
            for net in ["G", "D", "Gs"]:
                resume_data[net] = copy.deepcopy(resume_data[net]).eval().requires_grad_(False).to(device)
            nets = (resume_data["G"], resume_data["D"], resume_data["Gs"])

    return nets

def save_nets(G, D, Gs, cur_nimg, dataset_args, run_dir, distributed, last_snapshots, log):
    snapshot_data = dict(dataset_args = dict(dataset_args))

    for name, net in [("G", G), ("D", D), ("Gs", Gs)]:
        if net is not None:
            if distributed:
                torch_misc.assert_ddp_consistency(net, ignore_regex = r".*\.w_avg")
            net = copy.deepcopy(net).eval().requires_grad_(False).cpu()
        snapshot_data[name] = net
        del net

    snapshot_pkl = os.path.join(run_dir, f"network-snapshot-{cur_nimg//1000:06d}.pkl")
    if log:
        with open(snapshot_pkl, "wb") as f:
            pickle.dump(snapshot_data, f)

        if last_snapshots > 0:
            misc.rm(sorted(glob.glob(os.path.join(run_dir, "network*.pkl")))[:-last_snapshots])

    return snapshot_data, snapshot_pkl

# Print network summary tables
def print_nets(G, D, batch_gpu, device, log):
    if not log:
        return
    z = torch.empty([batch_gpu, *G.input_shape[1:]], device = device)
    c = torch.empty([batch_gpu, *G.cond_shape[1:]], device = device)
    img = torch_misc.print_module_summary(G, [z, c])[0]
    torch_misc.print_module_summary(D, [img, c])

# Training and optimization
# ----------------------------------------------------------------------------

# Initialize cuda according to command line arguments
def init_cuda(rank, cudnn_benchmark, allow_tf32):
    device = torch.device("cuda", rank)
    torch.backends.cudnn.benchmark = cudnn_benchmark    # Improves training speed
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32  # Allow PyTorch to internally use tf32 for matmul
    torch.backends.cudnn.allow_tf32 = allow_tf32        # Allow PyTorch to internally use tf32 for convolutions
    conv2d_gradfix.enabled = True                       # Improves training speed
    return device

# Setup training stages (alternating optimization of G and D, and for )
def setup_training_stages(loss_args, G, cG, D, cD, ddp_nets, device, log):
    misc.log("Setting up training stages...", "white", log)
    loss = dnnlib.util.construct_class_by_name(device = device, **ddp_nets, **loss_args) # subclass of training.loss.Loss
    stages = []

    for name, net, config in [("G", G, cG), ("D", D, cD)]:
        if config.reg_interval is None:
            opt = dnnlib.util.construct_class_by_name(params = net.parameters(), **config.opt_args) # subclass of torch.optimOptimizer
            stages.append(dnnlib.EasyDict(name = f"{name}_both", net = net, opt = opt, interval = 1))
        # Lazy regularization
        else: 
            mb_ratio = config.reg_interval / (config.reg_interval + 1)
            opt_args = dnnlib.EasyDict(config.opt_args)
            opt_args.lr = opt_args.lr * mb_ratio
            opt_args.betas = [beta ** mb_ratio for beta in opt_args.betas]
            opt = dnnlib.util.construct_class_by_name(net.parameters(), **opt_args) # subclass of torch.optimOptimizer
            stages.append(dnnlib.EasyDict(name = f"{name}_main", net = net, opt = opt, interval = 1))
            stages.append(dnnlib.EasyDict(name = f"{name}_reg", net = net, opt = opt, interval = config.reg_interval))

    for stage in stages:
        stage.start_event = None
        stage.end_event = None
        if log:
            stage.start_event = torch.cuda.Event(enable_timing = True)
            stage.end_event = torch.cuda.Event(enable_timing = True)

    return loss, stages

# Compute gradients and update the network weights for the current training stage
def run_training_stage(loss, stage, device, real_img, real_c, gen_z, gen_c, batch_size, batch_gpu, num_gpus):
    # Initialize gradient accumulation
    if stage.start_event is not None:
        stage.start_event.record(torch.cuda.current_stream(device))
    
    stage.opt.zero_grad(set_to_none = True)
    stage.net.requires_grad_(True)

    # Accumulate gradients over multiple rounds
    for round_idx, (x, cx, z, cz) in enumerate(zip(real_img, real_c, gen_z, gen_c)):
        sync = (round_idx == batch_size // (batch_gpu * num_gpus) - 1)
        loss.accumulate_gradients(stage = stage.name, real_img = x, real_c = cx, 
            gen_z = z, gen_c = cz, sync = sync, gain = stage.interval)

    # Update weights
    stage.net.requires_grad_(False)
    with torch.autograd.profiler.record_function(stage.name + "_opt"):
        for param in stage.net.parameters():
            if param.grad is not None:
                torch_misc.nan_to_num(param.grad, nan = 0, posinf = 1e5, neginf=-1e5, out = param.grad)
        stage.opt.step()
    
    if stage.end_event is not None:
        stage.end_event.record(torch.cuda.current_stream(device))

# Update Gs -- the exponential moving average weights copy of G
def update_ema_network(G, Gs, batch_size, cur_nimg, ema_kimg, ema_rampup):
    with torch.autograd.profiler.record_function("Gs"):
        ema_nimg = ema_kimg * 1000

        if ema_rampup is not None:
            ema_nimg = min(ema_nimg, cur_nimg * ema_rampup)
        ema_beta = 0.5 ** (batch_size / max(ema_nimg, 1e-8))

        for p_ema, p in zip(Gs.parameters(), G.parameters()):
            p_ema.copy_(p.lerp(p_ema, ema_beta))

        for b_ema, b in zip(Gs.buffers(), G.buffers()):
            b_ema.copy_(b)

# Evaluate a model over a list of metrics and report the results
def evaluate(Gs, snapshot_pkl, metrics, eval_images_num, dataset_args, num_gpus, rank, device, log, 
        logger = None, run_dir = None, print_progress = True):
    for metric in metrics:
        result_dict = metric_main.compute_metric(metric = metric, max_items = eval_images_num, 
            G = Gs, dataset_args = dataset_args, num_gpus = num_gpus, rank = rank, device = device,
            progress = metric_utils.ProgressMonitor(verbose = log))
        if log:
            metric_main.report_metric(result_dict, run_dir = run_dir, snapshot_pkl = snapshot_pkl)
        if logger is not None:
            logger.metrics.update(result_dict.results)

# Snapshots and logging
# ----------------------------------------------------------------------------

# Initialize image grid, of both real and generated sampled
def init_img_grid(dataset, input_shape, device, run_dir, log): 
    if not log:
        return None, None, None
    grid_size, images, labels = misc.setup_snapshot_img_grid(dataset)
    misc.save_img_grid(images, os.path.join(run_dir, "reals.png"), drange = [0, 255], grid_size = grid_size)
    grid_z = torch.randn([labels.shape[0], *input_shape[1:]], device = device)
    grid_c = torch.from_numpy(labels).to(device)
    return grid_size, grid_z, grid_c

# Save a snapshot of the sampled grid for the given latents/labels
def snapshot_img_grid(Gs, drange_net, grid_z, grid_c, grid_size, batch_gpu, truncation_psi, suffix = "init"):
    images = torch.cat([Gs(z, c, truncation_psi, noise_mode = "const").cpu() for z, c in zip(grid_z.split(batch_gpu), grid_c.split(batch_gpu))]).numpy()
    misc.save_img_grid(images, os.path.join(run_dir, f"fakes_{suffix}.png"), drange = drange_net, grid_size = grid_size)

# Initialize logs (tracking metrics, json log file, tfevent files, etc.)
def init_logger(run_dir, log):
    logger = dnnlib.EasyDict({
        "collector": training_stats.Collector(regex = ".*"), 
        "metrics": {}, 
        "json": None, 
        "tfevents": None
    })

    if log:
        logger.json = open(os.path.join(run_dir, "stats.jsonl"), "wt")
        try:
            import torch.utils.tensorboard as tensorboard
            logger.tfevents = tensorboard.SummaryWriter(run_dir)
        except ImportError as err:
            print("Skipping tfevents export:", err)

    return logger

# Collect statistics from each training stage across the processes/GPUs
def collect_stats(logger, stages):
    for stage in stages:
        value = []
        if (stage.start_event is not None) and (stage.end_event is not None):
            stage.end_event.synchronize()
            value = stage.start_event.elapsed_time(stage.end_event)
        training_stats.report0("Timing/" + stage.name, value)
    logger.collector.update()
    stats = logger.collector.as_dict()
    return stats

# Update the logs (json and tfevents files) with the new info in stats
def update_logger(logger, stats, cur_nimg, start_time):
    timestamp = time.time()
    if logger.json is not None:
        fields = dict(stats, timestamp = timestamp)
        logger.json.write(json.dumps(fields) + "\n")
        logger.json.flush()
    if logger.tfevents is not None:
        global_step = int(cur_nimg / 1e3)
        walltime = timestamp - start_time
        for name, value in stats.items():
            logger.tfevents.add_scalar(name, value.mean, global_step = global_step, walltime = walltime)
        for name, value in logger.metrics.items():
            logger.tfevents.add_scalar(f"Metrics/{name}", value, global_step = global_step, walltime = walltime)
        logger.tfevents.flush()

# Training Loop
# ----------------------------------------------------------------------------
# 1. Sets up the environment and data
# 2. Builds the generator (g) and discriminator (d) networks
# 3. Manages the training process
# 4. Runs periodic evaluations on specified metrics
# 5. Produces sample images over the course of training

def training_loop(
    # General configuration
    train                   = False,    # Training mode
    eval                    = False,    # Evaluation mode
    vis                     = False,    # Visualization mode        
    run_dir                 = ".",      # Output directory
    num_gpus                = 1,        # Number of GPUs participating in the training
    rank                    = 0,        # Rank of the current process in [0, num_gpus]
    cG                      = {},       # Options for generator network
    cD                      = {},       # Options for discriminator network
    # Data
    dataset_args            = {},       # Options for training set
    drange_net              = [-1,1],   # Dynamic range used when feeding image data to the networks
    # Optimization
    loss_args               = {},       # Options for loss function
    total_kimg              = 25000,    # Total length of the training, measured in thousands of real images
    batch_size              = 4,        # Total batch size for one training iteration. Can be larger than batch_gpu * num_gpus
    batch_gpu               = 4,        # Number of samples processed at a time by one GPU
    ema_kimg                = 10.0,     # Half-life of the exponential moving average (EMA) of generator weights
    ema_rampup              = None,     # EMA ramp-up coefficient
    cudnn_benchmark         = True,     # Enable torch.backends.cudnnbenchmark?
    allow_tf32              = False,    # Enable torch.backends.cuda.matmul.allow_tf32 and torch.backends.cudnnallow_tf32?
    # Logging
    resume_pkl              = None,     # Network pickle to resume training from
    resume_kimg             = 0.0,      # Assumed training progress at the beginning
                                        # Affects reporting and training schedule    
    kimg_per_tick           = 8,        # Progress snapshot interval
    img_snapshot_ticks      = 3,        # How often to save image snapshots? None = disable
    network_snapshot_ticks  = 3,        # How often to save network snapshots? None = disable    
    last_snapshots          = 10,       # Maximal number of prior snapshots to save
    printname               = "",       # Experiment name for logging
    # Evaluation
    vis_args                = {},       # Options for vis.vis
    metrics                 = [],       # Metrics to evaluate during training
    eval_images_num         = 50000,    # Sample size for the metrics
    truncation_psi          = 0.7       # Style strength multiplier for the truncation trick (used for visualizations only)
):  
    # Initialize
    start_time = time.time()
    device = init_cuda(rank, cudnn_benchmark, allow_tf32)
    log = (rank == 0)

    dataset, dataset_iter = load_dataset(dataset_args, batch_size, rank, num_gpus, log) # Load training set

    nets = construct_nets(cG, cD, dataset, device, log) if train else None        # Construct networks
    G, D, Gs = load_nets(resume_pkl, nets, device, log)                           # Resume from existing pickle
    print_nets(G, D, batch_gpu, device, log)                                      # Print network summary tables

    if eval:
        misc.log("Run evaluation...", log = log)
        evaluate(Gs, resume_pkl, metrics, eval_images_num, dataset_args, num_gpus, rank, device, log)

    if vis and log:
        misc.log("Produce visualizations...")
        visualize.vis(Gs, dataset, device, batch_gpu, drange_net = drange_net, ratio = dataset.ratio, 
            truncation_psi = truncation_psi, **vis_args)

    if not train:
        exit()

    nets = distribute_nets(G, D, Gs, device, num_gpus, log)                                   # Distribute networks across GPUs
    loss, stages = setup_training_stages(loss_args, G, cG, D, cD, nets, device, log)          # Setup training stages (losses and optimizers)
    grid_size, grid_z, grid_c = init_img_grid(dataset, G.input_shape, device, run_dir, log)   # Initialize an image grid    
    logger = init_logger(run_dir, log)                                                        # Initialize logs

    # Train
    misc.log(f"Training for {total_kimg} kimg...", "white", log)    
    cur_nimg, cur_tick, batch_idx = int(resume_kimg * 1000), 0, 0
    tick_start_nimg, tick_start_time = cur_nimg, time.time()
    stats = None

    while True:
        # Fetch training data
        real_img, real_c, gen_zs, gen_cs = fetch_data(dataset, dataset_iter, G.input_shape, drange_net, 
            device, len(stages), batch_size, batch_gpu)

        # Execute training stages
        for stage, gen_z, gen_c in zip(stages, gen_zs, gen_cs):
            if batch_idx % stage.interval != 0:
                continue
            run_training_stage(loss, stage, device, real_img, real_c, gen_z, gen_c, batch_size, batch_gpu, num_gpus)

        # Update Gs
        update_ema_network(G, Gs, batch_size, cur_nimg, ema_kimg, ema_rampup)

        # Update state
        cur_nimg += batch_size
        batch_idx += 1

        # Perform maintenance tasks once per tick
        done = (cur_nimg >= total_kimg * 1000)
        if (not done) and (cur_tick != 0) and (cur_nimg < tick_start_nimg + kimg_per_tick * 1000):
            continue

        # Print status line and accumulate the info in logger.collector
        tick_end_time = time.time()
        if stats is not None:
            default = dnnlib.EasyDict({'mean': -1})
            fields = []
            fields.append("tick " + misc.bold(f"{training_stats.report0('Progress/tick', cur_tick):<5d}"))
            fields.append("kimg " + misc.bcolored(f"{training_stats.report0('Progress/kimg', cur_nimg / 1e3):<8.1f}", "red"))
            fields.append("")
            fields.append("loss/reg: G (" + misc.bcolored(f"{stats.get('Loss/G/loss', default).mean:>6.3f}", "blue"))
            fields.append(misc.bold(f"{stats.get('Loss/G/reg', default).mean:>6.3f}") + ")")
            fields.append("D "+ misc.bcolored(f"({stats.get('Loss/D/loss', default).mean:>6.3f}", "blue"))
            fields.append(misc.bold(f"{stats.get('Loss/D/reg', default).mean:>6.3f}") + ")")
            fields.append("")
            fields.append("time " + misc.bold(f"{dnnlib.util.format_time(training_stats.report0('Timing/total_sec', tick_end_time - start_time)):<12s}"))
            fields.append(f"sec/kimg {training_stats.report0('Timing/sec_per_kimg', (tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg) * 1e3):<7.2f}")
            fields.append(f"mem: GPU {training_stats.report0('Resources/cpu_mem_gb', psutil.Process(os.getpid()).memory_info().rss / 2**30):<6.2f}")
            fields.append(f"CPU {training_stats.report0('Resources/peak_gpu_mem_gb', torch.cuda.max_memory_allocated(device) / 2**30):<6.2f}")
            fields.append(misc.bold(printname))
            torch.cuda.reset_peak_memory_stats()
            training_stats.report0('Timing/total_hours', (tick_end_time - start_time) / (60 * 60))
            training_stats.report0('Timing/total_days', (tick_end_time - start_time) / (24 * 60 * 60))
            misc.log(" ".join(fields), log = log)

        # Save image snapshot
        if log and (img_snapshot_ticks is not None) and (done or cur_tick % img_snapshot_ticks == 0):
            visualize.vis(Gs, dataset, device, batch_gpu, training = True,
                step = cur_nimg // 1000, grid_size = grid_size, latents = grid_z, 
                labels = grid_c, drange_net = drange_net, ratio = dataset.ratio, **vis_args)

        # Save network snapshot
        if (network_snapshot_ticks is not None) and (done or cur_tick % network_snapshot_ticks == 0):
            snapshot_data, snapshot_pkl = save_nets(G, D, Gs, cur_nimg, dataset_args, run_dir, num_gpus > 1, last_snapshots, log)

            # Evaluate metrics
            evaluate(snapshot_data["Gs"], snapshot_pkl, metrics, eval_images_num,
                dataset_args, num_gpus, rank, device, log, logger, run_dir)
            del snapshot_data

        # Collect stats and update logs
        stats = collect_stats(logger, stages)
        update_logger(logger, stats, cur_nimg, start_time)

        cur_tick += 1
        tick_start_nimg, tick_start_time = cur_nimg, time.time()
        maintenance_time = tick_start_time - tick_end_time
        if done:
            break

    # Done
    misc.log("Done!", "blue")
