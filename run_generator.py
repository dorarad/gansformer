# Produces variety of visualizations:
### Sampling newly generated images
### Style mixing between different images
### Image variation when sampling stochastic local noise given fixed latents
### Impact of local noise in different layers on the output image
### Traversal along to different dimensions close to an initial latent value
### Image interpolation between latents

import argparse
import numpy as np
import PIL.Image
import dnnlib
import dnnlib.tflib as tflib
import re
import sys
import run_projector
import projector
import glob
import os
from training import misc

import pretrained_networks

# Generate images
# ----------------------------------------------------------------------------

def save_gif(imgs, fn):
    imgs[0].save(fn, save_all = True, append_imgs = imgs[1:], duration = 50, loop = 0)  

def broadcastLtnt(Gs, ltnt):
    return np.tile(ltnt[:, np.newaxis], [1, Gs.components.synthesis.input_shape[1], 1])

def generate_imgs(network_pkl, seeds, truncation_psi):
    print("Loading networks from %s..." % network_pkl)
    _G, _D, Gs = pretrained_networks.load_networks(network_pkl)[:3]
    noise_vars = [var for name, var in Gs.components.synthesis.vars.items() if name.startswith("noise")]

    Gs_kwargs = dnnlib.EasyDict()
    Gs_kwargs.output_transform = dict(func = tflib.convert_imgs_to_uint8, nchw_to_nhwc = True)
    Gs_kwargs.randomize_noise = False
    if truncation_psi is not None:
        Gs_kwargs.truncation_psi = truncation_psi

    for seed_idx, seed in enumerate(seeds):
        print("Generating image for seed %d (%d/%d)..." % (seed, seed_idx, len(seeds)))
        rnd = np.random.RandomState(seed)
        z = rnd.randn(1, *Gs.input_shape[1:]) # [minibatch, component]
        tflib.set_vars({var: rnd.randn(*var.shape.as_list()) for var in noise_vars}) # [height, width]
        imgs = Gs.run(z, None, **Gs_kwargs) # [minibatch, height, width, channel]
        misc.to_pil(imgs[0]).save(dnnlib.make_run_dir_path("seed%04d_t%04d.png" % (seed, truncation_psi)))

# Style mixing
# ----------------------------------------------------------------------------

def style_mixing_example(network_pkl, row_seeds, col_seeds, truncation_psi, col_styles, minibatch_size = 4):
    print("Loading networks from %s..." % network_pkl)
    _G, _D, Gs = pretrained_networks.load_networks(network_pkl)[:3]
    w_avg = Gs.get_var("dlatent_avg") # [component]

    Gs_syn_kwargs = dnnlib.EasyDict()
    Gs_syn_kwargs.output_transform = dict(func = tflib.convert_imgs_to_uint8, nchw_to_nhwc = True)
    Gs_syn_kwargs.randomize_noise = False
    Gs_syn_kwargs.minibatch_size = minibatch_size

    print("Generating W vectors...")
    all_seeds = list(set(row_seeds + col_seeds))
    all_z = np.stack([np.random.RandomState(seed).randn(*Gs.input_shape[1:]) for seed in all_seeds]) # [minibatch, component]
    all_w = Gs.components.mapping.run(all_z, None) # [minibatch, layer, component]
    all_w = w_avg + (all_w - w_avg) * truncation_psi # [minibatch, layer, component]
    w_dict = {seed: w for seed, w in zip(all_seeds, list(all_w))} # [layer, component]

    print("Generating images...")
    all_imgs = Gs.components.synthesis.run(all_w, **Gs_syn_kwargs) # [minibatch, height, width, channel]
    img_dict = {(seed, seed): img for seed, img in zip(all_seeds, list(all_imgs))}

    print("Generating style-mixed images...")
    for row_seed in row_seeds:
        for col_seed in col_seeds:
            w = w_dict[row_seed].copy()
            w[col_styles] = w_dict[col_seed][col_styles]
            img = Gs.components.synthesis.run(w[np.newaxis], **Gs_syn_kwargs)[0]
            img_dict[(row_seed, col_seed)] = img

    print("Saving images...")
    for (row_seed, col_seed), img in img_dict.items():
        misc.to_pil(img).save(dnnlib.make_run_dir_path("%d-%d.png" % (row_seed, col_seed)))

    print("Saving image grid...")
    _N, _C, H, W = Gs.output_shape
    canvas = PIL.Image.new("RGB", (W * (len(col_seeds) + 1), H * (len(row_seeds) + 1)), "black")
    for row_idx, row_seed in enumerate([None] + row_seeds):
        for col_idx, col_seed in enumerate([None] + col_seeds):
            if row_seed is None and col_seed is None:
                continue
            key = (row_seed, col_seed)
            if row_seed is None:
                key = (col_seed, col_seed)
            if col_seed is None:
                key = (row_seed, row_seed)
            canvas.paste(misc.to_pil(img_dict[key]), (W * col_idx, H * row_idx))
    canvas.save(dnnlib.make_run_dir_path("grid.png"))

# Noise variation
# ----------------------------------------------------------------------------

def generate_noisevar_imgs(network_pkl, seeds, num_samples, num_variants):
    print("Loading networks from %s..." % network_pkl)
    _G, _D, Gs = pretrained_networks.load_networks(network_pkl)[:3]
    Gs_kwargs = dnnlib.EasyDict()
    Gs_kwargs.truncation_psi = 1
    Gs_kwargs.output_transform = dict(func = tflib.convert_imgs_to_uint8, nchw_to_nhwc = True)
    Gs_kwargs.minibatch_size = 4
    _, _, H, W = Gs.output_shape
      
    for seed_idx, seed in enumerate(seeds):
        print("Generating image for seed %d (%d/%d)..." % (seed, seed_idx, len(seeds)))
        canvas = PIL.Image.new("RGB", (W * (num_variants + 2), H), "white")

        z = np.stack([np.random.RandomState(seed).randn(Gs.input_shape[1])] * num_samples) # [minibatch, component]
        imgs = Gs.run(z, None, **Gs_kwargs) # [minibatch, height, width, channel]

        npimgs = imgs
        imgs = [misc.to_pil(img) for img in imgs]
        
        save_gif(imgs, dnnlib.make_run_dir_path("noisevar%04d.gif" % seed))
        for i in range(num_variants + 1):
            canvas.paste(imgs[i], (i * W, 0))

        diff = np.std(np.mean(npimgs, axis = 3), axis = 0) * 4
        diff = np.clip(diff + 0.5, 0, 255).astype(np.uint8)            
        canvas.paste(PIL.Image.fromarray(diff, "L"), (W * (num_variants + 1), 0))

        canvas.save(dnnlib.make_run_dir_path("noisevar%04d.png" % seed))

# Noise components
# ----------------------------------------------------------------------------

def generate_noisecomp_imgs(network_pkl, seeds, noise_ranges):
    print("Loading networks from %s..." % network_pkl)
    _G, _D, Gs = pretrained_networks.load_networks(network_pkl)[:3]
    Gsc = Gs.clone()
    
    noise_vars = [var for name, var in Gsc.components.synthesis.vars.items() if name.startswith("noise")]
    noise_pairs = list(zip(noise_vars, tflib.run(noise_vars))) # [(var, val), ...]

    Gs_kwargs = dnnlib.EasyDict()
    Gs_kwargs.output_transform = dict(func = tflib.convert_imgs_to_uint8, nchw_to_nhwc = True)
    Gs_kwargs.randomize_noise = False
    Gs_kwargs.truncation_psi = 1

    _, _, H, W = Gs.output_shape
    for seed_idx, seed in enumerate(seeds):
        print("Generating images for seed %d (%d/%d)..." % (seed, seed_idx, len(seeds)))
        canvas = PIL.Image.new("RGB", (W * len(noise_ranges), H), "white")

        z = np.random.RandomState(seed).randn(1, *Gsc.input_shape[1:]) # [minibatch, component]
        
        for i, noise_range in enumerate(noise_ranges):
            tflib.set_vars({var: val * (1 if vi in noise_range else 0) for vi, (var, val) in enumerate(noise_pairs)})
            imgs = Gsc.run(z, None, **Gs_kwargs) # [minibatch, height, width, channel]
            canvas.paste(misc.to_pil(imgs[0]), (i * W, 0))
        
        canvas.save(dnnlib.make_run_dir_path("noisecomp%04d.png" % seed))

# Latents sensitivity
# ----------------------------------------------------------------------------

def generate_ltntprtrb_imgs(network_pkl, seeds, num_samples, noise_range, dltnt, group_size):
    print("Loading networks from %s..." % network_pkl)
    _G, _D, Gs = pretrained_networks.load_networks(network_pkl)[:3]
    _, _, H, W = Gs.output_shape

    noise_vars = [var for name, var in Gs.components.synthesis.vars.items() if name.startswith("noise")]

    Gs_kwargs = dnnlib.EasyDict()
    Gs_kwargs.output_transform = dict(func = tflib.convert_imgs_to_uint8, nchw_to_nhwc = True)
    Gs_kwargs.randomize_noise = False
    Gs_kwargs.truncation_psi = 1
    Gs_kwargs.minibatch_size = 4

    ltnt_size = Gs.components.synthesis.input_shape[2] if dltnt else Gs.input_shape[1]

    stds = np.ones(ltnt_size)
    if dltnt:
        samples = np.random.randn(num_samples, Gs.input_shape[1]) # [minibatch, component]
        dlanents = Gs.components.mapping.run(samples, None, minibatch_size = 32)[:, 0]
        stds = np.std(dlanents, axis = 1)

    for seed_idx, seed in enumerate(seeds):
        print("Generating image for seed %d (%d/%d)..." % (seed, seed_idx, len(seeds)))
        rnd = np.random.RandomState(seed)

        if seed_idx == 0 and dltnt:
            ltnt = Gs.get_var("dlatent_avg")[np.newaxis]
        else:
            ltnt = rnd.randn(1, Gs.input_shape[1])
            if dltnt:
                ltnt = Gs.components.mapping.run(ltnt, None)[:, 0]

        ltnt = np.tile(ltnt, (ltnt_size * len(noise_range), 1)) 
        tflib.set_vars({var: rnd.randn(*var.shape.as_list()) for var in noise_vars}) # [height, width]

        idx = 0
        for j in range(ltnt_size):
            for r in noise_range:
                ltnt[idx, j] += r * stds[j]
                idx += 1

        if dltnt:
            imgs = Gs.components.synthesis.run(broadcastLtnt(Gs, ltnt), **Gs_kwargs)
        else:
            imgs = Gs.run(ltnt, None, **Gs_kwargs) # [minibatch, height, width, channel]
        
        num_crops = int(ltnt_size / group_size)
        frameIdx, imgIdx = 0, 0
        for m in range(num_crops):
            canvas = PIL.Image.new("RGB", (W * len(noise_range), H * group_size), "white")
            
            for i in range(group_size):
                imgStart = frameIdx
                for j in range(len(noise_range)):
                    canvas.paste(misc.to_pil(imgs[frameIdx]), (j * W, i * H))
                    frameIdx += 1  

                imgsOut = [misc.to_pil(img) for img in imgs[imgStart:frameIdx]]
                save_gif(imgsOut, dnnlib.make_run_dir_path("latent_perturbation_%s_%04d_%04d.gif" % \
                    (("w" if dltnt else "z"), seed, imgIdx)))
                imgIdx += 1

            canvas.save(dnnlib.make_run_dir_path("latent_perturbation_%s_%04d_%04d.png" % \
                (("w" if dltnt else "z"), seed, m)))

# Interpolations
# ----------------------------------------------------------------------------

# Linear interpolation
def lerp(a, b, ts):
    ts = ts[:, np.newaxis]
    return a + (b - a) * ts

def normalize(v):
    return v / np.linalg.norm(v, axis = -1, keepdims = True)

# Spherical interpolation
def slerp(a, b, ts):
    ts = ts[:, np.newaxis]
    a = normalize(a)
    b = normalize(b)
    d = np.sum(a * b)
    p = ts * np.arccos(d)[np.newaxis]
    c = normalize(b - d * a)
    d = a * np.cos(p) + c * np.sin(p)
    return normalize(d)

def projectImage(Gs, projc, img_fn):
    img = PIL.Image.open(img_fn).convert("RGB")
    img = misc.pad_min_square(img)
    img = img.resize((Gs.output_shape[2], Gs.output_shape[3]), PIL.Image.ANTIALIAS) # resize mode? 
    img = np.asarray(img)

    channels = img.shape[1] if img.ndim == 3 else 1
    img = img[np.newaxis, :, :] if channels == 1 else img.transpose([2, 0, 1]) # HW => CHW # HWC => CHW

    assert img.shape == tuple(Gs.output_shape[1:])
    img = misc.adjust_dynamic_range(img, [0, 255], [-1, 1])[np.newaxis]
    return run_projector.project_img(projc, img) # w.append()       

def interpolate(network_pkl, seeds, dltnt, img_dir, samples_num, loss, lr):
    print("Loading networks from %s..." % network_pkl)
    _G, _D, Gs = pretrained_networks.load_networks(network_pkl)[:3]
    noise_vars = [var for name, var in Gs.components.synthesis.vars.items() if name.startswith("noise")]
    interp_range = np.linspace(0.0, 1.0, num = samples_num + 1, endpoint = True)
    ts = np.array(interp_range)
    proj = img_dir is not None

    mod = "w" if dltnt else "z"
    if proj: 
        mod = "p{}".format(img_dir.split("/")[-1])
        dltnt = True

    Gs_kwargs = dnnlib.EasyDict()
    Gs_kwargs.output_transform = dict(func = tflib.convert_imgs_to_uint8, nchw_to_nhwc = True)
    Gs_kwargs.randomize_noise = False
    Gs_kwargs.truncation_psi = 1
    Gs_kwargs.minibatch_size = 4

    # Start interpolation from input images that will be first projected to the latent space
    if proj:
        tflib.set_vars({var: np.random.randn(*var.shape.as_list()) for var in noise_vars}) # [height, width]

        projc = projector.Projector()
        projc.num_steps = 7000
        projc.lossType = loss
        projc.initial_learning_rate = lr
        projc.set_network(Gs)
        
        print("Loading images from %s" % img_dir)
        img_fns = sorted(glob.glob(os.path.join(img_dir, "*")))
        seeds = range(len(img_fns[1:]))

        if len(img_fns) == 0:
            print("Error: No input images found")
            sys.exit(1)

        imgControl = img_fns[0]
        w0 = projectImage(Gs, projc, imgControl)

    for seed_idx, seed in enumerate(seeds):
        print("Generating image for seed %d (%d/%d)..." % (seed, seed_idx, len(seeds)))
        rnd = np.random.RandomState(seed)        

        if not proj:
            tflib.set_vars({var: rnd.randn(*var.shape.as_list()) for var in noise_vars}) # [height, width]
            z = rnd.randn(2, *Gs.input_shape[1:]) # [minibatch, component]
        
        # interpolate either in w space (if dltnt) or z space
        if dltnt:
            if proj:
                w = [w0, projectImage(Gs, projc, img_fns[seed_idx + 1])]
            else:
                w = Gs.components.mapping.run(z, None)[:, 0]
            w = lerp(w[0], w[1], ts)
            imgs = Gs.components.synthesis.run(broadcastLtnt(Gs, w), **Gs_kwargs)
        else:
            z = slerp(z[0], z[1], ts)
            imgs = Gs.run(z, None, **Gs_kwargs) # [minibatch, height, width, channel]
        
        imgs = [misc.to_pil(img) for img in imgs]
        save_gif(imgs, dnnlib.make_run_dir_path("interpolation_%s_%04d.gif" % (mod, seed)))        

def _parse_num_range(s):
    # Accept either a comma separated list of numbers "a,b,c" or a range "a-c" and return as a list of ints
    range_re = re.compile(r"^(\d+)-(\d+)$")
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2))+1))
    vals = s.split(",")
    return [int(x) for x in vals]

def _parse_num_ranges(ls):
    return [_parse_num_range(s) for s in ls.split(",")]

_examples = """examples:

  # Generate ffhq uncurated images (matches paper Figure 12)
  python %(prog)s generate-imgs --network = gdrive:networks/stylegan2-ffhq-config-f.pkl --seeds = 6600-6625 --truncation-psi = 0.5

  # Generate ffhq curated images (matches paper Figure 11)
  python %(prog)s generate-imgs --network = gdrive:networks/stylegan2-ffhq-config-f.pkl --seeds = 66,230,389,1518 --truncation-psi = 1.0

  # Generate uncurated car images (matches paper Figure 12)
  python %(prog)s generate-imgs --network = gdrive:networks/stylegan2-car-config-f.pkl --seeds = 6000-6025 --truncation-psi = 0.5

  # Generate style mixing example (matches style mixing video clip)
  python %(prog)s style-mixing-example --network = gdrive:networks/stylegan2-ffhq-config-f.pkl --row-seeds = 85,100,75,458,1500 --col-seeds = 55,821,1789,293 --truncation-psi = 1.0
"""

# ----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description = """StyleGAN2 generator.

Run "python %(prog)s <subcommand> --help" for subcommand help.""",
        epilog = _examples,
        formatter_class = argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(help = "Sub-commands", dest = "command")

    parser_generate_imgs = subparsers.add_parser("generate-images",     help = "Generate images")
    parser_generate_imgs.add_argument("--network",                      help = "Network pickle filename", dest = "network_pkl", required = True)
    parser_generate_imgs.add_argument("--seeds",                        help = "List of random seeds", type = _parse_num_range, required = True)
    parser_generate_imgs.add_argument("--truncation-psi",               help = "Truncation psi (default: %(default)s)", default = 0.5, type = float)
    parser_generate_imgs.add_argument("--result-dir",                   help = "Root directory for run results (default: %(default)s)", default = "results", metavar = "DIR")

    parser_style_mixing_example = subparsers.add_parser("style-mixing", help = "Generate style mixing video")
    parser_style_mixing_example.add_argument("--network",               help = "Network pickle filename", dest = "network_pkl", required = True)
    parser_style_mixing_example.add_argument("--row-seeds",             help = "Random seeds to use for image rows", type = _parse_num_range, required = True)
    parser_style_mixing_example.add_argument("--col-seeds",             help = "Random seeds to use for image columns", type = _parse_num_range, required = True)
    parser_style_mixing_example.add_argument("--col-styles",            help = "Style layer range (default: %(default)s)", type = _parse_num_range, default = "0-6")
    parser_style_mixing_example.add_argument("--truncation-psi",        help = "Truncation psi (default: %(default)s)", default = 0.5, type = float)
    parser_style_mixing_example.add_argument("--result-dir",            help = "Root directory for run results (default: %(default)s)", default = "results", metavar = "DIR")

    parser_noisevar_imgs = subparsers.add_parser("noise-variance")
    parser_noisevar_imgs.add_argument("--network",                      help = "Network pickle filename", dest = "network_pkl", required = True)
    parser_noisevar_imgs.add_argument("--seeds",                        help = "List of random seeds", type = _parse_num_range, required = True)
    parser_noisevar_imgs.add_argument("--num-samples",                  help = "Number of sampled image to compute noise variance (default: %(default)s)", default = 100, type = int)
    parser_noisevar_imgs.add_argument("--num-variants",                 help = "Number of samples to show (default: %(default)s)", default = 8, type = int)
    parser_noisevar_imgs.add_argument("--result-dir",                   help = "Root directory for run results (default: %(default)s)", default = "results", metavar = "DIR")

    parser_noisecomp_imgs = subparsers.add_parser("noise-samples")
    parser_noisecomp_imgs.add_argument("--network",                     help = "Network pickle filename", dest = "network_pkl", required = True)
    parser_noisecomp_imgs.add_argument("--seeds",                       help = "List of random seeds", type = _parse_num_range, required = True)
    parser_noisecomp_imgs.add_argument("--noise-ranges",                help = "Ranges of layers noise is applied to", type = _parse_num_ranges, default = [range(0, 18), range(0, 0), range(8, 18), range(0, 8)])
    parser_noisecomp_imgs.add_argument("--result-dir",                  help = "Root directory for run results (default: %(default)s)", default = "results", metavar = "DIR")

    parser_ltntprtrb_imgs = subparsers.add_parser("latent-perturbation")
    parser_ltntprtrb_imgs.add_argument("--network",                     help = "Network pickle filename", dest = "network_pkl", required = True)
    parser_ltntprtrb_imgs.add_argument("--seeds",                       help = "List of random seeds", type = _parse_num_range, required = True)
    parser_ltntprtrb_imgs.add_argument("--num-samples",                 help = "Number of sampled image to w latent space variance (default: %(default)s)", default = 1000, type = int)
    parser_ltntprtrb_imgs.add_argument("--noise-range",                 help = "Range of perturbation for any latent dimension (default: %(default)s)", default = np.linspace(0.0, 5.0, num = 41, endpoint = True), type = float, nargs = "*")
    parser_ltntprtrb_imgs.add_argument("--dltnt",                       help = "Whether to explore perturbations in the z or w space (before of after the mapping network) (default: %(default)s)", default = False, action = "store_true")
    parser_ltntprtrb_imgs.add_argument("--group-size",                  help = "Number of images to group in one table (default: %(default)s)", default = 8, type = int)
    parser_ltntprtrb_imgs.add_argument("--result-dir",                  help = "Root directory for run results (default: %(default)s)", default = "results", metavar = "DIR")

    parser_interpolate_imgs = subparsers.add_parser("interpolate",      help = "Generate images")
    parser_interpolate_imgs.add_argument("--network",                   help = "Network pickle filename", dest = "network_pkl", required = True)
    parser_interpolate_imgs.add_argument("--seeds",                     help = "List of random seeds", type = _parse_num_range) # , required = True
    parser_interpolate_imgs.add_argument("--dltnt",                     help = "Whether to explore perturbations in the z or w space (before of after the mapping network) (default: %(default)s)", default = False, action = "store_true")
    parser_interpolate_imgs.add_argument("--img-dir",                   help = "A directory of source images to perform interpolation between", default = None, type = str)
    parser_interpolate_imgs.add_argument("--samples-num",               help = "Number of samples when interpolating between two image endpoints", default = 200, type = float)
    parser_interpolate_imgs.add_argument("--loss",                      help = "Loss to be used then projecting a source image to the latent space (default: %(default)s)", default = "l1", type = str)
    parser_interpolate_imgs.add_argument("--lr",                        help = "Learning rate to be used then projecting a source image to the latent space (default: %(default)s)", default = 0.1, type = floata)
    parser_interpolate_imgs.add_argument("--result-dir",                help = "Root directory for run results (default: %(default)s)", default = "results", metavar = "DIR")

    args = parser.parse_args()
    kwargs = dnnlib.EasyDict(vars(args))
    subcmd = kwargs.pop("command")

    if subcmd is None:
        print ("Error: missing subcommand. Re-run with --help for usage.")
        sys.exit(1)

    sc = dnnlib.SubmitConfig()
    sc.num_gpus = 1
    sc.submit_target = dnnlib.SubmitTarget.LOCAL
    sc.local.do_not_copy_source_files = True
    sc.run_desc = sc.run_dir_root = kwargs.pop("result_dir")

    func_name_map = {
        "generate-images": "run_generator.generate_imgs",
        "style-mixing": "run_generator.style_mixing_example",
        "noise-variance": "run_generator.generate_noisevar_imgs",
        "noise-samples": "run_generator.generate_noisecomp_imgs",
        "latent-perturbation": "run_generator.generate_ltntprtrb_imgs",
        "interpolate": "run_generator.interpolate"
    }
    dnnlib.submit_run(sc, func_name_map[subcmd], **kwargs)

if __name__ == "__main__":
    main()
