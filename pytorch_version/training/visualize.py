import shutil
import numpy as np
import PIL.Image

from tqdm import tqdm, trange
from training import misc
import os
import torch

# Compute the effective batch size given a total number of elements, the batch index, and the
# batch size. For the last batch, its effective size might be smaller if the total_num is not
# evenly divided by the batch_size.
def curr_section_size(total_num, idx, batch_size):
    start = idx * batch_size
    end = min((idx + 1) * batch_size, total_num)
    return end - start

def run(G, zs, cs, batch_size, truncation_psi, **kwargs):
    if cs.shape[0] == 1:
        cs = cs.repeat(zs.shape[0], 1)
    outs = [G(z, c, truncation_psi = truncation_psi, **kwargs) for z, c in zip(zs.split(batch_size), cs.split(batch_size))]
    outs = zip(*outs)
    outs = [torch.cat([batch.cpu() for batch in out]).numpy() for out in outs]
    return outs

# Mathematical utilities
# ----------------------------------------------------------------------------

# Linear interpolation
def lerp(a, b, ts):
    ts = ts[:, None]
    return a + (b - a) * ts

def normalize(v):
    return v / v.norm(dim = -1, keepdim = True)

# Spherical interpolation
def slerp(a, b, ts):
    ts = ts[:, None]
    a = normalize(a)
    b = normalize(b)
    d = (a * b).sum(dim = -1, keepdim = True)
    p = ts * torch.acos(d)
    c = normalize(b - d * a)
    d = a * torch.cos(p) + c * torch.sin(p)
    return normalize(d)

# Evaluate: generate variety of samples
# ----------------------------------------------------------------------------

## Creates variety of visualizations. Supported types
# imgs           : Save image samples
# ltnts          : Save latent vectors
# maps           : Save attention maps (for GANformer only)
# layer_maps     : Save attention maps for all layers (for GANformer only)
# interpolations : Create latent interpolations
# noise_var      : Create noise variation visualization
# style_mix      : Create style mixing visualization
def vis(G,
    dataset,                          # The dataset object for accessing the data
    device,                           # Device to run visualization on
    batch_size,                       # Visualization batch size
    run_dir             = ".",        # Output directory
    training            = False,      # Training mode
    latents             = None,       # Source latents to generate images from
    labels              = None,       # Source labels to generate images from (0 if no labels are used)
    ratio               = 1.0,        # Image height/width ratio in the dataset
    truncation_psi      = 0.7,        # Style strength multiplier for the truncation trick (used for visualizations only)
    # Model settings
    k                   = 1,          # Number of components the model has
    drange_net          = [-1,1],     # Model image output range
    attention           = False,      # Whereas the model produces attention maps (for visualization)
    num_heads           = 1,          # Number of attention heads
    # Visualization settings
    vis_types           = None,       # Visualization types to be created
    num                 = 100,        # Number of produced samples
    rich_num            = 5,          # Number of samples for which richer visualizations will be created
                                      # (requires more memory and disk space, and therefore rich_num <= num)
    grid                = None,       # Whether to save the samples in one large grid files
                                      # or in separated files one per sample
    grid_size           = None,       # Grid proportions (w, h)
    step                = None,       # Step number to be used in visualization filenames
    verbose             = None,       # Verbose print progress messages
    keep_samples        = True,       # Keep all prior samples during training
    # Visualization-specific settings
    alpha               = 0.3,        # Proportion for generated images and attention maps blends
    intrp_density       = 8,          # Number of samples in between two end points of an interpolation
    intrp_per_component = False,      # Whether to perform interpolation along particular latent components (True)
                                      # or all of them at once (False)
    noise_samples_num   = 100,        # Number of samples used to compute noise variation visualization
    section_size        = 100):       # Visualization section size (section_size <= num) for reducing memory footprint
    
    def prefix(step): return "" if step is None else f"{step:06d}_"
    def pattern_of(dir, step, suffix): return f"{run_dir}/visuals/{dir}/{prefix(step)}%06d.{suffix}"

    # Set default options
    if verbose is None: verbose = not training # Disable verbose during training
    if grid is None: grid = training # Save image samples in one grid file during training    
    if grid_size is not None: section_size = rich_num = num = np.prod(grid_size) # If grid size is provided, set images number accordingly

    _labels, _latents = labels, latents
    if _latents is not None: assert num == _latents.shape[0]
    if _labels  is not None: assert num == _labels.shape[0]
    assert rich_num <= section_size

    vis = vis_types
    # For time efficiency, during training save only image and map samples rather than richer visualizations
    if training:
        vis = {"imgs"} # , "maps"
        # if num_heads == 1:
        #    vis.add("layer_maps")
    else:
        vis = vis or {"imgs", "maps", "ltnts", "interpolations", "noise_var"}

    # Build utility functions
    save_images = misc.save_images_builder(drange_net, ratio, grid_size, grid, verbose)
    save_blends = misc.save_blends_builder(drange_net, ratio, grid_size, grid, verbose, alpha)

    crange = trange if verbose else range
    section_of = lambda a, i, n: a[i * n: (i + 1) * n]

    get_rnd_latents = lambda n: torch.randn([n, *G.input_shape[1:]], device = device)
    get_rnd_labels = lambda n: torch.from_numpy(dataset.get_random_labels(n)).to(device)

    # Create directories
    dirs = []
    if "imgs" in vis:            dirs += ["images"]
    if "ltnts" in vis:           dirs += ["latents-z", "latents-w"]
    if "maps" in vis:            dirs += ["maps", "softmaps", "blends", "softblends"]
    if "layer_maps" in vis:      dirs += ["layer_maps"]
    if "interpolations" in vis:  dirs += ["interpolations-z", "interpolation-w"]

    if not keep_samples:
        shutil.rmtree(f"{run_dir}/visuals")
    for dir in dirs:
        os.makedirs(f"{run_dir}/visuals/{dir}", exist_ok = True)

    if verbose:
        print("Running network and saving samples...")
    
    # Produce visualizations
    for idx in crange(0, num, section_size):
        curr_size = curr_section_size(num, idx, section_size)

        # Compute source latents/labels that images will be produced from
        latents = get_rnd_latents(curr_size) if _latents is None else section_of(_latents, idx, section_size)
        labels  = get_rnd_labels(curr_size)  if _labels  is None else section_of(_labels,  idx, section_size)
        if idx == 0:
            latents0, labels0 = latents, labels
        # Run network over latents and produce images and attention maps
        ret = run(G, latents, labels, batch_size, truncation_psi, noise_mode = "const", 
            return_att = True, return_ws = True)
        # For memory efficiency, save full information only for a small amount of images
        images, attmaps_all_layers, wlatents_all_layers = ret
        soft_maps = attmaps_all_layers[:,:,-1,0] if attention else None
        attmaps_all_layers = attmaps_all_layers[:rich_num]
        wlatents = wlatents_all_layers[:,:,0]
        # Save image samples
        if "imgs" in vis:
            save_images(images, pattern_of("images", step, "png"), idx)

        # Save latent vectors
        if "ltnts" in vis:
            misc.save_npys(latents, pattern_of("latents-z", step, "npy"), verbose, idx)
            misc.save_npys(wlatents, pattern_of("latents-w", step, "npy"), verbose, idx)

        # For the GANformer model, save attention maps
        if attention:
            if "maps" in vis:
                pallete = np.expand_dims(misc.get_colors(k - 1), axis = [2, 3])
                maps = (soft_maps == np.amax(soft_maps, axis = 1, keepdims = True)).astype(float)

                soft_maps = np.sum(pallete * np.expand_dims(soft_maps, axis = 2), axis = 1)
                maps = np.sum(pallete * np.expand_dims(maps, axis = 2), axis = 1)

                save_images(soft_maps, pattern_of("softmaps", step, "png"), idx)
                save_images(maps, pattern_of("maps", step, "png"), idx)

                save_blends(soft_maps, images, pattern_of("softblends", step, "png"), idx)
                save_blends(maps, images, pattern_of("blends", step, "png"), idx)

            # Save maps from all attention heads and layers
            # (for efficiency, only for a small number of images)
            if "layer_maps" in vis:
                all_maps = []
                maps_fakes = np.split(attmaps_all_layers, attmaps_all_layers.shape[2], axis = 2)
                for layer, lmaps in enumerate(maps_fakes):
                    lmaps = np.split(np.squeeze(lmaps, axis = 2), lmaps.shape[2], axis = 2)
                    for head, hmap in enumerate(lmaps):
                        hmap = (hmap == np.amax(hmap, axis = 1, keepdims = True)).astype(float)
                        hmap = np.sum(pallete * hmap, axis = 1)
                        all_maps.append((hmap, f"l{layer}_h{head}"))

                if not grid:
                    for i in range(rich_num):
                        stepdir = "" if step is None else (f"/{step:06d}")
                        os.makedirs(f"{run_dir}/visuals/layer_maps/%06d" % i + stepdir, exist_ok = True)
                for maps, name in all_maps:
                    if grid:
                        pattern = f"{run_dir}/visuals/layer_maps/{prefix(step)}%06d-{name}.png"
                    else:
                        pattern = f"{run_dir}/visuals/layer_maps/%06d/{stepdir}/{name}.png"
                    save_images(maps, pattern, idx)

    # Produce interpolations between pairs or source latents
    # In the GANformer case, varying one component at a time
    if "interpolations" in vis:
        ts = torch.linspace(0.0, 1.0, steps = intrp_density)

        if verbose:
            print("Generating interpolations...")
        for i in crange(rich_num):
            os.makedirs(f"{run_dir}/visuals/interpolations-z/%06d" % i, exist_ok = True)
            os.makedirs(f"{run_dir}/visuals/interpolations-w/%06d" % i, exist_ok = True)

            z = get_rnd_latents(2)
            z[0] = latents0[i]
            c = labels0[i:i+1]
            
            w = run(G, z, c, batch_size, truncation_psi, noise_mode = "const", return_ws = True)[-1]

            def update(t, fn, ts, dim):
                if dim == 3:
                    ts = ts[:, None]
                t_ups = []

                if intrp_per_component:
                    for c in range(k - 1):
                        # copy over all the components except component c that will get interpolated
                        t_up = torch.clone(t[0]).unsqueeze(0).repeat((intrp_density, ) + tuple([1] * dim))
                        # interpolate component c
                        t_up[:,c] = fn(t[0, c], t[1, c], ts)
                        t_ups.append(t_up)

                    t_up = torch.cat(t_ups, dim = 0)
                else:
                    t_up = fn(t[0], t[1], ts.unsqueeze(1))

                return t_up

            z_up = update(z, slerp, ts, 2)
            w_up = update(w, lerp, ts, 3)

            imgs1 = run(G, z_up, c, batch_size, truncation_psi, noise_mode = "const")[0]
            imgs2 = run(G, w_up, c, batch_size, truncation_psi, noise_mode = "const", take_w = True)[0]

            def save_interpolation(imgs, name):
                imgs = np.split(imgs, k - 1, axis = 0)
                for c in range(k - 1):
                    filename = f"{run_dir}/visuals/interpolations-{name}/{i:06d}/{c:02d}"
                    imgs[c] = [misc.to_pil(img, drange = drange_net) for img in imgs[c]]
                    imgs[c][-1].save(f"{filename}.png")
                    misc.save_gif(imgs[c], f"{filename}.gif")

            save_interpolation(imgs1, "z")
            save_interpolation(imgs2, "w")

    # Compute noise variance map
    # Shows what areas vary the most given fixed source
    # latents due to the use of stochastic local noise
    if "noise_var" in vis:
        if verbose:
            print("Generating noise variance...")

        z = get_rnd_latents(1).repeat(noise_samples_num, 1, 1)
        c = get_rnd_labels(1)
        imgs = run(G, z, c, batch_size, truncation_psi)[0]
        imgs = np.stack([misc.to_pil(img, drange = drange_net) for img in imgs], axis = 0)
        diff = np.std(np.mean(imgs, axis = 3), axis = 0) * 4
        diff = np.clip(diff + 0.5, 0, 255).astype(np.uint8)
        PIL.Image.fromarray(diff, "L").save(f"{run_dir}/visuals/noise-variance.png")

    # Compute style mixing table, varying using the latent A in some of the layers and latent B in rest.
    # For the GANformer, also produce component mixes (using latents from A in some of the components,
    # and latents from B in the rest.
    if "style_mix" in vis:
        if verbose:
            print("Generating style mixes...")
        cols, rows = 4, 2
        row_lens = np.array([2, 5, 8, 11])
        c = get_rnd_labels(1)

        # Create latent mixes
        mixes = {
            "layer": (np.arange(wlatents_all_layers.shape[2]) < row_lens[:,None]).astype(np.float32)[:,None,None,None,:,None],
            "component": (np.arange(wlatents_all_layers.shape[1]) < row_lens[:,None]).astype(np.float32)[:,None,None,:,None,None]
        }
        ws = wlatents_all_layers[:cols+rows]
        orig_imgs = images[:cols+rows]
        col_z = wlatents_all_layers[:cols][None, None]
        row_z = wlatents_all_layers[cols:cols+rows][None,:,None]

        for name, mix in mixes.items():
            # Produce image mixes
            mix_z = mix * row_z  + (1 - mix) * col_z
            mix_z = torch.from_numpy(np.reshape(mix_z, [-1, *wlatents_all_layers.shape[1:]])).to(device)
            mix_imgs = run(G, mix_z, c, batch_size, truncation_psi, noise_mode = "const", take_w = True)[0]
            mix_imgs = np.reshape(mix_imgs, [len(row_lens) * rows, cols, *mix_imgs.shape[1:]])

            # Create image table canvas
            H, W = mix_imgs.shape[-2:]
            canvas = PIL.Image.new("RGB", (W * (cols + 1), H * (len(row_lens) * rows + 1)), "black")

            # Place image mixes respectively at each position (row_idx, col_idx)
            for row_idx, row_elem in enumerate([None] + list(range(len(row_lens) * rows))):
                for col_idx, col_elem in enumerate([None] + list(range(cols))):
                    if (row_elem, col_elem) == (None, None):  continue
                    if row_elem is None:                    img = orig_imgs[col_elem]
                    elif col_elem is None:                  img = orig_imgs[cols + (row_elem % rows)]
                    else:                                   img = mix_imgs[row_elem, col_elem]

                    canvas.paste(misc.to_pil(img, drange = drange_net), (W * col_idx, H * row_idx))

            canvas.save(f"{run_dir}/visuals/{name}-mixing.png")

    if verbose:
        misc.log("Visualizations Completed!", "blue")
