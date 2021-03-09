
def evaluate():
    def save_imgs(imgs, path, counter = 0):
        imgs = list(imgs)
        for i, img in tqdm(enumerate(imgs), total = len(imgs)):
            misc.convert_to_pil_img(img, drange = drange_net).save(dnnlib.make_run_dir_path(path % (counter + i)))

    def save_bimgs(imgs, bimgs, path, counter = 0, alpha = 0.3, drange = drange_net):
        img_pairs = zip(list(imgs), list(bimgs))
        for i, (img, bimg) in tqdm(enumerate(img_pairs), total = len(imgs)):
            img = misc.convert_to_pil_img(img, drange = drange)
            bimg = misc.convert_to_pil_img(bimg, drange = drange)
            PIL.Image.blend(bimg, img, alpha = alpha).save(dnnlib.make_run_dir_path(path % (counter + i)))

    def save_npys(npys, path, counter = 0):
        npys = list(npys)
        for i, npy in tqdm(enumerate(npys), total = len(npys)):
            npsave(dnnlib.make_run_dir_path(path % (counter + i)), npy)

    print("Making dirs...")
    for dirname in ["gen", "vmap", "map", "smap", "vsmap", "vlmap", "lmap", "comps", "intr_z", "intr_w",
        "latents", "dlatents", "prtrb_z", "prtrb_w"]:
        misc.mkdir(dnnlib.make_run_dir_path("eval/{}".format(dirname)))

    eval_batch_size = 100

    for counter in range(0, eval_size, eval_batch_size):
        print("Batch {}/{}".format(counter, eval_size))
        eval_latents = np.random.randn(eval_batch_size, *G.input_shape[1:])

        print("Running network...")
        ret = gnet.run(eval_latents, None, None, randomize_noise = False,
            is_visualization = True, minibatch_size = sched.minibatch_gpu, return_dlatents = True)

        eval_fakes = ret[0]
        eval_dlatents_layers = ret[4]
        eval_dlatents = ret[4][:,:,0]
        eval_map_fakes_all = ret[3] if len(ret[3]) > 3 else None

        print("Saving samples...")
        save_imgs(eval_fakes, "eval/gen/%06d.png", counter)

        print("Saving latents...")
        save_npys(eval_latents, "eval/latents/%06d.npy", counter)
        save_npys(eval_dlatents, "eval/dlatents/%06d.npy", counter)
        if attention:
            eval_smaps = eval_map_fakes_all[:,:,-1,0]

            pallete = np.expand_dims(get_colors(component_num), axis = [2, 3])
            eval_maps = (eval_smaps == np.amax(eval_smaps, axis = 1, keepdims = True)).astype(float) # np.expand_dims(, axis = 2)

            eval_vsmaps = np.sum(pallete * np.expand_dims(eval_smaps, axis = 2), axis = 1)
            eval_vmaps = np.sum(pallete * np.expand_dims(eval_maps, axis = 2), axis = 1)

            print("Saving maps...")
            save_npys(eval_smaps, "eval/smap/%06d.npy", counter)
            save_imgs(eval_vsmaps, "eval/vsmap/%06d.png", counter)
            save_npys(eval_maps, "eval/map/%06d.npy", counter)
            save_imgs(eval_vmaps, "eval/vmap/%06d.png", counter)

            save_bimgs(eval_vmaps, eval_fakes, "eval/vmap/b_%06d.png", counter)
            save_bimgs(eval_vsmaps, eval_fakes, "eval/vsmap/b_%06d.png", counter)

        if attention:
            maps = []
            maps_fakes = np.split(eval_map_fakes_all, eval_map_fakes_all.shape[2], axis = 2)
            for layer, amaps in enumerate(maps_fakes):
                amaps = np.split(np.squeeze(amaps, axis = 2), mapfakes.shape[3], axis = 2)
                for head, amap in enumerate(amaps):
                    hmap = (amap == np.amax(amap, axis = 1, keepdims = True)).astype(float)
                    vamap = np.sum(pallete * amap, axis = 1)
                    amap = np.squeeze(amap, axis = 2)
                    maps.append((amap, vamap, "l{}_h{}".format(layer, head)))

            print("Saving layer maps...")
            for i in trange(rich_eval_size):
                for s in ["lmap", "vlmap"]:
                    misc.mkdir(dnnlib.make_run_dir_path("eval/%s/%06d" % (s, i)))
            for amaps, vamaps, name in tqdm(maps, total = len(maps)):
                if False:
                    save_npys(amaps, "eval/lmap/%06d/" + "{}.npy".format(name))
                save_imgs(vamaps, "eval/vlmap/%06d/" + "{}.png".format(name), counter)

    if component_num > 1:
        gif_rate = 8

        ts = np.array(np.linspace(0.0, 1.0, num = gif_rate, endpoint = True))

        print("Generating interpolations...")
        for i in trange(rich_eval_num):
            for s in ["intr_z", "intr_w"]:
                misc.mkdir(dnnlib.make_run_dir_path("eval/%s/%06d" % (s, i)))
            if i == 0:
                for s in ["prtrb_z", "prtrb_w"]:
                   misc.mkdir(dnnlib.make_run_dir_path("eval/%s/%06d" % (s, i)))

            z = np.random.randn(2, *gnet.input_shape[1:])
            z[0] = eval_latents[i:i+1]
            w = gnet.run(z, None, None, randomize_noise = False, return_dlatents = True,
                minibatch_size = sched.minibatch_gpu)[-1]

            def update(t, fn, ts, dim):
                if dim == 3:
                    ts = ts[:, np.newaxis]
                t_ups = []
                for o in range(component_num):
                    t_up = np.tile(np.copy(t[0])[None], [gif_rate] + [1] * dim)
                    t_up[:,o] = fn(t[0,o], t[1,o], ts)
                    t_ups.append(t_up)
                t_up = np.concatenate(t_ups, axis = 0)
                return t_up

            z_up = update(z, slerp, ts, 2)
            w_up = update(w, lerp, ts, 3)

            imgs1 = gnet.run(z_up, None, None, randomize_noise = False,
                minibatch_size = sched.minibatch_gpu)[0]
            imgs2 = gnet.run(w_up, None, None, randomize_noise = False, take_dlatents = True,
                minibatch_size = sched.minibatch_gpu)[0]

            def save_gif(imgs, name):
                imgs = np.split(imgs, component_num, axis = 0)
                for c in range(component_num):
                    imgs[c] = [misc.convert_to_pil_img(img, drange = drange_net) for img in imgs[c]]
                    saveGIF(imgs[c], dnnlib.make_run_dir_path("eval/intr_%s/%06d/%02d.gif" % (name, i, c)))
                    imgs[c][-1].save(dnnlib.make_run_dir_path("eval/intr_%s/%06d/%02d.png" % (name, i, c)))

            save_gif(imgs1, "z")
            save_gif(imgs2, "w")

    print("Generating noise variance...")
    noise_samples_num = 100
    z = np.tile(np.random.randn(1, *gnet.input_shape[1:]), [noise_samples_num, 1, 1])
    imgs = gnet.run(z, None, None, minibatch_size = sched.minibatch_gpu)[0]
    imgs = np.stack([misc.convert_to_pil_img(img, drange = drange_net) for img in imgs], axis = 0)
    diff = np.std(np.mean(imgs, axis = 3), axis = 0) * 4
    diff = np.clip(diff + 0.5, 0, 255).astype(np.uint8)
    PIL.Image.fromarray(diff, "L").save(dnnlib.make_run_dir_path("eval/noisevar.png"))

    print("Generating style mixes...")
    cols, rows = 4, 2
    row_lens = np.array([2, 5, 8, 11])
    mixes = {}
    mixes["l"] = (np.arange(eval_dlatents_layers.shape[2]) < row_lens[:,None]).astype(np.float32)[:,None,None,None,:,None]
    mixes["o"] = (np.arange(eval_dlatents_layers.shape[1]) < row_lens[:,None]).astype(np.float32)[:,None,None,:,None,None]

    ws = eval_dlatents_layers[:cols+rows]
    orig_imgs = eval_fakes[:cols+rows]
    col_ltnts = eval_dlatents_layers[:cols][None, None]
    row_ltnts = eval_dlatents_layers[cols:cols+rows][None,:,None]

    for name, mix in mixes.items():
        mix_ltnts = mix * row_ltnts  + (1 - mix) * col_ltnts
        mix_ltnts = np.reshape(mix_ltnts, [-1, *eval_dlatents_layers.shape[1:]])
        mix_imgs = gnet.run(mix_ltnts, None, None, randomize_noise = False, take_dlatents = True,
            minibatch_size = sched.minibatch_gpu)[0]
        mix_imgs = np.reshape(mix_imgs, [len(row_lens) * rows, cols, *mix_imgs.shape[1:]])

        H, W = mix_imgs.shape[-2:]
        canvas = PIL.Image.new("RGB", (W * (cols + 1), H * (len(row_lens) * rows + 1)), "black")
        for row_idx, row_elem in enumerate([None] + list(range(len(row_lens) * rows))):
            for col_idx, col_elem in enumerate([None] + list(range(cols))):
                if row_elem is None and col_elem is None:
                    continue
                if row_elem is None:
                    img = orig_imgs[col_elem]
                elif col_elem is None:
                    img = orig_imgs[cols + (row_elem % rows)]
                else:
                    img = mix_imgs[row_elem, col_elem]

                canvas.paste(misc.convert_to_pil_img(img, drange = drange_net), (W * col_idx, H * row_idx))
        canvas.save(dnnlib.make_run_dir_path("eval/grid_{}.png".format(name)))

    print(misc.bold("Finished evaluation."))

def visualize():
    ret = gnet.run(grid_latents, grid_labels, None, randomize_noise = False,
        is_visualization = True, minibatch_size = sched.minibatch_gpu)
    grid_fakes = ret[0]
    map_fakes = ret[-1][:,:,-1:] if len(ret[-1].shape) > 3 else None
    del ret

    misc.save_img_grid(grid_fakes, dnnlib.make_run_dir_path("fakes_init.png"), drange = drange_net, grid_size = grid_size)

    if map_fakes is not None and map_fakes.ndim > 1:
        if not keep_samples:
            misc.rm(glob.glob(dnnlib.make_run_dir_path("maps_*")))
        pallete = np.expand_dims(get_colors(component_num), axis = [2, 3, 4, 5] if attention or merge else [2, 3, 4])
        choices = np.expand_dims((map_fakes == np.amax(map_fakes, axis = 1, keepdims = True)).astype(float), axis = 2)
        mapfakes = np.sum(pallete * choices, axis = 1)
        mapfakes = np.concatenate([mapfakes, np.sum(map_fakes, axis = 1, keepdims = True)], axis = 1)

        maps = [(mapfakes, "")]
        if attention or merge:
            maps = []
            maps_fakes = np.split(mapfakes, mapfakes.shape[2], axis = 2)
            for layer, amaps in enumerate(maps_fakes):
                amaps = np.split(np.squeeze(amaps, axis = 2), mapfakes.shape[3], axis = 2)
                for head, amap in enumerate(amaps):
                    maps.append((np.squeeze(amap, axis = 2), "l{}_h{}".format(layer, head)))
        else:
            maps = []
            maps_fakes = np.split(mapfakes, mapfakes.shape[2], axis = 2)
            for head, amap in enumerate(maps_fakes):
                maps.append((np.squeeze(amap, axis = 2), "h{}".format(head)))

        for amap, name in maps:
            misc.save_img_grid(amap, dnnlib.make_run_dir_path("maps_init_%s.png" % (name)),
                drange = drange_net, grid_size = grid_size)

            misc.save_img_grid(amap[:,:-1], dnnlib.make_run_dir_path("map_init_%s.png" % (name)),
                drange = drange_net, grid_size = grid_size)


    tflib.set_vars(noise_var_vals) # [height, width]
    gnet = cG.gpu
    ret = gnet.run(grid_latents, grid_labels, None, is_validation = True, is_visualization = True, randomize_noise = False,
        minibatch_size = sched.minibatch_gpu)

    grid_fakes = ret[0]
    map_fakes = ret[-1][:,:,-1:] if len(ret[-1].shape) > 3 else None

    misc.save_img_grid(grid_fakes, dnnlib.make_run_dir_path("fakes%06d.jpg" % (cur_nimg // 1000)),
        drange = drange_net, grid_size = grid_size)

    if map_fakes is not None and map_fakes.ndim > 1:
        if not keep_samples:
            misc.rm(glob.glob(dnnlib.make_run_dir_path("map*")))
        pallete = np.expand_dims(get_colors(component_num), axis = [2, 3, 4, 5] if attention or merge else [2, 3, 4])
        choices = np.expand_dims((map_fakes == np.amax(map_fakes, axis = 1, keepdims = True)).astype(float), axis = 2)
        mapfakes = np.sum(pallete * choices, axis = 1)
        mapfakes = np.concatenate([mapfakes, np.sum(map_fakes, axis = 1, keepdims = True)], axis = 1)

        maps = [(mapfakes, "")]
        if attention or merge:
            maps = []
            maps_fakes = np.split(mapfakes, mapfakes.shape[2], axis = 2)
            for layer, amaps in enumerate(maps_fakes):
                amaps = np.split(np.squeeze(amaps, axis = 2), mapfakes.shape[3], axis = 2)
                for head, amap in enumerate(amaps):
                    maps.append((np.squeeze(amap, axis = 2), "l{}_h{}".format(layer, head)))
        else:
            maps = []
            maps_fakes = np.split(mapfakes, mapfakes.shape[2], axis = 2)
            for head, amap in enumerate(maps_fakes):
                maps.append((np.squeeze(amap, axis = 2), "h{}".format(head)))

        for amap, name in maps:
            misc.save_img_grid(amap, dnnlib.make_run_dir_path("maps_%06d_%s.png" % (cur_nimg // 1000, name)),
                drange = drange_net, grid_size = grid_size)

            misc.save_img_grid(amap[:,:-1], dnnlib.make_run_dir_path("map_%06d_%s.png" % (cur_nimg // 1000, name)),
                drange = drange_net, grid_size = grid_size)