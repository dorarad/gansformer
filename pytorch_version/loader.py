import click
import pickle
import re
import copy
import numpy as np
import torch
import dnnlib
from torch_utils import misc

pretrained_networks = {
    "gdrive:clevr-snapshot.pkl":        "https://drive.google.com/uc?id=1sYtuNEi0HGBH1F8GW2JlF4mCRGGrtZa9",
    "gdrive:cityscapes-snapshot.pkl":   "https://drive.google.com/uc?id=1iL8S26IYCNAQmUS-6GYCVheKFtCE4ktQ",
    "gdrive:ffhq-snapshot.pkl":         "https://drive.google.com/uc?id=1-b0vwevUQs6LI_EybdO8XJ5uYSx63vEa",
    "gdrive:bedrooms-snapshot.pkl":     "https://drive.google.com/uc?id=1lrB4t3hOOpn8YiwHEduIZYzndHrsKX8z"
}

eval_pretrained_networks = pretrained_networks.copy()
eval_pretrained_networks.update({
#    "gdrive:cityscapes-snapshot-2048.pkl":   "https://drive.google.com/uc?id=1Zw1cFxxN6-iC_M4x6Zbf9lwH9wKryW3p",
    "gdrive:ffhq-snapshot-1024.pkl":         "https://drive.google.com/uc?id=1OcsibUthk2y8y0slf2lQnrYBAtmdJYCQ"
})

def get_path_or_url(path_or_gdrive_path, eval = False):
    nets = eval_pretrained_networks if eval else pretrained_networks
    return nets.get(path_or_gdrive_path, path_or_gdrive_path)

def load_network(filename, eval = False):
    filename = get_path_or_url(filename, eval)
    with dnnlib.util.open_url(filename) as f:
        network = load_network_pkl(f)
    return network

def load_network_pkl(f):
    data = _LegacyUnpickler(f).load()

    # Legacy TensorFlow pickle => convert
    if isinstance(data, tuple) and len(data) == 3 and all(isinstance(net, _TFNetworkStub) for net in data):
        tf_G, tf_D, tf_Gs = data
        G = convert_tf_generator(tf_G)
        D = convert_tf_discriminator(tf_D)
        Gs = convert_tf_generator(tf_Gs)
        data = dict(G = G, D = D, Gs = Gs)

    # Validate contents
    assert isinstance(data["G"], torch.nn.Module)
    assert isinstance(data["D"], torch.nn.Module)
    assert isinstance(data["Gs"], torch.nn.Module)
    return data

#----------------------------------------------------------------------------

class _TFNetworkStub(dnnlib.EasyDict):
    pass

class _LegacyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "dnnlib.tflib.network" and name == "Network":
            return _TFNetworkStub
        return super().find_class(module, name)

def _collect_tf_params(tf_net):
    tf_params = dict()
    def recurse(prefix, tf_net):
        for name, value in tf_net.variables:
            tf_params[prefix + name] = value
        for name, comp in tf_net.components.items():
            recurse(prefix + name + "/", comp)
    recurse("", tf_net)
    return tf_params

def _populate_module_params(module, *patterns):
    for name, tensor in misc.named_params_and_buffers(module):
        found = False
        value = None
        for pattern, value_fn in zip(patterns[0::2], patterns[1::2]):
            match = re.fullmatch(pattern, name)
            if match:
                found = True
                if value_fn is not None:
                    value = value_fn(*match.groups())
                break
        try:
            assert found
            if value is not None:
                tensor.copy_(torch.from_numpy(np.array(value)))
        except:
            print(name, list(tensor.shape))
            raise

#----------------------------------------------------------------------------

def convert_tf_generator(tf_G):
    if tf_G.version < 4:
        raise ValueError("TensorFlow pickle version too low")

    # Collect kwargs
    tf_kwargs = tf_G.static_kwargs
    def kwarg(tf_names, default = None, none = None):
        if not isinstance(tf_names, list):
            tf_names = [tf_names]
        val = default
        for tf_name in tf_names:
            if tf_name in tf_kwargs:
                val = tf_kwargs[tf_name]
        return val if val is not None else none

    # Convert kwargs
    kwargs = dnnlib.EasyDict(
        z_dim                   = kwarg(["latent_size",  "z_dim"], 512),
        c_dim                   = kwarg(["label_size",   "c_dim"], 0),
        w_dim                   = kwarg(["dlatent_size", "w_dim"], 512),
        k                       = kwarg("components_num",       1) + int(tf_G.static_kwargs.get("transformer", False)),
        img_resolution          = kwarg("resolution",           1024),
        img_channels            = kwarg("num_channels",         3),
        mapping_kwargs = dnnlib.EasyDict(
            num_layers          = kwarg(["mapping_layersnum", "mapping_num_layers"], 8),
            layer_dim           = kwarg("mapping_dim",          None),
            act                 = kwarg("mapping_nonlinearity", "lrelu"),
            lrmul               = kwarg("mapping_lrmul",        0.01),
            w_avg_beta          = kwarg(["dlatent_avg_beta", "w_avg_beta"], 0.995,  none = 1),
            resnet              = kwarg("mapping_resnet",       False),
            ltnt2ltnt           = kwarg("mapping_ltnt2ltnt",    False),
            transformer         = kwarg("transformer",          False),
            num_heads           = kwarg("num_heads",            1),
            attention_dropout   = kwarg("attention_dropout",    0.12),
            ltnt_gate           = kwarg("ltnt_gate",            False),
            use_pos             = kwarg("use_pos",              False),
            normalize_global    = False,
        ),
        synthesis_kwargs = dnnlib.EasyDict(
            crop_ratio          = kwarg("crop_ratio",           None),
            channel_base        = kwarg(["fmap_base", "channel_base"],   16 << 10) * 2,
            channel_max         = kwarg(["fmap_max",  "channel_max"],    512),
            architecture        = kwarg("architecture",         "skip"),
            resample_kernel     = kwarg("resample_kernel",      [1,3,3,1]),
            local_noise         = kwarg("local_noise",          True),
            act                 = kwarg("nonlinearity",         "lrelu"),
            ltnt_stem           = kwarg(["latent_stem", "ltnt_stem"],    False),
            style               = kwarg("style",                True),
            transformer         = kwarg("transformer",          False),
            start_res           = kwarg("start_res",            0),
            end_res             = kwarg("end_res",              8),
            num_heads           = kwarg("num_heads",            1),
            attention_dropout   = kwarg("attention_dropout",    0.12),
            ltnt_gate           = kwarg("ltnt_gate",            False),
            img_gate            = kwarg("img_gate",             False),
            integration         = kwarg("integration",          "add"),
            norm                = kwarg("norm",                 None),
            kmeans              = kwarg("kmeans",               False),
            kmeans_iters        = kwarg("kmeans_iters",         1),
            iterative           = kwarg("iterative",            False),
            use_pos             = kwarg("use_pos",              False),
            pos_dim             = kwarg("pos_dim",              None),
            pos_type            = kwarg("pos_type",             "sinus"),
            pos_init            = kwarg("pos_init",             "uniform"),
            pos_directions_num  = kwarg("pos_directions_num",   2),
        ),
    )

    # Collect params
    tf_params = _collect_tf_params(tf_G)
    for name, value in list(tf_params.items()):
        match = re.fullmatch(r"ToRGB_lod(\d+)/(.*)", name)
        if match:
            r = kwargs.img_resolution // (2 ** int(match.group(1)))
            tf_params[f"{r}x{r}/ToRGB/{match.group(2)}"] = value
            kwargs.synthesis.kwargs.architecture = "orig"
    # for name, value in tf_params.items(): print(f"{name:<50s}{list(value.shape)}")

    # Convert params
    from training import networks
    G = networks.Generator(**kwargs).eval().requires_grad_(False)
    index = lambda r, i: "" if int(r) == 4 else f"{i}{['_up',''][int(i)]}"
    plural = lambda s: {"queries": "query", "keys": "key", "values": "value"}[s]
    global_fix = lambda s: "global/" if "global" in s else ""
    z_dim = tf_G.static_kwargs.get("latent_size") or tf_G.static_kwargs.get("z_dim") or 512

    _populate_module_params(G,
        r"pos",                                             lambda:          tf_params["ltnt_emb/emb"],
        # Mapping network
        r"mapping\.w_avg",                                  lambda:          tf_params["dlatent_avg"],
        r"mapping\.embed\.weight",                          lambda:          tf_params["mapping/LabelConcat/weight"].transpose(),
        r"mapping\.embed\.bias",                            lambda:          np.zeros([z_dim]),
        r"mapping\.([a-z_]+)\.l(\d+)\.fc(\d+)\.weight",     lambda s, i, j:  tf_params[f"mapping/{global_fix(s)}Dense{i}_{j}/weight"].transpose(),
        r"mapping\.([a-z_]+)\.l(\d+)\.fc(\d+)\.bias",       lambda s, i, j:  tf_params[f"mapping/{global_fix(s)}Dense{i}_{j}/bias"],
        r"mapping\.([a-z_]+)\.out_layer\.weight",           lambda s:        tf_params[f"mapping/{global_fix(s)}Dense3/weight"].transpose(),
        r"mapping\.([a-z_]+)\.out_layer\.bias",             lambda s:        tf_params[f"mapping/{global_fix(s)}Dense3/bias"],
        r"mapping\.mlp\.l(\d+)\.fc(\d+)\.weight",           lambda i, j:     tf_params[f"mapping/Dense{i}_{j}/weight"].transpose(),
        r"mapping\.mlp\.l(\d+)\.fc(\d+)\.bias",             lambda i, j:     tf_params[f"mapping/Dense{i}_{j}/bias"],
        r"mapping\.mlp\.out_layer\.weight",                 lambda:          tf_params[f"mapping/Dense3/weight"].transpose(),
        r"mapping\.mlp\.out_layer\.bias",                   lambda:          tf_params[f"mapping/Dense3/bias"],
        # Mapping ltnt2ltnt
        r"mapping\.mlp\.sa(\d+)\.to_([a-z]+)\.weight",        lambda i, s:   tf_params[f"mapping/AttLayer_{i}/weight_{plural(s)}"].transpose(),
        r"mapping\.mlp\.sa(\d+)\.to_([a-z]+)\.bias",          lambda i, s:   tf_params[f"mapping/AttLayer_{i}/bias_{plural(s)}"],
        r"mapping\.mlp\.sa(\d+)\.([a-z]+)_pos_map\.weight",   lambda i, s:   tf_params[f"mapping/AttLayer_{i}/weight_{s}_pos"].transpose(),
        r"mapping\.mlp\.sa(\d+)\.([a-z]+)_pos_map\.bias",     lambda i, s:   tf_params[f"mapping/AttLayer_{i}/bias_{s}_pos"],
        r"mapping\.mlp\.sa(\d+)\.modulation\.weight",         lambda i:      tf_params[f"mapping/AttLayer_{i}/weight_out"].transpose(),
        r"mapping\.mlp\.sa(\d+)\.modulation\.bias",           lambda i:      tf_params[f"mapping/AttLayer_{i}/bias_out"],
        r"mapping\.mlp\.sa(\d+)\.centroids",                  lambda i:      tf_params[f"mapping/AttLayer_{i}/toasgn_init"],
        r"mapping\.mlp\.sa(\d+)\.queries2centroids",          lambda i:      tf_params[f"mapping/AttLayer_{i}/weight_key2"].transpose(),
        r"mapping\.mlp\.sa(\d+)\.queries2centroids",          lambda i:      tf_params[f"mapping/AttLayer_{i}/bias_key2"],        
        r"mapping\.mlp\.sa(\d+)\.att_weight",                 lambda i:      tf_params[f"mapping/AttLayer_{i}/iter_0/st_weights"],
        # Synthesis Network
        r"synthesis\.b4\.const",                            lambda:          tf_params[f"synthesis/4x4/Const/const"][0],
        r"synthesis\.b(\d+)\.conv0\.weight",                lambda r:        tf_params[f"synthesis/{r}x{r}/Conv0_up/weight"][::-1, ::-1].transpose(3, 2, 0, 1),
        r"synthesis\.b(\d+)\.conv1\.weight",                lambda r:        tf_params[f"synthesis/{r}x{r}/Conv{index(r,1)}/weight"].transpose(3, 2, 0, 1),
        r"synthesis\.b(\d+)\.conv(\d+)\.biasAct\.bias",     lambda r, i:     tf_params[f"synthesis/{r}x{r}/Conv{index(r,i)}/bias"],
        r"synthesis\.b(\d+)\.conv(\d+)\.noise_const",       lambda r, i:     tf_params[f"synthesis/noise{int(np.log2(int(r)))*2-5+int(i)}"][0, 0],
        r"synthesis\.b(\d+)\.conv(\d+)\.noise_strength",    lambda r, i:     tf_params[f"synthesis/{r}x{r}/Conv{index(r,i)}/noise_strength"],
        r"synthesis\.b(\d+)\.conv(\d+)\.affine\.weight",    lambda r, i:     tf_params[f"synthesis/{r}x{r}/Conv{index(r,i)}/mod_weight"].transpose(),
        r"synthesis\.b(\d+)\.conv(\d+)\.affine\.bias",      lambda r, i:     tf_params[f"synthesis/{r}x{r}/Conv{index(r,i)}/mod_bias"] + 1,
        # Synthesis Network: Latents to Image
        r"synthesis\.b(\d+)\.conv(\d+)\.transformer\.to_([a-z]+)\.weight",        lambda r, i, s:   tf_params[f"synthesis/{r}x{r}/Conv{index(r,i)}/AttLayer_l2n/weight_{plural(s)}"].transpose(),
        r"synthesis\.b(\d+)\.conv(\d+)\.transformer\.to_([a-z]+)\.bias",          lambda r, i, s:   tf_params[f"synthesis/{r}x{r}/Conv{index(r,i)}/AttLayer_l2n/bias_{plural(s)}"],
        r"synthesis\.b(\d+)\.conv(\d+)\.transformer\.([a-z]+)_pos_map\.weight",   lambda r, i, s:   tf_params[f"synthesis/{r}x{r}/Conv{index(r,i)}/AttLayer_l2n/weight_{s}_pos"].transpose(),
        r"synthesis\.b(\d+)\.conv(\d+)\.transformer\.([a-z]+)_pos_map\.bias",     lambda r, i, s:   tf_params[f"synthesis/{r}x{r}/Conv{index(r,i)}/AttLayer_l2n/bias_{s}_pos"],
        r"synthesis\.b(\d+)\.conv(\d+)\.transformer\.modulation\.weight",         lambda r, i:      tf_params[f"synthesis/{r}x{r}/Conv{index(r,i)}/AttLayer_l2n/weight_out"].transpose(),
        r"synthesis\.b(\d+)\.conv(\d+)\.transformer\.modulation\.bias",           lambda r, i:      tf_params[f"synthesis/{r}x{r}/Conv{index(r,i)}/AttLayer_l2n/bias_out"],
        r"synthesis\.b(\d+)\.conv(\d+)\.transformer\.centroids",                  lambda r, i:      tf_params[f"synthesis/{r}x{r}/Conv{index(r,i)}/AttLayer_l2n/toasgn_init"],
        r"synthesis\.b(\d+)\.conv(\d+)\.transformer\.queries2centroids",          lambda r, i:      tf_params[f"synthesis/{r}x{r}/Conv{index(r,i)}/AttLayer_l2n/weight_key2"].transpose(),
        r"synthesis\.b(\d+)\.conv(\d+)\.transformer\.queries2centroids",          lambda r, i:      tf_params[f"synthesis/{r}x{r}/Conv{index(r,i)}/AttLayer_l2n/bias_key2"],        
        r"synthesis\.b(\d+)\.conv(\d+)\.transformer\.att_weight",                 lambda r, i:      tf_params[f"synthesis/{r}x{r}/Conv{index(r,i)}/AttLayer_l2n/iter_0/st_weights"],
        # Synthesis Network's RGB layer
        r"synthesis\.b(\d+)\.torgb\.weight",                lambda r:   tf_params[f"synthesis/{r}x{r}/ToRGB/weight"].transpose(3, 2, 0, 1),
        r"synthesis\.b(\d+)\.torgb\.biasAct\.bias",         lambda r:   tf_params[f"synthesis/{r}x{r}/ToRGB/bias"],
        r"synthesis\.b(\d+)\.torgb\.affine\.weight",        lambda r:   tf_params[f"synthesis/{r}x{r}/ToRGB/mod_weight"].transpose(),
        r"synthesis\.b(\d+)\.torgb\.affine\.bias",          lambda r:   tf_params[f"synthesis/{r}x{r}/ToRGB/mod_bias"] + 1,
        r"synthesis\.b(\d+)\.skip\.weight",                 lambda r:   tf_params[f"synthesis/{r}x{r}/Skip/weight"][::-1, ::-1].transpose(3, 2, 0, 1),
        r"synthesis\.b256\.conv_last\.weight",              lambda:     tf_params[f"synthesis/256x256/ToRGB/extraLayer/weight"].transpose(3, 2, 0, 1),
        r"synthesis\.b256\.conv_last\.affine\.weight",      lambda:     tf_params[f"synthesis/256x256/ToRGB/extraLayer/mod_weight"].transpose(),
        r"synthesis\.b256\.conv_last\.affine\.bias",        lambda:     tf_params[f"synthesis/256x256/ToRGB/extraLayer/mod_bias"] + 1,
        r".*\.resample_kernel",                             None,
        r".*\.grid_pos",                                    None,
    )
    return G

#----------------------------------------------------------------------------

def convert_tf_discriminator(tf_D):
    if tf_D.version < 4:
        raise ValueError("TensorFlow pickle version too low")

    # Collect kwargs
    tf_kwargs = tf_D.static_kwargs
    def kwarg(tf_names, default = None):
        if not isinstance(tf_names, list):
            tf_names = [tf_names]
        for tf_name in tf_names:
            if tf_name in tf_kwargs:
                return tf_kwargs[tf_name]
        return default

    # Convert kwargs
    kwargs = dnnlib.EasyDict(
        c_dim                   = kwarg(["label_size", "c_dim"], 0),
        img_resolution          = kwarg("resolution",           1024),
        img_channels            = kwarg("num_channels",         3),
        architecture            = kwarg("architecture",         "resnet"),
        channel_base            = kwarg("fmap_base",            16 << 10) * 2,
        channel_max             = kwarg("fmap_max",             512),
        crop_ratio              = kwarg("crop_ratio",           None),
        block_kwargs = dnnlib.EasyDict(
            act                 = kwarg("nonlinearity",         "lrelu"),
            resample_kernel     = kwarg("resample_kernel",      [1,3,3,1]),
        ),
        epilogue_kwargs = dnnlib.EasyDict(
            act                 = kwarg("nonlinearity",         "lrelu"),
            mbstd_group_size    = kwarg("mbstd_group_size",     4),
            mbstd_num_channels  = kwarg("mbstd_num_features",   1),
        ),
    )

    # Collect params
    tf_params = _collect_tf_params(tf_D)
    for name, value in list(tf_params.items()):
        match = re.fullmatch(r"FromRGB_lod(\d+)/(.*)", name)
        if match:
            r = kwargs.img_resolution // (2 ** int(match.group(1)))
            tf_params[f"{r}x{r}/FromRGB/{match.group(2)}"] = value
            kwargs.architecture = "orig"
    #for name, value in tf_params.items(): print(f"{name:<50s}{list(value.shape)}")

    # Convert params
    from training import networks
    D = networks.Discriminator(**kwargs).eval().requires_grad_(False)
    _populate_module_params(D,
        r"b(\d+)\.fromrgb\.weight",             lambda r:       tf_params[f"{r}x{r}/FromRGB/weight"].transpose(3, 2, 0, 1),
        r"b(\d+)\.fromrgb\.biasAct\.bias",      lambda r:       tf_params[f"{r}x{r}/FromRGB/bias"],
        r"b(\d+)\.conv(\d+)\.weight",           lambda r, i:    tf_params[f"{r}x{r}/Conv{i}{['','_down'][int(i)]}/weight"].transpose(3, 2, 0, 1),
        r"b(\d+)\.conv(\d+)\.biasAct\.bias",    lambda r, i:    tf_params[f"{r}x{r}/Conv{i}{['','_down'][int(i)]}/bias"],
        r"b(\d+)\.skip\.weight",                lambda r:       tf_params[f"{r}x{r}/Skip/weight"].transpose(3, 2, 0, 1),
        r"b4\.conv\.weight",                    lambda:         tf_params[f"4x4/Conv/weight"].transpose(3, 2, 0, 1),
        r"b4\.conv\.biasAct\.bias",             lambda:         tf_params[f"4x4/Conv/bias"],
        r"b4\.fc\.weight",                      lambda:         tf_params[f"4x4/Dense0/weight"].transpose(),
        r"b4\.fc\.bias",                        lambda:         tf_params[f"4x4/Dense0/bias"],
        r"b4\.out\.weight",                     lambda:         tf_params[f"Output/weight"].transpose(),
        r"b4\.out\.bias",                       lambda:         tf_params[f"Output/bias"],
        r".*\.resample_kernel",                 None,
    )

    return D

#----------------------------------------------------------------------------

@click.command()
@click.option("--source", help="Input pickle", required=True, metavar="PATH")
@click.option("--dest", help="Output pickle", required=True, metavar="PATH")
def convert_network_pickle(source, dest):
    """Convert legacy network pickle into the native PyTorch format.

    The tool is able to load the main network configurations exported using the TensorFlow version of GANFormer.

    Example: python loader.py --source=checkpoint-tf.pkl --dest=checkpoint.pkl
    """
    print(f"Loading {source}...")
    with dnnlib.util.open_url(source) as f:
        data = load_network_pkl(f)
    print(f"Saving {dest}...")
    with open(dest, "wb") as f:
        pickle.dump(data, f)
    print("Done.")

if __name__ == "__main__":
    convert_network_pickle()
