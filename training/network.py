# Network architectures for the GANSformer model, as well as multiple baselines such as
# vanilla GAN, StyleGAN2, k-GAN and SAGAN, all implemented as extensions of the same 
# model skeleton for most precise comparability under same settings.
# See readme for architecture overview.

import numpy as np
import tensorflow as tf
import dnnlib
import dnnlib.tflib as tflib
from dnnlib.tflib.ops.upfirdn_2d import upsample_2d, downsample_2d, upsample_conv_2d, conv_downsample_2d
from dnnlib.tflib.ops.fused_bias_act import fused_bias_act
from dnnlib import EasyDict
from training import misc
import math
import copy

# Shape operations
# ----------------------------------------------------------------------------

# Return the shape of tensor as a list, preferring static dimensions when available
def get_shape(x):
    shape, dyn_shape = copy.deepcopy(x.shape.as_list()), tf.shape(x)
    for index, dim in enumerate(shape):
        if dim is None:
            shape[index] = dyn_shape[index]
    return shape

# Given a tensor with elements [batch_size, ...] compute the size of each element in the batch
def element_dim(x):
    return np.prod(get_shape(x)[1:])

# Flatten all dimensions of a tensor except the last one
def to_2d(x):
    shape = get_shape(x)
    if len(shape) == 2:
        return x
    return tf.reshape(x, [np.prod(shape[:-1]), shape[-1]])

# Linear layer
# ----------------------------------------------------------------------------

# Get/create a weight tensor for a convolution or fully-connected layer
def get_weight(shape, gain = 1, use_wscale = True, lrmul = 1, weight_var = "weight"):
    fan_in = np.prod(shape[:-1])
    he_std = gain / np.sqrt(fan_in)

    # Equalized learning rate and custom learning rate multiplier
    if use_wscale:
        init_std = 1.0 / lrmul
        runtime_coef = he_std * lrmul
    else:
        init_std = he_std / lrmul
        runtime_coef = lrmul

    # Create variable
    init = tf.initializers.random_normal(0, init_std)
    return tf.get_variable(weight_var, shape = shape, initializer = init) * runtime_coef

# Linear dense layer (doesn't include biases. For that see function below)
def dense_layer(x, dim, gain = 1, use_wscale = True, lrmul = 1, weight_var = "weight", name = None):
    if name is not None:
        weight_var = "{}_{}".format(weight_var, name)

    if len(get_shape(x)) > 2:
        x = tf.reshape(x, (-1, element_dim(x)))

    w = get_weight([get_shape(x)[1], dim], gain = gain, use_wscale = use_wscale, 
        lrmul = lrmul, weight_var = weight_var)

    return tf.matmul(x, w)

# Apply bias and optionally an activation function
def apply_bias_act(x, act = "linear", alpha = None, gain = None, lrmul = 1, bias_var = "bias", name = None):
    if name is not None:
        bias_var = "{}_{}".format(bias_var, name)
    b = tf.get_variable(bias_var, shape = [get_shape(x)[1]], initializer = tf.initializers.zeros()) * lrmul
    return fused_bias_act(x, b = b, act = act, alpha = alpha, gain = gain)

# Feature normalization
# ----------------------------------------------------------------------------

# Apply feature normalization, either instance, batch or layer normalization.
# x shape is NCHW
def norm(x, norm_type, parametric = True):
    if norm_type == "instance":
        x = tf.contrib.layers.instance_norm(x, data_format = "NCHW", center = parametric, scale = parametric)
    elif norm_type == "batch":
        x = tf.contrib.layers.batch_norm(x, data_format = "NCHW", center = parametric, scale = parametric)
    else norm_type == "layer":
        x = tf.contrib.layers.layer_norm(inputs = x, begin_norm_axis = -1, begin_params_axis = -1)
    return x

# Normalization operation used in attention layers. Does not scale back features x (the image) 
# with parametrized gain and bias, since these will be controlled by the additive/multiplicative 
# integration of as part of the transformer layer (where the latent y will modulate the image features x)
# after x gets normalized, by controlling their scale and bias (similar to the FiLM and StyleGAN approaches).
# 
# Arguments:
# - x: [batch_size * num, channels]
# - num: number of elements in the x set (e.g. number of positions WH)
# - integration: type of integration -- additive, multiplicative or both
# - norm: normalization type -- instance or layer-wise
# Returns: normalized x tensor
def att_norm(x, num, integration, norm):
    shape = get_shape(x)
    x = tf.reshape(x, [-1, num] + get_shape(x)[1:])
    x = tf.cast(x, tf.float32)

    # instance axis if norm == "instance" and channel axis if norm == "layer"
    norm_axis = 1 if norm == "instance" else 2

    if integration in ["add", "both"]:
        x -= tf.reduce_mean(x, axis = norm_axis, keepdims = True)
    if integration in ["mul", "both"]:
        x *= tf.rsqrt(tf.reduce_mean(tf.square(x), axis = norm_axis, keepdims = True) + 1e-8)
    
    # return x to its original shape
    x = tf.reshape(x, shape)
    return x

# Minibatch standard deviation layer (used in the Discriminator, see StyleGAN model and prior
# work about GANs for further details).
def minibatch_stddev_layer(x, group_size = 4, num_new_features = 1, sdims = 2):
    # Minibatch must be divisible by (or smaller than) group_size
    group_size = tf.minimum(group_size, get_shape(x)[0])    
    s = x.shape # [NCHW] 
    # Split minibatch into M groups of size G. Split channels into n channel groups c
    y = tf.reshape(x, [group_size, -1, num_new_features, s[1]//num_new_features, s[2]] + ([s[3]] if sdims == 2 else [])) # [GMncHW]
    y = tf.cast(y, tf.float32) # [GMncHW]

    # Subtract group mean, then compute variance and stddev
    y -= tf.reduce_mean(y, axis = 0, keepdims = True) # [GMncHW] 
    y = tf.reduce_mean(tf.square(y), axis = 0) # [MncHW]
    y = tf.sqrt(y + 1e-8) # [MncHW]

    # Take average over dim and pixels
    y = tf.reduce_mean(y, axis = [2, 3] + ([4] if sdims == 2 else []), keepdims = True) # [Mn111] 
    # Split channels into c channel groups
    y = tf.reduce_mean(y, axis = [2]) # [Mn11]
    # Replicate over group and pixels
    y = tf.tile(y, [group_size, 1, s[2]] + ([s[3]] if sdims == 2 else [])) # [NnHW]
    # Append as new fmap
    return tf.concat([x, y], axis = 1) # [NCHW]

# Dropout and masking
# ----------------------------------------------------------------------------

# Create a random mask of a chosen shape and with probability 'dropout' to be dropped (=0)
def random_dp_binary(shape, dropout):
    if dropout == 0:
        return tf.ones(shape)
    eps = tf.random.uniform(shape)
    keep_mask = (eps >= dropout)
    return keep_mask

# Perform dropout
def dropout(x, dp, noise_shape = None):
    if dp is None or dp == 0.0:
        return x
    return tf.nn.dropout(x, keep_prob = 1.0 - dp, noise_shape = noise_shape)

# Set a mask for logits (set -Inf where mask is 0)
def logits_mask(x, mask):
    return x + tf.cast(1 - tf.cast(mask, tf.int32), tf.float32) * -10000.0

# Positional encoding
# ----------------------------------------------------------------------------

# 2d linear embeddings [size, size, dim] in a [-rng, rng] range where size = grid size and 
# dim = the embedding dimension. Each embedding consists of 'num' parts, with each part measuring 
# positional similarity along another direction, uniformly spanning the 2d space.
def get_linear_embeddings(size, dim, num, rng = 1.0):
    pi = tf.constant(math.pi)
    theta = tf.range(0, pi, pi / num)
    dirs = tf.stack([tf.cos(theta), tf.sin(theta)], axis = -1)
    embs = tf.get_variable(name = "emb", shape = [num, int(dim / num)], 
        initializer = tf.random_uniform_initializer())
    
    c = tf.linspace(-rng, rng, size)
    x = tf.tile(tf.expand_dims(c, axis = 0), [size, 1])
    y = tf.tile(tf.expand_dims(c, axis = 1), [1, size])
    xy = tf.stack([x,y], axis = -1)
    
    lens = tf.reduce_sum(tf.expand_dims(xy, axis = 2) * dirs, axis = -1, keepdims = True)  
    emb = tf.reshape(lens * embs, [size, size, dim])
    return emb

# 2d sinusoidal embeddings [size, size, dim] with size = grid size and dim = embedding dimension 
# (see "Attention is all you need" paper)
def get_sinusoidal_embeddings(size, dim, num = 2): 
    # Standard positional embeddings in the two spatial w,h directions
    if num == 2:
        c = tf.expand_dims(tf.to_float(tf.linspace(-1.0, 1.0, size)), axis = -1)
        i = tf.to_float(tf.range(int(dim / 4))) 

        peSin = tf.sin(c / (tf.pow(10000.0, 4 * i / dim)))
        peCos = tf.cos(c / (tf.pow(10000.0, 4 * i / dim)))

        peSinX = tf.tile(tf.expand_dims(peSin, axis = 0), [size, 1, 1])
        peCosX = tf.tile(tf.expand_dims(peCos, axis = 0), [size, 1, 1])
        peSinY = tf.tile(tf.expand_dims(peSin, axis = 1), [1, size, 1])
        peCosY = tf.tile(tf.expand_dims(peCos, axis = 1), [1, size, 1]) 

        emb = tf.concat([peSinX, peCosX, peSinY, peCosY], axis = -1)
    # Extension to 'num' spatial directions. Each embedding consists of 'num' parts, with each  
    # part measuring positional similarity along another direction, uniformly spanning the 2d space.
    # Each such part has a sinus and cosine components.
    else:
        pi = tf.constant(math.pi)
        theta = tf.range(0, pi, pi / num)
        dirs = tf.stack([tf.cos(theta), tf.sin(theta)], axis = -1)

        c = tf.linspace(-1.0, 1.0, size)
        x = tf.tile(tf.expand_dims(c, axis = 0), [size, 1])
        y = tf.tile(tf.expand_dims(c, axis = 1), [1, size])
        xy = tf.stack([x,y], axis = -1)

        lens = tf.reduce_sum(tf.expand_dims(xy, axis = -2) * dirs, axis = -1, keepdims = True)

        i = tf.to_float(tf.range(int(dim / (2 * num))))
        sins = tf.sin(lens / (tf.pow(10000.0, 2 * num * i / dim)))
        coss = tf.cos(lens / (tf.pow(10000.0, 2 * num * i / dim)))
        emb = tf.reshape(tf.concat([sins, coss], axis = -1), [size, size, dim])

    return emb

# 2d positional embeddings of dimension 'dim' in a range of resolutions from 2x2 up to 'max_res x max_res'
# 
# pos_type: supports several types of embedding schemes:
# - sinus: (see "Attention is all you need")
# - linear: where each position gets a value of [-1, 1] * trainable_vector, in each spatial
#   direction based on its location.
# - trainable: where an embedding of position [w,h] is [emb_w, emb_h] (independent parts in 
#   each spatial direction)
# - trainable2d: where an embedding of position [w,h] is emb_{w,h} (a different embedding for 
#   each position)
# 
# dir_num: Each embedding consists of 'dir_num' parts, with each path measuring positional similarity 
# along another direction, uniformly spanning the 2d space.
# 
# shared: True for using same embeddings for all directions / parts
# init: uniform or normal distribution for trainable embeddings initialization
def get_positional_embeddings(max_res, dim, pos_type = "sinus", dir_num = 2, init = "uniform", shared = False):
    embs = []
    initializer = tf.random_uniform_initializer() if init == "uniform" else tf.initializers.random_normal()
    for res in range(max_res + 1):
        with tf.variable_scope("pos_emb%d" % res):
            size = 2 ** res            
            if pos_type == "sinus":
                emb = get_sinusoidal_embeddings(size, dim, num = dir_num)
            elif pos_type == "linear":
                emb = get_linear_embeddings(size, dim, num = dir_num)
            elif pos_type == "trainable2d":
                emb = tf.get_variable(name = "emb", shape = [size, size, dim], initializer = initializer)
            else pos_type == "trainable":
                xemb = tf.get_variable(name = "x_emb", shape = [size, int(dim / 2)], initializer = initializer)
                yemb = xemb if shared else tf.get_variable(name = "y_emb", shape = [size, int(dim / 2)], initializer = initializer)
                xemb = tf.tile(tf.expand_dims(xemb, axis = 0), [size, 1, 1])
                yemb = tf.tile(tf.expand_dims(yemb, axis = 1), [1, size, 1])
                emb = tf.concat([xemb, yemb], axis = -1)
            embs.append(emb)
    return embs

# Produce trainable embeddings of shape [size, dim], uniformly/normally initialized 
def get_embeddings(size, dim, init = "uniform", name = None):
    initializer = tf.random_uniform_initializer() if init == "uniform" else tf.initializers.random_normal()
    with tf.variable_scope(name):    
        emb = tf.get_variable(name = "emb", shape = [size, dim], initializer = initializer)
    return emb

# Produce relative embeddings
def get_relative_embeddings(l, dim, embs):
    diffs = tf.expand_dims(tf.range(l), axis = -1) - tf.range(l)
    diffs -= tf.reduce_min(diffs)
    ret = tf.gather(embs, tf.reshape(diffs, [-1]))
    ret = tf.reshape(ret, [1, l, l, dim])
    return ret

# Non-Linear networks
# ----------------------------------------------------------------------------

# Non-linear layer with a resnet connection. Accept features 'x' of dimension 'dim', 
# and use nonlinearty 'act'. Optionally perform attention from x to y 
# (meaning information flows y -> x).
def nnlayer(x, dim, act, lrmul, y = None, ff = True, name = "", **kwargs):
    _x = x

    if y is not None:
        x = transformer(from_tensor = x, to_tensor = y, dim = dim, name = name, **kwargs)[0]

    if ff:
        with tf.variable_scope("Dense%s_0" % name):
            x = apply_bias_act(dense_layer(x, dim, lrmul = lrmul), act = act, lrmul = lrmul)
        with tf.variable_scope("Dense%s_1" % name):
            x = apply_bias_act(dense_layer(x, dim, lrmul = lrmul), lrmul = lrmul)  
 
        x = tf.nn.leaky_relu(x + _x)

    return x

# Multi-layer network with 'layers_num' layers, dimension 'dim', and nonlinearity 'act'.
# Optionally use resnet connections and self-attention.
# If x dimensions are not [batch_size, dim], then first pool x, according to a pooling scheme:
# - mean: mean pooling
# - cnct: concatenate all spatial features together to one vector and compress them to dimension 'dim'
# - 2d: turn a tensor [..., dim] into [-1, dim] so to create one large batch with all the elements from the last axis.
def mlp(x, resnet, layers_num, dim, act, lrmul, pooling = "mean", attention = False, norm_type = None, **kwargs): 
    shape = get_shape(x)
    
    if len(get_shape(x)) > 2:
        if pooling == "cnct":
            with tf.variable_scope("Dense_pool"):
                x = apply_bias_act(dense_layer(x, dim), act = act)
        elif pooling == "batch":
            x = to_2d(x)
        else:
            pool_shape = (get_shape(x)[-2], get_shape(x)[-1])
            x = tf.nn.avg_pool(x, pool_shape, pool_shape, padding = "SAME", data_format = "NCHW")
            x = tf.reshape(x, [get_shape(x)[0], element_dim(x)])

    if resnet:
        half_layers_num = int(layers_num / 2)
        for layer_idx in range(half_layers_num):
            y = x if attention else None
            x = nnlayer(x, dim, act, lrmul, y = y, name = layer_idx, **kwargs)
            x = norm(x, norm_type)

        with tf.variable_scope("Dense%d" % layer_idx):
            x = apply_bias_act(dense_layer(x, dim, lrmul = lrmul), act = act, lrmul = lrmul) 

    else:    
        for layer_idx in range(layers_num):
            with tf.variable_scope("Dense%d" % layer_idx):
                x = apply_bias_act(dense_layer(x, dim, lrmul = lrmul), act = act, lrmul = lrmul)
                x = norm(x, norm_type)
    
    x = tf.reshape(x, shape[:-1] + [dim])
    return x

# Convolution layer and optional enhancements
# ----------------------------------------------------------------------------

# Convolution layer with optional upsampling or downsampling
def conv2d_layer(x, dim, kernel, up = False, down = False, resample_kernel = None, gain = 1, 
        use_wscale = True, lrmul = 1, weight_var = "weight"):
    assert not (up and down)
    assert kernel >= 1 and kernel % 2 == 1
    w = get_weight([kernel, kernel, get_shape(x)[1], dim], gain = gain, use_wscale = use_wscale, 
        lrmul = lrmul, weight_var = weight_var)
    if up:
        x = upsample_conv_2d(x, w, data_format = "NCHW", k = resample_kernel)
    elif down:
        x = conv_downsample_2d(x, w, data_format = "NCHW", k = resample_kernel)
    else:
        x = tf.nn.conv2d(x, w, data_format = "NCHW", strides = [1, 1, 1, 1], padding = "SAME")
    return x

# Modulated convolution layer
def modulated_conv2d_layer(x, y, dim, kernel, # Convolution parameters
    up = False,             # Whether to upsample features after convolution
    down = False,           # Whether to downsample features before convolution
    resample_kernel = None, # resample_kernel for up/downsampling
    modulate = True,        # Whether y should modulate x (control its gain/bias) 
    demodulate = True,      # Whether to normalize/scale back x gain/bias
    fused_modconv = True,   # Whether to fuse the convlution and modulate operations together (see StyleGAN2)
    noconv = False,         # Whether to skip convlution itself (so to perform only modulation)
    gain = 1, use_wscale = True, lrmul = 1, # Parameters for creating the convolution weight 
    weight_var = "weight", mod_weight_var = "mod_weight", mod_bias_var = "mod_bias"):

    assert not (up and down)
    assert kernel >= 1 and kernel % 2 == 1

    w = get_weight([kernel, kernel, get_shape(x)[1], dim], gain = gain, use_wscale = use_wscale, 
        lrmul = lrmul, weight_var = weight_var)
    ww = w[np.newaxis]

    # Modulate and demodulate
    # Transform incoming W to style and add bias (initially 1)
    s = dense_layer(y, dim = get_shape(x)[1], weight_var = mod_weight_var) # [BI] 
    s = apply_bias_act(s, bias_var = mod_bias_var) + 1 # [BI] 
    
    if modulate:
        # Scale input features
        ww *= s[:, np.newaxis, np.newaxis, :, np.newaxis] # [BkkIO]
        if demodulate:
            # Scale output features according to Scaling factor
            d = tf.rsqrt(tf.reduce_sum(tf.square(ww), axis = [1, 2, 3]) + 1e-8) # [BO] 
            ww *= d[:, np.newaxis, np.newaxis, np.newaxis, :] # [BkkIO]
    else:
        ww += tf.zeros_like(s[:, np.newaxis, np.newaxis, :, np.newaxis])

    # Reshape/scale input
    # If fused, reshape to convolution groups. Otherwise, scale input activations directly.
    if fused_modconv:
        x = tf.reshape(x, [1, -1, get_shape(x)[-2], get_shape(x)[-1]]) 
        w = tf.reshape(tf.transpose(ww, [1, 2, 3, 0, 4]), [ww.shape[1], ww.shape[2], ww.shape[3], -1])
    else:
        if modulate:
            x *= s[:, :, np.newaxis, np.newaxis] # [BIhw]

    # Convolution with optional up/downsampling
    if noconv:
        if up:
            x = upsample_2d(x, k = resample_kernel)
        elif down:
            x = downsample_2d(x, k = resample_kernel)
    else:
        if up:
            x = upsample_conv_2d(x, w, data_format = "NCHW", k = resample_kernel)
        elif down:
            x = conv_downsample_2d(x, w, data_format = "NCHW", k = resample_kernel)
        else:
            x = tf.nn.conv2d(x, w, data_format = "NCHW", strides = [1,1,1,1], padding = "SAME")

    # Reshape/scale output
    # If fused,  reshape convolution groups back to batch. Otherwise, scale back output activations.
    if fused_modconv:
        x = tf.reshape(x, [-1, dim] + get_shape(x)[-2:])
    elif modulate and demodulate:
        x *= d[:, :, np.newaxis, np.newaxis] # [BOhw] 

    return x

# Local attention block, potentially replacing convolution (not used by default in GANsformer)
# Grid is positional embeddings that match the size of a convolution receptive window
# X shape is [NCHW]
def att2d_layer(x, dim, kernel, grid, up = False, down = False, resample_kernel = None, num_heads = 4):
    if num_heads == "max":
        num_heads = dim

    if down:
        x = downsample_2d(x, k = resample_kernel)

    x = conv2d_layer(x, dim = dim, kernel = 1, weight_var = "newdim")

    # Transpose channel axis to last one
    depth = get_shape(x)[1]
    x = tf.transpose(x, [0, 2, 3, 1]) # [N, H, W, C]

    # extract local patches for each image position
    x_patches = tf.image.extract_patches(x, sizes = [1, kernel, kernel, 1], 
        strides = [1, 1, 1, 1], rates = [1, 1, 1, 1], padding = "SAME") # [N, H, W, k*k, C]
    x_patches = tf.reshape(x_patches, [-1, kernel * kernel, depth]) # [N * H * W, k*k, C]
    x_flat = tf.reshape(x, [-1, 1, depth]) # [N * H * W, 1, C]

    kwargs = {
        "num_heads": num_heads, 
        "integration": "add", 
        "from_pos": tf.zeros([1, depth]),
        "to_pos": grid
    }

    # Perform local attention between each image position and its local neighborhood
    x_flat = transformer(from_tensor = x_flat, to_tensor = x_patches, dim = dim, name = "att2d", **kwargs)[0] # [N * H * W, 1, C]
    x = tf.transpose(tf.reshape(x_flat, get_shape(x)), [0, 3, 1, 2]) # [N, C, H, W]

    if up:
        x = upsample_2d(x, k = resample_kernel)

    return x

# Transformer layer
# ----------------------------------------------------------------------------

# A helper function for the transformer
def transpose_for_scores(x, batch_size, num_heads, elem_num, head_size):  
    x = tf.reshape(x, [batch_size, elem_num, num_heads, head_size]) # [B, N, H, S]
    x = tf.transpose(x, [0, 2, 1, 3]) # [B, H, N, S]
    return x

##########################################################################################################################
# Note that the code of this function is not yet re-factored and cleaned-up. Will be fixed in a few hours, Stay Tuned!
##########################################################################################################################

# comment about direction from to
def transformer(from_tensor, to_tensor, dim, att_vars, from_len = None, to_len = None, att_mask = None, to_gate = False, 
        img_gate = False, num_heads = 1, att_dp = 0.12, from_pos = None, to_pos = None, 
        inputs = "both", kmeans = False, kmeans_iters = 1, integration = "add", normalize = False, 
        norm = None, q_act = "linear", v_act = "linear", c_act = "linear", k_act = "linear", name = ""):
    from_asgn = att_vars["from_asgn"]
    to_asgn = att_vars["to_asgn"]
    
    with tf.variable_scope("AttLayer_{}".format(name)):
        batch_size, from_shape = None, None
        attention_scores = tf.constant(0.0)
        size_head = int(dim / num_heads)

        outdim = get_shape(from_tensor)[-1]
        if integration == "both": 
            outdim *= 2

        if len(get_shape(from_tensor)) > 3 or len(get_shape(to_tensor)) > 3:
            raise
        if len(get_shape(from_tensor)) > 2:
            from_shape = get_shape(from_tensor) 
            batch_size = from_shape[0]
            from_len = from_shape[1]
            from_tensor = to_2d(from_tensor)
        if len(get_shape(to_tensor)) > 2:
            to_len = get_shape(to_tensor)[1]
            to_tensor = to_2d(to_tensor)

        if batch_size is None:
            batch_size = tf.cast(get_shape(from_tensor)[0] / from_len, tf.int32) 

        value_layer = to_tensor
        value_layer = apply_bias_act(dense_layer(value_layer, dim, name = "value"), name = "value", act = v_act) # [B*T, N*H]
        value_layer = transpose_for_scores(value_layer, batch_size, num_heads, to_len, size_head) # [B, N, T, H] int(dim / num_heads)

        if from_pos is not None:
            from_pos = tf.tile(to_2d(from_pos), [batch_size, 1])

        if to_pos is not None:
            to_pos = tf.tile(to_2d(to_pos), [batch_size, 1])

        def cond_downsample(mp):
            if mp is not None and get_shape(mp)[-2] > from_len: # NHWC 
                s = int(math.sqrt(get_shape(mp)[-2]))
                new_s = int(math.sqrt(from_len))
                factor = int(s / new_s)
                mp = downsample_2d(tf.reshape(mp, [batch_size, s, s, to_len]), factor = factor, data_format = "NHWC")
                mp = tf.reshape(mp, [batch_size, num_heads, from_len, to_len])
            return mp

        sh_f = (2 * size_head) if inputs == "both" else size_head  

        if from_pos is not None:
            fp = apply_bias_act(dense_layer(from_pos, dim, name = "from_pos"), name = "from_pos", act = q_act) # _from_pos
        if to_pos is not None:
            tp = apply_bias_act(dense_layer(to_pos, dim, name = "to_pos"), name = "to_pos", act = k_act) # _to_pos

        query_layer = apply_bias_act(dense_layer(from_tensor, dim, name = "query"), name = "query", act = q_act) # [B*F, N*H]
        key_layer = apply_bias_act(dense_layer(to_tensor, dim, name = "key"), name = "key", act = k_act) # [B*T, N*H]

        to_sign = tf.tile(tf.one_hot(tf.range(to_len), to_len)[np.newaxis, np.newaxis], [batch_size, num_heads, 1, 1])
        sh_t = to_len

        from_sign = {"both": tf.concat([query_layer, fp], axis = -1), "content": query_layer, "pos": fp}.get(inputs)
        
        _from_sign = from_sign = transpose_for_scores(from_sign, batch_size, num_heads, from_len, sh_f) # [B, N, F, H]
        _to_sign = to_sign = transpose_for_scores(to_sign, batch_size, num_heads, to_len, sh_t) # [B, N, T, H]

        if from_pos is not None:
            query_layer += fp 
        if to_pos is not None:
            key_layer += tp 

        if from_asgn is not None:
            if get_shape(from_asgn)[-2] < from_len: # NHWC 
                s = int(math.sqrt(get_shape(from_asgn)[-2]))
                from_asgn = upsample_2d(tf.reshape(from_asgn, [batch_size * num_heads, s, s, to_len]), factor = 2, data_format = "NHWC")
                from_asgn = tf.reshape(from_asgn, [batch_size, num_heads, from_len, to_len])
            if get_shape(from_asgn)[-1] < to_len:
                s = int(math.sqrt(get_shape(to_len)[-1]))
                from_asgn = upsample_2d(tf.reshape(from_asgn, [batch_size * num_heads, from_len, s, s]), factor = 2, data_format = "NCHW")
                from_asgn = tf.reshape(from_asgn, [batch_size, num_heads, from_len, to_len])
            from_asgn = tf.matmul(from_asgn, to_sign) 
        if to_asgn is not None:
            if get_shape(to_asgn)[-2] < to_len:
                s = int(math.sqrt(get_shape(to_asgn)[-2]))
                to_asgn = upsample_2d(tf.reshape(to_asgn, [batch_size * num_heads, s, s, from_len]), factor = 2, data_format = "NHWC")
                to_asgn = tf.reshape(to_asgn, [batch_size, num_heads, to_len, from_len])
            if get_shape(to_asgn)[-1] < from_len:
                s = int(math.sqrt(get_shape(to_asgn)[-1]))
                to_asgn = upsample_2d(tf.reshape(to_asgn, [batch_size * num_heads, to_len, s, s]), factor = 2, data_format = "NCHW")
                to_asgn = tf.reshape(to_asgn, [batch_size, num_heads, to_len, from_len])
            to_asgn = tf.matmul(to_asgn, from_sign)

        if not kmeans:
            if from_asgn is not None:
                from_asgn_2d = tf.transpose(from_asgn, [0, 2, 1, 3])
                from_asgn_2d = tf.reshape(from_asgn_2d, [np.prod(get_shape(from_asgn_2d)[:2]), np.prod(get_shape(from_asgn_2d)[2:])])
                query_layer += apply_bias_act(dense_layer(from_asgn_2d, dim, name = "from_asgn"), name = "from_asgn") # [B*F, N*H]
            if to_asgn is not None:
                to_asgn_2d = tf.transpose(to_asgn, [0, 2, 1, 3])
                to_asgn_2d = tf.reshape(to_asgn_2d, [np.prod(get_shape(to_asgn_2d)[:2]), np.prod(get_shape(to_asgn_2d)[2:])])
                key_layer += apply_bias_act(dense_layer(to_asgn_2d, dim, name = "to_asgn"), name = "to_asgn") # [B*F, N*H]

        query_layer = transpose_for_scores(query_layer, batch_size, num_heads, from_len, size_head) # [B, N, F, H]
        key_layer = transpose_for_scores(key_layer, batch_size, num_heads, to_len, size_head) # [B, N, T, H]

        att_scores = tf.matmul(query_layer, key_layer, transpose_b = True) # [B, N, F, T]

        if kmeans:
            att_scores = tf.constant(0.0)
            to_asgn = tf.tile(tf.get_variable("toasgn_init", shape = [1, num_heads, to_len, sh_f], 
                initializer = tf.initializers.random_normal()), [batch_size, 1, 1, 1]) 

        for i in range(kmeans_iters):
            with tf.variable_scope("iter_{}".format(i)):
                if kmeans:
                    if i > 0:
                        if from_asgn is not None:
                            from_asgn = tf.matmul(from_asgn, _to_sign) 
                        if to_asgn is not None:
                            to_asgn = tf.matmul(to_asgn, _from_sign)

                    new_scores = tf.constant(0.0)

                    if from_asgn is not None:
                        w = tf.get_variable(name = "sf_weight", shape = [num_heads, 1, get_shape(from_asgn)[-1]], initializer = tf.ones_initializer())
                        new_scores += tf.matmul(from_asgn * w, to_sign, transpose_b = True)

                    if to_asgn is not None:
                        w = tf.get_variable(name = "st_weights", shape = [num_heads, 1, get_shape(from_sign)[-1]], initializer = tf.ones_initializer())
                        new_scores += tf.matmul(from_sign * w, to_asgn, transpose_b = True) 

                    att_scores = (att_scores + new_scores) if (not kmeans) else new_scores

                attention_scores = tf.multiply(att_scores, 1.0 / math.sqrt(float(size_head)))

                if att_mask is not None:
                    if len(get_shape(att_mask)) == 2:
                        value_layer *= tf.cast(att_mask[:, np.newaxis, :, np.newaxis], tf.float32) # tf.expand_dims(, axis = [1, 3])
                    else:
                        if len(get_shape(att_mask)) == 3:
                            att_mask = tf.expand_dims(att_mask, axis = 1) # [B, 1, F, T]
                        attention_scores = logits_mask(attention_scores, att_mask)

                attention_probs = tf.nn.softmax(attention_scores) # [B, N, F, T]
                attention_probs = dropout(attention_probs, attention_dropout / 2)
                attention_probs = dropout(attention_probs, attention_dropout / 2, [get_shape(attention_scores)[0], num_heads, 1, to_len])

                from_asgn = attention_probs
                to_asgn = from_asgn
                to_asgn = tf.transpose((to_asgn / (tf.reduce_sum(to_asgn, axis = -2, keepdims = True) + 1e-8)), [0, 1, 3, 2])

                if kmeans: 
                    from_asgn = None
                else:
                    from_asgn = to_asgn = None

        attention_probs = tf.nn.softmax(attention_scores) # [B, N, F, T]
        attention_probs = dropout(attention_probs, attention_dropout)
        attention_probs = dropout(attention_probs, attention_dropout, [get_shape(attention_scores)[0], num_heads, 1, to_len])

        if to_gate:
            exist_mask = apply_bias_act(dense_layer(to_tensor, num_heads, name = "e_cont"), name = "e_cont")
            if to_pos is not None:
                exist_mask += apply_bias_act(dense_layer(to_pos, num_heads, name = "e_pos"), name = "e_pos")
            exist_mask = tf.sigmoid(exist_mask)
            exist_mask = tf.transpose(tf.reshape(exist_mask, [batch_size, 1, to_len, num_heads]), [0, 3, 1, 2])
            attention_probs *= exist_mask # tf.expand_dims(exist_mask, axis = 1)

        if img_gate:
            gate = apply_bias_act(dense_layer(from_tensor, num_heads, name = "n_cont"), name = "n_cont")
            if from_pos is not None:
                gate += apply_bias_act(dense_layer(from_pos, num_heads, name = "n_pos"), name = "n_pos") # _from_pos

            gate = tf.sigmoid(gate + 1)
            gate_att = tf.transpose(tf.reshape(gate, [batch_size, from_len, num_heads, 1]), [0, 2, 1, 3])
            attention_probs *= gate_att

        context_layer = tf.matmul(attention_probs, value_layer) # [B, N, F, H]
        context_layer = tf.transpose(context_layer, [0, 2, 1, 3]) # [B, F, N, H]
        context_layer = tf.reshape(context_layer, [batch_size * from_len, dim]) # [B * F, N*H] # batch_size * from_len

        context_layer = apply_bias_act(dense_layer(context_layer, outdim, name = "out"), act = c_act, name = "out")

        out = from_tensor
        if norm is not None: 
            out = att_norm(out, from_len, integration, norm)
        a_context_layer = m_context_layer = context_layer
        if integration == "full": 
            m_context_layer, a_context_layer = tf.split(context_layer, 2, axis = -1)
        if integration != "add":
            out *= (m_context_layer + 1) 
        if integration != "mul":
            out += a_context_layer

        if from_shape is not None:
            out = tf.reshape(out, from_shape)

        from_asgn = attention_probs
        to_asgn = from_asgn
        to_asgn = tf.transpose((to_asgn / (tf.reduce_sum(to_asgn, axis = -2, keepdims = True) + 1e-8)), [0, 1, 3, 2])

        if kmeans: 
            from_asgn = None
        else:
            from_asgn = to_asgn = None

    return out, attention_probs, (from_asgn, to_asgn)

# (Not used be default)
# Merge multiple images together through a weighted sum to create one image (Used by k-GAN only)
# Also merge their the latent vectors that have used to create the images respectively
# Arguments:
# - x: the image features, [batch_size * k, C, H, W]
# - k: the number of images to merge together  
# - type: the type of the merge (sum, softmax, max, leaves)
# - same: whether to merge all images equally along all WH positions
# Returns the merged images [batch_size, C, H, W] and latents [batch_size, k, layers_num, dim]
def merge_images(x, dlatents, k, type, same = False):
    batch_size = tf.cast(get_shape(x)[0] / k, tf.int32)
    x = tf.reshape(x, [batch_size, k] + list(x.shape[-3:])) # [B, k, C, H, W]

    # Compute scores w to be used in a following weighted sum of the images x
    scores = conv2d_layer(x, dim = 1, kernel = 1)
    scores = tf.reshape(scores, [batch_size, k] + list(scores.shape[-3:])) # [B, k, 1, H, W]
    if same:
        scores = tf.reduce_sum(scores, axis = [-2, -1], keepdims = True) # [B, k, 1, 1, 1]

    # Compute soft probabilities for weighted sum
    if type == "softmax":
        scores = tf.nn.softmax(scores, axis = 1) # [B, k, 1, H, W]
    # Take only image of highest score
    elif type == "max":
        scores = tf.one_hot(tf.math.argmax(scores, axis = 1), k, axis = 1) # [B, k, 1, H, W]
    # Merge the images recursively using a falling-leaves scheme (see k-GAN paper for details)
    elif type == "leaves":
        score_list = tf.split(scores, k, axis = 1) # [[B, 1, H, W],...]
        alphas = tf.ones_like(score_list[0]) # [B, 1, H, W]
        for d, weight in enumerate(score_list[1:]): # Process the k images iteratively
            max_weight = tf.math.reduce_max(scores[:,:d + 1], ) # Compute image currently most "in-front"
            new_alphas = tf.sigmoid(weight - max_weight) # Compute alpha values based on "distance" to the new image
            alphas = tf.concat([alphas * (1 - new_alphas), new_alphas], axis = 1) # Compute recursive alpha values
        scores = alphas
    else: # sum
        scores = tf.ones_like(scores) # [B, k, 1, H, W]

    # Compute a weighted sum of the images
    x = tf.reduce_sum(x * scores, axis = 1) # [B, C, H, W]

    if dlatents is not None:
        scores = tf.reduce_mean(tf.reduce_mean(scores, axis = 2), axis = [-2, -1], keepdims = True) # [B, k, 1, 1]
        dlatents = tf.tile(tf.reduce_sum(dlatents * scores, axis = 1, keepdims = True), [1, k, 1, 1]) # [B, k, L, D]
    
    return x

# Generator
# ----------------------------------------------------------------------------

# Generator network, composed of two sub-networks: mapping and synthesis, as defined below
# This function is mostly similar to the original StyleGAN2 version
def G_GANsformer(
    latents_in,                               # First input: Latent vectors (z) [batch_size, latent_size]
    labels_in,                                # Second input (optional): Conditioning labels [batch_size, label_size]
    # General arguments 
    is_training             = False,          # Network is under training? Enables and disables specific features
    is_validation           = False,          # Network is under validation? Chooses which value to use for truncation_psi
    return_dlatents         = False,          # Return dlatents in addition to the images?
    take_dlatents           = False,          # Use latents_in as dlatents (skip mapping network)
    is_template_graph       = False,          # True = template graph constructed by the Network class, False = actual evaluation
    subnets                 = EasyDict(),     # Container for sub-networks. Retained between calls
    mapping_func            = "G_mapping",    # Function name of the mapping network
    synthesis_func          = "G_synthesis",  # Function name of the synthesis network
    # Truncation cutoff
    truncation_psi          = 0.65,           # Style strength multiplier for the truncation trick. None = disable
    truncation_cutoff       = None,           # Number of layers for which to apply the truncation trick. None = disable
    truncation_psi_val      = None,           # Value for truncation_psi to use during validation
    truncation_cutoff_val   = None,           # Value for truncation_cutoff to use during validation
    # W latent space mean tracking
    dlatent_avg_beta        = 0.995,          # Decay for tracking the moving average of W during training. None = disable
    # Mixing
    style_mixing            = 0.9,            # Probability of mixing styles during training. None = disable
    component_mixing        = 0.0,            # Probability of mixing components during training. None = disable
    component_dropout       = 0.0,            # Dropout over the k latent components 0 = disable    
    **kwargs):                                # Arguments for sub-networks (mapping and synthesis)

    # Validate arguments
    assert not is_training or not is_validation
    assert isinstance(subnets, EasyDict)
    latents_in = tf.cast(latents_in, tf.float32)
    labels_in = tf.cast(labels_in, tf.float32)

    # Set options for training/validation
    # Set truncation_psi values if validations
    if is_validation:
        truncation_psi = truncation_psi_val
        truncation_cutoff = truncation_cutoff_val
    # Turn off truncation cutoff for training
    if is_training or (truncation_psi is not None and not tflib.is_tf_expression(truncation_psi) and truncation_psi == 1):
        truncation_psi = None
    if is_training:
        truncation_cutoff = None
    # Turn off update of w latent mean when not training
    if not is_training or (dlatent_avg_beta is not None and not tflib.is_tf_expression(dlatent_avg_beta) and dlatent_avg_beta == 1):
        dlatent_avg_beta = None
    # Turn off style and component mixing when not training
    if not is_training or (style_mixing is not None and not tflib.is_tf_expression(style_mixing) and style_mixing <= 0):
        style_mixing = None
    if not is_training or (component_mixing is not None and not tflib.is_tf_expression(component_mixing) and component_mixing <= 0):
        component_mixing = None
    # Turn off dropout when not training
    if not is_training:
        kwargs["attention_dropout"] = 0.0

    # Useful variables 
    k = kwargs["component_num"] 
    latent_size = kwargs["latent_size"]
    dlatent_size = kwargs["dlatent_size"]

    latents_num = k + int(kwargs.get("attention", False)) # k attention region-based latents + 1 global latent (like StyleGAN)

    resolution = kwargs["resolution"]
    resolution_log2 = int(np.log2(kwargs["resolution"]))

    # Initialize trainable positional embeddings for k latent components
    latent_pos = get_embeddings(k, dlatent_size, name = "ltnt_emb")
    # Initialize component dropout mask (disabled by default)
    component_mask = random_dp_binary([batch_size, k], component_dropout)
    component_mask = tf.expand_dims(component_mask, axis = 1)

    # Setup sub-networks
    # Set synthesis network
    if "synthesis" not in subnets:
        subnets.synthesis = tflib.Network("G_synthesis", func_name = globals()[synthesis_func], **kwargs)
    num_layers = subnets.synthesis.input_shape[2]
    # Set mapping network
    if "mapping" not in subnets:
        subnets.mapping = tflib.Network("G_mapping", func_name = globals()[mapping_func], 
            dlatent_broadcast = num_layers, **kwargs)

    # If latents_in used for dlatents (skipping mapping network), 
    # then need latent per each synthesis network layer, see StyleGAN for details.
    if take_dlatents: 
        latents_in.set_shape([None, latents_num, num_layers, latent_size])
    else:
        latents_in.set_shape([None, latents_num, latent_size])
    batch_size = get_shape(latents_in)[0]

    # Setup variables
    dlatent_avg = tf.get_variable("dlatent_avg", shape = [dlatent_size], initializer = tf.initializers.zeros(), trainable = False)

    if take_dlatents:
        dlatents = latents_in 
    else:
        # Evaluate mapping network
        dlatents = subnets.mapping.get_output_for(latents_in, labels_in, latent_pos, component_mask, is_training = is_training, **kwargs)
        dlatents = tf.cast(dlatents, tf.float32)

    # Update moving average of W latent space
    if dlatent_avg_beta is not None:
        with tf.variable_scope("DlatentAvg"):
            batch_avg = tf.reduce_mean(dlatents[:, :, 0], axis = [0, 1])
            update_op = tf.assign(dlatent_avg, tflib.lerp(batch_avg, dlatent_avg, dlatent_avg_beta))
            with tf.control_dependencies([update_op]):
                dlatents = tf.identity(dlatents)

    # Mixing (see StyleGAN): mixes together some of the W latents mapped from one set of 
    # source latents A, and another set of source latents B. Mixing can be between latents 
    # that correspond either to different layers or to the different k components used in 
    # the GANsformer.
    # Used only during training, and by default only layers are mixes as in StyleGAN.
    def mixing(latents_in, dlatents, prob, num, idx):
        if prob is None or prob == 0:
            return dlatents

        with tf.variable_scope("StyleMix"):
            latents2 = tf.random_normal(get_shape(latents_in))
            dlatents2 = subnets.mapping.get_output_for(latents2, labels_in, latent_pos, map_mask, is_training = is_training, **kwargs)
            dlatents2 = tf.cast(dlatents2, tf.float32)
            mixing_cutoff = tf.cond(
                tf.random_uniform([], 0.0, 1.0) < prob,
                lambda: tf.random_uniform([], 1, num, dtype = tf.int32),
                lambda: num)
            dlatents = tf.where(tf.broadcast_to(idx < mixing_cutoff, get_shape(dlatents)), dlatents, dlatents2)
        return dlatents

    # Perform style mixing regularization
    layer_idx = np.arange(num_layers)[np.newaxis, np.newaxis, :, np.newaxis]
    dlatents = mixing(latents_in, dlatents, style_mixing, num_layers, layer_idx)
    # Perform component mixing regularization
    ltnt_idx = np.arange(latents_num)[np.newaxis, :, np.newaxis, np.newaxis]
    dlatents = mixing(latents_in, dlatents, component_mixing, k, ltnt_idx)

    # Apply truncation (not in training or evaluation, only when sampling images, see StyleGAN for details)
    if truncation_psi is not None:
        with tf.variable_scope("Truncation"):
            layer_idx = np.arange(num_layers)[np.newaxis, np.newaxis, :, np.newaxis]
            layer_psi = np.ones(layer_idx.shape, dtype = np.float32)
            if truncation_cutoff is None:
                layer_psi *= truncation_psi
            else:
                layer_psi = tf.where(layer_idx < truncation_cutoff, layer_psi * truncation_psi, layer_psi)
            dlatents = tflib.lerp(dlatent_avg, dlatents, layer_psi)

    # Evaluate synthesis network
    images_out, maps_out = subnets.synthesis.get_output_for(dlatents, latent_pos, component_mask, 
        is_training = is_training, force_clean_graph = is_template_graph, **kwargs) 

    # Return requested outputs
    images_out = tf.identity(images_out, name = "images_out")
    maps_out = tf.identity(maps_out, name = "maps_out")
    ret = (images_out, maps_out)

    if return_dlatents:
        dlatents = tf.identity(dlatents, name = "dlatents_out")
        ret += (dlatents,)

    return ret

# Mapping network
# Mostly similar to the StylGAN version, with new options for the GANsformer such as:
# self-attention among latents, shared mapping of components, and support for mapping_dim != output_dim
def G_mapping(
    # Tensor inputs
    latents_in,                             # First input: Latent vectors (z) [batch_size, latent_size]
    labels_in,                              # Second input (optional): Conditioning labels [batch_size, label_size]
    latent_pos,                             # Third input: Positional embeddings for latents
    component_mask,                         # Fourth input: Component dropout mask (not used by default)
    # Dimensions
    component_num           = 1,            # Number of latent (z) vector components z_1,...,z_k
    latent_size             = 512,          # Latent (z) dimensionality per component
    dlatent_size            = 512,          # Disentangled latent (w) dimensionality
    label_size              = 0,            # Label dimensionality, 0 if no labels
    # General settings
    dlatent_broadcast       = None,         # Output disentangled latent (w) as [batch_size, dlatent_size] 
                                            # or [batch_size, dlatent_broadcast, dlatent_size]
    normalize_latents       = True,         # Normalize latent vectors (z) before feeding them to the mapping layers?
    attention               = False,        # Whereas the generator uses attention or not (i.e. GANsformer or StyleGAN)
    # Mapping network
    mapping_layersnum       = 8,            # Number of mapping layers
    mapping_dim             = None,         # Number of activations in the mapping layers
    mapping_lrmul           = 0.01,         # Learning rate multiplier for the mapping layers
    mapping_nonlinearity    = "lrelu",      # Activation function: relu, lrelu, etc.
    mapping_ltnt2ltnt       = False,        # Add self-attention over latents in the mapping network
    mapping_resnet          = False,        # Use resnet connections in mapping network
    mapping_shared_dim      = 0,            # Perform a shared mapping with that dimension to all latents concatenated together
    # Attention options
    num_heads               = 1,            # Number of attention heads 
    use_pos                 = False,        # Use positional encoding for latents
    ltnt_gate               = False,        # Gate attention scores so that info may not be sent/received when value is low
    attention_dropout       = 0.12,         # Attention dropout rate
    **_kwargs):                             # Ignore unrecognized keyword args
    
    # Short names
    act = mapping_nonlinearity
    k = component_num
    global_latent = attention # Whether to add a global latent vector to the k attention region-based latents
    latents_num = k + int(global_latent) # k region-based latents + 1 global latent (like StyleGAN)

    net_dim = mapping_dim
    layersnum = mapping_layersnum
    lrmul = mapping_lrmul
    ltnt2ltnt = mapping_ltnt2ltnt
    resnet = mapping_resnet
    shared_dim = mapping_shared_dim

    # Inputs
    latents_in.set_shape([None, latents_num, latent_size])
    labels_in.set_shape([None, label_size])  
    latent_pos.set_shape([k, dlatent_size])
    component_mask.set_shape([None, 1, k])

    batch_size = get_shape(latents_in)[0]

    if not use_pos: 
        latent_pos = None

    # Set internal and output dimensions
    if net_dim is None:
        net_dim = dlatent_size
        output_dim = None # Since mapping dimension is already dlatent_size
    else:
        output_dim = dlatent_size
        # Map inputs to the mapping dimension
        latents_in = tf.reshape(apply_bias_act(dense_layer(to_2d(latents_in), net_dim, name = "map_start"), name = "map_start"),
            [batch_size, latents_num, net_dim])
        latent_pos = apply_bias_act(dense_layer(latent_pos, net_dim, name = "map_pos"), name = "map_pos")

    # Concatenate labels (False by default)
    if label_size:
        with tf.variable_scope("LabelConcat"):
            w = tf.get_variable("weight", shape = [label_size, latent_size], initializer = tf.initializers.random_normal())
            l = tf.tile(tf.expand_dims(tf.matmul(labels_in, w), axis = 1), (1, latents_num, 1))
            x = tf.concat([x, l], axis = 1)

    # Split latents to region-based and global
    if global_latent:
        x, g = tf.split(latents_in, [k, 1], axis = 1)
        g = tf.squeeze(g, axis = 1)

    # Normalize latents
    if normalize_latents:
        with tf.variable_scope("Normalize"):
            x *= tf.rsqrt(tf.reduce_mean(tf.square(x), axis = -1, keepdims = True) + 1e-8)

    # Set optional attention arguments (By default, none since mapping_ltnt2ltnt = False)
    mlp_kwargs = EasyDict()
    if ltnt2ltnt:
        mlp_kwargs.update({        "attention": ltnt2ltnt, 
           "num_heads": num_heads, "att_dp": attention_dropout,
           "from_gate": ltnt_gate, "to_gate": ltnt_gate, 
           "from_pos": latent_pos, "to_pos": latent_pos
           "from_len": k,          "to_len": k, 
        })

    # Mapping layers
    if k == 0:
        x = tf.zeros([batch_size, 0, net_dim])
    else:
        # If shared mapping is set, then:
        if shared_dim > 0:
            # Concatenate all latents together (taken care by dense_layer function)
            x = apply_bias_act(dense_layer(x, shared_dim, name = "inmap"), name = "inmap")
            # Map the latents to the w space
            x = mlp(x, resnet, layersnum, shared_dim, act, lrmul, **mlp_kwargs)
            # Split the latents back to the k components
            x = tf.reshape(apply_bias_act(dense_layer(x, k * net_dim, name = "outmap"), name = "outmap"),
                [batch_size, k, net_dim])
        else:
            x = mlp(x, resnet, layersnum, net_dim, act, lrmul, pooling = "batch", 
                att_mask = component_mask, **kwargs)

    if global_latent:
        with tf.variable_scope("global"):
            # Map global latent to the w space
            g = mlp(g, resnet, layersnum, net_dim, act, lrmul)
        # Concatenate back region-based and global latents
        x = tf.concat([x, tf.expand_dims(g, axis = 1)], axis = 1)

    # Map latents to output dimension
    if output_dim is not None:
        x = tf.reshape(apply_bias_act(dense_layer(to_2d(x), output_dim, name = "map_end"), name = "map_end"),
            [batch_size, latents_num, output_dim])

    # Broadcast latents to all layers (so to control each of the resolution layers of the Synthesis network)
    if dlatent_broadcast is not None:
        with tf.variable_scope("Broadcast"):
            x = tf.tile(x[:, :, np.newaxis], [1, 1, dlatent_broadcast, 1])

    # Output
    x = tf.identity(x, name = "dlatents_out")
    return x

# Synthesis network
# Main differences from the StyleGAN version include the incorporation of transformer layers.
# This function supports different generator forms:
# * GANsformer (--attention)
# * GAN (--latent-stem --style = False)
# * StyleGAN (--style) 
# * SAGAN (--img2img)
# * k-GAN (--merge)
def G_synthesis(
    # Tensor inputs
    dlatents_in,                        # First input: Disentangled latents (W) [batch_size, num_layers, dlatent_size]
    latent_pos,                         # Second input: Positional embeddings for latents
    component_mask,                     # Third input: Component dropout mask (not used by default)
    # Dimensions and resolution
    dlatent_size        = 512,          # Disentangled latent (W) dimensionality
    pos_dim             = None,         # Positional embeddings dimension
    num_channels        = 3,            # Number of output color channels
    resolution          = 1024,         # Output resolution
    fmap_base           = 16 << 10,     # Overall multiplier for the network dimension
    fmap_decay          = 1.0,          # log2 network dimension reduction when doubling the resolution
    fmap_min            = 1,            # Minimum network dimension in any layer
    fmap_max            = 512,          # Maximum network dimension in any layer
    # General settings (StyleGAN/GANsformer related)
    architecture        = "skip",       # Architecture: orig, skip, resnet
    nonlinearity        = "lrelu",      # Activation function: relu, lrelu", etc.
    resample_kernel     = [1, 3, 3, 1], # Low-pass filter to apply when resampling activations, None = no filtering
    fused_modconv       = True,         # Implement modulated_conv2d_layer() as a single fused op?
    latent_stem         = False,        # If True, map latents to initial 4x4 image grid. Otherwise, use a trainable constant grid
    style               = True,         # Use global style modulation (StyleGAN)
    local_noise         = True,         # Add local stochastic noise during image synthesis
    randomize_noise     = True,         # If True, randomize noise inputs every time (non-deterministic)
                                        # Otherwise, read noise inputs from variables
    tanh                = False,        # Use tanh on output activations (turns out to be unnecessary and suboptimal)
    # GANsformer settings
    attention           = False,        # Whereas the generator uses attention or not (i.e. GANsformer or StyleGAN)
    component_num       = 1,            # Number of Latent (z) vector components z_1,...,z_k
    num_heads           = 1,            # Number of attention heads 
    attention_inputs    = "both",       # Attention function inputs: content, position or both
    attention_dropout   = 0.12,         # Attention dropout rate
    ltnt_gate           = False,        # Gate attention from latents, such that components may not send information 
                                        # when gate value is low
    img_gate            = False,        # Gate attention for images, such that some image positions may not get updated
                                        # or receive information when gate value is low
    integration         = "add",        # Feature integration type: additive, multiplicative or both
    norm                = None,         # Feature normalization type (optional): instance, batch or layer
    # Extra attention options, related to key-value duplex attention
    kmeans              = False,        # Track and update image-to-latents assignment centroids, used in the duplex attention
    kmeans_iters        = 1,            # Number of K-means iterations per transformer layer
                                        # Note that centroids are carried from layer to layer
    # Attention directions and layers: 
    # image->latents, latents->image, image->image (SAGAN), resolution to be applied at
    start_res           = 0,            # Transformer minimum resolution layer to be used at
    end_res             = 20,           # Transformer maximum resolution layer to be used at
    img2img             = 0,            # Add self-attention between images positions in that layer (SAGAN, not used by default)
    img2ltnt            = False,        # Add image to latents attention (bottom-up)
    ltnt2ltnt           = False,        # Add self-attention between the latents (not used by default)
    local_attention     = 0,            # Use Local attention operations instead of convolution up to this layer (not used by default)
    # Positional encoding
    use_pos             = False,        # Use positional encoding for latents
    pos_type            = "sinus",      # Positional encoding type: linear, sinus, trainable, trainable2d
    pos_init            = "uniform",    # Positional encoding initialization distribution: normal or uniform
    pos_directions_num  = 2,            # Positional encoding number of spatial directions
    # k-GAN options (not used by default)
    merge               = False,        # Generate component_num images and then merge them
    merge_layer         = -1,           # Merge layer, where images get combined through alpha-composition (-1 for last layer)
    merge_type          = "sum",        # Merge type: sum, softmax, max, leaves
    merge_same          = False,        # Merge images with same alpha weights across all spatial positions
    **_kwargs):                         # Ignore unrecognized keyword args

    # Set variables
    k = component_num
    act = nonlinearity
    latents_num = k + int(attention)
    resolution_log2 = int(np.log2(resolution))
    num_layers = resolution_log2 * 2 - 1
    if pos_dim is None:
        pos_dim = dlatent_size

    assert resolution == 2**resolution_log2 and resolution >= 4
    assert architecture in ["orig", "skip", "resnet"]

    # Network dimension, is set to fmap_base and then reduces with increasing stage 
    # according to fmap_decay, in the range of [fmap_min, fmap_max] (see StyleGAN)
    def nf(stage): return np.clip(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_min, fmap_max)
    # Get global latent (the last one)
    def get_global(dlatents): return dlatents[:, -1]

    # k-GAN variables (not used by default)
    m = merge_layer
    merge_final_layer = (m == -1)
    if m == -1: 
        m = resolution_log2 + 1
    
    # Inputs
    dlatents_in.set_shape([None, latents_num, num_layers, dlatent_size])
    component_mask.set_shape([None, 1, k]) 
    latent_pos.set_shape([k, dlatent_size])  
    # Disable latent_pos if we don't want to use it
    if not use_pos: 
        latent_pos = None

    batch_size = get_shape(dlatents_in)[0]

    # Positional encodings (spatial)
    # Image positional encodings, dictionary at different resolutions (used in GANsformer)
    grid_poses = get_positional_embeddings(resolution_log2, pos_dim, pos_type, pos_directions_num, pos_init)
    # Encodings for Local attention (not used by default)
    if local_attention:
        att2d_grid = get_embeddings(3 * 3, 16, name = "att2d_grid")

    # Local noise inputs to add to features during image synthesis (if --local-noise)
    noise_layers = []
    for layer_idx in range(num_layers - 1):
        # Infer layer resolution from its index
        res = (layer_idx + 5) // 2
        noise_shape = [kk, 1, 2**res, 2**res]  

        # For k-GAN only, need noise for each image out of the k 
        if merge and res < m:
            noise_shape[0] = k

        # Create new noise variable
        noise_layers.append(tf.get_variable("noise%d" % layer_idx, shape = noise_shape, 
            initializer = tf.initializers.random_normal(), trainable = False))

    def add_noise(x, layer_idx):
        if randomize_noise:
            # Random noise is same for all channels 
            shape = get_shape(x) # NCHW
            shape[1] = 1
            noise = tf.random_normal(shape)
        else:
            # For k-GAN only, before merging, features have dimension batch_size * k
            if merge:
                batch_mul = tf.cast(get_shape(x)[0] / get_shape(noise)[0], tf.int32)
                noise = tf.tile(noise, (batch_mul, 1, 1, 1))
        
        strength = tf.get_variable("noise_strength", shape = [], initializer = tf.initializers.zeros())
        x += strength * noise
        return x

    # Single generator layer with transformer (optionally)
    def layer(x, dlatents, layer_idx, dim, kernel, att_vars, up = False):
        att_map = None
        # Image features resolution of current layer
        res = (layer_idx + 5) // 2
        # Global latent for features global modulation (as in StyleGAN)
        dlatent_global = get_global(dlatents)[:, layer_idx + 1] 
        # If bottom-up connections exist, use the (iteratively updated) latents
        # Otherwise, use the input latents corresponding to the current layer
        new_dlatents = None
        if dlatents is None:
            dlatents = dlatents_in[:, :-1, layer_idx + 1]

        # If local_attention, use it instead of convolution (not used by default)
        noconv = False
        if local_attention and res < local_attention:
            x = att2d_layer(x, dim, kernel, grid = att2d_grid)
            fused_modconv, noconv = False, True

        # Modulate image features and perform convolution
        x = modulated_conv2d_layer(x, dlatent_global, dim, kernel, up = up, 
            resample_kernel = resample_kernel, fused_modconv = fused_modconv, modulate = style, noconv = noconv) 
        shape = get_shape(x)

        # Transformer layer
        if attention and res >= start_res and res < end_res: 
            # Reshape image features for transformer layer [batch_size (B), img_feats (WH), feat_dim (C)]
            shape = get_shape(x)
            x = tf.transpose(tf.reshape(x, [shape[0], shape[1], shape[2] * shape[3]]), [0, 2, 1])

            ######## Top-down latents->image ########
            # Main Transformer layer: arguments and function call
            kwargs = EasyDict({
                "num_heads": num_heads,         # Number of attention heads
                "inputs": attention_inputs,     # Attention input: position, content, or both
                "integration": integration,     # Integration mode: additive, multiplicative, or both
                "norm": norm,                   # Feature normalization type: batch, instance of layer
                "att_mask": component_mask,     # Attention mask (to disable particular latents)
                "att_dp": attention_dropout,    # Attention dropout rate
                "from_gate": img_gate,          # Gate attention flow from images (see G_synthesis signature for details)
                "to_gate": ltnt_gate,           # Gate attention flow to latents (see G_synthesis signature for details)
                "from_pos": grid_poses[res]     # Positional encodings for the image
                "to_pos": latent_pos,           # Positional encodings for the latents
                "kmeans": kmeans,               # Track and update image-to-latents assignment centroids (in duplex)
                "kmeans_iters": kmeans_iters,   # Number of K-means iterations per layer
                "att_vars": att_vars,           # K-means variables carried over from layer to layer (only when --kmeans)
            })
            # Perform attention from image to latents, meaning that information will flow 
            # latents -> image (from/to naming matches with Google BERT repository)
            x, att_map, att_vars = transformer(from_tensor = x, to_tensor = dlatents, dim = dim, 
                name = "l2n", **kwargs)

            ##### Optional transformer layers (image->image, image->latents, latents->latents) #####                
            kwargs = EasyDict({
                "num_heads": num_heads, 
                "from_gate": ltnt_gate, 
                "to_pos": latent_pos, 
                "att_dp": attention_dropout, 
                "from_pos": latent_pos
            })
            
            # Latent->Latent self-attention
            if ltnt2ltnt:
                new_dlatents = nnlayer(dlatents, dlatent_size, y = dlatents, act = "lrelu", 
                    name = "l2l", to_pos = latent_pos, to_gate = ltnt_gate, **kwargs)
            
            # Bottom-Up: Image->Latent attention
            if img2ltnt:
                new_dlatents = nnlayer(dlatents, dlatent_size, y = x, act = "lrelu", 
                    name = "n2l", to_pos = grid_poses[res], to_gate = img_gate, **kwargs)
            
            # Image->Image self-attention (SAGAN) (not used by default)
            if res == img2img:
                kwargs["from_pos"] = kwargs["to_pos"] = grid_poses[res]
                kwargs["from_gate"] = kwargs["to_gate"] = img_gate
                x = transformer(from_tensor = x, to_tensor = x, dim = dim, name = "n2n", **kwargs)[0]

            # Reshape image features back to standard [NCHW]
            x = tf.reshape(tf.transpose(x, [0, 2, 1]), shape)

        # Add local stochastic noise to image features
        if local_noise:
            x = add_noise(x, layer_idx)
        # Biases and nonlinearity
        x = apply_bias_act(x, act = act)

        return x, new_dlatents, att_map, att_vars

    # The building block for the generator (two layers with a resnet connection)
    def block(x, res, dlatents, dim, att_vars, up = True): # res = 3..resolution_log2  = (None, None, None)
        t = x
        with tf.variable_scope("Conv0_up"):
            x, dlatents, att_map1, att_vars = layer(x, dlatents, layer_idx = res*2-5, 
                dim = dim, kernel = 3, up = up, att_vars = att_vars)

        with tf.variable_scope("Conv1"):
            x, dlatents, att_map2, att_vars = layer(x, dlatents, layer_idx = res*2-4, 
                dim = dim, kernel = 3, att_vars = att_vars)

        if architecture == "resnet":
            with tf.variable_scope("Skip"):
                t = conv2d_layer(t, dim = dim, kernel = 1, up = up, resample_kernel = resample_kernel)
                x = (x + t) * (1 / np.sqrt(2))

        att_maps = [att_map1, att_map2]
        return x, dlatents, att_maps, att_vars

    def upsample(y):
        with tf.variable_scope("Upsample"):
            return upsample_2d(y, k = resample_kernel)

    # Convert image features to output image (e.g. RGB)
    # - For GANsformer: optionally perform transformer on highest resolution (if end_res >= resolution_log2)
    # - For k-GAN: Merge the k images, if didn't merge so far
    # - optional tanh activation on outputs (not used by default)
    # - skip connections (for --g-arch skip)
    def torgb(t, y, res, dlatents): # res = 2..resolution_log2
        with tf.variable_scope("ToRGB"):
            if res == resolution_log2:
                # for k-GAN only: if didn't merge so far, merge now
                if merge and resolution_log2 <= m:
                    with tf.variable_scope("merge%s" % (res)):
                        t, dlatents = merge_images(t, dlatents, k, merge_type, merge_same)

                # If end_res >= resolution_log2, need to perform transformer on highest resolution too
                if attention and res <= end_res:
                    with tf.variable_scope("extraLayer"):
                        t = modulated_conv2d_layer(t, dlatents[:, res*2-3], dim = nf(res-1), 
                            kernel = 3, fused_modconv = fused_modconv, modulate = style) 

            # Convert image features to output image (with num_channels, e.g. RGB)
            t = modulated_conv2d_layer(t, dlatents[:, res*2-2], dim = num_channels, 
                kernel = 1, demodulate = False, fused_modconv = fused_modconv, modulate = style)
            t = apply_bias_act(t)

            # Optionally convert features to the range [-1,1] (disabled, turns out to not be useful)
            if tanh:
                t = tf.math.tanh(t)

            # Optional skip connections (see StyleGAN)
            if y is not None:
                t += y

            return t 

    # Early layers
    y, dlatents = None, None
    att_vars = {"from_asgn": None, "to_asgn": None}
    att_vars = {"assignments": None, "centroids": None}
    att_maps = []

    # Stem: initial 4x4 image grid: either constant (StyleGAN) or mapped from the latents (standard GAN)
    with tf.variable_scope("4x4"):
        if not latent_stem: # default
            with tf.variable_scope("Const"):
                stem_size = k if merge else 1
                x = tf.get_variable("const", shape = [stem_size, nf(1), 4, 4], initializer = tf.initializers.random_normal())
                x = tf.tile(x, [batch_size, 1, 1, 1])
        else:
            with tf.variable_scope("Dense"):
                dlatents_stem = get_global(dlatents_in)
                stem_inp = dlatents_stem[:, 0]

                x = dense_layer(stem_inp, dim = nf(1) * 16, gain = np.sqrt(2) / 4)
                x = tf.reshape(x, [-1, nf(1), 4, 4])

                if local_noise:
                    x = add_noise(x, layer_idx)
                x = apply_bias_act(x, act = act)

        with tf.variable_scope("Conv"):
            x, dlatents, amap, att_vars = layer(x, dlatents,
                layer_idx = 0, dim = nf(1), kernel = 3, att_vars = att_vars)
            att_maps.append(amap)
        if architecture == "skip":
            y = torgb(x, y, 2, get_global(dlatents_in))
    
    # Main layers
    sep = True
    for res in range(3, resolution_log2 + 1):
        with tf.variable_scope("%dx%d" % (2**res, 2**res)):
            should_merge = merge and sep and res >= m
            x, dlatents, amaps, att_vars = block(x, res, dlatents, 
                dim = nf(res-1), att_vars = att_vars)
            att_maps += amaps
            
            if should_merge: 
                with tf.variable_scope("merge%s" % (res)):
                    x, dlatents_in = merge_images(x, dlatents_in, k, merge_type, merge_same)
                sep = False

            if architecture == "skip":
                y = upsample(y)
            if architecture == "skip" or res == resolution_log2:
                y = torgb(x, y, res, get_global(dlatents_in))

    images_out = y

    def list2tensor(att_list):
        maps_out = []
        for amap in att_list:
            if amap is not None:
                s = int(math.sqrt(get_shape(amap)[2])) 
                amap = tf.transpose(tf.reshape(amap, [-1, s, s, k]), [0, 3, 1, 2])
                if s < resolution:
                    amap = upsample_2d(amap, factor = int(resolution / s))
                amap = tf.reshape(amap, [-1, num_heads, k, resolution, resolution])
                maps_out.append(amap)

        if len(maps_out) == 0:
            return None

        maps_out = tf.transpose(tf.stack(maps_out, axis = 1), [0, 3, 1, 2, 4, 5]) 
        return maps_out

    if attention:
        maps_out = list2tensor(att_maps)

    # Output
    if maps_out is None:
        maps_out = tf.zeros(1)

    return images_out, maps_out

# Discriminator 
# ----------------------------------------------------------------------------
##########################################################################################################################
# Note that the code of this function is not yet re-factored and cleaned-up. Will be fixed in a few hours, Stay Tuned!
##########################################################################################################################
def D_GANsformer(
    images_in,                          # First input: Images [batch_size, channel, height, width]
    labels_in,                          # Second input: Labels [batch_size, label_size]
    # Dimensions and resolution
    latent_size         = 512,          #
    label_size          = 0,            # Dimensionality of the labels, 0 if no labels
    pos_dim             = None,         # Positional embeddings dimension
    num_channels        = 3,            # Number of input color channels
    resolution          = 1024,         # Input resolution
    fmap_base           = 16 << 10,     # Overall multiplier for the network dimension
    fmap_decay          = 1.0,          # log2 network dimension reduction when doubling the resolution
    fmap_min            = 1,            # Minimum network dimension in any layer
    fmap_max            = 512,          # Maximum network dimension in any layer
    # General settings
    architecture        = "resnet",     # Architecture: orig, skip, resnet
    nonlinearity        = "lrelu",      # Activation function: relu, lrelu, etc
    mbstd_group_size    = 4,            # Group size for the minibatch standard deviation layer, 0 = disable
    mbstd_num_features  = 1,            # Number of features for the minibatch standard deviation layer
    resample_kernel     = [1, 3, 3, 1], # Low-pass filter to apply when resampling activations, None = no filtering
    # GANsformer settings
    attention           = False,        # Whereas the discriminator uses attention or not
    component_num       = 1,            # Number of aggregator variables
    num_heads           = 1,            # Number of attention heads 
    attention_inputs    = "both",       # Attention function inputs: content, position or both
    attention_dropout   = 0.12,         # Attention dropout rate
    ltnt_gate           = False,        # Gate attention from latents, such that components may not send information 
                                        # when gate value is low
    img_gate            = False,        # Gate attention for images, such that some image positions may not get updated
                                        # or receive information when gate value is low
    # Attention directions and layers: 
    # image->latents, latents->image, image->image (SAGAN), resolution to be applied at
    start_res           = 0,            # Transformer minimum resolution layer to be used at
    end_res             = 20,           # Transformer maximum resolution layer to be used at
    img2img             = 0,            # Add self-attention between images positions in that layer (SAGAN, not used by default)
    ltnt2ltnt           = False,        # Add self-attention between the aggregator variables (not used by default)
    ltnt2img            = False,        # Add aggregators to image attention (top-down)
    local_attention     = 0,            # Use Local attention operations instead of convolution to up this layer (not used by default)
    # Positional encoding
    use_pos             = False,        # Use positional encoding for latents
    pos_type            = "sinus",      # Positional encoding type: linear, sinus, trainable, trainable2d
    pos_init            = "uniform",    # Positional encoding initialization distribution: normal or uniform
    pos_directions_num  = 2,            # Positional encoding number of spatial directions    
    **_kwargs):                         # Ignore unrecognized keyword args
    
    # Set variables
    act = nonlinearity
    resolution_log2 = int(np.log2(resolution))
    if pos_dim is None:
        pos_dim = latent_size

    assert architecture in ["orig", "skip", "resnet"]
    assert resolution == 2**resolution_log2 and resolution >= 4

    # Network dimension, is set to fmap_base and then reduces with increasing stage 
    # according to fmap_decay, in the range of [fmap_min, fmap_max] (see StyleGAN)
    def nf(stage): return np.clip(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_min, fmap_max)

    # Inputs
    images_in.set_shape([None, num_channels, resolution, resolution])
    labels_in.set_shape([None, label_size])

    images_in = tf.cast(images_in, dtype)
    labels_in = tf.cast(labels_in, dtype)

    batch_size = get_shape(images_in)[0]

    # Positional encodings
    latent_pos = None
    if use_pos: 
        latent_pos = get_embeddings(component_num, latent_size, name = "ltnt_emb")    

    grid_poses = get_positional_embeddings(resolution_log2, pos_dim, pos_type, pos_directions_num, pos_init)
    if local_attention:
        att2d_grid = get_embeddings(3 * 3, 16, name = "att2d_grid")

    # Convert input image (e.g. RGB) to image features 
    def fromrgb(x, y, res, nf): # res = 2..resolution_log2
        with tf.variable_scope("FromRGB"):
            t = apply_bias_act(conv2d_layer(y, dim = nf(res-1), kernel = 1), act = act)
            # Optional skip connections (see StyleGAN)
            if x is not None:
                t += x
            return t

    # The building block for the discriminator
    def block(x, res, dlatent_objs, m, nfunc): # res = 2..resolution_log2
        if attention and res >= start_res and res < end_res:
            shape = get_shape(x)
            x = tf.transpose(tf.reshape(x, [batch_size, shape[1], shape[2] * shape[3]]), [0, 2, 1])

            if img2img > 0:
                if res == img2img:
                    kwargs = EasyDict({"num_heads": num_heads, "from_gate": img_gate, "to_gate": img_gate, 
                        "from_pos": grid_poses[res], "to_pos": grid_poses[res], "pos_type": pos_type, 
                        "att_dp": attention_dropout, "inputs": attention_inputs})

                    x = transformer(from_tensor = x, to_tensor = x, dim = latent_size, name = "n2n", **kwargs)[0]

            kwargs = EasyDict({"num_heads": num_heads, "to_gate": ltnt_gate, "from_gate": img_gate,
                "to_pos": grid_poses[res], "pos_type": pos_type, 
                "from_pos": latent_pos, "att_dp": attention_dropout, "resample_kernel": resample_kernel})

            dlatent_objs, att_map_n2l, _ = transformer(from_tensor = dlatent_objs, to_tensor = x, dim = latent_size, name = "n2l", **kwargs)

            if ltnt2ltnt:
                kwargs = EasyDict({"num_heads": num_heads, "from_gate": ltnt_gate, "to_gate": ltnt_gate, 
                    "from_pos": latent_pos, "to_pos": latent_pos, "att_dp": attention_dropout, "resample_kernel": resample_kernel})

                dlatent_objs = nnlayer(dlatent_objs, latent_size, act = "lrelu", y = dlatent_objs, 
                    name = "l2l", **kwargs) 

            if ltnt2img:
                kwargs = EasyDict({"num_heads": num_heads, "from_gate": img_gate, "to_gate": ltnt_gate,
                    "to_pos": latent_pos, "from_pos": grid_poses[res], "pos_type": pos_type, "resample_kernel": resample_kernel})

                x = transformer(from_tensor = x, to_tensor = dlatent_objs, dim = get_shape(x_flat)[-1], name = "l2n", **kwargs)[0]

            x = tf.reshape(tf.transpose(x, [0, 2, 1]), get_shape(x))

        t = x
        local_att = local_attention and (res < local_attention)
        kernel_size = 1 if local_att else 3
        if local_att:
            x = att2d_layer(x, nfunc(res-1), 3, att2d_grid)

        with tf.variable_scope("Conv0"):
            x = apply_bias_act(conv2d_layer(x, dim = nfunc(res-1), kernel = kernel_size), act = act)

        with tf.variable_scope("Conv1_down"):
            x = apply_bias_act(conv2d_layer(x, dim = nfunc(res-2), kernel = kernel_size, down = True, resample_kernel = resample_kernel), act = act)

        if architecture == "resnet":
            with tf.variable_scope("Skip"):
                t = conv2d_layer(t, dim = nfunc(res-2), kernel = 1, down = True, resample_kernel = resample_kernel)
                x = (x + t) * (1 / np.sqrt(2))
        return x, dlatent_objs
    
    def downsample(y):
        with tf.variable_scope("Downsample"):
            return downsample_2d(y, k = resample_kernel)

    # Main layers
    x, y = None, images_in
    
    dlatent_objs = None
    if False: # seg_mod = obj
        dlatent_objs = tf.get_variable(name = "dobjects", shape = [component_num, latent_size], initializer = tf.random_uniform_initializer())
        dlatent_objs = tf.tile(tf.expand_dims(dlatent_objs, axis = 0), [batch_size, 1, 1])

    for res in range(resolution_log2, 2, -1):
        with tf.variable_scope("%dx%d" % (2**res, 2**res)):
            if architecture == "skip" or res == resolution_log2:
                x = fromrgb(x, y, res, nf)
            x, dlatent_objs = block(x, res, dlatent_objs, m, nf)
            if architecture == "skip":
                y = downsample(y)

    # Final layers
    with tf.variable_scope("4x4"):         
        if architecture == "skip":
            x = fromrgb(x, y, 2, nf)
        if mbstd_group_size > 1:
            with tf.variable_scope("MinibatchStddev"):
                x = minibatch_stddev_layer(x, mbstd_group_size, mbstd_num_features)
        with tf.variable_scope("Conv"):
            x = apply_bias_act(conv2d_layer(x, dim = nf(1), kernel = 3), act = act)
        with tf.variable_scope("Dense0"):
            x = apply_bias_act(dense_layer(x, dim = nf(0)), act = act)

    # Output layer with label conditioning from "Which Training Methods for GANs do actually Converge?"
    with tf.variable_scope("Output"):
        x = apply_bias_act(dense_layer(x, dim = max(labels_in.shape[1], 1)))
        if labels_in.shape[1] > 0:
            x = tf.reduce_sum(x * labels_in, axis = 1, keepdims = True)

    if attention:
        with tf.variable_scope("Obj_scoring"):  
            if mbstd_group_size > 1:
                with tf.variable_scope("MinibatchStddev"):
                    dlatent_objs = tf.transpose(minibatch_stddev_layer(tf.transpose(dlatent_objs, [0, 2, 1]), mbstd_group_size, mbstd_num_features, sdims = 1), [0, 2, 1])
            shape = get_shape(dlatent_objs)
            dlatent_objs = to_2d(dlatent_objs)
            with tf.variable_scope("Dense"):
                dlatent_objs = apply_bias_act(dense_layer(dlatent_objs, latent_size), act = act)
            with tf.variable_scope("Output"):
                o = apply_bias_act(dense_layer(dlatent_objs, 1))
            o = tf.reshape(o, shape[:-1])
            x = tf.concat([x, o], axis = -1)

    # Output
    scores_out = tf.identity(x, name = "scores_out")
    return scores_out
