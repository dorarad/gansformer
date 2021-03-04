####################################################################################################################################################
# Note that the code from line 468 until the end of the file is not cleaned up yet! Code will get cleaned-up by March 4. Stay Tuned!
####################################################################################################################################################

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

# Tensor basic operations
# ----------------------------------------------------------------------------

# Returns a list of the shape of tensor, preferring static dimensions when available
def get_shape(tensor):
    shape = tensor.shape.as_list()
    non_static_indexes = [index for (index, dim) in enumerate(shape) if dim is None]

    if not non_static_indexes:
        return shape

    dyn_shape = tf.shape(tensor)
    for index in non_static_indexes:
        shape[index] = dyn_shape[index]
    return shape

# Given a tensor [batch_size, ...] computes the size of each element in the batch
def element_dim(x):
    return np.prod(get_shape(tensor)[1:])

# Flattens all dimensions of tensor except last one
def to_2d(x):
    if len(get_shape(x)) == 2:
        return x
    shape = get_shape(x)
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
    w = tf.cast(w, x.dtype)

    return tf.matmul(x, w)

# Apply bias and optionally an activation function
def apply_bias_act(x, act = "linear", alpha = None, gain = None, lrmul = 1, bias_var = "bias", name = None):
    if name is not None:
        bias_var = "{}_{}".format(bias_var, name)
    b = tf.get_variable(bias_var, shape=[get_shape(x)[1]], initializer = tf.initializers.zeros()) * lrmul
    return fused_bias_act(x, b = tf.cast(b, x.dtype), act = act, alpha = alpha, gain = gain)

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

# Normalization to be used in attention layers. Does not scale back features with scale and bias 
# Arguments:
# - x: [batch_size * x_len, channels]
# - x_len: number of elements in the x set (e.g. number of positions WH)
# - integration: type of integration: additive, multiplicative or both
# - norm: normalization type: instance or layer-wise
# Returns: normalized x tensor
def att_norm(x, x_len, integration, norm):
    shape = get_shape(x)
    x = tf.reshape(x, [-1, x_len] + get_shape(x)[1:])
    x = tf.cast(x, tf.float32)

    # instance axis if norm == "instance" and channel axis if norm == "layer"
    norm_axis = 1 if norm == "instance" else 2

    if integration in ["add", "both"]:
        x -= tf.reduce_mean(x, axis = norm_axis, keepdims = True)
    if integration in ["mul", "both"]:
        x *= tf.rsqrt(tf.reduce_mean(tf.square(x), axis = norm_axis, keepdims = True) + 1e-8)
    
    # return x to original shape
    x = tf.reshape(x, shape)
    return x

# Minibatch standard deviation layer
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
    y = tf.cast(y, x.dtype) # [Mn11]
    # Replicate over group and pixels
    y = tf.tile(y, [group_size, 1, s[2]] + ([s[3]] if sdims == 2 else [])) # [NnHW]
    # Append as new fmap
    return tf.concat([x, y], axis = 1) # [NCHW]

# Dropout and masking
# ----------------------------------------------------------------------------

# Create a random mask of chosen shape and with probability rate to be dropped
def random_dp_binary(shape, dropout):
    if dropout == 0:
        return tf.ones(shape, dtype = tf.int32)
    eps = tf.random.uniform(shape)
    keep_mask = tf.cast(eps >= dropout, dtype = tf.int32)
    return keep_mask

# Perform dropout
def dropout(x, dp, noise_shape = None):
    if dp is None or dp == 0.0:
        return x
    return tf.nn.dropout(x, keep_prob = 1.0 - dp, noise_shape = noise_shape)

# Set a mask on logits (-Inf where mask is 0)
def logits_mask(x, mask):
    return x + tf.cast(1 - tf.cast(mask, tf.int32), tf.float32) * -10000.0

# Positional encoding
# ----------------------------------------------------------------------------

# 2d linear embeddings [s, s, dim] in a [-rng, rng] range where s is the grid size and dim 
# the embedding dimension. Each embedding consists of 'num' units, with each unit measuring 
# positional similarity along another direction, uniformly spanning the 2d space.
def get_linear_embeddings(size, dim, num, rng = 1.0):
    pi = tf.constant(math.pi)
    theta = tf.range(0, pi, pi / num)
    dirs = tf.stack([tf.cos(theta), tf.sin(theta)], axis = -1)
    embs = tf.get_variable(name = "emb", shape = [num, int(dim / num)], initializer = tf.random_uniform_initializer())
    
    c = tf.linspace(-rng, rng, size)
    x = tf.tile(tf.expand_dims(c, axis = 0), [size, 1])
    y = tf.tile(tf.expand_dims(c, axis = 1), [1, size])
    xy = tf.stack([x,y], axis = -1)
    
    lens = tf.reduce_sum(tf.expand_dims(xy, axis = 2) * dirs, axis = -1, keepdims = True) #  
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
    # Extension to 'num' spatial directions. Each embedding consists of 'num' units, with each  
    # unit measuring positional similarity along another direction, uniformly spanning the 2d space.
    # Each such unit has a sinus and cosine components.
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

# 2d positional embeddings of dimension 'dim' in a range of resolution from 2x2 up to 'max_res x max_res'
# 
# pos_type: supports several types of embedding schemes:
# - sinus: (see "Attention is all you need")
# - linear: where each position gets a value of [-1, 1] * trainable_vector, in each spatial
#   direction based on its location.
# - trainable: where embedding of position [w,h] is [emb_w, emb_h] (independently in each spatial direction)
# - trainable2d: whereembedding of position [w,h] is emb_{w,h} (a different embedding for each position)
# 
# dir_num: Each embedding consists of 'dir_num' units, with each unit measuring positional similarity 
# along another direction, uniformly spanning the 2d space.
# 
# shared: True for using same embeddings for all directions
# init: uniform or normal for trainable embeddings initialization
def get_positional_embeddings(max_res, dim, pos_type = "sinus", dir_num = 2, shared = False, init = "uniform"):
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

# Non-linear layer with resnet connection. Accepts features 'x' of dimension 'dim', 
# and uses nonlinearty 'act'. Optionally performs attention from x to y 
# (meaning information flows y -> x)
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
# Optionally uses resnet connections and self-attention.
# If x dimensions are not [batch_size, x_dim], then first pools x, according to pooling:
# - mean: mean pooling
# - cnct: concatenate all spatial features together to one vector and maps them to dimension 'dim'
# - 2d: turns a tensor [..., dim] into [-1, dim] so to make a large batch with all elements of last axis.
def mlp(x, resnet, layers_num, dim, act, lrmul, pooling = "mean", attention = False, norm_type = None, **kwargs): 
    shape = get_shape(x)
    
    if len(get_shape(x)) > 2:
        if pooling == "cnct":
            with tf.variable_scope("Dense_pool"):
                x = apply_bias_act(dense_layer(x, dim), act = act)
        elif pooling == "2d":
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
        x = upsample_conv_2d(x, tf.cast(w, x.dtype), data_format = "NCHW", k = resample_kernel)
    elif down:
        x = conv_downsample_2d(x, tf.cast(w, x.dtype), data_format = "NCHW", k = resample_kernel)
    else:
        x = tf.nn.conv2d(x, tf.cast(w, x.dtype), data_format = "NCHW", strides = [1, 1, 1, 1], 
            padding = "SAME")
    return x

# Modulated convolution layer
def modulated_conv2d_layer(x, y, dim, kernel, up = False, down = False, demodulate = True, modulate = True, 
    resample_kernel = None, gain = 1, use_wscale = True, lrmul = 1, fused_modconv = True, noconv = False,
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
        # Scale input feature maps
        ww *= tf.cast(s[:, np.newaxis, np.newaxis, :, np.newaxis], w.dtype) # [BkkIO]
        if demodulate:
            # Scale output feature maps according to Scaling factor
            d = tf.rsqrt(tf.reduce_sum(tf.square(ww), axis = [1, 2, 3]) + 1e-8) # [BO] 
            ww *= d[:, np.newaxis, np.newaxis, np.newaxis, :] # [BkkIO]
    else:
        ww += tf.zeros_like(s[:, np.newaxis, np.newaxis, :, np.newaxis])

    # Reshape/scale input
    # If fused, reshape minibatch to convolution groups. Otherwise, scale input activations directly.
    if fused_modconv:
        x = tf.reshape(x, [1, -1, get_shape(x)[-2], get_shape(x)[-1]]) 
        w = tf.reshape(tf.transpose(ww, [1, 2, 3, 0, 4]), [ww.shape[1], ww.shape[2], ww.shape[3], -1])
    else:
        if modulate:
            x *= tf.cast(s[:, :, np.newaxis, np.newaxis], x.dtype) # [BIhw]

    # Convolution with optional up/downsampling
    if noconv:
        if up:
            x = upsample_2d(x, k = resample_kernel)
        elif down:
            x = downsample_2d(x, k = resample_kernel)
    else:
        if up:
            x = upsample_conv_2d(x, tf.cast(w, x.dtype), data_format = "NCHW", k = resample_kernel)
        elif down:
            x = conv_downsample_2d(x, tf.cast(w, x.dtype), data_format = "NCHW", k = resample_kernel)
        else:
            x = tf.nn.conv2d(x, tf.cast(w, x.dtype), data_format = "NCHW", strides = [1,1,1,1], padding = "SAME")

    # Reshape/scale output
    # If fused,  reshape convolution groups back to minibatch. Otherwise, scale back output activations.
    if fused_modconv:
        x = tf.reshape(x, [-1, dim, get_shape(x)[-2], get_shape(x)[-1]])
    elif modulate and demodulate:
        x *= tf.cast(d[:, :, np.newaxis, np.newaxis], x.dtype) # [BOhw] 

    return x

# Local attention block, potentially replacing convolution (Not used by default in GANsformer)
def att2d_layer(x, dim, kernel, grid, up = False, down = False, resample_kernel = None, num_heads = 4):
    if num_heads == "max":
        num_heads = dim

    if down:
        x = downsample_2d(x, k = resample_kernel)

    x = conv2d_layer(x, dim = dim, kernel = 1, weight_var = "newdim")

    # Transpose channel axis to last one
    shape = get_shape(x)
    depth = shape[1]
    x = tf.transpose(x, [0, 2, 3, 1])

    # extract local patches for each image position
    x_patches = tf.reshape(tf.image.extract_patches(x, sizes = [1, kernel, kernel, 1], 
        strides = [1, 1, 1, 1], rates = [1, 1, 1, 1], padding = "SAME"), [-1, kernel * kernel, depth])
    x_flat = tf.reshape(x, [-1, 1, depth])

    kwargs = {
        "num_heads": num_heads, 
        "integration": "add", 
        "from_pos": tf.zeros([1, depth]), 
        "to_pos": grid
    }

    # Perform local attention between each image position and its local neighborhood
    x_flat = transformer(from_tensor = x_flat, to_tensor = x_patches, dim = dim, name = "att2d", **kwargs)[0]
    x = tf.reshape(tf.transpose(x_flat, [0, 2, 1]), shape)

    if up:
        x = upsample_2d(x, k = resample_kernel)

    return x

# Transformer layer
# ----------------------------------------------------------------------------

# A helper function for the transformer
def transpose_for_scores(x, batch_size, num_heads, seq_len, head_size):  
    x = tf.reshape(x, [batch_size, seq_len, num_heads, head_size]) 
    x = tf.transpose(x, [0, 2, 1, 3])
    return x

####################################################################################################################################################
# Note that the code from this point until the end of the file is not cleaned up yet! Code will get cleaned-up by March 4. Stay Tuned!
####################################################################################################################################################

# comment about direction from to
def transformer(from_tensor, to_tensor, dim, from_len = None, to_len = None, att_mask = None, ltnt_gate = False, 
        img_gate = False, num_heads = 1, attention_dropout = 0.06, col_dp = 0.06, from_pos = None, to_pos = None, 
        pos_type = False, from_frozen = None, to_frozen = None, from_asgn = None, to_asgn = None,
        asgn_direct = False, attention_inputs = "both", kmeans = "non", 
        kmeans_iters = 1, integration = "add", normalize = False, 
        norm = None, q_act = "linear", v_act = "linear", c_act = "linear", k_act = "linear", 
        fixed_gate = False, name = ""):

    with tf.variable_scope("AttLayer_{}".format(name)):
        batch_size, from_shape = None, None
        attention_scores = tf.constant(0.0)
        size_head = int(dim / num_heads)

        outdim = get_shape(from_tensor)[-1]
        if integration == "both": 
            outdim *= 2

        if to_tensor is None:
            to_tensor = to_frozen
            to_frozen = None

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

        if from_frozen is not None:
            from_frozen = to_2d(from_frozen)
        if to_frozen is not None:
            to_frozen = to_2d(to_frozen)

        sh_f = (2 * size_head) if attention_inputs == "both" else size_head  

        if from_pos is not None:
            fp = apply_bias_act(dense_layer(from_pos, dim, name = "from_pos"), name = "from_pos", act = q_act) # _from_pos
        if to_pos is not None:
            tp = apply_bias_act(dense_layer(to_pos, dim, name = "to_pos"), name = "to_pos", act = k_act) # _to_pos

        query_layer = apply_bias_act(dense_layer(from_tensor, dim, name = "query"), name = "query", act = q_act) # [B*F, N*H]
        key_layer = apply_bias_act(dense_layer(to_tensor, dim, name = "key"), name = "key", act = k_act) # [B*T, N*H]

        to_sign = tf.tile(tf.one_hot(tf.range(to_len), to_len)[np.newaxis, np.newaxis], [batch_size, num_heads, 1, 1])
        sh_t = to_len

        from_sign = {"both": tf.concat([query_layer, fp], axis = -1), "content": query_layer, "pos": fp}.get(attention_inputs)
        
        _from_sign = from_sign = transpose_for_scores(from_sign, batch_size, num_heads, from_len, sh_f) # [B, N, F, H]
        _to_sign = to_sign = transpose_for_scores(to_sign, batch_size, num_heads, to_len, sh_t) # [B, N, T, H]

        if from_frozen is not None:
            query_layer += apply_bias_act(dense_layer(from_frozen, dim, name = "from_froz"), name = "from_froz") # [B*F, N*H]
        if to_frozen is not None:
            key_layer += apply_bias_act(dense_layer(to_frozen, dim, name = "to_froz"), name = "to_froz") # [B*F, N*H]
            value_layer += apply_bias_act(dense_layer(to_frozen, dim, name = "up_froz"), name = "up_froz") # [B*F, N*H]

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

        if not asgn_direct:
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

        if asgn_direct:
            att_scores = tf.constant(0.0)
            to_asgn = tf.tile(tf.get_variable("toasgn_init", shape=[1, num_heads, to_len, sh_f], 
                initializer = tf.initializers.random_normal()), [batch_size, 1, 1, 1]) 

        for i in range(kmeans_iters):
            with tf.variable_scope("iter_{}".format(i)):
                if asgn_direct:
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

                    att_scores = (att_scores + new_scores) if (not asgn_direct) else new_scores

                attention_scores = tf.multiply(att_scores, 1.0 / math.sqrt(float(size_head)))

                if att_mask is not None:
                    if len(get_shape(att_mask)) == 2:
                        value_layer *= tf.cast(att_mask[:, np.newaxis, :, np.newaxis], tf.float32) # tf.expand_dims(, axis = [1, 3])
                    else:
                        if len(get_shape(att_mask)) == 3:
                            att_mask = tf.expand_dims(att_mask, axis = 1) # [B, 1, F, T]
                        attention_scores = logits_mask(attention_scores, att_mask)

                attention_probs = tf.nn.softmax(attention_scores) # [B, N, F, T]
                attention_probs = dropout(attention_probs, attention_dropout)
                attention_probs = dropout(attention_probs, col_dp, [get_shape(attention_scores)[0], num_heads, 1, to_len])

                from_asgn = attention_probs
                to_asgn = from_asgn
                to_asgn = tf.transpose((to_asgn / (tf.reduce_sum(to_asgn, axis = -2, keepdims = True) + 1e-8)), [0, 1, 3, 2])

                if kmeans: 
                    from_asgn = None
                else:
                    from_asgn = to_asgn = None

        attention_probs = tf.nn.softmax(attention_scores) # [B, N, F, T]
        attention_probs = dropout(attention_probs, attention_dropout)
        attention_probs = dropout(attention_probs, col_dp, [get_shape(attention_scores)[0], num_heads, 1, to_len])

        if ltnt_gate:
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

        if fixed_gate:
            context_layer *= tf.reshape(tf.reduce_sum(attention_probs, axis = -1), [batch_size * from_len, 1])
        out = from_tensor
        if norm is not None: 
            out = att_norm(out, from_len, integration, norm)
        a_context_layer = m_context_layer = context_layer
        if integration == "full": 
            m_context_layer, a_context_layer = tf.split(context_layer, 2, axis = -1)
        if integration != "add":
            if c_act == "sigmoid":
                out *= m_context_layer 
            else:
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

    return out, (attention_probs, attention_scores), (None, from_asgn, to_asgn)

# 
def merge_images(x, dlatents, weights, res, component_mask, k, merge_type, merge_after, is_visualization = False,
        merge_channelwise = False, name = ""):
    _x, _dlatents = x, dlatents
    maps_out = None
    batch_size =  tf.cast(get_shape(x)[0] / k, tf.int32)
    x = tf.reshape(x, [batch_size, k] + list(x.shape[-3:]))

    with tf.variable_scope("weights%s" % (res)):
        if merge_type == "auto":
            weights = conv2d_layer(_x, dim = 1, kernel = 1)
            weights = tf.reshape(weights, [batch_size, k] + list(weights.shape[-3:]))
            if merge_channelwise:
                weights = tf.reduce_sum(weights, axis = [-2, -1], keepdims = True)
        component_mask = tf.reshape(component_mask, [batch_size, k] + [1 for _ in range(len(get_shape(weights)) - 2)])
        if merge_type == "softmax":
            weights = logits_mask(weights, component_mask)
        else:
            weights *= component_mask

        if merge_type in ["softmax", "auto"]:
            pre_weights = weights
            weights = tf.nn.softmax(weights, axis = 1)
            if merge_type == "auto" and not merge_after:
                maps_out = weights 
        elif merge_type == "max":
            weights = tf.one_hot(tf.math.argmax(weights, axis = 1), k, axis = 1)
        elif merge_type == "leaves":
            weight_list = tf.split(weights, k, axis = 1)
            alphas = tf.ones_like(weight_list[0]) 
            for d, weight in enumerate(weight_list[1:]):
                max_weight = tf.math.reduce_max(weights[:,:d + 1], )
                new_alphas = tf.sigmoid(weight - max_weight) 
                alphas = tf.concat([alphas * (1 - new_alphas), new_alphas], axis = 1)
            weights = alphas

    x = tf.reduce_sum(x * weights, axis = 1)

    if dlatents is not None and len(get_shape(_dlatents)) == 4:
        weights = tf.reduce_mean(tf.reduce_mean(weights, axis = 2), axis = [-2, -1], keepdims = True)
        dlatents = tf.tile(tf.reduce_sum(dlatents * weights, axis = 1, keepdims = True), [1, k, 1, 1])

    if is_visualization:
        x = tf.concat([x, _x], axis = 0)
        if dlatents is not None:
            if len(get_shape(_dlatents)) == 4:
                _dlatentsn = tf.reshape(tf.transpose(tf.tile(_dlatents[:, :, np.newaxis], [1, 1, k, 1, 1]), [0, 2, 1, 3, 4]), [get_shape(dlatents)[0] * k] + get_shape(dlatents)[1:])
            else:
                _dlatentsn = tf.reshape(tf.tile(_dlatents[:, np.newaxis], [1, k, 1, 1]), [get_shape(dlatents)[0] * k] + get_shape(dlatents)[1:])
            dlatents = tf.concat([dlatents, _dlatentsn], axis = 0) 
    
    return x, dlatents, maps_out

# Generator, composed of two sub-networks: mapping and synthesis, as defined below
# ----------------------------------------------------------------------------

def G_GANsformer(
    latents_in,                                         # First input: Latent vectors (Z) [minibatch, latent_size].
    labels_in,                                          # Second input: Conditioning labels [minibatch, label_size].
    truncation_psi          = 0.65,                     # Style strength multiplier for the truncation trick. None = disable.
    truncation_cutoff       = None,                     # Number of layers for which to apply the truncation trick. None = disable.
    truncation_psi_val      = None,                     # Value for truncation_psi to use during validation
    truncation_cutoff_val   = None,                     # Value for truncation_cutoff to use during validation
    dlatent_avg_beta        = 0.995,                    # Decay for tracking the moving average of W during training. None = disable.
    style_mixing       = 0.9,                           # Probability of mixing styles during training. None = disable.
    component_mixing         = 0.0,                     # Probability of mixing styles during training. None = disable.
    is_training             = False,                    # Network is under training? Enables and disables specific features.
    is_validation           = False,                    # Network is under validation? Chooses which value to use for truncation_psi.
    return_dlatents         = False,                    # Return dlatents in addition to the images?
    return_comps            = False,
    take_dlatents           = False,
    is_template_graph       = False,                    # True = template graph constructed by the Network class, False = actual evaluation.
    obj_vis                 = None,
    component_dropout       = 0.0,
    components              = EasyDict(),               # Container for sub-networks. Retained between calls.
    mapping_func            = "G_mapping",              # Build func name for the mapping network.
    synthesis_func          = "G_synthesis",            # Build func name for the synthesis network.
    skip_layers             = False,
    const_dim               = 0,
    **kwargs): # Arguments for sub-networks (mapping and synthesis)

    # Validate arguments
    assert not is_training or not is_validation
    assert isinstance(components, EasyDict)
    if is_validation:
        truncation_psi = truncation_psi_val
        truncation_cutoff = truncation_cutoff_val
    if is_training or (truncation_psi is not None and not tflib.is_tf_expression(truncation_psi) and truncation_psi == 1):
        truncation_psi = None
    if is_training:
        truncation_cutoff = None
    if not is_training or (dlatent_avg_beta is not None and not tflib.is_tf_expression(dlatent_avg_beta) and dlatent_avg_beta == 1):
        dlatent_avg_beta = None
    if not is_training or (style_mixing is not None and not tflib.is_tf_expression(style_mixing) and style_mixing <= 0):
        style_mixing = None
    if not is_training or (component_mixing is not None and not tflib.is_tf_expression(component_mixing) and component_mixing <= 0):
        component_mixing = None

    if not is_training:
        kwargs["attention_dropout"] = 0.0 # None

    k = component_num = kwargs["component_num"] # components.synthesis.input_shape[1]
    dlatent_size = kwargs["dlatent_size"] # components.synthesis.input_shape[3]

    batch_size = get_shape(maps_in)[0]

    resolution = kwargs["resolution"]
    resolution_log2 = int(np.log2(kwargs["resolution"]))
    ltnt_pos = get_embeddings(component_num, dlatent_size, name = "ltnt_emb")

    map_mask = syn_mask = tf.ones([batch_size, component_num], dtype = tf.int32)
    component_mask = random_dp_binary([batch_size, component_num], component_dropout)
    if obj_vis is not None:
        indx = tf.one_hot(obj_vis, component_num, dtype = tf.int32) 

        component_mask = tf.tile(indx[np.newaxis], [batch_size, 1])
    map_mask = syn_mask = component_mask 

    # Setup components
    g_kwargs = kwargs.copy()
    if gen_mod != "non" or gen_cond:
        g_kwargs["img_gate"] = False     

    if "synthesis" not in components:
        components.synthesis = tflib.Network("G_synthesis", func_name = globals()[synthesis_func], **g_kwargs) # ltnt_pos = ltnt_pos, grid_poses = grid_poses, 

    num_layers = components.synthesis.input_shape[2]

    if "mapping" not in components:
        components.mapping = tflib.Network("G_mapping", func_name = globals()[mapping_func], 
            dlatent_broadcast = num_layers, **kwargs)

    if take_dlatents:
        latents_in.set_shape([None, kwargs["component_num"] + int(kwargs.get("attention", False)), num_layers, kwargs["latent_size"]])
    else:
        latents_in.set_shape([None, kwargs["component_num"] + int(kwargs.get("attention", False)), kwargs["latent_size"]])

    # Setup variables
    lod_in = tf.get_variable("lod", initializer = np.float32(0), trainable = False)
    dlatent_avg = tf.get_variable("dlatent_avg", shape=[dlatent_size], initializer = tf.initializers.zeros(), trainable = False)

    if take_dlatents:
        dlatents = latents_in 
    else:
        # Evaluate mapping network
        dlatents = components.mapping.get_output_for(latents_in, labels_in, ltnt_pos, map_mask, is_training = is_training, **kwargs)
        dlatents = tf.cast(dlatents, tf.float32)

    # Update moving average of W
    if dlatent_avg_beta is not None and (not kwargs.get("ltnt2ltnt")):
        with tf.variable_scope("DlatentAvg"):
            batch_avg = tf.reduce_mean(dlatents[:, :, 0], axis = [0, 1])
            update_op = tf.assign(dlatent_avg, tflib.lerp(batch_avg, dlatent_avg, dlatent_avg_beta))
            with tf.control_dependencies([update_op]):
                dlatents = tf.identity(dlatents)

    # Perform style mixing regularization.
    if style_mixing is not None and style_mixing > 0:
        with tf.variable_scope("StyleMix"):
            latents2 = tf.random_normal(get_shape(latents_in))
            dlatents2 = components.mapping.get_output_for(latents2, labels_in, ltnt_pos, map_mask, is_training = is_training, **kwargs)
            dlatents2 = tf.cast(dlatents2, tf.float32)
            layer_idx = np.arange(num_layers)[np.newaxis, np.newaxis, :, np.newaxis]
            cur_layers = num_layers - tf.cast(lod_in, tf.int32) * 2
            mixing_cutoff = tf.cond(
                tf.random_uniform([], 0.0, 1.0) < style_mixing,
                lambda: tf.random_uniform([], 1, cur_layers, dtype = tf.int32),
                lambda: cur_layers)
            dlatents = tf.where(tf.broadcast_to(layer_idx < mixing_cutoff, get_shape(dlatents)), dlatents, dlatents2)

    if component_mixing is not None:
        with tf.variable_scope("ObjMix"):
            latents3 = tf.random_normal(get_shape(latents_in))
            dlatents3 = components.mapping.get_output_for(latents3, labels_in, ltnt_pos, map_mask, is_training = is_training, **kwargs)
            dlatents3 = tf.cast(dlatents3, tf.float32)
            obj_idx = np.arange(component_num + int(kwargs.get("attention", False)))[np.newaxis, :, np.newaxis, np.newaxis]
            mixing_cutoff = tf.cond(
                tf.random_uniform([], 0.0, 1.0) < component_mixing,
                lambda: tf.random_uniform([], 1, component_num, dtype = tf.int32),
                lambda: component_num)
            dlatents = tf.where(tf.broadcast_to(obj_idx < mixing_cutoff, get_shape(dlatents)), dlatents, dlatents3)

    # Apply truncation trick
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
    deps = []

    if "lod" in components.synthesis.vars:
        deps.append(tf.assign(components.synthesis.vars["lod"], lod_in))
    with tf.control_dependencies(deps):

        if return_dlatents:
            dlatents = tf.identity(dlatents, name = "dlatents_out")
        dlatents_all = dlatents

        images_out, comps_out, maps_out = components.synthesis.get_output_for(dlatents_all, ltnt_pos, syn_mask, segs, None, is_training = is_training, 
            force_clean_graph = is_template_graph, **g_kwargs) # grid_poses = grid_poses, 

    # Return requested outputs
    images_out = tf.identity(images_out, name = "images_out")
    maps_out = tf.identity(maps_out, name = "maps_out")

    ret = (images_out, maps_out)
    if return_comps:
        comps_out = tf.identity(comps_out, name = "comps_out")
        ret += (comps_out, )
    if return_dlatents:
        dlatents = tf.identity(dlatents, name = "dlatents")
        ret += (dlatents, )
    return ret

# Mapping network
def G_mapping(
    latents_in,                             # First input: Latent vectors (Z) [minibatch, latent_size].
    labels_in,                              # Second input: Conditioning labels [minibatch, label_size].
    ltnt_pos,
    component_mask,
    component_num                 = 1,            # Latent vector (Z) dimensionality.
    latent_size             = 512,          # Latent vector (Z) dimensionality.
    label_size              = 0,            # Label dimensionality, 0 if no labels.
    dlatent_size            = 512,          # Disentangled latent (W) dimensionality.
    dlatent_broadcast       = None,         # Output disentangled latent (W) as [minibatch, dlatent_size] or [minibatch, dlatent_broadcast, dlatent_size].
    mapping_layersnum          = 8,            # Number of mapping layers.
    mapping_dim             = None,          # Number of activations in the mapping layers.
    mapping_lrmul           = 0.01,         # Learning rate multiplier for the mapping layers.
    mapping_nonlinearity    = "lrelu",      # Activation function: "relu", "lrelu", etc.
    normalize_latents       = True,         # Normalize latent vectors (Z) before feeding them to the mapping layers?
    resnet_mlp              = False,
    dtype                   = "float32",    # Data type to use for activations and outputs.
    mapping_ltnt2ltnt              = False,
    attention                     = False,
    ltnt_gate              = False,
    img_gate               = False,
    num_heads               = 1,
    attention_dropout                  = 0.12,
    use_pos                 = False,
    mapping_shared          = 0,
    **_kwargs):                             # Ignore unrecognized keyword args

    act = mapping_nonlinearity
    k = component_num
    # dropout is divided by 2 since we apply two forms of dropout
    attention_dropout /= 2

    # Inputs
    latents_in.set_shape([None, k + int(attention), latent_size])
    latents_in = tf.cast(latents_in, dtype)

    batch_size = get_shape(latents_in)[0]

    ltnt_pos.set_shape([k, dlatent_size])
    ltnt_pos = tf.cast(ltnt_pos, dtype)

    if mapping_dim is not None:
        out_dim = dlatent_size
        dlatent_size = mapping_dim

        latents_in = tf.reshape(apply_bias_act(dense_layer(to_2d(latents_in), dlatent_size, name = "map_start"), name = "map_start"),
            [batch_size, k + int(attention), dlatent_size])

        if ltnt_pos is not None:
            ltnt_pos = apply_bias_act(dense_layer(ltnt_pos, dlatent_size, name = "map_pos"), name = "map_pos")

    x, b = tf.split(latents_in, [k, int(attention)], axis = 1)
    if attention:
        b = tf.squeeze(b, axis = 1)

    labels_in.set_shape([None, label_size])  
    labels_in = tf.cast(labels_in, dtype)

    component_mask.set_shape([None, k])
    component_mask = tf.cast(component_mask, dtype)

    if label_size:
        with tf.variable_scope("LabelConcat"):
            w = tf.get_variable("weight", shape=[label_size, latent_size], initializer = tf.initializers.random_normal())
            y = tf.tile(tf.expand_dims(tf.matmul(labels_in, tf.cast(w, dtype)), axis = 1), (1, k, 1))
            x = tf.concat([x, y], axis = 1)

    # Normalize latents
    if normalize_latents:
        with tf.variable_scope("Normalize"):
            x *= tf.rsqrt(tf.reduce_mean(tf.square(x), axis = -1, keepdims = True) + 1e-8)

    kwargs = EasyDict()
    if mapping_ltnt2ltnt:
        kwargs.update({"attention": mapping_ltnt2ltnt, "from_len": k, "to_len": k, "num_heads": num_heads, 
                       "ltnt_gate": ltnt_gate, "img_gate": img_gate, "attention_dropout": attention_dropout, "col_dp": attention_dropout})
        if use_pos:
            kwargs.update({"from_pos": ltnt_pos, "to_pos": ltnt_pos})

    # Mapping layers
    if k > 0:
        if mapping_shared > 0:
            x = apply_bias_act(dense_layer(x, mapping_shared, name = "inmap"), name = "inmap")
            x = mlp(x, resnet_mlp, mapping_layersnum, mapping_shared, act, mapping_lrmul, **kwargs)
            x = tf.reshape(apply_bias_act(dense_layer(x, k * dlatent_size, name = "outmap"), name = "outmap"),
                [batch_size, k, dlatent_size])
        else:
            x = mlp(x, resnet_mlp, mapping_layersnum, dlatent_size, act, mapping_lrmul, pooling = "2d", **kwargs)
    else:
        x = tf.zeros([batch_size, 0, dlatent_size])

    if attention:
        with tf.variable_scope("global"):
            b = mlp(b, resnet_mlp, mapping_layersnum, dlatent_size, act, mapping_lrmul)

        x = tf.concat([x, tf.expand_dims(b, axis = 1)], axis = 1)

    if mapping_dim is not None:
        x = tf.reshape(apply_bias_act(dense_layer(to_2d(x), out_dim, name = "map_end"), name = "map_end"),
            [batch_size, k + int(attention), out_dim])

    # Broadcast
    if dlatent_broadcast is not None:
        with tf.variable_scope("Broadcast"):
            x = tf.tile(x[:, :, np.newaxis], [1, 1, dlatent_broadcast, 1])

    # Output
    assert x.dtype == tf.as_dtype(dtype)
    x = tf.identity(x, name = "dlatents_out")
    return x

# Synthesis network
def G_synthesis(
    dlatents_in,                        # Input: Disentangled latents (W) [minibatch, num_layers, dlatent_size].
    ltnt_pos,
    component_mask,
    condition,
    dlatent_size        = 512,          # Disentangled latent (W) dimensionality.
    pos_dim             = None,
    num_channels        = 3,            # Number of output color channels.
    resolution          = 1024,         # Output resolution.
    fmap_base           = 16 << 10,     # Overall multiplier for the number of feature maps.
    fmap_decay          = 1.0,          # log2 feature map reduction when doubling the resolution.
    fmap_min            = 1,            # Minimum number of feature maps in any layer.
    fmap_max            = 512,          # Maximum number of feature maps in any layer.
    randomize_noise     = True,         # True = randomize noise inputs every time (non-deterministic), False = read noise inputs from variables.
    architecture        = "skip",       # Architecture: "orig", "skip", "resnet".
    nonlinearity        = "lrelu",      # Activation function: "relu", "lrelu", etc.
    dtype               = "float32",    # Data type to use for activations and outputs.
    resample_kernel     = [1, 3, 3, 1], # Low-pass filter to apply when resampling activations. None = no filtering.
    fused_modconv       = True,         # Implement modulated_conv2d_layer() as a single fused op?
    latent_stem         = False,
    style               = True,
    local_noise           = True,
    new_size            = True,
    is_visualization    = False,
    component_num             = 1,
    merge             = False,
    merge_type          = None,
    merge_channelwise       = False, 
    merge_layer     = -1,    
    attention                 = False,
    num_heads           = 1,
    attention_dropout              = 0.06,
    ltnt_gate          = False,
    img_gate           = False,
    use_pos             = False,
    pos_type            = "sinus",
    pos_init            = "uniform",
    grid_refine         = 0,
    img2ltnt  = False,
    ltnt2ltnt   = False,
    start_res           = 0, # att_idx
    end_res             = 100, # att_idx
    only_frozen         = False,
    masking_type        = "full",
    integration         = "add",
    attention_inputs            = "both",
    kmeans          = False,    
    asgn_direct         = False,
    kmeans_iters          = 1,
    norm          = None,
    fixed_gate          = False,
    tanh                = False,
    cond_dim            = 0,
    const_dim           = 0,
    skip_layers         = False,
    pos_directions_num  = 2,
    local_attention     = False,
    **_kwargs):                         # Ignore unrecognized keyword args

    k = component_num
    l = merge_layer
    resolution_log2 = int(np.log2(resolution))
    assert resolution == 2**resolution_log2 and resolution >= 4

    att2d_grid = get_embeddings(3 * 3, 16, name = "att2d_grid") if local_attention else None

    def nf(stage): 
        ret = np.clip(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_min, fmap_max)
        if merge and not attention:
            ret = int(max(ret, k) / k)
        if const_dim > 0:
            ret = const_dim
        return ret 

    assert architecture in ["orig", "skip", "resnet"]
    act = nonlinearity
    num_layers = resolution_log2 * 2 - 2 + int(new_size)
    
    merge_after = (l == -1)
    if l == -1: 
        l = resolution_log2 + 1

    images_out = None

    dnum = k + int(attention)
    # Primary inputs
    dlatents_in.set_shape([None, dnum, num_layers, dlatent_size])
    dlatents_in = tf.cast(dlatents_in, dtype)
    batch_size = get_shape(dlatents_in)[0]

    component_mask.set_shape([None, k]) 
    component_mask = tf.cast(component_mask, dtype)

    condition.set_shape([None, cond_dim, resolution, resolution])
    condition = tf.cast(condition, dtype)

    conditions = {resolution_log2: condition}
    for res in range(resolution_log2, 1, -1):
        conditions[res - 1] = downsample_2d(conditions[res])

    ltnt_pos.set_shape([k, dlatent_size])
    ltnt_pos = tf.cast(ltnt_pos, dtype)

    grid_poses = get_positional_embeddings(resolution_log2, pos_dim or dlatent_size, pos_type, pos_directions_num, init = pos_init, **_kwargs)

    weight_inputs = [1 for _ in range(resolution_log2 + 1)]
    if merge or (attention and latent_stem):
        weight_inputs = []
        for layer_idx in range(resolution_log2 + 1): 
            shape = [1, k, nf(layer_idx), 1, 1] 
            weights = tf.get_variable("component_%d" % layer_idx, shape = shape, 
                initializer = tf.ones_initializer(), trainable = False)
            weight_inputs.append(weights)

        shape = [1, k, num_channels, 1, 1]  
        weights = tf.get_variable("component_%d" % (resolution_log2 + 1), shape = shape, 
            initializer = tf.ones_initializer(), trainable = False)

        weight_inputs.append(weights)

    # Noise inputs
    noise_inputs = []
    for layer_idx in range(num_layers - 1):
        res = (layer_idx + 5) // 2
        kk = k if merge and res < l else 1 
        shape = [kk, 1, 2**res, 2**res]  
        noise_inputs.append(tf.get_variable("noise%d" % layer_idx, shape = shape, initializer = tf.initializers.random_normal(), trainable = False))

    def glob(d, res):
        if (merge and (res <= l)):
            if attention:
                d = d[:,:k]
            return tf.reshape(d, (-1, num_layers, dlatent_size)) 
        return d[:, -1]

    # Single convolution layer with all the bells and whistles.    
    def layer(x, dlatents, dlatents_next, layer_idx, dim, kernel, up = False, iterative = (None, None, None), att_map = None, dis_att = False):
        att_map_l2n = None
        res = (layer_idx + 5) // 2
        dlatent_global = glob(dlatents, res)[:, layer_idx + int(new_size)] 
        merge_params = (weight_inputs[res - 1], component_mask, merge_type, integration == "conv", att_map)

        if local_attention and res < 7:
            x = att2d_layer(x, dim, kernel, grid = att2d_grid)
            x = modulated_conv2d_layer(x, dlatent_global, dim, kernel, up = up, 
                resample_kernel = resample_kernel, fused_modconv = False, modulate = style, noconv = True) 
        else:
            x = modulated_conv2d_layer(x, dlatent_global, dim, kernel, up = up, 
                resample_kernel = resample_kernel, fused_modconv = fused_modconv, modulate = style) 
        if attention and not dis_att and res >= start_res and res < end_res: 
            shape = get_shape(x)

            x = tf.transpose(tf.reshape(x, [shape[0], shape[1], shape[2] * shape[3]]), [0, 2, 1])
            dlatent_objs = dlatents_next
            if dlatent_objs is None:
                dlatent_objs = dlatents[:, :-1, layer_idx + int(new_size)] #  if dlatents_next is None else dlatents_next
            if only_frozen:
                dlatent_objs = None 

            kwargs = EasyDict()
            kwargs.update({"num_heads": num_heads, "ltnt_gate": ltnt_gate, "img_gate": img_gate,
                "from_pos": grid_poses[res], "pos_type": pos_type, 
                "attention_dropout": attention_dropout, "col_dp": attention_dropout,
                "integration": integration, "norm": norm, "fixed_gate": fixed_gate,
                "norm": norm})

            if use_pos:
                kwargs.to_pos = ltnt_pos
            if kmeans:
                kwargs.attention_inputs = attention_inputs
                kwargs.kmeans = kmeans
                kwargs.asgn_direct = asgn_direct
                kwargs.kmeans_iters = kmeans_iters
                kwargs.from_asgn = iterative[1]
                kwargs.to_asgn = iterative[2]

            x, att_map_l2n, iterative = transformer(from_tensor = x, to_tensor = dlatent_objs, dim = dim, name = "l2n", 
                **kwargs)

            if ltnt2ltnt:
                y = x if img2ltnt else dlatent_objs 
                
                kwargs = EasyDict()
                to_pos = grid_poses[res] if img2ltnt else None
                kwargs.update({"num_heads": num_heads, "ltnt_gate": ltnt_gate, "img_gate": img_gate, 
                    "to_pos": to_pos, "pos_type": pos_type and img2ltnt, 
                    "attention_dropout": attention_dropout, "col_dp": attention_dropout})

                if use_pos:
                    kwargs.from_pos = ltnt_pos
                    if not img2ltnt:
                        kwargs.to_pos = ltnt_pos

                dlatents_next = nnlayer(dlatent_objs, dlatent_size, act = "lrelu", lrmul = 1, y = y, 
                    name = "nl2l", **kwargs)

            x = tf.reshape(tf.transpose(x, [0, 2, 1]), shape)

        if res == grid_refine and grid_refine > 0:
            shape = get_shape(x)
            x = tf.transpose(tf.reshape(x, [shape[0], shape[1], shape[2] * shape[3]]), [0, 2, 1])

            kwargs = EasyDict()
            kwargs.update({"num_heads": num_heads, "ltnt_gate": ltnt_gate, "img_gate": img_gate, 
                "from_pos": grid_poses[res], "to_pos": grid_poses[res], "pos_type": pos_type,
                "attention_dropout": attention_dropout, "col_dp": attention_dropout})

            x, att_map_n2n, _ = transformer(from_tensor = x, to_tensor = x, dim = dim, name = "n2n", **kwargs)
            x = tf.reshape(tf.transpose(x, [0, 2, 1]), shape)

        if randomize_noise:
            noise = tf.random_normal([get_shape(x)[0], 1, get_shape(x)[-2], get_shape(x)[-1]], dtype = x.dtype)
        else:
            noise = tf.cast(noise_inputs[layer_idx], x.dtype)
            noise = tf.tile(noise, (tf.cast(get_shape(x)[0] / get_shape(noise)[0], tf.int32), 1, 1, 1))
        noise_strength = tf.get_variable("noise_strength", shape = [], initializer = tf.initializers.zeros())
        if local_noise:
            x += noise * tf.cast(noise_strength, x.dtype)
        return apply_bias_act(x, act = act), dlatents_next, att_map_l2n, iterative

    # Building blocks for main layers
    def block(x, res, dlatents, dlatents_next, dim, iterative, att_map, up = True, dis_att = False, skip_layers = False): # res = 3..resolution_log2  = (None, None, None)
        t = x
        aa = []
        inds = [res*2-5, res*2-4]
        _iterative, _dlatents_next = iterative, dlatents_next
        with tf.variable_scope("Conv0_up"):
            x, dlatents_next, a1, iterative = layer(x, dlatents, dlatents_next, layer_idx = inds[0], dim = dim, kernel = 3, up = up, iterative = iterative, att_map = att_map, dis_att = dis_att)
            aa.append(a1)

        if not skip_layers:
            with tf.variable_scope("Conv1"):
                x, dlatents_next, a2, iterative = layer(x, dlatents, dlatents_next, layer_idx = inds[1], dim = dim, kernel = 3, iterative = iterative, att_map = a1, dis_att = dis_att)
                aa.append(a2)

        if architecture == "resnet":
            with tf.variable_scope("Skip"):
                t = conv2d_layer(t, dim = dim, kernel = 1, up = up, resample_kernel = resample_kernel)
                x = (x + t) * (1 / np.sqrt(2))

        return x, dlatents_next, aa, iterative

    def upsample(y):
        with tf.variable_scope("Upsample"):
            return upsample_2d(y, k = resample_kernel)

    def torgb(x, y, res, dlatents, att_map):
        maps_out = None
        merge_params = (weight_inputs[res - 1], component_mask, merge_type, integration == "conv", att_map)
        ind = res*2-3
        with tf.variable_scope("ToRGB"):
            t = x
            if merge_after:
                t = modulated_conv2d_layer(t, dlatents[:, res*2-3], dim = num_channels,
                        kernel = 1, demodulate = False, fused_modconv = fused_modconv, modulate = style) 
                t = apply_bias_act(t)

            if merge and res < l:
                weights = weight_inputs[-1 if merge_after else (res - 1)]
                t, dlatents_merged, maps_out = merge_images(t, None if attention else dlatents, weights, res,
                    component_mask, k, merge_type, merge_after, 
                    is_visualization and not attention, merge_channelwise = merge_channelwise)
                if dlatents_merged is not None:
                    dlatents = dlatents_merged

            if not merge_after:
                if end_res > 7:
                    with tf.variable_scope("extraLayer"):
                        t = modulated_conv2d_layer(t, dlatents[:, res*2-3], dim = nf(res-1), 
                            kernel = 3, fused_modconv = fused_modconv, modulate = style) 

                t = modulated_conv2d_layer(t, dlatents[:, res*2-2], dim = num_channels, 
                    kernel = 1, demodulate = False, fused_modconv = fused_modconv, modulate = style)
                t = apply_bias_act(t)

            if tanh:
                t = tf.math.tanh(t)
            return t if y is None else y + t

    # Early layers
    y, comps_out, maps_out, dlatents_next = None, None, None, None 
    iterative = (None, None, None)
    att_maps_l2n = []
    last_or_none = lambda a: a[-1][0] if len(a) > 0 and isinstance(a, tuple) else None

    with tf.variable_scope("4x4"):
        if not latent_stem:
            with tf.variable_scope("Const"):
                stem_size = k if merge else 1
                x = tf.get_variable("const", shape = [stem_size, nf(1), 4, 4], initializer = tf.initializers.random_normal())
                x = tf.tile(tf.cast(x, dtype), [batch_size, 1, 1, 1])
        else:
            with tf.variable_scope("Dense"):
                dlatents_stem = glob(dlatents_in, False, res = 2)
                stem_inp = dlatents_stem[:, 0]

                x = dense_layer(stem_inp, dim = nf(1) * 16, gain = np.sqrt(2) / 4)
                x = tf.reshape(x, [-1, nf(1), 4, 4])

                if randomize_noise:
                    noise = tf.random_normal([get_shape(x)[0], 1, get_shape(x)[-2], get_shape(x)[-1]], dtype = x.dtype)
                else:
                    noise = tf.cast(noise_inputs[0], x.dtype)
                    noise = tf.tile(noise, (tf.cast(get_shape(x)[0] / get_shape(noise)[0], tf.int32), 1, 1, 1))
                noise_strength = tf.get_variable("noise_strength", shape = [], initializer = tf.initializers.zeros())
                if local_noise:
                    x += noise * tf.cast(noise_strength, x.dtype)
                x = apply_bias_act(x, act = act)

        with tf.variable_scope("Conv"):
            x, dlatents_next, amap, iterative = layer(x, dlatents_in, dlatents_next, 
                layer_idx = 0, dim = nf(1), kernel = 3, iterative = iterative, att_map = last_or_none(att_maps_l2n))
            att_maps_l2n.append(amap)

        if architecture == "skip":
            y = torgb(x, y, 2, glob(dlatents_in, res = 2), last_or_none(att_maps_l2n)) # cout, 
    
    # Main layers
    sep = True
    for res in range(3, resolution_log2 + 1):
        with tf.variable_scope("%dx%d" % (2**res, 2**res)):
            should_merge = merge and sep and res >= l
            x, dlatents_next, amaps, iterative = block(x, res, dlatents_in, dlatents_next, 
                dim = nf(res-1), iterative = iterative, att_map = last_or_none(att_maps_l2n), 
                skip_layers = skip_layers)
            att_maps_l2n += amaps
            
            if should_merge: 
                x, dlatents_merged, maps_out = merge_images(x, None if attention else dlatents_in, weight_inputs[res - 1], res, # comps_out, 
                    component_mask, k, merge_type, merge_after, 
                    is_visualization and not attention)
                if dlatents_merged is not None:
                    dlatents_in = dlatents_merged
                sep = False
                if attention:
                    amap0 = tf.transpose(tf.reshape(maps_out[0], get_shape(maps_out[0])[:2] + [(2**res) * (2**res)]), [0, 2, 1])[:, np.newaxis]
                    amap1 = tf.transpose(tf.reshape(maps_out[1], get_shape(maps_out[1])[:2] + [(2**res) * (2**res)]), [0, 2, 1])[:, np.newaxis]

                    att_maps_l2n.append((amap0, amap1))

            if architecture == "skip":
                y = upsample(y)
            if architecture == "skip" or res == resolution_log2:
                y = torgb(x, y, res, glob(dlatents_in, res), att_map = last_or_none(att_maps_l2n)) # cout, 

    images_out = y
    if merge and is_visualization and not attention:
        images_out, comps_out = tf.split(images_out, [batch_size, k * batch_size], axis = 0)
        comps_out = tf.reshape(comps_out, [batch_size, k] + get_shape(comps_out)[1:])

    def list2tensor(att_list):
        maps_out = []
        for amap in att_list:
            if amap is not None:
                if attention:
                    s = int(math.sqrt(get_shape(amap)[2])) 
                    amap = tf.transpose(tf.reshape(amap, [-1, s, s, k]), [0, 3, 1, 2])
                s = get_shape(amap)[2]
                if s < resolution:
                    amap = upsample_2d(amap, factor = int(resolution / s))
                amap = tf.reshape(amap, [-1, num_heads, k, resolution, resolution])
                maps_out.append(amap)
        if len(maps_out) > 0:
            maps_out = tf.transpose(tf.stack(maps_out, axis = 1), [0, 3, 1, 2, 4, 5]) 
        else:
            maps_out = None
        return maps_out

    if attention:
        att_maps_l2n = [a for a in att_maps_l2n if a is not None]
        att_probs_l2n, att_scores_l2n = zip(*att_maps_l2n)
        maps_out = list2tensor(att_probs_l2n)

    assert images_out.dtype == tf.as_dtype(dtype)

    if comps_out is None:
        comps_out = tf.zeros(1)
    else:
        assert comps_out.dtype == tf.as_dtype(dtype)    

    if maps_out is None:
        maps_out = tf.zeros(1)

    return images_out, comps_out, maps_out

# Discriminator 
# ----------------------------------------------------------------------------

def D_GANsformer(
    images_in,                          # First input: Images [minibatch, channel, height, width].
    labels_in,                          # Second input: Labels [minibatch, label_size].
    latents_in,
    dlatents_in,
    num_channels        = 3,            # Number of input color channels. Overridden based on dataset.
    resolution          = 1024,         # Input resolution. Overridden based on dataset.
    label_size          = 0,            # Dimensionality of the labels, 0 if no labels. Overridden based on dataset.
    pos_dim             = None,
    fmap_base           = 16 << 10,     # Overall multiplier for the number of feature maps.
    fmap_decay          = 1.0,          # log2 feature map reduction when doubling the resolution.
    fmap_min            = 1,            # Minimum number of feature maps in any layer.
    fmap_max            = 512,          # Maximum number of feature maps in any layer.
    architecture        = "resnet",     # Architecture: "orig", "skip", "resnet".
    nonlinearity        = "lrelu",      # Activation function: "relu", "lrelu", etc.
    mbstd_group_size    = 4,            # Group size for the minibatch standard deviation layer, 0 = disable.
    mbstd_num_features  = 1,            # Number of features for the minibatch standard deviation layer.
    dtype               = "float32",    # Data type to use for activations and outputs.
    resample_kernel     = [1, 3, 3, 1], # Low-pass filter to apply when resampling activations. None = no filtering.
    resnet_mlp          = False,
    component_num             = 1,
    attention           = False,
    num_heads           = 1,
    attention_dropout              = 0.06,
    ltnt_gate          = False,
    img_gate           = False,
    use_pos             = False,
    pos_type             = "sinus",
    latent_size          = 512,
    pos_init            = "uniform",
    ltnt2ltnt   = False,
    d_grid_latent_refine  = False,
    img2img       = False,
    dlatent_init        = "uniform",
    start_res             = 0,
    end_res               = 100, 
    no_background         = False,
    c_act                 = "linear",
    v_act                 = "linear",
    pos_directions_num    = 2,
    local_attention       = False,
    **_kwargs):                         # Ignore unrecognized keyword args.

    max_k = obj_max_num
    resolution_log2 = int(np.log2(resolution))
    assert resolution == 2**resolution_log2 and resolution >= 4

    def nf(stage): return np.clip(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_min, fmap_max)

    assert architecture in ["orig", "skip", "resnet"]
    act = nonlinearity

    images_in.set_shape([None, num_channels, resolution, resolution])
    labels_in.set_shape([None, label_size])
    batch_size = get_shape(images_in)[0]

    images_in = tf.cast(images_in, dtype)
    labels_in = tf.cast(labels_in, dtype)

    ltnt_pos = get_embeddings(max_k, latent_size, name = "ltnt_emb")
    grid_poses = get_positional_embeddings(resolution_log2, pos_dim or latent_size, pos_type, pos_directions_num, init = pos_init, **_kwargs)

    att2d_grid = get_embeddings(3 * 3, 16, name = "att2d_grid") if local_attention else None

    # Building blocks for main layers
    def fromrgb(x, y, res, nf): # res = 2..resolution_log2
        with tf.variable_scope("FromRGB"):
            t = apply_bias_act(conv2d_layer(y, dim = nf(res-1), kernel = 1), act = act)
            return t if x is None else x + t
    
    def block(x, res, dlatent_objs, m, nfunc): # res = 2..resolution_log2
        if seg_mod == "obj" or (attention and res >= start_res and res < end_res):
            shape = get_shape(x)
            x_flat = tf.transpose(tf.reshape(x, [batch_size, shape[1], shape[2] * shape[3]]), [0, 2, 1])

            if img2img > 0:
                if res == img2img:
                    kwargs = EasyDict()
                    kwargs.update({"num_heads": num_heads, "ltnt_gate": ltnt_gate, "img_gate": img_gate, 
                        "from_pos": grid_poses[res], "to_pos": grid_poses[res], "pos_type": pos_type, 
                        "attention_dropout": attention_dropout, "col_dp": attention_dropout})

                    x_flat, att_map_n2n, _ = transformer(from_tensor = x_flat, to_tensor = x_flat, dim = latent_size, name = "n2n", **kwargs)
            else:
                kwargs = EasyDict()
                kwargs.update({"num_heads": num_heads, "ltnt_gate": ltnt_gate, "img_gate": img_gate,
                    "to_pos": grid_poses[res], "pos_type": pos_type, 
                    "attention_dropout": attention_dropout, "col_dp": attention_dropout, "resample_kernel": resample_kernel})
                if use_pos:
                    kwargs.from_pos = ltnt_pos

                dlatent_objs, att_map_n2l, _ = transformer(from_tensor = dlatent_objs, to_tensor = x_flat, dim = latent_size, name = "n2l", **kwargs)

                if ltnt2ltnt:
                    kwargs = EasyDict()
                    kwargs.update({"num_heads": num_heads, "ltnt_gate": ltnt_gate, "img_gate": img_gate, 
                        "attention_dropout": attention_dropout, "col_dp": attention_dropout, "resample_kernel": resample_kernel})
                    if use_pos:
                        kwargs.update({"from_pos": ltnt_pos, "to_pos": ltnt_pos})                    

                    dlatent_objs = nnlayer(dlatent_objs, latent_size, act = "lrelu", lrmul = 1, y = dlatent_objs, 
                        name = "l2l", **kwargs) 

                if d_grid_latent_refine:
                    kwargs = EasyDict()
                    kwargs.update({"num_heads": num_heads, "ltnt_gate": ltnt_gate, "img_gate": img_gate,
                        "from_pos": grid_poses[res], "pos_type": pos_type, "resample_kernel": resample_kernel})
                    if use_pos:
                        kwargs.to_pos = ltnt_pos

                    x_flat, att_map_l2n, _ = transformer(from_tensor = x_flat, to_tensor = dlatent_objs, dim = get_shape(x_flat)[-1], name = "l2n", **kwargs)
            x = tf.reshape(tf.transpose(x_flat, [0, 2, 1]), get_shape(x))

        t = x
        att_cond = local_attention and (res < 7)
        if att_cond:
            x = att2d_layer(x, nfunc(res-1), 3, att2d_grid)

        with tf.variable_scope("Conv0"):
            x = apply_bias_act(conv2d_layer(x, dim = nfunc(res-1), kernel = 1 if att_cond else 3), act = act)

        with tf.variable_scope("Conv1_down"):
            x = apply_bias_act(conv2d_layer(x, dim = nfunc(res-2), kernel = 1 if att_cond else 3, down = True, resample_kernel = resample_kernel), act = act)

        if architecture == "resnet":
            with tf.variable_scope("Skip"):
                t = conv2d_layer(t, dim = nfunc(res-2), kernel = 1, down = True, resample_kernel = resample_kernel)
                x = (x + t) * (1 / np.sqrt(2))
        return x, dlatent_objs
    
    def downsample(y):
        with tf.variable_scope("Downsample"):
            return downsample_2d(y, k = resample_kernel)

    # Main layers
    if seg_mod in ["feat", "both", "shared", "obj", "bothobj", "shrbth", "patch"]:
        x = None
        y = images_in
        
        dlatent_objs = None
        if attention or seg_mod in ["obj", "bothobj"]:
            initializer = tf.random_uniform_initializer() if dlatent_init == "uniform" else tf.initializers.random_normal()
            dlatent_objs = tf.get_variable(name = "dobjects", shape = [max_k, latent_size], initializer = initializer)
            dlatent_objs = tf.tile(tf.expand_dims(dlatent_objs, axis = 0), [batch_size, 1, 1])

        for res in range(resolution_log2, 2, -1):
            with tf.variable_scope("%dx%d" % (2**res, 2**res)):
                if architecture == "skip" or res == resolution_log2:
                    x = fromrgb(x, y, res, nf)
                if seg_mod in ["shared", "shrbth", "patch"] and res > 3:
                    x = tf.concat([x, m], axis = 1)
                x, dlatent_objs = block(x, res, dlatent_objs, m, nf)
                if architecture == "skip":
                    y = downsample(y)
                m = downsample_2d(m)

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

    pre_x = x
    # Output layer with label conditioning from "Which Training Methods for GANs do actually Converge?"
    with tf.variable_scope("Output"):
        x = apply_bias_act(dense_layer(x, dim = max(labels_in.shape[1], 1)))
        if labels_in.shape[1] > 0:
            x = tf.reduce_sum(x * labels_in, axis = 1, keepdims = True)

    if attention or seg_mod in ["obj", "bothobj"]:
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
            if agg_scores:
                o = tf.reduce_mean(o, axis = -1, keepdims = True)
            else:
                x = tf.concat([x, o], axis = -1)

    scores_out = x

    # Output
    assert scores_out.dtype == tf.as_dtype(dtype)
    scores_out = tf.identity(scores_out, name = "scores_out")
    return scores_out
