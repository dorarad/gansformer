import numpy as np
import torch
import math
from training import misc
from torch_utils import misc as torch_misc
from torch_utils import persistence
from torch_utils.ops import conv2d_resample
from torch_utils.ops import upfirdn2d
from torch_utils.ops import bias_act
from torch_utils.ops import fma
from dnnlib import EasyDict

def float_dtype():
    return torch.float32

# Flatten all dimensions of a tensor except the fist/last one
def to_2d(x, mode):
    if len(x.shape) == 2:
        return x
    if mode == "last":
        return x.flatten(end_dim = -2) # x.reshape(-1, x.shape[-1])
    else:
        return x.flatten(1) # x.reshape(x.shape[0], element_dim(x))

# Normalize a tensor 
@torch_misc.profiled_function
def normalize(x, eps = 1e-8, mode = "l2"):
    dims = list(range(1, len(x.shape)))
    x = x.to(float_dtype())
    if mode == "l2":
        factor = (x.square().mean(dim = dims, keepdim = True) + eps).rsqrt()
    if mode == "inf":
        factor = 1 / x.norm(float("inf"), dim = dims, keepdim = True)
    return x * factor

# Get suitable memory format
def memory_format(channels_last):
    return torch.channels_last if channels_last else torch.contiguous_format

# Convert tensor to memory format
def format_memory(w, channels_last):
    if channels_last:
        return w.to(memory_format = torch.channels_last)
    return w

# Convert tensor to requested dtype and memory format
def convert(x, dtype = None, channels_last = False):
    return x.to(dtype = dtype or float_dtype(), memory_format = memory_format(channels_last))

# Return a nearest neighbors upsampling kernel
def nearest_neighbors_kernel(device, factor = 2):
    return upfirdn2d.setup_filter([1] * factor, device = device)

# Convert a torch.nn.Parameter to the necessary dtype and apply gain, to be used within 'forward'
def get_param(param, dtype, gain, reorder = False):
    if param is None:
        return None
    if gain != 1 and reorder:
        param = param * gain
    param = param.to(dtype)
    if gain != 1 and not reorder:
        param = param * gain
    return param

# Create a weight variable for a convolution or fully-connected layer. lrmul means learning-rate multiplier 
def get_weight(shape, gain = 1, use_wscale = True, lrmul = 1, channels_last = False):
    fan_in = np.prod(shape[1:])
    he_std = gain / np.sqrt(fan_in)

    # Equalized learning rate and custom learning rate multiplier
    if use_wscale:
        init_std = 1.0 / lrmul
        runtime_coef = he_std * lrmul
    else:
        init_std = he_std / lrmul
        runtime_coef = lrmul

    # Create variable
    w = torch.randn(shape) * init_std
    w = torch.nn.Parameter(format_memory(w, channels_last))
    return w, runtime_coef

# Create a bias variable for a convolution or fully-connected layer
def get_bias(num_channels, bias_init = 0, lrmul = 1):
    b = torch.nn.Parameter(torch.full([num_channels], np.float32(bias_init)))
    return b, lrmul

# Return the padding size for convolution
def get_padding(kernel_size, mode):
    if mode == "VALID":
        return 0
    else: # "SAME"
        return kernel_size // 2

# Return a mapping from features resolution to their dimension (number of channels)
def get_res2channels(channel_base, channel_max):
    return lambda res: int(min(channel_base // res, channel_max))

# Return the gain value for each architecture (to be multiplied by its output)
# For resent connections we divide the activations by sqrt(2) (following StyleGAN2)
def get_gain(arch):
    return np.sqrt(0.5) if arch == "resnet" else 1

# Return the global component from the latent variable (that globally modulate all the image features)
def get_global(ws): 
    return ws[:, -1]

# Return the local components from the latent variable (that interact with the image through spatial attention)
def get_components(ws): 
    return ws[:, :-1]    

# Bias-Activation layer act(x + b)
@persistence.persistent_class
class BiasActLayer(torch.nn.Module):
    def __init__(self, num_channels, bias = True, act = "linear", lrmul = 1, bias_init = 0, clamp = None, gain = 1):
        super().__init__()
        self.bias, self.b_gain = get_bias(num_channels, bias_init, lrmul) if bias else (None, None)
        self.out_gain = bias_act.activation_funcs[act].def_gain * gain
        self.out_clamp = clamp * gain if clamp is not None else None
        self.act = act

    def forward(self, x):
        b = get_param(self.bias, x.dtype, self.b_gain)
        return bias_act.bias_act(x, b, act = self.act, gain = self.out_gain, clamp = self.out_clamp)

# Fully-connected layer act(x@w + b)
@persistence.persistent_class
class FullyConnectedLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, bias = True, act = "linear", gain = 1, lrmul = 1, bias_init = 0):
        super().__init__()
        self.weight, self.w_gain = get_weight([out_channels, in_channels], gain = gain, lrmul = lrmul)
        self.bias,   self.b_gain = get_bias(out_channels, bias_init, lrmul) if bias else None
        self.act = act

    def forward(self, x, _x = None): # _x is added to the signature for backward-compatibility and isn't used
        w = get_param(self.weight, x.dtype, self.w_gain)
        b = get_param(self.bias, x.dtype, self.b_gain)

        if len(x.shape) > 2:
            x = to_2d(x, "first")

        if self.act == "linear" and b is not None:
            x = torch.addmm(b.unsqueeze(0), x, w.t())
        else:
            x = x.matmul(w.t())
            x = bias_act.bias_act(x, b, act = self.act)
        return x

# Resent Layer
@persistence.persistent_class
class ResnetLayer(torch.nn.Module):
    def __init__(self, channels, act = "linear", lrmul = 1, sa = False):
        super().__init__()
        self.fc0 = FullyConnectedLayer(channels, channels, act = act, lrmul = lrmul)
        self.fc1 = FullyConnectedLayer(channels, channels, lrmul = lrmul)

    def forward(self, x, _x):
        shape = x.shape
        batch = len(shape) > 2

        if batch:
            x = to_2d(x, "last")
        x = self.fc0(x)
        x = self.fc1(x)
        if batch:
            x = x.reshape(shape)

        x = torch.nn.functional.leaky_relu(x + _x, negative_slope = 0.2) 
        return x

# Multi-layer network with 'layers_num' layers, dimension 'dim', and nonlinearity 'act'.
# Optionally use resnet connections and self-attention.
# If x dimensions are not [batch_size, dim], then turn the input tensor [..., dim] into [-1, dim] 
# so to create one large batch with all the elements from the last axis.
@persistence.persistent_class
class MLP(torch.nn.Module):
    def __init__(self, channels, act, resnet = False, sa = False, batch = True, lrmul = 1, **sa_kwargs):
        super().__init__()

        self.layers_num = int(len(channels) / 2) if resnet else (len(channels) - 1)
        self.out_layer = FullyConnectedLayer(channels[-1], channels[-1], act = act, lrmul = lrmul)
        self.batch = batch
        self.sa = sa

        for idx in range(self.layers_num):
            in_dim = channels[idx]
            out_dim = channels[idx + 1]
            if sa:
                sa_layer = TransformerLayer(dim = in_dim, pos_dim = in_dim, 
                    from_dim = in_dim, to_dim = in_dim, **sa_kwargs)
                setattr(self, f"sa{idx}", sa_layer)

            if resnet:
                layer = ResnetLayer(in_dim, act = act, lrmul = lrmul)
                assert in_dim == out_dim
            else:
                layer = FullyConnectedLayer(in_dim, out_dim, act = act, lrmul = lrmul)
            
            setattr(self, f"l{idx}", layer)

    def forward(self, x, pos = None, mask = None):
        shape = x.shape
        if len(x.shape) > 2:
            x = to_2d(x, "last" if self.batch else "first") 

        for idx in range(self.layers_num):
            _x = x
            if self.sa:
                sa = getattr(self, f"sa{idx}")
                x = sa(from_tensor = x, to_tensor = x, from_pos = pos, to_pos = pos, att_mask = mask.unsqueeze(1))[0]

            layer = getattr(self, f"l{idx}")
            x = layer(x, _x)

        x = self.out_layer(x)

        x = x.reshape(*shape[:-1], -1)
        return x

# Convolution layer with optional upsampling or downsampling
@persistence.persistent_class
class Conv2dLayer(torch.nn.Module):
    def __init__(self,
        in_channels,                    # Number of input channels
        out_channels,                   # Number of output channels
        kernel_size,                    # Width and height of the convolution kernel
        bias            = True,         # Apply additive bias before the activation function?
        act             = "linear",     # Activation function: "relu", "lrelu", etc
        up              = 1,            # Integer upsampling factor
        down            = 1,            # Integer downsampling factor
        resample_kernel = [1,3,3,1],    # Low-pass kernel to apply when resampling activations
        gain            = 1,            # Scaling factor for the output tensor
    ):
        super().__init__()
        self.up = up
        self.down = down
        self.weight, self.w_gain = get_weight([out_channels, in_channels, kernel_size, kernel_size])
        self.biasAct = BiasActLayer(out_channels, bias, act, gain = gain)
        self.register_buffer("resample_kernel", upfirdn2d.setup_filter(resample_kernel))
        self.kernel_size = kernel_size

    def forward(self, x):
        # 'reorder' for first multiplying by gain and then type-converting to match original StyleGAN2-ADA implementation
        w = get_param(self.weight, x.dtype, self.w_gain, reorder = True) 
        x = conv2d_resample.conv2d_resample(x = x, w = w, f = self.resample_kernel, up = self.up, down = self.down, 
            padding = get_padding(self.kernel_size, "SAME"), flip_weight = (self.up == 1))
        return self.biasAct(x)

@torch_misc.profiled_function
def modulated_conv2d(
    x,                          # Input tensor of shape [batch_size, in_channels, in_height, in_width]
    weight,                     # Weight tensor of shape [out_channels, in_channels, kernel_height, kernel_width]
    styles,                     # Modulation coefficients of shape [batch_size, in_channels]
    noise           = None,     # Optional noise tensor to add to the output activations
    up              = 1,        # Integer upsampling factor
    down            = 1,        # Integer downsampling factor
    padding         = 0,        # Padding with respect to the upsampled image
    resample_kernel = None,     # Low-pass kernel to apply when resampling activations. Must be prepared beforehand by calling upfirdn2d.setup_filter()
    demodulate      = True,     # Apply weight demodulation?
    flip_weight     = True,     # False = convolution, True = correlation (matches torch.nn.functional.conv2d)
    fused_modconv   = True,     # Perform modulation, convolution, and demodulation as a single fused operation?
    modulate        = True      # Perform modulation or apply standard convolution
):
    # Shape assertions
    if not modulate:
        x = conv2d_resample.conv2d_resample(x, weight, f = resample_kernel, up = up, padding = padding, flip_weight = flip_weight)
        if noise is not None:
            x = x.add_(noise)
        return x

    s = styles
    batch_size = x.shape[0]
    out_channels, in_channels, kh, kw = weight.shape
    torch_misc.assert_shape(weight, [out_channels, in_channels, kh, kw]) # [OIkk]
    torch_misc.assert_shape(x, [batch_size, in_channels, None, None]) # [NIHW]
    torch_misc.assert_shape(styles, [batch_size, in_channels]) # [NI]
    if fused_modconv:
        # Assert input shape
        with torch_misc.suppress_tracer_warnings(): # this value will be treated as a constant
            batch_size = int(batch_size)
        torch_misc.assert_shape(x, [batch_size, in_channels, None, None])

    # Calculate per-sample weights and demodulation coefficients
    if demodulate or fused_modconv:
        w = weight.unsqueeze(0) # [1OIkk]
        w = w * s.reshape(batch_size, 1, -1, 1, 1) # [NOIkk]
    if demodulate:
        d = (w.square().sum(dim = [2,3,4]) + 1e-8).rsqrt() # [NO]
    if demodulate and fused_modconv:
        w = w * d.reshape(batch_size, -1, 1, 1, 1) # [NOIkk]

    if fused_modconv:
        # Type conversations
        w = w.to(x.dtype)

        # Execute as one fused op using grouped convolution
        x = x.reshape(1, -1, *x.shape[2:])
        w = w.reshape(-1, in_channels, kh, kw)
        x = conv2d_resample.conv2d_resample(x = x, w = w, f = resample_kernel, 
            up = up, down = down, padding = padding, groups = batch_size, flip_weight = flip_weight)
        
        # Reshape back to output shape and add noise
        x = x.reshape(batch_size, -1, *x.shape[2:])
        if noise is not None:
            x = x.add_(noise)    
    else:
        # Type conversations
        # w, s, d, noise = w.to(x.dtype), s.to(x.dtype), d.to(x.dtype), noise.to(x.dtype)

        # Modulate the activations before the convolution
        x = x * s.reshape(batch_size, -1, 1, 1)

        # Convolution
        x = conv2d_resample.conv2d_resample(x = x, w = weight, f = resample_kernel, 
            up = up, down = down, padding = padding, flip_weight = flip_weight)

        # Demodulation and noise
        if demodulate and noise is not None:
            x = fma.fma(x, d.reshape(batch_size, -1, 1, 1), noise)
        elif demodulate:
            x = x * d.reshape(batch_size, -1, 1, 1)
        elif noise is not None:
            x = x.add_(noise)

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
    if norm is None:
        return x

    shape = x.shape
    x = x.reshape([-1, num] + list(shape[1:])).to(float_dtype())

    # instance axis if norm == "instance" and channel axis if norm == "layer"
    norm_axis = 1 if norm == "instance" else 2

    if integration in ["add", "both"]:
        x = x - x.mean(dim = norm_axis, keepdim = True)
    if integration in ["mul", "both"]:
        x = x * torch.rsqrt(torch.square(x).mean(dim = norm_axis, keepdim = True) + 1e-8)

    # return x to its original shape
    x = x.reshape(shape)
    return x

# Dropout and masking
# ----------------------------------------------------------------------------

# Create a random mask of a chosen shape and with probability 'dropout' to be dropped (=0)
def random_dp_binary(shape, dropout, training, device):
    if not training or dropout == 0.0:
        return torch.ones(shape, device = device)
    eps = torch.rand(shape, device = device)
    keep_mask = (eps >= dropout)
    return keep_mask

# Perform dropout
def dropout(x, dp_func, noise_shape = None):
    noise_shape = noise_shape or x.shape
    return dp_func(torch.ones(noise_shape, device = x.device)) * x

# Set a mask for logits (set -Inf where mask is 0)
def logits_mask(x, mask): 
    return x + (1 - mask.to(torch.int32)).to(float_dtype()) * -10000.0

# Positional encoding
# ----------------------------------------------------------------------------

# 2d linear embeddings [size, size, dim] in a [-rng, rng] range where size = grid size and
# dim = the embedding dimension. Each embedding consists of 'num' parts, with each part measuring
# positional similarity along another direction, uniformly spanning the 2d space.
def get_linear_encoding(size, dim, num, rng = 1.0):
    theta = torch.arange(0, math.pi, step = math.pi / num)
    dirs = torch.stack([torch.cos(theta), torch.sin(theta)], dim = -1)
    embs = torch.nn.Parameter(torch.rand([num, int(dim / num)]))

    c = torch.linspace(-rng, rng, size)
    x = c.unsqueeze(0).tile([size, 1])
    y = c.unsqueeze(1).tile([1, size])
    xy = torch.stack([x, y], dim = -1)

    lens = (xy.unsqueeze(2) * dirs).sum(dim = -1, keepdim = True)
    emb = (lens * embs).reshape(size, size, dim)
    return emb

# 2d sinusoidal embeddings [size, size, dim] with size = grid size and dim = embedding dimension
# (see "Attention is all you need" paper)
def get_sinusoidal_encoding(size, dim, num = 2):
    # Standard positional encoding in the two spatial w,h directions
    if num == 2:
        c = torch.linspace(-1.0, 1.0, size).unsqueeze(-1)
        i = torch.arange(int(dim / 4)).to(float_dtype())

        peSin = torch.sin(c / (torch.pow(10000.0, 4 * i / dim)))
        peCos = torch.cos(c / (torch.pow(10000.0, 4 * i / dim)))

        peSinX = peSin.unsqueeze(0).tile([size, 1, 1])
        peCosX = peCos.unsqueeze(0).tile([size, 1, 1])
        peSinY = peSin.unsqueeze(1).tile([1, size, 1])
        peCosY = peCos.unsqueeze(1).tile([1, size, 1])

        emb = torch.cat([peSinX, peCosX, peSinY, peCosY], dim = -1)
    # Extension to 'num' spatial directions. Each embedding consists of 'num' parts, with each
    # part measuring positional similarity along another direction, uniformly spanning the 2d space.
    # Each such part has a sinus and cosine components.
    else:
        theta = torch.arange(0, math.pi, math.pi / num)
        dirs = torch.stack([torch.cos(theta), torch.sin(theta)], dim = -1)

        c = torch.linspace(-1.0, 1.0, size)
        x = c.unsqueeze(0).tile([size, 1])
        y = c.unsqueeze(1).tile([1, size])
        xy = torch.stack([x, y], dim = -1)

        lens = (xy.unsqueeze(2) * dirs).sum(dim = -1, keepdim = True)

        i = torch.arange(int(dim / (2 * num))).to(float_dtype())
        sins = torch.sin(lens / (torch.pow(10000.0, 2 * num * i / dim)))
        coss = torch.cos(lens / (torch.pow(10000.0, 2 * num * i / dim)))
        emb = torch.cat([sins, coss], dim = -1).reshape(size, size, dim)

    return emb

# 2d positional encoding of dimension 'dim' in a range of resolutions from 2x2 up to 'max_res x max_res'
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
def get_positional_encoding(
        res, 
        pos_dim,                # Positional encoding dimension
        pos_type = "sinus",     # Positional encoding type: linear, sinus, trainable, trainable2d
        pos_init = "uniform",   # Positional encoding initialization distribution: normal or uniform
        pos_directions_num = 2, # Positional encoding number of spatial directions
        shared = False,         # Share embeddings for x and y axes
        crop_ratio = None,      # Crop the embedding features to the ratio 
        **_kwargs):             # Ignore unrecognized keyword args

    params = []
    initializer = torch.rand if pos_init == "uniform" else torch.randn
    if pos_type == "sinus":
        emb = get_sinusoidal_encoding(res, pos_dim, num = pos_directions_num)
    elif pos_type == "linear":
        emb = get_linear_encoding(res, pos_dim, num = pos_directions_num)
    elif pos_type == "trainable2d":
        emb = torch.nn.Parameter(initializer([res, res, pos_dim]))
        params = [emb]
    else: # pos_type == "trainable"
        xemb = torch.nn.Parameter(initializer([res, int(pos_dim / 2)]))
        yemb = xemb if shared else torch.nn.Parameter(initializer(res, int(pos_dim / 2)))
        params = [xemb, yemb]
        xemb = xemb.unsqueeze(0).tile([res, 1, 1])
        yemb = yemb.unsqueeze(1).tile([1, res, 1])
        emb = torch.cat([xemb, yemb], dim = -1)

    emb = misc.crop_tensor(emb, crop_ratio)
    return emb, params

# Produce trainable embeddings of shape [size, dim], uniformly/normally initialized
def get_embeddings(size, dim, init = "uniform", name = None):
    if size == 0:
        return None
    initializer = torch.rand if init == "uniform" else torch.randn
    emb = torch.nn.Parameter(initializer([size, dim]))
    return emb

############################################# Transformer #############################################
# -----------------------------------------------------------------------------------------------------

# Transpose tensor to scores
def transpose_for_scores(x, num_heads, elem_num, head_size):
    x = x.reshape(-1, elem_num, num_heads, head_size) # [B, N, H, S]
    x = x.permute(0, 2, 1, 3) # [B, H, N, S]
    return x

# Compute attention probabilities: perform softmax on att_scores and dropout
def compute_probs(scores, dp_func):
    # Compute attention probabilities
    probs = torch.nn.functional.softmax(scores, dim = -1) # [B, N, F, T]
    shape = [int(d) for d in probs.shape]
    shape[-2] = 1
    # Dropout over random cells and over random full rows (randomly don't use a 'to' element)
    probs = dropout(probs, dp_func)
    probs = dropout(probs, dp_func, shape)
    return probs

# Compute relative weights of different 'from' elements for each 'to' centroid.
# Namely, compute assignments of 'from' elements to 'to' elements, by normalizing the
# attention distribution over the rows, to obtain the weight contribution of each
# 'from' element to the 'to' centroid.
# Returns [batch_size, num_heads, to_len, from_len] for each element in 'to'
# the relative weights of assigned 'from' elements (their weighted sum is the respective centroid)
def compute_assignments(att_probs):
    centroid_assignments = (att_probs / (att_probs.sum(dim = -2, keepdim = True) + 1e-8))
    centroid_assignments = centroid_assignments.permute(0, 1, 3, 2) # [B, N, T, F]
    return centroid_assignments

# (Optional, only used when --ltnt-gate, --img-gate)
#
# Gate attention values either row-wise (from) or column-wise so that some of the elements
# in the from/to_tensor will not participate in sending/receiving information, when gate
# value is low for them.
@persistence.persistent_class
class GateAttention(torch.nn.Module):
    def __init__(self, should_gate, dim, pos_dim, num_heads, from_len, to_len, gate_bias = 0):
        super().__init__()
        self.should_gate = should_gate
        self.from_len = from_len
        self.to_len = to_len
        self.num_heads = num_heads
        self.gate_bias = gate_bias

        if should_gate:
            self.gate = FullyConnectedLayer(dim, num_heads)
            self.gate_pos = FullyConnectedLayer(pos_dim, num_heads)

    def forward(self, att_probs, tensor, pos):
        if not self.should_gate:
            return att_probs
        gate = self.gate(tensor)
        if pos is not None:
            gate = gate + self.gate_pos(pos)
        gate = torch.sigmoid(gate + self.gate_bias)
        gate = gate.reshape(-1, self.from_len, self.to_len, self.num_heads).permute(0, 3, 1, 2)
        att_probs = att_probs * gate
        return att_probs

@persistence.persistent_class
class TransformerLayer(torch.nn.Module):
    def __init__(self,
            dim,                                    # The layer dimension
            pos_dim,                                # Positional encoding dimension
            from_len,           to_len,             # The from/to tensors length (must be specified if from/to has 2 dims)
            from_dim,           to_dim,             # The from/to tensors dimensions
            from_gate = False,  to_gate = False,    # Add sigmoid gate on from/to, so that info may not be sent/received
                                                    # when gate is low (i.e. the attention probs may not sum to 1)
            # Additional options
            num_heads           = 1,                # Number of attention heads
            attention_dropout   = 0.12,             # Attention dropout rate
            integration         = "add",            # Feature integration type: additive, multiplicative or both
            norm                = None,             # Feature normalization type (optional): instance, batch or layer

            # k-means options (optional, duplex)
            kmeans              = False,            # Track and update image-to-latents assignment centroids, used in the duplex attention
            kmeans_iters        = 1,                # Number of K-means iterations per transformer layer
            iterative           = False,            # Carry over attention assignments across transformer layers of different resolutions
                                                    # If True, centroids are carried from layer to layer            
            **_kwargs):                             # Ignore unrecognized keyword args

        super().__init__()
        self.dim = dim
        self.pos_dim = pos_dim
        self.from_len = from_len
        self.to_len = to_len
        self.from_dim = from_dim
        self.to_dim = to_dim
        
        self.num_heads = num_heads
        self.size_head = int(dim / num_heads)

        # We divide by 2 since we apply the dropout layer twice, over elements and over columns
        self.att_dp = torch.nn.Dropout(p = attention_dropout / 2) 

        self.norm = norm
        self.integration = integration        
        
        self.parametric = not iterative
        self.centroid_dim = 2 * self.size_head
        self.kmeans = kmeans
        self.kmeans_iters = kmeans_iters

        # Query, Key and Value mappings
        self.to_queries = FullyConnectedLayer(from_dim, dim)
        self.to_keys    = FullyConnectedLayer(to_dim, dim)
        self.to_values  = FullyConnectedLayer(to_dim, dim)

        # Positional encodings
        self.from_pos_map = FullyConnectedLayer(pos_dim, dim)
        self.to_pos_map   = FullyConnectedLayer(pos_dim, dim)

        # Attention gates
        self.to_gate_attention   = GateAttention(to_gate, dim, pos_dim, num_heads, from_len = 1, to_len = to_len)
        self.from_gate_attention = GateAttention(from_gate, dim, pos_dim, num_heads, from_len = from_len, to_len = 1, gate_bias = 1)

        # Features Integration
        control_dim = (2 * self.dim) if self.integration == "both" else self.dim 
        self.modulation = FullyConnectedLayer(self.dim, control_dim)

        # Centroids
        if self.kmeans:
            self.att_weight = torch.nn.Parameter(torch.ones(num_heads, 1, self.centroid_dim))
            if self.parametric:
                self.centroids = torch.nn.Parameter(torch.randn([1, num_heads, to_len, self.centroid_dim]))
            else:
                self.queries2centroids = FullyConnectedLayer(dim, dim * num_heads)

    # Validate transformer input shape for from/to_tensor and reshape to 2d
    def process_input(self, t, t_pos, name):
        shape = t.shape
        t_len = getattr(self, f"{name}_len")
        t_dim = getattr(self, f"{name}_dim")

        # from/to_tensor should be either 2 or 3 dimensions. If it's 3, then t_len should be specified.
        if len(shape) > 3:
            misc.error(f"Transformer {name}_tensor has {shape} shape. should be up to 3 dims.")
        elif len(shape) == 3:
            torch_misc.assert_shape(t, [None, t_len, t_dim])
            batch_size = shape[0]
        else:
            # Infer batch size for the 2-dims case
            torch_misc.assert_shape(t, [None, t_dim])
            batch_size = int(shape[0] / t_len)

        # Reshape tensors to 2d
        t = to_2d(t, "last")
        if t_pos is not None:
            t_pos = to_2d(t_pos, "last")
            torch_misc.assert_shape(t_pos, [t_len, self.pos_dim])
            t_pos = t_pos.tile([batch_size, 1])

        return t, t_pos, shape

    # Normalizes the 'tensor' elements, and then integrate the new information from
    # 'control' with 'tensor', where 'control' controls the bias/gain of 'tensor'.
    # norm types: batch, instance, layers
    # integration types: add, mul, both
    def integrate(self, tensor, tensor_len, control): # integration, norm
        # Normalize tensor
        tensor = att_norm(tensor, tensor_len, self.integration, self.norm)

        # Compute gain/bias
        bias = gain = control = self.modulation(control)
        if self.integration == "both":
            gain, bias = torch.split(control, 2, dim = -1)

        # Modulate the bias/gain of 'tensor'
        if self.integration != "add":
            tensor = tensor * (gain + 1)
        if self.integration != "mul":
            tensor = tensor + bias

        return tensor

    #### K-means (as part of Duplex Attention)
    # Basically, given the attention scores between 'from' elements to 'to' elements, compute
    # the 'to' centroids of the inferred assignments relations, as in the k-means algorithm.
    #
    # (Intuitively, given that the bed region will get assigned to one latent, and the chair region
    # will get assigned to another latent, we will compute the centroid/mean of that region and use
    # it as a representative of that region/object).
    # 
    # Given queries (function of the 'from' elements) and the centroid_assignemnts
    # between 'from' and 'to' elements, compute the centroid/mean queries.
    #
    # Some of the code here meant to be backward compatible with the pretrained networks
    # and may improve in further versions of the repository.
    def compute_centroids(self, _queries, queries, to_from, hw_shape):
        # We use [_queries, queries - _queries] for backward compatibility with the pretrained models
        from_elements = torch.cat([_queries, queries - _queries], dim = -1)
        from_elements = transpose_for_scores(from_elements, self.num_heads, self.from_len, self.centroid_dim) # [B, N, F, H]
        hw_shape = [int(s / 2) for s in hw_shape]
        # to_from represent centroid_assignments of 'from' elements to 'to' elements
        # [batch_size, num_head, to_len, from_len]
        if to_from is not None:
            # upsample centroid_assignments from the prior generator layer
            # (where image grid dimensions were x2 smaller)
            if to_from.shape[-2] < self.to_len:
                # s = int(math.sqrt(to_from.shape[-2]))
                to_from = upfirdn2d.upsample2d(to_from.reshape(-1, *hw_shape, self.from_len).permute(0, 3, 1, 2), 
                    f = nearest_neighbors_kernel(queries.device))
                to_from = to_from.permute(0, 2, 3, 1).reshape(-1, self.num_heads, self.to_len, self.from_len)

            if to_from.shape[-1] < self.from_len:
                # s = int(math.sqrt(to_from.shape[-1]))
                to_from = upfirdn2d.upsample2d(to_from.reshape(-1, self.to_len, *hw_shape), 
                    f = nearest_neighbors_kernel(queries.device))
                to_from = to_from.reshape(-1, self.num_heads, self.to_len, self.from_len)

            # Given:
            # 1. Centroid assignments of 'from' elements to 'to' centroid
            # 2. 'from' elements (queries)
            # Compute the 'to' respective centroids
            to_centroids = to_from.matmul(from_elements)

        # Centroids initialization
        if to_from is None or self.parametric:
            if self.parametric:
                to_centroids = self.centroids.tile([from_elements.shape[0], 1, 1, 1])
            else:
                to_centroids = self.queries2centroids(queries)
                to_centroids = transpose_for_scores(to_centroids, self.num_heads, self.to_len, self.centroid_dim)

        return from_elements, to_centroids

    # Transformer (multi-head attention) function originated from the Google-BERT repository.
    # https://github.com/google-research/bert/blob/master/modeling.py#L558
    #
    # We adopt their from/to notation:
    # from_tensor: [batch_size, from_len, dim] a list of 'from_len' elements
    # to_tensor: [batch_size, to_len, dim] a list of 'to_len' elements
    #
    # Each element in 'from_tensor' attends to elements from 'to_tensor',
    # Then we compute a weighted sum over the 'to_tensor' elements, and use it to update
    # the elements at 'from_tensor' (through additive/multiplicative integration).
    #
    # Overall it means that information flows in the direction to->from, or that the 'to'
    # modulates the 'from'. For instance, if from=image, and to=latents, then the latents
    # will control the image features. If from = to then this implements self-attention.
    #
    # We first project 'from_tensor' into a 'query', and 'to_tensor' into 'key' and 'value'.
    # Then, the query and key tensors are dot-producted and softmaxed to obtain
    # attention distribution over the to_tensor elements. The values are then
    # interpolated (weighted-summed) using this distribution, to get 'context'.
    # The context is used to modulate the bias/gain of the 'from_tensor' (depends on 'intervention').
    # Notation: B - batch_size, F - from_len, T - to_len, N - num_heads, H - head_size
    # Other arguments:
    # - att_vars: K-means variables carried over from layer to layer (only when --kmeans)
    # - att_mask: Attention mask to block from/to elements [batch_size, from_len, to_len]
    def forward(self, from_tensor, to_tensor, from_pos, to_pos, 
            att_vars = None, att_mask = None, hw_shape = None):
        # Validate input shapes and map them to 2d
        from_tensor, from_pos, from_shape = self.process_input(from_tensor, from_pos, "from")
        to_tensor,   to_pos,   to_shape   = self.process_input(to_tensor, to_pos, "to")

        att_vars = att_vars or {}
        to_from = att_vars.get("centroid_assignments")

        # Compute queries, keys and values
        queries = self.to_queries(from_tensor)
        keys    = self.to_keys(to_tensor)
        values  = self.to_values(to_tensor)
        _queries = queries

        # Add positional encodings to queries and keys
        if from_pos is not None:
            queries = queries + self.from_pos_map(from_pos)
        if to_pos is not None:
            keys = keys + self.to_pos_map(to_pos)

        if self.kmeans:
            from_elements, to_centroids = self.compute_centroids(_queries, queries, to_from, hw_shape)

        # Reshape queries, keys and values, and then compute att_scores
        values = transpose_for_scores(values,  self.num_heads, self.to_len,   self.size_head)  # [B, N, T, H]
        queries = transpose_for_scores(queries, self.num_heads, self.from_len, self.size_head)  # [B, N, F, H]
        keys = transpose_for_scores(keys,    self.num_heads, self.to_len,   self.size_head)  # [B, N, T, H]

        att_scores = queries.matmul(keys.permute(0, 1, 3, 2)) # [B, N, F, T]
        att_probs = None

        for i in range(self.kmeans_iters):
            if self.kmeans:
                if i > 0:
                    # Compute relative weights of different 'from' elements for each 'to' centroid
                    to_from = compute_assignments(att_probs)
                    # Given:
                    # 1. Centroid assignments of 'from' elements to 'to' centroid
                    # 2. 'from' elements (queries)
                    # Compute the 'to' respective centroids
                    to_centroids = to_from.matmul(from_elements)

                # Compute attention scores based on dot products between
                # 'from' queries and the 'to' centroids.
                att_scores = (from_elements * self.att_weight).matmul(to_centroids.permute(0, 1, 3, 2))

            # Scale attention scores given head size (see BERT)
            att_scores = att_scores / math.sqrt(float(self.size_head))
            # (optional, not used by default)
            # Mask attention logits using att_mask (to mask some components)
            if att_mask is not None:
                att_scores = logits_mask(att_scores, att_mask.unsqueeze(1))
            # Turn attention logits to probabilities (softmax + dropout)
            att_probs = compute_probs(att_scores, self.att_dp)
        # Gate attention values for the from/to elements
        att_probs = self.to_gate_attention(att_probs, to_tensor, to_pos)
        att_probs = self.from_gate_attention(att_probs, from_tensor, from_pos)

        # Compute relative weights of different 'from' elements for each 'to' centroid
        if self.kmeans:
            to_from = compute_assignments(att_probs)

        # Compute weighted-sum of the values using the attention distribution
        control = att_probs.matmul(values)      # [B, N, F, H]
        control = control.permute(0, 2, 1, 3)   # [B, F, N, H]
        control = control.reshape(-1, self.dim) # [B*F, N*H]
        # This newly computed information will control the bias/gain of the new from_tensor
        from_tensor = self.integrate(from_tensor, self.from_len, control)

        # Reshape from_tensor to its original shape (if 3 dimensions)
        if len(from_shape) > 2:
            from_tensor = from_tensor.reshape(from_shape)

        if hw_shape is not None:
            att_probs = att_probs.reshape(-1, *hw_shape, self.to_len).permute(0, 3, 1, 2) # [NCHW]

        return from_tensor, att_probs, {"centroid_assignments": to_from}

############################################## Generator ##############################################
# -----------------------------------------------------------------------------------------------------

# A mapping network from normally-sampled latents z1,...,zk to intermediate latents w1,...,wk
# The architecture includes feed-forward and optionally self-attention layers, 
# for the global (z_{k}) and local (z_{1},...,z_{k-1}) components
@persistence.persistent_class
class MappingNetwork(torch.nn.Module):
    def __init__(self,
        z_dim             = 512,      # Input latent (Z) dimensionality, 0 = no latent
        c_dim             = 0,        # Conditioning label (C) dimensionality, 0 = no label
        w_dim             = 512,      # Intermediate latent (W) dimensionality
        k                 = 1,        # Number of latent vector components z_1,...,z_k
        num_broadcast     = None,     # Number of intermediate latents to output, None = do not broadcast
        num_layers        = 8,        # Number of mapping layers
        embed_dim         = None,     # Label embedding dimensionality, None = same as w_dim
        layer_dim         = None,     # Number of intermediate channels in the mapping layers, None = same as w_dim
        act               = "lrelu",  # Activation function: "relu", "lrelu", etc
        lrmul             = 0.01,     # Learning rate multiplier for the mapping layers
        w_avg_beta        = 0.995,    # Decay for tracking the moving average of W during training, None = do not track
        # GANformer-related options
        transformer       = False,    # Whereas the generator uses attention or not (i.e. GANformer or StyleGAN)
        resnet            = False,    # Use resnet connections in mapping network
        shared            = False,    # Perform a shared mapping to all latents concatenated together
        ltnt2ltnt         = False,    # Add self-attention over latents in the mapping network
        ltnt_gate         = False,    # Gate attention scores so that info may not be sent/received when value is low
        normalize_global  = True,     # Normalize the input global latent 
                                      # (True is recommended. The flag is introduced for backwards compatibility)
        use_pos           = False,    # Use positional encoding for latents
        **transformer_kwargs          # Arguments for SynthesisLayer
    ):

        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.k = k
        self.num_broadcast = num_broadcast
        self.num_layers = num_layers
        self.w_avg_beta = w_avg_beta
        self.normalize_global = normalize_global
        self.use_pos = use_pos

        layer_dim = layer_dim or w_dim
        embed_dim = (embed_dim or z_dim) if c_dim > 0 else 0

        if c_dim > 0:
            self.embed = FullyConnectedLayer(c_dim, embed_dim)

        sa_kwargs = {
           "sa": ltnt2ltnt and not shared,   "batch":   not shared,
           "from_len":  k - 1,               "to_len":  k - 1,
           "from_gate": ltnt_gate,           "to_gate": ltnt_gate,
        }
        sa_kwargs.update(transformer_kwargs)

        layers = [layer_dim] * (num_layers - 1) + [w_dim]

        self.global_mlp = MLP([z_dim + embed_dim] + layers, act = act, resnet = resnet, lrmul = lrmul)
        if transformer:
            self.mlp = MLP([z_dim] + layers, act = act, resnet = resnet, lrmul = lrmul, **sa_kwargs)

        if num_broadcast is not None and w_avg_beta is not None:
            self.register_buffer("w_avg", torch.zeros([w_dim]))

        # GANformer-related setup
        self.transformer = transformer

    def forward(self, z, c, pos = None, mask = None, truncation_psi = 1, truncation_cutoff = None, skip_w_avg_update = False):
        # Embed, normalize, and concatenate inputs
        with torch.autograd.profiler.record_function("input"):
            torch_misc.assert_shape(z, [None, self.k, self.z_dim])
            if self.transformer:
                z, g = torch.split(z, [self.k - 1, 1], dim = 1)
                if self.normalize_global: 
                    g = normalize(g)
            
            z = normalize(z)
            x = g if self.transformer else z
            
            if self.c_dim > 0:
                torch_misc.assert_shape(c, [None, self.c_dim])
                y = self.embed(c.to(torch.float32))
                y = normalize(y)
                x = torch.cat([x, y.unsqueeze(1)], dim = -1)

        if mask is not None:
            torch_misc.assert_shape(mask, [None, self.k - 1])
        if pos is not None:
            torch_misc.assert_shape(pos, [self.k - 1, self.w_dim])

        # Main layers
        x = self.global_mlp(x)

        if self.transformer:
            p = self.mlp(z, pos = pos if self.use_pos else None, mask = mask)
            x = torch.cat([p, x], dim = 1)

        # Update moving average of W
        if self.w_avg_beta is not None and self.training and not skip_w_avg_update:
            with torch.autograd.profiler.record_function("update_w_avg"):
                self.w_avg.copy_(x.detach().mean(dim = (0, 1)).lerp(self.w_avg, self.w_avg_beta))

        # Broadcast
        if self.num_broadcast is not None:
            with torch.autograd.profiler.record_function("broadcast"):
                x = x.unsqueeze(2).repeat([1, 1, self.num_broadcast, 1])

        # Apply truncation
        if truncation_psi != 1:
            with torch.autograd.profiler.record_function("truncate"):
                assert self.w_avg_beta is not None
                if self.num_broadcast is None or truncation_cutoff is None:
                    x = self.w_avg.lerp(x, truncation_psi)
                else:
                    x[:, :, :truncation_cutoff] = self.w_avg.lerp(x[:, :, :truncation_cutoff], truncation_psi)
        return x

# A synthesis layer where intermediate latents (y) modulate the evolving image features (x)
@persistence.persistent_class
class SynthesisLayer(torch.nn.Module):
    def __init__(self,
        in_channels,                    # Number of input channels
        out_channels,                   # Number of output channels
        y_dim,                          # modulator dimensionality (W intermediate latents)
        k,                              # Number of latent vector components w_1,...,w_k
        out_resolution,                 # Resolution of this layer
        kernel_size     = 3,            # Convolution kernel size
        up              = 1,            # Integer upsampling factor
        local_noise     = True,         # Enable noise input?
        bias            = True,         # Add bias
        act             = "lrelu",      # Activation function: "relu", "lrelu", etc
        resample_kernel = [1,3,3,1],    # Low-pass kernel to apply when resampling activations
        gain            = 1,            # Scaling factor for the output tensor        
        style           = True,         # Use modulated convolution
        transformer     = False,        # Whereas the generator uses attention or not (i.e. GANformer or StyleGAN)
        use_pos         = False,        # Use positional encoding for latents
        ltnt_gate       = False,        # Gate attention from latents, such that components may not send information
                                        # when gate value is low
        img_gate        = False,        # Gate attention for images, such that some image positions may not get updated
                                        # or receive information when gate value is low
        **transformer_kwargs            # Arguments for SynthesisLayer
    ):
        super().__init__()

        self.affine = FullyConnectedLayer(y_dim, in_channels, bias_init = 1)
        self.weight, self.w_gain = get_weight([out_channels, in_channels, kernel_size, kernel_size])
        self.biasAct = BiasActLayer(out_channels, act = act, gain = gain) if bias else None
        self.style = style
        self.kernel_size = kernel_size

        self.out_res = out_resolution
        self.in_res = self.out_res // up
        self.up = up

        self.register_buffer("resample_kernel", upfirdn2d.setup_filter(resample_kernel))

        self.local_noise = local_noise
        if local_noise:
            noise_shape = list(misc.crop_tensor_shape((self.out_res, self.out_res), 
                transformer_kwargs.get("crop_ratio")))
            self.register_buffer("noise_const", torch.randn(noise_shape))
            self.noise_strength = torch.nn.Parameter(torch.zeros([]))

        self.transformer = None
        self.use_pos = use_pos

        if transformer:
            transformer_kwargs["pos_dim"] = transformer_kwargs.get("pos_dim") or y_dim
            grid_pos, params = get_positional_encoding(out_resolution, **transformer_kwargs)
            # Register positional encoding params/buffers
            if len(params) > 0:
                self.grid_pos = grid_pos
                for i, param in enumerate(params):
                    setattr(self, f"pos{i}", param)
            else:
                self.register_buffer("grid_pos", grid_pos)

            kwargs = {
                "from_len":  self.out_res * self.out_res, "to_len":  k - 1,         # The from/to tensor lengths
                "from_dim":  out_channels,                "to_dim":  y_dim,         # The from/to tensor dimensions
                "from_gate": img_gate,                    "to_gate": ltnt_gate,     # Gate attention flow between from/to tensors
            }
            kwargs.update(transformer_kwargs)
            self.transformer = TransformerLayer(dim = out_channels, **kwargs)

    def forward(self, x, y, att_vars = None, pos = None, mask = None, noise_mode = "random", fused_modconv = True):
        assert noise_mode in ["random", "const", "none"]
        torch_misc.assert_shape(x, [None, self.weight.shape[1], self.in_res, self.in_res])
        
        att_map, noise = None, None
        if self.local_noise and noise_mode != "none":
            if noise_mode == "random":
                noise = torch.randn([x.shape[0], 1, self.out_res, self.out_res], device = x.device)
            if noise_mode == "const":
                noise = self.noise_const
            noise = noise * self.noise_strength

        x = modulated_conv2d(x = x, weight = self.weight * self.w_gain, styles = self.affine(get_global(y)), 
            modulate = self.style, up = self.up, padding = get_padding(self.kernel_size, "SAME"),
            resample_kernel = self.resample_kernel, flip_weight = (self.up == 1), fused_modconv = fused_modconv)

        if self.transformer is not None:
            shape = x.shape
            x = x.reshape(shape[0], shape[1], -1).permute(0, 2, 1)
            x, att_map, att_vars = self.transformer(
                from_tensor = x,            to_tensor = get_components(y), 
                from_pos = self.grid_pos,   to_pos = pos if self.use_pos else None,
                att_vars = att_vars,        att_mask = mask.unsqueeze(1),
                hw_shape = shape[-2:]
            )
            x = x.permute(0, 2, 1).reshape(shape)

        if noise is not None:
            x = x.add_(noise) 

        if self.biasAct:
            x = self.biasAct(x)

        return x, att_map, att_vars

# An RGB-layer that maps the dense image features to RGB
@persistence.persistent_class
class ToRGBLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, y_dim, kernel_size = 1, style = True):
        super().__init__()
        self.affine = FullyConnectedLayer(y_dim, in_channels, bias_init = 1)
        self.weight, self.w_gain = get_weight([out_channels, in_channels, kernel_size, kernel_size])
        self.biasAct = BiasActLayer(out_channels)
        self.style = style

    def forward(self, x, y, fused_modconv):
        # Multiply styles by weight_gains instead of weight to match original StyleGAN2-ADA implementation
        styles = self.affine(get_global(y))
        weight = self.weight
        if self.style:
            styles = styles * self.w_gain
        else:
            weight = self.weight * self.w_gain
        x = modulated_conv2d(x = x, weight = weight, styles = styles, modulate = self.style, demodulate = False, 
            fused_modconv = fused_modconv)
        x = convert(self.biasAct(x))
        return x

# A synthesis block that composes together multiple synthesis layers (through standard, resnet or skip connections)
# Optionally includes a stem initialization for the first block or an RGB layer for the last one
@persistence.persistent_class
class SynthesisBlock(torch.nn.Module):
    def __init__(self,
        in_channels,                        # Number of input channels, 0 = first block
        out_channels,                       # Number of output channels
        w_dim,                              # Intermediate latent (W) dimensionality
        resolution,                         # Resolution of this block
        img_channels,                       # Number of output color channels
        is_last,                            # Is this the last block?
        architecture        = "skip",       # Architecture: "orig", "skip", "resnet"
        resample_kernel     = [1,3,3,1],    # Low-pass kernel to apply when resampling activations
        latent_stem         = False,        # If True, map latents to initial 4x4 image grid. Otherwise, use a trainable constant grid
        style               = True,         # Use modulated convolution
        **layer_kwargs                      # Arguments for SynthesisLayer
    ):
        assert architecture in ["orig", "skip", "resnet"]
        super().__init__()
        self.in_channels = in_channels
        self.img_channels = img_channels
        self.res = resolution
        self.w_dim = w_dim
        self.stem = (in_channels == 0)
        self.latent_stem = latent_stem
        self.is_last = is_last
        self.architecture = architecture
        self.register_buffer("resample_kernel", upfirdn2d.setup_filter(resample_kernel))

        self.num_conv, self.num_torgb = 0, 0

        if self.stem:
            self.init_shape = misc.crop_tensor_shape((self.res, self.res), layer_kwargs.get("crop_ratio"))
            if latent_stem:
                # optional todo: add local noise (to comply with TF version)
                self.conv_stem = FullyConnectedLayer(self.w_dim, out_channels * np.prod(self.init_shape), 
                    act = layer_kwargs.get(act, "lrelu"), gain = np.sqrt(2) / 4)                
                self.num_conv += 1
            else:
                self.const = torch.nn.Parameter(torch.randn([out_channels, *self.init_shape]))
        else:
            self.conv0 = SynthesisLayer(in_channels, out_channels, out_resolution = self.res, up = 2, resample_kernel = resample_kernel, 
                y_dim = w_dim, style = style, **layer_kwargs)
            self.num_conv += 1

        self.conv1 = SynthesisLayer(out_channels, out_channels, out_resolution = self.res, 
            gain = 1 if self.stem else get_gain(architecture), 
            y_dim = w_dim, style = style, **layer_kwargs)
        self.num_conv += 1

        if is_last or architecture == "skip":
            self.torgb = ToRGBLayer(out_channels, img_channels, y_dim = w_dim, style = style)
            self.num_torgb += 1

        if (not self.stem) and architecture == "resnet":
            self.skip = Conv2dLayer(in_channels, out_channels, kernel_size = 1, bias = False, up = 2,
                resample_kernel = resample_kernel, gain = get_gain(architecture))

        if is_last: # For backward compatibility with TF version
            conv_last_kwargs = layer_kwargs.copy()
            for disabled in ["transformer", "bias", "local_noise"]:
                conv_last_kwargs[disabled] = False
            self.conv_last = SynthesisLayer(out_channels, out_channels, out_resolution = self.res,
                y_dim = w_dim, style = style, **conv_last_kwargs)
            self.num_conv += 1

    def forward(self, x, img, ws, att_vars, fused_modconv = None, **layer_kwargs):
        torch_misc.assert_shape(ws, [None, None, self.num_conv + self.num_torgb, self.w_dim])
        w_iter = iter(ws.unbind(dim = 2))

        if fused_modconv is None:
            with torch_misc.suppress_tracer_warnings(): # this value will be treated as a constant
                fused_modconv = not self.training

        # Input
        if self.stem:
            batch_size = ws.shape[0]
            if self.latent_stem:
                x = self.conv_stem(get_global(next(w_iter)))
                x = x.reshape(batch_size, -1, *self.init_shape)
            else:
                x = self.const.unsqueeze(0).repeat([batch_size, 1, 1, 1])
        else:
            torch_misc.assert_shape(x, [None, self.in_channels, self.res // 2, self.res // 2])
        x = convert(x)

        # Main layers
        att_maps = [None, None]
        if self.stem:
            x, att_maps[0], att_vars = self.conv1(x, next(w_iter), att_vars, fused_modconv = fused_modconv, **layer_kwargs)
        elif self.architecture == "resnet":
            y = self.skip(x)
            x, att_maps[0], att_vars = self.conv0(x, next(w_iter), att_vars, fused_modconv = fused_modconv, **layer_kwargs)
            x, att_maps[1], att_vars = self.conv1(x, next(w_iter), att_vars, fused_modconv = fused_modconv, **layer_kwargs)
            x = y.add_(x)
        else:
            x, att_maps[0], att_vars = self.conv0(x, next(w_iter), att_vars, fused_modconv = fused_modconv, **layer_kwargs)
            x, att_maps[1], att_vars = self.conv1(x, next(w_iter), att_vars, fused_modconv = fused_modconv, **layer_kwargs)
        
        # ToRGB
        if img is not None:
            torch_misc.assert_shape(img, [None, self.img_channels, self.res // 2, self.res // 2])
            img = upfirdn2d.upsample2d(img, self.resample_kernel)
        if self.is_last: # For backward compatibility with TF version
            x = self.conv_last(x, next(w_iter), fused_modconv = fused_modconv, **layer_kwargs)[0]
        if self.is_last or self.architecture == "skip":
            y = self.torgb(x, next(w_iter), fused_modconv = fused_modconv)
            img = img.add_(y) if img is not None else y
        return x, img, att_maps, att_vars

# A synthesis network that maps the intermediate latents w1,...,wk into an output image x
# 
# Main differences from the StyleGAN version include the incorporation of transformer layers.
# This function supports different generator forms:
# - GANformer (--transformer)
# - GAN (--latent-stem --style = False)
# - StyleGAN (--style)
# See TF version for additional baselines (k-GAN and SAGAN)
@persistence.persistent_class
class SynthesisNetwork(torch.nn.Module):
    def __init__(self,
        w_dim,                         # Intermediate latent (W) dimensionality
        k,                             # Number of latent vector components w_1,...,w_k
        img_resolution,                # Output image resolution
        img_channels,                  # Number of color channels
        channel_base       = 32 << 10, # Overall multiplier for the number of channels
        channel_max        = 512,      # Maximum number of channels in any layer
        transformer        = False,    # Whereas the generator uses attention or not (i.e. GANformer or StyleGAN)
        start_res          = 0,        # Transformer minimum resolution layer to be used at
        end_res            = 20,       # Transformer maximum resolution layer to be used at
        **block_kwargs                 # Arguments for SynthesisBlock
    ):
        assert img_resolution >= 4 and img_resolution & (img_resolution - 1) == 0
        super().__init__()
        self.w_dim = w_dim
        self.k = k
        self.img_res = img_resolution
        self.img_channels = img_channels
        self.block_resolutions = [2 ** i for i in range(2, int(np.log2(img_resolution)) + 1)]
        channels_num = get_res2channels(channel_base, channel_max)
        self.num_heads = block_kwargs.get("num_heads", 1)

        self.num_ws = 0
        for res in self.block_resolutions:
            in_channels = channels_num(res // 2) if res > 4 else 0
            out_channels = channels_num(res)
            is_last = (res == self.img_res)
            use_transformer = transformer and np.log2(res) >= start_res and np.log2(res) < end_res
            block = SynthesisBlock(in_channels, out_channels, w_dim = w_dim, k = k, resolution = res,
                img_channels = img_channels, is_last = is_last,
                transformer = use_transformer, **block_kwargs)
            self.num_ws += block.num_conv
            if is_last:
                self.num_ws += block.num_torgb
            setattr(self, f"b{res}", block)

    # Convert the list of all attention maps from all layers into one tensor
    def list2tensor(self, att_list, device):
        att_list = [att_map for att_map in att_list if att_map is not None]
        if len(att_list) == 0:
            return torch.zeros([1], device = device)

        maps_out = []
        for att_map in att_list:
            # Reshape attention map into spatial
            # s = int(math.sqrt(int(att_map.shape[2])))
            # att_map = att_map.reshape(-1, s, s, self.k - 1).permute(0, 3, 1, 2) # [NCHW]
            # Upsample attention map to final image resolution
            # (since attention map of early generator layers have lower resolutions)
            s = att_map.shape[-1]
            if s < self.img_res:
                factor = int(self.img_res / s)
                att_map = upfirdn2d.upsample2d(att_map, f = nearest_neighbors_kernel(att_map.device, factor), up = factor)
            att_map = att_map.reshape(-1, self.num_heads, self.k - 1, self.img_res, self.img_res) # [NhkHW]
            maps_out.append(att_map)

        maps_out = torch.stack(maps_out, dim = 1).permute(0, 3, 1, 2, 4, 5) # [NklhHW]
        return maps_out

    def forward(self, ws, **block_kwargs):
        block_ws = []
        with torch.autograd.profiler.record_function("split_ws"):
            torch_misc.assert_shape(ws, [None, self.k, self.num_ws, self.w_dim])
            ws = ws.to(torch.float32)
            w_idx = 0
            for res in self.block_resolutions:
                block = getattr(self, f"b{res}")
                block_ws.append(ws.narrow(2, w_idx, block.num_conv + block.num_torgb))
                w_idx += block.num_conv

        x, img, att_maps = None, None, []
        att_vars = {"centroid_assignments": None}

        for res, cur_ws in zip(self.block_resolutions, block_ws):
            block = getattr(self, f"b{res}")
            x, img, _att_maps, att_vars = block(x, img, cur_ws, att_vars, **block_kwargs)
            att_maps += _att_maps
        att_maps = self.list2tensor(att_maps, ws.device)

        return img, att_maps

@persistence.persistent_class
class Generator(torch.nn.Module):
    def __init__(self,
        z_dim,                      # Input latent (Z) dimensionality
        c_dim,                      # Conditioning label (C) dimensionality
        w_dim,                      # Intermediate latent (W) dimensionality
        k,                          # Number of latent vector components z_1,...,z_k
        img_resolution,             # Output resolution
        img_channels,               # Number of output color channels
        component_dropout   = 0.0,  # Dropout over the latent components, 0 = disable
        mapping_kwargs      = {},   # Arguments for MappingNetwork
        synthesis_kwargs    = {},   # Arguments for SynthesisNetwork
        **_kwargs                   # Ignore unrecognized keyword args
    ):
        super().__init__()
        
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.k = k
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.component_dropout = component_dropout

        self.input_shape = [None, k, z_dim]
        self.cond_shape  = [None, c_dim]

        self.pos = get_embeddings(k - 1, w_dim)

        self.synthesis = SynthesisNetwork(w_dim = w_dim, k = k, img_resolution = img_resolution, 
            img_channels = img_channels, **synthesis_kwargs)
        self.num_ws = self.synthesis.num_ws

        self.mapping = MappingNetwork(z_dim = z_dim, c_dim = c_dim, w_dim = w_dim, k = k, 
            num_broadcast = self.num_ws, **mapping_kwargs)

    def forward(self, z = None, c = None, ws = None, truncation_psi = 1, truncation_cutoff = None, return_img = True, 
        return_att = False, return_ws = False, subnet = None, **synthesis_kwargs):
        return_tensor = False
        if subnet is not None:
            return_ws = (subnet == "mapping")
            return_img = (subnet == "synthesis")
            return_att = False
            return_tensor = True

        _input = z if z is not None else ws
        mask = random_dp_binary([_input.shape[0], self.k - 1], self.component_dropout, self.training, _input.device)

        if ws is None:
            ws = self.mapping(z, c, pos = self.pos, mask = mask, truncation_psi = truncation_psi, truncation_cutoff = truncation_cutoff)
        torch_misc.assert_shape(ws, [None, self.k, self.num_ws, self.w_dim])

        ret = ()
        if return_img or return_att:
            img, att_maps = self.synthesis(ws, pos = self.pos, mask = mask, **synthesis_kwargs)
            if return_img:  ret += (img, )
            if return_att:  ret += (att_maps, )

        if return_ws:  ret += (ws, )

        if return_tensor:
            ret = ret[0]

        return ret

############################################ Discriminator ############################################
# -----------------------------------------------------------------------------------------------------

@persistence.persistent_class
class DiscriminatorBlock(torch.nn.Module):
    def __init__(self,
        in_channels,                        # Number of input channels
        out_channels,                       # Number of output channels
        resolution,                         # Resolution of this block
        img_channels,                       # Number of input color channels
        stem                = False,        # First block (processing image)
        architecture        = "resnet",     # Architecture: "orig", "skip", "resnet"
        act                 = "lrelu",      # Activation function: "relu", "lrelu", etc
        resample_kernel     = [1,3,3,1],    # Low-pass kernel to apply when resampling activations
        **_kwargs                           # Ignore unrecognized keyword args
    ):
        assert architecture in ["orig", "skip", "resnet"]
        super().__init__()
        self.in_channels = in_channels
        self.img_channels = img_channels
        self.resolution = resolution
        self.architecture = architecture
        self.stem = stem

        self.register_buffer("resample_kernel", upfirdn2d.setup_filter(resample_kernel))

        if stem or architecture == "skip":
            self.fromrgb = Conv2dLayer(img_channels, in_channels, kernel_size = 1, act = act)

        self.conv0 = Conv2dLayer(in_channels, in_channels, kernel_size = 3, act = act)

        self.conv1 = Conv2dLayer(in_channels, out_channels, kernel_size = 3, down = 2, resample_kernel = resample_kernel,
            act = act, gain = get_gain(architecture))

        if architecture == "resnet":
            self.skip = Conv2dLayer(in_channels, out_channels, kernel_size = 1, bias = False, down = 2,
                resample_kernel = resample_kernel, gain = get_gain(architecture))

    def forward(self, x, img):
        # Input
        if x is not None:
            # torch_misc.assert_shape(x, [None, self.in_channels, self.resolution, self.resolution])
            x = convert(x)

        # FromRGB
        if self.stem or self.architecture == "skip":
            # torch_misc.assert_shape(img, [None, self.img_channels, self.resolution, self.resolution])
            y = self.fromrgb(convert(img))
            x = x + y if x is not None else y
            img = upfirdn2d.downsample2d(img, self.resample_kernel) if self.architecture == "skip" else None

        # Main layers
        if self.architecture == "resnet":
            y = self.skip(x)
            x = self.conv0(x)
            x = self.conv1(x)
            x = y.add_(x)
        else:
            x = self.conv0(x)
            x = self.conv1(x)

        return x, img

@persistence.persistent_class
class MinibatchStdLayer(torch.nn.Module):
    def __init__(self, group_size, num_channels = 1):
        super().__init__()
        self.group_size = group_size
        self.num_channels = num_channels

    def forward(self, x):
        N, C, H, W = x.shape
        with torch_misc.suppress_tracer_warnings(): # as_tensor results are registered as constants
            G = torch.min(torch.as_tensor(self.group_size), torch.as_tensor(N)) if self.group_size is not None else N
        F = self.num_channels
        c = C // F

        y = x.reshape(G, -1, F, c, H, W)    # [GnFcHW] Split minibatch N into n groups of size G, and channels C into F groups of size c
        y = y - y.mean(dim = 0)             # [GnFcHW] Subtract mean over group
        y = y.square().mean(dim = 0)        # [nFcHW]  Compute variance over group
        y = (y + 1e-8).sqrt()               # [nFcHW]  Compute stddev over group
        y = y.mean(dim=[2,3,4])             # [nF]     Take average over channels and pixels
        y = y.reshape(-1, F, 1, 1)          # [nF11]   Add missing dimensions
        y = y.repeat(G, 1, H, W)            # [NFHW]   Replicate over group and pixels
        x = torch.cat([x, y], dim = 1)      # [NCHW]   Append to input as new channels
        return x

@persistence.persistent_class
class DiscriminatorEpilogue(torch.nn.Module):
    def __init__(self,
        in_channels,                    # Number of input channels
        c_dim,                          # Dimensionality of conditioning label, 0 = no label
        resolution,                     # Resolution of this block
        img_channels,                   # Number of input color channels
        architecture        = "resnet", # Architecture: "orig", "skip", "resnet"
        mbstd_group_size    = 4,        # Group size for the minibatch standard deviation layer, None = entire minibatch
        mbstd_num_channels  = 1,        # Number of channels for the minibatch standard deviation layer, 0 = disable
        act                 = "lrelu",  # Activation function: "relu", "lrelu", etc
    ):
        assert architecture in ["orig", "skip", "resnet"]
        super().__init__()
        self.in_channels = in_channels
        self.c_dim = c_dim
        self.resolution = resolution
        self.img_channels = img_channels
        self.architecture = architecture

        if architecture == "skip":
            self.fromrgb = Conv2dLayer(img_channels, in_channels, kernel_size = 1, act = act)
        
        self.mbstd = MinibatchStdLayer(group_size = mbstd_group_size, num_channels = mbstd_num_channels) if mbstd_num_channels > 0 else None
        self.conv = Conv2dLayer(in_channels + mbstd_num_channels, in_channels, kernel_size = 3, act = act)
        
        self.fc =  FullyConnectedLayer(in_channels * (resolution ** 2), in_channels, act = act)
        self.out = FullyConnectedLayer(in_channels, max(c_dim, 1))

    def forward(self, x, img, c):
        # torch_misc.assert_shape(x, [None, self.in_channels, self.resolution, self.resolution]) # [NCHW]

        # FromRGB
        x = convert(x)
        if self.architecture == "skip":
            x = x + self.fromrgb(convert(img))

        # Main layers
        if self.mbstd is not None:
            x = self.mbstd(x)
        x = self.conv(x)
        x = self.fc(x.flatten(1))
        x = self.out(x)

        # Conditioning
        if self.c_dim > 0:
            torch_misc.assert_shape(c, [None, self.c_dim])
            x = (x * c).sum(dim = 1, keepdim = True)

        return x

@persistence.persistent_class
class Discriminator(torch.nn.Module):
    def __init__(self,
        c_dim,                          # Conditioning label (C) dimensionality
        img_resolution,                 # Input resolution
        img_channels,                   # Number of input color channels
        crop_ratio          = None,     # Crop image features to output ratio during model's computation
        architecture        = "resnet", # Architecture: "orig", "skip", "resnet"
        channel_base        = 32 << 10, # Overall multiplier for the number of channels
        channel_max         = 512,      # Maximum number of channels in any layer
        block_kwargs        = {},       # Arguments for DiscriminatorBlock
        epilogue_kwargs     = {},       # Arguments for DiscriminatorEpilogue
        **_kwargs                       # Ignore unrecognized keyword args
    ):
        super().__init__()
        self.c_dim = c_dim
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.crop_ratio = crop_ratio
        self.block_resolutions = [2 ** i for i in range(int(np.log2(img_resolution)), 2, -1)]
        channels_num = get_res2channels(channel_base, channel_max)

        common_kwargs = dict(img_channels = img_channels, architecture = architecture)
        for res in self.block_resolutions:
            in_channels = channels_num(res)
            out_channels = channels_num(res // 2)
            block = DiscriminatorBlock(in_channels, out_channels, resolution = res, stem = (res == img_resolution),
                **block_kwargs, **common_kwargs)
            setattr(self, f"b{res}", block)

        self.b4 = DiscriminatorEpilogue(channels_num(4), c_dim, resolution = 4, **epilogue_kwargs, **common_kwargs)

    def forward(self, img, c, **block_kwargs):
        img = misc.crop_tensor(img, self.crop_ratio)
        img_shape = misc.crop_tensor_shape((self.img_resolution, self.img_resolution), self.crop_ratio)
        torch_misc.assert_shape(img, [None, self.img_channels, *img_shape])

        x = None
        for res in self.block_resolutions:
            block = getattr(self, f"b{res}")
            x, img = block(x, img, **block_kwargs)
        x = self.b4(x, img, c)
        return x
