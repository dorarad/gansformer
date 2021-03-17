# Loss functions for the generator and the discriminator
from warnings import simplefilter
simplefilter(action = "ignore", category = FutureWarning)

import numpy as np
import tensorflow as tf
import dnnlib.tflib as tflib
from dnnlib.tflib.autosummary import autosummary

def G_loss(G, D,
        dataset,                 # The dataset object for the real images
        minibatch_size,          # size of each minibatch
        loss_type,               # The loss type: logistic, hinge, wgan
        reg_weight = 1.0,        # Regularization strength
        pathreg = False,         # Path regularization
        pl_minibatch_shrink = 2, # Minibatch shrink (for path regularization only)
        pl_decay = 0.01,         # Decay (for path regularization only)
        pl_weight = 2.0,         # Weight (for path regularization only)
        **kwargs):

    latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
    labels = dataset.get_random_labels_tf(minibatch_size)
    fake_imgs_out = G.get_output_for(latents, labels, is_training = True)[0]
    fake_scores_out = D.get_output_for(fake_imgs_out, labels, is_training = True)

    if loss_type == "logistic":
        loss = -tf.nn.softplus(fake_scores_out)
    elif loss_type == "logistic_ns":
        loss = tf.nn.softplus(-fake_scores_out)
    elif loss_type == "hinge":
        loss = -tf.maximum(0.0, 1.0 + fake_scores_out)
    elif loss_type == "wgan":
        loss = -fake_scores_out

    reg = None
    if pathreg:
        with tf.name_scope("PathReg"):
            # Evaluate the regularization term using a smaller minibatch to conserve memory
            if pl_minibatch_shrink > 1:
                pl_minibatch = minibatch_size // pl_minibatch_shrink
                pl_latents = tf.random_normal([pl_minibatch] + G.input_shapes[0][1:])
                pl_labels = dataset.get_random_labels_tf(pl_minibatch)
                ret = G.get_output_for(pl_latents, pl_labels, is_training = True, return_dlatents = True)
                fake_imgs_out, dlatents = ret[0], ret[-1]
            # Compute |J*y|
            pl_noise = tf.random_normal(tf.shape(fake_imgs_out)) / np.sqrt(np.prod(G.output_shape[2:]))
            pl_grads = tf.gradients(tf.reduce_sum(fake_imgs_out * pl_noise), [dlatents])[0]
            pl_lengths = tf.sqrt(tf.reduce_mean(tf.reduce_sum(tf.square(pl_grads), axis = 3), axis = [1, 2]))
            pl_lengths = autosummary("Loss/pl_lengths", pl_lengths)

            # Track exponential moving average of |J*y|
            with tf.control_dependencies(None):
                pl_mean_var = tf.Variable(name = "pl_mean", trainable = False, initial_value = 0.0, dtype = tf.float32)
            pl_mean = pl_mean_var + pl_decay * (tf.reduce_mean(pl_lengths) - pl_mean_var)
            pl_update = tf.assign(pl_mean_var, pl_mean)

            # Calculate (|J*y|-a)^2
            with tf.control_dependencies([pl_update]):
                pl_penalty = tf.square(pl_lengths - pl_mean)
                pl_penalty = autosummary("Loss/pl_penalty", pl_penalty)

            reg = pl_penalty * pl_weight

    if reg is not None:
        reg *= reg_weight

    return loss, reg

def D_loss(G, D,
        reals,                  # A batch of real images
        labels,                 # A batch of labels (default 0s if no labels)
        minibatch_size,         # Size of each minibatch
        loss_type,              # Loss type: logistic, hinge, wgan
        reg_type,               # Regularization type: r1, t2, gp (mixed)
        gamma = 10.0,           # Regularization strength
        wgan_epsilon = 0.001,   # Wasserstein epsilon (for wgan only)
        wgan_target = 1.0,      # Wasserstein target (for wgan only)
        **kwargs):

    latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
    fake_imgs_out = G.get_output_for(latents, labels, is_training = True)[0]

    real_scores_out = D.get_output_for(reals, labels, is_training = True)
    fake_scores_out = D.get_output_for(fake_imgs_out, labels, is_training = True)

    real_scores_out = autosummary("Loss/scores/real", real_scores_out)
    fake_scores_out = autosummary("Loss/scores/fake", fake_scores_out)

    if loss_type == "logistic":
        loss = tf.nn.softplus(fake_scores_out)
        loss += tf.nn.softplus(-real_scores_out)
    elif loss_type == "hinge":
        loss = tf.maximum(0.0, 1.0 + fake_scores_out)
        loss += tf.maximum(0.0, 1.0 - real_scores_out)
    elif loss_type == "wgan":
        loss = fake_scores_out - real_scores_out
        with tf.name_scope("EpsilonPenalty"):
            epsilon_penalty = autosummary("Loss/epsilon_penalty", tf.square(real_scores_out))
            loss += epsilon_penalty * wgan_epsilon

    reg = None
    with tf.name_scope("GradientPenalty"):
        if reg_type in ["r1", "r2"]:
            if reg_type == "r1":
                grads = tf.gradients(tf.reduce_sum(real_scores_out), [reals])[0]
            else:
                grads = tf.gradients(tf.reduce_sum(fake_scores_out), [fake_imgs_out])[0]
            gradient_penalty = tf.reduce_sum(tf.square(grads), axis = [1, 2, 3])
            gradient_penalty = autosummary("Loss/gradient_penalty", gradient_penalty)
            reg = gradient_penalty * (gamma * 0.5)
        elif reg_type == "gp":
            mixing_factors = tf.random_uniform([minibatch_size, 1, 1, 1], 0.0, 1.0, dtype = fake_imgs_out.dtype)
            mixed_imgs_out = tflib.lerp(tf.cast(reals, fake_imgs_out.dtype), fake_imgs_out, mixing_factors)
            mixed_scores_out = D.get_output_for(mixed_imgs_out, labels, is_training = True)
            mixed_scores_out = autosummary("Loss/scores/mixed", mixed_scores_out)
            mixed_grads = tf.gradients(tf.reduce_sum(mixed_scores_out), [mixed_imgs_out])[0]
            mixed_norms = tf.sqrt(tf.reduce_sum(tf.square(mixed_grads), axis = [1, 2, 3]))
            mixed_norms = autosummary("Loss/mixed_norms", mixed_norms)
            gradient_penalty = tf.square(mixed_norms - wgan_target)
            reg = gradient_penalty * (gamma / (wgan_target ** 2))

    return loss, reg
