# A project class to perform optimization for finding a latent for which the generator produces
# an output that approximates a target image.
import numpy as np
import tensorflow as tf
import dnnlib
import dnnlib.tflib as tflib

from training import misc

class Projector:
    def __init__(self):
        self.num_steps                  = 1000
        self.dlatent_avg_samples        = 10000
        self.initial_learning_rate      = 0.1
        self.initial_noise_factor       = 0.05
        self.lr_rampdown_length         = 0.25
        self.lr_rampup_length           = 0.05
        self.noise_ramp_length          = 0.75
        self.regularize_noise_weight    = 1e5
        self.verbose                    = False
        self.clone_net                  = True
        self.lossType                   = "l2"
        self.loss                       = 0.0
        self._lpips                     = None
        self._cur_step                  = None

    def _info(self, *args):
        if self.verbose:
            print("Projector: ", *args)

    def set_network(self, Gs, minibatch_size = 1):
        assert minibatch_size == 1
        self._Gs = Gs
        self._minibatch_size = minibatch_size
        if self._Gs is None:
            return
        if self.clone_net:
            self._Gs = self._Gs.clone()

        # Find dlatent stats
        self._info("Finding W midpoint and stddev using %d samples..." % self.dlatent_avg_samples)
        latent_samples = np.random.RandomState(123).randn(self.dlatent_avg_samples, *self._Gs.input_shapes[0][1:])
        dlatent_samples = self._Gs.components.mapping.run(latent_samples, None)[:, :, :1, :] # [N, 1, 512]
        self._dlatent_avg = np.mean(dlatent_samples, axis = 0, keepdims = True) # [1, 1, 512]
        self._dlatent_std = (np.sum((dlatent_samples - self._dlatent_avg) ** 2) / self.dlatent_avg_samples) ** 0.5
        self._info("std = %g" % self._dlatent_std)

        # Find noise inputs
        self._info("Setting up noise inputs...")
        self._noise_vars = []
        noise_init_ops = []
        noise_normalize_ops = []
        while True:
            n = "G_synthesis/noise%d" % len(self._noise_vars)
            if n not in self._Gs.vars:
                break
            v = self._Gs.vars[n]
            self._noise_vars.append(v)
            noise_init_ops.append(tf.assign(v, tf.random_normal(tf.shape(v), dtype = tf.float32)))
            noise_mean = tf.reduce_mean(v)
            noise_std = tf.reduce_mean((v - noise_mean)**2)**0.5
            noise_normalize_ops.append(tf.assign(v, (v - noise_mean) / noise_std))
            self._info(n, v)
        self._noise_init_op = tf.group(*noise_init_ops)
        self._noise_normalize_op = tf.group(*noise_normalize_ops)

        # Find weight inputs (relevant only for k-GAN)
        self._info("Setting up weight inputs...")
        self._weight_vars = []
        weight_init_ops = []
        while True:
            n = "G_synthesis/component_%d" % len(self._weight_vars)
            if n not in self._Gs.vars:
                break
            v = self._Gs.vars[n]
            self._weight_vars.append(v)
            weight_init_ops.append(tf.assign(v, tf.ones(tf.shape(v), dtype = tf.float32)))
            self._info(n, v)
        self._weight_init_op = tf.group(*weight_init_ops)

        # Image output graph
        self._info("Building image output graph...")
        self._dlatents_var = tf.Variable(tf.zeros([self._minibatch_size] + list(self._dlatent_avg.shape[1:])), name = "dlatents_var")
        self._noise_in = tf.placeholder(tf.float32, [], name = "noise_in")
        dlatents_noise = tf.random.normal(shape = self._dlatents_var.shape) * self._noise_in
        self._dlatents_expr = tf.tile(self._dlatents_var + dlatents_noise, [1, 1, self._Gs.components.synthesis.input_shape[1], 1])
        self._imgs_expr = self._Gs.components.synthesis.get_output_for(self._dlatents_expr, randomize_noise = False)

        # Downsample image to 256x256 if it's larger than that. VGG was built for 224x224 images
        proc_imgs_expr = (self._imgs_expr + 1) * (255 / 2)
        sh = proc_imgs_expr.shape.as_list()
        if sh[2] > 256:
            factor = sh[2] // 256
            proc_imgs_expr = tf.reduce_mean(tf.reshape(proc_imgs_expr, [-1, sh[1], sh[2] // factor, factor, sh[2] // factor, factor]), axis = [3, 5])

        # Loss graph
        self._info("Building loss graph...")
        self._target_imgs_var = tf.Variable(tf.zeros(proc_imgs_expr.shape), name = "target_imgs_var")

        if self.lossType == "l2":
            self._loss = tf.reduce_mean((proc_imgs_expr - self._target_imgs_var) ** 2, axis = [1, 2, 3])
        elif self.lossType == "l1":
            self._loss = tf.reduce_mean(tf.abs(proc_imgs_expr - self._target_imgs_var), axis = [1, 2, 3])
        else:
            if self._lpips is None:
                self._lpips = misc.load_pkl("http://d36zk2xti64re0.cloudfront.net/stylegan1/networks/metrics/vgg16_zhang_perceptual.pkl")
            self._loss = self._lpips.get_output_for(proc_imgs_expr, self._target_imgs_var)
        self._loss = self._dist = tf.reduce_sum(self._loss)  

        # Noise regularization graph
        self._info("Building noise regularization graph...")
        reg_loss = 0.0
        for v in self._noise_vars:
            sz = v.shape[2]
            while True:
                reg_loss += tf.reduce_mean(v * tf.roll(v, shift = 1, axis = 3))**2 + \
                tf.reduce_mean(v * tf.roll(v, shift = 1, axis = 2))**2
                if sz <= 8:
                    break # Small enough already
                v = tf.reshape(v, [1, 1, sz//2, 2, sz//2, 2]) # Downscale
                v = tf.reduce_mean(v, axis = [3, 5])
                sz = sz // 2
        self._loss += reg_loss * self.regularize_noise_weight

        # Optimizer
        self._info("Setting up optimizer...")
        self._lrate_in = tf.placeholder(tf.float32, [], name = "lrate_in")
        self._opt = dnnlib.tflib.Optimizer(learning_rate = self._lrate_in)
        self._opt.register_gradients(self._loss, [self._dlatents_var] + self._noise_vars + self._weight_vars)
        self._opt_step = self._opt.apply_updates()

    def run(self, target_imgs):
        # Run to completion
        self.start(target_imgs)
        while self._cur_step < self.num_steps:
            self.step()

        # Collect results
        pres = dnnlib.EasyDict()
        pres.dlatents = self.get_dlatents()
        pres.noises = self.get_noises()
        pres.imgs = self.get_imgs()
        return pres

    def start(self, target_imgs):
        assert self._Gs is not None

        # Prepare target images
        self._info("Preparing target images...")
        target_imgs = np.asarray(target_imgs, dtype = "float32")
        target_imgs = (target_imgs + 1) * (255 / 2)
        sh = target_imgs.shape
        assert sh[0] == self._minibatch_size
        if sh[2] > self._target_imgs_var.shape[2]:
            factor = sh[2] // self._target_imgs_var.shape[2]
            target_imgs = np.reshape(target_imgs, [-1, sh[1], sh[2] // factor, factor, sh[3] // factor, factor]).mean((3, 5))

        # Initialize optimization state
        self._info("Initializing optimization state...")
        tflib.set_vars({self._target_imgs_var: target_imgs, 
            self._dlatents_var: np.tile(self._dlatent_avg, (self._minibatch_size, 1, 1, 1))})
        tflib.run(self._noise_init_op)
        tflib.run(self._weight_init_op)
        self._opt.reset_optimizer_state()
        self._cur_step = 0

    def step(self):
        assert self._cur_step is not None
        if self._cur_step >= self.num_steps:
            return
        if self._cur_step == 0:
            self._info("Running...")

        # Hyperparameters
        t = self._cur_step / self.num_steps
        noise_strength = self._dlatent_std * self.initial_noise_factor * max(0.0, 1.0 - t / self.noise_ramp_length) ** 2
        lr_ramp = min(1.0, (1.0 - t) / self.lr_rampdown_length)
        lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
        lr_ramp = lr_ramp * min(1.0, t / self.lr_rampup_length)
        learning_rate = self.initial_learning_rate * lr_ramp

        # Train
        feed_dict = {self._noise_in: noise_strength, self._lrate_in: learning_rate}
        _, dist_value, loss_value = tflib.run([self._opt_step, self._dist, self._loss], feed_dict)
        tflib.run(self._noise_normalize_op)

        self.loss = loss_value
        # Print status
        self._cur_step += 1
        if self._cur_step == self.num_steps or self._cur_step % 10 == 0:
            self._info("%-8d%-12g%-12g" % (self._cur_step, dist_value, loss_value))
        if self._cur_step == self.num_steps:
            self._info("Done")

    def get_cur_step(self):
        return self._cur_step

    def get_loss(self):
        return self.loss     

    def get_dlatents(self):
        return tflib.run(self._dlatents_expr, {self._noise_in: 0})

    def get_noises(self):
        return tflib.run(self._noise_vars)

    def get_imgs(self):
        return tflib.run(self._imgs_expr, {self._noise_in: 0})
