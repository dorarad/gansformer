# Loss functions for the generator and the discriminator
import numpy as np
import torch
from torch_utils import training_stats
from torch_utils import misc
from torch_utils.ops import conv2d_gradfix

#----------------------------------------------------------------------------

class Loss:
    def accumulate_gradients(self, stage, real_img, real_c, gen_z, gen_c, sync, gain): # to be overridden by subclass
        raise NotImplementedError()

#----------------------------------------------------------------------------

class StyleGAN2Loss(Loss):
    def __init__(self, device, G, D, g_loss = "logistic_ns", d_loss = "logistic", # G_mapping, G_synthesis, 
            style_mixing = 0.9, component_mixing = 0.0, r1_gamma = 10, pl_batch_shrink = 2, pl_decay = 0.01, 
            pl_weight = 2.0, wgan_epsilon = 0.001):
        super().__init__()
        self.device = device
        self.G = G
        self.D = D

        self.g_loss = g_loss                            # logistic, logistic_ns, hinge, or wgan
        self.d_loss = d_loss                            # logistic, hinge, or wgan

        self.style_mixing = style_mixing
        self.component_mixing = component_mixing

        self.r1_gamma = r1_gamma                        # D's regularization weight
        self.wgan_epsilon = wgan_epsilon                # For D's optional wgan-loss regularization
        self.pl_batch_shrink = pl_batch_shrink          # For G's optional path length regularization
        self.pl_decay = pl_decay                        # For G's optional path length regularization
        self.pl_weight = pl_weight                      # For G's optional path length regularization
        self.pl_mean = torch.zeros([], device = device)

    def run_G(self, z, c, sync):
        with misc.ddp_sync(self.G, sync):
            ws = self.G(z, c, subnet = "mapping")
            if self.style_mixing > 0:
                with torch.autograd.profiler.record_function("style_mixing"):
                    cutoff = torch.empty([], dtype = torch.int64, device = ws.device).random_(1, ws.shape[2])
                    cutoff = torch.where(torch.rand([], device = ws.device) < self.style_mixing, cutoff, torch.full_like(cutoff, ws.shape[2]))
                    ws[:, :, cutoff:] = self.G(torch.randn_like(z), c, skip_w_avg_update = True, subnet = "mapping")[:, :, cutoff:]
            if self.component_mixing > 0:
                with torch.autograd.profiler.record_function("component_mixing"):
                    cutoff = torch.empty([], dtype = torch.int64, device = ws.device).random_(1, ws.shape[1])
                    cutoff = torch.where(torch.rand([], device = ws.device) < self.style_mixing, cutoff, torch.full_like(cutoff, ws.shape[1]))
                    ws[:, cutoff:] = self.G(torch.randn_like(z), c, skip_w_avg_update = True, subnet = "mapping")[:, cutoff:]
        # with misc.ddp_sync(self.G, sync):
            img = self.G(ws = ws, subnet = "synthesis")
        return img, ws

    def run_D(self, img, c, sync):
        with misc.ddp_sync(self.D, sync):
            logits = self.D(img, c)
        return logits

    def accumulate_gradients(self, stage, real_img, real_c, gen_z, gen_c, sync, gain):
        assert stage in ["G_main", "G_reg", "G_both", "D_main", "D_reg", "D_both"]
        G_main = (stage in ["G_main", "G_both"])
        D_main = (stage in ["D_main", "D_both"])
        G_pl   = (stage in ["G_reg", "G_both"]) and (self.pl_weight != 0)
        D_r1   = (stage in ["D_reg", "D_both"]) and (self.r1_gamma != 0)

        # G_main: Maximize logits for generated images
        if G_main:
            with torch.autograd.profiler.record_function("G_main_forward"):
                gen_img, _gen_ws = self.run_G(gen_z, gen_c, sync = (sync and not G_pl)) # May get synced by G_pl
                gen_logits = self.run_D(gen_img, gen_c, sync = False)
                training_stats.report("Loss/scores/fake", gen_logits)
                training_stats.report("Loss/signs/fake", gen_logits.sign())
    
                if self.g_loss == "logistic":
                    loss_G_main = -torch.nn.functional.softplus(gen_logits) # -log(sigmoid(gen_logits))
                elif self.g_loss == "logistic_ns":
                    loss_G_main = torch.nn.functional.softplus(-gen_logits) # -log(sigmoid(gen_logits))
                elif self.g_loss == "hinge":
                    loss_G_main = -torch.clamp(1.0 + gen_logits, min = 0)
                elif self.g_loss == "wgan":
                    loss_G_main = -gen_logits

                training_stats.report("Loss/G/loss", loss_G_main)
            with torch.autograd.profiler.record_function("G_main_backward"):
                loss_G_main.mean().mul(gain).backward()

        # G_pl: Apply path length regularization
        if G_pl:
            with torch.autograd.profiler.record_function("G_pl_forward"):
                batch_size = gen_z.shape[0] // self.pl_batch_shrink
                gen_img, gen_ws = self.run_G(gen_z[:batch_size], gen_c[:batch_size], sync = sync)
                pl_noise = torch.randn_like(gen_img) / np.sqrt(gen_img.shape[2] * gen_img.shape[3])
                with torch.autograd.profiler.record_function("pl_grads"), conv2d_gradfix.no_weight_gradients():
                    pl_grads = torch.autograd.grad(outputs=[(gen_img * pl_noise).sum()], inputs=[gen_ws], create_graph = True, only_inputs = True)[0]
                pl_lengths = pl_grads.square().sum(2).mean(1).sqrt()
                pl_mean = self.pl_mean.lerp(pl_lengths.mean(), self.pl_decay)
                self.pl_mean.copy_(pl_mean.detach())
                pl_penalty = (pl_lengths - pl_mean).square()
                training_stats.report("Loss/pl_penalty", pl_penalty)
                loss_G_pl = pl_penalty * self.pl_weight
                training_stats.report("Loss/G/reg", loss_G_pl)
            with torch.autograd.profiler.record_function("G_pl_backward"):
                (gen_img[:, 0, 0, 0] * 0 + loss_G_pl).mean().mul(gain).backward()

        # D_main: Minimize logits for generated images
        loss_D_gen = 0
        if D_main:
            with torch.autograd.profiler.record_function("D_gen_forward"):
                gen_img, _gen_ws = self.run_G(gen_z, gen_c, sync = False)
                gen_logits = self.run_D(gen_img, gen_c, sync = False) # Gets synced by loss_D_real
                training_stats.report("Loss/scores/fake", gen_logits)
                training_stats.report("Loss/signs/fake", gen_logits.sign())
            
                if self.d_loss == "logistic":
                    loss_D_gen = torch.nn.functional.softplus(gen_logits) # -log(1 - sigmoid(gen_logits))
                elif self.d_loss == "hinge":
                    loss_D_gen = torch.clamp(1.0 + gen_logits, min = 0)
                elif self.d_loss == "wgan":
                    loss_D_gen = gen_logits

            with torch.autograd.profiler.record_function("D_gen_backward"):
                loss_D_gen.mean().mul(gain).backward()

        # D_main: Maximize logits for real images
        # D_r1: Apply R1 regularization
        if D_main or D_r1:
            name = "D_real_D_r1" if D_main and D_r1 else "D_real" if D_main else "D_r1"
            with torch.autograd.profiler.record_function(name + "_forward"):
                real_img_tmp = real_img.detach().requires_grad_(D_r1)
                real_logits = self.run_D(real_img_tmp, real_c, sync = sync)
                training_stats.report("Loss/scores/real", real_logits)
                training_stats.report("Loss/signs/real", real_logits.sign())

                loss_D_real = 0
                if D_main:
                    if self.d_loss == "logistic":
                        loss_D_real = torch.nn.functional.softplus(-real_logits) # -log(sigmoid(real_logits))
                    elif self.d_loss == "hinge":
                        loss_D_real = torch.clamp(1.0 - real_logits, min = 0)
                    elif self.d_loss == "wgan":
                        loss_D_real = -real_logits + tf.square(real_logits) * wgan_epsilon

                    training_stats.report("Loss/D/loss", loss_D_gen + loss_D_real)

                loss_D_r1 = 0
                if D_r1:
                    with torch.autograd.profiler.record_function("r1_grads"), conv2d_gradfix.no_weight_gradients():
                        r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp], create_graph = True, only_inputs = True)[0]
                    r1_penalty = r1_grads.square().sum([1,2,3])
                    loss_D_r1 = r1_penalty * (self.r1_gamma / 2)
                    training_stats.report("Loss/r1_penalty", r1_penalty)
                    training_stats.report("Loss/D/reg", loss_D_r1)

            with torch.autograd.profiler.record_function(name + "_backward"):
                (real_logits * 0 + loss_D_real + loss_D_r1).mean().mul(gain).backward()
