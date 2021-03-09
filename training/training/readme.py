The model consists of two networks:
- Generator: which produces the images (x) given randomly sampled latents (z). 
  The latent z has a shape [batch_size, component_num, latent_dim], where 
  component_num = 1 by default (Vanilla GAN, StyleGAN) but is > 1 for the GANsformer model.
  We can define the latent components by splitting z along the second dimension to obtain
  z_1,...,z_k latent components.
  The generator likewise consists of two parts:
  - Mapping network: converts sampled latents from a normal distribution (z)
    to the intermediate space (w). A series of Feed-forward layers.
    The k latent components either are mapped independently from the z space to the w space
    or interact with each other through self-attention (optional flag).

  - Synthesis network: the intermediate latents w are used to guide the generation of new images.
    Images features begin from a small constant/sampled grid of 4x4, and then go through multiple
    layers of convolution and up-sampling until reaching the desirable resolution (e.g. 256x256).
    After each convolution, the image features are modulated (meaning that their variance and bias 
    are controlled) by the intermediate latent vectors w.
    While in the StyleGAN model there is one global w vectors that controls all the features equally.
    The GANsformer uses attention so that the k latent components specialize to control different 
    regions in the image to create it cooperatively, and therefore perform better especially in 
    generating images depicting multi-object scenes.
    
    Attention can be used in several ways: 
    - Simplex Attention: when attention is applied in one direction only from the latents to the 
      image features (top-down).
    - Duplex Attention: when attention is applied in the two directions: latents to image features 
      (top-down) and then image features back to latents (bottom-up), so that each representation 
      informs the other iteratively.
    - Self-Attention between latents: can also be used so to each direct interactions between the latents.
    - Self-Attention between image features (SAGAN model): prior approaches used attention directly
      between the image features, but this method does not scale well due to the quadratic number of features
      which becomes very high for high-resolutions.
     
- Discriminator: Receives and image and has to predict whether it is real or fake -- originating
  from the dataset or the generator. The model perform multiple layers of convolution and downsampling
  on the image, reducing the representation's resolution gradually until making final prediction.
  
  Optionally, attention can be incorporated into the discriminator as well where it has multiple (k)
  aggregator variables, that use attention to adaptively collect information from the image while 
  being processed. We observe small improvements in model performance when attention is used in the 
  discriminator, although note that most of the gain in using attention based on our observations arises
  from the generator.